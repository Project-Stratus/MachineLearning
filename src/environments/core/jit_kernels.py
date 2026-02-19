# environments/core/jit_kernels.py
from numba import njit
import numpy as np
import math

# ISA constants (duplicated from constants.py for Numba; kept in sync)
_T0 = 288.15
_LAPSE = 0.0065
_TROPO_ALT = 11_000.0
_T_TROPO = _T0 - _LAPSE * _TROPO_ALT  # ~216.65 K
_G = 9.81
_R = 8.314462618
_M_AIR = 0.0289647
_GM_RL = _G * _M_AIR / (_R * _LAPSE)  # exponent for tropospheric pressure

# Sutherland's law constants
_MU_REF = 1.716e-5
_T_REF = 273.15
_S_SUTH = 110.4

# Geometry
_PI = math.pi
_FOUR_THIRDS_PI = 4.0 / 3.0 * _PI


@njit(cache=True, fastmath=True)
def temperature_numba(alt: float) -> float:
    """ISA temperature (K) at altitude."""
    if alt <= _TROPO_ALT:
        return _T0 - _LAPSE * alt
    return _T_TROPO


@njit(cache=True, fastmath=True)
def pressure_numba(p0: float, alt: float) -> float:
    """ISA pressure (Pa) at altitude."""
    if alt <= _TROPO_ALT:
        return p0 * (1.0 - _LAPSE * alt / _T0) ** _GM_RL
    # Stratosphere
    p_tropo = p0 * (_T_TROPO / _T0) ** _GM_RL
    scale_h = _R * _T_TROPO / (_M_AIR * _G)
    return p_tropo * math.exp(-(alt - _TROPO_ALT) / scale_h)


@njit(cache=True, fastmath=True)
def density_numba(p0: float, M: float, alt: float) -> float:
    """ISA density (kg/m^3) at altitude."""
    p = pressure_numba(p0, alt)
    T = temperature_numba(alt)
    return p * M / (_R * T)


@njit(cache=True, fastmath=True)
def dynamic_viscosity_numba(alt: float) -> float:
    """Dynamic viscosity of air (Pa·s) via Sutherland's law."""
    T = temperature_numba(alt)
    return _MU_REF * (T / _T_REF) ** 1.5 * (_T_REF + _S_SUTH) / (T + _S_SUTH)


@njit(cache=True, fastmath=True)
def sphere_area_from_volume(volume: float) -> float:
    """Frontal (cross-sectional) area of a sphere given its volume."""
    r = (volume / _FOUR_THIRDS_PI) ** (1.0 / 3.0)
    return _PI * r * r


@njit(cache=True, fastmath=True)
def morrison_cd(Re: float) -> float:
    """
    Drag coefficient for a sphere using the Morrison (2013) correlation.
    Valid for Re from 0 to ~1e6.  Returns ~0.44 in the Newton regime
    (1e3 < Re < 2e5) and captures the drag crisis.
    """
    if Re < 1e-8:
        return 0.0  # no flow, no drag
    term1 = 24.0 / Re
    term2 = 2.6 * (Re / 5.0) / (1.0 + (Re / 5.0) ** 1.52)
    term3 = 0.411 * (Re / 263000.0) ** (-7.94) / (1.0 + (Re / 263000.0) ** (-8.0))
    term4 = Re ** 0.80 / 461000.0
    return term1 + term2 + term3 + term4


@njit(cache=True, fastmath=True)
def wind_sample_idx_numba(x: float, y: float, z: float,
                          x0: float, inv_dx: float,
                          y0: float, inv_dy: float,
                          z0: float, inv_dz: float,
                          cells: int,
                          fx_grid: np.ndarray, fy_grid: np.ndarray) -> (float, float):
    ix = int((x - x0) * inv_dx)
    iy = int((y - y0) * inv_dy)
    iz = int((z - z0) * inv_dz)
    if ix < 0: ix = 0
    elif ix >= cells: ix = cells - 1
    if iy < 0: iy = 0
    elif iy >= cells: iy = cells - 1
    if iz < 0: iz = 0
    elif iz >= cells: iz = cells - 1
    return fx_grid[ix, iy, iz], fy_grid[ix, iy, iz]


@njit(cache=True, fastmath=True)
def physics_step_numba(pos: np.ndarray, vel: np.ndarray,
                       dt: float,
                       mass: float,
                       G: float,
                       rho_air: float, volume: float,
                       external_force: np.ndarray, control_force: np.ndarray,
                       n_dim: int, vel_max: float) -> None:
    """
    In-place integration for 1D/2D/3D with volume-dependent drag.

    Drag coefficient and frontal area are derived from the current balloon
    volume and flow conditions (Morrison CD correlation + sphere geometry).
    """
    z_idx = n_dim - 1
    alt = pos[z_idx]

    # Volume-dependent frontal area
    area = sphere_area_from_volume(volume)

    # Diameter for Reynolds number
    r = (volume / _FOUR_THIRDS_PI) ** (1.0 / 3.0)
    diameter = 2.0 * r

    # Speed
    speed2 = 0.0
    for i in range(n_dim):
        vi = vel[i]
        speed2 += vi * vi

    have_speed = speed2 > 1e-24  # SPEED_EPS**2
    speed = math.sqrt(speed2) if have_speed else 0.0
    inv_speed = 1.0 / speed if have_speed else 0.0

    # Reynolds-dependent drag coefficient
    if have_speed:
        mu = dynamic_viscosity_numba(alt)
        Re = rho_air * speed * diameter / mu
        cd = morrison_cd(Re)
        f_mag = 0.5 * cd * area * rho_air * speed2
    else:
        f_mag = 0.0

    buoy_z = rho_air * G * volume - mass * G  # only affects vertical axis

    # Integrate per-axis
    for i in range(n_dim):
        drag_i = (-f_mag * vel[i] * inv_speed) if have_speed else 0.0
        f_i = external_force[i] + control_force[i] + drag_i
        if i == z_idx:
            f_i += buoy_z
        # acceleration
        a_i = f_i / mass
        # vel update + clip
        vi = vel[i] + a_i * dt
        if vi > vel_max:
            vi = vel_max
        elif vi < -vel_max:
            vi = -vel_max
        vel[i] = vi
        # pos update
        pos[i] += vi * dt

    # ground clamp on vertical axis
    if pos[z_idx] < 0.0:
        pos[z_idx] = 0.0
        vel[z_idx] = 0.0
