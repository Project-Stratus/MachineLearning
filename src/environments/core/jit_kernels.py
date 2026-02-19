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
def _compute_drag(vel: np.ndarray, wind_vel: np.ndarray,
                  rho_air: float, area: float, diameter: float,
                  alt: float, n_dim: int) -> (float, float, float):
    """
    Compute drag magnitude and relative-velocity direction.

    Drag is proportional to |v_rel|^2 where v_rel = v_balloon - v_wind,
    directed opposite to v_rel.

    Returns (f_mag, rel_speed, inv_rel_speed).
    """
    rel_speed2 = 0.0
    for i in range(n_dim):
        dv = vel[i] - wind_vel[i]
        rel_speed2 += dv * dv

    have_rel = rel_speed2 > 1e-24
    if not have_rel:
        return 0.0, 0.0, 0.0

    rel_speed = math.sqrt(rel_speed2)
    inv_rel_speed = 1.0 / rel_speed

    mu = dynamic_viscosity_numba(alt)
    Re = rho_air * rel_speed * diameter / mu
    cd = morrison_cd(Re)
    f_mag = 0.5 * cd * area * rho_air * rel_speed2

    return f_mag, rel_speed, inv_rel_speed


@njit(cache=True, fastmath=True)
def _compute_accel(pos: np.ndarray, vel: np.ndarray,
                   wind_vel: np.ndarray, external_force: np.ndarray,
                   mass: float, G: float,
                   rho_air: float, volume: float, area: float, diameter: float,
                   n_dim: int,
                   accel_out: np.ndarray) -> None:
    """Compute per-axis acceleration into *accel_out* (pre-allocated)."""
    z_idx = n_dim - 1
    alt = pos[z_idx]

    f_mag, rel_speed, inv_rel_speed = _compute_drag(
        vel, wind_vel, rho_air, area, diameter, alt, n_dim)
    have_rel = rel_speed > 1e-12

    buoy_z = rho_air * G * volume - mass * G

    for i in range(n_dim):
        # Drag opposes relative velocity (v_balloon - v_wind)
        if have_rel:
            drag_i = -f_mag * (vel[i] - wind_vel[i]) * inv_rel_speed
        else:
            drag_i = 0.0
        f_i = external_force[i] + drag_i
        if i == z_idx:
            f_i += buoy_z
        accel_out[i] = f_i / mass


@njit(cache=True, fastmath=True)
def physics_step_numba(pos: np.ndarray, vel: np.ndarray,
                       dt: float,
                       mass: float,
                       G: float,
                       rho_air: float, volume: float,
                       wind_vel: np.ndarray, external_force: np.ndarray,
                       n_dim: int, vel_max: float,
                       p0: float, M_air: float) -> None:
    """
    Velocity-Verlet integration for 1D/2D/3D.

    Drag uses relative velocity (v_balloon - v_wind) with volume-dependent
    area and Morrison CD correlation.

    Algorithm:
      1. a_old = F(pos, vel) / m
      2. pos += vel * dt + 0.5 * a_old * dt^2
      3. Recompute rho at new altitude
      4. a_new = F(pos_new, vel) / m
      5. vel += 0.5 * (a_old + a_new) * dt
    """
    z_idx = n_dim - 1

    # Geometry from current volume
    area = sphere_area_from_volume(volume)
    r = (volume / _FOUR_THIRDS_PI) ** (1.0 / 3.0)
    diameter = 2.0 * r

    # --- Step 1: acceleration at current state ---
    a_old = np.empty(n_dim, dtype=np.float64)
    _compute_accel(pos, vel, wind_vel, external_force,
                   mass, G, rho_air, volume, area, diameter, n_dim, a_old)

    # --- Step 2: update position ---
    half_dt2 = 0.5 * dt * dt
    for i in range(n_dim):
        pos[i] += vel[i] * dt + a_old[i] * half_dt2

    # --- Step 3: recompute density at new altitude ---
    rho_new = density_numba(p0, M_air, pos[z_idx])

    # --- Step 4: acceleration at new position (with old velocity) ---
    a_new = np.empty(n_dim, dtype=np.float64)
    _compute_accel(pos, vel, wind_vel, external_force,
                   mass, G, rho_new, volume, area, diameter, n_dim, a_new)

    # --- Step 5: update velocity ---
    half_dt = 0.5 * dt
    for i in range(n_dim):
        vi = vel[i] + (a_old[i] + a_new[i]) * half_dt
        if vi > vel_max:
            vi = vel_max
        elif vi < -vel_max:
            vi = -vel_max
        vel[i] = vi

    # Ground clamp
    if pos[z_idx] < 0.0:
        pos[z_idx] = 0.0
        vel[z_idx] = 0.0
