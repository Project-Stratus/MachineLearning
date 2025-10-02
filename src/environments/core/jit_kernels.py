# environments/core/jit_kernels.py
from numba import njit
import numpy as np
import math

@njit(cache=True, fastmath=True)
def pressure_numba(p0: float, scale_h: float, alt: float) -> float:
    return p0 * np.exp(-alt / scale_h)

@njit(cache=True, fastmath=True)
def density_numba(p0: float, scale_h: float, T: float, M: float, R: float, alt: float) -> float:
    p = pressure_numba(p0, scale_h, alt)
    return p * M / (R * T)

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
                       G: float, CD: float, AREA: float,
                       rho_air: float, volume: float,
                       external_force: np.ndarray, control_force: np.ndarray,
                       n_dim: int) -> None:
    """
    In-place integration for 1D/2D/3D.
    Uses the last axis (index n_dim-1) as vertical for buoyancy/weight.
    No out-of-bounds indexing, ever.
    """
    # speed^2 across active dims
    speed2 = 0.0
    for i in range(n_dim):
        vi = vel[i]
        speed2 += vi * vi

    have_speed = speed2 > 1e-16
    inv_speed = 1.0 / math.sqrt(speed2) if have_speed else 0.0
    f_mag = 0.5 * CD * AREA * rho_air * speed2 if have_speed else 0.0

    z_idx = n_dim - 1
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
        if vi > 200.0:
            vi = 200.0
        elif vi < -200.0:
            vi = -200.0
        vel[i] = vi
        # pos update
        pos[i] += vi * dt

    # ground clamp on vertical axis
    if pos[z_idx] < 0.0:
        pos[z_idx] = 0.0
        vel[z_idx] = 0.0
