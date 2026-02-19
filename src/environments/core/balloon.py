import math
import numpy as np

from environments.core.atmosphere import Atmosphere
from environments.core.constants import (
    G, R, VOL_MAX, VOL_MIN, VEL_MAX, MASS, M_HE, T_BALLOON,
    ALT_DEFAULT, OSCILLATION_AMP, OSCILLATION_PERIOD, SPEED_EPS,
    MU_REF, T_REF, S_SUTH,
)

try:
    from environments.core.jit_kernels import (
        physics_step_numba, density_numba, sphere_area_from_volume, morrison_cd,
    )
    _JIT_OK = True
except Exception:
    _JIT_OK = False

_PI = math.pi
_FOUR_THIRDS_PI = 4.0 / 3.0 * _PI


class Balloon:
    """
    Unified balloon model for 1-D, 2-D and 3-D Gym environments.

    Gas tracking
    ------------
    Internal state tracks moles of helium (`self.n_gas`).  Volume is derived
    each step via the ideal gas law: V = n * R * T_balloon / P_ambient.
    As the balloon ascends, ambient pressure drops and volume grows
    automatically (passive expansion).  Agent inflate/deflate actions add or
    remove moles.

    Drag model
    ----------
    Frontal area is derived from volume (sphere assumption) and the drag
    coefficient uses the Morrison (2013) Reynolds-number correlation.
    """

    def __init__(
        self,
        dim: int = 1,
        mass: float = MASS,
        position=None,
        velocity=None,
        atmosphere: Atmosphere | None = None,
        oscillate: bool = False,
    ):
        self.dim = dim
        self.mass = mass
        self.atmosphere = atmosphere if atmosphere is not None else Atmosphere()

        # State ----------------------------------------------------------------
        if position is None:
            position = np.zeros(dim, dtype=float)
            position[-1] = ALT_DEFAULT
        if velocity is None:
            velocity = np.zeros(dim, dtype=float)

        self.pos = np.ascontiguousarray(position, dtype=np.float64)
        self.vel = np.ascontiguousarray(velocity, dtype=np.float64)
        self._zero_force = np.zeros(dim, dtype=np.float64)
        self.t = 0.0

        # Gas state ------------------------------------------------------------
        # Compute initial moles so volume equals neutral-buoyancy volume at
        # the starting altitude (mass / rho_air).
        alt = self.pos[-1]
        p_amb = self.atmosphere.pressure(alt)
        rho_air = self.atmosphere.density(alt)
        self.stationary_volume = self.mass / rho_air
        # n = P * V / (R * T_balloon)
        self.n_gas = p_amb * self.stationary_volume / (R * T_BALLOON)
        self.oscillate = oscillate

    # -- Volume from gas law --------------------------------------------------
    def _gas_law_volume(self) -> float:
        """Ideal gas law volume: V = n·R·T_balloon / P_ambient, clamped."""
        p_amb = self.atmosphere.pressure(self.pos[-1])
        vol = self.n_gas * R * T_BALLOON / p_amb
        return max(VOL_MIN, min(vol, VOL_MAX))

    # 1-D env (altitude) --------------------------------------------------
    @property
    def altitude(self):
        return self.pos[-1]

    @altitude.setter
    def altitude(self, z):
        self.pos[-1] = z

    @property
    def velocity(self):
        return self.vel[-1]

    @velocity.setter
    def velocity(self, vz):
        self.vel[-1] = vz

    # 2-D env (x–y) --------------------------------------------------------
    @property
    def x(self):
        return self.pos[0] if self.dim >= 1 else 0.0

    @x.setter
    def x(self, x_):
        self.pos[0] = x_

    @property
    def y(self):
        return self.pos[1] if self.dim >= 2 else 0.0

    @y.setter
    def y(self, y_):
        self.pos[1] = y_

    @property
    def vx(self):
        return self.vel[0] if self.dim >= 1 else 0.0

    @vx.setter
    def vx(self, vx_):
        self.vel[0] = vx_

    @property
    def vy(self):
        return self.vel[1] if self.dim >= 2 else 0.0

    @vy.setter
    def vy(self, vy_):
        self.vel[1] = vy_

    # Volume and buoyancy ---------------------------------------------------
    @property
    def volume(self):
        return self.dynamic_volume(self.t)

    @property
    def extra_volume(self):
        """Difference between current gas-law volume and stationary volume.

        Provided for backward compatibility with tests that inspect
        extra_volume after inflate/deflate calls.
        """
        return self._gas_law_volume() - self.stationary_volume

    def apply_volume_change(self, delta: float) -> None:
        """Add/remove gas to change volume by approximately *delta* m³.

        Converts the requested volume change to moles at the current ambient
        pressure so the effect is altitude-aware.
        """
        p_amb = self.atmosphere.pressure(self.pos[-1])
        # dn = P * dV / (R * T)
        dn = p_amb * delta / (R * T_BALLOON)
        self.n_gas += dn

    def inflate(self, delta: float) -> None:
        """Alias kept for env compatibility."""
        self.apply_volume_change(delta)

    @property
    def is_deflated(self) -> bool:
        """True if balloon has lost too much volume (helium released)."""
        return self.volume <= VOL_MIN

    # -------------------------------------------------------------------------
    # Core physics helpers
    # -------------------------------------------------------------------------
    def dynamic_volume(self, t: float) -> float:
        vol = self._gas_law_volume()
        if self.oscillate:
            amp = OSCILLATION_AMP * self.stationary_volume
            vol += amp * np.sin(2.0 * np.pi * t / OSCILLATION_PERIOD)
        return max(VOL_MIN, min(vol, VOL_MAX))

    def buoyant_force(self, t: float, rho_air: float | None = None) -> np.ndarray:
        if rho_air is None:
            rho_air = self.atmosphere.density(self.pos[-1])
        f = np.zeros(self.dim)
        f[-1] = rho_air * G * self.dynamic_volume(t)
        return f

    def weight(self) -> np.ndarray:
        f = np.zeros(self.dim)
        f[-1] = -self.mass * G
        return f

    def drag_force(self, rho_air: float | None = None) -> np.ndarray:
        """Drag with volume-dependent area and Reynolds-dependent CD."""
        speed = np.linalg.norm(self.vel)
        if speed < SPEED_EPS:
            return np.zeros(self.dim)
        if rho_air is None:
            rho_air = self.atmosphere.density(self.pos[-1])

        vol = self.dynamic_volume(self.t)
        area = _sphere_area(vol)
        diameter = 2.0 * (vol / _FOUR_THIRDS_PI) ** (1.0 / 3.0)

        # Reynolds number
        T = self.atmosphere.temperature(self.pos[-1])
        mu = MU_REF * (T / T_REF) ** 1.5 * (T_REF + S_SUTH) / (T + S_SUTH)
        Re = rho_air * speed * diameter / mu
        cd = _morrison_cd(Re)

        f_mag = 0.5 * cd * area * rho_air * speed**2
        return -f_mag * (self.vel / speed)

    def update(self, *args, external_force=None, control_force=None):
        if len(args) == 1:
            dt = float(args[0])
            t = self.t + dt
        elif len(args) >= 2:
            t, dt = map(float, args[:2])
            self.t = t
        else:
            raise TypeError("update() expects (dt) or (t, dt)")

        if external_force is None:
            external_force = self._zero_force
        elif not isinstance(external_force, np.ndarray) or external_force.dtype != np.float64:
            external_force = np.asarray(external_force, dtype=np.float64)
        if control_force is None:
            control_force = self._zero_force
        elif not isinstance(control_force, np.ndarray) or control_force.dtype != np.float64:
            control_force = np.asarray(control_force, dtype=np.float64)

        # Compute density once for this step
        z = self.pos[-1]
        if _JIT_OK:
            rho_air = float(density_numba(self.atmosphere.p0, self.atmosphere.molar_mass, z))
        else:
            rho_air = self.atmosphere.density(z)
        vol = self.dynamic_volume(t)

        if _JIT_OK:
            physics_step_numba(
                self.pos, self.vel, dt, self.mass, G,
                rho_air, vol,
                external_force, control_force, int(self.dim), VEL_MAX,
            )
        else:
            # Fallback: pure-Python path
            f_net = (self.buoyant_force(t, rho_air) + self.weight() +
                     self.drag_force(rho_air) + external_force + control_force)
            acc = f_net / self.mass
            self.vel += acc * dt
            self.vel = np.clip(self.vel, -VEL_MAX, VEL_MAX)
            self.pos += self.vel * dt
            if self.pos[-1] < 0.0:
                self.pos[-1] = 0.0
                self.vel[-1] = 0.0

        self.t = t


# ---- Module-level helpers (pure Python, used by Balloon.drag_force) ---------

def _sphere_area(volume: float) -> float:
    r = (volume / _FOUR_THIRDS_PI) ** (1.0 / 3.0)
    return _PI * r * r


def _morrison_cd(Re: float) -> float:
    if Re < 1e-8:
        return 0.0
    term1 = 24.0 / Re
    term2 = 2.6 * (Re / 5.0) / (1.0 + (Re / 5.0) ** 1.52)
    term3 = 0.411 * (Re / 263000.0) ** (-7.94) / (1.0 + (Re / 263000.0) ** (-8.0))
    term4 = Re ** 0.80 / 461000.0
    return term1 + term2 + term3 + term4
