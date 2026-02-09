import numpy as np

from environments.core.atmosphere import Atmosphere
from environments.core.constants import (
    G, CD, AREA, R, VOL_MAX, VOL_MIN, VEL_MAX, MASS,
    ALT_DEFAULT, OSCILLATION_AMP, OSCILLATION_PERIOD, SPEED_EPS,
)

try:
    from environments.core.jit_kernels import physics_step_numba, density_numba
    _JIT_OK = True
except Exception:
    _JIT_OK = False


class Balloon:
    """
    Unified balloon model that works in 1-D (altitude only) and 2-D (xy) Gym
    environments

    • Positional state is kept in self.pos  (shape = (dim,))
    • Velocity       ″            self.vel  (shape = (dim,))
    • Internal clock self.t is advanced automatically when using the 1-arg
      update(dt) call that the 1-D env makes.
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
        self._zero_force = np.zeros(dim, dtype=np.float64)  # reusable zero vector
        self.t = 0.0  # internal time (s)

        # Buoyancy --------------------------------------------------------------
        rho_air = self.atmosphere.density(self.pos[-1])
        self.stationary_volume = self.mass / rho_air
        self.extra_volume = 0.0  # volume added by “inflate”
        self.oscillate = oscillate  # sinusoidal breathing

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

    # 2-D env (x–y) ------------------------------------------------------------
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

    def apply_volume_change(self, delta: float) -> None:
        self.extra_volume += delta

    # -------------------------------------------------------------------------
    # Public helper expected by the environments
    # -------------------------------------------------------------------------
    def inflate(self, delta: float) -> None:
        """Alias kept for 1-D env compatibility."""
        self.apply_volume_change(delta)

    @property
    def is_deflated(self) -> bool:
        """True if balloon has lost too much volume (helium released)."""
        return self.volume <= VOL_MIN

    # -------------------------------------------------------------------------
    # Core physics helpers
    # -------------------------------------------------------------------------
    def dynamic_volume(self, t: float) -> float:
        vol = self.stationary_volume + self.extra_volume
        if self.oscillate:
            amp = OSCILLATION_AMP * self.stationary_volume
            vol += amp * np.sin(2.0 * np.pi * t / OSCILLATION_PERIOD)
        # Clamp to physical bounds: cannot go below minimum or above maximum
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
        speed = np.linalg.norm(self.vel)
        if speed < SPEED_EPS:
            return np.zeros(self.dim)
        if rho_air is None:
            rho_air = self.atmosphere.density(self.pos[-1])
        f_mag = 0.5 * CD * AREA * rho_air * speed**2
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

        # Compute density once for this step (used by JIT and fallback paths)
        z = self.pos[-1]
        if _JIT_OK:
            rho_air = float(density_numba(self.atmosphere.p0, self.atmosphere.scale_height, self.atmosphere.temperature, self.atmosphere.molar_mass, R, z))
        else:
            rho_air = self.atmosphere.density(z)
        vol = self.dynamic_volume(t)

        if _JIT_OK:
            physics_step_numba(self.pos, self.vel, dt, self.mass, G, CD, AREA, rho_air, vol, external_force, control_force, int(self.dim), VEL_MAX)
        else:
            # Fallback: pure-Python path using cached rho_air
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
