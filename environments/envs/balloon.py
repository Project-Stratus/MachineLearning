import numpy as np

from environments.envs.atmosphere import Atmosphere
from environments.constants import G, CD, AREA


# class Balloon:
#     """General balloon model supporting 1D, 2D and 3D motion."""
#     def __init__(self, dim: int = 1, mass: float = 2.0, position=None, velocity=None, atmosphere: Atmosphere | None = None, oscillate: bool = False):
#         self.dim = dim
#         self.mass = mass
#         self.atmosphere = atmosphere if atmosphere is not None else Atmosphere()
#         if position is None:
#             position = np.zeros(dim)
#             position[-1] = 25000.0
#         if velocity is None:
#             velocity = np.zeros(dim)
#         self.pos = np.array(position, dtype=float)
#         self.vel = np.array(velocity, dtype=float)
#         rho_air = self.atmosphere.density(self.pos[-1])
#         self.stationary_volume = self.mass / rho_air
#         self.extra_volume = 0.0
#         self.oscillate = oscillate

#     def dynamic_volume(self, t: float) -> float:
#         vol = self.stationary_volume + self.extra_volume
#         if self.oscillate:
#             amplitude_fraction = 0.05
#             amplitude = amplitude_fraction * self.stationary_volume
#             phase = 2.0 * np.pi * (t / 60.0)
#             vol += amplitude * np.sin(phase)
#         return vol

#     def apply_volume_change(self, delta: float) -> None:
#         self.extra_volume += delta

#     def buoyant_force(self, t: float) -> np.ndarray:
#         rho_air = self.atmosphere.density(self.pos[-1])
#         vol = self.dynamic_volume(t)
#         force = np.zeros(self.dim)
#         force[-1] = rho_air * G * vol
#         return force

#     def weight(self) -> np.ndarray:
#         force = np.zeros(self.dim)
#         force[-1] = -self.mass * G
#         return force

#     def drag_force(self) -> np.ndarray:
#         speed = np.linalg.norm(self.vel)
#         if speed < 1e-8:
#             return np.zeros(self.dim)
#         rho_air = self.atmosphere.density(self.pos[-1])
#         f_mag = 0.5 * CD * AREA * rho_air * speed ** 2
#         return -f_mag * (self.vel / speed)

#     def update(self, t: float, dt: float, external_force=None, control_force=None) -> None:
#         if external_force is None:
#             external_force = np.zeros(self.dim)
#         if control_force is None:
#             control_force = np.zeros(self.dim)
#         f_net = self.buoyant_force(t) + self.weight() + self.drag_force() + external_force + control_force
#         acc = f_net / self.mass
#         self.vel += acc * dt
#         max_velocity = 200.0
#         self.vel = np.clip(self.vel, -max_velocity, max_velocity)
#         self.pos += self.vel * dt
#         if self.pos[-1] < 0:
#             self.pos[-1] = 0
#             self.vel[-1] = 0


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
        mass: float = 2.0,
        position=None,
        velocity=None,
        atmosphere: Atmosphere | None = None,
        oscillate: bool = False,
    ):
        assert dim in (1, 2), (
            f"Only 1-D or 2-D motion is supported by the envs. Got dim={dim}."
        )
        self.dim = dim
        self.mass = mass
        self.atmosphere = atmosphere if atmosphere is not None else Atmosphere()

        # State ----------------------------------------------------------------
        if position is None:
            position = np.zeros(dim, dtype=float)
            position[-1] = 25_000.0  # start high in the air
        if velocity is None:
            velocity = np.zeros(dim, dtype=float)

        self.pos = np.asarray(position, dtype=float)
        self.vel = np.asarray(velocity, dtype=float)
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

    # -------------------------------------------------------------------------
    # Core physics helpers (unchanged from your original)
    # -------------------------------------------------------------------------
    def dynamic_volume(self, t: float) -> float:
        vol = self.stationary_volume + self.extra_volume
        if self.oscillate:
            amp = 0.05 * self.stationary_volume
            vol += amp * np.sin(2.0 * np.pi * t / 60.0)
        return vol

    def buoyant_force(self, t: float) -> np.ndarray:
        rho_air = self.atmosphere.density(self.pos[-1])
        f = np.zeros(self.dim)
        f[-1] = rho_air * G * self.dynamic_volume(t)
        return f

    def weight(self) -> np.ndarray:
        f = np.zeros(self.dim)
        f[-1] = -self.mass * G
        return f

    def drag_force(self) -> np.ndarray:
        speed = np.linalg.norm(self.vel)
        if speed < 1e-8:
            return np.zeros(self.dim)
        rho_air = self.atmosphere.density(self.pos[-1])
        f_mag = 0.5 * CD * AREA * rho_air * speed**2
        return -f_mag * (self.vel / speed)

    # -------------------------------------------------------------------------
    # UNIVERSAL update – works for *both* env calling conventions
    # -------------------------------------------------------------------------
    def update(self, *args, external_force=None, control_force=None):
        """
        Two call signatures accepted:

            • update(dt)                            ← 1-D env
            • update(t, dt, external_force=…,       ← 2-D env
                     control_force=…)

        The extra keyword arguments are optional in either case.
        """
        # Decode which variant we have been given ------------------------------
        if len(args) == 1:
            # Old 1-D env style: only Δt
            dt = float(args[0])
            t = self.t + dt  # advance internal clock
        elif len(args) >= 2:
            # New 2-D env style: explicit (t, dt)
            t, dt = map(float, args[:2])
            self.t = t  # sync internal clock
        else:
            raise TypeError("update() expects (dt) or (t, dt)")

        # Default forces --------------------------------------------------------
        if external_force is None:
            external_force = np.zeros(self.dim)
        if control_force is None:
            control_force = np.zeros(self.dim)

        # Net force -------------------------------------------------------------
        f_net = (
            self.buoyant_force(t)
            + self.weight()
            + self.drag_force()
            + np.asarray(external_force)
            + np.asarray(control_force)
        )
        acc = f_net / self.mass

        # Integrate -------------------------------------------------------------
        self.vel += acc * dt
        self.vel = np.clip(self.vel, -200.0, 200.0)
        self.pos += self.vel * dt

        # Keep above ground -----------------------------------------------------
        if self.pos[-1] < 0.0:
            self.pos[-1] = 0.0
            self.vel[-1] = 0.0

        self.t = t  # ensure time is consistent
