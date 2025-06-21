"""
Version by AS
Attempt to combine 1D, 2D and 3D balloon environments into a single file.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from environments.envs.atmosphere import Atmosphere

G = 9.81
R = 8.314462618
T_BALLOON = 273.15 + 20
T_AIR = 273.15 + 15
M_AIR = 0.0289647
P0 = 101325
SCALE_HEIGHT = 8500

DT = 1.0
EPISODE_LENGTH = 300
CD = 0.5
AREA = 1.0


# class Atmosphere:
#     """Simple exponential atmosphere model."""
#     def __init__(self, p0=P0, scale_height=SCALE_HEIGHT, temperature=T_AIR, molar_mass=M_AIR):
#         self.p0 = p0
#         self.scale_height = scale_height
#         self.temperature = temperature
#         self.molar_mass = molar_mass

#     def pressure(self, altitude: float) -> float:
#         return self.p0 * np.exp(-altitude / self.scale_height)

#     def density(self, altitude: float) -> float:
#         p = self.pressure(altitude)
#         return p * self.molar_mass / (R * self.temperature)


class Balloon:
    """General balloon model supporting 1D, 2D and 3D motion."""
    def __init__(self, dim: int = 1, mass: float = 2.0, position=None, velocity=None, atmosphere: Atmosphere | None = None, oscillate: bool = False):
        self.dim = dim
        self.mass = mass
        self.atmosphere = atmosphere if atmosphere is not None else Atmosphere()
        if position is None:
            position = np.zeros(dim)
            position[-1] = 25000.0
        if velocity is None:
            velocity = np.zeros(dim)
        self.pos = np.array(position, dtype=float)
        self.vel = np.array(velocity, dtype=float)
        rho_air = self.atmosphere.density(self.pos[-1])
        self.stationary_volume = self.mass / rho_air
        self.extra_volume = 0.0
        self.oscillate = oscillate

    def dynamic_volume(self, t: float) -> float:
        vol = self.stationary_volume + self.extra_volume
        if self.oscillate:
            amplitude_fraction = 0.05
            amplitude = amplitude_fraction * self.stationary_volume
            phase = 2.0 * np.pi * (t / 60.0)
            vol += amplitude * np.sin(phase)
        return vol

    def apply_volume_change(self, delta: float) -> None:
        self.extra_volume += delta

    def buoyant_force(self, t: float) -> np.ndarray:
        rho_air = self.atmosphere.density(self.pos[-1])
        vol = self.dynamic_volume(t)
        force = np.zeros(self.dim)
        force[-1] = rho_air * G * vol
        return force

    def weight(self) -> np.ndarray:
        force = np.zeros(self.dim)
        force[-1] = -self.mass * G
        return force

    def drag_force(self) -> np.ndarray:
        speed = np.linalg.norm(self.vel)
        if speed < 1e-8:
            return np.zeros(self.dim)
        rho_air = self.atmosphere.density(self.pos[-1])
        f_mag = 0.5 * CD * AREA * rho_air * speed ** 2
        return -f_mag * (self.vel / speed)

    def update(self, t: float, dt: float, external_force=None, control_force=None) -> None:
        if external_force is None:
            external_force = np.zeros(self.dim)
        if control_force is None:
            control_force = np.zeros(self.dim)
        f_net = self.buoyant_force(t) + self.weight() + self.drag_force() + external_force + control_force
        acc = f_net / self.mass
        self.vel += acc * dt
        max_velocity = 200.0
        self.vel = np.clip(self.vel, -max_velocity, max_velocity)
        self.pos += self.vel * dt
        if self.pos[-1] < 0:
            self.pos[-1] = 0
            self.vel[-1] = 0


class BalloonEnvBase(gym.Env):
    """Base environment shared by 1D, 2D and 3D variants."""
    metadata = {"render.modes": ["human"]}

    def __init__(self, dim: int, target=None):
        super().__init__()
        self.dim = dim
        self.target = np.zeros(dim) if target is None else np.array(target, dtype=float)
        self.balloon = Balloon(dim=dim, oscillate=dim > 1)
        self.time = 0.0
        self.step_count = 0

    # --- Methods subclasses should override ---
    def interpret_action(self, action):
        raise NotImplementedError

    def external_force(self) -> np.ndarray:
        return np.zeros(self.dim)

    def get_obs(self):
        raise NotImplementedError

    def compute_reward(self):
        raise NotImplementedError

    # --- Gym API ---
    def step(self, action):
        control_force, vol_change = self.interpret_action(action)
        if vol_change != 0:
            self.balloon.apply_volume_change(vol_change)
        self.balloon.update(self.time, DT, external_force=self.external_force(), control_force=control_force)
        self.time += DT
        self.step_count += 1
        obs = self.get_obs()
        reward = self.compute_reward()
        done = self.step_count >= EPISODE_LENGTH
        info = {}
        return obs, reward, done, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.balloon = Balloon(dim=self.dim, oscillate=self.dim > 1)
        self.time = 0.0
        self.step_count = 0
        return self.get_obs(), {}


# --- 1D Environment -------------------------------------------------------
class Balloon1DEnv(BalloonEnvBase):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    TIME_MAX = 400
    PUNISHMENT = -400

    def __init__(self, render_mode=None):
        super().__init__(dim=1)
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.max_volume = 20.0
        self.max_altitude = 20000.0
        self.max_velocity = 50.0
        self.max_pressure = 100000
        self.target = np.array([self.np_random.uniform(low=0, high=self.max_altitude)])
        self.observation_space = spaces.Dict({
            "target": spaces.Box(low=0, high=1.0, shape=(1,), dtype=float),
            "volume": spaces.Box(low=0, high=1.0, shape=(1,), dtype=float),
            "altitude": spaces.Box(low=0, high=1.0, shape=(1,), dtype=float),
            "velocity": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=float),
            "pressure": spaces.Box(low=0, high=1.0, shape=(1,), dtype=float),
            "wind": spaces.Box(low=-20.0, high=20.0, shape=(3,), dtype=float),
        })
        self.action_space = spaces.Discrete(3)
        self.truncated = False
        self.final_obs = None

    def interpret_action(self, action):
        if action == 0:  # inflate
            return np.zeros(1), 0.02
        if action == 1:  # deflate
            return np.zeros(1), -0.02
        return np.zeros(1), 0.0

    def external_force(self) -> np.ndarray:
        x_idx = GRID_CELLS // 2
        y_idx = np.searchsorted(y_edges, self.balloon.pos[-1]) - 1
        y_idx = np.clip(y_idx, 0, GRID_CELLS - 1)
        fy = fy_grid[y_idx, x_idx]
        return np.array([fy])

    def get_local_wind(self):
        x_idx = GRID_CELLS // 2
        y_idx = np.searchsorted(y_edges, self.balloon.pos[-1]) - 1
        y_idx = np.clip(y_idx, 0, GRID_CELLS - 1)
        values = []
        for j in range(y_idx - 1, y_idx + 2):
            j_c = np.clip(j, 0, GRID_CELLS - 1)
            values.append(fy_grid[j_c, x_idx])
        return np.array(values, dtype=float)

    def get_obs(self):
        return {
            "target": np.array([self.target[0] / self.max_altitude]),
            "volume": np.array([(self.balloon.extra_volume + self.balloon.stationary_volume) / self.max_volume]),
            "altitude": np.array([self.balloon.pos[-1] / self.max_altitude]),
            "velocity": np.array([self.balloon.vel[-1] / self.max_velocity]),
            "pressure": np.array([self.balloon.atmosphere.pressure(self.balloon.pos[-1]) / self.max_pressure]),
            "wind": self.get_local_wind(),
        }

    def compute_reward(self):
        goal_dist = (1 - abs(self.balloon.pos[-1] - self.target[0]) / self.max_altitude) ** 8
        return goal_dist if self.balloon.pos[-1] > 0 else self.PUNISHMENT

    def step(self, action):
        obs, reward, done, info = super().step(action)
        terminated = self.balloon.pos[-1] <= 0
        truncated = self.step_count >= self.TIME_MAX
        if truncated and not terminated:
            self.truncated = True
        if terminated or truncated:
            self.final_obs = obs
        if self.render_mode == "human":
            self._render_frame()
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        self.target = np.array([self.np_random.uniform(low=0, high=self.max_altitude)])
        if self.render_mode == "human":
            self._render_frame()
        return observation, info

    # Rendering code borrowed from original implementation
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        import pygame
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((512, 512))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        canvas = pygame.Surface((512, 512))
        canvas.fill((255, 255, 255))
        pygame.draw.circle(canvas, (255, 0, 0), (256, 512 - (self.balloon.pos[-1] / self.max_altitude) * 512), 20)
        pygame.draw.circle(canvas, (0, 255, 0), (256, 512 - (self.target[0] / self.max_altitude) * 512), 20)
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        import pygame
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


# --- 2D Environment -------------------------------------------------------
GRID_CELLS = 40
X_RANGE = (-2000, 2000)
Y_RANGE = (20000, 30000)
FORCE_MAG = 10.0
x_edges = np.linspace(X_RANGE[0], X_RANGE[1], GRID_CELLS + 1)
y_edges = np.linspace(Y_RANGE[0], Y_RANGE[1], GRID_CELLS + 1)
x_centers = (x_edges[:-1] + x_edges[1:]) / 2
y_centers = (y_edges[:-1] + y_edges[1:]) / 2
x_forces, y_forces = np.meshgrid(x_centers, y_centers)
fx_grid = (FORCE_MAG / 2) * (np.sin(2 * np.pi * x_forces / (X_RANGE[1] - X_RANGE[0])) + 0.5 * np.sin(4 * np.pi * x_forces / (X_RANGE[1] - X_RANGE[0])))
fy_grid = (FORCE_MAG / 2) * (np.cos(2 * np.pi * y_forces / (Y_RANGE[1] - Y_RANGE[0])) + 0.5 * np.cos(4 * np.pi * y_forces / (Y_RANGE[1] - Y_RANGE[0])))


class Balloon2DEnv(BalloonEnvBase):
    """2D balloon environment using the shared base class."""
    def __init__(self):
        super().__init__(dim=2, target=(0.0, 25000.0))
        low_obs = np.concatenate((np.array([X_RANGE[0], Y_RANGE[0], -200.0, -200.0]), np.full(18, -20.0)))
        high_obs = np.concatenate((np.array([X_RANGE[1], Y_RANGE[1], 200.0, 200.0]), np.full(18, 20.0)))
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)
        self.action_space = spaces.Discrete(9)
        self.control_force_mag_x = 500.0
        self.control_force_mag_y = 500.0
        self.fig = None
        self.ax = None
        self.circle = None
        self.target_marker = None

    def interpret_action(self, action):
        ax = (action % 3) - 1
        ay = (action // 3) - 1
        control_x = ax * self.control_force_mag_x
        control_y = ay * self.control_force_mag_y
        return np.array([control_x, control_y]), 0.0

    def external_force(self) -> np.ndarray:
        x_idx = np.searchsorted(x_edges, self.balloon.pos[0]) - 1
        y_idx = np.searchsorted(y_edges, self.balloon.pos[1]) - 1
        x_idx = np.clip(x_idx, 0, GRID_CELLS - 1)
        y_idx = np.clip(y_idx, 0, GRID_CELLS - 1)
        fx = fx_grid[y_idx, x_idx]
        fy = fy_grid[y_idx, x_idx]
        return np.array([fx, fy])

    def get_local_wind(self):
        x_idx = np.searchsorted(x_edges, self.balloon.pos[0]) - 1
        y_idx = np.searchsorted(y_edges, self.balloon.pos[1]) - 1
        x_idx = np.clip(x_idx, 0, GRID_CELLS - 1)
        y_idx = np.clip(y_idx, 0, GRID_CELLS - 1)
        wind_values = []
        for j in range(y_idx - 1, y_idx + 2):
            for i in range(x_idx - 1, x_idx + 2):
                i_c = np.clip(i, 0, GRID_CELLS - 1)
                j_c = np.clip(j, 0, GRID_CELLS - 1)
                wind_values.append(fx_grid[j_c, i_c])
                wind_values.append(fy_grid[j_c, i_c])
        return np.array(wind_values, dtype=np.float32)

    def get_obs(self):
        local_wind = self.get_local_wind()
        return np.concatenate((np.array([self.balloon.pos[0], self.balloon.pos[1], self.balloon.vel[0], self.balloon.vel[1]]), local_wind))

    def compute_reward(self):
        dx = self.balloon.pos[0] - self.target[0]
        dy = self.balloon.pos[1] - self.target[1]
        return -np.sqrt(dx * dx + dy * dy)

    def render(self, mode="human"):
        import matplotlib.pyplot as plt
        if self.fig is None:
            self.fig, self.ax = plt.subplots()
            self.ax.set_title("Balloon Environment")
            self.ax.set_xlabel("X position (m)")
            self.ax.set_ylabel("Altitude (m)")
            self.ax.set_xlim(X_RANGE)
            self.ax.set_ylim(Y_RANGE)
            self.circle = plt.Circle((self.balloon.pos[0], self.balloon.pos[1]), 50, color='red')
            self.ax.add_patch(self.circle)
            self.target_marker, = self.ax.plot(self.target[0], self.target[1], marker='*', color='green', markersize=15)
        self.circle.center = (self.balloon.pos[0], self.balloon.pos[1])
        plt.pause(0.001)

    def close(self):
        import matplotlib.pyplot as plt
        if self.fig:
            plt.close(self.fig)


# --- Placeholder for 3D Environment --------------------------------------
class Balloon3DEnv(BalloonEnvBase):
    """Skeleton for a future 3D balloon environment."""
    def __init__(self):
        super().__init__(dim=3, target=(0.0, 0.0, 25000.0))
        # Observation and action spaces would be defined similar to the 2D case
        self.action_space = spaces.Discrete(27)  # Example: 3x3x3 force grid
        obs_low = np.array([-np.inf, -np.inf, 0, -np.inf, -np.inf, -np.inf])
        obs_high = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

    def interpret_action(self, action):
        # Map action to 3D control force (placeholder)
        ax = (action % 3) - 1
        ay = ((action // 3) % 3) - 1
        az = (action // 9) - 1
        mag = 500.0
        return np.array([ax * mag, ay * mag, az * mag]), 0.0

    def get_obs(self):
        # Placeholder observation consisting of position and velocity
        return np.concatenate((self.balloon.pos, self.balloon.vel))

    def compute_reward(self):
        return -np.linalg.norm(self.balloon.pos - self.target)

