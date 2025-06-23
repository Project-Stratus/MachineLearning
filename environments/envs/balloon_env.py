"""
Version by TS
2D loon implementation only
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

from environments.envs.balloon import Balloon
from environments.constants import DT

# -------------------------------
# GLOBAL CONSTANTS
# -------------------------------
EPISODE_LENGTH = 300  # max steps per episode

# Force grid parameters (for the wind field)
GRID_CELLS = 40
X_RANGE = (-2000, 2000)
Y_RANGE = (20000, 30000)
FORCE_MAG = 10.0

# Define grid edges and cell centers for the wind field.
x_edges = np.linspace(X_RANGE[0], X_RANGE[1], GRID_CELLS + 1)
y_edges = np.linspace(Y_RANGE[0], Y_RANGE[1], GRID_CELLS + 1)
x_centers = (x_edges[:-1] + x_edges[1:]) / 2
y_centers = (y_edges[:-1] + y_edges[1:]) / 2
x_forces, y_forces = np.meshgrid(x_centers, y_centers)

# Create a complex yet gentle wind field.
fx_grid = (FORCE_MAG / 2) * (
    np.sin(2 * np.pi * x_forces / (X_RANGE[1] - X_RANGE[0]))
    + 0.5 * np.sin(4 * np.pi * x_forces / (X_RANGE[1] - X_RANGE[0]))
)
fy_grid = (FORCE_MAG / 2) * (
    np.cos(2 * np.pi * y_forces / (Y_RANGE[1] - Y_RANGE[0]))
    + 0.5 * np.cos(4 * np.pi * y_forces / (Y_RANGE[1] - Y_RANGE[0]))
)


# -------------------------------
# CUSTOM GYM ENVIRONMENT
# -------------------------------
class Balloon2DEnv(gym.Env):
    """
    A Gym environment for 2D balloon control.

    - **Observation (22 dimensions):**
          [x, y, vx, vy, local wind info (9 cells × 2 values)]
      where the local wind info is extracted from a 3x3 grid (the current cell plus its 8 neighbors).
    - **Action (Discrete 9):**
      Each action corresponds to a 2D control force:
          horizontal force = (action mod 3 – 1) * control_force_mag_x
          vertical force   = (action // 3 – 1) * control_force_mag_y
    - **Reward:**
          Negative Euclidean distance from the balloon's position to the target (0, 25000).
    """

    # metadata = {"render.modes": ["human"]}
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super(Balloon2DEnv, self).__init__()
        # Observation: [x, y, vx, vy, 18 wind components] → 22 values.
        low_obs = np.concatenate(
            (np.array([X_RANGE[0], Y_RANGE[0], -200.0, -200.0], dtype=np.float32), np.full(18, -20.0, dtype=np.float32))    # gym.spaces.Box requires float32 dtype
        )
        high_obs = np.concatenate(
            (np.array([X_RANGE[1], Y_RANGE[1], 200.0, 200.0], dtype=np.float32), np.full(18, 20.0, dtype=np.float32))       # gym.spaces.Box requires float32 dtype
        )
        self.observation_space = spaces.Box(
            low=low_obs, high=high_obs, dtype=np.float32
        )
        # Action space: 9 discrete actions (3×3 grid).
        self.action_space = spaces.Discrete(9)

        # Larger control forces allowed.
        self.control_force_mag_x = 500.0
        self.control_force_mag_y = 500.0
        # New target position.
        self.target_x = 0.0
        self.target_y = 25000.0
        self.balloon = Balloon()
        self.time = 0.0
        self.step_count = 0

        # For rendering.
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def get_local_wind(self):
        """
        Returns a flattened array of the wind forces (fx, fy) from the 3x3 grid of cells
        surrounding the current cell where the balloon is located.
        """
        x_idx = np.searchsorted(x_edges, self.balloon.x) - 1
        y_idx = np.searchsorted(y_edges, self.balloon.y) - 1
        x_idx = np.clip(x_idx, 0, GRID_CELLS - 1)
        y_idx = np.clip(y_idx, 0, GRID_CELLS - 1)
        wind_values = []
        for j in range(y_idx - 1, y_idx + 2):
            for i in range(x_idx - 1, x_idx + 2):
                i_clipped = np.clip(i, 0, GRID_CELLS - 1)
                j_clipped = np.clip(j, 0, GRID_CELLS - 1)
                wind_values.append(fx_grid[j_clipped, i_clipped])
                wind_values.append(fy_grid[j_clipped, i_clipped])
        return np.array(wind_values, dtype=np.float32)

    def step(self, action):
        # Map the discrete action (0–8) to a 2D control force.
        action_x = (action % 3) - 1  # yields -1, 0, or 1.
        action_y = (action // 3) - 1  # yields -1, 0, or 1.
        control_x = action_x * self.control_force_mag_x
        control_y = action_y * self.control_force_mag_y
        self.balloon.update(self.time, DT, control_force=(control_x, control_y))
        self.time += DT
        self.step_count += 1

        local_wind = self.get_local_wind()
        obs = np.concatenate(
            (
                np.array(
                    [self.balloon.x, self.balloon.y, self.balloon.vx, self.balloon.vy]
                ),
                local_wind,
            )
        )
        dx = self.balloon.x - self.target_x
        dy = self.balloon.y - self.target_y
        reward = -np.sqrt(dx * dx + dy * dy)
        # done = self.step_count >= EPISODE_LENGTH
        terminated = False
        truncated = self.step_count >= EPISODE_LENGTH
        info = {}
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # ---- 1. sample a random start position ------------------------
        while True:
            x0 = self.np_random.uniform(*X_RANGE)
            y0 = self.np_random.uniform(*Y_RANGE)
            if np.hypot(x0 - self.target_x, y0 - self.target_y) > 500.0:
                break

        self.balloon = Balloon(dim=2, position=[x0, y0], velocity=[0.0, 0.0])
        self.time = 0.0
        self.step_count = 0
        local_wind = self.get_local_wind()
        info = {"seed": seed}
        return np.concatenate(
            (
                np.array(
                    [self.balloon.x, self.balloon.y, self.balloon.vx, self.balloon.vy]
                ),
                local_wind,
            )
        ), info

    def render(self, mode="human"):
        if mode not in self.metadata["render_modes"]:
            raise ValueError(f"Invalid render mode: {mode}. ")
        self.render_mode = mode
        return self._render_frame()

    def _render_frame(self):
        # --------- 1. Lazy window / clock creation ----------
        if self.window is None and self.render_mode == "human":
            import pygame
            pygame.init()
            pygame.display.init()
            # choose a square window or keep aspect ratio; here 600×600
            self.window_size = 600
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        import pygame
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))  # white background

        # --------- 2. Coordinate transform helpers ----------
        def to_screen(x_m, y_m):
            """Convert world metres to screen pixels (origin top-left)."""
            sx = (x_m - X_RANGE[0]) / (X_RANGE[1] - X_RANGE[0])
            sy = (y_m - Y_RANGE[0]) / (Y_RANGE[1] - Y_RANGE[0])
            # flip y because Pygame's y grows downward
            return (int(sx * self.window_size),
                    int((1.0 - sy) * self.window_size))

        # --------- 3. Draw balloon  -------------------------
        bx, by = to_screen(self.balloon.x, self.balloon.y)
        pygame.draw.circle(canvas, (255, 0, 0), (bx, by), 10)

        # --------- 4. Draw target ---------------------------
        tx, ty = to_screen(self.target_x, self.target_y)
        pygame.draw.circle(canvas, (0, 255, 0), (tx, ty), 5)

        # --------- 4.5  Draw wind arrows -----------------------------
        # Convert each grid centre (x_forces, y_forces) to screen space,
        # scale the wind vector for visibility, and draw an arrow.
        ARROW_COLOR = (0, 128, 255)    # light blue
        ARROW_SCALE = 0.1              # metres → pixels (tweak to taste)
        HEAD_ANGLE = np.radians(23)    # opening angle of the arrow head
        HEAD_LEN = 8                   # pixels
        STEP = 3                       # subsample factor

        # fx_grid, fy_grid, x_forces, y_forces are 2-D numpy arrays
        for j in range(0, GRID_CELLS, STEP):
            for i in range(0, GRID_CELLS, STEP):
                fx = fx_grid[j, i]
                fy = fy_grid[j, i]
                if fx == 0.0 and fy == 0.0:
                    continue

                # start point = centre of that grid cell
                x0 = x_forces[j, i]
                y0 = y_forces[j, i]
                x1 = x0 + fx / ARROW_SCALE      # tip point in metres
                y1 = y0 + fy / ARROW_SCALE

                sx0, sy0 = to_screen(x0, y0)
                sx1, sy1 = to_screen(x1, y1)
                # pygame.draw.line(canvas, ARROW_COLOR, (sx0, sy0), (sx1, sy1), 1)

                # main shaft
                pygame.draw.line(canvas, ARROW_COLOR, (sx0, sy0), (sx1, sy1), 1)

                # direction unit vector in screen space
                dx, dy = sx1 - sx0, sy1 - sy0
                # length = np.hypot(dx, dy)
                # if length == 0:
                if dx == 0 and dy == 0:
                    continue

                base_angle = np.arctan2(dy, dx)  # angle of the arrow shaft
                left_angle = base_angle + HEAD_ANGLE
                right_angle = base_angle - HEAD_ANGLE

                lx = sx1 - HEAD_LEN * np.cos(left_angle)
                ly = sy1 - HEAD_LEN * np.sin(left_angle)
                rx = sx1 - HEAD_LEN * np.cos(right_angle)
                ry = sy1 - HEAD_LEN * np.sin(right_angle)

                pygame.draw.line(canvas, ARROW_COLOR, (sx1, sy1), (lx, ly), 1)
                pygame.draw.line(canvas, ARROW_COLOR, (sx1, sy1), (rx, ry), 1)

        # --------- 5. Blit & cap FPS ------------------------
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.display.update()
            pygame.event.pump()
            for e in pygame.event.get(pygame.QUIT):
                self.close()
                raise SystemExit
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None

    def save_wind_field(self, filename="wind_field.png"):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("Wind Field")
        ax.set_xlabel("X position (m)")
        ax.set_ylabel("Altitude (m)")
        ax.set_xlim(X_RANGE)
        ax.set_ylim(Y_RANGE)
        # q = ax.quiver(x_forces, y_forces, fx_grid, fy_grid, color='blue', alpha=0.5)          # Unused so far
        for xe in x_edges:
            ax.axvline(x=xe, color="gray", linestyle="--", alpha=0.3)
        for ye in y_edges:
            ax.axhline(y=ye, color="gray", linestyle="--", alpha=0.3)
        fig.savefig(filename)
        plt.close(fig)


if __name__ == "__main__":
    # Quick test of the environment.
    env = Balloon2DEnv()
    obs = env.reset()
    for _ in range(100):
        obs, reward, done, info = env.step(env.action_space.sample())
        env.render()
        print(f"Obs: {obs}, Reward: {reward}")
    env.save_wind_field("test_wind_field.png")
    env.close()
