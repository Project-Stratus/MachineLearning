"""
Multi dimensional (1D, 2D, 3D) balloon Gymnasium environment
==================================================================

Single Gymnasium environment that supports 1D, 2D and 3D balloon
environments with the same API.

=========
Dimensions
=========
+ 1D: Altitude only (z).
+ 2D: Horizontal plane (x, y) with fixed altitude (z).
+ 3D: Full 3D space (x, y, z).

Choose mode with constructor argument `dim=1|2|3`.

=========
Action space
=========
Discrete action space with three actions:

+ 0: Inflate balloon (increase volume).
+ 1: Deflate balloon (decrease volume).
+ 2: Do nothing (no volume change).

=========
Observation space
=========
Observation space is a dictionary with the following keys, each
scaled to [0,1]:

+ goal      : (dim,)    # Target position
+ volume    : (1,)      # Current balloon volume
+ position  : (dim,)    # Current balloon position
+ velocity  : (dim,)    # Current balloon velocity
+ pressure  : (1,)      # Ambient pressure at current altitude
+ wind      : (dim,)    # Instantaneous wind vector at current position

=========
Reward & Termination
=========
Reward is negative Euclidean distance to the goal.
Additional penalty of -400 is applied if the balloon crashes.

Termination occurs when:
+ Balloon altitude ≤ 0 m (crash).
+ Time limit reached (default 1000 steps).

=========
Visualisation
=========
Pygame window with two panels:
+ Left 75%: Top-down map (x-y) with wind arrows.
+ Right 25%: Vertical altitude bar.

=========
Configuration
=========
Tunable constants are in DEFAULTS dict, pass `config` argument
to override any of them without sub-classing.

"""
from __future__ import annotations

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from enum import Enum
from typing import Literal, Dict, Any, Tuple
import pygame
import json

# -----------------------------------------------------------------------------
# External domain classes
# -----------------------------------------------------------------------------
from environments.core.balloon import Balloon
from environments.core.atmosphere import Atmosphere
from environments.core.wind_field import WindField
from environments.envs.reward import distance_reward
from environments.render.pygame_render import PygameRenderer
from environments.core.constants import VOL_MAX, ALT_MAX, VEL_MAX, P_MAX, DT


# -----------------------------------------------------------------------------
class Actions(Enum):
    inflate = 0
    deflate = 1
    nothing = 2


class Balloon3DEnv(gym.Env):
    """Unified 1D/2D/3D balloon environment."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 150}

    # sensible defaults – override by *config*
    DEFAULTS: Dict[str, Any] = dict(
        dim=3,                    # 1, 2 or 3 dimensions
        time_max=1_000,           # steps per episode
        punishment=-400.0,        # reward on crash
        x_range=(-2_000.0, 2_000.0),
        y_range=(-2_000.0, 2_000.0),
        z_range=(0.0, ALT_MAX),
        wind_mag=10.0,           # max wind speed [m/s]
        wind_cells=40,           # grid for wind visualisation
        inflate_rate=0.02,       # Δvolume per *inflate* action
        window_size=(800, 600),  # pygame window (w,h)
        wind_pattern="split_fork",      # wind pattern: "sinusoid", "linear_right", "linear_up", "split_fork"
    )

    def __init__(self,
                 dim: Literal[1, 2, 3] = 3,
                 render_mode: str | None = None,
                 *,
                 config: Dict[str, Any] | None = None):
        # merge defaults with overrides
        cfg = {**self.DEFAULTS, **(config or {}), "dim": dim}
        self.cfg = cfg
        self.dim: int = cfg["dim"]
        self.wind_cfg_path = "environments/winds.json"

        assert self.dim in (1, 2, 3), "dim must be 1, 2 or 3"
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # ------------------------------------------------------------------
        # Ranges & grids
        # ------------------------------------------------------------------
        self.x_range: Tuple[float, float] = cfg["x_range"]
        self.y_range: Tuple[float, float] = cfg["y_range"]
        self.z_range: Tuple[float, float] = cfg["z_range"]
        self.z0 = 0.5 * (self.z_range[0] + self.z_range[1])   # reference altitude
        # self._wind_grid_init()
        self.wind = WindField(
            x_range=self.x_range,
            y_range=self.y_range,
            z_range=self.z_range,
            cells=cfg["wind_cells"],
            pattern=self.cfg.get("wind_pattern", "sinusoid"),
            default_mag=cfg["wind_mag"],
            wind_cfg_path=self.wind_cfg_path
        )

        self.x_centers = self.wind.x_centers
        self.y_centers = self.wind.y_centers
        self.z_centers = self.wind.z_centers
        self.wind_cells = self.wind.cells

        # ------------------------------------------------------------------
        # Spaces (all normalised to [0,1])
        # ------------------------------------------------------------------
        obs_low, obs_high = self._build_observation_bounds()
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.action_space = spaces.Discrete(len(Actions))

        # ------------------------------------------------------------------
        # Runtime state containers
        # ------------------------------------------------------------------
        self._atmosphere: Atmosphere | None = None
        self._balloon: Balloon | None = None
        self.goal: np.ndarray | None = None
        self._time: int = 0

        # bookkeeping for gym
        self.final_obs = None
        self.truncated = False

        # pygame
        self.window: pygame.Surface | None = None
        self.clock: pygame.time.Clock | None = None
        self.window_w, self.window_h = cfg["window_size"]
        self.renderer: PygameRenderer | None = None   # Create renderer lazily

    # ----------------------------------------------------------------------
    # Wind field initialisation & evaluation
    # ----------------------------------------------------------------------
    # def _wind_grid_init(self):
    #     cells = self.cfg["wind_cells"]
    #     self.wind_cells = cells
    #     xr = self.x_range
    #     yr = self.y_range
    #     zr = self.z_range
    #     # 3‑D grid: for 1‑D or 2‑D we keep thickness‑1 slices
    #     self.x_edges = np.linspace(xr[0], xr[1], cells + 1)
    #     self.y_edges = np.linspace(yr[0], yr[1], cells + 1)
    #     self.z_edges = np.linspace(zr[0], zr[1], cells + 1)
    #     self.x_centers = (self.x_edges[:-1] + self.x_edges[1:]) / 2
    #     self.y_centers = (self.y_edges[:-1] + self.y_edges[1:]) / 2
    #     self.z_centers = (self.z_edges[:-1] + self.z_edges[1:]) / 2

    #     # # build mesh for sinusoidal wind – simple but rich dynamics
    #     # X, Y, Z = np.meshgrid(self.x_centers, self.y_centers, self.z_centers, indexing="ij")
    #     # mag = self.cfg["wind_mag"]
    #     # self._fx_grid = (mag / 2) * (np.sin(2 * np.pi * X / (xr[1] - xr[0])) + 0.5 * np.sin(4 * np.pi * X / (xr[1] - xr[0])))
    #     # self._fy_grid = (mag / 2) * (np.cos(2 * np.pi * Y / (yr[1] - yr[0])) + 0.5 * np.cos(4 * np.pi * Y / (yr[1] - yr[0])))
    #     # # add gentle altitude‑dependent shear on x
    #     # self._fx_grid += (mag / 4) * np.sin(2 * np.pi * Z / (zr[1] - zr[0]))

    #     # Wind patterns: sin, linear_x, linear_z, split
    #     pat = self.cfg.get("wind_pattern", "sinusoid")      # Name of wind pattern

    #     print(f"Initialised wind field with pattern: {pat}")

    #     with open(self.wind_cfg_path, 'r') as f:
    #         wind_cfg = json.load(f)

    #     if pat in wind_cfg:
    #         mag = wind_cfg[pat].get("wind_mag", None)
    #     else:
    #         raise

    #     X, Y, Z, = np.meshgrid(self.x_centers, self.y_centers, self.z_centers, indexing="ij")

    #     if pat == 'linear_right':
    #         self._fx_grid = np.full_like(X, mag)
    #         self._fy_grid = np.zeros_like(Y)

    #     elif pat == 'linear_up':
    #         self._fx_grid = np.zeros_like(X)
    #         self._fy_grid = np.full_like(Y, mag)

    #     elif pat == 'split_fork':
    #         # Normalise X ∈ [0,1] (0 at mid-plane) and |Y| ∈ [0,1]
    #         Xn = np.clip((X - 0.0) / (xr[1] - 0.0), 0.0, 1.0)      # 0 left, 1 right
    #         Yn = np.clip(np.abs(Y) / (0.5 * (yr[1] - yr[0])), 0.0, 1.0)

    #         alpha = Xn * Yn            # 0 along centre-line & entire left half,1 at far-right + top/bottom edges

    #         # Components: magnitude preserved, direction blends from (1,0) to (0,±1)
    #         self._fx_grid = mag * (1.0 - alpha)                # diminish x-push
    #         self._fy_grid = mag * alpha * np.sign(Y)           # outward ±y
    #     else:                                     # default sinusoid ("sin")
    #         self._fx_grid = (mag / 2) * (np.sin(2*np.pi*X/(xr[1]-xr[0])) /
    #                                      +0.5*np.sin(4*np.pi*X/(xr[1]-xr[0])))
    #         self._fy_grid = (mag / 2) * (np.cos(2*np.pi*Y/(yr[1]-yr[0])) /
    #                                      +0.5*np.cos(4*np.pi*Y/(yr[1]-yr[0])))
    #         # gentle altitude-shear
    #         self._fx_grid += (mag / 4) * np.sin(2*np.pi*Z/(zr[1]-zr[0]))

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------
    def _build_observation_bounds(self):
        dim = self.dim
        low = []  # collect pieces in order matching _get_obs
        high = []
        # goal, position, velocity, wind all have size *dim*
        low.extend([0.0] * dim)                      # goal
        low.append(0.0)                              # volume
        low.extend([0.0] * dim)                      # position (normalised 0‑1)
        low.extend([-1.0] * dim)                     # velocity
        low.append(0.0)                              # pressure
        low.extend([-1.0] * dim)                     # wind

        high.extend([1.0] * dim)                     # goal
        high.append(1.0)                             # volume
        high.extend([1.0] * dim)                     # position
        high.extend([1.0] * dim)                     # velocity
        high.append(1.0)                             # pressure
        high.extend([1.0] * dim)                     # wind

        return np.array(low, dtype=np.float32), np.array(high, dtype=np.float32)

    # pack observation in the same order as bounds above
    def _get_obs(self) -> np.ndarray:
        # dim = self.dim
        pos = self._balloon.pos  # native metres array (len == dim)
        vel = self._balloon.vel
        alt = pos[-1] if self.dim == 3 else self.z0
        pos_norm = self._normalise_position(pos)
        vel_norm = np.clip(vel / VEL_MAX, -1.0, 1.0)
        vol_norm = np.array([self._balloon.volume / VOL_MAX])
        goal_norm = self._normalise_position(self.goal)
        pressure_norm = np.array([self._atmosphere.pressure(alt) / P_MAX])
        # wind = self._wind_at(*self._full_coords(pos)) / self.cfg["wind_mag"]
        wind = self.wind.sample(*self._full_coords(pos)) / self.cfg["wind_mag"]
        wind = wind[:self.dim]  # slice to dim

        return np.concatenate([goal_norm, vol_norm, pos_norm, vel_norm, pressure_norm, wind])

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------
    def _normalise_position(self, pos: np.ndarray) -> np.ndarray:
        # expect shape (dim,)
        if self.dim == 1:
            z = pos[0]
            return np.array([(z - self.z_range[0]) / (self.z_range[1] - self.z_range[0])])
        elif self.dim == 2:
            x, y = pos[:2]
            return np.array([
                (x - self.x_range[0]) / (self.x_range[1] - self.x_range[0]),
                (y - self.y_range[0]) / (self.y_range[1] - self.y_range[0])
            ])
        else:  # 3‑D
            x, y, z = pos
            return np.array([
                (x - self.x_range[0]) / (self.x_range[1] - self.x_range[0]),
                (y - self.y_range[0]) / (self.y_range[1] - self.y_range[0]),
                (z - self.z_range[0]) / (self.z_range[1] - self.z_range[0])
            ])

    def _full_coords(self, pos: np.ndarray) -> Tuple[float, float, float]:
        """Return (x,y,z) regardless of *dim*, padding with zeros as needed."""
        if self.dim == 1:
            return 0.0, 0.0, float(pos[0])
        elif self.dim == 2:
            x, y = pos[:2]
            return float(x), 0.0, float(y)
        else:
            x, y, z = pos
            return float(x), float(y), float(z)

    # ------------------------------------------------------------------
    # Wind query – trilinear interpolation on the precomputed grid
    # ------------------------------------------------------------------
    # def _wind_at(self, x: float, y: float, z: float) -> np.ndarray:
    #     # clamp inside ranges
    #     xi = np.clip(x, *self.x_range)
    #     yi = np.clip(y, *self.y_range)
    #     zi = np.clip(z, *self.z_range)

    #     # indices
    #     ix = np.searchsorted(self.x_edges, xi) - 1
    #     iy = np.searchsorted(self.y_edges, yi) - 1
    #     iz = np.searchsorted(self.z_edges, zi) - 1
    #     ix = np.clip(ix, 0, self.wind_cells - 1)
    #     iy = np.clip(iy, 0, self.wind_cells - 1)
    #     iz = np.clip(iz, 0, self.wind_cells - 1)

    #     fx = self._fx_grid[ix, iy, iz]
    #     fy = self._fy_grid[ix, iy, iz]
    #     # no vertical wind for simplicity – extend as needed
    #     fz = 0.0
    #     if self.dim == 1:
    #         return np.array([fz])
    #     elif self.dim == 2:
    #         return np.array([fx, fy])
    #     else:
    #         return np.array([fx, fy, fz])

    # ------------------------------------------------------------------
    # Gym API – reset
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)

        self._time = 0
        self.truncated = False
        self.final_obs = None

        self._atmosphere = Atmosphere()

        # random starting position far enough from the goal (~500 m) to avoid trivial episodes
        while True:
            if self.dim == 1:
                z0 = self.np_random.uniform(*self.z_range)
                pos0 = [z0]
            elif self.dim == 2:
                x0 = self.np_random.uniform(*self.x_range)
                y0 = self.np_random.uniform(*self.y_range)
                pos0 = [x0, y0]
            else:
                x0 = self.np_random.uniform(*self.x_range)
                y0 = self.np_random.uniform(*self.y_range)
                z0 = self.np_random.uniform(*self.z_range)
                pos0 = [x0, y0, z0]

            # random goal
            if self.dim == 1:
                zg = self.np_random.uniform(*self.z_range)
                goal = np.array([zg])
                dist = abs(z0 - zg)
            elif self.dim == 2:
                xg = self.np_random.uniform(*self.x_range)
                yg = self.np_random.uniform(*self.y_range)
                goal = np.array([xg, yg])
                dist = math.hypot(x0 - xg, y0 - yg)
            else:
                xg = self.np_random.uniform(*self.x_range)
                yg = self.np_random.uniform(*self.y_range)
                zg = self.np_random.uniform(*self.z_range)
                goal = np.array([xg, yg, zg])
                dist = math.sqrt((x0 - xg) ** 2 + (y0 - yg) ** 2 + (z0 - zg) ** 2)

            if dist > 500.0:
                break

        self.goal = goal
        real_dim = 3 if self.dim == 2 else self.dim
        init_pos = pos0 + [self.z0] if self.dim == 2 else pos0
        self._balloon = Balloon(dim=real_dim,
                                atmosphere=self._atmosphere,
                                position=init_pos,
                                velocity=[0.0] * real_dim,
                                )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            # self._render_frame()
            self._ensure_renderer()
            self.renderer.draw(dict(
                dim=self.dim,
                balloon_pos=self._balloon.pos.copy(),
                goal_pos=self.goal.copy(),
                z0=self.z0,
                wind_sampler=self.wind.sample
            ))

        return observation, info

    # ------------------------------------------------------------------
    # Gym API – step
    # ------------------------------------------------------------------
    def step(self, action: int):
        assert self._balloon is not None, "Call reset() first."

        # --- execute action ------------------------------------------------
        if action == Actions.inflate.value:
            self._balloon.inflate(self.cfg["inflate_rate"])
        elif action == Actions.deflate.value:
            self._balloon.inflate(-self.cfg["inflate_rate"])
        # nothing → no volume change

        # horizontal wind force only (vertical handled by buoyancy)
        # wind = self._wind_at(*self._full_coords(self._balloon.pos))
        wind = self.wind.sample(*self._full_coords(self._balloon.pos))
        # pad to dim
        if self.dim == 1:
            control_force = [0.0]
        elif self.dim == 2:
            control_force = wind.tolist() + [0.0]
        else:
            control_force = wind.tolist()
        # update balloon physics
        self._balloon.update(DT, external_force=control_force)  # Balloon signature accepts *external_force*

        if self.dim == 2:          # keep altitude fixed
            self._balloon.pos[2] = self.z0
            self._balloon.vel[2] = 0.0

        # --- reward & termination -----------------------------------------
        self._time += 1
        alt = self._balloon.pos[-1]
        terminated = (self.dim == 3 and alt <= 0.0)  # crash to ground
        self.truncated = self._time >= self.cfg["time_max"]

        # # dist = np.linalg.norm(self._balloon.pos - self.goal)
        # if self.dim == 1:
        #     dist = abs(self._balloon.pos[-1] - self.goal[0])
        # elif self.dim == 2:
        #     dist = math.hypot(self._balloon.pos[0] - self.goal[0],
        #                       self._balloon.pos[1] - self.goal[1])
        # else:  # dim == 3
        #     dist = np.linalg.norm(self._balloon.pos - self.goal)

        # reward = -dist
        # if terminated:
        #     reward = self.cfg["punishment"]

        reward = distance_reward(
            balloon_pos=self._balloon.pos,
            goal_pos=self.goal,
            dim=self.dim,
            terminated=terminated,
            punishment=self.cfg["punishment"]
        )

        obs = self._get_obs()
        if terminated or self.truncated:
            self.final_obs = obs

        info = self._get_info()

        if self.render_mode == "human":
            # self._render_frame()
            self._ensure_renderer()
            self.renderer.draw(dict(
                dim=self.dim,
                balloon_pos=self._balloon.pos.copy(),
                goal_pos=self.goal.copy(),
                z0=self.z0,
                wind_sampler=self.wind.sample
            ))

        return obs, reward, terminated, self.truncated, info

    # ------------------------------------------------------------------
    def _get_info(self):
        return {
            "TimeLimit.truncated": self.truncated,
            "terminal_observation": self.final_obs,
        }

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------
    def render(self):
        # if self.render_mode == "rgb_array":
        #     return self._render_frame()
        if self.render_mode == "human":
            self._ensure_renderer()
            self.renderer.draw(dict(
                dim=self.dim,
                balloon_pos=self._balloon.pos,
                goal_pos=self.goal.copy(),
                z0=self.z0,
                wind_sampler=self.wind.sample
            ))

    # def _render_frame(self):
    #     if self.window is None and self.render_mode == "human":
    #         import os
    #         os.environ["SDL_VIDEO_WINDOW_POS"] = "0,0"        # always top-left
    #         os.environ.pop("SDL_VIDEO_CENTERED", None)        # cancel any centring

    #         pygame.init()
    #         pygame.display.init()
    #         self.window = pygame.display.set_mode((self.window_w, self.window_h), display=0)
    #     if self.clock is None and self.render_mode == "human":
    #         self.clock = pygame.time.Clock()

    #     canvas = pygame.Surface((self.window_w, self.window_h))
    #     canvas.fill((255, 255, 255))

    #     left_w = int(self.window_w * 0.75)
    #     right_w = self.window_w - left_w

    #     # ------------------------------------------------------------
    #     # LEFT PANEL – top‑down map (x‑y or x‑z)
    #     # ------------------------------------------------------------
    #     map_surface = pygame.Surface((left_w, self.window_h))
    #     map_surface.fill((240, 240, 240))

    #     # coordinate transforms
    #     def to_left_panel(px: float, py: float):
    #         # sx = (px - self.x_range[0]) / (self.x_range[1] - self.x_range[0])
    #         # sy = (py - self.y_range[0]) / (self.y_range[1] - self.y_range[0]) if self.dim == 3 else (py - self.z_range[0]) / (self.z_range[1] - self.z_range[0])
    #         # if self.dim == 3:
    #         #     sy = (py - self.y_range[0]) / (self.y_range[1] - self.y_range[0])
    #         # else:  # 1‑D/2‑D – use z slice
    #         # sy = (py - self.z_range[0]) / (self.z_range[1] - self.z_range[0])
    #         sx = (px - self.x_range[0]) / (self.x_range[1] - self.x_range[0])
    #         if self.dim == 3:
    #             sy = (py - self.y_range[0]) / (self.y_range[1] - self.y_range[0])
    #         elif self.dim == 2:
    #             sy = (py - self.y_range[0]) / (self.y_range[1] - self.y_range[0])
    #         else:                                # dim == 1 (unused map)
    #             sy = 0.5
    #         return int(sx * left_w), int((1.0 - sy) * self.window_h)

    #     # draw wind arrows (subsample for clarity)
    #     STEP = 4
    #     HEAD_LEN = 6
    #     ARROW_COL = (180, 180, 255)
    #     for ix in range(0, self.wind_cells, STEP):
    #         for iy in range(0, self.wind_cells, STEP):
    #             x = self.x_centers[ix]
    #             if self.dim == 3:
    #                 y = self.y_centers[iy]
    #                 z = self._balloon.pos[-1]  # current altitude slice
    #             else:  # 1‑D/2‑D – use ix & iy for x & z slices
    #                 if self.dim == 1:
    #                     continue  # no arrows
    #                 y = self.y_centers[iy]
    #                 z = self.z0
    #             # wind = self._wind_at(x, y, z)
    #             wind = self.wind.sample(x, y, z)
    #             if np.allclose(wind, 0.0):
    #                 continue
    #             # pick 2‑D projection
    #             wx, wy = wind[0], wind[1]
    #             scale = 5  # visual scaling
    #             x0, y0 = to_left_panel(x, y)
    #             x1, y1 = to_left_panel(x + wx * scale, y + wy * scale)
    #             pygame.draw.line(map_surface, ARROW_COL, (x0, y0), (x1, y1), 1)
    #             # arrowhead
    #             angle = math.atan2(y1 - y0, x1 - x0)
    #             for sign in (-1, 1):
    #                 dx = HEAD_LEN * math.cos(angle + sign * math.pi / 6)
    #                 dy = HEAD_LEN * math.sin(angle + sign * math.pi / 6)
    #                 pygame.draw.line(map_surface, ARROW_COL, (x1, y1), (x1 - dx, y1 - dy), 1)

    #     # draw balloon & goal
    #     # bx, by = to_left_panel(*self._balloon.pos[:2] if self.dim >= 2 else (0.0, self._balloon.pos[0]))
    #     if self.dim == 1:
    #         bx, by = to_left_panel(0.0, 0.0)
    #     elif self.dim == 2:
    #         bx, by = to_left_panel(self._balloon.pos[0], self._balloon.pos[1])
    #     else:  # dim == 3
    #         bx, by = to_left_panel(self._balloon.pos[0], self._balloon.pos[1])

    #     pygame.draw.circle(map_surface, (255, 0, 0), (bx, by), 8)
    #     # gx, gy = to_left_panel(*self.goal[:2] if self.dim >= 2 else (0.0, self.goal[0]))
    #     if self.dim == 1:
    #         gx, gy = to_left_panel(0.0, 0.0)
    #     elif self.dim == 2:
    #         gx, gy = to_left_panel(self.goal[0], self.goal[1])
    #     else:  # dim == 3
    #         gx, gy = to_left_panel(self.goal[0], self.goal[1])

    #     pygame.draw.circle(map_surface, (0, 200, 0), (gx, gy), 6)

    #     # ------------------------------------------------------------
    #     # RIGHT PANEL – altitude bar
    #     # ------------------------------------------------------------
    #     alt_surface = pygame.Surface((right_w, self.window_h))
    #     alt_surface.fill((250, 250, 250))
    #     # draw scale
    #     pygame.draw.line(alt_surface, (0, 0, 0), (right_w // 2, 0), (right_w // 2, self.window_h), 2)
    #     # balloon altitude
    #     bz = (1.0 - (self._balloon.pos[-1] - self.z_range[0]) / (self.z_range[1] - self.z_range[0])) * self.window_h
    #     pygame.draw.circle(alt_surface, (255, 0, 0), (right_w // 2, int(bz)), 6)
    #     # goal altitude
    #     gz = (1.0 - (self.goal[-1] - self.z_range[0]) / (self.z_range[1] - self.z_range[0])) * self.window_h
    #     pygame.draw.circle(alt_surface, (0, 200, 0), (right_w // 2, int(gz)), 5)

    #     # ------------------------------------------------------------
    #     # Compose panels
    #     # ------------------------------------------------------------
    #     canvas.blit(map_surface, (0, 0))
    #     canvas.blit(alt_surface, (left_w, 0))

    #     if self.render_mode == "human":
    #         self.window.blit(canvas, canvas.get_rect())
    #         pygame.display.update()
    #         pygame.event.pump()
    #         self.clock.tick(self.metadata["render_fps"])
    #     else:
    #         return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    # ------------------------------------------------------------------
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None

    def _ensure_renderer(self):
        """Create renderer instance on first use"""
        if self.renderer is None:
            self.renderer = PygameRenderer(
                window_size=(self.window_w, self.window_h),
                x_range=self.x_range,
                y_range=self.y_range,
                z_range=self.z_range,
                x_centers=self.x_centers,
                y_centers=self.y_centers,
                wind_cells=self.wind_cells
            )
