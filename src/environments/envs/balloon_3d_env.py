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
For most of the code, dim=1 is treated with only z coordinate but
dim=2|3 both are computed in 3 dimensions with a static z coordinate
for 2D.

=========
Action space
=========
Discrete action space with three actions:

+ 1: Inflate balloon (increase volume).
  0: Do nothing (no volume change).
- 1: Deflate balloon (decrease volume).

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

from environments.core.balloon import Balloon
from environments.core.atmosphere import Atmosphere
from environments.core.wind_field import WindField
from environments.core.reward import balloon_reward, l2_distance
from environments.render.pygame_render import PygameRenderer
from environments.core.constants import VOL_MAX, ALT_MAX, VEL_MAX, P_MAX, DT

_JIT_WARMED = False  # whether numba JIT has been warmed up


class Actions(Enum):
    inflate = 1
    nothing = 0
    deflate = -1


class Balloon3DEnv(gym.Env):
    """Unified 1D/2D/3D balloon environment."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 150}

    # sensible defaults – override by *config*
    DEFAULTS: Dict[str, Any] = dict(
        dim=3,                    # 1, 2 or 3 dimensions
        time_max=5_000,           # steps per episode
        punishment=-5.0,        # reward on crash
        x_range=(-2_000.0, 2_000.0),
        y_range=(-2_000.0, 2_000.0),
        z_range=(0.0, ALT_MAX),
        wind_mag=10.0,           # max wind speed [m/s]
        wind_cells=20,           # grid for wind visualisation
        inflate_rate=0.01,       # Δvolume per *inflate* action
        window_size=(800, 600),  # pygame window (w,h)
        wind_pattern="split_fork",      # wind pattern: "sinusoid", "linear_right", "linear_up", "split_fork"
        alpha=0.05,                   # velocity cost weight
        beta=0.01,                   # action-flip cost weight

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

        self.alpha = cfg["alpha"]  # velocity cost weight
        self.beta = cfg["beta"]    # action-flip cost weight

        assert self.dim in (1, 2, 3), f"dim must be 1, 2 or 3. Got {self.dim}."
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # ------------------------------------------------------------------
        # Ranges & grids
        # ------------------------------------------------------------------
        self.x_range: Tuple[float, float] = cfg["x_range"]
        self.y_range: Tuple[float, float] = cfg["y_range"]
        self.z_range: Tuple[float, float] = cfg["z_range"]
        self.z0 = 0.5 * (self.z_range[0] + self.z_range[1])   # reference altitude

        # precompute normalisation spans
        self.inv_xspan = 0.0 if self.dim == 1 else 1.0 / (self.x_range[1] - self.x_range[0])
        self.inv_yspan = 0.0 if self.dim <= 1 else 1.0 / (self.y_range[1] - self.y_range[0])
        self.inv_zspan = 1.0 / (self.z_range[1] - self.z_range[0])


        # ------------------------------------------------------------------
        # Wind field
        # ------------------------------------------------------------------
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
        self._obs_size = self.observation_space.shape[0]
        self._obs_buf = np.empty(self._obs_size, dtype=np.float32)
        self.goal_norm = None
        self.action_space = spaces.Discrete(3)      # inflate, deflate and nothing. Creates idx values 0,1,2
        self._action_lut = np.array([-1, 0, 1])     # map action index to effect on volume (0->-1, 1->0, 2->+1)

        # ------------------------------------------------------------------
        # Runtime state containers
        # ------------------------------------------------------------------
        self._atmosphere: Atmosphere | None = None
        self._balloon: Balloon | None = None
        self.goal: np.ndarray | None = None
        self._time: int = 0
        self.last_wind = np.zeros(3, dtype=np.float32)
        self.last_pressure_norm = 0.0

        # bookkeeping for gym
        self.final_obs = None
        self.truncated = False

        # pygame
        self.window: pygame.Surface | None = None
        self.clock: pygame.time.Clock | None = None
        self.window_w, self.window_h = cfg["window_size"]
        self.renderer: PygameRenderer | None = None   # Create renderer lazily

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
        low.extend([-1.0] * dim)                     # goal - position (normalised)
        low.extend([-1.0] * dim)                     # velocity
        low.append(0.0)                              # pressure
        low.extend([-1.0] * dim)                     # wind

        high.extend([1.0] * dim)                     # goal
        high.append(1.0)                             # volume
        high.extend([1.0] * dim)                     # position
        high.extend([1.0] * dim)                     # goal - position (normalised)
        high.extend([1.0] * dim)                     # velocity
        high.append(1.0)                             # pressure
        high.extend([1.0] * dim)                     # wind

        return np.array(low, dtype=np.float32), np.array(high, dtype=np.float32)

    # pack observation in the same order as bounds above
    def _get_obs(self) -> np.ndarray:
        pos = self._balloon.pos  # native metres array (len == dim)
        vel = self._balloon.vel

        pos_norm = self._normalise_position(pos).astype(np.float32)
        delta_norm = (self.goal_norm - pos_norm).astype(np.float32)
        vel_norm = np.clip(vel / VEL_MAX, -1.0, 1.0).astype(np.float32)
        vol_norm = np.array([self._balloon.volume / VOL_MAX]).astype(np.float32)
        pressure_norm = np.array([self.last_pressure_norm], dtype=np.float32)
        wind = self.last_wind[:self.dim] / self.cfg["wind_mag"]

        i = 0
        d = self.dim
        self._obs_buf[i:i+d] = self.goal_norm
        i += d
        self._obs_buf[i] = vol_norm[0]
        i += 1
        self._obs_buf[i:i+d] = pos_norm
        i += d
        self._obs_buf[i:i+d] = delta_norm
        i += d
        self._obs_buf[i:i+d] = vel_norm[:d]
        i += d
        self._obs_buf[i] = pressure_norm[0]
        i += 1
        self._obs_buf[i:i+d] = wind
        return self._obs_buf.copy()

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------
    def _normalise_position(self, pos: np.ndarray) -> np.ndarray:
        # expect shape (dim,)
        if self.dim == 1:
            z = pos[0]
            return np.array([(z - self.z_range[0]) * self.inv_zspan], dtype=np.float32)
        elif self.dim == 2:
            x, y = pos[:2]
            return np.array([(x - self.x_range[0]) * self.inv_xspan,
                             (y - self.y_range[0]) * self.inv_yspan], dtype=np.float32)
        else:  # 3‑D
            x, y, z = pos
            return np.array([(x - self.x_range[0]) * self.inv_xspan,
                             (y - self.y_range[0]) * self.inv_yspan,
                             (z - self.z_range[0]) * self.inv_zspan], dtype=np.float32)


    def _full_coords(self, pos: np.ndarray) -> Tuple[float, float, float]:
        """Return (x,y,z) regardless of *dim*, padding with zeros as needed."""
        if self.dim == 1:
            return 0.0, 0.0, float(pos[0])
        elif self.dim == 2:
            x, y = pos[:2]
            return float(x), float(y), self.z0
        else:
            x, y, z = pos
            return float(x), float(y), float(z)

    # ------------------------------------------------------------------
    # Gym API – reset
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        import numpy as np
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
        self.goal_norm = self._normalise_position(self.goal).astype(np.float32)
        real_dim = 3 if self.dim == 2 else self.dim
        init_pos = pos0 + [self.z0] if self.dim == 2 else pos0
        self._balloon = Balloon(dim=real_dim,
                                atmosphere=self._atmosphere,
                                position=init_pos,
                                velocity=[0.0] * real_dim,
                                )

        observation = self._get_obs()
        info = self._get_info()
        self._prev_distance = l2_distance(self._balloon.pos, self.goal, self.dim)

        try:
            self.prev_action = Actions.nothing.value  # no action on reset
        except AttributeError:
            self.prev_action = Actions.deflate.value  # first action is deflate

        if self.render_mode == "human":
            self._ensure_renderer()
            self.renderer.draw(dict(
                dim=self.dim,
                balloon_pos=self._balloon.pos.copy(),
                goal_pos=self.goal.copy(),
                z0=self.z0,
                wind_sampler=self.wind.sample
            ))

        # Warm up JIT to compile before training
        global _JIT_WARMED
        if not _JIT_WARMED:
            try:
                from environments.core.jit_kernels import pressure_numba, density_numba, wind_sample_idx_numba, physics_step_numba
                # warm-up calls with small dummy inputs (compile once)
                _ = pressure_numba(101325.0, 8500.0, 1000.0)
                _ = density_numba(101325.0, 8500.0, 288.15, 0.0289647, 8.314462618, 1000.0)
                # wind warmup
                wf = self.wind
                # ensure we touch the grids; use grid center and these parameters
                _ = wind_sample_idx_numba(wf.x_centers[0], wf.y_centers[0], wf.z_centers[0],
                                        wf.x_range[0], wf.inv_dx, wf.y_range[0], wf.inv_dy, wf.z_range[0], wf.inv_dz,
                                        wf.cells, wf._fx_grid, wf._fy_grid)
                # physics warmup (dim-aware)
                import numpy as np
                pos = self._balloon.pos.astype(np.float64, copy=True)
                vel = self._balloon.vel.astype(np.float64, copy=True)
                ext = np.zeros_like(pos)
                ctrl = np.zeros_like(pos)
                physics_step_numba(pos, vel, 0.01, self._balloon.mass, 9.81, 0.5, 1.0, 1.2, self._balloon.volume, ext, ctrl)
            except Exception:
                pass  # if numba missing or compilation deferred, no problem
            _JIT_WARMED = True

        return observation, info

    # ------------------------------------------------------------------
    # Gym API – step
    # ------------------------------------------------------------------
    def step(self, action: int):
        assert self._balloon is not None, "Call reset() first."

        # --- execute action ------------------------------------------------
        # if action == Actions.inflate.value:
        #     self._balloon.inflate(self.cfg["inflate_rate"])
        # elif action == Actions.deflate.value:
        #     self._balloon.inflate(-self.cfg["inflate_rate"])
        # nothing → no volume change
        effect = int(self._action_lut[action])  # -1, 0, +1

        if effect:
            self._balloon.inflate(effect * self.cfg["inflate_rate"])    # -1, 0, +1 * rate

        wind = self.wind.sample(*self._full_coords(self._balloon.pos))
        self.last_wind[:] = wind    # Cache for obs

        # pad to dim
        if self.dim == 1:
            control_force = np.array([0.0], dtype=np.float32)
        elif self.dim == 2:
            control_force = np.array([wind[0], wind[1], 0.0], dtype=np.float32)
        else:
            control_force = wind.astype(np.float32, copy=False)

        # update balloon physics
        self._balloon.update(DT, external_force=control_force)  # Balloon signature accepts *external_force*

        if self.dim == 2:          # keep altitude fixed
            self._balloon.pos[2] = self.z0
            self._balloon.vel[2] = 0.0

        # --- reward & termination -----------------------------------------
        self._time += 1
        alt = self._balloon.pos[-1] if self.dim in (1, 3) else self.z0
        self.last_pressure_norm = float(self._atmosphere.pressure(alt) / P_MAX)
        # terminated = (self.dim == 3 and alt <= 0.0)  # crash to ground
        if (self.dim in (1, 3)) and self._balloon.pos[-1] <= 0.0:
            terminated = True
        else:
            terminated = False
        self.truncated = self._time >= self.cfg["time_max"]

        reward_total, reward_components, self._prev_distance = balloon_reward(
            balloon_pos=self._balloon.pos,
            goal_pos=self.goal,
            velocity=self._balloon.vel,
            dim=self.dim,
            terminated=terminated,
            effect=effect,
            punishment=self.cfg["punishment"],
            prev_distance=getattr(self, "_prev_distance", float("inf")),
        )

        components_for_info = dict(reward_components)
        components_for_info["total"] = reward_total

        info = self._get_info()
        info.setdefault("reward_components", {}).update(components_for_info)

        self.prev_action = action

        # --- collect observation & info -----------------------------------
        obs = self._get_obs()
        if terminated or self.truncated:
            self.final_obs = obs
            info["terminal_observation"] = self.final_obs

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

        return obs, reward_total, terminated, self.truncated, info

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
        if self.render_mode == "human":
            self._ensure_renderer()
            self.renderer.draw(dict(
                balloon_pos=self._balloon.pos,
                goal_pos=self.goal.copy(),
                z0=self.z0,
                wind_sampler=self.wind.sample
            ))

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer.window = None
            self.renderer.clock = None

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
                wind_cells=self.wind_cells,
                dim=self.dim,
            )
