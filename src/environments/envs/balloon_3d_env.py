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
Discrete(3) with action indices 0, 1, 2 mapped to effects:

    Index | Effect | Description
    ------|--------|--------------------------------------
      0   |   -1   | Vent gas (release helium → descend)
      1   |    0   | Do nothing
      2   |   +1   | Drop ballast (reduce weight → ascend)

Both actions are irreversible: vented helium and dropped ballast
cannot be recovered.  The mapping is defined in `_action_lut`.

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
Perciatelli-style reward (inspired by Google BLE):
+ Inside station-keeping radius (default 10 km): flat reward of 1.0
+ Outside radius: exponential decay toward 0.0
+ On termination: reward is 0.0 (forfeiting future reward is the penalty)
+ Reward range: [0, 1]

Termination occurs when:
+ Balloon altitude ≤ 0 m (crash).
+ Balloon altitude ≥ altitude ceiling (pop).
+ Balloon fully deflated (helium loss).
+ Ballast exhausted (no ballast remaining).
+ Balloon exits XY bounds (2D/3D only).
+ Time limit reached (default 5000 steps).

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
from environments.core.constants import (
    VOL_MAX, ALT_MAX, XY_MAX, VEL_MAX, P_MAX, DT,
    P0, M_AIR, G, RHO_0,
    BALLAST_DROP, VENT_RATE,
    MIN_START_DISTANCE,
    INIT_VEL_SIGMA, INIT_GAS_FRAC_RANGE, INIT_BALLAST_LOSS_MAX,
)

_JIT_WARMED = False  # whether numba JIT has been warmed up


class Actions(Enum):
    """Effect values for balloon altitude control (not action indices).

    drop_ballast (+1): drop expendable ballast to reduce weight → ascend.
    nothing      ( 0): take no action.
    vent         (-1): vent helium to reduce buoyancy → descend.
    """
    drop_ballast = 1
    nothing = 0
    vent = -1


class Balloon3DEnv(gym.Env):
    """Unified 1D/2D/3D balloon environment."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 150}

    # sensible defaults – override by *config*
    DEFAULTS: Dict[str, Any] = dict(
        dim=3,                    # 1, 2 or 3 dimensions
        time_max=5_000,           # steps per episode
        x_range=(-XY_MAX, XY_MAX),
        y_range=(-XY_MAX, XY_MAX),
        z_range=(0.0, ALT_MAX),
        wind_mag=5.0,            # max wind speed [m/s]
        wind_cells=20,           # grid for wind visualisation
        vent_rate=VENT_RATE,      # gas volume equivalent vented per *vent* action (m³)
        window_size=(800, 600),  # pygame window (w,h)
        wind_pattern="split_fork",      # wind pattern: "sinusoid", "linear_right", "linear_up", "split_fork", "altitude_shear", "altitude_shear_2d"
        wind_layers=2,                # number of full wind rotations over altitude range (altitude_shear_2d only)
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

        # Precompute normalisation arrays for vectorized position normalisation
        # dim=1 uses z only, dim=2 uses x,y, dim=3 uses x,y,z
        if self.dim == 1:
            self._norm_offsets = np.array([self.z_range[0]], dtype=np.float32)
            self._norm_scales = np.array([1.0 / (self.z_range[1] - self.z_range[0])], dtype=np.float32)
            self._ranges = [self.z_range]
        elif self.dim == 2:
            self._norm_offsets = np.array([self.x_range[0], self.y_range[0]], dtype=np.float32)
            self._norm_scales = np.array([
                1.0 / (self.x_range[1] - self.x_range[0]),
                1.0 / (self.y_range[1] - self.y_range[0])
            ], dtype=np.float32)
            self._ranges = [self.x_range, self.y_range]
        else:  # dim == 3
            self._norm_offsets = np.array([self.x_range[0], self.y_range[0], self.z_range[0]], dtype=np.float32)
            self._norm_scales = np.array([
                1.0 / (self.x_range[1] - self.x_range[0]),
                1.0 / (self.y_range[1] - self.y_range[0]),
                1.0 / (self.z_range[1] - self.z_range[0])
            ], dtype=np.float32)
            self._ranges = [self.x_range, self.y_range, self.z_range]


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
            wind_cfg_path=self.wind_cfg_path,
            wind_layers=cfg["wind_layers"],
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
        self._inv_wind_mag = 1.0 / cfg["wind_mag"]
        self.goal_norm = None
        self.action_space = spaces.Discrete(3)      # vent, nothing, drop ballast. Creates idx values 0,1,2
        self._action_lut = np.array([-1, 0, 1])     # map action index to effect (0->vent, 1->nothing, 2->drop ballast)

        # ------------------------------------------------------------------
        # Runtime state containers
        # ------------------------------------------------------------------
        self._atmosphere: Atmosphere | None = None
        self._balloon: Balloon | None = None
        self.goal: np.ndarray | None = None
        self._time: int = 0
        self.last_wind = np.zeros(3, dtype=np.float32)
        self._wind_vel_buf = np.zeros(3, dtype=np.float64)  # reusable wind velocity for physics
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
        d = self.dim
        inv_wind_mag = self._inv_wind_mag

        # Compute pos_norm in-place into obs_buf (goal_norm first, then pos_norm)
        i = 0
        self._obs_buf[i:i+d] = self.goal_norm
        i += d

        # volume
        self._obs_buf[i] = self._balloon.volume / VOL_MAX
        i += 1

        # position normalised
        pos_start = i
        for j in range(d):
            self._obs_buf[i + j] = (pos[j] - self._norm_offsets[j]) * self._norm_scales[j]
        i += d

        # delta (goal - position) normalised
        for j in range(d):
            self._obs_buf[i + j] = self.goal_norm[j] - self._obs_buf[pos_start + j]
        i += d

        # velocity normalised and clipped
        for j in range(d):
            v = vel[j] / VEL_MAX
            if v > 1.0:
                v = 1.0
            elif v < -1.0:
                v = -1.0
            self._obs_buf[i + j] = v
        i += d

        # pressure
        self._obs_buf[i] = self.last_pressure_norm
        i += 1

        # wind
        for j in range(d):
            self._obs_buf[i + j] = self.last_wind[j] * inv_wind_mag
        i += d

        return self._obs_buf.copy()

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------
    def _normalise_position(self, pos: np.ndarray) -> np.ndarray:
        """Normalise position to [0,1] using pre-computed offsets and scales."""
        return ((pos[:self.dim] - self._norm_offsets) * self._norm_scales).astype(np.float32)


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

        # Determine goal based on wind pattern
        wind_pattern = self.cfg.get("wind_pattern", "sinusoid")

        if self.dim == 1:
            # 1D: altitude only — goal is a target altitude
            if wind_pattern == "altitude_shear":
                goal = np.array([self.z0], dtype=np.float64)
            else:
                goal = np.array([self.np_random.uniform(*self.z_range)], dtype=np.float64)

            while True:
                pos0 = np.array([self.np_random.uniform(*self.z_range)], dtype=np.float64)
                if abs(pos0[0] - goal[0]) > MIN_START_DISTANCE:
                    break
        else:
            # 2D/3D: goal is x,y center (station-keeping).
            # Altitude is the agent's control mechanism, not an objective.
            goal_xy = np.array([0.0, 0.0], dtype=np.float64)
            if self.dim == 2:
                goal = goal_xy
            else:  # dim == 3
                goal = np.array([0.0, 0.0, self.z0], dtype=np.float64)

            # Balloon starts within 50% of XY_MAX so it's always closer to
            # the target than to the nearest edge.
            spawn_ranges = []
            for lo, hi in self._ranges:
                half = (hi - lo) * 0.25  # 25% from center = 50% of range
                mid = (lo + hi) * 0.5
                spawn_ranges.append((mid - half, mid + half))

            while True:
                pos0 = np.array([self.np_random.uniform(*r) for r in spawn_ranges], dtype=np.float64)
                # Check horizontal distance only for spawn validation
                dist_xy = float(np.sqrt((pos0[0] - goal[0])**2 + (pos0[1] - goal[1])**2))
                if dist_xy > MIN_START_DISTANCE:
                    break

        self.goal = goal
        self.goal_norm = self._normalise_position(self.goal).astype(np.float32)

        real_dim = 3 if self.dim == 2 else self.dim
        # For 2D mode, balloon is internally 3D with fixed altitude z0
        init_pos = np.append(pos0, self.z0) if self.dim == 2 else pos0
        self._balloon = Balloon(dim=real_dim,
                                atmosphere=self._atmosphere,
                                position=init_pos,
                                velocity=[0.0] * real_dim,
                                )

        # --- Domain randomisation of initial conditions ---
        # Simulates variability after reaching float altitude: turbulence,
        # imprecise inflation, and ballast spent during ascent.
        # 1. Initial velocity perturbation (turbulence at float arrival)
        init_vel = self.np_random.normal(0.0, INIT_VEL_SIGMA, size=real_dim)
        if self.dim == 2:
            init_vel[2] = 0.0  # no vertical velocity in 2D mode
        self._balloon.vel[:] = init_vel

        # 2. Gas imbalance (±INIT_GAS_FRAC_RANGE of neutral amount)
        gas_frac = self.np_random.uniform(-INIT_GAS_FRAC_RANGE, INIT_GAS_FRAC_RANGE)
        self._balloon.n_gas *= (1.0 + gas_frac)

        # 3. Ballast variation (some ballast may have been spent during ascent)
        ballast_loss = self.np_random.uniform(0.0, INIT_BALLAST_LOSS_MAX)
        self._balloon.ballast_mass = max(0.0, self._balloon.ballast_mass - ballast_loss)

        observation = self._get_obs()
        info = self._get_info()
        self._prev_distance = l2_distance(self._balloon.pos, self.goal, self.dim)

        self.prev_action = 1  # action index 1 = nothing (effect 0)

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
                from environments.core.jit_kernels import (
                    pressure_numba, density_numba, temperature_numba,
                    wind_sample_idx_numba, physics_step_numba,
                    sphere_area_from_volume, morrison_cd, dynamic_viscosity_numba,
                )
                # warm-up calls with small dummy inputs (compile once)
                _ = temperature_numba(1000.0)
                _ = pressure_numba(P0, 1000.0)
                _ = density_numba(P0, M_AIR, 1000.0)
                _ = dynamic_viscosity_numba(1000.0)
                _ = sphere_area_from_volume(1.0)
                _ = morrison_cd(1000.0)
                # wind warmup
                wf = self.wind
                _ = wind_sample_idx_numba(wf.x_centers[0], wf.y_centers[0], wf.z_centers[0],
                                        wf.x_range[0], wf.inv_dx, wf.y_range[0], wf.inv_dy, wf.z_range[0], wf.inv_dz,
                                        wf.cells, wf._fx_grid, wf._fy_grid)
                # physics warmup (dim-aware)
                pos = self._balloon.pos.astype(np.float64, copy=True)
                vel = self._balloon.vel.astype(np.float64, copy=True)
                wv = np.zeros_like(pos)
                ext = np.zeros_like(pos)
                physics_step_numba(pos, vel, DT, self._balloon.mass, G, RHO_0, self._balloon.volume, wv, ext, self.dim, VEL_MAX, P0, M_AIR)
            except Exception:
                pass  # if numba missing or compilation deferred, no problem
            _JIT_WARMED = True

        return observation, info

    # ------------------------------------------------------------------
    # Gym API – step
    # ------------------------------------------------------------------
    def step(self, action: int):
        assert self._balloon is not None, "Call reset() first."

        # Map action index to balloon control
        effect = int(self._action_lut[action])

        if effect == 1:       # drop ballast → ascend
            self._balloon.drop_ballast(BALLAST_DROP)
        elif effect == -1:    # vent gas → descend
            self._balloon.vent_gas(self.cfg["vent_rate"])

        wind = self.wind.sample(*self._full_coords(self._balloon.pos))
        self.last_wind[:] = wind    # Cache for obs (copies before buffer reuse)

        # Build wind velocity vector for relative-velocity drag
        if self.dim == 1:
            self._wind_vel_buf[0] = 0.0
            wind_vel = self._wind_vel_buf[:1]
        elif self.dim == 2:
            self._wind_vel_buf[0] = wind[0]
            self._wind_vel_buf[1] = wind[1]
            self._wind_vel_buf[2] = 0.0
            wind_vel = self._wind_vel_buf
        else:
            self._wind_vel_buf[0] = wind[0]
            self._wind_vel_buf[1] = wind[1]
            self._wind_vel_buf[2] = wind[2]
            wind_vel = self._wind_vel_buf

        # update balloon physics (wind passed as velocity, not force)
        self._balloon.update(DT, wind_vel=wind_vel)

        if self.dim == 2:          # keep altitude fixed
            self._balloon.pos[2] = self.z0
            self._balloon.vel[2] = 0.0

        # --- reward & termination -----------------------------------------
        self._time += 1
        alt = self._balloon.pos[-1] if self.dim in (1, 3) else self.z0
        self.last_pressure_norm = float(self._atmosphere.pressure(alt) / P_MAX)

        # Check termination conditions
        crashed = (self.dim in (1, 3)) and self._balloon.pos[-1] <= 0.0
        popped = (self.dim in (1, 3)) and self._balloon.pos[-1] >= self.z_range[1]
        deflated = self._balloon.is_deflated
        ballast_empty = self._balloon.is_ballast_empty
        out_of_bounds = False
        if self.dim in (2, 3):
            x, y = self._balloon.pos[0], self._balloon.pos[1]
            out_of_bounds = (
                x <= self.x_range[0] or x >= self.x_range[1]
                or y <= self.y_range[0] or y >= self.y_range[1]
            )
        terminated = crashed or popped or deflated or ballast_empty or out_of_bounds
        self.truncated = self._time >= self.cfg["time_max"]

        reward_total, reward_components, self._prev_distance = balloon_reward(
            balloon_pos=self._balloon.pos,
            goal_pos=self.goal,
            dim=self.dim,
            terminated=terminated,
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
            if self.truncated and not terminated:
                info["termination_reason"] = "All timesteps completed"
            elif crashed:
                info["termination_reason"] = "Crashed (altitude reached zero)"
            elif popped:
                info["termination_reason"] = "Popped (altitude limit exceeded)"
            elif deflated:
                info["termination_reason"] = "Deflated (helium fully lost)"
            elif ballast_empty:
                info["termination_reason"] = "Ballast exhausted (no ballast remaining)"
            elif out_of_bounds:
                info["termination_reason"] = "Out of bounds (XY limit exceeded)"

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
