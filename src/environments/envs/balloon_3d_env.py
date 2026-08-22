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

    Index | Effect | ZP (zero_pressure)              | SP (superpressure)
    ------|--------|---------------------------------|--------------------------
      0   |   -1   | Vent helium → descend           | Pump air in → descend
      1   |    0   | Do nothing                      | Do nothing
      2   |   +1   | Drop ballast → ascend           | Pump air out → ascend

ZP actions are irreversible (finite resource budget).
SP actions are reversible (air pumped in can be pumped out again).
The effect-to-index mapping is defined in `_action_lut`.

=========
Observation space
=========
A flat ``float32`` Box of width :data:`OBS_WIDTH` (143), **identical for
dim 1, 2 and 3** — fields that are meaningless in a given dimension are
zeroed, never omitted (Layer 1 contract §1, roadmap §2.1).  Later layers
populate the stub fields in place, so crossing a layer boundary costs a
retrain of weights rather than a re-architecture.

    indices   0..122 : wind column, 41 levels x 3 channels (mag, bearing,
                       uncertainty), 250 m apart, centred on the balloon's
                       own altitude and ordered low to high.
    indices 123..142 : 20 ambient scalars, see :data:`AMBIENT_FIELDS`.

Levels outside the operational band ``[ALT_SAFE_MIN, ALT_SAFE_MAX]`` carry
the *limit triple* ``(1.0, 1.0, 0.0)`` — "confidently, a fast wind straight
away from the goal" — so an unreachable altitude is self-evidently bad.

=========
Reward & Termination
=========
Perciatelli-style reward (inspired by Google BLE), scaled by a
multiplicative resource-consumption factor:
+ Inside station-keeping radius (default 10 km): flat reward of 1.0
+ Outside radius: exponential decay toward 0.0
+ Actuating costs a factor of ``0.97 - 0.3*omega`` held for a full
  decision interval (see ``_omega_steps``)
+ On termination: reward is 0.0 (forfeiting future reward is the penalty)
+ Reward range: [0, 1]

Altitude no longer terminates: the safety layer clamps the balloon into
``[ALT_SAFE_MIN, ALT_SAFE_MAX]`` and raises the ``at_alt_min`` /
``at_alt_max`` observation flags instead (roadmap §3.4).  Horizontal
bounds are soft — the balloon may drift arbitrarily far and ride a wind
back (roadmap §3.2) — with ``XY_ABORT`` as a pure numerical runaway guard.

Termination occurs when:
+ Balloon fully deflated / ballast exhausted (ZP only — SP envelope is sealed and reversible).
+ |x| or |y| exceeds XY_ABORT (2D/3D only, numerical guard).
+ Time limit reached (default TIME_MAX = 43,200 steps = 12 h).

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
import warnings
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from enum import Enum
from typing import Literal, Dict, Any, Tuple
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)
import pygame

try:
    from numba import njit as _njit
except Exception:  # pragma: no cover - numba is a hard dependency, but stay honest
    def _njit(**_kwargs):
        def _deco(fn):
            return fn
        return _deco

from environments.core.balloon import Balloon, BalloonSP
from environments.core.atmosphere import Atmosphere
from environments.core.wind_field import WindField
from environments.core.reward import balloon_reward, l2_distance
from environments.render.pygame_render import PygameRenderer
from environments.core.constants import (
    VOL_MAX, ALT_MAX, XY_MAX, VEL_MAX, VEL_Z_OBS_NORM, P_MAX, DT,
    P0, M_AIR, G, RHO_0,
    BALLAST_DROP, BALLAST_INITIAL,
    MIN_START_DISTANCE,
    INIT_VEL_SIGMA, INIT_GAS_FRAC_RANGE, INIT_BALLAST_LOSS_MAX,
    AIR_BLADDER_MAX,
    TIME_MAX, DECISION_INTERVAL,
    ALT_SAFE_MIN, ALT_SAFE_MAX, ALT_DEFAULT, XY_ABORT,
    WIND_COL_LEVELS, WIND_COL_SPACING, WIND_MAG_NORM,
    OBS_WIDTH, DIST_NORM,
    STATION_RADIUS, REWARD_HALFLIFE,
    STATION_RADIUS_1D, REWARD_HALFLIFE_1D, DIST_NORM_1D,
)

_JIT_WARMED = False  # whether numba JIT has been warmed up

# --------------------------------------------------------------------------- #
# Frozen observation layout — Layer 1 contract §1 (mirrored in agents.baselines)
# --------------------------------------------------------------------------- #
# This module is the *producer* of the layout; ``agents.baselines`` mirrors it
# so the agent package stays importable without the physics stack.  The two are
# cross-checked in ``tests/envs/test_observation.py`` — if they ever drift, the
# GreedyWind baseline silently misreads the column, which is exactly the kind of
# failure that does not announce itself.
WIND_COL_CHANNELS = 3
WIND_COL_WIDTH = WIND_COL_LEVELS * WIND_COL_CHANNELS      # 123
WIND_COL_CENTRE = WIND_COL_LEVELS // 2                    # 20 -> balloon altitude

CH_MAG = 0          # |wind| / WIND_MAG_NORM, in [0, 1]
CH_BEARING = 1      # signed angle(goal_dir -> wind_dir) / pi, in [-1, 1]
CH_UNCERTAINTY = 2  # STUB in Layer 1 (always 0.0); Layer 3 populates it

#: Levels outside the operational band read as "confidently, a fast wind
#: straight away from the goal", mirroring Loon's (1, 1, 0) encoding.
LIMIT_TRIPLE = (1.0, 1.0, 0.0)

AMBIENT_START = WIND_COL_WIDTH                            # 123
AMBIENT_FIELDS: Tuple[str, ...] = (
    "alt_norm",           # 123
    "goal_dz_norm",       # 124
    "pressure_norm",      # 125
    "vel_z_norm",         # 126
    "dist_norm",          # 127
    "heading_sin",        # 128
    "heading_cos",        # 129
    "resource_a",         # 130
    "resource_b",         # 131
    "volume_norm",        # 132
    "last_action_down",   # 133
    "last_action_stay",   # 134
    "last_action_up",     # 135
    "at_alt_min",         # 136
    "at_alt_max",         # 137
    "resource_a_low",     # 138
    "resource_b_low",     # 139
    "solar_elevation",    # 140  STUB (Layer 2)
    "solar_phase_sin",    # 141  STUB (Layer 2)
    "solar_phase_cos",    # 142  STUB (Layer 2)
)
AMBIENT_IDX: Dict[str, int] = {
    name: AMBIENT_START + i for i, name in enumerate(AMBIENT_FIELDS)
}

IDX_ALT_NORM = AMBIENT_IDX["alt_norm"]
IDX_GOAL_DZ_NORM = AMBIENT_IDX["goal_dz_norm"]
IDX_PRESSURE_NORM = AMBIENT_IDX["pressure_norm"]
IDX_VEL_Z_NORM = AMBIENT_IDX["vel_z_norm"]
IDX_DIST_NORM = AMBIENT_IDX["dist_norm"]
IDX_HEADING_SIN = AMBIENT_IDX["heading_sin"]
IDX_HEADING_COS = AMBIENT_IDX["heading_cos"]
IDX_RESOURCE_A = AMBIENT_IDX["resource_a"]
IDX_RESOURCE_B = AMBIENT_IDX["resource_b"]
IDX_VOLUME_NORM = AMBIENT_IDX["volume_norm"]
IDX_LAST_ACTION_DOWN = AMBIENT_IDX["last_action_down"]
IDX_LAST_ACTION_STAY = AMBIENT_IDX["last_action_stay"]
IDX_LAST_ACTION_UP = AMBIENT_IDX["last_action_up"]
IDX_AT_ALT_MIN = AMBIENT_IDX["at_alt_min"]
IDX_AT_ALT_MAX = AMBIENT_IDX["at_alt_max"]
IDX_RESOURCE_A_LOW = AMBIENT_IDX["resource_a_low"]
IDX_RESOURCE_B_LOW = AMBIENT_IDX["resource_b_low"]
IDX_SOLAR_ELEVATION = AMBIENT_IDX["solar_elevation"]
IDX_SOLAR_PHASE_SIN = AMBIENT_IDX["solar_phase_sin"]
IDX_SOLAR_PHASE_COS = AMBIENT_IDX["solar_phase_cos"]

assert AMBIENT_START + len(AMBIENT_FIELDS) == OBS_WIDTH, "observation layout drifted from OBS_WIDTH"

#: Layer 2 replaces these with a live diurnal model; Layer 1 is daytime-only.
SOLAR_ELEVATION_STUB = 0.7
SOLAR_PHASE_SIN_STUB = 0.0
SOLAR_PHASE_COS_STUB = 1.0

#: Below this fraction the corresponding ``resource_*_low`` flag is raised.
RESOURCE_LOW_FRAC = 0.05

# Operational band geometry, precomputed.
ALT_SAFE_SPAN = ALT_SAFE_MAX - ALT_SAFE_MIN
_INV_ALT_SAFE_SPAN = 1.0 / ALT_SAFE_SPAN
_INV_WIND_MAG_NORM = 1.0 / WIND_MAG_NORM
_INV_DIST_NORM = 1.0 / DIST_NORM
_INV_VEL_MAX = 1.0 / VEL_MAX
# Observation scaling only; the physics clamp stays VEL_MAX (roadmap §3.10).
_INV_VEL_Z_OBS_NORM = 1.0 / VEL_Z_OBS_NORM
_TWO_PI = 2.0 * math.pi
_EPS = 1e-9

# Action indices (Discrete(3)); the one-hot slots above are in this order.
ACTION_DOWN = 0
ACTION_STAY = 1
ACTION_UP = 2


@_njit(cache=True, fastmath=True)
def _write_wind_column(col, out, z, spacing, goal_dir, has_bearing,
                       alt_min, alt_max, inv_mag_norm):
    """Write the (mag, bearing) channels of the wind column into ``out``.

    One kernel rather than a chain of numpy ufuncs: this runs on *every*
    physics step (43,200 per episode), and at 41 levels the per-call overhead
    of a dozen ufuncs dwarfs the arithmetic — the numpy version measured ~30 us
    against ~4 us for the physics it accompanies.

    ``out`` is the float32 observation buffer, written in place at stride
    ``WIND_COL_CHANNELS``; the uncertainty channel is left untouched (it is a
    Layer 3 stub pinned to 0.0 at construction).
    """
    levels = col.shape[0]
    half = levels // 2
    two_pi = 2.0 * math.pi
    for i in range(levels):
        j = i * WIND_COL_CHANNELS
        zi = z + (i - half) * spacing
        if zi < alt_min or zi > alt_max:
            # Limit triple: unreachable altitudes read as a fast wind straight
            # away from the goal, so they are self-evidently bad.
            out[j + CH_MAG] = 1.0
            out[j + CH_BEARING] = 1.0
            continue

        fx = col[i, 0]
        fy = col[i, 1]
        mag = math.sqrt(fx * fx + fy * fy) * inv_mag_norm
        if mag > 1.0:
            mag = 1.0
        out[j + CH_MAG] = mag

        if has_bearing and mag > 1e-9:
            bearing = math.atan2(fy, fx) - goal_dir
            if bearing > math.pi:
                bearing -= two_pi
            elif bearing < -math.pi:
                bearing += two_pi
            out[j + CH_BEARING] = bearing / math.pi
        else:
            # 1D has no horizontal geometry, and a dead-calm level has no
            # direction; both read as "toward the goal", which is inert
            # because the magnitude channel is what weights it.
            out[j + CH_BEARING] = 0.0


class Actions(Enum):
    """Effect values for balloon altitude control (not action indices).

    The effect values are model-agnostic; the underlying operation depends
    on ``balloon_type``:

    Effect | ZP (zero_pressure)            | SP (superpressure)
    -------|-------------------------------|-------------------------------
      +1   | drop_ballast  → ascend        | pump_out (air) → ascend
       0   | nothing                       | nothing
      -1   | vent_gas (helium) → descend   | pump_in  (air) → descend

    ZP actions are irreversible (finite resource budget).
    SP actions are reversible (air pumped in can be pumped out again).
    """
    drop_ballast = 1   # ZP: drop sand ballast;  SP: pump air out
    nothing = 0
    vent = -1          # ZP: vent helium;         SP: pump air in


class Balloon3DEnv(gym.Env):
    """Unified 1D/2D/3D balloon environment."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 150}

    # sensible defaults – override by *config*
    DEFAULTS: Dict[str, Any] = dict(
        dim=3,                    # 1, 2 or 3 dimensions
        time_max=TIME_MAX,        # physics steps per episode (12 hours at DT=1s)
        decision_interval=DECISION_INTERVAL,  # physics steps one agent decision covers
        x_range=(-XY_MAX, XY_MAX),
        y_range=(-XY_MAX, XY_MAX),
        z_range=(0.0, ALT_MAX),
        wind_mag=5.0,            # nominal wind speed [m/s] (randomised per episode)
        wind_cells=20,           # horizontal (x, y) wind-grid resolution
        wind_cells_z=None,       # vertical wind-grid resolution; None -> Nyquist for WIND_COL_SPACING
        vent_rate_moles=None,     # reserved — vent rate now fixed in constants.py (VENT_RATE_MOLES)
        window_size=(800, 600),  # pygame window (w,h)
        wind_pattern="split_fork",      # wind pattern: "sinusoid", "linear_right", "linear_up", "split_fork", "altitude_shear", "altitude_shear_2d"
        wind_layers=2,                # number of full wind rotations over altitude range (altitude_shear_2d only)
        balloon_type="zero_pressure",  # "zero_pressure" (ZP + sand ballast) or "superpressure" (SP + air ballast)
        # --- scenario randomisation (roadmap §3.9) -----------------------
        randomise_scenario=True,      # draw a fresh scenario from np_random on reset
        wind_mag_range=None,          # (lo, hi) m/s; None -> (0.5x, 1.5x) the field's nominal
        wind_layers_range=(1.0, 3.0),  # rotations over z_range, altitude_shear_2d only
        goal_radius=15_000.0,         # goal drawn uniformly in a disc of this radius
        spawn_dist_range=(2_000.0, 30_000.0),   # horizontal spawn offset from the goal
        spawn_alt_range=(ALT_SAFE_MIN + 1_000.0, ALT_SAFE_MAX - 1_000.0),
        goal_alt_range=(ALT_SAFE_MIN + 1_000.0, ALT_SAFE_MAX - 1_000.0),  # 1D only
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
        self._balloon_type: str = cfg.get("balloon_type", "zero_pressure")

        # Reward/observation distance scales.  1-D measures |dz| against a target
        # altitude and is bounded by the safety band, so it needs its own (much
        # tighter) scales — see constants.py.  Overridable via config.
        _one_d = self.dim == 1
        self._station_radius: float = float(cfg.get(
            "station_radius", STATION_RADIUS_1D if _one_d else STATION_RADIUS))
        self._reward_halflife: float = float(cfg.get(
            "reward_halflife", REWARD_HALFLIFE_1D if _one_d else REWARD_HALFLIFE))
        self._inv_dist_norm: float = 1.0 / float(cfg.get(
            "dist_norm", DIST_NORM_1D if _one_d else DIST_NORM))
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
        # Fixed altitude for dim==2 and nominal goal altitude for dim==3.  The
        # operational band midpoint, not the z_range midpoint: the band is what
        # the balloon may actually occupy.
        self.z0 = ALT_DEFAULT

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
        # The vertical resolution is decoupled from the horizontal one: the
        # observation samples wind every WIND_COL_SPACING metres, so a cubic
        # grid coarse enough to be cheap in x/y would hand the agent 41 levels
        # carrying only a handful of distinct values.  ``wind_cells_z=None``
        # asks WindField for the Nyquist count for that spacing.
        self.wind = WindField(
            x_range=self.x_range,
            y_range=self.y_range,
            z_range=self.z_range,
            cells=cfg["wind_cells"],
            pattern=self.cfg.get("wind_pattern", "sinusoid"),
            default_mag=cfg["wind_mag"],
            wind_cfg_path=self.wind_cfg_path,
            wind_layers=cfg["wind_layers"],
            cells_z=cfg.get("wind_cells_z"),
        )
        # Nominal magnitude *after* the winds.json catalogue has had its say;
        # the per-episode randomisation is expressed relative to it.
        self._wind_mag_nominal = float(self.wind.mag)
        self._wind_layers_nominal = float(cfg["wind_layers"])
        mag_range = cfg.get("wind_mag_range")
        if mag_range is None:
            mag_range = (0.5 * self._wind_mag_nominal, 1.5 * self._wind_mag_nominal)
        self._wind_mag_range = (float(mag_range[0]), float(mag_range[1]))

        self.x_centers = self.wind.x_centers
        self.y_centers = self.wind.y_centers
        self.z_centers = self.wind.z_centers
        self.wind_cells = self.wind.cells

        # ------------------------------------------------------------------
        # Spaces
        # ------------------------------------------------------------------
        obs_low, obs_high = self._build_observation_bounds()
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self._obs_size = self.observation_space.shape[0]
        self._obs_buf = np.zeros(self._obs_size, dtype=np.float32)
        # Channels that never change in Layer 1 are written once, here.
        self._obs_buf[CH_UNCERTAINTY:WIND_COL_WIDTH:WIND_COL_CHANNELS] = LIMIT_TRIPLE[2]
        self._obs_buf[IDX_SOLAR_ELEVATION] = SOLAR_ELEVATION_STUB
        self._obs_buf[IDX_SOLAR_PHASE_SIN] = SOLAR_PHASE_SIN_STUB
        self._obs_buf[IDX_SOLAR_PHASE_COS] = SOLAR_PHASE_COS_STUB

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
        self._init_n_gas = 1.0
        self.prev_action = ACTION_STAY
        self._seed: int | None = None
        self._scenario: Dict[str, Any] = {}
        self._prev_distance = 0.0

        # Altitude-safety flags for the current step
        self._at_alt_min = False
        self._at_alt_max = False

        # Resource-penalty hold — see `_charge_resources`.
        self._decision_interval = int(cfg["decision_interval"])
        self._omega = 0.0
        self._omega_steps = 0

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
    def _build_observation_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Per-field bounds for the frozen layout (contract §1).

        Deliberately *not* a blanket 0..1 box: the bearing channel and five of
        the ambient scalars are signed, and an observation space that lies
        about that is a silent trap for anything that samples or normalises it.
        """
        low = np.zeros(OBS_WIDTH, dtype=np.float32)
        high = np.ones(OBS_WIDTH, dtype=np.float32)

        # wind column: mag [0,1], bearing [-1,1], uncertainty [0,1]
        low[CH_BEARING:WIND_COL_WIDTH:WIND_COL_CHANNELS] = -1.0

        for name in ("goal_dz_norm", "vel_z_norm", "heading_sin", "heading_cos",
                     "solar_phase_sin", "solar_phase_cos"):
            low[AMBIENT_IDX[name]] = -1.0

        return low, high

    def _resource_fractions(self) -> Tuple[float, float]:
        """``(resource_a, resource_b)`` in [0, 1] for the active balloon model.

        ZP: remaining ballast / remaining helium, both irreversible.
        SP: bladder fill and its complement — the fill fraction alone would
        leave "how much further can I descend" implicit.
        """
        b = self._balloon
        if self._balloon_type == "superpressure":
            fill = b.air_bladder_mass / AIR_BLADDER_MAX
            fill = 0.0 if fill < 0.0 else (1.0 if fill > 1.0 else fill)
            return fill, 1.0 - fill
        ballast = min(b.ballast_mass / BALLAST_INITIAL, 1.0)
        gas = min(b.n_gas / self._init_n_gas, 1.0)
        return max(ballast, 0.0), max(gas, 0.0)

    def _fill_wind_column(self, x: float, y: float, z: float, goal_dir: float) -> None:
        """Write the 41x3 egocentric wind column into the observation buffer.

        ``goal_dir`` is the world-frame angle of the balloon→goal vector; the
        bearing channel is measured *relative* to it so the column is
        rotation-invariant: "would this level carry me toward the station" is
        the same question at every compass heading.
        """
        col = self.wind.sample_column(x, y, z)   # (levels, 2) — reused buffer
        _write_wind_column(
            col, self._obs_buf, z, WIND_COL_SPACING, goal_dir, self.dim != 1,
            ALT_SAFE_MIN, ALT_SAFE_MAX, _INV_WIND_MAG_NORM,
        )

    def _get_obs(self) -> np.ndarray:
        """Pack the frozen 143-wide observation (contract §1)."""
        buf = self._obs_buf
        b = self._balloon
        dim = self.dim
        x, y, z = self._full_coords(b.pos)

        # --- goal geometry ------------------------------------------------
        if dim == 1:
            goal_z = float(self.goal[0])
            distance = abs(z - goal_z)
            goal_dir = 0.0
            heading_sin = 0.0
            heading_cos = 0.0
            goal_dz = (goal_z - z) * _INV_ALT_SAFE_SPAN
        else:
            dx = float(self.goal[0]) - x
            dy = float(self.goal[1]) - y
            distance = math.hypot(dx, dy)
            goal_dir = math.atan2(dy, dx) if distance > _EPS else 0.0
            heading_sin = math.sin(goal_dir)
            heading_cos = math.cos(goal_dir)
            # Altitude is the control mechanism in 2D/3D, not an objective.
            goal_dz = 0.0

        self._fill_wind_column(x, y, z, goal_dir)

        # --- ambient scalars ----------------------------------------------
        alt_norm = (z - ALT_SAFE_MIN) * _INV_ALT_SAFE_SPAN
        buf[IDX_ALT_NORM] = 0.0 if alt_norm < 0.0 else (1.0 if alt_norm > 1.0 else alt_norm)
        buf[IDX_GOAL_DZ_NORM] = -1.0 if goal_dz < -1.0 else (1.0 if goal_dz > 1.0 else goal_dz)

        pressure_norm = float(self._atmosphere.pressure(z)) / P_MAX
        self.last_pressure_norm = pressure_norm
        buf[IDX_PRESSURE_NORM] = min(pressure_norm, 1.0)

        vel_z = float(b.vel[-1]) * _INV_VEL_Z_OBS_NORM
        buf[IDX_VEL_Z_NORM] = -1.0 if vel_z < -1.0 else (1.0 if vel_z > 1.0 else vel_z)

        buf[IDX_DIST_NORM] = min(distance * self._inv_dist_norm, 1.0)
        buf[IDX_HEADING_SIN] = heading_sin
        buf[IDX_HEADING_COS] = heading_cos

        res_a, res_b = self._resource_fractions()
        buf[IDX_RESOURCE_A] = res_a
        buf[IDX_RESOURCE_B] = res_b
        buf[IDX_RESOURCE_A_LOW] = 1.0 if res_a < RESOURCE_LOW_FRAC else 0.0
        buf[IDX_RESOURCE_B_LOW] = 1.0 if res_b < RESOURCE_LOW_FRAC else 0.0

        buf[IDX_VOLUME_NORM] = min(b.volume / VOL_MAX, 1.0)

        # One-hot of the previous action; the three slots are contiguous and in
        # action-index order, so the offset *is* the action.
        buf[IDX_LAST_ACTION_DOWN] = 0.0
        buf[IDX_LAST_ACTION_STAY] = 0.0
        buf[IDX_LAST_ACTION_UP] = 0.0
        buf[IDX_LAST_ACTION_DOWN + int(self.prev_action)] = 1.0

        buf[IDX_AT_ALT_MIN] = 1.0 if self._at_alt_min else 0.0
        buf[IDX_AT_ALT_MAX] = 1.0 if self._at_alt_max else 0.0

        # solar stubs were written at construction and never change in Layer 1.
        return buf.copy()

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
            x, y, z = pos[:3]
            return float(x), float(y), float(z)

    # ------------------------------------------------------------------
    # Decision interval
    # ------------------------------------------------------------------
    def set_decision_interval(self, decision_interval: int) -> None:
        """Tell the env how many physics steps one agent decision covers.

        The env holds the resource penalty for exactly this many steps, so it
        must agree with :class:`~environments.wrappers.decision_interval.DecisionIntervalWrapper`.
        The wrapper calls this on construction; nothing else needs to.
        """
        self._decision_interval = max(1, int(decision_interval))
        self.cfg["decision_interval"] = self._decision_interval

    # ------------------------------------------------------------------
    # Scenario randomisation (roadmap §3.9)
    # ------------------------------------------------------------------
    def _draw_scenario(self) -> Dict[str, Any]:
        """Draw goal, spawn and wind-pattern parameters from ``self.np_random``.

        Everything that distinguishes one episode from another is drawn here
        and recorded in the returned dict, so ``env.reset(seed=s)`` replays a
        scenario exactly and ``info["scenario"]`` describes what was flown.
        Without this the agent trains on one scenario repeated: a goal pinned
        to the origin, spawn in the central 50% of the box, and a single fixed
        shear.
        """
        rng = self.np_random
        cfg = self.cfg
        pattern = cfg.get("wind_pattern", "sinusoid")
        randomise = bool(cfg.get("randomise_scenario", True))

        # --- wind field ----------------------------------------------------
        if randomise:
            wind_mag = float(rng.uniform(*self._wind_mag_range))
        else:
            wind_mag = self._wind_mag_nominal
        if randomise and pattern == "altitude_shear_2d":
            lo, hi = cfg["wind_layers_range"]
            wind_layers = float(rng.uniform(lo, hi))
        else:
            wind_layers = self._wind_layers_nominal

        if wind_mag != self.wind.mag or wind_layers != self.wind.wind_layers:
            self.wind.mag = wind_mag
            self.wind.wind_layers = wind_layers
            self.wind._build_grid()

        # --- goal & spawn --------------------------------------------------
        alt_lo, alt_hi = cfg["spawn_alt_range"]
        if self.dim == 1:
            goal_lo, goal_hi = cfg["goal_alt_range"]
            if randomise:
                goal_z = float(rng.uniform(goal_lo, goal_hi))
                spawn_z = float(rng.uniform(alt_lo, alt_hi))
                for _ in range(100):
                    if abs(spawn_z - goal_z) > MIN_START_DISTANCE:
                        break
                    spawn_z = float(rng.uniform(alt_lo, alt_hi))
                else:  # pragma: no cover - only if the ranges are pathological
                    spawn_z = float(np.clip(goal_z + 2.0 * MIN_START_DISTANCE, alt_lo, alt_hi))
            else:
                goal_z = 0.5 * (goal_lo + goal_hi)
                spawn_z = ALT_DEFAULT
            goal = np.array([goal_z], dtype=np.float64)
            spawn = np.array([spawn_z], dtype=np.float64)
        else:
            if randomise:
                # Uniform over the *area* of the disc, not over the radius —
                # otherwise the goal clusters at the centre.
                goal_r = math.sqrt(float(rng.uniform(0.0, 1.0))) * float(cfg["goal_radius"])
                goal_th = float(rng.uniform(0.0, _TWO_PI))
                gx, gy = goal_r * math.cos(goal_th), goal_r * math.sin(goal_th)
                # Spawn on a ring around the goal: guarantees the minimum start
                # distance by construction instead of by rejection sampling.
                dist = float(rng.uniform(*cfg["spawn_dist_range"]))
                th = float(rng.uniform(0.0, _TWO_PI))
                sx, sy = gx + dist * math.cos(th), gy + dist * math.sin(th)
                spawn_z = float(rng.uniform(alt_lo, alt_hi))
            else:
                gx = gy = 0.0
                dist = float(cfg["spawn_dist_range"][0])
                sx, sy = dist, 0.0
                spawn_z = ALT_DEFAULT

            if self.dim == 2:
                goal = np.array([gx, gy], dtype=np.float64)
                spawn = np.array([sx, sy], dtype=np.float64)
            else:
                goal = np.array([gx, gy, self.z0], dtype=np.float64)
                spawn = np.array([sx, sy, spawn_z], dtype=np.float64)

        return {
            "seed": self._seed,
            "dim": self.dim,
            "balloon_type": self._balloon_type,
            "wind_pattern": pattern,
            "wind_mag": wind_mag,
            "wind_layers": wind_layers,
            "goal": tuple(float(v) for v in goal),
            "spawn": tuple(float(v) for v in spawn),
            "_goal": goal,
            "_spawn": spawn,
        }

    # ------------------------------------------------------------------
    # Gym API – reset
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._seed = int(seed)

        self._time = 0
        self.truncated = False
        self.final_obs = None
        self._at_alt_min = False
        self._at_alt_max = False
        self._omega = 0.0
        self._omega_steps = 0
        self.prev_action = ACTION_STAY

        self._atmosphere = Atmosphere()

        scenario = self._draw_scenario()
        goal = scenario.pop("_goal")
        pos0 = scenario.pop("_spawn")
        self._scenario = scenario
        self.goal = goal

        real_dim = 3 if self.dim == 2 else self.dim
        # For 2D mode, balloon is internally 3D with fixed altitude z0
        init_pos = np.append(pos0, self.z0) if self.dim == 2 else pos0

        if self._balloon_type == "superpressure":
            self._balloon = BalloonSP(dim=real_dim,
                                      atmosphere=self._atmosphere,
                                      position=init_pos,
                                      velocity=[0.0] * real_dim,
                                      )
            self._init_n_gas = 1.0  # unused for SP; set to avoid AttributeError

            # --- SP domain randomisation ---
            # 1. Initial velocity perturbation (turbulence at float arrival)
            init_vel = self.np_random.normal(0.0, INIT_VEL_SIGMA, size=real_dim)
            if self.dim == 2:
                init_vel[2] = 0.0
            self._balloon.vel[:] = init_vel

            # 2. Bladder perturbation (±INIT_GAS_FRAC_RANGE × AIR_BLADDER_MAX around midpoint)
            bladder_delta = self.np_random.uniform(-INIT_GAS_FRAC_RANGE, INIT_GAS_FRAC_RANGE) * AIR_BLADDER_MAX
            self._balloon.air_bladder_mass = float(
                np.clip(self._balloon.air_bladder_mass + bladder_delta, 0.0, AIR_BLADDER_MAX)
            )
        else:
            self._balloon = Balloon(dim=real_dim,
                                    atmosphere=self._atmosphere,
                                    position=init_pos,
                                    velocity=[0.0] * real_dim,
                                    )

            # Store neutral-buoyancy gas amount for normalising the gas observation
            self._init_n_gas = self._balloon.n_gas

            # --- ZP domain randomisation ---
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

        self._prev_distance = l2_distance(self._balloon.pos, self.goal, self.dim)
        self.last_wind[:] = 0.0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._ensure_renderer()
            self.renderer.draw(self._render_state())

        # Warm up JIT to compile before training
        global _JIT_WARMED
        if not _JIT_WARMED:
            try:
                from environments.core.jit_kernels import (
                    pressure_numba, density_numba, temperature_numba,
                    gas_temperature_numba,
                    wind_sample_idx_numba, wind_sample_column_numba,
                    physics_step_numba,
                    sphere_area_from_volume, morrison_cd, dynamic_viscosity_numba,
                )
                # warm-up calls with small dummy inputs (compile once)
                _ = temperature_numba(1000.0)
                _ = gas_temperature_numba(1000.0)
                _ = pressure_numba(P0, 1000.0)
                _ = density_numba(P0, M_AIR, 1000.0)
                _ = dynamic_viscosity_numba(1000.0)
                _ = sphere_area_from_volume(1.0)
                _ = morrison_cd(1000.0)
                # wind warmup
                wf = self.wind
                _ = wind_sample_idx_numba(wf.x_centers[0], wf.y_centers[0], wf.z_centers[0],
                                        wf.x_range[0], wf.inv_dx, wf.y_range[0], wf.inv_dy, wf.z_range[0], wf.inv_dz,
                                        wf.cells, wf.cells_z, wf._fx_grid, wf._fy_grid)
                wind_sample_column_numba(0.0, 0.0, 20000.0, WIND_COL_SPACING,
                                         wf.x_range[0], wf.x_range[1], wf.inv_dx,
                                         wf.y_range[0], wf.y_range[1], wf.inv_dy,
                                         wf.z_range[0], wf.z_range[1], wf.inv_dz,
                                         wf.cells, wf.cells_z, wf._fx_grid, wf._fy_grid,
                                         np.zeros((WIND_COL_LEVELS, 2)))
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
    # Resource accounting
    # ------------------------------------------------------------------
    def _charge_resources(self, consumed_frac: float) -> float:
        """Return the resource ``omega`` in force for this physics step.

        The subtlety this exists for: with a decision interval the agent's
        action fires on **one** physics step and the other 59 run as NOOPs.
        Charging omega only on the firing step spreads a 3% penalty over 60
        rewards — 0.05% per step — which shapes nothing.  So a consuming action
        latches its omega for a full decision interval, making the penalty a
        property of the *decision* rather than of one arbitrary physics tick.

        Implemented here rather than in the wrapper so the env is correct
        whether or not it is wrapped: the wrapper only decides *when* actions
        fire, it must not own what they cost.
        """
        if consumed_frac > 0.0:
            self._omega = max(self._omega, consumed_frac) if self._omega_steps > 0 else consumed_frac
            self._omega_steps = self._decision_interval

        if self._omega_steps <= 0:
            return 0.0

        omega = self._omega
        self._omega_steps -= 1
        if self._omega_steps == 0:
            self._omega = 0.0
        return omega

    def _actuate(self, effect: int) -> float:
        """Apply the control effect and return the fraction of budget spent.

        The fraction is measured from the balloon's *actual* state change, so
        an action that hits an empty tank (or a full bladder) costs nothing.
        """
        b = self._balloon
        if effect == 0:
            return 0.0

        if self._balloon_type == "superpressure":
            before = b.air_bladder_mass
            if effect == 1:      # pump air out → lighter → ascend
                b.pump_out()
            else:                # pump air in  → heavier → descend
                b.pump_in()
            return abs(b.air_bladder_mass - before) / AIR_BLADDER_MAX

        if effect == 1:          # drop ballast → ascend
            before = b.ballast_mass
            b.drop_ballast(BALLAST_DROP)
            return max(0.0, before - b.ballast_mass) / BALLAST_INITIAL

        before = b.n_gas         # vent gas → descend
        b.vent_gas()
        return max(0.0, before - b.n_gas) / max(self._init_n_gas, _EPS)

    # ------------------------------------------------------------------
    # Gym API – step
    # ------------------------------------------------------------------
    def step(self, action: int):
        assert self._balloon is not None, "Call reset() first."

        # Map action index to balloon control
        effect = int(self._action_lut[action])
        omega = self._charge_resources(self._actuate(effect))

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

        # --- altitude safety layer ----------------------------------------
        # Clamp into the operational band instead of terminating: the agent
        # learns the constraint from the flags rather than dying at it
        # (roadmap §3.4).
        self._at_alt_min, self._at_alt_max = self._balloon.clamp_altitude(
            ALT_SAFE_MIN, ALT_SAFE_MAX
        )

        # --- reward & termination -----------------------------------------
        self._time += 1

        deflated = self._balloon.is_deflated
        ballast_empty = self._balloon.is_ballast_empty
        runaway = False
        if self.dim in (2, 3):
            # Soft horizontal bounds: no distance termination at all, so the
            # agent can discover that riding an unfavourable wind out and back
            # is worth it.  XY_ABORT only catches numerical runaway.
            runaway = bool(abs(self._balloon.pos[0]) > XY_ABORT
                           or abs(self._balloon.pos[1]) > XY_ABORT)
        terminated = bool(deflated or ballast_empty or runaway)
        self.truncated = self._time >= self.cfg["time_max"]

        reward_total, reward_components, self._prev_distance = balloon_reward(
            balloon_pos=self._balloon.pos,
            goal_pos=self.goal,
            dim=self.dim,
            terminated=terminated,
            resource_consumed_frac=omega,
            station_radius=self._station_radius,
            reward_halflife=self._reward_halflife,
        )

        components_for_info = dict(reward_components)
        components_for_info["total"] = reward_total

        self.prev_action = int(action)

        info = self._get_info()
        info["reward_components"] = components_for_info
        info["resource_consumed_frac"] = omega

        # --- collect observation & info -----------------------------------
        obs = self._get_obs()
        if terminated or self.truncated:
            self.final_obs = obs
            info["terminal_observation"] = self.final_obs
            if self.truncated and not terminated:
                info["termination_reason"] = "All timesteps completed"
            elif deflated:
                info["termination_reason"] = "Deflated (helium fully lost)"
            elif ballast_empty:
                info["termination_reason"] = "Ballast exhausted (no ballast remaining)"
            elif runaway:
                info["termination_reason"] = "Numerical abort (|x| or |y| beyond XY_ABORT)"

        if self.render_mode == "human":
            self._ensure_renderer()
            self.renderer.draw(self._render_state())

        return obs, reward_total, terminated, self.truncated, info

    # ------------------------------------------------------------------
    def _get_info(self) -> Dict[str, Any]:
        """Per-step info.

        ``distance`` is authoritative for evaluation (contract §5): TWR is
        computed from it directly rather than re-derived from positions, so it
        must be present on **every** step.
        """
        return {
            "TimeLimit.truncated": self.truncated,
            "terminal_observation": self.final_obs,
            "distance": float(self._prev_distance),
            "scenario": self._scenario,
        }

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------
    def _render_state(self) -> Dict[str, Any]:
        """State dict for :class:`PygameRenderer`.

        The renderer reads ``goal_pos[:2]`` for the map and ``goal_pos[-1]``
        for the altitude bar, so it needs a 3-vector goal.  In 1D and 2D the
        goal has no such shape — pad it here rather than in the renderer,
        which this package does not own.
        """
        if self.dim == 1:
            goal = np.array([0.0, 0.0, float(self.goal[0])], dtype=np.float64)
        elif self.dim == 2:
            goal = np.array([float(self.goal[0]), float(self.goal[1]), self.z0], dtype=np.float64)
        else:
            goal = self.goal.copy()
        return dict(
            dim=self.dim,
            balloon_pos=self._balloon.pos.copy(),
            goal_pos=goal,
            z0=self.z0,
            wind_sampler=self.wind.sample,
        )

    def render(self):
        if self.render_mode == "human":
            self._ensure_renderer()
            self.renderer.draw(self._render_state())

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer.window = None
            self.renderer.clock = None

    def _ensure_renderer(self):
        """Create renderer instance on first use."""
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


class BalloonSP3DEnv(Balloon3DEnv):
    """Entry point for superpressure balloon environments.

    Identical to Balloon3DEnv with ``balloon_type='superpressure'`` locked in
    so that passing additional config kwargs to ``gym.make()`` (e.g.
    ``config={"time_max": 3600}``) does not accidentally revert to the default
    zero-pressure model.  Any config items supplied at make-time are merged on
    top of the SP default.
    """

    def __init__(self, dim: int = 3, render_mode=None, *, config=None):
        super().__init__(
            dim=dim,
            render_mode=render_mode,
            config={"balloon_type": "superpressure", **(config or {})},
        )
