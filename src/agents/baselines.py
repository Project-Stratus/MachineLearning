"""
agents.baselines
----------------

Non-learning reference policies for Layer 1 evaluation.

Layer 1's exit criterion is "beats passive-drift and random baselines on a
frozen held-out scenario set". A score is meaningless without a floor to
compare it against, so this module supplies four:

``PassiveDriftAgent``
    Never actuates. The "do nothing" floor — whatever the wind field hands
    you for free. Also the cheapest possible resource policy, so it is the
    reference any resource-spending controller has to justify itself against.

``RandomAgent``
    Uniform over the three actions, seeded. Distinct from passive drift
    because random actuation still *moves* the balloon through the shear,
    and on a strongly sheared field that alone can beat doing nothing.

``GreedyWindAgent``
    The heuristic floor: read the wind column out of the observation, score
    every reachable level, and climb/descend toward the best one. This is the
    baseline that matters. Loon's equivalent heuristic scored 40.5% TWR50
    against their learned controller's 55.1% — a learned agent that cannot
    clear the heuristic has learned nothing worth having.

``BangBangAgent`` (1-D only)
    Altitude-hold bang-bang with a velocity lead term. ``GreedyWindAgent`` has
    no horizontal objective to exploit in 1D and degenerates there, so 1D needs
    its own heuristic floor — otherwise its only reference is passive drift and
    the exit criterion is unmeasurable. Use :func:`baselines_for_dim` rather
    than iterating :data:`BASELINES` directly, so this one stays out of the 2D
    and 3D tables.

Each exposes the SB3 policy interface

    predict(observation, state=None, episode_start=None, deterministic=True)
        -> (action, None)

so they drop into the same evaluation loop as a trained ``QRDQN`` without any
branching at the call site (see :mod:`agents.evaluation`).

This module is also the agent-side single source of truth for the frozen
observation layout (Layer 1 contract §1): every index lives in the constant
block below and nothing else in ``src/agents`` should hardcode an offset.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# Observation layout — Layer 1 contract §1. Width 143, identical for dim 1/2/3.
# --------------------------------------------------------------------------- #
# The physics package owns these numbers; we mirror them so that `agents` stays
# importable without the environment stack (numba, pygame, ...) — useful for
# scoring saved traces offline and for tests that stub the env out. The values
# below are the contract values and must not drift from constants.py.
try:  # pragma: no cover - trivial import shim
    from environments.core.constants import (
        WIND_COL_LEVELS,
        WIND_COL_SPACING,
        WIND_MAG_NORM,
        OBS_WIDTH,
        STATION_RADIUS,
        DIST_NORM,
        STATION_RADIUS_1D,
        ALT_SAFE_MIN,
        ALT_SAFE_MAX,
        VEL_Z_OBS_NORM,
    )
except ImportError:  # pragma: no cover - environment package unavailable
    WIND_COL_LEVELS = 41
    WIND_COL_SPACING = 250.0
    WIND_MAG_NORM = 30.0
    OBS_WIDTH = 143
    STATION_RADIUS = 10_000.0
    DIST_NORM = 100_000.0
    STATION_RADIUS_1D = 500.0
    ALT_SAFE_MIN = 15_000.0
    ALT_SAFE_MAX = 25_000.0
    VEL_Z_OBS_NORM = 5.0

#: Span of the operational band (m). ``goal_dz_norm`` and ``alt_norm`` are both
#: normalised by this, so it is the factor that turns them back into metres.
ALT_SAFE_SPAN = ALT_SAFE_MAX - ALT_SAFE_MIN

# --- wind column: indices 0..122, 41 levels x 3 channels, ordered low to high.
WIND_COL_CHANNELS = 3
WIND_COL_WIDTH = WIND_COL_LEVELS * WIND_COL_CHANNELS  # 123
WIND_COL_CENTRE = WIND_COL_LEVELS // 2  # 20 -> balloon's own altitude

CH_MAG = 0  # |wind| / WIND_MAG_NORM, in [0, 1]
CH_BEARING = 1  # signed angle(goal_dir -> wind_dir) / pi, in [-1, 1]; 0 == toward goal
CH_UNCERTAINTY = 2  # STUB in Layer 1 (always 0.0); Layer 3 populates it

#: Levels outside ``[ALT_SAFE_MIN, ALT_SAFE_MAX]`` carry this triple — read as
#: "confidently, a fast wind straight away from the goal". Unreachable levels are
#: therefore self-evidently bad even to a policy that does not special-case them,
#: but :class:`GreedyWindAgent` masks them outright so they can never be chosen.
LIMIT_TRIPLE = (1.0, 1.0, 0.0)

# --- ambient block: indices 123..142, 20 scalars in exactly this order.
AMBIENT_START = WIND_COL_WIDTH
AMBIENT_FIELDS: tuple[str, ...] = (
    "alt_norm",  # 123
    "goal_dz_norm",  # 124
    "pressure_norm",  # 125
    "vel_z_norm",  # 126
    "dist_norm",  # 127
    "heading_sin",  # 128
    "heading_cos",  # 129
    "resource_a",  # 130
    "resource_b",  # 131
    "volume_norm",  # 132
    "last_action_down",  # 133
    "last_action_stay",  # 134
    "last_action_up",  # 135
    "at_alt_min",  # 136
    "at_alt_max",  # 137
    "resource_a_low",  # 138
    "resource_b_low",  # 139
    "solar_elevation",  # 140  STUB (Layer 2)
    "solar_phase_sin",  # 141  STUB (Layer 2)
    "solar_phase_cos",  # 142  STUB (Layer 2)
)
AMBIENT_IDX: dict[str, int] = {
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

# --------------------------------------------------------------------------- #
# Action layout — Discrete(3), matching `Balloon3DEnv._action_lut = [-1, 0, +1]`
# and the last-action one-hot ordering above.
# --------------------------------------------------------------------------- #
ACTION_DOWN = 0  # ZP: vent helium.   SP: pump air in.
ACTION_STAY = 1  # do nothing
ACTION_UP = 2  # ZP: drop ballast.  SP: pump air out.
N_ACTIONS = 3


# --------------------------------------------------------------------------- #
# Observation accessors
# --------------------------------------------------------------------------- #
def wind_column(obs: np.ndarray) -> np.ndarray:
    """View of the wind column as ``(WIND_COL_LEVELS, 3)``, ordered low to high.

    Row ``i`` describes the wind at ``z_balloon + (i - WIND_COL_CENTRE) *
    WIND_COL_SPACING``; row ``WIND_COL_CENTRE`` is the balloon's own altitude.
    """
    arr = np.asarray(obs)
    return arr[..., :WIND_COL_WIDTH].reshape(
        *arr.shape[:-1], WIND_COL_LEVELS, WIND_COL_CHANNELS
    )


def ambient(obs: np.ndarray) -> np.ndarray:
    """View of the 20 ambient scalars (indices 123..142)."""
    return np.asarray(obs)[..., AMBIENT_START:]


def level_offsets_m() -> np.ndarray:
    """Altitude of each wind-column level *relative to the balloon*, in metres."""
    return (
        np.arange(WIND_COL_LEVELS, dtype=np.float64) - WIND_COL_CENTRE
    ) * WIND_COL_SPACING


def limit_mask(column: np.ndarray) -> np.ndarray:
    """Boolean mask of levels carrying the limit triple (i.e. outside the band).

    Compared with a tight tolerance rather than exactly, because the observation
    is ``float32`` and may be widened to ``float64`` before it reaches us.
    """
    col = np.asarray(column, dtype=np.float64)
    return (
        np.isclose(col[..., CH_MAG], LIMIT_TRIPLE[0], atol=1e-6)
        & np.isclose(col[..., CH_BEARING], LIMIT_TRIPLE[1], atol=1e-6)
        & np.isclose(col[..., CH_UNCERTAINTY], LIMIT_TRIPLE[2], atol=1e-6)
    )


# --------------------------------------------------------------------------- #
# Base class
# --------------------------------------------------------------------------- #
class BaselinePolicy:
    """SB3-compatible policy base: implement :meth:`_action` and you are done.

    Handles the two observation shapes an evaluation loop can hand a policy:
    a single ``(OBS_WIDTH,)`` observation from a plain gymnasium env, and a
    batched ``(n_envs, OBS_WIDTH)`` observation from a ``VecEnv``. Returns
    ``(action, None)`` in both cases, mirroring SB3's ``predict``: the second
    element is the recurrent state, which none of these policies has.
    """

    n_actions: int = N_ACTIONS

    #: Dimensionalities this policy is meaningful in; ``None`` means all of
    #: them. Only :class:`BangBangAgent` narrows it — see
    #: :func:`baselines_for_dim` for why the benchmark needs to know.
    dims: tuple[int, ...] | None = None

    def predict(
        self,
        observation: np.ndarray,
        state: tuple | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, None]:
        obs = np.asarray(observation, dtype=np.float64)
        if obs.shape[-1] != OBS_WIDTH:
            raise ValueError(
                f"{type(self).__name__} expects observations of width {OBS_WIDTH} "
                f"(Layer 1 contract §1), got {obs.shape[-1]}."
            )
        if obs.ndim == 1:
            return np.int64(self._action(obs)), None
        if obs.ndim == 2:
            return np.array([self._action(o) for o in obs], dtype=np.int64), None
        raise ValueError(f"Expected a 1-D or 2-D observation, got shape {obs.shape}.")

    def reset(self, seed: int | None = None) -> None:
        """Reset any per-episode/per-run state. No-op unless overridden."""

    def _action(self, obs: np.ndarray) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
class PassiveDriftAgent(BaselinePolicy):
    """Never actuates: the balloon floats at its equilibrium and drifts.

    The floor for both TWR *and* resource spend — a controller that beats this
    on TWR but empties its ballast doing so has not obviously won.
    """

    def _action(self, obs: np.ndarray) -> int:
        return ACTION_STAY


class RandomAgent(BaselinePolicy):
    """Uniform over the three actions.

    Seeded through its own ``Generator`` rather than the global numpy state, so
    an evaluation run is reproducible regardless of what else is drawing
    randomness in the process (env resets, torch, other baselines).
    """

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> None:
        """Restart the action stream from ``seed`` (default: the ctor seed)."""
        self.seed = self.seed if seed is None else seed
        self._rng = np.random.default_rng(self.seed)

    def _action(self, obs: np.ndarray) -> int:
        return int(self._rng.integers(N_ACTIONS))

    def __repr__(self) -> str:
        return f"RandomAgent(seed={self.seed})"


class GreedyWindAgent(BaselinePolicy):
    """Greedy best-wind altitude selection — the heuristic floor.

    Scores every level of the wind column and commands one step toward the best
    one. The score interpolates between two regimes using ``dist_norm``
    (observation index 127), because what counts as a "good" wind depends
    entirely on where you are:

    *Far from the station* — you need to be carried back, so the best level is
    the one with the largest wind component pointing at the goal. Speed is an
    asset. (Loon's sensitivity analysis found its learned controller doing
    exactly this: deliberately seeking *fast* winds beyond the radius in order
    to return.)

    *On station* — you need to not be carried anywhere, so the best level is the
    calmest one. Speed is a liability, whatever its direction.

    The blend weight is ``far = clip(dist_norm / near_far_scale, 0, 1)``, with
    ``near_far_scale`` defaulting to the station radius expressed in ``dist_norm``
    units. So the policy is fully in the "get back" regime at and beyond the
    radius and slides into the "hold still" regime as it closes in.

    Both terms are normalised to [0, 1] so the blend is a genuine convex
    combination rather than an accidental reweighting:

    ``far_score  = 0.5 * (1 + mag * cos(pi * bearing))``
        1.0 for a maximal wind straight at the goal, 0.5 for dead calm,
        0.0 for a maximal wind straight away.
    ``near_score = 1 - mag``
        1.0 for dead calm, 0.0 for a maximal wind in any direction.

    A small ``level_cost`` penalises distant levels. It is deliberately tiny —
    it breaks ties toward the current altitude and stops the policy chasing a
    marginally better level 5 km away, without materially reordering genuinely
    different winds. Levels carrying the limit triple are scored ``-inf`` and can
    never win, so the policy will not steer at an altitude the safety layer
    would only clamp it out of.
    """

    def __init__(
        self,
        *,
        near_far_scale: float | None = None,
        level_cost: float = 0.01,
        deadband_levels: int = 0,
    ) -> None:
        if near_far_scale is None:
            near_far_scale = STATION_RADIUS / DIST_NORM
        if near_far_scale <= 0.0:
            raise ValueError("near_far_scale must be positive.")
        if deadband_levels < 0:
            raise ValueError("deadband_levels must be non-negative.")
        self.near_far_scale = float(near_far_scale)
        self.level_cost = float(level_cost)
        self.deadband_levels = int(deadband_levels)
        # |i - centre| normalised to [0, 1]; precomputed, this runs every step.
        self._level_penalty = (
            np.abs(np.arange(WIND_COL_LEVELS, dtype=np.float64) - WIND_COL_CENTRE)
            / WIND_COL_CENTRE
        )

    # -- scoring ---------------------------------------------------------- #
    def score_levels(self, column: np.ndarray, dist_norm: float) -> np.ndarray:
        """Per-level desirability, higher is better. ``-inf`` marks unreachable."""
        col = np.asarray(column, dtype=np.float64)
        mag = col[:, CH_MAG]
        bearing = col[:, CH_BEARING]

        far = float(np.clip(dist_norm / self.near_far_scale, 0.0, 1.0))
        toward = np.cos(np.pi * bearing)  # +1 at the goal, -1 away
        far_score = 0.5 * (1.0 + mag * toward)  # goalward speed, in [0, 1]
        near_score = 1.0 - mag  # calmness, in [0, 1]

        score = far * far_score + (1.0 - far) * near_score
        score -= self.level_cost * self._level_penalty
        score[limit_mask(col)] = -np.inf
        return score

    def best_level(self, obs: np.ndarray) -> int:
        """Index of the highest-scoring reachable level, or the centre if none is."""
        obs = np.asarray(obs, dtype=np.float64)
        scores = self.score_levels(wind_column(obs), float(obs[IDX_DIST_NORM]))
        if not np.isfinite(scores).any():
            return WIND_COL_CENTRE
        return int(np.argmax(scores))

    # -- policy ----------------------------------------------------------- #
    def _action(self, obs: np.ndarray) -> int:
        delta = self.best_level(obs) - WIND_COL_CENTRE
        if abs(delta) <= self.deadband_levels:
            return ACTION_STAY
        return ACTION_UP if delta > 0 else ACTION_DOWN

    def __repr__(self) -> str:
        return (
            f"GreedyWindAgent(near_far_scale={self.near_far_scale:g}, "
            f"level_cost={self.level_cost:g}, deadband_levels={self.deadband_levels})"
        )


class BangBangAgent(BaselinePolicy):
    """Hand-rolled altitude-hold bang-bang — the 1-D heuristic floor.

    :class:`GreedyWindAgent` is the reference that matters in 2D/3D, but in 1D
    it has nothing to work with: there is no horizontal objective, so every
    column bearing is meaningless and the policy degenerates. 1D needs its own
    reference, and without one its only floor is passive drift at TWR 0.011 —
    a bar a trained agent clears without having learned anything.

    The 1D objective is to reach a target altitude and stay there, which is a
    textbook bang-bang problem: drive full-up when below, full-down when above,
    coast inside a deadband. The one refinement over naive position switching is
    a velocity lead term. A balloon is a second-order system with real inertia
    and a very weak actuator, so switching on position alone overshoots the
    target and then oscillates across it, burning consumables on every reversal.
    Switching on the *predicted* error instead starts the arrest before arrival:

    ``predicted = error_m - vel_ms * lead_time_s``
        ``> +deadband_m`` -> ascend, ``< -deadband_m`` -> descend, else coast.

    The output is still bang-bang; what the lead term buys is a sloped switching
    surface rather than a vertical one, which is the bang-bang controller's
    standard fix for exactly this failure mode.

    Tuning, and what it revealed
    ----------------------------
    ``lead_time_s`` and ``deadband_m`` were swept on the *training* seed range
    (:func:`agents.evaluation.make_scenario_set` with ``held_out=False``) and
    only then scored on the held-out set, so the number this posts is not fitted
    to the set it is reported on.

    The sweep is worth knowing about, because the result is lopsided. The
    deadband is nearly irrelevant — 60 m and 250 m score within 0.001 of each
    other. The lead time is everything: TWR climbs from 0.17 at
    ``lead_time_s=0`` to 0.92 at 1200 s, then falls off by 1800 s. At the
    default the position term is so heavily led that the policy is effectively
    regulating *vertical velocity* toward ``error / lead_time_s``, and only
    incidentally regulating position. That the optimum sits at 20 minutes —
    twenty decision intervals — is a statement about how slowly this platform
    responds to a 0.01 kg impulse, not about bang-bang control.

    The practical consequence: a two-parameter heuristic holds station 92% of
    the time in 1D. 1D is a debugging mode, not a meaningful test of a learned
    controller.
    """

    dims = (1,)

    def __init__(
        self,
        *,
        lead_time_s: float = 1200.0,
        deadband_m: float | None = None,
        respect_limits: bool = True,
    ) -> None:
        if lead_time_s < 0.0:
            raise ValueError("lead_time_s must be non-negative.")
        if deadband_m is None:
            # Half the full-reward radius: tight enough to sit well inside the
            # station, wide enough not to chatter on every metre of drift.
            deadband_m = STATION_RADIUS_1D / 2.0
        if deadband_m < 0.0:
            raise ValueError("deadband_m must be non-negative.")
        self.lead_time_s = float(lead_time_s)
        self.deadband_m = float(deadband_m)
        self.respect_limits = bool(respect_limits)

    def predicted_error_m(self, obs: np.ndarray) -> float:
        """Altitude error in metres, led by ``lead_time_s`` of current velocity.

        Positive means "the target is above where this policy expects to be".
        Both inputs are de-normalised back to physical units so the constants
        above read as seconds and metres rather than as fractions of whatever
        the observation happened to be scaled by.
        """
        obs = np.asarray(obs, dtype=np.float64)
        error_m = float(obs[IDX_GOAL_DZ_NORM]) * ALT_SAFE_SPAN
        vel_ms = float(obs[IDX_VEL_Z_NORM]) * VEL_Z_OBS_NORM
        return error_m - vel_ms * self.lead_time_s

    def _action(self, obs: np.ndarray) -> int:
        predicted = self.predicted_error_m(obs)
        if predicted > self.deadband_m:
            action = ACTION_UP
        elif predicted < -self.deadband_m:
            action = ACTION_DOWN
        else:
            return ACTION_STAY

        # Commanding further into a limit the safety layer is already clamping
        # spends an irreversible consumable for no displacement. GreedyWindAgent
        # avoids the same waste by masking out-of-band levels.
        if self.respect_limits:
            at_max = float(obs[IDX_AT_ALT_MAX]) > 0.5
            at_min = float(obs[IDX_AT_ALT_MIN]) > 0.5
            if (action == ACTION_UP and at_max) or (action == ACTION_DOWN and at_min):
                return ACTION_STAY
        return action

    def __repr__(self) -> str:
        return (
            f"BangBangAgent(lead_time_s={self.lead_time_s:g}, "
            f"deadband_m={self.deadband_m:g}, respect_limits={self.respect_limits})"
        )


#: Convenient registry for evaluation scripts: name -> zero-argument factory.
BASELINES: dict[str, type[BaselinePolicy]] = {
    "passive": PassiveDriftAgent,
    "random": RandomAgent,
    "greedy_wind": GreedyWindAgent,
    "bang_bang": BangBangAgent,
}


def baselines_for_dim(dim: int) -> list[str]:
    """Names of the baselines that are meaningful in ``dim`` dimensions.

    ``bang_bang`` holds an altitude against a target altitude, and in 2D/3D
    there is no such target — ``goal_dz_norm`` is pinned to 0, so it would coast
    forever and post a duplicate of the passive row. Filtering here keeps the
    benchmark table free of rows that only look like results.
    """
    return [
        name for name, cls in BASELINES.items() if cls.dims is None or dim in cls.dims
    ]


def make_baseline(name: str, **kwargs) -> BaselinePolicy:
    """Instantiate a baseline by name (see :data:`BASELINES` for the keys)."""
    try:
        cls = BASELINES[name]
    except KeyError:
        raise KeyError(
            f"Unknown baseline {name!r}. Available: {sorted(BASELINES)}"
        ) from None
    return cls(**kwargs)
