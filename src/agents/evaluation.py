"""
agents.evaluation
-----------------

Time-within-radius (TWR) evaluation against a frozen, held-out scenario set.

Why this exists
===============
Layer 1's exit criterion is "performs well", which was previously unmeasurable.
The old setup ran SB3's ``EvalCallback`` for 5 episodes on a single eval env
built from the *same* config as training and scored summed reward: no held-out
set, no train/test split, no baselines. The symptom was an early-stop threshold
of 83,000 against a maximum episode return of 86,400 — i.e. stopping at ~96%
time-within-radius, which says the environment is trivial, not that the agent
is good.

Three things fix that, and this module provides them:

**A time-within-radius metric.** Fraction of the flight spent inside
``STATION_RADIUS``. Interpretable on its own, comparable across layers as the
reward shaping changes underneath it, and it transfers to hardware unchanged —
unlike summed reward, which is only comparable to itself.

**A frozen held-out scenario set.** :func:`make_scenario_set` draws scenario
seeds from a range *disjoint* from the training range (see
:data:`TRAIN_SEED_RANGE` / :data:`HELD_OUT_SEED_RANGE`), so held-out scenarios
are genuinely unseen. The env randomises goal, spawn and wind-pattern
parameters from ``self.np_random`` on reset, so pinning the reset seed pins the
whole scenario.

**Baselines.** See :mod:`agents.baselines`. They share the SB3 ``predict``
interface, so :func:`evaluate_policy_twr` scores a baseline and a trained
``QRDQN`` through exactly the same code path — which is the only way the
comparison means anything.

*Prior art:* Loon benchmarked on 6,000 flights (5 locations x 100 dates x 12
seeds) with training years excluded, stratified into difficulty bands. Their
best controller reached 55.1% TWR50 against a ~57% estimated ceiling; the
heuristic baseline reached 40.5%.
"""

from __future__ import annotations

import os
from collections import Counter
from typing import Any, Iterable, Sequence

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

# The physics package owns the reward/TWR definitions; mirror them only as a
# fallback so `agents.evaluation` stays importable (and testable) without the
# environment stack. Values are the Layer 1 contract values.
try:  # pragma: no cover - trivial import shim
    from environments.core.constants import STATION_RADIUS
    from environments.core.reward import time_within_radius
except ImportError:  # pragma: no cover - environment package unavailable
    STATION_RADIUS = 10_000.0

    def time_within_radius(distances, station_radius: float = STATION_RADIUS) -> float:
        """Fraction of samples within ``station_radius`` (boundary counts as inside)."""
        arr = np.asarray(distances, dtype=np.float64).ravel()
        if arr.size == 0:
            return 0.0
        return float(np.count_nonzero(arr <= station_radius) / arr.size)


# --------------------------------------------------------------------------- #
# Scenario seeds — the train/test split
# --------------------------------------------------------------------------- #
# The split is by *seed range*, not by draw. Training seeds and held-out seeds
# come from two half-open intervals that cannot overlap, so no amount of
# resampling, reseeding or rerunning can leak a held-out scenario into training.
# That is the entire point: a split enforced by construction needs no bookkeeping
# and cannot silently rot as the training script changes.
#
# Anything that seeds a *training* env (qrdqn.SEED, the per-worker `seed + i`
# offsets in `_build_vec_env`) must stay inside TRAIN_SEED_RANGE.
TRAIN_SEED_RANGE: tuple[int, int] = (0, 1_000_000)
HELD_OUT_SEED_RANGE: tuple[int, int] = (1_000_000, 2_000_000)


def is_held_out_seed(seed: int) -> bool:
    """True if ``seed`` lies in the held-out range."""
    lo, hi = HELD_OUT_SEED_RANGE
    return lo <= int(seed) < hi


def seed_range(held_out: bool) -> tuple[int, int]:
    """The half-open seed interval reserved for held-out (or training) scenarios."""
    return HELD_OUT_SEED_RANGE if held_out else TRAIN_SEED_RANGE


def make_scenario_set(
    n: int,
    seed: int,
    held_out: bool = True,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build a frozen set of ``n`` scenario descriptors.

    A scenario is just a reset seed plus optional env config overrides: the env
    draws goal position, spawn position and wind-pattern parameters from
    ``self.np_random``, so ``env.reset(seed=s)`` reproduces a scenario exactly.

    Parameters
    ----------
    n
        Number of scenarios. Must be positive.
    seed
        Meta-seed for *drawing* the scenario seeds. Deterministic: the same
        ``(n, seed, held_out)`` always yields the same list. The ``held_out``
        flag is folded into the generator's entropy as well as selecting the
        range, so a training set and a held-out set built from the same ``seed``
        are unrelated draws rather than the same draw shifted by an offset.
    held_out
        Draw from :data:`HELD_OUT_SEED_RANGE` (default) or
        :data:`TRAIN_SEED_RANGE`.
    config
        Env config overrides attached to every scenario. These are for the
        *caller* to apply when constructing the env — a scenario cannot
        reconfigure an env that is already built.

    Returns
    -------
    list of dict
        Keys: ``scenario_id``, ``seed``, ``held_out``, ``config``.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}.")

    lo, hi = seed_range(held_out)
    rng = np.random.default_rng([int(seed), int(held_out)])

    seeds: list[int] = []
    seen: set[int] = set()
    while len(seeds) < n:
        candidate = int(rng.integers(lo, hi))
        if candidate not in seen:
            seen.add(candidate)
            seeds.append(candidate)

    tag = "heldout" if held_out else "train"
    overrides = dict(config or {})
    return [
        {
            "scenario_id": f"{tag}-{i:04d}",
            "seed": s,
            "held_out": bool(held_out),
            "config": dict(overrides),
        }
        for i, s in enumerate(seeds)
    ]


def scenario_seeds(scenarios: Iterable[dict[str, Any]]) -> list[int]:
    """Extract the reset seeds from a scenario set."""
    return [int(s["seed"]) for s in scenarios]


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def _as_env_action(action: Any) -> Any:
    """Normalise a policy's action to something ``env.step`` accepts.

    SB3 returns a 0-d array for a single observation and a ``(1,)`` array when
    the obs was auto-batched; baselines return a numpy scalar. All of these are
    the same discrete action.
    """
    arr = np.asarray(action)
    if arr.ndim == 0 or arr.size == 1:
        return arr.reshape(-1)[0].item()
    return action


def _termination_reason(info: dict[str, Any], terminated: bool, truncated: bool) -> str:
    if terminated:
        return str(info.get("termination_reason", "terminated"))
    if truncated:
        return str(info.get("termination_reason", "time_limit"))
    return "unknown"


def evaluate_policy_twr(
    policy: Any,
    env: Any,
    scenarios: Sequence[dict[str, Any]],
    *,
    deterministic: bool = True,
    station_radius: float | None = None,
    max_episode_steps: int | None = None,
) -> dict[str, Any]:
    """Run ``policy`` on every scenario and score it by time-within-radius.

    ``env`` must be a single (non-vectorised) gymnasium-style env whose
    ``info`` carries ``"distance"`` every step — the current horizontal distance
    to the goal. That value is taken as authoritative and never re-derived from
    positions, so this function stays correct as the env's internal state
    representation changes.

    Episode-length normalisation
    ----------------------------
    Per-episode TWR is ``in-radius steps / horizon``, where ``horizon`` is the
    *scheduled* episode length rather than the achieved one. If it were the
    achieved length, an agent that parked itself on station and then died after
    ten steps would score a perfect 1.0 — early termination would be rewarded.
    Normalising by the horizon prices the forfeited flight time at zero, which
    is the same logic behind the reward being 0.0 on termination.

    ``horizon`` comes from ``max_episode_steps`` if given, else from
    ``env.spec.max_episode_steps``, else from the longest observed episode.

    Returns
    -------
    dict
        ``twr`` (mean per-episode TWR), ``mean_return``, ``mean_final_distance``,
        ``termination_counts``, ``n_episodes``, plus the per-episode breakdown
        (``episode_twr``, ``episode_returns``, ``episode_lengths``), the
        step-weighted ``twr_pooled`` and the ``horizon`` used.
    """
    # The radius must match the one the env rewards against, or the metric and
    # the reward disagree.  1-D uses a much tighter altitude tolerance, so
    # defaulting to the horizontal constant would score every 1-D run at TWR 1.0
    # while the return told a completely different story.
    if station_radius is None:
        station_radius = float(getattr(
            getattr(env, "unwrapped", env), "_station_radius", STATION_RADIUS))

    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    episode_in_radius: list[int] = []
    final_distances: list[float] = []
    all_distances: list[float] = []
    termination_counts: Counter[str] = Counter()

    for scenario in scenarios:
        # The scenario seed pins the env only. Stateful policies are deliberately
        # *not* reset between scenarios: re-seeding `RandomAgent` every episode
        # would make it replay one identical action sequence across the whole set.
        # Run-level reproducibility comes from how the policy was constructed.
        obs, _info = env.reset(seed=int(scenario["seed"]))

        done = False
        ep_return = 0.0
        ep_distances: list[float] = []
        info: dict[str, Any] = {}
        terminated = truncated = False

        while not done:
            action, _state = policy.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(_as_env_action(action))
            ep_return += float(reward)
            if "distance" not in info:
                raise KeyError(
                    "info['distance'] is required for TWR evaluation (Layer 1 "
                    "contract §5) but the env did not provide it."
                )
            ep_distances.append(float(info["distance"]))
            done = bool(terminated) or bool(truncated)

        episode_returns.append(ep_return)
        episode_lengths.append(len(ep_distances))
        arr = np.asarray(ep_distances, dtype=np.float64)
        episode_in_radius.append(int(np.count_nonzero(arr <= station_radius)))
        final_distances.append(float(ep_distances[-1]) if ep_distances else float("nan"))
        all_distances.extend(ep_distances)
        termination_counts[_termination_reason(info, terminated, truncated)] += 1

    n_episodes = len(episode_returns)
    if n_episodes == 0:
        return {
            "twr": 0.0,
            "mean_return": 0.0,
            "mean_final_distance": float("nan"),
            "termination_counts": {},
            "n_episodes": 0,
            "twr_pooled": 0.0,
            "mean_episode_length": 0.0,
            "episode_twr": [],
            "episode_returns": [],
            "episode_lengths": [],
            "horizon": 0,
        }

    horizon = max_episode_steps
    if horizon is None:
        spec = getattr(env, "spec", None)
        horizon = getattr(spec, "max_episode_steps", None) if spec is not None else None
    if horizon is None:
        horizon = max(episode_lengths)
    horizon = int(horizon)

    if horizon > 0:
        episode_twr = [min(1.0, n_in / horizon) for n_in in episode_in_radius]
    else:
        episode_twr = [0.0] * n_episodes

    return {
        "twr": float(np.mean(episode_twr)),
        "mean_return": float(np.mean(episode_returns)),
        "mean_final_distance": float(np.mean(final_distances)),
        "termination_counts": dict(termination_counts),
        "n_episodes": n_episodes,
        "twr_pooled": float(time_within_radius(all_distances, station_radius)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "episode_twr": episode_twr,
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "horizon": horizon,
    }


# --------------------------------------------------------------------------- #
# Training callback
# --------------------------------------------------------------------------- #
class TWREvalCallback(BaseCallback):
    """Periodic held-out evaluation that selects checkpoints by TWR, not return.

    Drop-in alongside (or instead of) SB3's ``EvalCallback``. The difference
    that matters is the selection criterion: mean episode return depends on the
    reward shaping, which changes between layers as resource penalties and
    dropoffs are retuned, so "best return" checkpoints are not comparable across
    runs. TWR is fixed by the mission definition, so it is.

    Logs ``eval/twr`` and ``eval/mean_return`` (plus ``eval/mean_final_distance``
    and ``eval/mean_ep_length``) to TensorBoard.

    Parameters
    ----------
    eval_env
        A single, non-vectorised gymnasium env, built from the *held-out* config.
        Do not reuse the training env.
    scenarios
        Frozen scenario set, normally ``make_scenario_set(n, seed, held_out=True)``.
    eval_freq
        Evaluate every ``eval_freq`` calls of the callback. As with SB3's
        ``EvalCallback``, with ``n`` parallel envs one callback call is ``n``
        environment steps, so pass ``max(eval_freq // n_envs, 1)`` if you want
        the period measured in environment steps.
    best_model_save_path
        Directory for the best-by-TWR checkpoint. ``None`` disables saving.
    best_model_name
        Filename stem, deliberately *not* ``best_model`` so this can run
        alongside SB3's ``EvalCallback`` without either clobbering the other.
    log_path
        Directory for ``twr_evaluations.npz`` (timesteps / twr / mean_return).
    callback_on_new_best
        Fired on every TWR improvement — e.g. a stop-on-threshold callback.
    """

    def __init__(
        self,
        eval_env: Any,
        scenarios: Sequence[dict[str, Any]],
        *,
        eval_freq: int = 50_000,
        best_model_save_path: str | None = None,
        best_model_name: str = "best_twr_model",
        log_path: str | None = None,
        deterministic: bool = True,
        station_radius: float | None = None,
        max_episode_steps: int | None = None,
        callback_on_new_best: BaseCallback | None = None,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.eval_env = eval_env
        self.scenarios = list(scenarios)
        self.eval_freq = int(eval_freq)
        self.best_model_save_path = best_model_save_path
        self.best_model_name = best_model_name
        self.log_path = log_path
        self.deterministic = deterministic
        # None -> derive from the eval env, so the logged TWR always matches the
        # radius the env actually rewards against (see evaluate_policy_twr).
        self.station_radius = (
            None if station_radius is None else float(station_radius))
        self.max_episode_steps = max_episode_steps
        self.callback_on_new_best = callback_on_new_best

        self.best_twr: float = -np.inf
        self.last_twr: float = -np.inf
        self.last_results: dict[str, Any] = {}
        self.evaluations_timesteps: list[int] = []
        self.evaluations_twr: list[float] = []
        self.evaluations_return: list[float] = []

    def _init_callback(self) -> None:
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(self.log_path, exist_ok=True)
        if self.callback_on_new_best is not None:
            self.callback_on_new_best.init_callback(self.model)

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True
        return self.evaluate()

    def evaluate(self) -> bool:
        """Run one evaluation, log it, and update the best-by-TWR checkpoint.

        Returns ``False`` to stop training (only ever via ``callback_on_new_best``).
        """
        results = evaluate_policy_twr(
            self.model,
            self.eval_env,
            self.scenarios,
            deterministic=self.deterministic,
            station_radius=self.station_radius,
            max_episode_steps=self.max_episode_steps,
        )
        self.last_results = results
        self.last_twr = float(results["twr"])

        self.evaluations_timesteps.append(int(self.num_timesteps))
        self.evaluations_twr.append(self.last_twr)
        self.evaluations_return.append(float(results["mean_return"]))

        if self.log_path is not None:
            np.savez(
                os.path.join(self.log_path, "twr_evaluations.npz"),
                timesteps=np.asarray(self.evaluations_timesteps),
                twr=np.asarray(self.evaluations_twr),
                mean_return=np.asarray(self.evaluations_return),
            )

        if self.verbose >= 1:
            print(
                f"Eval @ {self.num_timesteps:,} steps | "
                f"TWR={self.last_twr:.3f} | "
                f"return={results['mean_return']:.1f} | "
                f"final dist={results['mean_final_distance']:,.0f} m | "
                f"{results['n_episodes']} episodes"
            )

        self.logger.record("eval/twr", self.last_twr)
        self.logger.record("eval/mean_return", float(results["mean_return"]))
        self.logger.record("eval/mean_final_distance", float(results["mean_final_distance"]))
        self.logger.record("eval/mean_ep_length", float(results["mean_episode_length"]))
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        self.logger.dump(self.num_timesteps)

        continue_training = True
        if self.last_twr > self.best_twr:
            self.best_twr = self.last_twr
            if self.best_model_save_path is not None:
                self.model.save(os.path.join(self.best_model_save_path, self.best_model_name))
            if self.verbose >= 1:
                print(f"New best TWR: {self.best_twr:.3f}")
            if self.callback_on_new_best is not None:
                continue_training = self.callback_on_new_best.on_step()
        return continue_training
