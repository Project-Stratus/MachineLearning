import glob
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback, ProgressBarCallback

from agents.baselines import ACTION_DOWN, ACTION_STAY, ACTION_UP, AMBIENT_IDX

# The physics package owns the operational band; mirror it as a fallback so
# `agents.utils` stays importable (and testable) without the environment stack,
# matching the shim in `agents.baselines`.
try:  # pragma: no cover - trivial import shim
    from environments.core.constants import ALT_SAFE_MIN, ALT_SAFE_MAX, WIND_COL_SPACING
except ImportError:  # pragma: no cover - environment package unavailable
    ALT_SAFE_MIN, ALT_SAFE_MAX = 15_000.0, 25_000.0
    WIND_COL_SPACING = 250.0

#: Observation index of `alt_norm` (Layer 1 contract §1). Read from the layout
#: table in `agents.baselines` rather than hardcoded — the layout has exactly
#: one owner and this is not it.
IDX_ALT_NORM = AMBIENT_IDX["alt_norm"]


def _gather_monitor_csvs(log_dir: str) -> pd.DataFrame:
    """
    Reads VecMonitor/Monitor CSVs and returns a tidy DataFrame with:
      columns = ['episode_idx', 'r', 'l', 't', 'source']
      r = episode return, l = episode length (steps), t = time since start (seconds)
    """
    files = sorted(glob.glob(os.path.join(log_dir, "train_monitor*.csv")))
    dfs = []
    for f in files:
        # Monitor CSVs have commented headers with metadata; use comment='#'
        d = pd.read_csv(f, comment="#")
        # Add a monotonically increasing episode index per file then combine
        d["episode_idx"] = range(1, len(d) + 1)
        d["source"] = os.path.basename(f)
        dfs.append(d[["episode_idx", "r", "l", "t", "source"]])
    if not dfs:
        return pd.DataFrame(columns=["episode_idx", "r", "l", "t", "source"])
    df = pd.concat(dfs, ignore_index=True)
    # If you want a single global episode index across all vec envs:
    df["global_episode"] = range(1, len(df) + 1)
    return df


class TerminationTracker(BaseCallback):
    """Tracks termination reasons from VecEnv infos and prints a summary at the end."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.counts: dict[str, int] = defaultdict(int)

    def _on_step(self) -> bool:
        # SB3 VecEnv stores episode info in self.locals["infos"]
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", np.zeros(len(infos), dtype=bool))
        for info, done in zip(infos, dones):
            if not done:
                continue
            reason = info.get("termination_reason", None)
            if reason is None:
                # Check inside the terminal_observation wrapper used by VecEnv
                ep_info = info.get("terminal_info", info)
                reason = ep_info.get("termination_reason", "Unknown")
            self.counts[reason] += 1
        return True

    def _on_training_end(self) -> None:
        total = sum(self.counts.values())
        if total == 0:
            return
        print(f"\n{'='*55}")
        print(f"  Termination Breakdown ({total:,} episodes)")
        print(f"{'='*55}")
        for reason, count in sorted(self.counts.items(), key=lambda x: -x[1]):
            pct = 100.0 * count / total
            print(f"  {reason:<45s} {count:>6,}  ({pct:5.1f}%)")
        print(f"{'='*55}\n")


class InfoProgressBar(ProgressBarCallback):
    """Progress bar callback with custom description and postfix."""

    def __init__(self, description: str, postfix: dict | None = None):
        super().__init__()
        self._description = description
        self._postfix = postfix or {}

    def _resolve_bar(self):
        return getattr(self, "progress_bar", None) or getattr(self, "pbar", None)

    def _on_training_start(self) -> None:
        super()._on_training_start()
        bar = self._resolve_bar()
        if bar is not None:
            bar.set_description_str(self._description)
            if self._postfix:
                bar.set_postfix(self._postfix, refresh=False)

    def _on_step(self) -> bool:
        bar = self._resolve_bar()
        if bar is not None and self._postfix:
            bar.set_postfix(self._postfix, refresh=False)
        return super()._on_step()


# --------------------------------------------------------------------------- #
# Momentum exploration (roadmap §3.8)
# --------------------------------------------------------------------------- #
class MomentumExplorer:
    """Persistent target-altitude random walk — an exploratory *intention*.

    Why this exists
    ---------------
    Per-step epsilon-greedy is close to useless on a balloon. One exploratory
    action is a single ~0.01 kg impulse; over one 60 s decision it moves the
    balloon by metres out of a 10 km band, and the *next* exploratory action is
    drawn independently, so a uniform-random action stream is a zero-mean
    dither. The balloon stays where it was, sees the same wind, and the agent
    never observes the states that actually differ — a different altitude, and
    therefore a different wind. Exploration that cannot change the macroscopic
    state is not exploration.

    The fix is to randomise the *intention* rather than the action: sample a
    target altitude uniformly from the operational band, hold it for
    ``perturb_every`` decisions, perturb it with Gaussian noise, and emit
    whichever action moves the balloon toward it. Consecutive actions are then
    correlated over hundreds of steps and the balloon genuinely traverses the
    column.

    *Prior art:* Loon sampled a random altitude setpoint, perturbed it with
    Gaussian noise, and interleaved 4 h greedy / 2 h exploratory phases on 80%
    of trials.

    State is **per environment**: with ``n_envs`` parallel actors each carries
    its own target, its own perturbation phase and its own countdown, so the
    actors explore different parts of the column instead of moving in lockstep.

    Parameters
    ----------
    alt_min, alt_max
        Operational band the target is drawn from and clipped to. Defaults to
        the safety layer's ``[ALT_SAFE_MIN, ALT_SAFE_MAX]``: a target the safety
        layer would only clamp out of is not worth walking toward.
    perturb_every
        Decisions between Gaussian perturbations of the target. The default 30
        is 30 minutes at ``DECISION_INTERVAL = 60`` — long enough that the
        balloon can actually reach a target before it moves.
    perturb_sigma
        Standard deviation of the perturbation, in metres. 750 m is ~7.5% of
        the band, so the target diffuses across it over an episode rather than
        teleporting.
    deadband
        Half-width, in metres, of the "close enough" zone around the target
        where the explorer commands ``stay``. Deliberately smaller than
        ``WIND_COL_SPACING`` (250 m), so holding station inside the deadband
        cannot hide a wind level from the agent.
    seed
        Seeds a private ``Generator``, so exploration is reproducible
        independently of whatever else is drawing randomness in the process.
    """

    def __init__(
        self,
        *,
        alt_min: float = ALT_SAFE_MIN,
        alt_max: float = ALT_SAFE_MAX,
        perturb_every: int = 30,
        perturb_sigma: float = 750.0,
        deadband: float = 0.5 * WIND_COL_SPACING,
        seed: int | None = None,
    ) -> None:
        if alt_max <= alt_min:
            raise ValueError(f"alt_max must exceed alt_min, got ({alt_min}, {alt_max}).")
        if perturb_every < 1:
            raise ValueError(f"perturb_every must be >= 1, got {perturb_every}.")
        if perturb_sigma < 0.0 or deadband < 0.0:
            raise ValueError("perturb_sigma and deadband must be non-negative.")

        self.alt_min = float(alt_min)
        self.alt_max = float(alt_max)
        self.span = self.alt_max - self.alt_min
        self.perturb_every = int(perturb_every)
        self.perturb_sigma = float(perturb_sigma)
        self.deadband = float(deadband)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.target_alt: np.ndarray = np.empty(0, dtype=np.float64)
        self._countdown: np.ndarray = np.empty(0, dtype=np.int64)

    # -- state management ------------------------------------------------- #
    @property
    def n_envs(self) -> int:
        return int(self.target_alt.size)

    def _resize(self, n_envs: int) -> None:
        """(Re)allocate per-env state for ``n_envs`` actors and seed it."""
        self.target_alt = np.empty(n_envs, dtype=np.float64)
        # Random phase per env, so the actors do not all perturb on the same
        # step — otherwise n_envs actors are one actor with n_envs copies.
        self._countdown = self.rng.integers(1, self.perturb_every + 1, size=n_envs).astype(np.int64)
        self.reset_envs(np.ones(n_envs, dtype=bool))

    def reset_envs(self, mask: np.ndarray | None = None) -> None:
        """Draw fresh targets for the envs selected by ``mask`` (default: all).

        Called on episode boundaries: a new episode is a new scenario, so the
        intention carried over from the previous one means nothing.
        """
        if self.n_envs == 0:
            return
        if mask is None:
            mask = np.ones(self.n_envs, dtype=bool)
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        k = int(np.count_nonzero(mask))
        if k == 0:
            return
        self.target_alt[mask] = self.rng.uniform(self.alt_min, self.alt_max, size=k)
        self._countdown[mask] = self.perturb_every

    def _advance(self) -> None:
        """Tick every env's countdown, perturbing the targets that come due."""
        self._countdown -= 1
        due = self._countdown <= 0
        k = int(np.count_nonzero(due))
        if k:
            noise = self.rng.normal(0.0, self.perturb_sigma, size=k)
            self.target_alt[due] = np.clip(
                self.target_alt[due] + noise, self.alt_min, self.alt_max
            )
            self._countdown[due] = self.perturb_every

    # -- policy ------------------------------------------------------------ #
    def act(self, obs: np.ndarray) -> np.ndarray:
        """Actions moving each env's balloon toward its target altitude.

        Accepts a single ``(OBS_WIDTH,)`` observation or a batched
        ``(n_envs, OBS_WIDTH)`` one; always returns a ``(n_envs,)`` int array.
        Advances the target walk by one decision.
        """
        arr = np.atleast_2d(np.asarray(obs, dtype=np.float64))
        n = arr.shape[0]
        if n != self.n_envs:
            self._resize(n)
        self._advance()

        alt = self.alt_min + np.clip(arr[:, IDX_ALT_NORM], 0.0, 1.0) * self.span
        delta = self.target_alt - alt

        actions = np.full(n, ACTION_STAY, dtype=np.int64)
        actions[delta > self.deadband] = ACTION_UP
        actions[delta < -self.deadband] = ACTION_DOWN
        return actions

    def __repr__(self) -> str:
        return (
            f"MomentumExplorer(band=({self.alt_min:g}, {self.alt_max:g}), "
            f"perturb_every={self.perturb_every}, perturb_sigma={self.perturb_sigma:g}, "
            f"deadband={self.deadband:g}, seed={self.seed}, n_envs={self.n_envs})"
        )