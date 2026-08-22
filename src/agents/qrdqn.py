# qr_dqn_runner.py
import os
import time
import multiprocessing as mp

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from sb3_contrib import QRDQN

import environments  # registers the Balloon3D-v0 environment
from environments.core.constants import DECISION_INTERVAL, TIME_MAX
from environments.wrappers.decision_interval import DecisionIntervalWrapper
from agents.baselines import baselines_for_dim, make_baseline
from agents.evaluation import (
    TRAIN_SEED_RANGE,
    TWREvalCallback,
    evaluate_policy_twr,
    make_scenario_set,
)
from agents.utils import (
    _gather_monitor_csvs,
    InfoProgressBar,
    MomentumExplorer,
    TerminationTracker,
)

# ---- Config ----
_ENV_NAMES = {
    "zero_pressure": "environments/Balloon3D-v0",
    "superpressure": "environments/BalloonSP3D-v0",
}
BASE_SAVE_PATH = "./src/models/qr_dqn_model/"
VIDEO_PATH = "./figs/qr_dqn_figs/performance_video"  # (unused here but kept for parity)

#: Seed for the *training* envs. MUST stay inside `evaluation.TRAIN_SEED_RANGE`
#: — the train/held-out split is enforced by seed range and nothing else. The
#: per-worker offsets below (`seed + i`) must stay inside it too; that is what
#: `_check_train_seeds` enforces, so a future bump cannot silently leak
#: held-out scenarios into training. See `agents.evaluation`.
SEED = 42

#: Meta-seed for *drawing* the held-out scenario seeds. Not itself an env seed,
#: so it is unconstrained: `make_scenario_set(held_out=True)` draws the actual
#: scenario seeds from HELD_OUT_SEED_RANGE by construction.
SCENARIO_SEED = 2026

#: Agent decisions per episode: 43,200 physics steps / 60 = 720.
DECISIONS_PER_EPISODE = TIME_MAX // DECISION_INTERVAL

# ---- Shared architecture (Loon-style QR-DQN — same for both balloon types) ----
POLICY_KWARGS = dict(
    # Four hidden layers. The observation went 19 -> 143 when the wind column
    # landed (Layer 1 contract §1), and roadmap §3.8 gates depth on exactly that:
    # Loon ablated depth directly and found performance still climbing at ~7
    # layers — but against a 1,099-dim input. Scaled to our 143 dims, four
    # layers is the proportionate step; revisit again when Layer 2 widens the
    # column with real weather.
    net_arch=[512, 512, 512, 512],
    activation_fn=torch.nn.ReLU,
    # Loon-style quantile head. Do NOT tune this yet: per roadmap §2.2 the
    # dynamics are deterministic through Layers 1-2, so the return distribution
    # from any state-action pair is a point mass and every quantile collapses to
    # the same value. QR-DQN is DQN with 51x the output width until Layer 3
    # introduces uncertainty. The architecture is right for where we are going;
    # the knob is simply inert here.
    n_quantiles=51,
)

# ---- Held-out evaluation during training (roadmap §3.1) ----
#: Scenarios per evaluation. Each is a full 12 h episode (720 decisions,
#: ~2 s of physics), so 12 scenarios is ~25 s per evaluation.
N_EVAL_SCENARIOS = 12
#: Environment steps between evaluations. At 15M steps that is ~60 evaluations,
#: i.e. ~25 min of eval over a multi-hour run. The callback counts *calls*, so
#: this is divided by n_envs at construction.
EVAL_FREQ = 250_000

#: Reference TWR of the non-learning policies, measured on *this* scenario set
#: (N_EVAL_SCENARIOS, SCENARIO_SEED, held_out=True, zero_pressure) — the same set
#: TWREvalCallback scores during training, so these are directly comparable to
#: the `eval/twr` curve. Annotation only: `python main.py --benchmark` is the
#: authoritative measurement and must be re-run whenever the scenario set, the
#: wind family or the reward changes.
#:
#: Keyed by dim, because the bar is not the same policy in each. In 3D the wind
#: column carries exploitable signal and `greedy_wind` separates 3.4x from
#: passive. In 2D and 1D it does not separate at all — see BASELINE_BAR.
BASELINE_REFERENCE_TWR: dict[int, dict[str, float]] = {
    1: {"bang_bang": 0.919, "random": 0.009, "passive": 0.008, "greedy_wind": 0.008},
    2: {"passive": 0.025, "random": 0.025, "greedy_wind": 0.025},
    3: {"greedy_wind": 0.126, "passive": 0.037, "random": 0.037},
}

#: The policy a trained agent has to beat in each dim, and the floor it must not
#: fall below. Reading the bar off `max()` would be fragile in 2D, where all
#: three baselines tie — naming it makes the claim explicit and reviewable.
BASELINE_BAR: dict[int, str] = {1: "bang_bang", 2: "greedy_wind", 3: "greedy_wind"}


def baseline_reference(dim: int) -> tuple[dict[str, float], str]:
    """Held-out baseline TWRs for ``dim``, and the name of the bar to clear.

    Raises rather than falling back to the 3D numbers: plotting a 3D bar on a
    1D training curve would make a 0.5 TWR agent look like a triumph when it is
    less than half of what a two-parameter heuristic already does.
    """
    try:
        return BASELINE_REFERENCE_TWR[dim], BASELINE_BAR[dim]
    except KeyError:
        raise KeyError(
            f"No measured baselines for dim={dim}. Run `python main.py "
            f"--benchmark --dim {dim}` and record the result."
        ) from None


# ---- Per-type training config ----
# Each dict is independent so ZP and SP can be tuned without affecting each other.
#
# Discount (roadmap §3.2). The effective horizon of gamma is 1/(1-gamma)
# decisions, and one decision is DECISION_INTERVAL seconds:
#
#     gamma = 0.995 -> 1/0.005 =  200 decisions x 60 s =  3.3 h
#     gamma = 0.997 -> 1/0.003 =  333 decisions x 60 s =  5.6 h   <-- chosen
#
# against a 12 h episode. 0.995 saw barely a quarter of the flight, which is not
# enough to price "ride an unfavourable wind out now, come back later". 0.997
# covers ~46% of the episode, which is the same fraction Loon ran (0.993 at
# 180 s = 7.1 h against 2-day episodes is ~15%, but their episodes were 4x
# longer relative to the control timescale).
#
# RE-DERIVE THIS WHENEVER `DECISION_INTERVAL` OR `EPISODE_HOURS` CHANGES:
#     gamma = 1 - DECISION_INTERVAL / (target_horizon_hours * 3600)
_TRAIN_CFG = {
    "zero_pressure": dict(
        learning_rate=3e-4,
        gamma=0.997,  # ~5.6 h horizon at a 60 s decision; see above
        buffer_size=1_000_000,
        learning_starts=50_000,
        train_freq=4,  # collect 4 env steps between gradient updates
        gradient_steps=4,  # 4 gradient steps per update
        target_update_interval=10_000,
        batch_size=512,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        exploration_fraction=0.3,  # portion of training over which epsilon decays
        verbose=0,
    ),
    "superpressure": dict(
        learning_rate=3e-4,
        gamma=0.997,
        buffer_size=1_000_000,
        learning_starts=50_000,
        train_freq=4,
        gradient_steps=4,
        target_update_interval=10_000,
        batch_size=512,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        exploration_fraction=0.3,
        verbose=0,
    ),
}

_TOTAL_TIMESTEPS = {
    "zero_pressure": 15_000_000,
    "superpressure": 15_000_000,
}

# ---- Per-type environment config ----
_ENV_CONFIG = {
    "zero_pressure": dict(
        wind_pattern="altitude_shear_2d",
    ),
    "superpressure": dict(
        wind_pattern="altitude_shear_2d",
    ),
}

MAX_ENVS = max(8, os.cpu_count() - 2)
N_ENVS = min(MAX_ENVS, max(1, os.cpu_count() // 2))  # overridden by train(n_envs=...)


# --------------------------------------------------------------------------- #
# Train / held-out split guard (roadmap §3.1)
# --------------------------------------------------------------------------- #
def _check_train_seeds(seed: int, n_envs: int) -> None:
    """Assert every training env seed lands inside ``TRAIN_SEED_RANGE``.

    ``agents.evaluation`` splits train from held-out purely by seed range, so
    the split has exactly one failure mode: a training seed drifting into the
    held-out interval. Nothing downstream would notice — the numbers would just
    quietly become train scores wearing a held-out label. This makes that a
    loud, immediate failure instead.
    """
    lo, hi = TRAIN_SEED_RANGE
    highest = int(seed) + max(1, int(n_envs)) - 1  # workers use `seed + i`
    if int(seed) < lo or highest >= hi:
        raise ValueError(
            f"Training seeds [{seed}, {highest}] fall outside TRAIN_SEED_RANGE "
            f"[{lo}, {hi}). Held-out scenarios are drawn from "
            f"{list(TRAIN_SEED_RANGE)}-disjoint seeds; a training seed in that "
            f"range would silently leak the evaluation set into training."
        )


# Fail at import if the module-level default is already outside the range.
_check_train_seeds(SEED, MAX_ENVS)


# --------------------------------------------------------------------------- #
# Momentum exploration (roadmap §3.8)
# --------------------------------------------------------------------------- #
class MomentumQRDQN(QRDQN):
    """QR-DQN whose *exploratory* actions come from a target-altitude walk.

    Only **what** an exploratory action is changes. SB3's epsilon schedule still
    decides **when** to explore, so `exploration_initial_eps`,
    `exploration_final_eps` and `exploration_fraction` keep their meaning and the
    change ablates cleanly against plain epsilon-greedy
    (``momentum_exploration=False`` recovers stock behaviour exactly).

    Two differences from ``QRDQN._sample_action``:

    1. The exploratory branch calls :class:`~agents.utils.MomentumExplorer`
       instead of ``action_space.sample()``. See that class for why uniform
       random actions explore nothing on a balloon.
    2. The explore/exploit coin is flipped **per environment**. SB3's
       ``DQN.predict`` draws a single ``np.random.rand()`` for the whole batch,
       so all n_envs actors explore or exploit together — which turns n_envs
       actors into one actor with n_envs copies exactly when diversity matters.

    The warm-up phase (``num_timesteps < learning_starts``) is fully
    exploratory, as in SB3, and is where momentum exploration matters most: it
    is the only time the buffer is filled by exploration alone.
    """

    def __init__(
        self,
        *args,
        momentum_exploration: bool = True,
        momentum_kwargs: dict | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.momentum_exploration = bool(momentum_exploration)
        self.momentum_kwargs = dict(momentum_kwargs or {})
        self._explorer = MomentumExplorer(seed=self.seed, **self.momentum_kwargs)

    def _excluded_save_params(self) -> list[str]:
        # The explorer is training-only state (and holds a Generator); a saved
        # policy must not depend on it to load.
        return super()._excluded_save_params() + ["_explorer"]

    def _sample_action(self, learning_starts: int, action_noise=None, n_envs: int = 1):
        if not self.momentum_exploration or not isinstance(self._last_obs, np.ndarray):
            return super()._sample_action(learning_starts, action_noise, n_envs)

        warmup = self.num_timesteps < learning_starts
        if warmup:
            explore = np.ones(n_envs, dtype=bool)
        else:
            explore = self._explorer.rng.random(n_envs) < self.exploration_rate

        # The target walk advances every step whether or not this env is
        # exploring: the target is a latent intention, not a per-action coin
        # flip. Stalling it while the greedy policy is in control would make
        # exploratory stretches restart from a stale target.
        actions = self._explorer.act(self._last_obs)

        if not explore.all():
            greedy, _ = self.policy.predict(self._last_obs, deterministic=True)
            actions = np.where(explore, actions, np.asarray(greedy).reshape(-1))

        actions = np.asarray(actions, dtype=np.int64).reshape(-1)
        return actions, actions

    def _store_transition(
        self, replay_buffer, buffer_action, new_obs, reward, dones, infos
    ):
        super()._store_transition(
            replay_buffer, buffer_action, new_obs, reward, dones, infos
        )
        # A new episode is a new scenario (new goal, spawn and shear), so the
        # intention carried over from the last one is meaningless.
        if self.momentum_exploration and np.any(dones):
            self._explorer.reset_envs(np.asarray(dones, dtype=bool))


# --------------------------------------------------------------------------- #
# Environment construction
# --------------------------------------------------------------------------- #
def _make_vec_env_fn(env_id: str, dim: int, seed: int, config: dict):
    """Return a picklable factory for SubprocVecEnv workers."""

    def _init():
        env = gym.make(
            env_id, render_mode=None, dim=dim, disable_env_checker=True, config=config
        )
        env = DecisionIntervalWrapper(env)
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _init


def _build_vec_env(
    n_envs: int,
    env_id: str,
    dim: int,
    seed: int = SEED,
    save_path: str = None,
    env_config: dict = None,
):
    _check_train_seeds(seed, n_envs)
    cfg = {**(env_config or {})}
    sp = save_path if save_path is not None else BASE_SAVE_PATH
    if n_envs != 1:
        if mp.get_start_method(allow_none=True) != "spawn":
            mp.set_start_method("spawn", force=True)
        try:
            venv = SubprocVecEnv(
                [_make_vec_env_fn(env_id, dim, seed + i, cfg) for i in range(n_envs)]
            )
        except Exception as e:
            print(f"SubprocVecEnv failed ({e}). Falling back to DummyVecEnv.")
            venv = DummyVecEnv(
                [_make_vec_env_fn(env_id, dim, seed + i, cfg) for i in range(n_envs)]
            )
    else:
        venv = DummyVecEnv([_make_vec_env_fn(env_id, dim, seed, cfg)])

    os.makedirs(sp, exist_ok=True)
    monitor_file = os.path.join(sp, "train_monitor")
    return VecMonitor(venv, filename=monitor_file)


def _build_eval_env(env_id: str, dim: int, env_config: dict) -> gym.Env:
    """A single, unmonitored env for held-out evaluation.

    Deliberately *not* the training env and deliberately not vectorised:
    :func:`agents.evaluation.evaluate_policy_twr` pins one scenario per
    ``reset(seed=...)`` and needs `info["distance"]` per step, which a VecEnv
    would bury behind autoreset.
    """
    env = gym.make(
        env_id, render_mode=None, dim=dim, disable_env_checker=True, config=env_config
    )
    return DecisionIntervalWrapper(env)


def _resolve_model_path(save_path: str) -> str | None:
    """Prefer the best-by-TWR checkpoint, fall back to the final one."""
    for stem in ("best_twr_model", "qr_dqn"):
        path = os.path.join(save_path, stem)
        if os.path.exists(path + ".zip"):
            return path
    return None


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(
    dim: int,
    verbose: int = 0,
    render_freq=None,
    use_gpu: bool = False,
    hpc: bool = False,
    n_envs: int = None,
    balloon_type: str = "zero_pressure",
    momentum_exploration: bool = True,
    total_timesteps: int | None = None,
) -> pd.DataFrame:
    """
    Train QR-DQN on the balloon environment. Returns a DataFrame of episode returns/lengths.

    Checkpoint selection is by **held-out TWR**, not by episode return. Return is
    not a usable selection criterion any more: the reward carries a multiplicative
    resource penalty, so the achievable maximum is neither 43,200 nor analytically
    obvious, and it moves whenever the shaping is retuned. The old
    ``StopTrainingOnRewardThreshold`` at 83,000/86,400 was an early stop at ~96%
    time-within-radius — a statement that the environment was trivial, not that
    the agent was good (roadmap §3.1). TWR is fixed by the mission definition, so
    it stays comparable across layers.

    ``total_timesteps`` overrides :data:`_TOTAL_TIMESTEPS` for this run. Its
    reason to exist is the pilot run: the full budget is 15M steps, and there is
    no point discovering that checkpointing or the TWR callback is broken three
    days in. When the override is short enough that the normal
    :data:`EVAL_FREQ` would never fire, the evaluation cadence is compressed so
    a pilot still exercises the eval and best-checkpoint paths — a smoke test
    that skips the very machinery it is meant to smoke-test is worthless.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)

    device = (
        torch.device("cuda")
        if (use_gpu and torch.cuda.is_available())
        else torch.device("cpu")
    )

    env_name = _ENV_NAMES.get(balloon_type, _ENV_NAMES["zero_pressure"])
    save_path = os.path.join(BASE_SAVE_PATH, balloon_type)
    model_path = os.path.join(save_path, "qr_dqn")
    train_cfg = _TRAIN_CFG[balloon_type]
    if total_timesteps is None:
        total_timesteps = _TOTAL_TIMESTEPS[balloon_type]
    else:
        total_timesteps = int(total_timesteps)
        if total_timesteps <= 0:
            raise ValueError(f"total_timesteps must be positive, got {total_timesteps}.")
    env_config = {**_ENV_CONFIG[balloon_type], "balloon_type": balloon_type}

    n = n_envs if n_envs is not None else N_ENVS
    _check_train_seeds(SEED, n)
    print(
        f"Training with {n} environments, dim={dim}, balloon_type={balloon_type}, "
        f"momentum_exploration={momentum_exploration}."
    )

    env = _build_vec_env(
        n, env_name, dim=dim, seed=SEED, save_path=save_path, env_config=env_config
    )

    model = MomentumQRDQN(
        policy="MlpPolicy",
        env=env,
        tensorboard_log=save_path,
        device=device,
        **train_cfg,
        policy_kwargs=POLICY_KWARGS,
        seed=SEED,
        momentum_exploration=momentum_exploration,
    )

    # Held-out evaluation (roadmap §3.1). Scenario seeds come from a range
    # disjoint from the training seeds by construction, so these episodes are
    # genuinely unseen.
    scenarios = make_scenario_set(
        N_EVAL_SCENARIOS, seed=SCENARIO_SEED, held_out=True, config=env_config
    )
    eval_env = _build_eval_env(env_name, dim=dim, env_config=env_config)

    # SB3's own EvalCallback is deliberately not used: it evaluates on the
    # training config, selects on mean return, and would only duplicate the
    # TensorBoard series TWREvalCallback already logs.
    # Normally EVAL_FREQ; compressed for a short pilot so the eval and
    # best-checkpoint paths actually run at least twice (see docstring).
    eval_every = EVAL_FREQ
    if total_timesteps < 2 * EVAL_FREQ:
        eval_every = max(total_timesteps // 2, 1)
        print(
            f"Short run ({total_timesteps:,} steps): eval cadence compressed "
            f"{EVAL_FREQ:,} -> {eval_every:,} steps so the pilot exercises evaluation."
        )

    twr_cb = TWREvalCallback(
        eval_env,
        scenarios,
        eval_freq=max(
            eval_every // n, 1
        ),  # callback counts calls; one call == n env steps
        best_model_save_path=save_path,
        log_path=save_path,
        deterministic=True,  # greedy action at eval
        max_episode_steps=DECISIONS_PER_EPISODE,
        verbose=1,
    )

    callbacks = [twr_cb]
    if hpc:
        callbacks.append(TerminationTracker())
    else:
        tqdm_cb = InfoProgressBar(
            description=f"QR-DQN | steps={total_timesteps:,} | envs={n} | device={device} |",
            postfix=dict(gamma=train_cfg["gamma"]),
        )
        callbacks.insert(0, tqdm_cb)
    callback = CallbackList(callbacks)

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        tb_log_name=f"QRDQN_run_{balloon_type}_dim{dim}",
    )
    model.save(model_path)
    env.close()
    eval_env.close()

    # Training summary
    evaluated = np.isfinite(twr_cb.best_twr)
    print(f"\n{'='*50}")
    print("Training complete")
    print(f"  Balloon type:     {balloon_type}")
    print(
        f"  Best held-out TWR: {twr_cb.best_twr:.3f}"
        if evaluated
        else "  Best held-out TWR: (never evaluated)"
    )
    print(
        f"  Last held-out TWR: {twr_cb.last_twr:.3f}"
        if evaluated
        else f"  Eval every:       {EVAL_FREQ:,} env steps"
    )
    print(f"  Total timesteps:  {model.num_timesteps:,} / {total_timesteps:,}")
    print(f"  Model saved to:   {os.path.abspath(model_path)}")
    print(
        f"  Best-TWR model:   {os.path.abspath(os.path.join(save_path, 'best_twr_model'))}"
    )
    if hpc:
        print(f"  Device:           {device}")
    print(f"{'='*50}\n")
    print("Compare against the baselines with:")
    print(f"  python main.py --benchmark --dim {dim} --balloon-type {balloon_type}\n")

    return _gather_monitor_csvs(save_path)


# --------------------------------------------------------------------------- #
# Benchmarking against the baselines
# --------------------------------------------------------------------------- #
def benchmark(
    dim: int = 3,
    balloon_type: str = "zero_pressure",
    n_scenarios: int = N_EVAL_SCENARIOS,
    use_gpu: bool = False,
    include_model: bool = True,
    scenario_seed: int = SCENARIO_SEED,
) -> dict:
    """Score every baseline (and the trained agent) on one held-out scenario set.

    The whole point of Layer 1's exit criterion is a comparison, so this runs
    every policy through the *same* scenarios and the *same* eval loop. Loon's
    heuristic scored 40.5% TWR50 against their learned controller's 55.1%: an
    agent that does not clear the greedy-wind row below has learned nothing
    worth having, whatever its reward curve looks like.

    Returns ``{policy_name: results_dict}`` from :func:`evaluate_policy_twr`.
    """
    env_name = _ENV_NAMES.get(balloon_type, _ENV_NAMES["zero_pressure"])
    save_path = os.path.join(BASE_SAVE_PATH, balloon_type)
    env_config = {**_ENV_CONFIG[balloon_type], "balloon_type": balloon_type}

    scenarios = make_scenario_set(
        n_scenarios, seed=scenario_seed, held_out=True, config=env_config
    )
    env = _build_eval_env(env_name, dim=dim, env_config=env_config)

    policies: dict[str, object] = {}
    for name in baselines_for_dim(dim):
        policies[name] = (
            make_baseline(name, seed=SEED) if name == "random" else make_baseline(name)
        )

    model_path = _resolve_model_path(save_path) if include_model else None
    if include_model:
        if model_path is None:
            print(
                f"No trained model under {os.path.abspath(save_path)} — baselines only."
            )
        else:
            device = (
                torch.device("cuda")
                if (use_gpu and torch.cuda.is_available())
                else torch.device("cpu")
            )
            policies["qr_dqn"] = QRDQN.load(model_path, device=device)
            print(f"Loaded agent from {os.path.abspath(model_path)}.zip")

    results: dict[str, dict] = {}
    for name, policy in policies.items():
        t0 = time.time()
        results[name] = evaluate_policy_twr(
            policy,
            env,
            scenarios,
            deterministic=True,
            max_episode_steps=DECISIONS_PER_EPISODE,
        )
        print(f"  {name:<12s} done in {time.time() - t0:5.1f} s")
    env.close()

    _print_benchmark_table(
        results,
        dim=dim,
        balloon_type=balloon_type,
        n_scenarios=n_scenarios,
        scenario_seed=scenario_seed,
    )
    return results


def _print_benchmark_table(
    results: dict, *, dim: int, balloon_type: str, n_scenarios: int, scenario_seed: int
) -> None:
    header = (
        f"Held-out benchmark | {balloon_type} | dim={dim} | "
        f"{n_scenarios} scenarios (meta-seed {scenario_seed})"
    )
    print(f"\n{header}")
    print("=" * len(header))
    print(
        f"{'policy':<14s}{'TWR':>8s}{'mean return':>14s}{'final dist (m)':>17s}{'ep len':>9s}"
    )
    print("-" * len(header))
    for name, r in results.items():
        print(
            f"{name:<14s}{r['twr']:>8.3f}{r['mean_return']:>14.1f}"
            f"{r['mean_final_distance']:>17,.0f}{r['mean_episode_length']:>9.1f}"
        )
    print("-" * len(header))

    heuristic = results.get("greedy_wind", {}).get("twr")
    agent = results.get("qr_dqn", {}).get("twr")
    if heuristic is not None and agent is not None:
        verdict = "CLEARS" if agent > heuristic else "DOES NOT CLEAR"
        print(
            f"Agent {verdict} the greedy-wind heuristic "
            f"({agent:.3f} vs {heuristic:.3f}, {agent - heuristic:+.3f})."
        )
    print()


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def test(dim: int, use_gpu: bool = False, balloon_type: str = "zero_pressure") -> None:
    """
    Load the saved QR-DQN and run a few episodes with greedy actions.
    """
    from environments.envs.balloon_3d_env import Actions  # your enum
    from environments.envs.balloon_3d_env import Balloon3DEnv, BalloonSP3DEnv

    _EnvCls = BalloonSP3DEnv if balloon_type == "superpressure" else Balloon3DEnv

    device = (
        torch.device("cuda")
        if (use_gpu and torch.cuda.is_available())
        else torch.device("cpu")
    )

    env_name = _ENV_NAMES.get(balloon_type, _ENV_NAMES["zero_pressure"])
    save_path = os.path.join(BASE_SAVE_PATH, balloon_type)
    model_path = _resolve_model_path(save_path)
    if model_path is None:
        raise FileNotFoundError(
            f"No QR-DQN checkpoint under {os.path.abspath(save_path)} "
            "(looked for best_twr_model.zip and qr_dqn.zip). Train one first."
        )
    env_config = {**_ENV_CONFIG[balloon_type], "balloon_type": balloon_type}

    # Human-render env for demo (shorter episode for interactive viewing)
    test_config = {
        **env_config,
        "time_max": 7_200,
    }  # 2 hours of physics -> 120 decisions
    env: gym.Env = Monitor(
        DecisionIntervalWrapper(
            gym.make(
                env_name,
                render_mode="human",
                dim=dim,
                disable_env_checker=True,
                config=test_config,
            )
        )
    )

    # Load model
    model: QRDQN = QRDQN.load(model_path, device=device)
    print(f"Loaded {os.path.abspath(model_path)}.zip")

    # (Optional) inspect Q-values for a single obs
    env_temp = DecisionIntervalWrapper(
        _EnvCls(dim=dim, render_mode=None, config=env_config)
    )
    obs, _ = env_temp.reset(seed=42)
    obs_tensor = torch.as_tensor(
        obs, dtype=torch.float32, device=model.device
    ).unsqueeze(0)
    with torch.no_grad():
        quantiles = model.quantile_net(obs_tensor)  # shape: [1, n_quantiles, n_actions]
        q_values = quantiles.mean(dim=1)  # mean across quantiles -> [1, n_actions]
        print("Q-values:", q_values.cpu().numpy())
    env_temp.close()

    # Roll a few episodes
    for episode in range(10):
        state, info = env.reset()
        done = False
        t0 = time.time()
        steps = 0
        while not done:
            steps += 1
            # Greedy at test time (deterministic=True)
            action_idx, _ = model.predict(state, deterministic=True)
            next_state, reward, terminated, truncated, info = env.step(action_idx)

            effect = int(env.unwrapped._action_lut[action_idx])
            act = (
                Actions(effect).name if effect in Actions._value2member_map_ else "?"
            ).upper()[:3]

            pos = env.unwrapped._balloon.pos
            if env.unwrapped.dim == 1:
                pos_str = f"z={pos[0]:+8.1f}"
            elif env.unwrapped.dim == 2:
                pos_str = f"{pos[0]:+8.1f},{pos[1]:+8.1f}"
            else:
                pos_str = f"{pos[0]:+8.1f},{pos[1]:+8.1f},{pos[2]:+8.1f}"

            c = info.get("reward_components", {})
            print(
                f"E{episode+1:<2}|S{steps:>5}|A:{act:<3}"
                f"|Pos:{pos_str}"
                f"|R:{reward:+8.3f}"
                f"|dist:{info.get('distance', float('nan')):>9,.0f}"
                f"|base:{c.get('base', 0):6.3f}"
                f" res:{c.get('resource_factor', 1):6.3f}"
            )

            state = next_state
            done = terminated or truncated

            # Check renderer flags (events are processed inside draw())
            renderer = env.unwrapped.renderer
            if renderer is not None:
                if renderer.quit_requested:
                    env.close()
                    return
                if renderer.skip_requested:
                    renderer.skip_requested = False
                    done = True

        t1 = time.time()
        print(f"{steps / (t1 - t0):.2f} steps/second")

        # Show end screen with termination reason
        reason = info.get("termination_reason", "Episode ended")
        renderer = env.unwrapped.renderer
        if renderer is not None:
            renderer.show_end_screen(reason)

    env.close()
