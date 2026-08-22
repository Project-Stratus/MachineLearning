"""Tests for the QR-DQN runner: config sanity, the seed guard, and the
momentum-exploration subclass.

Deliberately no training. `model.learn()` is never called — these tests verify
that the model *builds* against the frozen 143-wide observation (Layer 1
contract §1), that it predicts, and that the exploration path behaves. Whether
the agent is any good is a question for `python main.py --benchmark`, not for
the unit suite.
"""

from pathlib import Path
import numpy as np
import pytest
import torch

import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv

import agents.qrdqn as qrdqn
from agents.baselines import OBS_WIDTH, N_ACTIONS
from agents.evaluation import HELD_OUT_SEED_RANGE, TRAIN_SEED_RANGE
from agents.qrdqn import MomentumQRDQN, POLICY_KWARGS, _TRAIN_CFG, _check_train_seeds
from agents.utils import MomentumExplorer
from environments.core.constants import DECISION_INTERVAL, TIME_MAX
from environments.wrappers.decision_interval import DecisionIntervalWrapper


BALLOON_TYPES = ("zero_pressure", "superpressure")


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _make_env(dim: int = 3):
    env = gym.make(
        "environments/Balloon3D-v0",
        render_mode=None,
        dim=dim,
        disable_env_checker=True,
        config={
            "wind_pattern": "altitude_shear_2d",
            "balloon_type": "zero_pressure",
            "time_max": 600,
        },
    )
    return DecisionIntervalWrapper(env)


@pytest.fixture(scope="module")
def vec_env():
    """Two real envs — enough to exercise per-env exploration state."""
    venv = DummyVecEnv([lambda: _make_env(), lambda: _make_env()])
    yield venv
    venv.close()


@pytest.fixture(scope="module")
def model(vec_env):
    """A real MomentumQRDQN on the real 143-wide observation space.

    Module-scoped: building the four-layer policy is the expensive part and
    nothing here mutates the weights.
    """
    cfg = {**_TRAIN_CFG["zero_pressure"], "buffer_size": 1_000, "learning_starts": 10}
    return MomentumQRDQN(
        policy="MlpPolicy",
        env=vec_env,
        device=torch.device("cpu"),
        policy_kwargs=POLICY_KWARGS,
        seed=qrdqn.SEED,
        **cfg,
    )


# --------------------------------------------------------------------------- #
# Config sanity
# --------------------------------------------------------------------------- #
class TestTrainingConfig:
    @pytest.mark.parametrize("balloon_type", BALLOON_TYPES)
    def test_gamma_covers_roughly_half_the_episode(self, balloon_type):
        gamma = _TRAIN_CFG[balloon_type]["gamma"]
        assert gamma == 0.997
        # 1/(1-gamma) decisions x DECISION_INTERVAL seconds, against a 12 h episode.
        horizon_h = (1.0 / (1.0 - gamma)) * DECISION_INTERVAL / 3600.0
        episode_h = TIME_MAX * 1.0 / 3600.0
        assert horizon_h == pytest.approx(5.56, abs=0.05)
        assert 0.35 < horizon_h / episode_h < 0.6, (
            "discount horizon no longer covers a useful fraction of the episode — "
            "re-derive gamma (see the comment above _TRAIN_CFG)"
        )

    def test_network_is_four_layers_of_512(self):
        assert POLICY_KWARGS["net_arch"] == [512, 512, 512, 512]

    def test_quantile_head_is_untouched(self):
        # Roadmap §2.2: inert until Layer 3. Changing it here would be noise.
        assert POLICY_KWARGS["n_quantiles"] == 51

    def test_reward_threshold_early_stopping_is_gone(self):
        # Roadmap §3.1: a return threshold calibrated to a trivial env is exactly
        # what hid the problem. Nothing may reintroduce one. Selection is by
        # held-out TWR, via TWREvalCallback.
        assert not hasattr(qrdqn, "_REWARD_THRESHOLD")
        assert not hasattr(qrdqn, "StopTrainingOnRewardThreshold")
        assert not hasattr(qrdqn, "EvalCallback")
        assert hasattr(qrdqn, "TWREvalCallback")

    def test_decisions_per_episode(self):
        assert qrdqn.DECISIONS_PER_EPISODE == 720


# --------------------------------------------------------------------------- #
# Train / held-out seed split
# --------------------------------------------------------------------------- #
class TestSeedGuard:
    def test_default_seed_is_inside_the_training_range(self):
        _check_train_seeds(qrdqn.SEED, qrdqn.MAX_ENVS)  # must not raise
        lo, hi = TRAIN_SEED_RANGE
        assert lo <= qrdqn.SEED < hi

    def test_guard_fires_on_a_held_out_seed(self):
        held_out_lo = HELD_OUT_SEED_RANGE[0]
        with pytest.raises(ValueError, match="TRAIN_SEED_RANGE"):
            _check_train_seeds(held_out_lo, n_envs=1)

    def test_guard_fires_when_worker_offsets_cross_the_boundary(self):
        # seed itself is legal; seed + (n_envs - 1) is not.
        with pytest.raises(ValueError, match="TRAIN_SEED_RANGE"):
            _check_train_seeds(TRAIN_SEED_RANGE[1] - 4, n_envs=8)

    def test_guard_fires_below_the_range(self):
        with pytest.raises(ValueError):
            _check_train_seeds(-1, n_envs=1)

    def test_build_vec_env_enforces_the_guard(self):
        with pytest.raises(ValueError, match="TRAIN_SEED_RANGE"):
            qrdqn._build_vec_env(
                1, "environments/Balloon3D-v0", dim=1, seed=HELD_OUT_SEED_RANGE[0]
            )


# --------------------------------------------------------------------------- #
# The model builds and predicts against the frozen observation
# --------------------------------------------------------------------------- #
class TestModelBuildsAgainstFrozenObservation:
    def test_observation_space_is_143_wide(self, vec_env):
        assert vec_env.observation_space.shape == (OBS_WIDTH,)
        assert vec_env.action_space.n == N_ACTIONS

    def test_policy_input_and_output_widths(self, model):
        layers = [
            m
            for m in model.policy.quantile_net.quantile_net
            if isinstance(m, torch.nn.Linear)
        ]
        assert layers[0].in_features == OBS_WIDTH
        assert [layer.out_features for layer in layers[:-1]] == POLICY_KWARGS[
            "net_arch"
        ]
        # Quantile head: n_quantiles x n_actions, flattened.
        assert layers[-1].out_features == POLICY_KWARGS["n_quantiles"] * N_ACTIONS

    def test_predicts_on_a_batched_observation(self, model, vec_env):
        obs = vec_env.reset()
        assert obs.shape == (2, OBS_WIDTH)
        action, state = model.predict(obs, deterministic=True)
        assert state is None
        assert action.shape == (2,)
        assert np.all((action >= 0) & (action < N_ACTIONS))

    def test_predicts_on_a_single_observation(self, model, vec_env):
        obs = vec_env.reset()[0]
        action, _ = model.predict(obs, deterministic=True)
        assert 0 <= int(action) < N_ACTIONS

    def test_quantile_net_forward(self, model, vec_env):
        obs = torch.as_tensor(vec_env.reset(), dtype=torch.float32)
        with torch.no_grad():
            quantiles = model.quantile_net(obs)
        assert quantiles.shape == (2, POLICY_KWARGS["n_quantiles"], N_ACTIONS)


# --------------------------------------------------------------------------- #
# Momentum exploration wiring
# --------------------------------------------------------------------------- #
class TestMomentumWiring:
    def test_model_owns_an_explorer(self, model):
        assert model.momentum_exploration is True
        assert isinstance(model._explorer, MomentumExplorer)

    def test_explorer_is_excluded_from_saved_params(self, model):
        assert "_explorer" in model._excluded_save_params()

    def test_warmup_uses_momentum_not_uniform(self, model, vec_env):
        model._last_obs = vec_env.reset()
        model.num_timesteps = 0
        actions, buffer_actions = model._sample_action(learning_starts=1_000, n_envs=2)
        assert actions.shape == (2,) and actions.dtype == np.int64
        np.testing.assert_array_equal(actions, buffer_actions)
        # The explorer was consulted, so per-env targets now exist.
        assert model._explorer.n_envs == 2

    def test_epsilon_zero_gives_the_greedy_action(self, model, vec_env):
        model._last_obs = vec_env.reset()
        model.num_timesteps = 10_000_000
        model.exploration_rate = 0.0
        actions, _ = model._sample_action(learning_starts=10, n_envs=2)
        greedy, _ = model.policy.predict(model._last_obs, deterministic=True)
        np.testing.assert_array_equal(actions, np.asarray(greedy).reshape(-1))

    def test_epsilon_one_gives_the_momentum_action(self, model, vec_env):
        """With epsilon pinned at 1, every action must point at that env's target."""
        from agents.utils import ALT_SAFE_MIN, IDX_ALT_NORM

        model._last_obs = vec_env.reset()
        model.num_timesteps = 10_000_000
        model.exploration_rate = 1.0
        model._explorer = MomentumExplorer(seed=0)
        actions, _ = model._sample_action(learning_starts=10, n_envs=2)

        explorer = model._explorer
        alt = ALT_SAFE_MIN + model._last_obs[:, IDX_ALT_NORM] * explorer.span
        delta = explorer.target_alt - alt
        expected = np.where(
            delta > explorer.deadband, 2, np.where(delta < -explorer.deadband, 0, 1)
        )
        np.testing.assert_array_equal(actions, expected)

    def test_exploration_stream_is_temporally_correlated(self, model, vec_env):
        """The whole point (roadmap §3.8), measured through the model itself."""
        obs = vec_env.reset()
        model.num_timesteps = 10_000_000
        model.exploration_rate = 1.0  # always explore
        model._explorer = MomentumExplorer(seed=0)

        streams = []
        for _ in range(150):
            model._last_obs = obs
            actions, _ = model._sample_action(learning_starts=10, n_envs=2)
            obs, _r, _d, _i = vec_env.step(actions)
            streams.append(actions)
        stream = np.asarray(streams)  # (150, 2)

        for env_idx in range(2):
            a = stream[:, env_idx]
            persistence = float(np.mean(a[1:] == a[:-1]))
            assert persistence > 0.75, (
                f"env {env_idx}: exploratory actions are not persistent "
                f"({persistence:.3f}); uniform random would score ~0.33"
            )

    def test_per_env_exploration_state_is_independent(self, model, vec_env):
        model._last_obs = vec_env.reset()
        model.num_timesteps = 0
        model._sample_action(learning_starts=1_000, n_envs=2)

        targets = model._explorer.target_alt.copy()
        assert targets[0] != targets[1], "both actors drew the same target altitude"

        # An episode ending in env 0 must not disturb env 1's intention.
        model._explorer.reset_envs(np.array([True, False]))
        assert model._explorer.target_alt[0] != targets[0]
        assert model._explorer.target_alt[1] == targets[1]

    def test_disabling_momentum_restores_stock_epsilon_greedy(self, vec_env):
        cfg = {
            **_TRAIN_CFG["zero_pressure"],
            "buffer_size": 1_000,
            "learning_starts": 10,
        }
        plain = MomentumQRDQN(
            policy="MlpPolicy",
            env=vec_env,
            device=torch.device("cpu"),
            policy_kwargs=dict(net_arch=[16], n_quantiles=5),
            seed=qrdqn.SEED,
            momentum_exploration=False,
            **cfg,
        )
        plain._last_obs = vec_env.reset()
        plain.num_timesteps = 0
        actions, _ = plain._sample_action(learning_starts=1_000, n_envs=2)
        assert actions.shape == (2,)
        # Stock SB3 warm-up samples uniformly and never touches the explorer.
        assert plain._explorer.n_envs == 0

    def test_uniform_exploration_is_not_persistent(self, vec_env):
        """Control for the correlation test above: the thing we replaced."""
        rng = np.random.default_rng(0)
        obs = vec_env.reset()
        stream = []
        for _ in range(150):
            actions = rng.integers(N_ACTIONS, size=2)
            obs, _r, _d, _i = vec_env.step(actions)
            stream.append(actions)
        a = np.asarray(stream)
        for env_idx in range(2):
            col = a[:, env_idx]
            assert float(np.mean(col[1:] == col[:-1])) < 0.55


# --------------------------------------------------------------------------- #
# Evaluation / benchmark wiring
# --------------------------------------------------------------------------- #
class TestBenchmarkWiring:
    def test_resolve_model_path_prefers_best_twr(self, tmp_path):
        (tmp_path / "qr_dqn.zip").write_bytes(b"")
        assert qrdqn._resolve_model_path(str(tmp_path)).endswith("qr_dqn")
        (tmp_path / "best_twr_model.zip").write_bytes(b"")
        assert qrdqn._resolve_model_path(str(tmp_path)).endswith("best_twr_model")

    def test_resolve_model_path_returns_none_when_untrained(self, tmp_path):
        assert qrdqn._resolve_model_path(str(tmp_path)) is None

    def test_eval_scenarios_are_held_out(self):
        scenarios = qrdqn.make_scenario_set(
            qrdqn.N_EVAL_SCENARIOS, seed=qrdqn.SCENARIO_SEED, held_out=True
        )
        lo, hi = HELD_OUT_SEED_RANGE
        assert len(scenarios) == qrdqn.N_EVAL_SCENARIOS
        assert all(lo <= s["seed"] < hi for s in scenarios)

    def test_benchmark_covers_every_baseline(self):
        from agents.baselines import BASELINES

        assert set(BASELINES) == {"passive", "random", "greedy_wind", "bang_bang"}

    def test_benchmark_scores_bang_bang_in_1d_only(self):
        """bang_bang is an altitude-hold policy; 2D/3D pin goal_dz_norm to 0, so
        scoring it there would just duplicate the passive row."""
        from agents.baselines import baselines_for_dim

        assert "bang_bang" in baselines_for_dim(1)
        assert "bang_bang" not in baselines_for_dim(2)
        assert "bang_bang" not in baselines_for_dim(3)
        # Everything else is dimension-agnostic and must survive the filter.
        for dim in (1, 2, 3):
            assert {"passive", "random", "greedy_wind"}.issubset(baselines_for_dim(dim))


class TestBaselineReference:
    """The recorded baselines are annotation, but a wrong annotation is worse
    than none: it is what the training curve draws as the bar to clear."""

    def test_every_dim_has_a_bar_and_a_floor(self):
        for dim in (1, 2, 3):
            ref, bar = qrdqn.baseline_reference(dim)
            assert bar in ref, f"dim={dim} names a bar it has no measurement for"
            assert "passive" in ref, f"dim={dim} has no passive floor"
            assert all(0.0 <= v <= 1.0 for v in ref.values())

    def test_bar_is_the_strongest_measured_baseline(self):
        for dim in (1, 2, 3):
            ref, bar = qrdqn.baseline_reference(dim)
            assert ref[bar] == max(ref.values()), f"dim={dim} bar is not the strongest"

    def test_bang_bang_is_the_1d_bar_not_greedy_wind(self):
        """greedy_wind has no horizontal objective in 1D and ties with passive,
        so using it as the 1D bar would set the target ~100x too low."""
        ref, bar = qrdqn.baseline_reference(1)
        assert bar == "bang_bang"
        assert ref["greedy_wind"] == pytest.approx(ref["passive"], abs=0.005)

    def test_unmeasured_dim_raises_rather_than_defaulting(self):
        with pytest.raises(KeyError, match="--benchmark"):
            qrdqn.baseline_reference(4)


class TestTimestepOverride:
    """`--timesteps` exists so a pilot run can prove the pipeline before a
    multi-day one commits to it."""

    def test_train_accepts_a_total_timesteps_override(self):
        import inspect
        sig = inspect.signature(qrdqn.train)
        assert "total_timesteps" in sig.parameters
        assert sig.parameters["total_timesteps"].default is None

    def test_default_budget_is_still_the_module_constant(self):
        assert qrdqn._TOTAL_TIMESTEPS["zero_pressure"] == 15_000_000
        assert qrdqn._TOTAL_TIMESTEPS["superpressure"] == 15_000_000

    @pytest.mark.parametrize("bad", [0, -1, -15_000_000])
    def test_rejects_non_positive_budgets(self, bad, monkeypatch):
        """Caught before any env or model is built, so a typo fails in a second
        rather than after the vec-env spins up."""
        def boom(*a, **k):
            raise AssertionError("built the env despite an invalid budget")
        monkeypatch.setattr(qrdqn, "_build_vec_env", boom)
        with pytest.raises(ValueError, match="total_timesteps must be positive"):
            qrdqn.train(dim=3, total_timesteps=bad)

    def test_cli_exposes_the_flag(self):
        """main.py is the only entry point most runs go through."""
        main_src = (Path(__file__).resolve().parents[2] / "main.py").read_text()
        assert "--timesteps" in main_src
        assert "total_timesteps=args.timesteps" in main_src

    def test_short_runs_compress_the_eval_cadence(self):
        """A pilot shorter than 2x EVAL_FREQ would otherwise never evaluate,
        so it would not test the callback or best-checkpoint path at all."""
        for budget in (100_000, 200_000, 2 * qrdqn.EVAL_FREQ - 1):
            assert budget < 2 * qrdqn.EVAL_FREQ
            assert max(budget // 2, 1) < qrdqn.EVAL_FREQ

    def test_full_budget_keeps_the_normal_cadence(self):
        assert qrdqn._TOTAL_TIMESTEPS["zero_pressure"] >= 2 * qrdqn.EVAL_FREQ
