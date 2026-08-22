"""Tests for the TWR evaluation framework.

The env is stubbed rather than real: these tests are about the *metric* and the
*scenario bookkeeping*, and a scripted distance trace is the only way to assert
an exact TWR. It also keeps them independent of the physics core.
"""

import os
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger

from agents.baselines import ACTION_STAY, OBS_WIDTH, PassiveDriftAgent, RandomAgent
from agents.evaluation import (
    HELD_OUT_SEED_RANGE,
    TRAIN_SEED_RANGE,
    TWREvalCallback,
    evaluate_policy_twr,
    is_held_out_seed,
    make_scenario_set,
    scenario_seeds,
    seed_range,
)

RADIUS = 10_000.0


# --------------------------------------------------------------------------- #
# Stubs
# --------------------------------------------------------------------------- #
class StubEnv:
    """Minimal gymnasium-style env replaying a scripted distance trace.

    Each episode emits ``distances`` one per step, then ends. ``info['distance']``
    is the only channel the evaluator is allowed to read distances from, so this
    stub proves the contract rather than the physics.
    """

    def __init__(self, distances, rewards=None, termination_reason=None,
                 terminate=False, max_episode_steps=None):
        self.distances = list(distances)
        self.rewards = list(rewards) if rewards is not None else [1.0] * len(self.distances)
        self.termination_reason = termination_reason
        self.terminate = terminate
        self.spec = SimpleNamespace(
            max_episode_steps=max_episode_steps if max_episode_steps is not None
            else len(self.distances)
        )
        self.seeds_seen = []
        self.actions_seen = []
        self._t = 0

    def _obs(self):
        return np.zeros(OBS_WIDTH, dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.seeds_seen.append(seed)
        self._t = 0
        return self._obs(), {}

    def step(self, action):
        self.actions_seen.append(action)
        distance = self.distances[self._t]
        reward = self.rewards[self._t]
        self._t += 1
        last = self._t >= len(self.distances)
        terminated = bool(last and self.terminate)
        truncated = bool(last and not self.terminate)
        info = {"distance": distance}
        if last and self.termination_reason is not None:
            info["termination_reason"] = self.termination_reason
        return self._obs(), reward, terminated, truncated, info


class NoDistanceEnv(StubEnv):
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info.pop("distance")
        return obs, reward, terminated, truncated, info


class _RecordingLogger(Logger):
    """Logger that keeps a snapshot of every dump so tests can inspect keys."""

    def __init__(self):
        super().__init__(folder=None, output_formats=[])
        self.dumps = []

    def dump(self, step=0):
        self.dumps.append(dict(self.name_to_value))
        super().dump(step)


class StubModel:
    """Stands in for a trained SB3 model: `predict`, `save`, `logger`, timesteps."""

    def __init__(self, action=ACTION_STAY):
        self.action = action
        self.logger = _RecordingLogger()
        self.num_timesteps = 1_000
        self.saved = []

    def predict(self, obs, deterministic=True, state=None, episode_start=None):
        return np.int64(self.action), None

    def save(self, path):
        self.saved.append(path)


# --------------------------------------------------------------------------- #
# Scenario sets
# --------------------------------------------------------------------------- #
class TestMakeScenarioSet:
    def test_shape_and_keys(self):
        scenarios = make_scenario_set(5, seed=0)
        assert len(scenarios) == 5
        for i, sc in enumerate(scenarios):
            assert set(sc) == {"scenario_id", "seed", "held_out", "config"}
            assert sc["scenario_id"] == f"heldout-{i:04d}"
            assert sc["held_out"] is True
            assert isinstance(sc["seed"], int)

    def test_deterministic_for_a_given_seed(self):
        assert make_scenario_set(20, seed=42) == make_scenario_set(20, seed=42)

    def test_different_meta_seeds_differ(self):
        assert scenario_seeds(make_scenario_set(20, seed=1)) != scenario_seeds(
            make_scenario_set(20, seed=2)
        )

    def test_seeds_are_unique_within_a_set(self):
        seeds = scenario_seeds(make_scenario_set(200, seed=3))
        assert len(set(seeds)) == 200

    def test_held_out_seeds_are_in_the_held_out_range(self):
        lo, hi = HELD_OUT_SEED_RANGE
        assert all(lo <= s < hi for s in scenario_seeds(make_scenario_set(100, seed=7)))

    def test_training_seeds_are_in_the_training_range(self):
        lo, hi = TRAIN_SEED_RANGE
        seeds = scenario_seeds(make_scenario_set(100, seed=7, held_out=False))
        assert all(lo <= s < hi for s in seeds)

    def test_train_and_held_out_are_disjoint(self):
        """The whole point: no held-out scenario can ever have been trained on."""
        train = set(scenario_seeds(make_scenario_set(500, seed=11, held_out=False)))
        held = set(scenario_seeds(make_scenario_set(500, seed=11, held_out=True)))
        assert train.isdisjoint(held)
        # Disjoint by construction, not by luck — the ranges cannot overlap.
        assert TRAIN_SEED_RANGE[1] <= HELD_OUT_SEED_RANGE[0]

    def test_same_meta_seed_gives_unrelated_draws_not_a_shift(self):
        offset = HELD_OUT_SEED_RANGE[0] - TRAIN_SEED_RANGE[0]
        train = scenario_seeds(make_scenario_set(50, seed=5, held_out=False))
        held = scenario_seeds(make_scenario_set(50, seed=5, held_out=True))
        assert [s + offset for s in train] != held

    def test_config_overrides_are_attached_and_copied(self):
        cfg = {"wind_pattern": "altitude_shear_2d"}
        scenarios = make_scenario_set(3, seed=1, config=cfg)
        assert all(sc["config"] == cfg for sc in scenarios)
        scenarios[0]["config"]["wind_pattern"] = "mutated"
        assert scenarios[1]["config"]["wind_pattern"] == "altitude_shear_2d"
        assert cfg["wind_pattern"] == "altitude_shear_2d"

    def test_rejects_non_positive_n(self):
        with pytest.raises(ValueError):
            make_scenario_set(0, seed=1)

    def test_is_held_out_seed(self):
        assert is_held_out_seed(HELD_OUT_SEED_RANGE[0])
        assert is_held_out_seed(HELD_OUT_SEED_RANGE[1] - 1)
        assert not is_held_out_seed(HELD_OUT_SEED_RANGE[1])
        assert not is_held_out_seed(42)  # qrdqn.SEED lives in the training range

    def test_seed_range_helper(self):
        assert seed_range(True) == HELD_OUT_SEED_RANGE
        assert seed_range(False) == TRAIN_SEED_RANGE


# --------------------------------------------------------------------------- #
# evaluate_policy_twr
# --------------------------------------------------------------------------- #
class TestEvaluatePolicyTWR:
    def test_twr_on_a_scripted_trace(self):
        # 3 of 5 samples inside the radius; the boundary sample counts as inside.
        env = StubEnv([0.0, 5_000.0, 10_000.0, 15_000.0, 20_000.0])
        scenarios = make_scenario_set(4, seed=0)
        results = evaluate_policy_twr(PassiveDriftAgent(), env, scenarios,
                                      station_radius=RADIUS)

        assert results["twr"] == pytest.approx(0.6)
        assert results["twr_pooled"] == pytest.approx(0.6)
        assert results["mean_return"] == pytest.approx(5.0)
        assert results["mean_final_distance"] == pytest.approx(20_000.0)
        assert results["n_episodes"] == 4
        assert results["mean_episode_length"] == pytest.approx(5.0)
        assert results["horizon"] == 5

    def test_all_in_radius_scores_one(self):
        env = StubEnv([0.0] * 10)
        results = evaluate_policy_twr(PassiveDriftAgent(), env,
                                      make_scenario_set(2, seed=0), station_radius=RADIUS)
        assert results["twr"] == pytest.approx(1.0)

    def test_none_in_radius_scores_zero(self):
        env = StubEnv([50_000.0] * 10)
        results = evaluate_policy_twr(PassiveDriftAgent(), env,
                                      make_scenario_set(2, seed=0), station_radius=RADIUS)
        assert results["twr"] == pytest.approx(0.0)

    def test_early_termination_is_not_rewarded(self):
        """Parking on station and then dying must not score a perfect TWR."""
        env = StubEnv([0.0, 0.0], terminate=True, termination_reason="deflated",
                      max_episode_steps=10)
        results = evaluate_policy_twr(PassiveDriftAgent(), env,
                                      make_scenario_set(3, seed=0), station_radius=RADIUS)
        # 2 in-radius steps out of a 10-step scheduled flight.
        assert results["twr"] == pytest.approx(0.2)
        # Step-weighted pooling is the number that *would* have said 1.0.
        assert results["twr_pooled"] == pytest.approx(1.0)

    def test_explicit_horizon_overrides_env_spec(self):
        env = StubEnv([0.0] * 5, max_episode_steps=5)
        results = evaluate_policy_twr(PassiveDriftAgent(), env,
                                      make_scenario_set(1, seed=0),
                                      station_radius=RADIUS, max_episode_steps=20)
        assert results["horizon"] == 20
        assert results["twr"] == pytest.approx(0.25)

    def test_horizon_falls_back_to_longest_episode(self):
        env = StubEnv([0.0] * 4)
        env.spec = None
        results = evaluate_policy_twr(PassiveDriftAgent(), env,
                                      make_scenario_set(2, seed=0), station_radius=RADIUS)
        assert results["horizon"] == 4
        assert results["twr"] == pytest.approx(1.0)

    def test_scenarios_pin_the_reset_seed(self):
        env = StubEnv([0.0, 0.0])
        scenarios = make_scenario_set(5, seed=13)
        evaluate_policy_twr(PassiveDriftAgent(), env, scenarios, station_radius=RADIUS)
        assert env.seeds_seen == scenario_seeds(scenarios)

    def test_termination_counts(self):
        env = StubEnv([1.0], terminate=True, termination_reason="ballast_empty")
        results = evaluate_policy_twr(PassiveDriftAgent(), env,
                                      make_scenario_set(3, seed=0), station_radius=RADIUS)
        assert results["termination_counts"] == {"ballast_empty": 3}

    def test_truncation_counts_as_time_limit(self):
        env = StubEnv([1.0])
        results = evaluate_policy_twr(PassiveDriftAgent(), env,
                                      make_scenario_set(2, seed=0), station_radius=RADIUS)
        assert results["termination_counts"] == {"time_limit": 2}

    def test_returns_are_summed_rewards(self):
        env = StubEnv([0.0, 0.0, 0.0], rewards=[1.0, 0.5, 0.25])
        results = evaluate_policy_twr(PassiveDriftAgent(), env,
                                      make_scenario_set(2, seed=0), station_radius=RADIUS)
        assert results["mean_return"] == pytest.approx(1.75)
        assert results["episode_returns"] == pytest.approx([1.75, 1.75])

    def test_policy_actions_reach_the_env(self):
        env = StubEnv([0.0] * 3)
        evaluate_policy_twr(PassiveDriftAgent(), env, make_scenario_set(2, seed=0),
                            station_radius=RADIUS)
        assert env.actions_seen == [ACTION_STAY] * 6
        assert all(isinstance(a, int) for a in env.actions_seen)

    def test_works_with_a_stateful_baseline(self):
        env = StubEnv([0.0] * 5)
        results = evaluate_policy_twr(RandomAgent(seed=4), env,
                                      make_scenario_set(3, seed=0), station_radius=RADIUS)
        assert results["n_episodes"] == 3
        assert set(env.actions_seen) <= {0, 1, 2}

    def test_empty_scenario_set(self):
        results = evaluate_policy_twr(PassiveDriftAgent(), StubEnv([0.0]), [])
        assert results["n_episodes"] == 0
        assert results["twr"] == 0.0
        assert results["termination_counts"] == {}

    def test_missing_distance_is_an_error(self):
        with pytest.raises(KeyError, match="distance"):
            evaluate_policy_twr(PassiveDriftAgent(), NoDistanceEnv([0.0]),
                                make_scenario_set(1, seed=0))

    def test_per_episode_breakdown(self):
        env = StubEnv([0.0, 20_000.0])
        results = evaluate_policy_twr(PassiveDriftAgent(), env,
                                      make_scenario_set(3, seed=0), station_radius=RADIUS)
        assert results["episode_twr"] == pytest.approx([0.5, 0.5, 0.5])
        assert results["episode_lengths"] == [2, 2, 2]


# --------------------------------------------------------------------------- #
# TWREvalCallback
# --------------------------------------------------------------------------- #
class TestTWREvalCallback:
    def _callback(self, env, save_path=None, log_path=None, **kwargs):
        cb = TWREvalCallback(
            env,
            make_scenario_set(2, seed=0),
            eval_freq=1,
            best_model_save_path=save_path,
            log_path=log_path,
            station_radius=RADIUS,
            verbose=0,
            **kwargs,
        )
        model = StubModel()
        cb.init_callback(model)
        return cb, model

    def test_logs_twr_and_mean_return(self):
        cb, model = self._callback(StubEnv([0.0, 0.0, 20_000.0, 20_000.0]))
        assert cb.on_step() is True
        assert len(model.logger.dumps) == 1
        recorded = model.logger.dumps[0]
        assert "eval/twr" in recorded
        assert "eval/mean_return" in recorded
        assert recorded["eval/twr"] == pytest.approx(0.5)
        assert recorded["eval/mean_return"] == pytest.approx(4.0)

    def test_tracks_last_and_best(self):
        cb, _ = self._callback(StubEnv([0.0, 20_000.0]))
        cb.on_step()
        assert cb.last_twr == pytest.approx(0.5)
        assert cb.best_twr == pytest.approx(0.5)
        assert cb.last_results["n_episodes"] == 2

    def test_saves_best_by_twr_not_by_return(self):
        """A higher-return but lower-TWR eval must not overwrite the checkpoint."""
        with tempfile.TemporaryDirectory() as tmp:
            good = StubEnv([0.0, 0.0, 0.0, 0.0], rewards=[1.0] * 4)     # twr 1.0, return 4
            cb, model = self._callback(good, save_path=tmp)
            cb.on_step()
            assert len(model.saved) == 1
            assert cb.best_twr == pytest.approx(1.0)
            assert os.path.basename(model.saved[0]) == "best_twr_model"

            # Same callback, worse TWR but far higher return.
            cb.eval_env = StubEnv([20_000.0] * 4, rewards=[100.0] * 4)
            cb.on_step()
            assert cb.last_twr == pytest.approx(0.0)
            assert cb.last_results["mean_return"] == pytest.approx(400.0)
            assert len(model.saved) == 1          # not overwritten
            assert cb.best_twr == pytest.approx(1.0)

    def test_saves_again_on_improvement(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb, model = self._callback(StubEnv([20_000.0, 20_000.0]), save_path=tmp)
            cb.on_step()
            assert len(model.saved) == 1  # first eval always sets the bar
            cb.eval_env = StubEnv([0.0, 0.0])
            cb.on_step()
            assert len(model.saved) == 2
            assert cb.best_twr == pytest.approx(1.0)

    def test_eval_freq_gates_evaluation(self):
        cb = TWREvalCallback(StubEnv([0.0]), make_scenario_set(1, seed=0),
                             eval_freq=3, station_radius=RADIUS, verbose=0)
        model = StubModel()
        cb.init_callback(model)
        for _ in range(2):
            cb.on_step()
        assert model.logger.dumps == []
        cb.on_step()
        assert len(model.logger.dumps) == 1

    def test_writes_evaluation_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            cb, _ = self._callback(StubEnv([0.0, 20_000.0]), log_path=tmp)
            cb.on_step()
            cb.on_step()
            data = np.load(os.path.join(tmp, "twr_evaluations.npz"))
            assert data["twr"].tolist() == pytest.approx([0.5, 0.5])
            assert len(data["timesteps"]) == 2

    def test_callback_on_new_best_can_stop_training(self):
        class _Stop(BaseCallback):
            def _on_step(self) -> bool:
                return False

        cb, _ = self._callback(StubEnv([0.0, 0.0]), callback_on_new_best=_Stop())
        assert cb.on_step() is False

    def test_no_saving_when_path_is_none(self):
        cb, model = self._callback(StubEnv([0.0, 0.0]))
        cb.on_step()
        assert model.saved == []
