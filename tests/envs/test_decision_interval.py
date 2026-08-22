"""Tests for DecisionIntervalWrapper — the only wrapper Layer 1 keeps.

The wrapper decouples the agent's decision rate (60 s) from the physics step
(1 s). Its one subtlety is what it deliberately does *not* do: the resource
penalty is charged by the environment, not here, because charging it on the
single sub-step where the action fires would dilute it 60x. Those tests live
alongside it because that is where the bug would be reintroduced.
"""

import numpy as np
import pytest

from environments.core.constants import (
    BALLAST_DROP,
    BALLAST_INITIAL,
    DECISION_INTERVAL,
    RESOURCE_PENALTY_BASE,
)
from environments.wrappers import DecisionIntervalWrapper
from tests.conftest import make_env


@pytest.fixture
def wrapped_1d():
    env = DecisionIntervalWrapper(make_env(1, time_max=6_000))
    yield env
    env.close()


class TestWrapperBasics:
    def test_default_interval_comes_from_the_env_config(self, wrapped_1d):
        assert wrapped_1d.decision_interval == DECISION_INTERVAL

    def test_explicit_interval_overrides(self):
        env = DecisionIntervalWrapper(make_env(1, time_max=600), decision_interval=10)
        try:
            assert env.decision_interval == 10
        finally:
            env.close()

    def test_env_config_interval_is_honoured(self):
        env = DecisionIntervalWrapper(make_env(1, time_max=600, decision_interval=15))
        try:
            assert env.decision_interval == 15
            assert env.unwrapped._decision_interval == 15
        finally:
            env.close()

    def test_wrapper_tells_the_env_its_interval(self):
        """One source of truth: the env's resource hold must span the same steps."""
        env = DecisionIntervalWrapper(make_env(1, time_max=600), decision_interval=7)
        try:
            assert env.unwrapped._decision_interval == 7
        finally:
            env.close()

    def test_one_wrapper_step_is_n_physics_steps(self, wrapped_1d):
        wrapped_1d.reset(seed=1)
        wrapped_1d.step(1)
        assert wrapped_1d.unwrapped._time == DECISION_INTERVAL

    def test_action_fires_once_then_noops(self, wrapped_1d):
        """One valve opening per decision, then coast — matching the hardware."""
        wrapped_1d.reset(seed=1)
        before = wrapped_1d.unwrapped._balloon.ballast_mass
        wrapped_1d.step(2)
        dropped = before - wrapped_1d.unwrapped._balloon.ballast_mass
        assert dropped == pytest.approx(BALLAST_DROP)

    def test_reward_is_summed_over_the_interval(self, wrapped_1d):
        wrapped_1d.reset(seed=1)
        _, reward, _, _, _ = wrapped_1d.step(1)
        assert 0.0 < reward <= DECISION_INTERVAL

    def test_observation_and_info_come_from_the_last_sub_step(self, wrapped_1d):
        obs, _ = wrapped_1d.reset(seed=1)
        obs2, _, _, _, info = wrapped_1d.step(1)
        assert obs2.shape == obs.shape
        assert wrapped_1d.observation_space.contains(obs2)
        assert "distance" in info
        assert info["distance"] == pytest.approx(
            wrapped_1d.unwrapped._prev_distance)

    def test_distance_present_on_every_decision(self, wrapped_1d):
        wrapped_1d.reset(seed=1)
        for _ in range(10):
            _, _, term, trunc, info = wrapped_1d.step(1)
            assert "distance" in info
            if term or trunc:
                break

    def test_terminates_mid_interval(self):
        env = DecisionIntervalWrapper(make_env(1, time_max=6_000))
        try:
            env.reset(seed=1)
            env.unwrapped._balloon.n_gas = 0.0     # instantly "deflated"
            _, reward, terminated, _, info = env.step(1)
            assert terminated
            assert reward == 0.0, "the terminal sub-step forfeits the interval"
            assert env.unwrapped._time == 1, "must stop at the terminal sub-step"
            assert info["termination_reason"] == "Deflated (helium fully lost)"
        finally:
            env.close()

    def test_truncation_propagates(self):
        env = DecisionIntervalWrapper(make_env(1, time_max=DECISION_INTERVAL))
        try:
            env.reset(seed=1)
            _, _, terminated, truncated, _ = env.step(1)
            assert truncated and not terminated
        finally:
            env.close()


class TestResourcePenaltyIsNotDiluted:
    """The failure mode this design exists to prevent.

    With the action firing on sub-step 0 only, a penalty charged on that step
    alone contributes ``(59 + 0.97) / 60 = 0.9995`` of the interval's return —
    a 0.05% cost for actuating, which shapes nothing. Held across the interval
    it is the intended ~3%.
    """

    def _interval_return(self, action, seed=5, dim=2):
        env = DecisionIntervalWrapper(make_env(dim, time_max=6_000))
        try:
            env.reset(seed=seed)
            return env.step(action)[1]
        finally:
            env.close()

    def test_actuating_costs_the_full_factor_not_a_sixtieth(self):
        ratio = self._interval_return(2) / self._interval_return(1)
        assert ratio == pytest.approx(RESOURCE_PENALTY_BASE, abs=0.01)
        assert ratio < 0.99, "penalty diluted across the decision interval"

    def test_venting_costs_the_full_factor_too(self):
        ratio = self._interval_return(0) / self._interval_return(1)
        assert ratio == pytest.approx(RESOURCE_PENALTY_BASE, abs=0.01)

    def test_every_sub_step_of_the_interval_carries_the_factor(self):
        """Same check, from inside: no sub-step escapes the penalty."""
        env = make_env(1, time_max=6_000)
        try:
            env.reset(seed=5)
            factors = []
            for i in range(DECISION_INTERVAL):
                _, _, _, _, info = env.step(2 if i == 0 else 1)
                factors.append(info["reward_components"]["resource_factor"])
            expected = RESOURCE_PENALTY_BASE - 0.3 * (BALLAST_DROP / BALLAST_INITIAL)
            assert factors == pytest.approx([expected] * DECISION_INTERVAL)
        finally:
            env.close()

    def test_a_stay_decision_is_never_penalised(self):
        env = DecisionIntervalWrapper(make_env(1, time_max=6_000))
        try:
            env.reset(seed=5)
            for _ in range(5):
                _, _, _, _, info = env.step(1)
                assert info["reward_components"]["resource_factor"] == 1.0
        finally:
            env.close()

    def test_penalty_does_not_leak_into_the_next_decision(self):
        env = DecisionIntervalWrapper(make_env(1, time_max=6_000))
        try:
            env.reset(seed=5)
            env.step(2)                       # actuating decision
            _, _, _, _, info = env.step(1)    # the next one must be free
            assert info["reward_components"]["resource_factor"] == 1.0
        finally:
            env.close()

    def test_shorter_interval_shortens_the_hold(self):
        env = DecisionIntervalWrapper(make_env(1, time_max=6_000), decision_interval=5)
        try:
            base = env.unwrapped
            base.reset(seed=5)
            factors = []
            for i in range(8):
                _, _, _, _, info = base.step(2 if i == 0 else 1)
                factors.append(info["reward_components"]["resource_factor"])
            assert all(f < 1.0 for f in factors[:5])
            assert all(f == 1.0 for f in factors[5:])
        finally:
            env.close()


def test_wrapped_episode_runs_to_completion():
    """A full 12 h episode is 720 decisions; check the arithmetic end to end."""
    env = DecisionIntervalWrapper(make_env(3, time_max=3_600))
    try:
        env.reset(seed=1)
        decisions = 0
        rng = np.random.default_rng(0)
        while True:
            _, _, terminated, truncated, _ = env.step(int(rng.integers(3)))
            decisions += 1
            if terminated or truncated:
                break
        assert decisions == 3_600 // DECISION_INTERVAL
        assert truncated and not terminated
    finally:
        env.close()
