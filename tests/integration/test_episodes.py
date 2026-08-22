"""Integration tests for full episode rollouts."""

import numpy as np
import pytest

from agents.baselines import AMBIENT_IDX
from environments.core.constants import (
    ALT_DEFAULT, ALT_SAFE_MAX, ALT_SAFE_MIN, DIST_NORM, SP_VOL_FIXED,
    VEL_MAX, VEL_Z_OBS_NORM, VOL_MAX,
)
from environments.envs.balloon_3d_env import Balloon3DEnv

#: Read the observation through the *names* in the frozen layout (contract §1).
#: These tests previously indexed it with magic numbers from a since-replaced
#: 19-wide layout and kept passing on the wrong fields until they did not.
IDX_ALT_NORM = AMBIENT_IDX["alt_norm"]
IDX_VEL_Z_NORM = AMBIENT_IDX["vel_z_norm"]
IDX_DIST_NORM = AMBIENT_IDX["dist_norm"]
IDX_VOLUME_NORM = AMBIENT_IDX["volume_norm"]
IDX_AT_ALT_MIN = AMBIENT_IDX["at_alt_min"]
IDX_AT_ALT_MAX = AMBIENT_IDX["at_alt_max"]
IDX_LAST_ACTION_DOWN = AMBIENT_IDX["last_action_down"]


class TestFullEpisodeRollout:
    """Tests for complete episode execution."""

    @pytest.mark.integration
    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_episode_completes_all_dimensions(self, dim):
        """Full episode should complete without errors for all dimensions."""
        env = Balloon3DEnv(dim=dim, render_mode=None, config={"time_max": 100})
        try:
            obs, _ = env.reset(seed=42)
            done = False
            steps = 0

            while not done and steps < 200:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1

                # Verify each step produces valid output
                assert np.all(np.isfinite(obs))
                assert np.isfinite(reward)

            assert done, "Episode should complete within step limit"
        finally:
            env.close()

    @pytest.mark.integration
    def test_episode_reward_bounds(self):
        """Rewards live in [0, 1] — the Perciatelli-style reward has no penalties."""
        env = Balloon3DEnv(dim=1, render_mode=None, config={"time_max": 50})
        try:
            env.reset(seed=42)
            rewards = []

            for _ in range(50):
                _, reward, term, trunc, _ = env.step(env.action_space.sample())
                rewards.append(reward)
                if term or trunc:
                    break

            for r in rewards:
                assert 0.0 <= r <= 1.0, f"Reward {r} outside the [0, 1] contract"
        finally:
            env.close()

    @pytest.mark.integration
    def test_episode_determinism(self):
        """Same seed and actions should produce identical episode."""
        actions = [0, 1, 2, 1, 0, 2, 1, 1, 0, 2]

        def run_episode(seed):
            env = Balloon3DEnv(dim=1, render_mode=None, config={"time_max": 100})
            try:
                observations = []
                rewards = []

                obs, _ = env.reset(seed=seed)
                observations.append(obs.copy())

                for action in actions:
                    obs, reward, _, _, _ = env.step(action)
                    observations.append(obs.copy())
                    rewards.append(reward)

                return observations, rewards
            finally:
                env.close()

        obs1, rew1 = run_episode(seed=123)
        obs2, rew2 = run_episode(seed=123)

        for o1, o2 in zip(obs1, obs2):
            assert np.allclose(o1, o2), "Observations should be identical"
        for r1, r2 in zip(rew1, rew2):
            assert r1 == pytest.approx(r2), "Rewards should be identical"

    @pytest.mark.integration
    def test_episode_nondeterminism_different_seeds(self):
        """Different seeds should produce different episodes."""
        def run_episode(seed):
            env = Balloon3DEnv(dim=1, render_mode=None, config={"time_max": 100})
            try:
                obs, _ = env.reset(seed=seed)
                for _ in range(5):
                    obs, _, _, _, _ = env.step(1)
                return obs
            finally:
                env.close()

        obs1 = run_episode(seed=111)
        obs2 = run_episode(seed=222)

        assert not np.allclose(obs1, obs2), "Different seeds should produce different results"


class TestPhysicsConsistency:
    """Tests for physics consistency during episodes."""

    @pytest.mark.integration
    def test_drop_ballast_causes_rise_1d(self):
        """Dropping ballast repeatedly should cause balloon to rise in 1D."""
        env = Balloon3DEnv(dim=1, render_mode=None, config={"time_max": 200})
        try:
            env.reset(seed=42)
            initial_alt = env._balloon.altitude

            # Drop ballast repeatedly — need enough drops to overcome gas mass
            # deficit and the downward momentum from sinking while still heavy
            for _ in range(100):
                env.step(2)  # Drop ballast

            final_alt = env._balloon.altitude
            assert final_alt > initial_alt, "Balloon should rise when dropping ballast"
        finally:
            env.close()

    @pytest.mark.integration
    def test_vent_causes_fall_1d(self):
        """Venting gas repeatedly should eventually cause balloon to fall in 1D."""
        env = Balloon3DEnv(dim=1, render_mode=None, config={"time_max": 200})
        try:
            env.reset(seed=42)
            initial_alt = env._balloon.altitude

            # Vent gas many times to reduce buoyancy
            for _ in range(50):
                env.step(0)  # Vent gas

            final_alt = env._balloon.altitude
            final_extra_vol = env._balloon.extra_volume

            # At minimum, the extra volume should be negative (gas vented)
            assert final_extra_vol < 0, "Venting should reduce extra volume"

            # The balloon should eventually be falling or have fallen
            assert final_alt < initial_alt or env._balloon.vel[0] < 0, \
                "Balloon should fall or be falling when gas is vented"
        finally:
            env.close()

    @pytest.mark.integration
    def test_altitude_stays_within_safety_band(self):
        """The safety layer holds the balloon inside the operational band.

        Formerly "altitude should stay positive", which the ground clamp made
        trivially true.  The live invariant is the ``[ALT_SAFE_MIN,
        ALT_SAFE_MAX]`` band, and venting is the fastest way to lean on its
        floor (roadmap §3.4).
        """
        env = Balloon3DEnv(dim=1, render_mode=None, config={"time_max": 500})
        try:
            env.reset(seed=42)

            for _ in range(500):
                obs, _, terminated, _, _ = env.step(0)  # keep venting
                assert ALT_SAFE_MIN - 1e-6 <= env._balloon.altitude <= ALT_SAFE_MAX + 1e-6, \
                    "altitude escaped the operational band"
                assert obs[IDX_ALT_NORM] == pytest.approx(
                    (env._balloon.altitude - ALT_SAFE_MIN) / (ALT_SAFE_MAX - ALT_SAFE_MIN),
                    abs=1e-5,
                )
                if terminated:
                    break
        finally:
            env.close()

    @pytest.mark.integration
    def test_wind_affects_2d_motion(self):
        """Wind should affect balloon motion in 2D."""
        env = Balloon3DEnv(dim=2, render_mode=None, config={"time_max": 100})
        try:
            env.reset(seed=42)
            initial_pos = env._balloon.pos[:2].copy()

            # Take several steps
            for _ in range(50):
                env.step(1)  # Do nothing (let wind push)

            final_pos = env._balloon.pos[:2]

            # Position should have changed due to wind
            displacement = np.linalg.norm(final_pos - initial_pos)
            assert displacement > 0, "Wind should cause horizontal displacement"
        finally:
            env.close()

    @pytest.mark.integration
    def test_2d_altitude_constant(self):
        """2D mode should maintain constant altitude."""
        env = Balloon3DEnv(dim=2, render_mode=None, config={"time_max": 100})
        try:
            env.reset(seed=42)
            z0 = env.z0
            altitudes = []

            for _ in range(50):
                env.step(env.action_space.sample())
                altitudes.append(env._balloon.pos[2])

            for alt in altitudes:
                assert alt == pytest.approx(z0), "2D altitude should stay constant"
        finally:
            env.close()


class TestRewardConsistency:
    """Tests for reward consistency during episodes."""

    @pytest.mark.integration
    def test_reward_is_monotone_in_distance(self):
        """Being nearer the station is never worth less.

        Holding one action for the whole rollout keeps the resource factor
        constant, so reward becomes a pure function of distance and the
        invariant can be checked without a tolerance on the trajectory: sort
        the (distance, reward) pairs and reward must not increase with
        distance.  The previous version computed early/late averages, then
        asserted only ``r > -2`` — vacuous under a reward bounded below by 0.
        """
        env = Balloon3DEnv(dim=1, render_mode=None, config={"time_max": 200})
        try:
            env.reset(seed=42)
            action = 2 if env.goal[0] > env._balloon.altitude else 0

            pairs = []
            for _ in range(60):
                _, reward, term, trunc, info = env.step(action)
                if term:
                    # Termination forfeits reward regardless of distance.
                    break
                pairs.append((info["distance"], reward))
                if trunc:
                    break

            assert len(pairs) >= 10, "need a usable rollout to check the invariant"
            pairs.sort(key=lambda p: p[0])
            rewards = [r for _, r in pairs]
            assert all(0.0 <= r <= 1.0 for r in rewards)
            for near, far in zip(rewards, rewards[1:]):
                assert near >= far - 1e-9, "reward must not grow with distance"
        finally:
            env.close()

    @pytest.mark.integration
    def test_hitting_alt_min_clamps_and_flags_without_terminating(self):
        """The floor of the band clamps the balloon; it no longer kills it.

        Replaces ``test_crash_gives_punishment``: crash termination was removed
        with the altitude safety layer (contract §5, roadmap §3.4), so the
        behaviour under test is the clamp, the ``at_alt_min`` flag, and the
        *absence* of both termination and a punishment reward.
        """
        env = Balloon3DEnv(dim=1, render_mode=None, config={"time_max": 1000})
        try:
            env.reset(seed=42)
            # Drop the balloon onto the floor with enough speed that the clamp
            # has to act, rather than waiting on buoyancy to take it there.
            env._balloon.pos[-1] = ALT_SAFE_MIN + 20.0
            env._balloon.vel[-1] = -0.5 * VEL_MAX

            clamped = False
            for _ in range(100):
                obs, reward, terminated, truncated, _ = env.step(1)  # do nothing
                assert not terminated, "altitude must never terminate the episode"
                assert env._balloon.altitude >= ALT_SAFE_MIN - 1e-9
                assert reward > 0.0, "there is no punishment for touching the limit"
                if obs[IDX_AT_ALT_MIN] == 1.0:
                    clamped = True
                    assert env._balloon.altitude == pytest.approx(ALT_SAFE_MIN)
                    assert env._balloon.vel[-1] == pytest.approx(0.0)
                    assert obs[IDX_AT_ALT_MAX] == 0.0
                    assert obs[IDX_ALT_NORM] == pytest.approx(0.0)
                    break
                if truncated:
                    break

            assert clamped, "a hard descent should reach the lower altitude limit"
        finally:
            env.close()


class TestStateConsistency:
    """Tests for internal state consistency."""

    @pytest.mark.integration
    def test_balloon_state_matches_observation(self):
        """Balloon state should be reflected in the observation.

        Indexed by name from ``agents.baselines.AMBIENT_IDX`` rather than by
        position: the original read ``obs[2]`` for a normalised altitude, an
        index that under the frozen 143-wide layout is the wind column's
        uncertainty channel.  A layout change should break the *import*, not
        silently compare against an unrelated field.
        """
        env = Balloon3DEnv(dim=1, render_mode=None, config={"time_max": 100})
        try:
            obs, _ = env.reset(seed=42)
            span = ALT_SAFE_MAX - ALT_SAFE_MIN

            for _ in range(10):
                action = env.action_space.sample()
                obs, _, _, _, info = env.step(action)
                b = env._balloon

                assert obs[IDX_ALT_NORM] == pytest.approx(
                    np.clip((b.altitude - ALT_SAFE_MIN) / span, 0.0, 1.0), abs=1e-5)
                assert obs[IDX_VEL_Z_NORM] == pytest.approx(
                    np.clip(b.vel[-1] / VEL_Z_OBS_NORM, -1.0, 1.0), abs=1e-5)
                assert obs[IDX_VOLUME_NORM] == pytest.approx(
                    min(b.volume / VOL_MAX, 1.0), rel=1e-5)
                # Distance normaliser is per-dim (1-D measures |dz| against a
                # much tighter altitude scale), so read it off the env rather
                # than assuming the horizontal constant.
                assert obs[IDX_DIST_NORM] == pytest.approx(
                    min(info["distance"] * env._inv_dist_norm, 1.0), abs=1e-5)

                # Previous action is one-hot encoded in action-index order.
                one_hot = obs[IDX_LAST_ACTION_DOWN:IDX_LAST_ACTION_DOWN + 3]
                assert one_hot.sum() == pytest.approx(1.0)
                assert one_hot[action] == 1.0
        finally:
            env.close()

    @pytest.mark.integration
    def test_goal_constant_during_episode(self):
        """Goal should not change during an episode."""
        env = Balloon3DEnv(dim=1, render_mode=None, config={"time_max": 100})
        try:
            env.reset(seed=42)
            initial_goal = env.goal.copy()

            for _ in range(20):
                env.step(env.action_space.sample())
                assert np.allclose(env.goal, initial_goal), "Goal should not change"
        finally:
            env.close()

    @pytest.mark.integration
    def test_time_increments_correctly(self):
        """Internal time should increment by 1 each step."""
        env = Balloon3DEnv(dim=1, render_mode=None, config={"time_max": 100})
        try:
            env.reset(seed=42)
            assert env._time == 0

            for expected_time in range(1, 11):
                env.step(1)
                assert env._time == expected_time
        finally:
            env.close()


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.integration
    def test_multiple_resets(self):
        """Multiple resets should work correctly."""
        env = Balloon3DEnv(dim=1, render_mode=None, config={"time_max": 50})
        try:
            for i in range(5):
                obs, _ = env.reset(seed=i)
                assert np.all(np.isfinite(obs))

                for _ in range(10):
                    obs, _, term, trunc, _ = env.step(env.action_space.sample())
                    assert np.all(np.isfinite(obs))
                    if term or trunc:
                        break
        finally:
            env.close()

    @pytest.mark.integration
    def test_step_after_termination(self):
        """Stepping after termination should still work (for compatibility)."""
        env = Balloon3DEnv(dim=1, render_mode=None, config={"time_max": 5})
        try:
            env.reset(seed=42)

            # Run until truncation
            for _ in range(10):
                _, _, _, truncated, _ = env.step(1)
                if truncated:
                    break

            # Step again after truncation - should not crash
            # (Gym typically expects reset, but env should handle gracefully)
            obs, reward, _, _, _ = env.step(1)
            assert np.all(np.isfinite(obs))
        finally:
            env.close()

    @pytest.mark.integration
    def test_extreme_actions_sequence(self):
        """Rapid action changes should not break physics."""
        env = Balloon3DEnv(dim=1, render_mode=None, config={"time_max": 100})
        try:
            env.reset(seed=42)

            # Rapidly alternate between drop ballast and vent
            for i in range(50):
                action = 2 if i % 2 == 0 else 0
                obs, reward, term, _, _ = env.step(action)
                assert np.all(np.isfinite(obs))
                assert np.isfinite(reward)
                if term:
                    break
        finally:
            env.close()


class TestGymCompatibility:
    """Tests for Gymnasium API compatibility."""

    @pytest.mark.integration
    def test_gym_make_works(self):
        """Should be able to create environment via gym.make."""
        import gymnasium as gym
        env = gym.make("environments/Balloon3D-v0", dim=1, render_mode=None)
        try:
            obs, info = env.reset()
            assert obs is not None
            obs, reward, term, trunc, info = env.step(1)
            assert obs is not None
        finally:
            env.close()

    @pytest.mark.integration
    def test_observation_space_sample(self):
        """Observation space should support sampling."""
        env = Balloon3DEnv(dim=1, render_mode=None)
        try:
            sample = env.observation_space.sample()
            assert sample.shape == env.observation_space.shape
            assert env.observation_space.contains(sample)
        finally:
            env.close()

    @pytest.mark.integration
    def test_action_space_sample(self):
        """Action space should support sampling."""
        env = Balloon3DEnv(dim=1, render_mode=None)
        try:
            for _ in range(10):
                action = env.action_space.sample()
                assert 0 <= action < 3
                assert env.action_space.contains(action)
        finally:
            env.close()


class TestSPEpisodeRollout:
    """Integration tests for SP (superpressure + air ballast) episode rollouts."""

    @pytest.mark.integration
    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_sp_episode_completes_all_dimensions(self, dim):
        """SP full episode should complete without errors for all dimensions."""
        env = Balloon3DEnv(dim=dim, render_mode=None,
                           config={"time_max": 100, "balloon_type": "superpressure"})
        try:
            obs, _ = env.reset(seed=42)
            done = False
            steps = 0
            while not done and steps < 200:
                obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
                done = terminated or truncated
                steps += 1
                assert np.all(np.isfinite(obs))
                assert np.isfinite(reward)
            assert done, "SP episode should complete within step limit"
        finally:
            env.close()

    @pytest.mark.integration
    def test_sp_gym_make_works(self):
        """Should be able to create SP environment via gym.make with BalloonSP3D-v0."""
        import gymnasium as gym
        env = gym.make("environments/BalloonSP3D-v0", dim=1, render_mode=None)
        try:
            obs, info = env.reset()
            assert obs is not None
            obs, reward, term, trunc, info = env.step(1)
            assert obs is not None
            assert np.isfinite(reward)
        finally:
            env.close()

    @pytest.mark.integration
    def test_sp_episode_determinism(self):
        """Same seed and actions should produce identical SP episodes."""
        actions = [0, 1, 2, 1, 0, 2, 1, 1, 0, 2]

        def run_episode(seed):
            env = Balloon3DEnv(dim=1, render_mode=None,
                               config={"time_max": 100, "balloon_type": "superpressure"})
            try:
                obs, _ = env.reset(seed=seed)
                observations = [obs.copy()]
                for action in actions:
                    obs, _, _, _, _ = env.step(action)
                    observations.append(obs.copy())
                return observations
            finally:
                env.close()

        for o1, o2 in zip(run_episode(123), run_episode(123)):
            assert np.allclose(o1, o2), "SP observations should be identical for same seed"


class TestSPPhysicsConsistency:
    """Physics consistency tests for SP balloon episodes."""

    @pytest.mark.integration
    def test_sp_pump_out_causes_rise_1d(self):
        """Pumping air out repeatedly should cause SP balloon to rise in 1D.

        The balloon is placed at ALT_DEFAULT (neutral-buoyancy with midpoint
        bladder) so the passive restoring force is zero and pumping determines
        the direction of motion.
        """
        env = Balloon3DEnv(dim=1, render_mode=None,
                           config={"time_max": 200, "balloon_type": "superpressure"})
        try:
            env.reset(seed=42)
            env._balloon.pos[-1] = ALT_DEFAULT   # start at neutral-buoyancy altitude
            env._balloon.vel[-1] = 0.0
            initial_alt = env._balloon.altitude
            for _ in range(100):
                env.step(2)   # effect +1 → pump_out → ascend
            assert env._balloon.altitude > initial_alt, \
                "SP balloon should rise after pumping air out"
        finally:
            env.close()

    @pytest.mark.integration
    def test_sp_pump_in_causes_fall_1d(self):
        """Pumping air in repeatedly should cause SP balloon to fall in 1D.

        The balloon is placed at ALT_DEFAULT (neutral-buoyancy with midpoint
        bladder) so the passive restoring force is zero and pumping determines
        the direction of motion.
        """
        env = Balloon3DEnv(dim=1, render_mode=None,
                           config={"time_max": 200, "balloon_type": "superpressure"})
        try:
            env.reset(seed=42)
            env._balloon.pos[-1] = ALT_DEFAULT   # start at neutral-buoyancy altitude
            env._balloon.vel[-1] = 0.0
            initial_alt = env._balloon.altitude
            for _ in range(100):
                env.step(0)   # effect -1 → pump_in → descend
            assert env._balloon.altitude < initial_alt, \
                "SP balloon should fall after pumping air in"
        finally:
            env.close()

    @pytest.mark.integration
    def test_sp_altitude_stays_within_safety_band(self):
        """SP obeys the same altitude safety layer as ZP, and never terminates on it."""
        env = Balloon3DEnv(dim=1, render_mode=None,
                           config={"time_max": 500, "balloon_type": "superpressure"})
        try:
            env.reset(seed=42)
            for _ in range(500):
                obs, _, terminated, truncated, _ = env.step(0)  # pump_in repeatedly
                assert not terminated, "SP has no altitude termination"
                assert ALT_SAFE_MIN - 1e-6 <= env._balloon.altitude <= ALT_SAFE_MAX + 1e-6, \
                    "SP altitude escaped the operational band"
                if obs[IDX_AT_ALT_MIN] == 1.0:
                    assert env._balloon.altitude == pytest.approx(ALT_SAFE_MIN)
                if truncated:
                    break
        finally:
            env.close()

    @pytest.mark.integration
    def test_sp_volume_constant_throughout_episode(self):
        """SP balloon volume should remain fixed throughout an entire episode."""
        env = Balloon3DEnv(dim=1, render_mode=None,
                           config={"time_max": 100, "balloon_type": "superpressure"})
        try:
            env.reset(seed=42)
            for _ in range(100):
                _, _, term, trunc, _ = env.step(env.action_space.sample())
                assert env._balloon.volume == pytest.approx(SP_VOL_FIXED), \
                    "SP volume should be fixed throughout episode"
                if term or trunc:
                    break
        finally:
            env.close()

    @pytest.mark.integration
    def test_sp_wind_affects_horizontal_motion(self):
        """Wind should affect SP balloon horizontal motion the same as ZP."""
        env = Balloon3DEnv(dim=2, render_mode=None,
                           config={"time_max": 100, "balloon_type": "superpressure"})
        try:
            env.reset(seed=42)
            initial_pos = env._balloon.pos[:2].copy()
            for _ in range(50):
                env.step(1)   # do nothing — let wind push
            displacement = np.linalg.norm(env._balloon.pos[:2] - initial_pos)
            assert displacement > 0, "Wind should cause horizontal displacement for SP balloon"
        finally:
            env.close()
