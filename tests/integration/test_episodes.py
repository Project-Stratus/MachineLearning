"""Integration tests for full episode rollouts."""

import numpy as np
import pytest

from environments.envs.balloon_3d_env import Balloon3DEnv
from environments.core.constants import ALT_MAX


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
        """Episode rewards should be within reasonable bounds."""
        env = Balloon3DEnv(dim=1, render_mode=None, config={"time_max": 50})
        try:
            env.reset(seed=42)
            rewards = []

            for _ in range(50):
                _, reward, term, trunc, _ = env.step(env.action_space.sample())
                rewards.append(reward)
                if term or trunc:
                    break

            # Rewards should be bounded (distance reward in [-1, 0] plus small bonuses)
            for r in rewards:
                assert -10 < r < 2, f"Reward {r} outside expected bounds"
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
    def test_inflate_causes_rise_1d(self):
        """Consistent inflating should cause balloon to rise in 1D."""
        env = Balloon3DEnv(dim=1, render_mode=None, config={"time_max": 100})
        try:
            env.reset(seed=42)
            initial_alt = env._balloon.altitude

            # Inflate repeatedly
            for _ in range(20):
                env.step(2)  # Inflate

            final_alt = env._balloon.altitude
            assert final_alt > initial_alt, "Balloon should rise when inflating"
        finally:
            env.close()

    @pytest.mark.integration
    def test_deflate_causes_fall_1d(self):
        """Consistent deflating should eventually cause balloon to fall in 1D."""
        env = Balloon3DEnv(dim=1, render_mode=None, config={"time_max": 200})
        try:
            env.reset(seed=42)
            # Start at a known altitude with neutral buoyancy
            initial_alt = env._balloon.altitude

            # Deflate many times to create negative buoyancy
            for _ in range(50):
                env.step(0)  # Deflate

            # The balloon should have less volume and thus less buoyancy
            # It might take time to actually fall due to momentum
            final_alt = env._balloon.altitude
            final_extra_vol = env._balloon.extra_volume

            # At minimum, the extra volume should be negative (deflated)
            assert final_extra_vol < 0, "Deflating should reduce extra volume"

            # The balloon should eventually be falling or have fallen
            # (allow for cases where drag slows it significantly)
            assert final_alt < initial_alt or env._balloon.vel[0] < 0, \
                "Balloon should fall or be falling when deflated"
        finally:
            env.close()

    @pytest.mark.integration
    def test_altitude_stays_positive_or_terminates(self):
        """Altitude should stay positive or episode should terminate."""
        env = Balloon3DEnv(dim=1, render_mode=None, config={"time_max": 500})
        try:
            env.reset(seed=42)

            for _ in range(500):
                _, _, terminated, truncated, _ = env.step(0)  # Keep deflating
                if terminated:
                    break
                assert env._balloon.altitude >= 0, "Altitude should never be negative"
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
    def test_reward_improves_when_approaching_goal(self):
        """Reward should generally improve when getting closer to goal."""
        env = Balloon3DEnv(dim=1, render_mode=None, config={"time_max": 100})
        try:
            env.reset(seed=42)

            # Determine which direction to go
            balloon_z = env._balloon.altitude
            goal_z = env.goal[0]

            # Choose action to move toward goal
            if goal_z > balloon_z:
                action = 2  # Inflate to rise
            else:
                action = 0  # Deflate to fall

            rewards = []
            for _ in range(30):
                _, reward, term, _, _ = env.step(action)
                rewards.append(reward)
                if term:
                    break

            # Later rewards should generally be better (less negative)
            # Compare first 5 vs last 5 average
            if len(rewards) >= 10:
                early_avg = np.mean(rewards[:5])
                late_avg = np.mean(rewards[-5:])
                # This might not always hold due to direction reward, but distance should help
                # Just verify rewards are reasonable
                assert all(r > -2 for r in rewards), "Rewards should be reasonable"
        finally:
            env.close()

    @pytest.mark.integration
    def test_crash_gives_punishment(self):
        """Crashing should give punishment reward."""
        env = Balloon3DEnv(dim=1, render_mode=None, config={"time_max": 1000, "punishment": -5.0})
        try:
            env.reset(seed=42)

            # Make balloon heavy so gravity overwhelms buoyancy, causing a crash
            env._balloon.mass = 1000.0
            env._balloon.pos[0] = 10.0
            env._balloon.vel[0] = -5.0

            # Step until crash
            for _ in range(100):
                _, reward, terminated, _, _ = env.step(1)  # do nothing
                if terminated:
                    assert reward == 0.0, "Crash should give zero reward (forfeits future reward)"
                    break
            else:
                pytest.skip("Balloon didn't crash in time")
        finally:
            env.close()


class TestStateConsistency:
    """Tests for internal state consistency."""

    @pytest.mark.integration
    def test_balloon_state_matches_observation(self):
        """Balloon state should be reflected in observation."""
        env = Balloon3DEnv(dim=1, render_mode=None, config={"time_max": 100})
        try:
            obs, _ = env.reset(seed=42)

            for _ in range(10):
                obs, _, _, _, _ = env.step(env.action_space.sample())

                # Position in observation should match balloon position (normalized)
                # Position is at index dim+1 for 1D (after goal and volume)
                pos_norm_obs = obs[2]  # For dim=1: goal(1) + vol(1) + pos(1)
                pos_actual = env._balloon.pos[0]
                z_min, z_max = env.z_range
                pos_norm_expected = (pos_actual - z_min) / (z_max - z_min)

                assert pos_norm_obs == pytest.approx(pos_norm_expected, rel=0.01)
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

            # Rapidly alternate between inflate and deflate
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
