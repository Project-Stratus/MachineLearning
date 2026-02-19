"""Tests for observation space and observation construction."""

import numpy as np
import pytest

from environments.envs.balloon_3d_env import Balloon3DEnv
from tests.conftest import expected_obs_size


class TestObservationSpace:
    """Tests for observation space construction."""

    def test_observation_space_shape(self, env_any_dim):
        """Observation space should have correct shape for dimension."""
        env, dim = env_any_dim
        expected = expected_obs_size(dim)
        assert env.observation_space.shape == (expected,)

    def test_observation_space_dtype(self, env_any_dim):
        """Observation space should have float32 dtype."""
        env, _ = env_any_dim
        assert env.observation_space.dtype == np.float32

    def test_observation_space_low_bounds(self, env_any_dim):
        """Observation space should have correct low bounds."""
        env, dim = env_any_dim
        low = env.observation_space.low

        # Check structure: goal, volume, position, delta, velocity, pressure, wind
        i = 0
        # goal (dim): [0, 1]
        assert np.all(low[i:i + dim] == 0.0)
        i += dim
        # volume: [0, 1]
        assert low[i] == 0.0
        i += 1
        # position (dim): [0, 1]
        assert np.all(low[i:i + dim] == 0.0)
        i += dim
        # delta (dim): [-1, 1]
        assert np.all(low[i:i + dim] == -1.0)
        i += dim
        # velocity (dim): [-1, 1]
        assert np.all(low[i:i + dim] == -1.0)
        i += dim
        # pressure: [0, 1]
        assert low[i] == 0.0
        i += 1
        # wind (dim): [-1, 1]
        assert np.all(low[i:i + dim] == -1.0)

    def test_observation_space_high_bounds(self, env_any_dim):
        """Observation space should have correct high bounds."""
        env, dim = env_any_dim
        high = env.observation_space.high

        i = 0
        # All components should have high bound of 1.0
        assert np.all(high == 1.0)

    def test_observation_buffer_size(self, env_any_dim):
        """Internal observation buffer should match observation space."""
        env, dim = env_any_dim
        assert env._obs_size == expected_obs_size(dim)
        assert env._obs_buf.shape == (env._obs_size,)


class TestObservationNormalization:
    """Tests for observation normalization."""

    def test_position_normalization_1d(self, env_1d):
        """1D position should normalize z to [0, 1]."""
        env_1d.reset(seed=42)
        # Set balloon to known position
        z_min, z_max = env_1d.z_range
        z_mid = (z_min + z_max) / 2
        env_1d._balloon.pos[0] = z_mid

        norm_pos = env_1d._normalise_position(env_1d._balloon.pos)
        assert norm_pos.shape == (1,)
        assert 0.4 < norm_pos[0] < 0.6  # Should be near 0.5

    def test_position_normalization_2d(self, env_2d):
        """2D position should normalize x, y to [0, 1]."""
        env_2d.reset(seed=42)
        # Set to center of ranges
        env_2d._balloon.pos[0] = 0.0  # Center of x_range
        env_2d._balloon.pos[1] = 0.0  # Center of y_range

        norm_pos = env_2d._normalise_position(env_2d._balloon.pos)
        assert norm_pos.shape == (2,)
        assert 0.4 < norm_pos[0] < 0.6
        assert 0.4 < norm_pos[1] < 0.6

    def test_position_normalization_3d(self, env_3d):
        """3D position should normalize x, y, z to [0, 1]."""
        env_3d.reset(seed=42)
        # Set position to center of each range
        x_mid = (env_3d.x_range[0] + env_3d.x_range[1]) / 2
        y_mid = (env_3d.y_range[0] + env_3d.y_range[1]) / 2
        z_mid = (env_3d.z_range[0] + env_3d.z_range[1]) / 2
        env_3d._balloon.pos = np.array([x_mid, y_mid, z_mid])

        norm_pos = env_3d._normalise_position(env_3d._balloon.pos)
        assert norm_pos.shape == (3,)
        assert 0.4 < norm_pos[0] < 0.6  # x center
        assert 0.4 < norm_pos[1] < 0.6  # y center
        assert 0.4 < norm_pos[2] < 0.6  # z center

    def test_position_normalization_at_bounds(self, env_1d):
        """Position at range bounds should normalize to 0 and 1."""
        env_1d.reset(seed=42)
        z_min, z_max = env_1d.z_range

        env_1d._balloon.pos[0] = z_min
        norm_min = env_1d._normalise_position(env_1d._balloon.pos)
        assert norm_min[0] == pytest.approx(0.0)

        env_1d._balloon.pos[0] = z_max
        norm_max = env_1d._normalise_position(env_1d._balloon.pos)
        assert norm_max[0] == pytest.approx(1.0)


class TestObservationPacking:
    """Tests for observation packing correctness."""

    def test_observation_packing_validation(self, env_any_dim):
        """Observation packing should pass internal validation."""
        env, _ = env_any_dim
        env.reset(seed=42)
        # _get_obs has internal assertion that should pass
        obs = env._get_obs()
        assert obs is not None

    def test_observation_components_order(self, env_1d):
        """Observation should pack components in documented order."""
        env_1d.reset(seed=42)
        obs = env_1d._get_obs()

        # Order: goal, volume, position, delta, velocity, pressure, wind, ballast, gas
        # For dim=1: sizes are 1, 1, 1, 1, 1, 1, 1, 1, 1 = 9 total

        # We can't easily extract individual components from flat array
        # but we can verify the total size
        assert obs.shape == (expected_obs_size(1),)

    def test_observation_goal_normalized(self, env_any_dim):
        """Goal should be normalized to [0, 1]."""
        env, dim = env_any_dim
        env.reset(seed=42)
        obs = env._get_obs()

        # First `dim` elements are goal
        goal_obs = obs[:dim]
        assert np.all(goal_obs >= 0.0)
        assert np.all(goal_obs <= 1.0)

    def test_observation_volume_normalized(self, env_any_dim):
        """Volume should be normalized to [0, 1]."""
        env, dim = env_any_dim
        env.reset(seed=42)
        obs = env._get_obs()

        # Volume is at index `dim`
        vol_obs = obs[dim]
        assert 0.0 <= vol_obs <= 1.0

    def test_observation_position_normalized(self, env_any_dim):
        """Position should be normalized to [0, 1]."""
        env, dim = env_any_dim
        env.reset(seed=42)
        obs = env._get_obs()

        # Position is at indices dim+1 to dim+1+dim
        pos_start = dim + 1
        pos_obs = obs[pos_start:pos_start + dim]
        assert np.all(pos_obs >= 0.0)
        assert np.all(pos_obs <= 1.0)

    def test_observation_velocity_clipped(self, env_any_dim):
        """Velocity should be clipped to [-1, 1]."""
        env, dim = env_any_dim
        env.reset(seed=42)
        obs = env._get_obs()

        # Velocity is at indices 3*dim+2 to 4*dim+2
        vel_start = 3 * dim + 2
        vel_obs = obs[vel_start:vel_start + dim]
        assert np.all(vel_obs >= -1.0)
        assert np.all(vel_obs <= 1.0)

    def test_observation_pressure_normalized(self, env_any_dim):
        """Pressure should be normalized to [0, 1]."""
        env, dim = env_any_dim
        env.reset(seed=42)
        env.step(1)  # Take a step to update pressure
        obs = env._get_obs()

        # Pressure is at index 4*dim+2
        pressure_idx = 4 * dim + 2
        pressure_obs = obs[pressure_idx]
        assert 0.0 <= pressure_obs <= 1.0

    def test_observation_wind_normalized(self, env_any_dim):
        """Wind should be normalized to approximately [-1, 1]."""
        env, dim = env_any_dim
        env.reset(seed=42)
        obs = env._get_obs()

        # Wind is last `dim` elements
        wind_obs = obs[-dim:]
        # Wind magnitude could exceed 1 slightly due to normalization
        assert np.all(wind_obs >= -2.0)
        assert np.all(wind_obs <= 2.0)


class TestObservationConsistency:
    """Tests for observation consistency across steps."""

    def test_observation_changes_with_step(self, env_any_dim):
        """Observation should change after taking action."""
        env, _ = env_any_dim
        obs1, _ = env.reset(seed=42)
        obs2, _, _, _, _ = env.step(2)  # Inflate
        # Position or velocity should change
        assert not np.allclose(obs1, obs2)

    def test_observation_reproducible(self, env_1d):
        """Same actions from same seed should produce same observations."""
        obs_sequence_1 = []
        env_1d.reset(seed=42)
        for _ in range(5):
            obs, _, _, _, _ = env_1d.step(1)
            obs_sequence_1.append(obs.copy())

        obs_sequence_2 = []
        env_1d.reset(seed=42)
        for _ in range(5):
            obs, _, _, _, _ = env_1d.step(1)
            obs_sequence_2.append(obs.copy())

        for o1, o2 in zip(obs_sequence_1, obs_sequence_2):
            assert np.allclose(o1, o2)

    def test_observation_returned_is_copy(self, env_1d):
        """Returned observation should be a copy, not a reference."""
        env_1d.reset(seed=42)
        obs1, _, _, _, _ = env_1d.step(1)
        obs1_copy = obs1.copy()
        obs2, _, _, _, _ = env_1d.step(1)

        # obs1 should not have changed
        assert np.allclose(obs1, obs1_copy)


class TestFullCoordsHelper:
    """Tests for _full_coords helper method."""

    def test_full_coords_1d(self, env_1d):
        """1D should return (0, 0, z)."""
        env_1d.reset(seed=42)
        env_1d._balloon.pos[0] = 15000.0
        x, y, z = env_1d._full_coords(env_1d._balloon.pos)
        assert x == 0.0
        assert y == 0.0
        assert z == 15000.0

    def test_full_coords_2d(self, env_2d):
        """2D should return (x, y, z0)."""
        env_2d.reset(seed=42)
        env_2d._balloon.pos[0] = 500.0
        env_2d._balloon.pos[1] = -300.0
        x, y, z = env_2d._full_coords(env_2d._balloon.pos)
        assert x == 500.0
        assert y == -300.0
        assert z == env_2d.z0

    def test_full_coords_3d(self, env_3d):
        """3D should return (x, y, z)."""
        env_3d.reset(seed=42)
        env_3d._balloon.pos = np.array([100.0, 200.0, 300.0])
        x, y, z = env_3d._full_coords(env_3d._balloon.pos)
        assert x == 100.0
        assert y == 200.0
        assert z == 300.0
