"""Tests for reward functions."""

import math
import numpy as np
import pytest

from environments.core.reward import l2_distance, balloon_reward


class TestL2Distance:
    """Tests for l2_distance function across all dimensions."""

    def test_l2_distance_1d_simple(self):
        """1D distance should be absolute difference."""
        balloon_pos = np.array([1000.0])
        goal_pos = np.array([500.0])
        dist = l2_distance(balloon_pos, goal_pos, dim=1)
        assert dist == pytest.approx(500.0)

    def test_l2_distance_1d_uses_last_index(self):
        """1D should use last index of balloon_pos (for internal 3D representation)."""
        balloon_pos = np.array([100.0, 200.0, 1000.0])
        goal_pos = np.array([500.0])
        dist = l2_distance(balloon_pos, goal_pos, dim=1)
        assert dist == pytest.approx(500.0)

    def test_l2_distance_1d_symmetric(self):
        """1D distance should be symmetric."""
        pos_a = np.array([1000.0])
        pos_b = np.array([500.0])
        assert l2_distance(pos_a, pos_b, 1) == l2_distance(pos_b, pos_a, 1)

    def test_l2_distance_2d_simple(self):
        """2D distance should be Euclidean in x-y plane."""
        balloon_pos = np.array([300.0, 400.0, 10000.0])
        goal_pos = np.array([0.0, 0.0])
        dist = l2_distance(balloon_pos, goal_pos, dim=2)
        assert dist == pytest.approx(500.0)

    def test_l2_distance_2d_ignores_z(self):
        """2D distance should ignore z component."""
        balloon_pos = np.array([300.0, 400.0, 99999.0])
        goal_pos = np.array([0.0, 0.0])
        dist = l2_distance(balloon_pos, goal_pos, dim=2)
        assert dist == pytest.approx(500.0)

    def test_l2_distance_3d_uses_xy_only(self):
        """3D distance should use x,y only (altitude is not an objective)."""
        balloon_pos = np.array([100.0, 200.0, 300.0])
        goal_pos = np.array([400.0, 600.0, 900.0])
        # Should ignore z component
        expected = math.sqrt((100.0 - 400.0)**2 + (200.0 - 600.0)**2)
        dist = l2_distance(balloon_pos, goal_pos, dim=3)
        assert dist == pytest.approx(expected)

    def test_l2_distance_3d_ignores_z(self):
        """3D distance should be identical regardless of z values."""
        pos_a = np.array([100.0, 200.0, 0.0])
        pos_b = np.array([100.0, 200.0, 99999.0])
        goal = np.array([400.0, 500.0, 5000.0])
        assert l2_distance(pos_a, goal, 3) == pytest.approx(l2_distance(pos_b, goal, 3))

    def test_l2_distance_3d_symmetric(self):
        """3D distance should be symmetric."""
        pos_a = np.array([100.0, 200.0, 300.0])
        pos_b = np.array([400.0, 500.0, 600.0])
        assert l2_distance(pos_a, pos_b, 3) == pytest.approx(l2_distance(pos_b, pos_a, 3))

    def test_l2_distance_zero_when_same(self):
        """Distance should be zero when positions are identical."""
        for dim in [1, 2, 3]:
            if dim == 1:
                pos = np.array([1000.0])
                goal = pos.copy()
            else:
                pos = np.array([100.0, 200.0, 300.0])
                goal = pos.copy()  # same x,y (z ignored for dim 2/3)
            assert l2_distance(pos, goal, dim) == pytest.approx(0.0)


class TestBalloonReward:
    """Tests for the Perciatelli-style balloon_reward function."""

    @pytest.fixture
    def reward_kwargs(self):
        """Default kwargs for balloon_reward (unused params kept for signature compat)."""
        return dict(
            punishment=-100.0,
            prev_distance=20000.0,
            max_distance=100000.0,
            station_radius=10000.0,
            reward_dropoff=0.4,
            reward_halflife=20000.0,
        )

    def test_returns_tuple(self, reward_kwargs):
        """balloon_reward should return (total, components, new_distance)."""
        result = balloon_reward(
            balloon_pos=np.array([100.0]),
            goal_pos=np.array([0.0]),
            velocity=np.array([0.0]),
            dim=1,
            terminated=False,
            effect=0,
            **reward_kwargs,
        )
        assert isinstance(result, tuple)
        assert len(result) == 3
        total, components, new_dist = result
        assert isinstance(total, float)
        assert isinstance(components, dict)
        assert isinstance(new_dist, float)

    def test_components_present(self, reward_kwargs):
        """Reward components dict should contain expected keys."""
        _, components, _ = balloon_reward(
            balloon_pos=np.array([100.0]),
            goal_pos=np.array([0.0]),
            velocity=np.array([0.0]),
            dim=1,
            terminated=False,
            effect=0,
            **reward_kwargs,
        )
        assert "station" in components
        assert "decay" in components

    def test_total_is_sum(self, reward_kwargs):
        """Total reward should equal sum of components."""
        total, components, _ = balloon_reward(
            balloon_pos=np.array([500.0]),
            goal_pos=np.array([0.0]),
            velocity=np.array([-1.0]),
            dim=1,
            terminated=False,
            effect=0,
            **reward_kwargs,
        )
        expected = sum(components.values())
        assert total == pytest.approx(expected)

    def test_inside_radius_gives_one(self, reward_kwargs):
        """Inside station radius should give reward of 1.0."""
        total, components, _ = balloon_reward(
            balloon_pos=np.array([5000.0]),
            goal_pos=np.array([0.0]),
            velocity=np.array([0.0]),
            dim=1,
            terminated=False,
            effect=0,
            **reward_kwargs,
        )
        assert total == pytest.approx(1.0)
        assert components["station"] == pytest.approx(1.0)
        assert components["decay"] == pytest.approx(0.0)

    def test_at_boundary_gives_dropoff(self, reward_kwargs):
        """Exactly at station radius boundary should give reward_dropoff."""
        # Place balloon just outside the radius
        total, components, _ = balloon_reward(
            balloon_pos=np.array([10001.0]),
            goal_pos=np.array([0.0]),
            velocity=np.array([0.0]),
            dim=1,
            terminated=False,
            effect=0,
            **reward_kwargs,
        )
        assert total == pytest.approx(0.4, abs=0.01)

    def test_decay_at_one_halflife(self, reward_kwargs):
        """At one half-life past the radius, decay should be dropoff / 2."""
        # station_radius=10000, halflife=20000 -> at distance 30000 (excess=20000)
        total, _, _ = balloon_reward(
            balloon_pos=np.array([30000.0]),
            goal_pos=np.array([0.0]),
            velocity=np.array([0.0]),
            dim=1,
            terminated=False,
            effect=0,
            **reward_kwargs,
        )
        assert total == pytest.approx(0.2, rel=1e-3)

    def test_far_away_approaches_zero(self, reward_kwargs):
        """Very far from goal should give reward near zero."""
        total, _, _ = balloon_reward(
            balloon_pos=np.array([500000.0]),
            goal_pos=np.array([0.0]),
            velocity=np.array([0.0]),
            dim=1,
            terminated=False,
            effect=0,
            **reward_kwargs,
        )
        assert total < 0.001

    def test_closer_is_better(self, reward_kwargs):
        """Closer position should give higher reward."""
        total_far, _, _ = balloon_reward(
            balloon_pos=np.array([50000.0]),
            goal_pos=np.array([0.0]),
            velocity=np.array([0.0]),
            dim=1,
            terminated=False,
            effect=0,
            **reward_kwargs,
        )
        total_near, _, _ = balloon_reward(
            balloon_pos=np.array([15000.0]),
            goal_pos=np.array([0.0]),
            velocity=np.array([0.0]),
            dim=1,
            terminated=False,
            effect=0,
            **reward_kwargs,
        )
        assert total_near > total_far

    def test_reward_range_zero_to_one(self, reward_kwargs):
        """Reward should always be in [0, 1]."""
        for dist in [0, 500, 5000, 10000, 20000, 50000, 200000]:
            total, _, _ = balloon_reward(
                balloon_pos=np.array([float(dist)]),
                goal_pos=np.array([0.0]),
                velocity=np.array([0.0]),
                dim=1,
                terminated=False,
                effect=0,
                **reward_kwargs,
            )
            assert 0.0 <= total <= 1.0

    def test_terminated_gives_zero(self, reward_kwargs):
        """Terminated episode should return 0.0 reward."""
        total, components, _ = balloon_reward(
            balloon_pos=np.array([0.0]),
            goal_pos=np.array([1000.0]),
            velocity=np.array([0.0]),
            dim=1,
            terminated=True,
            effect=0,
            **reward_kwargs,
        )
        assert total == 0.0
        assert components["station"] == 0.0
        assert components["decay"] == 0.0

    def test_updates_distance(self, reward_kwargs):
        """Returned distance should match current position."""
        pos = np.array([750.0])
        goal = np.array([0.0])
        _, _, new_dist = balloon_reward(
            balloon_pos=pos,
            goal_pos=goal,
            velocity=np.array([0.0]),
            dim=1,
            terminated=False,
            effect=0,
            **reward_kwargs,
        )
        expected_dist = l2_distance(pos, goal, dim=1)
        assert new_dist == pytest.approx(expected_dist)

    def test_multidimensional(self, reward_kwargs):
        """Reward should work for all dimensions."""
        for dim in [1, 2, 3]:
            if dim == 1:
                pos = np.array([500.0])
                goal = np.array([0.0])
                vel = np.array([0.0])
            else:
                pos = np.array([300.0, 400.0, 500.0])
                goal = np.array([0.0, 0.0]) if dim == 2 else np.array([0.0, 0.0, 0.0])
                vel = np.array([0.0, 0.0, 0.0])

            total, components, dist = balloon_reward(
                balloon_pos=pos,
                goal_pos=goal,
                velocity=vel,
                dim=dim,
                terminated=False,
                effect=0,
                **reward_kwargs,
            )
            assert np.isfinite(total)
            assert all(np.isfinite(v) for v in components.values())
            assert np.isfinite(dist)
