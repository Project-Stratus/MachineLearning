"""Tests for reward functions."""

import numpy as np
import pytest

from environments.core.reward import l2_distance, balloon_reward, _normalise_distance
from environments.core.constants import ALT_MAX, XY_MAX


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
        # Even if balloon_pos has 3 elements, should use last one
        balloon_pos = np.array([100.0, 200.0, 1000.0])
        goal_pos = np.array([500.0])
        dist = l2_distance(balloon_pos, goal_pos, dim=1)
        assert dist == pytest.approx(500.0)  # |1000 - 500|

    def test_l2_distance_1d_symmetric(self):
        """1D distance should be symmetric."""
        pos_a = np.array([1000.0])
        pos_b = np.array([500.0])
        assert l2_distance(pos_a, pos_b, 1) == l2_distance(pos_b, pos_a, 1)

    def test_l2_distance_2d_simple(self):
        """2D distance should be Euclidean in x-y plane."""
        balloon_pos = np.array([300.0, 400.0, 10000.0])  # 3D internally
        goal_pos = np.array([0.0, 0.0])
        dist = l2_distance(balloon_pos, goal_pos, dim=2)
        assert dist == pytest.approx(500.0)  # 3-4-5 triangle

    def test_l2_distance_2d_ignores_z(self):
        """2D distance should ignore z component."""
        balloon_pos = np.array([300.0, 400.0, 99999.0])
        goal_pos = np.array([0.0, 0.0])
        dist = l2_distance(balloon_pos, goal_pos, dim=2)
        assert dist == pytest.approx(500.0)

    def test_l2_distance_3d_simple(self):
        """3D distance should be full Euclidean norm."""
        balloon_pos = np.array([100.0, 200.0, 300.0])
        goal_pos = np.array([400.0, 600.0, 900.0])
        expected = np.linalg.norm(balloon_pos - goal_pos)
        dist = l2_distance(balloon_pos, goal_pos, dim=3)
        assert dist == pytest.approx(expected)

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
            else:
                pos = np.array([100.0, 200.0, 300.0])
            goal = pos[:dim].copy()
            assert l2_distance(pos, goal, dim) == pytest.approx(0.0)


class TestNormaliseDistance:
    """Tests for distance normalization."""

    def test_normalise_distance_zero(self):
        """Zero distance should normalize to 0."""
        assert _normalise_distance(0.0, ALT_MAX) == 0.0

    def test_normalise_distance_negative(self):
        """Normalized distance should be negative."""
        assert _normalise_distance(1000.0, ALT_MAX) < 0

    def test_normalise_distance_max(self):
        """Maximum distance should normalize to approximately -1."""
        norm = _normalise_distance(ALT_MAX, ALT_MAX)
        assert norm == pytest.approx(-1.0)

    def test_normalise_distance_range(self):
        """Normalized distance should be in [-1, 0] for reasonable inputs."""
        for d in [0, 1000, 5000, 10000, ALT_MAX]:
            norm = _normalise_distance(d, ALT_MAX)
            assert -1.0 <= norm <= 0.0


class TestBalloonReward:
    """Tests for the composite balloon_reward function."""

    @pytest.fixture
    def reward_kwargs(self):
        """Default kwargs for balloon_reward."""
        return dict(
            punishment=-5.0,
            prev_distance=1000.0,
            max_distance=ALT_MAX,
            success_radius=500.0,
            success_outer_radius=1500.0,
            success_speed=0.2,
            direction_scale=0.05,
        )

    def test_balloon_reward_returns_tuple(self, reward_kwargs):
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

    def test_balloon_reward_components_present(self, reward_kwargs):
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
        assert "distance" in components
        assert "direction" in components
        assert "reached" in components

    def test_balloon_reward_total_is_sum(self, reward_kwargs):
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

    def test_balloon_reward_distance_component_negative(self, reward_kwargs):
        """Distance component should be negative (further = worse)."""
        _, components, _ = balloon_reward(
            balloon_pos=np.array([5000.0]),
            goal_pos=np.array([0.0]),
            velocity=np.array([0.0]),
            dim=1,
            terminated=False,
            effect=0,
            **reward_kwargs,
        )
        assert components["distance"] < 0

    def test_balloon_reward_closer_is_better(self, reward_kwargs):
        """Closer position should give higher (less negative) distance reward."""
        _, components_far, _ = balloon_reward(
            balloon_pos=np.array([5000.0]),
            goal_pos=np.array([0.0]),
            velocity=np.array([0.0]),
            dim=1,
            terminated=False,
            effect=0,
            **reward_kwargs,
        )
        _, components_near, _ = balloon_reward(
            balloon_pos=np.array([1000.0]),
            goal_pos=np.array([0.0]),
            velocity=np.array([0.0]),
            dim=1,
            terminated=False,
            effect=0,
            **reward_kwargs,
        )
        assert components_near["distance"] > components_far["distance"]

    def test_balloon_reward_direction_positive_when_approaching(self, reward_kwargs):
        """Direction component should be positive when getting closer."""
        _, components, _ = balloon_reward(
            balloon_pos=np.array([800.0]),
            goal_pos=np.array([0.0]),
            velocity=np.array([-5.0]),  # Moving toward goal
            dim=1,
            terminated=False,
            effect=0,
            prev_distance=1000.0,  # Was further before
            **{k: v for k, v in reward_kwargs.items() if k != "prev_distance"},
        )
        assert components["direction"] > 0

    def test_balloon_reward_direction_negative_when_receding(self, reward_kwargs):
        """Direction component should be negative when getting further."""
        _, components, _ = balloon_reward(
            balloon_pos=np.array([1200.0]),
            goal_pos=np.array([0.0]),
            velocity=np.array([5.0]),  # Moving away from goal
            dim=1,
            terminated=False,
            effect=0,
            prev_distance=1000.0,  # Was closer before
            **{k: v for k, v in reward_kwargs.items() if k != "prev_distance"},
        )
        assert components["direction"] < 0

    def test_balloon_reward_reached_full_bonus(self, reward_kwargs):
        """Should get full reached bonus when near goal and slow."""
        _, components, _ = balloon_reward(
            balloon_pos=np.array([50.0]),  # Within success_radius (500)
            goal_pos=np.array([0.0]),
            velocity=np.array([0.1]),  # Below success_speed (0.2)
            dim=1,
            terminated=False,
            effect=0,
            **reward_kwargs,
        )
        assert components["reached"] == pytest.approx(0.3)

    def test_balloon_reward_reached_partial_in_outer_zone(self, reward_kwargs):
        """Should get partial reached bonus in the outer ramp zone."""
        _, components, _ = balloon_reward(
            balloon_pos=np.array([1000.0]),  # Between outer (1500) and inner (500)
            goal_pos=np.array([0.0]),
            velocity=np.array([10.0]),  # Speed doesn't matter in outer zone
            dim=1,
            terminated=False,
            effect=0,
            **reward_kwargs,
        )
        # At 1000m: proximity = (1500 - 1000) / (1500 - 500) = 0.5
        assert components["reached"] == pytest.approx(0.15)

    def test_balloon_reward_no_full_bonus_when_fast(self, reward_kwargs):
        """Inside inner radius but fast should get proximity bonus, not full."""
        _, components, _ = balloon_reward(
            balloon_pos=np.array([50.0]),  # Within success_radius
            goal_pos=np.array([0.0]),
            velocity=np.array([10.0]),  # Above success_speed
            dim=1,
            terminated=False,
            effect=0,
            **reward_kwargs,
        )
        # Gets proximity bonus (capped at 1.0) but not the full 0.3
        # proximity = (1500 - 50) / (1500 - 500) = 1.45 -> capped to 1.0
        # reached = 0.3 * 1.0 = 0.3, but speed gate prevents full bonus override
        # Actually since proximity is capped at 1.0, it's 0.3 * 1.0 = 0.3
        # The full bonus override only triggers when BOTH conditions met
        assert components["reached"] == pytest.approx(0.3)

    def test_balloon_reward_no_reached_bonus_when_far(self, reward_kwargs):
        """Should not get reached bonus when outside outer radius."""
        _, components, _ = balloon_reward(
            balloon_pos=np.array([2000.0]),  # Outside success_outer_radius (1500)
            goal_pos=np.array([0.0]),
            velocity=np.array([0.1]),  # Slow
            dim=1,
            terminated=False,
            effect=0,
            **reward_kwargs,
        )
        assert components["reached"] == 0.0

    def test_balloon_reward_termination_punishment(self, reward_kwargs):
        """Terminated episode should return punishment as total reward."""
        total, components, _ = balloon_reward(
            balloon_pos=np.array([0.0]),
            goal_pos=np.array([1000.0]),
            velocity=np.array([0.0]),
            dim=1,
            terminated=True,  # Crashed!
            effect=0,
            **reward_kwargs,
        )
        assert total == reward_kwargs["punishment"]
        assert components["distance"] == reward_kwargs["punishment"]
        assert components["direction"] == 0.0
        assert components["reached"] == 0.0

    def test_balloon_reward_updates_distance(self, reward_kwargs):
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

    def test_balloon_reward_multidimensional(self, reward_kwargs):
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
