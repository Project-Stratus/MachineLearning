"""Tests for reward functions."""

import math
import numpy as np
import pytest

from environments.core.constants import (
    STATION_RADIUS, REWARD_DROPOFF, REWARD_HALFLIFE,
    RESOURCE_PENALTY_BASE, RESOURCE_PENALTY_SLOPE,
)
from environments.core.reward import (
    l2_distance, balloon_reward, time_within_radius, _resource_factor,
)


EXPECTED_KEYS = {"station", "decay", "base", "resource_factor"}


def _reward_1d(distance, **kwargs):
    """balloon_reward at a 1-D offset of *distance* metres from the goal."""
    return balloon_reward(
        balloon_pos=np.array([float(distance)]),
        goal_pos=np.array([0.0]),
        dim=1,
        terminated=kwargs.pop("terminated", False),
        **kwargs,
    )


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
        expected = math.sqrt((100.0 - 400.0) ** 2 + (200.0 - 600.0) ** 2)
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
        assert l2_distance(pos_a, pos_b, 3) == pytest.approx(
            l2_distance(pos_b, pos_a, 3)
        )

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


class TestRewardStructure:
    """Return shape, component keys and the relations between components."""

    def test_returns_triple(self):
        result = _reward_1d(100.0)
        assert isinstance(result, tuple) and len(result) == 3
        total, components, distance = result
        assert isinstance(total, float)
        assert isinstance(components, dict)
        assert isinstance(distance, float)

    def test_component_keys_exact(self):
        """Component keys must be exactly the four contracted names."""
        for terminated in (False, True):
            _, components, _ = _reward_1d(25_000.0, terminated=terminated)
            assert set(components.keys()) == EXPECTED_KEYS

    @pytest.mark.parametrize("distance", [0.0, 5_000.0, 10_000.0, 25_000.0, 90_000.0])
    @pytest.mark.parametrize("omega", [0.0, 0.25, 1.0])
    def test_base_is_station_plus_decay(self, distance, omega):
        _, c, _ = _reward_1d(distance, resource_consumed_frac=omega)
        assert c["base"] == pytest.approx(c["station"] + c["decay"])

    @pytest.mark.parametrize("distance", [0.0, 5_000.0, 10_000.0, 25_000.0, 90_000.0])
    @pytest.mark.parametrize("omega", [0.0, 0.25, 1.0])
    def test_total_is_base_times_factor(self, distance, omega):
        """Total is the *product* of base and the resource factor, not a sum."""
        total, c, _ = _reward_1d(distance, resource_consumed_frac=omega)
        assert total == pytest.approx(c["base"] * c["resource_factor"])

    def test_returns_current_distance(self):
        pos = np.array([750.0])
        goal = np.array([0.0])
        _, _, distance = balloon_reward(
            balloon_pos=pos, goal_pos=goal, dim=1, terminated=False,
        )
        assert distance == pytest.approx(l2_distance(pos, goal, dim=1))

    def test_all_dims_finite(self):
        for dim in [1, 2, 3]:
            if dim == 1:
                pos, goal = np.array([500.0]), np.array([0.0])
            else:
                pos = np.array([30_000.0, 40_000.0, 18_000.0])
                goal = np.array([0.0, 0.0]) if dim == 2 else np.array([0.0, 0.0, 0.0])
            total, components, distance = balloon_reward(
                balloon_pos=pos, goal_pos=goal, dim=dim, terminated=False,
                resource_consumed_frac=0.4,
            )
            assert np.isfinite(total)
            assert all(np.isfinite(v) for v in components.values())
            assert np.isfinite(distance)

    def test_defaults_come_from_constants(self):
        """Omitting the shaping kwargs must reproduce the constants."""
        implicit, _, _ = _reward_1d(STATION_RADIUS + REWARD_HALFLIFE)
        explicit, _, _ = _reward_1d(
            STATION_RADIUS + REWARD_HALFLIFE,
            station_radius=STATION_RADIUS,
            reward_dropoff=REWARD_DROPOFF,
            reward_halflife=REWARD_HALFLIFE,
        )
        assert implicit == pytest.approx(explicit)


class TestDistanceShaping:
    """Flat plateau inside, cliff at the boundary, exponential decay outside."""

    @pytest.mark.parametrize(
        "distance", [0.0, 1.0, 2_500.0, 9_999.0, STATION_RADIUS]
    )
    def test_flat_one_inside_radius(self, distance):
        """Reward is a flat 1.0 anywhere inside the radius, boundary included."""
        total, c, _ = _reward_1d(distance)
        assert total == pytest.approx(1.0)
        assert c["station"] == pytest.approx(1.0)
        assert c["decay"] == pytest.approx(0.0)
        assert c["base"] == pytest.approx(1.0)

    def test_cliff_at_boundary(self):
        """Crossing the boundary drops the reward from 1.0 straight to dropoff."""
        inside, _, _ = _reward_1d(STATION_RADIUS)
        just_outside, c, _ = _reward_1d(STATION_RADIUS + 1e-6)
        assert inside == pytest.approx(1.0)
        assert just_outside == pytest.approx(REWARD_DROPOFF, rel=1e-9)
        assert c["station"] == 0.0
        assert c["decay"] == pytest.approx(REWARD_DROPOFF, rel=1e-9)
        # The cliff is a genuine discontinuity, not a steep ramp.
        assert inside - just_outside == pytest.approx(1.0 - REWARD_DROPOFF, rel=1e-6)

    @pytest.mark.parametrize("n_halflives", [1, 2, 3, 4])
    def test_halflife_actually_halves(self, n_halflives):
        """Each half-life past the boundary halves the reward."""
        total, _, _ = _reward_1d(STATION_RADIUS + n_halflives * REWARD_HALFLIFE)
        assert total == pytest.approx(REWARD_DROPOFF * 0.5 ** n_halflives, rel=1e-9)

    def test_halving_is_scale_free(self):
        """Ratio between points one half-life apart is 0.5 wherever you sample."""
        for start in (3_000.0, 17_500.0, 61_000.0):
            near, _, _ = _reward_1d(STATION_RADIUS + start)
            far, _, _ = _reward_1d(STATION_RADIUS + start + REWARD_HALFLIFE)
            assert far / near == pytest.approx(0.5, rel=1e-9)

    def test_custom_halflife_respected(self):
        total, _, _ = _reward_1d(
            15_000.0, station_radius=5_000.0, reward_dropoff=0.8,
            reward_halflife=10_000.0,
        )
        assert total == pytest.approx(0.4, rel=1e-9)

    def test_monotone_decreasing_outside(self):
        distances = [10_001.0, 15_000.0, 30_000.0, 50_000.0, 120_000.0]
        totals = [_reward_1d(d)[0] for d in distances]
        assert all(a > b for a, b in zip(totals, totals[1:]))

    def test_far_away_approaches_zero(self):
        total, _, _ = _reward_1d(500_000.0)
        assert 0.0 < total < 1e-3

    @pytest.mark.parametrize("omega", [0.0, 0.5, 1.0])
    @pytest.mark.parametrize(
        "distance", [0.0, 500.0, 5_000.0, 10_000.0, 20_000.0, 50_000.0, 200_000.0]
    )
    def test_reward_bounded_zero_to_one(self, distance, omega):
        total, c, _ = _reward_1d(distance, resource_consumed_frac=omega)
        assert 0.0 <= total <= 1.0
        assert all(0.0 <= v <= 1.0 for v in c.values())


class TestTermination:
    """terminated -> everything zero."""

    @pytest.mark.parametrize("distance", [0.0, 5_000.0, 250_000.0])
    @pytest.mark.parametrize("omega", [0.0, 0.5, 1.0])
    def test_terminated_zeroes_everything(self, distance, omega):
        total, c, _ = _reward_1d(
            distance, terminated=True, resource_consumed_frac=omega
        )
        assert total == 0.0
        assert set(c.keys()) == EXPECTED_KEYS
        assert all(v == 0.0 for v in c.values())

    def test_terminated_still_reports_distance(self):
        _, _, distance = _reward_1d(4_321.0, terminated=True)
        assert distance == pytest.approx(4_321.0)


class TestResourcePenalty:
    """The multiplicative resource-consumption factor."""

    def test_factor_is_one_when_nothing_consumed(self):
        """Explicit zero and the default argument both give exactly 1.0."""
        _, c_explicit, _ = _reward_1d(5_000.0, resource_consumed_frac=0.0)
        _, c_default, _ = _reward_1d(5_000.0)
        assert c_explicit["resource_factor"] == 1.0
        assert c_default["resource_factor"] == 1.0

    @pytest.mark.parametrize("omega", [0.05, 0.25, 0.5, 0.75, 1.0])
    def test_factor_formula(self, omega):
        _, c, _ = _reward_1d(5_000.0, resource_consumed_frac=omega)
        expected = RESOURCE_PENALTY_BASE - RESOURCE_PENALTY_SLOPE * omega
        assert c["resource_factor"] == pytest.approx(expected)

    def test_discontinuity_at_zero(self):
        """The first drop of consumption costs 1 - BASE outright."""
        _, c_none, _ = _reward_1d(5_000.0, resource_consumed_frac=0.0)
        _, c_epsilon, _ = _reward_1d(5_000.0, resource_consumed_frac=1e-12)
        assert c_none["resource_factor"] == 1.0
        assert c_epsilon["resource_factor"] == pytest.approx(RESOURCE_PENALTY_BASE)

    def test_applied_multiplicatively_inside_radius(self):
        omega = 0.6
        total, c, _ = _reward_1d(3_000.0, resource_consumed_frac=omega)
        factor = RESOURCE_PENALTY_BASE - RESOURCE_PENALTY_SLOPE * omega
        assert c["base"] == pytest.approx(1.0)
        assert total == pytest.approx(1.0 * factor)

    def test_applied_multiplicatively_outside_radius(self):
        """Far from the station the penalty scales the reward, it does not
        swamp it — an additive penalty would drive the total negative here."""
        omega = 1.0
        clean, _, _ = _reward_1d(80_000.0, resource_consumed_frac=0.0)
        spent, c, _ = _reward_1d(80_000.0, resource_consumed_frac=omega)
        factor = RESOURCE_PENALTY_BASE - RESOURCE_PENALTY_SLOPE * omega
        assert spent == pytest.approx(clean * factor)
        assert spent > 0.0

    def test_penalty_preserves_distance_ordering(self):
        """A fully-spent balloon on station still beats a full one far away —
        this is the property an additive penalty destroys."""
        near_spent, _, _ = _reward_1d(1_000.0, resource_consumed_frac=1.0)
        far_full, _, _ = _reward_1d(60_000.0, resource_consumed_frac=0.0)
        assert near_spent > far_full

    def test_monotone_decreasing_in_omega(self):
        omegas = [0.0, 0.1, 0.3, 0.6, 0.9, 1.0]
        totals = [_reward_1d(5_000.0, resource_consumed_frac=w)[0] for w in omegas]
        assert all(a > b for a, b in zip(totals, totals[1:]))

    def test_factor_floored_at_zero(self):
        """With a steep enough slope the factor floors at 0, never negative."""
        assert _resource_factor(1.0, base=0.5, slope=2.0) == 0.0
        assert _resource_factor(0.9, base=0.2, slope=5.0) == 0.0
        for omega in np.linspace(0.0, 1.0, 21):
            assert _resource_factor(float(omega), base=0.4, slope=10.0) >= 0.0

    def test_omega_above_one_is_clipped(self):
        """An over-budget accounting slip cannot run the factor negative."""
        _, c_one, _ = _reward_1d(5_000.0, resource_consumed_frac=1.0)
        _, c_over, _ = _reward_1d(5_000.0, resource_consumed_frac=7.5)
        assert c_over["resource_factor"] == pytest.approx(c_one["resource_factor"])

    def test_negative_omega_treated_as_unspent(self):
        _, c, _ = _reward_1d(5_000.0, resource_consumed_frac=-0.3)
        assert c["resource_factor"] == 1.0

    def test_nan_omega_treated_as_unspent(self):
        total, c, _ = _reward_1d(5_000.0, resource_consumed_frac=float("nan"))
        assert c["resource_factor"] == 1.0
        assert total == pytest.approx(1.0)


class TestTimeWithinRadius:
    """The TWR evaluation metric."""

    def test_known_fraction(self):
        distances = [0.0, 5_000.0, 9_999.0, 10_001.0, 50_000.0]
        assert time_within_radius(distances) == pytest.approx(0.6)

    def test_all_inside(self):
        assert time_within_radius([0.0, 100.0, 9_000.0]) == pytest.approx(1.0)

    def test_all_outside(self):
        assert time_within_radius([20_000.0, 30_000.0]) == pytest.approx(0.0)

    def test_boundary_counts_as_inside(self):
        """Inclusive boundary, matching balloon_reward's flat plateau."""
        assert time_within_radius([STATION_RADIUS]) == pytest.approx(1.0)
        assert time_within_radius(
            [STATION_RADIUS + 1e-6]
        ) == pytest.approx(0.0)

    def test_accepts_list_and_ndarray_alike(self):
        distances = [1_000.0, 12_000.0, 8_000.0, 40_000.0]
        assert time_within_radius(distances) == pytest.approx(
            time_within_radius(np.asarray(distances))
        )

    def test_accepts_integer_array(self):
        assert time_within_radius(np.array([0, 20_000, 5_000])) == pytest.approx(2 / 3)

    def test_stacked_traces_are_pooled(self):
        """A (n_episodes, n_steps) array is flattened and pooled."""
        traces = np.array([[0.0, 0.0], [20_000.0, 20_000.0]])
        assert time_within_radius(traces) == pytest.approx(0.5)

    def test_custom_radius(self):
        distances = [500.0, 1_500.0, 2_500.0, 3_500.0]
        twr = time_within_radius(distances, station_radius=2_000.0)
        assert twr == pytest.approx(0.5)

    def test_empty_input_returns_zero(self):
        """Empty input is defined as 0.0 (not NaN) so aggregation stays safe."""
        assert time_within_radius([]) == 0.0
        assert time_within_radius(np.array([])) == 0.0

    def test_result_in_unit_interval(self):
        rng = np.random.default_rng(0)
        for _ in range(10):
            distances = rng.uniform(0.0, 60_000.0, size=64)
            assert 0.0 <= time_within_radius(distances) <= 1.0

    def test_agrees_with_reward_plateau(self):
        """A sample is 'within radius' exactly when it earns the flat 1.0."""
        distances = np.array([0.0, 4_000.0, STATION_RADIUS, 11_000.0, 80_000.0])
        on_station = [
            _reward_1d(d)[1]["station"] == pytest.approx(1.0) for d in distances
        ]
        assert time_within_radius(distances) == pytest.approx(
            sum(on_station) / len(on_station)
        )
