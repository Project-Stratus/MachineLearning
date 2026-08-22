"""Tests for the Layer 1 baseline policies.

These build observations by hand rather than pulling them from the real env:
the observation layout is frozen by contract (§1), so a hand-built vector is a
*stricter* test than a live one — it fails loudly if either side drifts.
"""

import numpy as np
import pytest

from agents.baselines import (
    ACTION_DOWN,
    ACTION_STAY,
    ACTION_UP,
    AMBIENT_IDX,
    AMBIENT_START,
    CH_BEARING,
    CH_MAG,
    CH_UNCERTAINTY,
    IDX_DIST_NORM,
    LIMIT_TRIPLE,
    N_ACTIONS,
    OBS_WIDTH,
    WIND_COL_CENTRE,
    WIND_COL_LEVELS,
    WIND_COL_WIDTH,
    ALT_SAFE_SPAN,
    STATION_RADIUS_1D,
    VEL_Z_OBS_NORM,
    BangBangAgent,
    GreedyWindAgent,
    PassiveDriftAgent,
    RandomAgent,
    ambient,
    baselines_for_dim,
    level_offsets_m,
    limit_mask,
    make_baseline,
    wind_column,
)


# --------------------------------------------------------------------------- #
# Observation builder
# --------------------------------------------------------------------------- #
def make_obs(
    *,
    mag=0.5,
    bearing=0.5,
    dist_norm=0.5,
    levels=None,
    limit_levels=(),
    **ambient_overrides,
):
    """Build a well-formed 143-wide observation.

    ``levels`` overrides individual wind levels: ``{index: (mag, bearing)}``.
    ``limit_levels`` marks levels as outside the operational band.
    """
    obs = np.zeros(OBS_WIDTH, dtype=np.float32)

    column = np.zeros((WIND_COL_LEVELS, 3), dtype=np.float32)
    column[:, CH_MAG] = mag
    column[:, CH_BEARING] = bearing
    column[:, CH_UNCERTAINTY] = 0.0  # STUB in Layer 1
    for idx, (m, b) in (levels or {}).items():
        column[idx, CH_MAG] = m
        column[idx, CH_BEARING] = b
    for idx in limit_levels:
        column[idx, :] = LIMIT_TRIPLE

    obs[:WIND_COL_WIDTH] = column.reshape(-1)
    obs[IDX_DIST_NORM] = dist_norm
    obs[AMBIENT_IDX["alt_norm"]] = 0.5
    obs[AMBIENT_IDX["pressure_norm"]] = 0.5
    obs[AMBIENT_IDX["resource_a"]] = 1.0
    obs[AMBIENT_IDX["resource_b"]] = 1.0
    obs[AMBIENT_IDX["volume_norm"]] = 0.5
    obs[AMBIENT_IDX["last_action_stay"]] = 1.0
    obs[AMBIENT_IDX["solar_elevation"]] = 0.7  # STUB constants (Layer 2)
    obs[AMBIENT_IDX["solar_phase_cos"]] = 1.0
    for name, value in ambient_overrides.items():
        obs[AMBIENT_IDX[name]] = value
    return obs


# --------------------------------------------------------------------------- #
# Layout constants
# --------------------------------------------------------------------------- #
class TestObservationLayout:
    """The index block must match the frozen contract exactly."""

    def test_widths(self):
        assert OBS_WIDTH == 143
        assert WIND_COL_LEVELS == 41
        assert WIND_COL_WIDTH == 123
        assert WIND_COL_CENTRE == 20
        assert AMBIENT_START == 123

    def test_ambient_indices(self):
        assert len(AMBIENT_IDX) == 20
        assert AMBIENT_IDX["alt_norm"] == 123
        assert AMBIENT_IDX["dist_norm"] == 127
        assert AMBIENT_IDX["resource_a"] == 130
        assert AMBIENT_IDX["last_action_down"] == 133
        assert AMBIENT_IDX["at_alt_min"] == 136
        assert AMBIENT_IDX["solar_phase_cos"] == 142
        assert max(AMBIENT_IDX.values()) == OBS_WIDTH - 1

    def test_action_indices_match_env_lut(self):
        # `_action_lut = [-1, 0, +1]`: 0 descends, 1 does nothing, 2 ascends.
        assert (ACTION_DOWN, ACTION_STAY, ACTION_UP) == (0, 1, 2)
        assert N_ACTIONS == 3

    def test_wind_column_view_round_trips(self):
        obs = make_obs(levels={7: (0.25, -0.75)})
        column = wind_column(obs)
        assert column.shape == (WIND_COL_LEVELS, 3)
        assert column[7, CH_MAG] == pytest.approx(0.25)
        assert column[7, CH_BEARING] == pytest.approx(-0.75)
        assert np.all(column[:, CH_UNCERTAINTY] == 0.0)

    def test_ambient_view(self):
        obs = make_obs(dist_norm=0.3)
        amb = ambient(obs)
        assert amb.shape == (20,)
        assert amb[IDX_DIST_NORM - AMBIENT_START] == pytest.approx(0.3)

    def test_level_offsets(self):
        offsets = level_offsets_m()
        assert offsets.shape == (WIND_COL_LEVELS,)
        assert offsets[WIND_COL_CENTRE] == 0.0
        assert offsets[0] == pytest.approx(-5000.0)
        assert offsets[-1] == pytest.approx(5000.0)

    def test_limit_mask(self):
        obs = make_obs(limit_levels=(0, 1, 40))
        mask = limit_mask(wind_column(obs))
        assert mask.sum() == 3
        assert mask[0] and mask[1] and mask[40]
        assert not mask[WIND_COL_CENTRE]


# --------------------------------------------------------------------------- #
# PassiveDriftAgent
# --------------------------------------------------------------------------- #
class TestPassiveDriftAgent:
    def test_always_stay(self):
        agent = PassiveDriftAgent()
        rng = np.random.default_rng(0)
        for _ in range(25):
            obs = make_obs(
                mag=rng.random(), bearing=rng.uniform(-1, 1), dist_norm=rng.random()
            )
            action, state = agent.predict(obs, deterministic=True)
            assert int(action) == ACTION_STAY
            assert state is None

    def test_batched(self):
        agent = PassiveDriftAgent()
        batch = np.stack([make_obs() for _ in range(4)])
        actions, state = agent.predict(batch)
        assert actions.shape == (4,)
        assert np.all(actions == ACTION_STAY)
        assert state is None

    def test_rejects_wrong_width(self):
        agent = PassiveDriftAgent()
        with pytest.raises(ValueError, match="width 143"):
            agent.predict(np.zeros(19, dtype=np.float32))


# --------------------------------------------------------------------------- #
# RandomAgent
# --------------------------------------------------------------------------- #
class TestRandomAgent:
    def test_actions_are_valid(self):
        agent = RandomAgent(seed=0)
        obs = make_obs()
        actions = [int(agent.predict(obs)[0]) for _ in range(300)]
        assert set(actions) <= {ACTION_DOWN, ACTION_STAY, ACTION_UP}
        assert set(actions) == {ACTION_DOWN, ACTION_STAY, ACTION_UP}  # all three appear

    def test_reproducible_across_instances(self):
        obs = make_obs()
        first = _draw(RandomAgent(seed=7), obs, 100)
        second = _draw(RandomAgent(seed=7), obs, 100)
        assert first == second

    def test_different_seeds_differ(self):
        obs = make_obs()
        first = _draw(RandomAgent(seed=7), obs, 100)
        other = _draw(RandomAgent(seed=8), obs, 100)
        assert first != other

    def test_reset_restarts_stream(self):
        obs = make_obs()
        agent = RandomAgent(seed=123)
        first = _draw(agent, obs, 50)
        agent.reset()
        assert _draw(agent, obs, 50) == first

    def test_reset_with_new_seed(self):
        obs = make_obs()
        agent = RandomAgent(seed=1)
        first = _draw(agent, obs, 50)
        agent.reset(seed=2)
        assert _draw(agent, obs, 50) != first

    def test_does_not_touch_global_numpy_state(self):
        obs = make_obs()
        np.random.seed(0)
        expected = np.random.random()
        np.random.seed(0)
        _draw(RandomAgent(seed=99), obs, 50)
        assert np.random.random() == expected

    def test_roughly_uniform(self):
        obs = make_obs()
        counts = np.bincount(_draw(RandomAgent(seed=5), obs, 3000), minlength=3)
        assert np.all(counts > 800)  # ~1000 each


def _draw(agent, obs, n):
    return [int(agent.predict(obs)[0]) for _ in range(n)]


# --------------------------------------------------------------------------- #
# GreedyWindAgent
# --------------------------------------------------------------------------- #
class TestGreedyWindAgent:
    def test_commands_up_for_good_wind_above(self):
        """Far from station, one level above carries a fast goalward wind."""
        agent = GreedyWindAgent()
        obs = make_obs(
            mag=0.5,
            bearing=1.0,  # everywhere else: straight away from the goal
            dist_norm=0.5,  # 50 km out -> firmly in the "get back" regime
            levels={25: (1.0, 0.0)},  # 1.25 km up: fast, straight at the goal
        )
        assert agent.best_level(obs) == 25
        assert int(agent.predict(obs)[0]) == ACTION_UP

    def test_commands_down_for_good_wind_below(self):
        agent = GreedyWindAgent()
        obs = make_obs(mag=0.5, bearing=1.0, dist_norm=0.5, levels={12: (1.0, 0.0)})
        assert agent.best_level(obs) == 12
        assert int(agent.predict(obs)[0]) == ACTION_DOWN

    def test_stays_when_current_level_is_best(self):
        agent = GreedyWindAgent()
        obs = make_obs(
            mag=0.5,
            bearing=1.0,
            dist_norm=0.5,
            levels={WIND_COL_CENTRE: (1.0, 0.0)},
        )
        assert agent.best_level(obs) == WIND_COL_CENTRE
        assert int(agent.predict(obs)[0]) == ACTION_STAY

    def test_far_regime_prefers_goalward_wind_over_calm(self):
        """Far out, dead calm is useless — it never brings you home."""
        agent = GreedyWindAgent()
        obs = make_obs(
            mag=0.5,
            bearing=1.0,
            dist_norm=1.0,
            levels={
                15: (0.0, 0.0),  # dead calm, 1.25 km below
                25: (0.9, 0.0),
            },  # fast and goalward, 1.25 km above
        )
        assert agent.best_level(obs) == 25
        assert int(agent.predict(obs)[0]) == ACTION_UP

    def test_near_regime_prefers_calm_over_fast_goalward(self):
        """On station, the best wind is no wind — being carried anywhere is a loss."""
        agent = GreedyWindAgent()
        obs = make_obs(
            mag=0.8,
            bearing=0.5,
            dist_norm=0.0,
            levels={
                15: (0.0, 1.0),  # dead calm, direction irrelevant
                25: (0.9, 0.0),
            },  # fast, goalward
        )
        assert agent.best_level(obs) == 15
        assert int(agent.predict(obs)[0]) == ACTION_DOWN

    def test_regimes_flip_on_the_same_column(self):
        """Same wind column, different distance -> opposite command."""
        agent = GreedyWindAgent()
        kwargs = dict(mag=0.8, bearing=0.5, levels={15: (0.0, 1.0), 25: (0.9, 0.0)})
        assert int(agent.predict(make_obs(dist_norm=0.0, **kwargs))[0]) == ACTION_DOWN
        assert int(agent.predict(make_obs(dist_norm=0.9, **kwargs))[0]) == ACTION_UP

    def test_never_selects_a_limit_level(self):
        """Levels outside the operational band must be unreachable, always."""
        agent = GreedyWindAgent()
        above = tuple(range(WIND_COL_CENTRE + 1, WIND_COL_LEVELS))
        obs = make_obs(
            mag=0.9,
            bearing=1.0,
            dist_norm=0.5,
            levels={15: (0.9, 0.0)},  # the only decent wind is below
            limit_levels=above,
        )
        scores = agent.score_levels(wind_column(obs), obs[IDX_DIST_NORM])
        assert np.all(np.isneginf(scores[list(above)]))
        assert agent.best_level(obs) == 15
        assert int(agent.predict(obs)[0]) == ACTION_DOWN

    def test_limit_levels_lose_even_when_everything_else_is_bad(self):
        """Every real level is terrible; the limit levels must still not win."""
        agent = GreedyWindAgent()
        limits = tuple(range(WIND_COL_LEVELS))
        obs = make_obs(mag=1.0, bearing=1.0, dist_norm=0.5, limit_levels=limits)
        # Free two levels back up so there is a legal choice.
        column = wind_column(obs)
        column[WIND_COL_CENTRE] = (1.0, 0.9, 0.0)
        column[30] = (1.0, 0.95, 0.0)
        obs[:WIND_COL_WIDTH] = column.reshape(-1)

        assert agent.best_level(obs) not in set(limits) - {WIND_COL_CENTRE, 30}
        assert agent.best_level(obs) in (WIND_COL_CENTRE, 30)

    def test_all_levels_limited_falls_back_to_stay(self):
        agent = GreedyWindAgent()
        obs = make_obs(limit_levels=tuple(range(WIND_COL_LEVELS)))
        assert agent.best_level(obs) == WIND_COL_CENTRE
        assert int(agent.predict(obs)[0]) == ACTION_STAY

    def test_prefers_nearer_level_on_a_tie(self):
        """level_cost breaks ties toward the current altitude."""
        agent = GreedyWindAgent()
        obs = make_obs(
            mag=0.5, bearing=1.0, dist_norm=0.5, levels={22: (1.0, 0.0), 35: (1.0, 0.0)}
        )
        assert agent.best_level(obs) == 22

    def test_level_cost_does_not_reorder_real_differences(self):
        agent = GreedyWindAgent()
        obs = make_obs(
            mag=0.5, bearing=1.0, dist_norm=0.5, levels={22: (0.2, 0.0), 35: (1.0, 0.0)}
        )
        assert agent.best_level(obs) == 35

    def test_deadband_suppresses_small_corrections(self):
        obs = make_obs(mag=0.5, bearing=1.0, dist_norm=0.5, levels={22: (1.0, 0.0)})
        assert int(GreedyWindAgent().predict(obs)[0]) == ACTION_UP
        assert int(GreedyWindAgent(deadband_levels=3).predict(obs)[0]) == ACTION_STAY

    def test_returns_valid_actions_on_random_observations(self):
        agent = GreedyWindAgent()
        rng = np.random.default_rng(11)
        for _ in range(100):
            obs = make_obs(
                mag=rng.random(), bearing=rng.uniform(-1, 1), dist_norm=rng.random()
            )
            column = rng.random((WIND_COL_LEVELS, 3)).astype(np.float32)
            column[:, CH_BEARING] = rng.uniform(-1, 1, WIND_COL_LEVELS)
            column[:, CH_UNCERTAINTY] = 0.0
            obs[:WIND_COL_WIDTH] = column.reshape(-1)
            action, state = agent.predict(obs)
            assert int(action) in (ACTION_DOWN, ACTION_STAY, ACTION_UP)
            assert state is None

    def test_batched_predict(self):
        agent = GreedyWindAgent()
        up = make_obs(mag=0.5, bearing=1.0, dist_norm=0.5, levels={25: (1.0, 0.0)})
        down = make_obs(mag=0.5, bearing=1.0, dist_norm=0.5, levels={12: (1.0, 0.0)})
        actions, state = agent.predict(np.stack([up, down]))
        assert actions.tolist() == [ACTION_UP, ACTION_DOWN]
        assert state is None

    def test_bearing_sign_is_ignored(self):
        """+/- bearing is left/right of the goal; symmetric winds score equally."""
        agent = GreedyWindAgent()
        column = wind_column(make_obs(mag=0.5, bearing=0.0, dist_norm=0.5))
        left = column.copy()
        left[:, CH_BEARING] = 0.4
        right = column.copy()
        right[:, CH_BEARING] = -0.4
        np.testing.assert_allclose(
            agent.score_levels(left, 0.5), agent.score_levels(right, 0.5)
        )

    def test_rejects_bad_construction(self):
        with pytest.raises(ValueError):
            GreedyWindAgent(near_far_scale=0.0)
        with pytest.raises(ValueError):
            GreedyWindAgent(deadband_levels=-1)


# --------------------------------------------------------------------------- #
# BangBangAgent (1-D altitude hold)
# --------------------------------------------------------------------------- #
def _vel_norm(ms):
    """m/s -> the observation's normalised vertical velocity."""
    return ms / VEL_Z_OBS_NORM


def _dz_norm(metres):
    """metres of altitude error -> the observation's normalised goal_dz."""
    return metres / ALT_SAFE_SPAN


class TestBangBangAgent:
    """Switching logic only — the physics is the env's problem, not the policy's."""

    def test_is_declared_one_dimensional(self):
        assert BangBangAgent.dims == (1,)
        assert "bang_bang" in baselines_for_dim(1)
        assert "bang_bang" not in baselines_for_dim(3)

    def test_default_deadband_is_half_the_station_radius(self):
        assert BangBangAgent().deadband_m == pytest.approx(STATION_RADIUS_1D / 2.0)

    # -- position switching ------------------------------------------------ #
    def test_ascends_when_target_is_above(self):
        obs = make_obs(goal_dz_norm=_dz_norm(1_000.0), vel_z_norm=0.0)
        assert BangBangAgent()._action(obs) == ACTION_UP

    def test_descends_when_target_is_below(self):
        obs = make_obs(goal_dz_norm=_dz_norm(-1_000.0), vel_z_norm=0.0)
        assert BangBangAgent()._action(obs) == ACTION_DOWN

    def test_coasts_inside_the_deadband(self):
        obs = make_obs(goal_dz_norm=_dz_norm(100.0), vel_z_norm=0.0)
        assert BangBangAgent()._action(obs) == ACTION_STAY

    def test_deadband_edges_are_symmetric(self):
        agent = BangBangAgent(lead_time_s=0.0, deadband_m=250.0)
        assert agent._action(make_obs(goal_dz_norm=_dz_norm(251.0))) == ACTION_UP
        assert agent._action(make_obs(goal_dz_norm=_dz_norm(-251.0))) == ACTION_DOWN
        assert agent._action(make_obs(goal_dz_norm=_dz_norm(249.0))) == ACTION_STAY
        assert agent._action(make_obs(goal_dz_norm=_dz_norm(-249.0))) == ACTION_STAY

    # -- the velocity lead term -------------------------------------------- #
    def test_lead_term_stops_commanding_before_arrival(self):
        """500 m below target but climbing: naive position switching says UP,
        the lead term says the climb already covers it."""
        obs = make_obs(goal_dz_norm=_dz_norm(500.0), vel_z_norm=_vel_norm(3.0))
        assert BangBangAgent(lead_time_s=120.0)._action(obs) == ACTION_STAY
        # Without the lead term the same state commands UP — that is the overshoot.
        assert BangBangAgent(lead_time_s=0.0)._action(obs) == ACTION_UP

    def test_lead_term_actively_brakes_a_fast_approach(self):
        obs = make_obs(goal_dz_norm=_dz_norm(500.0), vel_z_norm=_vel_norm(8.0))
        assert BangBangAgent(lead_time_s=120.0)._action(obs) == ACTION_DOWN

    def test_lead_term_reinforces_when_moving_the_wrong_way(self):
        """Target above, descending: the lead term should push harder, not less."""
        obs = make_obs(goal_dz_norm=_dz_norm(500.0), vel_z_norm=_vel_norm(-5.0))
        assert BangBangAgent(lead_time_s=120.0)._action(obs) == ACTION_UP

    def test_predicted_error_denormalises_to_metres(self):
        obs = make_obs(goal_dz_norm=_dz_norm(1_000.0), vel_z_norm=_vel_norm(2.0))
        agent = BangBangAgent(lead_time_s=100.0)
        # 1000 m of error, minus 100 s of climbing at 2 m/s.
        assert agent.predicted_error_m(obs) == pytest.approx(800.0, abs=1.0)

    # -- safety-layer interaction ------------------------------------------ #
    def test_does_not_spend_ballast_against_the_ceiling(self):
        obs = make_obs(goal_dz_norm=_dz_norm(2_000.0), vel_z_norm=0.0, at_alt_max=1.0)
        assert BangBangAgent()._action(obs) == ACTION_STAY

    def test_does_not_spend_gas_against_the_floor(self):
        obs = make_obs(goal_dz_norm=_dz_norm(-2_000.0), vel_z_norm=0.0, at_alt_min=1.0)
        assert BangBangAgent()._action(obs) == ACTION_STAY

    def test_limit_flags_only_block_the_matching_direction(self):
        """At the ceiling, descending is still allowed — that is the way out."""
        obs = make_obs(goal_dz_norm=_dz_norm(-2_000.0), vel_z_norm=0.0, at_alt_max=1.0)
        assert BangBangAgent()._action(obs) == ACTION_DOWN

    def test_respect_limits_can_be_disabled(self):
        obs = make_obs(goal_dz_norm=_dz_norm(2_000.0), vel_z_norm=0.0, at_alt_max=1.0)
        assert BangBangAgent(respect_limits=False)._action(obs) == ACTION_UP

    # -- degeneracy that motivates the dim gate ----------------------------- #
    def test_coasts_forever_when_there_is_no_altitude_target(self):
        """2D/3D pin goal_dz_norm to 0, so this policy would duplicate passive
        drift there. Hence :func:`baselines_for_dim`."""
        agent = BangBangAgent()
        rng = np.random.default_rng(0)
        for _ in range(50):
            obs = make_obs(goal_dz_norm=0.0, vel_z_norm=float(rng.uniform(-0.05, 0.05)))
            if (
                abs(obs[AMBIENT_IDX["vel_z_norm"]]) * VEL_Z_OBS_NORM * agent.lead_time_s
                <= agent.deadband_m
            ):
                assert agent._action(obs) == ACTION_STAY

    # -- interface ---------------------------------------------------------- #
    def test_batched_predict(self):
        obs = np.stack(
            [
                make_obs(goal_dz_norm=_dz_norm(1_000.0)),
                make_obs(goal_dz_norm=_dz_norm(-1_000.0)),
                make_obs(goal_dz_norm=0.0),
            ]
        )
        actions, state = BangBangAgent().predict(obs)
        assert state is None
        assert list(actions) == [ACTION_UP, ACTION_DOWN, ACTION_STAY]

    def test_returns_valid_actions_on_random_observations(self):
        agent = BangBangAgent()
        rng = np.random.default_rng(7)
        for _ in range(200):
            obs = make_obs(
                goal_dz_norm=float(rng.uniform(-1.0, 1.0)),
                vel_z_norm=float(rng.uniform(-1.0, 1.0)),
                at_alt_min=float(rng.integers(2)),
                at_alt_max=float(rng.integers(2)),
            )
            assert agent._action(obs) in (ACTION_DOWN, ACTION_STAY, ACTION_UP)

    def test_rejects_wrong_width(self):
        with pytest.raises(ValueError, match="width"):
            BangBangAgent().predict(np.zeros(OBS_WIDTH - 1, dtype=np.float32))

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(lead_time_s=-1.0),
            dict(deadband_m=-1.0),
        ],
    )
    def test_rejects_bad_construction(self, kwargs):
        with pytest.raises(ValueError):
            BangBangAgent(**kwargs)

    def test_repr_round_trips_parameters(self):
        r = repr(BangBangAgent(lead_time_s=90.0, deadband_m=300.0))
        assert "lead_time_s=90" in r and "deadband_m=300" in r


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
class TestRegistry:
    @pytest.mark.parametrize("name", ["passive", "random", "greedy_wind", "bang_bang"])
    def test_make_baseline(self, name):
        agent = make_baseline(name)
        action, state = agent.predict(make_obs())
        assert int(action) in (0, 1, 2)
        assert state is None

    def test_make_baseline_passes_kwargs(self):
        assert make_baseline("random", seed=3).seed == 3

    def test_unknown_name(self):
        with pytest.raises(KeyError):
            make_baseline("perciatelli")
