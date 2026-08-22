"""Tests for the frozen observation layout (Layer 1 contract §1).

The layout is width 143 and **identical for dim 1, 2 and 3** — fields that are
meaningless in a dimension are zeroed, never omitted, so that later layers can
populate the stub fields in place without changing the network's input width
(roadmap §2.1).  These tests pin that contract from the producing side; the
consuming side (``agents.baselines``) mirrors the same indices and is
cross-checked here, because a silent disagreement between the two would show up
only as a baseline that quietly misreads the wind column.
"""

import math

import numpy as np
import pytest

from agents import baselines as bl
from environments.core.constants import (
    AIR_BLADDER_MAX,
    ALT_SAFE_MAX,
    ALT_SAFE_MIN,
    BALLAST_INITIAL,
    DIST_NORM,
    OBS_WIDTH,
    P_MAX,
    SP_VOL_FIXED,
    VEL_Z_OBS_NORM,
    VOL_MAX,
    WIND_COL_LEVELS,
    WIND_COL_SPACING,
    WIND_MAG_NORM,
)
from environments.envs import balloon_3d_env as env_mod
from environments.envs.balloon_3d_env import (
    AMBIENT_FIELDS,
    AMBIENT_IDX,
    AMBIENT_START,
    CH_BEARING,
    CH_MAG,
    CH_UNCERTAINTY,
    LIMIT_TRIPLE,
    SOLAR_ELEVATION_STUB,
    SOLAR_PHASE_COS_STUB,
    SOLAR_PHASE_SIN_STUB,
    WIND_COL_CENTRE,
    WIND_COL_CHANNELS,
    WIND_COL_WIDTH,
    Balloon3DEnv,
)
from tests.conftest import expected_obs_size, make_env

#: Ambient scalars whose declared range is [-1, 1] rather than [0, 1].
SIGNED_AMBIENT = (
    "goal_dz_norm", "vel_z_norm", "heading_sin", "heading_cos",
    "solar_phase_sin", "solar_phase_cos",
)


def wind_column(obs):
    """View the wind column as ``(levels, channels)``."""
    return np.asarray(obs)[:WIND_COL_WIDTH].reshape(WIND_COL_LEVELS, WIND_COL_CHANNELS)


def level_altitude(z: float, i: int) -> float:
    """Altitude of wind-column level ``i`` for a balloon at ``z``."""
    return z + (i - WIND_COL_CENTRE) * WIND_COL_SPACING


class TestLayoutIsFrozen:
    """Width and index constants — the part that must never move."""

    def test_width_is_143_for_every_dim(self, env_any_dim):
        env, dim = env_any_dim
        assert env.observation_space.shape == (OBS_WIDTH,) == (143,)
        obs, _ = env.reset(seed=42)
        assert obs.shape == (expected_obs_size(dim),)

    def test_sp_width_matches_zp(self, env_sp_any_dim):
        env, _ = env_sp_any_dim
        obs, _ = env.reset(seed=42)
        assert obs.shape == (OBS_WIDTH,)

    def test_wind_column_occupies_first_123(self):
        assert WIND_COL_WIDTH == 123
        assert WIND_COL_CENTRE == 20
        assert AMBIENT_START == 123
        assert AMBIENT_START + len(AMBIENT_FIELDS) == OBS_WIDTH

    def test_ambient_field_order(self):
        assert AMBIENT_FIELDS == (
            "alt_norm", "goal_dz_norm", "pressure_norm", "vel_z_norm", "dist_norm",
            "heading_sin", "heading_cos", "resource_a", "resource_b", "volume_norm",
            "last_action_down", "last_action_stay", "last_action_up",
            "at_alt_min", "at_alt_max", "resource_a_low", "resource_b_low",
            "solar_elevation", "solar_phase_sin", "solar_phase_cos",
        )
        assert AMBIENT_IDX["alt_norm"] == 123
        assert AMBIENT_IDX["solar_phase_cos"] == 142

    def test_agrees_with_agents_baselines(self):
        """The consumer's mirror of the layout must match this producer exactly.

        ``agents.baselines`` duplicates the index block so the agent package
        stays importable without the physics stack. Duplication is only safe if
        something fails loudly when the two drift.
        """
        assert bl.OBS_WIDTH == OBS_WIDTH
        assert bl.WIND_COL_WIDTH == WIND_COL_WIDTH
        assert bl.WIND_COL_CENTRE == WIND_COL_CENTRE
        assert bl.WIND_COL_CHANNELS == WIND_COL_CHANNELS
        assert (bl.CH_MAG, bl.CH_BEARING, bl.CH_UNCERTAINTY) == (CH_MAG, CH_BEARING, CH_UNCERTAINTY)
        assert bl.LIMIT_TRIPLE == LIMIT_TRIPLE
        assert bl.AMBIENT_FIELDS == AMBIENT_FIELDS
        assert bl.AMBIENT_IDX == AMBIENT_IDX
        assert (bl.ACTION_DOWN, bl.ACTION_STAY, bl.ACTION_UP) == (
            env_mod.ACTION_DOWN, env_mod.ACTION_STAY, env_mod.ACTION_UP)


class TestObservationSpaceBounds:
    """Bounds are per-field, not a blanket 0..1 box."""

    def test_dtype_is_float32(self, env_any_dim):
        env, _ = env_any_dim
        assert env.observation_space.dtype == np.float32
        obs, _ = env.reset(seed=1)
        assert obs.dtype == np.float32

    def test_wind_column_bounds(self, env_any_dim):
        env, _ = env_any_dim
        low, high = env.observation_space.low, env.observation_space.high
        assert np.all(low[CH_MAG:WIND_COL_WIDTH:WIND_COL_CHANNELS] == 0.0)
        assert np.all(low[CH_BEARING:WIND_COL_WIDTH:WIND_COL_CHANNELS] == -1.0)
        assert np.all(low[CH_UNCERTAINTY:WIND_COL_WIDTH:WIND_COL_CHANNELS] == 0.0)
        assert np.all(high[:WIND_COL_WIDTH] == 1.0)

    def test_ambient_bounds(self, env_any_dim):
        env, _ = env_any_dim
        low, high = env.observation_space.low, env.observation_space.high
        for name in AMBIENT_FIELDS:
            idx = AMBIENT_IDX[name]
            expected_low = -1.0 if name in SIGNED_AMBIENT else 0.0
            assert low[idx] == expected_low, f"{name} low bound"
            assert high[idx] == 1.0, f"{name} high bound"

    def test_bounds_are_not_blanket_zero_one(self, env_1d):
        """Regression guard: a signed field declared as [0,1] is a silent trap."""
        assert not np.all(env_1d.observation_space.low == 0.0)

    def test_buffer_matches_space(self, env_any_dim):
        env, _ = env_any_dim
        assert env._obs_size == OBS_WIDTH
        assert env._obs_buf.shape == (OBS_WIDTH,)

    @pytest.mark.parametrize("balloon_type", ["zero_pressure", "superpressure"])
    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_every_field_in_bounds_over_a_rollout(self, dim, balloon_type):
        env = make_env(dim, balloon_type=balloon_type, time_max=200)
        try:
            obs, _ = env.reset(seed=7)
            assert env.observation_space.contains(obs)
            rng = np.random.default_rng(0)
            for _ in range(200):
                obs, _, term, trunc, _ = env.step(int(rng.integers(3)))
                assert np.all(np.isfinite(obs))
                assert env.observation_space.contains(obs), (
                    np.flatnonzero((obs < env.observation_space.low)
                                   | (obs > env.observation_space.high))
                )
                if term or trunc:
                    break
        finally:
            env.close()


class TestWindColumn:
    """Centring, ordering, normalisation and the limit triple."""

    def test_centre_level_is_the_balloons_own_altitude(self, env_3d):
        env_3d.reset(seed=11)
        env_3d._balloon.pos[:] = [1_000.0, -2_000.0, 20_000.0]
        obs = env_3d._get_obs()
        col = wind_column(obs)

        fx, fy, _ = env_3d.wind.sample(1_000.0, -2_000.0, 20_000.0)
        expected = min(math.hypot(fx, fy) / WIND_MAG_NORM, 1.0)
        assert col[WIND_COL_CENTRE, CH_MAG] == pytest.approx(expected, abs=1e-6)

    def test_levels_are_ordered_low_to_high(self, env_3d):
        """Level i must be the wind at z + (i-20)*250, low to high."""
        env_3d.reset(seed=11)
        z = 20_000.0
        env_3d._balloon.pos[:] = [0.0, 0.0, z]
        col = wind_column(env_3d._get_obs())

        for i in range(WIND_COL_LEVELS):
            alt = level_altitude(z, i)
            if alt < ALT_SAFE_MIN or alt > ALT_SAFE_MAX:
                continue
            fx, fy, _ = env_3d.wind.sample(0.0, 0.0, alt)
            expected = min(math.hypot(fx, fy) / WIND_MAG_NORM, 1.0)
            assert col[i, CH_MAG] == pytest.approx(expected, abs=1e-6), f"level {i}"

    def test_uncertainty_channel_is_a_stub(self, env_any_dim):
        env, _ = env_any_dim
        obs, _ = env.reset(seed=5)
        col = wind_column(obs)
        assert np.all(col[:, CH_UNCERTAINTY] == 0.0)
        for _ in range(10):
            col = wind_column(env.step(2)[0])
            assert np.all(col[:, CH_UNCERTAINTY] == 0.0)

    def test_levels_below_the_band_carry_the_limit_triple(self, env_3d):
        env_3d.reset(seed=11)
        z = ALT_SAFE_MIN + 100.0        # levels 0..19 fall below the band
        env_3d._balloon.pos[:] = [0.0, 0.0, z]
        col = wind_column(env_3d._get_obs())

        for i in range(WIND_COL_CENTRE):
            assert level_altitude(z, i) < ALT_SAFE_MIN
            assert tuple(col[i]) == pytest.approx(LIMIT_TRIPLE), f"level {i}"
        assert tuple(col[WIND_COL_CENTRE]) != pytest.approx(LIMIT_TRIPLE)

    def test_levels_above_the_band_carry_the_limit_triple(self, env_3d):
        env_3d.reset(seed=11)
        z = ALT_SAFE_MAX - 100.0        # levels 21..40 rise above the band
        env_3d._balloon.pos[:] = [0.0, 0.0, z]
        col = wind_column(env_3d._get_obs())

        for i in range(WIND_COL_CENTRE + 1, WIND_COL_LEVELS):
            assert level_altitude(z, i) > ALT_SAFE_MAX
            assert tuple(col[i]) == pytest.approx(LIMIT_TRIPLE), f"level {i}"

    def test_band_edges_are_inclusive(self, env_3d):
        """At mid-band the column exactly spans the band; no level is masked."""
        env_3d.reset(seed=11)
        z = 0.5 * (ALT_SAFE_MIN + ALT_SAFE_MAX)
        env_3d._balloon.pos[:] = [0.0, 0.0, z]
        assert level_altitude(z, 0) == ALT_SAFE_MIN
        assert level_altitude(z, WIND_COL_LEVELS - 1) == ALT_SAFE_MAX

        col = wind_column(env_3d._get_obs())
        masked = bl.limit_mask(col)
        # A genuine (1, 1, 0) can only be a masked level here if the field
        # happens to saturate, which the test wind pattern never does.
        assert not masked.any()

    def test_limit_mask_agrees_with_the_baseline_reader(self, env_3d):
        env_3d.reset(seed=11)
        z = ALT_SAFE_MIN + 100.0
        env_3d._balloon.pos[:] = [0.0, 0.0, z]
        col = wind_column(env_3d._get_obs())
        assert bl.limit_mask(col)[:WIND_COL_CENTRE].all()
        assert not bl.limit_mask(col)[WIND_COL_CENTRE:].any()

    def test_magnitude_uses_the_fixed_normaliser(self, env_3d):
        """WIND_MAG_NORM is fixed at 30 m/s, independent of the field's mag."""
        env_3d.reset(seed=11)
        env_3d._balloon.pos[:] = [0.0, 0.0, 20_000.0]
        col = wind_column(env_3d._get_obs())
        mags = col[:, CH_MAG]
        assert np.all(mags >= 0.0) and np.all(mags <= 1.0)
        # The test field peaks well below the normaliser, so nothing saturates.
        assert mags.max() < 1.0

    def test_bearing_is_zero_when_the_wind_blows_at_the_goal(self):
        env = make_env(3, wind_pattern="linear_right", randomise_scenario=False)
        try:
            env.reset(seed=3)
            env._balloon.pos[:] = [0.0, 0.0, 20_000.0]
            env.goal = np.array([10_000.0, 0.0, env.z0])   # goal due +x, wind due +x
            col = wind_column(env._get_obs())
            assert col[WIND_COL_CENTRE, CH_BEARING] == pytest.approx(0.0, abs=1e-6)
        finally:
            env.close()

    def test_bearing_is_one_when_the_wind_blows_away(self):
        env = make_env(3, wind_pattern="linear_right", randomise_scenario=False)
        try:
            env.reset(seed=3)
            env._balloon.pos[:] = [0.0, 0.0, 20_000.0]
            env.goal = np.array([-10_000.0, 0.0, env.z0])  # goal due -x, wind due +x
            col = wind_column(env._get_obs())
            assert abs(col[WIND_COL_CENTRE, CH_BEARING]) == pytest.approx(1.0, abs=1e-6)
        finally:
            env.close()

    def test_bearing_sign_preserves_left_right(self):
        env = make_env(3, wind_pattern="linear_right", randomise_scenario=False)
        try:
            env.reset(seed=3)
            env._balloon.pos[:] = [0.0, 0.0, 20_000.0]

            env.goal = np.array([0.0, 10_000.0, env.z0])   # goal due +y
            right = wind_column(env._get_obs())[WIND_COL_CENTRE, CH_BEARING]
            env.goal = np.array([0.0, -10_000.0, env.z0])  # goal due -y
            left = wind_column(env._get_obs())[WIND_COL_CENTRE, CH_BEARING]

            assert right == pytest.approx(-0.5, abs=1e-6)
            assert left == pytest.approx(0.5, abs=1e-6)
        finally:
            env.close()

    def test_bearing_is_zero_in_1d(self, env_1d):
        """1D has no horizontal geometry, so a relative bearing is meaningless."""
        env_1d.reset(seed=5)
        for _ in range(5):
            col = wind_column(env_1d.step(1)[0])
            assert np.all(col[bl.limit_mask(col) == False, CH_BEARING] == 0.0)  # noqa: E712


class TestAmbientFields:
    """Content of indices 123..142."""

    def test_alt_norm_spans_the_operational_band(self, env_3d):
        env_3d.reset(seed=2)
        for z, expected in ((ALT_SAFE_MIN, 0.0), (20_000.0, 0.5), (ALT_SAFE_MAX, 1.0)):
            env_3d._balloon.pos[2] = z
            obs = env_3d._get_obs()
            assert obs[AMBIENT_IDX["alt_norm"]] == pytest.approx(expected, abs=1e-6)

    def test_goal_dz_is_meaningful_in_1d_and_zero_elsewhere(self, env_1d, env_3d):
        env_1d.reset(seed=2)
        env_1d._balloon.pos[0] = 18_000.0
        env_1d.goal = np.array([21_000.0])
        obs = env_1d._get_obs()
        assert obs[AMBIENT_IDX["goal_dz_norm"]] == pytest.approx(0.3, abs=1e-6)

        env_3d.reset(seed=2)
        assert env_3d._get_obs()[AMBIENT_IDX["goal_dz_norm"]] == 0.0

    def test_pressure_norm(self, env_3d):
        env_3d.reset(seed=2)
        env_3d._balloon.pos[2] = 20_000.0
        obs = env_3d._get_obs()
        expected = env_3d._atmosphere.pressure(20_000.0) / P_MAX
        assert obs[AMBIENT_IDX["pressure_norm"]] == pytest.approx(expected, abs=1e-6)
        assert 0.0 < obs[AMBIENT_IDX["pressure_norm"]] < 1.0

    def test_vel_z_norm_is_signed_and_clipped(self, env_3d):
        env_3d.reset(seed=2)
        env_3d._balloon.vel[2] = -0.5 * VEL_Z_OBS_NORM
        assert env_3d._get_obs()[AMBIENT_IDX["vel_z_norm"]] == pytest.approx(-0.5, abs=1e-6)
        env_3d._balloon.vel[2] = 10.0 * VEL_Z_OBS_NORM
        assert env_3d._get_obs()[AMBIENT_IDX["vel_z_norm"]] == 1.0

    def test_dist_norm_matches_the_reward_distance(self, env_3d):
        obs, info = env_3d.reset(seed=2)
        assert obs[AMBIENT_IDX["dist_norm"]] == pytest.approx(
            min(info["distance"] / DIST_NORM, 1.0), abs=1e-6)

    def test_dist_norm_saturates_rather_than_escaping_bounds(self, env_3d):
        env_3d.reset(seed=2)
        env_3d._balloon.pos[:2] = [10 * DIST_NORM, 0.0]
        assert env_3d._get_obs()[AMBIENT_IDX["dist_norm"]] == 1.0

    def test_heading_sin_cos(self, env_3d):
        env_3d.reset(seed=2)
        env_3d._balloon.pos[:] = [0.0, 0.0, 20_000.0]
        env_3d.goal = np.array([0.0, 5_000.0, env_3d.z0])   # due north
        obs = env_3d._get_obs()
        assert obs[AMBIENT_IDX["heading_sin"]] == pytest.approx(1.0, abs=1e-6)
        assert obs[AMBIENT_IDX["heading_cos"]] == pytest.approx(0.0, abs=1e-6)

    def test_heading_is_zero_in_1d(self, env_1d):
        obs, _ = env_1d.reset(seed=2)
        assert obs[AMBIENT_IDX["heading_sin"]] == 0.0
        assert obs[AMBIENT_IDX["heading_cos"]] == 0.0

    def test_zp_resources_are_ballast_and_gas(self, env_1d):
        env_1d.reset(seed=2)
        obs = env_1d._get_obs()
        assert obs[AMBIENT_IDX["resource_a"]] == pytest.approx(
            min(env_1d._balloon.ballast_mass / BALLAST_INITIAL, 1.0), abs=1e-6)
        assert obs[AMBIENT_IDX["resource_b"]] == pytest.approx(
            min(env_1d._balloon.n_gas / env_1d._init_n_gas, 1.0), abs=1e-6)

    def test_sp_resources_are_bladder_fill_and_headroom(self, env_sp_1d):
        env_sp_1d.reset(seed=2)
        obs = env_sp_1d._get_obs()
        fill = env_sp_1d._balloon.air_bladder_mass / AIR_BLADDER_MAX
        assert obs[AMBIENT_IDX["resource_a"]] == pytest.approx(fill, abs=1e-6)
        assert obs[AMBIENT_IDX["resource_b"]] == pytest.approx(1.0 - fill, abs=1e-6)

    def test_resource_low_flags(self, env_1d):
        env_1d.reset(seed=2)
        env_1d._balloon.ballast_mass = 0.04 * BALLAST_INITIAL
        obs = env_1d._get_obs()
        assert obs[AMBIENT_IDX["resource_a_low"]] == 1.0
        assert obs[AMBIENT_IDX["resource_b_low"]] == 0.0

    def test_volume_norm(self, env_1d):
        env_1d.reset(seed=2)
        obs = env_1d._get_obs()
        assert obs[AMBIENT_IDX["volume_norm"]] == pytest.approx(
            env_1d._balloon.volume / VOL_MAX, abs=1e-6)

    def test_sp_volume_norm_is_the_fixed_fraction(self, env_sp_1d):
        env_sp_1d.reset(seed=2)
        expected = SP_VOL_FIXED / VOL_MAX
        for _ in range(10):
            obs, _, term, trunc, _ = env_sp_1d.step(env_sp_1d.action_space.sample())
            assert obs[AMBIENT_IDX["volume_norm"]] == pytest.approx(expected, rel=1e-5)
            if term or trunc:
                break

    @pytest.mark.parametrize("action,field", [
        (0, "last_action_down"), (1, "last_action_stay"), (2, "last_action_up"),
    ])
    def test_last_action_one_hot(self, env_1d, action, field):
        env_1d.reset(seed=2)
        obs, _, _, _, _ = env_1d.step(action)
        one_hot = [obs[AMBIENT_IDX[n]] for n in
                   ("last_action_down", "last_action_stay", "last_action_up")]
        assert sum(one_hot) == 1.0
        assert obs[AMBIENT_IDX[field]] == 1.0

    def test_reset_reports_stay_as_the_last_action(self, env_1d):
        obs, _ = env_1d.reset(seed=2)
        assert obs[AMBIENT_IDX["last_action_stay"]] == 1.0

    def test_safety_flags_are_clear_at_spawn(self, env_any_dim):
        env, _ = env_any_dim
        obs, _ = env.reset(seed=2)
        assert obs[AMBIENT_IDX["at_alt_min"]] == 0.0
        assert obs[AMBIENT_IDX["at_alt_max"]] == 0.0

    def test_solar_fields_are_pinned_stubs(self, env_any_dim):
        """Layer 2 populates these; Layer 1 must keep them constant."""
        env, _ = env_any_dim
        obs, _ = env.reset(seed=2)
        for _ in range(20):
            assert obs[AMBIENT_IDX["solar_elevation"]] == SOLAR_ELEVATION_STUB
            assert obs[AMBIENT_IDX["solar_phase_sin"]] == SOLAR_PHASE_SIN_STUB
            assert obs[AMBIENT_IDX["solar_phase_cos"]] == SOLAR_PHASE_COS_STUB
            obs, _, term, trunc, _ = env.step(env.action_space.sample())
            if term or trunc:
                break


class TestObservationConsistency:
    """Buffer hygiene and reproducibility."""

    def test_returned_observation_is_a_copy(self, env_1d):
        env_1d.reset(seed=42)
        obs1, _, _, _, _ = env_1d.step(1)
        snapshot = obs1.copy()
        env_1d.step(1)
        assert np.array_equal(obs1, snapshot), "step() must not alias its own buffer"

    def test_observation_changes_with_step(self, env_any_dim):
        env, _ = env_any_dim
        obs1, _ = env.reset(seed=42)
        obs2, _, _, _, _ = env.step(2)
        assert not np.allclose(obs1, obs2)

    def test_observations_are_reproducible(self, env_1d):
        def rollout():
            env_1d.reset(seed=42)
            return [env_1d.step(1)[0].copy() for _ in range(5)]

        for a, b in zip(rollout(), rollout()):
            assert np.array_equal(a, b)

    def test_sp_and_zp_observations_are_the_same_shape(self):
        """The network is reusable across balloon types by construction."""
        for dim in (1, 2, 3):
            zp = make_env(dim)
            sp = make_env(dim, balloon_type="superpressure")
            try:
                assert zp.reset(seed=1)[0].shape == sp.reset(seed=1)[0].shape == (OBS_WIDTH,)
                assert zp.observation_space == sp.observation_space
            finally:
                zp.close()
                sp.close()


class TestFullCoordsHelper:
    """`_full_coords` pads to (x, y, z) whatever the dimension."""

    def test_full_coords_1d(self, env_1d):
        env_1d.reset(seed=42)
        env_1d._balloon.pos[0] = 19_000.0
        assert env_1d._full_coords(env_1d._balloon.pos) == (0.0, 0.0, 19_000.0)

    def test_full_coords_2d(self, env_2d):
        env_2d.reset(seed=42)
        env_2d._balloon.pos[:2] = [500.0, -300.0]
        assert env_2d._full_coords(env_2d._balloon.pos) == (500.0, -300.0, env_2d.z0)

    def test_full_coords_3d(self, env_3d):
        env_3d.reset(seed=42)
        env_3d._balloon.pos[:] = [100.0, 200.0, 19_000.0]
        assert env_3d._full_coords(env_3d._balloon.pos) == (100.0, 200.0, 19_000.0)


def test_observation_space_is_shared_by_gym_make():
    """gym.make must expose the same frozen space (no wrapper surprises)."""
    import gymnasium as gym

    env = gym.make("environments/Balloon3D-v0", dim=3, disable_env_checker=True,
                   config={"time_max": 10})
    try:
        assert env.observation_space.shape == (OBS_WIDTH,)
        obs, _ = env.reset(seed=1)
        assert env.observation_space.contains(obs)
    finally:
        env.close()


def test_sp_env_class_shares_the_layout():
    env = Balloon3DEnv(dim=3, config={"time_max": 10, "balloon_type": "superpressure"})
    try:
        assert env.observation_space.shape == (OBS_WIDTH,)
    finally:
        env.close()
