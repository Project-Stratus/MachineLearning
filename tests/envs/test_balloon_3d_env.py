"""Tests for Balloon3DEnv — the main Gymnasium environment.

Layer 1 changed the environment's shape in four ways that these tests pin:

* **No altitude terminations.** The safety layer clamps into
  ``[ALT_SAFE_MIN, ALT_SAFE_MAX]`` and raises an observation flag; crash and
  pop are gone (roadmap §3.4).
* **Soft horizontal bounds.** Leaving the old ±50 km box is merely bad, not
  fatal, so recovery behaviour is learnable (roadmap §3.2). ``XY_ABORT`` is a
  numerical guard, nothing more.
* **Resource rationing that survives the decision interval** (roadmap §3.3).
* **Scenario randomisation** so the held-out evaluation set means something
  (roadmap §3.9).
"""

import numpy as np
import pytest

from environments.core.balloon import BalloonSP
from environments.core.constants import (
    AIR_BLADDER_MAX,
    ALT_SAFE_MAX,
    ALT_SAFE_MIN,
    BALLAST_DROP,
    BALLAST_INITIAL,
    DECISION_INTERVAL,
    MIN_START_DISTANCE,
    RESOURCE_PENALTY_BASE,
    RESOURCE_PENALTY_SLOPE,
    SP_VOL_FIXED,
    TIME_MAX,
    WIND_COL_LEVELS,
    WIND_COL_SPACING,
    XY_ABORT,
    XY_MAX,
)
from environments.core.reward import l2_distance
from environments.envs.balloon_3d_env import (
    AMBIENT_IDX, WIND_COL_CHANNELS, WIND_COL_WIDTH, Actions, Balloon3DEnv,
)
from tests.conftest import expected_obs_size, make_env


class TestBalloon3DEnvInitialization:
    """Tests for environment initialization."""

    def test_initialization_all_dims(self, env_any_dim):
        env, dim = env_any_dim
        assert env.dim == dim
        assert env.observation_space.shape == (expected_obs_size(dim),)
        assert env.action_space.n == 3

    def test_invalid_dimension_raises(self):
        with pytest.raises(AssertionError):
            Balloon3DEnv(dim=4)

    def test_default_episode_is_twelve_hours(self):
        """time_max 86,400 (24 h) -> TIME_MAX 43,200 (12 h), our mission length."""
        assert Balloon3DEnv.DEFAULTS["time_max"] == TIME_MAX == 43_200

    def test_default_decision_interval(self):
        assert Balloon3DEnv.DEFAULTS["decision_interval"] == DECISION_INTERVAL

    def test_float_altitude_is_the_band_midpoint(self, env_3d):
        assert env_3d.z0 == pytest.approx(0.5 * (ALT_SAFE_MIN + ALT_SAFE_MAX))

    def test_config_override(self):
        env = Balloon3DEnv(dim=1, config={"time_max": 500, "wind_mag": 3.0})
        try:
            assert env.cfg["time_max"] == 500
            assert env.cfg["wind_mag"] == 3.0
        finally:
            env.close()

    def test_render_mode_none(self, env_1d):
        assert env_1d.render_mode is None

    def test_normalization_arrays_setup(self, env_any_dim):
        env, dim = env_any_dim
        assert env._norm_offsets.shape == (dim,)
        assert env._norm_scales.shape == (dim,)
        assert len(env._ranges) == dim


class TestWindColumnResolution:
    """The env must build a wind field the 41-level column can actually read.

    ``wind_cells=20`` on a cubic grid puts 1.5 km between vertical samples, so
    a column advertising 250 m spacing would hand the agent seven distinct
    winds dressed up as forty-one — the above/below signal it is supposed to
    learn from, blunted at source.
    """

    def test_vertical_grid_is_finer_than_the_column(self, env_3d):
        assert env_3d.wind.dz <= 0.5 * WIND_COL_SPACING
        assert env_3d.wind.cells_z > env_3d.wind.cells

    def test_horizontal_grid_still_follows_wind_cells(self, env_3d):
        assert env_3d.wind.cells == env_3d.cfg["wind_cells"]
        assert env_3d.wind._fx_grid.shape == (
            env_3d.wind.cells, env_3d.wind.cells, env_3d.wind.cells_z)

    def test_vertical_resolution_is_overridable(self):
        env = make_env(3, wind_cells_z=31)
        try:
            assert env.wind.cells_z == 31
        finally:
            env.close()

    def test_observed_column_levels_are_distinct(self, env_3d):
        """The payoff: adjacent levels of the *observation* carry distinct wind.

        ``altitude_shear_2d`` varies with altitude alone, and the env is placed
        mid-band so every level is inside ``[ALT_SAFE_MIN, ALT_SAFE_MAX]`` and
        therefore a real sample rather than the limit triple.
        """
        assert env_3d.cfg["wind_pattern"] == "altitude_shear_2d"
        obs, _ = env_3d.reset(seed=3)
        env_3d._balloon.pos[-1] = 0.5 * (ALT_SAFE_MIN + ALT_SAFE_MAX)
        obs, _, _, _, _ = env_3d.step(1)

        col = obs[:WIND_COL_WIDTH].reshape(WIND_COL_LEVELS, WIND_COL_CHANNELS)
        mag_bearing = col[:, :2]
        deltas = np.abs(np.diff(mag_bearing, axis=0)).sum(axis=1)
        assert np.all(deltas > 1e-6), \
            "adjacent column levels share a wind cell — vertical grid too coarse"


class TestBalloon3DEnvReset:
    """Tests for environment reset."""

    def test_reset_returns_observation_and_info(self, env_any_dim):
        env, dim = env_any_dim
        obs, info = env.reset(seed=42)
        assert obs.shape == (expected_obs_size(dim),)
        assert isinstance(info, dict)

    def test_reset_observation_finite_and_in_bounds(self, env_any_dim):
        env, _ = env_any_dim
        obs, _ = env.reset(seed=42)
        assert np.all(np.isfinite(obs))
        assert env.observation_space.contains(obs)

    def test_reset_info_structure(self, env_any_dim):
        env, _ = env_any_dim
        _, info = env.reset(seed=42)
        assert info["TimeLimit.truncated"] is False
        assert "terminal_observation" in info
        assert "distance" in info
        assert "scenario" in info

    def test_reset_initializes_goal_and_balloon(self, env_any_dim):
        env, dim = env_any_dim
        env.reset(seed=42)
        assert env.goal is not None and env.goal.shape == (dim,)
        assert env._balloon is not None
        assert env._time == 0

    def test_reset_clears_safety_flags_and_resource_hold(self, env_1d):
        env_1d.reset(seed=42)
        env_1d.step(2)                      # start a resource hold
        assert env_1d._omega_steps > 0
        env_1d.reset(seed=42)
        assert env_1d._omega_steps == 0
        assert env_1d._omega == 0.0
        assert env_1d._at_alt_min is False and env_1d._at_alt_max is False

    def test_reset_deterministic_with_seed(self, env_1d):
        obs1, _ = env_1d.reset(seed=123)
        obs2, _ = env_1d.reset(seed=123)
        assert np.array_equal(obs1, obs2)

    def test_reset_different_with_different_seed(self, env_1d):
        obs1, _ = env_1d.reset(seed=123)
        obs2, _ = env_1d.reset(seed=456)
        assert not np.allclose(obs1, obs2)

    def test_spawn_and_goal_are_far_enough_apart(self, env_any_dim):
        env, dim = env_any_dim
        for seed in range(10):
            env.reset(seed=seed)
            assert l2_distance(env._balloon.pos, env.goal, dim) > MIN_START_DISTANCE

    def test_spawn_altitude_is_inside_the_operational_band(self, env_any_dim):
        env, _ = env_any_dim
        for seed in range(10):
            env.reset(seed=seed)
            assert ALT_SAFE_MIN <= env._balloon.pos[-1] <= ALT_SAFE_MAX


class TestScenarioRandomisation:
    """Roadmap §3.9 — one scenario repeated is not a training distribution."""

    def _scenarios(self, env, seeds):
        out = []
        for s in seeds:
            _, info = env.reset(seed=s)
            out.append(dict(info["scenario"]))
        return out

    def test_scenario_reports_what_was_drawn(self, env_3d):
        _, info = env_3d.reset(seed=99)
        sc = info["scenario"]
        assert sc["seed"] == 99
        assert sc["dim"] == 3
        assert sc["balloon_type"] == "zero_pressure"
        assert sc["wind_pattern"] == env_3d.cfg["wind_pattern"]
        assert sc["wind_mag"] == pytest.approx(env_3d.wind.mag)
        assert sc["wind_layers"] == pytest.approx(env_3d.wind.wind_layers)
        assert sc["goal"] == pytest.approx(tuple(env_3d.goal))
        assert len(sc["spawn"]) == 3

    def test_scenario_is_carried_on_every_step(self, env_any_dim):
        env, _ = env_any_dim
        _, reset_info = env.reset(seed=7)
        for _ in range(5):
            _, _, _, _, info = env.step(1)
            assert info["scenario"] == reset_info["scenario"]

    def test_goal_varies_across_seeds(self, env_3d):
        goals = {s["goal"] for s in self._scenarios(env_3d, range(10))}
        assert len(goals) == 10, "goal must not be pinned to the origin"

    def test_spawn_varies_across_seeds(self, env_3d):
        spawns = {s["spawn"] for s in self._scenarios(env_3d, range(10))}
        assert len(spawns) == 10

    def test_wind_magnitude_varies_across_seeds(self, env_3d):
        mags = {round(s["wind_mag"], 9) for s in self._scenarios(env_3d, range(10))}
        assert len(mags) == 10

    def test_wind_layers_vary_for_altitude_shear_2d(self, env_3d):
        assert env_3d.cfg["wind_pattern"] == "altitude_shear_2d"
        layers = {round(s["wind_layers"], 9) for s in self._scenarios(env_3d, range(10))}
        assert len(layers) == 10

    def test_wind_layers_fixed_for_other_patterns(self):
        env = make_env(3, wind_pattern="split_fork")
        try:
            layers = {s["wind_layers"] for s in self._scenarios(env, range(5))}
            assert layers == {float(env.cfg["wind_layers"])}
        finally:
            env.close()

    def test_wind_field_is_rebuilt_not_just_recorded(self, env_3d):
        env_3d.reset(seed=1)
        grid_a = env_3d.wind._fx_grid.copy()
        env_3d.reset(seed=2)
        assert not np.allclose(grid_a, env_3d.wind._fx_grid)

    def test_same_seed_replays_the_whole_episode(self):
        """Evaluation pins scenarios by seed alone — this is what makes that work."""
        actions = [0, 1, 2, 1, 0, 2, 1, 1, 0, 2]

        def rollout(seed):
            env = make_env(3)
            try:
                obs, info = env.reset(seed=seed)
                trace = [obs.copy()]
                rewards = []
                for a in actions:
                    obs, r, _, _, _ = env.step(a)
                    trace.append(obs.copy())
                    rewards.append(r)
                return dict(info["scenario"]), trace, rewards
            finally:
                env.close()

        sc_a, trace_a, rew_a = rollout(2024)
        sc_b, trace_b, rew_b = rollout(2024)
        assert sc_a == sc_b
        for a, b in zip(trace_a, trace_b):
            assert np.array_equal(a, b)
        assert rew_a == rew_b

    def test_randomisation_can_be_disabled(self):
        env = make_env(3, randomise_scenario=False)
        try:
            scenarios = self._scenarios(env, range(5))
            assert len({s["goal"] for s in scenarios}) == 1
            assert len({s["spawn"] for s in scenarios}) == 1
            assert len({s["wind_mag"] for s in scenarios}) == 1
        finally:
            env.close()


class TestBalloon3DEnvStep:
    """Tests for environment step."""

    def test_step_returns_five_tuple(self, env_any_dim):
        env, _ = env_any_dim
        env.reset(seed=42)
        assert len(env.step(1)) == 5

    def test_step_observation_valid(self, env_any_dim):
        env, dim = env_any_dim
        env.reset(seed=42)
        obs, _, _, _, _ = env.step(1)
        assert obs.shape == (expected_obs_size(dim),)
        assert np.all(np.isfinite(obs))
        assert env.observation_space.contains(obs)

    def test_step_flags_are_bools(self, env_any_dim):
        env, _ = env_any_dim
        env.reset(seed=42)
        _, _, term, trunc, _ = env.step(1)
        assert isinstance(term, (bool, np.bool_))
        assert isinstance(trunc, (bool, np.bool_))

    def test_step_reward_in_range(self, env_any_dim):
        env, _ = env_any_dim
        env.reset(seed=42)
        _, reward, _, _, _ = env.step(1)
        assert np.isfinite(reward)
        assert 0.0 <= reward <= 1.0

    def test_step_time_increments(self, env_any_dim):
        env, _ = env_any_dim
        env.reset(seed=42)
        for expected in range(1, 6):
            env.step(1)
            assert env._time == expected


class TestInfoDict:
    """`info` is the evaluation interface (contract §5)."""

    def test_distance_present_on_every_step(self, env_any_dim):
        env, _ = env_any_dim
        _, info = env.reset(seed=42)
        assert "distance" in info
        for _ in range(20):
            _, _, term, trunc, info = env.step(env.action_space.sample())
            assert "distance" in info, "TWR evaluation reads info['distance'] every step"
            assert np.isfinite(info["distance"])
            if term or trunc:
                break

    def test_distance_is_the_reward_distance(self, env_any_dim):
        env, dim = env_any_dim
        env.reset(seed=42)
        for _ in range(5):
            _, _, _, _, info = env.step(1)
            expected = l2_distance(env._balloon.pos, env.goal, dim)
            assert info["distance"] == pytest.approx(expected)

    def test_reward_components(self, env_any_dim):
        env, _ = env_any_dim
        env.reset(seed=42)
        _, reward, _, _, info = env.step(1)
        c = info["reward_components"]
        assert set(c) == {"station", "decay", "base", "resource_factor", "total"}
        assert c["base"] == pytest.approx(c["station"] + c["decay"])
        assert c["total"] == pytest.approx(c["base"] * c["resource_factor"])
        assert reward == pytest.approx(c["total"])

    def test_termination_reason_on_truncation(self, env_short_episode):
        env_short_episode.reset(seed=42)
        for _ in range(20):
            _, _, _, truncated, info = env_short_episode.step(1)
            if truncated:
                assert info["termination_reason"] == "All timesteps completed"
                assert info["terminal_observation"] is not None
                break
        else:
            pytest.fail("episode never truncated")


class TestBalloon3DEnvActions:
    """Tests for action effects."""

    def test_action_lut_mapping(self, env_any_dim):
        env, _ = env_any_dim
        assert env._action_lut[0] == -1  # vent / pump in  -> descend
        assert env._action_lut[1] == 0   # nothing
        assert env._action_lut[2] == 1   # drop ballast / pump out -> ascend

    def test_drop_ballast_reduces_mass(self, env_1d):
        env_1d.reset(seed=42)
        before = env_1d._balloon.mass
        env_1d.step(2)
        assert env_1d._balloon.mass < before

    def test_vent_decreases_volume(self, env_1d):
        env_1d.reset(seed=42)
        before = env_1d._balloon.volume
        env_1d.step(0)
        assert env_1d._balloon.volume < before

    def test_nothing_changes_no_mass(self, env_1d):
        env_1d.reset(seed=42)
        before = env_1d._balloon.mass
        env_1d.step(1)
        assert env_1d._balloon.mass == pytest.approx(before)

    def test_all_actions_valid(self, env_any_dim):
        env, _ = env_any_dim
        for action in range(3):
            env.reset(seed=42)
            obs, reward, _, _, _ = env.step(action)
            assert np.all(np.isfinite(obs))
            assert np.isfinite(reward)


class TestAltitudeSafetyLayer:
    """Roadmap §3.4 — clamp into the band, flag it, never die of it."""

    @pytest.mark.parametrize("dim", [1, 3])
    def test_ceiling_clamps_and_flags_without_terminating(self, dim):
        env = make_env(dim)
        try:
            env.reset(seed=1)
            env._balloon.pos[-1] = ALT_SAFE_MAX + 500.0
            env._balloon.vel[-1] = 5.0
            obs, reward, terminated, _, _ = env.step(1)

            assert not terminated, "hitting the ceiling must not end the episode"
            assert env._balloon.pos[-1] == pytest.approx(ALT_SAFE_MAX)
            assert env._balloon.vel[-1] == 0.0
            assert obs[AMBIENT_IDX["at_alt_max"]] == 1.0
            assert obs[AMBIENT_IDX["at_alt_min"]] == 0.0
            assert reward > 0.0, "still on station, still earning"
        finally:
            env.close()

    @pytest.mark.parametrize("dim", [1, 3])
    def test_floor_clamps_and_flags_without_terminating(self, dim):
        env = make_env(dim)
        try:
            env.reset(seed=1)
            env._balloon.pos[-1] = ALT_SAFE_MIN - 500.0
            env._balloon.vel[-1] = -5.0
            obs, _, terminated, _, _ = env.step(1)

            assert not terminated
            assert env._balloon.pos[-1] == pytest.approx(ALT_SAFE_MIN)
            assert env._balloon.vel[-1] == 0.0
            assert obs[AMBIENT_IDX["at_alt_min"]] == 1.0
            assert obs[AMBIENT_IDX["at_alt_max"]] == 0.0
        finally:
            env.close()

    def test_ground_no_longer_crashes(self, env_1d):
        """The old 'altitude <= 0 -> terminate' path is gone entirely."""
        env_1d.reset(seed=1)
        env_1d._balloon.pos[-1] = 10.0
        env_1d._balloon.vel[-1] = -50.0
        _, _, terminated, _, info = env_1d.step(1)
        assert not terminated
        assert "crash" not in str(info.get("termination_reason", "")).lower()
        assert env_1d._balloon.pos[-1] == pytest.approx(ALT_SAFE_MIN)

    def test_ceiling_no_longer_pops(self, env_1d):
        env_1d.reset(seed=1)
        env_1d._balloon.pos[-1] = env_1d.z_range[1] + 5_000.0
        _, _, terminated, _, info = env_1d.step(1)
        assert not terminated
        assert "pop" not in str(info.get("termination_reason", "")).lower()
        assert env_1d._balloon.pos[-1] == pytest.approx(ALT_SAFE_MAX)

    def test_altitude_stays_in_band_under_persistent_ascent(self):
        """Spamming 'up' presses the balloon against the ceiling; it stays there."""
        env = make_env(1, time_max=400)
        try:
            env.reset(seed=1)
            pressed = 0
            for _ in range(400):
                obs, _, terminated, truncated, _ = env.step(2)
                assert ALT_SAFE_MIN <= env._balloon.pos[-1] <= ALT_SAFE_MAX
                if obs[AMBIENT_IDX["at_alt_max"]] == 1.0:
                    pressed += 1
                if terminated or truncated:
                    break
            assert pressed > 0, "400 ballast drops should reach the ceiling"
        finally:
            env.close()

    def test_flag_stays_raised_while_pressed_against_the_limit(self, env_1d):
        """Not a one-step blip: the constraint is observable while it binds."""
        env_1d.reset(seed=1)
        env_1d._balloon.pos[-1] = ALT_SAFE_MAX
        env_1d._balloon.vel[-1] = 0.0
        raised = 0
        for _ in range(30):
            obs, _, _, _, _ = env_1d.step(2)   # keep pushing up
            raised += int(obs[AMBIENT_IDX["at_alt_max"]] == 1.0)
        assert raised >= 25, f"flag raised on only {raised}/30 steps while pressed"

    def test_2d_altitude_is_pinned_and_never_flags(self, env_2d):
        env_2d.reset(seed=1)
        for _ in range(20):
            obs, _, _, _, _ = env_2d.step(env_2d.action_space.sample())
            assert env_2d._balloon.pos[2] == pytest.approx(env_2d.z0)
            assert env_2d._balloon.vel[2] == 0.0
            assert obs[AMBIENT_IDX["at_alt_min"]] == 0.0
            assert obs[AMBIENT_IDX["at_alt_max"]] == 0.0


class TestSoftHorizontalBounds:
    """Roadmap §3.2 — terminating walls make recovery unlearnable."""

    @pytest.mark.parametrize("dim", [2, 3])
    def test_leaving_the_old_50km_box_does_not_terminate(self, dim):
        env = make_env(dim)
        try:
            env.reset(seed=1)
            env._balloon.pos[0] = 3.0 * XY_MAX     # 150 km out
            env._balloon.pos[1] = -3.0 * XY_MAX
            obs, reward, terminated, _, info = env.step(1)

            assert not terminated, "the ±50 km box must no longer be fatal"
            assert info["distance"] > XY_MAX
            assert reward >= 0.0
            assert env.observation_space.contains(obs)
        finally:
            env.close()

    @pytest.mark.parametrize("dim", [2, 3])
    def test_far_drift_still_pays_reward(self, dim):
        """Reward decays with distance but never stops — that is the gradient home."""
        env = make_env(dim)
        try:
            env.reset(seed=1)
            env._balloon.pos[:2] = [80_000.0, 0.0]
            _, reward, _, _, _ = env.step(1)
            assert reward > 0.0
        finally:
            env.close()

    @pytest.mark.parametrize("dim", [2, 3])
    def test_xy_abort_is_a_numerical_guard(self, dim):
        env = make_env(dim)
        try:
            env.reset(seed=1)
            env._balloon.pos[0] = XY_ABORT + 1_000.0
            _, reward, terminated, _, info = env.step(1)
            assert terminated
            assert reward == 0.0
            assert "XY_ABORT" in info["termination_reason"]
        finally:
            env.close()

    def test_1d_has_no_horizontal_abort(self, env_1d):
        env_1d.reset(seed=1)
        for _ in range(10):
            _, _, terminated, _, _ = env_1d.step(1)
            assert not terminated


class TestResourceAccounting:
    """Roadmap §3.3 — the penalty must survive the decision interval."""

    def test_noop_costs_nothing(self, env_1d):
        env_1d.reset(seed=1)
        _, _, _, _, info = env_1d.step(1)
        assert info["resource_consumed_frac"] == 0.0
        assert info["reward_components"]["resource_factor"] == 1.0

    def test_omega_is_the_fraction_of_the_initial_budget(self, env_1d):
        env_1d.reset(seed=1)
        _, _, _, _, info = env_1d.step(2)      # drop ballast
        assert info["resource_consumed_frac"] == pytest.approx(BALLAST_DROP / BALLAST_INITIAL)

        env_1d.reset(seed=1)
        n0 = env_1d._balloon.n_gas
        _, _, _, _, info = env_1d.step(0)      # vent gas
        expected = (n0 - env_1d._balloon.n_gas) / env_1d._init_n_gas
        assert info["resource_consumed_frac"] == pytest.approx(expected)
        assert 0.0 < expected < 1.0

    def test_actuating_multiplies_the_reward_down(self, env_1d):
        env_1d.reset(seed=1)
        _, _, _, _, info = env_1d.step(2)
        omega = info["resource_consumed_frac"]
        expected = RESOURCE_PENALTY_BASE - RESOURCE_PENALTY_SLOPE * omega
        assert info["reward_components"]["resource_factor"] == pytest.approx(expected)
        assert expected < 1.0

    def test_penalty_is_held_for_a_whole_decision_interval(self, env_1d):
        """The subtle one.

        The wrapper fires the real action on sub-step 0 and NOOPs for the other
        59. If the penalty were charged only on the firing step it would be
        diluted 60x — a 3% cost becoming 0.05% — and would shape nothing. The
        env therefore holds omega for the whole interval.
        """
        env_1d.reset(seed=1)
        factors = []
        for i in range(DECISION_INTERVAL + 5):
            _, _, _, _, info = env_1d.step(2 if i == 0 else 1)
            factors.append(info["reward_components"]["resource_factor"])

        charged = factors[:DECISION_INTERVAL]
        assert all(f < 1.0 for f in charged), "penalty diluted across the interval"
        assert len(set(charged)) == 1, "the same omega must apply to every sub-step"
        assert all(f == 1.0 for f in factors[DECISION_INTERVAL:]), "hold must expire"

    def test_hold_restarts_on_a_fresh_actuation(self, env_1d):
        env_1d.reset(seed=1)
        env_1d.step(2)
        for _ in range(DECISION_INTERVAL // 2):
            env_1d.step(1)
        env_1d.step(2)                       # re-arm mid-hold
        for _ in range(DECISION_INTERVAL - 1):
            _, _, _, _, info = env_1d.step(1)
            assert info["reward_components"]["resource_factor"] < 1.0
        _, _, _, _, info = env_1d.step(1)
        assert info["reward_components"]["resource_factor"] == 1.0

    def test_interval_return_is_penalised_by_the_full_factor(self):
        """End-to-end check against the dilution failure mode.

        Two identical 2D episodes: one actuates on the first sub-step, the
        other does not. In 2D the altitude is pinned, so the action barely
        perturbs the trajectory and the summed returns are comparable. A
        correctly held penalty gives a ratio near RESOURCE_PENALTY_BASE
        (~0.97); a diluted one gives ~0.9995.
        """
        def interval_return(first_action):
            env = make_env(2)
            try:
                env.reset(seed=5)
                return sum(env.step(first_action if i == 0 else 1)[1]
                           for i in range(DECISION_INTERVAL))
            finally:
                env.close()

        ratio = interval_return(2) / interval_return(1)
        assert ratio == pytest.approx(RESOURCE_PENALTY_BASE, abs=0.01)
        assert ratio < 0.99, "resource penalty is being diluted across the interval"

    def test_an_action_that_consumes_nothing_is_free(self, env_sp_1d):
        """A pump against a full bladder moves no air, so it costs nothing."""
        env_sp_1d.reset(seed=1)
        env_sp_1d._balloon.air_bladder_mass = AIR_BLADDER_MAX
        _, _, _, _, info = env_sp_1d.step(0)   # pump in — already full
        assert info["resource_consumed_frac"] == 0.0
        assert info["reward_components"]["resource_factor"] == 1.0

    def test_sp_omega_is_bladder_fraction(self, env_sp_1d):
        env_sp_1d.reset(seed=1)
        before = env_sp_1d._balloon.air_bladder_mass
        _, _, _, _, info = env_sp_1d.step(0)
        moved = abs(env_sp_1d._balloon.air_bladder_mass - before)
        assert info["resource_consumed_frac"] == pytest.approx(moved / AIR_BLADDER_MAX)

    def test_set_decision_interval_changes_the_hold(self, env_1d):
        env_1d.reset(seed=1)
        env_1d.set_decision_interval(5)
        factors = []
        for i in range(8):
            _, _, _, _, info = env_1d.step(2 if i == 0 else 1)
            factors.append(info["reward_components"]["resource_factor"])
        assert all(f < 1.0 for f in factors[:5])
        assert all(f == 1.0 for f in factors[5:])


class TestTermination:
    """What is left that can end an episode early."""

    def test_zp_terminates_when_deflated(self, env_1d):
        env_1d.reset(seed=1)
        env_1d._balloon.n_gas = 0.0
        _, reward, terminated, _, info = env_1d.step(1)
        assert terminated
        assert reward == 0.0
        assert info["termination_reason"] == "Deflated (helium fully lost)"

    def test_zp_terminates_when_ballast_exhausted(self, env_1d):
        env_1d.reset(seed=1)
        env_1d._balloon.ballast_mass = 0.5 * BALLAST_DROP
        _, reward, terminated, _, info = env_1d.step(2)
        assert terminated
        assert reward == 0.0
        assert info["termination_reason"] == "Ballast exhausted (no ballast remaining)"

    def test_no_termination_in_a_nominal_episode(self, env_any_dim):
        env, _ = env_any_dim
        env.reset(seed=1)
        for _ in range(100):
            _, _, terminated, truncated, _ = env.step(1)
            assert not terminated
            if truncated:
                break

    def test_sp_never_deflates_or_runs_out(self, env_sp_1d):
        env_sp_1d.reset(seed=1)
        for action in (0, 2):
            env_sp_1d.reset(seed=1)
            for _ in range(100):
                _, _, terminated, truncated, info = env_sp_1d.step(action)
                assert not terminated
                if truncated:
                    break


class TestTruncation:
    """Tests for truncation (time limit)."""

    def test_truncation_on_time_limit(self, env_short_episode):
        env_short_episode.reset(seed=42)
        for _ in range(20):
            _, _, _, truncated, _ = env_short_episode.step(1)
            if truncated:
                break
        assert truncated

    def test_truncation_at_exact_time(self):
        env = Balloon3DEnv(dim=1, config={"time_max": 5})
        try:
            env.reset(seed=42)
            for i in range(5):
                _, _, _, truncated, _ = env.step(1)
                assert truncated == (i == 4)
        finally:
            env.close()

    def test_terminal_observation_on_truncation(self, env_short_episode):
        env_short_episode.reset(seed=42)
        for _ in range(20):
            _, _, _, truncated, info = env_short_episode.step(1)
            if truncated:
                assert info["terminal_observation"] is not None
                assert info["terminal_observation"].shape == (expected_obs_size(1),)
                break


class TestActionsEnum:
    """Tests for Actions enum."""

    def test_actions_enum_values(self):
        assert Actions.drop_ballast.value == 1
        assert Actions.nothing.value == 0
        assert Actions.vent.value == -1

    def test_actions_enum_lookup(self):
        assert Actions(1) == Actions.drop_ballast
        assert Actions(0) == Actions.nothing
        assert Actions(-1) == Actions.vent


class TestBalloon3DEnvSP:
    """Tests specific to the SP (superpressure + air ballast) balloon type."""

    def test_sp_reset_creates_balloon_sp(self, env_sp_1d):
        env_sp_1d.reset(seed=42)
        assert isinstance(env_sp_1d._balloon, BalloonSP)
        assert hasattr(env_sp_1d._balloon, "air_bladder_mass")

    def test_sp_action_pump_out_reduces_mass(self, env_sp_1d):
        env_sp_1d.reset(seed=42)
        before = env_sp_1d._balloon.mass
        env_sp_1d.step(2)
        assert env_sp_1d._balloon.mass < before

    def test_sp_action_pump_in_increases_mass(self, env_sp_1d):
        env_sp_1d.reset(seed=42)
        before = env_sp_1d._balloon.mass
        env_sp_1d.step(0)
        assert env_sp_1d._balloon.mass > before

    def test_sp_action_nothing_no_mass_change(self, env_sp_1d):
        env_sp_1d.reset(seed=42)
        before = env_sp_1d._balloon.mass
        env_sp_1d.step(1)
        assert env_sp_1d._balloon.mass == pytest.approx(before)

    def test_sp_volume_constant(self, env_sp_1d):
        env_sp_1d.reset(seed=42)
        for action in (0, 1, 2):
            env_sp_1d.step(action)
            assert env_sp_1d._balloon.volume == pytest.approx(SP_VOL_FIXED)

    def test_sp_action_lut_same_as_zp(self, env_sp_any_dim):
        env, _ = env_sp_any_dim
        assert list(env._action_lut) == [-1, 0, 1]

    def test_sp_all_actions_produce_valid_obs(self, env_sp_any_dim):
        env, dim = env_sp_any_dim
        for action in range(3):
            env.reset(seed=42)
            obs, reward, _, _, _ = env.step(action)
            assert obs.shape == (expected_obs_size(dim),)
            assert env.observation_space.contains(obs)
            assert np.isfinite(reward)

    def test_sp_altitude_safety_layer_applies(self, env_sp_1d):
        env_sp_1d.reset(seed=1)
        env_sp_1d._balloon.pos[-1] = ALT_SAFE_MAX + 500.0
        obs, _, terminated, _, _ = env_sp_1d.step(1)
        assert not terminated
        assert env_sp_1d._balloon.pos[-1] == pytest.approx(ALT_SAFE_MAX)
        assert obs[AMBIENT_IDX["at_alt_max"]] == 1.0
