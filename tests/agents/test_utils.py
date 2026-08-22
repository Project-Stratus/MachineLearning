"""Tests for agent utility functions."""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from agents.baselines import ACTION_DOWN, ACTION_STAY, ACTION_UP, OBS_WIDTH
from agents.utils import (
    _gather_monitor_csvs,
    ALT_SAFE_MAX,
    ALT_SAFE_MIN,
    IDX_ALT_NORM,
    InfoProgressBar,
    MomentumExplorer,
)


class TestGatherMonitorCSVs:
    """Tests for _gather_monitor_csvs function."""

    @pytest.fixture
    def mock_monitor_dir(self):
        """Create a temporary directory with mock monitor CSV files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock CSV files with Monitor format
            # Monitor CSVs have a comment header like:
            # #{"t_start": 0.0, "env_id": "Balloon3D-v0"}
            # r,l,t

            # File 1: 3 episodes
            with open(os.path.join(tmpdir, "train_monitor.csv"), "w") as f:
                f.write('#{"t_start": 0.0}\n')
                f.write("r,l,t\n")
                f.write("100.5,50,1.0\n")
                f.write("200.3,75,2.0\n")
                f.write("-50.0,30,3.0\n")

            # File 2: 2 episodes (from parallel env)
            with open(os.path.join(tmpdir, "train_monitor_1.csv"), "w") as f:
                f.write('#{"t_start": 0.0}\n')
                f.write("r,l,t\n")
                f.write("150.0,60,1.5\n")
                f.write("80.0,40,2.5\n")

            yield tmpdir

    @pytest.fixture
    def empty_monitor_dir(self):
        """Create an empty temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_gather_returns_dataframe(self, mock_monitor_dir):
        """Should return a pandas DataFrame."""
        df = _gather_monitor_csvs(mock_monitor_dir)
        assert isinstance(df, pd.DataFrame)

    def test_gather_correct_columns(self, mock_monitor_dir):
        """DataFrame should have expected columns."""
        df = _gather_monitor_csvs(mock_monitor_dir)
        expected_cols = ["episode_idx", "r", "l", "t", "source", "global_episode"]
        for col in expected_cols:
            assert col in df.columns

    def test_gather_correct_row_count(self, mock_monitor_dir):
        """Should have correct number of rows (all episodes from all files)."""
        df = _gather_monitor_csvs(mock_monitor_dir)
        # 3 episodes from first file + 2 from second = 5 total
        assert len(df) == 5

    def test_gather_episode_idx_per_file(self, mock_monitor_dir):
        """episode_idx should reset per file."""
        df = _gather_monitor_csvs(mock_monitor_dir)
        # Each file should have episode_idx starting from 1
        file1_episodes = df[df["source"] == "train_monitor.csv"]["episode_idx"]
        file2_episodes = df[df["source"] == "train_monitor_1.csv"]["episode_idx"]

        assert list(file1_episodes) == [1, 2, 3]
        assert list(file2_episodes) == [1, 2]

    def test_gather_global_episode_sequential(self, mock_monitor_dir):
        """global_episode should be sequential across all episodes."""
        df = _gather_monitor_csvs(mock_monitor_dir)
        assert list(df["global_episode"]) == [1, 2, 3, 4, 5]

    def test_gather_source_tracking(self, mock_monitor_dir):
        """source column should track which file each row came from."""
        df = _gather_monitor_csvs(mock_monitor_dir)
        sources = df["source"].unique()
        assert "train_monitor.csv" in sources
        assert "train_monitor_1.csv" in sources

    def test_gather_reward_values(self, mock_monitor_dir):
        """Should correctly parse reward values."""
        df = _gather_monitor_csvs(mock_monitor_dir)
        rewards = df["r"].tolist()
        assert 100.5 in rewards
        assert 200.3 in rewards
        assert -50.0 in rewards
        assert 150.0 in rewards
        assert 80.0 in rewards

    def test_gather_empty_directory(self, empty_monitor_dir):
        """Should return empty DataFrame for empty directory."""
        df = _gather_monitor_csvs(empty_monitor_dir)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        # Should still have correct columns
        assert "episode_idx" in df.columns
        assert "r" in df.columns

    def test_gather_no_matching_files(self, empty_monitor_dir):
        """Should return empty DataFrame when no train_monitor*.csv files exist."""
        # Create a file with different name
        with open(os.path.join(empty_monitor_dir, "other_file.csv"), "w") as f:
            f.write("r,l,t\n1,2,3\n")

        df = _gather_monitor_csvs(empty_monitor_dir)
        assert len(df) == 0

    def test_gather_handles_comment_lines(self, mock_monitor_dir):
        """Should correctly skip comment lines in CSV files."""
        # The mock files have comment lines - if they're parsed incorrectly,
        # we'd get wrong data or errors
        df = _gather_monitor_csvs(mock_monitor_dir)
        # Should not have any NaN values from mis-parsing
        assert not df["r"].isna().any()
        assert not df["l"].isna().any()
        assert not df["t"].isna().any()


class TestInfoProgressBar:
    """Tests for InfoProgressBar callback class."""

    def test_info_progress_bar_initialization(self):
        """InfoProgressBar should initialize with description and postfix."""
        bar = InfoProgressBar(
            description="Test Progress",
            postfix={"gamma": 0.99, "lr": 0.001}
        )
        assert bar._description == "Test Progress"
        assert bar._postfix == {"gamma": 0.99, "lr": 0.001}

    def test_info_progress_bar_default_postfix(self):
        """InfoProgressBar should default to empty postfix if not provided."""
        bar = InfoProgressBar(description="Test")
        assert bar._postfix == {}

    def test_info_progress_bar_none_postfix(self):
        """InfoProgressBar should handle None postfix."""
        bar = InfoProgressBar(description="Test", postfix=None)
        assert bar._postfix == {}

    def test_info_progress_bar_inherits_from_progress_bar_callback(self):
        """InfoProgressBar should inherit from ProgressBarCallback."""
        from stable_baselines3.common.callbacks import ProgressBarCallback
        bar = InfoProgressBar(description="Test")
        assert isinstance(bar, ProgressBarCallback)

    def test_info_progress_bar_resolve_bar_method(self):
        """_resolve_bar should return None when no bar is set."""
        bar = InfoProgressBar(description="Test")
        # Before training, there's no progress bar
        assert bar._resolve_bar() is None

    def test_info_progress_bar_on_step_returns_true(self):
        """_on_step should return True to continue training."""
        bar = InfoProgressBar(description="Test")
        # _on_step calls parent which should return True
        # We can't easily test this without a full model, but we can verify
        # the method exists and is callable
        assert hasattr(bar, "_on_step")
        assert callable(bar._on_step)


# --------------------------------------------------------------------------- #
# Momentum exploration (roadmap §3.8)
# --------------------------------------------------------------------------- #
BAND = ALT_SAFE_MAX - ALT_SAFE_MIN


def _obs(altitudes) -> np.ndarray:
    """Minimal batched observation carrying only ``alt_norm``."""
    alts = np.atleast_1d(np.asarray(altitudes, dtype=np.float64))
    obs = np.zeros((alts.size, OBS_WIDTH), dtype=np.float32)
    obs[:, IDX_ALT_NORM] = (alts - ALT_SAFE_MIN) / BAND
    return obs


def _roll_toy_balloon(action_fn, n_steps: int = 600, rate: float = 50.0,
                      alt0: float = 20_000.0):
    """Drive a deliberately sluggish toy balloon and record what happened.

    ``rate`` metres per decision is the whole point of the exercise: one
    exploratory action barely moves the balloon, so only a *sustained* run of
    the same action changes its altitude appreciably. Returns
    ``(actions, altitudes)``.
    """
    alt = float(alt0)
    actions, altitudes = [], []
    for _ in range(n_steps):
        action = int(action_fn(_obs(alt)))
        alt = float(np.clip(alt + (action - ACTION_STAY) * rate, ALT_SAFE_MIN, ALT_SAFE_MAX))
        actions.append(action)
        altitudes.append(alt)
    return np.asarray(actions), np.asarray(altitudes)


def _persistence(actions: np.ndarray) -> float:
    """Fraction of consecutive action pairs that are equal. Uniform random -> 1/3."""
    return float(np.mean(actions[1:] == actions[:-1]))


class TestMomentumExplorerState:
    """Per-env state, bounds and the target random walk."""

    def test_targets_start_inside_the_band(self):
        ex = MomentumExplorer(seed=0)
        ex.act(_obs([20_000.0] * 16))
        assert ex.n_envs == 16
        assert np.all(ex.target_alt >= ALT_SAFE_MIN)
        assert np.all(ex.target_alt <= ALT_SAFE_MAX)

    def test_targets_stay_inside_the_band_under_perturbation(self):
        # Tiny band + huge sigma: every perturbation would escape if unclipped.
        ex = MomentumExplorer(alt_min=19_000.0, alt_max=21_000.0, perturb_every=1,
                              perturb_sigma=50_000.0, seed=1)
        for _ in range(200):
            ex.act(_obs([20_000.0, 20_000.0]))
            assert np.all(ex.target_alt >= 19_000.0)
            assert np.all(ex.target_alt <= 21_000.0)

    def test_target_is_held_then_perturbed(self):
        ex = MomentumExplorer(perturb_every=10, perturb_sigma=500.0, seed=2)
        ex.act(_obs([20_000.0]))
        ex.reset_envs()                      # resets the countdown to perturb_every
        held = ex.target_alt.copy()
        for _ in range(9):                   # 9 ticks: not yet due
            ex.act(_obs([20_000.0]))
            assert ex.target_alt == pytest.approx(held)
        ex.act(_obs([20_000.0]))             # 10th tick: perturbation lands
        assert ex.target_alt != pytest.approx(held)

    def test_per_env_targets_are_independent(self):
        ex = MomentumExplorer(seed=3)
        ex.act(_obs([20_000.0] * 8))
        assert len(np.unique(ex.target_alt)) == 8, "all 8 actors drew the same target"

    def test_reset_envs_only_touches_the_masked_envs(self):
        ex = MomentumExplorer(seed=4)
        ex.act(_obs([20_000.0] * 4))
        before = ex.target_alt.copy()
        ex.reset_envs(np.array([True, False, False, True]))
        assert ex.target_alt[0] != before[0]
        assert ex.target_alt[3] != before[3]
        np.testing.assert_array_equal(ex.target_alt[1:3], before[1:3])

    def test_resizes_when_n_envs_changes(self):
        ex = MomentumExplorer(seed=5)
        ex.act(_obs([20_000.0]))
        assert ex.n_envs == 1
        ex.act(_obs([20_000.0] * 5))
        assert ex.n_envs == 5 and ex.target_alt.shape == (5,)

    def test_seeded_explorers_are_reproducible(self):
        a = MomentumExplorer(seed=7).act(_obs([20_000.0] * 4))
        b = MomentumExplorer(seed=7).act(_obs([20_000.0] * 4))
        np.testing.assert_array_equal(a, b)

    @pytest.mark.parametrize("bad", [
        dict(alt_min=20_000.0, alt_max=19_000.0),
        dict(perturb_every=0),
        dict(perturb_sigma=-1.0),
        dict(deadband=-1.0),
    ])
    def test_rejects_nonsense_config(self, bad):
        with pytest.raises(ValueError):
            MomentumExplorer(**bad)


class TestMomentumExplorerActions:
    """The action actually emitted, given a target and a current altitude."""

    def test_commands_up_when_below_target_and_down_when_above(self):
        ex = MomentumExplorer(deadband=100.0, seed=8)
        ex.act(_obs([20_000.0, 20_000.0]))
        ex.target_alt[:] = [22_000.0, 18_000.0]
        actions = ex.act(_obs([20_000.0, 20_000.0]))
        assert actions[0] == ACTION_UP
        assert actions[1] == ACTION_DOWN

    def test_stays_inside_the_deadband(self):
        ex = MomentumExplorer(deadband=200.0, seed=9)
        ex.act(_obs([20_000.0]))
        ex.target_alt[:] = 20_000.0
        assert int(ex.act(_obs([20_050.0]))[0]) == ACTION_STAY

    def test_per_env_actions_differ_when_states_differ(self):
        ex = MomentumExplorer(perturb_every=10_000, deadband=100.0, seed=10)
        ex.act(_obs([20_000.0, 20_000.0, 20_000.0]))
        ex.target_alt[:] = 20_000.0
        actions = ex.act(_obs([16_000.0, 20_000.0, 24_000.0]))
        np.testing.assert_array_equal(actions, [ACTION_UP, ACTION_STAY, ACTION_DOWN])

    def test_accepts_a_single_unbatched_observation(self):
        ex = MomentumExplorer(seed=11)
        actions = ex.act(np.zeros(OBS_WIDTH, dtype=np.float32))
        assert actions.shape == (1,)


class TestMomentumExplorationIsCorrelated:
    """The reason momentum exploration exists (roadmap §3.8).

    Uniform ε-greedy is a zero-mean dither: consecutive actions are independent,
    so the balloon never travels far enough to see a different wind. These tests
    assert momentum exploration is *measurably* more persistent, and that the
    persistence translates into macroscopic altitude variation.
    """

    N_STEPS = 600

    def _uniform_stream(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        return _roll_toy_balloon(lambda obs: rng.integers(3), n_steps=self.N_STEPS)

    def _momentum_stream(self, seed: int = 0):
        ex = MomentumExplorer(seed=seed)
        return _roll_toy_balloon(lambda obs: ex.act(obs)[0], n_steps=self.N_STEPS)

    def test_actions_are_far_more_persistent_than_uniform_random(self):
        momentum_actions, _ = self._momentum_stream()
        uniform_actions, _ = self._uniform_stream()

        momentum_p = _persistence(momentum_actions)
        uniform_p = _persistence(uniform_actions)

        assert uniform_p == pytest.approx(1 / 3, abs=0.08), "uniform baseline drifted"
        assert momentum_p > 0.9, f"momentum exploration is not persistent ({momentum_p:.3f})"
        assert momentum_p > 2.0 * uniform_p

    def test_persistence_produces_macroscopic_altitude_variation(self):
        """Correlated actions must translate into the balloon actually moving.

        Averaged over seeds rather than asserted per seed: an unforced random
        walk occasionally wanders a long way, and the toy dynamics here are
        kinder to it than the real env is (no buoyant restoring force, no
        finite ballast, no altitude clamp doing most of the work). The claim
        being tested is about the *distribution*, so measure it that way.
        """
        momentum_span, uniform_span = [], []
        for seed in range(8):
            _, momentum_alts = self._momentum_stream(seed=seed)
            _, uniform_alts = self._uniform_stream(seed=seed)
            momentum_span.append(np.ptp(momentum_alts))
            uniform_span.append(np.ptp(uniform_alts))

        assert np.mean(momentum_span) > 2.0 * np.mean(uniform_span)
        assert np.mean(momentum_span) > 2_500.0, "momentum exploration barely moved the balloon"

    def test_holds_a_direction_for_many_consecutive_decisions(self):
        actions, _ = self._momentum_stream()
        # Longest run of one action, ignoring which.
        boundaries = np.flatnonzero(np.diff(actions)) + 1
        runs = np.diff(np.concatenate(([0], boundaries, [actions.size])))
        assert runs.max() > 50, f"longest action run was only {runs.max()} decisions"

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_persistence_holds_across_seeds(self, seed):
        actions, _ = self._momentum_stream(seed=seed)
        assert _persistence(actions) > 0.85


class TestInfoProgressBarIntegration:
    """Integration tests for InfoProgressBar with actual training."""

    @pytest.mark.slow
    def test_info_progress_bar_with_short_training(self):
        """InfoProgressBar should work with actual (short) training."""
        pytest.importorskip("stable_baselines3")

        import torch
        from sb3_contrib import QRDQN
        from stable_baselines3.common.callbacks import CallbackList
        from environments.envs.balloon_3d_env import Balloon3DEnv

        device = "cpu"  # Force CPU for short test to avoid SB3 MLP-on-GPU warning
        env = Balloon3DEnv(dim=1, render_mode=None, config={"time_max": 10})
        try:
            callback = InfoProgressBar(
                description="Test Training",
                postfix={"test": True}
            )

            model = QRDQN("MlpPolicy", env, verbose=0, learning_starts=0, device=device)
            # Very short training just to verify no errors
            model.learn(total_timesteps=16, callback=callback)
        finally:
            env.close()
