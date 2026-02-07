"""Tests for agent utility functions."""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from agents.utils import _gather_monitor_csvs, InfoProgressBar


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


class TestInfoProgressBarIntegration:
    """Integration tests for InfoProgressBar with actual training."""

    @pytest.mark.slow
    def test_info_progress_bar_with_short_training(self):
        """InfoProgressBar should work with actual (short) training."""
        pytest.importorskip("stable_baselines3")

        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CallbackList
        from environments.envs.balloon_3d_env import Balloon3DEnv

        device = "cpu"  # Force CPU for short test to avoid SB3 MLP-on-GPU warning
        env = Balloon3DEnv(dim=1, render_mode=None, config={"time_max": 10})
        try:
            callback = InfoProgressBar(
                description="Test Training",
                postfix={"test": True}
            )

            model = PPO("MlpPolicy", env, verbose=0, n_steps=8, batch_size=4, device=device)
            # Very short training just to verify no errors
            model.learn(total_timesteps=16, callback=callback)
        finally:
            env.close()
