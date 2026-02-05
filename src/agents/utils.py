import glob
import os
import pandas as pd
from stable_baselines3.common.callbacks import ProgressBarCallback


def _gather_monitor_csvs(log_dir: str) -> pd.DataFrame:
    """
    Reads VecMonitor/Monitor CSVs and returns a tidy DataFrame with:
      columns = ['episode_idx', 'r', 'l', 't', 'source']
      r = episode return, l = episode length (steps), t = time since start (seconds)
    """
    files = sorted(glob.glob(os.path.join(log_dir, "train_monitor*.csv")))
    dfs = []
    for f in files:
        # Monitor CSVs have commented headers with metadata; use comment='#'
        d = pd.read_csv(f, comment="#")
        # Add a monotonically increasing episode index per file then combine
        d["episode_idx"] = range(1, len(d) + 1)
        d["source"] = os.path.basename(f)
        dfs.append(d[["episode_idx", "r", "l", "t", "source"]])
    if not dfs:
        return pd.DataFrame(columns=["episode_idx", "r", "l", "t", "source"])
    df = pd.concat(dfs, ignore_index=True)
    # If you want a single global episode index across all vec envs:
    df["global_episode"] = range(1, len(df) + 1)
    return df


class InfoProgressBar(ProgressBarCallback):
    """Progress bar callback with custom description and postfix."""

    def __init__(self, description: str, postfix: dict | None = None):
        super().__init__()
        self._description = description
        self._postfix = postfix or {}

    def _resolve_bar(self):
        return getattr(self, "progress_bar", None) or getattr(self, "pbar", None)

    def _on_training_start(self) -> None:
        super()._on_training_start()
        bar = self._resolve_bar()
        if bar is not None:
            bar.set_description_str(self._description)
            if self._postfix:
                bar.set_postfix(self._postfix, refresh=False)

    def _on_step(self) -> bool:
        bar = self._resolve_bar()
        if bar is not None and self._postfix:
            bar.set_postfix(self._postfix, refresh=False)
        return super()._on_step()