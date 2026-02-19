import glob
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback, ProgressBarCallback


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


class TerminationTracker(BaseCallback):
    """Tracks termination reasons from VecEnv infos and prints a summary at the end."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.counts: dict[str, int] = defaultdict(int)

    def _on_step(self) -> bool:
        # SB3 VecEnv stores episode info in self.locals["infos"]
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", np.zeros(len(infos), dtype=bool))
        for info, done in zip(infos, dones):
            if not done:
                continue
            reason = info.get("termination_reason", None)
            if reason is None:
                # Check inside the terminal_observation wrapper used by VecEnv
                ep_info = info.get("terminal_info", info)
                reason = ep_info.get("termination_reason", "Unknown")
            self.counts[reason] += 1
        return True

    def _on_training_end(self) -> None:
        total = sum(self.counts.values())
        if total == 0:
            return
        print(f"\n{'='*55}")
        print(f"  Termination Breakdown ({total:,} episodes)")
        print(f"{'='*55}")
        for reason, count in sorted(self.counts.items(), key=lambda x: -x[1]):
            pct = 100.0 * count / total
            print(f"  {reason:<45s} {count:>6,}  ({pct:5.1f}%)")
        print(f"{'='*55}\n")


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