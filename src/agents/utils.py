import glob
import os
import pandas as pd

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