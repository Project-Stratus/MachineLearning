# qr_dqn_runner.py
import os
import glob
import time
import gymnasium as gym
import torch
import pandas as pd
from typing import Callable
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, ProgressBarCallback, CallbackList
from sb3_contrib import QRDQN
from tqdm.auto import tqdm

from src.agents.utils import _gather_monitor_csvs

# ---- Config ----
ENVIRONMENT_NAME = "environments/Balloon3D-v0"
SAVE_PATH = "./src/models/qr_dqn_model/"
MODEL_PATH = os.path.join(SAVE_PATH, "qr_dqn")
VIDEO_PATH = "./figs/qr_dqn_figs/performance_video"  # (unused here but kept for parity)
USE_GPU = False
SEED = 42

# QR-DQN training config (strong defaults; adjust to taste)
TRAIN_CFG = dict(
    learning_rate = 3e-4,
    gamma = 0.99,
    buffer_size = 1_000_000,
    learning_starts = 50_000,
    train_freq = 32,                  # steps between gradient steps
    gradient_steps = 32,              # steps per env step collected (classic DQN setup is 1)
    target_update_interval = 20_000,  # soft: use tau if preferred; here: periodic hard update
    batch_size = 256,
    exploration_initial_eps = 1.0,
    exploration_final_eps = 0.05,
    exploration_fraction = 0.2,      # portion of training over which epsilon decays
    verbose = 0,
)

# Policy network; start modest. If you want Loon-style capacity, set [600]*7.
POLICY_KWARGS = dict(
    net_arch=[256, 256],  # try [600, 600, 600, 600, 600, 600, 600] to mimic Loon
    activation_fn=torch.nn.ReLU,
    n_quantiles=25,                # Loon-style head size is 51 quantiles/action
)

TOTAL_TIMESTEPS = 1_500_000
EVAL_FREQ = 1_000_000
REWARD_THRESHOLD = 10_000  # stop early on good performance


# def _gather_monitor_csvs(log_dir: str) -> pd.DataFrame:
#     """
#     Reads Monitor CSVs and returns tidy DataFrame:
#     columns = ['episode_idx','r','l','t','source','global_episode']
#     """
#     os.makedirs(log_dir, exist_ok=True)
#     files = sorted(glob.glob(os.path.join(log_dir, "train_monitor*.csv")))
#     dfs = []
#     for f in files:
#         d = pd.read_csv(f, comment="#")
#         d["episode_idx"] = range(1, len(d) + 1)
#         d["source"] = os.path.basename(f)
#         dfs.append(d[["episode_idx", "r", "l", "t", "source"]])
#     if not dfs:
#         return pd.DataFrame(columns=["episode_idx", "r", "l", "t", "source", "global_episode"])
#     df = pd.concat(dfs, ignore_index=True)
#     df["global_episode"] = range(1, len(df) + 1)
#     return df


def _make_env(env_id: str, dim: int, seed: int, monitor_file: str) -> Monitor:
    env = gym.make(env_id, render_mode=None, dim=dim, disable_env_checker=True)
    env = Monitor(env, filename=monitor_file)  # writes train_monitor.csv
    env.reset(seed=seed)
    return env


def train(dim: int, verbose: int = 0, render_freq=None) -> pd.DataFrame:
    """
    Train QR-DQN on the same environment. Returns a DataFrame of episode returns/lengths.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)

    device = torch.device("cuda") if (USE_GPU and torch.cuda.is_available()) else torch.device("cpu")

    os.makedirs(SAVE_PATH, exist_ok=True)
    monitor_file = os.path.join(SAVE_PATH, "train_monitor")
    env = _make_env(ENVIRONMENT_NAME, dim=dim, seed=SEED, monitor_file=monitor_file)

    model = QRDQN(
        policy="MlpPolicy",
        env=env,
        tensorboard_log=SAVE_PATH,
        device=device,
        **TRAIN_CFG,
        policy_kwargs=POLICY_KWARGS,
        seed=SEED,
    )

    # Evaluation env (separate Monitor file)
    eval_env = Monitor(
        gym.make(ENVIRONMENT_NAME, render_mode=None, dim=dim, disable_env_checker=True),
        filename=os.path.join(SAVE_PATH, "eval_monitor.csv")
    )

    stop_cb = StopTrainingOnRewardThreshold(reward_threshold=REWARD_THRESHOLD, verbose=1)

    class InfoProgressBar(ProgressBarCallback):
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

    eval_cb = EvalCallback(
        eval_env,
        callback_on_new_best=stop_cb,
        best_model_save_path=SAVE_PATH,
        log_path=SAVE_PATH,
        eval_freq=EVAL_FREQ,
        deterministic=True,    # greedy action at eval
        render=False
    )

    tqdm_cb = InfoProgressBar(
        description=f"QR-DQN | steps={TOTAL_TIMESTEPS:,} | device={device} |",
        postfix=dict(gamma=TRAIN_CFG["gamma"])
    )
    callback = CallbackList([tqdm_cb, eval_cb])

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, tb_log_name=f"QRDQN_run_dim{dim}")
    model.save(MODEL_PATH)
    env.close()
    eval_env.close()

    return _gather_monitor_csvs(SAVE_PATH)


def test(dim: int) -> None:
    """
    Load the saved QR-DQN and run a few episodes with greedy actions.
    Mirrors your PPO test loop (minus policy distribution prints).
    """
    import pygame
    from environments.envs.balloon_3d_env import Actions  # your enum
    from environments.envs.balloon_3d_env import Balloon3DEnv

    device = torch.device("cuda") if (USE_GPU and torch.cuda.is_available()) else torch.device("cpu")

    # Human-render env for demo
    env: gym.Env = Monitor(
        gym.make(ENVIRONMENT_NAME, render_mode="human", dim=dim, disable_env_checker=True, config={"time_max": 2_000})
    )

    # Load model
    model: QRDQN = QRDQN.load(MODEL_PATH, device=device)

    # (Optional) inspect Q-values for a single obs
    env_temp = Balloon3DEnv(dim=1, render_mode=None)
    obs, _ = env_temp.reset(seed=42)
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=model.device).unsqueeze(0)
    with torch.no_grad():
        q_values = model.q_net(obs_tensor)
        print("Q-values:", q_values.detach().cpu().numpy())   # shape: [1, n_actions]

    # Roll a few episodes
    for episode in range(10):
        state, info = env.reset()
        done = False
        t0 = time.time()
        steps = 0
        while not done:
            steps += 1
            # Greedy at test time (deterministic=True)
            action_idx, _ = model.predict(state, deterministic=True)
            next_state, reward, terminated, truncated, info = env.step(action_idx)

            # Pretty print (mirrors your PPO tester)
            effect = int(env.unwrapped._action_lut[action_idx])
            text_action = (Actions(effect).name if effect in Actions._value2member_map_ else "UNKNOWN").upper()

            comps = info.get("reward_components", {})
            print(
                f"|| Ep {episode+1} || Step {steps:>6} || Action: {text_action} "
                f"|| Reward: {reward:+.4f} || Components: "
                f"[Dist.: {comps.get('distance', 0.0):+.4f}, "
                f"Dir.: {comps.get('direction', 0.0):+.4f}, "
                f"Rea.: {comps.get('reached', 0.0):+.4f}, "
                f"Eff.: {comps.get('effect', 0.0):+.4f}] ||"
            )

            state = next_state
            done = terminated or truncated

            if pygame.event.peek(pygame.QUIT):
                env.close()
                return

        t1 = time.time()
        print(f"{steps / (t1 - t0):.2f} steps/second")

    env.close()
