# qr_dqn_runner.py
import os
import time
import multiprocessing as mp

import gymnasium as gym
import pandas as pd
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from sb3_contrib import QRDQN

import environments  # registers the Balloon3D-v0 environment
from agents.utils import _gather_monitor_csvs, InfoProgressBar

# ---- Config ----
ENVIRONMENT_NAME = "environments/Balloon3D-v0"
SAVE_PATH = "./src/models/qr_dqn_model/"
MODEL_PATH = os.path.join(SAVE_PATH, "qr_dqn")
VIDEO_PATH = "./figs/qr_dqn_figs/performance_video"  # (unused here but kept for parity)
SEED = 42

# QR-DQN training config (strong defaults; adjust to taste)
TRAIN_CFG = dict(
    learning_rate = 3e-4,
    gamma = 0.995,
    buffer_size = 1_000_000,
    learning_starts = 50_000,
    train_freq = 4,                   # collect 4 env steps between gradient updates
    gradient_steps = 1,               # 1 gradient step per update (standard DQN ratio)
    target_update_interval = 20_000,  # soft: use tau if preferred; here: periodic hard update
    batch_size = 256,
    exploration_initial_eps = 1.0,
    exploration_final_eps = 0.01,
    exploration_fraction = 0.3,      # portion of training over which epsilon decays
    verbose = 0,
)

# Policy network; scaling toward Loon-style [600]*7 as wind fields get harder.
POLICY_KWARGS = dict(
    net_arch=[512, 512, 256],      # capacity for 2D shear; next step: [600]*5+ for realistic winds
    activation_fn=torch.nn.ReLU,
    n_quantiles=51,                # Loon-style quantile head
)

TOTAL_TIMESTEPS = 15_000_000
EVAL_FREQ = 250_000
REWARD_THRESHOLD = 4_800  # ~96% of max episode reward (5000); proves sustained station-keeping

# Environment config overrides (passed to Balloon3DEnv)
ENV_CONFIG = dict(
    wind_pattern="altitude_shear_2d",  # wind direction rotates with altitude (N→E→S→W)
)

MAX_ENVS = 4 if os.cpu_count() <= 8 else 8
N_ENVS = min(MAX_ENVS, max(1, os.cpu_count() // 2))


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


def _make_env(env_id: str, dim: int, seed: int, monitor_file: str, config: dict = None) -> Monitor:
    cfg = {**ENV_CONFIG, **(config or {})}
    env = gym.make(env_id, render_mode=None, dim=dim, disable_env_checker=True, config=cfg)
    env = Monitor(env, filename=monitor_file)  # writes train_monitor.csv
    env.reset(seed=seed)
    return env


def _make_vec_env_fn(env_id: str, dim: int, seed: int, config: dict):
    """Return a picklable factory for SubprocVecEnv workers."""
    def _init():
        env = gym.make(env_id, render_mode=None, dim=dim, disable_env_checker=True, config=config)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def _build_vec_env(n_envs: int, env_id: str, dim: int, seed: int = SEED):
    cfg = dict(ENV_CONFIG)
    if n_envs != 1:
        if mp.get_start_method(allow_none=True) != "spawn":
            mp.set_start_method("spawn", force=True)
        try:
            venv = SubprocVecEnv([_make_vec_env_fn(env_id, dim, seed + i, cfg) for i in range(n_envs)])
        except Exception as e:
            print(f"SubprocVecEnv failed ({e}). Falling back to DummyVecEnv.")
            venv = DummyVecEnv([_make_vec_env_fn(env_id, dim, seed + i, cfg) for i in range(n_envs)])
    else:
        venv = DummyVecEnv([_make_vec_env_fn(env_id, dim, seed, cfg)])

    os.makedirs(SAVE_PATH, exist_ok=True)
    monitor_file = os.path.join(SAVE_PATH, "train_monitor")
    return VecMonitor(venv, filename=monitor_file)


def train(dim: int, verbose: int = 0, render_freq=None, use_gpu: bool = False, hpc: bool = False) -> pd.DataFrame:
    """
    Train QR-DQN on the same environment. Returns a DataFrame of episode returns/lengths.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)

    device = torch.device("cuda") if (use_gpu and torch.cuda.is_available()) else torch.device("cpu")

    print(f"Training with {N_ENVS} environments, dim={dim}.")

    env = _build_vec_env(N_ENVS, ENVIRONMENT_NAME, dim=dim, seed=SEED)

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
        gym.make(ENVIRONMENT_NAME, render_mode=None, dim=dim, disable_env_checker=True, config=ENV_CONFIG),
        filename=os.path.join(SAVE_PATH, "eval_monitor.csv")
    )

    stop_cb = StopTrainingOnRewardThreshold(reward_threshold=REWARD_THRESHOLD, verbose=1)

    eval_cb = EvalCallback(
        eval_env,
        callback_on_new_best=stop_cb,
        best_model_save_path=SAVE_PATH,
        log_path=SAVE_PATH,
        eval_freq=EVAL_FREQ,
        deterministic=True,    # greedy action at eval
        render=False
    )

    callbacks = [eval_cb]
    if not hpc:
        tqdm_cb = InfoProgressBar(
            description=f"QR-DQN | steps={TOTAL_TIMESTEPS:,} | envs={N_ENVS} | device={device} |",
            postfix=dict(gamma=TRAIN_CFG["gamma"])
        )
        callbacks.insert(0, tqdm_cb)
    callback = CallbackList(callbacks)

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, tb_log_name=f"QRDQN_run_dim{dim}")
    model.save(MODEL_PATH)
    env.close()
    eval_env.close()

    # Training summary
    early_stopped = model.num_timesteps < TOTAL_TIMESTEPS
    best_reward = eval_cb.best_mean_reward
    print(f"\n{'='*50}")
    print(f"Training complete {'(early stopped)' if early_stopped else '(full run)'}")
    print(f"  Best eval reward: {best_reward:.2f}")
    print(f"  Total timesteps:  {model.num_timesteps:,} / {TOTAL_TIMESTEPS:,}")
    print(f"  Model saved to:   {os.path.abspath(MODEL_PATH)}")
    if hpc:
        print(f"  Device:           {device}")
    print(f"{'='*50}\n")

    return _gather_monitor_csvs(SAVE_PATH)


def test(dim: int, use_gpu: bool = False) -> None:
    """
    Load the saved QR-DQN and run a few episodes with greedy actions.
    Mirrors your PPO test loop (minus policy distribution prints).
    """
    import pygame
    from environments.envs.balloon_3d_env import Actions  # your enum
    from environments.envs.balloon_3d_env import Balloon3DEnv

    device = torch.device("cuda") if (use_gpu and torch.cuda.is_available()) else torch.device("cpu")

    # Human-render env for demo
    test_config = {**ENV_CONFIG, "time_max": 2_000}
    env: gym.Env = Monitor(
        gym.make(ENVIRONMENT_NAME, render_mode="human", dim=dim, disable_env_checker=True, config=test_config)
    )

    # Load model
    model: QRDQN = QRDQN.load(MODEL_PATH, device=device)

    # (Optional) inspect Q-values for a single obs
    env_temp = Balloon3DEnv(dim=dim, render_mode=None, config=ENV_CONFIG)
    obs, _ = env_temp.reset(seed=42)
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=model.device).unsqueeze(0)
    with torch.no_grad():
        quantiles = model.quantile_net(obs_tensor)  # shape: [1, n_quantiles, n_actions]
        q_values = quantiles.mean(dim=1)  # mean across quantiles -> [1, n_actions]
        print("Q-values:", q_values.cpu().numpy())

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

            effect = int(env.unwrapped._action_lut[action_idx])
            act = (Actions(effect).name if effect in Actions._value2member_map_ else "?").upper()[:3]

            pos = env.unwrapped._balloon.pos
            if env.unwrapped.dim == 1:
                pos_str = f"z={pos[0]:+.1f}"
            elif env.unwrapped.dim == 2:
                pos_str = f"{pos[0]:+.1f},{pos[1]:+.1f}"
            else:
                pos_str = f"{pos[0]:+.1f},{pos[1]:+.1f},{pos[2]:+.1f}"

            c = info.get("reward_components", {})
            print(
                f"E{episode+1}|S{steps:>5}|A:{act:<3}"
                f"|Pos:{pos_str}"
                f"|R:{reward:+.3f}"
                f"|stn:{c.get('station',0):.3f}"
                f" dec:{c.get('decay',0):.3f}"
            )

            state = next_state
            done = terminated or truncated

            # Check renderer flags (events are processed inside draw())
            renderer = env.unwrapped.renderer
            if renderer is not None:
                if renderer.quit_requested:
                    env.close()
                    return
                if renderer.skip_requested:
                    renderer.skip_requested = False
                    done = True

        t1 = time.time()
        print(f"{steps / (t1 - t0):.2f} steps/second")

        # Show end screen with termination reason
        reason = info.get("termination_reason", "Episode ended")
        renderer = env.unwrapped.renderer
        if renderer is not None:
            renderer.show_end_screen(reason)

    env.close()
