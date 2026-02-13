import os
import time
import multiprocessing as mp
from typing import Callable

import gymnasium as gym
import pygame
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor

from agents.utils import _gather_monitor_csvs, InfoProgressBar
"""
Time param cheat sheet:
- BATCH_SIZE:   Samples per SGD minibatch. How many samples the optimiser processes before one weight step.
                Larger = smoother grad estimate, heavier CPU load. Smaller = noiser grads, more optimiser steps per update.
- EPOCHS:       Passes over each rollout. How many times PPO re-shuffles the same chunk of data.
                Larger = Better utilisation of each rollout, slower wall-clock per update, risk of overfitting. Smaller = Faster updates, each sample influences policy fewer times.
- EPISODES:     Not SB3 param. Affects total_timesteps.
"""


ENVIRONMENT_NAME = "environments/Balloon3D-v0"
SAVE_PATH = "./src/models/ppo_model/"
MODEL_PATH = SAVE_PATH + "ppo"
VIDEO_PATH = "./figs/ppo_figs/performance_video"
SEED = 42

TRAIN_CFG = dict(
    batch_size = 128,           # Samples per SGD minibatch. How many samples the optimiser processes before one weight step.
    n_epochs = 4,
    n_steps = 256,              # rollout size = N_ENVS * N_STEPS (~2048-4096)
    learning_rate = 3e-4,       # [2e-4, 1e-3],
    gamma = 0.99,
    clip_range = 0.2,
    ent_coef = 0.01,            # 0.0 = no entropy bonus, 0.01 = small bonus to encourage exploration (add/remove randomness)
)

POLICY_KWARGS = dict(
    net_arch = dict(pi=[64, 64], vf=[64, 64]),        # [[256, 256, 256], [256, 256, 256]],
    activation_fn = torch.nn.Tanh
)

TOTAL_TIMESTEPS = 5_000_000  # Total training steps
EVAL_FREQ = 500_000  # Evaluate every n steps
REWARD_THRESHOLD = 10_000  # Stop training when the model reaches this reward

MAX_ENVS = 4 if os.cpu_count() <= 8 else 8
N_ENVS = min(MAX_ENVS, max(1, os.cpu_count() //2))

def make_env_fn(dim: int) -> Callable[[], gym.Env]:
    def _thunk():
        # IMPORTANT: avoid passing large unpicklable objects via closure.
        env = gym.make(ENVIRONMENT_NAME, render_mode=None, dim=dim, disable_env_checker=True)
        return env
    return _thunk


# Create and train a PPO model from scratch. Returns a dataframe containing the reward attained at each episode.
def train(dim, verbose=0, render_freq=None, use_gpu: bool = False, hpc: bool = False) -> "pd.DataFrame":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)

    # Use a GPU if possible
    device = torch.device("cuda") if use_gpu and torch.cuda.is_available() else torch.device("cpu")

    def make_env(env_id, seed, dim):
        # Define at module level so it’s picklable for SubprocVecEnv
        def _init():
            env = gym.make(
                env_id,
                render_mode=None,
                dim=dim,
                disable_env_checker=True,
            )
            env = Monitor(env)
            env.reset(seed=seed)
            return env
        return _init

    def build_vec_env(n_envs, env_id, dim, seed=SEED):
        if n_envs != 1:
            # On macOS/Windows, SubprocVecEnv needs "spawn"
            if mp.get_start_method(allow_none=True) != "spawn":
                mp.set_start_method("spawn", force=True)
            try:
                venv = SubprocVecEnv([make_env(env_id, seed + i, dim) for i in range(n_envs)])
            except Exception as e:
                print(f"SubprocVecEnv failed ({e}). Falling back to DummyVecEnv.")
                venv = DummyVecEnv([make_env(env_id, seed + i, dim) for i in range(n_envs)])
        else:
            venv = DummyVecEnv([make_env(env_id, seed, dim)])

        # Write monitor CSV to disk to build training curves post-hoc
        os.makedirs(SAVE_PATH, exist_ok=True)
        monitor_file = os.path.join(SAVE_PATH, "train_monitor")     # We will create train_monitor.csv, train_monitor_1.csv, etc.
        return VecMonitor(venv, filename=monitor_file)

    print(f"Training with {N_ENVS} environments, dim={dim}.")

    env = VecMonitor(build_vec_env(N_ENVS, ENVIRONMENT_NAME, dim=dim, seed=SEED))

    model = PPO(
        "MlpPolicy",
        env,
        verbose=verbose,
        device=device,
        tensorboard_log=SAVE_PATH,  # Tensorboard log dir
        **TRAIN_CFG,
        policy_kwargs=POLICY_KWARGS
    )

    # Create an evaluation callback (no vectorisation needed)
    eval_env = Monitor(gym.make(ENVIRONMENT_NAME, render_mode=None, dim=dim, disable_env_checker=True), filename=os.path.join(SAVE_PATH, "eval_monitor.csv"))

    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=REWARD_THRESHOLD,
        verbose=1
    )
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_callback,
        best_model_save_path=SAVE_PATH,
        log_path=SAVE_PATH,
        eval_freq=EVAL_FREQ,
        deterministic=True, 
        render=False
    )

    callbacks = [eval_callback]
    if not hpc:
        tqdm_callback = InfoProgressBar(
            description=f"PPO | steps={TOTAL_TIMESTEPS:,} | envs={N_ENVS} | device={device} |",
            postfix=dict(gamma=TRAIN_CFG["gamma"], ent_coef=TRAIN_CFG["ent_coef"]),
        )
        callbacks.insert(0, tqdm_callback)
    callback = CallbackList(callbacks)

    # Train the model
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback, tb_log_name=f"PPO_run_dim{dim}")

    # Save the model
    model.save(MODEL_PATH)

    # Training summary
    early_stopped = model.num_timesteps < TOTAL_TIMESTEPS
    best_reward = eval_callback.best_mean_reward
    print(f"\n{'='*50}")
    print(f"Training complete {'(early stopped)' if early_stopped else '(full run)'}")
    print(f"  Best eval reward: {best_reward:.2f}")
    print(f"  Total timesteps:  {model.num_timesteps:,} / {TOTAL_TIMESTEPS:,}")
    print(f"  Model saved to:   {os.path.abspath(MODEL_PATH)}")
    if hpc:
        print(f"  Device:           {device}")
    print(f"{'='*50}\n")

    df = _gather_monitor_csvs(SAVE_PATH)
    return df


# Load the final model from the previous training run, and dipslay it playing the environment
def test(dim, use_gpu: bool = False) -> None:
    from environments.envs.balloon_3d_env import Actions

    env: gym.Env = Monitor(gym.make(ENVIRONMENT_NAME, render_mode="human", dim=dim, disable_env_checker=True, config={"time_max": 2_000}))

    # Use a GPU if possible
    device = torch.device("cuda") if (use_gpu and torch.cuda.is_available()) else torch.device("cpu")

    # Load the model
    model = PPO.load(MODEL_PATH, device=device)

    from environments.envs.balloon_3d_env import Balloon3DEnv

    model = PPO.load(MODEL_PATH, device="cpu")
    env_temp = Balloon3DEnv(dim=1, render_mode=None)

    obs, _ = env_temp.reset(seed=42)
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=model.device).unsqueeze(0)

    dist = model.policy.get_distribution(obs_tensor)
    print("action probs:", dist.distribution.probs.detach().cpu().numpy())
    print("logits:", dist.distribution.logits.detach().cpu().numpy())

    # Record the specified number of episodes
    for episode in range(10):
        state, info = env.reset()
        game_over: bool = False
        t0 = time.time()
        steps = 0
        while not game_over:
            steps += 1
            action_idx, _states = model.predict(state, deterministic=False)     # Switch to true when distribution learns to favour correct action
            # action_idx = env.action_space.sample()      # Uncomment to override with random actions
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
            game_over = terminated or truncated

            # Check renderer flags (events are processed inside draw())
            renderer = env.unwrapped.renderer
            if renderer is not None:
                if renderer.quit_requested:
                    env.close()
                    return
                if renderer.skip_requested:
                    renderer.skip_requested = False
                    game_over = True

        t1 = time.time()
        print(f"{steps / (t1 - t0):.2f} steps/second")

        # Show end screen with termination reason
        reason = info.get("termination_reason", "Episode ended")
        renderer = env.unwrapped.renderer
        if renderer is not None:
            renderer.show_end_screen(reason)

    return
