import gymnasium as gym
import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, ProgressBarCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
import pygame
import multiprocessing as mp
from typing import Callable
import time

"""
Time param cheat sheet:
- BATCH_SIZE:   Samples per SGD minibatch. How many samples the optimiser processes before one weight step.
                Larger = smoother grad estimate, heavier CPU load. Smaller = noiser grads, more optimiser steps per update.
- EPOCHS:       Passes over each rollout. How many times PPO re-shuffles the same chunk of data.
                Larger = Better utilisation of each rollout, slower wall-clock per update, risk of overfitting. Smaller = Faster updates, each sample influences policy fewer times.
- EPISODES:     Not SB3 param. Affects total_timesteps.
"""


ENVIRONMENT_NAME = "environments/Balloon3D-v0"
SAVE_PATH = "./models/ppo_model/"
VIDEO_PATH = "./figs/ppo_figs/performance_video"
USE_GPU = False

# EPISODES = 1_000_000
EPISODES = 200_000
BATCH_SIZE = 256                  # Samples per SGD minibatch. How many samples the optimiser processes before one weight step.
# BATCH_SIZE = 128
EPOCHS = 10
HIDDEN_SIZES = [[256, 256, 256], [256, 256, 256]]
EPSILON = 0.2
LEARNING_RATE = [2e-4, 1e-3]
GAMMA = 0.99
REWARD_THRESHOLD = 250

DIM = 1     # 1, 2, or 3 for 1D, 2D, or 3D environments respectively.

N_ENVS = min(8, max(2, os.cpu_count() - 1))
N_STEPS = 256           # rollout size = N_ENVS * N_STEPS (~2048-4096)


def make_env_fn(rank: int) -> Callable[[], gym.Env]:
    def _thunk():
        # IMPORTANT: avoid passing large unpicklable objects via closure.
        env = gym.make(ENVIRONMENT_NAME, render_mode=None, dim=DIM, disable_env_checker=True)
        return env
    return _thunk


# Create and train a PPO model from scratch. Returns a dataframe containing the reward attained at each episode.
def train(verbose=0, render_freq=None) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)

    # Use a GPU if possible
    device = torch.device("cuda") if USE_GPU and torch.cuda.is_available() else torch.device("cpu")

    # env: gym.Env = Monitor(gym.make(ENVIRONMENT_NAME, render_mode=None, dim=DIM, disable_env_checker=True))

    # Try SubprocVecEnv; fall back to DummyVecEnv if pickling fails
    try:
        # On macOS/Windows, safer to use "spawn"
        if mp.get_start_method(allow_none=True) != "spawn":
            mp.set_start_method("spawn", force=True)
        venv = SubprocVecEnv([make_env_fn(i) for i in range(N_ENVS)])
    except Exception as e:
        print(f"SubprocVecEnv failed ({e}). Falling back to DummyVecEnv (no true parallelism).")
        venv = DummyVecEnv([make_env_fn(i) for i in range(N_ENVS)])

    print(f"Training with {N_ENVS} environments.")

    env = VecMonitor(venv)

    model = PPO(
        "MlpPolicy",
        env,
        batch_size=BATCH_SIZE,
        n_steps=N_STEPS,
        n_epochs=EPOCHS,
        gamma=GAMMA,
        learning_rate=LEARNING_RATE[0],
        clip_range=EPSILON,
        verbose=verbose,
        device=device
    )

    # Create an evaluation callback (no vectorisation needed)
    eval_env = Monitor(gym.make(ENVIRONMENT_NAME, render_mode=None, dim=DIM, disable_env_checker=True))

    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=320,  # Set your desired reward threshold here
        verbose=1
    )
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_callback,
        best_model_save_path=SAVE_PATH,
        log_path=SAVE_PATH,
        eval_freq=100_000,
        deterministic=True,
        render=False
    )

    tqdm_callback = ProgressBarCallback()
    callback = CallbackList([tqdm_callback, eval_callback])

    # Train the model
    model.learn(total_timesteps=EPISODES*BATCH_SIZE, callback=callback)

    # Save the model
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    model.save(os.path.join(SAVE_PATH, "ppo"))


# Load the final model from the previous training run, and dipslay it playing the environment
def test() -> None:
    env: gym.Env = Monitor(gym.make(ENVIRONMENT_NAME, render_mode="human", dim=DIM, disable_env_checker=True))

    # Use a GPU if possible
    device = torch.device("cuda") if (USE_GPU and torch.cuda.is_available()) else torch.device("cpu")

    # Load the model
    model = PPO.load(os.path.join(SAVE_PATH, "ppo"), device=device)

    # Record the specified number of episodes
    for episode in range(10):
        state, info = env.reset()
        game_over: bool = False
        t0 = time.time()
        steps = 0
        while not game_over:
            steps += 1
            action, _states = model.predict(state, deterministic=True)
            # action = env.action_space.sample()      # Uncomment to override with random actions
            next_state, reward, terminated, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward}")
            state = next_state
            game_over = terminated or truncated

            if pygame.event.peek(pygame.QUIT):
                env.close()
                return

        t1 = time.time()
        print(f"{steps / (t1 - t0):.2f} steps/second")

    return
