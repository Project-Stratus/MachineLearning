from __future__ import annotations

import os
import sys
import gymnasium as gym
import torch
import pygame
import environments    # noqa - Required to register the custom environment

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)

ENVIRONMENT_NAME = "environments/Balloon2D-v0"

SAVE_PATH = "./models/dqn_model"
MODEL_FILE = os.path.join(SAVE_PATH, "dqn")  # SB3 adds .zip
USE_GPU = False

TOTAL_TIMESTEPS = 1_000_000
BUFFER_SIZE = 100_000
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
GAMMA = 0.99
TARGET_UPDATE = 10_000  # env steps
TRAIN_FREQ = 4
EPS_FRACTION = 0.1  # frac of steps over which epsilon decays
EPS_FINAL = 0.05
REWARD_THRESHOLD = 250  # early-stop criterion in callback


# Create and train a DQN model from scratch. Returns a dataframe containing the reward attained at each episode.
def train() -> None:
    os.makedirs(SAVE_PATH, exist_ok=True)

    env = Monitor(gym.make(ENVIRONMENT_NAME, disable_env_checker=True))
    eval_env = Monitor(gym.make(ENVIRONMENT_NAME, disable_env_checker=True))

    device = torch.device("cuda") if USE_GPU and torch.cuda.is_available() else "cpu"

    model = DQN(
        "MlpPolicy",
        env,
        buffer_size=BUFFER_SIZE,
        learning_starts=10_000,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        train_freq=TRAIN_FREQ,
        target_update_interval=TARGET_UPDATE,
        exploration_fraction=EPS_FRACTION,
        exploration_final_eps=EPS_FINAL,
        verbose=1,
        device=device,
    )

    callback = EvalCallback(
        eval_env,
        best_model_save_path=SAVE_PATH,
        log_path=SAVE_PATH,
        eval_freq=20_000,
        deterministic=True,
        callback_on_new_best=StopTrainingOnRewardThreshold(
            reward_threshold=REWARD_THRESHOLD,
            verbose=1,
        ),
    )

    # Train the model
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

    # Save the model
    model.save(MODEL_FILE)
    env.close()
    eval_env.close()


# Load final model from prev training run, and display it playing the environment
def test(episodes: int = 10) -> None:
    env = Monitor(
        gym.make(ENVIRONMENT_NAME, render_mode="human", disable_env_checker=True)
    )

    device = torch.device("cuda") if USE_GPU and torch.cuda.is_available() else "cpu"
    model = DQN.load(MODEL_FILE, device=device)

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if pygame.event.peek(pygame.QUIT):
                env.close()
                return
    env.close()


if __name__ == "__main__":
    if len(sys.argv) == 1 or sys.argv[1].lower() == "train":
        train()
    elif sys.argv[1].lower() == "test":
        test()
    else:
        print("Usage: python dqn.py [train|test]")
