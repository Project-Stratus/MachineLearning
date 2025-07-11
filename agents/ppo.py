import gymnasium as gym
import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, ProgressBarCallback, CallbackList
import pygame
from stable_baselines3.common.monitor import Monitor


ENVIRONMENT_NAME = "environments/Balloon3D-v0"
SAVE_PATH = "./models/ppo_model/"
VIDEO_PATH = "./figs/ppo_figs/performance_video"
USE_GPU = False

EPISODES = 1_000_000
# EPISODES = 1_000
BATCH_SIZE = 256
EPOCHS = 10
HIDDEN_SIZES = [[256, 256, 256], [256, 256, 256]]
EPSILON = 0.2
LEARNING_RATE = [2e-4, 1e-3]
GAMMA = 0.99
REWARD_THRESHOLD = 250

DIM = 1     # 1, 2, or 3 for 1D, 2D, or 3D environments respectively.


# Create and train a PPO model from scratch. Returns a dataframe containing the reward attained at each episode.
def train(verbose=0, render_freq=None) -> None:

    env: gym.Env = Monitor(gym.make(ENVIRONMENT_NAME, render_mode=None, dim=DIM, disable_env_checker=True))

    # Use a GPU if possible
    device = torch.device("cuda") if USE_GPU and torch.cuda.is_available() else torch.device("cpu")

    model = PPO(
        "MlpPolicy",
        env,
        batch_size=BATCH_SIZE,
        n_epochs=EPOCHS,
        verbose=verbose,
        device=device
    )

    # Create an evaluation callback
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
        while not game_over:
            action, _states = model.predict(state, deterministic=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            # next_state, reward, terminated, truncated, info = env.step(env.action_space.sample())
            state = next_state
            game_over = terminated or truncated

            if pygame.event.peek(pygame.QUIT):
                env.close()
                return

    return
