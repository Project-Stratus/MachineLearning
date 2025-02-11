"""
My rendition of a PPO based on this paper:
https://arxiv.org/abs/1707.06347

PPO Summary:
A PPO is 'Proximal Policy Optimization'. That means:
 - Proximal = We don't optimize with respect to the raw reward, instead we optimize a "surrogate objective".
 - Policy Optimization = We optimize the policy pi based on rewards, as opposed to optimizing a value function like in Q-Learning.

These models can not only deal with continuous action spaces, but have higher reliability, meaning that they don't get 'stuck' with bad policies as often.
It's really good at utilizing data, making it a good fit for applications where data is expensive (like robotics). That attribute is not as applicable in these simulations.
This model is stochastic, meaning that actions are chosen by converting the output of the actor to a probability distribtion, and then picking one randomly.
It is also "On-Policy", which means that it updates the model based on actual decisions the model makes. 
It is also "Online", meaning that it learns as it plays, rather than learning after a long session of playtime to begin training.

PPOs have two models - an actor and a critic.
The actor is what plays the game - it explores the environment and tries to earn reward. 
We use the critic to evaluate the actor - this is what lets us know how to improve the actor. The critic tries to estimate the actor's average received value of each state,
based the reward recieved (and the expected value from future states). 
However, what the actor learns isn't a simple reward function. Instead, the actor learns a more complex formula called the surrogate objective (surrogate because it isn't just TD learning or straight reward).
We use the critic's value guesses about state-action pairs the to tell the actor what it does well, and what it should do differently. 

The PPO works by playing a small number of steps (the batch size), and then training multiple times on that batch of data. This is why the PPO is so data efficient; it re-uses data.
The reason that it can re-use data better than other models can is because it scales the amount it learns based on how different the current model is from when the data is collected.
The model is only sort of "Online" - it doesn't train after each action, instead it trains after a relatively small number of actions, called a batch. This is why it is important to
check how different the model is after each training step - each time we train on an item from the batch, the policy gets more and more different from the policy used to make those
actions, and the data becomes less relevant.
PPOs also limit how much it will increase the likelyhood of an action, so avoid overfitting on high rewards on single steps. 

The surrogate objective is: min( r*A, clip(r, 1-e, 1+e)*A )
Where r is the difference between the policy used to , e is some small number, and A is the advantage for the current state-action pair. A = discounted rewards - critic's estimate
The "clip" is what limits change based on model difference, and the "min" ensures that we don't increase the likelyhood of any action too much in a single step.
"""

import gymnasium as gym
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.distributions import Categorical
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

import loon_v0


ENVIRONMENT_NAME = "loon_v0/LoonEnv-v0"
SAVE_PATH = "./model_saves/"
VIDEO_PATH = "/performance_video/"
USE_GPU = False

EPISODES = 1000000
BATCH_SIZE = 256
EPOCHS = 10
HIDDEN_SIZES = [[256, 256, 256], [256, 256, 256]]
EPSILON = 0.2
LEARNING_RATE = [2e-4, 1e-3]
GAMMA = 0.99
REWARD_THRESHOLD = 250


# Create and train a PPO model from scratch. Returns a dataframe containing the reward attained at each episode.
def train(verbose=False, render_freq=None) -> pd.DataFrame:

    env: gym.Env = gym.make(ENVIRONMENT_NAME, render_mode=None, disable_env_checker=True)

    # Use a GPU if possible
    device = torch.device("cuda") if USE_GPU and torch.cuda.is_available() else torch.device("cpu")

    model = PPO(
        "MultiInputPolicy",
        env,
        batch_size=BATCH_SIZE,
        n_epochs=EPOCHS,
        verbose=1,
        device=device
    )

    # Create an evaluation callback
    eval_env = gym.make(ENVIRONMENT_NAME, render_mode=None, disable_env_checker=True)
    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=320,  # Set your desired reward threshold here
        verbose=1
    )
    callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_callback,
        best_model_save_path=SAVE_PATH,
        log_path=SAVE_PATH,
        eval_freq=10000,
        deterministic=True,
        render=False
    )

    # Train the model
    model.learn(total_timesteps=EPISODES*BATCH_SIZE, callback=callback)

    # Save the model
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    model.save(os.path.join(SAVE_PATH, "ppo"))

    return pd.DataFrame()


# Load the final model from the previous training run, and dipslay it playing the environment
def test() -> None:
    env: gym.Env = gym.make(ENVIRONMENT_NAME, render_mode="human", disable_env_checker=True)

    # Use a GPU if possible
    device = torch.device("cuda") if (USE_GPU and torch.cuda.is_available()) else torch.device("cpu")

    # Load the model
    model = PPO.load(os.path.join(SAVE_PATH, "ppo"), device=device)

    # Record the specified number of episodes
    for episode in range(10):
        state, info = env.reset()
        game_over: bool = False
        while not game_over:
            env.render()
            action, _states = model.predict(state, deterministic=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            state = next_state
            game_over = terminated or truncated

    return
