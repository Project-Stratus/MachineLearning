"""
test_dqn.py
-----------
Play back a trained DQN controller in the 2-D balloon environment.

• Opens one matplotlib window that updates every step.
• Close the window or press Ctrl-C to abort early.
"""

# import os
import sys
from pathlib import Path
import time
# import numpy as np
import torch

from agents.dqn_agent import DQNAgent
from environments.envs.balloon_env import Balloon2DEnv


# ----------------------------------------------------------------------
# User-tweakable parameters
# ----------------------------------------------------------------------
MODEL_PATH = Path("models/dqn_balloon_model.pth")  # where train.py saved the weights
NUM_EPISODES = 5  # number of demo episodes
RENDER_DELAY_S = 0.02  # extra pause after each frame


def load_agent(env: Balloon2DEnv, model_path: Path, device: torch.device) -> DQNAgent:
    """Reconstruct the DQN agent and load its trained weights."""
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)
    checkpoint = torch.load(model_path, map_location=device)
    agent.policy_net.load_state_dict(checkpoint)
    agent.policy_net.eval()  # inference mode
    agent.epsilon = 0.0  # pure exploitation

    return agent


def dqn_test() -> None:
    if not MODEL_PATH.exists():
        sys.exit(f"[ERROR] Cannot find model weights at {MODEL_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = Balloon2DEnv()  # render_mode is None; we call env.render() manually
    agent = load_agent(env, MODEL_PATH, device)

    try:
        for ep in range(NUM_EPISODES):
            state, info = env.reset()
            done = False
            episode_reward = 0.0

            while not done:
                env.render()  # draws / updates the matplotlib figure
                # time.sleep(RENDER_DELAY_S)

                action = agent.select_action(state)  # epsilon = 0 → greedy
                state, reward, done, _ = env.step(action)
                episode_reward += reward

            print(f"Episode {ep + 1}: reward = {episode_reward: .2f}")

    finally:
        env.close()


if __name__ == "__main__":
    dqn_test()
