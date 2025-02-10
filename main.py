import ppo
import argparse
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import os


if __name__ == "__main__":

    # Optional arguments
    parser = argparse.ArgumentParser(description="Train or Run a PPO on our Loon Environment.")
    parser.add_argument('--train', action='store_true', help='Run the mdoel in training mode.')
    args = parser.parse_args()

    # If the agent is in testing mode, load the saved model
    if not args.train:
        ppo.test()
    else:
        reward_df = ppo.train(verbose=True)

        # Plot the average reward over time
        sns.lineplot(x='episode', y='reward', data=reward_df)
        plt.title(f"Reward Over Episodes")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.savefig(f"{os.getcwd() + ppo.SAVE_PATH}/PPO_reward_over_episodes.png")
        plt.show()