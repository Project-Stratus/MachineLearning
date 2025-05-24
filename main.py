import ppo
import argparse
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
        ppo.train(verbose=True)
