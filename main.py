import agents.ppo as ppo
from agents.train_dqn import train as dqn_train
import argparse


if __name__ == "__main__":

    # Optional arguments
    parser = argparse.ArgumentParser(description="Train or Run a PPO on our Loon Environment.")
    parser.add_argument('--train', action='store_true', help='Run the mdoel in training mode.')
    parser.add_argument('model', choices=['ppo', 'dqn'], help='Name of agent to load/train.')
    args = parser.parse_args()

    # If the agent is in testing mode, load the saved model
    if not args.train:
        if args.model == 'ppo':
            ppo.test()
        else:
            raise NotImplementedError("DQN not implemented without training.")
    else:
        if args.model == 'ppo':
            ppo.train(verbose=True)
        else:
            dqn_train(num_episodes=500, target_update=10)
