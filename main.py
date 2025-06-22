import agents.ppo as ppo
from agents.train_dqn import train as dqn_train
from agents.test_dqn import dqn_test
import argparse


if __name__ == "__main__":

    # Optional arguments
    parser = argparse.ArgumentParser(description="Train or Run a PPO on our Loon Environment.")
    parser.add_argument('--train', action='store_true', help='Run the mdoel in training mode.')
    parser.add_argument('model', choices=['ppo', 'dqn'], help='Name of agent to load/train.')
    args = parser.parse_args()

    if args.model == 'ppo':
        if args.train:
            ppo.train(verbose=True)
        else:
            ppo.test()

    elif args.model == 'dqn':
        if args.train:
            dqn_train(num_episodes=500, target_update=10)
        else:
            # raise NotImplementedError("Testing mode for DQN is not implemented. Please use the --train flag.")
            dqn_test()

    else:
        raise ValueError(f"Unknown model type: {args.model}")
