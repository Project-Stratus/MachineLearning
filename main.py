import agents.ppo as ppo
import agents.dqn as dqn
import argparse


if __name__ == "__main__":

    # Args
    parser = argparse.ArgumentParser(description="Train or Run a PPO/DQN on our Loon Environment.")
    parser.add_argument('--train', action='store_true', help='Run the mdoel in training mode.')
    parser.add_argument('model', choices=['ppo', 'dqn'], help='Name of agent to load/train.')
    args = parser.parse_args()

    # Call
    if args.model == 'ppo':
        if args.train:
            ppo.train(verbose=True)
        else:
            ppo.test()

    elif args.model == 'dqn':
        if args.train:
            dqn.train()
        else:
            dqn.test()

    else:
        raise ValueError(f"Unknown model type: {args.model}")
