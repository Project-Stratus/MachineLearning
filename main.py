import agents.ppo as ppo
import agents.dqn as dqn
import argparse


if __name__ == "__main__":

    # Args
    parser = argparse.ArgumentParser(description="Train or Run a PPO/DQN on our Loon Environment.")
    parser.add_argument('-t','--train', action='store_true', help='Train the model.')
    parser.add_argument('-d', '--dim', type=int, default=1, choices=[1,2,3], help='Dimensionality of the environment (1D, 2D or 3D).')
    parser.add_argument('model', choices=['ppo', 'dqn'], help='Name of agent to load/train.')
    args = parser.parse_args()

    # Call
    if args.model == 'ppo':
        if args.train:
            ppo.train(verbose=True, dim=args.dim)
        else:
            ppo.test(dim=args.dim)

    elif args.model == 'dqn':
        if args.train:
            dqn.train(dim=args.dim)
        else:
            dqn.test(dim=args.dim)

    else:
        raise ValueError(f"Unknown model type: {args.model}. Please choose 'ppo' or 'dqn'.")
