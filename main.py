import sys
from pathlib import Path

# Add src/ to path so we can import agents and environments
sys.path.insert(0, str(Path(__file__).parent / "src"))

import agents.ppo as ppo
import agents.qrdqn as qrdqn
import argparse
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # Args
    parser = argparse.ArgumentParser(description="Train or Run a PPO/QR-DQN agent on our Loon Environment.")
    parser.add_argument('-t','--train', action='store_true', help='Train the model.')
    parser.add_argument('-g', '--gpu', action='store_true', help='Use GPU for training/inference if available.')
    parser.add_argument('--hpc', action='store_true', help='HPC mode: disables progress bars for non-interactive SLURM jobs.')
    parser.add_argument('-d', '--dim', type=int, default=3, choices=[1,2,3], help='Dimensionality of the environment (1D, 2D or 3D).')
    parser.add_argument('-sf', '--save_fig', action='store_true', help='Save training figure to disk (only in train mode).')
    parser.add_argument('model', choices=['ppo', 'qrdqn'], default='qrdqn', help='Name of agent to load/train.')
    args = parser.parse_args()

    model = str(args.model).lower()

    # Call
    if model == 'ppo':
        if args.train:
            df = ppo.train(verbose=True, dim=args.dim, use_gpu=args.gpu, hpc=args.hpc)
            if args.save_fig:
                save_dir = "src/models/ppo_model"

                plt.figure(figsize=(10, 6))
                plt.plot(df["global_episode"], df["r"], alpha=0.4, label="Episode reward")
                rolling = df["r"].rolling(window=100, min_periods=1).mean()
                plt.plot(df["global_episode"], rolling, color="black", linestyle="--", linewidth=1.5, label="100-episode avg")
                plt.xlabel("Episode")
                plt.ylabel("Reward")
                plt.title("PPO Training Reward Curve")
                plt.legend()
                plt.grid()
                plt.savefig(f"{save_dir}/training_curve_dim{args.dim}.png")
                plt.close()
                print(f"Training figure saved to {save_dir}/training_curve_dim{args.dim}.png")

                eval_file = f"{save_dir}/evaluations.npz"
                if Path(eval_file).exists():
                    ev = np.load(eval_file)
                    timesteps, results = ev["timesteps"], ev["results"]
                    mean_r = results.mean(axis=1)
                    std_r = results.std(axis=1)
                    plt.figure(figsize=(10, 6))
                    plt.plot(timesteps, mean_r, marker="o", label="Mean eval reward")
                    plt.fill_between(timesteps, mean_r - std_r, mean_r + std_r, alpha=0.2)
                    plt.xlabel("Timestep")
                    plt.ylabel("Reward")
                    plt.title("PPO Evaluation Reward Curve")
                    plt.legend()
                    plt.grid()
                    plt.savefig(f"{save_dir}/eval_curve_dim{args.dim}.png")
                    plt.close()
                    print(f"Eval figure saved to {save_dir}/eval_curve_dim{args.dim}.png")
        else:
            ppo.test(dim=args.dim, use_gpu=args.gpu)

    elif model == 'qrdqn':
        if args.train:
            df = qrdqn.train(dim=args.dim, use_gpu=args.gpu, hpc=args.hpc)
            if args.save_fig:
                save_dir = "src/models/qr_dqn_model"

                plt.figure(figsize=(10, 6))
                plt.plot(df["global_episode"], df["r"], alpha=0.4, label="Episode reward")
                rolling = df["r"].rolling(window=100, min_periods=1).mean()
                plt.plot(df["global_episode"], rolling, color="black", linestyle="--", linewidth=1.5, label="100-episode avg")
                plt.xlabel("Episode")
                plt.ylabel("Reward")
                plt.title("QR-DQN Training Reward Curve")
                plt.legend()
                plt.grid()
                plt.savefig(f"{save_dir}/training_curve_dim{args.dim}.png")
                plt.close()
                print(f"Training figure saved to {save_dir}/training_curve_dim{args.dim}.png")

                eval_file = f"{save_dir}/evaluations.npz"
                if Path(eval_file).exists():
                    ev = np.load(eval_file)
                    timesteps, results = ev["timesteps"], ev["results"]
                    mean_r = results.mean(axis=1)
                    std_r = results.std(axis=1)
                    plt.figure(figsize=(10, 6))
                    plt.plot(timesteps, mean_r, marker="o", label="Mean eval reward")
                    plt.fill_between(timesteps, mean_r - std_r, mean_r + std_r, alpha=0.2)
                    plt.xlabel("Timestep")
                    plt.ylabel("Reward")
                    plt.title("QR-DQN Evaluation Reward Curve")
                    plt.legend()
                    plt.grid()
                    plt.savefig(f"{save_dir}/eval_curve_dim{args.dim}.png")
                    plt.close()
                    print(f"Eval figure saved to {save_dir}/eval_curve_dim{args.dim}.png")
        else:
            qrdqn.test(dim=args.dim, use_gpu=args.gpu)

    else:
        raise ValueError(f"Unknown model type: {model}. Please choose 'ppo' or 'qrdqn'.")
