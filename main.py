import sys
from pathlib import Path

# Add src/ to path so we can import agents and environments
sys.path.insert(0, str(Path(__file__).parent / "src"))

import agents.qrdqn as qrdqn
import argparse
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # Args
    parser = argparse.ArgumentParser(description="Train or Run a QR-DQN agent on our Loon Environment.")
    parser.add_argument('-t','--train', action='store_true', help='Train the model.')
    parser.add_argument('-b', '--benchmark', action='store_true',
                        help='Score the baselines (and the trained agent, if present) on the held-out scenario set.')
    parser.add_argument('-g', '--gpu', action='store_true', help='Use GPU for training/inference if available.')
    parser.add_argument('--hpc', action='store_true', help='HPC mode: disables progress bars for non-interactive SLURM jobs.')
    parser.add_argument('-d', '--dim', type=int, default=3, choices=[1,2,3], help='Dimensionality of the environment (1D, 2D or 3D).')
    parser.add_argument('-sf', '--save_fig', action='store_true', help='Save training figure to disk (only in train mode).')
    parser.add_argument('--n-envs', type=int, default=None, help='Number of parallel environments (overrides auto-detected default). Use to limit memory on HPC.')
    parser.add_argument('--balloon-type', type=str, default='zero_pressure',
                        choices=['zero_pressure', 'superpressure'],
                        help='Balloon physics model to train/test.')
    parser.add_argument('--no-momentum', action='store_true',
                        help='Ablation: explore with plain per-step epsilon-greedy instead of the '
                             'target-altitude random walk (roadmap §3.8).')
    parser.add_argument('--n-scenarios', type=int, default=qrdqn.N_EVAL_SCENARIOS,
                        help='Held-out scenarios to score in --benchmark mode.')
    parser.add_argument('-ts', '--timesteps', type=int, default=None,
                        help='Override the training budget (default: 15M). Use for a short '
                             'pilot run before committing to the full one; the eval cadence '
                             'compresses automatically so the pilot still tests checkpointing.')
    args = parser.parse_args()

    if args.benchmark:
        qrdqn.benchmark(dim=args.dim, balloon_type=args.balloon_type,
                        n_scenarios=args.n_scenarios, use_gpu=args.gpu)
    elif args.train:
        df = qrdqn.train(dim=args.dim, use_gpu=args.gpu, hpc=args.hpc, n_envs=args.n_envs,
                         balloon_type=args.balloon_type,
                         momentum_exploration=not args.no_momentum,
                         total_timesteps=args.timesteps)
        if args.save_fig:
            save_dir = f"src/models/qr_dqn_model/{args.balloon_type}"

            plt.figure(figsize=(10, 6))
            plt.plot(df["global_episode"], df["r"], alpha=0.4, label="Episode reward")
            rolling = df["r"].rolling(window=100, min_periods=1).mean()
            plt.plot(df["global_episode"], rolling, color="black", linestyle="--", linewidth=1.5, label="100-episode avg")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.title(f"QR-DQN Training Reward Curve ({args.balloon_type})")
            plt.legend()
            plt.grid()
            plt.savefig(f"{save_dir}/training_curve_dim{args.dim}_{args.balloon_type}.png")
            plt.close()
            print(f"Training figure saved to {save_dir}/training_curve_dim{args.dim}_{args.balloon_type}.png")

            # Written by TWREvalCallback. Plotted as TWR rather than return: the
            # reward's multiplicative resource penalty makes returns incomparable
            # between runs, TWR is fixed by the mission definition.
            eval_file = f"{save_dir}/twr_evaluations.npz"
            if Path(eval_file).exists():
                ev = np.load(eval_file)
                timesteps, twr = ev["timesteps"], ev["twr"]
                plt.figure(figsize=(10, 6))
                plt.plot(timesteps, twr, marker="o", label="Held-out TWR")
                # The bar is dim-specific: greedy_wind is the reference in 3D but
                # degenerates to passive in 1D, where bang_bang is the real floor.
                ref, bar = qrdqn.baseline_reference(args.dim)
                plt.axhline(ref[bar], color="tab:orange", linestyle="--",
                            label=f"{bar} ({ref[bar]:.3f})")
                plt.axhline(ref["passive"], color="tab:red", linestyle=":",
                            label=f"Passive drift ({ref['passive']:.3f})")
                plt.xlabel("Timestep")
                plt.ylabel("Time within radius")
                plt.ylim(0.0, 1.0)
                plt.title(f"QR-DQN Held-out TWR ({args.balloon_type})")
                plt.legend()
                plt.grid()
                plt.savefig(f"{save_dir}/eval_curve_dim{args.dim}_{args.balloon_type}.png")
                plt.close()
                print(f"Eval figure saved to {save_dir}/eval_curve_dim{args.dim}_{args.balloon_type}.png")
    else:
        qrdqn.test(dim=args.dim, use_gpu=args.gpu, balloon_type=args.balloon_type)
