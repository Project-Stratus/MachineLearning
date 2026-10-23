# Onboarding: reinforcement learning from scratch

Four notebooks that teach the basics of reinforcement learning (RL) with the
project's own balloon physics. You don't need any RL experience. You do need
some Python: functions, loops, and a little NumPy.

The notebooks don't try to cover the whole codebase. They cover the essentials
the project is built on, using one small task throughout: holding a balloon at
a target altitude.

| # | Notebook | What you'll learn | Time |
|---|---|---|---|
| 1 | [The RL loop and Gymnasium](01_rl_loop_and_gymnasium.ipynb) | What RL is, which library does what, Gymnasium's `reset`/`step` API, and how the balloon behaves | ~1 h |
| 2 | [Defining the objective](02_build_an_environment.ipynb) | Building your own environment, why the observation needs velocity, and testing a reward before training | ~1½ h |
| 3 | [Q-learning by hand](03_q_learning.ipynb) | How an agent learns from reward alone: exploration, the discount γ, the learning rate α, and evaluation | ~1½ h |
| 4 | [Neural networks and Stable-Baselines3](04_neural_networks_and_sb3.ipynb) | Q-networks in PyTorch, DQN and PPO with SB3, TensorBoard, and how it all maps onto Stratus | ~1½ h |

Do them in order. Each one stands on its own, though, so you can pick up at any
notebook.

## Setup

1. Install the project by following **Getting started** in the
   [main README](../README.md). The `dev` extra includes Jupyter.
2. Make your own branch from the latest `main`:

   ```bash
   git switch main
   git pull
   git switch -c onboarding/<your-name>
   ```

3. Open the notebooks, either with
   - **JupyterLab:** run `jupyter lab` from the repo root and open `onboarding/`, or
   - **VS Code:** open a notebook and pick the project's Python environment as
     the kernel.

## How the exercises work

- Cells marked ✏️ are exercises. Replace the `raise NotImplementedError` line
  with your own code.
- Most exercises are followed by a check that prints ✅ when you've got it right
  and ❌ when you haven't. Plots and tables tell you the rest.
- Sections marked 🚀 are optional stretch goals. They're open-ended and have no
  model answers.
- Full answers are in [`solutions/`](solutions/). Try the exercise first, but
  don't stay stuck for long: reading a solution and working out why it works
  is learning too.
- [`toy_env.py`](toy_env.py) holds the task, the finished environment and the
  small helpers the notebooks share.

## Saving your work

Commit to your branch after each notebook:

```bash
git add onboarding/01_rl_loop_and_gymnasium.ipynb
git commit -m "Onboarding notebook 1"
git push -u origin onboarding/<your-name>   # just `git push` after the first time
```

If the commit fails with `files were modified by this hook`, that's the
pre-commit hooks stripping the notebook's outputs (or tidying whitespace).
Run the same `git add` and `git commit` again and it will go through.

Onboarding branches are for your own practice and don't get merged, so there's
no need to open a pull request.

## For maintainers

The solution notebooks are the single source of truth. To change a notebook,
edit its copy in `solutions/`, then regenerate the exercise copies:

```bash
python onboarding/make_exercises.py
```

In the solutions, wrap answer code in `# BEGIN SOLUTION` / `# END SOLUTION`
and answer text in `<!-- BEGIN SOLUTION -->` / `<!-- END SOLUTION -->`. Tag a
cell `solution` to leave it out of the exercise copy entirely.
`make_exercises.py` explains the rules in full.

The notebooks run a few short training runs (the longest is about a minute).
After changing `toy_env.py` or the balloon physics, run the solution notebooks
from top to bottom to check that the ✅ checks still pass.
