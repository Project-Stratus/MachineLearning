# Project Stratus MachineLearning Repo
Here is the Project Stratus repository for developing and maintaining our RL agent for controlling an inflatable balloon with the aim of station-keeping it around a target location.

## Objective
_Build an RL agent which can autonomously control an atmospheric balloon. The agent's goal is to station-keep around a target location by increasing or decreasing its altitude and taking advantage of wind currents._

### Why?
Station-keeping high-altitude balloons have several key advantages over drones, satellites, and ground-based systems:
- Persistent, low-cost aerial platforms that can loiter for weeks or months
- Large-scale wireless connectivity to remote and rural areas
- Rapid-response imagery and communications for natural disaster sites
- Short/medium-range weather forecasting from the stratosphere
- They're cool

## How it works
The agent interacts with a Gym-compatible 3D balloon environment at each timestep. It observes its position, velocity, altitude, ambient pressure, wind vector, and distance to a target location — all normalised to [0, 1]. It then chooses one of three discrete actions: **inflate**, **deflate**, or **do nothing**, which adjusts the balloon's buoyancy and therefore its altitude. By changing altitude, the agent moves into different wind layers and exploits wind currents to navigate toward the target.

Training uses [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3). Vectorised environments collect rollouts in parallel, and the agent (PPO or QR-DQN) optimises a composite reward that balances proximity to the goal, approach direction, and penalties for crashes or leaving bounds. The best checkpoint is saved automatically when evaluation reward improves.

## Getting started
### Prerequisites
- Python 3.11 (via `conda`, `pyenv`, or system install)
- Recent pip and build tooling:
  - `python -m pip install -U pip hatchling build`
  - A C/C++ toolchain (Linux: `build-essential`; macOS: Xcode CLT; Windows: MSVC Build Tools)
- Optional (for `gpu` extra): NVIDIA driver compatible with CUDA 12.4 and CUDA 12.4 runtime

### Set up your workspace
1. Clone the repository and `cd MachineLearning`
2. Create an isolated environment:
   - Conda: `conda create -n Stratus python=3.11`
   - venv: `python -m venv .venv`
3. Activate it:
    - Conda: `conda activate Stratus`
    - venv (bash/zsh): `source .venv/bin/activate`

### Install from pyproject
Run ONE of these options depending on your intention:
- Core runtime: `pip install -e .`
- Development tooling (recommended): `pip install -e .[dev]`
- GPU-enabled training stack: `pip install -e .[dev,gpu]`

### Smoke tests
1. Run install check: `python tests/check_install.py --build --pip-check`
2. Run unit/integration tests: `pytest`

**For contribution guidelines and PR expectations, see `CONTRIBUTING.md`.**

### Pre-commit hooks
We use `black` (formatting) and `ruff` (linting) enforced via pre-commit hooks. Set them up once after cloning:
```bash
pre-commit install
```
Hooks run automatically on `git commit`. To run them manually across the whole repo:
```bash
pre-commit run --all-files
```

### Training & Running
```bash
python main.py ppo --train --dim 1           # Train PPO in 1D
python main.py qrdqn --train --dim 2         # Train QR-DQN in 2D
python main.py ppo --dim 3                   # Test PPO in 3D (no --train = inference)
python main.py ppo --train --dim 1 --save_fig  # Train and save reward curve plot
python main.py qrdqn --train --dim 3 -g      # Train QR-DQN on GPU
python main.py qrdqn --train --dim 3 -g --hpc  # GPU training, no progress bar (for SLURM jobs)
```

### Accessing TensorBoard during training
#### Local training
1. Run `tensorboard --logdir ./src/models/ --port 6006` in your terminal during training to view logs for all agents.
2. Open `http://localhost:6006` in your browser.

#### Monitoring cluster (SLURM) training
TensorBoard logs are written to `src/models/<agent>_model/` as normal when training with `--hpc`. To monitor from your laptop, periodically sync the logs and run TensorBoard locally:
```bash
# Pull logs from the cluster (run periodically or with watch):
rsync -avz <user>@<cluster>:/path/to/MachineLearning/src/models/ ./src/models/

# Then view locally:
tensorboard --logdir ./src/models/ --port 6006
```
TensorBoard picks up new event files as they arrive, so re-running the rsync is enough to refresh.

## Basic repo layout:
- `EDA/`: exploratory analyses, notebooks, and supporting scripts for balloon data
- `src/agents/`: reinforcement-learning agents (PPO, QR-DQN) and their training/eval logic
- `src/environments/`: gym-compatible Balloon3D environment, physics core, renderers, rewards
- `src/models/`: persisted checkpoints and training artefacts
- `tests/`: pytest suite
  - `core/`: atmosphere, balloon, reward, wind field tests
  - `envs/`: environment and observation tests
  - `agents/`: agent utility tests
  - `integration/`: episode integration tests

## Prior work: Google Loon
[Loon](https://en.wikipedia.org/wiki/Loon_LLC) was a Google/Alphabet project (2011-2021) with the same core goal — station-keeping high-altitude balloons using wind currents. Over its lifetime the project accumulated 218 flight-years and 127 million telemetry data points, demonstrating that RL-based autonomous navigation in the stratosphere is feasible at scale. Loon provided connectivity to remote regions including rural Kenya and disaster-hit Puerto Rico before being shut down in 2021.

Their flight dataset is publicly available and is a key resource for validating our environment and training on real atmospheric conditions:

https://zenodo.org/records/5119968#.YVNdiGZKio5

## Testing / CI / CD
- Run the whole suite locally with `pytest`; this mirrors the GitHub Actions workflow.
- Scope to a module while iterating, e.g. `pytest tests/envs/ -k balloon`.
- Ensure new features include coverage in `tests/` and update fixtures when interfaces change.
- Record any longer training validations in your PR description (see `CONTRIBUTING.md`).
- CI validates pull requests with the same commands; keep failures red/green before requesting review.
