# Project Stratus MachineLearning Repo
Here is the Project Stratus repository for developing and maintaining our RL agent for controlling an inflatable balloon with the aim of station-keeping it around a target location.

## Objective
_Build an RL agent which can autonomously control an atmospheric balloon. The agent's goal is to station-keep around a target location by increasing or decreasing its altitude and taking advantage of wind currents._

### Why?
Station-keeping high-altitude balloons have several key advantages over drones, satellites and ground-based systems. They can:
- Provide large-scale wireless connectivity to rural areas.
- Provide rapid response imagery/connectivity to sites of natural disasters
- Short/medium notice weather forecasting
- They're cool

### Tasks
From an ML perspective, there are several core areas of the project to be developed and enhanced.
- Develop an accurate 3-D virtual environment for training. This environment would need to simulate real-world atmospheric conditions and feed the agent with data in the same format as it would receive from a real balloon. 
- Develop an RL model which can control the balloon by increasing or decreasing altitude. Justify our RL framework based on the problem and literature and create an agent with a high performance in the virtual environment.
- Use real world data. Take advantage of publicly available flight data to improve training and validation.

## Getting started
### Prerequisites
- Python 3.11 (via `conda`, `pyenv`, or system install)
- Recent pip and build tooling:
  - `python -m pip install -U pip hatchling build`
  - A C/C++ toolchain (Linux: `build-essential`; macOS: Xcode CLT; Windows: MSVC Build Tools)
- Optional (for `gpu` extra): NVIDIA driver compatible with CUDA 12.4 and CUDA 12.4 runtime

### Set up your workspace
1. `git clone <git@github.com:Project-Stratus/MachineLearning.git>` (SSH)
2. `cd MachineLearning`
3. Create an isolated environment:
   - Conda: `conda create -n Stratus python=3.11`
   - venv: `python -m venv .venv`
4. Activate it:
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

### Training & Running
```bash
python main.py ppo --train --dim 1           # Train PPO in 1D
python main.py qrdqn --train --dim 2         # Train QR-DQN in 2D
python main.py ppo --dim 3                   # Test PPO in 3D (no --train = inference)
python main.py ppo --train --dim 1 --save_fig  # Train and save reward curve plot
```

### Accessing Tensorboard during training
1. Run `tensorboard --logdir ./src/models/ --port 6006` in your terminal during training to view logs for all agents.
2. Open `http://localhost:6006` in your browser.

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

## Progress
We have PPO and QR-DQN agents supporting 1D, 2D, and 3D environments. Current development areas:
1. Train and evaluate agents in 3D environments with realistic wind fields.
2. Develop a generative weather model and train on ECMWF data. 

## Loon data:
Data below is flight data from _Loon_, a Google project with the similar goal of using high-altitude balloons. Below is a comparison of data recorded by the Loon and Stratus teams. Loon recorded flights between 2011 and 2021, a total of 218 flight-years and 127 million telemetry points.

| Value                      | Loon | Stratus |
|----------------------------|------|---------|
| time                       | ✔    | ✔      |
| latitude                   | ✔    | ✔      |
| longitude                  | ✔    | ✔      |
| altitude                   | ✔    | ✔      |
| temperature                | ✔    | ✔      |
| pressure                   | ✔    | ✔      |
| earth_ir                   | ✔    | X      |
| earth_ir_sensor_config     | ✔    | X      |
| acs                        | ✔    | X      |
| flight_id                  | ✔    | N/A    |
| propeller_on               | ✔    | N/A    |

<br>
Data:

https://zenodo.org/records/5119968#.YVNdiGZKio5

## Testing / CI / CD
- Run the whole suite locally with `pytest`; this mirrors the GitHub Actions workflow.
- Scope to a module while iterating, e.g. `pytest tests/envs/ -k balloon`.
- Ensure new features include coverage in `tests/` and update fixtures when interfaces change.
- Record any longer training validations in your PR description (see `CONTRIBUTING.md`).
- CI validates pull requests with the same commands; keep failures red/green before requesting review.
