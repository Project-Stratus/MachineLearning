# Project Stratus MachineLearning Repo
Here is the Project Stratus repository for developing and maintaining **[name TBC]**, our RL agent for controlling an inflatable balloon with the aim of station-keeping it around a target location.

## Objective
_Build an RL agent which can autonomously control an atmospheric balloon. The agent's goal is to station-keep around a target location by increasing or decreasing it's altitude and taking advantage of wind currents._

### Why?
Station-keeping high-altitude balloons have several key advantages over drones, satellites and ground-based systems. They can:
- Provide large-scale wireless connectivity to rural areas.
- Provide rapid response imagery/connectivity to sites of natural disasters
- Short notice weather forecasting
- They're cool

### Tasks
From an ML perspective, there are several core areas of the project to be developed and enhanced.
- Develop an accurate 3-D virtual environment for training. This environment would need to simulate real-world atmospheric conditions and feed the agent with data in the same format as it would receive from a real balloon.
- Develop an RL model which can control the balloon by increasing or decreasing altitude. Justify our RL framework based on the problem and literature and create an agent with a high performance in the virtual environment.
- Use real world data. Take advantage of publicly available flight data to improve training and validation.

## Getting started
### Prerequisites
- Python 3.11 (via `conda`, `pyenv`, or system install)
- `pip` 23+ and a working C/C++ toolchain (build-essential / Xcode CLT / MSVC)
- Optional: NVIDIA driver + CUDA 12.4 runtime for the `gpu` extra

### Set up your workspace
1. `git clone <git@github.com:Project-Stratus/MachineLearning.git>` (SSH)
2. `cd MachineLearning`
3. Create an isolated environment
   - Conda: `conda create -n Stratus python=3.11`
   - venv: `python -m venv .venv`
4. Activate it (`conda activate Stratus` or `source .venv/bin/activate`)

### Install from pyproject
- Core runtime only: `pip install -e .`
- Development tooling (recommended): `pip install -e .[dev]`
- GPU-enabled training stack: `pip install -e .[dev,gpu]`
- Legacy scripts that still consume `requirements.txt` can run `pip install -r requirements.txt`; it resolves to the editable install above.

### Smoke tests
- Run unit/integration tests: `pytest`
- Check inference: `python main.py ppo`

**For contribution guidelines and PR expectations, see `CONTRIBUTING.md`.**

## Basic repo layout:
- `agents/`: reinforcement-learning agents (PPO, DQN) and their training/eval logic
- `EDA/`: exploratory analyses, notebooks, and supporting scripts for balloon data
- `environments/`: gym-compatible Balloon3D environment, physics core, renderers, rewards
- `models/`: persisted checkpoints and training artefacts
- `tests/`: pytest suite covering atmosphere, balloon physics, environment, and rewards

## Loon data:
Data below is flight data from _Loon_, a Google project with the similar goal of using high-altitude balloons. Below is a comparison of data recorded by the Loon and Stratus teams. Loon recorded flights between 2011 and 2021, a total of 218 flight-years and 127 million telemetry points.

| Value                      | Loon | Stratus |
|----------------------------|------|---------|
| flight_id                  | ✔    | X      |
| time                       | ✔    | ✔      |
| latitude                   | ✔    | ✔      |
| longitude                  | ✔    | ✔      |
| altitude                   | ✔    | ✔      |
| temperature                | ✔    | ✔      |
| pressure                   | ✔    | ✔      |
| earth_ir                   | ✔    | X      |
| earth_ir_sensor_config     | ✔    | X      |
| acs                        | ✔    | X      |
| propeller_on               | ✔    | X      |

<br>
Data:

https://zenodo.org/records/5119968#.YVNdiGZKio5

## Testing / CI / CD
- Run the whole suite locally with `pytest`; this mirrors the GitHub Actions workflow.
- Scope to a module while iterating, e.g. `pytest tests/envs_test.py -k balloon`.
- Ensure new features include coverage in `tests/` and update fixtures when interfaces change.
- Record any longer training validations in your PR description (see `CONTRIBUTING.md`).
- CI validates pull requests with the same commands; keep failures red/green before requesting review.
