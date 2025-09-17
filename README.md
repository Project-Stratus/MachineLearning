# Project Stratus MachineLearning Repo
Here is the Project Stratus repository for developing and maintaining **[name TBC]**, our RL agent for controlling an inflatable balloon with the aim of station-keeping it around a target location.

## Goal
_Build an RL agent which can autonomously increase and decrease the altitude of an atmospheric balloon, taking advantage of different wind currents, to station-keep around a target position._

## Purpose
- Provide wireless connectivity to rural areas.
- Provide rapid response imagery/connectivity to sites of natural disasters
- Weather forecasting
- It's cool

## Tasks
- Develop a virtual environment for training.
    - Accurate 3-D weather conditions
    - Environment variables delivered to the agent similarly to how onboard electronics would. 
- Develop an RL model which can control the balloon by increasing or decreasing altitude.
    - Justify our RL framework based on the problem and literature.
- Use real world data
  - Take advantage of publicly available flight data to improve training.

## Getting started
- Setup the conda environment using either `requirements.txt` or `pyproject.toml`.
- Set python version to 3.11
- To run an existing model, run `main.py` with one of the following arguments:
    - `ppo`
    - `dqn` (not currently in use)
- The `--train` or `-t` flag can be used to retrain either of these models.

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
Currently tested via github actions or manually by running `pytest` in home directory:<br>

**Atmosphere class**<br>
  - Test 1: Basic init, pressure and density

**Balloon class**<br>
  - Test 2: buoyancy balance
  - Test 3: Inflate/update integration

**Balloon3DEnv (used for 1D/2D/3D projections)**<br>
  - Tests 4–6: reset/observation validation across dim ∈ {1,2,3}
  - Tests 7–9: single-step dynamics and reward-component logging
  - Tests 10–12: episode termination via crash or time limit

**Reward shaping helpers (`balloon_reward`)**<br>
  - Test 13: directional progress reporting
  - Test 14: crash punishment handling
