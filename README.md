# MachineLearning
Here is the Project Stratus repository for developing and maintaining **[name TBC]**, our RL agent for controlling an inflatable balloon with the aim of station-keeping it around a target location.

# Goal
_Build an RL agent which can autonomously increase and decrease its altitude, taking advantage of different wind currents, to station-keep around a target position._ 

# Purpose
- Provide wireless connectivity to rural areas.
- Provide rapid response imagery/connectivity to sites of natural disasters
- Weather forecasting
- It's cool

# Tasks
- Develop a virtual environment for training.
    - Accurate 3-D weather conditions
    - Environment variables delivered to model similarly to how onboard electronics would. 
- Develop an RL model which can control the balloon by increasing or decreasing altitude.
    - Justify our RL framework based on the problem and literature

## Getting started
- Setup the conda environment using either `requirements.txt` or `pyproject.toml`.
- To run an existing model, run `main.py` with one of the following arguments:
    - `ppo` for the 1-D altitude model.
    - `dqn` for the 2-D x-y model.
- The `--train` flag can be used to retrain either of these models.

## Loon data:
Data below is flight data from _Loon_, a Google project with the similar goal of using high-altitude balloons. Below is a comparison of values recorded by the Loon and Stratus teams. Loon recorded flights between 2011 and 2021, a total of 218 flight-years and 127 million telemetry points.

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
Data:<br>

https://zenodo.org/records/5119968#.YVNdiGZKio5


## Testing & CI
Currently tested via github actions:
- Atmosphere class
    1. Basic init, pressure and density
- Balloon class

    2. Bouyancy
    3. Inflate, update
- Balloon1DEnv class

    4. Reset 
    5. Step
    6. Runs to completion
- Balloon2DEnv class

    7. Reset
    8. Step
    9. Local wind size
    10. Runs to completion