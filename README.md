# MachineLearning
Here is the Project Stratus repository for developing and maintaining [name TBC], our RL agent for controlling an inflatable balloon with the aim of station-keeping it around a target location.
---
### Goals
- Develop a balloon object (Currently operational in 2D)
- Develop an RL model which can control said balloon object by increasing or decreasing altitude (Currently operational in 1D)
- Develop a realistic amtospheric environment to train the model in (Not implemented yet)

### Getting started
etc etc

### Loon data:
https://zenodo.org/records/5119968#.YVNdiGZKio5


### Testing
**Actions not active yet**
Currently tested in github actions:
- Atmosphere class
    - Basic init, pressure and density
- Balloon class
    - Bouyancy
    - Inflate, update
- Balloon1DEnv class
    - Create instance
    - Reset 
    - Step
    - Runs to completion
- Balloon2DEnv class
    - Create instance
    - Reset
    - Step
    - Local wind size
    - Runs to completion