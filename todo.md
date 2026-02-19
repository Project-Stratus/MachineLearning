## Known simplifications
- [ ] Altitude-dependent gas temperature: replace constant T_BALLOON (20°C) with a model that accounts for ambient cooling, solar heating, and time-of-day effects. Currently the gas is assumed 20°C at all altitudes, which overestimates volume at high altitude (ambient ~-57°C in the stratosphere).
- [ ] (Low priority) Add vertical wind component: the wind field currently has no vertical component (fz = 0). Stratospheric vertical winds are small but non-zero; adding them would improve realism.
- [ ] (Low priority) Recompute volume at Verlet half-step: during integration, density is recomputed at the updated altitude but volume (V = nRT/P) is not. For DT=1s the error is negligible, but recomputing would make the two force evaluations fully consistent.
- [ ] (Low priority) Extend ISA beyond two layers: the atmosphere model covers the troposphere and stratosphere only. Adding the mesosphere and above would allow operations beyond ~50 km, but is unnecessary for the current ~25 km ceiling.

## Training performance
- [ ] (Low priority) Increase `train_freq` (currently 4) to 8-16 to reduce gradient updates per env step. Trades sample efficiency for wall-clock speed — not worth doing unless training time becomes a bottleneck again, since GPU and vectorised envs already address the main performance issues.

## Next phase
- [ ] Integrate weather VAE: use a variational autoencoder to generate realistic, diverse wind fields for training instead of hand-crafted patterns.
- [ ] Swap wind conditions for sensor readings: replace direct wind vector in the observation space with simulated sensor data (e.g. GPS drift, pressure tendency) to close the sim-to-real gap.

## Project-level goals
- [ ] Develop an accurate 3-D virtual environment for training. This environment would need to simulate real-world atmospheric conditions and feed the agent with data in the same format as it would receive from a real balloon.
- [ ] Develop an RL model which can control the balloon by increasing or decreasing altitude. Justify our RL framework based on the problem and literature and create an agent with a high performance in the virtual environment.
- [ ] Use real world data. Take advantage of publicly available flight data to improve training and validation.
