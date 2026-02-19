- [x] Unify drag model to use relative velocity: drag proportional to `(balloon_vel - wind_vel)^2`. Wind passed as velocity vector, not force. Balloon drifting with wind experiences zero drag.
- [x] Altitude-dependent temperature model: ISA lapse rate (6.5 K/km troposphere, constant stratosphere). Pressure uses barometric formula. Density, buoyancy, and drag all altitude-aware.
- [x] Volume-dependent cross-sectional area and drag coefficient: frontal area derived from sphere geometry, CD from Morrison (2013) Reynolds-number correlation. Includes Sutherland's law for dynamic viscosity.
- [x] Upgrade integrator from forward Euler to velocity Verlet (symplectic, second-order). Two force evaluations per step with density recomputed at the updated position. Better energy conservation and stability at DT=1.0s.
- [x] Passive gas expansion/compression with altitude: balloon tracks gas moles, volume derived via ideal gas law (V = nRT/P). Gas expands/compresses automatically with altitude changes.
- [x] Variable balloon mass: total mass = payload + ballast + gas (n_gas × M_HE). Actions swapped from inflate/deflate to drop ballast (ascend) and vent gas (descend) — both irreversible. `mass` is now a property recomputed each step.

## Training performance
- [ ] (Low priority) Increase `train_freq` (currently 4) to 8-16 to reduce gradient updates per env step. Trades sample efficiency for wall-clock speed — not worth doing unless training time becomes a bottleneck again, since GPU and vectorised envs already address the main performance issues.

## Next phase
- [ ] Integrate weather VAE: use a variational autoencoder to generate realistic, diverse wind fields for training instead of hand-crafted patterns.
- [ ] Swap wind conditions for sensor readings: replace direct wind vector in the observation space with simulated sensor data (e.g. GPS drift, pressure tendency) to close the sim-to-real gap.

## Project-level goals
- [ ] Develop an accurate 3-D virtual environment for training. This environment would need to simulate real-world atmospheric conditions and feed the agent with data in the same format as it would receive from a real balloon.
- [ ] Develop an RL model which can control the balloon by increasing or decreasing altitude. Justify our RL framework based on the problem and literature and create an agent with a high performance in the virtual environment.
- [ ] Use real world data. Take advantage of publicly available flight data to improve training and validation.
