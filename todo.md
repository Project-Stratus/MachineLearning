- [ ] Unify drag model to use relative velocity: drag should be proportional to `(balloon_vel - wind_vel)^2` rather than the current setup where wind is an additive external force and drag opposes absolute velocity independently. Current approach works at low wind speeds but will matter as wind fields become more realistic.
- [x] Altitude-dependent temperature model: ISA lapse rate (6.5 K/km troposphere, constant stratosphere). Pressure uses barometric formula. Density, buoyancy, and drag all altitude-aware.
- [x] Volume-dependent cross-sectional area and drag coefficient: frontal area derived from sphere geometry, CD from Morrison (2013) Reynolds-number correlation. Includes Sutherland's law for dynamic viscosity.
- [ ] Upgrade integrator from forward Euler to symplectic (e.g. velocity Verlet): at `DT=1.0s` Euler is adequate for gentle manoeuvres but accumulates energy error and can become unstable during rapid altitude changes or strong wind shear transitions.
- [x] Passive gas expansion/compression with altitude: balloon tracks gas moles, volume derived via ideal gas law (V = nRT/P). Gas expands/compresses automatically with altitude changes.
- [ ] Variable balloon mass: `MASS = 2.0` is constant. Real balloons lose mass through gas venting/leakage and ballast drops. Relevant if the action space is eventually expanded to include gas management.

## Training performance
- [ ] (Low priority) Increase `train_freq` (currently 4) to 8-16 to reduce gradient updates per env step. Trades sample efficiency for wall-clock speed — not worth doing unless training time becomes a bottleneck again, since GPU and vectorised envs already address the main performance issues.

## Next phase
- [ ] Integrate weather VAE: use a variational autoencoder to generate realistic, diverse wind fields for training instead of hand-crafted patterns.
- [ ] Swap wind conditions for sensor readings: replace direct wind vector in the observation space with simulated sensor data (e.g. GPS drift, pressure tendency) to close the sim-to-real gap.

## Project-level goals
- [ ] Develop an accurate 3-D virtual environment for training. This environment would need to simulate real-world atmospheric conditions and feed the agent with data in the same format as it would receive from a real balloon.
- [ ] Develop an RL model which can control the balloon by increasing or decreasing altitude. Justify our RL framework based on the problem and literature and create an agent with a high performance in the virtual environment.
- [ ] Use real world data. Take advantage of publicly available flight data to improve training and validation.
