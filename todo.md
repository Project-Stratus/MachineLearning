- [ ] Unify drag model to use relative velocity: drag should be proportional to `(balloon_vel - wind_vel)^2` rather than the current setup where wind is an additive external force and drag opposes absolute velocity independently. Current approach works at low wind speeds but will matter as wind fields become more realistic.
- [ ] Altitude-dependent temperature model: replace isothermal atmosphere (`T_AIR = 288.15K` constant) with a temperature profile that decreases ~6.5K/km in the troposphere and levels off in the stratosphere. Affects density, pressure, buoyancy, and drag across the 0–21km operating range.
- [ ] Volume-dependent cross-sectional area and drag coefficient: `AREA` and `CD` are currently fixed constants. Both should vary with balloon volume — frontal area scales with radius (from volume), and CD varies with Reynolds number. Inflate/deflate actions currently have no aerodynamic side-effect.
- [ ] Upgrade integrator from forward Euler to symplectic (e.g. velocity Verlet): at `DT=1.0s` Euler is adequate for gentle manoeuvres but accumulates energy error and can become unstable during rapid altitude changes or strong wind shear transitions.
- [ ] Passive gas expansion/compression with altitude: balloon volume currently only changes via agent inflate/deflate actions. Real balloon gas expands as ambient pressure drops with altitude (ideal gas law), which is central to superpressure and zero-pressure balloon behaviour.
- [ ] Variable balloon mass: `MASS = 2.0` is constant. Real balloons lose mass through gas venting/leakage and ballast drops. Relevant if the action space is eventually expanded to include gas management.

## Training performance
- [ ] Enable GPU training: `USE_GPU = False` in `qrdqn.py`. Essential now that network is `[512, 512, 256]` with 51 quantiles (~440K params, ~5x previous). CPU can't keep up as architecture scales toward Loon.
- [ ] Vectorised environments for QR-DQN: PPO uses `SubprocVecEnv` but QR-DQN runs a single env. Add 4-8 parallel envs to increase data throughput.
- [ ] Consider increasing `train_freq` (currently 4) to 8-16 to reduce gradient updates per env step, trading sample efficiency for wall-clock speed.

## Project-level goals
- [ ] Develop an accurate 3-D virtual environment for training. This environment would need to simulate real-world atmospheric conditions and feed the agent with data in the same format as it would receive from a real balloon.
- [ ] Develop an RL model which can control the balloon by increasing or decreasing altitude. Justify our RL framework based on the problem and literature and create an agent with a high performance in the virtual environment.
- [ ] Use real world data. Take advantage of publicly available flight data to improve training and validation.
