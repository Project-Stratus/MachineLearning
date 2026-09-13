# Initial Conditions: Simulation vs Real Life

## Real-world launch profile

A high-altitude balloon launch follows a predictable sequence:

1. **Ground fill** — the envelope is filled with helium, deliberately overfilled to produce 10-20% excess buoyancy ("free lift") above total weight.
2. **Passive ascent** — the balloon rises through the troposphere for 60-90 minutes. The pilot has limited control during this phase; the balloon simply goes up.
3. **Float arrival** — the balloon reaches its target altitude band where buoyancy roughly balances weight. For a zero-pressure balloon this happens when the envelope is fully expanded and begins venting excess gas. For a super-pressure balloon the rigid envelope constrains further expansion.
4. **Station-keeping** — the balloon adjusts altitude to exploit wind shear and maintain position near a target. This is the phase where the RL agent operates.

## Why we skip the ascent

The agent's task is station-keeping, not managing a passive ascent. Starting each training episode from the ground would mean:

- Thousands of wasted timesteps climbing through the troposphere with no meaningful control decisions.
- Diluted reward signal — the agent can't earn station-keeping reward until it reaches the operating band.
- Slower training with no benefit, since the ascent phase is deterministic and not part of the control problem.

Instead, the simulation initialises the balloon **near its float altitude at neutral buoyancy**. This is physically equivalent to "the balloon has completed its ascent and is beginning its mission." The neutral buoyancy baseline is computed exactly, accounting for the mass of the helium gas itself (not just structural mass).

## Domain randomisation for robustness

A real balloon never arrives at float altitude in a perfectly clean state. To make the agent robust to realistic variability, each episode perturbs the initial conditions around the neutral-buoyancy baseline:

### 1. Initial velocity perturbation
- **What:** each velocity component is drawn from N(0, INIT_VEL_SIGMA).
- **Why:** the balloon arrives at float with residual vertical velocity and is buffeted by turbulence. The agent must learn to damp out initial motion rather than relying on a perfectly still start.
- **Default:** sigma = 2.0 m/s.

### 2. Gas imbalance
- **What:** the neutral gas amount (n_gas) is scaled by a uniform factor in [1 - f, 1 + f] where f = INIT_GAS_FRAC_RANGE.
- **Why:** real inflation is imprecise. Slight over-inflation means the balloon starts with net upward force; under-inflation means net downward. The agent must learn to correct either direction immediately.
- **Default:** f = 0.05 (plus or minus 5%).

### 3. Ballast variation
- **What:** a random amount of ballast (up to INIT_BALLAST_LOSS_MAX kg) is subtracted from the starting ballast.
- **Why:** during ascent, operators may drop some ballast to control the ascent rate or avoid obstacles. The agent should not assume it always starts with a full ballast budget.
- **Default:** 0 to 0.5 kg lost (out of 5.0 kg initial).

### Combined effect

These three perturbations are independent and applied every episode, so the agent sees a wide variety of "arrival states." This acts as a form of domain randomisation that:

- Prevents the policy from overfitting to a single clean starting condition.
- Builds robustness to the kinds of perturbations that occur in real deployments.
- Keeps the simulation grounded in physical reality without simulating the full (uninteresting) ascent phase.

## Configuration

All randomisation parameters are defined in `src/environments/core/constants.py` under the "Initial Condition Randomisation" section and can be overridden via the environment's `config` dict if needed.
