# Reward Design & RL Incentives for Balloon Station-Keeping

Why a simple reward function works better than a complex one, and how
DQN/QR-DQN agents are inherently motivated by temporal difference learning.

---

## 1. What the Agent Actually Optimises

A DQN agent does not optimise per-step reward. It learns the **Q-function** --
the expected discounted return from taking action `a` in state `s` and then
following the optimal policy:

```
Q(s, a) = E[ r_0 + gamma * r_1 + gamma^2 * r_2 + ... ]
```

where `gamma` (discount factor) controls how much the agent values future
reward. With `gamma = 0.995`, the sum of an infinite stream of reward 1.0
per step converges to:

```
V = 1.0 / (1 - 0.995) = 200.0
```

This means the agent values *being in a good state* (one that leads to high
future reward) far more than any single-step bonus we could design.

---

## 2. Temporal Difference Learning Creates the Gradient

DQN learns via the Bellman update:

```
Q(s, a) <-- r + gamma * max_a' Q(s', a')
```

This is the key mechanism. Consider three states at different distances from
the station-keeping target:

- State A: inside station (reward = 1.0/step)
- State B: 5 minutes from station (reward ~ 0.2/step)
- State C: 15 minutes from station (reward ~ 0.05/step)

Through repeated Bellman updates, the Q-values propagate backward:

1. State A has high Q because it directly receives 1.0 reward per step.
2. State B has moderately high Q because `gamma * Q(A)` is large -- being
   near high-value states is itself valuable.
3. State C has lower Q, but the gradient toward B is encoded in
   `gamma * max Q(s', a')`.

**The agent doesn't need an explicit "direction bonus" to know that moving
toward the station is good.** The Q-function already encodes this: every
state closer to the goal has higher Q because it's nearer to the 1.0/step
reward stream. TD learning propagates this signal backward automatically.

---

## 3. Why Shaping Rewards Hurt More Than They Help

### 3.1 Redundant signals add noise

A direction reward says "reward for getting closer." But gamma-discounting
already does this. Adding a shaping reward creates a second gradient that
may not perfectly align with the optimal policy. The agent must now learn to
balance two competing signals instead of following one clean one.

### 3.2 Composite rewards create local optima

A reward with 5+ components (distance, direction, reached, survival, action
cost) creates a complex optimisation landscape. In practice:

- The agent discovered that collecting a survival bonus (+0.1/step) while
  doing nothing was a local optimum, because the survival signal dominated
  the diluted distance gradient.
- With a single signal (1.0 inside station, decaying outside, 0.0 on death),
  there is essentially one gradient: get to the station and stay there.

### 3.3 Action costs distort the optimal policy

Penalising inflate/deflate actions tells the agent "the optimal policy uses
fewer actions." This is not necessarily true -- the optimal station-keeping
strategy may involve constant small adjustments (Google's real Loon balloons
adjusted altitude frequently). An action cost biases toward passivity, which
is a *different objective* than station-keeping.

### 3.4 Normalisation interacts badly with domain size

A linear distance reward normalised by `max_distance` (domain diagonal)
dilutes the gradient as the domain grows. When XY_MAX went from 10km to
50km, the distance gradient weakened by ~4x, making it smaller than the
survival bonus. Exponential decay avoids this entirely -- the gradient is
proportional to the current reward value at every scale.

---

## 4. How Survival and Resource Conservation Emerge Naturally

### Survival

The agent does not need an explicit survival bonus because dying means
termination, which forfeits all future reward. With `gamma = 0.995` and
an in-station reward of 1.0/step:

- Agent at step 1000 with 4000 steps remaining: ~200 expected discounted
  future reward.
- Agent that dies at step 1000: 0 future reward.

The Q-function in early-episode states encodes the full future reward
stream. States with high termination probability have lower Q, naturally
discouraging risky behaviour.

### Resource conservation (helium/ballast)

No explicit "conserve helium" reward is needed. Running out of helium
causes termination, which forfeits all future reward. The Q-function
learns that low-helium states have lower value because they carry higher
termination risk. The agent conserves helium not because we told it to,
but because wasting helium reduces expected future return.

This is a general principle: **any failure mode that causes termination
is automatically penalised by the loss of future reward, proportional to
how much episode remains.**

---

## 5. The Perciatelli Reward Function

Google's Balloon Learning Environment uses a reward function that
demonstrates these principles (named after their Perciatelli44 agent):

```
Inside station_radius:   reward = 1.0   (flat, no gradient to center)
Outside station_radius:  reward = dropoff * 2^(-(distance - radius) / halflife)
On termination:          reward = 0.0   (forfeit all future reward)
```

### Why flat inside the radius

Once the balloon is "close enough," there is no benefit to incentivising
it to get closer to the exact center. The flat reward avoids unnecessary
oscillation and lets the agent focus on *staying* within the radius rather
than chasing a moving optimum.

### Why exponential decay outside

The gradient of exponential decay is proportional to the function value:

```
d/dx [A * e^(-kx)] = -kA * e^(-kx)
```

This means the signal stays meaningful at every distance:

| Distance from station | Reward | Gradient    |
|-----------------------|--------|-------------|
| Inside radius         | 1.0    | --          |
| Just outside          | 0.40   | Steep       |
| +1 half-life          | 0.20   | Moderate    |
| +2 half-lives         | 0.10   | Meaningful  |
| +4 half-lives         | 0.025  | Fading      |

Compare with linear normalisation: gradient = 1/max_distance (constant,
tiny in large domains).

### Why termination gives 0 instead of a large negative penalty

With reward in [0, 1], a terminated state simply receives 0.0. The
"punishment" is implicit -- the agent loses all future reward it would
have collected. This is more effective than a fixed penalty because:

1. The implicit penalty scales with how much episode remains. Dying
   early is much worse than dying near the end.
2. There is no need to tune the penalty magnitude relative to per-step
   reward.
3. The reward range stays bounded and well-behaved for Q-learning.

---

## 6. QR-DQN Specifics

QR-DQN (Quantile Regression DQN) extends standard DQN by learning the
full distribution of returns rather than just the mean. Instead of
estimating `E[return]`, it estimates quantiles:

```
Standard DQN:  Q(s, a) = E[r_0 + gamma * r_1 + ...]
QR-DQN:        theta_i(s, a) = i-th quantile of the return distribution
```

This matters for balloon station-keeping because:

1. **Risk sensitivity**: The return distribution for balloon control is
   multi-modal -- sometimes the balloon reaches the station (high return),
   sometimes it gets blown out of bounds (zero return). QR-DQN captures
   both modes, while standard DQN averages them into a misleading mean.

2. **Better gradient signal**: Quantile regression uses a pinball loss
   that provides gradients even when the mean Q-value is uninformative.
   This helps learning in sparse-reward regions (far from station).

3. **Implicit risk management**: By seeing the full distribution, the
   agent can distinguish between "reliably moderate reward" and "either
   great or terrible." This naturally leads to policies that avoid
   high-variance actions (like aggressive altitude changes that might
   cause termination).

The choice of 51 quantiles (matching Google's Loon setup) provides a
good balance between distribution resolution and computational cost.

---

## 7. Summary of Design Principles

1. **Let gamma do the work.** Discounted future reward already creates
   gradients toward good states, survival incentives, and resource
   conservation. Don't add shaping rewards for things TD learning
   handles naturally.

2. **One clear signal.** A single reward objective (station proximity)
   avoids conflicting gradients and local optima from component
   interactions.

3. **Exponential decay over linear normalisation.** Maintains meaningful
   gradients regardless of domain size.

4. **Terminate, don't penalise.** Lost future reward is a stronger and
   more naturally-scaled punishment than any fixed penalty.

5. **Only add complexity for information the primary reward can't
   capture.** Google's only addition was a power penalty for real-world
   energy constraints -- a genuine operational constraint that
   station-keeping reward alone cannot encode.

---

## References

- Bellemare et al., "Autonomous navigation of stratospheric balloons
  using reinforcement learning" (Nature, 2020)
- google/balloon-learning-environment (GitHub)
- Dabney et al., "Distributional Reinforcement Learning with Quantile
  Regression" (AAAI, 2018)
