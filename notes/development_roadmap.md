# Development Layers: Stratus Capability Roadmap

What we are building, in five layers ordered by implementation and validation
cost. Each layer adds one class of functionality and is independently
trainable and measurable before the next begins.

Google Loon is used throughout as prior art — a check that a design decision has
already been shown to work at scale — rather than as the organising principle.
Each item leads with our rationale; the `Prior art:` lines are supporting
evidence.

Sources:
- Paper: [Autonomous navigation of stratospheric balloons using reinforcement learning](https://www.nature.com/articles/s41586-020-2939-8), Nature 588, 77–82 (2020). [Full text PDF](https://gwern.net/doc/reinforcement-learning/model-free/2020-bellemare.pdf)
- Code: [google/balloon-learning-environment](https://github.com/google/balloon-learning-environment) (archived Sept 2025)
- Docs: [BLE environment reference](https://balloon-learning-environment.readthedocs.io/en/latest/environment.html)

---

## 0. Project Frame

| Constraint | Value | Consequence |
|---|---|---|
| Flight duration | 12 hours | No multi-day power budgeting, battery fatigue, or helium-loss ageing |
| Target flight window | Any 12h window, including full night | Diurnal thermal is in scope — but lands at the end of Layer 2, not in Layer 1 |
| Primary platform | Zero-pressure, then small superpressure | Irreversible resource control is the core problem |
| Scale | Amateur, small envelopes | Loon's ~1,800 m³ / ~100 kg is not the target |
| Consumables | Ballast + helium budget is binding | Rationing is a first-class control problem |
| Hardware | Flying during this project | Sensor and actuator realism get their own layer |

Two framing points that cut across every layer:

1. **Our hardest problem is not Loon's hardest problem.** Theirs was overnight
   *power* management on a reversible actuator. Ours is overnight *thermal*
   collapse on an irreversible one. Different physics, same structural shape: a
   scarce resource spent against an uncertain future.
2. **Layers 1–2 are deterministic by construction.** Worth knowing what that
   implies for the agent — see §2.2.

---

## 1. The Five Layers

| Layer | Adds | Exit criterion |
|---|---|---|
| **1 — Basic usable** | Trainable agent on analytic winds, daytime-only, full information | Beats passive-drift and random baselines on a frozen held-out scenario set, measured in TWR |
| **2 — Weather** | Real wind fields and atmosphere, still fully known. Day/night thermal at the end | Same benchmark, real winds; policy survives a sunset without crashing |
| **3 — Uncertainty** | Agent no longer sees truth: partial observability, forecast error, measurement noise | TWR degrades gracefully vs Layer 2 rather than collapsing; agent demonstrably explores |
| **4 — Hardware** | Sensor models, actuator realism, latency, airspace limits, SP envelope state | Simulated flight matches a real flight trace within tolerance |
| **5 — Long-duration SP** | Battery, solar power, ACS power, ageing, fleet | Not addressed |

Layers 1–2 are deterministic. Layer 3 is where the environment becomes genuinely
stochastic. Layer 4 is where it becomes ours specifically.

---

## 2. Cross-Cutting Decisions

Two things that must be settled in Layer 1 because retrofitting them is
expensive.

### 2.1 Freeze the observation layout in Layer 1, stub the later fields

This is what makes the layering pay off rather than costing three full retrains.

Changing the observation width changes the network input dimension, which
invalidates every checkpoint. So the *final* observation layout should be
decided in Layer 1, with fields that later layers will populate present from the
start and pinned to constants:

| Field | Layer 1 | Populated by |
|---|---|---|
| Wind column (N levels above/below, altitude-centred) | Exact values from the analytic field | — |
| Per-level uncertainty channel | Pinned to 0 (perfect knowledge) | Layer 3 |
| Solar elevation, sin/cos diurnal phase | Pinned to a fixed daytime value | Layer 2 (end) |
| Safety flags (at altitude limit, etc.) | Live from Layer 1 | — |
| Resource fractions | Live from Layer 1 | — |
| Sensor-error indicators | Pinned to 0 | Layer 4 |

The wind column is the important case. **Layer 1 gives the agent a *perfect*
column; Layer 3 degrades it to a *believed* one.** The layout is identical —
only the content changes — so Layer 3 costs a retrain of weights, not a
re-architecture.

This also resolves the apparent tension between "wind column" and "partial
observability" being in different layers. They are different things: the column
is observation *structure* (deterministic, Layer 1); partial observability is
observation *degradation* (stochastic, Layer 3).

### 2.2 Distributional RL does nothing until Layer 3

Worth knowing so nobody is surprised by a null result. With deterministic
dynamics, the return distribution from any state-action pair is a **point mass**
— every quantile collapses to the same value. Through Layers 1 and 2, QR-DQN is
DQN with 51× the output width and no information gain.

This is not an argument to change algorithm. The architecture is right for where
we are going, and switching later would cost a retrain. But it does mean:
- Do not expect distributional RL to show a benefit in Layer 1–2 benchmarks.
- `n_quantiles` is not worth tuning until Layer 3.
- Layer 3 is where the quantile spread becomes a usable signal — it is a direct
  readout of how much the agent thinks it does not know.

*Prior art:* Loon measured only a +1.2% TWR50 gain from going distributional,
and that was in a fully stochastic setting. The gain is real but small; the
value is more in the uncertainty readout than the score.

---

## 3. Layer 1 — Basic Usable Version (Deterministic, Daytime)

**Goal:** a fully trainable agent that performs well on simple analytic winds,
daytime only, with complete information and no stochastic dynamics.

**Exit criterion:** beats passive-drift and random baselines on a frozen
held-out scenario set, measured in time-within-radius.

> **Status: implemented, not yet trained.** Everything in this section is in the
> code and the suite is green (748 tests). The exit criterion is not yet met
> because no training run has been done — that is the next action, not
> outstanding work.
>
> Measured held-out baselines (meta-seed 2026, 12 scenarios, ZP), which are the
> bar a trained agent has to clear. Regenerate with
> `python main.py --benchmark --dim N`; `qrdqn.baseline_reference(dim)` is the
> in-code copy and deliberately raises for an unmeasured dim rather than falling
> back to the 3D row.
>
> | policy | 1D TWR | 2D TWR | 3D TWR |
> | --- | --- | --- | --- |
> | passive drift | 0.008 | 0.025 | 0.037 |
> | random | 0.009 | 0.025 | 0.037 |
> | greedy wind | 0.008 | 0.025 | **0.126** |
> | hand-rolled bang-bang (1D only) | **0.919** | — | — |
>
> GreedyWind separating 3.4x from passive in 3D is the evidence that the wind
> column carries exploitable signal. It is the only dim where that holds: in 1D
> and 2D it ties with passive to three decimal places, and in 2D it is actively
> worse than doing nothing — same TWR, but it terminates at 625/720 decisions,
> so it spends a consumable to buy nothing.
>
> **1D is now a debugging mode rather than a test.** `BangBangAgent`
> (`src/agents/baselines.py`) is two parameters — a switching deadband and a
> velocity lead time — and holds station 92% of the time. The earlier 0.136
> figure recorded here was an undertuned version of the same controller, not a
> different one; sweeping its lead time on the *training* seeds moved it from
> 0.17 to 0.92, and it generalises (0.923 train, 0.919 held out). A 1D agent
> that scores 0.5 has not half-solved the problem, it has lost to a heuristic.
> **Judge Layer 1 on 3D.**
>
> That the lead time optimises at ~1200 s — twenty decision intervals — is worth
> carrying into §3.7 and §6.3: it is a measurement of how slowly this platform
> answers a 0.01 kg impulse, and it suggests the 60 s decision interval is much
> finer than the plant's response time.

### 3.1 Evaluation framework — build this first

Layer 1's exit criterion is "performs well", which is currently unmeasurable.
`EvalCallback` runs the default 5 episodes on a single eval env built from the
*same* config as training (`src/agents/qrdqn.py:188-193`) — same wind pattern,
same everything — and scores summed reward. No held-out set, no train/test
split, no baselines.

Needed:
- A **time-within-radius metric**. Interpretable, comparable across layers, and
  it transfers to hardware unchanged.
- A **frozen held-out scenario set** with parameters disjoint from training.
- **Baselines**: passive drift, uniform random, and a simple heuristic
  (greedy-best-wind altitude selection). Without these there is no floor to
  compare against.

Symptom of the current gap: `_REWARD_THRESHOLD = 83_000` against a max of 86,400
is an early-stop at **~96% time-within-radius**.

*Prior art:* Loon's benchmark was 6,000 flights across 5 locations × 100 dates ×
12 seeds, training years excluded, stratified into 5 difficulty bands. Their
best controller scored 55.1% TWR50 against an estimated ceiling of ~57%; the
heuristic baseline scored 40.5%, passive drift and random far below. A 96%
threshold tells us the environment is trivial, not that the agent is good.

### 3.2 Episode and reward calibration

- **`time_max` 86,400 → 43,200** (24h → 12h).
- **Discount horizon.** γ = 0.995 at a 60 s decision interval gives a ~3.3 h
  effective horizon against a 12 h episode. Needs raising, and re-checking
  whenever the decision interval changes. *Prior art:* Loon used γ = 0.993 at
  180 s ≈ 7.1 h against 2-day episodes.
- **Station radius, decay half-life, and box size.** Currently 10 km / 20 km /
  ±50 km, inherited from a scaled-down Loon rather than derived from our
  mission. See §8.1.
- **Terminating walls.** Exiting the ±50 km box ends the episode, so drifting
  far is fatal rather than merely bad. This makes recovery behaviour unlearnable
  — the agent can never discover that riding a fast unfavourable wind out and
  back is worth it. *Prior art:* Loon began station-keeping from up to 300 km
  out and never terminated on distance; sensitivity analysis showed its
  controller deliberately sought *fast* winds beyond the radius to return.

### 3.3 Resource rationing in the reward

The highest-value reward change. Ballast and helium are irreversible and the
budget binds, but there is currently no cost to venting or dropping beyond
eventual terminal death — nothing stops the agent spending its entire budget in
the first hour.

Add a graded consumption penalty. *Prior art:* Loon penalised power the same way
and found a **multiplicative** penalty (×0.95 − 0.3ω) clearly better than an
additive one, which "dominates the action-value estimates far from the station."
The mechanism transfers directly even though power itself does not.

Note this will need re-tuning at the end of Layer 2 — daytime-only flight is
stable, so rationing pressure is much weaker than it becomes once a sunset can
force a hard ballast dump to arrest a descent.

### 3.4 Altitude safety layer

Hitting ground or ceiling currently terminates the episode with reward 0.
`notes/altitude_control_instability.md` already documents pop and crash
terminations dominating training. Clamping to a permitted band instead — and
exposing "at limit" flags in the observation so the agent learns the constraint
rather than dying at it — is cheap and directly addresses that.

Also: `z_range` is 0 → ALT_MAX ≈ 40 km with float at ~20 km. Nothing prevents
flight to 40 km or into the ground. A realistic permitted band for our platform
is needed, especially with airspace constraints coming in Layer 4.

*Prior art:* Loon ran every controller inside a safety layer present in both
training and flight, confined to 5–14 kPa (≈15–20 km), with limit flags in the
observation vector.

### 3.5 Observation design

Current: 19 dims (`5·dim + 4`) — goal, volume, position, delta, velocity,
pressure, local wind, 2 resource fractions.

Changes:
- **Wind column, exact.** Sample the analytic field at N levels above and below
  the balloon. Deterministic and complete — this is added information, not
  uncertainty.
- **Altitude-centred frame.** Centre the column on the balloon's current
  altitude and drop absolute x/y. The natural symmetry is "is it better above or
  below me", and an egocentric frame expresses that directly. *Prior art:* Loon
  called this out as their key inductive bias — it "supports a simple strategy:
  ascend or descend when the winds above or below offer better returns, stay if
  they do not. Because poor decisions take the balloon to a lower-return state,
  this strategy corrects for many mistakes by design."
- **Heading to station as sin/cos** rather than raw delta.
- **Last-action encoding.**
- **Stub fields per §2.1** — uncertainty channels at 0, solar phase fixed,
  sensor-error indicators at 0.

### 3.6 Daytime superheat constant

Layer 1 is daytime-only, so a single calibrated offset suffices — but the
current value is wrong. `T_BALLOON = 293.15 K`
(`src/environments/core/constants.py:16`) against a stratospheric ambient of
216.65 K is a **+76 K** superheat. Real daytime ZP superheat is on the order of
+10 to +30 K.

Replace the fixed absolute temperature with a superheat offset *above ambient*.
That is both more accurate now and the natural hook for the radiative model at
the end of Layer 2 — the offset simply stops being constant.

This matters more than it looks: ZP volume is $V = nRT_b/P$, so a 76 K error is
a direct error in buoyancy, float altitude, and every derived control response.

### 3.7 Action semantics and actuator rates

Needs a deliberate decision rather than an inherited default.

Loon's ascend/descend/stay are *setpoints* to a closed-loop system — "ascend"
means "go to the top of the permitted band". That abstraction assumes a
**reversible** actuator. On ZP it would be actively dangerous: one command could
spend the entire ballast budget.

So ZP likely wants impulse-style actions, which is what we have
(`src/environments/wrappers/decision_interval.py:53-54`: one 0.01 kg impulse,
then 59 s of coasting). But `BALLAST_DROP` and `AIR_PUMP_RATE` at 0.01 kg per
decision are placeholders, not measurements — see §8.1. Setpoints can be
revisited for small SP in Layer 4.

### 3.8 Agent and training

| Hyperparameter | Ours | Loon (reference) |
|---|---|---|
| Algorithm | QR-DQN | QR-DQN |
| Quantiles | 51 | 51 |
| Network | [512, 512, 256] | 7 × 600 |
| Update horizon | 1-step | 5-step |
| Discount | 0.995 @ 60 s | 0.993 @ 180 s |
| Adam step size | 3e-4 constant | 2e-6 → 4e-7 over 5M updates |
| Minibatch | 512 | 32 |
| Replay | 1M | 4 × 500k = 2M |
| Actors | ~8 (`cpu_count // 2`) | 100 |
| Target update | every 10,000 steps | every 100 updates |
| Gradient updates | ~1.9e6 at n_envs=8 | 1.1e9 |

- **Momentum exploration** is the highest-value agent-side change. Per-step
  ε-greedy is nearly useless here: a single random 0.01 kg impulse barely moves
  a balloon, so random actions produce almost no macroscopic state variation.
  Sampling a target altitude and random-walking toward it explores the space
  that actually matters. *Prior art:* Loon sampled a random setpoint, perturbed
  it with Gaussian noise, and interleaved 4 h greedy / 2 h exploratory on 80% of
  trials.
- **Network depth** is gated on §3.5. Three layers may be fine for 19 dims;
  once the column lands and the input is in the hundreds, revisit. *Prior art:*
  Loon ablated this directly and found performance climbing to ~7 layers — but
  against a 1,099-dim input.
- **n-step returns** need a custom buffer; sb3-contrib 2.5.0 exposes no `n_step`
  parameter.
- **Compute** will bind once the environment stops being trivial. We are running
  ~1.9M gradient updates; Loon's comparison controllers used 300M and their
  deployed one 1.1e9.

### 3.9 Scenario diversity within the analytic family

The goal is hard-coded to the origin in 2D/3D
(`src/environments/envs/balloon_3d_env.py:387-391`), spawn is uniform in the
central 50% of the box, and both balloon types train on a single
`altitude_shear_2d` pattern. Randomisation is ±2 m/s initial velocity, ±5% gas,
≤0.5 kg ballast. Effectively one scenario, repeated.

Vary goal position, spawn, and shear parameters within the existing analytic
family. This is what makes §3.1's held-out set meaningful.

### 3.10 Validate the physics core against real flight data

**Done.** `scripts/validate_physics_vs_loon.py` scores the atmosphere and our
velocity assumptions against the Loon Q2-2021 CSV (Zenodo 5119968 — 36 MB,
`EDA/Data/`, gitignored). 825,853 samples across 18 flights, 15–20 km, which is
almost exactly our operating band. Loon flew superpressure and we fly
zero-pressure, so the *platform* numbers do not transfer — but the atmosphere is
the atmosphere, and it is the layer everything else sits on.

| Check | Result | Verdict |
|---|---|---|
| ISA pressure vs measured | −2.5% median in our band (p5..p95: −5.8%..−0.1%) | Good enough for Layer 1 |
| ISA temperature vs measured | **+16.1 K too warm** (ISA 216.65 K vs measured median 200.5 K) | Open — see below |
| Vertical rates | real p50 0.16 m/s, p95 0.59 m/s, p99.9 6.2 m/s | Prompted a fix — see below |
| Horizontal wind magnitude | p50 5.8, p95 20.9, max 48.6 m/s | `WIND_MAG_NORM = 30` validated (clips 1.6%) |

**The temperature result is the one that matters.** ISA's isothermal 216.65 K
stratosphere is a mid-latitude annual mean; Loon's Q2 flights were tropical
(~24°S in the sample) where the tropopause is far colder. The gap is +16.1 K —
*larger than `SUPERHEAT_DAY` itself*. Since `T_gas = T_ambient + SUPERHEAT_DAY`
and ZP volume is `V = nRT_g/P`, flying a tropical mission on ISA ambient would
more than double the intended superheat and inflate buoyancy with it. §3.6
removed the +76 K error from the old absolute `T_BALLOON`; this is a smaller
version of the same failure that now lives in the *ambient* model rather than
the offset. It is not fixable until we know where we fly — **open question §9.5
(launch latitude and season) now blocks §3.6 as well as §4.4**, and §4.2's move
to reanalysis profiles is the real fix.

**The vertical-rate result produced a fix.** Real stratospheric vertical motion
is slow — sub-m/s in the median. Our observation normalised `vel_z` by
`VEL_MAX = 200 m/s`, which is the *numerical runaway clamp*, not a physical
scale. Measured across held-out episodes the channel spanned |v|/VEL_MAX ≤ 0.022
with a median of exactly 0.0: roughly 1% of its range, i.e. a dead input the
network could not have learned from. Split the two roles —
`VEL_Z_OBS_NORM = 5.0 m/s` now scales the observation, `VEL_MAX` stays the
clamp — and the channel spans up to 0.89 with p95 ≈ 0.70 under active control,
without clipping. Layout width is unchanged, so this is a weight-level change
and not a §2.1 re-architecture; it was free to make before the first training
run and would have been expensive after it. All held-out baseline TWRs are
unchanged by it.

This is exactly what §3.10 was for: the check paid for itself before a single
gradient step.

---

## 4. Layer 2 — Weather (Deterministic)

**Goal:** real wind fields and atmosphere, still fully known to the agent. No
forecast error, no measurement noise — the agent sees weather truth.

**Exit criterion:** same benchmark on real winds; policy survives a sunset
without crashing.

### 4.1 Wind field generation

Already tracked in `todo.md` as the weather VAE. *Prior art:* Loon used ERA5
reanalysis modified with Perlin-style procedural noise, varying the seed to
generate unlimited scenarios; BLE ships a VAE for the same purpose.

Note the seed-varying procedural noise served two purposes for Loon —
scenario generation *and* forecast-error emulation. Only the first belongs in
this layer; the second is Layer 3.

### 4.2 Real atmosphere

- Ambient temperature and tropopause height from reanalysis rather than
  two-layer ISA.
- Extend ISA beyond two layers (`todo.md`).
- Vertical wind component — currently `fz = 0` (`todo.md`).

### 4.3 Geographic and seasonal diversity

Real launch sites and seasonal wind statistics, and a difficulty-stratified
scenario set built from them. This converts Layer 1's evaluation framework from
a sanity check into a meaningful benchmark.

*Prior art:* Loon defined "wind diversity" (opposing winds at different
altitudes), estimated 67% availability across the tropics from ~18,000
gridpoints × hourly queries 2000–2019, and used it to *reject* impossible
training scenarios — a useful idea for keeping the benchmark honest.

### 4.4 Day/night radiative thermal — end of Layer 2

The last thing in this layer, and the one that changes flight character most.

A ZP balloon's volume is $V = nRT_b/P$. At sunset $T_b$ falls toward ambient,
volume collapses, buoyancy drops, and the balloon descends hard — the classic ZP
sunset drop. With superheat pinned to the Layer 1 constant this event does not
exist in our simulator.

Needed:
- Solar elevation from a real solar-position calculation (pure astronomy —
  deterministic, no weather model or power system required).
- Radiative balance: direct solar, Earth IR, convective coupling to ambient.
- Cloud cover modulating IR and direct solar, deterministic from reanalysis.
- Un-pin the solar phase fields reserved in §2.1.

Then re-tune the resource penalty from §3.3 — a sunset that can force a hard
ballast dump changes the economics of rationing substantially.

*Prior art:* BLE's `thermal.py`.

**Keep separate from the power system.** Solar *geometry* is free astronomy and
belongs here. Solar *panels and batteries* are Layer 5. Conflating them would
block a cheap deterministic model behind an expensive one we have deferred.

---

## 5. Layer 3 — Uncertainty

**Goal:** the agent stops seeing truth. Everything it knows about the wind field
is estimated from what it has actually measured.

**Exit criterion:** TWR degrades gracefully relative to Layer 2 rather than
collapsing; the agent demonstrably probes unexplored altitudes.

### 5.1 Partial observability

A balloon measures wind only where it is and where it has been. A balloon is a
Lagrangian tracer, so its own horizontal velocity *is* the local wind — that part
stays observable. The column above and below is not.

- Restrict direct measurement to the balloon's position and wake.
- Estimate the rest of the column, blending measurements against a forecast
  prior.
- Decay confidence with time since measurement and distance from it.

*Prior art:* Loon used a Gaussian process with the ECMWF forecast as prior mean.
This is the paper's central contribution — "the nature of this partial
observability alone limits the usefulness of conventional control techniques."

### 5.2 Uncertainty as an observation

Populate the per-level uncertainty channels reserved in §2.1 with the posterior
variance of the estimate.

*Prior art:* "By treating uncertainty as an input, we avoid explicitly
enumerating plausible scenarios — a considerable computational advantage over
search methods." Loon also encoded inaccessible levels as a limit triple
(1, 1, 0), semantically "maximally confident the wind blows infinitely fast away
from the station" — a neat way to make unreachable options self-evidently bad.

### 5.3 Forecast and measurement error

- Forecast error statistics. *Prior art:* Loon reported wind-heading prediction
  errors exceeding 90° as frequent near the equator. That magnitude is what
  makes exploration mandatory rather than optional.
- Sensor measurement noise on the wind estimate (distinct from Layer 4's
  hardware-specific sensor models).
- Stochastic thermal: superheat variability, cloud variability.

### 5.4 What this layer unlocks

- **Exploration becomes strategic.** Until now, exploration was a training
  device. Here it becomes part of the optimal policy — probing an unmeasured
  altitude has genuine information value.
- **The quantile spread becomes meaningful** (§2.2). It is a direct readout of
  the agent's own uncertainty and can be used for diagnostics or risk-sensitive
  action selection.

---

## 6. Layer 4 — Hardware

**Goal:** everything that matters only because we are flying real equipment.

**Exit criterion:** a simulated flight matches a real flight trace within
tolerance.

### 6.1 Sensor model

The observation must be constructible from what is actually onboard. Auditing
the current 19 dims:

| Obs component | Measurable in flight? |
|---|---|
| Position (x, y, z) | Yes — GPS |
| Velocity | Yes — GPS derivative |
| Pressure | Yes — barometer |
| Local wind vector | Yes, implicitly — horizontal velocity ≈ local wind |
| Resource fractions | Approximately — count valve actuations |
| **Volume** | **No** |

`volume / VOL_MAX` (`src/environments/envs/balloon_3d_env.py:288`) has no sensor
behind it on a ZP balloon. Inferring it needs pressure, gas temperature and
known moles — so it depends on Layer 2's thermal model. Either drop it or
replace it with the estimate the flight computer would actually have, including
that estimate's error.

Then: sensor noise, dropout, and the error indicators stubbed in §2.1.

### 6.2 Actuator realism

Valve response time, ballast hopper discretisation, non-ideal vent and drop
behaviour, and calibrated rates replacing the §3.7 placeholders.

### 6.3 Timing and compute

- Real decision cadence. Ours is 60 s by assumption; Loon used 180 s. The EDA
  notebook was scoping environment change rates for exactly this question.
- Sensor and command latency.
- Onboard compute limits — can the policy actually run on the flight computer?

### 6.4 Airspace and recovery constraints

Hard altitude and geographic limits belong in the §3.4 safety layer rather than
as soft reward penalties, so the agent cannot trade them away.

### 6.5 Ascent phase

If the 12 h window includes ascent, that is a distinct control regime and is
currently not modelled at all. See §8.1.

### 6.6 Superpressure envelope state — when we move to SP

`BalloonSP.dynamic_volume()` returns a constant
(`src/environments/core/balloon.py:434`). To fly small SP we need the internal
differential pressure state: burst conditions, and the **zeropressure event** —
an SP balloon crossing sunset with insufficient superpressure margin degrades
into a zero-pressure balloon mid-flight and loses altitude control.

Depends on Layer 2's thermal model. Our current "popped" termination is a purely
geometric altitude ceiling, unrelated to envelope stress.

*Prior art:* BLE's `envelope_safety.py`; the BLE info dict exposes both
`envelope_burst` and `zeropressure`.

---

## 7. Layer 5 — Long-Duration Superpressure (Not Addressed)

Recorded so we recognise these when they surface, and so the parts that *do*
transfer are not deferred along with them.

- **Battery and solar power generation.** BLE's `solar.py`, `power_table.py`.
  Loon's features include battery charge and "excess energy available";
  `out_of_power` is a termination condition.
- **ACS power draw and its asymmetry.** "Pumping air into the chamber requires
  energy but releasing it does not, creating asymmetry in control dynamics."
  BLE's `acs.py` models pump efficiency against pressure ratio.
- **Power safety layer** (`power_safety.py`) and power as a deployment
  constraint — Loon required average power ≤ the incumbent's; in flight, 29 W vs
  33 W.
- **Ageing.** Helium loss and battery fatigue over months. Irrelevant at 12 h.
- **Fleet coordination.** Dispatch assigning stations, a separate approach
  controller handing off to station-keeping, and a GP ingesting measurements
  from other nearby balloons.
- **Multi-day episodes.** Loon's 2-day trials existed to force overnight
  recovery. Our 12 h window with a night in it captures that at a fraction of
  the cost.

**What transfers out of this layer anyway:**

| Idea | Lands in | Why it separates cleanly |
|---|---|---|
| Solar elevation as a clock | Layer 2 (§4.4) | Astronomy, not electronics |
| Multiplicative resource penalty | Layer 1 (§3.3) | Applies to ballast and helium instead of watts |
| Safety limits exposed as observations | Layer 1 (§3.4) | Independent of what the limit is about |

---

## 8. Sequencing Notes

**Dependencies that constrain the order:**

- Observation layout (§2.1) blocks everything — decide it before the first long
  training run.
- Resource penalty tuning (§3.3) must be revisited after day/night (§4.4).
- Volume-from-sensors (§6.1) depends on the thermal model (§4.4).
- SP envelope state (§6.6) depends on the thermal model (§4.4).
- Network depth (§3.8) is gated on the wind column landing (§3.5).

**Where retraining is unavoidable:** every layer boundary costs a retrain of
weights. §2.1 ensures none of them costs a re-architecture.

**Known re-tunes at layer boundaries:** the resource penalty at the end of
Layer 2, and the discount horizon whenever the decision interval changes.

---

## 9. Open Questions

Parameters not answerable from the codebase, roughly in the order they block
work:

1. **Mission geometry** (blocks §3.2). What station-keeping radius and operating
   box are realistic for us? 10 km / 20 km half-life / ±50 km are inherited from
   a scaled-down Loon.
2. **ZP platform sizing** (blocks §3.6). Envelope volume, payload, ballast mass,
   helium fill. `VOL_MAX = 180.6 m³`, `PAYLOAD_MASS = 2.0 kg`,
   `BALLAST_INITIAL = 5.0 kg` were chosen to make ALT_MAX come out at 40 km.
3. **Actuator rates** (blocks §3.7). What can our valve and ballast hopper
   actually do per actuation?
4. **Does the 12 h window include ascent** (blocks §6.5), or is it 12 h at float?
5. **Launch latitude and season** (blocks §4.4). Needed for solar elevation and
   realistic winds.
6. **Sensor suite** (blocks §6.1). Fixes the observable set.
7. **Real decision cadence** (blocks §6.3). What did the EDA latency work
   conclude?
8. **Airspace and recovery limits** (blocks §6.4).
