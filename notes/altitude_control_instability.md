# Altitude Control Instability in Zero-Pressure Balloons

Why pop and crash terminations dominate training, why constant vent/drop rates
are fundamentally asymmetric, and what the air-ballast alternative solves.

---

## 1. The Zero-Pressure Balloon Model

The simulation uses a **zero-pressure balloon**: helium is contained in an
elastic envelope that expands or contracts freely to equalise internal and
external pressure. The balloon volume at altitude $h$ follows the ideal gas law:

$$V(h) = \frac{n_{He} \, R \, T_b}{P(h)}$$

where $n_{He}$ is the current moles of helium, $T_b = 293.15\,\text{K}$ is the
assumed internal gas temperature, and $P(h)$ is ambient pressure. Volume and
pressure are inversely coupled — as the balloon rises and $P$ falls, $V$ grows.

The atmosphere in the stratosphere (above $h_{tropo} = 11{,}000\,\text{m}$)
follows an exponential pressure profile at constant temperature
$T_{strato} = 216.65\,\text{K}$:

$$P(h) = P_{tropo} \exp\!\left(-\frac{h - h_{tropo}}{H_s}\right), \qquad H_s = \frac{R \, T_{strato}}{M_{air} \, g} \approx 6{,}340\,\text{m}$$

So volume grows **exponentially** with altitude. The balloon has a hard ceiling
at $V_{max} = 180.6\,\text{m}^3$ — pop occurs if $V$ reaches this limit.

---

## 2. Neutral Buoyancy and Stratospheric Neutral Stability

The balloon floats when buoyancy equals total weight:

$$\rho_{air}(h) \cdot V(h) \cdot g = m_{total} \cdot g$$

Substituting the ideal gas expressions for both $\rho_{air}$ and $V$:

$$\frac{P(h) \, M_{air}}{R \, T(h)} \cdot \frac{n_{He} \, R \, T_b}{P(h)} = m_{payload} + m_{ballast} + n_{He} \, M_{He}$$

The $P(h)$ terms cancel exactly, giving:

$$\boxed{n_{He}\!\left(\frac{M_{air} \, T_b}{T(h)} - M_{He}\right) = m_{payload} + m_{ballast}}$$

In the **stratosphere**, $T(h) = T_{strato}$ is constant. The right-hand side
is fixed for given masses, and the bracket on the left is altitude-independent.
This means the neutral buoyancy condition is **the same at every stratospheric
altitude** — there is no restoring force pulling the balloon back to a
particular height. The balloon is **neutrally stable** vertically.

The practical consequence: a balloon that is even slightly over-buoyant
(too much $n_{He}$ for its mass) will rise continuously until it hits
$V_{max}$ and pops. There is no altitude at which it naturally levels off.
Conversely, a slightly under-buoyant balloon sinks until it exits the
stratosphere, enters the troposphere (where buoyancy increases with altitude
— an unstable regime), and accelerates downward toward a crash.

---

## 3. Why Constant Drop/Vent Rates Fail

The agent controls buoyancy through two irreversible discrete actions:

| Action | Effect |
|---|---|
| Drop ballast | Reduces $m_{ballast}$ by $\Delta m = 0.01\,\text{kg}$ |
| Vent gas | Removes a fixed volume $\Delta V_{vent} = 0.05\,\text{m}^3$ of helium |

### 3.1 Moles removed per vent is altitude-dependent

Venting $\Delta V_{vent}$ at altitude $h$ removes:

$$\Delta n_{He}^{vent}(h) = \frac{P(h) \cdot \Delta V_{vent}}{R \, T_b}$$

Because $P(h)$ falls exponentially with altitude, the vent action removes
**exponentially fewer moles the higher the balloon is**. The immediate
volume reduction $\Delta V_{vent} = 0.05\,\text{m}^3$ is the same at any
altitude, but the permanent reduction in $n_{He}$ — and therefore the
lasting change in buoyancy — shrinks with altitude.

### 3.2 Moles needed to compensate one ballast drop are fixed

Dropping ballast reduces the right-hand side of the neutral buoyancy equation
by $\Delta m$. The $n_{He}$ surplus that must be vented to restore balance is:

$$\Delta n_{He}^{req} = \frac{\Delta m_{ballast}}{M_{air} \, T_b / T_{strato} - M_{He}} = \frac{0.01}{0.03519} \approx 0.284\,\text{mol}$$

This is altitude-independent. The number of vent actions required to
compensate a single ballast drop is therefore:

$$N_{vents}(h) = \frac{\Delta n_{He}^{req}}{\Delta n_{He}^{vent}(h)} = \frac{\Delta m_{ballast}}{(M_{air} T_b / T_{strato} - M_{He})} \cdot \frac{R \, T_b}{P(h) \cdot \Delta V_{vent}}$$

Substituting constants:

$$N_{vents}(h) = \frac{0.284 \times 2{,}438}{P(h) \times 0.05} = \frac{692}{P(h)}$$

| Altitude | $P(h)$ (Pa) | $N_{vents}$ to offset one ballast drop |
|---|---|---|
| 15 km | 12,100 | 0.06 |
| 20 km | 5,470 | **2.5** |
| 25 km | 2,470 | **5.6** |
| 30 km | 1,120 | **12.2** |
| 35 km | 560 | **24.4** |

The balloon operates between $h_{default} \approx 19{,}550\,\text{m}$ and
$h_{max} \approx 39{,}100\,\text{m}$. As the balloon climbs toward the
ceiling, corrective vent actions become an order of magnitude less effective
than the ballast drops that caused the ascent.

### 3.3 Force asymmetry

The net vertical force change from each action is:

$$\Delta F_{ballast} = +g \cdot \Delta m_{ballast} = +0.098\,\text{N} \quad \text{(upward, altitude-independent)}$$

$$\Delta F_{vent}(h) = -\rho_{air}(h) \cdot g \cdot \Delta V_{vent}$$

| Altitude | $\rho_{air}$ (kg/m³) | $|\Delta F_{vent}|$ (N) | Ratio $\Delta F_{ballast} / |\Delta F_{vent}|$ |
|---|---|---|---|
| 20 km | 0.0880 | 0.043 | **2.3×** |
| 25 km | 0.0396 | 0.019 | **5.1×** |
| 30 km | 0.0182 | 0.0089 | **11.0×** |

At 30 km altitude, one ballast drop produces **11× more vertical force** than
one vent action. The actions are not symmetric, and the asymmetry worsens
exactly where it matters most — near the ceiling.

---

## 4. The Pop-Crash Cycle

The combination of neutral stability and action asymmetry produces a
characteristic failure cycle:

1. **Agent drops ballast** to gain altitude and reach a favourable wind layer.
   The balloon becomes over-buoyant by $\Delta n_{He}^{req} \approx 0.284\,\text{mol}$.

2. **Uncorrected over-buoyancy** causes the balloon to rise continuously
   (no restoring force in the stratosphere). Each 6,340 m of ascent
   halves the ambient pressure, doubling the remaining volume and
   halving vent authority.

3. **Agent vents** to compensate, but requires an increasing number of vent
   actions per ballast drop as it ascends. With one decision per 60 seconds,
   the balloon may climb 60–200 m per decision interval before the correction
   takes effect.

4. **At the ceiling** ($V_{max}$): episode terminates with a pop. Or the agent
   over-corrects by venting too aggressively, becomes under-buoyant, sinks
   to the troposphere, and crashes.

5. **No recovery path**: both resources (ballast and helium) are spent
   irreversibly. The agent cannot undo a poor sequence of actions.

This explains the training termination breakdown from the most recent run:

- **Popped 67.7%** — dominant; the asymmetry makes descent much harder than
  ascent, so over-buoyancy accumulates.
- **Crashed 14.2%** — over-correction or exhausted gas.
- **Completed 12.1%** — the rare case where the agent maintains near-perfect
  balance throughout the episode.

---

## 5. The Air Ballast Alternative (Project Loon Model)

Project Loon solved this with a fundamentally different design:

- The **helium envelope is superpressure** — sealed and fixed in volume.
  $V$ is constant; buoyancy changes only because $\rho_{air}$ changes with
  altitude.
- An **internal air bladder (ballonet)** can be inflated or deflated by a
  pump. Pumping air **in** increases total mass → descend. Pumping air
  **out** decreases mass → ascend.

### 5.1 What this changes physically

With fixed $V$, the buoyancy force is:

$$F_{buoy}(h) = \rho_{air}(h) \cdot g \cdot V_{fixed}$$

This still decreases with altitude (as $\rho_{air}$ falls), but now the
**control action — adding or removing air mass — has the same authority at
all altitudes**:

$$\Delta F_{air}(h) = g \cdot \Delta m_{air} \quad \text{(altitude-independent)}$$

Both ascent and descent actions produce equal and opposite force changes.
The action space is **symmetric and reversible**: air pumped in can be
pumped out again. Neither resource is permanently consumed.

### 5.2 Why the cycle breaks

| Property | Zero-pressure (Stratus) | Superpressure + air ballast (Loon) |
|---|---|---|
| Volume | Varies with $P(h)$ — grows near ceiling | Fixed |
| Ascent action | Drop ballast (irreversible) | Pump air out (reversible) |
| Descent action | Vent helium (irreversible) | Pump air in (reversible) |
| Descent authority at altitude | Degrades exponentially | Constant |
| Resource budget | Two finite consumables | Unlimited (solar-powered pump) |
| Neutral stability | Yes — no restoring force | Yes — same property |
| Over-correction cost | Permanent resource loss | Zero |

The air ballast model does not eliminate the neutral stability problem —
a superpressure balloon is also neutrally stable in the stratosphere — but
it eliminates the resource depletion and the asymmetric authority that
cause the pop-crash cycle in practice.

---

## 6. Implications for Project Stratus

The current physics model accurately represents a zero-pressure balloon, which
is the harder control problem. The 67.7% pop rate in training is not primarily
a learning failure — it reflects a genuine physical asymmetry that the agent
must overcome. Key consequences:

- **More vent actions are needed than ballast drops** to maintain balance, so
  the optimal policy is heavily biased toward venting over ballasting.
- **Training at higher altitudes is harder** than lower altitudes because vent
  authority degrades. The agent's difficulty scales with how often it is near
  $h_{max}$.
- **The problem becomes significantly more tractable** if the model is updated
  to superpressure + air ballast before scaling to realistic atmospheric data,
  since the symmetric, reversible action space will make the control policy
  learnable with fewer samples and produce a more directly transferable
  skill to real balloon systems following the Loon design.
