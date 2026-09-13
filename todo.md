# Near-term tasks

Small, tactical work items. For sequencing and the staged development plan, see
[`notes/development_roadmap.md`](notes/development_roadmap.md) — this file tracks
individual chores, the roadmap tracks capability layers. Where an item belongs to
a layer, it is cross-referenced.

## Known simplifications
- [x] ~~Altitude-dependent gas temperature: replace constant T_BALLOON (20°C)~~ — done in §3.6. `T_BALLOON = 293.15 K` is gone; gas temperature is now `T_ambient(z) + SUPERHEAT_DAY` (15 K). The full radiative model is still Layer 2 §4.4.
- [ ] Ambient temperature is still ISA, which §3.10 measured at **+16.1 K too warm** against tropical flight data — a bigger error than the superheat offset it carries. Blocked on open question §9.5 (launch latitude); the fix is Layer 2 §4.2's reanalysis profiles. *Roadmap: §3.10.*
- [ ] (Low priority) Add vertical wind component: the wind field currently has no vertical component (fz = 0). Stratospheric vertical winds are small but non-zero; adding them would improve realism. *Roadmap: Layer 2 §4.2.*
- [ ] (Low priority) Recompute volume at Verlet half-step: during integration, density is recomputed at the updated altitude but volume (V = nRT/P) is not. For DT=1s the error is negligible, but recomputing would make the two force evaluations fully consistent.
- [ ] (Low priority) Extend ISA beyond two layers: the atmosphere model covers the troposphere and stratosphere only. Adding the mesosphere and above would allow operations beyond ~50 km, but is unnecessary for the current ~25 km ceiling. *Roadmap: Layer 2 §4.2 (superseded if we move to reanalysis profiles).*

## Training performance
- [ ] (Low priority) Increase `train_freq` (currently 4) to 8-16 to reduce gradient updates per env step. Trades sample efficiency for wall-clock speed — not worth doing unless training time becomes a bottleneck again, since GPU and vectorised envs already address the main performance issues.

## Next phases
Superseded by [`notes/development_roadmap.md`](notes/development_roadmap.md).
The three items previously listed here map onto the layered plan as:

- Weather VAE → **Layer 2** (§4.1)
- Sensor readings in place of the true wind vector → **Layer 4** (§6.1)
- Real-world flight data → **Layer 1** §3.10 (validate the physics core against the
  Loon flight CSV) and **Layer 2** §4.3 (geographic and seasonal diversity)

Do not add forward-looking plans here — put them in the roadmap so sequencing
stays in one place.
