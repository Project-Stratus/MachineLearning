# Papers

Reference PDFs kept in-repo (not linked out) so they stay available offline and
byte-identical for anyone — human or AI tool — reading the codebase. See
`CLAUDE.md` for how Loon/this literature should be framed: prior art for
validation, not the thing being replicated.

- **Bellemare et al., "Autonomous navigation of stratospheric balloons using
  reinforcement learning"** (*Nature*, 2020) — Google Loon's RL flight
  controller for superpressure balloons, validated with a 39-day controlled
  experiment over the Pacific; the main prior-art reference for "RL station-
  keeping works at scale."
- **Corum & Watrous, "A Comparative Analysis of Reinforcement Learning
  Methods for Stratospheric Balloon Station Keeping"** (project report) —
  benchmarks PPO, DQN, discrete SAC, and QR-DQN against Loon's own
  controllers in the Balloon Learning Environment; useful as a sanity check
  for which algorithms are known to struggle (discrete SAC, QR-DQN) versus
  perform adequately (PPO, DQN) on this task family.
- **Saunders et al., "Resource-Constrained Station-Keeping for Helium
  Balloons using Reinforcement Learning"** — soft actor-critic control for
  vent/ballast (zero-pressure-style) balloons rather than air-pump
  superpressure balloons, with an explicit resource-consumption objective —
  closest prior art to this project's ZP focus and consumption-penalty reward.
- **Xu, Liu, Du & Lv, "Station-keeping for high-altitude balloon with
  reinforcement learning"** (*Chinese Journal of Aeronautics*, 2022) —
  dueling double Q-learning with prioritized experience replay, trained
  against a wind field built from historical real wind data rather than a
  synthetic model.
- **Du, Lv, Li, Zhu, Zhang & Wu, "Station-keeping performance analysis for
  high altitude balloon with altitude control system"** (*Aerospace Science
  and Technology*, 2019) — non-RL, physics/simulation-based analysis of how
  much station-keeping endurance an altitude control system buys in a real
  wind field; useful as a non-learned baseline perspective.
