# CLAUDE.md

## Project Overview

Project Stratus is a reinforcement learning framework for training autonomous agents to control high-altitude balloons for station-keeping. Agents learn to adjust balloon altitude through inflation/deflation, using wind currents to maintain position near target locations.

## Project Scope & Constraints

Amateur-scale project — **not** a Loon reimplementation. These constraints drive design decisions:

- **12-hour missions** (`TIME_MAX = 43_200` physics steps = 720 decisions), any window including full night. Day/night thermal lands at the end of Layer 2; Layer 1 is daytime-only via a constant `SUPERHEAT_DAY` offset above ambient.
- **Zero-pressure is the primary platform**; small superpressure comes later. Both models live in `balloon.py`, but ZP is the focus and SP is untuned — it is kept green in tests, not optimised.
- **Ballast and helium budgets bind.** The reward carries a multiplicative consumption penalty (roadmap §3.3). The omega is held across the whole decision interval — do not "simplify" it to fire only on the actuating sub-step, which dilutes it 60× and makes it shape nothing.
- **Hardware will fly during this project.** Observations must be constructible from real onboard sensors. `volume_norm` is in the observation and has no sensor behind it — a deliberate Layer 4 stub (roadmap §6.1).
- Station-keeping geometry is still inherited from a scaled-down Loon rather than derived from our mission — an open question (roadmap §9). 3D uses a 10 km radius / 20 km half-life; 1D uses 500 m / 1 km, because reusing the horizontal scales made every 1D state a full-reward state and all baselines scored TWR 1.000.

**Layer 1 is implemented.** Held-out baselines (meta-seed 2026, **12** scenarios, ZP) — regenerate with `python main.py --benchmark --dim N`:

| policy | 1D | 2D | 3D |
| --- | --- | --- | --- |
| passive drift | 0.008 | 0.025 | 0.037 |
| random | 0.009 | 0.025 | 0.037 |
| greedy_wind | 0.008 | 0.025 | **0.126** |
| bang_bang (1D only) | **0.919** | — | — |

The bar is **dim-specific** and `qrdqn.baseline_reference(dim)` is the source of truth — it raises rather than defaulting, because plotting the 3D bar on a 1D curve flatters a bad agent by ~100x. `greedy_wind` only separates from passive in 3D (3.4x); in 1D and 2D it ties with doing nothing, and in 2D it also dies early (ep len 625/720) spending consumables for no gain.

**1D is a debugging mode, not a test.** A two-parameter bang-bang holds station 92% of the time, so a 1D result says nothing about whether the agent learned anything. Judge Layer 1 on 3D against greedy_wind 0.126.

Google Loon is used as **prior art for validation**, not as the thing being replicated. When citing it, frame it as evidence a design works at scale rather than as the reason to adopt it.

## Development Roadmap

Work is staged into five layers, each independently trainable and measurable before the next begins: deterministic basics → deterministic weather (day/night at the end) → uncertainty → hardware → deferred long-duration SP. See `notes/development_roadmap.md`.

**Before changing the observation space, read roadmap §2.1.** The layout is meant to be frozen early with later-layer fields present but stubbed at constants (uncertainty channels at 0, solar phase fixed, sensor-error indicators at 0), so that crossing a layer boundary costs a retrain of weights rather than a re-architecture.

Also worth knowing (roadmap §2.2): Layers 1–2 are deterministic, so the return distribution from any state-action pair is a point mass and QR-DQN's quantiles carry no information. Distributional RL earns nothing until Layer 3 — don't tune `n_quantiles` before then, and don't read a null result there as a bug.

## Commands

### Installation
```bash
pip install -e .[dev]        # Development (recommended)
pip install -e .[dev,gpu]    # GPU-enabled training
```

### Testing
```bash
pytest                           # Full test suite
pytest tests/envs_test.py -k balloon  # Scoped testing
python tests/check_install.py --build --pip-check  # Smoke test
```

### Training & Running
```bash
python main.py --train --dim 1    # Train QR-DQN in 1D
python main.py --train --dim 2    # Train QR-DQN in 2D
python main.py --dim 3            # Test QR-DQN in 3D (no --train = inference)
```

### Code Quality
Pre-commit hooks enforce `black` and `ruff`. Run `pre-commit install` to set up.

## Architecture

```
src/
├── agents/              # QR-DQN agent wrapping sb3-contrib
├── environments/
│   ├── core/            # Physics engine (balloon, atmosphere, wind_field, reward, jit_kernels)
│   ├── envs/            # Gym-compatible Balloon3DEnv (supports 1D/2D/3D via dim parameter)
│   ├── render/          # Pygame visualization
│   └── wrappers/        # Gym wrappers for action/observation modifications
└── models/              # Trained model checkpoints
```

### Key Design Patterns

- **Single environment for all dimensions**: `Balloon3DEnv` handles 1D/2D/3D via `dim` constructor parameter
- **Action space**: Discrete(3) - inflate, nothing, deflate
- **Reward composition**: `balloon_reward()` returns `(total, components_dict, new_distance)` for debugging
- **Agent configs**: Module-level dicts (TRAIN_CFG, POLICY_KWARGS) rather than config files
- **Physics acceleration**: Numba JIT (`@njit`) in `jit_kernels.py`
- **Parallel training**: `SubprocVecEnv` / `DummyVecEnv` for vectorized environments

### Environment Registration
```python
# Registered in src/environments/__init__.py
gymnasium.make("environments/Balloon3D-v0", dim=3)
```

### Imports
Use relative imports within `src/` subpackages. There is intentionally no top-level `src/__init__.py`.

## Tensorboard
Logs stored in `./src/models/<agent>_model/`. View with:
```bash
tensorboard --logdir ./src/models/qr_dqn_model/
```
