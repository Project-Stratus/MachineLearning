# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Project Stratus is a reinforcement learning framework for training autonomous agents to control high-altitude balloons for station-keeping. Agents learn to adjust balloon altitude through inflation/deflation, using wind currents to maintain position near target locations.

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
python main.py ppo --train --dim 1    # Train PPO in 1D
python main.py qrdqn --train --dim 2  # Train QR-DQN in 2D
python main.py ppo --dim 3            # Test PPO in 3D (no --train = inference)
```

### Code Quality
Pre-commit hooks enforce `black` and `ruff`. Run `pre-commit install` to set up.

## Architecture

```
src/
├── agents/              # RL implementations (PPO, DQN, QR-DQN) wrapping Stable-Baselines3
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
tensorboard --logdir ./src/models/ppo_model/
```
