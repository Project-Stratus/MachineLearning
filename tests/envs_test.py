"""
Pytest unit tests for the environments in the `environments.envs` package.
"""

import math
import pytest
import numpy as np

from environments.core.atmosphere import Atmosphere
from environments.core.balloon import Balloon
from environments.envs.old.balloon_1d_env import Balloon1DEnv
from environments.envs.old.balloon_2d_env import Balloon2DEnv


# ---------------------------------------------------------------------
# Atmosphere
# ---------------------------------------------------------------------
def test_atmosphere():
    atm = Atmosphere()

    # Sea-level numbers are in a reasonable range.
    p0 = atm.pressure(0.0)
    rho0 = atm.density(0.0)
    assert 9e4 < p0 < 1.05e5          # Pa
    assert 1.0 < rho0 < 1.4           # kg/m³

    # Pressure and density should decrease with altitude.
    assert atm.pressure(10_000.0) < p0
    assert atm.density(10_000.0) < rho0


# ---------------------------------------------------------------------
# Balloon
# ---------------------------------------------------------------------
def test_balloon_buoyancy():
    atm = Atmosphere()
    b = Balloon(atmosphere=atm)

    # At t=0 the buoyant force roughly balances the weight
    # (because stationary_volume ≈ mass / rho_air by construction).
    buoy = b.buoyant_force(0.0)[-1]     # upward (+)
    weight = -b.weight()[-1]            # make it positive
    assert abs(buoy - weight) / weight < 0.05

    # Inflating should *increase* buoyant force.
    before = b.buoyant_force(0.0)[-1]
    b.inflate(0.2 * b.stationary_volume)
    after = b.buoyant_force(0.0)[-1]
    assert after > before


def test_balloon_update_integrates():
    atm = Atmosphere()
    b = Balloon(atmosphere=atm)

    z0 = b.altitude
    b.inflate(0.1 * b.stationary_volume)
    b.update(1.0)                      # integrate 1 second (1-D signature)
    assert b.altitude >= z0            # should not fall immediately


# ---------------------------------------------------------------------
# Balloon1DEnv — basic API checks
# ---------------------------------------------------------------------
@pytest.fixture(scope="module")
def env_1d():
    """Create a single env instance and clean it up after all tests."""
    env = Balloon1DEnv(render_mode=None)
    yield env
    env.close()


def test_env_reset_returns_valid_obs(env_1d):
    obs, info = env_1d.reset(seed=123)

    # Right keys?
    expect_keys = {"goal", "volume", "altitude", "velocity", "pressure"}
    assert expect_keys == set(obs.keys())

    # All values are 1-D arrays in [0,1] or [-1,1] (velocity).
    for k, v in obs.items():
        assert v.shape == (1,)
        if k == "velocity":
            assert -1.0 <= v[0] <= 1.0
        else:
            assert 0.0 <= v[0] <= 1.0

    # Info dict has the standard monitor keys.
    assert "TimeLimit.truncated" in info
    assert "terminal_observation" in info
    assert info["terminal_observation"] is None


def test_env_single_step(env_1d):
    env_1d.reset(seed=0)
    a = env_1d.action_space.sample()
    obs, reward, terminated, truncated, info = env_1d.step(a)

    # Observation structure same as after reset.
    assert set(obs.keys()) == {"goal", "volume", "altitude", "velocity", "pressure"}

    # Reward is finite.
    assert math.isfinite(reward)

    # Terminated / truncated are booleans.
    assert isinstance(terminated, (bool, np.bool_)), f"Expected bool, got {type(terminated)}: {terminated}"
    assert isinstance(truncated, (bool, np.bool_)), f"Expected bool, got {type(truncated)}: {truncated}"
    assert isinstance(info, dict), f"Expected dict, got {type(info)}: {info}"


def test_env_runs_until_done(env_1d):
    """Run the environment until it signals the end of an episode."""
    env_1d.reset(seed=42)
    for _ in range(Balloon1DEnv.TIME_MAX + 5):    # definitely enough steps
        _, _, term, trunc, _ = env_1d.step(env_1d.action_space.sample())
        if term or trunc:
            break
    assert term or trunc, "Episode did not finish within TIME_MAX steps"


# ---------------------------------------------------------------------
# Balloon2DEnv — basic API checks
# ---------------------------------------------------------------------
@pytest.fixture(scope="module")
def env_2d():
    env = Balloon2DEnv()          # render_mode=None by default
    yield env
    env.close()


# Reset
def test_2d_reset_shape(env_2d):
    obs, info = env_2d.reset()
    assert obs.shape == (22,), "Observation should be 22-dim flat vector"
    # First four entries are x, y, vx, vy.
    x, y, vx, vy = obs[:4]
    assert np.isfinite([x, y, vx, vy]).all()


# One step
def test_2d_single_step(env_2d):
    env_2d.reset()
    a = env_2d.action_space.sample()
    obs, reward, terminated, truncated, info = env_2d.step(a)

    # Observation OK?
    assert obs.shape == (22,)
    # Reward finite (should be <= 0 except when exactly on target).
    assert np.isfinite(reward)
    # done is bool or numpy.bool_
    assert isinstance(terminated, (bool, np.bool_))
    assert isinstance(info, dict)


# Local-wind helper
def test_2d_local_wind_size(env_2d):
    env_2d.reset()
    wind = env_2d.get_local_wind()
    # 3×3 grid × 2 components = 18
    assert wind.shape == (18,)
    assert wind.dtype == np.float32


# Episode terminates in finite time
def test_2d_runs_until_done(env_2d):
    env_2d.reset()
    max_rollout = 5_000            # should far exceed EPISODE_LENGTH
    done = False
    for _ in range(max_rollout):
        _, _, terminated, truncated, _ = env_2d.step(env_2d.action_space.sample())
        if terminated or truncated:
            done = True
            break
    assert done, "Episode should finish within EPISODE_LENGTH steps"
