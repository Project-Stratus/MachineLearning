"""Pytest unit tests for the core balloon environment and helpers."""

import math
import numpy as np
import pytest

from environments.core.atmosphere import Atmosphere
from environments.core.balloon import Balloon
from environments.envs.balloon_3d_env import Balloon3DEnv
from environments.core.reward import balloon_reward


# ---------------------------------------------------------------------
# Atmosphere
# ---------------------------------------------------------------------
def test_atmosphere():
    atm = Atmosphere()

    p0 = atm.pressure(0.0)
    rho0 = atm.density(0.0)
    assert 9e4 < p0 < 1.05e5          # Pa
    assert 1.0 < rho0 < 1.4           # kg/m³

    assert atm.pressure(10_000.0) < p0
    assert atm.density(10_000.0) < rho0


# ---------------------------------------------------------------------
# Balloon
# ---------------------------------------------------------------------
def test_balloon_buoyancy():
    atm = Atmosphere()
    balloon = Balloon(atmosphere=atm)

    buoy = balloon.buoyant_force(0.0)[-1]
    weight = -balloon.weight()[-1]
    assert abs(buoy - weight) / weight < 0.05

    before = balloon.buoyant_force(0.0)[-1]
    balloon.inflate(0.2 * balloon.stationary_volume)
    after = balloon.buoyant_force(0.0)[-1]
    assert after > before


def test_balloon_update_integrates():
    atm = Atmosphere()
    balloon = Balloon(atmosphere=atm)

    z0 = balloon.altitude
    balloon.inflate(0.1 * balloon.stationary_volume)
    balloon.update(1.0)
    assert balloon.altitude >= z0


# ---------------------------------------------------------------------
# Balloon3DEnv (multi-dimensional)
# ---------------------------------------------------------------------
def _expected_obs_size(dim: int) -> int:
    # goal(d) + volume(1) + position(d) + delta(d) + velocity(d) + pressure(1) + wind(d)
    return 5 * dim + 2


@pytest.fixture(params=[1, 2, 3])
def balloon_env(request):
    dim = request.param
    env = Balloon3DEnv(dim=dim, render_mode=None, config={"time_max": 50})
    try:
        yield env, dim
    finally:
        env.close()


def test_balloon3d_reset_shapes(balloon_env):
    env, dim = balloon_env
    obs, info = env.reset(seed=123)

    assert obs.shape == (_expected_obs_size(dim),)
    assert np.all(np.isfinite(obs))

    assert info["TimeLimit.truncated"] is False
    assert info["terminal_observation"] is None


def test_balloon3d_single_step(balloon_env):
    env, dim = balloon_env
    obs0, _ = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(1)  # index 1 => no volume change

    assert obs.shape == obs0.shape == (_expected_obs_size(dim),)
    assert math.isfinite(reward)
    assert isinstance(terminated, (bool, np.bool_))
    assert isinstance(truncated, (bool, np.bool_))

    components = info.get("reward_components")
    assert components is not None
    for key in ("distance", "direction", "reached", "total"):
        assert key in components
        assert np.isfinite(components[key])
    assert math.isclose(components["total"], reward, rel_tol=1e-5, abs_tol=1e-7)


def test_balloon3d_episode_finishes(balloon_env):
    env, _ = balloon_env
    env.reset(seed=999)
    done = False
    for _ in range(60):
        _, _, term, trunc, _ = env.step(env.action_space.sample())
        if term or trunc:
            done = True
            break
    assert done, "Episode should terminate via crash or time limit."


def test_balloon3d_reward_direction_positive():
    pos = np.array([10.0])
    goal = np.array([0.0])
    vel = np.array([-1.5])  # moving toward the goal
    total, components, updated = balloon_reward(
        balloon_pos=pos,
        goal_pos=goal,
        velocity=vel,
        dim=1,
        terminated=False,
        punishment=-5.0,
        prev_distance=12.0,
        success_radius=5.0,
        success_speed=0.2,
        direction_scale=0.05,
    )

    assert updated == pytest.approx(np.linalg.norm(pos - goal))
    assert components["direction"] >= 0.0
    assert math.isfinite(total)


def test_balloon_reward_crash_returns_punishment():
    pos = np.array([0.0, 0.0, 0.0])
    goal = np.array([0.0, 0.0, 10.0])
    vel = np.zeros(3)
    total, components, updated = balloon_reward(
        balloon_pos=pos,
        goal_pos=goal,
        velocity=vel,
        dim=3,
        terminated=True,
        punishment=-5.0,
        prev_distance=15.0,
        success_radius=5.0,
        success_speed=0.2,
        direction_scale=0.05,
    )

    assert total == -5.0
    assert components == dict(distance=-5.0, direction=0.0, reached=0.0)
    assert updated == pytest.approx(np.linalg.norm(pos - goal))
