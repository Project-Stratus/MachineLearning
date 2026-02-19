"""Shared pytest fixtures for the Project Stratus test suite."""

import numpy as np
import pytest

from environments.core.atmosphere import Atmosphere
from environments.core.balloon import Balloon
from environments.core.wind_field import WindField
from environments.envs.balloon_3d_env import Balloon3DEnv


# -----------------------------------------------------------------------------
# Atmosphere fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def atmosphere():
    """Fresh Atmosphere instance with default parameters."""
    return Atmosphere()


# -----------------------------------------------------------------------------
# Balloon fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def balloon_1d(atmosphere):
    """1D balloon at default altitude."""
    return Balloon(dim=1, atmosphere=atmosphere, position=[15_000.0])


@pytest.fixture
def balloon_3d(atmosphere):
    """3D balloon at default position."""
    return Balloon(dim=3, atmosphere=atmosphere, position=[0.0, 0.0, 15_000.0])


@pytest.fixture(params=[1, 2, 3])
def balloon_any_dim(request, atmosphere):
    """Parametrized balloon for all dimensions."""
    dim = request.param
    if dim == 1:
        pos = [15_000.0]
    elif dim == 2:
        pos = [0.0, 0.0, 15_000.0]  # 2D internally uses 3D
    else:
        pos = [0.0, 0.0, 15_000.0]
    return Balloon(dim=3 if dim == 2 else dim, atmosphere=atmosphere, position=pos), dim


# -----------------------------------------------------------------------------
# Wind field fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def wind_field():
    """Wind field with default parameters."""
    return WindField(
        x_range=(-2000.0, 2000.0),
        y_range=(-2000.0, 2000.0),
        z_range=(0.0, 30000.0),
        cells=10,
        pattern="sinusoid",
        default_mag=10.0,
    )


# -----------------------------------------------------------------------------
# Environment fixtures
# -----------------------------------------------------------------------------
@pytest.fixture(params=[1, 2, 3])
def env_any_dim(request):
    """Parametrized environment for all dimensions."""
    dim = request.param
    env = Balloon3DEnv(dim=dim, render_mode=None, config={"time_max": 100})
    yield env, dim
    env.close()


@pytest.fixture
def env_1d():
    """1D environment."""
    env = Balloon3DEnv(dim=1, render_mode=None, config={"time_max": 100})
    yield env
    env.close()


@pytest.fixture
def env_2d():
    """2D environment."""
    env = Balloon3DEnv(dim=2, render_mode=None, config={"time_max": 100})
    yield env
    env.close()


@pytest.fixture
def env_3d():
    """3D environment."""
    env = Balloon3DEnv(dim=3, render_mode=None, config={"time_max": 100})
    yield env
    env.close()


@pytest.fixture
def env_short_episode():
    """Environment with very short time limit for quick termination tests."""
    env = Balloon3DEnv(dim=1, render_mode=None, config={"time_max": 10})
    yield env
    env.close()


# -----------------------------------------------------------------------------
# Utility fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def rng():
    """Seeded numpy random generator for reproducible tests."""
    return np.random.default_rng(seed=42)


def expected_obs_size(dim: int) -> int:
    """Calculate expected observation size for a given dimension.

    Observation layout: goal(d) + volume(1) + position(d) + delta(d) + velocity(d) + pressure(1) + wind(d) + ballast(1) + gas(1)
    """
    return 5 * dim + 4
