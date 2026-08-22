"""Shared pytest fixtures for the Project Stratus test suite."""

import numpy as np
import pytest

from environments.core.atmosphere import Atmosphere
from environments.core.balloon import Balloon, BalloonSP
from environments.core.constants import ALT_DEFAULT, OBS_WIDTH
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
#: Short episodes keep the suite fast; the wind pattern is the one the agents
#: actually train on, so tests exercise the configuration that matters.
ENV_TEST_CONFIG = {"time_max": 100, "wind_pattern": "altitude_shear_2d"}


def make_env(dim: int, **overrides) -> Balloon3DEnv:
    """Construct a test env: short episode, training wind pattern, no render."""
    return Balloon3DEnv(dim=dim, render_mode=None,
                        config={**ENV_TEST_CONFIG, **overrides})


@pytest.fixture(params=[1, 2, 3])
def env_any_dim(request):
    """Parametrized environment for all dimensions."""
    env = make_env(request.param)
    yield env, request.param
    env.close()


@pytest.fixture
def env_1d():
    """1D environment."""
    env = make_env(1)
    yield env
    env.close()


@pytest.fixture
def env_2d():
    """2D environment."""
    env = make_env(2)
    yield env
    env.close()


@pytest.fixture
def env_3d():
    """3D environment."""
    env = make_env(3)
    yield env
    env.close()


@pytest.fixture
def env_short_episode():
    """Environment with very short time limit for quick termination tests."""
    env = make_env(1, time_max=10)
    yield env
    env.close()


# -----------------------------------------------------------------------------
# SP Balloon fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def balloon_sp_1d(atmosphere):
    """1D SP balloon at default altitude."""
    return BalloonSP(dim=1, atmosphere=atmosphere, position=[ALT_DEFAULT])


@pytest.fixture
def balloon_sp_3d(atmosphere):
    """3D SP balloon at default position."""
    return BalloonSP(dim=3, atmosphere=atmosphere, position=[0.0, 0.0, ALT_DEFAULT])


@pytest.fixture(params=[1, 2, 3])
def env_sp_any_dim(request):
    """Parametrized SP environment for all dimensions."""
    env = make_env(request.param, balloon_type="superpressure")
    yield env, request.param
    env.close()


@pytest.fixture
def env_sp_1d():
    """1D SP environment."""
    env = make_env(1, balloon_type="superpressure")
    yield env
    env.close()


@pytest.fixture
def env_sp_3d():
    """3D SP environment."""
    env = make_env(3, balloon_type="superpressure")
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
    """Observation width for a given dimension.

    The layout is frozen at :data:`OBS_WIDTH` (143) and is **identical** for
    dim 1, 2 and 3 — fields meaningless in a dimension are zeroed, never
    omitted (Layer 1 contract §1, roadmap §2.1).  The ``dim`` argument is
    accepted only so call sites read naturally; it is deliberately ignored.
    """
    return OBS_WIDTH
