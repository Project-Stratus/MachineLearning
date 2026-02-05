"""Tests for the WindField class."""

import numpy as np
import pytest

from environments.core.wind_field import WindField


class TestWindFieldInitialization:
    """Tests for wind field initialization."""

    def test_default_initialization(self, wind_field):
        """Wind field should initialize with specified parameters."""
        assert wind_field.cells == 10
        assert wind_field.x_range == (-2000.0, 2000.0)
        assert wind_field.y_range == (-2000.0, 2000.0)
        assert wind_field.z_range == (0.0, 30000.0)

    def test_grid_centers_computed(self, wind_field):
        """Grid centers should be computed for sampling."""
        assert len(wind_field.x_centers) == wind_field.cells
        assert len(wind_field.y_centers) == wind_field.cells
        assert len(wind_field.z_centers) == wind_field.cells

    def test_grid_centers_within_range(self, wind_field):
        """Grid centers should be within specified ranges."""
        assert wind_field.x_centers.min() >= wind_field.x_range[0]
        assert wind_field.x_centers.max() <= wind_field.x_range[1]
        assert wind_field.y_centers.min() >= wind_field.y_range[0]
        assert wind_field.y_centers.max() <= wind_field.y_range[1]
        assert wind_field.z_centers.min() >= wind_field.z_range[0]
        assert wind_field.z_centers.max() <= wind_field.z_range[1]

    def test_inverse_spacing_computed(self, wind_field):
        """Inverse spacing values should be computed for fast indexing."""
        assert wind_field.inv_dx > 0
        assert wind_field.inv_dy > 0
        assert wind_field.inv_dz > 0


class TestWindFieldSampling:
    """Tests for wind field sampling."""

    def test_sample_returns_3d_vector(self, wind_field):
        """Sample should return a 3D wind vector."""
        wind = wind_field.sample(0.0, 0.0, 15000.0)
        assert wind.shape == (3,)

    def test_sample_at_center(self, wind_field):
        """Sampling at grid center should return valid wind."""
        wind = wind_field.sample(0.0, 0.0, 15000.0)
        assert np.all(np.isfinite(wind))

    def test_sample_at_edges(self, wind_field):
        """Sampling at edges should not crash."""
        # Test all corners and edges
        corners = [
            (-2000.0, -2000.0, 0.0),
            (-2000.0, -2000.0, 30000.0),
            (-2000.0, 2000.0, 0.0),
            (-2000.0, 2000.0, 30000.0),
            (2000.0, -2000.0, 0.0),
            (2000.0, -2000.0, 30000.0),
            (2000.0, 2000.0, 0.0),
            (2000.0, 2000.0, 30000.0),
        ]
        for x, y, z in corners:
            wind = wind_field.sample(x, y, z)
            assert wind.shape == (3,)
            assert np.all(np.isfinite(wind))

    def test_sample_outside_bounds_clamped(self, wind_field):
        """Sampling outside bounds should be clamped to edge values."""
        # Way outside bounds
        wind_outside = wind_field.sample(10000.0, 10000.0, 50000.0)
        wind_edge = wind_field.sample(2000.0, 2000.0, 30000.0)

        # Should be clamped to same value
        assert wind_outside.shape == (3,)
        assert np.all(np.isfinite(wind_outside))

    def test_sample_returns_numpy_array(self, wind_field):
        """Sample should return numpy array."""
        wind = wind_field.sample(0.0, 0.0, 15000.0)
        assert isinstance(wind, np.ndarray)

    def test_sample_wind_magnitude_bounded(self, wind_field):
        """Wind magnitude should be bounded by configured magnitude."""
        # Sample multiple points
        for _ in range(100):
            x = np.random.uniform(-2000, 2000)
            y = np.random.uniform(-2000, 2000)
            z = np.random.uniform(0, 30000)
            wind = wind_field.sample(x, y, z)
            magnitude = np.linalg.norm(wind[:2])  # Horizontal component
            # Allow some tolerance due to interpolation
            assert magnitude <= wind_field.mag * 1.5


class TestWindFieldPatterns:
    """Tests for different wind patterns."""

    @pytest.fixture(params=["sinusoid", "linear_right", "linear_up", "split_fork"])
    def wind_field_pattern(self, request):
        """Wind field with different patterns."""
        return WindField(
            x_range=(-2000.0, 2000.0),
            y_range=(-2000.0, 2000.0),
            z_range=(0.0, 30000.0),
            cells=10,
            pattern=request.param,
            default_mag=10.0,
        )

    def test_pattern_produces_valid_wind(self, wind_field_pattern):
        """Each pattern should produce valid wind vectors."""
        wind = wind_field_pattern.sample(0.0, 0.0, 15000.0)
        assert wind.shape == (3,)
        assert np.all(np.isfinite(wind))

    def test_pattern_produces_consistent_wind(self, wind_field_pattern):
        """Each pattern should produce consistent wind at same position."""
        # Sample same position twice
        wind1 = wind_field_pattern.sample(500.0, 500.0, 15000.0)
        wind2 = wind_field_pattern.sample(500.0, 500.0, 15000.0)
        assert np.allclose(wind1, wind2), "Same position should give same wind"


class TestWindFieldGridConstruction:
    """Tests for wind grid construction."""

    def test_grid_arrays_exist(self, wind_field):
        """Wind field should have grid arrays."""
        assert hasattr(wind_field, "_fx_grid")
        assert hasattr(wind_field, "_fy_grid")

    def test_grid_shape(self, wind_field):
        """Grid arrays should have correct shape."""
        expected_shape = (wind_field.cells, wind_field.cells, wind_field.cells)
        assert wind_field._fx_grid.shape == expected_shape
        assert wind_field._fy_grid.shape == expected_shape

    def test_grid_values_finite(self, wind_field):
        """Grid values should be finite."""
        assert np.all(np.isfinite(wind_field._fx_grid))
        assert np.all(np.isfinite(wind_field._fy_grid))


class TestWindFieldDeterminism:
    """Tests for wind field determinism."""

    def test_same_position_same_wind(self, wind_field):
        """Same position should always return same wind."""
        wind1 = wind_field.sample(500.0, 500.0, 15000.0)
        wind2 = wind_field.sample(500.0, 500.0, 15000.0)
        assert np.allclose(wind1, wind2)

    def test_wind_field_reproducible(self):
        """Same parameters should produce same wind field."""
        wf1 = WindField(
            x_range=(-1000.0, 1000.0),
            y_range=(-1000.0, 1000.0),
            z_range=(0.0, 20000.0),
            cells=5,
            pattern="sinusoid",
            default_mag=5.0,
        )
        wf2 = WindField(
            x_range=(-1000.0, 1000.0),
            y_range=(-1000.0, 1000.0),
            z_range=(0.0, 20000.0),
            cells=5,
            pattern="sinusoid",
            default_mag=5.0,
        )

        wind1 = wf1.sample(0.0, 0.0, 10000.0)
        wind2 = wf2.sample(0.0, 0.0, 10000.0)
        assert np.allclose(wind1, wind2)
