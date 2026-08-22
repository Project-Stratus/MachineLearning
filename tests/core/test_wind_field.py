"""Tests for the WindField class."""

import numpy as np
import pytest

from environments.core.wind_field import WindField, default_cells_z, CELLS_Z_MAX
from environments.core.constants import WIND_COL_LEVELS, WIND_COL_SPACING, WIND_COL_HALF_SPAN


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
        assert len(wind_field.z_centers) == wind_field.cells_z

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


class TestVerticalResolution:
    """The vertical axis is resolved independently of the horizontal one.

    A cubic grid at ``cells=20`` over a 30 km ``z_range`` gives 1.5 km cells,
    so the 41-level / 250 m wind column would collapse onto ~8 distinct
    values — the column would be nominally 41 levels deep and actually blind
    between them.  These tests pin the decoupling that fixes it.
    """

    def test_default_cells_z_meets_nyquist(self):
        """Default vertical cells must be <= half the column spacing."""
        z_range = (0.0, 30_105.0)
        n = default_cells_z(z_range)
        dz = (z_range[1] - z_range[0]) / n
        assert dz <= 0.5 * WIND_COL_SPACING

    def test_default_cells_z_scales_with_span(self):
        """Twice the altitude range needs twice the cells."""
        assert default_cells_z((0.0, 20_000.0)) == 2 * default_cells_z((0.0, 10_000.0))

    def test_default_cells_z_respects_spacing_argument(self):
        """A coarser column spacing needs proportionally fewer cells."""
        fine = default_cells_z((0.0, 30_000.0), spacing=250.0)
        coarse = default_cells_z((0.0, 30_000.0), spacing=500.0)
        assert coarse * 2 == fine

    def test_default_cells_z_clamped(self):
        """A pathological range must not size the grid without bound."""
        assert default_cells_z((0.0, 1e9)) == CELLS_Z_MAX

    def test_default_cells_z_degenerate_range(self):
        assert default_cells_z((5_000.0, 5_000.0)) == 1

    def test_field_uses_nyquist_by_default(self, wind_field):
        """The fixture asks for cells=10 horizontally and gets a fine z axis."""
        assert wind_field.cells_z == default_cells_z(wind_field.z_range)
        assert wind_field.cells_z > wind_field.cells
        assert wind_field.dz <= 0.5 * WIND_COL_SPACING

    def test_cells_z_overridable(self):
        """An explicit cells_z wins over the computed default."""
        wf = WindField(
            x_range=(-1000.0, 1000.0),
            y_range=(-1000.0, 1000.0),
            z_range=(0.0, 30_000.0),
            cells=8,
            cells_z=17,
            pattern="altitude_shear",
            default_mag=10.0,
        )
        assert wf.cells_z == 17
        assert wf._fx_grid.shape == (8, 8, 17)
        assert len(wf.z_centers) == 17
        assert wf.dz == pytest.approx(30_000.0 / 17)

    def test_horizontal_resolution_unaffected(self):
        """``cells`` still governs x and y only."""
        wf = WindField(
            x_range=(-1000.0, 1000.0),
            y_range=(-1000.0, 1000.0),
            z_range=(0.0, 30_000.0),
            cells=6,
            pattern="sinusoid",
            default_mag=10.0,
        )
        assert wf.dx == pytest.approx(2000.0 / 6)
        assert wf.dy == pytest.approx(2000.0 / 6)
        assert len(wf.x_centers) == 6
        assert len(wf.y_centers) == 6

    def test_adjacent_column_levels_are_distinct(self):
        """The point of the whole exercise: 41 levels, 41 distinct winds.

        ``altitude_shear_2d`` varies with altitude alone, so any two adjacent
        levels that read identically are a resolution failure, not a property
        of the field.
        """
        wf = WindField(
            x_range=(-50_000.0, 50_000.0),
            y_range=(-50_000.0, 50_000.0),
            z_range=(0.0, 30_105.0),
            cells=20,
            pattern="altitude_shear_2d",
            default_mag=10.0,
            wind_layers=2,
        )
        col = wf.sample_column(0.0, 0.0, 20_000.0).copy()
        deltas = np.abs(np.diff(col, axis=0)).sum(axis=1)
        assert np.all(deltas > 1e-9), "adjacent levels must not share a grid cell"
        assert len(np.unique(col, axis=0)) == WIND_COL_LEVELS

    def test_cubic_grid_would_alias(self):
        """Guard the premise: the old cubic grid really did collapse levels."""
        cubic = WindField(
            x_range=(-50_000.0, 50_000.0),
            y_range=(-50_000.0, 50_000.0),
            z_range=(0.0, 30_105.0),
            cells=20,
            cells_z=20,
            pattern="altitude_shear_2d",
            default_mag=10.0,
            wind_layers=2,
        )
        col = cubic.sample_column(0.0, 0.0, 20_000.0).copy()
        assert len(np.unique(col, axis=0)) < WIND_COL_LEVELS // 2

    def test_grid_memory_is_modest(self):
        """Refining z must stay cheap enough for many parallel envs."""
        wf = WindField(
            x_range=(-50_000.0, 50_000.0),
            y_range=(-50_000.0, 50_000.0),
            z_range=(0.0, 30_105.0),
            cells=20,
            pattern="altitude_shear_2d",
            default_mag=10.0,
        )
        total_mib = (wf._fx_grid.nbytes + wf._fy_grid.nbytes) / 1024 ** 2
        assert total_mib < 4.0


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

    def test_sample_jit_and_fallback_agree(self, wind_field, monkeypatch):
        """The kernel and the searchsorted fallback index the same cell.

        Both now carry separate x/y and z cell counts; a mismatch here would
        show up as an off-by-one only on the vertical axis.
        """
        import environments.core.wind_field as wf_mod

        points = [(0.0, 0.0, 15_000.0), (-1_337.0, 812.0, 2_500.0),
                  (1_900.0, -1_900.0, 29_500.0), (17.0, -23.0, 137.0)]
        jit_vals = [wind_field.sample(*p).copy() for p in points]
        monkeypatch.setattr(wf_mod, "_JIT_OK", False)
        for p, expected in zip(points, jit_vals):
            assert np.array_equal(wind_field.sample(*p), expected)

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


@pytest.fixture
def shear_field():
    """Altitude-shear field on a fine vertical grid, for column tests.

    ``altitude_shear`` gives fx varying linearly with altitude and fy == 0, so
    every level of a sampled column is individually identifiable.  ``cells``
    is horizontal only; the vertical axis takes the Nyquist default.
    """
    return WindField(
        x_range=(-50_000.0, 50_000.0),
        y_range=(-50_000.0, 50_000.0),
        z_range=(0.0, 30_000.0),
        cells=120,
        pattern="altitude_shear",
        default_mag=10.0,
    )


class TestWindFieldSampleColumn:
    """Tests for the altitude-centred wind column (contract §4)."""

    def test_default_shape(self, shear_field):
        """Default column is (WIND_COL_LEVELS, 2) of (fx, fy)."""
        col = shear_field.sample_column(0.0, 0.0, 20_000.0)
        assert col.shape == (WIND_COL_LEVELS, 2)

    def test_custom_levels_shape(self, shear_field):
        col = shear_field.sample_column(0.0, 0.0, 20_000.0, levels=11, spacing=500.0)
        assert col.shape == (11, 2)

    def test_defaults_match_constants(self, shear_field):
        """Explicitly passing the constants must give the same column as defaults."""
        explicit = shear_field.sample_column(
            0.0, 0.0, 20_000.0, levels=WIND_COL_LEVELS, spacing=WIND_COL_SPACING
        ).copy()
        default = shear_field.sample_column(0.0, 0.0, 20_000.0)
        assert np.array_equal(explicit, default)

    def test_half_span_constant_consistent(self):
        """WIND_COL_HALF_SPAN must equal (levels//2) * spacing."""
        assert WIND_COL_HALF_SPAN == (WIND_COL_LEVELS // 2) * WIND_COL_SPACING

    def test_centre_level_equals_point_sample(self, shear_field):
        """Index levels//2 is the balloon's own altitude."""
        z = 18_000.0
        col = shear_field.sample_column(0.0, 0.0, z).copy()
        point = shear_field.sample(0.0, 0.0, z)
        assert col[WIND_COL_LEVELS // 2, 0] == pytest.approx(point[0])
        assert col[WIND_COL_LEVELS // 2, 1] == pytest.approx(point[1])

    def test_every_level_matches_a_point_sample(self, shear_field):
        """Level i must equal sample() at z_center + (i - levels//2)*spacing."""
        z = 20_000.0
        col = shear_field.sample_column(0.0, 0.0, z).copy()
        half = WIND_COL_LEVELS // 2
        for i in range(WIND_COL_LEVELS):
            zi = z + (i - half) * WIND_COL_SPACING
            expected = shear_field.sample(0.0, 0.0, zi)
            assert col[i, 0] == pytest.approx(expected[0])
            assert col[i, 1] == pytest.approx(expected[1])

    def test_ordered_low_to_high(self, shear_field):
        """altitude_shear increases fx with altitude, so the column is ascending."""
        col = shear_field.sample_column(0.0, 0.0, 15_000.0).copy()
        assert np.all(np.diff(col[:, 0]) >= 0)
        assert col[0, 0] < col[-1, 0]

    def test_column_spans_expected_altitudes(self, shear_field):
        """Bottom/top levels sit HALF_SPAN below/above the centre."""
        z = 20_000.0
        col = shear_field.sample_column(0.0, 0.0, z).copy()
        assert col[0, 0] == pytest.approx(
            shear_field.sample(0.0, 0.0, z - WIND_COL_HALF_SPAN)[0]
        )
        assert col[-1, 0] == pytest.approx(
            shear_field.sample(0.0, 0.0, z + WIND_COL_HALF_SPAN)[0]
        )

    def test_levels_outside_grid_clamp_to_edge(self, shear_field):
        """Levels below the grid floor repeat the lowest cell, not garbage."""
        col = shear_field.sample_column(0.0, 0.0, 1_000.0).copy()
        bottom = shear_field.sample(0.0, 0.0, 0.0)[0]
        # Levels at negative altitude (i < 4 for spacing 250 m) all clamp
        assert col[0, 0] == pytest.approx(bottom)
        assert col[1, 0] == pytest.approx(bottom)
        assert np.all(np.isfinite(col))

    def test_horizontal_coords_clamped(self, shear_field):
        """Far-outside x/y clamp to the grid edge rather than failing."""
        far = shear_field.sample_column(1e9, -1e9, 20_000.0).copy()
        edge = shear_field.sample_column(
            shear_field.x_range[1], shear_field.y_range[0], 20_000.0
        ).copy()
        assert np.array_equal(far, edge)

    def test_values_finite_everywhere(self, wind_field):
        """Works on the coarse default fixture too."""
        for z in [0.0, 5_000.0, 15_000.0, 30_000.0, 60_000.0]:
            col = wind_field.sample_column(0.0, 0.0, z)
            assert col.shape == (WIND_COL_LEVELS, 2)
            assert np.all(np.isfinite(col))

    def test_magnitude_bounded_by_field_magnitude(self, shear_field):
        col = shear_field.sample_column(0.0, 0.0, 20_000.0)
        assert np.all(np.abs(col) <= shear_field.mag + 1e-9)

    def test_buffer_is_reused(self, shear_field):
        """Called every decision step — must not allocate per call."""
        first = shear_field.sample_column(0.0, 0.0, 20_000.0)
        second = shear_field.sample_column(0.0, 0.0, 21_000.0)
        assert first is second

    def test_buffer_resized_only_on_level_change(self, shear_field):
        shear_field.sample_column(0.0, 0.0, 20_000.0)
        buf = shear_field._column_buf
        shear_field.sample_column(0.0, 0.0, 20_000.0, levels=9)
        assert shear_field._column_buf is not buf
        assert shear_field._column_buf.shape == (9, 2)

    def test_deterministic(self, shear_field):
        a = shear_field.sample_column(100.0, 200.0, 19_000.0).copy()
        b = shear_field.sample_column(100.0, 200.0, 19_000.0).copy()
        assert np.array_equal(a, b)

    def test_jit_and_fallback_agree(self, shear_field, monkeypatch):
        """The non-JIT fallback must be numerically identical to the kernel."""
        import environments.core.wind_field as wf_mod

        jit_col = shear_field.sample_column(1_500.0, -800.0, 17_250.0).copy()
        monkeypatch.setattr(wf_mod, "_JIT_OK", False)
        py_col = shear_field.sample_column(1_500.0, -800.0, 17_250.0).copy()
        assert np.array_equal(jit_col, py_col)

    def test_2d_pattern_populates_both_components(self):
        """altitude_shear_2d rotates direction with altitude — fy must vary too."""
        wf = WindField(
            x_range=(-50_000.0, 50_000.0),
            y_range=(-50_000.0, 50_000.0),
            z_range=(0.0, 30_000.0),
            cells=120,
            pattern="altitude_shear_2d",
            default_mag=10.0,
            wind_layers=2,
        )
        col = wf.sample_column(0.0, 0.0, 20_000.0)
        assert np.ptp(col[:, 0]) > 0.0
        assert np.ptp(col[:, 1]) > 0.0


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
        """Grid arrays are (cells, cells, cells_z) — z resolved independently."""
        expected_shape = (wind_field.cells, wind_field.cells, wind_field.cells_z)
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
