"""Tests for the Balloon class - physics simulation."""

import numpy as np
import pytest

from environments.core.balloon import Balloon
from environments.core.atmosphere import Atmosphere
from environments.core.constants import G, ALT_DEFAULT


class TestBalloonInitialization:
    """Tests for balloon initialization."""

    def test_default_initialization(self, atmosphere):
        """Balloon should initialize with sensible defaults."""
        balloon = Balloon(atmosphere=atmosphere)
        assert balloon.dim == 1
        assert balloon.mass == 2.0
        assert balloon.altitude == ALT_DEFAULT  # default starting altitude
        assert balloon.velocity == 0.0
        assert balloon.t == 0.0

    def test_custom_position_1d(self, atmosphere):
        """1D balloon should accept custom position."""
        balloon = Balloon(dim=1, atmosphere=atmosphere, position=[10_000.0])
        assert balloon.altitude == 10_000.0
        assert balloon.pos.shape == (1,)

    def test_custom_position_3d(self, atmosphere):
        """3D balloon should accept custom position."""
        balloon = Balloon(dim=3, atmosphere=atmosphere, position=[100.0, 200.0, 15_000.0])
        assert balloon.x == 100.0
        assert balloon.y == 200.0
        assert balloon.altitude == 15_000.0
        assert balloon.pos.shape == (3,)

    def test_custom_velocity(self, atmosphere):
        """Balloon should accept custom velocity."""
        balloon = Balloon(dim=3, atmosphere=atmosphere, velocity=[1.0, 2.0, 3.0])
        assert balloon.vx == 1.0
        assert balloon.vy == 2.0
        assert balloon.velocity == 3.0  # z-velocity property

    def test_stationary_volume_calculation(self, atmosphere):
        """Stationary volume should equal mass / air_density (neutral buoyancy)."""
        balloon = Balloon(dim=1, atmosphere=atmosphere, position=[15_000.0])
        rho_air = atmosphere.density(15_000.0)
        expected_volume = balloon.mass / rho_air
        assert np.isclose(balloon.stationary_volume, expected_volume, rtol=1e-6)


class TestBalloonVolume:
    """Tests for balloon volume control."""

    def test_inflate_increases_volume(self, balloon_1d):
        """Inflating should increase total volume."""
        initial_volume = balloon_1d.volume
        balloon_1d.inflate(0.1)
        assert balloon_1d.volume > initial_volume

    def test_deflate_decreases_volume(self, balloon_1d):
        """Deflating (negative inflate) should decrease extra volume."""
        balloon_1d.inflate(0.5)  # First add some volume
        volume_after_inflate = balloon_1d.volume
        balloon_1d.inflate(-0.2)
        assert balloon_1d.volume < volume_after_inflate

    def test_volume_changes_persist(self, balloon_1d):
        """Volume changes should accumulate."""
        balloon_1d.inflate(0.1)
        balloon_1d.inflate(0.1)
        balloon_1d.inflate(0.1)
        assert balloon_1d.extra_volume == pytest.approx(0.3)

    def test_apply_volume_change_alias(self, balloon_1d):
        """apply_volume_change should work like inflate."""
        v1 = balloon_1d.volume
        balloon_1d.apply_volume_change(0.2)
        assert balloon_1d.volume == pytest.approx(v1 + 0.2)


class TestBalloonForces:
    """Tests for force calculations."""

    def test_buoyant_force_direction(self, balloon_3d):
        """Buoyant force should act upward (positive z)."""
        buoy = balloon_3d.buoyant_force(0.0)
        assert buoy[-1] > 0, "Buoyant force should be positive (upward)"
        # x, y components should be zero
        assert buoy[0] == 0
        assert buoy[1] == 0

    def test_weight_direction(self, balloon_3d):
        """Weight should act downward (negative z)."""
        weight = balloon_3d.weight()
        assert weight[-1] < 0, "Weight should be negative (downward)"
        # x, y components should be zero
        assert weight[0] == 0
        assert weight[1] == 0

    def test_weight_equals_mg(self, balloon_1d):
        """Weight magnitude should equal mass * g."""
        weight = balloon_1d.weight()
        expected = -balloon_1d.mass * G
        assert weight[-1] == pytest.approx(expected)

    def test_neutral_buoyancy_at_stationary_volume(self, atmosphere):
        """At stationary volume, buoyancy should approximately equal weight."""
        balloon = Balloon(dim=1, atmosphere=atmosphere, position=[15_000.0])
        buoy = balloon.buoyant_force(0.0)[-1]
        weight = abs(balloon.weight()[-1])
        # Should be within 5% due to numerical precision
        assert abs(buoy - weight) / weight < 0.05

    def test_inflating_increases_buoyancy(self, balloon_1d):
        """Inflating should increase buoyant force."""
        buoy_before = balloon_1d.buoyant_force(0.0)[-1]
        balloon_1d.inflate(0.2 * balloon_1d.stationary_volume)
        buoy_after = balloon_1d.buoyant_force(0.0)[-1]
        assert buoy_after > buoy_before

    def test_drag_opposes_motion(self, balloon_3d):
        """Drag force should oppose velocity direction."""
        balloon_3d.vel = np.array([10.0, 5.0, 2.0])
        drag = balloon_3d.drag_force()

        # Drag should be opposite to velocity
        for i in range(3):
            if balloon_3d.vel[i] > 0:
                assert drag[i] < 0
            elif balloon_3d.vel[i] < 0:
                assert drag[i] > 0

    def test_drag_zero_when_stationary(self, balloon_3d):
        """Drag should be zero when velocity is zero."""
        balloon_3d.vel = np.zeros(3)
        drag = balloon_3d.drag_force()
        assert np.allclose(drag, 0)

    def test_drag_magnitude_increases_with_speed(self, balloon_3d):
        """Drag magnitude should increase with speed (quadratically)."""
        balloon_3d.vel = np.array([0.0, 0.0, 5.0])
        drag_slow = np.linalg.norm(balloon_3d.drag_force())

        balloon_3d.vel = np.array([0.0, 0.0, 10.0])
        drag_fast = np.linalg.norm(balloon_3d.drag_force())

        # Drag ~ v^2, so doubling speed should quadruple drag
        ratio = drag_fast / drag_slow
        assert 3.5 < ratio < 4.5  # Allow some tolerance


class TestBalloonCachedDensity:
    """Tests for cached density parameter in force calculations."""

    def test_buoyant_force_with_cached_density(self, balloon_1d, atmosphere):
        """Buoyant force should use cached density when provided."""
        rho = atmosphere.density(balloon_1d.altitude)
        buoy_cached = balloon_1d.buoyant_force(0.0, rho_air=rho)
        buoy_computed = balloon_1d.buoyant_force(0.0)
        assert np.allclose(buoy_cached, buoy_computed)

    def test_drag_force_with_cached_density(self, balloon_3d, atmosphere):
        """Drag force should use cached density when provided."""
        balloon_3d.vel = np.array([5.0, 3.0, 2.0])
        rho = atmosphere.density(balloon_3d.altitude)
        drag_cached = balloon_3d.drag_force(rho_air=rho)
        drag_computed = balloon_3d.drag_force()
        assert np.allclose(drag_cached, drag_computed)


class TestBalloonPhysicsIntegration:
    """Tests for physics integration step."""

    def test_update_advances_time(self, balloon_1d):
        """Update should advance internal time."""
        t0 = balloon_1d.t
        balloon_1d.update(1.0)
        assert balloon_1d.t == t0 + 1.0

    def test_update_with_buoyancy_excess_rises(self, balloon_1d):
        """Balloon with excess buoyancy should rise."""
        z0 = balloon_1d.altitude
        balloon_1d.inflate(0.5 * balloon_1d.stationary_volume)  # Excess buoyancy
        balloon_1d.update(1.0)
        assert balloon_1d.altitude > z0

    def test_update_with_buoyancy_deficit_falls(self, balloon_1d):
        """Balloon with buoyancy deficit should fall."""
        z0 = balloon_1d.altitude
        balloon_1d.inflate(-0.3 * balloon_1d.stationary_volume)  # Deficit
        balloon_1d.update(1.0)
        assert balloon_1d.altitude < z0

    def test_ground_clamping(self, atmosphere):
        """Balloon should not go below ground (z=0)."""
        balloon = Balloon(dim=1, atmosphere=atmosphere, position=[100.0], velocity=[-200.0])
        balloon.update(10.0)  # Large dt to ensure ground contact
        assert balloon.altitude >= 0.0
        assert balloon.velocity >= 0.0  # Velocity should be clamped too

    def test_velocity_clipping(self, balloon_1d):
        """Velocity should be clipped to +-200 m/s."""
        balloon_1d.vel = np.array([300.0])
        balloon_1d.update(0.001)  # Very small dt
        assert balloon_1d.velocity <= 200.0

    def test_external_force_applied(self, balloon_3d):
        """External force should affect motion."""
        balloon_3d.vel = np.zeros(3)
        x0 = balloon_3d.x

        # Apply horizontal force
        external = np.array([100.0, 0.0, 0.0])
        balloon_3d.update(1.0, external_force=external)

        # Should have moved in x direction
        assert balloon_3d.x > x0

    def test_control_force_applied(self, balloon_3d):
        """Control force should affect motion."""
        balloon_3d.vel = np.zeros(3)
        y0 = balloon_3d.y

        # Apply control force
        control = np.array([0.0, 50.0, 0.0])
        balloon_3d.update(1.0, control_force=control)

        # Should have moved in y direction
        assert balloon_3d.y > y0


class TestBalloonProperties:
    """Tests for balloon property accessors."""

    def test_altitude_property_1d(self, balloon_1d):
        """Altitude property should work for 1D balloon."""
        balloon_1d.altitude = 20_000.0
        assert balloon_1d.altitude == 20_000.0
        assert balloon_1d.pos[-1] == 20_000.0

    def test_velocity_property_1d(self, balloon_1d):
        """Velocity property should work for 1D balloon."""
        balloon_1d.velocity = 5.0
        assert balloon_1d.velocity == 5.0
        assert balloon_1d.vel[-1] == 5.0

    def test_xy_properties_3d(self, balloon_3d):
        """x, y properties should work for 3D balloon."""
        balloon_3d.x = 500.0
        balloon_3d.y = -300.0
        assert balloon_3d.x == 500.0
        assert balloon_3d.y == -300.0
        assert balloon_3d.pos[0] == 500.0
        assert balloon_3d.pos[1] == -300.0

    def test_vx_vy_properties_3d(self, balloon_3d):
        """vx, vy properties should work for 3D balloon."""
        balloon_3d.vx = 10.0
        balloon_3d.vy = -5.0
        assert balloon_3d.vx == 10.0
        assert balloon_3d.vy == -5.0
