"""Tests for the Balloon class - gas tracking, passive expansion, volume-dependent drag."""

import math
import numpy as np
import pytest

from environments.core.balloon import Balloon
from environments.core.atmosphere import Atmosphere
from environments.core.constants import (
    G, R, T_BALLOON, M_HE, ALT_DEFAULT, VOL_MAX, VOL_MIN,
)


class TestBalloonInitialization:
    """Tests for balloon initialization."""

    def test_default_initialization(self, atmosphere):
        """Balloon should initialize with sensible defaults."""
        balloon = Balloon(atmosphere=atmosphere)
        assert balloon.dim == 1
        assert balloon.mass == 2.0
        assert balloon.altitude == ALT_DEFAULT
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
        assert balloon.velocity == 3.0

    def test_initial_volume_is_neutral_buoyancy(self, atmosphere):
        """Initial volume should equal mass / rho_air (neutral buoyancy)."""
        balloon = Balloon(dim=1, atmosphere=atmosphere, position=[15_000.0])
        rho_air = atmosphere.density(15_000.0)
        expected_volume = balloon.mass / rho_air
        assert balloon.volume == pytest.approx(expected_volume, rel=1e-4)

    def test_initial_gas_moles_consistent(self, atmosphere):
        """Initial n_gas should satisfy V = nRT/P at the starting altitude."""
        balloon = Balloon(dim=1, atmosphere=atmosphere, position=[10_000.0])
        p = atmosphere.pressure(10_000.0)
        expected_vol = balloon.n_gas * R * T_BALLOON / p
        assert balloon.volume == pytest.approx(expected_vol, rel=1e-4)


class TestGasExpansion:
    """Tests for passive gas expansion/compression with altitude."""

    def test_volume_increases_at_higher_altitude(self, atmosphere):
        """Same gas moles should produce larger volume at higher altitude (lower pressure)."""
        b_low = Balloon(dim=1, atmosphere=atmosphere, position=[5_000.0])
        n_gas = b_low.n_gas

        b_high = Balloon(dim=1, atmosphere=atmosphere, position=[5_000.0])
        b_high.n_gas = n_gas
        b_high.pos[0] = 15_000.0  # move to higher altitude without changing gas

        assert b_high.volume > b_low.volume

    def test_volume_decreases_at_lower_altitude(self, atmosphere):
        """Same gas moles should produce smaller volume at lower altitude (higher pressure)."""
        b_high = Balloon(dim=1, atmosphere=atmosphere, position=[15_000.0])
        n_gas = b_high.n_gas

        b_low = Balloon(dim=1, atmosphere=atmosphere, position=[15_000.0])
        b_low.n_gas = n_gas
        b_low.pos[0] = 5_000.0

        assert b_low.volume < b_high.volume

    def test_ideal_gas_law_holds(self, atmosphere):
        """Volume should satisfy V = nRT/P at any altitude."""
        balloon = Balloon(dim=1, atmosphere=atmosphere, position=[8_000.0])
        for alt in [2_000.0, 8_000.0, 14_000.0, 20_000.0]:
            balloon.pos[0] = alt
            p = atmosphere.pressure(alt)
            expected = balloon.n_gas * R * T_BALLOON / p
            expected = max(VOL_MIN, min(expected, VOL_MAX))
            assert balloon.volume == pytest.approx(expected, rel=1e-4)

    def test_volume_clamped_to_max(self, atmosphere):
        """Volume should not exceed VOL_MAX even at very high altitude."""
        balloon = Balloon(dim=1, atmosphere=atmosphere, position=[5_000.0])
        balloon.n_gas *= 100  # absurdly large gas amount
        balloon.pos[0] = 30_000.0
        assert balloon.volume <= VOL_MAX

    def test_volume_clamped_to_min(self, atmosphere):
        """Volume should not drop below VOL_MIN even with very little gas."""
        balloon = Balloon(dim=1, atmosphere=atmosphere, position=[5_000.0])
        balloon.n_gas = 1e-10  # almost no gas
        assert balloon.volume >= VOL_MIN


class TestBalloonVolume:
    """Tests for balloon volume control via inflate/deflate."""

    def test_inflate_increases_volume(self, balloon_1d):
        """Inflating should increase total volume."""
        initial_volume = balloon_1d.volume
        balloon_1d.inflate(0.5)
        assert balloon_1d.volume > initial_volume

    def test_deflate_decreases_volume(self, balloon_1d):
        """Deflating should decrease volume."""
        balloon_1d.inflate(1.0)
        volume_after_inflate = balloon_1d.volume
        balloon_1d.inflate(-0.5)
        assert balloon_1d.volume < volume_after_inflate

    def test_volume_changes_accumulate(self, balloon_1d):
        """Multiple inflations should accumulate."""
        v0 = balloon_1d.volume
        balloon_1d.inflate(0.2)
        balloon_1d.inflate(0.2)
        balloon_1d.inflate(0.2)
        # Volume should have increased by approximately 0.6
        assert balloon_1d.volume > v0

    def test_inflate_adds_moles(self, balloon_1d):
        """Inflating should increase n_gas."""
        n0 = balloon_1d.n_gas
        balloon_1d.inflate(0.5)
        assert balloon_1d.n_gas > n0

    def test_deflate_removes_moles(self, balloon_1d):
        """Deflating should decrease n_gas."""
        balloon_1d.inflate(1.0)
        n_after = balloon_1d.n_gas
        balloon_1d.inflate(-0.5)
        assert balloon_1d.n_gas < n_after

    def test_apply_volume_change_alias(self, balloon_1d):
        """apply_volume_change should work like inflate."""
        v1 = balloon_1d.volume
        balloon_1d.apply_volume_change(0.5)
        assert balloon_1d.volume > v1


class TestBalloonForces:
    """Tests for force calculations."""

    def test_buoyant_force_direction(self, balloon_3d):
        """Buoyant force should act upward (positive z)."""
        buoy = balloon_3d.buoyant_force(0.0)
        assert buoy[-1] > 0, "Buoyant force should be positive (upward)"
        assert buoy[0] == 0
        assert buoy[1] == 0

    def test_weight_direction(self, balloon_3d):
        """Weight should act downward (negative z)."""
        weight = balloon_3d.weight()
        assert weight[-1] < 0, "Weight should be negative (downward)"
        assert weight[0] == 0
        assert weight[1] == 0

    def test_weight_equals_mg(self, balloon_1d):
        """Weight magnitude should equal mass * g."""
        weight = balloon_1d.weight()
        expected = -balloon_1d.mass * G
        assert weight[-1] == pytest.approx(expected)

    def test_neutral_buoyancy_at_initial_volume(self, atmosphere):
        """At initial volume, buoyancy should approximately equal weight."""
        balloon = Balloon(dim=1, atmosphere=atmosphere, position=[15_000.0])
        buoy = balloon.buoyant_force(0.0)[-1]
        weight = abs(balloon.weight()[-1])
        assert abs(buoy - weight) / weight < 0.05

    def test_inflating_increases_buoyancy(self, balloon_1d):
        """Inflating should increase buoyant force."""
        buoy_before = balloon_1d.buoyant_force(0.0)[-1]
        balloon_1d.inflate(0.5 * balloon_1d.stationary_volume)
        buoy_after = balloon_1d.buoyant_force(0.0)[-1]
        assert buoy_after > buoy_before

    def test_drag_opposes_motion(self, balloon_3d):
        """Drag force should oppose velocity direction."""
        balloon_3d.vel = np.array([10.0, 5.0, 2.0])
        drag = balloon_3d.drag_force()

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

    def test_drag_increases_with_speed(self, balloon_3d):
        """Drag magnitude should increase with speed (within the Newton regime)."""
        # Use low velocities to stay well within the Newton drag regime
        # (Re ~ 30k-70k) and avoid the drag crisis at Re > 200k.
        balloon_3d.vel = np.array([0.0, 0.0, 1.0])
        drag_slow = np.linalg.norm(balloon_3d.drag_force())

        balloon_3d.vel = np.array([0.0, 0.0, 2.0])
        drag_fast = np.linalg.norm(balloon_3d.drag_force())

        assert drag_fast > drag_slow

    def test_drag_depends_on_volume(self, atmosphere):
        """Larger balloon volume should produce more drag (bigger frontal area)."""
        b_small = Balloon(dim=1, atmosphere=atmosphere, position=[10_000.0])
        b_large = Balloon(dim=1, atmosphere=atmosphere, position=[10_000.0])
        b_large.inflate(5.0)  # inflate to get larger volume

        # Set same velocity
        b_small.vel = np.array([10.0])
        b_large.vel = np.array([10.0])

        drag_small = abs(b_small.drag_force()[0])
        drag_large = abs(b_large.drag_force()[0])
        assert drag_large > drag_small


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
        balloon_1d.inflate(0.5 * balloon_1d.stationary_volume)
        balloon_1d.update(1.0)
        assert balloon_1d.altitude > z0

    def test_update_with_buoyancy_deficit_falls(self, balloon_1d):
        """Balloon with buoyancy deficit should fall."""
        z0 = balloon_1d.altitude
        balloon_1d.inflate(-0.3 * balloon_1d.stationary_volume)
        balloon_1d.update(1.0)
        assert balloon_1d.altitude < z0

    def test_ground_clamping(self, atmosphere):
        """Balloon should not go below ground (z=0)."""
        balloon = Balloon(dim=1, atmosphere=atmosphere, position=[100.0], velocity=[-200.0])
        balloon.update(10.0)
        assert balloon.altitude >= 0.0
        assert balloon.velocity >= 0.0

    def test_velocity_clipping(self, balloon_1d):
        """Velocity should be clipped to +-200 m/s."""
        balloon_1d.vel = np.array([300.0])
        balloon_1d.update(0.001)
        assert balloon_1d.velocity <= 200.0

    def test_external_force_applied(self, balloon_3d):
        """External force should affect motion."""
        balloon_3d.vel = np.zeros(3)
        x0 = balloon_3d.x
        external = np.array([100.0, 0.0, 0.0])
        balloon_3d.update(1.0, external_force=external)
        assert balloon_3d.x > x0

    def test_control_force_applied(self, balloon_3d):
        """Control force should affect motion."""
        balloon_3d.vel = np.zeros(3)
        y0 = balloon_3d.y
        control = np.array([0.0, 50.0, 0.0])
        balloon_3d.update(1.0, control_force=control)
        assert balloon_3d.y > y0

    def test_passive_expansion_during_ascent(self, atmosphere):
        """Volume should increase as balloon ascends (gas expands).

        Uses a moderate inflation to keep terminal velocity low enough
        that the forward-Euler integrator remains stable at dt=1.0.
        """
        balloon = Balloon(dim=1, atmosphere=atmosphere, position=[8_000.0])
        balloon.inflate(0.2 * balloon.stationary_volume)  # gentle excess buoyancy
        n_gas_before = balloon.n_gas  # gas moles should not change
        v0 = balloon.volume
        for _ in range(500):
            balloon.update(1.0)
        # Gas moles unchanged (no inflate/deflate during ascent)
        assert balloon.n_gas == pytest.approx(n_gas_before)
        # Balloon should have risen
        assert balloon.altitude > 8_000.0
        # Volume grew passively (lower pressure at higher altitude)
        assert balloon.volume > v0


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

    def test_extra_volume_property(self, balloon_1d):
        """extra_volume should reflect difference from stationary volume."""
        ev_before = balloon_1d.extra_volume
        balloon_1d.inflate(0.5)
        ev_after = balloon_1d.extra_volume
        assert ev_after > ev_before
