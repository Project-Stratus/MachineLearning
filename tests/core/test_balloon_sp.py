"""Tests for BalloonSP - superpressure + air ballast balloon model.

Mirrors the structure of test_balloon.py for the ZP model.
"""

import numpy as np
import pytest

from environments.core.balloon import BalloonSP
from environments.core.atmosphere import Atmosphere
from environments.core.constants import (
    G, ALT_DEFAULT,
    SP_VOL_FIXED, SP_PAYLOAD_MASS,
    AIR_PUMP_RATE, AIR_BLADDER_MAX, AIR_BLADDER_INITIAL,
)


class TestBalloonSPInitialization:
    """Tests for SP balloon initialization."""

    def test_initial_neutral_buoyancy(self, atmosphere):
        """ρ_air × V_FIXED == total mass at ALT_DEFAULT with bladder at midpoint.

        By construction: m_he_fixed = ρ_air(ALT_DEFAULT)·V_FIXED - payload - bladder_initial,
        so buoyancy mass = ρ_air·V_FIXED = m_he_fixed + payload + bladder_initial = total mass.
        """
        b = BalloonSP(dim=1, atmosphere=atmosphere, position=[ALT_DEFAULT])
        rho_air = atmosphere.density(ALT_DEFAULT)
        buoyancy_mass = rho_air * SP_VOL_FIXED
        assert buoyancy_mass == pytest.approx(b.mass, rel=1e-6)

    def test_initial_bladder_at_midpoint(self, atmosphere):
        """Bladder should initialise at AIR_BLADDER_INITIAL (midpoint)."""
        b = BalloonSP(dim=1, atmosphere=atmosphere, position=[ALT_DEFAULT])
        assert b.air_bladder_mass == pytest.approx(AIR_BLADDER_INITIAL)

    def test_initial_position_default(self, atmosphere):
        """Default position should be ALT_DEFAULT."""
        b = BalloonSP(dim=1, atmosphere=atmosphere)
        assert b.altitude == pytest.approx(ALT_DEFAULT)

    def test_custom_position(self, atmosphere):
        """SP balloon should accept custom position."""
        b = BalloonSP(dim=3, atmosphere=atmosphere, position=[100.0, 200.0, 15_000.0])
        assert b.x == 100.0
        assert b.y == 200.0
        assert b.altitude == 15_000.0

    def test_m_he_fixed_is_positive(self, atmosphere):
        """Fixed helium mass must be positive for the balloon to float."""
        b = BalloonSP(dim=1, atmosphere=atmosphere, position=[ALT_DEFAULT])
        assert b.m_he_fixed > 0.0


class TestBalloonSPVolume:
    """Tests for fixed volume behaviour."""

    def test_volume_constant_at_various_altitudes(self, atmosphere):
        """Volume is unchanged regardless of altitude (fixed envelope)."""
        b = BalloonSP(dim=1, atmosphere=atmosphere, position=[ALT_DEFAULT])
        for alt in [10_000.0, 15_000.0, ALT_DEFAULT, 25_000.0, 35_000.0]:
            b.pos[0] = alt
            assert b.volume == pytest.approx(SP_VOL_FIXED)

    def test_volume_constant_after_pumping(self, atmosphere):
        """Pumping air does not change the envelope volume."""
        b = BalloonSP(dim=1, atmosphere=atmosphere, position=[ALT_DEFAULT])
        b.pump_in()
        assert b.volume == pytest.approx(SP_VOL_FIXED)
        b.pump_out()
        b.pump_out()
        assert b.volume == pytest.approx(SP_VOL_FIXED)


class TestBalloonSPMass:
    """Tests for variable mass from air bladder."""

    def test_pump_in_increases_mass(self, atmosphere):
        """Pumping air in should increase total mass."""
        b = BalloonSP(dim=1, atmosphere=atmosphere, position=[ALT_DEFAULT])
        m0 = b.mass
        b.pump_in()
        assert b.mass > m0

    def test_pump_out_decreases_mass(self, atmosphere):
        """Pumping air out should decrease total mass."""
        b = BalloonSP(dim=1, atmosphere=atmosphere, position=[ALT_DEFAULT])
        m0 = b.mass
        b.pump_out()
        assert b.mass < m0

    def test_pump_in_increases_mass_by_pump_rate(self, atmosphere):
        """Mass increase from pump_in should equal AIR_PUMP_RATE."""
        b = BalloonSP(dim=1, atmosphere=atmosphere, position=[ALT_DEFAULT])
        m0 = b.mass
        b.pump_in()
        assert b.mass - m0 == pytest.approx(AIR_PUMP_RATE)

    def test_pump_out_decreases_mass_by_pump_rate(self, atmosphere):
        """Mass decrease from pump_out should equal AIR_PUMP_RATE."""
        b = BalloonSP(dim=1, atmosphere=atmosphere, position=[ALT_DEFAULT])
        m0 = b.mass
        b.pump_out()
        assert m0 - b.mass == pytest.approx(AIR_PUMP_RATE)

    def test_mass_includes_all_components(self, atmosphere):
        """Total mass = payload + m_he_fixed + air_bladder."""
        b = BalloonSP(dim=1, atmosphere=atmosphere, position=[ALT_DEFAULT])
        expected = b.payload_mass + b.m_he_fixed + b.air_bladder_mass
        assert b.mass == pytest.approx(expected)


class TestBalloonSPPumpClamping:
    """Tests for bladder clamping at boundaries."""

    def test_pump_clamped_at_max(self, atmosphere):
        """Bladder should not exceed AIR_BLADDER_MAX."""
        b = BalloonSP(dim=1, atmosphere=atmosphere, position=[ALT_DEFAULT])
        for _ in range(1000):
            b.pump_in()
        assert b.air_bladder_mass == pytest.approx(AIR_BLADDER_MAX)
        assert b.is_bladder_full

    def test_pump_clamped_at_zero(self, atmosphere):
        """Bladder mass should never go negative."""
        b = BalloonSP(dim=1, atmosphere=atmosphere, position=[ALT_DEFAULT])
        for _ in range(1000):
            b.pump_out()
        assert b.air_bladder_mass == pytest.approx(0.0)
        assert b.is_bladder_empty

    def test_is_bladder_full_at_max(self, atmosphere):
        """is_bladder_full should be True when bladder reaches capacity."""
        b = BalloonSP(dim=1, atmosphere=atmosphere, position=[ALT_DEFAULT])
        b.air_bladder_mass = AIR_BLADDER_MAX
        assert b.is_bladder_full

    def test_is_bladder_empty_at_zero(self, atmosphere):
        """is_bladder_empty should be True when bladder is drained."""
        b = BalloonSP(dim=1, atmosphere=atmosphere, position=[ALT_DEFAULT])
        b.air_bladder_mass = 0.0
        assert b.is_bladder_empty


class TestBalloonSPFlags:
    """Tests for always-False legacy flags."""

    def test_is_deflated_always_false(self, atmosphere):
        """SP envelope cannot deflate — flag must always be False."""
        b = BalloonSP(dim=1, atmosphere=atmosphere, position=[ALT_DEFAULT])
        assert not b.is_deflated

    def test_is_deflated_false_after_pumping(self, atmosphere):
        """is_deflated stays False even with extreme bladder states."""
        b = BalloonSP(dim=1, atmosphere=atmosphere, position=[ALT_DEFAULT])
        for _ in range(1000):
            b.pump_out()
        assert not b.is_deflated

    def test_is_ballast_empty_always_false(self, atmosphere):
        """SP has no expendable ballast — flag must always be False."""
        b = BalloonSP(dim=1, atmosphere=atmosphere, position=[ALT_DEFAULT])
        assert not b.is_ballast_empty


class TestBalloonSPAltitudeControl:
    """Tests that pumping produces the expected altitude changes."""

    def test_pump_in_causes_descent(self, atmosphere):
        """Pumping air in (heavier) should cause the balloon to descend."""
        b = BalloonSP(dim=1, atmosphere=atmosphere, position=[ALT_DEFAULT])
        z0 = b.altitude
        for _ in range(50):
            b.pump_in()
        for _ in range(200):
            b.update(1.0)
        assert b.altitude < z0

    def test_pump_out_causes_ascent(self, atmosphere):
        """Pumping air out (lighter) should cause the balloon to ascend."""
        b = BalloonSP(dim=1, atmosphere=atmosphere, position=[ALT_DEFAULT])
        z0 = b.altitude
        for _ in range(50):
            b.pump_out()
        for _ in range(200):
            b.update(1.0)
        assert b.altitude > z0


class TestBalloonSPPassiveStability:
    """Tests for the passive restoring force of the SP balloon."""

    def test_passive_restoring_force_upward_displacement(self, atmosphere):
        """Upward displacement should yield a net downward force (restoring)."""
        b = BalloonSP(dim=1, atmosphere=atmosphere, position=[ALT_DEFAULT])
        # Move 500 m above float altitude
        b.pos[0] = ALT_DEFAULT + 500.0
        rho = atmosphere.density(b.altitude)
        F_net = rho * G * SP_VOL_FIXED - b.mass * G
        assert F_net < 0.0, "Upward displacement must produce net downward force"

    def test_passive_restoring_force_downward_displacement(self, atmosphere):
        """Downward displacement should yield a net upward force (restoring)."""
        b = BalloonSP(dim=1, atmosphere=atmosphere, position=[ALT_DEFAULT])
        # Move 500 m below float altitude
        b.pos[0] = ALT_DEFAULT - 500.0
        rho = atmosphere.density(b.altitude)
        F_net = rho * G * SP_VOL_FIXED - b.mass * G
        assert F_net > 0.0, "Downward displacement must produce net upward force"

    def test_passive_stability_tendency(self, atmosphere):
        """Balloon displaced up without control should trend back toward float altitude."""
        b = BalloonSP(dim=1, atmosphere=atmosphere, position=[ALT_DEFAULT + 500.0])
        b.vel[0] = 0.0
        # Run without control for 600 steps (10 min)
        for _ in range(600):
            b.update(1.0)
        # Should be closer to ALT_DEFAULT than initial displaced position
        assert abs(b.altitude - ALT_DEFAULT) < 500.0


class TestBalloonSPForceSymmetry:
    """Tests for equal and opposite authority of pump_in vs pump_out."""

    def test_action_force_symmetry(self, atmosphere):
        """pump_in and pump_out produce equal magnitude mass changes (within 1%)."""
        b_in = BalloonSP(dim=1, atmosphere=atmosphere, position=[ALT_DEFAULT])
        b_out = BalloonSP(dim=1, atmosphere=atmosphere, position=[ALT_DEFAULT])
        m0 = b_in.mass

        b_in.pump_in()
        b_out.pump_out()

        delta_in = b_in.mass - m0    # positive: heavier
        delta_out = m0 - b_out.mass  # positive: lighter

        assert abs(delta_in) == pytest.approx(abs(delta_out), rel=0.01)

    def test_net_force_symmetry_at_float_altitude(self, atmosphere):
        """Force change from pump_in equals force change from pump_out in magnitude."""
        b = BalloonSP(dim=1, atmosphere=atmosphere, position=[ALT_DEFAULT])
        rho = atmosphere.density(ALT_DEFAULT)

        # Net force at neutral buoyancy ≈ 0
        F_neutral = rho * G * SP_VOL_FIXED - b.mass * G

        b_in = BalloonSP(dim=1, atmosphere=atmosphere, position=[ALT_DEFAULT])
        b_out = BalloonSP(dim=1, atmosphere=atmosphere, position=[ALT_DEFAULT])
        b_in.pump_in()
        b_out.pump_out()

        F_after_in = rho * G * SP_VOL_FIXED - b_in.mass * G
        F_after_out = rho * G * SP_VOL_FIXED - b_out.mass * G

        delta_in = F_neutral - F_after_in    # magnitude of downward shift
        delta_out = F_after_out - F_neutral  # magnitude of upward shift

        assert abs(delta_in) == pytest.approx(abs(delta_out), rel=0.01)


class TestBalloonSPPhysicsIntegration:
    """Tests for physics integration with SP balloon."""

    def test_update_advances_time(self, atmosphere):
        """update() should advance internal time."""
        b = BalloonSP(dim=1, atmosphere=atmosphere, position=[ALT_DEFAULT])
        b.update(1.0)
        assert b.t == pytest.approx(1.0)

    def test_ground_clamping(self, atmosphere):
        """SP balloon should not go below ground."""
        b = BalloonSP(dim=1, atmosphere=atmosphere, position=[100.0])
        b.vel[0] = -200.0
        b.update(10.0)
        assert b.altitude >= 0.0

    def test_wind_moves_balloon_horizontally(self, atmosphere):
        """Wind should accelerate a stationary SP balloon via drag."""
        b = BalloonSP(dim=3, atmosphere=atmosphere, position=[0.0, 0.0, ALT_DEFAULT])
        b.vel[:] = 0.0
        x0 = b.x
        b.update(1.0, wind_vel=np.array([5.0, 0.0, 0.0]))
        assert b.x > x0
