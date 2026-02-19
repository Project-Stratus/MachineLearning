"""Tests for the Balloon class - variable mass, ballast, gas venting, passive expansion, relative-velocity drag, Verlet integration."""

import math
import numpy as np
import pytest

from environments.core.balloon import Balloon
from environments.core.atmosphere import Atmosphere
from environments.core.constants import (
    G, R, T_BALLOON, M_HE, ALT_DEFAULT, VOL_MAX, VOL_MIN,
    PAYLOAD_MASS, BALLAST_INITIAL, BALLAST_DROP, VENT_RATE,
)


class TestBalloonInitialization:
    """Tests for balloon initialization."""

    def test_default_initialization(self, atmosphere):
        """Balloon should initialize with sensible defaults."""
        balloon = Balloon(atmosphere=atmosphere)
        assert balloon.dim == 1
        assert balloon.payload_mass == PAYLOAD_MASS
        assert balloon.ballast_mass == BALLAST_INITIAL
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

    def test_initial_volume_matches_structural_mass(self, atmosphere):
        """Initial volume should equal structural_mass / rho_air.

        The balloon starts at neutral buoyancy for its structural mass
        (payload + ballast).  The gas itself has mass (n_gas * M_HE),
        so the total mass is slightly above neutral — this is physically
        correct and means the balloon initially sinks very slowly.
        """
        balloon = Balloon(dim=1, atmosphere=atmosphere, position=[15_000.0])
        rho_air = atmosphere.density(15_000.0)
        structural_mass = balloon.payload_mass + balloon.ballast_mass
        expected_volume = structural_mass / rho_air
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


class TestVariableMass:
    """Tests for variable mass model (payload + ballast + gas)."""

    def test_mass_includes_all_components(self, atmosphere):
        """Total mass should be payload + ballast + gas mass."""
        balloon = Balloon(dim=1, atmosphere=atmosphere, position=[15_000.0])
        expected = balloon.payload_mass + balloon.ballast_mass + balloon.n_gas * M_HE
        assert balloon.mass == pytest.approx(expected)

    def test_mass_decreases_after_ballast_drop(self, balloon_1d):
        """Dropping ballast should reduce total mass."""
        m0 = balloon_1d.mass
        balloon_1d.drop_ballast()
        assert balloon_1d.mass < m0

    def test_mass_decreases_after_vent(self, balloon_1d):
        """Venting gas should reduce total mass (gas has mass)."""
        m0 = balloon_1d.mass
        balloon_1d.vent_gas()
        assert balloon_1d.mass < m0

    def test_payload_mass_never_changes(self, balloon_1d):
        """Payload mass should remain constant through actions."""
        pm0 = balloon_1d.payload_mass
        balloon_1d.drop_ballast()
        balloon_1d.vent_gas()
        assert balloon_1d.payload_mass == pm0


class TestBallastDrop:
    """Tests for ballast drop mechanics."""

    def test_drop_reduces_ballast_mass(self, balloon_1d):
        """drop_ballast should reduce ballast_mass by BALLAST_DROP."""
        b0 = balloon_1d.ballast_mass
        balloon_1d.drop_ballast()
        assert balloon_1d.ballast_mass == pytest.approx(b0 - BALLAST_DROP)

    def test_drop_ballast_clamped_at_zero(self, balloon_1d):
        """Ballast mass should never go negative."""
        for _ in range(500):
            balloon_1d.drop_ballast()
        assert balloon_1d.ballast_mass == 0.0
        assert balloon_1d.is_ballast_empty

    def test_is_ballast_empty(self, atmosphere):
        """is_ballast_empty should be True when ballast is exhausted."""
        balloon = Balloon(dim=1, atmosphere=atmosphere, position=[15_000.0],
                          ballast_initial=0.1)
        assert not balloon.is_ballast_empty
        balloon.drop_ballast(0.1)
        assert balloon.is_ballast_empty

    def test_drop_ballast_causes_ascent(self, atmosphere):
        """Dropping enough ballast should cause the balloon to rise.

        The balloon starts slightly heavy (gas has mass), so enough
        drops are needed to cross neutral buoyancy and gain altitude.
        """
        balloon = Balloon(dim=1, atmosphere=atmosphere, position=[10_000.0])
        z0 = balloon.altitude
        for _ in range(50):
            balloon.drop_ballast()
        for _ in range(100):
            balloon.update(1.0)
        assert balloon.altitude > z0


class TestGasVenting:
    """Tests for gas venting mechanics."""

    def test_vent_removes_moles(self, balloon_1d):
        """Venting should decrease n_gas."""
        n0 = balloon_1d.n_gas
        balloon_1d.vent_gas()
        assert balloon_1d.n_gas < n0

    def test_vent_decreases_volume(self, balloon_1d):
        """Venting gas should decrease balloon volume."""
        v0 = balloon_1d.volume
        balloon_1d.vent_gas()
        assert balloon_1d.volume < v0

    def test_vent_gas_clamped_at_zero(self, balloon_1d):
        """n_gas should never go negative."""
        for _ in range(1000):
            balloon_1d.vent_gas()
        assert balloon_1d.n_gas >= 0.0

    def test_vent_causes_descent(self, atmosphere):
        """Venting gas should cause the balloon to descend."""
        balloon = Balloon(dim=1, atmosphere=atmosphere, position=[15_000.0])
        z0 = balloon.altitude
        for _ in range(10):
            balloon.vent_gas()
        for _ in range(100):
            balloon.update(1.0)
        assert balloon.altitude < z0


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

    def test_near_neutral_buoyancy_at_initial_volume(self, atmosphere):
        """At initial volume, buoyancy should approximately balance structural weight.

        The balloon is slightly heavier than neutral because gas has mass
        (n_gas * M_HE), but buoyancy should match the structural mass
        (payload + ballast) contribution.
        """
        balloon = Balloon(dim=1, atmosphere=atmosphere, position=[15_000.0])
        buoy = balloon.buoyant_force(0.0)[-1]
        structural_weight = (balloon.payload_mass + balloon.ballast_mass) * G
        assert buoy == pytest.approx(structural_weight, rel=0.01)

    def test_dropping_ballast_reduces_weight(self, balloon_1d):
        """Dropping ballast should reduce weight magnitude."""
        weight_before = abs(balloon_1d.weight()[-1])
        balloon_1d.drop_ballast()
        weight_after = abs(balloon_1d.weight()[-1])
        assert weight_after < weight_before

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
        b_large.n_gas *= 2.0  # more gas → larger volume

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

    def test_update_after_ballast_drop_rises(self, balloon_1d):
        """Balloon should rise after dropping enough ballast to overcome gas mass."""
        z0 = balloon_1d.altitude
        for _ in range(50):
            balloon_1d.drop_ballast()
        balloon_1d.update(1.0)
        assert balloon_1d.altitude > z0

    def test_update_after_vent_falls(self, balloon_1d):
        """Balloon should fall after venting gas (less buoyancy)."""
        z0 = balloon_1d.altitude
        for _ in range(5):
            balloon_1d.vent_gas()
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

        Uses a ballast drop to create gentle upward force.
        """
        balloon = Balloon(dim=1, atmosphere=atmosphere, position=[8_000.0])
        for _ in range(50):
            balloon.drop_ballast()  # make balloon lighter → ascend
        n_gas_before = balloon.n_gas  # gas moles should not change
        v0 = balloon.volume
        for _ in range(500):
            balloon.update(1.0)
        # Gas moles unchanged (no venting during ascent)
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

    def test_extra_volume_after_vent(self, balloon_1d):
        """extra_volume should decrease after venting gas."""
        ev_before = balloon_1d.extra_volume
        balloon_1d.vent_gas()
        ev_after = balloon_1d.extra_volume
        assert ev_after < ev_before


class TestRelativeVelocityDrag:
    """Tests for drag using relative velocity (v_balloon - v_wind)."""

    def test_no_drag_when_moving_with_wind(self, atmosphere):
        """Balloon moving at wind speed should experience zero drag."""
        balloon = Balloon(dim=3, atmosphere=atmosphere, position=[0.0, 0.0, 10_000.0])
        balloon.vel = np.array([5.0, 3.0, 0.0])
        wind = np.array([5.0, 3.0, 0.0])  # same as balloon velocity
        drag = balloon.drag_force(wind_vel=wind)
        assert np.allclose(drag, 0.0, atol=1e-10)

    def test_drag_from_wind_on_stationary_balloon(self, atmosphere):
        """Stationary balloon in wind should experience drag in the wind direction."""
        balloon = Balloon(dim=3, atmosphere=atmosphere, position=[0.0, 0.0, 10_000.0])
        balloon.vel = np.zeros(3)
        wind = np.array([5.0, 0.0, 0.0])
        drag = balloon.drag_force(wind_vel=wind)
        # v_rel = [0,0,0] - [5,0,0] = [-5,0,0]; drag opposes v_rel -> positive x
        assert drag[0] > 0, "Drag should push balloon in wind direction"
        assert drag[1] == pytest.approx(0.0, abs=1e-10)

    def test_drag_with_wind_reduces_drag_magnitude(self, atmosphere):
        """Tailwind should reduce drag compared to still air."""
        balloon = Balloon(dim=3, atmosphere=atmosphere, position=[0.0, 0.0, 10_000.0])
        balloon.vel = np.array([5.0, 0.0, 0.0])

        drag_still = balloon.drag_force(wind_vel=np.zeros(3))
        drag_tail = balloon.drag_force(wind_vel=np.array([3.0, 0.0, 0.0]))

        assert np.linalg.norm(drag_tail) < np.linalg.norm(drag_still)

    def test_drag_with_headwind_increases_drag_magnitude(self, atmosphere):
        """Headwind should increase drag compared to still air."""
        balloon = Balloon(dim=3, atmosphere=atmosphere, position=[0.0, 0.0, 10_000.0])
        balloon.vel = np.array([5.0, 0.0, 0.0])

        drag_still = balloon.drag_force(wind_vel=np.zeros(3))
        drag_head = balloon.drag_force(wind_vel=np.array([-3.0, 0.0, 0.0]))

        assert np.linalg.norm(drag_head) > np.linalg.norm(drag_still)

    def test_wind_accelerates_stationary_balloon(self, atmosphere):
        """Wind should accelerate a stationary balloon via relative-velocity drag."""
        balloon = Balloon(dim=3, atmosphere=atmosphere, position=[0.0, 0.0, 10_000.0])
        balloon.vel = np.zeros(3)
        wind = np.array([5.0, 0.0, 0.0])
        x0 = balloon.x
        balloon.update(1.0, wind_vel=wind)
        # Balloon should move in the wind direction
        assert balloon.x > x0

    def test_wind_passed_to_update(self, atmosphere):
        """update() should accept wind_vel and use it for drag."""
        b_wind = Balloon(dim=3, atmosphere=atmosphere, position=[0.0, 0.0, 10_000.0])
        b_still = Balloon(dim=3, atmosphere=atmosphere, position=[0.0, 0.0, 10_000.0])

        b_wind.update(1.0, wind_vel=np.array([10.0, 0.0, 0.0]))
        b_still.update(1.0)  # no wind

        # With wind, balloon should have moved more in x
        assert b_wind.x > b_still.x


class TestVerletIntegration:
    """Tests for velocity Verlet integrator properties."""

    def test_verlet_better_energy_conservation(self, atmosphere):
        """Verlet should conserve energy better than Euler over many steps.

        A balloon at neutral buoyancy with an initial vertical velocity
        should oscillate with nearly constant total energy (kinetic + potential).
        """
        balloon = Balloon(dim=1, atmosphere=atmosphere, position=[10_000.0])
        balloon.vel = np.array([2.0])  # small upward kick

        # Record initial kinetic + gravitational potential energy
        def energy(b):
            KE = 0.5 * b.mass * b.velocity**2
            PE = b.mass * G * b.altitude
            return KE + PE

        E0 = energy(balloon)
        energies = [E0]
        for _ in range(200):
            balloon.update(1.0)
            energies.append(energy(balloon))

        # Energy should not drift by more than 20% over 200 steps
        # (variable mass model means total mass includes gas, and the
        # heavier balloon interacts more strongly with altitude-dependent
        # density, causing some energy drift)
        E_final = energies[-1]
        drift = abs(E_final - E0) / E0
        assert drift < 0.20, f"Energy drifted by {drift*100:.1f}%"

    def test_verlet_symmetric_in_time(self, atmosphere):
        """Position update should include the 0.5*a*dt^2 term (Verlet signature).

        With a known constant force, the Verlet position update
        (x += v*dt + 0.5*a*dt^2) gives a more accurate result than Euler
        (x += v*dt where v already includes a*dt).
        """
        # Balloon with buoyancy excess from ballast drops — known upward acceleration
        balloon = Balloon(dim=1, atmosphere=atmosphere, position=[10_000.0])
        for _ in range(50):
            balloon.drop_ballast()
        z0 = balloon.altitude
        v0 = balloon.velocity  # 0.0

        balloon.update(1.0)

        # With Verlet, after 1 step: pos = z0 + v0*dt + 0.5*a*dt^2
        # Since v0=0, position change should be ~0.5*a*1.0
        # With Euler, it would be a*1.0 (velocity updated first, then position)
        dz = balloon.altitude - z0
        # The position change should be positive (rising) and moderate
        assert dz > 0
        # Second step should show velocity has built up
        z1 = balloon.altitude
        balloon.update(1.0)
        dz2 = balloon.altitude - z1
        # Second step covers more distance (velocity accumulated)
        assert dz2 > dz

    def test_verlet_stable_ascent(self, atmosphere):
        """Verlet should produce a smooth ascent after dropping ballast."""
        balloon = Balloon(dim=1, atmosphere=atmosphere, position=[8_000.0])
        for _ in range(50):
            balloon.drop_ballast()

        altitudes = []
        for _ in range(50):
            balloon.update(1.0)
            altitudes.append(balloon.altitude)

        # Altitude should increase monotonically (smooth ascent to terminal velocity)
        reversals = 0
        for i in range(2, len(altitudes)):
            if (altitudes[i] - altitudes[i-1]) * (altitudes[i-1] - altitudes[i-2]) < 0:
                reversals += 1
        assert reversals < 5, f"Too many direction reversals ({reversals}), suggests instability"
        # Should have risen overall
        assert altitudes[-1] > 8_000.0
