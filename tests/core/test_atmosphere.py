"""Tests for the Atmosphere class - ISA temperature, pressure and density."""

import numpy as np
import pytest

from environments.core.atmosphere import Atmosphere
from environments.core.constants import (
    R, P0, T0, LAPSE_RATE, TROPOPAUSE_ALT, T_TROPOPAUSE,
)


class TestAtmosphereTemperature:
    """Tests for ISA temperature profile."""

    def test_temperature_at_sea_level(self, atmosphere):
        """Temperature at sea level should be 288.15 K."""
        T = atmosphere.temperature(0.0)
        assert T == pytest.approx(T0)

    def test_temperature_decreases_in_troposphere(self, atmosphere):
        """Temperature should decrease with altitude in the troposphere."""
        temps = [atmosphere.temperature(alt) for alt in [0, 3000, 6000, 9000]]
        for i in range(1, len(temps)):
            assert temps[i] < temps[i - 1]

    def test_temperature_lapse_rate(self, atmosphere):
        """Temperature drop should match the ISA lapse rate (6.5 K/km)."""
        T_0 = atmosphere.temperature(0.0)
        T_5km = atmosphere.temperature(5000.0)
        expected_drop = LAPSE_RATE * 5000.0  # 32.5 K
        assert (T_0 - T_5km) == pytest.approx(expected_drop, rel=1e-6)

    def test_temperature_at_tropopause(self, atmosphere):
        """Temperature at the tropopause should be ~216.65 K."""
        T = atmosphere.temperature(TROPOPAUSE_ALT)
        assert T == pytest.approx(T_TROPOPAUSE, rel=1e-6)

    def test_temperature_constant_in_stratosphere(self, atmosphere):
        """Temperature should be constant above the tropopause."""
        T_12km = atmosphere.temperature(12_000.0)
        T_20km = atmosphere.temperature(20_000.0)
        T_30km = atmosphere.temperature(30_000.0)
        assert T_12km == pytest.approx(T_TROPOPAUSE)
        assert T_20km == pytest.approx(T_TROPOPAUSE)
        assert T_30km == pytest.approx(T_TROPOPAUSE)


class TestAtmospherePressure:
    """Tests for atmospheric pressure calculations."""

    def test_pressure_at_sea_level(self, atmosphere):
        """Pressure at sea level should match P0 (approximately 101325 Pa)."""
        p0 = atmosphere.pressure(0.0)
        assert p0 == pytest.approx(P0, rel=1e-6)

    def test_pressure_decreases_with_altitude(self, atmosphere):
        """Pressure should decrease monotonically with altitude."""
        altitudes = [0, 5000, 10000, 15000, 20000, 25000, 30000]
        pressures = [atmosphere.pressure(alt) for alt in altitudes]

        for i in range(1, len(pressures)):
            assert pressures[i] < pressures[i - 1], (
                f"Pressure at {altitudes[i]}m ({pressures[i]}) >= "
                f"pressure at {altitudes[i-1]}m ({pressures[i-1]})"
            )

    def test_pressure_at_10km(self, atmosphere):
        """Pressure at 10km should be roughly 26% of sea level (26.5 kPa)."""
        p_10km = atmosphere.pressure(10_000.0)
        assert 2.0e4 < p_10km < 3.5e4, f"Pressure at 10km ({p_10km}) outside expected range"

    def test_pressure_continuous_at_tropopause(self, atmosphere):
        """Pressure should be continuous across the tropopause boundary."""
        p_below = atmosphere.pressure(TROPOPAUSE_ALT - 1.0)
        p_at = atmosphere.pressure(TROPOPAUSE_ALT)
        p_above = atmosphere.pressure(TROPOPAUSE_ALT + 1.0)
        # Should be monotonically decreasing and close together
        assert p_below > p_at > p_above
        assert abs(p_below - p_at) / p_at < 0.001

    def test_pressure_positive_at_high_altitude(self, atmosphere):
        """Pressure should remain positive even at very high altitudes."""
        p_high = atmosphere.pressure(50_000.0)
        assert p_high > 0, "Pressure should be positive"

    def test_pressure_type(self, atmosphere):
        """Pressure should return a float."""
        p = atmosphere.pressure(10_000.0)
        assert isinstance(p, (float, np.floating))


class TestAtmosphereDensity:
    """Tests for atmospheric density calculations."""

    def test_density_at_sea_level(self, atmosphere):
        """Density at sea level should be approximately 1.225 kg/m^3."""
        rho0 = atmosphere.density(0.0)
        assert 1.0 < rho0 < 1.4, f"Sea level density {rho0} outside expected range"

    def test_density_decreases_with_altitude(self, atmosphere):
        """Density should decrease monotonically with altitude."""
        altitudes = [0, 5000, 10000, 15000, 20000, 25000, 30000]
        densities = [atmosphere.density(alt) for alt in altitudes]

        for i in range(1, len(densities)):
            assert densities[i] < densities[i - 1], (
                f"Density at {altitudes[i]}m ({densities[i]}) >= "
                f"density at {altitudes[i-1]}m ({densities[i-1]})"
            )

    def test_density_at_10km(self, atmosphere):
        """Density at 10km should be roughly 34% of sea level (ISA value)."""
        rho_10km = atmosphere.density(10_000.0)
        rho_0 = atmosphere.density(0.0)
        ratio = rho_10km / rho_0
        assert 0.3 < ratio < 0.5, f"Density ratio at 10km ({ratio}) outside expected range"

    def test_density_positive_at_high_altitude(self, atmosphere):
        """Density should remain positive even at very high altitudes."""
        rho_high = atmosphere.density(50_000.0)
        assert rho_high > 0, "Density should be positive"

    def test_density_type(self, atmosphere):
        """Density should return a float."""
        rho = atmosphere.density(10_000.0)
        assert isinstance(rho, (float, np.floating))


class TestAtmosphereConsistency:
    """Tests for consistency between pressure, temperature and density."""

    def test_ideal_gas_law_consistency(self, atmosphere):
        """Pressure and density should be consistent with ideal gas law at all altitudes."""
        for alt in [0, 5000, 10000, 11000, 15000, 20000]:
            p = atmosphere.pressure(alt)
            rho = atmosphere.density(alt)
            T = atmosphere.temperature(alt)
            M = atmosphere.molar_mass

            # From ideal gas: P = rho * R * T / M
            p_calculated = rho * R * T / M
            assert np.isclose(p, p_calculated, rtol=1e-6), (
                f"Ideal gas law mismatch at {alt}m: P={p}, calculated={p_calculated}"
            )

    def test_atmosphere_parameters_reasonable(self, atmosphere):
        """Atmosphere parameters should have physically reasonable values."""
        assert 100_000 < atmosphere.p0 < 110_000, "P0 should be ~101325 Pa"
        assert 0.025 < atmosphere.molar_mass < 0.035, "Molar mass of air should be ~0.029 kg/mol"


class TestAtmosphereViscosity:
    """Tests for dynamic viscosity via Sutherland's law."""

    def test_viscosity_at_sea_level(self, atmosphere):
        """Dynamic viscosity at sea level should be ~1.79e-5 Pa·s."""
        mu = atmosphere.dynamic_viscosity(0.0)
        assert 1.7e-5 < mu < 1.9e-5

    def test_viscosity_decreases_with_altitude(self, atmosphere):
        """Viscosity should decrease with altitude (lower temperature)."""
        mu_0 = atmosphere.dynamic_viscosity(0.0)
        mu_10km = atmosphere.dynamic_viscosity(10_000.0)
        assert mu_10km < mu_0

    def test_viscosity_constant_in_stratosphere(self, atmosphere):
        """Viscosity should be constant in the stratosphere (constant T)."""
        mu_15km = atmosphere.dynamic_viscosity(15_000.0)
        mu_25km = atmosphere.dynamic_viscosity(25_000.0)
        assert mu_15km == pytest.approx(mu_25km, rel=1e-6)

    def test_viscosity_positive(self, atmosphere):
        """Viscosity should always be positive."""
        for alt in [0, 5000, 11000, 20000, 40000]:
            assert atmosphere.dynamic_viscosity(alt) > 0
