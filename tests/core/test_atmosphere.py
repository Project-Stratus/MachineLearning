"""Tests for the Atmosphere class - pressure and density calculations."""

import numpy as np
import pytest

from environments.core.atmosphere import Atmosphere
from environments.core.constants import R


class TestAtmospherePressure:
    """Tests for atmospheric pressure calculations."""

    def test_pressure_at_sea_level(self, atmosphere):
        """Pressure at sea level should match P0 (approximately 101325 Pa)."""
        p0 = atmosphere.pressure(0.0)
        assert 1.0e5 < p0 < 1.05e5, f"Sea level pressure {p0} outside expected range"

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
        # Standard atmosphere: ~26.5 kPa at 10km
        assert 2.0e4 < p_10km < 3.5e4, f"Pressure at 10km ({p_10km}) outside expected range"

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
        """Density at 10km should be roughly 40% of sea level."""
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
    """Tests for consistency between pressure and density."""

    def test_ideal_gas_law_consistency(self, atmosphere):
        """Pressure and density should be consistent with ideal gas law: P = rho * R * T / M."""
        for alt in [0, 5000, 10000, 15000, 20000]:
            p = atmosphere.pressure(alt)
            rho = atmosphere.density(alt)
            T = atmosphere.temperature
            M = atmosphere.molar_mass

            # From ideal gas: P = rho * R * T / M
            p_calculated = rho * R * T / M
            assert np.isclose(p, p_calculated, rtol=1e-6), (
                f"Ideal gas law mismatch at {alt}m: P={p}, calculated={p_calculated}"
            )

    def test_atmosphere_parameters_reasonable(self, atmosphere):
        """Atmosphere parameters should have physically reasonable values."""
        assert 100_000 < atmosphere.p0 < 110_000, "P0 should be ~101325 Pa"
        assert 7000 < atmosphere.scale_height < 9000, "Scale height should be ~8500 m"
        assert 250 < atmosphere.temperature < 320, "Temperature should be reasonable"
        assert 0.025 < atmosphere.molar_mass < 0.035, "Molar mass of air should be ~0.029 kg/mol"
