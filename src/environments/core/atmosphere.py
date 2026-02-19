import numpy as np

from environments.core.constants import (
    P0, R, M_AIR, G,
    T0, LAPSE_RATE, TROPOPAUSE_ALT, T_TROPOPAUSE,
    MU_REF, T_REF, S_SUTH,
)

try:
    from environments.core.jit_kernels import pressure_numba, density_numba, temperature_numba
    _JIT_OK = True
except Exception:
    _JIT_OK = False


class Atmosphere:
    """
    International Standard Atmosphere (ISA) model.

    Temperature profile:
      - Troposphere (0–11 km): T decreases linearly at 6.5 K/km
      - Stratosphere (>11 km): T is constant at ~216.65 K

    Pressure and density follow from the hydrostatic equation and ideal gas law.
    """

    def __init__(self, p0=P0, molar_mass=M_AIR):
        self.p0 = p0
        self.molar_mass = molar_mass

    def temperature(self, altitude):
        """Returns ISA temperature (K) at a given altitude."""
        if _JIT_OK:
            return float(temperature_numba(altitude))
        if altitude <= TROPOPAUSE_ALT:
            return T0 - LAPSE_RATE * altitude
        return T_TROPOPAUSE

    def pressure(self, altitude):
        """
        Returns atmospheric pressure (Pa) at a given altitude using the ISA
        barometric formula.

        Troposphere: power-law  P = P0 * (T/T0)^(gM/RL)
        Stratosphere: exponential  P = P_tropo * exp(-gM(h-h_tropo)/(RT_tropo))
        """
        if _JIT_OK:
            return float(pressure_numba(self.p0, altitude))
        return self._pressure_py(altitude)

    def density(self, altitude):
        """
        Returns air density (kg/m^3) at a given altitude via the ideal gas law:
        rho = P * M_air / (R * T).
        """
        if _JIT_OK:
            return float(density_numba(self.p0, self.molar_mass, altitude))
        p = self._pressure_py(altitude)
        T = self.temperature(altitude)
        return p * self.molar_mass / (R * T)

    def dynamic_viscosity(self, altitude):
        """
        Returns dynamic viscosity of air (Pa·s) at a given altitude using
        Sutherland's law: mu = mu_ref * (T/T_ref)^(3/2) * (T_ref + S) / (T + S).
        """
        T = self.temperature(altitude)
        return MU_REF * (T / T_REF) ** 1.5 * (T_REF + S_SUTH) / (T + S_SUTH)

    def _pressure_py(self, altitude):
        """Pure-Python ISA pressure (used when JIT is unavailable)."""
        gM_RL = G * self.molar_mass / (R * LAPSE_RATE)
        if altitude <= TROPOPAUSE_ALT:
            return self.p0 * (1.0 - LAPSE_RATE * altitude / T0) ** gM_RL
        # Stratosphere: exponential decay from tropopause values
        p_tropo = self.p0 * (T_TROPOPAUSE / T0) ** gM_RL
        scale_h = R * T_TROPOPAUSE / (self.molar_mass * G)
        return p_tropo * np.exp(-(altitude - TROPOPAUSE_ALT) / scale_h)
