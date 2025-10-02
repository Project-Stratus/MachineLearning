import numpy as np

from environments.core.constants import SCALE_HEIGHT, P0, R, T_AIR, M_AIR

try:
    from environments.core.jit_kernels import pressure_numba, density_numba
    _JIT_OK = True
except Exception:
    _JIT_OK = False


class Atmosphere:
    """
    Simple model of the atmosphere to retrieve pressure and density
    as functions of altitude.
    """
    def __init__(self,
                 p0=P0,
                 scale_height=SCALE_HEIGHT,
                 temperature=T_AIR,
                 molar_mass=M_AIR
                 ):
        self.p0 = p0
        self.scale_height = scale_height
        self.temperature = temperature
        self.molar_mass = molar_mass

    def pressure(self, altitude):
        """
        Returns external pressure (Pa) at a given altitude
        using the exponential model: P(h) = P0 * exp(-h/H).
        """
        if _JIT_OK:
            return float(pressure_numba(self.p0, self.scale_height, altitude))
        return self.p0 * np.exp(-altitude / self.scale_height)

    def density(self, altitude):
        """
        Returns air density (kg/m^3) at a given altitude
        via the ideal gas law: rho = P * M_air / (R * T).
        """
        if _JIT_OK:
            return float(density_numba(self.p0, self.scale_height, self.temperature, self.molar_mass, R, altitude))
        p = self.pressure(altitude)
        rho = p * self.molar_mass / (R * self.temperature)
        return rho
