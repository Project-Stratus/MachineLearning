import numpy as np

from environments.constants import SCALE_HEIGHT, P0, R, T_AIR, M_AIR

# SCALE_HEIGHT = 8500      # scale height (m)
# P0 = 101325              # sea-level atmospheric pressure (Pa)
# R = 8.314462618          # universal gas constant (J/(mol·K))
# T_AIR = 273.15 + 15      # ambient air temperature (K)
# M_AIR = 0.0289647        # molar mass of air (kg/mol)


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
        return self.p0 * np.exp(-altitude / self.scale_height)

    def density(self, altitude):
        """
        Returns air density (kg/m^3) at a given altitude
        via the ideal gas law: rho = P * M_air / (R * T).
        """
        p = self.pressure(altitude)
        rho = p * self.molar_mass / (R * self.temperature)
        return rho
