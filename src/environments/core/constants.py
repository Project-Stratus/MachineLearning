import numpy as np

# ----- UNIVERSAL CONSTANTS -----
G = 9.81                    # gravitational acceleration (m/s^2)
R = 8.314462618             # universal gas constant (J/(mol*K))
P0 = 101325                 # sea-level standard atmospheric pressure (Pa)
M_AIR = 0.0289647           # molar mass of air (kg/mol)

# ----- ENVIRONMENT CONSTANTS -----
T_BALLOON = 273.15 + 20     # internal gas temperature (K) (~20°C)
T_AIR = 273.15 + 15         # ambient air temperature (K)
SCALE_HEIGHT = 8500         # scale height (m)
XY_MAX = 10_000.0           # Maximum horizontal operating distance from origin (m)
DT = 1.0                    # Time step (s)

# ----- DERIVED CONSTANTS -----
RHO_0 = P0 * M_AIR / (R * T_AIR)  # Sea-level air density (≈1.225 kg/m³)

# ----- BALLOON CONSTANTS -----
MASS = 2.0                  # Balloon mass (kg)
VOL_MAX = 180.6             # Maximum balloon volume (m³) — gives ALT_MAX ≈ 40 km
VOL_MIN = 0.1               # Minimum volume before deflation (m³)
CD = 0.5                    # Drag coefficient (dimensionless)
AREA = 1.0                  # Cross-sectional area of the balloon (m²)
OSCILLATION_AMP = 0.05      # Breathing oscillation amplitude (fraction of stationary volume)
OSCILLATION_PERIOD = 60.0   # Breathing oscillation period (s)

# Derived altitude ceiling — balloon pops when neutral buoyancy volume = VOL_MAX
ALT_MAX = SCALE_HEIGHT * np.log(RHO_0 * VOL_MAX / MASS)  # ≈21,300 m
ALT_DEFAULT = 0.5 * ALT_MAX  # Default starting altitude (m) — midpoint of operating range
VEL_MAX = 200.0             # Maximum velocity (m/s) - physics clamp limit
P_MAX = 1.0e5               # Maximum pressure (Pa)
SPEED_EPS = 1e-12           # Speed threshold for drag computation (m/s²)

# ----- RESET CONSTANTS -----
MIN_START_DISTANCE = 500.0  # Minimum distance between start and goal on reset (m)

# ----- CODE CONSTANTS -----
SEED = 42                    # Random seed for reproducibility
