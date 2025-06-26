# ----- UNIVERSAL CONSTANTS -----
G = 9.81                    # gravitational acceleration (m/s^2)
R = 8.314462618             # universal gas constant (J/(mol*K))
P0 = 101325                 # sea-level standard atmospheric pressure (Pa)
M_AIR = 0.0289647           # molar mass of air (kg/mol)

# ----- ENVIRONMENT CONSTANTS -----
T_BALLOON = 273.15 + 20     # internal gas temperature (K) (~20°C)
T_AIR = 273.15 + 15         # ambient air temperature (K)
SCALE_HEIGHT = 8500         # scale height (m)
ALT_MAX = 20_000.0          # Maximum altitude (m)
DT = 1.0                    # Time step (s)

# ----- BALLOON CONSTANTS -----
VOL_MAX = 20.0              # Maximum balloon volume (m³)
VEL_MAX = 50.0              # Maximum velocity (m/s)
P_MAX = 1.0e5               # Maximum pressure (Pa)
CD = 0.5                    # Drag coefficient (dimensionless)
AREA = 1.0                  # Cross-sectional area of the balloon (m²)

# ----- CODE CONSTANTS -----
SEED = 42                    # Random seed for reproducibility
