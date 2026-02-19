import numpy as np

# ----- UNIVERSAL CONSTANTS -----
G = 9.81                    # gravitational acceleration (m/s^2)
R = 8.314462618             # universal gas constant (J/(mol*K))
P0 = 101325                 # sea-level standard atmospheric pressure (Pa)
M_AIR = 0.0289647           # molar mass of air (kg/mol)

# ----- ISA TEMPERATURE MODEL -----
T0 = 288.15                 # sea-level standard temperature (K)
LAPSE_RATE = 0.0065         # tropospheric lapse rate (K/m)
TROPOPAUSE_ALT = 11_000.0   # tropopause altitude (m)
T_TROPOPAUSE = T0 - LAPSE_RATE * TROPOPAUSE_ALT  # ~216.65 K

# ----- ENVIRONMENT CONSTANTS -----
T_BALLOON = 273.15 + 20     # internal gas temperature (K) (~20°C)
XY_MAX = 50_000.0           # Maximum horizontal operating distance from origin (m)
DT = 1.0                    # Time step (s)

# ----- BALLOON CONSTANTS -----
PAYLOAD_MASS = 2.0          # Fixed structural mass: envelope + gondola + electronics (kg)
BALLAST_INITIAL = 5.0       # Expendable ballast mass at launch (kg)
BALLAST_DROP = 0.02         # Ballast mass dropped per "drop" action (kg)
VENT_RATE = 0.19            # Gas volume equivalent vented per "vent" action (m³)
MASS = PAYLOAD_MASS + BALLAST_INITIAL  # Total structural mass (kg), used for ALT_MAX derivation
VOL_MAX = 180.6             # Maximum balloon volume (m³) — gives ALT_MAX ≈ 40 km
VOL_MIN = 0.1               # Minimum volume before deflation (m³)
M_HE = 0.004002602          # Molar mass of helium (kg/mol)
OSCILLATION_AMP = 0.05      # Breathing oscillation amplitude (fraction of stationary volume)
OSCILLATION_PERIOD = 60.0   # Breathing oscillation period (s)

# ----- DYNAMIC VISCOSITY (Sutherland's law reference) -----
MU_REF = 1.716e-5           # reference dynamic viscosity of air (Pa·s) at T_REF
T_REF = 273.15              # reference temperature for Sutherland's law (K)
S_SUTH = 110.4              # Sutherland's constant for air (K)

# ----- DERIVED CONSTANTS (ISA model) -----
# Sea-level air density via ideal gas law
RHO_0 = P0 * M_AIR / (R * T0)  # ≈ 1.225 kg/m³

# Altitude ceiling — computed from ISA pressure model.
# Balloon pops when neutral-buoyancy volume = VOL_MAX, i.e. MASS / rho_air = VOL_MAX.
# We solve for the altitude where rho_air = MASS / VOL_MAX using the ISA model.
# In the troposphere: rho(h) = rho0 * (1 - L*h/T0)^(g*M/(R*L) - 1)
_EXP_TROPO = G * M_AIR / (R * LAPSE_RATE) - 1.0
_RHO_TARGET = MASS / VOL_MAX
_RHO_RATIO = _RHO_TARGET / RHO_0

# Check if target density falls within troposphere
_RHO_TROPOPAUSE = RHO_0 * (1.0 - LAPSE_RATE * TROPOPAUSE_ALT / T0) ** _EXP_TROPO
if _RHO_TARGET >= _RHO_TROPOPAUSE:
    # Target altitude is in the troposphere
    ALT_MAX = (T0 / LAPSE_RATE) * (1.0 - _RHO_RATIO ** (1.0 / _EXP_TROPO))
else:
    # Target altitude is in the stratosphere
    _P_TROPO = P0 * (T_TROPOPAUSE / T0) ** (G * M_AIR / (R * LAPSE_RATE))
    _RHO_TROPO_EXACT = _P_TROPO * M_AIR / (R * T_TROPOPAUSE)
    _SCALE_H_STRATO = R * T_TROPOPAUSE / (M_AIR * G)
    ALT_MAX = TROPOPAUSE_ALT + _SCALE_H_STRATO * np.log(_RHO_TROPO_EXACT / _RHO_TARGET)

ALT_DEFAULT = 0.5 * ALT_MAX  # Default starting altitude (m) — midpoint of operating range
VEL_MAX = 200.0             # Maximum velocity (m/s) - physics clamp limit
P_MAX = 1.0e5               # Maximum pressure (Pa)
SPEED_EPS = 1e-12           # Speed threshold for drag computation (m/s²)

# ----- RESET CONSTANTS -----
MIN_START_DISTANCE = 500.0  # Minimum distance between start and goal on reset (m)

# ----- CODE CONSTANTS -----
SEED = 42                    # Random seed for reproducibility
