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

# Altitude ceiling — where balloon at VOL_MAX reaches neutral buoyancy.
# Accounts for helium gas mass:
#   rho_air * VOL_MAX = MASS + P * VOL_MAX * M_HE / (R * T_BALLOON)
# Rearranged: rho_eff(h) = rho_air - P * M_HE / (R * T_BALLOON) = MASS / VOL_MAX
_GM_RL = G * M_AIR / (R * LAPSE_RATE)
_EXP_TROPO = _GM_RL - 1.0
_MASS_OVER_VOL = MASS / VOL_MAX

# Tropopause reference values
_P_TROPO = P0 * (T_TROPOPAUSE / T0) ** _GM_RL
_RHO_TROPO = _P_TROPO * M_AIR / (R * T_TROPOPAUSE)
_RHO_EFF_TROPO = _RHO_TROPO - _P_TROPO * M_HE / (R * T_BALLOON)

if _MASS_OVER_VOL >= _RHO_EFF_TROPO:
    # Target altitude is in the troposphere (unlikely with current params).
    # Bisection: rho_eff(h) has no closed form when T varies with h.
    _lo, _hi = 0.0, TROPOPAUSE_ALT
    for _ in range(60):
        _mid = 0.5 * (_lo + _hi)
        _T = T0 - LAPSE_RATE * _mid
        _P = P0 * (_T / T0) ** _GM_RL
        _rho_eff = _P * M_AIR / (R * _T) - _P * M_HE / (R * T_BALLOON)
        if _rho_eff > _MASS_OVER_VOL:
            _lo = _mid
        else:
            _hi = _mid
    ALT_MAX = 0.5 * (_lo + _hi)
else:
    # Target altitude is in the stratosphere (T = T_TROPOPAUSE = const).
    # Analytical: P_target = (MASS/VOL_MAX) / (M_AIR/(R*T_tropo) - M_HE/(R*T_balloon))
    _SCALE_H_STRATO = R * T_TROPOPAUSE / (M_AIR * G)
    _K = M_AIR / (R * T_TROPOPAUSE) - M_HE / (R * T_BALLOON)
    _P_TARGET = _MASS_OVER_VOL / _K
    ALT_MAX = TROPOPAUSE_ALT + _SCALE_H_STRATO * np.log(_P_TROPO / _P_TARGET)

ALT_DEFAULT = 0.5 * ALT_MAX  # Default starting altitude (m) — midpoint of operating range
VEL_MAX = 200.0             # Maximum velocity (m/s) - physics clamp limit
P_MAX = 1.0e5               # Maximum pressure (Pa)
SPEED_EPS = 1e-12           # Speed threshold for drag computation (m/s²)

# ----- RESET CONSTANTS -----
MIN_START_DISTANCE = 500.0  # Minimum distance between start and goal on reset (m)

# ----- INITIAL CONDITION RANDOMISATION -----
# Domain randomisation of initial conditions for robustness.
# Simulates the variability a balloon experiences after reaching float altitude.
INIT_VEL_SIGMA = 2.0       # Std dev of initial velocity perturbation (m/s)
INIT_GAS_FRAC_RANGE = 0.05 # Max fractional deviation from neutral gas amount (±5%)
INIT_BALLAST_LOSS_MAX = 0.5 # Max ballast (kg) that may have been spent during ascent

# ----- CODE CONSTANTS -----
SEED = 42                    # Random seed for reproducibility
