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

# ----- THERMAL MODEL -----
# Layer 1 is daytime-only, so a single calibrated superheat offset above the
# ambient ISA temperature suffices.  The gas temperature is therefore
#   T_gas(z) = T_ambient(z) + SUPERHEAT_DAY
# rather than a fixed absolute (the old ``T_BALLOON = 293.15 K`` implied a
# +76 K superheat against a 216.65 K stratosphere — see roadmap §3.6).
# Layer 2 replaces the constant with a radiative day/night model; the hook is
# ``Atmosphere.gas_temperature``.
SUPERHEAT_DAY = 15.0        # K above ambient (real daytime ZP: +10 to +30 K)

# ----- ENVIRONMENT CONSTANTS -----
XY_MAX = 50_000.0           # Wind-grid horizontal half-extent from origin (m)
DT = 1.0                    # Time step (s)

# ----- EPISODE -----
EPISODE_HOURS = 12.0                            # Mission duration (h)
TIME_MAX = int(EPISODE_HOURS * 3600 / DT)       # 43_200 physics steps
DECISION_INTERVAL = 60      # Physics steps per agent decision -> 720 decisions

# ----- OPERATIONAL ALTITUDE BAND -----
# The safety layer clamps the balloon into this band instead of terminating.
ALT_SAFE_MIN = 15_000.0     # Lower operational limit (m)
ALT_SAFE_MAX = 25_000.0     # Upper operational limit (m)

# ----- HORIZONTAL BOUNDS -----
# Soft bounds: there is no distance termination.  XY_ABORT exists purely as a
# numerical runaway guard.
XY_ABORT = 500_000.0        # Abort if |x| or |y| exceeds this (m)

# ----- WIND COLUMN OBSERVATION -----
WIND_COL_LEVELS = 41        # Number of altitude levels in the wind column
WIND_COL_SPACING = 250.0    # Vertical spacing between levels (m)
WIND_COL_HALF_SPAN = 5000.0 # Half-span of the column (m) == 20 * SPACING
WIND_MAG_NORM = 30.0        # Fixed wind-magnitude normaliser (m/s)

# ----- OBSERVATION -----
OBS_WIDTH = 143             # Frozen observation width (identical for dim 1/2/3)
DIST_NORM = 100_000.0       # Horizontal distance normaliser (m)

# ----- REWARD -----
STATION_RADIUS = 10_000.0        # Full-reward radius around the station (m)
REWARD_DROPOFF = 0.4             # Reward at the station-radius boundary
REWARD_HALFLIFE = 20_000.0       # Distance over which reward halves (m)

# 1-D reward scales.  In 1-D the "station" is a target *altitude* and distance is
# |dz|, which is bounded by the safety band (ALT_SAFE_MAX - ALT_SAFE_MIN =
# 10 km).  Reusing the horizontal 10 km radius therefore makes every state a
# full-reward state — the task becomes trivially solved by doing nothing, and
# all baselines score TWR 1.000.  The 1-D scales are sized to the band instead,
# so holding a target altitude is an actual control problem.
STATION_RADIUS_1D = 500.0        # Full-reward altitude tolerance (m)
REWARD_HALFLIFE_1D = 1_000.0     # Altitude error over which reward halves (m)
DIST_NORM_1D = 10_000.0          # Altitude-error normaliser for the observation (m)
RESOURCE_PENALTY_BASE = 0.97     # Multiplicative factor at omega -> 0+
RESOURCE_PENALTY_SLOPE = 0.3     # Extra penalty per unit consumed fraction

# ----- BALLOON CONSTANTS -----
PAYLOAD_MASS = 2.0          # Fixed structural mass: envelope + gondola + electronics (kg)
BALLAST_INITIAL = 5.0       # Expendable ballast mass at launch (kg)
BALLAST_DROP = 0.01         # Ballast mass dropped per "drop" action (kg)
# VENT_RATE / VENT_RATE_MOLES are derived from the float altitude — see below.
MASS = PAYLOAD_MASS + BALLAST_INITIAL  # Total structural mass (kg), used for ALT_MAX derivation
VOL_MAX = 450.0             # Maximum (design/burst) balloon volume (m³) — gives ALT_MAX ≈ 30 km
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

_GM_RL = G * M_AIR / (R * LAPSE_RATE)
_P_TROPO = P0 * (T_TROPOPAUSE / T0) ** _GM_RL
_H_STRATO = R * T_TROPOPAUSE / (M_AIR * G)   # stratospheric pressure scale height (m)


def _isa_temperature(h: float) -> float:
    """ISA ambient temperature (K) — duplicated here to keep constants.py leaf-level."""
    if h <= TROPOPAUSE_ALT:
        return T0 - LAPSE_RATE * h
    return T_TROPOPAUSE


def _isa_gas_temperature(h: float) -> float:
    """Balloon gas temperature (K): ambient plus the daytime superheat offset."""
    return _isa_temperature(h) + SUPERHEAT_DAY


def _isa_pressure(h: float) -> float:
    """ISA pressure (Pa)."""
    if h <= TROPOPAUSE_ALT:
        return P0 * (1.0 - LAPSE_RATE * h / T0) ** _GM_RL
    return _P_TROPO * np.exp(-(h - TROPOPAUSE_ALT) / _H_STRATO)


def _rho_effective(h: float) -> float:
    """Buoyant density net of the helium the envelope must itself carry.

    A balloon of volume V floats where  rho_air*V = m_struct + n*M_HE  with
    n = P*V/(R*T_gas), i.e. where

        rho_eff(h) = rho_air(h) - P(h)*M_HE/(R*T_gas(h)) = m_struct / V.

    T_gas now varies with altitude, so there is no closed form — bisect.
    rho_eff is monotonically decreasing in h, which makes bisection safe.
    """
    p = _isa_pressure(h)
    return p * M_AIR / (R * _isa_temperature(h)) - p * M_HE / (R * _isa_gas_temperature(h))


def _solve_ceiling(mass_over_vol: float, hi: float = 100_000.0) -> float:
    """Altitude (m) where rho_eff(h) == mass_over_vol, by bisection."""
    lo = 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _rho_effective(mid) > mass_over_vol:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# Altitude ceiling — where the balloon at VOL_MAX reaches neutral buoyancy.
# In the isothermal stratosphere a zero-pressure balloon with fixed moles has
# altitude-independent buoyant mass, so the ceiling is set entirely by the
# volume clamp at VOL_MAX.
ALT_MAX = _solve_ceiling(MASS / VOL_MAX)   # ≈ 30_105 m with VOL_MAX = 450 m³

# Default float altitude — midpoint of the *operational* band.  Previously this
# was 0.5*ALT_MAX (a leftover from when the operating range was [0, ALT_MAX]);
# with an explicit safety band the band midpoint is the meaningful default.
ALT_DEFAULT = 0.5 * (ALT_SAFE_MIN + ALT_SAFE_MAX)   # 20_000 m

# Sanity: the ceiling must sit above the band, otherwise the agent cannot reach
# the top of its own operating range.  VOL_MAX is the knob if this ever fails.
assert ALT_SAFE_MIN < ALT_DEFAULT < ALT_SAFE_MAX, "ALT_DEFAULT outside safety band"
assert ALT_MAX > ALT_SAFE_MAX, "VOL_MAX too small: ceiling below ALT_SAFE_MAX"

# ----- VENT CALIBRATION -----
# Float-altitude reference state.
_P_FLOAT = _isa_pressure(ALT_DEFAULT)
_T_GAS_FLOAT = _isa_gas_temperature(ALT_DEFAULT)

# Vent size, derived rather than fixed: one vent action must change the net
# vertical force by exactly as much as one ballast drop, at float altitude.
# Venting loses buoyancy rho_air*V but also sheds the helium's own weight, so
# the net is the *effective* buoyant density:
#     rho_eff(ALT_DEFAULT) * VENT_RATE == BALLAST_DROP.
# This is the ZP counterpart of the SP rule "AIR_PUMP_RATE == BALLAST_DROP for
# force symmetry", and it is *derived* so it stays correct if ALT_DEFAULT moves.
# (A hardcoded 0.05 m³ was calibrated against a ~12 km float altitude; at 20 km
# the envelope is 3.5x larger and the same volume is 3.5x less authority.)
VENT_RATE = BALLAST_DROP / _rho_effective(ALT_DEFAULT)  # Gas volume per action (m³)

# Molar vent rate: removes a fixed number of moles per vent action, matching the
# volume-based rate VENT_RATE at float altitude.  Descent authority is then
# constant across the operating range instead of degrading exponentially with
# altitude.  See notes/altitude_control_instability.md.
VENT_RATE_MOLES = _P_FLOAT * VENT_RATE / (R * _T_GAS_FLOAT)  # mol removed per vent action
VEL_MAX = 200.0             # Maximum velocity (m/s) - physics clamp limit

# Observation normaliser for vertical velocity, deliberately separate from the
# physics clamp above (roadmap §3.10).  VEL_MAX is a numerical runaway guard set
# ~40x above anything physical; using it to scale the observation made
# ``vel_z_norm`` a dead channel — measured across held-out episodes it spanned
# |v|/VEL_MAX <= 0.022 with a median of exactly 0.0, i.e. ~1% of its [-1, 1]
# range, so the network saw a near-constant input where a control-relevant one
# was intended.
#
# 5 m/s is set from measurement, not taste.  Loon Q2-2021 flight data
# (scripts/validate_physics_vs_loon.py) puts real stratospheric vertical rates
# at p50 0.16 m/s, p95 0.59 m/s, p99.9 6.2 m/s; our own simulator tops out
# around 4.5 m/s under both passive and greedy-wind control.  Clipping above
# 5 m/s therefore costs almost nothing and buys ~40x the resolution.
#
# This changes observation *scaling*, not the frozen layout (§2.1): the width is
# unchanged, so it costs a retrain of weights and not a re-architecture.
VEL_Z_OBS_NORM = 5.0        # Vertical-velocity observation normaliser (m/s)
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

# ----- SUPERPRESSURE + AIR BALLAST MODEL -----
SP_VOL_FIXED        = 100.0             # Fixed outer volume (m³)
SP_PAYLOAD_MASS     = 2.0               # Same as ZP for comparability (kg)
AIR_PUMP_RATE       = 0.01              # Air mass pumped per action (kg) — equals BALLAST_DROP for force symmetry
AIR_BLADDER_MAX     = 5.0              # Maximum air mass in bladder (kg)
AIR_BLADDER_INITIAL = AIR_BLADDER_MAX / 2  # Start at midpoint for equal up/down authority (2.5 kg)
# m_he_fixed is computed dynamically in BalloonSP.__init__ to ensure neutral buoyancy at ALT_DEFAULT
