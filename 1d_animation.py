import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# -------------------------------
# GLOBAL CONSTANTS
# -------------------------------
G = 9.81                 # gravitational acceleration (m/s^2)
R = 8.314462618          # universal gas constant (J/(mol*K))
T_BALLOON = 273.15 + 20  # internal gas temperature (K) (~20°C)
T_AIR = 273.15 + 15      # ambient air temperature (K)
M_AIR = 0.0289647        # molar mass of air (kg/mol)
P0 = 101325              # sea-level standard atmospheric pressure (Pa)
SCALE_HEIGHT = 8500      # scale height (m)

DT = 1.0         # Time step (s)
T_MAX = 400.0     # Simulation duration (s)

# Volume fluctuation parameters
FLUCTUATION_FRACTION = 0.05   # ±5% of the stationary volume
FLUCTUATION_PERIOD = 60.0     # seconds for one full sine-wave cycle

class Atmosphere:
    """
    Simple model of the atmosphere to retrieve pressure and density
    as functions of altitude.
    """
    def __init__(self, p0=P0, scale_height=SCALE_HEIGHT, 
                 temperature=T_AIR, molar_mass=M_AIR):
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

class Balloon:
    """
    A weather balloon that would be stationary at the initial altitude
    if its volume were constant, but which now oscillates around that
    'stationary' volume, causing gentle up-and-down motion.
    """
    def __init__(self, 
                 mass_balloon=2.0,   
                 altitude=25000.0,    # start altitude in meters
                 velocity=0.0,
                 atmosphere=Atmosphere()):
        
        self.mass_balloon = mass_balloon
        self.altitude = altitude
        self.velocity = velocity
        self.atmosphere = atmosphere
        
        # Compute the "stationary volume" that exactly balances
        # the balloon's weight at this initial altitude:
        rho_air = self.atmosphere.density(self.altitude)
        self.stationary_volume = self.mass_balloon / rho_air

    def dynamic_volume(self, t):
        """
        Slightly oscillate around the stationary volume:
          V(t) = V_stationary + amplitude * sin(...)
        with amplitude = (FLUCTUATION_FRACTION * V_stationary).
        """
        amplitude = FLUCTUATION_FRACTION * self.stationary_volume
        phase = 2.0 * np.pi * (t / FLUCTUATION_PERIOD)
        
        return self.stationary_volume + amplitude * np.sin(phase)
    
    def buoyant_force(self, t):
        """
        Buoyant force = (density at altitude) * g * [dynamic volume].
        """
        rho_air = self.atmosphere.density(self.altitude)
        return rho_air * G * self.dynamic_volume(t)
    
    def weight(self):
        """
        Weight = mass * g.
        """
        return self.mass_balloon * G
    
    def net_force(self, t):
        """
        Net force = buoyant force - weight.
        """
        return self.buoyant_force(t) - self.weight()
    
    def update(self, t, dt):
        """
        Update balloon's velocity and altitude (simple Euler method).
        """
        a = self.net_force(t) / self.mass_balloon
        self.velocity += a * dt
        self.altitude += self.velocity * dt
        
        # Prevent negative altitude:
        if self.altitude < 0:
            self.altitude = 0
            self.velocity = 0

# ----------------------------
# RUN THE SIMULATION
# ----------------------------
balloon = Balloon(
    mass_balloon=2.0,
    altitude=25000.0,
    velocity=0.0
)

times = []
altitudes = []
volumes = []

time = 0.0
while time <= T_MAX:
    times.append(time)
    altitudes.append(balloon.altitude)
    volumes.append(balloon.dynamic_volume(time))
    
    balloon.update(time, DT)
    time += DT

# ----------------------------
# ANIMATION
# ----------------------------
fig, ax = plt.subplots()
ax.set_title("Slight Balloon Volume Fluctuation")
ax.set_xlabel("X position (m)")
ax.set_ylabel("Altitude (m)")

# We'll allow y to range from 0 to 40,000 to visualize the motion
ax.set_xlim(-20, 20)
ax.set_ylim(20000, 30000)

# Create the circle object; we'll update its center & radius in the animation.
initial_center = (0, altitudes[0])
# We'll scale radius with the cubic root of the volume just for display
initial_radius = (volumes[0]) ** (1.0/3.0)
circle = plt.Circle(initial_center, initial_radius, color='red')
ax.add_patch(circle)

def init():
    circle.center = (0, altitudes[0])
    circle.radius = (volumes[0]) ** (1.0/3.0)
    return (circle,)

def animate(i):
    y = altitudes[i]
    circle.center = (0, y)
    
    # Volume-based radius:  scale with cubic root (arbitrary for appearance)
    r = (volumes[i]) ** (1.0/3.0)
    circle.radius = r
    return (circle,)

ani = animation.FuncAnimation(
    fig,
    animate,
    frames=len(times),
    init_func=init,
    interval=50,
    blit=True,
    repeat=False
)

plt.show()
