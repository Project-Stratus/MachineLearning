import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

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



Cd = 0.285

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
        p = self.pressure(altitude[1])
        rho = p * self.molar_mass / (R * self.temperature)
        return rho
    
    def get_wind_speed(self, position):
        return -1 if position[1] > 25000 else 1 if position[1] > 10000 else -1

class Balloon:
    """
    A weather balloon that would be stationary at the initial altitude
    if its volume were constant, but which now oscillates around that
    'stationary' volume, causing gentle up-and-down motion.
    """
    def __init__(self, 
                 mass_balloon=2.0,   # kg, mass of balloon + payload
                 altitude=(0, 25000),   # m (start around 25km)
                 velocity=(0, 0),       # m/s
                 atmosphere=Atmosphere()):
        
        self.mass_balloon = mass_balloon
        self.altitude = altitude
        self.velocity = velocity
        self.atmosphere = atmosphere
        
        # Compute the "stationary volume" that exactly balances
        # the balloon's weight at this initial altitude:
        rho_air = self.atmosphere.density(self.altitude[1])
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
        (drag_x, drag_y) = self.drag_force(t)
        return (drag_x, drag_y + self.buoyant_force(t) - self.weight())

    def cross_area(self, t): # assuming area sphere
        radius = ((self.dynamic_volume(t) / np.pi) * 3 / 4) ** (1/3)
        return (radius * radius * np.pi, radius * radius * np.pi)

    def drag_force(self, t):
        wind = self.atmosphere.get_wind_speed(self.altitude)
        Densityat = self.atmosphere.density(self.altitude)

        (a1, a2) = self.cross_area(t)
        def calc(a, v):
            if v > 0:
                return -Cd * Densityat * v * v * a / 2
            else:
                return Cd * Densityat * v * v * a / 2
        return (calc(a1, self.velocity[0] - wind), calc(a2, self.velocity[1]))
    
    def update(self, t, dt):
        """
        Update balloon's velocity and altitude (simple Euler method).
        """
        a = div(self.net_force(t), self.mass_balloon)
        self.velocity = add(self.velocity, mul(a, dt))
        self.altitude = add(self.altitude, mul(self.velocity, dt))
        
        # Keep altitude non-negative
        if self.altitude[1] < 0:
            self.altitude = (0, 0)
            self.velocity = (0, 0)

def add(z1, z2):
    (z11, z12) = z1 
    (z21, z22) = z2 
    return (z11 + z21, z12 + z22)

def div(z1, z2):
    (z11, z12) = z1 
    return (z11 / z2, z12 / z2)

def mul(z1, z2):
    (z11, z12) = z1 
    return (z11 * z2, z12 * z2)

# ----------------------------
# RUN THE SIMULATION
# ----------------------------
balloon = Balloon(
    mass_balloon=2.0,   # kg
    altitude=(0, 25000),   # start altitude ~25km
    velocity=(0, 0)
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

# Create the circle object. We'll dynamically change its center and radius.
initial_center = altitudes[0]
initial_radius = (volumes[0]) ** (1.0/3.0)  # just a placeholder, will update in animate()
circle = plt.Circle(initial_center, initial_radius, color='red')
ax.add_patch(circle)

x = np.arange(-100, 100, 3)
y = np.arange(0, 40000, 1000)
x, y = np.meshgrid(x, y)

u = np.zeros_like(x, dtype=float)
v = np.zeros_like(y, dtype=float)

for i in range(x.shape[0]):
    for j in range(y.shape[1]):
        print((x[i, j], y[i, j]))
        speed = balloon.atmosphere.get_wind_speed((x[i, j], y[i, j]))
        u[i, j] = speed
        print(speed)

# Create the quiver plot
plt.quiver(x, y, u, v, scale=1, scale_units='xy', angles='xy')

# Keep aspect ratio 'equal' or 'auto' so the circle doesn't look squashed.
ax.set_aspect('auto')  # or 'equal', but that can make the axis extremely tall

def init():
    circle.center = altitudes[0]
    circle.radius = (volumes[0]) ** (1.0/3.0)
    return circle,

def animate(i):
    y = altitudes[i]
    print(y)
    circle.center = y
    
    # Volume-based radius:  scale with cubic root (arbitrary for appearance)
    r = (volumes[i]) ** (1.0/3.0)
    circle.radius = r
    return (circle,)

ani = animation.FuncAnimation(
    fig,
    animate, 
    frames=len(times), 
    init_func=init, 
    interval=1, 
    blit=True, 
    repeat=False
)

plt.show()
