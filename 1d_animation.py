import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math

# --- CONSTANTS ---
G = 9.81                 # gravitational acceleration (m/s^2)
R = 8.314462618          # universal gas constant (J/(mol*K))
T_BALLOON = 273.15 + 20  # internal gas temperature (K) (~20°C)
T_AIR = 273.15 + 15      # ambient air temperature (K) (simplified)
M_AIR = 0.0289647        # molar mass of air (kg/mol)
P0 = 101325              # sea-level standard atmospheric pressure (Pa)
SCALE_HEIGHT = 8500      # scale height (m)
N_MOLES = 10.0           # number of moles of gas (arbitrary choice)



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
    Weather balloon with methods to update position and a time-varying volume 
    (sinusoidal around the 'ideal' volume) that causes up-and-down motion.
    """
    def __init__(self, 
                 mass_balloon=2.0,   # kg, mass of balloon + payload
                 altitude=(0, 25000),   # m (start around 25km)
                 velocity=(0, 0),       # m/s
                 n_moles=N_MOLES,    # moles of lifting gas
                 atmosphere=Atmosphere()):
        
        self.mass_balloon = mass_balloon
        self.altitude = altitude
        self.velocity = velocity
        self.n_moles = n_moles
        
        # Atmosphere object for external conditions
        self.atmosphere = atmosphere
        
    def ideal_volume(self):
        """
        Ideal volume from the Ideal Gas Law: V = (n * R * T_in) / P_ext.
        """
        p_ext = self.atmosphere.pressure(self.altitude[1])
        return (self.n_moles * R * T_BALLOON) / p_ext
    
    def dynamic_volume(self, t):
        """
        Volume fluctuates around the ideal volume with a large sinusoidal swing
        so we can see a big difference in buoyancy and altitude.
        """
        v_ideal = self.ideal_volume()
        # 50% amplitude for a more dramatic effect:
        amplitude = 0.5 * v_ideal
        # Shorter period so it oscillates more frequently:
        period = 60.0  # seconds
        phase = 2.0 * np.pi * t / period
        
        return v_ideal + amplitude * np.sin(phase)
    
    def buoyant_force(self, t):
        """
        Buoyant force = rho_air(altitude) * g * volume(t).
        """
        rho_air = self.atmosphere.density(self.altitude)
        vol = self.dynamic_volume(t)
        return rho_air * G * vol
    
    def weight(self):
        """
        Weight = total mass * g.
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
        Update balloon velocity and altitude with simple Euler integration.
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



# --- SIMULATION PARAMETERS ---
dt = 1.0         # time step (s)
t_max = 400.0    # total simulation time (s)

balloon = Balloon(
    mass_balloon=20.0,   # kg
    altitude=(0, 25000),   # start altitude ~25km
    velocity=(0, 0),
    n_moles=N_MOLES
)

times = []
altitudes = []
volumes = []

time = 0.0
while time <= t_max:
    times.append(time)
    altitudes.append(balloon.altitude)
    volumes.append(balloon.dynamic_volume(time))
    
    balloon.update(time, dt)
    time += dt

# --- ANIMATION ---
fig, ax = plt.subplots()
ax.set_title("Weather Balloon (side view)")
ax.set_xlabel("X position (m)")
ax.set_ylabel("Altitude (m)")

# We give plenty of space in the x-axis so the balloon's radius won't be cut off.
ax.set_xlim(-100, 100)
ax.set_ylim(0, 40000)

# Create the circle object. We'll dynamically change its center and radius.
initial_center = altitudes[0]
initial_radius = 1.0  # just a placeholder, will update in animate()
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
    return circle,

def animate(i):
    # Update the balloon center to (0, altitude)
    y = altitudes[i]
    print(y)
    circle.center = y
    
    # Update radius based on the volume (scale with cubic root).
    # volumes[i] is in m^3, so radius ~ (volume)^(1/3) to visualize a sphere's characteristic size.
    # Adjust "radius_factor" to make the balloon bigger or smaller in the plot.
    radius_factor = 1.0
    new_radius = radius_factor * (volumes[i])**(1/3)
    print(new_radius)
    circle.radius = new_radius
    
    return circle,

ani = animation.FuncAnimation(
    fig, animate, 
    frames=len(times), 
    init_func=init, 
    interval=1, 
    blit=True, 
    repeat=False
)

plt.show()
