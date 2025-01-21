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

DT = 1.0                 # Time step (s)
T_MAX = 60.0             # Simulation duration (s) - shortened for demonstration

# Volume fluctuation parameters
FLUCTUATION_FRACTION = 0.5   # ±5% of the stationary volume
FLUCTUATION_PERIOD = 60.0     # seconds for one full sine-wave cycle

# Drag parameters
CD = 0.5        # Drag coefficient (dimensionless)
AREA = 1.0      # Cross-sectional area of the balloon (m^2), purely for demonstration

# External force parameters (e.g., a "wind gust" or horizontal oscillation)
EXT_FORCE_AMPLITUDE = 100.0   # Newtons
EXT_FORCE_PERIOD = 30.0       # seconds for a full sine-wave cycle


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
    A weather balloon that would be nearly stationary at the initial altitude
    if its volume were constant, but which now oscillates around that
    'stationary' volume. We also add:
      - Drag force that opposes motion.
      - External (time-varying) horizontal force.
      - 2D motion (x, y).
    """
    def __init__(self, 
                 mass_balloon=2.0,
                 x=0.0,               # start x-position
                 y=25000.0,          # start y-position (altitude)
                 vx=0.0,             # start x-velocity
                 vy=0.0,             # start y-velocity
                 atmosphere=Atmosphere()):
        
        self.mass_balloon = mass_balloon
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.atmosphere = atmosphere
        
        # Compute the "stationary volume" that balances weight at the initial altitude:
        rho_air = self.atmosphere.density(self.y)
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
        Buoyant force (in Newtons) is upward along +y.
        magnitude = rho_air * g * volume.
        We'll return a 2D vector = (0, +F_buoy).
        """
        rho_air = self.atmosphere.density(self.y)
        vol = self.dynamic_volume(t)
        F_mag = rho_air * G * vol
        return (0.0, F_mag)
    
    def weight(self):
        """
        Weight is downward along -y.
        We'll return a 2D vector = (0, -mg).
        """
        return (0.0, -self.mass_balloon * G)
    
    def external_force(self, t):
        """
        Time-varying external force, e.g., a sinusoidal horizontal "gust."
        We'll apply it in the +x direction with a sinusoidal pattern.

        Fx(t) = EXT_FORCE_AMPLITUDE * sin(2π * t/EXT_FORCE_PERIOD)
        Fy(t) = 0
        """
        phase = 2.0 * np.pi * (t / EXT_FORCE_PERIOD)
        Fx = EXT_FORCE_AMPLITUDE * np.sin(phase)
        Fy = 0.0
        return (Fx, Fy)
    
    def drag_force(self):
        """
        Drag force in 2D:
          F_drag = -1/2 * Cd * A * rho * v^2 * (v̂)
        where (v̂) is the velocity unit vector.
        
        We use the local air density at balloon's altitude.
        """
        # If speed is zero, drag is zero
        speed = np.sqrt(self.vx**2 + self.vy**2)
        if speed < 1e-8:
            return (0.0, 0.0)
        
        # Local air density
        rho_air = self.atmosphere.density(self.y)

        # Magnitude of the drag
        F_drag_mag = 0.5 * CD * AREA * rho_air * (speed**2)
        
        # Direction opposite velocity
        drag_x = -F_drag_mag * (self.vx / speed)
        drag_y = -F_drag_mag * (self.vy / speed)
        return (drag_x, drag_y)
    
    def net_force(self, t):
        """
        Sum of all forces in 2D:
          buoyant (up), weight (down), drag (opposite velocity), external (time-varying).
        Returns a tuple (Fx, Fy).
        """
        Fx_buoy, Fy_buoy = self.buoyant_force(t)
        Fx_wt, Fy_wt = self.weight()
        Fx_drag, Fy_drag = self.drag_force()
        Fx_ext, Fy_ext = self.external_force(t)
        
        Fx_net = Fx_buoy + Fx_wt + Fx_drag + Fx_ext
        Fy_net = Fy_buoy + Fy_wt + Fy_drag + Fy_ext
        return (Fx_net, Fy_net)
    
    def update(self, t, dt):
        """
        Update balloon's velocity and position using a simple Euler step in 2D.
        We'll also print the forces each time step.
        """
        # Get each force individually, just for printing
        (Fx_buoy, Fy_buoy) = self.buoyant_force(t)
        (Fx_wt, Fy_wt) = self.weight()
        (Fx_drag, Fy_drag) = self.drag_force()
        (Fx_ext, Fy_ext) = self.external_force(t)
        
        # Summation for net
        Fx_net = Fx_buoy + Fx_wt + Fx_drag + Fx_ext
        Fy_net = Fy_buoy + Fy_wt + Fy_drag + Fy_ext
        
        # Print all force values
        print(f"t={t:4.1f}s | "
              f"Buoy=({Fx_buoy:.2f},{Fy_buoy:.2f}) N, "
              f"Weight=({Fx_wt:.2f},{Fy_wt:.2f}) N, "
              f"Drag=({Fx_drag:.2f},{Fy_drag:.2f}) N, "
              f"External=({Fx_ext:.2f},{Fy_ext:.2f}) N, "
              f"Net=({Fx_net:.2f},{Fy_net:.2f}) N")
        
        # Acceleration
        ax = Fx_net / self.mass_balloon
        ay = Fy_net / self.mass_balloon
        
        # Update velocities
        self.vx += ax * dt
        self.vy += ay * dt
        
        # Update positions
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        # Prevent negative altitude:
        if self.y < 0:
            self.y = 0
            self.vy = 0


# ----------------------------
# RUN THE SIMULATION
# ----------------------------
balloon = Balloon(
    mass_balloon=2.0,
    x=0.0,
    y=25000.0,
    vx=0.0,
    vy=0.0
)

times = []
xs = []
ys = []
volumes = []

time = 0.0
while time <= T_MAX:
    times.append(time)
    xs.append(balloon.x)
    ys.append(balloon.y)
    volumes.append(balloon.dynamic_volume(time))
    
    balloon.update(time, DT)
    time += DT

# ----------------------------
# ANIMATION
# ----------------------------
fig, ax = plt.subplots()
ax.set_title("2D Balloon Motion with Drag & External Force")
ax.set_xlabel("X position (m)")
ax.set_ylabel("Altitude (m)")

# Let's focus on the region around x in [-2000,2000], y in [20000,30000]
ax.set_xlim(-2000, 2000)
ax.set_ylim(20000, 30000)

# Create the circle object; we'll update its center & radius in the animation.
# For the radius, we scale with the cubic root of volume (arbitrary for aesthetics).
initial_center = (xs[0], ys[0])
initial_radius = (volumes[0]) ** (1.0/3.0)
circle = plt.Circle(initial_center, initial_radius, color='red')
ax.add_patch(circle)

def init():
    circle.center = (xs[0], ys[0])
    circle.radius = (volumes[0]) ** (1.0/3.0)
    return (circle,)

def animate(i):
    cx = xs[i]
    cy = ys[i]
    circle.center = (cx, cy)
    
    # Recompute radius based on volume
    r = (volumes[i]) ** (1.0/3.0)
    circle.radius = r
    return (circle,)

ani = animation.FuncAnimation(
    fig,
    animate,
    frames=len(times),
    init_func=init,
    interval=300,  # slowed down so you can see it move step by step
    blit=True,
    repeat=False
)

plt.show()
