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
T_MAX = 4000.0             # Simulation duration (s)

# Drag parameters
CD = 0.5                 # Drag coefficient (dimensionless)
AREA = 1.0               # Cross-sectional area of the balloon (m^2)

# Force grid parameters
GRID_SIZE = 20           # 20x20 grid
X_RANGE = (-2000, 2000)  # Range of x positions (m)
Y_RANGE = (20000, 30000) # Range of altitudes (m)
FORCE_MAG = 10.0        # Magnitude scale for forces (N)

# Generate coordinate axes for the force grid
x_grid = np.linspace(X_RANGE[0], X_RANGE[1], GRID_SIZE)
y_grid = np.linspace(Y_RANGE[0], Y_RANGE[1], GRID_SIZE)
x_forces, y_forces = np.meshgrid(x_grid, y_grid)

# Randomize forces for each cell in the grid
# fx_grid[j, i] and fy_grid[j, i] = force in cell (i, j)
fx_grid = np.random.uniform(FORCE_MAG, FORCE_MAG, size=(GRID_SIZE, GRID_SIZE))
fy_grid = np.random.uniform(FORCE_MAG, FORCE_MAG, size=(GRID_SIZE, GRID_SIZE))




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
        """ Exponential atmosphere model for pressure """
        return self.p0 * np.exp(-altitude / self.scale_height)
    
    def density(self, altitude):
        """
        Ideal gas law for density:
        rho = (P * M_air) / (R * T).
        """
        p = self.pressure(altitude)
        rho = p * self.molar_mass / (R * self.temperature)
        return rho


class Balloon:
    """
    A balloon with:
      - slightly oscillating internal volume (thus changing buoyant force).
      - weight, drag, and an external force determined by its cell in a 20x20 grid.
    """
    def __init__(self, mass_balloon=2.0, x=0.0, y=25000.0, vx=0.0, vy=0.0,
                 atmosphere=Atmosphere()):
        
        self.mass_balloon = mass_balloon
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.atmosphere = atmosphere

        # Compute the "stationary" volume from initial altitude 
        rho_air = self.atmosphere.density(self.y)
        self.stationary_volume = self.mass_balloon / rho_air

    def dynamic_volume(self, t):
        """
        Oscillate around the stationary volume by +/- 5%.
        """
        amplitude_fraction = 0.05
        amplitude = amplitude_fraction * self.stationary_volume
        phase = 2.0 * np.pi * (t / 60.0)  # one oscillation in 60s
        return self.stationary_volume + amplitude * np.sin(phase)
    
    def buoyant_force(self, t):
        """ Buoyant force = (0, rho_air * g * volume) """
        rho_air = self.atmosphere.density(self.y)
        vol = self.dynamic_volume(t)
        F_mag = rho_air * G * vol
        return (0.0, F_mag)
    
    def weight(self):
        """ Weight = (0, -m * g) """
        return (0.0, -self.mass_balloon * G)
    
    def external_force(self):
        """
        Find which grid cell the balloon is in and return 
        the corresponding force (Fx, Fy).
        """
        # Determine grid indices 
        # (searchsorted -> which bin the balloon falls into)
        x_idx = np.searchsorted(x_grid, self.x) - 1
        y_idx = np.searchsorted(y_grid, self.y) - 1
        
        # Clip to valid range
        x_idx = np.clip(x_idx, 0, GRID_SIZE - 1)
        y_idx = np.clip(y_idx, 0, GRID_SIZE - 1)
        
        # Force from the grid at (y_idx, x_idx)
        Fx = fx_grid[y_idx, x_idx]
        Fy = fy_grid[y_idx, x_idx]
        return (Fx, Fy)
    
    def drag_force(self):
        """
        Drag = -1/2 * Cd * A * rho * v^2 * (v_hat).
        """
        speed = np.sqrt(self.vx**2 + self.vy**2)
        if speed < 1e-8:
            return (0.0, 0.0)
        rho_air = self.atmosphere.density(self.y)
        F_drag_mag = 0.5 * CD * AREA * rho_air * speed**2
        drag_x = -F_drag_mag * (self.vx / speed)
        drag_y = -F_drag_mag * (self.vy / speed)
        return (drag_x, drag_y)
    
    def net_force(self, t):
        """
        Sum of buoyant, weight, drag, and external forces.
        """
        Fx_buoy, Fy_buoy = self.buoyant_force(t)
        Fx_wt, Fy_wt = self.weight()
        Fx_drag, Fy_drag = self.drag_force()
        Fx_ext, Fy_ext = self.external_force()
        
        Fx_net = Fx_buoy + Fx_wt + Fx_drag + Fx_ext
        Fy_net = Fy_buoy + Fy_wt + Fy_drag + Fy_ext
        return (Fx_net, Fy_net)
    
    def update(self, t, dt):
        """
        Advance the balloon state using a simple Euler integrator.
        """
        Fx_net, Fy_net = self.net_force(t)
        ax = Fx_net / self.mass_balloon
        ay = Fy_net / self.mass_balloon
        
        # Update velocity
        self.vx += ax * dt
        self.vy += ay * dt
        
        # Update position
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        # Prevent negative altitude
        if self.y < 0:
            self.y = 0
            self.vy = 0


# ----------------------------
# RUN THE SIMULATION
# ----------------------------
balloon = Balloon()

times = []
xs = []
ys = []

time = 0.0
while time <= T_MAX:
    times.append(time)
    xs.append(balloon.x)
    ys.append(balloon.y)
    
    balloon.update(time, DT)
    time += DT

# ----------------------------
# ANIMATION
# ----------------------------
fig, ax = plt.subplots()
ax.set_title("2D Balloon Motion with Grid-Based External Forces")
ax.set_xlabel("X position (m)")
ax.set_ylabel("Altitude (m)")
ax.set_xlim(X_RANGE)
ax.set_ylim(Y_RANGE)

# Plot quivers for the force grid (blue arrows)
quiver = ax.quiver(x_forces, y_forces, fx_grid, fy_grid, color='blue', alpha=0.5)

# Create rectangle patches for each cell in the 20×20 grid 
# We'll highlight the active cell in red in the animation
rectangles = []
for j in range(GRID_SIZE - 1):
    row_rects = []
    for i in range(GRID_SIZE - 1):
        # The rectangle spans from (x_grid[i], y_grid[j]) 
        # to (x_grid[i+1], y_grid[j+1])
        rect_width = x_grid[i+1] - x_grid[i]
        rect_height = y_grid[j+1] - y_grid[j]
        
        rect = plt.Rectangle(
            (x_grid[i]-rect_width/2, y_grid[j]-rect_height/2),
            rect_width,
            rect_height,
            fill=False,            # no fill by default
            edgecolor='gray',
            linewidth=1.0,
            alpha=0.3
        )
        ax.add_patch(rect)
        row_rects.append(rect)
    rectangles.append(row_rects)

# Create the circle object for the balloon
initial_center = (xs[0], ys[0])
circle = plt.Circle(initial_center, 50, color='red')  # fixed radius for visibility
ax.add_patch(circle)

def init():
    circle.center = (xs[0], ys[0])
    return (circle,)

def animate(frame):
    # Update balloon position
    cx = xs[frame]
    cy = ys[frame]
    circle.center = (cx, cy)
    
    # Determine which cell we're in, and highlight it
    x_idx = np.searchsorted(x_grid, cx) - 1
    y_idx = np.searchsorted(y_grid, cy) - 1
    x_idx = np.clip(x_idx, 0, GRID_SIZE - 2)  # up to GRID_SIZE-2 since we have GRID_SIZE-1 rectangles
    y_idx = np.clip(y_idx, 0, GRID_SIZE - 2)
    
    # Reset all rectangles to default
    for rj in range(GRID_SIZE - 1):
        for ri in range(GRID_SIZE - 1):
            rectangles[rj][ri].set_edgecolor('gray')
            rectangles[rj][ri].set_linewidth(1.0)
    
    # Highlight the active cell
    active_rect = rectangles[y_idx][x_idx]
    active_rect.set_edgecolor('red')
    active_rect.set_linewidth(2.0)
    
    return (circle,)

ani = animation.FuncAnimation(
    fig,
    animate,
    frames=len(times),
    init_func=init,
    interval=100,
    blit=True,
    repeat=False
)

plt.show()
