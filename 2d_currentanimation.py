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
T_MAX = 4000.0           # Simulation duration (s)

# Drag parameters
CD = 0.5                 # Drag coefficient (dimensionless)
AREA = 1.0               # Cross-sectional area of the balloon (m^2)

# Force grid parameters
# We want to divide the domain into cells (boxes) and assign one force per cell.
GRID_CELLS = 20          # number of cells in each direction
X_RANGE = (-2000, 2000)  # Range of x positions (m)
Y_RANGE = (20000, 30000) # Range of altitudes (m)
FORCE_MAG = 10.0         # Magnitude scale for forces (N)

# Define grid edges (cell boundaries) for x and y.
x_edges = np.linspace(X_RANGE[0], X_RANGE[1], GRID_CELLS + 1)
y_edges = np.linspace(Y_RANGE[0], Y_RANGE[1], GRID_CELLS + 1)

# Compute the centers of the cells to place the wind arrows.
x_centers = (x_edges[:-1] + x_edges[1:]) / 2
y_centers = (y_edges[:-1] + y_edges[1:]) / 2
x_forces, y_forces = np.meshgrid(x_centers, y_centers)

# --- Create a smooth (less noisy) wind field using sine and cosine ---
fx_grid = FORCE_MAG * np.sin(2 * np.pi * x_forces / (X_RANGE[1]-X_RANGE[0]))
fy_grid = FORCE_MAG * np.cos(2 * np.pi * y_forces / (Y_RANGE[1]-Y_RANGE[0]))

# -------------------------------
# ATMOSPHERE & BALLOON CLASSES
# -------------------------------
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
        """Exponential atmosphere model for pressure."""
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
      - slightly oscillating internal volume (thus changing buoyant force),
      - weight, drag, and an external force determined solely by its cell (box) in a grid.
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
        phase = 2.0 * np.pi * (t / 60.0)  # one oscillation in 60 s
        return self.stationary_volume + amplitude * np.sin(phase)
    
    def buoyant_force(self, t):
        """Buoyant force = (0, rho_air * g * volume)."""
        rho_air = self.atmosphere.density(self.y)
        vol = self.dynamic_volume(t)
        F_mag = rho_air * G * vol
        return (0.0, F_mag)
    
    def weight(self):
        """Weight = (0, -m * g)."""
        return (0.0, -self.mass_balloon * G)
    
    def external_force(self):
        """
        Determine the cell (box) that the balloon is in using x_edges and y_edges,
        and return the corresponding wind force (Fx, Fy) from the grid.
        """
        x_idx = np.searchsorted(x_edges, self.x) - 1
        y_idx = np.searchsorted(y_edges, self.y) - 1
        
        # Ensure indices are valid
        x_idx = np.clip(x_idx, 0, GRID_CELLS - 1)
        y_idx = np.clip(y_idx, 0, GRID_CELLS - 1)
        
        Fx = fx_grid[y_idx, x_idx]
        Fy = fy_grid[y_idx, x_idx]
        return (Fx, Fy)
    
    def drag_force(self):
        """
        Drag = -1/2 * Cd * A * rho * v^2 * (v_hat).
        """
        speed = np.hypot(self.vx, self.vy)
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
        
        # Update velocity and position.
        self.vx += ax * dt
        self.vy += ay * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        # Prevent negative altitude.
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
# ANIMATION SETUP
# ----------------------------
fig, ax = plt.subplots()
ax.set_title("2D Balloon Motion with Smooth Wind Fields")
ax.set_xlabel("X position (m)")
ax.set_ylabel("Altitude (m)")
ax.set_xlim(X_RANGE)
ax.set_ylim(Y_RANGE)

# Plot quivers for the wind field (blue arrows) at cell centers.
quiver = ax.quiver(x_forces, y_forces, fx_grid, fy_grid, color='blue', alpha=0.5)

# Create rectangle patches for each cell.
rectangles = []
for j in range(GRID_CELLS):
    row_rects = []
    for i in range(GRID_CELLS):
        # Each cell extends from x_edges[i] to x_edges[i+1] and similarly for y.
        rect_width = x_edges[i+1] - x_edges[i]
        rect_height = y_edges[j+1] - y_edges[j]
        # Place the rectangle so that its center is at the midpoint of the cell.
        rect = plt.Rectangle(
            (x_edges[i], y_edges[j]),
            rect_width,
            rect_height,
            fill=False,
            edgecolor='gray',
            linewidth=1.0,
            alpha=0.3
        )
        ax.add_patch(rect)
        row_rects.append(rect)
    rectangles.append(row_rects)

# Create the circle object for the balloon.
initial_center = (xs[0], ys[0])
circle = plt.Circle(initial_center, 50, color='red')  # fixed radius for visibility
ax.add_patch(circle)

# Create a text element for the balloon volume in the top right.
volume_text = ax.text(0.95, 0.95, '', transform=ax.transAxes,
                      ha='right', va='top', fontsize=12, color='black')

def init():
    circle.center = (xs[0], ys[0])
    volume_text.set_text(f"Volume: {balloon.dynamic_volume(times[0]):.2f} m³")
    return (circle, volume_text)

def animate(frame):
    # Update balloon position.
    cx = xs[frame]
    cy = ys[frame]
    circle.center = (cx, cy)
    
    # Update volume text using the dynamic volume at the current time.
    current_volume = balloon.dynamic_volume(times[frame])
    volume_text.set_text(f"Volume: {current_volume:.2f} m³")
    
    # Determine which cell the balloon is in.
    x_idx = np.searchsorted(x_edges, cx) - 1
    y_idx = np.searchsorted(y_edges, cy) - 1
    x_idx = np.clip(x_idx, 0, GRID_CELLS - 1)
    y_idx = np.clip(y_idx, 0, GRID_CELLS - 1)
    
    # Reset all rectangles to default.
    for row in rectangles:
        for rect in row:
            rect.set_edgecolor('gray')
            rect.set_linewidth(1.0)
    
    # Highlight the active cell.
    rectangles[y_idx][x_idx].set_edgecolor('red')
    rectangles[y_idx][x_idx].set_linewidth(2.0)
    
    return (circle, volume_text)

ani = animation.FuncAnimation(
    fig,
    animate,
    frames=len(times),
    init_func=init,
    interval=100,
    blit=False,  # disable blitting since several artists update
    repeat=False
)

plt.show()
