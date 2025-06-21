"""
Version by TS
2D loon implementation only
"""


import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from environments.envs.atmosphere import Atmosphere

# -------------------------------
# GLOBAL CONSTANTS
# -------------------------------
G = 9.81                 # gravitational acceleration (m/s²)
R = 8.314462618          # universal gas constant (J/(mol·K))
T_BALLOON = 273.15 + 20  # internal gas temperature (K)
T_AIR = 273.15 + 15      # ambient air temperature (K)
M_AIR = 0.0289647        # molar mass of air (kg/mol)
P0 = 101325              # sea-level atmospheric pressure (Pa)
SCALE_HEIGHT = 8500      # scale height (m)

DT = 1.0                 # simulation time step (s)
EPISODE_LENGTH = 300     # max steps per episode

# Drag parameters
CD = 0.5                 
AREA = 1.0               

# Force grid parameters (for the wind field)
GRID_CELLS = 40          
X_RANGE = (-2000, 2000)  
Y_RANGE = (20000, 30000) 
FORCE_MAG = 10.0         

# Define grid edges and cell centers for the wind field.
x_edges = np.linspace(X_RANGE[0], X_RANGE[1], GRID_CELLS + 1)
y_edges = np.linspace(Y_RANGE[0], Y_RANGE[1], GRID_CELLS + 1)
x_centers = (x_edges[:-1] + x_edges[1:]) / 2
y_centers = (y_edges[:-1] + y_edges[1:]) / 2
x_forces, y_forces = np.meshgrid(x_centers, y_centers)

# Create a complex yet gentle wind field.
fx_grid = (FORCE_MAG / 2) * (
    np.sin(2 * np.pi * x_forces / (X_RANGE[1] - X_RANGE[0])) +
    0.5 * np.sin(4 * np.pi * x_forces / (X_RANGE[1] - X_RANGE[0]))
)
fy_grid = (FORCE_MAG / 2) * (
    np.cos(2 * np.pi * y_forces / (Y_RANGE[1] - Y_RANGE[0])) +
    0.5 * np.cos(4 * np.pi * y_forces / (Y_RANGE[1] - Y_RANGE[0]))
)

# -------------------------------
# SIMULATION CLASSES
# # -------------------------------
# class Atmosphere:
#     def __init__(self, p0=P0, scale_height=SCALE_HEIGHT, temperature=T_AIR, molar_mass=M_AIR):
#         self.p0 = p0
#         self.scale_height = scale_height
#         self.temperature = temperature
#         self.molar_mass = molar_mass
        
#     def pressure(self, altitude):
#         return self.p0 * np.exp(-altitude / self.scale_height)
    
#     def density(self, altitude):
#         p = self.pressure(altitude)
#         rho = p * self.molar_mass / (R * self.temperature)
#         return rho


class Balloon:
    def __init__(self, mass_balloon=2.0, x=0.0, y=25000.0, vx=0.0, vy=0.0, atmosphere=None):
        self.mass_balloon = mass_balloon
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.atmosphere = atmosphere if atmosphere is not None else Atmosphere()
        # Stationary volume based on initial air density.
        rho_air = self.atmosphere.density(self.y)
        self.stationary_volume = self.mass_balloon / rho_air

    def dynamic_volume(self, t):
        amplitude_fraction = 0.05
        amplitude = amplitude_fraction * self.stationary_volume
        phase = 2.0 * np.pi * (t / 60.0)  # one oscillation every 60 seconds
        return self.stationary_volume + amplitude * np.sin(phase)
    
    def buoyant_force(self, t):
        rho_air = self.atmosphere.density(self.y)
        vol = self.dynamic_volume(t)
        F_mag = rho_air * G * vol
        return (0.0, F_mag)
    
    def weight(self):
        return (0.0, -self.mass_balloon * G)
    
    def external_force(self):
        x_idx = np.searchsorted(x_edges, self.x) - 1
        y_idx = np.searchsorted(y_edges, self.y) - 1
        x_idx = np.clip(x_idx, 0, GRID_CELLS - 1)
        y_idx = np.clip(y_idx, 0, GRID_CELLS - 1)
        Fx = fx_grid[y_idx, x_idx]
        Fy = fy_grid[y_idx, x_idx]
        return (Fx, Fy)
    
    def drag_force(self):
        speed = np.hypot(self.vx, self.vy)
        if speed < 1e-8:
            return (0.0, 0.0)
        rho_air = self.atmosphere.density(self.y)
        F_drag_mag = 0.5 * CD * AREA * rho_air * speed**2
        drag_x = -F_drag_mag * (self.vx / speed)
        drag_y = -F_drag_mag * (self.vy / speed)
        return (drag_x, drag_y)
    
    def net_force(self, t):
        Fx_buoy, Fy_buoy = self.buoyant_force(t)
        Fx_wt, Fy_wt = self.weight()
        Fx_drag, Fy_drag = self.drag_force()
        Fx_ext, Fy_ext = self.external_force()
        Fx_net = Fx_buoy + Fx_wt + Fx_drag + Fx_ext
        Fy_net = Fy_buoy + Fy_wt + Fy_drag + Fy_ext
        return (Fx_net, Fy_net)
    
    def update(self, t, dt, control_force=(0.0, 0.0)):
        # Add the control force in both x and y directions.
        Fx_net, Fy_net = self.net_force(t)
        Fx_net += control_force[0]
        Fy_net += control_force[1]
        ax = Fx_net / self.mass_balloon
        ay = Fy_net / self.mass_balloon
        self.vx += ax * dt
        self.vy += ay * dt
        # Clamp velocities to prevent runaway speeds.
        max_velocity = 200.0
        self.vx = np.clip(self.vx, -max_velocity, max_velocity)
        self.vy = np.clip(self.vy, -max_velocity, max_velocity)
        self.x += self.vx * dt
        self.y += self.vy * dt
        if self.y < 0:
            self.y = 0
            self.vy = 0


# -------------------------------
# CUSTOM GYM ENVIRONMENT
# -------------------------------
class Balloon2DEnv(gym.Env):
    """
    A Gym environment for 2D balloon control.
    
    - **Observation (22 dimensions):**
          [x, y, vx, vy, local wind info (9 cells × 2 values)]
      where the local wind info is extracted from a 3x3 grid (the current cell plus its 8 neighbors).
    - **Action (Discrete 9):**  
      Each action corresponds to a 2D control force:
          horizontal force = (action mod 3 – 1) * control_force_mag_x  
          vertical force   = (action // 3 – 1) * control_force_mag_y  
    - **Reward:**  
          Negative Euclidean distance from the balloon's position to the target (0, 25000).
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(Balloon2DEnv, self).__init__()
        # Observation: [x, y, vx, vy, 18 wind components] → 22 values.
        low_obs = np.concatenate((
            np.array([X_RANGE[0], Y_RANGE[0], -200.0, -200.0]),
            np.full(18, -20.0)
        ))
        high_obs = np.concatenate((
            np.array([X_RANGE[1], Y_RANGE[1], 200.0, 200.0]),
            np.full(18, 20.0)
        ))
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)
        # Action space: 9 discrete actions (3×3 grid).
        self.action_space = spaces.Discrete(9)
        
        # Larger control forces allowed.
        self.control_force_mag_x = 500.0
        self.control_force_mag_y = 500.0
        # New target position.
        self.target_x = 0.0
        self.target_y = 25000.0
        self.balloon = Balloon()
        self.time = 0.0
        self.step_count = 0
        
        # For rendering.
        self.fig = None
        self.ax = None
        self.circle = None
        self.target_marker = None
    
    def get_local_wind(self):
        """
        Returns a flattened array of the wind forces (fx, fy) from the 3x3 grid of cells
        surrounding the current cell where the balloon is located.
        """
        x_idx = np.searchsorted(x_edges, self.balloon.x) - 1
        y_idx = np.searchsorted(y_edges, self.balloon.y) - 1
        x_idx = np.clip(x_idx, 0, GRID_CELLS - 1)
        y_idx = np.clip(y_idx, 0, GRID_CELLS - 1)
        wind_values = []
        for j in range(y_idx - 1, y_idx + 2):
            for i in range(x_idx - 1, x_idx + 2):
                i_clipped = np.clip(i, 0, GRID_CELLS - 1)
                j_clipped = np.clip(j, 0, GRID_CELLS - 1)
                wind_values.append(fx_grid[j_clipped, i_clipped])
                wind_values.append(fy_grid[j_clipped, i_clipped])
        return np.array(wind_values, dtype=np.float32)
    
    def step(self, action):
        # Map the discrete action (0–8) to a 2D control force.
        action_x = (action % 3) - 1   # yields -1, 0, or 1.
        action_y = (action // 3) - 1  # yields -1, 0, or 1.
        control_x = action_x * self.control_force_mag_x
        control_y = action_y * self.control_force_mag_y
        self.balloon.update(self.time, DT, control_force=(control_x, control_y))
        self.time += DT
        self.step_count += 1
        
        local_wind = self.get_local_wind()
        obs = np.concatenate((
            np.array([self.balloon.x, self.balloon.y, self.balloon.vx, self.balloon.vy]),
            local_wind
        ))
        dx = self.balloon.x - self.target_x
        dy = self.balloon.y - self.target_y
        reward = -np.sqrt(dx * dx + dy * dy)
        done = self.step_count >= EPISODE_LENGTH
        info = {}
        return obs, reward, done, info

    def reset(self):
        self.balloon = Balloon()
        self.time = 0.0
        self.step_count = 0
        local_wind = self.get_local_wind()
        return np.concatenate((
            np.array([self.balloon.x, self.balloon.y, self.balloon.vx, self.balloon.vy]),
            local_wind
        ))
    
    def render(self, mode='human'):
        if self.fig is None:
            self.fig, self.ax = plt.subplots()
            self.ax.set_title("Balloon Environment")
            self.ax.set_xlabel("X position (m)")
            self.ax.set_ylabel("Altitude (m)")
            self.ax.set_xlim(X_RANGE)
            self.ax.set_ylim(Y_RANGE)
            self.circle = plt.Circle((self.balloon.x, self.balloon.y), 50, color='red')
            self.ax.add_patch(self.circle)
            # Plot target marker.
            self.target_marker, = self.ax.plot(self.target_x, self.target_y, marker='*', color='green', markersize=15)
        self.circle.center = (self.balloon.x, self.balloon.y)
        plt.pause(0.001)
    
    def close(self):
        if self.fig:
            plt.close(self.fig)
    
    def save_wind_field(self, filename="wind_field.png"):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("Wind Field")
        ax.set_xlabel("X position (m)")
        ax.set_ylabel("Altitude (m)")
        ax.set_xlim(X_RANGE)
        ax.set_ylim(Y_RANGE)
        # q = ax.quiver(x_forces, y_forces, fx_grid, fy_grid, color='blue', alpha=0.5)          # Unused so far
        for xe in x_edges:
            ax.axvline(x=xe, color='gray', linestyle='--', alpha=0.3)
        for ye in y_edges:
            ax.axhline(y=ye, color='gray', linestyle='--', alpha=0.3)
        fig.savefig(filename)
        plt.close(fig)


if __name__ == "__main__":
    # Quick test of the environment.
    env = Balloon2DEnv()
    obs = env.reset()
    for _ in range(10):
        obs, reward, done, info = env.step(env.action_space.sample())
        env.render()
        print(f"Obs: {obs}, Reward: {reward}")
    env.save_wind_field("test_wind_field.png")
    env.close()
