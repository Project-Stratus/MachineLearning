from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np


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
T_MAX = 400.0            # Simulation duration (s)

# Volume fluctuation parameters
FLUCTUATION_FRACTION = 0.05   # ±5% of the stationary volume
FLUCTUATION_PERIOD = 60.0     # seconds for one full sine-wave cycle


class Actions(Enum):
    inflate = 0
    deflate = 1


class LoonEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.window_size = 512  # The size of the PyGame window

        # Define ranges for each observation
        self.max_mass = 20.0
        self.max_altitude = 20000.0
        self.max_velocity = 50.0
        self.max_pressure = 100000

        # Observations are dictionaries with the agent's and the target's location.
        self.observation_space = spaces.Dict(
            {
                "mass":      spaces.Box(low=0,                  high=self.max_mass,     dtype=float),
                "altitude":  spaces.Box(low=0,                  high=self.max_altitude, dtype=float),
                "velocity":  spaces.Box(low=-self.max_velocity, high=self.max_velocity, dtype=float),
                "pressure":  spaces.Box(low=0,                  high=self.max_pressure, dtype=float),
            }
        )


        self.action_space = spaces.Discrete(2)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"mass":     np.array([self._balloon.mass_balloon]),
                "altitude": np.array([self._balloon.altitude]),
                "velocity": np.array([self._balloon.velocity]),
                "pressure": np.array([self._atmosphere.pressure(self._balloon.altitude)])}

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        # Track the number of timesteps in the env
        self._time = 0

        # Randomly initialize the balloon and the atmosphere
        self._atmosphere = Atmosphere()
        self._balloon = Balloon(self._atmosphere,
                                mass_balloon=2.0,
                                altitude=2000.0,
                                velocity=0.0)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):

        # Update the Balloon's position
        self._balloon.update(self._time, 1.0)
        
        # Episodes finish after a number of time steps
        self._time += 1
        terminated = self._time >= 120

        # Binary sparse rewards
        reward = self._balloon.altitude 

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        # First we draw the balloon
        pygame.draw.circle(
            canvas,
            (255, 0, 0),
            (self.window_size/2.0, (self._balloon.altitude/self.max_altitude)*self.window_size),
            20
        )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


class Atmosphere:
    """
    Simple model of the atmosphere to retrieve pressure and density
    as functions of altitude.
    """
    def __init__(self, 
                 p0=P0, 
                 scale_height=SCALE_HEIGHT, 
                 temperature=T_AIR, 
                 molar_mass=M_AIR
                 ):
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
                 atmosphere:    Atmosphere,
                 mass_balloon:  float,          # kg, mass of balloon + payload
                 altitude:      float,          # m (start around 25km)
                 velocity:      float,          # m/s
    ):
        
        self.mass_balloon = mass_balloon
        self.altitude = altitude
        self.velocity = velocity

        # Atmosphere object for external conditions
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