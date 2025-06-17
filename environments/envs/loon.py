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


class Actions(Enum):
    inflate = 0
    deflate = 1
    nothing = 2


class LoonEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    TIME_MAX = 400
    PUNISHMENT = -400

    def __init__(self, render_mode=None):
        self.window_size = 512  # The size of the PyGame window

        # Define ranges for each observation
        self.max_volume = 20.0
        self.max_altitude = 20000.0
        self.max_velocity = 50.0
        self.max_pressure = 100000

        # Observations are dictionaries with the agent's and the target's location.
        self.observation_space = spaces.Dict(
            {
                "goal":      spaces.Box(low=0,    high=1.0, shape=(1,), dtype=float),
                "volume":    spaces.Box(low=0,    high=1.0, shape=(1,), dtype=float),
                "altitude":  spaces.Box(low=0,    high=1.0, shape=(1,), dtype=float),
                "velocity":  spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=float),
                "pressure":  spaces.Box(low=0,    high=1.0, shape=(1,), dtype=float),
            }
        )

        # The action space consists of two discrete actions: inflate and deflate
        self.action_space = spaces.Discrete(3)

        # Define a random goal for the agento to reach
        self.goal = self.np_random.uniform(low=0, high=self.max_altitude, size=(1,))

        self.final_obs = None
        self.truncated = False

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
        return {
            "goal":      np.array([self.goal / self.max_altitude]),
            "volume":    np.array([self._balloon.volume / self.max_volume]),
            "altitude":  np.array([self._balloon.altitude / self.max_altitude]),
            "velocity":  np.array([self._balloon.velocity / self.max_velocity]),
            "pressure":  np.array([self._atmosphere.pressure(self._balloon.altitude) / self.max_pressure])
        }

    def _get_info(self):
        return {
            "TimeLimit.truncated": self.truncated,
            "terminal_observation": self.final_obs,
            }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Track the number of timesteps in the env
        self._time = 0

        # Randomly initialize the balloon and the atmosphere
        self._atmosphere = Atmosphere()
        self._balloon = Balloon(self._atmosphere,
                                mass_balloon=2.0,
                                altitude=np.random.uniform(low=0, high=self.max_altitude),
                                velocity=0.0)

        # Randomly reset the goal
        self.goal = self.np_random.uniform(low=0, high=self.max_altitude)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):

        # Inflate or deflate the balloon
        if action == Actions.inflate.value:
            self._balloon.inflate(0.02)
        elif action == Actions.deflate.value:
            self._balloon.inflate(-0.02)

        # Update the Balloon's position
        self._balloon.update(DT)
        goal_dist = (1 - abs(self._balloon.altitude - self.goal)/self.max_altitude)**8  # Only reward if balloon is near the goal

        # Episodes finish after a number of time steps
        self._time += 1
        terminated = self._balloon.altitude <= 0
        truncated = self._time >= self.TIME_MAX

        if truncated and not terminated:
            self.truncated = True

        # Reward the player based on distance to the goal
        reward = goal_dist if not terminated else self.PUNISHMENT

        observation = self._get_obs()
        if terminated or truncated:
            self.final_obs = observation

        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

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
            (self.window_size/2.0, self.window_size - (self._balloon.altitude/self.max_altitude)*self.window_size),
            20
        )

        # Then we draw the goal
        pygame.draw.circle(
            canvas,
            (0, 255, 0),
            (self.window_size/2.0, self.window_size - (self.goal/self.max_altitude)*self.window_size),
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
        self.volume = self.mass_balloon / rho_air  # Default to a volume that would be stationary

    def buoyant_force(self):
        """
        Buoyant force = (density at altitude) * g * [dynamic volume].
        """
        rho_air = self.atmosphere.density(self.altitude)
        return rho_air * G * self.volume

    def weight(self):
        """
        Weight = mass * g.
        """
        return self.mass_balloon * G

    def net_force(self):
        """
        Net force = buoyant force - weight.
        """
        return self.buoyant_force() - self.weight()

    def inflate(self, delta_volume):
        """
        Inflate or deflate the balloon by a small amount.
        """
        self.volume += delta_volume

    def update(self, dt):
        """
        Update balloon's velocity and altitude (simple Euler method).
        """

        a = self.net_force() / self.mass_balloon
        self.velocity += a * dt
        self.altitude += self.velocity * dt
