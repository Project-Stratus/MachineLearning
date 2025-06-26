"""
Version by WS
1D env with altitude control
"""

from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from environments.core.atmosphere import Atmosphere
from environments.core.balloon import Balloon

from environments.core.constants import DT, VOL_MAX, ALT_MAX, VEL_MAX, P_MAX


class Actions(Enum):
    inflate = 0
    deflate = 1
    nothing = 2


class Balloon1DEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 150}

    TIME_MAX = 1_000
    PUNISHMENT = -400

    def __init__(self, render_mode=None):
        self.window_size = 512  # The size of the PyGame window

        # Define ranges for each observation
        self.max_volume = VOL_MAX
        self.max_altitude = ALT_MAX
        self.max_velocity = VEL_MAX
        self.max_pressure = P_MAX

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
        self._balloon = Balloon(atmosphere=self._atmosphere,
                                mass=2.0,
                                # altitude=np.random.uniform(low=0, high=self.max_altitude),
                                position=[np.random.uniform(0, self.max_altitude)],  # altitude → position[0]
                                velocity=[0.0])

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
