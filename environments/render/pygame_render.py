"""
render.pygame_renderer
----------------------

A standalone viewer: `PygameRenderer.draw(state_dict)` produces
the same split-panel visual as the original `_render_frame()`.

The renderer is completely stateless except for the Pygame window/clock,
so you can reuse one instance across episodes or even different envs.
"""
from __future__ import annotations
import math
import os
from typing import Tuple, Dict, Any
import numpy as np
import pygame


class PygameRenderer:
    def __init__(
        self,
        window_size: Tuple[int, int],
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        z_range: Tuple[float, float],
        x_centers: np.ndarray,
        y_centers: np.ndarray,
        wind_cells: int,
    ):
        self.window_w, self.window_h = window_size
        self.x_range, self.y_range, self.z_range = x_range, y_range, z_range
        self.x_centers, self.y_centers = x_centers, y_centers
        self.wind_cells = wind_cells

        self.left_w = int(self.window_w * 0.75)
        self.right_w = self.window_w - self.left_w

        os.environ["SDL_VIDEO_WINDOW_POS"] = "0,0"
        os.environ.pop("SDL_VIDEO_CENTERED", None)

        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode(window_size, display=0)
        self.clock = pygame.time.Clock()
        self.fps = 60

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def draw(self, state: Dict[str, Any]) -> None:
        """
        `state` must provide:
            • dim            (int)
            • balloon_pos    (np.ndarray)
            • goal_pos       (np.ndarray)
            • z0             (float)         – fixed altitude for dim==2
            • wind_sampler   (callable (x,y,z)->np.ndarray)
        """
        dim = state["dim"]
        balloon_pos = state["balloon_pos"]
        goal_pos = state["goal_pos"]
        z0 = state["z0"]
        wind_sampler = state["wind_sampler"]

        canvas = pygame.Surface((self.window_w, self.window_h))
        canvas.fill((255, 255, 255))

        # ------------- coordinate helper ---------------------------------
        def to_left(px: float, py: float):
            sx = (px - self.x_range[0]) / (self.x_range[1] - self.x_range[0])
            sy_range = self.y_range if dim == 3 else self.y_range
            sy = (py - sy_range[0]) / (sy_range[1] - sy_range[0])
            return int(sx * self.left_w), int((1.0 - sy) * self.window_h)

        # ------------- map panel -----------------------------------------
        map_surface = pygame.Surface((self.left_w, self.window_h))
        map_surface.fill((240, 240, 240))

        STEP, HEAD, COL = 4, 6, (180, 180, 255)
        for ix in range(0, self.wind_cells, STEP):
            for iy in range(0, self.wind_cells, STEP):
                x = self.x_centers[ix]
                y = self.y_centers[iy]
                z = balloon_pos[-1] if dim == 3 else z0
                wx, wy, _ = wind_sampler(x, y, z)
                if wx == wy == 0.0:
                    continue
                scale = 5
                x0, y0 = to_left(x, y)
                x1, y1 = to_left(x + wx * scale, y + wy * scale)
                pygame.draw.line(map_surface, COL, (x0, y0), (x1, y1), 1)
                ang = math.atan2(y1 - y0, x1 - x0)
                for s in (-1, 1):
                    dx = HEAD * math.cos(ang + s * math.pi / 6)
                    dy = HEAD * math.sin(ang + s * math.pi / 6)
                    pygame.draw.line(map_surface, COL, (x1, y1), (x1 - dx, y1 - dy), 1)

        # balloon & goal
        bx, by = to_left(*(balloon_pos[:2] if dim >= 2 else (0.0, 0.0)))
        gx, gy = to_left(*(goal_pos[:2] if dim >= 2 else (0.0, 0.0)))
        pygame.draw.circle(map_surface, (255, 0, 0), (bx, by), 8)
        pygame.draw.circle(map_surface, (0, 200, 0), (gx, gy), 6)

        # ------------- altitude bar --------------------------------------
        alt_surface = pygame.Surface((self.right_w, self.window_h))
        alt_surface.fill((250, 250, 250))
        pygame.draw.line(alt_surface, (0, 0, 0),
                         (self.right_w // 2, 0),
                         (self.right_w // 2, self.window_h), 2)
        z_b = balloon_pos[-1]
        z_g = goal_pos[-1]
        bz = (1.0 - (z_b - self.z_range[0]) / (self.z_range[1] - self.z_range[0])) * self.window_h
        gz = (1.0 - (z_g - self.z_range[0]) / (self.z_range[1] - self.z_range[0])) * self.window_h
        pygame.draw.circle(alt_surface, (255, 0, 0), (self.right_w // 2, int(bz)), 6)
        pygame.draw.circle(alt_surface, (0, 200, 0), (self.right_w // 2, int(gz)), 5)

        # ------------- blit ----------------------------------------------
        canvas.blit(map_surface, (0, 0))
        canvas.blit(alt_surface, (self.left_w, 0))
        self.window.blit(canvas, canvas.get_rect())
        pygame.display.update()
        pygame.event.pump()
        self.clock.tick(self.fps)

    # ------------------------------------------------------------------
    def close(self):
        pygame.display.quit()
        pygame.quit()
