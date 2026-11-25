"""
render.pygame_renderer
----------------------

A standalone viewer: `PygameRenderer.draw(state_dict)` produces
the same split-panel visual as the original `_render_frame()`.

The renderer is completely stateless except for the Pygame window/clock,
so you can reuse one instance across episodes or even different envs.
"""

from __future__ import annotations
import colorsys
import math
import os
import warnings
from typing import Tuple, Dict, Any
import numpy as np

warnings.filterwarnings(
    "ignore", message="pkg_resources is deprecated", category=UserWarning
)
import pygame  # noqa: E402 -- must follow the filterwarnings() call above

_TWO_PI = 2.0 * math.pi

# Minor gradation spacing on the altitude axis (m).
ALT_MINOR_TICK = 500.0

# Axis furniture. Majors and minors are both black, so the hierarchy between
# them has to come from tick *length* rather than colour.
AXIS_COL = (0, 0, 0)
PLOT_FILL = (240, 240, 240)
# Majors rule right across the altitude column; minors are short marks in from
# each edge. Both are black, so length alone separates them.
TICK_MINOR = 7

# Wind arrows are fixed-length with speed shown by colour, so direction stays
# legible at any zoom. The speed reference is fixed rather than per-frame, so
# a colour means the same speed in every episode.
ARROW_PX = 13
ARROW_SPEED_MAX = 25.0  # m/s mapped to the top of the colour ramp

# Vertical wind profile beside the altitude column.
WIND_PROFILE_STEP = 2_000.0  # m between sampled levels
WIND_PROFILE_PX = 11


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
        dim: int,
        fps: int = 60,
    ):
        self.window_w, self.window_h = window_size
        self.x_range, self.y_range, self.z_range = x_range, y_range, z_range
        self.x_centers, self.y_centers = x_centers, y_centers
        self.wind_cells = wind_cells
        self.dim = dim

        self.left_w = int(self.window_w * 0.75)
        self.right_w = self.window_w - self.left_w

        # Gutters for the axis gradations. The map is plotted inside these so
        # the labels sit off the grid rather than over the wind field.
        self.pad_l = 44  # y labels, left of the map
        self.pad_b = 22  # x labels, below the map
        self.plot_w = self.left_w - self.pad_l
        self.plot_h = self.window_h - self.pad_b

        # The altitude column is framed like the map plot, with its km labels
        # outside to the right. The top margin clears the skip button, which
        # is centred above it.
        self.alt_top = 34
        self.alt_col_w = 60
        self.alt_x = self.right_w // 2  # axis centre, panel-local

        os.environ["SDL_VIDEO_WINDOW_POS"] = "0,0"
        os.environ.pop("SDL_VIDEO_CENTERED", None)

        pygame.init()
        pygame.display.init()
        pygame.display.set_caption(f"Project Stratus - {self.dim}D Balloon Environment")
        self.window = pygame.display.set_mode(window_size, display=0)
        self.clock = pygame.time.Clock()

        # Built once: SysFont() parses the font cache on every call, which is
        # wasted work in a method that runs every frame.
        pygame.font.init()
        self.font = pygame.font.SysFont(None, 22)
        self.tick_font = pygame.font.SysFont(None, 18)
        self.big_font = pygame.font.SysFont(None, 36)
        # One decision = `decision_interval` sub-steps, each rendered once, each
        # throttled to `fps` -> decisions/sec = fps / decision_interval (60 sub-steps
        # by default). fps=60 is real-time-feeling (~1 decision/s); scale it up to
        # fast-forward playback.
        self.fps = fps
        self.skip_requested = False
        self.quit_requested = False

    # ------------------------------------------------------------------ #
    # axes
    # ------------------------------------------------------------------ #
    @staticmethod
    def _tick_step(span_m: float, target_ticks: int = 5) -> float:
        """A round 1/2/5 x 10^n step giving roughly ``target_ticks`` ticks."""
        raw = span_m / max(1, target_ticks)
        exp = math.floor(math.log10(raw)) if raw > 0 else 0
        base = 10.0**exp
        for mult in (1.0, 2.0, 5.0):
            if raw <= mult * base:
                return mult * base
        return 10.0 * base

    @staticmethod
    def _dashed_circle(surf, colour, centre, radius, width=2, dash=10, gap=8) -> None:
        """A dashed circle; pygame.draw has no dash support.

        ``dash``/``gap`` are arc lengths in pixels, so the dash density stays
        the same whatever the radius.
        """
        if radius <= 0:
            return
        cx, cy = centre
        circumference = 2.0 * math.pi * radius
        n = max(4, int(circumference / max(1.0, dash + gap)))
        arc = _TWO_PI / n
        on = arc * dash / max(1e-6, dash + gap)
        for i in range(n):
            a0 = i * arc
            pts = [
                (
                    cx + radius * math.cos(a0 + t * on),
                    cy + radius * math.sin(a0 + t * on),
                )
                for t in (0.0, 0.25, 0.5, 0.75, 1.0)
            ]
            pygame.draw.lines(surf, colour, False, pts, width)

    @staticmethod
    def _speed_colour(speed: float):
        """Blue -> amber -> red ramp over [0, ARROW_SPEED_MAX] m/s.

        Every stop is saturated, so a slow wind is still clearly visible —
        colour carries speed without visibility depending on it. The reference
        is a fixed constant rather than the frame's maximum, so a colour means
        the same thing across episodes and between runs.
        """
        t = min(1.0, max(0.0, speed / ARROW_SPEED_MAX))
        # Interpolate hue, not RGB: blending blue to red through RGB passes
        # through grey and olive, which read as "faded" rather than "slower".
        # Rotating the hue from blue to red keeps every speed fully saturated.
        hue = (210.0 / 360.0) * (1.0 - t)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.80)
        return (int(r * 255), int(g * 255), int(b * 255))

    @staticmethod
    def _draw_arrow(surf, colour, centre, angle: float, length: int) -> None:
        """A fixed-length arrow centred on ``centre``, pointing at ``angle``.

        ``angle`` is in screen space (x right, y down) and ``length`` is in
        pixels, so the arrow is legible at any zoom and for any wind speed.
        """
        cx, cy = centre
        dx, dy = math.cos(angle), math.sin(angle)
        half = length / 2.0
        tail = (cx - dx * half, cy - dy * half)
        tip = (cx + dx * half, cy + dy * half)
        pygame.draw.line(surf, colour, tail, tip, 2)
        head = max(4.0, length * 0.42)
        for s in (-1, 1):
            a = angle + s * (math.pi - math.pi / 6)
            pygame.draw.line(
                surf,
                colour,
                tip,
                (tip[0] + head * math.cos(a), tip[1] + head * math.sin(a)),
                2,
            )

    def _draw_wind_profile(self, surf, wind_sampler, x: float, y: float) -> None:
        """Wind bearing against altitude, in the gutter left of the column.

        The agent's only control is altitude, and on these fields the wind
        turns with height while barely varying horizontally — so this, not the
        map, is the picture of what it is choosing between.
        """
        cx = (self.alt_x - self.alt_col_w // 2) // 2
        cap = self.tick_font.render("wind", True, AXIS_COL)
        surf.blit(cap, (cx - cap.get_width() // 2, self.alt_top - cap.get_height() - 4))
        z = math.ceil(self.z_range[0] / WIND_PROFILE_STEP) * WIND_PROFILE_STEP
        while z <= self.z_range[1]:
            wx, wy, _ = wind_sampler(x, y, z)
            speed = math.hypot(wx, wy)
            if speed > 0.0:
                self._draw_arrow(
                    surf,
                    self._speed_colour(speed),
                    (cx, self._to_alt_y(z)),
                    math.atan2(-wy, wx),
                    WIND_PROFILE_PX,
                )
            z += WIND_PROFILE_STEP

    def _to_alt_y(self, z: float) -> int:
        """Altitude in metres to a y pixel in the altitude panel.

        Single source of truth for the vertical mapping: the balloon marker,
        the gradations and the 1D goal and its tolerance band all come through
        here, so the column's extent is set in one place.
        """
        frac = (z - self.z_range[0]) / (self.z_range[1] - self.z_range[0])
        return int(self.alt_top + (1.0 - frac) * (self.plot_h - self.alt_top))

    @staticmethod
    def _km(metres: float) -> str:
        """Signed km label, with the station itself as a plain "0"."""
        return "0" if abs(metres) < 1.0 else f"{metres / 1000:+.0f}"

    def _draw_map_axes(self, surf, goal_x: float, goal_y: float, to_left) -> None:
        """Tick marks and km labels in the map gutters, relative to the goal.

        Ticks run outwards from the goal in both directions so the station is
        always "0 km" and every other label reads as a distance from it.
        """
        axis = AXIS_COL
        pygame.draw.rect(surf, axis, (self.pad_l, 0, self.plot_w, self.plot_h), width=1)
        step = self._tick_step(self.x_range[1] - self.x_range[0])

        def ticks(centre: float, lo: float, hi: float):
            k = math.ceil((lo - centre) / step)
            while centre + k * step <= hi:
                yield centre + k * step, (centre + k * step - centre)
                k += 1

        for wx, off in ticks(goal_x, *self.x_range):
            px, _ = to_left(wx, self.y_range[0])
            pygame.draw.line(surf, axis, (px, self.plot_h), (px, self.plot_h + 4))
            label = self.tick_font.render(self._km(off), True, axis)
            surf.blit(label, (px - label.get_width() // 2, self.plot_h + 6))

        for wy, off in ticks(goal_y, *self.y_range):
            _, py = to_left(self.x_range[0], wy)
            pygame.draw.line(surf, axis, (self.pad_l - 4, py), (self.pad_l, py))
            label = self.tick_font.render(self._km(off), True, axis)
            surf.blit(
                label,
                (self.pad_l - 7 - label.get_width(), py - label.get_height() // 2),
            )

        units = self.tick_font.render("km from station", True, axis)
        surf.blit(units, (self.pad_l + 6, 6))

        # Speed key: without it the arrow colours carry no meaning.
        bar_w, bar_h = 70, 7
        bx = self.left_w - bar_w - 48
        for i in range(bar_w):
            pygame.draw.line(
                surf,
                self._speed_colour(ARROW_SPEED_MAX * i / (bar_w - 1)),
                (bx + i, 8),
                (bx + i, 8 + bar_h),
            )
        pygame.draw.rect(surf, axis, (bx, 8, bar_w, bar_h), width=1)
        lo = self.tick_font.render("0", True, axis)
        hi = self.tick_font.render(f"{ARROW_SPEED_MAX:.0f} m/s", True, axis)
        surf.blit(lo, (bx - lo.get_width() - 4, 6))
        surf.blit(hi, (bx + bar_w + 4, 6))

    def _draw_alt_axis(self, surf) -> None:
        """The framed altitude column, its gradations and km labels.

        Framed and filled like the map plot, with the labels outside the frame
        to the right — the same convention the map uses for its gutters.
        """
        step = self._tick_step(self.z_range[1] - self.z_range[0])
        col_x = self.alt_x - self.alt_col_w // 2
        col = pygame.Rect(
            col_x, self.alt_top, self.alt_col_w, self.plot_h - self.alt_top
        )
        pygame.draw.rect(surf, PLOT_FILL, col)
        pygame.draw.rect(surf, AXIS_COL, col, width=1)

        # Minors are drawn in from the frame edges rather than out from the
        # centre line, so they read as a scale on the column.
        z = math.ceil(self.z_range[0] / ALT_MINOR_TICK) * ALT_MINOR_TICK
        while z <= self.z_range[1]:
            py = self._to_alt_y(z)
            pygame.draw.line(
                surf, AXIS_COL, (col.left + 1, py), (col.left + 1 + TICK_MINOR, py)
            )
            pygame.draw.line(
                surf, AXIS_COL, (col.right - 2 - TICK_MINOR, py), (col.right - 2, py)
            )
            z += ALT_MINOR_TICK

        z = math.ceil(self.z_range[0] / step) * step
        while z <= self.z_range[1]:
            py = self._to_alt_y(z)
            pygame.draw.line(surf, AXIS_COL, (col.left + 1, py), (col.right - 2, py))
            label = self.tick_font.render(f"{z / 1000:.0f} km", True, AXIS_COL)
            ly = min(
                max(0, py - label.get_height() // 2), self.plot_h - label.get_height()
            )
            surf.blit(label, (col.right + 8, ly))
            z += step

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def draw(self, state: Dict[str, Any]) -> None:
        """
        `state` must provide:
            • dim            (int)
            • balloon_pos    (np.ndarray)
            • goal_pos       (np.ndarray)
            • z0             (float)         - fixed altitude for dim==2
            • wind_sampler   (callable (x,y,z)->np.ndarray)
            • station_radius (float)         - optional, full-reward region
        """
        balloon_pos = state["balloon_pos"]
        goal_pos = state["goal_pos"]
        z0 = state["z0"]
        wind_sampler = state["wind_sampler"]
        station_radius = state.get("station_radius")

        canvas = pygame.Surface((self.window_w, self.window_h))
        canvas.fill((255, 255, 255))

        # ------------- coordinate helper ---------------------------------
        # Plots into the padded rect; the gutters carry the axis labels.
        def to_left(px: float, py: float):
            sx = (px - self.x_range[0]) / (self.x_range[1] - self.x_range[0])
            sy = (py - self.y_range[0]) / (self.y_range[1] - self.y_range[0])
            return (
                int(self.pad_l + sx * self.plot_w),
                int((1.0 - sy) * self.plot_h),
            )

        # ------------- map panel -----------------------------------------
        map_surface = pygame.Surface((self.left_w, self.window_h))
        map_surface.fill((255, 255, 255))
        pygame.draw.rect(
            map_surface, (240, 240, 240), (self.pad_l, 0, self.plot_w, self.plot_h)
        )

        # Fixed-length arrows: direction from the wind vector, speed from
        # colour. Scaling the shaft by the wind vector in world metres makes
        # it sub-pixel at this zoom (13 m on a 100 km map), which collapses
        # every arrow onto one pixel and leaves atan2(0, 0) == 0 — i.e. every
        # arrow pointing due east whatever the wind was doing.
        STEP = 4
        for ix in range(0, self.wind_cells, STEP):
            for iy in range(0, self.wind_cells, STEP):
                x = self.x_centers[ix]
                y = self.y_centers[iy]
                z = balloon_pos[-1] if self.dim == 3 else z0
                wx, wy, _ = wind_sampler(x, y, z)
                speed = math.hypot(wx, wy)
                if speed == 0.0:
                    continue
                self._draw_arrow(
                    map_surface,
                    self._speed_colour(speed),
                    to_left(x, y),
                    # Screen y grows downward while world y grows north, so
                    # the vertical component flips.
                    math.atan2(-wy, wx),
                    ARROW_PX,
                )

        # ------------- axis gradations -----------------------------------
        # Ticks are placed at round distances from the goal, so "0 km" marks
        # the station and the rest read as offsets from it. The goal is fixed
        # for an episode, so the gradations are static for a whole run.
        gx_w, gy_w = (
            (float(goal_pos[0]), float(goal_pos[1])) if self.dim >= 2 else (0.0, 0.0)
        )
        self._draw_map_axes(map_surface, gx_w, gy_w, to_left)

        # balloon, goal and the full-reward region
        bx, by = to_left(*(balloon_pos[:2] if self.dim >= 2 else (0.0, 0.0)))
        gx, gy = to_left(*(goal_pos[:2] if self.dim >= 2 else (0.0, 0.0)))
        if station_radius and self.dim >= 2:
            # x and y share a scale, so the region is a true circle.
            r_px = int(
                station_radius * self.plot_w / (self.x_range[1] - self.x_range[0])
            )
            self._dashed_circle(map_surface, (0, 0, 0), (gx, gy), r_px, width=2)
        pygame.draw.circle(map_surface, (255, 0, 0), (bx, by), 8)
        pygame.draw.circle(map_surface, (0, 200, 0), (gx, gy), 6)

        # ------------- altitude bar --------------------------------------
        alt_surface = pygame.Surface((self.right_w, self.window_h))
        alt_surface.fill((250, 250, 250))
        # Altitude is only an objective in 1D; in 2D/3D the reward is the
        # horizontal distance alone (reward.l2_distance) and `goal_pos[-1]`
        # is an unused coordinate, so drawing it as a target is misleading.
        show_goal_alt = self.dim == 1

        self._draw_alt_axis(alt_surface)
        self._draw_wind_profile(
            alt_surface,
            wind_sampler,
            float(balloon_pos[0]) if self.dim >= 2 else 0.0,
            float(balloon_pos[1]) if self.dim >= 2 else 0.0,
        )

        # The tolerance band is clipped to the framed column so it reads as
        # part of the scale rather than a stripe across the whole panel.
        if show_goal_alt and station_radius:
            top = self._to_alt_y(goal_pos[-1] + station_radius)
            bot = self._to_alt_y(goal_pos[-1] - station_radius)
            band = pygame.Surface((self.alt_col_w - 2, max(1, bot - top)))
            band.set_alpha(70)
            band.fill((0, 200, 0))
            alt_surface.blit(band, (self.alt_x - self.alt_col_w // 2 + 1, top))

        pygame.draw.line(
            alt_surface,
            AXIS_COL,
            (self.alt_x, self.alt_top),
            (self.alt_x, self.plot_h),
            2,
        )

        if show_goal_alt:
            pygame.draw.circle(
                alt_surface, (0, 200, 0), (self.alt_x, self._to_alt_y(goal_pos[-1])), 5
            )
        pygame.draw.circle(
            alt_surface, (255, 0, 0), (self.alt_x, self._to_alt_y(balloon_pos[-1])), 6
        )

        # ------------- blit ----------------------------------------------
        canvas.blit(map_surface, (0, 0))
        canvas.blit(alt_surface, (self.left_w, 0))
        self.window.blit(canvas, canvas.get_rect())

        # ------------- "Next (N)" skip button ----------------------------
        # Centred in the altitude panel's top margin, above the column.
        btn_w, btn_h = 80, 24
        btn_x = self.left_w + (self.right_w - btn_w) // 2
        btn_y = (self.alt_top - btn_h) // 2
        btn_rect = pygame.Rect(btn_x, btn_y, btn_w, btn_h)
        pygame.draw.rect(self.window, (60, 60, 60), btn_rect, border_radius=5)
        pygame.draw.rect(
            self.window, (180, 180, 180), btn_rect, width=1, border_radius=5
        )
        btn_text = self.tick_font.render("Next (N)", True, (220, 220, 220))
        self.window.blit(
            btn_text,
            (
                btn_x + (btn_w - btn_text.get_width()) // 2,
                btn_y + (btn_h - btn_text.get_height()) // 2,
            ),
        )

        pygame.display.update()

        # Process events here (pump would discard them before the test loop sees them)
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self.quit_requested = True
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_n:
                self.skip_requested = True
            elif ev.type == pygame.MOUSEBUTTONDOWN and btn_rect.collidepoint(ev.pos):
                self.skip_requested = True

        self.clock.tick(self.fps)

    # ------------------------------------------------------------------
    def show_end_screen(self, reason: str, duration: float = 3.0) -> bool:
        """
        Freeze the current frame and overlay the termination reason for
        *duration* seconds.  Returns True if the user pressed N or clicked
        "Next" to skip ahead early, False otherwise.
        """
        font = self.big_font
        small_font = self.font

        # Semi-transparent overlay
        overlay = pygame.Surface((self.window_w, self.window_h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 140))
        self.window.blit(overlay, (0, 0))

        # Reason text (centered)
        text_surf = font.render(reason, True, (255, 255, 255))
        tx = (self.window_w - text_surf.get_width()) // 2
        ty = (self.window_h - text_surf.get_height()) // 2 - 20
        self.window.blit(text_surf, (tx, ty))

        # "Next" button
        btn_w, btn_h = 100, 36
        btn_x = (self.window_w - btn_w) // 2
        btn_y = ty + text_surf.get_height() + 24
        btn_rect = pygame.Rect(btn_x, btn_y, btn_w, btn_h)
        pygame.draw.rect(self.window, (70, 70, 70), btn_rect, border_radius=6)
        pygame.draw.rect(
            self.window, (200, 200, 200), btn_rect, width=1, border_radius=6
        )
        btn_text = small_font.render("Next (N)", True, (255, 255, 255))
        self.window.blit(
            btn_text,
            (
                btn_x + (btn_w - btn_text.get_width()) // 2,
                btn_y + (btn_h - btn_text.get_height()) // 2,
            ),
        )

        pygame.display.update()

        start = pygame.time.get_ticks()
        while (pygame.time.get_ticks() - start) < duration * 1000:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    return False
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_n:
                    return True
                if ev.type == pygame.MOUSEBUTTONDOWN and btn_rect.collidepoint(ev.pos):
                    return True
            self.clock.tick(30)
        return False

    # ------------------------------------------------------------------
    def close(self):
        pygame.display.quit()
        pygame.quit()
