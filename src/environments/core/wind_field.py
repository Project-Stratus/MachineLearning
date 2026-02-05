"""
core.wind_field
---------------

Builds a 3-D wind-velocity field on a regular grid and provides fast
trilinear interpolation.  Patterns currently supported:

    • "sinusoid"       - original wavy field (2D)
    • "linear_right"   - constant +X wind (2D)
    • "linear_up"      - constant +Y wind (2D)
    • "split_fork"     - fan-out pattern (2D)
    • "altitude_shear" - east/west wind based on altitude (3D)
                         west wind below midpoint, east wind above

Extend `_build_grid()` to add more patterns.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Tuple
import numpy as np

try:
    from environments.core.jit_kernels import wind_sample_idx_numba
    _JIT_OK = True
except Exception:
    _JIT_OK = False


class WindField:
    def __init__(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        z_range: Tuple[float, float],
        cells: int = 40,
        pattern: str = "sinusoid",
        default_mag: float = 10.0,
        wind_cfg_path: str | Path | None = None,
    ):
        self.x_range, self.y_range, self.z_range = x_range, y_range, z_range
        self.cells = cells
        self.pattern = pattern

        # --- magnitude -----------------------------------------------------
        self.mag = default_mag
        if wind_cfg_path:
            try:
                cfg = json.loads(Path(wind_cfg_path).read_text())
                if pattern in cfg:
                    self.mag = cfg[pattern].get("wind_mag", default_mag)
            except FileNotFoundError:
                pass  # silently ignore missing catalogue

        # --- grids ---------------------------------------------------------
        self.x_edges = np.linspace(x_range[0], x_range[1], cells + 1)
        self.y_edges = np.linspace(y_range[0], y_range[1], cells + 1)
        self.z_edges = np.linspace(z_range[0], z_range[1], cells + 1)
        self.x_centers = (self.x_edges[:-1] + self.x_edges[1:]) / 2
        self.y_centers = (self.y_edges[:-1] + self.y_edges[1:]) / 2
        self.z_centers = (self.z_edges[:-1] + self.z_edges[1:]) / 2

        self._build_grid()  # fills self._fx_grid, self._fy_grid

        self.dx = (x_range[1] - x_range[0]) / self.cells
        self.dy = (y_range[1] - y_range[0]) / self.cells
        self.dz = (z_range[1] - z_range[0]) / self.cells
        self.inv_dx = 1.0 / self.dx
        self.inv_dy = 1.0 / self.dy
        self.inv_dz = 1.0 / self.dz

    def _to_idx(self, xi, x0, inv_dx, cells):
        ix = int((xi - x0) * inv_dx)
        if ix < 0:
            ix = 0
        elif ix >= cells:
            ix = cells - 1
        return ix

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    # def sample(self, x: float, y: float, z: float) -> np.ndarray:
    #     """Return (fx, fy, fz) at continuous point (x,y,z)."""
    #     xi = np.clip(x, *self.x_range)
    #     yi = np.clip(y, *self.y_range)
    #     zi = np.clip(z, *self.z_range)

    #     ix = np.clip(np.searchsorted(self.x_edges, xi) - 1, 0, self.cells - 1)
    #     iy = np.clip(np.searchsorted(self.y_edges, yi) - 1, 0, self.cells - 1)
    #     iz = np.clip(np.searchsorted(self.z_edges, zi) - 1, 0, self.cells - 1)

    #     fx = self._fx_grid[ix, iy, iz]
    #     fy = self._fy_grid[ix, iy, iz]
    #     return np.array([fx, fy, 0.0], dtype=np.float32)

    def sample(self, x: float, y: float, z: float) -> np.ndarray:
        xi = x if x >= self.x_range[0] else self.x_range[0]
        xi = xi if xi <= self.x_range[1] else self.x_range[1]
        yi = y if y >= self.y_range[0] else self.y_range[0]
        yi = yi if yi <= self.y_range[1] else self.y_range[1]
        zi = z if z >= self.z_range[0] else self.z_range[0]
        zi = zi if zi <= self.z_range[1] else self.z_range[1]

        if _JIT_OK:
            fx, fy = wind_sample_idx_numba(xi, yi, zi,
                                             self.x_range[0], self.inv_dx,
                                             self.y_range[0], self.inv_dy,
                                             self.z_range[0], self.inv_dz,
                                             self.cells,
                                             self._fx_grid, self._fy_grid)
        else:
            # Fallback: previous searchsorted approach
            ix = np.clip(np.searchsorted(self.x_edges, xi) - 1, 0, self.cells - 1)
            iy = np.clip(np.searchsorted(self.y_edges, yi) - 1, 0, self.cells - 1)
            iz = np.clip(np.searchsorted(self.z_edges, zi) - 1, 0, self.cells - 1)
            fx = self._fx_grid[ix, iy, iz]
            fy = self._fy_grid[ix, iy, iz]

        # ix = self._to_idx(xi, self.x_range[0], self.inv_dx, self.cells)
        # iy = self._to_idx(yi, self.y_range[0], self.inv_dy, self.cells)
        # iz = self._to_idx(zi, self.z_range[0], self.inv_dz, self.cells)

        # # avoid new array allocation on hot path
        # fx = float(self._fx_grid[ix, iy, iz])
        # fy = float(self._fy_grid[ix, iy, iz])
        return np.array([fx, fy, 0.0], dtype=np.float32)

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _build_grid(self) -> None:
        X, Y, Z = np.meshgrid(
            self.x_centers, self.y_centers, self.z_centers, indexing="ij"
        )
        mag = self.mag
        xr, yr, zr = self.x_range, self.y_range, self.z_range

        if self.pattern == "linear_right":
            self._fx_grid = np.full_like(X, mag)
            self._fy_grid = np.zeros_like(Y)

        elif self.pattern == "linear_up":
            self._fx_grid = np.zeros_like(X)
            self._fy_grid = np.full_like(Y, mag)

        elif self.pattern == "split_fork":
            Xn = np.clip((X - xr[0]) / (xr[1] - xr[0]), 0.0, 1.0)
            Yn = np.clip(np.abs(Y) / (yr[1] - yr[0]), 0.0, 1.0)
            alpha = Xn * Yn
            self._fx_grid = mag * (1.0 - alpha)
            self._fy_grid = mag * alpha * np.sign(Y)

        elif self.pattern == "altitude_shear":
            # Normalize Z to [0, 1] where 0 = low altitude, 1 = high altitude
            Zn = (Z - zr[0]) / (zr[1] - zr[0])
            # Linear shear: -mag at bottom (west), +mag at top (east)
            # Crossover (zero wind) at midpoint altitude
            self._fx_grid = mag * (2.0 * Zn - 1.0)
            self._fy_grid = np.zeros_like(Y)

        else:  # "sinusoid" default
            self._fx_grid = (mag * 0.5 * (np.sin(2 * np.pi * X / (xr[1] - xr[0]))
                             + 0.5 * np.sin(4 * np.pi * X / (xr[1] - xr[0]))))
            self._fy_grid = (mag * 0.5 * (np.cos(2 * np.pi * Y / (yr[1] - yr[0]))
                             + 0.5 * np.cos(4 * np.pi * Y / (yr[1] - yr[0]))))
            # gentle altitude shear
            self._fx_grid += (mag / 4) * np.sin(2 * np.pi * Z / (zr[1] - zr[0]))
