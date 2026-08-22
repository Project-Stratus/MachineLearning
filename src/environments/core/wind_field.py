"""
core.wind_field
---------------

Builds a 3-D wind-velocity field on a regular grid and provides fast
trilinear interpolation.  Patterns currently supported:

    • "sinusoid"       - original wavy field (2D)
    • "linear_right"   - constant +X wind (2D)
    • "linear_up"      - constant +Y wind (2D)
    • "split_fork"     - fan-out pattern (2D)
    • "altitude_shear"    - east/west wind based on altitude (3D)
                           west wind below midpoint, east wind above
    • "altitude_shear_2d" - wind direction rotates with altitude (3D)
                           N→E→S→W, repeating `wind_layers` times over
                           the full altitude range (default 2)

Extend `_build_grid()` to add more patterns.

The grid is deliberately **anisotropic**: `cells` governs x and y, while
`cells_z` governs the vertical axis and defaults to at least Nyquist for the
observation's wind column (see :func:`default_cells_z`).  A cubic grid coarse
enough to be cheap horizontally cannot resolve two adjacent 250 m column
levels, which is where the agent's only above/below signal lives.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Tuple
import numpy as np

from environments.core.constants import WIND_COL_LEVELS, WIND_COL_SPACING

try:
    from environments.core.jit_kernels import (
        wind_sample_idx_numba, wind_sample_column_numba,
    )
    _JIT_OK = True
except Exception:
    _JIT_OK = False


#: Upper bound on the automatic vertical cell count.  A pathological `z_range`
#: would otherwise size the grid into the hundreds of megabytes per env, and
#: these are constructed once per parallel worker.
CELLS_Z_MAX = 2048


def default_cells_z(z_range: Tuple[float, float],
                    spacing: float = WIND_COL_SPACING,
                    max_cells: int = CELLS_Z_MAX) -> int:
    """Vertical cell count that resolves a column sampled every ``spacing``.

    Nyquist: a column at 250 m spacing needs cells of at most 125 m, otherwise
    adjacent levels land in the same cell and read as identical wind.  Computed
    from the range rather than hardcoded so it stays correct if ``ALT_MAX`` or
    ``WIND_COL_SPACING`` move, and clamped to ``max_cells`` so an absurd
    ``z_range`` cannot blow up the grid.
    """
    span = abs(z_range[1] - z_range[0])
    if span <= 0.0 or spacing <= 0.0:
        return 1
    n = int(math.ceil(span / (0.5 * spacing)))
    return max(1, min(n, max_cells))


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
        wind_layers: int = 2,
        cells_z: int | None = None,
    ):
        self.x_range, self.y_range, self.z_range = x_range, y_range, z_range
        self.cells = cells
        #: Vertical resolution is decoupled from horizontal: the wind column is
        #: the agent's main new signal and it is sampled every 250 m, so the
        #: grid must be fine enough to give adjacent levels distinct values.
        self.cells_z = int(cells_z) if cells_z is not None else default_cells_z(z_range)
        self.pattern = pattern
        self.wind_layers = wind_layers

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
        self.z_edges = np.linspace(z_range[0], z_range[1], self.cells_z + 1)
        self.x_centers = (self.x_edges[:-1] + self.x_edges[1:]) / 2
        self.y_centers = (self.y_edges[:-1] + self.y_edges[1:]) / 2
        self.z_centers = (self.z_edges[:-1] + self.z_edges[1:]) / 2

        self._build_grid()  # fills self._fx_grid, self._fy_grid
        self._sample_buf = np.zeros(3, dtype=np.float32)  # reusable return buffer
        # Reusable (levels, 2) buffer for sample_column — sized on first use and
        # only reallocated if the caller ever asks for a different level count.
        self._column_buf = np.zeros((WIND_COL_LEVELS, 2), dtype=np.float64)

        self.dx = (x_range[1] - x_range[0]) / self.cells
        self.dy = (y_range[1] - y_range[0]) / self.cells
        self.dz = (z_range[1] - z_range[0]) / self.cells_z
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
                                             self.cells, self.cells_z,
                                             self._fx_grid, self._fy_grid)
        else:
            # Fallback must mirror the kernel's arithmetic exactly — the older
            # `searchsorted` version disagreed with it by one cell on points
            # sitting exactly on a grid edge, which is where regular test
            # coordinates land.
            ix = self._to_idx(xi, self.x_range[0], self.inv_dx, self.cells)
            iy = self._to_idx(yi, self.y_range[0], self.inv_dy, self.cells)
            iz = self._to_idx(zi, self.z_range[0], self.inv_dz, self.cells_z)
            fx = self._fx_grid[ix, iy, iz]
            fy = self._fy_grid[ix, iy, iz]

        self._sample_buf[0] = fx
        self._sample_buf[1] = fy
        # _sample_buf[2] stays 0.0 from init
        return self._sample_buf

    def sample_column(self, x: float, y: float, z_center: float,
                      levels: int = WIND_COL_LEVELS,
                      spacing: float = WIND_COL_SPACING) -> np.ndarray:
        """Return shape ``(levels, 2)`` of ``(fx, fy)`` on a vertical column.

        Level ``i`` (``i = 0 .. levels-1``) is sampled at
        ``z_center + (i - levels // 2) * spacing``, ordered low to high, so an
        odd ``levels`` puts the balloon's own altitude at index ``levels // 2``.

        Sampling coordinates are clamped into the grid range exactly as
        :meth:`sample` does — for altitude-only patterns that is horizontally
        exact, which is what makes the soft horizontal bounds free.

        The returned array is a **reused internal buffer**: this is called once
        per decision step, so it must not allocate.  Copy it if you need to
        keep the values across calls.
        """
        buf = self._column_buf
        if buf.shape[0] != levels:
            buf = np.zeros((levels, 2), dtype=np.float64)
            self._column_buf = buf

        if _JIT_OK:
            wind_sample_column_numba(
                x, y, z_center, spacing,
                self.x_range[0], self.x_range[1], self.inv_dx,
                self.y_range[0], self.y_range[1], self.inv_dy,
                self.z_range[0], self.z_range[1], self.inv_dz,
                self.cells, self.cells_z, self._fx_grid, self._fy_grid, buf,
            )
            return buf

        # --- non-JIT fallback (must stay numerically identical) ---
        half = levels // 2
        xi = min(max(x, self.x_range[0]), self.x_range[1])
        yi = min(max(y, self.y_range[0]), self.y_range[1])
        ix = self._to_idx(xi, self.x_range[0], self.inv_dx, self.cells)
        iy = self._to_idx(yi, self.y_range[0], self.inv_dy, self.cells)
        for i in range(levels):
            zi = z_center + (i - half) * spacing
            zi = min(max(zi, self.z_range[0]), self.z_range[1])
            iz = self._to_idx(zi, self.z_range[0], self.inv_dz, self.cells_z)
            buf[i, 0] = self._fx_grid[ix, iy, iz]
            buf[i, 1] = self._fy_grid[ix, iy, iz]
        return buf

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

        elif self.pattern == "altitude_shear_2d":
            # Wind direction rotates smoothly with altitude,
            # repeating `wind_layers` full rotations over the altitude range.
            Zn = (Z - zr[0]) / (zr[1] - zr[0])
            theta = 2.0 * np.pi * self.wind_layers * Zn
            self._fx_grid = mag * np.sin(theta)
            self._fy_grid = mag * np.cos(theta)

        else:  # "sinusoid" default
            self._fx_grid = (mag * 0.5 * (np.sin(2 * np.pi * X / (xr[1] - xr[0]))
                             + 0.5 * np.sin(4 * np.pi * X / (xr[1] - xr[0]))))
            self._fy_grid = (mag * 0.5 * (np.cos(2 * np.pi * Y / (yr[1] - yr[0]))
                             + 0.5 * np.cos(4 * np.pi * Y / (yr[1] - yr[0]))))
            # gentle altitude shear
            self._fx_grid += (mag / 4) * np.sin(2 * np.pi * Z / (zr[1] - zr[0]))
