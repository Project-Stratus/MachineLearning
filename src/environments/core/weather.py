"""Weather data backends for deterministic Layer 2 simulations.

The runtime deliberately consumes a compact, dependency-free ``.npz`` cube.
Downloading and decoding ERA5 is an offline concern (see
``scripts/download_era5.py`` and ``scripts/prepare_era5.py``); training workers
must never contact CDS or open GRIB files in their step loop.

All gridded fields use the axis order ``(time, altitude, y, x)``.  Coordinates
are seconds from the cube epoch and metres in a local east/north/up frame.  A
scenario chooses an offset into the time coordinate, after which
``set_time(elapsed_seconds)`` advances the deterministic weather clock.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

from environments.core.constants import (
    M_AIR,
    R,
    SUPERHEAT_DAY,
    WIND_COL_LEVELS,
    WIND_COL_SPACING,
)

BoundsPolicy = Literal["clip", "raise"]
_REQUIRED_FIELDS = ("u", "v", "temperature", "pressure")
_OPTIONAL_FIELDS = ("w", "radiative_forcing")

try:
    from numba import njit
except Exception:  # pragma: no cover - numba is a project dependency

    def njit(**_kwargs):
        def decorate(function):
            return function

        return decorate


@njit(cache=True, fastmath=True)
def _bracket_numba(axis: np.ndarray, value: float) -> tuple[int, int, float]:
    if axis.size == 1:
        return 0, 0, 0.0
    clipped = min(max(value, axis[0]), axis[-1])
    hi = int(np.searchsorted(axis, clipped, side="right"))
    hi = min(max(hi, 1), axis.size - 1)
    lo = hi - 1
    fraction = (clipped - axis[lo]) / (axis[hi] - axis[lo])
    return lo, hi, min(max(fraction, 0.0), 1.0)


@njit(cache=True, fastmath=True)
def _interpolate_field_numba(
    field: np.ndarray,
    t0: int,
    t1: int,
    ft: float,
    z0: int,
    z1: int,
    fz: float,
    y0: int,
    y1: int,
    fy: float,
    x0: int,
    x1: int,
    fx: float,
) -> float:
    value = 0.0
    for mask in range(16):
        use_t1 = mask & 1
        use_z1 = mask & 2
        use_y1 = mask & 4
        use_x1 = mask & 8
        it = t1 if use_t1 else t0
        iz = z1 if use_z1 else z0
        iy = y1 if use_y1 else y0
        ix = x1 if use_x1 else x0
        weight = (ft if use_t1 else 1.0 - ft) * (fz if use_z1 else 1.0 - fz)
        weight *= (fy if use_y1 else 1.0 - fy) * (fx if use_x1 else 1.0 - fx)
        value += weight * field[it, iz, iy, ix]
    return value


@njit(cache=True, fastmath=True)
def _sample_fields_numba(
    time_s: np.ndarray,
    altitude_m: np.ndarray,
    y_m: np.ndarray,
    x_m: np.ndarray,
    source_time: float,
    z: float,
    y: float,
    x: float,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    temperature: np.ndarray,
    pressure: np.ndarray,
    radiation: np.ndarray,
    have_w: bool,
    have_radiation: bool,
) -> tuple[float, float, float, float, float, float]:
    t0, t1, ft = _bracket_numba(time_s, source_time)
    z0, z1, fz = _bracket_numba(altitude_m, z)
    y0, y1, fy = _bracket_numba(y_m, y)
    x0, x1, fx = _bracket_numba(x_m, x)

    values = np.empty(6, dtype=np.float64)
    values[0] = _interpolate_field_numba(
        u, t0, t1, ft, z0, z1, fz, y0, y1, fy, x0, x1, fx
    )
    values[1] = _interpolate_field_numba(
        v, t0, t1, ft, z0, z1, fz, y0, y1, fy, x0, x1, fx
    )
    values[2] = (
        _interpolate_field_numba(w, t0, t1, ft, z0, z1, fz, y0, y1, fy, x0, x1, fx)
        if have_w
        else 0.0
    )
    values[3] = _interpolate_field_numba(
        temperature, t0, t1, ft, z0, z1, fz, y0, y1, fy, x0, x1, fx
    )
    values[4] = _interpolate_field_numba(
        pressure, t0, t1, ft, z0, z1, fz, y0, y1, fy, x0, x1, fx
    )
    values[5] = (
        _interpolate_field_numba(
            radiation, t0, t1, ft, z0, z1, fz, y0, y1, fy, x0, x1, fx
        )
        if have_radiation
        else 0.0
    )
    return values[0], values[1], values[2], values[3], values[4], values[5]


@njit(cache=True, fastmath=True)
def _sample_column_numba(
    time_s: np.ndarray,
    altitude_m: np.ndarray,
    y_m: np.ndarray,
    x_m: np.ndarray,
    source_time: float,
    x: float,
    y: float,
    z_center: float,
    spacing: float,
    u: np.ndarray,
    v: np.ndarray,
    out: np.ndarray,
) -> None:
    t0, t1, ft = _bracket_numba(time_s, source_time)
    y0, y1, fy = _bracket_numba(y_m, y)
    x0, x1, fx = _bracket_numba(x_m, x)
    half = out.shape[0] // 2
    for i in range(out.shape[0]):
        z = z_center + (i - half) * spacing
        z0, z1, fz = _bracket_numba(altitude_m, z)
        out[i, 0] = _interpolate_field_numba(
            u, t0, t1, ft, z0, z1, fz, y0, y1, fy, x0, x1, fx
        )
        out[i, 1] = _interpolate_field_numba(
            v, t0, t1, ft, z0, z1, fz, y0, y1, fy, x0, x1, fx
        )


@dataclass(frozen=True)
class WeatherSample:
    """One interpolated atmosphere sample in SI units."""

    u: float
    v: float
    w: float
    temperature: float
    pressure: float
    density: float
    radiative_forcing: float = 0.0


@dataclass
class WeatherCube:
    """A validated local weather cube suitable for fast deterministic replay."""

    time_s: np.ndarray
    altitude_m: np.ndarray
    y_m: np.ndarray
    x_m: np.ndarray
    u: np.ndarray
    v: np.ndarray
    temperature: np.ndarray
    pressure: np.ndarray
    w: np.ndarray | None = None
    radiative_forcing: np.ndarray | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        for name in ("time_s", "altitude_m", "y_m", "x_m"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.ndim != 1 or value.size == 0:
                raise ValueError(f"{name} must be a non-empty one-dimensional array")
            if value.size > 1 and np.any(np.diff(value) <= 0.0):
                raise ValueError(f"{name} must be strictly increasing")
            setattr(self, name, np.ascontiguousarray(value))

        shape = (
            self.time_s.size,
            self.altitude_m.size,
            self.y_m.size,
            self.x_m.size,
        )
        for name in _REQUIRED_FIELDS:
            value = np.asarray(getattr(self, name), dtype=np.float32)
            self._validate_field(name, value, shape)
            setattr(self, name, np.ascontiguousarray(value))

        for name in _OPTIONAL_FIELDS:
            value = getattr(self, name)
            if value is None:
                continue
            value = np.asarray(value, dtype=np.float32)
            self._validate_field(name, value, shape)
            setattr(self, name, np.ascontiguousarray(value))

        if np.any(self.temperature <= 0.0):
            raise ValueError("temperature must be strictly positive Kelvin")
        if np.any(self.pressure <= 0.0):
            raise ValueError("pressure must be strictly positive Pa")
        self.metadata = dict(self.metadata or {})

    @staticmethod
    def _validate_field(name: str, value: np.ndarray, shape: tuple[int, ...]) -> None:
        if value.shape != shape:
            raise ValueError(f"{name} has shape {value.shape}; expected {shape}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} contains non-finite values")

    def save(self, path: str | Path) -> None:
        """Write the portable runtime format without pickle/object arrays."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "time_s": self.time_s,
            "altitude_m": self.altitude_m,
            "y_m": self.y_m,
            "x_m": self.x_m,
            "u": self.u,
            "v": self.v,
            "temperature": self.temperature,
            "pressure": self.pressure,
            "metadata_json": np.asarray(json.dumps(self.metadata, sort_keys=True)),
        }
        if self.w is not None:
            arrays["w"] = self.w
        if self.radiative_forcing is not None:
            arrays["radiative_forcing"] = self.radiative_forcing
        np.savez_compressed(destination, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "WeatherCube":
        source = Path(path)
        with np.load(source, allow_pickle=False) as data:
            missing = [
                name
                for name in ("time_s", "altitude_m", "y_m", "x_m", *_REQUIRED_FIELDS)
                if name not in data
            ]
            if missing:
                raise ValueError(f"weather cube {source} is missing {missing}")
            metadata: dict[str, Any] = {}
            if "metadata_json" in data:
                metadata = json.loads(str(data["metadata_json"].item()))
            kwargs = {name: np.array(data[name]) for name in _REQUIRED_FIELDS}
            for name in _OPTIONAL_FIELDS:
                kwargs[name] = np.array(data[name]) if name in data else None
            return cls(
                time_s=np.array(data["time_s"]),
                altitude_m=np.array(data["altitude_m"]),
                y_m=np.array(data["y_m"]),
                x_m=np.array(data["x_m"]),
                metadata=metadata,
                **kwargs,
            )


def _bracket(
    axis: np.ndarray, value: float, policy: BoundsPolicy
) -> tuple[int, int, float]:
    """Return lower/upper indices and interpolation fraction for one axis."""

    if axis.size == 1:
        if policy == "raise" and value != axis[0]:
            raise ValueError(
                f"coordinate {value} lies outside singleton axis {axis[0]}"
            )
        return 0, 0, 0.0

    if value < axis[0] or value > axis[-1]:
        if policy == "raise":
            raise ValueError(
                f"coordinate {value} lies outside weather domain [{axis[0]}, {axis[-1]}]"
            )
        value = min(max(value, float(axis[0])), float(axis[-1]))

    hi = int(np.searchsorted(axis, value, side="right"))
    hi = min(max(hi, 1), axis.size - 1)
    lo = hi - 1
    span = float(axis[hi] - axis[lo])
    frac = 0.0 if span == 0.0 else (float(value) - float(axis[lo])) / span
    return lo, hi, min(max(frac, 0.0), 1.0)


class ReanalysisWeatherProvider:
    """Interpolate a :class:`WeatherCube` as deterministic weather truth.

    The provider intentionally mirrors the small portion of :class:`WindField`
    used by the environment (``sample`` and ``sample_column``).  This keeps the
    observation, renderer and baselines independent of where weather came from.
    """

    is_reanalysis = True

    def __init__(
        self,
        cube: WeatherCube | str | Path,
        *,
        start_offset_s: float = 0.0,
        bounds_policy: BoundsPolicy = "clip",
    ) -> None:
        self.cube = WeatherCube.load(cube) if isinstance(cube, (str, Path)) else cube
        if bounds_policy not in ("clip", "raise"):
            raise ValueError("bounds_policy must be 'clip' or 'raise'")
        self.bounds_policy = bounds_policy
        self.start_offset_s = float(start_offset_s)
        self.elapsed_s = 0.0

        self.x_centers = self.cube.x_m
        self.y_centers = self.cube.y_m
        self.z_centers = self.cube.altitude_m
        self.x_range = (float(self.cube.x_m[0]), float(self.cube.x_m[-1]))
        self.y_range = (float(self.cube.y_m[0]), float(self.cube.y_m[-1]))
        self.z_range = (
            float(self.cube.altitude_m[0]),
            float(self.cube.altitude_m[-1]),
        )
        self.cells = int(min(self.cube.x_m.size, self.cube.y_m.size))
        self.cells_z = int(self.cube.altitude_m.size)
        self.dz = (
            float(np.min(np.diff(self.cube.altitude_m)))
            if self.cube.altitude_m.size > 1
            else np.inf
        )
        self._sample_buf = np.zeros(3, dtype=np.float32)
        self._column_buf = np.empty((0, 2), dtype=np.float64)
        # Optional arrays use an existing grid as a typed placeholder when the
        # corresponding flag is false; this avoids allocating all-zero weather
        # cubes merely to satisfy a compiled function signature.
        self._w = self.cube.w if self.cube.w is not None else self.cube.u
        self._radiation = (
            self.cube.radiative_forcing
            if self.cube.radiative_forcing is not None
            else self.cube.u
        )

    @property
    def source_time_s(self) -> float:
        return self.start_offset_s + self.elapsed_s

    def set_time(self, elapsed_s: float) -> None:
        self.elapsed_s = float(elapsed_s)

    def validate_window(self, duration_s: float) -> None:
        """Fail early if an episode would outlast the source time axis.

        Spatial clipping is an explicit policy because a trajectory can leave
        the downloaded box. Time clipping is never acceptable for a scenario:
        it would silently replay the final analysis hour until the episode ends.
        """

        source_start = self.start_offset_s
        source_end = source_start + float(duration_s)
        if source_start < self.cube.time_s[0] or source_end > self.cube.time_s[-1]:
            raise ValueError(
                "weather cube does not cover the requested episode window: "
                f"[{source_start}, {source_end}] s versus "
                f"[{self.cube.time_s[0]}, {self.cube.time_s[-1]}] s"
            )

    def contains_horizontal(self, x: float, y: float) -> bool:
        """Return whether a point is inside the downloaded horizontal domain."""

        return bool(
            self.cube.x_m[0] <= x <= self.cube.x_m[-1]
            and self.cube.y_m[0] <= y <= self.cube.y_m[-1]
        )

    def _check_bounds(self, x: float, y: float, z: float) -> None:
        if self.bounds_policy != "raise":
            return
        # Reuse the Python helper for its precise, user-facing error.
        _bracket(self.cube.time_s, self.source_time_s, "raise")
        _bracket(self.cube.altitude_m, z, "raise")
        _bracket(self.cube.y_m, y, "raise")
        _bracket(self.cube.x_m, x, "raise")

    def _sample_values(self, x: float, y: float, z: float):
        self._check_bounds(x, y, z)
        return _sample_fields_numba(
            self.cube.time_s,
            self.cube.altitude_m,
            self.cube.y_m,
            self.cube.x_m,
            self.source_time_s,
            z,
            y,
            x,
            self.cube.u,
            self.cube.v,
            self._w,
            self.cube.temperature,
            self.cube.pressure,
            self._radiation,
            self.cube.w is not None,
            self.cube.radiative_forcing is not None,
        )

    def sample_weather(self, x: float, y: float, z: float) -> WeatherSample:
        u, v, w, temperature, pressure, radiative = self._sample_values(x, y, z)
        density = pressure * M_AIR / (R * temperature)
        return WeatherSample(
            u=u,
            v=v,
            w=w,
            temperature=temperature,
            pressure=pressure,
            density=density,
            radiative_forcing=radiative,
        )

    def sample(self, x: float, y: float, z: float) -> np.ndarray:
        u, v, w, _temperature, _pressure, _radiative = self._sample_values(x, y, z)
        self._sample_buf[:] = (u, v, w)
        return self._sample_buf

    def sample_column(
        self,
        x: float,
        y: float,
        z_center: float,
        levels: int = WIND_COL_LEVELS,
        spacing: float = WIND_COL_SPACING,
    ) -> np.ndarray:
        if self._column_buf.shape[0] != levels:
            self._column_buf = np.zeros((levels, 2), dtype=np.float64)
        if self.bounds_policy == "raise":
            self._check_bounds(x, y, z_center)
            _bracket(
                self.cube.altitude_m,
                z_center - (levels // 2) * spacing,
                "raise",
            )
            _bracket(
                self.cube.altitude_m,
                z_center + (levels // 2) * spacing,
                "raise",
            )
        _sample_column_numba(
            self.cube.time_s,
            self.cube.altitude_m,
            self.cube.y_m,
            self.cube.x_m,
            self.source_time_s,
            x,
            y,
            z_center,
            spacing,
            self.cube.u,
            self.cube.v,
            self._column_buf,
        )
        return self._column_buf

    def radiative_forcing(self, x: float, y: float, z: float) -> float:
        return float(self._sample_values(x, y, z)[5])


class ReanalysisAtmosphere:
    """Altitude API expected by ``Balloon``, backed by local reanalysis truth.

    ``Balloon`` historically asks its atmosphere about altitude alone.  The
    environment sets horizontal position and time once per one-second physics
    step; altitude remains the varying coordinate during integration.
    """

    use_isa_numba = False

    def __init__(self, provider: ReanalysisWeatherProvider) -> None:
        self.provider = provider
        self.p0 = 101_325.0  # compatibility only; never used for reanalysis
        self.molar_mass = M_AIR
        self._x = 0.0
        self._y = 0.0
        self._cache_key: tuple[float, float, float, float] | None = None
        self._cache_value: WeatherSample | None = None

    def set_context(self, x: float, y: float, elapsed_s: float) -> None:
        self._x = float(x)
        self._y = float(y)
        self.provider.set_time(elapsed_s)
        self._cache_key = None

    def _sample(self, altitude: float) -> WeatherSample:
        key = (self._x, self._y, float(altitude), self.provider.source_time_s)
        if key != self._cache_key:
            self._cache_value = self.provider.sample_weather(self._x, self._y, altitude)
            self._cache_key = key
        assert self._cache_value is not None
        return self._cache_value

    def temperature(self, altitude: float) -> float:
        return self._sample(altitude).temperature

    def pressure(self, altitude: float) -> float:
        return self._sample(altitude).pressure

    def density(self, altitude: float) -> float:
        return self._sample(altitude).density

    def gas_temperature(self, altitude: float) -> float:
        # Layer 2.2 retains the calibrated daytime offset. Layer 2.4 replaces
        # this with the stateful radiative model.
        return self.temperature(altitude) + SUPERHEAT_DAY

    def dynamic_viscosity(self, altitude: float) -> float:
        from environments.core.constants import MU_REF, S_SUTH, T_REF

        temperature = self.temperature(altitude)
        return (
            MU_REF
            * (temperature / T_REF) ** 1.5
            * (T_REF + S_SUTH)
            / (temperature + S_SUTH)
        )
