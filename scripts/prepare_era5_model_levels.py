#!/usr/bin/env python3
"""Convert ERA5 model-level GRIB inputs into a Stratus weather cube.

ERA5's 137 model levels are hybrid pressure/sigma surfaces, not fixed
altitudes.  Pressure is reconstructed from the GRIB ``pv`` coefficients and
surface pressure.  Geopotential is then integrated upward from the surface
using temperature and specific humidity, following ECMWF's documented
procedure, before all fields are interpolated onto geometric altitude.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from environments.core.constants import G, M_AIR, R  # noqa: E402
from environments.core.weather import WeatherCube  # noqa: E402

EARTH_RADIUS_M = 6_371_000.0
R_DRY_AIR = 287.06
VIRTUAL_TEMPERATURE_Q_FACTOR = 0.609133
EXPECTED_MODEL_LEVELS = np.arange(1, 138, dtype=np.int64)

MODEL_ALIASES = {
    "u": ("u", "u_component_of_wind"),
    "v": ("v", "v_component_of_wind"),
    "temperature": ("t", "temperature"),
    "specific_humidity": ("q", "specific_humidity"),
    "omega": ("w", "vertical_velocity"),
}
SURFACE_ALIASES = {
    "geopotential": ("z", "geopotential"),
    "lnsp": ("lnsp", "logarithm_of_surface_pressure"),
}


def _first_present(dataset, names: tuple[str, ...]) -> str:
    for name in names:
        if name in dataset:
            return name
    raise ValueError(f"none of {names} are present in {list(dataset.data_vars)}")


def _dimension_name(dataset, names: tuple[str, ...]) -> str:
    for name in names:
        if name in dataset.dims:
            return name
    raise ValueError(f"none of dimension names {names} are present in {dataset.dims}")


def _field_values(dataset, name: str, dimensions: tuple[str, ...]) -> np.ndarray:
    field = dataset[name]
    for dimension in tuple(field.dims):
        if dimension in dimensions:
            continue
        if field.sizes[dimension] != 1:
            raise ValueError(
                f"{name} has unexpected non-singleton dimension {dimension!r}"
            )
        field = field.isel({dimension: 0}, drop=True)
    missing = [dimension for dimension in dimensions if dimension not in field.dims]
    if missing:
        raise ValueError(f"{name} is missing dimensions {missing}")
    return np.asarray(field.transpose(*dimensions).values, dtype=np.float64)


def _local_xy(latitude: np.ndarray, longitude: np.ndarray, lat0: float, lon0: float):
    lat_rad = np.deg2rad(latitude)
    lon_delta = (longitude - lon0 + 180.0) % 360.0 - 180.0
    y = EARTH_RADIUS_M * (lat_rad - np.deg2rad(lat0))
    x = EARTH_RADIUS_M * np.cos(np.deg2rad(lat0)) * np.deg2rad(lon_delta)
    return y, x


def read_hybrid_coefficients(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the 138 ERA5 half-level A/B coefficients from a GRIB header."""

    try:
        from eccodes import (
            codes_get_array,
            codes_grib_new_from_file,
            codes_is_defined,
            codes_release,
        )
    except ImportError as exc:
        raise SystemExit(
            "ecCodes support is not installed. Run `pip install -e '.[weather]'`."
        ) from exc

    coefficients = None
    with Path(path).open("rb") as handle:
        while True:
            message = codes_grib_new_from_file(handle)
            if message is None:
                break
            try:
                if codes_is_defined(message, "pv"):
                    candidate = np.asarray(
                        codes_get_array(message, "pv"), dtype=np.float64
                    )
                    if candidate.size == 2 * (EXPECTED_MODEL_LEVELS.size + 1):
                        coefficients = candidate
                        break
            finally:
                codes_release(message)
    if coefficients is None:
        raise ValueError(f"{path} has no ERA5 L137 pv coefficients")
    midpoint = coefficients.size // 2
    return coefficients[:midpoint], coefficients[midpoint:]


def hybrid_pressures(
    ln_surface_pressure: np.ndarray,
    a_coefficients: np.ndarray,
    b_coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return half- and full-level pressures in Pa."""

    lnsp = np.asarray(ln_surface_pressure, dtype=np.float64)
    a = np.asarray(a_coefficients, dtype=np.float64)
    b = np.asarray(b_coefficients, dtype=np.float64)
    if a.ndim != 1 or b.shape != a.shape or a.size < 2:
        raise ValueError("hybrid A/B coefficients must be equal one-dimensional arrays")
    surface_pressure = np.exp(lnsp)
    expand = (1, a.size) + (1,) * (lnsp.ndim - 1)
    a_grid = a.reshape(expand)
    b_grid = b.reshape(expand)
    pressure_half = a_grid + b_grid * np.expand_dims(surface_pressure, axis=1)
    pressure_full = 0.5 * (pressure_half[:, :-1] + pressure_half[:, 1:])
    return pressure_half, pressure_full


def model_level_geopotential(
    temperature: np.ndarray,
    specific_humidity: np.ndarray,
    pressure_half: np.ndarray,
    surface_geopotential: np.ndarray,
) -> np.ndarray:
    """Integrate geopotential upward using ECMWF's L137 hydrostatic method."""

    temperature = np.asarray(temperature, dtype=np.float64)
    humidity = np.asarray(specific_humidity, dtype=np.float64)
    pressure_half = np.asarray(pressure_half, dtype=np.float64)
    surface = np.asarray(surface_geopotential, dtype=np.float64)
    if humidity.shape != temperature.shape:
        raise ValueError("temperature and specific humidity shapes differ")
    if pressure_half.shape[1] != temperature.shape[1] + 1:
        raise ValueError("half-level pressure count must be model-level count + 1")
    if pressure_half.shape[:1] + pressure_half.shape[2:] != surface.shape:
        raise ValueError("surface geopotential shape does not match model fields")

    geopotential = np.empty_like(temperature, dtype=np.float64)
    half_level_geopotential = surface.copy()
    for index in range(temperature.shape[1] - 1, -1, -1):
        pressure_top = pressure_half[:, index]
        pressure_bottom = pressure_half[:, index + 1]
        virtual_temperature = temperature[:, index] * (
            1.0 + VIRTUAL_TEMPERATURE_Q_FACTOR * humidity[:, index]
        )
        if index == 0:
            delta_log_pressure = np.log(pressure_bottom / 0.1)
            alpha = np.log(2.0)
        else:
            delta_log_pressure = np.log(pressure_bottom / pressure_top)
            alpha = 1.0 - (
                pressure_top / (pressure_bottom - pressure_top) * delta_log_pressure
            )
        virtual_temperature_r = R_DRY_AIR * virtual_temperature
        geopotential[:, index] = half_level_geopotential + virtual_temperature_r * alpha
        half_level_geopotential += virtual_temperature_r * delta_log_pressure
    return geopotential


def geometric_height(geopotential: np.ndarray) -> np.ndarray:
    """Convert geopotential to geometric altitude above mean sea level."""

    geopotential_height = np.asarray(geopotential, dtype=np.float64) / G
    return EARTH_RADIUS_M * geopotential_height / (EARTH_RADIUS_M - geopotential_height)


def _open_grib(path: Path):
    try:
        import xarray as xr
    except ImportError as exc:
        raise SystemExit(
            "xarray is not installed. Run `pip install -e '.[weather]'`."
        ) from exc
    try:
        return xr.open_dataset(
            path,
            engine="cfgrib",
            backend_kwargs={"indexpath": ""},
        )
    except ValueError as exc:
        if "unrecognized engine" in str(exc) or "cfgrib" in str(exc):
            raise SystemExit(
                "cfgrib is not installed. Run `pip install -e '.[weather]'`."
            ) from exc
        raise


def convert(args: argparse.Namespace) -> WeatherCube:
    model = _open_grib(args.model_input)
    surface = _open_grib(args.surface_input)

    model_time = _dimension_name(model, ("time", "valid_time"))
    surface_time = _dimension_name(surface, ("time", "valid_time"))
    level_name = _dimension_name(model, ("hybrid", "model_level", "level"))
    model_lat = _dimension_name(model, ("latitude", "lat"))
    model_lon = _dimension_name(model, ("longitude", "lon"))
    surface_lat = _dimension_name(surface, ("latitude", "lat"))
    surface_lon = _dimension_name(surface, ("longitude", "lon"))

    # Noon corpus batches concatenate 12:00--23:00 messages for several dates
    # with their following 00:00 boundary messages. GRIB message order is then
    # not chronological even though the timestamps are valid; sort coordinates
    # before selecting an individual 12-hour window.
    model = model.sortby(model_time)
    surface = surface.sortby(surface_time)

    if args.start is not None:
        start = np.datetime64(args.start)
        end = start + np.timedelta64(int(args.duration_hours * 3_600), "s")
        model = model.sel({model_time: slice(start, end)})
        surface = surface.sel({surface_time: slice(start, end)})
    if model.sizes.get(model_time, 0) < 2:
        raise ValueError("the selected interval needs at least two ERA5 time samples")

    times = np.asarray(model[model_time].values).astype("datetime64[s]")
    surface_times = np.asarray(surface[surface_time].values).astype("datetime64[s]")
    if not np.array_equal(times, surface_times):
        raise ValueError("model and surface GRIB files have different time axes")

    latitudes = np.asarray(model[model_lat].values, dtype=np.float64)
    longitudes = np.asarray(model[model_lon].values, dtype=np.float64)
    if not np.array_equal(
        latitudes, np.asarray(surface[surface_lat].values)
    ) or not np.array_equal(longitudes, np.asarray(surface[surface_lon].values)):
        raise ValueError("model and surface GRIB files have different spatial grids")
    y_m, x_m = _local_xy(
        latitudes, longitudes, float(args.latitude), float(args.longitude)
    )
    y_order = np.argsort(y_m)
    x_order = np.argsort(x_m)
    y_m = y_m[y_order]
    x_m = x_m[x_order]

    levels = np.asarray(model[level_name].values, dtype=np.int64)
    level_order = np.argsort(levels)
    levels = levels[level_order]
    if not np.array_equal(levels, EXPECTED_MODEL_LEVELS):
        raise ValueError(
            "model-level conversion requires every ERA5 level 1-137; "
            f"received {levels.tolist()}"
        )

    model_dimensions = (model_time, level_name, model_lat, model_lon)
    model_fields = {}
    for target, aliases in MODEL_ALIASES.items():
        values = _field_values(model, _first_present(model, aliases), model_dimensions)
        model_fields[target] = values[:, level_order][:, :, y_order][:, :, :, x_order]

    surface_dimensions = (surface_time, surface_lat, surface_lon)
    surface_fields = {}
    for target, aliases in SURFACE_ALIASES.items():
        values = _field_values(
            surface, _first_present(surface, aliases), surface_dimensions
        )
        surface_fields[target] = values[:, y_order][:, :, x_order]

    a_coefficients, b_coefficients = read_hybrid_coefficients(args.model_input)
    pressure_half, pressure_full = hybrid_pressures(
        surface_fields["lnsp"], a_coefficients, b_coefficients
    )
    geopotential = model_level_geopotential(
        model_fields["temperature"],
        model_fields["specific_humidity"],
        pressure_half,
        surface_fields["geopotential"],
    )
    height = geometric_height(geopotential)

    time_s = (times - times[0]).astype("timedelta64[s]").astype(np.float64)
    altitude_m = np.arange(
        float(args.altitude_min),
        float(args.altitude_max) + 0.5 * float(args.altitude_step),
        float(args.altitude_step),
        dtype=np.float64,
    )
    shape = (time_s.size, altitude_m.size, y_m.size, x_m.size)
    outputs = {
        name: np.empty(shape, dtype=np.float32)
        for name in ("u", "v", "temperature", "pressure", "w")
    }

    for it in range(time_s.size):
        for iy in range(y_m.size):
            for ix in range(x_m.size):
                heights = height[it, :, iy, ix]
                order = np.argsort(heights)
                heights = heights[order]
                if altitude_m[0] < heights[0] or altitude_m[-1] > heights[-1]:
                    raise ValueError(
                        "requested altitude grid is outside the reconstructed model "
                        f"levels at t/y/x={it}/{iy}/{ix}: "
                        f"[{heights[0]:.0f}, {heights[-1]:.0f}] m"
                    )
                for name in ("u", "v", "temperature"):
                    profile = model_fields[name][it, :, iy, ix][order]
                    outputs[name][it, :, iy, ix] = np.interp(
                        altitude_m, heights, profile
                    )
                pressure = np.exp(
                    np.interp(
                        altitude_m,
                        heights,
                        np.log(pressure_full[it, :, iy, ix][order]),
                    )
                )
                outputs["pressure"][it, :, iy, ix] = pressure
                omega = np.interp(
                    altitude_m,
                    heights,
                    model_fields["omega"][it, :, iy, ix][order],
                )
                density = pressure * M_AIR / (R * outputs["temperature"][it, :, iy, ix])
                outputs["w"][it, :, iy, ix] = -omega / (density * G)

    start_time = np.datetime_as_string(times[0], unit="s") + "Z"
    return WeatherCube(
        time_s=time_s,
        altitude_m=altitude_m,
        y_m=y_m,
        x_m=x_m,
        metadata={
            "source": "ERA5 model levels",
            "start_time_utc": start_time,
            "site_name": args.site_name,
            "site_latitude": float(args.latitude),
            "site_longitude": float(args.longitude),
            "model_source_file": args.model_input.name,
            "surface_source_file": args.surface_input.name,
            "vertical_interpolation": (
                "ECMWF L137 hydrostatic geopotential; geometric height"
            ),
        },
        **outputs,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-input", type=Path, required=True)
    parser.add_argument("--surface-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", help="UTC ISO timestamp, e.g. 2021-04-15T00:00")
    parser.add_argument("--duration-hours", type=float, default=12.0)
    parser.add_argument("--site-name", default="London")
    parser.add_argument("--latitude", type=float, default=51.5074)
    parser.add_argument("--longitude", type=float, default=-0.1278)
    parser.add_argument("--altitude-min", type=float, default=10_000.0)
    parser.add_argument("--altitude-max", type=float, default=30_000.0)
    parser.add_argument("--altitude-step", type=float, default=250.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cube = convert(args)
    cube.save(args.output)
    print(
        f"wrote {args.output}: shape={cube.u.shape}, {cube.metadata['start_time_utc']}"
    )


if __name__ == "__main__":
    main()
