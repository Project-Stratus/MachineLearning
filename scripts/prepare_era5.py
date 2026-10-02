#!/usr/bin/env python3
"""Convert an ERA5 pressure-level NetCDF file to a Stratus weather cube.

ERA5 pressure surfaces are not fixed geometric altitudes.  This conversion
uses geopotential height at every time/horizontal grid point, interpolates
wind and temperature onto a regular geometric-altitude grid, and interpolates
pressure in log space.  The output has no xarray/NetCDF dependency and can be
loaded cheaply by parallel training workers.
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
ALIASES = {
    "u": ("u", "u_component_of_wind"),
    "v": ("v", "v_component_of_wind"),
    "temperature": ("t", "temperature"),
    "geopotential": ("z", "geopotential"),
    "omega": ("w", "vertical_velocity"),
}


def _first_present(dataset, names: tuple[str, ...]) -> str:
    for name in names:
        if name in dataset:
            return name
    raise ValueError(f"none of {names} are present in {list(dataset.data_vars)}")


def _coord_name(dataset, names: tuple[str, ...]) -> str:
    for name in names:
        if name in dataset.coords:
            return name
    raise ValueError(f"none of coordinate names {names} are present")


def _local_xy(latitude: np.ndarray, longitude: np.ndarray, lat0: float, lon0: float):
    lat_rad = np.deg2rad(latitude)
    lon_delta = (longitude - lon0 + 180.0) % 360.0 - 180.0
    y = EARTH_RADIUS_M * (lat_rad - np.deg2rad(lat0))
    x = EARTH_RADIUS_M * np.cos(np.deg2rad(lat0)) * np.deg2rad(lon_delta)
    return y, x


def convert(args: argparse.Namespace) -> WeatherCube:
    try:
        import xarray as xr
    except ImportError as exc:
        raise SystemExit(
            "xarray/NetCDF support is not installed. Run `pip install -e '.[weather]'`."
        ) from exc

    dataset = xr.open_dataset(args.input)
    time_name = _coord_name(dataset, ("valid_time", "time"))
    level_name = _coord_name(dataset, ("pressure_level", "level", "isobaricInhPa"))
    lat_name = _coord_name(dataset, ("latitude", "lat"))
    lon_name = _coord_name(dataset, ("longitude", "lon"))

    if args.start is not None:
        start = np.datetime64(args.start)
        end = start + np.timedelta64(int(args.duration_hours * 3600), "s")
        dataset = dataset.sel({time_name: slice(start, end)})
    if dataset.sizes.get(time_name, 0) < 2:
        raise ValueError("the selected interval needs at least two ERA5 time samples")

    latitudes = np.asarray(dataset[lat_name].values, dtype=np.float64)
    longitudes = np.asarray(dataset[lon_name].values, dtype=np.float64)
    y_m, x_m = _local_xy(
        latitudes, longitudes, float(args.latitude), float(args.longitude)
    )
    y_order = np.argsort(y_m)
    x_order = np.argsort(x_m)
    y_m = y_m[y_order]
    x_m = x_m[x_order]

    levels_hpa = np.asarray(dataset[level_name].values, dtype=np.float64)
    field_values: dict[str, np.ndarray] = {}
    for target, aliases in ALIASES.items():
        name = _first_present(dataset, aliases)
        field = (
            dataset[name]
            .squeeze(drop=True)
            .transpose(time_name, level_name, lat_name, lon_name)
        )
        values = np.asarray(field.values, dtype=np.float64)
        field_values[target] = values[:, :, y_order, :][:, :, :, x_order]

    times = np.asarray(dataset[time_name].values).astype("datetime64[s]")
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
    pressure_levels_pa = levels_hpa * 100.0

    for it in range(time_s.size):
        for iy in range(y_m.size):
            for ix in range(x_m.size):
                heights = field_values["geopotential"][it, :, iy, ix] / G
                order = np.argsort(heights)
                heights = heights[order]
                if altitude_m[0] < heights[0] or altitude_m[-1] > heights[-1]:
                    raise ValueError(
                        "requested altitude grid is outside the downloaded pressure "
                        f"surfaces at t/y/x={it}/{iy}/{ix}: "
                        f"[{heights[0]:.0f}, {heights[-1]:.0f}] m"
                    )
                for name in ("u", "v", "temperature"):
                    profile = field_values[name][it, :, iy, ix][order]
                    outputs[name][it, :, iy, ix] = np.interp(
                        altitude_m, heights, profile
                    )
                log_pressure = np.log(pressure_levels_pa[order])
                pressure = np.exp(np.interp(altitude_m, heights, log_pressure))
                outputs["pressure"][it, :, iy, ix] = pressure

                omega = np.interp(
                    altitude_m,
                    heights,
                    field_values["omega"][it, :, iy, ix][order],
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
            "source": "ERA5 pressure levels",
            "start_time_utc": start_time,
            "site_name": args.site_name,
            "site_latitude": float(args.latitude),
            "site_longitude": float(args.longitude),
            "source_file": args.input.name,
            "vertical_interpolation": "geopotential height; pressure log-linear",
        },
        **outputs,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", help="UTC ISO timestamp, e.g. 2024-03-15T00:00")
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
