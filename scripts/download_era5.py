#!/usr/bin/env python3
"""Download a bounded ERA5 pressure-level input file for preprocessing.

This script is intentionally not imported by the runtime environment.  It
requires the optional ``weather`` dependencies and a CDS account whose terms
have been accepted.  Start with a one-day spike before requesting months or
years of data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PRESSURE_LEVELS = [
    "300",
    "250",
    "225",
    "200",
    "175",
    "150",
    "125",
    "100",
    "70",
    "50",
    "30",
    "20",
    "10",
]
VARIABLES = [
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
]
HOURS = [f"{hour:02d}:00" for hour in range(24)]


def request_payload(args: argparse.Namespace) -> dict:
    # North, west, south, east. Roughly 650 km around London; this is wider
    # than the station radius because recovery trajectories may travel far out.
    half_span = float(args.half_span_degrees)
    area = [
        args.latitude + half_span,
        args.longitude - half_span,
        args.latitude - half_span,
        args.longitude + half_span,
    ]
    return {
        "product_type": ["reanalysis"],
        "variable": VARIABLES,
        "pressure_level": PRESSURE_LEVELS,
        "year": [str(args.year)],
        "month": [f"{args.month:02d}"],
        "day": [f"{day:02d}" for day in args.days],
        "time": HOURS,
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": area,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, choices=(3, 4), required=True)
    parser.add_argument("--days", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--latitude", type=float, default=51.5074)
    parser.add_argument("--longitude", type=float, default=-0.1278)
    parser.add_argument("--half-span-degrees", type=float, default=6.0)
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Print the CDS request without requiring credentials or downloading.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = request_payload(args)
    if args.plan:
        print(json.dumps(payload, indent=2))
        return
    try:
        import cdsapi
    except ImportError as exc:
        raise SystemExit(
            "cdsapi is not installed. Run `pip install -e '.[weather]'`."
        ) from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()
    client.retrieve("reanalysis-era5-pressure-levels", payload, str(args.output))


if __name__ == "__main__":
    main()
