#!/usr/bin/env python3
"""Build a date-split, difficulty-stratified Layer 2 scenario manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from environments.core.weather_scenarios import build_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weather_cubes", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heldout-years", type=int, nargs="+", required=True)
    parser.add_argument("--site-name", default="London")
    parser.add_argument("--latitude", type=float, default=51.5074)
    parser.add_argument("--longitude", type=float, default=-0.1278)
    parser.add_argument(
        "--window-stride-hours",
        type=float,
        default=12.0,
        help="Spacing between 12-hour windows within each cube.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest(
        args.weather_cubes,
        output_path=args.output,
        heldout_years=set(args.heldout_years),
        site_name=args.site_name,
        latitude=args.latitude,
        longitude=args.longitude,
        window_stride_s=args.window_stride_hours * 3600.0,
    )
    counts = {split: len(manifest.for_split(split)) for split in ("train", "heldout")}
    print(f"wrote {args.output}: {counts}")


if __name__ == "__main__":
    main()
