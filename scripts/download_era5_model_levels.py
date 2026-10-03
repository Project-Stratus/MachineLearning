#!/usr/bin/env python3
"""Download ERA5 model-level inputs for a bounded Stratus weather cube.

The model-level product lives in ``reanalysis-era5-complete`` and uses MARS
request syntax.  Two GRIB files are required: three-dimensional model fields,
and the surface geopotential/log-pressure fields needed to reconstruct height
and pressure on the hybrid levels.  Keep requests short until the conversion
and domain size have been validated locally.
"""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path

DATASET = "reanalysis-era5-complete"
MODEL_LEVELS = "1/to/137/by/1"
MODEL_PARAMETERS = "130/131/132/133/135"  # t/u/v/q/omega
SURFACE_PARAMETERS = "129/152"  # surface geopotential/log(surface pressure)
PRESSURE_LEVELS = "10/20/30/50/70/100/125/150/175/200/225/250/300"
PRESSURE_PARAMETERS = "129/130/131/132/135"  # z/t/u/v/omega
HOURS = "00/to/23/by/1"


def _dates(args: argparse.Namespace) -> str:
    explicit_dates = getattr(args, "dates", None)
    if explicit_dates is not None:
        return "/".join(
            value.isoformat() if isinstance(value, date) else str(value)
            for value in explicit_dates
        )
    values = []
    for day in args.days:
        values.append(date(args.year, args.month, day).isoformat())
    return "/".join(values)


def request_payloads(args: argparse.Namespace) -> tuple[dict, dict]:
    """Return model-field and support-field MARS requests."""

    half_span = float(args.half_span_degrees)
    area = [
        args.latitude + half_span,
        args.longitude - half_span,
        args.latitude - half_span,
        args.longitude + half_span,
    ]
    common = {
        "date": _dates(args),
        "levtype": "ml",
        "stream": "oper",
        "time": getattr(args, "hours", HOURS),
        "type": "an",
        "area": area,
        "grid": [float(args.grid_degrees), float(args.grid_degrees)],
        "format": "grib",
    }
    model = {
        **common,
        "levelist": MODEL_LEVELS,
        "param": MODEL_PARAMETERS,
    }
    surface = {
        **common,
        "levelist": "1",
        "param": SURFACE_PARAMETERS,
    }
    return model, surface


def pressure_comparison_payload(args: argparse.Namespace) -> dict:
    """Return an optional same-date pressure-level validation request."""

    half_span = float(args.half_span_degrees)
    return {
        "date": _dates(args),
        "levtype": "pl",
        "stream": "oper",
        "time": getattr(args, "hours", HOURS),
        "type": "an",
        "area": [
            args.latitude + half_span,
            args.longitude - half_span,
            args.latitude - half_span,
            args.longitude + half_span,
        ],
        "grid": [float(args.grid_degrees), float(args.grid_degrees)],
        "format": "grib",
        "levelist": PRESSURE_LEVELS,
        "param": PRESSURE_PARAMETERS,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, choices=range(1, 13), required=True)
    parser.add_argument("--days", type=int, nargs="+", required=True)
    parser.add_argument(
        "--hours",
        default=HOURS,
        help=(
            "MARS time expression (default: 00/to/23/by/1). Corpus tooling "
            "uses shorter 13-hour windows to avoid unnecessary downloads."
        ),
    )
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--surface-output", type=Path, required=True)
    parser.add_argument(
        "--pressure-comparison-output",
        type=Path,
        help=(
            "Optional same-date pressure-level GRIB used only to validate the "
            "model-level conversion."
        ),
    )
    parser.add_argument("--latitude", type=float, default=51.5074)
    parser.add_argument("--longitude", type=float, default=-0.1278)
    parser.add_argument(
        "--half-span-degrees",
        type=float,
        default=10.0,
        help="Latitude/longitude half-span around the site (default: 10 degrees).",
    )
    parser.add_argument(
        "--grid-degrees",
        type=float,
        default=0.25,
        help="Regular output grid spacing in degrees (default: 0.25).",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Print both MARS requests without contacting CDS.",
    )
    parser.add_argument(
        "--only",
        choices=("both", "model", "surface", "pressure-comparison"),
        default="both",
        help=(
            "Submit both production inputs or resume one request. The validation "
            "request also requires --pressure-comparison-output."
        ),
    )
    return parser.parse_args()


def _retrieve(client, payload: dict, output: Path) -> None:
    if output.exists():
        print(f"keeping existing {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".part")
    if partial.exists():
        partial.unlink()
    client.retrieve(DATASET, payload, str(partial))
    if not partial.exists() or partial.stat().st_size == 0:
        raise RuntimeError(f"CDS retrieval produced no data for {output}")
    partial.replace(output)


def main() -> None:
    args = parse_args()
    model, surface = request_payloads(args)
    comparison = pressure_comparison_payload(args)
    if args.plan:
        print(
            json.dumps(
                {
                    "dataset": DATASET,
                    "model": model,
                    "surface": surface,
                    "pressure_comparison": comparison,
                },
                indent=2,
            )
        )
        return

    try:
        import cdsapi
    except ImportError as exc:
        raise SystemExit(
            "cdsapi is not installed. Run `pip install -e '.[weather]'`."
        ) from exc

    client = cdsapi.Client()
    if args.only in ("both", "model"):
        _retrieve(client, model, args.model_output)
    if args.only in ("both", "surface"):
        _retrieve(client, surface, args.surface_output)
    if args.only == "pressure-comparison" and args.pressure_comparison_output is None:
        raise SystemExit(
            "--only pressure-comparison requires --pressure-comparison-output"
        )
    if args.pressure_comparison_output is not None and args.only in (
        "both",
        "pressure-comparison",
    ):
        _retrieve(client, comparison, args.pressure_comparison_output)


if __name__ == "__main__":
    main()
