#!/usr/bin/env python3
"""Plan or execute the sampled London ERA5 model-level corpus on the HPC.

The default is a read-only plan. ``--execute`` is deliberately required before
the script contacts CDS, writes GRIB/cube assets, or builds a manifest.

Profiles
--------
``smoke``
    Two 00:00 UTC cubes: 15 April 2018 for training and 15 April 2021 held out.
    This is the minimum leakage-safe corpus for the 60k-step Phase 2 smoke job.
``sampled``
    The full initial design: 1, 8, 15 and 22 March/April for 2010--2025, with
    2019--2021 held out. Days 1/15 start at 00:00 UTC and days 8/22 at 12:00
    UTC, yielding 104 train and 24 held-out scenarios.

Requests are batched by year and start hour. A 12:00 scenario needs 12:00--23:00
on its start date plus 00:00 on the following date; the two valid GRIB streams
are concatenated before conversion. Runs are resumable: existing raw batches
and runtime cubes are kept.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from environments.core.weather_scenarios import build_manifest  # noqa: E402
from scripts.download_era5_model_levels import (  # noqa: E402
    DATASET,
    _retrieve,
    request_payloads,
)
from scripts.prepare_era5_model_levels import convert  # noqa: E402

HELD_OUT_YEARS = frozenset({2019, 2020, 2021})
ALL_YEARS = tuple(range(2010, 2026))
SAMPLED_MONTHS = (3, 4)
SAMPLED_DAYS = (1, 8, 15, 22)


@dataclass(frozen=True)
class ScenarioPlan:
    start: datetime
    split: str

    @property
    def start_hour(self) -> int:
        return self.start.hour

    @property
    def slug(self) -> str:
        return f"london-{self.start:%Y-%m-%dT%H}"

    def cube_path(self, root: Path) -> Path:
        return root / "cubes" / f"{self.slug}-model-levels.npz"


@dataclass(frozen=True)
class RequestChunk:
    dates: tuple[date, ...]
    hours: str


@dataclass(frozen=True)
class AcquisitionBatch:
    year: int
    start_hour: int
    scenarios: tuple[ScenarioPlan, ...]

    @property
    def slug(self) -> str:
        return f"london-{self.year}-spring-T{self.start_hour:02d}"

    @property
    def chunks(self) -> tuple[RequestChunk, ...]:
        starts = tuple(scenario.start.date() for scenario in self.scenarios)
        if self.start_hour == 0:
            return (RequestChunk(starts, "00/to/12/by/1"),)
        boundaries = tuple(value + timedelta(days=1) for value in starts)
        return (
            RequestChunk(starts, "12/to/23/by/1"),
            RequestChunk(boundaries, "00"),
        )

    def raw_path(self, root: Path, field: str) -> Path:
        return root / "raw" / f"{self.slug}-{field}.grib"


def corpus_scenarios(profile: str) -> list[ScenarioPlan]:
    """Return the deterministic train/held-out schedule for ``profile``."""

    if profile == "smoke":
        starts = (
            datetime(2018, 4, 15, 0, tzinfo=timezone.utc),
            datetime(2021, 4, 15, 0, tzinfo=timezone.utc),
        )
    elif profile == "sampled":
        starts = tuple(
            datetime(
                year,
                month,
                day,
                0 if day in (1, 15) else 12,
                tzinfo=timezone.utc,
            )
            for year in ALL_YEARS
            for month in SAMPLED_MONTHS
            for day in SAMPLED_DAYS
        )
    else:
        raise ValueError(f"unknown corpus profile {profile!r}")
    return [
        ScenarioPlan(
            start=start,
            split="heldout" if start.year in HELD_OUT_YEARS else "train",
        )
        for start in starts
    ]


def acquisition_batches(scenarios: list[ScenarioPlan]) -> list[AcquisitionBatch]:
    grouped: dict[tuple[int, int], list[ScenarioPlan]] = defaultdict(list)
    for scenario in scenarios:
        grouped[(scenario.start.year, scenario.start_hour)].append(scenario)
    return [
        AcquisitionBatch(
            year=year,
            start_hour=start_hour,
            scenarios=tuple(sorted(members, key=lambda item: item.start)),
        )
        for (year, start_hour), members in sorted(grouped.items())
    ]


def _request_args(chunk: RequestChunk, args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        dates=chunk.dates,
        year=chunk.dates[0].year,
        month=chunk.dates[0].month,
        days=[value.day for value in chunk.dates],
        hours=chunk.hours,
        latitude=args.latitude,
        longitude=args.longitude,
        half_span_degrees=args.half_span_degrees,
        grid_degrees=args.grid_degrees,
    )


def _combine_grib(parts: list[Path], destination: Path) -> None:
    if destination.exists():
        print(f"keeping existing {destination}")
        return
    if not parts or any(
        not part.exists() or part.stat().st_size == 0 for part in parts
    ):
        raise ValueError(f"cannot combine incomplete GRIB parts for {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as output:
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, output, length=1024 * 1024)
    temporary.replace(destination)
    for part in parts:
        part.unlink()


def _download_batch(client, batch: AcquisitionBatch, root: Path, args) -> None:
    model_output = batch.raw_path(root, "ml")
    surface_output = batch.raw_path(root, "surface")
    need_model = not model_output.exists()
    need_surface = not surface_output.exists()
    if not need_model and not need_surface:
        print(f"keeping existing raw batch {batch.slug}")
        return

    chunks = batch.chunks
    if len(chunks) == 1:
        model_payload, surface_payload = request_payloads(
            _request_args(chunks[0], args)
        )
        if need_model:
            _retrieve(client, model_payload, model_output)
        if need_surface:
            _retrieve(client, surface_payload, surface_output)
        return

    parts_dir = root / "raw" / ".parts"
    model_parts: list[Path] = []
    surface_parts: list[Path] = []
    for index, chunk in enumerate(chunks):
        model_payload, surface_payload = request_payloads(_request_args(chunk, args))
        model_part = parts_dir / f"{batch.slug}-{index}-ml.grib"
        surface_part = parts_dir / f"{batch.slug}-{index}-surface.grib"
        if need_model:
            _retrieve(client, model_payload, model_part)
            model_parts.append(model_part)
        if need_surface:
            _retrieve(client, surface_payload, surface_part)
            surface_parts.append(surface_part)
    if need_model:
        _combine_grib(model_parts, model_output)
    if need_surface:
        _combine_grib(surface_parts, surface_output)


def _convert_scenario(
    scenario: ScenarioPlan,
    batch: AcquisitionBatch,
    root: Path,
    args: argparse.Namespace,
) -> None:
    output = scenario.cube_path(root)
    if output.exists():
        print(f"keeping existing {output}")
        return
    conversion_args = argparse.Namespace(
        model_input=batch.raw_path(root, "ml"),
        surface_input=batch.raw_path(root, "surface"),
        output=output,
        start=scenario.start.strftime("%Y-%m-%dT%H:%M"),
        duration_hours=12.0,
        site_name="London",
        latitude=args.latitude,
        longitude=args.longitude,
        altitude_min=10_000.0,
        altitude_max=30_000.0,
        altitude_step=250.0,
    )
    cube = convert(conversion_args)
    cube.save(output)
    print(f"wrote {output}: shape={cube.u.shape}")


def _default_manifest(root: Path, profile: str) -> Path:
    name = "london-smoke.json" if profile == "smoke" else "london-spring.json"
    return root / "manifests" / name


def print_plan(
    scenarios: list[ScenarioPlan],
    batches: list[AcquisitionBatch],
    root: Path,
    manifest: Path,
) -> None:
    counts = {
        split: sum(scenario.split == split for scenario in scenarios)
        for split in ("train", "heldout")
    }
    print(f"dataset: {DATASET}")
    print(f"assets: {root}")
    print(f"manifest: {manifest}")
    print(
        f"scenarios: {len(scenarios)} "
        f"({counts['train']} train, {counts['heldout']} held out)"
    )
    print(f"CDS batches: {len(batches)}")
    for batch in batches:
        dates = ", ".join(
            scenario.start.strftime("%Y-%m-%dT%H") for scenario in batch.scenarios
        )
        chunks = "; ".join(
            f"{','.join(value.isoformat() for value in chunk.dates)} @ {chunk.hours}"
            for chunk in batch.chunks
        )
        print(f"  {batch.slug}: {dates} | {chunks}")
    print("dry run only; pass --execute on the HPC to contact CDS and write assets")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "sampled"), default="smoke")
    parser.add_argument("--root", type=Path, default=Path("weather_data"))
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--latitude", type=float, default=51.5074)
    parser.add_argument("--longitude", type=float, default=-0.1278)
    parser.add_argument("--half-span-degrees", type=float, default=10.0)
    parser.add_argument("--grid-degrees", type=float, default=0.25)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--plan",
        action="store_true",
        help="Print the plan without contacting CDS (the default).",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Contact CDS, convert cubes, and build the manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root
    scenarios = corpus_scenarios(args.profile)
    batches = acquisition_batches(scenarios)
    manifest = args.manifest_output or _default_manifest(root, args.profile)
    if not args.execute:
        print_plan(scenarios, batches, root, manifest)
        return

    try:
        import cdsapi
    except ImportError as exc:
        raise SystemExit(
            "cdsapi is not installed. Run `pip install -e '.[weather]'`."
        ) from exc

    client = cdsapi.Client()
    for batch in batches:
        pending = [
            scenario
            for scenario in batch.scenarios
            if not scenario.cube_path(root).exists()
        ]
        if not pending:
            print(f"keeping existing cubes for {batch.slug}")
            continue
        _download_batch(client, batch, root, args)
        for scenario in pending:
            _convert_scenario(scenario, batch, root, args)

    build_manifest(
        [scenario.cube_path(root) for scenario in scenarios],
        output_path=manifest,
        heldout_years=set(HELD_OUT_YEARS),
        site_name="London",
        latitude=args.latitude,
        longitude=args.longitude,
    )
    print(f"wrote {manifest}")


if __name__ == "__main__":
    main()
