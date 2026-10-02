"""Frozen, date-based scenario manifests for Layer 2 weather.

Seed ranges are sufficient for analytic weather because the seed creates the
entire field.  They are not a train/test boundary for reanalysis: neighbouring
seeds can select neighbouring hours from the same storm.  These manifests make
the weather asset, UTC window, year split and content hash explicit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np

from environments.core.constants import ALT_SAFE_MAX, ALT_SAFE_MIN
from environments.core.weather import WeatherCube

Split = Literal["train", "heldout"]
DifficultyBand = Literal["easy", "medium", "hard", "unclassified"]
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class WeatherScenario:
    scenario_id: str
    weather_path: str
    start_time_utc: str
    start_offset_s: float
    duration_s: float
    split: Split
    year: int
    difficulty_band: DifficultyBand
    difficulty_score: float
    wind_diversity: float
    mean_wind_speed: float
    vertical_shear: float
    sha256: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "WeatherScenario":
        return cls(**value)


@dataclass
class WeatherScenarioManifest:
    site_name: str
    latitude: float
    longitude: float
    scenarios: list[WeatherScenario]
    source: str = "ERA5"
    schema_version: int = SCHEMA_VERSION
    path: Path | None = None

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported weather manifest schema {self.schema_version}; "
                f"expected {SCHEMA_VERSION}"
            )
        ids = [scenario.scenario_id for scenario in self.scenarios]
        if len(ids) != len(set(ids)):
            raise ValueError("weather scenario IDs must be unique")
        train_years = {s.year for s in self.scenarios if s.split == "train"}
        heldout_years = {s.year for s in self.scenarios if s.split == "heldout"}
        overlap = train_years & heldout_years
        if overlap:
            raise ValueError(
                "weather split leakage: years occur in both train and heldout: "
                f"{sorted(overlap)}"
            )

    @property
    def base_dir(self) -> Path:
        return self.path.parent if self.path is not None else Path.cwd()

    def for_split(self, split: Split) -> list[WeatherScenario]:
        return [scenario for scenario in self.scenarios if scenario.split == split]

    def by_id(self, scenario_id: str) -> WeatherScenario:
        for scenario in self.scenarios:
            if scenario.scenario_id == scenario_id:
                return scenario
        raise KeyError(f"unknown weather scenario {scenario_id!r}")

    def resolve_weather_path(self, scenario: WeatherScenario) -> Path:
        path = Path(scenario.weather_path)
        return path if path.is_absolute() else self.base_dir / path

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.schema_version,
            "source": self.source,
            "site": {
                "name": self.site_name,
                "latitude": self.latitude,
                "longitude": self.longitude,
            },
            "scenarios": [asdict(scenario) for scenario in self.scenarios],
        }
        destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        self.path = destination.resolve()

    @classmethod
    def load(cls, path: str | Path) -> "WeatherScenarioManifest":
        source = Path(path).resolve()
        payload = json.loads(source.read_text())
        site = payload["site"]
        return cls(
            site_name=site["name"],
            latitude=float(site["latitude"]),
            longitude=float(site["longitude"]),
            scenarios=[WeatherScenario.from_dict(s) for s in payload["scenarios"]],
            source=payload.get("source", "ERA5"),
            schema_version=int(payload.get("schema_version", 0)),
            path=source,
        )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def weather_difficulty(
    cube: WeatherCube,
    *,
    start_offset_s: float = 0.0,
    duration_s: float | None = None,
) -> dict[str, float]:
    """Return transparent wind-only difficulty features for one weather cube.

    The score is a *stratification proxy*, not a claim about optimal control.
    Fast, similarly directed winds are harder because they remove altitude
    choices and carry the balloon away quickly.  Directional diversity and
    vertical shear provide choices and therefore reduce the score.
    """

    z_mask = (cube.altitude_m >= ALT_SAFE_MIN) & (cube.altitude_m <= ALT_SAFE_MAX)
    if not np.any(z_mask):
        raise ValueError("weather cube does not overlap the operational altitude band")
    iy = int(np.argmin(np.abs(cube.y_m)))
    ix = int(np.argmin(np.abs(cube.x_m)))
    if duration_s is None:
        time_mask = cube.time_s >= start_offset_s
    else:
        time_mask = (cube.time_s >= start_offset_s) & (
            cube.time_s <= start_offset_s + duration_s
        )
    if not np.any(time_mask):
        raise ValueError("weather window contains no time samples")
    u = cube.u[time_mask][:, z_mask, iy, ix].astype(np.float64)
    v = cube.v[time_mask][:, z_mask, iy, ix].astype(np.float64)
    speed = np.hypot(u, v)
    unit_u = u / np.maximum(speed, 1e-6)
    unit_v = v / np.maximum(speed, 1e-6)
    resultant = np.hypot(np.mean(unit_u, axis=1), np.mean(unit_v, axis=1))
    diversity = float(np.mean(1.0 - resultant))

    if u.shape[1] > 1:
        shear = np.hypot(np.diff(u, axis=1), np.diff(v, axis=1))
        vertical_shear = float(np.mean(shear))
    else:
        vertical_shear = 0.0
    mean_speed = float(np.mean(speed))

    # Bounded and deliberately simple. Population quantiles, not absolute
    # thresholds, assign easy/medium/hard below.
    speed_term = min(mean_speed / 40.0, 1.0)
    choice_term = 1.0 - min(diversity + vertical_shear / 20.0, 1.0)
    score = float(0.55 * speed_term + 0.45 * choice_term)
    return {
        "difficulty_score": score,
        "wind_diversity": diversity,
        "mean_wind_speed": mean_speed,
        "vertical_shear": vertical_shear,
    }


def build_manifest(
    weather_paths: Iterable[str | Path],
    *,
    output_path: str | Path,
    heldout_years: set[int],
    site_name: str = "London",
    latitude: float = 51.5074,
    longitude: float = -0.1278,
    duration_s: float = 43_200.0,
    window_stride_s: float = 43_200.0,
) -> WeatherScenarioManifest:
    """Build a frozen manifest and stratify each split into difficulty thirds."""

    output = Path(output_path).resolve()
    site_slug = "-".join(
        part
        for part in "".join(
            character if character.isalnum() else "-" for character in site_name.lower()
        ).split("-")
        if part
    )
    site_slug = site_slug or "site"
    pending: list[dict[str, Any]] = []
    for raw_path in sorted(Path(p).resolve() for p in weather_paths):
        cube = WeatherCube.load(raw_path)
        epoch_text = str(cube.metadata.get("start_time_utc", ""))
        if not epoch_text:
            raise ValueError(f"{raw_path} has no metadata.start_time_utc")
        epoch = datetime.fromisoformat(epoch_text.replace("Z", "+00:00"))
        if epoch.tzinfo is None:
            epoch = epoch.replace(tzinfo=timezone.utc)
        # Keep manifests movable between the laptop and HPC. The usual layout
        # places cubes beside (not beneath) the manifests directory, for which
        # Path.relative_to would incorrectly force an absolute path.
        relative_path = os.path.relpath(raw_path, output.parent)
        latest_start = float(cube.time_s[-1]) - float(duration_s)
        if latest_start < float(cube.time_s[0]):
            raise ValueError(
                f"{raw_path} is shorter than the {duration_s / 3600:.1f} h window"
            )
        if window_stride_s <= 0.0:
            raise ValueError("window_stride_s must be positive")
        first_start = float(cube.time_s[0])
        count = int(np.floor((latest_start - first_start) / window_stride_s)) + 1
        offsets = first_start + np.arange(count) * float(window_stride_s)
        digest = sha256_file(raw_path)
        for offset in offsets:
            start_dt = epoch + timedelta(seconds=float(offset))
            end_dt = start_dt + timedelta(seconds=float(duration_s))
            # Whole-year splitting is only leakage-safe if a window itself
            # does not straddle the boundary between train and held-out years.
            if end_dt.year != start_dt.year:
                continue
            start_time = start_dt.isoformat().replace("+00:00", "Z")
            year = start_dt.year
            metrics = weather_difficulty(
                cube, start_offset_s=float(offset), duration_s=duration_s
            )
            pending.append(
                {
                    "scenario_id": (
                        f"era5-{site_slug}-"
                        + start_time.replace(":", "").replace("-", "")
                    ),
                    "weather_path": relative_path,
                    "start_time_utc": start_time,
                    "start_offset_s": float(offset),
                    "duration_s": float(duration_s),
                    "split": "heldout" if year in heldout_years else "train",
                    "year": year,
                    "sha256": digest,
                    **metrics,
                }
            )

    if not pending:
        raise ValueError("cannot build an empty weather manifest")

    for split in ("train", "heldout"):
        members = [item for item in pending if item["split"] == split]
        members.sort(key=lambda item: (item["difficulty_score"], item["scenario_id"]))
        count = len(members)
        for rank, item in enumerate(members):
            fraction = (rank + 0.5) / count
            item["difficulty_band"] = (
                "easy"
                if fraction < 1.0 / 3.0
                else "medium" if fraction < 2.0 / 3.0 else "hard"
            )

    manifest = WeatherScenarioManifest(
        site_name=site_name,
        latitude=latitude,
        longitude=longitude,
        scenarios=[WeatherScenario(**item) for item in pending],
    )
    manifest.save(output)
    return manifest


def make_weather_scenario_set(
    manifest: WeatherScenarioManifest | str | Path,
    *,
    n: int,
    seed: int,
    held_out: bool,
) -> list[dict[str, Any]]:
    """Create evaluation descriptors pinned to distinct manifest records."""

    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    loaded = (
        WeatherScenarioManifest.load(manifest)
        if isinstance(manifest, (str, Path))
        else manifest
    )
    split: Split = "heldout" if held_out else "train"
    candidates = loaded.for_split(split)
    if n > len(candidates):
        raise ValueError(
            f"requested {n} {split} scenarios but manifest contains {len(candidates)}"
        )
    rng = np.random.default_rng([int(seed), int(held_out)])
    chosen = rng.choice(len(candidates), size=n, replace=False)
    seed_lo = 1_000_000 if held_out else 0
    reset_seeds = rng.choice(1_000_000, size=n, replace=False) + seed_lo
    return [
        {
            "scenario_id": candidates[int(index)].scenario_id,
            "seed": int(reset_seed),
            "held_out": bool(held_out),
            "config": {"weather_scenario_id": candidates[int(index)].scenario_id},
        }
        for index, reset_seed in zip(chosen, reset_seeds)
    ]
