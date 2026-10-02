"""Date-based Layer 2 weather scenario manifests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from environments.core.weather import WeatherCube
from environments.core.weather_scenarios import (
    WeatherScenario,
    WeatherScenarioManifest,
    build_manifest,
    make_weather_scenario_set,
    sha256_file,
    weather_difficulty,
)


def write_cube(path, year: int, speed: float, *, start_time: str | None = None) -> None:
    time = np.array([0.0, 43_200.0])
    altitude = np.array([15_000.0, 20_000.0, 25_000.0])
    y = x = np.array([-50_000.0, 0.0, 50_000.0])
    shape = (time.size, altitude.size, y.size, x.size)
    u = np.empty(shape, dtype=np.float32)
    v = np.empty(shape, dtype=np.float32)
    for iz in range(altitude.size):
        angle = iz * np.pi / 2.0
        u[:, iz] = speed * np.cos(angle)
        v[:, iz] = speed * np.sin(angle)
    WeatherCube(
        time_s=time,
        altitude_m=altitude,
        y_m=y,
        x_m=x,
        u=u,
        v=v,
        w=np.zeros(shape),
        temperature=np.full(shape, 205.0),
        pressure=np.full(shape, 7_000.0),
        metadata={"start_time_utc": start_time or f"{year}-03-15T00:00:00Z"},
    ).save(path)


def test_manifest_round_trip_and_year_split(tmp_path):
    paths = []
    for index, year in enumerate((2019, 2020, 2021, 2022, 2023, 2024)):
        path = tmp_path / f"weather-{year}.npz"
        write_cube(path, year, speed=5.0 + index)
        paths.append(path)

    manifest_path = tmp_path / "manifest.json"
    built = build_manifest(
        paths,
        output_path=manifest_path,
        heldout_years={2023, 2024},
    )
    restored = WeatherScenarioManifest.load(manifest_path)
    assert len(built.for_split("train")) == len(restored.for_split("train")) == 4
    assert len(restored.for_split("heldout")) == 2
    assert {s.year for s in restored.for_split("train")} == {2019, 2020, 2021, 2022}
    assert {s.year for s in restored.for_split("heldout")} == {2023, 2024}
    assert all(not Path(s.weather_path).is_absolute() for s in restored.scenarios)
    assert all(restored.resolve_weather_path(s).exists() for s in restored.scenarios)
    assert all(
        s.sha256 == sha256_file(restored.resolve_weather_path(s))
        for s in restored.scenarios
    )

    selected = make_weather_scenario_set(restored, n=2, seed=7, held_out=True)
    assert len({s["scenario_id"] for s in selected}) == 2
    assert all(s["held_out"] for s in selected)
    assert all("weather_scenario_id" in s["config"] for s in selected)


def test_manifest_rejects_year_leakage():
    common = dict(
        weather_path="a.npz",
        start_time_utc="2024-03-01T00:00:00Z",
        start_offset_s=0.0,
        duration_s=43_200.0,
        year=2024,
        difficulty_band="medium",
        difficulty_score=0.5,
        wind_diversity=0.5,
        mean_wind_speed=10.0,
        vertical_shear=2.0,
        sha256="abc",
    )
    with pytest.raises(ValueError, match="split leakage"):
        WeatherScenarioManifest(
            site_name="London",
            latitude=51.5,
            longitude=-0.1,
            scenarios=[
                WeatherScenario(scenario_id="train", split="train", **common),
                WeatherScenario(scenario_id="heldout", split="heldout", **common),
            ],
        )


def test_manifest_drops_windows_that_cross_a_year_boundary(tmp_path):
    path = tmp_path / "weather.npz"
    write_cube(
        path,
        2023,
        speed=10.0,
        start_time="2023-12-31T18:00:00Z",
    )
    with pytest.raises(ValueError, match="empty weather manifest"):
        build_manifest(
            [path],
            output_path=tmp_path / "manifest.json",
            heldout_years={2024},
        )


def test_difficulty_features_are_finite(tmp_path):
    path = tmp_path / "weather.npz"
    write_cube(path, 2024, speed=15.0)
    metrics = weather_difficulty(WeatherCube.load(path))
    assert set(metrics) == {
        "difficulty_score",
        "wind_diversity",
        "mean_wind_speed",
        "vertical_shear",
    }
    assert all(np.isfinite(value) for value in metrics.values())
