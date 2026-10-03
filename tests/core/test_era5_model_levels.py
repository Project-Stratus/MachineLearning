"""ERA5 L137 acquisition plans and hybrid-level reconstruction."""

from __future__ import annotations

import argparse
from datetime import timedelta

import numpy as np
import pytest

from environments.core.constants import G
from scripts.acquire_era5_corpus import (
    _combine_grib,
    acquisition_batches,
    corpus_scenarios,
)
from scripts.download_era5_model_levels import (
    DATASET,
    _retrieve,
    pressure_comparison_payload,
    request_payloads,
)
from scripts.prepare_era5_model_levels import (
    R_DRY_AIR,
    geometric_height,
    hybrid_pressures,
    model_level_geopotential,
)


def download_args(**overrides) -> argparse.Namespace:
    values = {
        "year": 2021,
        "month": 4,
        "days": [15],
        "latitude": 51.5074,
        "longitude": -0.1278,
        "half_span_degrees": 10.0,
        "grid_degrees": 0.25,
        "hours": "00/to/23/by/1",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_model_level_plan_uses_complete_era5_and_required_fields():
    model, surface = request_payloads(download_args())
    assert DATASET == "reanalysis-era5-complete"
    assert model["date"] == "2021-04-15"
    assert model["time"] == "00/to/23/by/1"
    assert model["levelist"] == "1/to/137/by/1"
    assert model["param"] == "130/131/132/133/135"
    assert surface["levelist"] == "1"
    assert surface["param"] == "129/152"
    assert model["area"] == pytest.approx([61.5074, -10.1278, 41.5074, 9.8722])
    assert model["grid"] == [0.25, 0.25]


def test_model_level_plan_rejects_invalid_calendar_date():
    with pytest.raises(ValueError):
        request_payloads(download_args(month=4, days=[31]))


def test_pressure_comparison_uses_same_complete_dataset_grid_and_date():
    payload = pressure_comparison_payload(download_args())
    assert DATASET == "reanalysis-era5-complete"
    assert payload["date"] == "2021-04-15"
    assert payload["levtype"] == "pl"
    assert payload["levelist"] == "10/20/30/50/70/100/125/150/175/200/225/250/300"
    assert payload["param"] == "129/130/131/132/135"
    assert payload["area"] == pytest.approx([61.5074, -10.1278, 41.5074, 9.8722])
    assert payload["grid"] == [0.25, 0.25]


def test_model_level_plan_accepts_a_bounded_time_window():
    model, surface = request_payloads(download_args(hours="00/to/12/by/1"))
    assert model["time"] == "00/to/12/by/1"
    assert surface["time"] == "00/to/12/by/1"


def test_model_level_plan_accepts_explicit_dates_across_months():
    args = download_args()
    args.dates = ["2018-03-15", "2018-04-15"]
    model, _surface = request_payloads(args)
    assert model["date"] == "2018-03-15/2018-04-15"


def test_smoke_corpus_has_one_train_and_one_heldout_cube():
    scenarios = corpus_scenarios("smoke")
    assert [scenario.start.isoformat() for scenario in scenarios] == [
        "2018-04-15T00:00:00+00:00",
        "2021-04-15T00:00:00+00:00",
    ]
    assert [scenario.split for scenario in scenarios] == ["train", "heldout"]


def test_sampled_corpus_has_expected_split_and_alternating_starts():
    scenarios = corpus_scenarios("sampled")
    assert len(scenarios) == 128
    assert sum(scenario.split == "train" for scenario in scenarios) == 104
    assert sum(scenario.split == "heldout" for scenario in scenarios) == 24
    assert all(
        scenario.start.hour == (0 if scenario.start.day in (1, 15) else 12)
        for scenario in scenarios
    )
    assert len(acquisition_batches(scenarios)) == 32


def test_noon_batch_adds_the_following_midnight_boundary():
    scenarios = [
        scenario
        for scenario in corpus_scenarios("sampled")
        if scenario.start.year == 2018 and scenario.start.hour == 12
    ]
    batch = acquisition_batches(scenarios)[0]
    assert len(batch.chunks) == 2
    assert batch.chunks[0].hours == "12/to/23/by/1"
    assert batch.chunks[1].hours == "00"
    assert batch.chunks[1].dates == tuple(
        scenario.start.date() + timedelta(days=1) for scenario in scenarios
    )


def test_grib_parts_are_combined_in_order_and_removed(tmp_path):
    first = tmp_path / "first.grib"
    second = tmp_path / "second.grib"
    output = tmp_path / "combined.grib"
    first.write_bytes(b"GRIB-first")
    second.write_bytes(b"GRIB-second")

    _combine_grib([first, second], output)

    assert output.read_bytes() == b"GRIB-firstGRIB-second"
    assert not first.exists()
    assert not second.exists()


def test_retrieval_is_published_atomically(tmp_path):
    class FakeClient:
        def retrieve(self, dataset, payload, target):
            assert dataset == DATASET
            assert payload == {"request": "bounded"}
            assert target.endswith(".grib.part")
            with open(target, "wb") as handle:
                handle.write(b"GRIB-data")

    output = tmp_path / "weather.grib"
    _retrieve(FakeClient(), {"request": "bounded"}, output)

    assert output.read_bytes() == b"GRIB-data"
    assert not output.with_suffix(".grib.part").exists()


def test_hybrid_pressures_follow_a_plus_b_surface_pressure():
    lnsp = np.log(np.array([[[100_000.0]], [[80_000.0]]]))
    a = np.array([0.0, 1_000.0, 0.0])
    b = np.array([0.0, 0.4, 1.0])
    half, full = hybrid_pressures(lnsp, a, b)

    assert half.shape == (2, 3, 1, 1)
    assert full.shape == (2, 2, 1, 1)
    assert half[0, :, 0, 0] == pytest.approx([0.0, 41_000.0, 100_000.0])
    assert full[0, :, 0, 0] == pytest.approx([20_500.0, 70_500.0])
    assert half[1, :, 0, 0] == pytest.approx([0.0, 33_000.0, 80_000.0])


def test_geopotential_integrates_from_surface_upward():
    temperature = np.array([[[[250.0]], [[300.0]]]])
    humidity = np.zeros_like(temperature)
    pressure_half = np.array([[[[0.0]], [[50_000.0]], [[100_000.0]]]])
    surface = np.array([[[100.0]]])

    geopotential = model_level_geopotential(
        temperature, humidity, pressure_half, surface
    )

    bottom_delta_log = np.log(2.0)
    bottom_alpha = 1.0 - bottom_delta_log
    expected_bottom = 100.0 + R_DRY_AIR * 300.0 * bottom_alpha
    expected_top = (
        100.0 + R_DRY_AIR * 300.0 * bottom_delta_log + R_DRY_AIR * 250.0 * np.log(2.0)
    )
    assert geopotential[0, 1, 0, 0] == pytest.approx(expected_bottom)
    assert geopotential[0, 0, 0, 0] == pytest.approx(expected_top)
    assert geopotential[0, 0, 0, 0] > geopotential[0, 1, 0, 0]


def test_geometric_height_applies_earth_curvature_correction():
    geopotential_height = np.array([0.0, 20_000.0, 30_000.0])
    result = geometric_height(geopotential_height * G)
    assert result[0] == 0.0
    assert np.all(result[1:] > geopotential_height[1:])


def test_geopotential_rejects_inconsistent_shapes():
    with pytest.raises(ValueError, match="humidity shapes differ"):
        model_level_geopotential(
            np.ones((1, 2, 1, 1)),
            np.ones((1, 1, 1, 1)),
            np.ones((1, 3, 1, 1)),
            np.ones((1, 1, 1)),
        )
