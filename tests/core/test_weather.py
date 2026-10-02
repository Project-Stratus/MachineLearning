"""Layer 2 deterministic weather-cube interpolation and persistence."""

from __future__ import annotations

import numpy as np
import pytest

from environments.core.constants import M_AIR, R, SUPERHEAT_DAY
from environments.core.weather import (
    ReanalysisAtmosphere,
    ReanalysisWeatherProvider,
    WeatherCube,
)


def make_cube(*, start_time: str = "2020-03-01T00:00:00Z", wind_scale: float = 1.0):
    time = np.array([0.0, 3_600.0, 43_200.0])
    altitude = np.array([10_000.0, 20_000.0, 30_000.0])
    y = np.array([-100_000.0, 0.0, 100_000.0])
    x = np.array([-100_000.0, 0.0, 100_000.0])
    tt, zz, yy, xx = np.meshgrid(time, altitude, y, x, indexing="ij")
    u = wind_scale * (2.0 + tt / 3_600.0 + zz / 10_000.0 + xx / 100_000.0)
    v = wind_scale * (-1.0 + 0.5 * zz / 10_000.0 + yy / 100_000.0)
    w = 0.01 + zz * 1e-6
    temperature = 190.0 + zz * 5e-4 + tt / 43_200.0
    pressure = 20_000.0 - zz * 0.5 + tt / 100.0
    radiation = 500.0 - tt / 100.0
    return WeatherCube(
        time_s=time,
        altitude_m=altitude,
        y_m=y,
        x_m=x,
        u=u,
        v=v,
        w=w,
        temperature=temperature,
        pressure=pressure,
        radiative_forcing=radiation,
        metadata={"source": "synthetic", "start_time_utc": start_time},
    )


class TestWeatherCube:
    def test_round_trip_without_pickle(self, tmp_path):
        source = make_cube()
        path = tmp_path / "weather.npz"
        source.save(path)
        restored = WeatherCube.load(path)
        assert restored.metadata == source.metadata
        for name in (
            "time_s",
            "altitude_m",
            "y_m",
            "x_m",
            "u",
            "v",
            "w",
            "temperature",
            "pressure",
            "radiative_forcing",
        ):
            assert np.array_equal(getattr(restored, name), getattr(source, name))

    def test_rejects_non_monotonic_coordinates(self):
        cube = make_cube()
        cube.x_m[:] = [0.0, -1.0, 1.0]
        with pytest.raises(ValueError, match="strictly increasing"):
            WeatherCube(**cube.__dict__)

    def test_rejects_wrong_field_shape(self):
        cube = make_cube()
        cube.u = cube.u[:, :-1]
        with pytest.raises(ValueError, match="expected"):
            WeatherCube(**cube.__dict__)


class TestReanalysisProvider:
    def test_multilinear_interpolation_is_exact_for_linear_field(self):
        provider = ReanalysisWeatherProvider(make_cube(), bounds_policy="raise")
        provider.set_time(1_800.0)
        sample = provider.sample_weather(50_000.0, -50_000.0, 15_000.0)
        assert sample.u == pytest.approx(2.0 + 0.5 + 1.5 + 0.5)
        assert sample.v == pytest.approx(-1.0 + 0.75 - 0.5)
        assert sample.w == pytest.approx(0.025)
        assert sample.temperature == pytest.approx(190.0 + 7.5 + 1_800 / 43_200)
        expected_p = 20_000.0 - 7_500.0 + 18.0
        assert sample.pressure == pytest.approx(expected_p)
        assert sample.density == pytest.approx(
            expected_p * M_AIR / (R * sample.temperature)
        )

    def test_time_offset_selects_later_weather(self):
        provider = ReanalysisWeatherProvider(make_cube(), start_offset_s=3_600.0)
        assert provider.sample_weather(0.0, 0.0, 20_000.0).u == pytest.approx(5.0)

    def test_bounds_can_raise_or_clip(self):
        strict = ReanalysisWeatherProvider(make_cube(), bounds_policy="raise")
        with pytest.raises(ValueError, match="outside weather domain"):
            strict.sample(1_000_000.0, 0.0, 20_000.0)

        clipped = ReanalysisWeatherProvider(make_cube(), bounds_policy="clip")
        assert np.array_equal(
            clipped.sample(1_000_000.0, 0.0, 20_000.0).copy(),
            clipped.sample(100_000.0, 0.0, 20_000.0).copy(),
        )

    def test_episode_window_must_fit_the_time_axis(self):
        provider = ReanalysisWeatherProvider(make_cube(), start_offset_s=3_600.0)
        provider.validate_window(39_600.0)
        with pytest.raises(ValueError, match="does not cover"):
            provider.validate_window(43_200.0)

    def test_column_is_low_to_high_and_uses_current_time(self):
        provider = ReanalysisWeatherProvider(make_cube())
        provider.set_time(3_600.0)
        column = provider.sample_column(0.0, 0.0, 20_000.0, levels=3, spacing=5_000.0)
        assert column.shape == (3, 2)
        assert np.all(np.diff(column[:, 0]) > 0.0)


class TestReanalysisAtmosphere:
    def test_uses_weather_instead_of_isa_and_keeps_day_offset(self):
        provider = ReanalysisWeatherProvider(make_cube())
        atmosphere = ReanalysisAtmosphere(provider)
        atmosphere.set_context(0.0, 0.0, 1_800.0)
        sample = provider.sample_weather(0.0, 0.0, 20_000.0)
        assert atmosphere.use_isa_numba is False
        assert atmosphere.temperature(20_000.0) == pytest.approx(sample.temperature)
        assert atmosphere.pressure(20_000.0) == pytest.approx(sample.pressure)
        assert atmosphere.density(20_000.0) == pytest.approx(sample.density)
        assert atmosphere.gas_temperature(20_000.0) == pytest.approx(
            sample.temperature + SUPERHEAT_DAY
        )
