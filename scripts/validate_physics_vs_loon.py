"""Roadmap §3.10 — validate the physics core against real flight measurements.

Checks the ISA atmosphere and our assumed vertical dynamics against the Loon
Q2-2021 flight CSV (Zenodo record 5119968). Loon flew superpressure balloons and
we fly zero-pressure, so the *platform* numbers do not transfer — but the
atmosphere does, and it is the layer everything else is built on.

Usage:  python scripts/validate_physics_vs_loon.py [--csv PATH] [--rows N]
"""
from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from environments.core.atmosphere import Atmosphere            # noqa: E402
from environments.core.constants import (                      # noqa: E402
    ALT_SAFE_MIN, ALT_SAFE_MAX, T_TROPOPAUSE, VEL_MAX, SUPERHEAT_DAY,
)

DEFAULT_CSV = Path(__file__).resolve().parents[1] / "EDA/Data/loon-flights-2021Q2.csv.gz"


def load(csv_path: Path, rows: int | None) -> pd.DataFrame:
    opener = gzip.open if csv_path.suffix == ".gz" else open
    with opener(csv_path, "rt") as fh:
        df = pd.read_csv(
            fh, nrows=rows,
            usecols=["flight_id", "time", "altitude", "temperature", "pressure",
                     "velocity_u", "velocity_v", "solar_elevation", "is_daytime"],
        )
    df["time"] = pd.to_datetime(df["time"], format="ISO8601")
    return df.dropna(subset=["altitude", "pressure", "temperature"])


def section(title: str) -> None:
    print(f"\n{title}\n{'=' * len(title)}")


def check_pressure(df: pd.DataFrame, atmo: Atmosphere) -> None:
    """Measured pressure vs ISA at the same altitude. Loon reports hPa; we use Pa."""
    section("1. ISA pressure vs measured")
    alt = df["altitude"].to_numpy()
    p_meas = df["pressure"].to_numpy() * 100.0                  # hPa -> Pa
    p_isa = np.array([atmo.pressure(float(z)) for z in alt])
    rel = (p_isa - p_meas) / p_meas * 100.0

    print(f"  samples                {len(alt):,}")
    print(f"  altitude range         {alt.min():,.0f} - {alt.max():,.0f} m")
    print(f"  measured pressure      {p_meas.min():,.0f} - {p_meas.max():,.0f} Pa")
    print(f"  ISA error (median)     {np.median(rel):+.1f} %")
    print(f"  ISA error (mean)       {rel.mean():+.1f} %")
    print(f"  ISA error (p5..p95)    {np.percentile(rel, 5):+.1f} .. {np.percentile(rel, 95):+.1f} %")

    band = (alt >= ALT_SAFE_MIN) & (alt <= ALT_SAFE_MAX)
    if band.any():
        print(f"  -- within our operating band ({ALT_SAFE_MIN:,.0f}-{ALT_SAFE_MAX:,.0f} m), "
              f"n={band.sum():,}")
        print(f"     median error        {np.median(rel[band]):+.1f} %")


def check_temperature(df: pd.DataFrame, atmo: Atmosphere) -> None:
    """Measured ambient temperature vs the ISA isothermal stratosphere."""
    section("2. ISA temperature vs measured")
    alt = df["altitude"].to_numpy()
    t_meas = df["temperature"].to_numpy()
    t_isa = np.array([atmo.temperature(float(z)) for z in alt])
    err = t_isa - t_meas

    print(f"  measured T             {t_meas.min():.1f} - {t_meas.max():.1f} K "
          f"(median {np.median(t_meas):.1f})")
    print(f"  ISA T_TROPOPAUSE       {T_TROPOPAUSE:.2f} K (constant above 11 km)")
    print(f"  ISA error (median)     {np.median(err):+.1f} K")
    print(f"  ISA error (p5..p95)    {np.percentile(err, 5):+.1f} .. {np.percentile(err, 95):+.1f} K")

    band = (alt >= ALT_SAFE_MIN) & (alt <= ALT_SAFE_MAX)
    if band.any():
        print(f"  -- within our operating band, n={band.sum():,}")
        print(f"     measured T median   {np.median(t_meas[band]):.1f} K")
        print(f"     ISA error median    {np.median(err[band]):+.1f} K")
        print(f"     spread (p5..p95)    {np.percentile(t_meas[band], 5):.1f} .. "
              f"{np.percentile(t_meas[band], 95):.1f} K")
    print(f"\n  Note: SUPERHEAT_DAY = {SUPERHEAT_DAY:.1f} K is applied on top of ISA ambient,")
    print("  so an ISA ambient error feeds straight into gas temperature and buoyancy.")


def check_vertical_rates(df: pd.DataFrame) -> None:
    """Real ascent/descent rates, from finite differences of altitude per flight."""
    section("3. Vertical rates (measured)")
    rates = []
    for _, g in df.groupby("flight_id", sort=False):
        g = g.sort_values("time")
        dz = g["altitude"].diff().to_numpy()[1:]
        dt = g["time"].diff().dt.total_seconds().to_numpy()[1:]
        ok = (dt > 0) & (dt < 300) & np.isfinite(dz)
        if ok.any():
            rates.append(dz[ok] / dt[ok])
    if not rates:
        print("  no usable samples")
        return
    v = np.concatenate(rates)
    absv = np.abs(v)
    print(f"  samples                {len(v):,} across {df['flight_id'].nunique()} flights")
    print(f"  vertical rate p50      {np.median(absv):.3f} m/s")
    print(f"  vertical rate p95      {np.percentile(absv, 95):.2f} m/s")
    print(f"  vertical rate p99.9    {np.percentile(absv, 99.9):.2f} m/s")
    print(f"  observed max           {absv.max():.2f} m/s")
    print(f"  our VEL_MAX clamp      {VEL_MAX:.0f} m/s")
    print(f"  -> clamp headroom      {VEL_MAX / max(absv.max(), 1e-9):.0f}x the fastest real sample")


def check_horizontal_winds(df: pd.DataFrame) -> None:
    """Balloon horizontal velocity is the local wind (Lagrangian tracer)."""
    section("4. Horizontal wind magnitudes (measured)")
    u = df["velocity_u"].to_numpy()
    w = df["velocity_v"].to_numpy()
    mag = np.hypot(u, w)
    mag = mag[np.isfinite(mag)]
    print(f"  |wind| p50             {np.median(mag):.1f} m/s")
    print(f"  |wind| p95             {np.percentile(mag, 95):.1f} m/s")
    print(f"  |wind| max             {mag.max():.1f} m/s")
    print("  our WIND_MAG_NORM      30.0 m/s (observation normaliser)")
    over = (mag > 30.0).mean() * 100.0
    print(f"  -> {over:.2f} % of real samples exceed the normaliser (would clip to 1.0)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--rows", type=int, default=2_000_000,
                    help="Rows to read (default 2M; use 0 for all).")
    args = ap.parse_args()

    if not args.csv.exists():
        sys.exit(
            f"Flight data not found at {args.csv}.\n"
            "Download loon-flights-2021Q2.csv.gz (36 MB) from "
            "https://zenodo.org/records/5119968 into EDA/Data/."
        )

    df = load(args.csv, args.rows or None)
    atmo = Atmosphere()
    print(f"Loon Q2-2021 | {len(df):,} rows | {df['flight_id'].nunique()} flights")
    check_pressure(df, atmo)
    check_temperature(df, atmo)
    check_vertical_rates(df)
    check_horizontal_winds(df)


if __name__ == "__main__":
    main()
