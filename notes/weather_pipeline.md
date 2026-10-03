# Layer 2 Weather Pipeline

## Decision

ERA5 reanalysis is the authoritative deterministic weather source for Layer 2.
ECMWF WeatherGenerator and a custom Loon-style VAE are not dependencies. They
remain optional future backends if the historical catalogue proves too narrow;
that decision must be driven by held-out results rather than assumed up front.

The analytic wind field and ISA atmosphere remain available for unit tests,
debugging, and ablations. They are no longer the intended Layer 2 training
distribution.

## Runtime architecture

Training workers consume compact `.npz` weather cubes, never CDS, GRIB, NetCDF,
or an ML weather model directly. Every field has shape
`(time, altitude, y, x)` in a local east/north/up frame and SI units.

Required fields are eastward and northward wind (`u`, `v`, m/s), ambient
temperature (K), and ambient pressure (Pa). Geometric vertical wind (`w`, m/s)
and net radiative forcing (W/m²) are optional fields supported by the format.

`ReanalysisWeatherProvider` performs deterministic four-dimensional
interpolation. `ReanalysisAtmosphere` exposes the altitude-based interface used
by the balloon while the environment supplies horizontal position and time.
The Numba physics path accepts sampled temperature and density explicitly; it
does not call the hard-coded ISA functions.

## ERA5 conversion

Install the offline tooling separately from runtime dependencies:

```bash
pip install -e '.[dev,weather]'
```

CDS requires an account, acceptance of the dataset licence, and a
`~/.cdsapirc` API token. Start with a one- or two-day spike:

```bash
python scripts/download_era5.py \
  --year 2024 --month 3 --days 15 16 \
  --output weather_data/raw/london-2024-03-15.nc

python scripts/prepare_era5.py \
  --input weather_data/raw/london-2024-03-15.nc \
  --output weather_data/cubes/london-2024-03-15.npz
```

The preprocessing step does not treat pressure level as altitude. At each
time/horizontal location it converts geopotential to height, interpolates wind
and temperature by height, interpolates pressure in log space, and converts
ERA5 pressure velocity `omega` to geometric vertical velocity using
`w = -omega/(rho*g)`.

The first real-data spike is complete; see
`notes/era5_spike_2024-03-15.md`. The regular pressure-level product has only
four native levels inside 15–25 km, separated by roughly 2–3 km. The converter
is numerically faithful, but interpolating those profiles onto the 250 m
observation grid does not create information. Production acquisition should
therefore move to ERA5's 137 model levels and validate the additional resolved
shear before a training corpus is built.

The model-level path uses the separate **Complete ERA5 global atmospheric
reanalysis** catalogue entry (`reanalysis-era5-complete`), not the single-level
or pressure-level entries. It retrieves GRIB so the hybrid A/B coefficients are
kept in the file header. Temperature, specific humidity, surface geopotential,
and logarithmic surface pressure reconstruct model-level pressure and height;
wind and pressure velocity then follow the same geometric-altitude conversion
as the runtime cube:

```bash
python scripts/download_era5_model_levels.py \
  --year 2021 --month 4 --days 15 \
  --model-output weather_data/raw/london-2021-04-15-ml.grib \
  --surface-output weather_data/raw/london-2021-04-15-surface.grib \
  --pressure-comparison-output \
    weather_data/raw/london-2021-04-15-pressure-comparison.grib

python scripts/prepare_era5_model_levels.py \
  --model-input weather_data/raw/london-2021-04-15-ml.grib \
  --surface-input weather_data/raw/london-2021-04-15-surface.grib \
  --start 2021-04-15T00:00 \
  --output weather_data/cubes/london-2021-04-15T00-model-levels.npz
```

The bounded pilot is ±10° at 0.25° resolution. See
`notes/era5_model_level_spike_2021-04-15.md` for its passing validation results
and `notes/weather_corpus_plan.md` for the sampled-year design. The optional
pressure comparison is for pilot validation only; omit it from corpus
downloads.

## HPC acquisition and training workflow

The corpus driver is dry-run by default and has two fixed profiles:

```bash
# Two-cube, leakage-safe integration test: 2018 train / 2021 held out.
python scripts/acquire_era5_corpus.py --profile smoke --plan
python scripts/acquire_era5_corpus.py --profile smoke --execute

# Initial 2010--2025 sampled corpus: 104 train / 24 held out.
python scripts/acquire_era5_corpus.py --profile sampled --plan
python scripts/acquire_era5_corpus.py --profile sampled --execute
```

Run `--execute` only on the HPC login/data-transfer node after installing the
`weather` extras and configuring CDS credentials. The driver batches requests
by year and start hour, requests only the 13 hourly endpoints needed by each
12-hour scenario, resumes existing raw/cube files, converts them, and writes
either `london-smoke.json` or `london-spring.json`. Noon windows automatically
include 00:00 on the following date.

Inspect any prepared cube without opening GRIB:

```bash
python scripts/plot_weather_cube.py \
  --input weather_data/cubes/london-2018-04-15T00-model-levels.npz \
  --output weather_data/plots/london-2018-04-15.png
```

The plot shows a horizontal wind field, a London time/height section, and
optional passive/greedy trajectories through the same runtime environment.

The Phase 2 SLURM script is intentionally a 60,000-step smoke run:

```bash
sbatch jobscripts/train_phase2_era5_slurm.sh
```

It uses one worker and one held-out scenario. This is enough to cross the
50,000-step learning warm-up, but it is not a performance result. After the
sampled profile is acquired and checked, edit the flags identified in the job
header before production training, including increasing the two-hour walltime
using the measured smoke runtime. Phase 1 remains available independently as
`jobscripts/train_phase1_slurm.sh`.

After either run, benchmark the corresponding held-out split with the same
manifest used for training (use one scenario for smoke and 12 for sampled):

```bash
python main.py --benchmark --dim 3 --balloon-type zero_pressure \
  --n-scenarios 1 \
  --weather-manifest weather_data/manifests/london-smoke.json
```

Analytic and ERA5 checkpoints are isolated. Supplying `--weather-manifest`
places checkpoints and TensorBoard data below the model's `era5/` directory;
old 143-wide checkpoints are skipped with an explanation rather than crashing
a 144-wide Layer 2 benchmark. `--baseline-only` avoids checkpoint loading
entirely.

## Scenario manifests and leakage control

Build manifests from one or more cubes:

```bash
python scripts/build_weather_manifest.py \
  weather_data/cubes/*.npz \
  --heldout-years 2023 2024 2025 \
  --output weather_data/manifests/london-spring.json
```

Multi-window cubes are divided into non-overlapping 12-hour windows by default,
so a one- or two-day source need not be duplicated for every scenario. Keep
source cubes short: each worker loads a complete cube into memory, so monthly
files are a poor runtime format. Every record pins its source and SHA-256
digest, UTC start and offset, whole-year split, wind features, and an
easy/medium/hard band assigned by within-split score quantile.
Cube paths are stored relative to the manifest so the asset tree can be copied
unchanged to the HPC. Workers verify each source digest on first use and reject
scenarios shorter than the configured episode instead of freezing the final
weather timestep.

A calendar year may not occur in both splits. This prevents adjacent windows
from the same synoptic event appearing as nominally independent train and test
episodes. The difficulty score is only a transparent stratification proxy;
after baselines are run, their TWR should be added to the analysis before the
benchmark is considered final.

The planned held-out years are 2019–2021. They overlap the mature end of the
published Google/Loon flight record, while the locally available Q2-2021 file
provides a directly inspectable atmosphere/flight validation set. This is a
temporal validation link only: Loon's global tracks are not treated as London
weather observations.

Training and evaluation select the appropriate manifest split automatically:

```bash
python main.py --train --dim 3 \
  --weather-manifest weather_data/manifests/london-spring.json
python main.py --benchmark --dim 3 \
  --weather-manifest weather_data/manifests/london-spring.json
```

## Radiative observability decision (point 6)

Variable cloud/radiative forcing cannot affect buoyancy while remaining hidden
without making Layer 2 partially observable. The observation therefore gains
one signed scalar, `radiative_forcing_norm`, taking the layout from 143 to 144
values. The scale is ±1000 W/m².

It is zero for analytic weather and ERA5 cubes without a forcing field. The
Layer 2.4 thermal model will populate it with the same net environmental
forcing used by its energy balance. This is deliberately net forcing rather
than raw cloud fraction: at stratospheric altitude most clouds are below the
balloon and principally modify upwelling longwave and reflected shortwave, not
the direct solar beam.

This one-time correction invalidates Layer 1 network checkpoints. Layer 2
already requires retraining, and preserving an incomplete input contract would
be worse than making the incompatibility explicit before the expensive run.

## Current validation boundary

Synthetic tests cover persistence, interpolation, time offsets, vertical wind,
real-atmosphere balloon dynamics, manifest leakage guards, clipping telemetry,
and observation population. The 15 April 2021 model-level pilot passes
integrity, same-date pressure-level agreement, vertical resolution, physical
range, storage, and full-flight runtime checks. Its ±10° domain has zero
clipping across 15 pilot baseline flights, but that result must continue to be
checked for every sampled date. A real ERA5 benchmark cannot be claimed until
the sampled corpus is built and split into frozen manifests.
