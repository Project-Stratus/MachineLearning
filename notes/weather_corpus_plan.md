# Layer 2 Weather Corpus Plan

## Fixed decisions

- Operational site: London, centred on 51.5074° N, 0.1278° W.
- Truth source: `reanalysis-era5-complete`, using all 137 ERA5 model levels.
- Season: March and April, matching the initial project scope.
- Runtime windows: deterministic 12-hour weather scenarios.
- Weather assets are local/HPC files under `weather_data/`, not storage attached
  to the Copernicus account and not Git artifacts.

## Held-out years

Use **2019, 2020, and 2021** as complete-year held-out weather splits.

The published Loon Stratospheric Sensor Data covers the project's full flight
history from 2011 through May 2021. These final three years represent mature
operations, and the 2020 Nature report documents real controller deployments
and a 39-day Pacific experiment from this period. Most importantly, the local
EDA asset contains 826,541 telemetry samples from 18 flights between 1 April
and 18 May 2021, so 2021 can be checked without acquiring the 127-million-row
full archive.

This is a temporal correspondence for external atmosphere/flight validation,
not a claim that the Loon balloons flew over London. The London benchmark and
the globally distributed Loon telemetry answer different questions and must
not be compared as if their wind fields were colocated.

## Sampled corpus

The 15 April 2021 model-level pilot passes the conversion, resolution, physical
range, runtime, and horizontal-clipping gates. See
`notes/era5_model_level_spike_2021-04-15.md`.

Use four dates per month (1, 8, 15, and 22 March and April) in each year:

- Train: 2010–2018 and 2022–2025 (13 years, 104 sampled dates).
- Held out: 2019–2021 (3 years, 24 sampled dates).

Dates are spaced by a week to reduce redundant adjacent synoptic states. Store
one 12-hour window per date, alternating start time to cover both halves of the
day: days 1 and 15 start at 00:00 UTC; days 8 and 22 start at 12:00 UTC. This
yields 104 train and 24 held-out scenarios without doubling the worker-cache
and long-term storage cost.

The pilot cube is 108.9 MiB compressed and 131.8 MiB in runtime arrays. At the
same compression ratio, 128 processed cubes require about 13.6 GiB. The core
pilot raw inputs were 226.8 MiB for a full 24 hours. The corpus driver requests
only the 13 endpoints needed per scenario, reducing the initial raw estimate to
about 15.4 GiB. Raw GRIB is reproducible from CDS: retain request metadata and
hashes, verify each cube, then raw inputs may be removed once the corpus is
backed up. Keeping both starts for every date remains a later expansion, not
the initial corpus.

## Pilot gate

The first model-level experiment used 15 April 2021, a ±10° London box, and a
0.25° grid. It passed because:

1. All 137 model levels, hybrid coefficients, and support fields are present.
2. Reconstructed pressure decreases monotonically and has p95 0.044% relative
   error against same-date pressure-level fields in the flight band.
3. The 15–25 km band has 25 native levels at 332–519 m spacing, rather than
   four pressure surfaces separated by roughly 2–3 km.
4. Temperature, pressure, density, horizontal wind, and converted vertical
   wind are physically coherent.
5. Fifteen passive/random/greedy flights complete with 0.00% horizontal
   clipping after removal of the stale 500 km numerical abort.
6. A 108.9 MiB compressed cube and 131.8 MiB worker array footprint are
   acceptable for a one-cube worker cache.

The ±10° extent is provisional across dates. Any non-negligible clipping in a
new cube's baseline runs blocks that cube from the frozen training manifest
until the acquisition extent is reviewed.
