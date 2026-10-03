# ERA5 London Model-Level Spike — 15 April 2021

## Scope

This is the production-source validation for Layer 2 weather. It uses the
**Complete ERA5 global atmospheric reanalysis** catalogue entry
(`reanalysis-era5-complete`), all 137 model levels, and the surface fields
needed to reconstruct pressure and geopotential. The request covers 15 April
2021, 00:00–23:00 UTC, a ±10° box around London, and a 0.25° grid. The first
12 hours were converted to the runtime cube.

A 13-pressure-level file for the same date, hours, area, and grid was retrieved
from the same catalogue entry solely for a conversion-fidelity comparison. It
is not part of the production corpus.

Local assets are gitignored:

- Model-level GRIB: `weather_data/raw/london-2021-04-15-ml.grib`
- Surface-support GRIB: `weather_data/raw/london-2021-04-15-surface.grib`
- Pressure comparison GRIB:
  `weather_data/raw/london-2021-04-15-pressure-comparison.grib`
- Runtime cube:
  `weather_data/cubes/london-2021-04-15T00-model-levels.npz`
- Model-level SHA-256:
  `13bf2a407045a9eba9fb143522f1d9591f8df90aee385e9ebaee12cf0a3d43d7`
- Surface-support SHA-256:
  `1739575df39d818d036700113619cd5288300ce9d4444eb018edb60bb6f3d27e`
- Pressure-comparison SHA-256:
  `ec14b1f02c6bb3748d84da8d1fd6aa457586019da0799c527c93dc8c9ec91f03`
- Runtime-cube SHA-256:
  `32115169595a7f69ad67959648375a206ace39c2c18be76ce907d0d0c492ab06`

## Integrity and physical checks

| Check | Result | Verdict |
|---|---:|---|
| Raw model dimensions | 24 times × 137 levels × 81 × 81 | Pass |
| Hybrid coefficients | All 138 half-level A/B pairs present | Pass |
| Cube dimensions | 13 times × 81 altitudes × 81 × 81 | Pass |
| Time grid | 0–43,200 s, hourly | Pass |
| Altitude grid | 10–30 km, 250 m | Pass |
| Finite fields | All wind, temperature, and pressure values finite | Pass |
| Pressure monotonicity | 100% of adjacent altitude pairs decrease | Pass |
| Same-date pressure agreement | p95 relative error 0.044%; max 0.085% in 15–25 km | Pass |
| Same-date temperature agreement | p95 0.033 K; max 0.103 K in 15–25 km | Pass |
| Same-date horizontal-wind agreement | p95 0.148 m/s; max 0.369 m/s vector error in 15–25 km | Pass |
| Same-date vertical-wind agreement | p95 0.00061 m/s; max 0.00248 m/s in 15–25 km | Pass |

The comparison resamples the model-level cube at the geometric altitudes of
the independent pressure-level fields. The small residual includes both
interpolation error and the way ERA5 represents fields on the two vertical
coordinates; it is not a validation of ERA5 against observations.

Across the full 15–25 km operational band:

- Temperature: 210.81–222.47 K; median 216.55 K.
- Pressure: 2.403–12.178 kPa; median 5.446 kPa.
- Density: 0.0394–0.1969 kg/m³; median 0.0878 kg/m³.
- Horizontal wind: 0.01–22.14 m/s; median 8.25 m/s; p95 13.37 m/s.
- No horizontal-wind values exceed the 30 m/s observation normaliser.
- Converted geometric vertical wind: −0.0616 to +0.0659 m/s; median
  +0.00029 m/s.
- Horizontal vector shear: median 2.52 and p95 5.43 m/s per vertical
  kilometre on the 250 m output grid.

These ranges are physically coherent for the lower stratosphere and show no
unit or sign failure.

## Vertical-resolution verdict

At the grid point centred exactly on London, every one of the 13 converted
hours has 25 native model levels inside 15–25 km. Their gaps range from 332 to
519 m, with a median of 382 m and p95 of 503 m. Native vector shear has a
median of 3.39 and p95 of 6.64 m/s per kilometre.

The earlier pressure-level spike had only four independent surfaces in the
same band, with gaps of roughly 2–3 km. The model-level source therefore
provides 24 independently resolved vertical intervals instead of three before
the common 250 m interpolation. It passes the production vertical-resolution
gate.

## Size and runtime

- Core raw inputs: 226.8 MiB (226.1 MiB model fields plus 0.7 MiB support).
- Optional pressure comparison: 19.7 MiB.
- Compressed runtime cube: 108.9 MiB.
- Runtime arrays in memory: 131.8 MiB; measured process peak while loading:
  248.6 MiB including Python and imported libraries.
- Cube load: 1.73 s on the development laptop.
- GRIB-to-cube conversion: under two minutes on the development laptop.

One cube per worker remains practical. Keeping a short cube cache is still
important; workers must not load a month-sized array.

## Horizontal-domain and runtime verdict

The ±10° request maps to approximately ±692 km east/west and ±1,112 km
north/south at London. Passive, random, and greedy-wind policies were each run
on five seeded 12-hour flights. All 15 completed with a 0.00% weather-clipped
fraction after the numerical guard was corrected.

The first run exposed a separate legacy limit: the old 500 km `XY_ABORT`
terminated five otherwise valid flights. Real ERA5 drift can reach that
distance without any numerical failure. The guard is now 10,000 km—beyond the
maximum displacement possible under the velocity clamp in a default 12-hour
episode—and also rejects non-finite positions. Weather-domain clipping remains
reported independently in per-step info, evaluation results, TensorBoard, and
benchmark tables.

One date cannot prove every sampled weather window fits the domain. The corpus
keeps the ±10° extent provisionally, and every baseline benchmark must retain a
zero or negligible clipping rate; a sampled date that violates that condition
forces a domain review before training.

## Conclusion

The model-level request, hybrid pressure reconstruction, hydrostatic height
integration, altitude interpolation, atmosphere coupling, runtime format, and
full-flight simulation pass the pilot. This removes the pressure-level
resolution blocker. The next data step is the sampled 2010–2025 corpus and its
frozen train/held-out manifest; the next independent code capability remains
the stateful day/night thermal model.
