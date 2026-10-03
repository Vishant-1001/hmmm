# Data sources

All sources below were verified by direct inspection of the public WeatherBench 2 bucket
(`gs://weatherbench2`, anonymous read) on **2026-09-28**, and cross-checked against the
[WeatherBench 2 data guide](https://weatherbench2.readthedocs.io/en/latest/data-guide.html).
Machine-readable provenance is written to `artifacts/dataset_manifest.json` by the pipeline.

| Role | Store | Verified facts |
|---|---|---|
| Forecast (features) | `gs://weatherbench2/datasets/ifs_ens/2018-2022-64x32_equiangular_conservative.zarr` | ECMWF IFS ENS, 50 members (`number` 1..50), initialisations 00/12 UTC 2018-01-01 .. 2022-12-31 (3652), `prediction_timedelta` 0..360 h every 6 h (int64, attribute `units: hours`), levels 500/700/850 hPa, 64×32 conservative grid (5.625°), geopotential in m² s⁻². Zarr chunks `(4 inits, 50, 1 lead, 3, 64, 32)` ≈ 3.5 MB compressed. |
| Verification reference (labels only) | `gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr` | ERA5, 6-hourly to 2023-01-10T18, 13 levels, same 64×32 grid. Bucket contains the Copernicus licence. |
| Forecast wind / pressure (v2 DYN features) | same IFS ENS store | u, v at 500/700/850 hPa and mean sea-level pressure for exactly the cached initialisations; the 50 members are reduced to ensemble mean and std (ddof = 1) at download (`python -m forecast_bust.data.wb2 extras`). 457/457 blocks, 1,828 initialisations 2018-01-01 .. 2022-12-31, validated for alignment, variables, value ranges and completeness (`artifacts/extras_validation.json`). Anomalies use the ERA5 1990–2017 u/v/MSLP climatology. |
| Anomaly climatology | `gs://weatherbench2/datasets/era5-hourly-climatology/1990-2017_6h_64x32_equiangular_conservative.zarr` | ERA5 1990–2017, hour (0/6/12/18) × day-of-year. Pre-dates every experiment year → no leakage. |

## Why the 5.625° product (and not 1.5° or 0.25°)

The build machine's link to Google Cloud Storage measured **~1.07 MB/s** (single and
6-way parallel reads, measured 2026-09-28). Member-level IFS ENS chunks are:

| Product | Chunk | Cost for one initialisation, Day 1–10 |
|---|---|---|
| 240×121 (1.5°) | 1 init × 50 members × 8 leads × 3 levels, ~92 MB | ~550 MB (≈ 9 min) |
| 1440×721 (0.25°) | 1 init × 50 × 1 lead × 3 levels, global | several GB |
| **64×32 (5.625°)** | 4 inits × 50 × 1 lead × 3 levels, ~3.5 MB | **~9 MB (≈ 9 s)** |

Ensemble *members* are required (spread, IQR, skewness, sign agreement, rank histograms), so
the 1.5° ensemble-mean store alone is insufficient. The 64×32 product is therefore the only
member-level product that permits a multi-year chronological experiment on this link. The
consequences (one grid box per region; RMSE over a region reduces to the absolute error of the
box-mean Z500; failure signatures computed on a 7×7-box neighbourhood) are listed in
`docs/limitations.md`. The code is resolution-agnostic (`region_members`, `weighted_rmse`) and
would use 3–4 grid points per region on the 1.5° product.

## Sampling

The forecast store is downloaded chunk-wise in coarse-to-fine order (`block_order`), keeping
every 2nd chunk (`block_stride: 2`): two consecutive days (00 and 12 UTC) out of every four.
That gives ~365 initialisations per year. The exact set of cached chunks is recorded in
`artifacts/dataset_manifest.json`.

## Acquisition

```bash
scripts/download_all.sh          # resumable; ~4.5 h at 1 MB/s; ~4.5 GB on disk
python scripts/smoke_test.py     # GATE 1 real-data smoke test (streams directly from the bucket)
```

Nothing under `data/` is committed.

## NCMRWF

NCMRWF NEPS ensemble members are listed in the ECMWF ECDS TIGGE catalogue (origin `dems`, Z500, Day 1–10,
2018–2022). `forecast_bust.data.providers.NCMRWFTIGGEProvider` implements retrieval and verification, but **no NCMRWF file
has been retrieved** because there is no ECDS token in this environment. See `docs/ncmrwf_tigge_availability.md`.
No NCMRWF data is fabricated, and every result is evaluated on historical ECMWF IFS ENS data only.

## Licensing

ERA5: Copernicus licence (file `datasets/era5/LICENSE` in the bucket). IFS ENS: ECMWF data made
available through WeatherBench 2 — check the WeatherBench 2 terms before redistribution. Raw data
is never committed; only derived metrics and a small number of derived replay fields are.
