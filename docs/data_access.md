# Data access

How the pipeline reaches real data. What the sources are and why this resolution was chosen:
`docs/data_sources.md`.

## Access path

| Step | Code | Notes |
|---|---|---|
| Open store | `forecast_bust.data.wb2._open` | `xarray.open_zarr(..., storage_options={"token": "anon"})` via `gcsfs`; no credentials |
| Forecast download | `python -m forecast_bust.data.wb2 forecasts` | one netCDF per 4-init store chunk → `data/cache/ens/block_NNNN.nc`; written to `.tmp` then renamed, so a killed download never leaves a partial block that is later skipped |
| Reference | `python -m forecast_bust.data.wb2 reference` | ERA5 Z500 → `data/cache/era5_z.nc` |
| Climatology | `python -m forecast_bust.data.wb2 climatology` | ERA5 1990–2017 → `data/cache/era5_clim_1990_2017_z.nc` |
| Assemble | `forecast_bust.data.assemble` | member statistics (mean, std, IQR, P10–P90, skewness, sign agreement), ERA5 at `valid_time = init + lead`, climatology at valid hour/day-of-year |

`scripts/download_all.sh` runs these in order and is resumable (cached blocks are skipped).
Progress goes to `logs/download.log`; the run is complete when the log ends with `DOWNLOAD_DONE`.

## Data contract

`forecast_bust.data.adapters.validate_forecast` checks, for any source, before use:
required dimensions, geopotential units (m² s⁻²), strictly ascending latitude, ascending
longitude in [0, 360), a regular latitude grid, and a minimum member count. `normalise_coords`
converts other coordinate conventions (descending latitude, −180..180 longitude) first.
Violations raise `DataContractError` (tested in `tests/test_adapters_failure_modes.py`).

## Adapters

`ForecastSourceAdapter` is the integration boundary (`list_initialisations`, `load`).

* `IFSENSWeatherBenchAdapter`: implemented, used for every result in this repository.
* `NCMRWFAdapter`: **interface only**. It raises `NotImplementedError` because no NCMRWF NEPS
  data was available to this build. No NCMRWF integration is claimed.

## Smoke test

`python scripts/smoke_test.py` streams one real initialisation and lead time straight from the bucket, runs
valid-time alignment → ERA5 match → regional RMSE, and writes `artifacts/smoke_test.json` (GATE 1).
