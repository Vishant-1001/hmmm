# NCMRWF NEPS via TIGGE/ECDS: what has been checked

Status: **NCMRWF/TIGGE CATALOGUE CONFIRMED, RETRIEVAL NOT YET VERIFIED** (as of 2026-10-04)

Machine-readable record: `artifacts/ncmrwf_tigge_availability.json`. Regenerate it with
`.venv/bin/python scripts/ncmrwf_tigge_check.py`.

This report keeps four levels of evidence apart:

| Level | Meaning | Status here |
|---|---|---|
| Catalogue availability | The ECDS catalogue and constraints list the data | **Confirmed** (public endpoint, 2026-10-03 21:05 UTC) |
| Retrieval demonstrated | A GRIB file was downloaded and checked field by field | **Not demonstrated**: no ECDS token in this environment (HTTP 401) |
| Historical coverage demonstrated | The same check passed for one date in each year 2018–2022 | **Not demonstrated** |
| Operational access | Real-time or near-real-time NEPS feed | **Not addressed.** TIGGE is a delayed research archive |

## 1. What was verified

* The ECMWF Data Store (ECDS) dataset `tigge-forecasts` is reachable at
  <https://ecds.ecmwf.int/datasets/tigge-forecasts> (HTTP 200). Its retrieve process lists
  `ncmrwf` as an origin.
* The public **constraints** endpoint (`POST https://ecds.ecmwf.int/api/retrieve/v1/processes/tigge-forecasts/constraints`)
  was queried with origin = ncmrwf, forecast_type = perturbed_forecast, level_type = pressure,
  variable = geopotential_height, level = 500 hPa, once per year. For **each of 2018, 2019, 2020, 2021 and 2022** it lists:
  all 12 months, cycles 00 and 12 UTC, `geopotential_height` at `500_hpa`, `perturbed_forecast`,
  and lead times 0–240 h every 6 h, so Day 1–10 is covered. NCMRWF entries begin in August 2017 (00 UTC only that year).
* These are **catalogue facts**. The constraints endpoint does not resolve individual missing days,
  members or fields, so it does not show that every date, member and field is complete.

## 2. What was actually retrieved

**Nothing.** The smoke request was sent to the ECDS execution endpoint without credentials. ECDS answered
`HTTP 401 {"title": "permission denied", "detail": "authentication required"}` (the trace id is in the JSON).
No NCMRWF GRIB file exists in this repository or its caches.

## 3–6. Provider/origin, variables, levels and leads requested by the smoke test

| Item | Value |
|---|---|
| Archive / dataset | TIGGE, ECDS dataset `tigge-forecasts` |
| Origin | `ncmrwf` (MARS origin `dems`, GRIB2 centre 29, "New Delhi") |
| Forecast type | `perturbed_forecast` (ensemble members) |
| Variable | `geopotential_height` (TIGGE param 156, units gpm) |
| Level | 500 hPa |
| Cycle | 2021-07-15 00 UTC (pre-declared; not chosen from any result) |
| Leads | 24, 48, 72, 96, 120, 144, 168, 192, 216, 240 h |
| Area (N/W/S/E) | 42, 56, −8, 108: the 5.625° target cells (−5.6…39.4 N, 59.1…104.1 E) plus padding |
| Format | GRIB |

The exact request is stored under `smoke_request` in the JSON manifest.

## 7. Actual member count

**Unknown.** NCMRWF documentation describes the operational NEPS as 23 members (1 control + 22 perturbed).
The code does **not** use that number. The member count, the member IDs and the finite-ensemble correction in
`spread_thr_ratio` (√(1+1/M)) all come from the returned GRIB. Once a file is retrieved, the documented
and archived counts are both logged in the bundle metadata (`notes`).

## 8. Tested years and dates

Pre-declared coverage probes: 15 July 00 UTC of 2018, 2019, 2020, 2021 and 2022, at Day 1, 5 and 10.
**None was attempted** because there is no token. Catalogue listing for these years is recorded under `catalogue.years`.

## 9. Missing dates and fields

Not measurable without retrieval. Catalogue-level: nothing is missing for Z500 perturbed members in 2018–2022.

## 10. Retrieval mechanism

`forecast_bust.data.providers.NCMRWFTIGGEProvider` uses the current ECDS/CDS-API engine through `cdsapi`
(`https://ecds.ecmwf.int/api`). It does not use the old ECMWF Web API. The GRIB is parsed with cfgrib/ecCodes. Each
file is then checked:

* GRIB centre must be `dems`/29 (a file from any other origin is rejected)
* `gh` at 500 hPa, units gpm, converted to geopotential (m² s⁻²) for the canonical contract
* member IDs and steps are read from the file
* coordinates are normalised (ascending latitude, longitude in [0, 360))
* missing values (GRIB bitmap → NaN) are excluded point by point

The native field is then aggregated onto the 5.625° grid used by the ECMWF benchmark and ERA5 verification. The
method is a cos-latitude-weighted mean of the native points inside each cell, and a cell with no valid point stays NaN.
`ncmrwf_check.verify_retrieval` runs the 16-point checklist (origin, non-synthetic, members, Z500, init/valid
times, all leads, domain, coordinates, units, plausibility, missing values, mean, spread). These checks are tested on
eccodes-written GRIB **test fixtures** (`tests/test_ncmrwf_provider.py`). The fixtures are not NCMRWF data.

## 11. Authentication and access requirements

1. An ECMWF account (free registration).
2. Log in to ECDS and accept the ECDS terms of use and the **TIGGE licence** on the dataset page.
3. Copy the personal access token into `~/.cdsapirc`:
   ```
   url: https://ecds.ecmwf.int/api
   key: <personal-access-token>
   ```
   or export `FBS_ECDS_KEY=<token>`.
4. `pip install -r requirements-ncmrwf.txt`, then run `.venv/bin/python scripts/ncmrwf_tigge_check.py`.

With a token, the script runs the smoke retrieval, the checklist, the five yearly probes and a B2 smoke
inference with real ERA5 verification labels for that single case. It then rewrites the JSON, sets the status line and
updates the `providers` block of `artifacts/dataset_manifest.json`.

## 12. Known limitations

* **No retrieval was performed.** Nothing about NCMRWF data contents is confirmed beyond the catalogue.
* TIGGE carries an archive delay and limited fields and resolution. It is a research archive, not operational access.
* Regridding to 5.625° gives up NCMRWF's native resolution. This is intentional so that the B2 inputs, regions and
  ERA5 verification match the existing benchmark exactly.
* The frozen B2 was trained on ECMWF IFS ENS spread. On NCMRWF inputs, `b2_ncmrwf` uses ECMWF TRAIN
  references and the ECMWF validation calibrator, and every output is labelled
  `"... (cross-system transfer; NOT validated for this provider)"`. An NCMRWF calibrator must be fitted on an
  NCMRWF **validation** period only. The code refuses one fitted on anything else.
* A statistically meaningful NCMRWF benchmark needs years of daily retrievals (train/validation/test) with the same
  ERA5 verification. No such benchmark exists, and none will be built from a handful of dates.

## 13. Can NCMRWF replace synthetic B2 inputs?

There are **no synthetic B2 inputs to replace.** The B2 and V4 scientific path already runs on real ECMWF IFS ENS
(50 members) verified against ERA5. Synthetic arrays exist only as labelled test fixtures and in
`SyntheticDemoProvider` (`synthetic: true`, `demo_only: true`), which no pipeline, benchmark or served view uses.

Whether NCMRWF can serve as an **additional real B2 provider** is **not yet established**. The provider path is
implemented and has been checked to reproduce the ECMWF benchmark B2 exactly (below). Real NCMRWF retrieval has not
been demonstrated.

## Provider-path equivalence (real ECMWF data)

`scripts/build_b2_references.py` runs `ECMWFResearchProvider → b2_provider.run_b2("b2_ecmwf")` on the first
validation initialisation (2021-01-01 00 UTC, 640 region-days) and compares the output with the v2 pipeline's stored
values. `spread_m`, `spread_pct` and `spread_thr_ratio` are identical (max |Δ| = 0), and the B2 probability matches to
max |Δ| = 1.5 × 10⁻⁸ (`artifacts/b2/ecmwf/b2_manifest.json`). The provider abstraction therefore leaves B2 unchanged.

Cost of the B2 path (real ECMWF, 4 validation initialisations, this machine):
spread + features + B2 ≈ **76 ms per initialisation** (640 region-days), reading one cached initialisation ≈ 0.8 s,
peak RSS **≈ 397 MB** for the whole process. B2 is not retrained: it is the frozen v2 booster, with no QRF, no
quantile reconstruction and no hyperparameter search.

## Optional second phase (not started)

The catalogue also lists NCMRWF `temperature`, `u/v_component_of_wind` and `specific_humidity` on
200–1000 hPa, plus `mean_sea_level_pressure`. These would belong to a **separate** "NCMRWF forecast-state" experiment,
never to B2. Nothing has been retrieved or implemented for it.
