# Model card — Forecast Bust Sentinel (generated from artifacts)

## Intended use

Research decision support: regional Day 1-10 probability that an existing NWP ensemble-mean Z500 forecast exceeds a project-defined large-error threshold; forecaster triage.

## Not intended for

* weather-event probabilities
* replacement NWP forecast
* local/point forecasts
* operational warnings without forecaster review
* NCMRWF forecasts (not trained on them)

## Data

* Forecasts: ECMWF IFS ENS (via WeatherBench 2) (`gs://weatherbench2/datasets/ifs_ens/2018-2022-64x32_equiangular_conservative.zarr`)
* Verification: ERA5 (WeatherBench 2) - verification reference analysis, not perfect truth
* Anomaly climatology: ERA5 1990–2017 (pre-dates all experiment years)
* Chronological split: embargo 2020-12-24..2021-12-31 (18 inits); test 2022-01-01..2022-12-31 (366 inits); train 2018-01-01..2020-12-29 (1096 inits); validation 2021-01-01..2021-12-28 (364 inits)

## Target and label

* bust = normalized_error > Q90 of TRAIN normalized_error within (region, lead_day, season). The 90th-percentile threshold is a project-defined operational bust criterion.
* Sensitivity: Q95 of the same TRAIN distribution

## Model

* XGBoost (hist, CPU) + isotonic calibration on validation; hyperparameters fixed in `config/model_v2.yaml`; early stopping on validation.
* Feature groups: SPREAD, ATM, ENS, PAT, EVO, MEM, REC, DYN; FULL uses ATM, EVO, MEM, REC (validation-selected); learner: standard.

## Validation / test methodology

* Development on a separate dev split (2018–19 / 2020 / 2021); final 2022 test evaluated once per run.
* Test-set history: 2022 was scored once before, by the v1 pipeline (commit be8123b; Sentinel == B2). All v2 choices (B2 spread/threshold ratio, DYN and tendency features, hyper-parameters, learner, feature groups) were made on the dev split (train 2018-19, validation 2020, dev-test 2021) and locked in this file before v2 scored 2022 once. No v2 decision used 2022 data, but 2022 is a second look, not an untouched test set.
* Primary metric AUPRC; verdict rule: material improvement iff 95% block-bootstrap CI of AUPRC(FULL)-AUPRC(B2) excludes 0 AND relative gain >= 5% (rule fixed before test evaluation).

## Results (TEST)

| Model | AUPRC | Brier | ROC AUC |
|---|---:|---:|---:|
| B0 | 0.091 | 0.0804 | 0.517 |
| B1 | 0.131 | — | 0.609 |
| B2 | 0.143 | 0.0788 | 0.619 |
| M1 | 0.143 | 0.0790 | 0.616 |
| M2 | 0.142 | 0.0789 | 0.614 |
| M3 | 0.146 | 0.0787 | 0.622 |
| M4 | 0.143 | 0.0787 | 0.621 |
| M5 | 0.143 | 0.0788 | 0.619 |
| M6 | 0.143 | 0.0788 | 0.622 |
| M7 | 0.145 | 0.0788 | 0.617 |
| ALL | 0.147 | 0.0788 | 0.622 |
| FULL | 0.143 | 0.0789 | 0.620 |
| EXP_RESIDUAL_B2 | 0.143 | 0.0788 | 0.618 |

Improvement over B2 material: **False** (AUPRC gain -0.0007, CI -0.0028..0.0017).

## Known failure modes and limitations

* Weak historical support for novel states (reported as evidence/support categories).
* Relationships are specific to ECMWF IFS ENS 2018–2022 and to the 5.625° grid; NWP upgrades cause drift.
* ERA5 is a reference analysis, not truth; Q90 bust criterion is project-defined; Z500 only.
* Full list: `docs/limitations.md`. Full metrics: `docs/evaluation.md`.
