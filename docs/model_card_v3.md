# Model card — Forecast Bust Sentinel v3 (quantile gradient boosting)

**Status: production model (served by `config.served_run()` = artifacts/v3).** v2's B2/Sentinel XGBoost
classifiers are archived benchmarks (`docs/model_card_v2.md`); the abandoned QRF attempt is described in
`docs/scientific_methodology.md` §0.8.

## What it predicts

For an existing IFS ENS forecast initialised at *T*, per region (64 native 5.625° boxes, 5°S–40°N,
60°E–105°E) and lead day 1–10: the conditional quantiles q10, q25, q50, q75, q90, q95 of the **normalized
regional Z500 forecast error** (`norm_error`), and from them the probability that the error exceeds the
project's fixed bust threshold (TRAIN Q90 per region × lead × season). It does not forecast weather and
does not replace NWP; it is a reliability layer over an existing forecast.

| Output | Meaning |
|---|---|
| `expected_error` = q50 | central (median) predicted error, × normal; not a conditional mean |
| `uncertainty_low`/`uncertainty_high` = q25/q75 | central predicted error range; not a confidence interval |
| `upper_tail_error` = q95 | upper-tail predicted error; not a maximum possible error |
| `estimated_exceedance_probability` | P(error > threshold) estimated from the six quantiles (not an exact CDF) |
| `calibrated_bust_probability` | the above after validation-only isotonic calibration; drives alerts |
| `confidence` | 1 − calibrated bust probability |

## Model

* Estimator: `sklearn.ensemble.HistGradientBoostingRegressor(loss='quantile')`, one estimator per quantile, fitted sequentially, one shared
  configuration: learning_rate 0.05, max_iter 200, max_leaf_nodes 31, min_samples_leaf 100, l2_regularization 1.0; early_stopping False;
  seed 20260928. scikit-learn 1.9.1 (pinned).
* Inputs: 70 prediction-time-safe features in groups SPREAD + ATM + ENS + PAT + EVO + MEM + REC + DYN
  (`features/build.py`, unchanged). No B2 output, no verification quantity, no future forecast cycle.
* Crossing: rearrangement (row-wise sort) at prediction time; raw crossing on 2021 validation
  0.36% of rows.
* Exceedance: piecewise-linear CDF between the quantiles, exponential tails beyond q10/q95.
* Calibration: isotonic regression on 2021 validation (230,656 rows,
  125 knots), frozen.
* Training: 2018–2020 (698,624 region-days). Hyper-parameters chosen on the
  dev split only (train 2018–19 / val 2020 / dev-test 2021).
* Artifacts: `models/v3/` (local) and the identical tracked copy `artifacts/v3/demo/model/v3/`
  (`qgb_q10..q95.joblib`, `calibration.joblib`, `metadata.json` with features, schema, params, quantiles,
  threshold/calibration metadata, splits, software versions, commit `045a9e8`).
  Loader: `forecast_bust.models.qgb.V3PredictiveModel`.

Model-level importance (permutation, q90 estimator, validation; association only, not per-row):
lead day (0.0115), season (0.0017), spread percentile vs training (same region/lead/season) (0.0015), domain-mean spread (m) (0.0009), recent verified normalized error, neighbours (0.0007), 500 hPa vector-wind spread (m/s) (0.0005), region (0.0005), analogue 90th pct normalized error (0.0004), spread relative to domain mean (0.0002), pattern PC3 coordinate (0.0001).

## Measured performance (details: `docs/evaluation_v3.md`)

2022, V3 first evaluation of this period (2022 was previously read by v1/v2; not an untouched test):
AUPRC 0.1449 (base rate 0.088; B0 climatology 0.0908),
ROC AUC 0.636, Brier 0.0785 (climatology 0.0802), ECE 0.0045,
hidden-bust recall 0.075; MAE(q50) 0.201,
coverage q10..q95 0.113 / 0.280 / 0.540 / 0.784 / 0.915 / 0.957.

On the controlled dev comparison v3 is **below** the archived spread-only B2 classifier (−0.018 AUPRC,
95% CI [−0.027, −0.011]). v3 is a calibrated, distribution-producing reliability model with a weak
signal; it is not shown to outperform B2, ECMWF products, commercial systems or other SIH submissions.

## Intended use and limits

Decision support for forecasters reviewing an existing Day 1–10 forecast, on historical replay cases.
Not validated for operational use, other variables, other grids or NCMRWF data (adapter only). See
`docs/limitations.md`.
