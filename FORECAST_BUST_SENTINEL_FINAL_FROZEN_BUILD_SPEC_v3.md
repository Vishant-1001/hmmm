# Forecast Bust Sentinel — final frozen build specification, v3

## Flow-Dependent Forecast Error Intelligence (SIH26079)

```
existing IFS ENS forecast
  -> prediction-time-safe forecast-state features (SPREAD ATM ENS PAT EVO MEM REC DYN; 70)
  -> quantile gradient boosting: HistGradientBoostingRegressor(loss="quantile") x {q10,q25,q50,q75,q90,q95}
  -> conditional regional normalized Z500 error distribution (rearranged if quantiles cross)
  -> central error (q50) + central range (q25-q75) + upper tail (q95)
  -> estimated exceedance probability of the TRAIN Q90 bust threshold (piecewise-linear CDF, exponential tails)
  -> validation-only isotonic calibration -> calibrated bust probability -> confidence = 1 - p
  -> region x Day 1-10 risk map, priority queue
  -> causal historical memory evidence + support/OOD
  -> blind replay -> reveal -> ERA5 verification + failure fingerprint (post-verification only)
```

| Component | v3 status |
|---|---|
| Quantile gradient boosting (`models/qgb.py`) | **production model** |
| QRF (`quantile-forest`) | retired: repeated OOM on the 8 GB dev machine; dependency removed |
| B2 spread-only XGBoost classifier | archived benchmark only; not a feature, fallback, calibrator or output |
| v2 Sentinel XGBoost classifier | retired from production; artifacts kept under `artifacts/v2`, `models/v2` |
| Data, alignment, regional error, normalisation, Q90/Q95 thresholds, features, memory, support, fingerprint | unchanged (thresholds verified identical to v2) |

## Frozen configuration (`config/model_v3.yaml`)

* Quantiles 0.10 0.25 0.50 0.75 0.90 0.95; one shared configuration: learning_rate 0.05, max_iter 200,
  max_leaf_nodes 31, min_samples_leaf 100, l2_regularization 1.0, early_stopping False, seed 20260928.
* Chosen on the dev split only (train 2018–19, val 2020, dev-test 2021) from an 8-config grid
  (learning_rate {0.03, 0.05} × max_leaf_nodes {15, 31} × min_samples_leaf {50, 100}); rule: max validation
  AUPRC among configs within 1% of the best mean pinball loss.
* Final fit: train 2018–2020, calibration 2021, evaluated once on 2022. Model commit `045a9e8`
  (experiment `v3-final-045a9e8`); result commit `45c63b9`.
* Bust = `norm_error > q_primary` (TRAIN Q90 per region × lead × season); Q95 kept as sensitivity.
* Hidden bust = bust ∧ spread percentile ≤ TRAIN 25th percentile (unchanged).

## Measured results (see `docs/evaluation_v3.md`)

| | dev-test 2021 | 2022 (V3 first evaluation; previously read by v1/v2) |
|---|---|---|
| AUPRC calibrated (B0) | 0.1396 (0.0932) | 0.1449 (0.0908) |
| Brier (climatology) | 0.0783 (0.0796) | 0.0785 (0.0802) |
| ECE | 0.0132 | 0.0045 |
| MAE(q50) / mean pinball | 0.204 / 0.0641 | 0.201 / 0.0635 |
| Coverage q10..q95 | .115 .284 .543 .782 .912 .956 | .113 .280 .540 .784 .915 .957 |
| Hidden-bust recall | 0.085 | 0.075 |
| B2 reference | 0.1578 (fresh, same rows): v3 − B2 = −0.018 [−0.027, −0.011] | v2 historical 0.1434 (not controlled) |

Engineering: works. Scientifically: valid, calibrated, weak signal above climatology. Outperforms B2: **no**.

## Serving

`config.served_run()` → `artifacts/v3`; demo engine loads `artifacts/v3/demo/model/v3`
(`V3PredictiveModel`); `/api/health` reports `served_run`, `model_type` = `quantile_gradient_boosting` and
`experiment_id`. Static `/api/forecast`, `/api/replay` routes serve the archived v2 replay report, labelled.

## Reproduce

`scripts/run_v3.sh dev` then `scripts/run_v3.sh final` (needs the data cache). Software versions, features,
schema, params, splits, threshold and calibration metadata: `artifacts/v3/demo/model/v3/metadata.json`,
`artifacts/v3/experiment_manifest.json`; environment `requirements-lock.txt`.

## Rules carried forward

No tuning after 2022. 2022 has now been read three times (v1, v2, v3) and cannot support a test claim for
any future change. One predictive model family; no stacking, no second classifier, no model zoo.
