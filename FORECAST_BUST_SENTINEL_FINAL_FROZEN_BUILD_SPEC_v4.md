# Forecast Bust Sentinel — final frozen build specification, V4

## Pattern-aware forecast bust detection (SIH26079)

> Forecast Bust Sentinel estimates where and when the NWP forecast is likely to suffer a severe bust by combining
> unusually large Z500 error with a spatial pattern/phase failure.

```
existing IFS ENS forecast (WB2 64x32, 50 members) + ERA5 verification (labels only)
  -> prediction-time-safe forecast state (SPREAD ATM ENS PAT EVO MEM REC DYN; 70 features)
  -> ONE shared XGBoost classifier (hist, CPU; v2 locked params) -> raw probability
  -> isotonic calibration on the validation year -> calibrated pattern-aware bust probability, confidence
  -> region x Day 1-10 map, trajectory, priority queue
  -> TreeSHAP + causal analogue evidence (large-error / pattern-failure / support levels) + support/OOD
  -> blind replay -> reveal: normalized error vs Q90, local ACC vs Q10, both criteria, fingerprint
```

| Item | Frozen value |
|---|---|
| Target | `pattern_bust = norm_error > TRAIN Q90 AND local 3x3 ACC < TRAIN Q10` (region x lead x season) |
| Local ACC | centred area-weighted anomaly correlation, ensemble mean vs ERA5, anomalies vs ERA5 1990-2017 climatology |
| Pre-registration | `config/model_v4.yaml`, commit `47ce766` (before any result) |
| Dev gate | PROMOTE 11/11 (`artifacts/v4/dev/metrics.json`, commit `405d2aa`) |
| Final model | `v4-final-405d2aa`: train 2018-20, early stopping + calibration 2021, 604 trees |
| 2022 | evaluated once (commit `b34b3a5`); previously observed by v1/v2/V3, not a pristine project-wide test |
| Serving | `config.served_run()` -> `artifacts/v4`; `/api/health` model_type `pattern_aware_xgboost` |

## Measured (details `docs/evaluation_v4.md`)

| | dev-test 2021 | 2022 |
|---|---|---|
| Prevalence | 0.0103 | 0.0104 |
| AUPRC (lift) | 0.0230 (2.24) | 0.0255 (2.45) |
| ROC AUC | 0.674 | 0.714 |
| Brier skill / ECE | +0.003 / 0.002 | +0.007 / 0.0003 |
| Precision / recall at alert | 0.026 / 0.263 | 0.028 / 0.305 |
| ROC AUC gain vs same model on magnitude target | +0.033 [+0.012, +0.054] | +0.092 [+0.071, +0.111] |

Engineering: works. Scientific: passes the pre-registered development gate, but with low precision, weak skill on the
large-error component and a partly artifactual pattern component (`docs/limitations_v4.md`). Product: a triage aid.

## Archive (untouched, not used by V4)

v1/v2 (`artifacts/`, `artifacts/v2`), V3 quantile gradient boosting (`artifacts/v3`), BMA negative result
(`artifacts/bma/dev`). All were measured on the magnitude-only target.

## Hard stop

This was the final predictive-core change. No further model or target changes; 2022 cannot support another test claim.
