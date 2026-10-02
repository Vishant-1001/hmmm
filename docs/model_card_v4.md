# Model card — Forecast Bust Sentinel V4 (pattern-aware forecast bust)

**Status: production model (served by `config.served_run()` = artifacts/v4, `model_type` = `pattern_aware_xgboost`).**
Archived, not used by V4: v2 B2/Sentinel (`docs/model_card_v2.md`), V3 quantile gradient boosting
(`docs/model_card_v3.md`), BMA (`docs/model_card_bma.md`, negative result).

## What it estimates

For an existing IFS ENS forecast initialised at *T*, per region (64 native 5.625° boxes) and lead day 1–10: the
probability of a **pattern-aware forecast bust** — the region's normalized Z500 error is above the TRAIN Q90 for its
region × lead × season **and** the local spatial anomaly correlation (ACC) between the ensemble-mean forecast and ERA5
over the 3 × 3 boxes around it is below the TRAIN Q10. It does not forecast weather and does not replace NWP.

| Output | Meaning |
|---|---|
| `raw_probability` | XGBoost probability, before calibration |
| `calibrated_bust_probability` (= `bust_probability`) | validation-only (2021) isotonic calibration of the above |
| `confidence` | LOW if ≥ the validation 10%-FAR alert threshold (0.0205); MODERATE if ≥ the training event rate (1.3%); else HIGH |
| `magnitude_criterion`, `pattern_criterion`, `historical_support` | rates of large-error / pattern-failure / pattern-aware busts among the nearest **verified historical** analogues vs training rates (HIGH ≥ 2×, ELEVATED > 1.25×) — evidence, not model output |

## Model

* ONE shared `xgboost.XGBClassifier` (tree_method hist, CPU) via the existing `CalibratedGBM`; v2's locked Sentinel
  configuration (learning_rate 0.05, subsample 0.8, reg_lambda 10.0, n_estimators 3000, max_depth 3, min_child_weight 50, colsample_bytree 0.5); early stopping and isotonic calibration on
  2021; 604 trees. No search, no stacking, no other model's output as input.
* Inputs: the existing 70 prediction-time-safe features (SPREAD ATM ENS PAT EVO MEM REC DYN).
  `local_acc`, thresholds, labels and every verification quantity are excluded (FORBIDDEN_INPUTS + tests).
* Training 2018–2020 (698,624 region-days), calibration 2021, evaluation 2022.
* Explanations: TreeSHAP of the V4 booster (associations, not causes) + analogue evidence.
* Model-level gain importance: 500 hPa zonal wind anomaly (m/s) (0.145), recent verified normalized error, domain (0.046), spread percentile vs training (same region/lead/season) (0.042), pattern PC5 coordinate (0.037), spread relative to the bust threshold (spread / TRAIN Q90 error, m/m) (0.037), region (0.034), zonal Z500 gradient (m/100 km) (0.034), 500 hPa meridional wind anomaly (m/s) (0.033).
* Artifacts: `models/v4/models.joblib`, `metadata.json`; tracked copy `artifacts/v4/demo/model/v4/`; thresholds
  `artifacts/v4/acc_thresholds.json` (Q10) and `artifacts/v3/thresholds.json` (Q90, unchanged since v2).

## Measured performance (`docs/evaluation_v4.md`)

| | dev-test 2021 | 2022 (final; not a pristine project-wide test) |
|---|---|---|
| Event prevalence | 0.0103 | 0.0104 |
| AUPRC (lift) | 0.0230 (2.24) | 0.0255 (2.45) |
| ROC AUC | 0.674 | 0.714 |
| Brier skill vs climatology / ECE | +0.0027 / 0.0020 | +0.0070 / 0.0003 |
| Recall @10% FAR; precision at alert | 0.236; 0.026 | 0.251; 0.028 |
| Pattern hidden-bust recall | 0.003 | 0.023 |
| ROC AUC gain over the same model on the magnitude target | +0.033 [+0.012, +0.054] | +0.092 [+0.071, +0.111] |

## Claims policy

Allowed: we estimate the probability of a pattern-aware forecast bust; we identify regions and lead times where large
error and pattern failure are jointly more likely than usual; we compare the current forecast state with verified
historical forecast-error behaviour; we provide region-wise Day 1–10 confidence; we verify against ERA5.
Not supported: outperforming NCMRWF/ECMWF/commercial products or other SIH submissions; detecting every bust (recall at
alert ≈ 0.3, precision ≈ 0.03); knowing the physical cause of a bust; novelty of the target, the ACC criterion or XGBoost.
