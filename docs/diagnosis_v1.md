# v1 audit and failure diagnosis

Produced by `scripts/audit_v1.py` → `artifacts/diagnosis/v1_audit.json` (2026-09-30).

## 1. Reproduction of the frozen v1 evaluation (exact)

Test predictions were regenerated from `models/models.joblib` on `data/interim/table.parquet` and scored
with scikit-learn directly, independently of the project's metric code.

| Model | AUPRC (reproduced = published) | ROC AUC | max \|Δp\| vs stored predictions |
|---|---:|---:|---:|
| B0 climatology | 0.0908 | 0.517 | 0 |
| B2 calibrated spread-only | 0.1360 | 0.609 | 0 |
| Sentinel (FULL) | 0.1360 | 0.609 | 0 |

FULL − B2: the largest absolute probability difference on the test set is 0, so the predictions are identical.
The published result stands: **no incremental skill over the spread baseline.**

## 2. Integrity checks (all pass)

* Target: `bust == (normalized_error > q_primary)` on all 1,169,920 rows, and `q_primary` equals the TRAIN-only
  Q90 per (region, lead day, season) exactly (max |diff| 0).
* Alignment: `valid_time == init_time + lead_day days` on every row. Lead days are exactly 1–10.
* Split: train inits 2018-01-01 → 2020-12-29; validation 2021; test 2022; row-level embargo purge. The latest
  TRAIN `valid_time` (2020-12-31 12 UTC) is before the first validation initialisation (2021-01-01 00 UTC), so
  no training label is verified after a validation forecast is issued.
* Hidden bust = bust ∧ spread ≤ TRAIN P25 holds on every row.
* Earlier tests cover the causal analogue memory, TRAIN-only normalisation and thresholds, and validation-only
  isotonic calibration.

## 3. Why hidden busts are not caught (validation 2021 only, no test data)

| Quantity (validation 2021) | Value |
|---|---:|
| Low-spread rows (spread ≤ TRAIN P25) | 24.0% of rows |
| Bust rate: low-spread / other | 5.0% / 9.8% |
| Share of all busts that are hidden (low-spread) | 13.8% |
| B2 alert threshold (10% FAR, chosen on validation) | 0.128 |
| Highest B2 probability of any low-spread row | 0.117 |
| Low-spread rows that can ever alert | **0** |

**Mechanism 1: structural.** A calibrated spread-driven probability is low wherever spread is low, because
busts really are about half as frequent there. With one global operating threshold, every low-spread row lies
below the threshold, so hidden-bust recall is 0 **by construction**, not because of a bug.

**Mechanism 2: weak within-regime signal.** Inside the low-spread regime (55,392 validation rows, 2,762
busts, base rate 0.050), no available forecast-time information separates busts well:

| Model (validation, low-spread rows only) | AUPRC | ROC AUC | lift over base rate |
|---|---:|---:|---:|
| B2 | 0.061 | 0.575 | 1.22× |
| best candidate group (M3 PCA pattern) | 0.064 | 0.582 | 1.29× |
| ALL groups | 0.065 | 0.596 | 1.29× |

The strongest single features within the regime (spread measures, Z500/Z700 anomaly) reach only |AUC − 0.5| ≤ 0.072.
A regime-specific threshold (10% FAR within low-spread rows, validation) would recover 17.6% of hidden busts,
but at a precision of 0.075 against a 0.050 base rate: about 1.5× chance, with roughly 12 false alarms per
detected bust.

## 4. Decision

* **The frozen v1 Sentinel is not changed.** Nothing in the diagnosis points to a defect in the target,
  alignment, split, calibration or threshold selection. The v1 result is kept as measured.
* **A regime-specific hidden-bust threshold is not adopted.** On validation its precision is only about 1.5× chance.
  Adopting it would add false alarms without a demonstrated operational benefit. This decision was made on
  validation data alone.
* **The justified next step is the one the specification already defines.** The geopotential-only features carry
  almost no within-regime information, and the missing information class is the flow itself: wind, shear,
  vorticity, divergence and MSLP (the v2 DYN group, spec §27 Group B). v2 evaluates it under the pre-registered
  protocol (`docs/v2_changes.md`). It reports within-low-spread AUPRC and hidden-bust recall at fixed FAR,
  and it applies the unchanged 5% materiality rule. If DYN does not pass validation, the result stays negative.
