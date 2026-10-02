# Evaluation — V4 (pattern-aware forecast bust)

Generated from `artifacts/v4/dev/{target_audit,metrics,proxy_diagnostic}.json` and `artifacts/v4/{target_audit,metrics}.json`
(`python -m forecast_bust.pipeline_v4`, `python -m forecast_bust.evaluation.run_v4`). Nothing is hand-entered.
Target, model and gate were pre-registered in `config/model_v4.yaml` (commit `47ce766`) before any ACC value or result.

## Three separate claims

* **Engineering success: yes.** Local ACC for all 1.17 M rows in seconds; V4 fit 26–60 s, < 2 GB RAM; live inference
  ~0.25 s per case; tests and browser walkthrough pass; served model reports `pattern_aware_xgboost`.
* **Scientific predictive success: partial.** The pre-registered development gate passed 11/11, and on 2022 V4 ranks
  pattern-aware busts clearly above climatology and the spread score (ROC AUC 0.714, lift 2.45). But (i) the event is rare
  (~1%) and precision at the alert threshold is ~3%; (ii) V4 ranks the *large-error* component weakly (ROC AUC 0.54);
  (iii) low local ACC is partly a statistical artifact of weak forecast-anomaly fields, which are visible at forecast
  time, so part of the target's extra predictability is not "forecast-bust" information (§4).
* **Product success: limited.** The workflow (map, confidence, evidence, replay, verification) works on the new event,
  but a ~3%-precision alert is a triage aid, not a reliable bust warning; hidden pattern busts are rarely flagged.

## 1. Target audit (before any model)

Development split (thresholds from TRAIN 2018–19):

| Split | rows | pattern-bust prev. | magnitude-bust prev. | pattern-failure prev. | positive rows | positive inits / inits | share of magnitude busts that are pattern busts | joint / independence |
|---|---|---|---|---|---|---|---|---|
| train | 464,256 | 0.0137 | 0.1021 | 0.1022 | 6,376 | 710 / 730 | 0.134 | 1.32 |
| validation | 230,144 | 0.0120 | 0.1003 | 0.0993 | 2,761 | 328 / 364 | 0.120 | 1.20 |
| test | 234,240 | 0.0103 | 0.0872 | 0.0865 | 2,407 | 343 / 366 | 0.118 | 1.36 |

Local ACC coverage: test 100.0%, train 100.0%, validation 100.0% (invalid rows: {'test': 0, 'train': 0, 'validation': 0}).

Final split (thresholds from TRAIN 2018–20; 2022 rows labelled only for the single evaluation):

| Split | rows | pattern-bust prev. | magnitude-bust prev. | pattern-failure prev. | positive rows | positive inits / inits | share of magnitude busts that are pattern busts | joint / independence |
|---|---|---|---|---|---|---|---|---|
| train | 698,624 | 0.0131 | 0.1013 | 0.1013 | 9,170 | 1042 / 1096 | 0.130 | 1.28 |
| validation | 230,656 | 0.0103 | 0.0866 | 0.0870 | 2,373 | 334 / 364 | 0.119 | 1.37 |
| test | 234,240 | 0.0104 | 0.0879 | 0.0884 | 2,439 | 337 / 366 | 0.118 | 1.34 |

Local ACC coverage: test 100.0%, train 100.0%, validation 100.0% (invalid rows: {'test': 0, 'train': 0, 'validation': 0}).

Large error and low local pattern agreement are **nearly independent** at this scale: the joint rate is only ~1.2–1.4×
what independence would give, and only ~12–13% of magnitude busts are pattern failures. The pattern-aware event is
therefore a small, specific subset of the old bust population, not a refinement of it.

## 2. Development split (validation 2020, dev-test 2021)

### Dev-test 2021

| Model (pattern-aware target) | Prevalence | AUPRC | Lift | ROC AUC | Brier | BSS vs val. clim. | ECE | Recall @10% FAR | Precision / recall at alert | Hidden-bust recall |
|---|---|---|---|---|---|---|---|---|---|---|
| **V4 calibrated** | 0.0103 | 0.0230 | 2.24 | 0.674 | 0.01015 | +0.0027 | 0.0020 | 0.236 | 0.026 / 0.263 | 0.003 |
| B0 pattern-bust climatology | 0.0103 | 0.0107 | 1.04 | 0.505 | 0.01026 | -0.0083 | 0.0034 | 0.096 | 0.011 / 0.096 | 0.057 |
| B1 spread percentile | 0.0103 | 0.0160 | 1.56 | 0.596 | 0.37892 | -36.2474 | 0.5292 | 0.191 | 0.019 / 0.202 | 0.000 |

Raw (uncalibrated) V4: AUPRC 0.0244, ROC AUC 0.677, Brier 0.01014, ECE 0.0036.

**Magnitude-only diagnostic** (old target, prevalence 0.087): the same model trained on it scores AUPRC 0.1558, ROC AUC 0.640, magnitude hidden-bust recall 0.000; V4's probability scored against the magnitude label: ROC AUC 0.555.

ROC AUC (V4 on pattern target) − ROC AUC (same model on magnitude target): +0.0333, 95% init-day block-bootstrap CI [+0.0122, +0.0536]. V4 − B1 AUPRC: +0.0070 [+0.0042, +0.0104].

**Pattern-specific check** — alerts at the validation 10%-FAR threshold: 25,747; 14.5% are large-error busts and 12.6% pattern failures (training rates ≈ 10% each); median normalized error alerts 0.373 vs others 0.234; median local ACC 0.812 vs 0.909; V4 ROC AUC for the large-error criterion alone 0.555, for the pattern criterion alone 0.633.

| Lead day | events | prevalence | AUPRC | lift | ROC AUC | AUPRC B1 |
|---|---|---|---|---|---|---|
| 1 | 285 | 0.0122 | 0.0223 | 1.83 | 0.689 | 0.0141 |
| 2 | 213 | 0.0091 | 0.0168 | 1.85 | 0.684 | 0.0119 |
| 3 | 206 | 0.0088 | 0.0179 | 2.03 | 0.698 | 0.0128 |
| 4 | 218 | 0.0093 | 0.0242 | 2.60 | 0.708 | 0.0131 |
| 5 | 255 | 0.0109 | 0.0273 | 2.51 | 0.670 | 0.0209 |
| 6 | 255 | 0.0109 | 0.0328 | 3.02 | 0.686 | 0.0186 |
| 7 | 208 | 0.0089 | 0.0260 | 2.93 | 0.686 | 0.0192 |
| 8 | 252 | 0.0108 | 0.0266 | 2.47 | 0.636 | 0.0190 |
| 9 | 241 | 0.0103 | 0.0316 | 3.07 | 0.680 | 0.0185 |
| 10 | 274 | 0.0117 | 0.0247 | 2.11 | 0.688 | 0.0186 |

Regions: 58 of 64 have ≥ 20 events; ROC AUC > 0.5 in 100% of them; median 0.656 (10th–90th pct 0.570–0.769). Warning lead (mean lead day of detected busts) 7.12; peak-risk-day error 2.96 days (within ±1 day 0.39).

### Validation 2020 (early stopping and calibration fitted here)

| Model (pattern-aware target) | Prevalence | AUPRC | Lift | ROC AUC | Brier | BSS vs val. clim. | ECE | Recall @10% FAR | Precision / recall at alert | Hidden-bust recall |
|---|---|---|---|---|---|---|---|---|---|---|
| **V4 calibrated** | 0.0120 | 0.0265 | 2.21 | 0.686 | 0.01177 | +0.0069 | 0.0000 | 0.277 | 0.033 / 0.277 | 0.006 |
| B0 pattern-bust climatology | 0.0120 | 0.0127 | 1.06 | 0.518 | 0.01192 | -0.0059 | 0.0017 | 0.094 | 0.012 / 0.094 | 0.070 |
| B1 spread percentile | 0.0120 | 0.0179 | 1.50 | 0.602 | 0.37833 | -30.9189 | 0.5318 | 0.175 | 0.021 / 0.175 | 0.000 |

Raw (uncalibrated) V4: AUPRC 0.0266, ROC AUC 0.683, Brier 0.01181, ECE 0.0017.

**Magnitude-only diagnostic** (old target, prevalence 0.100): the same model trained on it scores AUPRC 0.1800, ROC AUC 0.638, magnitude hidden-bust recall 0.000; V4's probability scored against the magnitude label: ROC AUC 0.555.

ROC AUC (V4 on pattern target) − ROC AUC (same model on magnitude target): +0.0480, 95% init-day block-bootstrap CI [+0.0258, +0.0675]. V4 − B1 AUPRC: +0.0085 [+0.0059, +0.0119].

**Pattern-specific check** — alerts at the validation 10%-FAR threshold: 24,347; 17.3% are large-error busts and 16.6% pattern failures (training rates ≈ 10% each); median normalized error alerts 0.407 vs others 0.245; median local ACC 0.754 vs 0.901; V4 ROC AUC for the large-error criterion alone 0.555, for the pattern criterion alone 0.647.

| Lead day | events | prevalence | AUPRC | lift | ROC AUC | AUPRC B1 |
|---|---|---|---|---|---|---|
| 1 | 277 | 0.0119 | 0.0214 | 1.80 | 0.683 | 0.0120 |
| 2 | 263 | 0.0113 | 0.0220 | 1.95 | 0.688 | 0.0148 |
| 3 | 265 | 0.0114 | 0.0263 | 2.30 | 0.690 | 0.0172 |
| 4 | 256 | 0.0111 | 0.0311 | 2.80 | 0.729 | 0.0179 |
| 5 | 251 | 0.0109 | 0.0285 | 2.61 | 0.675 | 0.0193 |
| 6 | 274 | 0.0119 | 0.0355 | 2.98 | 0.694 | 0.0197 |
| 7 | 302 | 0.0132 | 0.0317 | 2.40 | 0.712 | 0.0215 |
| 8 | 248 | 0.0109 | 0.0267 | 2.45 | 0.731 | 0.0171 |
| 9 | 275 | 0.0121 | 0.0241 | 2.00 | 0.668 | 0.0225 |
| 10 | 350 | 0.0154 | 0.0304 | 1.98 | 0.641 | 0.0224 |

Regions: 63 of 64 have ≥ 20 events; ROC AUC > 0.5 in 98% of them; median 0.678 (10th–90th pct 0.569–0.773). Warning lead (mean lead day of detected busts) 7.07; peak-risk-day error 2.93 days (within ±1 day 0.38).

### Pre-registered gate

| Check | Result |
|---|---|
| acc_coverage | pass |
| positive_rows | pass |
| positive_inits | pass |
| positives_per_lead | pass |
| rocauc_gain_vs_magnitude | pass |
| rocauc_gain_ci_lower_above_0 | pass |
| validation_rocauc_gain_positive | pass |
| beats_spread_score | pass |
| positive_brier_skill | pass |
| lift_every_lead | pass |
| regional_auc | pass |

**Decision: PROMOTE** (all checks pass). The gate was applied as written; nothing was changed after results.

## 3. 2022 — V4 final evaluation after freeze

2022 was previously observed by earlier project generations (v1, v2, V3); therefore the V4 result is not a pristine
project-wide unseen test. V4 was frozen first (target, TRAIN 2018–20 Q90 and Q10 thresholds, features, v2's locked
hyper-parameters, early stopping and isotonic calibration on 2021, model `v4-final-405d2aa`), then evaluated once.
Early stopping kept 604 trees here versus 35 on the dev split — the amount of learnable structure varies by year.

| Model (pattern-aware target) | Prevalence | AUPRC | Lift | ROC AUC | Brier | BSS vs val. clim. | ECE | Recall @10% FAR | Precision / recall at alert | Hidden-bust recall |
|---|---|---|---|---|---|---|---|---|---|---|
| **V4 calibrated** | 0.0104 | 0.0255 | 2.45 | 0.714 | 0.01023 | +0.0070 | 0.0003 | 0.251 | 0.028 / 0.305 | 0.023 |
| B0 pattern-bust climatology | 0.0104 | 0.0107 | 1.03 | 0.503 | 0.01036 | -0.0057 | 0.0027 | 0.110 | 0.012 / 0.110 | 0.107 |
| B1 spread percentile | 0.0104 | 0.0138 | 1.33 | 0.580 | 0.40450 | -38.2570 | 0.5534 | 0.158 | 0.016 / 0.208 | 0.000 |

Raw (uncalibrated) V4: AUPRC 0.0265, ROC AUC 0.715, Brier 0.01024, ECE 0.0028.

**Magnitude-only diagnostic** (old target, prevalence 0.088): the same model trained on it scores AUPRC 0.1471, ROC AUC 0.622, magnitude hidden-bust recall 0.000; V4's probability scored against the magnitude label: ROC AUC 0.536.

ROC AUC (V4 on pattern target) − ROC AUC (same model on magnitude target): +0.0919, 95% init-day block-bootstrap CI [+0.0707, +0.1110]. V4 − B1 AUPRC: +0.0116 [+0.0081, +0.0153].

**Pattern-specific check** — alerts at the validation 10%-FAR threshold: 28,536; 11.9% are large-error busts and 17.5% pattern failures (training rates ≈ 10% each); median normalized error alerts 0.278 vs others 0.236; median local ACC 0.812 vs 0.914; V4 ROC AUC for the large-error criterion alone 0.536, for the pattern criterion alone 0.690.

| Lead day | events | prevalence | AUPRC | lift | ROC AUC | AUPRC B1 |
|---|---|---|---|---|---|---|
| 1 | 267 | 0.0114 | 0.0252 | 2.21 | 0.732 | 0.0133 |
| 2 | 215 | 0.0092 | 0.0214 | 2.33 | 0.734 | 0.0122 |
| 3 | 188 | 0.0080 | 0.0223 | 2.78 | 0.743 | 0.0097 |
| 4 | 194 | 0.0083 | 0.0288 | 3.47 | 0.738 | 0.0123 |
| 5 | 216 | 0.0092 | 0.0371 | 4.02 | 0.765 | 0.0119 |
| 6 | 244 | 0.0104 | 0.0250 | 2.40 | 0.710 | 0.0154 |
| 7 | 245 | 0.0105 | 0.0251 | 2.40 | 0.701 | 0.0164 |
| 8 | 259 | 0.0111 | 0.0215 | 1.95 | 0.685 | 0.0145 |
| 9 | 301 | 0.0129 | 0.0309 | 2.41 | 0.680 | 0.0166 |
| 10 | 310 | 0.0132 | 0.0326 | 2.46 | 0.658 | 0.0170 |

Regions: 60 of 64 have ≥ 20 events; ROC AUC > 0.5 in 98% of them; median 0.702 (10th–90th pct 0.617–0.793). Warning lead (mean lead day of detected busts) 6.03; peak-risk-day error 2.90 days (within ±1 day 0.38).

## 4. Proxy diagnostic (after the dev gate; cannot change it)

Pattern-failure rate by quintile of the forecast-only 3×3 anomaly standard deviation (weak → strong):
16.2%, 9.9%, 8.9%, 5.8%, 2.6%;
the magnitude-bust rate is flat across the same quintiles (8.0%, 8.3%, 8.7%, 8.6%, 10.0%).
That single forecast-time quantity ranks pattern failure at ROC AUC 0.669 (V4: 0.633)
and pattern-aware busts at 0.586 (V4: 0.674); V4's probability correlates only
0.068 with it. Interpretation: Low local ACC is partly a statistical artifact of weak (flat) anomaly fields, which are visible at forecast time; part of the pattern target's extra predictability over the magnitude target reflects that. V4's ranking is not explained by that proxy alone (corr 0.07; AUC 0.674 vs 0.586 for pattern busts), and its alerts have higher error and lower ACC than non-alerts.

## 5. Archived generations (different target; history only)

v2 B2 / Sentinel, V3 and BMA were evaluated on the magnitude-only target (`docs/evaluation_v2.md`,
`docs/evaluation_v3.md`, `docs/model_card_bma.md`). Their AUPRC values are not comparable with V4's (different event,
~9× different prevalence). On the magnitude target nothing beat the spread-only B2 baseline; V4 does not claim to.
