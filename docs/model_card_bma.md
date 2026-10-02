# Model card — BMA experiment (Bayesian Model Averaging of the IFS ENS members)

**Status: NOT SELECTED. Negative result. BMA is not sufficiently validated as the final predictive core.**
It failed the pre-registered go/no-go gate on the development split, so it was **not** evaluated on 2022
and production serving was **not** changed (the live demo still serves V3). Artifacts are kept as a
scientifically valid negative result: `artifacts/bma/dev/metrics.json`, `artifacts/bma/dev/experiment_manifest.json`,
code `models/bma.py`, `pipeline_bma.py`, `evaluation/run_bma.py`, config and gate `config/model_bma.yaml`.

## Purpose and research basis

To test whether ensemble-member probabilistic post-processing extracts more bust information than the
spread-only B2 classifier or V3. Published research supports BMA for calibrating ensemble forecasts
(Raftery et al. 2005) and for probabilistic 500 hPa geopotential height forecasts from multi-model
ensembles (Ji et al. 2021, who found it better than EMOS at longer leads and slightly worse at Day 1–4).
That justified testing it; it did not imply it would work for this project's target, which is an adaptation.

* Raftery, A. E., Gneiting, T., Balabdaoui, F., & Polakowski, M. (2005). Using Bayesian Model Averaging to
  Calibrate Forecast Ensembles. *Monthly Weather Review*, 133, 1155–1174.
* Ji, L., Luo, Q., Ji, Y., & Zhi, X. (2021). Probabilistic Forecasting of the 500 hPa Geopotential Height over
  the Northern Hemisphere Using TIGGE Multi-model Ensemble Forecasts. *Atmosphere*, 12(2), 253.
  doi:10.3390/atmos12020253.

## Model

* **Ensemble-member representation**: the 50 genuine IFS ENS perturbed members of Z500 (WB2 64×32 store,
  `data/cache/ens`, all 1,828 initialisations 2018–2022, no missing member), read at the 64 target boxes
  (one native 5.625° box per region — the existing regionalisation). Their mean reproduces the stored
  ensemble mean that defines the target **exactly** (max difference 0.0 m). No member was synthesised.
* **Target (unchanged)**: `norm_error` = |ensemble mean − ERA5| / TRAIN scale(region, season); bust =
  `norm_error` > TRAIN Q90 (`q_primary`) — the same rows and thresholds as B2/V2/V3 (v3 table).
* **Formulation**: ERA5 Z500 anomaly z ~ (1/50) Σ_k N(a + b f_k, σ²) per region × lead (exchangeable members:
  equal weights, common a, b, σ). (a, b) by least squares on pooled members, σ by EM (Raftery et al. 2005).
  Y = |m − z|/s has an exact mixture CDF; outputs: mean, SD, q10–q95, the BMA mixture exceedance probability
  P(Y > q_primary), and a separately labelled validation-calibrated bust probability.
* **Training**: sliding 60-day window of cases verified before the forecast day (≥ 20 cases), refitted daily,
  one window choice fixed before results; no search. Calibration: isotonic on validation 2020 only.
* **Weights**: fixed at 1/50. The original free-weight EM on TRAIN cases (8 random region × lead groups, 720
  cases each) put 19–26 of 50 members below 0.001 and a different "top" member in every group (max weight
  0.167): unstable, no reproducible member distinction — consistent with exchangeable
  perturbations. A weight would only mean contribution to the fitted mixture, never physical causation.
* **Parameter stability (2019–2022 fits, 1st/50th/99th pct)**: b 0.27 / 0.82 / 1.04;
  a -10.2 / 0.07 / 13.5 m; σ 0.9 / 3.9 / 27.2 m.
* **Compute**: member extraction 58 s (234 MB); 577,152 causal parameter sets in 61 min single-core
  (~4 s per forecast day for 64 regions × 10 leads); whole pipeline 71 min, peak RSS 1.4 GB.

## Leakage controls

Parameters for day D use only cases with valid time ≤ D 00 UTC (asserted for every parameter set;
unit test shows future outcomes cannot change them); the target's scale and Q90 are the existing TRAIN-only
quantities; calibration uses validation rows only; 2022 was not touched. The sliding window uses earlier
verified cases of the evaluated year — the same causal rule as the existing MEM/REC features; B2 does not
use them, so this favoured BMA, not the baselines.

## Evaluation (development split: validation 2020, dev-test 2021; identical rows for all models, 234,240 dev-test rows, none missing)

### Dev-test 2021

| Model | AUPRC | ROC AUC | Brier | BSS vs val. clim. | ECE | Recall @10% FAR | Hidden-bust recall |
|---|---|---|---|---|---|---|---|
| B0 climatology | 0.0932 | 0.523 | 0.0798 | -0.001 | 0.0150 | 0.098 | 0.070 |
| B2 spread-only (archival dev reference) | 0.1578 | 0.638 | 0.0777 | +0.026 | 0.0138 | 0.233 | 0.000 |
| V3 quantile gradient boosting | 0.1396 | 0.629 | 0.0783 | +0.019 | 0.0132 | 0.190 | 0.085 |
| BMA mixture exceedance (raw) | 0.1210 | 0.606 | 0.0897 | -0.125 | 0.0604 | 0.173 | 0.051 |
| BMA calibrated (isotonic, validation) | 0.1188 | 0.605 | 0.0788 | +0.012 | 0.0080 | 0.171 | 0.046 |

BMA − B2 AUPRC -0.0365 (95% block-bootstrap CI [-0.0449, -0.0269]); BMA − V3 -0.0185 ([-0.0256, -0.0107]).

| Error distribution | MAE (q50) | RMSE (q50) | coverage q10 / q25 / q50 / q75 / q90 / q95 |
|---|---|---|---|
| BMA | 0.222 | 0.312 | 0.131 / 0.309 / 0.576 / 0.807 / 0.925 / 0.962 |
| V3_QGB | 0.204 | 0.289 | 0.115 / 0.284 / 0.543 / 0.782 / 0.912 / 0.956 |

| Lead day | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| AUPRC B2 | 0.143 | 0.131 | 0.137 | 0.151 | 0.167 | 0.168 | 0.156 | 0.159 | 0.184 | 0.192 |
| AUPRC V3 | 0.142 | 0.125 | 0.141 | 0.131 | 0.149 | 0.153 | 0.134 | 0.132 | 0.155 | 0.166 |
| AUPRC BMA | 0.138 | 0.118 | 0.125 | 0.123 | 0.129 | 0.124 | 0.108 | 0.109 | 0.123 | 0.130 |

Regions where BMA AUPRC ≥ B2: 8% of 64; median regional AUPRC BMA 0.1205, B2 0.1519, V3 0.1282.

### Validation 2020 (reproducibility check; calibration fitted here, so calibrated rows are in-sample)

| Model | AUPRC | ROC AUC | Brier | BSS vs val. clim. | ECE | Recall @10% FAR | Hidden-bust recall |
|---|---|---|---|---|---|---|---|
| B0 climatology | 0.0981 | 0.488 | 0.0902 | -0.000 | 0.0019 | 0.071 | 0.056 |
| B2 spread-only (archival dev reference) | 0.1747 | 0.639 | 0.0876 | +0.029 | 0.0000 | 0.221 | 0.000 |
| V3 quantile gradient boosting | 0.1485 | 0.616 | 0.0887 | +0.017 | 0.0000 | 0.191 | 0.074 |
| BMA mixture exceedance (raw) | 0.1301 | 0.596 | 0.1032 | -0.143 | 0.0700 | 0.156 | 0.074 |
| BMA calibrated (isotonic, validation) | 0.1305 | 0.597 | 0.0893 | +0.010 | 0.0000 | 0.141 | 0.071 |

BMA − B2 AUPRC -0.0445 (95% block-bootstrap CI [-0.0538, -0.0361]); BMA − V3 -0.0180 ([-0.0262, -0.0101]).

| Error distribution | MAE (q50) | RMSE (q50) | coverage q10 / q25 / q50 / q75 / q90 / q95 |
|---|---|---|---|
| BMA | 0.237 | 0.336 | 0.136 / 0.314 / 0.577 / 0.804 / 0.924 / 0.960 |
| V3_QGB | 0.214 | 0.311 | 0.109 / 0.269 / 0.524 / 0.760 / 0.900 / 0.947 |

| Lead day | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| AUPRC B2 | 0.147 | 0.144 | 0.156 | 0.166 | 0.186 | 0.203 | 0.188 | 0.175 | 0.191 | 0.196 |
| AUPRC V3 | 0.129 | 0.124 | 0.140 | 0.155 | 0.175 | 0.178 | 0.162 | 0.141 | 0.163 | 0.166 |
| AUPRC BMA | 0.149 | 0.138 | 0.138 | 0.136 | 0.133 | 0.141 | 0.137 | 0.108 | 0.120 | 0.129 |

Regions where BMA AUPRC ≥ B2: 2% of 64; median regional AUPRC BMA 0.1248, B2 0.1638, V3 0.1462.

## Pre-registered gate (`config/model_bma.yaml`, commit `483fcea`, written before any BMA result)

| Check | Result |
|---|---|
| devtest_auprc_gain_vs_B2 | **fail** |
| devtest_ci_lower_above_0 | **fail** |
| validation_auprc_gain_vs_B2 | **fail** |
| brier_not_worse_than_B2 | **fail** |
| ece_ok | pass |
| lead_days_beating_B2 | **fail** |
| coverage_of_rows | pass |

**Decision: NO-GO.** BMA beat B2 on 0 of 10 dev-test lead days.

## Why it did not help (diagnosis, not tuning)

* The kernel σ is constant within each 60-day window and about as large as the flow-dependent member
  dispersion (median σ / (|b| × member spread) = 0.91), so roughly half of BMA's predictive spread does not
  vary from case to case. That dilutes exactly the signal B2 exploits (spread relative to the threshold):
  `spread_thr_ratio` alone scores dev-test AUPRC 0.163; BMA's exceedance correlates with it at only 0.55.
* The raw mixture is too wide for the error target (q50 coverage 0.576, ECE 0.060, BSS −0.125); isotonic
  calibration repairs reliability (ECE 0.008) but cannot repair ranking.
* No implementation defect was found: member mean = target ensemble mean exactly, exact CDF matches Monte
  Carlo, quantiles ordered, causal fitting asserted, all dev rows scored.

## Answers

1. Genuine ensemble-member information: **yes** (50 real members; nothing reconstructed).
2. Leakage: **none found** (causal window asserted and tested; TRAIN-only thresholds; validation-only calibration).
3. Target: the unchanged `norm_error` of the ensemble mean and its Q90 bust event.
4. Distribution: equal-weight 50-component Gaussian mixture for the ERA5 Z500 anomaly, mapped exactly to Y.
5. Weights: 1/50 each (exchangeable); free weights were unstable (above).
6–9. Dev-test AUPRC: climatology 0.0932, B2 0.1578, V3 0.1396, BMA 0.1210 (calibrated 0.1188). BMA beats
   climatology, but is significantly below B2 (−0.037) and below V3 (−0.019).
10. Bust ranking improved: **no**. 11. Probability quality: raw **worse**; calibrated Brier 0.0788 vs B2 0.0777.
12. Improvement large enough: **there is no improvement**.
13. Stable across lead days: consistently below B2 (0 of 10 days). 14. Regions: below B2 in 92% of regions.
15. Computationally practical: yes (1.4 GB, ~4 s per forecast day).
16. Enough evidence for production: **no. BMA is not sufficiently validated as the final predictive core.**

"BMA did not establish sufficient evidence of improvement on this dataset."
