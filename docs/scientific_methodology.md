# Scientific methodology

This document is our implementation. The official SIH26079 wording is in `docs/sih_source.md`
and is not paraphrased here.

> **v3 (current production model).** Forecast Bust Sentinel models the conditional distribution of future
> normalized regional Z500 forecast error using quantile gradient boosting applied to prediction-time-safe
> NWP forecast-state features. The model's upper-tail error distribution is used to estimate the
> probability of exceeding the project-defined bust threshold, followed by validation-only probability
> calibration. Section 0 describes this predictive core; sections 1-13 (data, target, labels, features,
> memory, support, fingerprint, protocol) are unchanged and still apply. The B2/Sentinel classifiers of
> sections 5-6 and 9 are v1/v2 history: archived benchmarks, not used in v3 inference.

## 0. v3 predictive core: quantile gradient boosting

**What is established and what is ours.** Quantile regression (Koenker & Bassett 1978) and gradient-boosted
quantile regression fitted one quantile at a time (e.g. the GEFCom2014 winning wind-power method, Landry et
al. 2016) are established. Probabilistic post-processing of NWP output with tree ensembles (Taillardat et
al. 2016; Schulz & Lerch 2022) and learning NWP forecast errors from forecast-state predictors (Ben
Bouallègue et al. 2022, ECMWF TM 896) are established. Neither the method nor the idea is new here. The
project's contribution is the application: the regional Day 1-10 *normalized Z500 error* of an existing
IFS ENS forecast, turned into a calibrated probability of exceeding the project's fixed bust threshold, with
causal historical evidence, support/OOD and blind verification around it. Measured results are in
`docs/evaluation_v3.md`; nothing here asserts that the method improves on any baseline.

1. **Target**: `norm_error` (section 4), unchanged. **Inputs**: the existing groups SPREAD, ATM, ENS, PAT,
   EVO, MEM, REC, DYN (70 features, all available at *T*; section 6). No B2 output, no verification quantity.
2. **Estimator**: `sklearn.ensemble.HistGradientBoostingRegressor(loss="quantile", quantile=τ)` for
   τ ∈ {0.10, 0.25, 0.50, 0.75, 0.90, 0.95}, fitted one after another on TRAIN with one shared
   configuration (learning rate 0.05, 200 iterations, 31 leaves, min 100 samples per leaf, L2 1.0, no early
   stopping). The six estimators are one quantile model family, not competing models. NaN is handled
   natively (no imputation).
3. **Quantile crossing**: independently fitted quantiles can cross. Crossing is measured (dev validation:
   0.58% of rows, almost all q90 > q95, median size 0.011) and removed at prediction time by
   *rearrangement*: sorting each row's six values (Chernozhukov, Fernández-Val & Galichon 2010), which does
   not increase quantile estimation error. The raw crossing rate is stored in the model metadata.
4. **Estimated exceedance probability**: with the training-derived threshold *c* = `q_primary` (Q90 of
   TRAIN `norm_error` per region × lead × season, unchanged), the CDF at *c* is interpolated linearly
   between the bracketing predicted quantiles. Beyond q95 the survival function decays exponentially
   through (q90, 0.10) and (q95, 0.05): S(c) = 0.05 · 2^(−(c−q95)/(q95−q90)), with the mirror form below
   q10. Then `estimated_exceedance_probability = 1 − F(c)`, clipped only for floating-point noise. This is
   an **estimate from six predicted quantiles**, not an exact CDF.
5. **Calibration**: isotonic regression of the estimated exceedance probability against the bust label,
   fitted on VALIDATION rows only, frozen → `calibrated_bust_probability`. Calibration changes probability
   reliability, not ranking; AUPRC (discrimination), Brier, ECE and quantile quality are reported separately.
6. **Products**: per region × lead day: q10..q95; "expected error" = q50, the *central (median) predicted
   error* (not a conditional mean); *central predicted error range* = q25-q75 (not a confidence interval);
   *upper-tail error* = q95 (not a maximum); both probabilities; confidence = 1 − calibrated probability.
7. **Explanations**: no exact per-row attribution exists for this estimator here and none is fabricated.
   The UI separates MODEL OUTPUT (quantiles, threshold, probabilities), INPUT EVIDENCE (this row's values
   and training percentiles for the model-level most important inputs, by permutation importance of the q90
   estimator on validation; evidence items A-D) and INTERPRETATION (sentences generated from those numbers,
   "associated with", never "caused").
8. **Why not QRF**: a quantile regression forest was tried first; its fitted leaf-response storage was
   repeatedly OOM-killed on the 8 GB development machine. Histogram gradient boosting bins the features and
   stores only trees (all six estimators ≈ 5 MB; full dev fit 72-95 s, 2.5 GB peak RSS). This is an
   implementation change, not a different scientific method.

## 0b. BMA experiment (ensemble-member post-processing; see `docs/model_card_bma.md` for the outcome)

**Why BMA.** An ensemble forecast is a set of plausible future atmospheric states. BMA turns the members
into one predictive distribution, a weighted mixture of member-specific distributions with weights that
reflect historical predictive contribution. It does not reduce the ensemble to a single spread statistic,
and it is not a generic supervised classifier (Raftery et al. 2005). Published work applied BMA to 500 hPa
geopotential height and reported better probabilistic forecasts than the raw ensembles (Ji et al. 2021).
That justifies *testing* it here, not expecting it to work: this is an adaptation to a different target
(the regional error of the ensemble mean) on this project's data, and BMA is not new.

```
50 IFS ENS member Z500 forecasts at the region box (data/cache/ens; member mean == stored ensemble mean)
  -> member-specific predictive distributions N(a + b f_k, sigma^2) for the verifying ERA5 Z500 anomaly
  -> equal-weight mixture (exchangeable perturbed members: w_k = 1/50, common a, b, sigma per region x lead)
  -> regional Z500 error distribution: Y = |ensemble mean - Z| / TRAIN scale, exact mixture CDF
  -> BMA mixture exceedance probability P(Y > q_primary)  (TRAIN Q90, unchanged)
  -> optional validation-only isotonic calibration -> bust sentinel
```

* **Target, rows, thresholds** are exactly those of v2/v3/B2 (v3 namespace table); nothing is redefined.
* **Fitting**: per region × lead, (a, b) by least squares of the verifying anomaly on the pooled member
  anomalies and sigma by EM, as in Raftery et al. (2005), with weights fixed at 1/50. Members are
  exchangeable perturbations, so member-specific weights would carry no reproducible meaning; a free-weight
  EM fit on TRAIN cases is reported only as a diagnostic.
* **Sliding training window**: 60 days of cases *verified before* the forecast day (valid time ≤ 00 UTC of
  the initialisation day), refitted daily, at least 20 cases. One choice, fixed before any result
  (~1 initialisation per day per region and lead in the 50%-sampled archive → ~60 cases for 3 parameters).
  Like the existing causal memory features, the window uses earlier verified cases of the evaluated year;
  no case's own or any later outcome is used.
* **Outputs**: predictive mean and standard deviation of Y (folded-normal moments), quantiles q10..q95
  (bisection on the exact mixture CDF), the BMA mixture exceedance probability (the CDF of the fitted model,
  not an empirical CDF), and a separately labelled validation-calibrated bust probability.
* **Gate**: pre-registered in `config/model_bma.yaml` (commit `483fcea`) before any BMA result existed.
* **Outcome: NO-GO.** Dev-test 2021 AUPRC 0.121 vs B2 0.158 and V3 0.140; worse on all 10 lead days. Not
  evaluated on 2022; not served. Diagnosis: the window-constant kernel σ is as large as the flow-dependent
  member dispersion, diluting the spread signal B2 uses.

## 1. Quantity predicted

For every initialisation *T*, region *r* and lead day *d* ∈ {1..10}:

    P( Bust(r, d) = 1 | information available at T )

where Bust is defined on the **existing** IFS ENS ensemble-mean Z500 forecast. This is a
probability of a large *forecast error*, not of any weather event. Day-wise risks are predicted
independently by one shared model with lead-day context; peak-risk day, first alert day and
the deterioration window are derived products.

## 2. Alignment

`valid_time = init_time + lead` (`verification/alignment.py`). ERA5 is indexed at valid time;
tests cover +24, +48, +72 and +240 h, year crossing and missing reference times.

## 3. Regional error

    RMSE_r = sqrt( Σ w_i (F̄_i − A_i)² / Σ w_i ),   w_i = sin(φ_i + Δ/2) − sin(φ_i − Δ/2)

(exact grid-box area weights, ∝ cos φ). F̄ = 50-member mean, A = ERA5. With the 5.625° product
each region is one grid box, so RMSE_r equals |F̄ − A| of the box-mean Z500.

## 4. Normalisation and labels (TRAIN only)

* `scale(r, season)` = std of the ERA5 Z500 anomaly (vs 1990–2017 climatology) over TRAIN valid times.
* `normalized_error = RMSE_r / scale(r, season)`.
* Seasons: JF, MAM, JJAS, OND (by initialisation month).
* `bust = normalized_error > Q90_train(r, d, season)` — **the 90th-percentile threshold is a
  project-defined operational bust criterion.** Q95 is the sensitivity criterion.
* Because the threshold is conditioned on (region, lead, season) and the scale on (region, season),
  normalisation does not change which cases are busts; it makes errors comparable across regions
  for the historical-memory statistics.
* **Hidden bust** (diagnostic): bust AND spread ≤ TRAIN P25 of spread for (r, d, season).

## 5. Baselines

| | Definition |
|---|---|
| B0 | TRAIN bust frequency per (region, lead, season) |
| B1 | TRAIN-percentile of the regional ensemble spread (ranking score) |
| B2 | Same XGBoost learner, protocol and (v2) tuning budget as Sentinel, inputs: spread, spread percentile, **spread_thr_ratio** (v2), lead day, region, season, initialisation hour (00/12 UTC); isotonic calibration on VALIDATION |

B2 is deliberately strong: it learns region/lead/season-specific spread-skill relationships.
`spread_thr_ratio = spread·√(1 + 1/M) / (Q90_TRAIN · scale_TRAIN)` (M = 50 members) is the spread relative to
the bust threshold in metres; for a reliable Gaussian ensemble P(bust) = 2Φ(−1/ratio). It uses only the
forecast spread and TRAIN constants. It was added because, on the dev split, this textbook spread probability
beat the v1 tree B2 on validation (0.181 vs 0.170 AUPRC): trees approximate the ratio poorly.

## 6. Sentinel features (all available at T)

| Group | Features |
|---|---|
| ATM atmospheric state | Z500/700/850 anomalies of the ensemble mean, 500–850 hPa thickness anomaly, zonal/meridional Z500 gradient and magnitude, Z500 Laplacian, 3×3 neighbourhood anomaly |
| ENS ensemble behaviour | member IQR, P10–P90, skewness, anomaly-sign agreement, neighbourhood spread and heterogeneity, 700/850 spread, spread growth vs Day 1, domain spread, spread relative to domain |
| PAT large-scale pattern | 8 PCs of the ensemble-mean Z500 anomaly over the context domain (20°E–146°E, 31°S–65°N), fitted on TRAIN only and frozen; PC-space norm; unexplained-variance fraction |
| EVO forecast evolution | revision vs the cycle 24 h earlier for the same valid time (regional, neighbourhood RMS, PC-space), spread change; missing when that cycle is not in the downloaded sample (XGBoost handles NaN) |
| MEM historical memory | see §7 |
| DYN (v2) wind / pressure state | ensemble-mean u/v anomaly (500 hPa), wind speed (500/850), 500–850 hPa vector shear, relative vorticity and divergence (500/850; spherical centred differences), MSLP anomaly and gradient, vector-wind spread (500/850), MSLP spread, and in-forecast tendencies \|dZ500/dt\|, \|dMSLP/dt\|, \|dζ500/dt\| around the target lead (same forecast only) — `features/dynamics.py` |
| REC recent verified error | mean normalized error, mean signed error and bust fraction of this region's Day 1–3 forecasts verified in the 5 days up to initialisation; neighbour and domain means (`analogues/recent.py`) |

**Model inputs vs displayed evidence.** Counts that grow monotonically with the size of the memory
(`an_n_eligible`, `an_n_within`, `rec_n`) act as hidden timestamps and are excluded from model inputs;
they are shown to the forecaster as evidence only.

**FULL model.** v1: a feature group enters FULL only if "B2 + group" beats B2 on VALIDATION AUPRC (never
on test). v2: groups, hyper-parameters and learner were chosen on the dev split and locked in
`config/model_v2.yaml` (§13a). "ALL" (every group) is also reported so the effect of the selection is visible.

**In-sample quota effect (found on the dev split).** Because bust labels are TRAIN quantiles within
each (region, lead, season) group, a training case whose analogues were busts is slightly *less*
likely to be a bust itself (the group's 10% quota is used up). Label-rate features therefore show a
reversed relationship inside TRAIN (e.g. analogue bust rate AUC 0.45 in train vs 0.53 on validation
in the dev split). Validation-based group selection guards against this leaking into FULL.

## 7. Historical forecast-state memory

* Space: 8 pattern PCs + regional spread + regional anomaly + P10–P90, standardised with TRAIN statistics.
* Candidates: same region and lead day, **verified before the query was issued**
  (`candidate.valid_time ≤ query.init_time`); no self-retrieval is possible.
* k = 30 nearest; features: eligible count, count within radius (radius = TRAIN median distance
  to the 30th analogue), nearest and mean distance, analogue bust rate, median and Q90 of
  analogue normalized error.
* Mode `causal_online` (default) lets the memory grow with verified cases, as it would operationally;
  `frozen` (train + validation only) is reported as a sensitivity result.

## 8. Support / OOD and evidence strength

* Support distance: Ledoit-Wolf regularised Mahalanobis distance in the analogue space vs the TRAIN
  distribution for that lead day. Categories by TRAIN quantiles 0.90 / 0.975 / 0.995:
  NORMAL / MODERATE / WEAK / INSUFFICIENT HISTORICAL SUPPORT.
* Evidence strength (product rule, `support/ood.py::evidence_strength`):
  INSUFFICIENT if support is INSUFFICIENT or < 5 analogues within radius; STRONG if NORMAL
  support, ≥ 20 analogues within radius and analogue bust rate within 0.25 of the model
  probability; MODERATE if support ≤ MODERATE and ≥ 10 analogues within radius; else WEAK.
  A high probability can carry WEAK evidence.

## 9. Calibration and confidence

Isotonic regression on VALIDATION predictions, frozen. `confidence = 1 − p_bust`.
`confidence_disagreement = p_Sentinel − p_B2` (percentage points) — a project-level
decision-support diagnostic, not a new scientific metric.

## 10. Failure signature (deterministic, project-defined)

Over a 7×7-box window (≈39°, the synoptic wavelength scale) around the region, the exact MSE decomposition

    MSE = (F̄−Ā)² + (σF−σA)² + 2σFσA(1−r)

gives a pattern/phase share. POSITION_PHASE if phase share ≥ 0.6, AMPLITUDE_STRUCTURE if
bias+amplitude share ≥ 0.6, otherwise RESIDUAL_MIXED. Before verification the UI shows the
*distribution of actual signatures among similar historical busts* ("historical failure
signature among similar forecast states"); after verification it shows the actual fingerprint.
Top-1 agreement and probability assigned to the actual signature are measured against a
climatological reference (always the most frequent TRAIN signature).

## 11. Priority queue

`score = p · 0.5^((d−1)/5) · w(evidence) + 0.25 · max(0, p − p_B2)` with w = 1 / 0.85 / 0.7 / 0.5.
A transparent triage rule; the weights are design choices, not fitted constants.

## 12. Explanations

Deterministic templates filled with computed numbers (spread percentile, anomaly percentile,
analogue counts and bust rates, revision magnitudes, baseline disagreement), plus TreeSHAP
attributions of the pre-calibration XGBoost model. No LLM is used anywhere. Wording follows
"associated with / historically similar", never "caused by".

## 13. Evaluation protocol

Chronological split (train 2018–2020, validation 2021, test 2022) with purge at period
boundaries. Model development used a separate dev split (train 2018–19, validation 2020,
dev-test 2021). The primary metric is AUPRC. Material improvement over B2 requires, by a rule
fixed before the test evaluation, (i) the 95% block-bootstrap CI (blocks = initialisation days)
of AUPRC(FULL) − AUPRC(B2) to exclude 0 and (ii) a relative gain ≥ 5%.

## 13a. v2 optimisation protocol (dev split only)

Full log with every experiment: `docs/optimization_v2.md` (generated from `artifacts/v2/dev/optimization/`).

1. **Diagnosis** on train 2018–19 / validation 2020: prevalence, spread-only predictability ceiling (AUPRC a
   spread-only model could reach if the ensemble were perfectly reliable), spread-conditional information of
   every feature (ROC AUC within lead-day × spread-decile strata), low-spread (hidden-bust) separability, and
   the v1 hyper-parameters on v2 features (Sentinel stopped after 57 trees vs 821 for B2).
2. **Search**: the same 12-configuration grid (depth 2/3/5 × min_child_weight 50/300 × colsample 0.5/1.0) and
   an 8-configuration regularised refinement for B2, the standard Sentinel and the residual (B2-margin)
   learner; class weights 3 and 9; seeds 1–3. Primary criterion validation AUPRC.
3. **Ablation** (B2 + each group) with 3-seed means.
4. **One dev-test check** on 2021 of the validation-selected configuration: the +2.8% validation gain became
   −0.6% (CI [−0.0038, +0.0019]).
5. **Two-period stability rule** (fixed before it ran): keep a group only if its 3-seed mean gain over B2 is
   positive on both 2020 and 2021. Result: ATM, EVO, MEM, REC kept; DYN, ENS, PAT rejected.
6. **Lock** (`config/model_v2.yaml`, commit `8b196ee`) → final protocol: train 2018–20, early stopping and
   calibration on 2021, one scoring of 2022 (a disclosed second look; v1 scored it first).

## References (positioning, not copied)

* Rasp et al. (2024) *WeatherBench 2*, JAMES 16, e2023MS004019.
* Hersbach et al. (2020) *The ERA5 global reanalysis*, QJRMS 146, 1999–2049.
* Fortin et al. (2014) *Why should ensemble spread match the RMSE of the ensemble mean?*, J. Hydrometeorol. 15, 1708–1713.
* Whitaker & Loughe (1998) *The relationship between ensemble spread and ensemble mean skill*, MWR 126, 3292–3302.
* Murphy (1988) *Skill scores based on the mean square error and their relationships to the correlation coefficient*, MWR 116, 2417–2424.
* Rodwell et al. (2013) *Characteristics of occasional poor medium-range weather forecasts for Europe*, BAMS 94, 1393–1405.
* Hauser et al. *Exceptionally poor and good medium-range forecasts of the large-scale circulation over Europe in ERA5 reforecasts*, QJRMS, doi:10.1002/qj.70117 (related work on forecast busts).
* Lorenz (1969) *Atmospheric predictability as revealed by naturally occurring analogues*, JAS 26, 636–646.
* Lundberg et al. (2020) *From local explanations to global understanding with explainable AI for trees*, Nat. Mach. Intell. 2, 56–67.
* Chen & Guestrin (2016) *XGBoost*, KDD.
* Koenker & Bassett (1978) *Regression quantiles*, Econometrica 46, 33–50.
* Taillardat, Mestre, Zamo & Naveau (2016) *Calibrated ensemble forecasts using quantile regression forests
  and ensemble model output statistics*, MWR 144, 2375–2393, doi:10.1175/MWR-D-15-0260.1.
* Ben Bouallègue, Cooper, Chantry, Düben, Bechtold & Sandu (2022) *Statistical modelling of 2m temperature
  and 10m wind speed forecast errors*, ECMWF Technical Memorandum 896,
  https://www.ecmwf.int/en/elibrary/81297-statistical-modelling-2m-temperature-and-10m-wind-speed-forecast-errors
  (DOI 10.21957/vdcccja3f as given in the project brief; not independently resolved).
* Landry, Erlinger, Patschke & Varrichio (2016) *Probabilistic gradient boosting machines for GEFCom2014
  wind forecasting*, Int. J. Forecasting 32, 1061–1066, doi:10.1016/j.ijforecast.2016.02.002 (one GBM per
  quantile on NWP predictors - the pattern used here).
* Schulz & Lerch (2022) *Machine learning methods for postprocessing ensemble forecasts of wind gusts: a
  systematic comparison*, MWR 150, 235–257 (gradient-boosting EMOS and QRF among the compared methods).
* Chernozhukov, Fernández-Val & Galichon (2010) *Quantile and probability curves without crossing*,
  Econometrica 78, 1093–1125.
* Raftery, Gneiting, Balabdaoui & Polakowski (2005) *Using Bayesian Model Averaging to Calibrate Forecast
  Ensembles*, MWR 133, 1155–1174.
* Ji, Luo, Ji & Zhi (2021) *Probabilistic Forecasting of the 500 hPa Geopotential Height over the Northern
  Hemisphere Using TIGGE Multi-model Ensemble Forecasts*, Atmosphere 12(2), 253, doi:10.3390/atmos12020253.
* scikit-learn `HistGradientBoostingRegressor` (loss="quantile"), https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html
