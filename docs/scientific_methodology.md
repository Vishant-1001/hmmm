# Scientific methodology

This document is our implementation. The official SIH26079 wording is in `docs/sih_source.md`
and is not paraphrased here.

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
