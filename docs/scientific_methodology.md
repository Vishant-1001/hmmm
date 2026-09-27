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
| B2 | Same XGBoost learner and protocol as Sentinel, inputs: spread, spread percentile, lead day, region, season, initialisation hour (00/12 UTC); isotonic calibration on VALIDATION |

B2 is deliberately strong: it learns region/lead/season-specific spread-skill relationships.

## 6. Sentinel features (all available at T)

| Group | Features |
|---|---|
| ATM atmospheric state | Z500/700/850 anomalies of the ensemble mean, 500–850 hPa thickness anomaly, zonal/meridional Z500 gradient and magnitude, Z500 Laplacian, 3×3 neighbourhood anomaly |
| ENS ensemble behaviour | member IQR, P10–P90, skewness, anomaly-sign agreement, neighbourhood spread and heterogeneity, 700/850 spread, spread growth vs Day 1, domain spread, spread relative to domain |
| PAT large-scale pattern | 8 PCs of the ensemble-mean Z500 anomaly over the context domain (20°E–146°E, 31°S–65°N), fitted on TRAIN only and frozen; PC-space norm; unexplained-variance fraction |
| EVO forecast evolution | revision vs the cycle 24 h earlier for the same valid time (regional, neighbourhood RMS, PC-space), spread change; missing when that cycle is not in the downloaded sample (XGBoost handles NaN) |
| MEM historical memory | see §7 |
| REC recent verified error | mean normalized error, mean signed error and bust fraction of this region's Day 1–3 forecasts verified in the 5 days up to initialisation; neighbour and domain means (`analogues/recent.py`) |

**Model inputs vs displayed evidence.** Counts that grow monotonically with the size of the memory
(`an_n_eligible`, `an_n_within`, `rec_n`) act as hidden timestamps and are excluded from model inputs;
they are shown to the forecaster as evidence only.

**FULL model.** A feature group enters FULL only if "B2 + group" beats B2 on VALIDATION AUPRC (never
on test). "ALL" (every group) is also reported so the effect of the selection is visible.

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
