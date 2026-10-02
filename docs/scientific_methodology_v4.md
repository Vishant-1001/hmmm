# Scientific methodology — V4 (pattern-aware forecast bust)

> Forecast Bust Sentinel identifies severe regional forecast-bust risk using a pattern-aware definition that
> combines unusually large Z500 forecast error with unusually poor local spatial anomaly agreement. The
> thresholds for both conditions are estimated from training data, and an XGBoost classifier estimates the
> probability of this event from prediction-time-safe forecast-state information.

Neither the joint error/pattern criterion, the anomaly correlation coefficient, nor XGBoost is new. The contribution is
the application: a regional Day 1–10 reliability workflow over an existing NWP ensemble, with calibrated probability,
causal historical evidence, support/OOD, blind replay and ERA5 verification. Everything below §2 of
`docs/scientific_methodology.md` (data, alignment, regional error, normalisation, Q90, features, memory, support,
failure fingerprint, evaluation protocol) is unchanged and still applies.

## 1. Why the target changed

The magnitude-only event (normalized regional error > TRAIN Q90) could not be ranked better than a spread-only baseline
by any learner tried (v1/v2 XGBoost, V3 quantile gradient boosting, BMA). Large-scale bust studies characterise severe
busts by large RMSE *and* low anomaly correlation together, so that the event reflects both error magnitude and a
pattern/phase discrepancy (Rodwell et al. 2013 used both over Europe). V4 adopts that joint definition at regional
scale, with training-percentile thresholds instead of European constants.

## 2. Definition

For each region × initialisation × lead day:

1. `norm_error` — unchanged (|ensemble mean − ERA5| at the region box / TRAIN scale(region, season)).
2. `local_acc` — centred, area-weighted Pearson correlation between forecast and ERA5 Z500 **anomalies** over the
   3 × 3 native 5.625° boxes centred on the region box. Anomalies use the existing ERA5 1990–2017 climatology at the
   valid time (it predates every split); weights are the existing exact grid-box areas. Windows that would leave the
   cached array are invalid (never padded); in practice coverage is 100 % because the cached context domain
   (31°S–65°N, 22–147°E) surrounds the target domain.
3. Thresholds (TRAIN only, region × lead × season): `q_primary` = Q90 of `norm_error` (unchanged since v2);
   `acc_q10` = Q10 of `local_acc`.
4. `pattern_bust = (norm_error > q_primary) AND (local_acc < acc_q10)` — AND, fixed before results.
   The magnitude-only label is kept as a diagnostic, not trained as a second production model.

`local_acc` is computed from verification and is a label-construction variable only; it, its threshold and the labels
are in `FORBIDDEN_INPUTS` and are tested never to reach the feature matrix or the pre-reveal API.

## 3. Model and calibration

One shared `CalibratedGBM` (XGBoost hist) on the existing 70 prediction-time-safe features with v2's locked
hyper-parameters (no search); early stopping and isotonic calibration on the validation year only. `raw_probability`
and `calibrated_bust_probability` are kept distinct.

## 4. Protocol

Pre-registration (`config/model_v4.yaml`, commit `47ce766`) → target audit on the dev split → A/B adequacy fit (same
model on the pattern and on the magnitude target; compared by prevalence-independent ROC AUC, with AUPRC/lift reported)
→ dev gate (PROMOTE, 11/11) → freeze (train 2018–20, calibration 2021) → one 2022 evaluation. 2022 was previously
observed by earlier generations; the V4 result is not a pristine project-wide unseen test.

## 5. Interpretation

A pattern-aware bust means large error together with poor local spatial agreement, which is related to phase or
displacement error. Low ACC does not establish a physical cause; the post-verification failure fingerprint gives the
more detailed decomposition. At this resolution low local ACC is also more frequent when the anomaly field is weak
(an ill-conditioned correlation over 9 points), which is partly visible at forecast time — see
`docs/limitations_v4.md`.

## References

* Rodwell, M. J., et al. (2013). Characteristics of occasional poor medium-range weather forecasts for Europe.
  *Bulletin of the American Meteorological Society*, 94, 1393–1405.
* Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD*.
* Lundberg, S. M., et al. (2020). From local explanations to global understanding with explainable AI for trees.
  *Nature Machine Intelligence*, 2, 56–67.
