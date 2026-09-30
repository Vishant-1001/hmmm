# Explainability

Every explanation is computed from numbers the system produced. There is no LLM and no
free-text cause. Code: `src/forecast_bust/explainability/explain.py` (explanations),
`support/ood.py` (support and evidence strength), `labels/signature.py` (failure signatures),
`explainability/priority.py` (review queue).

## 1. Model attribution

TreeSHAP contributions (`CalibratedGBM.contributions`, XGBoost `pred_contribs`) of the
Sentinel (FULL) model. They are on the log-odds scale, **before** isotonic calibration.

* **Top drivers:** the 6 features with the largest |contribution|, each with its value, its
  percentile in the TRAIN distribution, and the direction ("raises risk" / "lowers risk").
* **Group attribution:** contributions summed per feature group (SPREAD, ATM, ENS, PAT, EVO,
  MEM, REC). Only groups the FULL model actually uses appear.

These show what the model relied on. They are associations, not physical causes, and the API
returns this caveat (`attribution_note`) alongside the contributions.

**v2 note.** The served v2 Sentinel uses MEM (analogue) features. They are recomputed live and causally by
the engine *before* the booster runs, so the TreeSHAP attributions of MEM inputs describe the same values the
prediction used (parity-tested against the research pipeline).

## 2. Evidence statements

Template sentences filled with computed values:

| Kind | Content |
|---|---|
| A. Ensemble | spread (m), its TRAIN percentile for region × lead × season, member sign agreement |
| B. Atmospheric state | Z500 anomaly and its TRAIN percentile, pattern distance in PC space |
| C. Historical | bust rate among the 30 most similar *already-verified* historical states, how many verified cases were eligible, and the climatological rate |
| C2. Recent verified error | mean normalized error / bias / bust fraction of forecasts verified in the 5 days before initialisation |
| D. Forecast evolution | revision versus the cycle 24 h earlier for the same valid time |
| E. Baseline disagreement | Sentinel − B2 probability ("Confidence Disagreement", a project diagnostic) |

A statement that cannot be computed says so (for example "Historical support unavailable…").
Nothing is filled with a default value.

## 3. Support and evidence strength (kept separate from risk)

* **Support level:** Ledoit-Wolf regularised Mahalanobis distance of the case from the TRAIN
  distribution of the same lead day, in the analogue space. Categories are set by TRAIN
  quantiles 0.90 / 0.975 / 0.995: NORMAL / MODERATE / WEAK / INSUFFICIENT HISTORICAL SUPPORT.
* **Evidence strength:** deterministic rules combining support level, number of analogues
  within the similarity radius, and agreement between the analogue bust rate and the model
  probability (see `evidence_strength` docstring). A high-risk case can still have weak
  evidence, and the UI shows both.

## 4. Failure signature

* **Before verification (historical signature):** distribution of verified failure classes
  among the retrieved analogues that busted (or among all retrieved analogues when fewer than 3
  busted; the `basis` field says which). Its top-1 agreement with the actual fingerprint is
  scored against a climatological reference in `artifacts/fingerprint_metrics.json`.
* **After verification (fingerprint):** exact Murphy (1988) MSE decomposition of the
  ensemble-mean vs ERA5 Z500 over a 7×7-box window: bias² + amplitude² + pattern/phase term.
  POSITION_PHASE if the pattern term is ≥ 60% of MSE, AMPLITUDE_STRUCTURE if bias + amplitude
  is ≥ 60%, otherwise RESIDUAL_MIXED. These are project rules, not meteorological categories,
  and they carry no causal claim.

## 5. Priority queue

`score = p_bust · 0.5^((lead−1)/5) · evidence_weight + 0.25 · max(0, p_bust − p_B2)`.
This is a triage rule for the product. The weights are design choices; they were not fitted
or validated.
