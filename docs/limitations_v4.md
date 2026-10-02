# Limitations — V4 (in addition to `docs/limitations.md`)

1. **Rare event, low precision.** Pattern-aware busts are ~1% of region-days. At the validation 10%-FAR operating
   point 2022 precision is 0.028 and recall 0.305 (745 hits, 26,230 false alarms): about 36 alerts per correctly
   flagged pattern bust. Use as triage.
2. **The large-error component is weakly predicted.** V4's ROC AUC for the large-error criterion alone is 0.54 (2022);
   most of its skill is on the pattern criterion. Alerts' median normalized error is only modestly higher (0.28 vs 0.24).
3. **Part of the pattern signal is a statistical artifact.** Correlation over a 3 × 3 window is ill-conditioned when
   anomalies are weak: the pattern-failure rate is 16.2% in the weakest forecast-anomaly quintile vs 2.6% in the
   strongest (dev-test), and that forecast-time quantity alone predicts pattern failure (ROC AUC 0.669) as well as V4
   does. Part of the target's extra predictability over the magnitude target reflects this, not forecast-bust physics.
4. **Large error and pattern failure are nearly independent here** (joint rate ≈ 1.2–1.4× independence; ~12% of
   magnitude busts are pattern busts). At 5.625° single-box scale the joint criterion behaves differently from
   domain-scale bust studies; the event is a small subset of large-error cases.
5. **Hidden busts are rarely flagged** (pattern hidden-bust recall 0.003 dev-test, 0.023 in 2022).
6. **Instability across fitting years.** Early stopping kept 35 trees on the dev split and 604 on the final split.
7. **2022 is not a pristine test.** It was read by v1, v2 and V3; V4 is its fourth reading (first for this target).
8. **Criterion levels are analogue evidence, not model output.** "Large-error / pattern-failure HIGH" means similar
   verified past states had those outcomes more often than usual; it is not a separate prediction.
9. **Archived engines.** The current demo engine serves V4 only; V3/BMA artifacts remain for reproduction from their
   commits, not as live alternatives.
