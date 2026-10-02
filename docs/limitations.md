# Limitations

1. **Atmospheric predictability has a physical ceiling.** Some large errors arise from
   unpredictable growth of small initial uncertainties; no reliability model removes that.
2. **Ensemble spread is imperfect but is a strong baseline.** It is treated as the baseline (B2),
   not as a straw man.
3. **ERA5 is a verification/reference analysis, not perfect truth.** In data-sparse tropical
   regions ERA5 uncertainty can be a non-negligible part of the "error".
4. **The Q90 bust definition is project-defined** (with Q95 sensitivity). Other definitions
   (absolute thresholds, anomaly correlation drops) would give different labels.
5. **The MVP target is Z500 only.** Z500 is the first validated target variable; this does not
   mean the system handles monsoon depressions, heavy rainfall, cyclones or heat waves from Z500
   alone. In the tropics Z500 variability is small, so busts there are small in absolute metres.
6. **NCMRWF operational integration depends on data access.** None was available; the adapter is a
   documented boundary only.
7. **Novel atmospheric states may have weak historical support.** The system says so
   (support / evidence categories) rather than hiding it.
8. **Historical failure signatures are evidence-derived, not causal proof.**
9. **Historical relationships change when the NWP system changes** (IFS cycle upgrades during
   2018–2022 are part of the data). A drift monitor is future work.
10. **Performance must be measured before claiming improvement** — see `docs/evaluation_v3.md` (current
    model) and `docs/evaluation_v2.md` (archived) for the measured results, including comparisons with B2.
11. **Research-dataset performance ≠ operational NCMRWF performance.**
12. **Regional outputs are 5.625° boxes**, not high-resolution local predictions.

## Limitations specific to this build

* **Resolution.** Bandwidth (~1 MB/s) forced the WB2 64×32 (5.625°) member product. Each region
  is one grid box, so "regional RMSE" equals the absolute error of the box-mean Z500 and
  failure signatures are computed over a 7×7-box (≈39°) window; a 3×3 window was tried on dev training data and put 93% of busts in one class (bias-dominated), so it was uninformative. The prompt-level
  5°×5° cell target is approximated by 5.625° boxes.
* **Sampling.** Every second 4-initialisation chunk was downloaded (~50% of initialisations);
  forecast-evolution features exist only when the cycle 24 h earlier is in the sample (≈ half of
  cases) and never for Day 10 (needs +264 h, not downloaded).
* **50 perturbed members only.** The WB2 IFS ENS store has no control member.
* **Short history.** Three training years; Q90 thresholds per (region, lead, season) rest on
  ~100–300 training cases each. Validation (2021) is used for early stopping and calibration; hyper-parameters
  and groups were chosen on the dev split (2020 validation, 2021 dev-test).
* **Single test year (2022).** Year-to-year variability of skill is not captured by one year;
  bootstrap CIs account for sampling within the year only.
* **Variables.** Geopotential (500/700/850 hPa), u/v wind (500/700/850 hPa) and MSLP are used, as
  ensemble mean/std only (members were reduced at download time). Humidity, precipitation and surface
  temperature are not used. Vorticity/divergence at 5.625° describe the resolved synoptic scale only.
* **Year-to-year non-stationarity.** On the dev split the wind/MSLP group improved 2020 validation AUPRC
  by +0.0056 but lost 0.0005 on 2021; the anomaly-sign signal was stronger in 2020 than in 2018–19. With
  only 2–3 training years, beyond-spread relationships measured in one year may not hold in the next.
* **Recent verified-error features assume timely verification.** They use ERA5 up to the
  initialisation time; ERA5 is released with ~5-day latency, so operationally the NWP centre's own
  analysis would have to stand in for the most recent days.
* **In-sample quota effect.** Training labels are TRAIN quantiles, which slightly distorts label-rate
  features inside the training period (see methodology). Validation-based group selection mitigates it.

## Findings from the BMA experiment (dev split only; NO-GO)

* Exchangeable-member BMA on the 50 genuine IFS ENS members is significantly worse than B2 at bust ranking
  (dev-test 2021 AUPRC 0.121 vs 0.158, CI of the difference [−0.045, −0.027]) and worse than V3 (0.140), on
  all 10 lead days and in 92% of regions. Raw mixture probabilities are over-dispersed (ECE 0.060).
* Not evaluated on 2022 and not served. No other model was tried afterwards (pre-agreed stop rule).
* Across V1/V2/V3/BMA, nothing has beaten a calibrated spread-to-threshold baseline at this 5.625°
  single-box scale. The limiting factor appears to be the information in the data, not the learner.

## Findings from the v3 run (quantile gradient boosting; 2022 read for the third time)

* **Does not outperform the spread-only baseline.** On the controlled dev split v3 is below a fresh B2
  (dev-test 2021 AUPRC 0.1396 vs 0.1578; −0.018, CI [−0.027, −0.011]); the single input `spread_thr_ratio`
  alone ranks busts better than v3's probability. Boosting had converged and the exceedance construction
  was checked; no implementation defect was found. v3 is the production model by project decision.
* **Weak discrimination.** 2022 AUPRC 0.1449 (base rate 0.088), ROC AUC 0.636, Brier 0.0785 vs
  climatology 0.0802. The calibrated probability is reliable (ECE 0.0045) but rarely far from climatology.
* **2022 is not an untouched test.** It was read by v1 and v2; v3's result is its first look at 2022 with
  everything frozen beforehand, and still one year only.
* **Quantiles are near-nominal on average, slightly too high in the body** (2022 coverage of q50 0.540,
  q75 0.784): the central error is mildly over-predicted for 2021–22. Coverage is not checked per region.
* **Exceedance probability is an estimate** from six quantiles with assumed exponential tails, not an
  exact CDF; the threshold lies above q95 for 15–16% of rows, where the tail assumption decides the value.
* **Crossing.** Independently fitted quantiles cross on 0.36–0.58% of rows (mostly q90 > q95, small);
  they are rearranged (sorted) at prediction time.
* **Hidden busts mostly missed** (recall 0.075 at the validation 10%-FAR operating point; realised 2022
  FAR at that threshold 0.127, above the 0.10 target).
* **Explanations are model-level.** No per-row attribution is computed for this estimator; the UI shows
  this row's input values next to model-level permutation importance, which is an association only.
* **Static report routes are archival.** `/api/forecast/*` and `/api/replay/*` still serve the v2 replay
  report (B2/Sentinel outputs), labelled as archived; the live demo routes serve v3.

## Findings from the v2 run (2022 second look, configuration locked at 8b196ee)

* **Incremental value over B2 not established.** Sentinel AUPRC 0.1428 vs B2 0.1434; difference −0.0007,
  95% init-day block-bootstrap CI [−0.0028, +0.0017]. The Sentinel beats B2 in 28 of 64 regions.
* **A stronger baseline, measured.** Adding the spread-only `spread_thr_ratio` input raised B2 from 0.1360
  (v1) to 0.1434 on 2022; a textbook spread probability is hard to beat at single-grid-point scale.
* **Hidden busts are still not caught** at the validation-chosen 10% FAR threshold (recall 0 on 2,537
  hidden busts for B2 and the Sentinel; also 0 at 5% FAR).
* **2022 is a second look.** v1 scored it first; v2 choices were made on the dev split only. Test-year
  ablation rows (e.g. ALL 0.147, M3 0.146) were rejected on validation before 2022 was read and are not claimed.
* **Validation optimism is real.** The dev-split selection gained +2.8% on 2020 validation and −0.6% on 2021;
  selection used a two-period stability rule as a consequence (`docs/optimization_v2.md`).

## Findings from the v1 final 2022 test run (historical record)

* **No incremental skill over the spread baseline.** No feature group beat B2 on 2021 validation AUPRC,
  so the Sentinel (FULL) is trained on B2's inputs and its 2022 predictions are identical to B2's
  (AUPRC 0.136 for both; the bootstrap CI of the difference is exactly [0, 0]). Test-year differences of
  individual ablations (up to +0.002 AUPRC) were not selected on validation and are not claimed.
* **Hidden busts are not caught.** At the operating threshold chosen on validation (10% FAR target;
  2022 FAR 0.128), recall on the 2,537 hidden-bust cases (bust with spread ≤ TRAIN P25) is 0.0 for B2
  and the Sentinel. A spread-driven score cannot flag low-spread cases at that threshold.
* **Replay stress cases were selected with 2022 verification** (count of hidden-bust region-days), never
  with model output, and are labelled "not representative"; the 8 random cases are the
  representative sample.
* **Dataset mismatch with the problem statement.** Regions are native ~5.625° boxes, not 5°×5°;
  v1 used geopotential only (v2 adds wind/MSLP); NCMRWF is an adapter interface with no data ingested.
