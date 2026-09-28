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
10. **Performance must be measured before claiming improvement** — see `docs/evaluation.md` for
    the measured result, including whether improvement over B2 was established.
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
  ~100–300 training cases each. Validation (2021) is used both for early stopping and calibration.
* **Single test year (2022).** Year-to-year variability of skill is not captured by one year;
  bootstrap CIs account for sampling within the year only.
* **Variables.** Only geopotential (500/700/850 hPa) was downloaded; winds, MSLP and humidity
  features are not used. "Wind" information enters only through Z500 gradients.
* **Recent verified-error features assume timely verification.** They use ERA5 up to the
  initialisation time; ERA5 is released with ~5-day latency, so operationally the NWP centre's own
  analysis would have to stand in for the most recent days.
* **In-sample quota effect.** Training labels are TRAIN quantiles, which slightly distorts label-rate
  features inside the training period (see methodology). Validation-based group selection mitigates it.

## Findings from the final 2022 test run

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
  the research data are geopotential only; NCMRWF is an adapter interface with no data ingested.
