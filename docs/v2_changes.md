# v2 — conformance rebuild (design, protocol, leakage controls)

v2 is a rebuild of the Sentinel model in its **own namespace** (`FBS_RUN=v2` →
`artifacts/v2/`, `models/v2/`, `data/interim/v2/`, `docs/*_v2.md`). The v1 final evaluation
(commit `be8123b`, `artifacts/`, `docs/evaluation.md`) is kept unchanged as the historical record.

## Why a v2

A conformance audit against the frozen build specification (SIH26079 v2.0) found the v1 system
complete on the product journey (real data, labels, B0/B1/B2, causal memory, support/OOD,
evidence, replay, verification, fingerprints, API, UI) with two scientific gaps:

| Spec item | v1 state | v2 change |
|---|---|---|
| §27 Feature Group B — 500/850 hPa wind, MSLP, vorticity, divergence | geopotential only (Z500/700/850, gradients, Laplacian) | **DYN group** from the same IFS ENS store: ensemble-mean/std of u, v (500/700/850 hPa) and MSLP |
| §22/§23 "does richer forecast state add information *beyond* a strong spread baseline?" | every Sentinel variant re-learns the spread relationship from the base rate; all early-stopped at 100–500 trees vs ~1100 for B2 and lost to B2 on validation | Sentinel **unchanged** (frozen single shared GBT). A residual learner (boost from cross-fitted B2 log-odds) is added as an **experimental comparison only** |

Both changes follow directly from the specification; neither was motivated by 2022 results.

## Feature group DYN (Group B completion)

`forecast_bust/features/dynamics.py`. Per region (native 5.625° box), from the ensemble-mean wind:
500 hPa u/v anomalies (vs ERA5 1990–2017 climatology), wind speed at 500/850 hPa, 500–850 hPa
vector shear, relative vorticity and divergence at 500/850 hPa (spherical centred differences,
unit-tested against solid-body rotation), MSLP anomaly and gradient; from the ensemble std:
vector-wind spread at 500/850 hPa and MSLP spread. All are forecast quantities available at
initialisation. Vorticity/divergence at 5.625° are resolved-scale quantities only.

Download: `scripts/download_extras.sh` (resumable). Members are reduced to mean/std at download
time for exactly the initialisations already cached for geopotential; ~100 s per 4-init block at
~1 MB/s (≈13 h for 457 blocks).

## Residual learner (EXPERIMENTAL comparison — not the Sentinel)

The frozen specification (§23) defines the Sentinel as ONE shared gradient-boosted tree classifier
on region, lead-day, season and forecast-state features. The residual learner is **not** equivalent:
it adds the output of a second model (B2) as a fixed offset, i.e. it is a stacked B2 + GBT model.
It is therefore fitted and reported as `EXP_RESIDUAL_B2` for comparison only and is never promoted
to the Sentinel, whatever its validation score (enforced in `pipeline.train`).

`CalibratedGBM(..., base=B2Margin(...))`. The booster's starting margin is B2's log-odds:

* TRAIN rows: **cross-fitted** margins from leave-one-year-out B2 refits (same features and
  early-stopped size as B2), so the residual trees see B2 as it behaves out of sample;
* validation/test rows: margin of the full B2 model (fitted on TRAIN only).

Early stopping on validation decides how many residual trees are supported; 0 trees = B2.
Isotonic calibration is then fitted on validation, as for every model.

## Selection protocol (validation only)

> **Superseded 2026-09-30 (before any v2 scoring of 2022).** The final v2 configuration was chosen by the
> dev-split optimisation in `docs/optimization_v2.md`: tuned hyper-parameters (same grid for B2 and Sentinel),
> learner by validation AUPRC (standard won; the residual learner stays an experimental comparison), and
> groups by a two-period stability rule, all locked in `config/model_v2.yaml`. B2 gained the spread-only
> `spread_thr_ratio` input and DYN gained in-forecast tendencies. The single-seed per-run rule below is still
> computed and recorded (`experiment_manifest.json: fitted.FULL.single_seed_rule_on_this_validation`) for
> information only.

Sentinel (standard learner, fixed a priori): B2 + each candidate group (M1…M7) is fitted; a group
is kept iff it beats B2's **validation** AUPRC; FULL is refitted on B2 + kept groups. The
experimental residual learner repeats the same validation-only group selection independently and
is reported as `EXP_RESIDUAL_B2`. All rules are in `config/model_v2.yaml` and were fixed before
any v2 test scoring. Development runs used the dev split (train 2018–19 / val 2020 /
dev-test 2021), where 2022 is `unused`.

Material-improvement rule (unchanged from v1): the 95% block-bootstrap CI of
AUPRC(FULL) − AUPRC(B2) excludes 0 **and** the relative gain is ≥ 5%.

## New evaluation outputs

* `disagreement_full_vs_b2` — distribution of p_FULL − p_B2 and, in bins (Sentinel higher by
  > 5 pp / within ±5 pp / lower by > 5 pp), observed bust rate vs both mean probabilities and Briers.
* `hidden_bust_at_far` — hidden-bust recall at overall 5% / 10% false-alarm rates (thresholds
  chosen on validation), per model (spec §21 fixed-FAR comparison).

## Test-set disclosure

2022 was scored once by v1. v2 scores it once more after all v2 choices were frozen. No v2
decision used 2022 data, but 2022 is a **second look**, not an untouched test set; this is
recorded in `metrics.json` (`test_history`) and shown in the UI.

## Interim dev result (v2a: no DYN, dev split, informational)

Standard learner: no group beat B2 on 2020 validation → FULL = B2 inputs (same as v1).
Experimental residual learner: all six groups gave small validation gains; on dev-test 2021 AUPRC
0.1520 vs B2 0.1506 (+0.9% relative, block-bootstrap CI [+0.0007, +0.0021]) — statistically
detectable but far below the 5% materiality rule; mean |p − p_B2| 0.38 pp.
