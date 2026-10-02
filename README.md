# Forecast Bust Sentinel

Research prototype for **SIH26079 — AI-Based Forecast Bust Detection for Medium-Range Weather Forecasts**
(Ministry of Earth Sciences / NCMRWF). The official problem statement, metadata and rubric are reproduced
verbatim, separately from our material, in [`docs/sih_source.md`](docs/sih_source.md).

> NWP tells you what it predicts. Forecast Bust Sentinel estimates when that prediction is likely to fail.

## 1. What it does

Forecast Bust Sentinel is a reliability layer over an **existing** medium-range NWP ensemble. For every
regional cell and lead day (Day 1–10) it estimates the calibrated probability that the NWP ensemble-mean
forecast will exceed a project-defined large-error ("bust") threshold, and it shows:

* **where** (regional risk map) and **when** (Day 1–10 reliability trajectory, peak-risk day),
* **why** (number-backed evidence: spread percentile, atmospheric state, similar historical forecast states,
  recent verified errors, cycle-to-cycle revisions, disagreement with the spread baseline),
* **how strong** the evidence is (historical support / out-of-distribution category, evidence strength),
* **what actually happened** (blind replay → reveal ERA5 verification → failure fingerprint),
* a transparent forecaster **review-priority queue**, and full **benchmark analytics**.

It does **not** forecast the weather and does not replace NWP. "Bust risk 64%" means a 64% probability that
the current forecast's error exceeds the bust threshold — not a 64% chance of any weather event.

## 2. Scientific target (MVP)

| Item | Choice |
|---|---|
| Variable | 500 hPa geopotential height (Z500) — the first validated target variable; not a claim to cover rainfall, cyclones or heat waves |
| Horizon | Day 1..10 = +24 h .. +240 h |
| Domain | target 5°S–40°N, 60°E–105°E; features use a wider context (31°S–65°N, 22°E–146°E) |
| Regions | 64 native 5.625° grid boxes (see §5 for why not 5°) |
| Error | area-weighted regional RMSE of the 50-member ensemble mean vs ERA5 |
| Bust | normalized error > TRAIN Q90 per (region, lead day, season); Q95 sensitivity. **Project-defined criterion.** |
| Hidden bust | bust with spread ≤ TRAIN P25 (diagnostic) |

## 3. Architecture

```
WB2 bucket ─► data (cache, adapters) ─► assemble ─► labels ─► features (ATM/ENS/PAT/EVO/DYN)
          ─► analogue memory + recent verified error (causal) ─► support/OOD
          ─► B0/B1/B2 + ablations + FULL (XGBoost + validation isotonic)
          ─► evaluation (2022) ─► replay (blind + verification) ─► FastAPI ─► React UI
```

Details: [`docs/architecture.md`](docs/architecture.md). Methodology: [`docs/scientific_methodology.md`](docs/scientific_methodology.md).

## 4. Data sources

Real data only (synthetic arrays appear only in unit tests, labelled TEST FIXTURES):

* **Forecasts:** ECMWF IFS ENS, 50 members, 2018–2022, via WeatherBench 2 — `gs://weatherbench2/datasets/ifs_ens/2018-2022-64x32_equiangular_conservative.zarr`: geopotential (500/700/850 hPa) and, from the same store and initialisations, u/v wind (500/700/850 hPa) and MSLP (ensemble mean/std)
* **Verification reference:** ERA5 (WeatherBench 2, 64×32). ERA5 is a reanalysis/reference product and is not perfect truth.
* **Anomaly climatology:** ERA5 1990–2017 (pre-dates all experiment years).

Provenance: `artifacts/dataset_manifest.json`; details and verification notes: [`docs/data_sources.md`](docs/data_sources.md).

## 5. Dataset acquisition

```bash
scripts/download_all.sh        # resumable, ~4.5 GB, ~4.5 h on a 1 MB/s link (geopotential)
scripts/download_extras.sh     # resumable, ~0.4 GB (wind/MSLP ensemble mean/std); validate: python -m forecast_bust.data.validate_extras
python scripts/smoke_test.py   # GATE 1: one real init, Day 3, Z500, ens mean/spread, ERA5 at valid time, regional RMSE
```

The build machine's link to GCS measured ~1 MB/s. Member-level IFS ENS at 1.5° costs ~550 MB per
initialisation (≈9 min), so the 5.625° member product (~9 MB per initialisation) was used and every second
store chunk (≈365 initialisations/year) downloaded. Each region is therefore one 5.625° grid box. See
[`docs/limitations.md`](docs/limitations.md).

## 6. Preprocessing

Ensemble mean/std (ddof = 1) and member statistics per box; Z converted to metres; valid time = init + lead;
TRAIN-only normalisation scale (std of ERA5 Z500 anomaly per region × season); TRAIN-only thresholds;
TRAIN-only PCA (8 components); purge of rows whose verification crosses a split boundary.

## 7. Model

XGBoost (`hist`, CPU), one shared model for all regions and lead days. Feature groups: ATM (atmospheric
state), ENS (ensemble behaviour), PAT (frozen PCA pattern), EVO (forecast evolution), MEM (historical
forecast-state memory / analogues), REC (recent verified error behaviour), DYN (v2: wind, shear, vorticity,
divergence, MSLP, and in-forecast Z500/MSLP/vorticity tendencies).

**BMA experiment (NO-GO, not served):** Bayesian Model Averaging of the 50 genuine IFS ENS members failed its
pre-registered gate on the dev split (dev-test AUPRC 0.121 vs B2 0.158 and V3 0.140) and was not evaluated on 2022.
It is kept as a negative result ([`docs/model_card_bma.md`](docs/model_card_bma.md)).

**v3 (served; archived experiment, not a validated final core): quantile gradient boosting.** Forecast Bust Sentinel models the conditional
distribution of future normalized regional Z500 forecast error using quantile gradient boosting
(`HistGradientBoostingRegressor(loss="quantile")`, q10/q25/q50/q75/q90/q95, one shared configuration) applied to
prediction-time-safe NWP forecast-state features (SPREAD+ATM+ENS+PAT+EVO+MEM+REC+DYN). The upper tail of that
distribution gives an estimated probability of exceeding the unchanged TRAIN-Q90 bust threshold, followed by
validation-only isotonic calibration. B2 and the v2 Sentinel are archived benchmarks, not used in v3 inference;
a QRF attempt was abandoned after repeated out-of-memory kills. Measured (`docs/evaluation_v3.md`): 2022 — V3's
first evaluation of a period already read by v1/v2 — AUPRC 0.1449 (base rate 0.088), Brier 0.0785
(climatology 0.0802), ECE 0.0045, quantile coverage near nominal. On the controlled dev split v3 is **below** the
spread-only B2 (−0.018 AUPRC). Method and references: `docs/scientific_methodology.md` §0; model card
`docs/model_card_v3.md`; run `scripts/run_v3.sh dev|final`.

**v2 (archived; superseded by v3):** hyper-parameters, learner and feature groups were chosen on the development split
only (train 2018–19, validation 2020, dev-test 2021) and locked in `config/model_v2.yaml` before 2022 was
scored ([`docs/optimization_v2.md`](docs/optimization_v2.md)). B2 and the Sentinel were tuned with the same
20-configuration grid and chose the same values. **Sentinel = B2 inputs + ATM + EVO + MEM + REC**: the groups
whose 3-seed mean validation gain over B2 was positive in *both* 2020 and 2021. DYN (wind/MSLP) improved 2020
by +0.0056 AUPRC but lost 0.0005 in 2021, so it was not used.

## 8. Baselines

B0 climatological bust frequency; B1 spread-percentile score; **B2 calibrated spread-only model** (same
learner, same tuning budget and protocol — deliberately strong). v2 adds one spread-only B2 input,
`spread_thr_ratio` = spread / TRAIN bust threshold in metres, after the diagnosis showed that the textbook
reliable-ensemble spread probability beat the v1 tree B2 on validation.

## 9. Calibration

Isotonic regression fitted on validation predictions only, then frozen. Confidence = 1 − bust probability.
Confidence disagreement = Sentinel − B2 (percentage points), a project-level diagnostic.

## 10. Evaluation

Chronological split: train 2018–2020, validation 2021, test 2022. All development used a separate dev split
(train 2018–19, validation 2020, dev-test 2021). Primary metric AUPRC; plus Brier, ECE, reliability curves
(also by lead day), ROC AUC, PR curves, recall at 5/10% FAR, hidden-bust recall, warning lead, peak-risk-day
error, spatial overlap, spread-skill ratio, rank histograms, and 95% initialisation-day block-bootstrap CIs.

**2022 provenance:** 2022 was scored once by v1 (commit `be8123b`). v2 scored it once more after every v2
choice was locked (commit `8b196ee`). No v2 decision used 2022, but this is a disclosed **second look**, not a
pristine test set.

**v2 result on 2022** (366 initialisations, 234,240 region×day cases, bust rate 0.088;
[`docs/evaluation_v2.md`](docs/evaluation_v2.md), generated from `artifacts/v2/metrics.json`):

| Model | AUPRC [95% CI] | ROC AUC | Brier | ECE | Recall @ 10% FAR |
|---|---:|---:|---:|---:|---:|
| B0 climatology | 0.091 | 0.517 | 0.0804 | 0.0134 | 0.075 |
| B1 spread score | 0.131 | 0.609 | — | — | 0.195 |
| B2 calibrated spread-only | 0.1434 [0.1308, 0.1572] | 0.619 | 0.0788 | 0.0061 | 0.213 |
| **Sentinel (FULL)** | 0.1428 [0.1306, 0.1565] | 0.620 | 0.0789 | 0.0082 | 0.212 |

Sentinel − B2 AUPRC = −0.0007 (−0.5%), 95% CI [−0.0028, +0.0017]. **Incremental predictive value over B2 was
not established.** Hidden-bust recall at the validation-chosen 10% FAR threshold is 0 for both (2,537 hidden
busts). The dev-split analysis explains why: the only group with a sizeable validation gain (wind/MSLP) did
not repeat it the next year, and the spread-conditional signals that do repeat are small. Test-year ablation
rows (e.g. ALL 0.147) are descriptive only; they were rejected on validation before 2022 was read.

v1 (for the record, [`docs/evaluation.md`](docs/evaluation.md)): B2 0.136, Sentinel 0.136 (identical
predictions). The strengthened v2 B2 improves on v1's B2 by 0.007 AUPRC on the same year.

## 11. Replay

`python -m forecast_bust.replay` builds precomputed **real** 2022 cases. Each case has a blind
`forecast.json` (information available at initialisation + model output) and a separate `verification.json`
served only on "Reveal". Case selection is documented: 4 hidden-bust stress cases chosen by verification only
(not representative) + 8 random test initialisations.

## Interactive demo (live inference)

`scripts/serve.sh`, then open http://127.0.0.1:8000. The locked v2 Sentinel/B2 models (served from
`artifacts/v2/demo/`; `config.served_run()`) execute on stored real 2022 forecast states for 5 registered
cases; historical memory features are computed live and causally; ERA5 verification appears only on *Reveal*.
The demo needs no download or retraining. See [docs/demo.md](docs/demo.md). Screenshots of the running app are
in `artifacts/screenshots/` (v1 era).

**Hosted demo (Render):** an API web service plus a static site for the UI, both from this repository
(`render.yaml`). Step-by-step setup: [docs/render_deploy.md](docs/render_deploy.md). The hosted API serves the
committed runtime bundle (`artifacts/v2/demo/`, `artifacts/v2/replay/`, v2 metrics JSONs) and never downloads
data or trains.

## Data status

| | Status |
|---|---|
| ECMWF IFS ENS geopotential (Z500/700/850, 50 members, 5.625°), 2018–2022 | **Available**: 457/457 blocks |
| IFS ENS wind (u/v at 500/700/850 hPa) and MSLP ensemble statistics | **Available**: 457/457 blocks, 1,828 initialisations, validated (`artifacts/extras_validation.json`); used by the v2 DYN group |
| ERA5 verification reference + 1990–2017 climatology | **Available** |
| NCMRWF NEPS | **Not available**: adapter interface only |

## 12–17. Install, train, evaluate, run

```bash
python3.12 -m venv .venv && uv pip install --python .venv/bin/python -r requirements-lock.txt && .venv/bin/pip install --no-deps -e .
scripts/download_all.sh && scripts/download_extras.sh
scripts/run_v2.sh dev                 # development split only (never 2022)
.venv/bin/python scripts/optimize_v2.py diagnose|search|refine|ablate|confirm|transfer|report
scripts/run_v2.sh final               # locked config: train -> evaluate 2022 -> replay -> reports -> UI build
FBS_RUN=v2 .venv/bin/python -m forecast_bust.demo.build   # live-demo bundle for the served run
scripts/serve.sh                      # FastAPI + built UI at http://127.0.0.1:8000
.venv/bin/python -m pytest            # backend tests
cd frontend && npm install && npm test && npm run dev   # UI dev server (proxies /api to :8000)
```

Individual stages: `python -m forecast_bust.train`, `python -m forecast_bust.evaluate`,
`python -m forecast_bust.replay`, `python scripts/write_reports.py`. API docs at `/docs`.

## 18. Limitations

See [`docs/limitations.md`](docs/limitations.md) — including the physical predictability ceiling, ERA5 not
being truth, the project-defined Q90 criterion, Z500-only scope, 5.625° resolution, weak support for novel
states, and model-specific (ECMWF IFS) relationships.

## 19. NCMRWF integration status

**Not integrated.** No NCMRWF NEPS data was accessible. `forecast_bust.data.adapters` defines the provider
contract and an `NCMRWFAdapter` boundary that raises `NotImplementedError`. *Research prototype evaluated on
historical ECMWF IFS ENS data. NCMRWF integration is architected but not claimed as operational.*

## 20. Reproducibility

Persisted: dataset manifest, split, preprocessing and normalisation parameters, thresholds, PCA, models,
calibrators, feature schema, analogue/support configuration, seeds, software versions
(`artifacts/experiment_manifest.json`), metrics and replay cases. Raw data is never committed.

## Documents

[architecture](docs/architecture.md) · [methodology](docs/scientific_methodology.md) ·
[data sources](docs/data_sources.md) · [leakage controls](docs/leakage_controls.md) ·
[evaluation v2](docs/evaluation_v2.md) · [optimisation v2](docs/optimization_v2.md) · [model card v2](docs/model_card_v2.md) ·
[evaluation v1](docs/evaluation.md) · [model card v1](docs/model_card.md) · [limitations](docs/limitations.md) ·
[spec compliance](docs/BUILD_SPEC_COMPLIANCE.md) ·
[demo](docs/demo.md) · [SIH source](docs/sih_source.md) · [build progress](BUILD_PROGRESS.md)
