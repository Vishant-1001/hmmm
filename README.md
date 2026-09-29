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
WB2 bucket ─► data (cache, adapters) ─► assemble ─► labels ─► features (ATM/ENS/PAT/EVO)
          ─► analogue memory + recent verified error (causal) ─► support/OOD
          ─► B0/B1/B2 + ablations + FULL (XGBoost + validation isotonic)
          ─► evaluation (2022, once) ─► replay (blind + verification) ─► FastAPI ─► React UI
```

Details: [`docs/architecture.md`](docs/architecture.md). Methodology: [`docs/scientific_methodology.md`](docs/scientific_methodology.md).

## 4. Data sources

Real data only (synthetic arrays appear only in unit tests, labelled TEST FIXTURES):

* **Forecasts:** ECMWF IFS ENS, 50 members, 2018–2022, via WeatherBench 2 — `gs://weatherbench2/datasets/ifs_ens/2018-2022-64x32_equiangular_conservative.zarr`
* **Verification reference:** ERA5 (WeatherBench 2, 64×32). ERA5 is a reanalysis/reference product and is not perfect truth.
* **Anomaly climatology:** ERA5 1990–2017 (pre-dates all experiment years).

Provenance: `artifacts/dataset_manifest.json`; details and verification notes: [`docs/data_sources.md`](docs/data_sources.md).

## 5. Dataset acquisition

```bash
scripts/download_all.sh        # resumable, ~4.5 GB, ~4.5 h on a 1 MB/s link
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

XGBoost (`hist`, CPU) with fixed, regularised hyperparameters (`config/model.yaml`), one shared model for all
regions and lead days. Feature groups: ATM (atmospheric state), ENS (ensemble behaviour), PAT (frozen PCA
pattern), EVO (forecast evolution), MEM (historical forecast-state memory / analogues), REC (recent verified
error behaviour). **FULL = B2 inputs + the groups that improved VALIDATION AUPRC over B2**; ALL (every group)
is reported as well.

## 8. Baselines

B0 climatological bust frequency; B1 spread-percentile score; **B2 calibrated spread-only model** (same
learner, same protocol — deliberately strong).

## 9. Calibration

Isotonic regression fitted on validation predictions only, then frozen. Confidence = 1 − bust probability.
Confidence disagreement = Sentinel − B2 (percentage points), a project-level diagnostic.

## 10. Evaluation

Chronological split: train 2018–2020, validation 2021, test 2022 (evaluated once). All development used a
separate dev split (train 2018–19, validation 2020, dev-test 2021). Primary metric AUPRC; plus Brier, ECE,
reliability curves, ROC AUC, recall at 5/10% FAR, hidden-bust recall, warning lead, peak-risk-day error,
spatial overlap, spread-skill ratio, rank histograms, block-bootstrap CIs. **Results:**
[`docs/evaluation.md`](docs/evaluation.md) (generated from `artifacts/metrics.json`).

**Final 2022 test result (scored once, 366 initialisations, 234,240 region×day cases, bust rate 0.088):**

| Model | AUPRC | ROC AUC | Brier | ECE | Recall @ 10% FAR |
|---|---:|---:|---:|---:|---:|
| B0 climatology | 0.091 | 0.517 | 0.0804 | 0.0134 | 0.075 |
| B1 spread score | 0.131 | 0.609 | — | — | 0.195 |
| B2 calibrated spread-only | 0.136 | 0.609 | 0.0791 | 0.0072 | 0.198 |
| Sentinel (FULL) | 0.136 | 0.609 | 0.0791 | 0.0072 | 0.198 |

**Incremental value over B2 was not established.** No feature group improved VALIDATION (2021) AUPRC over
B2, so under the selection rule fixed in advance FULL uses exactly B2's inputs and its predictions are
identical to B2's. Some ablations (M4, M6, ALL) score 0.138 on 2022, but they were not selected on
validation, so choosing one of them now would be selecting on the test year. At the validation-chosen
operating point, hidden-bust recall is 0.0 for both B2 and the Sentinel. The value demonstrated is the
calibrated Day 1–10 reliability workflow (baseline comparison, historical evidence, support levels, priority
queue, blind replay, verification fingerprint), not a skill gain over ensemble spread.

## 11. Replay

`python -m forecast_bust.replay` builds precomputed **real** 2022 cases. Each case has a blind
`forecast.json` (information available at initialisation + model output) and a separate `verification.json`
served only on "Reveal". Case selection is documented: 4 hidden-bust stress cases chosen by verification only
(not representative) + 8 random test initialisations.

## Interactive demo (live inference)

`scripts/serve.sh`, then open http://127.0.0.1:8000. The frozen Sentinel/B2 models execute on stored real 2022
forecast states for 5 registered cases, and ERA5 verification appears only on *Reveal*. The demo needs no
download or retraining. It uses the completed available research feature set; the wind/MSLP extension is
pending. See [docs/demo.md](docs/demo.md). Screenshots of the running app are in `artifacts/screenshots/`.

## 12–17. Install, train, evaluate, run

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
scripts/download_all.sh
scripts/run_all.sh                    # train -> evaluate -> replay -> reports -> UI build
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
[evaluation](docs/evaluation.md) · [model card](docs/model_card.md) · [limitations](docs/limitations.md) ·
[demo](docs/demo.md) · [SIH source](docs/sih_source.md) · [build progress](BUILD_PROGRESS.md)
