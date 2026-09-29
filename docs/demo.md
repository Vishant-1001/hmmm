# Demo guide: live-inference Forecast Bust Sentinel (about 3 minutes)

**Historical replay research prototype.** The frozen v1 Sentinel and B2 models **execute at request time** on
stored, real ECMWF IFS ENS (WeatherBench 2, 5.625 degrees) Z500 forecast states from the 2022 test year. ERA5
verification is stored in a separate file and served only after the user clicks *Reveal verification*.
There is no live feed and no NCMRWF data.

Current demo uses the completed available research feature set; wind/MSLP extension is pending (the download
was stopped, is incomplete, and is not used anywhere in the demo).

## Start

```bash
scripts/serve.sh                     # API + built UI on http://127.0.0.1:8000 (about 2 s to ready)
# first time only / after UI edits:
cd frontend && npm install && npm run build && cd ..
```

Open http://127.0.0.1:8000. You do not need a download, retraining or the research cache: everything the demo
loads is committed under `artifacts/demo/`. Stop the server with `kill $(cat logs/api.pid)`.

Scripted run of the real UI (headless Chromium). It checks every displayed value against the API and captures
screenshots and, optionally, a paced video:

```bash
.venv/bin/python scripts/demo_walkthrough.py --shots artifacts/screenshots
.venv/bin/python scripts/demo_walkthrough.py --video artifacts/demo_video --pace 3.2   # -> 2 min 50 s .webm (silent; add narration)
```

## What runs on "select case"

`POST /api/demo/cases/{id}/run` loads `artifacts/demo/cases/{id}/forecast_state.parquet` (64 regions x
Day 1-10, which holds no verification column) and then does the following:
1. it evaluates the exported Sentinel and B2 XGBoost boosters and their validation-fitted isotonic calibrators,
   plus the B0 climatology table;
2. it looks up the historical forecast-state memory. Candidates are restricted to the same region and lead
   day, verified no later than the init time;
3. it computes Mahalanobis support/OOD, evidence strength and the transparent priority score;
4. it computes TreeSHAP attributions of the Sentinel booster.

The measured cost is about 15 ms model inference and about 260 ms total per case. The sidebar shows the live
timing.

The model is `models/models.joblib` (v1 final, commit 603b3d3), exported unchanged to `artifacts/demo/model/`,
and a parity test shows it gives identical predictions. It was **not retrained**. Training covers 2018-2020;
isotonic calibration uses 2021 only.

The validated Sentinel equals B2. No feature group beat B2 on 2021 validation, so the Sentinel uses B2's six
inputs and the UI shows disagreement = 0 pp. This is shown as measured, not adjusted.

## Case registry (`artifacts/demo/registry.json`)

The rule was fixed before any demo output was viewed. From the existing 2022 replay set it takes the earliest
random (fixed-seed) case in each season, plus the earliest hidden-bust stress case.

| case_id | init (UTC) | season | selection |
|---|---|---|---|
| 2022012012 | 2022-01-20 12:00 | JF | random (default) |
| 2022020212 | 2022-02-02 12:00 | JF | **stress**: selected by verification, not representative |
| 2022031400 | 2022-03-14 00:00 | MAM | random |
| 2022060600 | 2022-06-06 00:00 | JJAS | random |
| 2022120612 | 2022-12-06 12:00 | OND | random |

## Routes

UI (hash routes): `#/overview`, `#/reliability`, `#/evidence` (Why flagged), `#/priority`, `#/verification`,
`#/trust`.

API (schemas in `src/forecast_bust/demo/schemas.py`):

| Method | Path | Returns |
|---|---|---|
| GET | `/api/demo/health` | status, startup time, memory size |
| GET | `/api/demo/cases` | case registry |
| GET | `/api/demo/model` | model version, windows, features, threshold |
| POST | `/api/demo/cases/{id}/run` | executes the models: 64 x 10 cells, lead summary, priority queue, timings (blind) |
| GET | `/api/demo/cases/{id}/regions/{rid}` | Day 1-10 trajectory |
| GET | `/api/demo/cases/{id}/regions/{rid}/explain?lead_day=d` | attribution, analogues, support, evidence, expected signature |
| GET | `/api/demo/cases/{id}/fields` | ensemble-mean / spread Z500 fields |
| POST | `/api/demo/cases/{id}/regions/{rid}/reveal?lead_day=d` | ERA5 verification, fingerprint, expected vs actual, memory update |

The evaluation artifacts on the Trust screen are the existing `/api/metrics*` endpoints. They were computed
once on 2022 and are not recomputed for the demo.

## Script (the default case; the region is the top entry of the model's own priority queue)

| Time | Screen | Show |
|---|---|---|
| 0:00 | Overview | Badges HISTORICAL REPLAY / RESEARCH PROTOTYPE, the case selector and the live inference time in the sidebar. |
| 0:15 | Case selector | Switch the case and the models re-run (under 1 s), then switch back. |
| 0:30 | Day 1-10 control, map | Bust-risk map and alerts per day. Alerts use the validation 10%-FAR threshold. |
| 0:55 | Priority regions | Click the top region. |
| 1:10 | Reliability | P(bust), reliability confidence, B2, Sentinel, disagreement, support, Day 1-10 trajectory. |
| 1:25 | Why flagged | TreeSHAP drivers with training percentiles, analogue counts and bust rate, support/OOD, evidence quality. |
| 2:05 | Expected failure signature | Distribution over similar past busts. This is historical evidence, not causal proof. |
| 2:15 | Verification | BLIND state, predicted risk and signature. |
| 2:25 | REVEAL VERIFICATION | Actual bust or no bust, normalized error vs threshold, the fingerprint, expected vs actual. |
| 2:55 | Memory update | Verified candidates before and after. The loop is FORECAST, then RELIABILITY, EVIDENCE, VERIFICATION, MEMORY. |
| 3:10 | Trust | B0/B1/B2/Sentinel AUPRC, Brier, calibration, hidden-bust recall 0.0 and warning lead, all as measured. |

Whatever the reveal shows is the real outcome. On the default case the top-priority region-day (R0201, Day 3)
was an alert that verified as **no bust**. Say so. The stress case is available but must be introduced as
selected by verification.

## Likely jury questions

* *Doesn't the ensemble already give uncertainty?* Yes — spread is the baseline (B2, calibrated, same learner).
  We test whether anything beyond spread helps; the Analytics tab shows the measured answer.
* *Is this just forecast-error prediction?* That concept exists. Our contribution is the connected, verified
  workflow: regional Day 1–10 bust probability, spread-baseline disagreement, historical evidence with support
  quality, blind replay and failure fingerprints.
* *Does it improve the forecast / predict weather?* No. It estimates the probability that the existing forecast
  exceeds a defined error threshold.
* *What if it doesn't beat spread?* Then we report that — see `docs/evaluation.md`.
* *NCMRWF?* The architecture supports NCMRWF-compatible ensemble ingestion (adapter contract), but the validated
  research prototype uses real historical ECMWF IFS ENS data.
* *Why Z500?* A coherent large-scale circulation target for the first validated experiment.
* *Why not a Transformer/GNN?* The MVP prioritises validation against a strong baseline; complexity must earn
  its place with measured gains.
