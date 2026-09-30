# Demo guide: live-inference Forecast Bust Sentinel (about 3 minutes)

**Historical replay research prototype.** The locked v2 Sentinel and B2 models **execute at request time** on
stored, real ECMWF IFS ENS (WeatherBench 2, 5.625 degrees) forecast states from the 2022 test year. Historical
memory (analogue) features are computed live from the memory store with the causal rule (only cases verified
by the initialisation time), before the model runs. ERA5 verification is stored in a separate file and served
only after the user clicks *Reveal verification*. There is no live feed and no NCMRWF data.

Served bundle: `artifacts/v2/demo/` (selected by `config.served_run()`; set `FBS_SERVE_RUN=` to fall back to
the v1 bundle in `artifacts/demo/`). The v2 Sentinel uses B2's inputs plus ATM, EVO, MEM and REC; wind/MSLP
(DYN) was evaluated and not selected. On 2022 the Sentinel and B2 are statistically indistinguishable
(`docs/evaluation_v2.md`); say so when presenting.

## Start

```bash
scripts/serve.sh                     # API + built UI on http://127.0.0.1:8000 (about 2 s to ready)
# first time only / after UI edits:
cd frontend && npm install && npm run build && cd ..
```

Open http://127.0.0.1:8000. You do not need a download, retraining or the research cache: everything the demo
loads is committed under `artifacts/v2/demo/`. Stop the server with `kill $(cat logs/api.pid)`.

Scripted run of the real UI (headless Chromium). It checks every displayed value against the API and captures
screenshots and, optionally, a paced video:

```bash
.venv/bin/python scripts/demo_walkthrough.py --shots artifacts/screenshots
.venv/bin/python scripts/demo_walkthrough.py --video artifacts/demo_video --pace 3.2   # -> 2 min 50 s .webm (silent; add narration)
```

## What runs on "select case"

`POST /api/demo/cases/{id}/run` loads `artifacts/v2/demo/cases/{id}/forecast_state.parquet` (64 regions x
Day 1-10, which holds no verification column) and then does the following:
1. it looks up the historical forecast-state memory. Candidates are restricted to the same region and lead
   day, verified no later than the init time; the resulting MEM features are Sentinel inputs in v2;
2. it evaluates the exported Sentinel and B2 XGBoost boosters and their validation-fitted isotonic calibrators,
   plus the B0 climatology table;
3. it computes Mahalanobis support/OOD, evidence strength and the transparent priority score;
4. it computes TreeSHAP attributions of the Sentinel booster.

The measured cost is about 15 ms model inference and about 260 ms total per case. The sidebar shows the live
timing.

The model is `models/v2/models.joblib` (v2 final, locked configuration `8b196ee`), exported unchanged to
`artifacts/v2/demo/model/`; a parity test shows the live engine reproduces the research pipeline's
predictions and memory features. It was **not retrained** for the demo. Training covers 2018-2020; early
stopping and isotonic calibration use 2021 only.

The v2 Sentinel = B2 inputs + ATM, EVO, MEM, REC. Its probabilities differ from B2's by 1.3 percentage points
on average on 2022 (|Δ| > 5 pp on 1.4% of region-days), and on 2022 it is not better than B2 (AUPRC 0.1428 vs
0.1434). Disagreement is shown as measured; it is not evidence of skill. The v1 bundle (Sentinel = B2) remains
in `artifacts/demo/` as the historical record.

## Case registry (`artifacts/v2/demo/registry.json`, same rule and cases as v1)

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
