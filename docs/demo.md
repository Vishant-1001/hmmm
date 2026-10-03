# Demo guide: Forecast Bust Sentinel MVP (about 3 minutes)

**Historical replay research prototype. One served model: B2.** The frozen B2 spread model
(`model_id = b2_spread_calibrated`, `B2 v2-final (locked 8b196ee)`, calibration `isotonic_validation_2021_v2`)
**runs at request time** on stored, real ECMWF IFS ENS forecast states (WeatherBench 2, 5.625°) from 2022. ERA5
verification is kept in a separate file and is served only after *Reveal*. There is no live feed and no NCMRWF data.
NCMRWF/TIGGE has a provider implementation and catalogue availability is confirmed, but authenticated retrieval is
pending.

> NWP tells you what it predicts. Forecast Bust Sentinel tells you when to be cautious about it.

The served bundle is `artifacts/v2/demo/`, selected by `config.served_run()`, which defaults to `v2`. The model card is
`artifacts/v2/demo/model/served_b2.json`.

## Start

```bash
scripts/serve.sh                     # API + built UI on http://127.0.0.1:8000
cd frontend && npm install && npm run build && cd ..   # only after UI edits
```

Scripted check of the real UI (headless Chromium). Every displayed value is compared with the API, and the run fails on
any browser console error:

```bash
.venv/bin/python scripts/demo_walkthrough.py --url http://127.0.0.1:8000 --shots artifacts/screenshots/mvp
.venv/bin/python scripts/demo_walkthrough.py --url https://<public-ui> [--api-url https://<public-api>]   # deployment smoke test
```

The walkthrough covers: case select → map → Day 5 → region → bust probability vs B0 → explanation → blind replay
(truth hidden, no `/reveal` request) → reveal (ERA5 error, bust, fingerprint) → reset (truth hidden again) → priority
→ trust (served model id) → direct navigation and hard refresh on every route → back/forward.

## What the numbers mean

* **Bust probability** (B2, calibrated): the estimated chance that this region-day's Z500 ensemble-mean error vs ERA5,
  normalized, exceeds the training 90th percentile for that region, lead day and season. It is **not** the chance of a
  weather event, and it is **not** a global forecast-quality score. About 10% of training region-days were busts.
* **Climatology (B0)**: the training bust rate for that region, lead day and season. This is the baseline comparison.
* **Risk level**: ALERT when the probability is at or above 13.9% (the threshold giving a 10% false-alarm rate on
  validation 2021), ABOVE CLIMATOLOGY when it is above B0, otherwise AT OR BELOW CLIMATOLOGY.
* **Evidence quality / support**: how well verified historical forecast states support this case (analogues and
  Mahalanobis distance). These are context, not model inputs. No "confidence %" is shown.
* **Explanation**: TreeSHAP over B2's 7 inputs, plus forecast-time context (spread, atmospheric state, causal analogues,
  forecast evolution, errors verified before initialisation). Nothing after initialisation is used before reveal.

## What runs on "select case"

`POST /api/demo/cases/{id}/run` loads the stored forecast state (64 regions × Day 1–10, no verification column). It
then evaluates the exported B2 booster and isotonic calibrator (about 50 ms) and the B0 table, runs the causal analogue
lookup (evidence only), and computes support, evidence quality and the priority score. The whole run takes about
0.3 s. `tests/test_demo_engine.py` checks that the served probabilities equal the benchmark's `p_B2` for every
served region-day.

## Case registry (`artifacts/v2/demo/registry.json`)

The selection rule was fixed before any demo output was viewed: the earliest random (fixed-seed) 2022 case in each
season, plus the earliest hidden-bust stress case.

| case_id | init (UTC) | season | selection |
|---|---|---|---|
| 2022012012 | 2022-01-20 12:00 | JF | random (default) |
| 2022020212 | 2022-02-02 12:00 | JF | **stress**: selected by verification, not representative |
| 2022031400 | 2022-03-14 00:00 | MAM | random |
| 2022060600 | 2022-06-06 00:00 | JJAS | random |
| 2022120612 | 2022-12-06 12:00 | OND | random |

## Routes

UI hash routes: `#/overview`, `#/reliability`, `#/evidence`, `#/priority`, `#/verification` (blind replay), `#/trust`.

| Method | Path | Returns |
|---|---|---|
| GET | `/health`, `/api/health` | liveness; served model id, version and calibration |
| GET | `/api/demo/cases` | case registry |
| GET | `/api/demo/model` | served model card (model_id, model_version, provider, calibration_version, dataset_mode) |
| POST | `/api/demo/cases/{id}/run` | B2 for 64 × 10 cells, lead summary, priority queue, `source` provenance (blind) |
| GET | `/api/demo/cases/{id}/regions/{rid}` | Day 1–10 trajectory |
| GET | `/api/demo/cases/{id}/regions/{rid}/explain?lead_day=d` | TreeSHAP, evidence, analogues, expected signature |
| POST | `/api/demo/cases/{id}/regions/{rid}/reveal?lead_day=d` | POST-VERIFICATION: ERA5 error, bust, fingerprint, memory update |
| GET | `/api/metrics` | unchanged B2 benchmark (`artifacts/v2/metrics.json`) |
| GET | `/api/providers` | ECMWF (real, served), NCMRWF/TIGGE (retrieval pending), synthetic (demo only, not served) |

## Script

| Time | Screen | Show |
|---|---|---|
| 0:00 | Overview | Badges: HISTORICAL REPLAY, RESEARCH PROTOTYPE, SOURCE · ECMWF IFS / ERA5. Risk map, Day 1–10 bar, alerts. |
| 0:30 | Priority regions | Click the top region. |
| 0:45 | Reliability | Bust probability vs B0, risk level, spread context, Day 1–10 trajectory against the B0 line. |
| 1:15 | Evidence | TreeSHAP drivers, analogue evidence and support, expected failure signature. |
| 2:00 | Blind replay | BLIND mode: prediction only. Click REVEAL, then show the ERA5 error vs threshold, the outcome and the fingerprint. Click RESET to hide them again. |
| 2:40 | Trust | Served model card. B0/B1/B2/Sentinel table: the Sentinel did not beat B2, so B2 is served. |

Whatever the reveal shows is the real outcome; an alert that verifies as no bust should be said out loud. The stress
case must be introduced as selected by verification.

## Likely jury questions

* *What does the model predict?* The probability that the project-defined large-error threshold is exceeded.
* *What data?* Real ECMWF IFS ENS forecasts with ERA5 verification. NCMRWF/TIGGE has a provider implementation, but
  authenticated retrieval has not been completed.
* *Why ensemble spread?* It is the strongest directly available uncertainty signal, and calibrated it is the validated
  baseline.
* *Does the Sentinel beat B2?* No. On 2022 the difference is −0.0007 AUPRC (95% CI −0.0028 to +0.0017), so incremental
  value was not demonstrated.
* *Synthetic data?* Only labelled test fixtures and a demo-only provider that the served app does not use. It is never
  evidence of skill.
* *Operational data?* The provider abstraction is built for that switch, and the NCMRWF/TIGGE code exists. It needs
  authenticated ECDS access.
* *Why not deep learning?* B2 is CPU-light, fast and explainable. No richer model showed a validated gain.
