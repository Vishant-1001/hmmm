# Architecture

```
 WeatherBench 2 bucket (real IFS ENS, ERA5, ERA5 climatology)
        │  data/wb2.py  (resumable, domain-subset cache; adapters.py = provider contract)
        ▼
 data/cache/*.nc ──► data/assemble.py ──► data/interim/states.nc
                      ensemble mean/std, member stats, ERA5 at valid time, climatology at valid time
        ▼
 labels/build.py  ── regional RMSE, TRAIN normalisation, TRAIN Q90/Q95, hidden busts, failure signatures
        ▼                                         (artifacts/preprocessing.json, thresholds.json)
 features/build.py ── ATM / ENS / PAT (frozen TRAIN PCA) / EVO           (models/pca.joblib, artifacts/pca.json)
 analogues/memory.py ── causal historical forecast-state memory          (artifacts/analogue_memory_*.json)
 support/ood.py ── Mahalanobis support + evidence strength               (artifacts/support_diagnostics.json)
        ▼
 models/sentinel.py ── B0, B1, B2, M1..M5, FULL (XGBoost hist + validation isotonic)  (models/models.joblib)
        ▼
 evaluation/run.py ── single TEST evaluation, ablation, calibration, spread-skill (artifacts/metrics.json ...)
 replay/build.py ── blind forecast.json + separate verification.json per case; fingerprint store
        ▼
 api/app.py (FastAPI) ──► frontend/ (React + TypeScript + Vite; served by FastAPI from frontend/dist)
```

## Product workflow represented in software

NWP ensemble → forecast state (`assemble`) → regional features (`features`) → spread baseline
(`B2`) → conditional Sentinel (`FULL`) → calibration → Day 1–10 risk → priority queue
(`explainability/priority.py`) → historical evidence (`analogues`, `explain.py`) → forecaster
review (UI) → ERA5 verification (`verification.json`, reveal) → failure fingerprint
(`fingerprints_test.parquet`) → historical memory update (verified cases become eligible
analogues via `candidate.valid_time ≤ query.init_time`).

## Technology choices

| Choice | Why |
|---|---|
| xarray + zarr + gcsfs | Reads WB2 zarr directly; only needed chunks are fetched |
| Local NetCDF cache | The link is ~1 MB/s; every chunk is downloaded once |
| XGBoost `hist` on CPU | Tabular, NaN-tolerant, fast on 4 cores, TreeSHAP attributions; no GPU needed |
| Isotonic calibration | Non-parametric, monotone; fitted on validation only |
| Brute-force analogue search per (region, lead) | ≤ ~2 000 cases per group; exact, simple and auditable |
| JSON + Parquet artifacts, no database | Read-only precomputed replay; reproducible; no server state. SQLite/PostgreSQL would add deployment weight without a query need |
| FastAPI | Typed, small, serves the built UI as well |
| React + TS + Vite, hand-written SVG charts | No charting dependency; scientific plots with exact control |

Deep sequence/graph models were not used: the MVP's question is whether any information beyond
spread helps; a strong, calibrated tree baseline answers it with far less risk.

## Directory map

| Path | Contents |
|---|---|
| `config/` | `data.yaml` (sources, domain, split), `model.yaml` (labels, memory, XGBoost, priority) |
| `src/forecast_bust/` | package (see diagram) — `train.py`, `evaluate.py`, `replay/` are the CLIs |
| `scripts/` | `download_all.sh`, `smoke_test.py`, `run_all.sh` |
| `tests/` | pytest suite (synthetic TEST FIXTURES only) |
| `frontend/` | dashboard; `src/test` holds fixture-based UI tests |
| `artifacts/` | committed metrics, manifests, replay cases |
| `models/` | fitted models (regenerated; `.joblib` files are small and committed for the demo) |
| `docs/` | methodology, data, leakage, evaluation, model card, limitations, demo |

The FastAPI app lives in `src/forecast_bust/api/` (no separate `backend/` directory is needed).
