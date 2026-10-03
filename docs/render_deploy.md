# Deploying the Forecast Bust Sentinel demo on Render

**Live:** `https://forecast-bust-sentinel-g0py.onrender.com` is one Render **Web Service**. The FastAPI app
(`forecast_bust.api.app:app`) serves the API under `/api/*` and the built React UI at `/` from the same origin, so the
UI needs no `VITE_API_URL` and makes no localhost calls. The UI bundle (`frontend/dist`) is committed, so the Python
service needs no Node build step. `tests/test_ui_bundle.py` fails if the committed bundle is stale.

The service serves the committed MVP bundle (`artifacts/v2/`: demo cases, the exported **B2** booster and calibrator,
and the unchanged metrics JSONs). It runs only B2 inference on stored real forecast states, and never downloads data,
trains or re-evaluates. It is a **historical replay** on real ECMWF IFS ENS cases, not a live NCMRWF feed. Measured
locally with the Render settings (`MALLOC_ARENA_MAX=2`, `OMP_NUM_THREADS=1`): RSS is 416 MB at startup and 436 MB after
the full judge flow, within the free tier's 512 MB.

**Deployment smoke test** (headless Chromium, the full judge flow, fails on any console error):
`.venv/bin/python scripts/demo_walkthrough.py --url https://forecast-bust-sentinel-g0py.onrender.com`.
The verification record is in `BUILD_PROGRESS.md`.

A separate **Static Site** for the UI (section 3) is optional. It is only needed to host the UI apart from the API.

## 1. Connect the GitHub repository

In Render, connect GitHub and grant it access to `Vishant-1001/hmmm`.

The quickest path is **New → Blueprint** with branch `main`: `render.yaml` creates both services and
asks for `VITE_API_URL` (step 3). To set things up by hand instead, use the values below.

## 2. API: Web Service

| Setting | Value |
|---|---|
| Runtime | Python 3 |
| Root Directory | *(empty: repository root)* |
| Build Command | `pip install -r requirements.txt && pip install --no-deps .` |
| Start Command | `python -m uvicorn forecast_bust.api.app:app --host 0.0.0.0 --port $PORT` |
| Health Check Path | `/health` |
| Environment | `PYTHON_VERSION=3.12.3`, `MALLOC_ARENA_MAX=2`, `OMP_NUM_THREADS=1` |

Once it's **Live**, check it:

```bash
curl https://<api>.onrender.com/health            # {"status":"ok"}
curl https://<api>.onrender.com/api/demo/health   # cases=5, memory_rows=1169920
```

## 3. UI: Static Site (optional)

| Setting | Value |
|---|---|
| Root Directory | `frontend` |
| Build Command | `npm ci && npm run build` |
| Publish Directory | `dist` |
| Environment | `VITE_API_URL=https://<api>.onrender.com` (the API URL from step 2, no trailing slash), `NODE_VERSION=22` |

`VITE_API_URL` is baked in at build time. If you change it, trigger **Manual Deploy → Clear build
cache & deploy**.

## 4. CORS

The API allows `https://*.onrender.com` (and `*.vercel.app`) plus local dev origins, with no
credentials. For a custom domain, set `FBS_CORS_ORIGINS=https://your.domain` on the API
(comma-separated for several).

## 5. Verify

Open the Static Site URL and walk through Overview → Reliability → select a region → Why Flagged →
Verification → Reveal → Model trust.

## Troubleshooting

- **The first load after idle shows "Running Sentinel…" for up to about a minute.** Free Render
  web services sleep when idle. The UI waits, and if a request fails it shows the error with a
  **Retry** button.
- **`DATA UNAVAILABLE (404)` in the UI**: `VITE_API_URL` wasn't set when the static site was built,
  so the UI called its own origin. Set it and redeploy the static site.
- **`FileNotFoundError ... config/model.yaml`**: the API isn't running from the repository checkout.
  `forecast_bust.config` finds the project root from the install location or the working directory;
  set `FBS_PROJECT_ROOT` to the checkout path if the layout differs.
