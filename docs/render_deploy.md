# Deploying the Forecast Bust Sentinel demo (Render API + Vercel UI)

- **API**: the existing FastAPI app (`forecast_bust.api.app:app`) as a native Python 3.12 web
  service on Render. It serves the committed runtime artifacts (`artifacts/demo/`,
  `artifacts/replay/`, metrics JSONs) and runs the frozen exported boosters on stored forecast
  states. It doesn't download data, train, or run evaluation.
- **UI**: the existing Vite/React app in `frontend/`, deployed as a static site on Vercel. It calls
  the API at `VITE_API_URL`.

The site is a **historical replay** demonstration on real ECMWF IFS ENS cases, not a live NCMRWF feed
(every API response carries this `mode` label and the UI shows it).

## 1. Connect the GitHub repository

Connect GitHub to both Render and Vercel and grant them access to `Vishant-1001/hmmm`.

## 2. Render: API web service

**New → Blueprint**, pick the repository, branch `main`. Render reads `render.yaml`:

| Setting | Value |
|---|---|
| Runtime | Python (native), `PYTHON_VERSION=3.12.3` |
| Build command | `pip install -r requirements.txt && pip install --no-deps .` |
| Start command | `python -m uvicorn forecast_bust.api.app:app --host 0.0.0.0 --port $PORT` |
| Health check path | `/health` |
| Env | `MALLOC_ARENA_MAX=2`, `OMP_NUM_THREADS=1`; `FBS_CORS_ORIGINS` optional (below) |

Apply, wait for **Live**, then check it:

```bash
curl https://<api>.onrender.com/health            # {"status":"ok"}
curl https://<api>.onrender.com/api/demo/health   # cases=5, memory_rows=1169920
```

## 3. Vercel: UI

**Add New → Project**, import the repository, then:

| Setting | Value |
|---|---|
| Root Directory | `frontend` |
| Framework preset | Vite (auto; `frontend/vercel.json`) |
| Build / output | `npm run build` → `dist` |
| Environment variable | `VITE_API_URL=https://<api>.onrender.com` (no trailing slash) |

Deploy. From the CLI instead: `cd frontend && npx vercel --prod`, with `VITE_API_URL` set in the
Vercel project settings.

## 4. CORS

The API allows `https://*.vercel.app` and local dev origins. For a custom domain, set
`FBS_CORS_ORIGINS=https://your.domain` on the Render service (comma-separated for several).

## 5. Verify

Open the Vercel URL and walk through Overview → Reliability → select a region → Why Flagged →
Verification → Reveal → Model trust.

## Troubleshooting

- **The first load after idle shows "Running Sentinel…" for up to about a minute.** Render free
  instances sleep when idle. The UI waits, and if a request fails it shows the error with a
  **Retry** button.
- **`DATA UNAVAILABLE (...)` / network error in the UI**: `VITE_API_URL` is missing or wrong. It's
  baked in at build time, so redeploy on Vercel after changing it.
- **`FileNotFoundError ... config/model.yaml`**: the service isn't running from the repository
  checkout. `forecast_bust.config` finds the project root from the install location or the working
  directory; set `FBS_PROJECT_ROOT` to the checkout path if the layout differs.
