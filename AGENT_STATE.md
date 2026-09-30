# AGENT_STATE

Persistent execution state for the autonomous final pass. Update at every checkpoint.

| Field | Value |
|---|---|
| CURRENT PHASE | 2: public frontend on Render |
| LAST COMPLETED ACTION | Wind/MSLP download supervisor started (single instance); committed same-origin UI bundle |
| CURRENT ACTION | Push, wait for Render auto-deploy, verify public UI at the backend URL |
| CURRENT GIT COMMIT | (see `git log -1`) |
| FILES CHANGED | .gitignore, frontend/dist/, tests/test_ui_bundle.py, scripts/download_supervisor.sh, docs/render_deploy.md, AGENT_STATE.md |
| TEST STATUS | 69 pytest + 8 vitest passed at 9967478; +2 bundle tests passed |
| RENDER STATUS | API live and verified (29/29 endpoints, 5 cases). Public UI: pending the auto-deploy of this commit |
| PUBLIC FRONTEND URL | https://forecast-bust-sentinel-g0py.onrender.com/ (after deploy, not yet verified) |
| BACKEND URL | https://forecast-bust-sentinel-g0py.onrender.com |
| WIND/MSLP DOWNLOAD STATUS | Running under scripts/download_supervisor.sh; 61/457 blocks at start (2026-09-30 06:40 IST) |
| DOWNLOAD CHECKPOINT | `ls data/cache/ens_extra/block_*.nc \| wc -l`; logs/download_supervisor.log, logs/download_extras.log |
| MODEL STATUS | v1 frozen (Sentinel = B2, no gain). v2 (DYN group) code exists and is unit-tested; not trained (data-blocked) |
| EVALUATION STATUS | v1 final 2022 evaluation complete (be8123b). v2 not run |
| KNOWN ISSUES | Render free tier: ~60 s cold start, ~6 s per model run. No Render API access from this machine |
| DATA-BLOCKED ITEMS | v2 DYN features, v2 dev and final evaluation |
| EXACT NEXT ACTION | Verify the public UI; then run the model-failure diagnosis on train/validation only while the download runs |

## Processes

| Purpose | Identity | Log | Stop condition |
|---|---|---|---|
| Wind/MSLP download supervisor | `logs/download_supervisor.pid` (flock `logs/download_supervisor.lock`) | logs/download_supervisor.log | exits when 457/457 extras blocks exist, or after 40 restarts |
| Downloader (child) | `scripts/download_extras.sh` → `python -m forecast_bust.data.wb2 extras` | logs/download_extras.log | exits when all blocks are done; killed by the supervisor after a 30 min stall |
