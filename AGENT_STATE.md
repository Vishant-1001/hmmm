# AGENT_STATE

Persistent execution state for the autonomous final pass. Update at every checkpoint.

| Field | Value |
|---|---|
| CURRENT PHASE | 4: v2 pipeline (data complete and validated) |
| LAST COMPLETED ACTION | Wind/MSLP download COMPLETE 17:17 IST (457/457, 0 failures, 0 restarts); validator: 457/457 valid, 1828 inits 2018-01-01 -> 2022-12-31, climatology OK (artifacts/extras_validation.json) |
| CURRENT ACTION | scripts/run_v2.sh dev (log logs/v2_dev.log) |
| CURRENT GIT COMMIT | (see `git log -1`) |
| FILES CHANGED | src/forecast_bust/demo/engine.py, AGENT_STATE.md |
| TEST STATUS | 79 pytest + 8 vitest passing (77a74f2) |
| RENDER STATUS | VERIFIED 2026-09-30 07:10 IST at 77a74f2: one web service serves the UI at / and the API at /api (auto-deploy from main) |
| PUBLIC FRONTEND URL | https://forecast-bust-sentinel-g0py.onrender.com/ (verified) |
| BACKEND URL | https://forecast-bust-sentinel-g0py.onrender.com |
| WIND/MSLP DOWNLOAD STATUS | DATA STATUS: COMPLETE and validated (2026-09-30 17:20 IST). Supervisor exited by itself |
| DOWNLOAD CHECKPOINT | `ls data/cache/ens_extra/block_*.nc \| wc -l`; logs/download_supervisor.log, logs/download_extras.log |
| MODEL STATUS | v1 frozen (Sentinel = B2, no gain). v2 (DYN group) code exists and is unit-tested; not trained (data-blocked) |
| EVALUATION STATUS | v1 final 2022 evaluation complete (be8123b) and reproduced exactly (artifacts/diagnosis/v1_audit.json). v2 not run |
| KNOWN ISSUES | Render free tier: ~60 s cold start, ~6 s per model run. No Render API access from this machine |
| DATA-BLOCKED ITEMS | none (wind/MSLP available) |
| EXACT NEXT ACTION | Finish scripts/run_v2.sh dev; review the dev/validation results; then scripts/run_v2.sh final (single disclosed second look at 2022) |

## Processes

| Purpose | Identity | Log | Stop condition |
|---|---|---|---|
| Wind/MSLP download supervisor | `logs/download_supervisor.pid` (flock `logs/download_supervisor.lock`) | logs/download_supervisor.log | exits when 457/457 extras blocks exist, or after 40 restarts |
| Downloader (child) | `scripts/download_extras.sh` → `python -m forecast_bust.data.wb2 extras` | logs/download_extras.log | exits when all blocks are done; killed by the supervisor after a 30 min stall |
