# AGENT_STATE

Persistent execution state for the autonomous final pass. Update at every checkpoint.

| Field | Value |
|---|---|
| CURRENT PHASE | 3: model/data (v1 audit done; waiting for wind/MSLP data) |
| LAST COMPLETED ACTION | Lazy per-row TreeSHAP (77a74f2): Render case run 4.0 s -> 1.7 s, outputs identical; public site re-verified after redeploy (13/13 checks, 591-state sweep, 0 problems) |
| CURRENT ACTION | Wind/MSLP download running under the supervisor |
| CURRENT GIT COMMIT | (see `git log -1`) |
| FILES CHANGED | src/forecast_bust/demo/engine.py, AGENT_STATE.md |
| TEST STATUS | 79 pytest + 8 vitest passing (77a74f2) |
| RENDER STATUS | VERIFIED 2026-09-30 07:10 IST at 77a74f2: one web service serves the UI at / and the API at /api (auto-deploy from main) |
| PUBLIC FRONTEND URL | https://forecast-bust-sentinel-g0py.onrender.com/ (verified) |
| BACKEND URL | https://forecast-bust-sentinel-g0py.onrender.com |
| WIND/MSLP DOWNLOAD STATUS | Running under scripts/download_supervisor.sh; 79/457 at 07:15 IST, ~100 s per block, ETA ~17:30 IST |
| DOWNLOAD CHECKPOINT | `ls data/cache/ens_extra/block_*.nc \| wc -l`; logs/download_supervisor.log, logs/download_extras.log |
| MODEL STATUS | v1 frozen (Sentinel = B2, no gain). v2 (DYN group) code exists and is unit-tested; not trained (data-blocked) |
| EVALUATION STATUS | v1 final 2022 evaluation complete (be8123b) and reproduced exactly (artifacts/diagnosis/v1_audit.json). v2 not run |
| KNOWN ISSUES | Render free tier: ~60 s cold start, ~6 s per model run. No Render API access from this machine |
| DATA-BLOCKED ITEMS | v2 DYN features, v2 dev and final evaluation |
| EXACT NEXT ACTION | When logs/download_supervisor.log says COMPLETE: validate extras (python -m forecast_bust.data.validate_extras), then scripts/run_v2.sh dev, then decide on dev/validation evidence only, then scripts/run_v2.sh final (a single disclosed second look at 2022) |

## Processes

| Purpose | Identity | Log | Stop condition |
|---|---|---|---|
| Wind/MSLP download supervisor | `logs/download_supervisor.pid` (flock `logs/download_supervisor.lock`) | logs/download_supervisor.log | exits when 457/457 extras blocks exist, or after 40 restarts |
| Downloader (child) | `scripts/download_extras.sh` → `python -m forecast_bust.data.wb2 extras` | logs/download_extras.log | exits when all blocks are done; killed by the supervisor after a 30 min stall |
