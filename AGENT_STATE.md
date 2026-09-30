# AGENT_STATE

Persistent execution state for the autonomous final pass. Update at every checkpoint.

| Field | Value |
|---|---|
| CURRENT PHASE | 3: model/data (v1 audit done; waiting for wind/MSLP data) |
| LAST COMPLETED ACTION | Extras validator (forecast_bust.data.validate_extras, 8 tests) and DYN integration dry run on 288 real inits: full coverage, 0 NaN, geostrophic check r=0.92, MSLP vs Z850 anomaly r=0.95 (scripts/dryrun_dyn.py) |
| CURRENT ACTION | Wind/MSLP download running under the supervisor |
| CURRENT GIT COMMIT | (see `git log -1`) |
| FILES CHANGED | src/forecast_bust/data/validate_extras.py, tests/test_validate_extras.py, scripts/dryrun_dyn.py, AGENT_STATE.md |
| TEST STATUS | 69 pytest + 8 vitest (9967478), +2 bundle tests, +8 validator tests: all passing |
| RENDER STATUS | VERIFIED 2026-09-30 06:45 IST: one web service serves the UI at / and the API at /api (auto-deploy from main) |
| PUBLIC FRONTEND URL | https://forecast-bust-sentinel-g0py.onrender.com/ (verified) |
| BACKEND URL | https://forecast-bust-sentinel-g0py.onrender.com |
| WIND/MSLP DOWNLOAD STATUS | Running under scripts/download_supervisor.sh; 61/457 at start (06:40 IST), 68/457 at 06:53 IST, ~100 s per block, ETA ~17:30 IST |
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
