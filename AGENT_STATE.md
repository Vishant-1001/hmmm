# AGENT_STATE

Persistent execution state. Update at every checkpoint.

| Field | Value |
|---|---|
| CURRENT PHASE | 5: v2 final run (configuration LOCKED at 8b196ee, before any v2 scoring of 2022) |
| LAST COMPLETED ACTION | Dev-split optimisation finished: diagnosis, 20-config search (B2 and Sentinel same grid), seed-averaged ablation, dev-test 2021 check (gain did not transfer), two-period stability rule → groups ATM+EVO+MEM+REC; DYN rejected |
| CURRENT ACTION | `scripts/run_v2.sh final` (detached; log `logs/v2_final.log`; lock hashes `logs/v2_lock_hashes.txt`) |
| LOCKED CONFIG | `config/model_v2.yaml` v2.b2_xgboost / sentinel_xgboost / sentinel_groups; learner standard |
| MODEL STATUS | v1 frozen (Sentinel = B2). v2 locked; final fit running |
| EVALUATION STATUS | v1 2022 result unchanged. v2 2022 = single disclosed second look (running) |
| SERVING | `config.SERVED_ARTIFACT_DIR`: artifacts/v2 once `artifacts/v2/metrics.json` and `artifacts/v2/demo/registry.json` exist, else v1 `artifacts/` |
| RENDER | Verified at 77a74f2 (v1). Needs redeploy + browser verification after v2 bundle is committed |
| PUBLIC URL | https://forecast-bust-sentinel-g0py.onrender.com/ |
| DATA | Geopotential 457/457 blocks; wind/MSLP extras 457/457 blocks (validated). Nothing to download |
| KNOWN ISSUES | Session crash on 2026-09-30 ~21:55 killed an in-session v2 run (it was a child of the session). All long jobs now run under `setsid nohup` |
| EXACT NEXT ACTION | When `RUN_V2_final_DONE` is in logs/v2_final.log: read artifacts/v2/metrics.json; `FBS_RUN=v2 python -m forecast_bust.demo.build`; full pytest + vitest; commit artifacts/v2 + bundle; push; verify Render |

## Background processes

| Purpose | Identity | Log | Stop condition |
|---|---|---|---|
| v2 final run | `pgrep -f "run_v2.sh final"` | logs/v2_final.log | prints RUN_V2_final_DONE or exits non-zero |

No download supervisors or log watchers are running.
