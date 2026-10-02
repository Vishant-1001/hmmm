# AGENT_STATE

Persistent execution state. Update at every checkpoint.

| Field | Value |
|---|---|
| CURRENT PHASE | v3 frozen (quantile gradient boosting); deployment verification |
| LAST COMPLETED ACTION | v3 final fit (045a9e8) + single 2022 evaluation (45c63b9); API/demo/frontend serve v3; docs; local QA (health v3, run 240 ms, RSS 425 MB) |
| LOCKED CONFIG | `config/model_v3.yaml` qgb.locked_params (lr 0.05, 200 iter, 31 leaves, min leaf 100, l2 1.0), locked at 045a9e8. v2 config unchanged |
| MODEL STATUS | v3 served: `models/v3/qgb_q*.joblib` + calibration → `artifacts/v3/demo/model/v3` (V3PredictiveModel). B2/Sentinel archived (v2), QRF retired |
| EVALUATION STATUS | v3 dev-test 2021 AUPRC 0.1396 (fresh B2 0.1578; Δ −0.018 CI [−0.027,−0.011]). 2022 (V3 first look; 3rd reading of 2022) AUPRC 0.1449, Brier 0.0785 (clim 0.0802), ECE 0.0045. v3 does NOT outperform B2 |
| TEST STATUS | 96 pytest (incl. headless E2E 16/16 UI = API) + 8 vitest passing |
| SERVING | `config.served_run()` → artifacts/v3 (FBS_SERVE_RUN=v2 serves the archive). Static /api/forecast,/api/replay → archived v2 replay report |
| RENDER | VERIFIED 2026-09-30 on the v2 deploy (https://forecast-bust-sentinel-g0py.onrender.com/). No Render dashboard/log access from this machine |
| DATA | Geopotential 457/457; wind/MSLP 457/457 (validated). Nothing to download |
| KNOWN ISSUES | Render free tier cold start; server RSS 425 MB of 512 MB. 2022 read three times (v1, v2, v3): no future test claim possible |
| EXACT NEXT ACTION | None required. Any new model idea must use FBS_SPLIT=dev and cannot claim a 2022 test result |

## Background processes

None. All v3 jobs ran under `setsid nohup` and have exited.
