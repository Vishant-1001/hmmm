# AGENT_STATE

Persistent execution state. Update at every checkpoint.

| Field | Value |
|---|---|
| CURRENT PHASE | 10: v2 frozen and deployed |
| LAST COMPLETED ACTION | Public Render verification of v2 in headless Chromium (all endpoints 200; 0 console errors; 0 failed requests; walkthrough 13/13 UI = API; blind until Reveal) |
| LOCKED CONFIG | `config/model_v2.yaml` at 8b196ee (B2/Sentinel hyper-parameters, standard learner, groups ATM+EVO+MEM+REC) |
| MODEL STATUS | v2 served: `models/v2/models.joblib` → `artifacts/v2/demo/model`. v1 kept as record (`artifacts/`, `models/models.joblib`) |
| EVALUATION STATUS | v2 2022 (second look): B2 0.1434, Sentinel 0.1428, Δ −0.0007 CI [−0.0028, +0.0017]; incremental value NOT established |
| TEST STATUS | 83 pytest + 8 vitest passing |
| SERVING | `config.served_run()` → artifacts/v2 (FBS_SERVE_RUN= falls back to v1) |
| RENDER | VERIFIED 2026-09-30 on the v2 deploy (https://forecast-bust-sentinel-g0py.onrender.com/). No Render dashboard/log access from this machine |
| DATA | Geopotential 457/457; wind/MSLP 457/457 (validated). Nothing to download |
| KNOWN ISSUES | Render free tier cold start (~5.6 s first open measured warm). 2022 cannot serve as an untouched test set again |
| EXACT NEXT ACTION | None required. Any new model idea: develop with `FBS_SPLIT=dev` (`scripts/optimize_v2.py`); it cannot reuse 2022 as a test claim |

## Background processes

None. All long jobs this session ran under `setsid nohup` and have exited.
