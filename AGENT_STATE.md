# AGENT_STATE

Persistent execution state. Update at every checkpoint.

| Field | Value |
|---|---|
| CURRENT PHASE | V4 pattern-aware bust: frozen and integrated (final predictive-core change; hard stop) |
| LAST COMPLETED ACTION | V4 pre-registered (47ce766), dev gate PROMOTE 11/11 (405d2aa), 2022 once (b34b3a5): AUPRC 0.0255 @1.04% prevalence, ROC AUC 0.714; API/frontend serve pattern_aware_xgboost |
| LOCKED CONFIG | `config/model_v3.yaml` qgb.locked_params (lr 0.05, 200 iter, 31 leaves, min leaf 100, l2 1.0), locked at 045a9e8. v2 config unchanged |
| MODEL STATUS | V4 served: models/v4/models.joblib -> artifacts/v4/demo/model/v4 (V4Model). v2/V3/BMA archived |
| EVALUATION STATUS | V4 2022 (4th reading of 2022; first for this target): AUPRC 0.0255 (lift 2.45), ROC AUC 0.714, BSS +0.007, ECE 0.0003, precision 0.028 at alert; large-error component weakly predicted (AUC 0.54) |
| TEST STATUS | 128 pytest (incl. NCMRWF provider GRIB-fixture tests, headless E2E) + 13 vitest passing (2026-10-04) |
| SERVING | `config.served_run()` -> artifacts/v4 (FBS_SERVE_RUN overrides). Static /api/forecast,/api/replay -> archived v2 replay report |
| RENDER | VERIFIED 2026-09-30 on the v2 deploy (https://forecast-bust-sentinel-g0py.onrender.com/). No Render dashboard/log access from this machine |
| DATA | Geopotential 457/457; wind/MSLP 457/457 (validated). Nothing to download |
| KNOWN ISSUES | Render free tier cold start; server RSS 425 MB of 512 MB. 2022 read three times (v1, v2, v3): no future test claim possible |
| NCMRWF | Provider path implemented (2026-10-04). Catalogue confirmed; retrieval NOT verified (no ECDS token, HTTP 401). artifacts/ncmrwf_tigge_availability.json |
| EXACT NEXT ACTION | Add an ECDS token (~/.cdsapirc, url https://ecds.ecmwf.int/api), then run scripts/ncmrwf_tigge_check.py. Hard stop on model/target changes otherwise |

## Background processes

None. All v3 jobs ran under `setsid nohup` and have exited.
