# AGENT_STATE

Persistent execution state. Update at every checkpoint.

| Field | Value |
|---|---|
| CURRENT PHASE | MVP / judging freeze: B2 is the one served model (2026-10-04) |
| LAST COMPLETED ACTION | V4 pre-registered (47ce766), dev gate PROMOTE 11/11 (405d2aa), 2022 once (b34b3a5): AUPRC 0.0255 @1.04% prevalence, ROC AUC 0.714; API/frontend serve pattern_aware_xgboost |
| LOCKED CONFIG | `config/model_v3.yaml` qgb.locked_params (lr 0.05, 200 iter, 31 leaves, min leaf 100, l2 1.0), locked at 045a9e8. v2 config unchanged |
| MODEL STATUS | Served: B2 (b2_spread_calibrated, artifacts/v2/demo/model/b2_booster.json + isotonic, served_b2.json). v2 Sentinel/V3/BMA/V4 archived |
| EVALUATION STATUS | V4 2022 (4th reading of 2022; first for this target): AUPRC 0.0255 (lift 2.45), ROC AUC 0.714, BSS +0.007, ECE 0.0003, precision 0.028 at alert; large-error component weakly predicted (AUC 0.54) |
| TEST STATUS | 128 pytest (incl. headless E2E) + 15 vitest passing (2026-10-04) |
| SERVING | `config.served_run()` -> artifacts/v2 (default). FBS_SERVE_RUN overrides |
| RENDER | VERIFIED 2026-10-04 on c0fab06 (https://forecast-bust-sentinel-g0py.onrender.com): public browser smoke test 24/24, 0 console errors |
| DATA | Geopotential 457/457; wind/MSLP 457/457 (validated). Nothing to download |
| KNOWN ISSUES | Render free tier cold start; server RSS 425 MB of 512 MB. 2022 read three times (v1, v2, v3): no future test claim possible |
| NCMRWF | Provider path implemented (2026-10-04). Catalogue confirmed; retrieval NOT verified (no ECDS token, HTTP 401). artifacts/ncmrwf_tigge_availability.json |
| EXACT NEXT ACTION | Freeze. Optional: ECDS token -> scripts/ncmrwf_tigge_check.py. No model changes |

## Background processes

None. All v3 jobs ran under `setsid nohup` and have exited.
