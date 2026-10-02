# AGENT_STATE

Persistent execution state. Update at every checkpoint.

| Field | Value |
|---|---|
| CURRENT PHASE | v3 (quantile gradient boosting) step 4: dev-split search + development evaluation |
| LAST COMPLETED ACTION | QRF retired (OOM); QGB smoke test on full dev rows passed: 6 quantiles fit in 72 s, peak RSS 2.5 GB, val coverage q10..q95 = .108/.269/.524/.760/.901/.948, crossing 0.63% of rows (mostly q90>q95, median 0.011), val AUPRC (calibrated) 0.147 at default params |
| LOCKED CONFIG | `config/model_v2.yaml` at 8b196ee (B2/Sentinel hyper-parameters, standard learner, groups ATM+EVO+MEM+REC) |
| MODEL STATUS | v2 served: `models/v2/models.joblib` → `artifacts/v2/demo/model`. v1 kept as record (`artifacts/`, `models/models.joblib`) |
| EVALUATION STATUS | v2 2022 (second look): B2 0.1434, Sentinel 0.1428, Δ −0.0007 CI [−0.0028, +0.0017]; incremental value NOT established |
| TEST STATUS | 83 pytest + 8 vitest passing |
| SERVING | `config.served_run()` → artifacts/v2 (FBS_SERVE_RUN= falls back to v1) |
| RENDER | VERIFIED 2026-09-30 on the v2 deploy (https://forecast-bust-sentinel-g0py.onrender.com/). No Render dashboard/log access from this machine |
| DATA | Geopotential 457/457; wind/MSLP 457/457 (validated). Nothing to download |
| KNOWN ISSUES | Render free tier cold start (~5.6 s first open measured warm). 2022 cannot serve as an untouched test set again |
| EXACT NEXT ACTION | Wait for `FBS_RUN=v3 FBS_SPLIT=dev python -m forecast_bust.train_qgb --reuse-table` (logs/v3_qgb_dev_train.log), then `FBS_RUN=v3 FBS_SPLIT=dev python -m forecast_bust.evaluate_qgb` = go/no-go gate. Do NOT touch 2022 before the gate passes |

## Background processes

2026-10-02: v3 QGB dev search under `setsid nohup` (logs/v3_qgb_dev_train.log). QRF leaf-map workaround kept only in `git stash` (retired).
