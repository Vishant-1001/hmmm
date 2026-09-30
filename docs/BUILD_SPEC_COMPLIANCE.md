# Build-spec compliance matrix

Source of truth: `FORECAST_BUST_SENTINEL_FINAL_FROZEN_BUILD_SPEC_v2.md` (SIH26079, v2.0).
Statuses: COMPLETE · PARTIAL · UNVERIFIED · DATA-BLOCKED · NOT IMPLEMENTED.
A row is COMPLETE only when the implementation exists **and** the named evidence was checked in this pass.

_Status as of: 2026-09-30, checkpoint 1 (state recovery). Rows marked "v2 pending" are updated when the v2 run finishes._

## SIH alignment (spec §3, §114)

| Requirement | Implementation | Evidence | Test / artifact | Status | Remaining work |
|---|---|---|---|---|---|
| Region-wise confidence | `confidence = 1 - p` per region, UI "Reliability Confidence" | `demo/engine.py`, `api/app.py` | `tests/test_demo_engine.py`, `tests/test_api.py` | COMPLETE | switch to v2 model after lock |
| Day 1–10 outputs | lead days 1–10 = +24…+240 h | `config/data.yaml:lead_hours` | `test_alignment_verification.py`, `test_deploy.py` (Day 1–10 indexing) | COMPLETE | – |
| Bust probability | calibrated Sentinel probability per region × day | `models/sentinel.py::CalibratedGBM` | `artifacts/metrics.json` | COMPLETE | v2 model pending |
| Error-prone areas | map + priority queue | `frontend/src/demo/screens.tsx`, `explainability/priority.py` | e2e test | COMPLETE | – |
| Explainable evidence | TreeSHAP of the actual prediction + analogues + disagreement | `explainability/explain.py`, `demo/engine.py` | `test_demo_engine.py` | COMPLETE | re-export for v2 model |
| Dashboard / API | FastAPI + React/Vite, Render | `api/app.py`, `frontend/` | `test_api.py`, vitest, public URL | COMPLETE (v1) | redeploy v2 |

## Scientific

| Requirement | Implementation | Evidence | Test / artifact | Status | Remaining work |
|---|---|---|---|---|---|
| Real forecast data | WB2 IFS ENS 64×32, 457 blocks, 1,828 inits 2018–2022 | `data/wb2.py`, `artifacts/dataset_manifest.json` | `test_real_pipeline.py` | COMPLETE | – |
| Wind / MSLP (Group B) | u/v 500/700/850, MSLP ens mean/std | `data/cache/ens_extra` 457/457 blocks | `artifacts/extras_validation.json` (457/457 valid) | COMPLETE | – |
| Real verification | ERA5 at valid time | `labels/build.py::base_table` | `test_alignment_verification.py` | COMPLETE | – |
| Z500 target | ensemble-mean Z500 error | `config/data.yaml:target_level` | `smoke_test.json` | COMPLETE | – |
| Alignment | valid = init + lead | `verification/alignment.py` | tests +24/+48/+72/+240 h | COMPLETE | – |
| 5°×5° regions | **Deviation**: native 5.625° boxes (bandwidth-limited 64×32 product); one grid point per region | `data/regions.py` | `docs/limitations.md` | COMPLETE (documented deviation) | – |
| Area weighting | exact sin-band cell areas | `verification/metrics.py` | `test_alignment_verification.py` | COMPLETE | trivial for 1-point regions (documented) |
| Training-only normalisation | scale(region, season) from TRAIN | `labels/build.py::fit_normalisation` | `test_normalisation_training_only` | COMPLETE | – |
| Q90 label | TRAIN Q90 per region × lead × season | `fit_thresholds` | `test_q90_q95_deterministic`, `v1_audit.json` | COMPLETE | – |
| Q95 sensitivity | same, Q95 | `fit_thresholds` | `metrics.json:q95_sensitivity` | COMPLETE | v2 pending |
| B0 / B1 / B2 | climatology / spread score / calibrated spread-only GBT | `models/sentinel.py` | `metrics.json` | COMPLETE | B2 re-tuned with same budget as Sentinel (v2) |
| Sentinel | single shared XGBoost (CPU hist) | `models/sentinel.py`, `pipeline.py` | `test_v2_components.py` | PARTIAL | v2 not yet trained/evaluated |
| Validation-only calibration | isotonic on validation | `CalibratedGBM.fit` | protocol in manifest | COMPLETE | – |
| Hidden-bust diagnostic | bust ∧ spread ≤ TRAIN P25 | `labels/build.py`, `evaluation/metrics.py` | `metrics.json:hidden_bust`, `docs/diagnosis_v1.md` | COMPLETE | v2 pending |
| Causal memory | `valid_time <= init`, no self | `analogues/memory.py` | `test_memory_temporal_cutoff_and_no_self` | COMPLETE | – |
| OOD / support | TRAIN Mahalanobis in PCA space, TRAIN-quantile levels | `support/ood.py` | `support_diagnostics.json` | COMPLETE | – |
| Traceable explanations | TreeSHAP of served model, provenance per value | `explainability/explain.py` | `test_demo_engine.py` | COMPLETE | – |
| Failure fingerprint | phase / amplitude / residual decomposition, post-verification | `labels/signature.py` | `test_failure_signature_*`, `fingerprint_metrics.json` | COMPLETE | – |

## Evaluation

| Requirement | Implementation | Evidence | Status | Remaining work |
|---|---|---|---|---|
| Chronological split | train 2018–20 / val 2021 / test 2022 + embargo; dev split 2018–19 / 2020 / 2021 | `config/data.yaml` | COMPLETE | – |
| AUPRC, Brier, calibration, precision, recall, fixed-FAR recall, hidden-bust recall, warning lead, peak-day error, spatial Jaccard, spread-skill, rank histogram | `evaluation/metrics.py`, `evaluation/run.py` | v1 `metrics.json` | COMPLETE (v1) | v2 pending |
| Ablations | M1–M7, ALL, FULL | `ablation_results.json` | COMPLETE (v1) | v2 validation ablation pending |
| Confidence intervals | init-day block bootstrap of ΔAUPRC | `metrics.json:full_vs_b2.bootstrap` | PARTIAL | CIs on each primary metric (v2) |

## Engineering

| Requirement | Evidence | Status | Remaining work |
|---|---|---|---|
| Python 3.12 | `.venv` Python 3.12.3 | COMPLETE | – |
| Dependency lock | `requirements.txt` pinned (runtime); no `uv.lock` | PARTIAL | generate `uv.lock` |
| Node 22 | node v22.23.3 | COMPLETE | – |
| Frontend build / critical tests | `npm run build`, 8 vitest | COMPLETE (v1) | re-run after v2 integration |
| Backend / API tests | FastAPI, `tests/test_api.py`, `test_deploy.py` | COMPLETE (v1) | re-run |
| Real-data replay, blind/reveal | `replay/build.py`, `demo/`, e2e test | COMPLETE (v1) | v2 model |
| Provenance | `/api/provenance`, manifests | COMPLETE | – |
| README / model card / docs / leakage docs | `README.md`, `docs/*` | PARTIAL | README still says wind/MSLP "pending"; v2 results to add |
| AGENT_STATE / BUILD_SPEC_COMPLIANCE | this file | PARTIAL | update at every checkpoint |

## Credibility

| Requirement | Status | Notes |
|---|---|---|
| No fabricated data / metrics | COMPLETE | all metrics generated from artifacts; `v1_audit.json` reproduces them independently |
| No fake NCMRWF integration | COMPLETE | adapter interface only, stated everywhere |
| 2022 provenance | COMPLETE | 2022 scored once by v1 (be8123b); any v2 score is a disclosed **second look** |
| Limitations clear | PARTIAL | add v2 findings |
