# Build-spec compliance matrix

Source of truth: `FORECAST_BUST_SENTINEL_FINAL_FROZEN_BUILD_SPEC_v2.md` (SIH26079, v2.0).
Statuses: COMPLETE · PARTIAL · UNVERIFIED · DATA-BLOCKED · NOT IMPLEMENTED.
A row is COMPLETE only when the implementation exists **and** the named evidence was checked in this pass.

_Status as of 2026-09-30, after the v2 final run, the documentation update and the public Render verification.
Served run: v2 (configuration locked at `8b196ee`)._

## SIH alignment (spec §3, §114)

| Requirement | Implementation | Evidence | Test / artifact | Status | Remaining work |
|---|---|---|---|---|---|
| Region-wise confidence | `confidence = 1 - p` per region, "Reliability Confidence" | `demo/engine.py`, `api/app.py` | `test_demo_engine.py`, public UI | COMPLETE | – |
| Day 1–10 outputs | lead days 1–10 = +24…+240 h | `config/data.yaml` | `test_alignment_verification.py`, `test_deploy.py` | COMPLETE | – |
| Bust probability | calibrated v2 Sentinel probability per region × day | `models/sentinel.py`, `artifacts/v2/demo/model` | parity test, `artifacts/v2/metrics.json` | COMPLETE | – |
| Error-prone areas | map + priority queue | `frontend/src/demo/screens.tsx`, `explainability/priority.py` | E2E (13/13 UI = API on public site) | COMPLETE | – |
| Explainable evidence | TreeSHAP of the served booster + analogues + disagreement | `explainability/explain.py`, `demo/engine.py` | `test_api_explain_and_reveal` | COMPLETE | – |
| Dashboard / API | FastAPI + React/Vite on Render | public URL | browser verification 2026-09-30 (0 console errors, 0 failed requests) | COMPLETE | – |

## Scientific

| Requirement | Implementation | Evidence | Test / artifact | Status | Remaining work |
|---|---|---|---|---|---|
| Real forecast data | WB2 IFS ENS 64×32, 457 blocks, 1,828 inits 2018–2022 | `artifacts/v2/dataset_manifest.json` | `test_real_pipeline.py` | COMPLETE | – |
| Wind / MSLP (Group B) | u/v 500/700/850, MSLP ens mean/std → DYN | `features/dynamics.py` | `extras_validation.json` 457/457; M7 in ablations | COMPLETE | evaluated; not selected (failed stability rule) |
| Real verification | ERA5 at valid time | `labels/build.py` | `test_alignment_verification.py` | COMPLETE | – |
| Z500 target | ensemble-mean Z500 error | `config/data.yaml` | `smoke_test.json` | COMPLETE | – |
| Alignment | valid = init + lead | `verification/alignment.py` | tests at +24/+48/+72/+240 h | COMPLETE | – |
| 5°×5° regions | **Documented deviation**: native 5.625° boxes (bandwidth-limited product), one grid point per region | `data/regions.py` | `docs/limitations.md` | COMPLETE (deviation documented) | – |
| Area weighting | exact sin-band cell areas | `verification/metrics.py` | `test_alignment_verification.py` | COMPLETE | trivial for 1-point regions (documented) |
| Training-only normalisation | scale(region, season) from TRAIN | `fit_normalisation` | `test_normalisation_training_only` | COMPLETE | – |
| Q90 label | TRAIN Q90 per region × lead × season | `fit_thresholds` | `test_q90_q95_deterministic`, `v1_audit.json` | COMPLETE | – |
| Q95 sensitivity | retrained B0/B2/FULL on Q95 | `pipeline.py` | `metrics.json:q95_sensitivity` (B2 0.090, FULL 0.092) | COMPLETE | – |
| B0 / B1 / B2 | climatology / spread score / calibrated spread-only GBT (+ `spread_thr_ratio`, same tuning as Sentinel) | `models/sentinel.py` | `metrics.json`, `optimization_v2.md` | COMPLETE | – |
| Sentinel | one shared XGBoost; B2 inputs + ATM + EVO + MEM + REC; locked hyper-parameters | `config/model_v2.yaml` | `experiment_manifest.json`, parity tests | COMPLETE | – |
| B2-conditioned formulation | residual learner boosting from cross-fitted B2 log-odds | `B2Margin` | search (0.1792 < 0.1816), `EXP_RESIDUAL_B2` on test (−0.0003) | COMPLETE (tested, not retained) | – |
| Validation-only calibration | isotonic on 2021 | `CalibratedGBM.fit` | `calibration.json` (+ by lead) | COMPLETE | – |
| Hidden-bust diagnostic | bust ∧ spread ≤ TRAIN P25; validation diagnosis + test recall at 5/10% FAR | `evaluation/metrics.py`, `optimize_v2.py diagnose` | `metrics.json:hidden_bust*`, `diagnosis.json` | COMPLETE | – |
| Causal memory | `valid_time <= init`, no self; engine computes it before inference | `analogues/memory.py`, `demo/engine.py` | `test_memory_*`, `test_live_inference_reproduces_frozen_pipeline` | COMPLETE | – |
| OOD / support | TRAIN Mahalanobis in PCA space, TRAIN-quantile levels | `support/ood.py` | `support_diagnostics.json` | COMPLETE | – |
| Traceable explanations | TreeSHAP of served model, feature/value/direction/contribution/provenance | `explain.py` | `test_api_explain_and_reveal` | COMPLETE | – |
| Failure fingerprint | phase/amplitude/residual decomposition, post-verification | `labels/signature.py`, `replay/build.py` | `fingerprint_metrics.json` (top-1 0.409 vs 0.332) | COMPLETE | – |

## Evaluation

| Requirement | Evidence | Status | Remaining work |
|---|---|---|---|
| Chronological split | train 2018–20 / val 2021 / test 2022 + embargo; dev split 2018–19 / 2020 / 2021 | COMPLETE | – |
| AUPRC, Brier, calibration, precision, recall | `artifacts/v2/metrics.json`, `calibration.json` | COMPLETE | – |
| Fixed-FAR recall | recall@5%/10% FAR | COMPLETE | – |
| Hidden-bust recall | 0 for B2 and Sentinel at 5% and 10% FAR (2,537 hidden busts) | COMPLETE | – |
| Warning lead / peak-day error / spatial | mean lead of detected busts 6.16 d (FULL); peak-day MAE 2.95 d; Jaccard 0.084 | COMPLETE | – |
| Spread-skill / rank histogram | `spread_skill.json` | COMPLETE | – |
| PR curves / calibration by lead / regional breakdown | `pr_curves.json`, `calibration.json:by_lead_day`, `metrics.json:by_region` | COMPLETE | – |
| Ablations | validation (`ablation_validation.json`, 3 seeds, `transfer.json`) and test (descriptive) | COMPLETE | – |
| Confidence intervals | 95% init-day block bootstrap for AUPRC, Brier, recall@10% FAR, hidden-bust recall and differences | COMPLETE | – |
| Optimisation without test contamination | dev split only; 2022 never read by `optimize_v2.py`; config locked before scoring | COMPLETE | – |

## Engineering

| Requirement | Evidence | Status | Remaining work |
|---|---|---|---|
| Python 3.12 | `.venv` 3.12.3 | COMPLETE | – |
| Dependency lock | `requirements-lock.txt` (exact env, uv) + pinned `requirements.txt`; no `uv.lock` | COMPLETE (equivalent lock, spec §77) | – |
| Node 22 | v22.23.3 | COMPLETE | – |
| Frontend build / critical tests | `npm run build`; 8 vitest | COMPLETE | – |
| Backend / API tests | 83 pytest incl. API, deploy, engine parity | COMPLETE | – |
| Real-data replay, blind/reveal | v2 replay + demo bundle; E2E and public-site walkthrough (reveal not requested before click) | COMPLETE | – |
| Provenance | `/api/provenance`, manifests incl. wind/MSLP | COMPLETE | – |
| README / model card / docs / leakage docs | README, `model_card_v2.md`, `evaluation_v2.md`, `optimization_v2.md`, updated docs | COMPLETE | – |
| AGENT_STATE / BUILD_SPEC_COMPLIANCE | this file, `AGENT_STATE.md` | COMPLETE | – |
| Render | public URL verified in browser (all endpoints 200, CORS as designed) | COMPLETE | Render dashboard logs not accessible from this machine |

## Credibility

| Requirement | Status | Notes |
|---|---|---|
| No fabricated data / metrics | COMPLETE | docs tables generated from artifacts (`write_reports.py`, `optimize_v2.py report`) |
| No fake NCMRWF integration | COMPLETE | adapter interface only |
| No unsupported novelty / causal claims | COMPLETE | – |
| Research vs operational status | COMPLETE | "Historical research replay" on every response and in the UI |
| 2022 provenance | COMPLETE | v1 scored 2022 first; v2 is a disclosed second look (README, metrics.json, UI) |
| Limitations | COMPLETE | `docs/limitations.md` incl. v2 findings and non-stationarity |
| Sentinel vs B2 | **Incremental value not established** | Δ AUPRC −0.0007, CI [−0.0028, +0.0017] |
