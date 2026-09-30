# BUILD_PROGRESS

_Last updated: 2026-09-30, after the v2 final run (`scripts/run_v2.sh final`, exit 0, 22 min)._

PROJECT: Forecast Bust Sentinel (SIH26079)

## CURRENT STATUS (v2, served)

| Item | Status |
|---|---|
| Environment | Python 3.12.3 `.venv`; `requirements-lock.txt` (exact, uv) + pinned `requirements.txt`; Node 22 |
| Data access | COMPLETE: geopotential 457/457 blocks; wind/MSLP 457/457 blocks (validated); ERA5 reference + climatology |
| Smoke test / verification / labels | COMPLETE (unchanged from v1; `artifacts/smoke_test.json`, `docs/diagnosis_v1.md` audit) |
| B0 / B1 | COMPLETE |
| B2 | COMPLETE, strengthened: + `spread_thr_ratio`; tuned with the same grid as the Sentinel |
| Sentinel | COMPLETE: B2 inputs + ATM + EVO + MEM + REC (locked `8b196ee`, dev-split selection) |
| Optimisation | COMPLETE: `docs/optimization_v2.md` (diagnosis, 20-config search, ablation, dev-test check, stability rule) |
| Calibration | Validation-only isotonic (2021) |
| Historical memory / OOD / explainability / failure signature | COMPLETE (unchanged mechanisms; MEM now a Sentinel input, computed live and causally) |
| Evaluation | COMPLETE: 2022 second look — B2 0.1434 [0.1308, 0.1572], Sentinel 0.1428 [0.1306, 0.1565]; Δ −0.0007 CI [−0.0028, +0.0017]; **incremental value not established**; hidden-bust recall 0 for both |
| API / Frontend / Replay | COMPLETE: serves `artifacts/v2` (`config.served_run()`); v2 replay + live-demo bundle |
| Tests | 83 pytest (incl. real-data pipeline, engine parity, headless E2E) + 8 vitest passing |
| Documentation | README, methodology, data, leakage, limitations, demo, architecture, evaluation_v2, model_card_v2, optimization_v2, BUILD_SPEC_COMPLIANCE |
| Git checkpoint | see `git log`; v2 lock `8b196ee`, evaluation `a30bff2`, integration `16918b2` |

LAST VERIFIED COMMAND: `scripts/run_v2.sh final` → `RUN_V2_final_DONE`; `pytest` 83 passed.

LAST VERIFIED RESULT: `artifacts/v2/metrics.json: full_vs_b2.material_improvement = false`.

BLOCKERS: none. RISKS: 2022 is a second look; one test year; year-to-year non-stationarity of beyond-spread signals.

NEXT SAFE STEP: presentation. Any further model change must be developed on the dev split and cannot reuse 2022 as a test claim.

### Change made after the v2 final run

Provenance only: `pipeline.write_dataset_manifest` now records the wind/MSLP source (`forecast_extras`) and
`artifacts/v2/dataset_manifest.json` was regenerated; the demo bundle's `exported_from` string was corrected. No
model, feature, threshold or metric changed.

---

# v1 record (2026-09-28)

_Last updated: 2026-09-28, after the final 2022 test run (`scripts/run_all.sh`, exit 0, `logs/run_all.log` kept locally)._

## FINAL STATUS

| Item | Status |
|---|---|
| Data | COMPLETE: 457/457 WB2 IFS ENS 64×32 (5.625°) chunks, 2018–2022 (every 2nd 4-init chunk), ERA5 reference, ERA5 1990–2017 climatology |
| Split | train 2018–2020 (1096 inits), validation 2021 (364), test 2022 (366), embargo 18 inits purged at boundaries |
| Final test | COMPLETE: 2022 scored once, after models and calibrators were frozen |
| B2 (calibrated spread-only) | AUPRC 0.136, ROC AUC 0.609, Brier 0.0791, ECE 0.0072, recall@10% FAR 0.198 |
| Sentinel (FULL) | AUPRC 0.136, ROC AUC 0.609, Brier 0.0791, ECE 0.0072, recall@10% FAR 0.198 |
| Sentinel vs B2 | **No improvement established.** No group beat B2 on validation → FULL = B2 inputs → identical predictions; bootstrap CI of ΔAUPRC [0, 0]; pre-registered verdict: `material_improvement: false` |
| Other baselines | B0 AUPRC 0.091 (ROC 0.517); B1 AUPRC 0.131 (ROC 0.609); test base rate 0.088 |
| Ablations (test, not selectable) | M1 0.135, M2 0.136, M3 0.136, M4 0.138, M5 0.137, M6 0.138, ALL 0.138 AUPRC |
| Q95 sensitivity | B0 0.042, B2 0.084, FULL 0.084 AUPRC |
| Calibration | Validation-only isotonic; test ECE 0.0072 (B2/FULL), 0.0134 (B0) |
| Hidden bust | 2,537 test hidden busts; recall 0.0 for B2 and Sentinel at the validation-chosen threshold (B0 0.073); low-spread AUPRC 0.060 |
| Operating point | threshold 0.128 (validation, 10% FAR target) → test precision 0.156, recall 0.246, FAR 0.128 |
| Warning lead / peak day / spatial | mean lead of detected busts 6.3 d; peak-risk-day MAE 3.07 d (35% within 1 d); mean Jaccard 0.082 |
| Spread-skill ratio | 0.79 (Day 1) rising to 0.98–1.02 (Days 3–10) |
| Failure fingerprint | analogue-expected signature top-1 agreement 0.409 vs climatological reference 0.332 |
| Replay | COMPLETE: 12 real 2022 cases (4 stress, selected by verification only; 8 random); blind file has no verification |
| Frontend | COMPLETE: built by run_all; consumes `/api/*` served from final `artifacts/` (no hardcoded metrics) |
| API | Verified with TestClient against final artifacts (health, cases, overview, regions, replay blind/verification, metrics, provenance) |
| Tests | 46 pytest passed, 0 skipped (incl. the real-data pipeline tests); 3 vitest passed |
| Docs | README results, limitations (final findings), data_access, explainability; evaluation.md + model_card.md generated from artifacts |

### Change made after the final run

`tests/test_real_pipeline.py`: Z500 lower sanity bound 4800 → 4500 m. The full dataset contains one
real value below 4800 m (4796.8 m at 64.7°N 146°E, Dec 2022; ERA5 reads 4788 m there) at the edge of
the context domain. This is a test-bound error; data, features and models were not changed.

## Known limitations

* Native ~5.625° grid boxes, not 5°×5° (bandwidth-limited data product).
* Geopotential only (Z500/700/850); no wind, MSLP or vorticity features.
* NCMRWF: adapter interface only, no NCMRWF data ingested.
* One test year; Sentinel shows no skill gain over spread; hidden busts are not detected.

See `docs/limitations.md`.

## Final commit

`be8123b` — "feat: complete forecast bust sentinel MVP (final 2022 evaluation)".

## Next step

SIH presentation/demo preparation (`docs/demo.md`). Any future model change must be developed on the
dev split (`FBS_SPLIT=dev`); 2022 is now inspected and can no longer serve as an untouched test set.

## Interactive demo (2026-09-29)

* Live-inference demo: `src/forecast_bust/demo/` (engine, bundle builder, API schemas), `/api/demo/*`, React
  app `frontend/src/demo/` (Overview, Reliability, Evidence/Why flagged, Priority, Verification, Trust).
* Model: the frozen v1 `models/models.joblib` exported unchanged to `artifacts/demo/model/`, with a parity
  test. It was not retrained and no model, threshold or feature changed. Sentinel = B2 is shown as measured.
* Bundle: `artifacts/demo/` (5 registered 2022 cases under a fixed rule, plus the historical memory store
  `memory.parquet`), committed so a clone runs without the research cache.
* Measured: server ready about 2 s; per-case run about 260 ms (model inference about 15 ms); UI screens under 1 s.
* Tests: pytest includes engine parity, causality, blind API, and a headless-browser E2E that checks
  UI = API values; vitest covers the demo flow on real API responses.
* Wind/MSLP: pending, not used. NCMRWF: adapter interface only.
