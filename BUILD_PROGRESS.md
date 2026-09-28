# BUILD_PROGRESS

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
