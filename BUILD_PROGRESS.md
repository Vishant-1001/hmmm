# BUILD_PROGRESS

_Last updated: 2026-09-28 03:45 local._

## PROJECT STATUS

| Component | Status |
|---|---|
| Data | REAL WB2 IFS ENS 64×32 (5.625°) members + ERA5 + ERA5 1990–2017 climatology. Sources verified 2026-09-28. Download of every 2nd 4-init chunk in progress (`logs/download.log`, resumable via `scripts/download_all.sh`). |
| Verification | DONE — valid-time alignment, exact area-weighted RMSE, tests (+24/48/72/240 h). GATE 1 smoke test passed on real data (`artifacts/smoke_test.json`). |
| Labels | DONE — TRAIN-only normalisation, Q90 (primary)/Q95 per region×lead×season, hidden busts, purge at split boundaries, failure signatures. |
| B0 | DONE (climatological frequency) |
| B1 | DONE (spread percentile score) |
| B2 | DONE (calibrated spread-only XGBoost, same protocol as Sentinel) |
| Sentinel | DONE — M1..M6 ablations, ALL, FULL = B2 + validation-selected groups |
| Calibration | DONE — validation-only isotonic, frozen |
| Analogue memory | DONE — causal (valid_time ≤ init), causal_online + frozen modes |
| OOD/support | DONE — Ledoit-Wolf Mahalanobis, TRAIN quantile categories; evidence-strength rules |
| API | DONE — FastAPI, `scripts/serve.sh` |
| Frontend | DONE — map, priority queue, trajectory, evidence, analogues, blind replay + reveal, analytics |
| Replay | DONE — blind forecast.json / separate verification.json; fingerprint store |
| Testing | 39 pytest + 3 vitest passing (synthetic TEST FIXTURES only) |
| Metrics | FINAL (2022 test) NOT YET COMPUTED. Only dev-split (2021 dev-test) runs so far, not reported as results. |

Git checkpoint: see `git log` (latest pushed to origin/main).

## Dev-split findings so far (development only; not final results)

* Spread-only B2 is strong; with ~170 train inits no extra feature group beat B2 on validation.
* In-sample "quota effect": analogue bust rate is anti-correlated with the label within TRAIN
  (thresholds are TRAIN quantiles) but positively correlated in validation/test.
* Monotone memory-size counts removed from model inputs (hidden timestamps).

## NEXT SAFE STEP

1. Wait for download to finish (`grep -c block logs/download.log`; 457 blocks).
2. `scripts/run_all.sh` (final split): train → evaluate (2022, once) → replay → reports.
3. Commit artifacts + docs; freeze MVP.

## RISKS / BLOCKERS

* ~1 MB/s link: full member-level 1.5° data infeasible (documented in docs/data_sources.md).
* Incremental value over B2 may not be established — will be reported honestly.
