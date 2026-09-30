# Leakage controls

The label depends on future verification; features must not. Each control below names the code
that enforces it and the test that checks it.

| Risk | Control | Code | Test |
|---|---|---|---|
| Future ERA5 used as a feature | Feature builders read only `ens_mean`, `ens_std`, `m_*` member stats and the 1990–2017 climatology. Verification fields are declared in `FORBIDDEN_INPUTS`. | `features/build.py` | `test_feature_builders_never_read_verification_fields`, `test_model_feature_sets_exclude_labels` |
| Normalisation fitted on all data | Scale = std of ERA5 Z500 anomaly over TRAIN valid times only | `labels/build.py::fit_normalisation` | `test_normalisation_training_only` |
| Threshold fitted on all data / tuned on test | Q90/Q95 of TRAIN normalized error per region × lead × season; fixed in `config/model.yaml` before any evaluation | `fit_thresholds` | `test_thresholds_ignore_non_training_rows`, `test_q90_q95_deterministic` |
| Train rows verifying inside validation | **Purge/embargo**: train (validation) rows whose valid time is after the period end are removed | `labels/build.py::base_table` | pipeline split summary |
| PCA fitted on all data | PCA fitted on TRAIN initialisations only, persisted (`models/pca.joblib`) and applied frozen | `features/build.py::fit_pca` | `artifacts/pca.json` records `fitted_on` |
| Spread percentile / hidden-bust threshold | TRAIN ECDF / TRAIN P25 per group | `spread_percentile`, `fit_thresholds` | `test_spread_percentile_uses_train_reference` |
| Self-analogue / future analogue | Candidate eligible iff `candidate.valid_time <= query.init_time` (so the candidate was initialised earlier *and* already verified) | `analogues/memory.py::eligible_mask` | `test_memory_temporal_cutoff_and_no_self`, `test_memory_features_do_not_use_future_labels` |
| Test labels improving test prediction via memory | Default `causal_online` uses only labels already verified at issue time (the operational situation); `frozen` mode (train+validation memory only) reported as sensitivity | `memory.py`, `pipeline.py` | `test_memory_frozen_mode_excludes_test`, `metrics.json:frozen_memory_sensitivity` |
| Recent-error features using unverified cases | REC uses only forecasts with valid_time ≤ init (same availability rule as memory) | `analogues/recent.py` | `test_recent_error_features_are_causal` |
| Hidden timestamps | Memory-size counts excluded from model inputs | `models/sentinel.py::NON_MODEL_FEATURES` | code review |
| Feature-group choice tuned on test | Groups selected on VALIDATION AUPRC only | `pipeline.py::train` | `experiment_manifest.json:selected_groups` |
| Future forecast cycles | Evolution features use the cycle 24 h *earlier* only | `features/build.py::grid_features` | code review |
| Calibration on test | Isotonic regression fitted on VALIDATION predictions, frozen | `models/sentinel.py::CalibratedGBM` | protocol in `experiment_manifest.json` |
| Operating threshold on test | Alert threshold chosen on VALIDATION (10% FAR) | `evaluation/run.py` | `metrics.json:operating_point.chosen_on` |
| Random splits | Chronological split: train 2018–2020, validation 2021, test 2022 | `config/data.yaml` | `test_season_and_split` |
| Re-using an inspected test year | 2022 was scored by v1 and once more by v2 (locked config); disclosed as a second look in `metrics.json:test_history`, README and UI | `config/model_v2.yaml:test_history` | – |
| Iterating against test | All development used a separate **dev split** (train 2018–19, validation 2020, dev-test 2021; `FBS_SPLIT=dev`). 2022 was evaluated once with the frozen configuration. | `config.py` | BUILD_PROGRESS log |
| v2 B2 input `spread_thr_ratio` | Forecast spread divided by TRAIN constants (Q90 × scale per region/lead/season); never reads the row's error or label | `labels/build.py::spread_threshold_ratio` | `test_spread_threshold_ratio_uses_no_verification` |
| v2 flow tendencies | Differences along the lead axis of the SAME forecast; no other cycle, no verification | `features/dynamics.py::tendency_fields` | `test_tendency_fields_are_within_forecast_rates` |
| v2 wind / MSLP features | Ensemble mean/std of the forecast only; anomalies vs 1990–2017 climatology | `features/dynamics.py` | `test_feature_builders_never_read_verification_fields`, `artifacts/extras_validation.json` |
| v2 hyper-parameter / group tuning on test | All search, ablation and selection ran on the dev split (train 2018–19, val 2020, dev-test 2021); `scripts/optimize_v2.py` asserts the dev namespace and raises if 2022 (`unused`) is requested; configuration locked (`8b196ee`) before v2 scored 2022 | `scripts/optimize_v2.py`, `config/model_v2.yaml` | `artifacts/v2/dev/optimization/*.json`, `logs` lock hashes |
| Live demo using memory built after the case | Engine computes MEM features before inference with the same `valid_time <= init` rule | `demo/engine.py::memory_lookup` | `test_live_inference_reproduces_frozen_pipeline`, `test_memory_is_causal` |
| Support thresholds from labels | Mahalanobis support thresholds are TRAIN distance quantiles; no labels used | `support/ood.py` | code review |

Known, documented compromise: XGBoost early stopping monitors VALIDATION log-loss/AUPRC, and the
isotonic calibrator is also fitted on VALIDATION. Both B2 and Sentinel use the identical protocol,
so the comparison stays fair; the test year is untouched by either step.
