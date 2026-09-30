"""End-to-end research pipeline on REAL cached WB2 data.

    python -m forecast_bust.train      # assemble -> labels -> features -> memory -> support -> fit models
    python -m forecast_bust.evaluate   # test-set metrics, ablation, calibration, diagnostics
    python -m forecast_bust.replay     # precomputed historical replay cases for the API/UI

The test split is touched exactly once, in `evaluate`, after every model and
calibrator is frozen.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import platform
import subprocess
import time

import joblib
import numpy as np
import pandas as pd

from forecast_bust.analogues.memory import compute_memory_features
from forecast_bust.analogues.recent import recent_error_features
from forecast_bust.config import RUN, clean_json, ARTIFACT_DIR, INTERIM_DIR, MODEL_DIR, REPO_ROOT, data_config, model_config
from forecast_bust.data.assemble import assemble, load_states, states_path
from forecast_bust.features.build import GROUPS, add_features
from forecast_bust.labels.build import build_cases
from forecast_bust.evaluation.metrics import auprc
from forecast_bust.models.sentinel import (CANDIDATE_GROUPS, DESCRIPTIONS, FEATURE_SETS, B0Climatology, B2Margin,
                                           CalibratedGBM, full_feature_set)
from forecast_bust.support.ood import EVIDENCE_LEVELS, SUPPORT_LEVELS, evidence_strength, fit_support, support_distance

log = logging.getLogger(__name__)
TABLE = INTERIM_DIR / "table.parquet"
PRED = INTERIM_DIR / "predictions.parquet"


def _git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def write_dataset_manifest(ds, cases: pd.DataFrame) -> None:
    cfg = data_config()
    splits = {s: {"first_init": str(g["init_time"].min()), "last_init": str(g["init_time"].max()),
                  "n_inits": int(g["init_time"].nunique())} for s, g in cases.groupby("split")}
    common = {"provider": "ECMWF, redistributed by Google Research WeatherBench 2 (public bucket gs://weatherbench2)",
              "documentation": "https://weatherbench2.readthedocs.io/en/latest/data-guide.html",
              "access": "anonymous read via gcsfs/xarray (storage_options token=anon)"}
    blocks = sorted(p.name for p in (REPO_ROOT / cfg["cache_dir"] / "ens").glob("block_*.nc"))
    manifest = {
        "forecast": {**common, "source_name": "ECMWF IFS ENS (operational ensemble), WB2 64x32 conservative regrid",
                     "source_url": cfg["source"]["forecast_store"], "dataset_version": "2018-2022 archive as listed in bucket",
                     "download_date": ds.attrs.get("downloaded", "see block files"),
                     "temporal_coverage": f"{str(ds.init.values.min())[:13]} .. {str(ds.init.values.max())[:13]} "
                                          f"({ds.sizes['init']} initialisations, 00/12 UTC)",
                     "sampling": f"every {cfg['block_stride']}nd store chunk of {cfg['init_chunk_size']} initialisations "
                                 f"({len(blocks)} chunks cached) - bandwidth constraint (~1 MB/s)",
                     "spatial_coverage": f"context lat {ds.latitude.values.min()}..{ds.latitude.values.max()}, "
                                         f"lon {ds.longitude.values.min()}..{ds.longitude.values.max()}",
                     "grid": cfg["source"]["grid"], "variables": "geopotential at 500, 700, 850 hPa",
                     "ensemble_structure": "50 perturbed members (control not included in WB2 store)",
                     "lead_times": "24..240 h every 24 h (Day 1..10)",
                     "preprocessing": "domain subset; Z/9.80665 -> metres; ensemble mean and std (ddof=1); member "
                                      "percentiles/skewness at target boxes",
                     "role": splits},
        "reference": {**common, "source_name": "ERA5 reanalysis (WB2 64x32 conservative regrid)",
                      "source_url": cfg["source"]["reference_store"], "role": "verification reference only (labels)",
                      "note": "ERA5 is a reanalysis/reference product and is not perfect truth.",
                      "preprocessing": "geopotential/9.80665 at valid time = init + lead"},
        "climatology": {**common, "source_name": "ERA5 1990-2017 hour-of-day x day-of-year climatology (WB2)",
                        "source_url": cfg["source"]["climatology_store"],
                        "role": "anomaly reference for features; pre-dates all experiment years (no leakage)"},
        "derived": {"regions": "native 5.625 deg boxes with centres inside 5S-40N, 60E-105E (64 regions)",
                    "labels": "artifacts/thresholds.json", "normalisation": "artifacts/preprocessing.json"},
        "licensing": "ERA5: Copernicus licence (LICENSE file in bucket). IFS ENS: ECMWF data terms - verify the "
                     "WeatherBench 2 licence notes before redistribution. Raw data is NOT committed to git.",
        "not_used": "No NCMRWF data is used; NCMRWF NEPS integration is architected (adapter interface) but not claimed.",
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "dataset_manifest.json").write_text(json.dumps(clean_json(manifest), indent=1, default=str))


def build_table(reassemble: bool = True) -> pd.DataFrame:
    if reassemble or not states_path().exists():
        assemble()
    ds = load_states()
    cases = build_cases(ds)
    write_dataset_manifest(ds, cases)
    df = add_features(cases, ds)
    df, meta, _ = compute_memory_features(df)
    df = recent_error_features(df)
    sup = fit_support(df[df["split"] == "train"])
    dist, lvl = support_distance(df, sup)
    df["support_distance"] = dist
    df["support_level"] = lvl
    df.to_parquet(TABLE, index=False)
    return df


def train(reassemble: bool = True) -> None:
    t0 = time.time()
    df = build_table(reassemble)
    tr, va = df[df["split"] == "train"], df[df["split"] == "validation"]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    models = {"B0": B0Climatology().fit(tr)}
    fitted = {}
    val_auprc = {}
    y_va = va["bust"].values

    v2 = model_config().get("v2", {})
    # hyper-parameters selected on the DEV split validation year (scripts/optimize_v2.py); B2 is tuned
    # with the same grid and budget as the Sentinel, so it is not weakened
    b2_params, s_params = v2.get("b2_xgboost"), v2.get("sentinel_xgboost")

    def _fit(name, feats, base=None, key=None):
        m = CalibratedGBM(feats, name, base=base, params=b2_params if name == "B2" else s_params).fit(tr, va)
        key = key or name
        val_auprc[key] = auprc(y_va, m.predict_raw(va))
        fitted[key] = {"best_iteration": m.best_iteration, "n_features": len(feats), "learner": m.learner,
                       "validation_auprc_raw": val_auprc[key]}
        log.info("fitted %s (%s, %d features, best_iter %d, val AUPRC %.4f)", key, m.learner, len(feats),
                 m.best_iteration, val_auprc[key])
        return m

    models["B2"] = _fit("B2", FEATURE_SETS["B2"])
    # Sentinel learner(s): v1 = "standard" only; v2 config also tries "residual_b2" (boost from the
    # cross-fitted B2 margin). Everything below is decided on VALIDATION only.
    learners = v2.get("learners", ["standard"])
    exp_learners = v2.get("experimental_learners", [])
    if len(learners) != 1 or learners[0] not in ("standard", "residual_b2"):
        raise ValueError("exactly one Sentinel learner (standard | residual_b2), fixed in config before the run")
    per_learner = {}
    for learner in learners + exp_learners:
        base = B2Margin(models["B2"], tr) if learner == "residual_b2" else None
        suffix = "" if learner == "standard" else f"@{learner}"
        cand = {name: _fit(name, feats, base, key=name + suffix) for name, feats in FEATURE_SETS.items()
                if name != "B2"}
        # Group selection: keep a group iff B2+group beats B2 on validation AUPRC. When the config locks
        # the groups (v2: chosen on the dev split with a 3-seed rule, confirmed once on dev-test 2021),
        # the locked set is used and this run's single-seed comparison is recorded for information only.
        rule_selected = [g for m, g in CANDIDATE_GROUPS.items() if val_auprc[m + suffix] > val_auprc["B2"]]
        selected = list(v2["sentinel_groups"]) if "sentinel_groups" in v2 else rule_selected
        cand["FULL"] = _fit("FULL", full_feature_set(selected), base, key="FULL" + suffix)
        fitted["FULL" + suffix]["selected_groups"] = selected
        fitted["FULL" + suffix]["single_seed_rule_on_this_validation"] = rule_selected
        per_learner[learner] = {"models": cand, "selected": selected, "base": base,
                                "val_auprc_full": val_auprc["FULL" + suffix]}
    # The Sentinel learner is fixed in config (selected on the dev split, never on this run's test year)
    chosen = learners[0]
    selected = per_learner[chosen]["selected"]
    models.update(per_learner[chosen]["models"])
    # Experimental comparison only: FULL of each experimental learner, kept under EXP_* names
    exp_models = {f"EXP_{l_.upper()}": per_learner[l_]["models"]["FULL"] for l_ in exp_learners}
    models.update(exp_models)
    FEATURE_SETS_RUN = dict(FEATURE_SETS)
    FEATURE_SETS_RUN["FULL"] = full_feature_set(selected)
    log.info("learner %s; FULL uses validated groups %s (val AUPRC %.4f vs B2 %.4f)", chosen, selected,
             per_learner[chosen]["val_auprc_full"], val_auprc["B2"])
    # Q95 sensitivity (same protocol, different project-defined criterion)
    tr95, va95 = tr.assign(bust=tr["bust_q95"]), va.assign(bust=va["bust_q95"])
    models["B0_q95"] = B0Climatology().fit(tr95)
    models["B2_q95"] = CalibratedGBM(FEATURE_SETS["B2"], "B2_q95", params=b2_params).fit(tr95, va95)
    base95 = B2Margin(models["B2_q95"], tr95) if chosen == "residual_b2" else None
    models["FULL_q95"] = CalibratedGBM(FEATURE_SETS_RUN["FULL"], "FULL_q95", base=base95,
                                       params=s_params).fit(tr95, va95)
    # Frozen-memory sensitivity: memory restricted to train+validation cases
    dff, _, _ = compute_memory_features(df.drop(columns=GROUPS["MEM"]), mode="frozen")
    models["FULL_frozen_memory"] = models["FULL"]  # same model, different memory feature inputs at test time
    dff[["case_id"] + GROUPS["MEM"]].to_parquet(INTERIM_DIR / "mem_frozen.parquet", index=False)
    joblib.dump(models, MODEL_DIR / "models.joblib")

    # predictions for validation and test rows (test is NOT scored here)
    rows = df[df["split"].isin(["validation", "test"])].copy()
    preds = pd.DataFrame({"case_id": rows["case_id"].values})
    preds["p_B0"] = models["B0"].predict(rows)
    preds["s_B1"] = rows["spread_pct"].values
    for name in FEATURE_SETS_RUN:
        preds[f"p_{name}"] = models[name].predict(rows)
    for name in exp_models:
        preds[f"p_{name}"] = models[name].predict(rows)
    preds["p_B0_q95"] = models["B0_q95"].predict(rows)
    preds["p_B2_q95"] = models["B2_q95"].predict(rows)
    preds["p_FULL_q95"] = models["FULL_q95"].predict(rows)
    fz = rows[["case_id"]].merge(dff[["case_id"] + GROUPS["MEM"]], on="case_id", how="left")
    frozen_rows = rows.copy()
    frozen_rows[GROUPS["MEM"]] = fz[GROUPS["MEM"]].values
    preds["p_FULL_frozen_memory"] = models["FULL"].predict(frozen_rows)
    ev = evidence_strength(rows["support_level"].values, rows["an_n_within"].values, rows["an_bust_rate"].values,
                           preds["p_FULL"].values, model_config()["analogues"]["min_analogues"])
    preds["evidence_level"] = ev
    preds.to_parquet(PRED, index=False)

    split_counts = df.groupby("split").agg(rows=("bust", "size"), inits=("init_time", "nunique"),
                                           bust_rate=("bust", "mean"), hidden_bust_rate=("hidden_bust", "mean"),
                                           first_init=("init_time", "min"), last_init=("init_time", "max"))
    manifest = {
        "created": dt.datetime.now(dt.timezone.utc).isoformat(), "git_rev": _git_rev(),
        "python": platform.python_version(), "seed": model_config()["seed"],
        "data_config": data_config(), "model_config": model_config(),
        "splits": json.loads(split_counts.to_json(orient="index", date_format="iso")),
        "feature_groups": GROUPS, "feature_sets": FEATURE_SETS_RUN, "selected_groups": selected, "descriptions": DESCRIPTIONS,
        "sentinel_learner": chosen, "experimental_learners": exp_learners,
        "experimental_models": {k: {"learner": m.learner, "selected_groups": per_learner[m.learner]["selected"],
                                    "n_features": len(m.features)} for k, m in exp_models.items()},
        "learner_rule": "Sentinel = one shared GBT; learner (standard, or boosting from the cross-fitted B2 "
                        "margin) and hyper-parameters fixed in config from the dev-split validation search; "
                        "experimental learners are reported for comparison only",
        "hyperparameters": {"B2": b2_params, "sentinel": s_params},
        "fitted": fitted, "support_levels": SUPPORT_LEVELS, "evidence_levels": EVIDENCE_LEVELS,
        "protocol": "fit on train (early stopping on validation) -> isotonic calibration on validation -> "
                    "frozen -> single evaluation on test",
        "run": RUN or "v1",
        "runtime_s": round(time.time() - t0, 1),
        "software": _versions(),
    }
    (ARTIFACT_DIR / "experiment_manifest.json").write_text(json.dumps(clean_json(manifest), indent=1, default=str))
    log.info("training done in %.0fs", time.time() - t0)


def _versions() -> dict:
    import sklearn
    import xarray
    import xgboost
    return {"numpy": np.__version__, "pandas": pd.__version__, "xarray": xarray.__version__,
            "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__}
