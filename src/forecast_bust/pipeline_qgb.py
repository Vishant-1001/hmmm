"""v3 training pipeline: quantile gradient boosting (QGB) production model + archival dev-split
B0/B2 reference comparison.

Reuses `pipeline.build_table()` unchanged (data assembly, labels, Q90/Q95 thresholds, features,
historical memory, support/OOD). Only the model-fitting stage changes: the retired B2/Sentinel
XGBoost classifiers (`models/sentinel.py`, historical only) and the abandoned QRF are replaced by
the QGB quantile family (`models/qgb.py`). On the dev split a fresh B2 is also fitted with the
frozen v2 hyper-parameters purely as a reference number - never a QGB input, never production.

    FBS_RUN=v3 FBS_SPLIT=dev python -m forecast_bust.train_qgb --reuse-table   # dev search + fit
    FBS_RUN=v3                python -m forecast_bust.train_qgb                 # final fit, locked params
"""
from __future__ import annotations

import datetime as dt
import gc
import json
import logging
import time

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from forecast_bust.config import DEV, RUN, clean_json, ARTIFACT_DIR, MODEL_DIR, REPO_ROOT, model_config
from forecast_bust.evaluation.metrics import auprc, brier, mae, pinball_loss, rmse
from forecast_bust.models.qgb import (CROSSING_FIX, DEFAULT_PARAMS, ESTIMATOR, EXCEEDANCE_METHOD, FIXED_PARAMS,
                                      MODEL_TYPE, QCOLS, QUANTILES, QGBErrorModel, crossing_stats)
from forecast_bust.models.sentinel import B0Climatology, CalibratedGBM, FEATURE_SETS
from forecast_bust.pipeline import PRED, TABLE, _git_rev, _versions, build_table

log = logging.getLogger(__name__)
FEATURE_GROUPS = ["SPREAD", "ATM", "ENS", "PAT", "EVO", "MEM", "REC", "DYN"]  # existing groups, unchanged
REGISTRY = REPO_ROOT / "artifacts" / "experiment_registry.json"  # accumulates across runs, top level
META_COLS = ["case_id", "init_time", "region_id", "lead_day", "season", "split", "norm_error", "q_primary",
             "bust", "bust_q95", "hidden_bust", "spread_pct"]
PINBALL_TOLERANCE = 0.01  # selection guardrail: mean pinball within 1% of the best candidate


def qgb_features() -> list[str]:
    """SPREAD + ATM + ENS + PAT + EVO + MEM + REC (+ DYN when enabled): FEATURE_SETS["ALL"]."""
    return FEATURE_SETS["ALL"]


def load_rows() -> pd.DataFrame:
    """Only the columns the model and its evaluation need, and only train/validation/test rows -
    one copy of the table in memory instead of all ~100 columns (2022 is 'unused' on the dev split)."""
    if not TABLE.exists():
        build_table(True)
    have = set(pq.ParquetFile(TABLE).schema_arrow.names)
    cols = [c for c in dict.fromkeys(META_COLS + qgb_features() + FEATURE_SETS["B2"]) if c in have]
    df = pd.read_parquet(TABLE, columns=cols, filters=[("split", "in", ["train", "validation", "test"])])
    return df.reset_index(drop=True)


def _val_scores(m: QGBErrorModel, va: pd.DataFrame) -> dict:
    y = va["norm_error"].to_numpy()
    raw = m.raw_quantiles(va)
    out = m.predict_all(va)
    pin = {c: pinball_loss(y, out[c].to_numpy(), ql) for c, ql in zip(QCOLS, QUANTILES)}
    return {"val_auprc": auprc(va["bust"].to_numpy(), out["calibrated_bust_probability"].to_numpy()),
            "val_auprc_raw_exceedance": auprc(va["bust"].to_numpy(), out["estimated_exceedance_probability"].to_numpy()),
            "val_brier_raw_exceedance": brier(va["bust"].to_numpy(), out["estimated_exceedance_probability"].to_numpy()),
            "val_mae_q50": mae(y, out["q50"].to_numpy()), "val_rmse_q50": rmse(y, out["q50"].to_numpy()),
            "val_pinball": pin, "val_pinball_mean": float(np.mean(list(pin.values()))),
            "val_coverage": {c: float((y <= out[c].to_numpy()).mean()) for c in QCOLS},
            "val_crossing": crossing_stats(raw)}


def search_qgb(tr: pd.DataFrame, va: pd.DataFrame, grid: dict, seed: int) -> dict:
    """Small dev-split-only grid; ONE configuration shared by all six quantiles. Selection: among
    candidates whose mean validation pinball loss is within PINBALL_TOLERANCE of the best (the
    distributional guardrail), the highest validation AUPRC of the calibrated bust probability.
    Only metrics are kept; every candidate model is released before the next is fitted."""
    feats = qgb_features()
    keys = list(grid)
    combos = [dict(zip(keys, v)) for v in __import__("itertools").product(*(grid[k] for k in keys))]
    log.info("qgb search: %d configs x %d quantiles on %d train rows", len(combos), len(QUANTILES), len(tr))
    runs = []
    for params in combos:
        params = {**DEFAULT_PARAMS, **params}
        t0 = time.time()
        m = QGBErrorModel(feats, params=params, seed=seed).fit(tr, va)
        runs.append({"params": params, **_val_scores(m, va), "fit_seconds": round(time.time() - t0, 1)})
        r = runs[-1]
        log.info("qgb search %s -> val AUPRC %.4f | pinball %.4f | mae %.4f | crossing %.4f (%.0fs)", params,
                 r["val_auprc"], r["val_pinball_mean"], r["val_mae_q50"], r["val_crossing"]["share_rows_crossing"],
                 r["fit_seconds"])
        (ARTIFACT_DIR / "qgb_search_partial.json").write_text(json.dumps(clean_json(runs), indent=1))
        del m
        gc.collect()
    best_pin = min(r["val_pinball_mean"] for r in runs)
    eligible = [r for r in runs if r["val_pinball_mean"] <= best_pin * (1 + PINBALL_TOLERANCE)]
    best = max(eligible, key=lambda r: r["val_auprc"])
    return {"grid": grid, "configs_tested": len(runs), "runs": runs,
            "selection_rule": f"max validation AUPRC (calibrated) among configs with mean pinball <= "
                              f"{1 + PINBALL_TOLERANCE:.2f} x best mean pinball",
            "eligible": len(eligible), "best_params": best["params"], "best_val_auprc": best["val_auprc"],
            "best_val_pinball_mean": best["val_pinball_mean"]}


def permutation_importance_q90(m: QGBErrorModel, va: pd.DataFrame, seed: int, n: int = 20000) -> dict:
    """Model-level importance: increase in q90 pinball loss when one feature is permuted (validation
    sample). q90 because the upper tail drives the bust probability. Not a per-row explanation."""
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import make_scorer, mean_pinball_loss
    sub = va.sample(n=min(n, len(va)), random_state=seed)
    r = permutation_importance(m.estimators["q90"], m._X(sub), sub["norm_error"].to_numpy(),
                               scoring=make_scorer(mean_pinball_loss, alpha=0.9, greater_is_better=False),
                               n_repeats=3, random_state=seed)
    return {f: float(v) for f, v in sorted(zip(m.features, r.importances_mean), key=lambda kv: -kv[1])}


def train_qgb(reassemble: bool = True, reuse_table: bool = False) -> None:
    t0 = time.time()
    if not (reuse_table and TABLE.exists()):
        build_table(reassemble)
    df = load_rows()
    tr, va = df[df["split"] == "train"], df[df["split"] == "validation"]
    qcfg = model_config()["qgb"]
    seed = model_config()["seed"]
    locked = qcfg.get("locked_params") or {}
    search = None
    if locked:
        params = {**DEFAULT_PARAMS, **locked}
        log.info("qgb: using locked params %s", params)
    else:
        if not DEV:
            raise RuntimeError("qgb.locked_params is empty: hyper-parameters are chosen on FBS_SPLIT=dev only")
        params = dict(DEFAULT_PARAMS)
        if qcfg.get("grid"):
            search = search_qgb(tr, va, qcfg["grid"], seed)
            params = search["best_params"]
            (ARTIFACT_DIR / "qgb_search.json").write_text(json.dumps(clean_json(search), indent=1))
        log.info("qgb: dev selection %s", params)

    feats = qgb_features()
    ft = time.time()
    qgb = QGBErrorModel(feats, params=params, seed=seed).fit(tr, va, model_dir=MODEL_DIR)
    fit_s = round(time.time() - ft, 1)
    importance = permutation_importance_q90(qgb, va, seed)

    # archival references only (never a QGB input / production): B0 always, fresh B2 on dev only
    b0 = B0Climatology().fit(tr)
    refs = {"B0": b0}
    if DEV:
        b2_params = model_config().get("v2", {}).get("b2_xgboost")
        refs["B2_dev_reference"] = CalibratedGBM(FEATURE_SETS["B2"], "B2_dev_reference", params=b2_params,
                                                 seed=seed).fit(tr, va)
    joblib.dump(refs, MODEL_DIR / "reference_models.joblib")

    rows = df[df["split"].isin(["validation", "test"])]
    preds = qgb.predict_all(rows)
    preds.insert(0, "case_id", rows["case_id"].to_numpy())
    preds["p_B0"] = b0.predict(rows)
    if "B2_dev_reference" in refs:
        preds["p_B2_dev_reference"] = refs["B2_dev_reference"].predict(rows)
    preds.to_parquet(PRED, index=False)

    thresholds = json.loads((ARTIFACT_DIR / "thresholds.json").read_text()) \
        if (ARTIFACT_DIR / "thresholds.json").exists() else None
    rev = _git_rev()
    splits = {s: {"rows": int(len(g)), "n_inits": int(g["init_time"].nunique()),
                  "first_init": str(g["init_time"].min()), "last_init": str(g["init_time"].max())}
              for s, g in df.groupby("split")}
    metadata = {
        "experiment_id": f"v3-{'dev' if DEV else 'final'}-{rev}", "commit_hash": rev,
        "created": dt.datetime.now(dt.timezone.utc).isoformat(), "run": RUN or "v1", "dev_split": DEV,
        "model_type": MODEL_TYPE, "estimator": ESTIMATOR, "quantiles": QUANTILES, "quantile_columns": QCOLS,
        "params": params, "fixed_params": FIXED_PARAMS, "seed": seed,
        "params_source": "locked_params (config/model_v3.yaml)" if locked else "dev-split selection",
        "features": feats, "feature_groups": FEATURE_GROUPS, "n_features": len(feats),
        "feature_schema": {c: str(df[c].dtype) for c in feats},
        "target": {"name": "norm_error", "definition": "area-weighted regional Z500 RMSE of the ensemble mean vs "
                   "ERA5, divided by the TRAIN climatological error for the region/lead/season (labels/build.py)"},
        "threshold": {"column": "q_primary", "definition": "TRAIN-derived Q90 of norm_error per region x lead x "
                      "season (unchanged from v1/v2); bust = norm_error > q_primary",
                      "file": "thresholds.json", "summary": thresholds and {k: thresholds[k] for k in thresholds
                                                                             if k != "thresholds"}},
        "exceedance_method": EXCEEDANCE_METHOD, "crossing_correction": CROSSING_FIX,
        "validation_crossing": qgb.val_crossing,
        "calibration": {"method": "IsotonicRegression(out_of_bounds='clip') on VALIDATION estimated exceedance "
                        "probability vs bust label; frozen", "n_rows": int(len(va)),
                        "n_knots": int(len(qgb.calibrator.X_thresholds_))},
        "model_level_importance": {"method": "permutation importance of the q90 estimator (increase in q90 "
                                   "pinball loss), 20k validation rows, 3 repeats", "values": importance},
        "splits": splits, "fit_seconds": fit_s, "software": _versions(),
        "b2_disposition": "ARCHIVAL REFERENCE ONLY - not a QGB input, not a fallback, not the production model",
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(clean_json(metadata), indent=1, default=str))
    manifest = {**metadata, "search": search, "runtime_s": round(time.time() - t0, 1),
                "model_dir": str(MODEL_DIR.relative_to(REPO_ROOT))}
    (ARTIFACT_DIR / "experiment_manifest.json").write_text(json.dumps(clean_json(manifest), indent=1, default=str))
    _append_registry(manifest)
    log.info("qgb training done in %.0fs (final fit %.0fs)", time.time() - t0, fit_s)


def _append_registry(manifest: dict) -> None:
    """Top-level artifacts/experiment_registry.json: one record per training run across v2/v3."""
    records = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else []
    records.append({k: manifest[k] for k in ("experiment_id", "created", "commit_hash", "run", "dev_split",
                                             "model_type", "features", "quantiles", "params", "params_source",
                                             "calibration", "seed")} | {"notes": manifest["b2_disposition"]})
    REGISTRY.write_text(json.dumps(clean_json(records), indent=1, default=str))
