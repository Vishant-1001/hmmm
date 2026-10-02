"""v3 training pipeline: QRF production model + archival dev-split B0/B2 reference comparison.

Reuses `pipeline.build_table()` unchanged (data assembly, labels, Q90/Q95 thresholds, features,
historical memory, support/OOD - none of that changes in v3). Only the model-fitting stage
changes: the retired B2/Sentinel XGBoost classifiers (`models/sentinel.py`, historical only) are
replaced by the QRF (`models/qrf.py`) as the production model. B2 is still fit here, on the same
rows, purely as an archival reference number - never as a QRF input, never promoted to
production (see `models/sentinel.py` module docstring and the frozen v3 spec).

    FBS_RUN=v3 FBS_SPLIT=dev python -m forecast_bust.train_qrf    # dev-split search + lock check
    FBS_RUN=v3                python -m forecast_bust.train_qrf    # final fit with locked params

(`train_qrf.py` is the thin CLI entry point, mirroring `train.py` / `pipeline.py`.)
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import time

import joblib
import numpy as np
import pandas as pd

from forecast_bust.config import DEV, RUN, clean_json, ARTIFACT_DIR, MODEL_DIR, REPO_ROOT, model_config
from forecast_bust.evaluation.metrics import auprc, mae, pinball_loss, rmse
from forecast_bust.models.qrf import QUANTILES, SEARCH_GRID_LEVELS, QRFErrorModel
from forecast_bust.models.sentinel import B0Climatology, CalibratedGBM, FEATURE_SETS
from forecast_bust.pipeline import PRED, TABLE, _git_rev, _versions, build_table

log = logging.getLogger(__name__)
QRF_FEATURES_GROUP = "ALL"  # SPREAD + every validated candidate group (features/build.py GROUPS)
REGISTRY = REPO_ROOT / "artifacts" / "experiment_registry.json"  # accumulates across runs, top level


def qrf_features() -> list[str]:
    return FEATURE_SETS[QRF_FEATURES_GROUP]


def _representative_configs(grid: dict) -> list[dict]:
    """3 representative configurations spanning the existing grid (conservative -> middle ->
    high-capacity), picked by inspection instead of the full 8-combination product. Values come
    directly from the existing grid in config/model_v3.yaml - nothing invented. n_estimators and
    max_depth increase capacity when larger; min_samples_leaf increases capacity when SMALLER."""
    n_est, depth, leaf = grid["n_estimators"], grid["max_depth"], grid["min_samples_leaf"]
    return [
        {"tag": "conservative", "n_estimators": n_est[0], "max_depth": depth[0], "min_samples_leaf": leaf[-1]},
        {"tag": "middle", "n_estimators": n_est[-1], "max_depth": depth[-1], "min_samples_leaf": leaf[-1]},
        {"tag": "high_capacity", "n_estimators": n_est[-1], "max_depth": depth[-1], "min_samples_leaf": leaf[0]},
    ]


def search_qrf(tr: pd.DataFrame, va: pd.DataFrame, grid: dict, seed: int) -> dict:
    """Reduced development-split-only search: 3 representative configurations (not the full
    8-combination product), and the coarse SEARCH_GRID_LEVELS (31 points over the same
    0.005..0.995 range as the final 199-point grid) for the exceedance-probability estimate
    used only to calibrate/score each candidate during search. Selection metric is unchanged:
    validation AUPRC of the calibrated bust probability, with MAE/RMSE/pinball loss recorded
    alongside as distributional guardrails. The winning configuration is refit with the
    default fine grid in `train_qrf` before anything is frozen or finally evaluated."""
    feats = qrf_features()
    configs = _representative_configs(grid)
    log.info("qrf search: %d representative configs (grid reduced 8->3, quantile grid reduced "
             "199->%d points, search-time only): %s", len(configs), len(SEARCH_GRID_LEVELS), configs)
    runs = []
    for cfg in configs:
        params = {k: v for k, v in cfg.items() if k != "tag"}
        t0 = time.time()
        m = QRFErrorModel(feats, params=params, seed=seed, grid_levels=SEARCH_GRID_LEVELS).fit(tr, va)
        p = m.calibrated_bust_probability(va)
        q = m.predict_quantiles(va)
        runtime_s = round(time.time() - t0, 1)
        runs.append({
            "tag": cfg["tag"], "params": params, "val_auprc": auprc(va["bust"].to_numpy(), p),
            "val_mae": mae(va["norm_error"].to_numpy(), q["q50"]),
            "val_rmse": rmse(va["norm_error"].to_numpy(), q["q50"]),
            "val_pinball_q90": pinball_loss(va["norm_error"].to_numpy(), q["q90"], 0.90),
            "fit_seconds": runtime_s,
        })
        log.info("qrf search [%s] %s -> val AUPRC %.4f (mae %.4f, rmse %.4f, %.1fs)",
                 cfg["tag"], params, runs[-1]["val_auprc"], runs[-1]["val_mae"], runs[-1]["val_rmse"], runtime_s)
    best = max(runs, key=lambda r: r["val_auprc"])
    return {"grid": grid, "search_quantile_grid_points": len(SEARCH_GRID_LEVELS),
           "configs_tested": len(configs), "runs": runs, "best_params": best["params"],
           "best_tag": best["tag"], "best_val_auprc": best["val_auprc"]}


def train_qrf(reassemble: bool = True, reuse_table: bool = False) -> None:
    """`reuse_table=True` skips data/label/feature/memory/support rebuild and reads the
    existing TABLE parquet directly (dev-iteration convenience only, e.g. re-running the QRF
    search after an unrelated code change; the table's own contents are unaffected by `reassemble`
    either way - this just avoids recomputing it)."""
    t0 = time.time()
    df = pd.read_parquet(TABLE) if reuse_table and TABLE.exists() else build_table(reassemble)
    tr, va = df[df["split"] == "train"], df[df["split"] == "validation"]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    qcfg = model_config()["qrf"]
    seed = model_config()["seed"]
    locked = qcfg.get("locked_params") or {}
    search_result = None
    if locked:
        params = locked
        log.info("qrf: using locked params %s", params)
    else:
        search_result = search_qrf(tr, va, qcfg["grid"], seed)
        params = search_result["best_params"]
        (ARTIFACT_DIR / "qrf_search.json").parent.mkdir(parents=True, exist_ok=True)
        (ARTIFACT_DIR / "qrf_search.json").write_text(json.dumps(clean_json(search_result), indent=1))
        log.info("qrf: no locked_params in config; searched and selected %s (val AUPRC %.4f)",
                 params, search_result["best_val_auprc"])

    feats = qrf_features()
    qrf = QRFErrorModel(feats, params=params, seed=seed).fit(tr, va)

    # Archival-only reference (never a QRF input, never production): B0 climatology always;
    # a FRESH B2 refit only on the dev split (spec §20/§21: do not manufacture a fresh B2
    # experiment against the final/2022 split merely for a convenient comparison - that
    # comparison instead reuses the already-frozen v2 `artifacts/v2/metrics.json` numbers,
    # reconciled at evaluation time in evaluation/run_qrf.py).
    b0 = B0Climatology().fit(tr)
    models = {"QRF": qrf, "B0": b0}
    b2 = None
    if DEV:
        b2_params = model_config().get("v2", {}).get("b2_xgboost")
        b2 = CalibratedGBM(FEATURE_SETS["B2"], "B2_dev_reference", params=b2_params, seed=seed).fit(tr, va)
        models["B2_dev_reference"] = b2
    joblib.dump(models, MODEL_DIR / "qrf_models.joblib")

    rows = df[df["split"].isin(["validation", "test"])].copy()
    preds = pd.DataFrame({"case_id": rows["case_id"].values})
    q = qrf.predict_quantiles(rows)
    for c, v in q.items():
        preds[c] = v
    preds["estimated_exceedance_probability"] = qrf.estimated_exceedance_probability(rows)
    preds["calibrated_bust_probability"] = qrf.calibrated_bust_probability(rows)
    preds["p_B0"] = b0.predict(rows)
    if b2 is not None:
        preds["p_B2_dev_reference"] = b2.predict(rows)
    preds.to_parquet(PRED, index=False)

    manifest = {
        "created": dt.datetime.now(dt.timezone.utc).isoformat(), "git_rev": _git_rev(),
        "run": RUN or "v1", "dev_split": DEV, "seed": seed,
        "model_type": "quantile_regression_forest", "quantiles": QUANTILES,
        "features": feats, "n_features": len(feats),
        "params": params, "params_source": "locked_params (config)" if locked else "dev-split grid search",
        "search": search_result,
        "b2_disposition": "ARCHIVAL REFERENCE ONLY - not a QRF input, not the production model",
        "exceedance_method": "interpolation over a 199-point quantile grid (0.005..0.995); "
                             "an ESTIMATE, not an exact CDF evaluation",
        "calibration": "isotonic regression fitted on VALIDATION predictions only, then frozen",
        "splits": {s: {"n_inits": int(g["init_time"].nunique()), "rows": int(len(g))}
                  for s, g in df.groupby("split")},
        "software": _versions(),
        "runtime_s": round(time.time() - t0, 1),
    }
    (ARTIFACT_DIR / "experiment_manifest_qrf.json").write_text(json.dumps(clean_json(manifest), indent=1, default=str))
    _append_registry(manifest)
    log.info("qrf training done in %.0fs", time.time() - t0)


def _append_registry(manifest: dict) -> None:
    """Top-level artifacts/experiment_registry.json (not RUN-namespaced): accumulates one
    record per training run across v2/v3 so experiments stay reconcilable (spec §30)."""
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    records = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else []
    records.append({
        "experiment_id": f"{manifest['run']}-{'dev' if manifest['dev_split'] else 'final'}-{manifest['git_rev']}",
        "created": manifest["created"], "commit_hash": manifest["git_rev"], "run": manifest["run"],
        "dev_split": manifest["dev_split"], "model_type": manifest["model_type"],
        "features": manifest["features"], "quantiles": manifest["quantiles"], "params": manifest["params"],
        "params_source": manifest["params_source"], "calibration": manifest["calibration"],
        "seed": manifest["seed"], "notes": manifest["b2_disposition"],
    })
    REGISTRY.write_text(json.dumps(clean_json(records), indent=1, default=str))


