"""Single evaluation of the v3 quantile-gradient-boosting model on whichever split
`df["split"]=="test"` currently means: dev-test 2021 under FBS_SPLIT=dev (the go/no-go gate),
2022 otherwise (run once, after the freeze). Writes <ARTIFACT_DIR>/metrics.json.

Labeling discipline: 2022 was already read by the retired v1/v2 GBT models. The final run is
"V3 first evaluation on the 2022 period", never "untouched"/"pristine".
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from forecast_bust.config import DEV, RUN, clean_json, ARTIFACT_DIR, MODEL_DIR, REPO_ROOT, model_config
from forecast_bust.evaluation import metrics as M
from forecast_bust.models.qgb import QCOLS, QUANTILES
from forecast_bust.pipeline_qgb import PRED, TABLE

log = logging.getLogger(__name__)
LEVELS = dict(zip(QCOLS, QUANTILES))
EVAL_COLS = ["case_id", "init_time", "region_id", "lead_day", "season", "split", "norm_error", "q_primary",
             "bust", "bust_q95", "hidden_bust", "low_spread", "spread_pct", "row", "col", "lat", "lon",
             "support_level"]


def _v2_historical_reference() -> dict | None:
    """Read-only quote of the frozen v2 2022 numbers (never recomputed)."""
    p = REPO_ROOT / "artifacts" / "v2" / "metrics.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text())["models"]
    return {"source": "artifacts/v2/metrics.json (v2 locked 8b196ee; 2022 read for the second time)",
            "B2": {"auprc": m["B2"]["auprc"], "brier": m["B2"]["brier"],
                   "hidden_bust_recall": m["B2"]["hidden_bust"]["hidden_bust_recall"]},
            "note": "V2 HISTORICAL RESULT - different model generation, not a controlled comparison."}


def _prob_block(te: pd.DataFrame, va: pd.DataFrame, col: str, far: float) -> dict:
    y, p = te["bust"].to_numpy(), te[col].to_numpy()
    thr = M.threshold_at_far(va["bust"].to_numpy(), va[col].to_numpy(), far)
    return {"auprc": M.auprc(y, p), "roc_auc": M.roc_auc(y, p), "brier": M.brier(y, p), "ece": M.ece(y, p),
            "brier_climatology": M.brier(y, np.full(len(y), te["bust"].mean())),
            **{f"recall_at_far_{int(f * 100)}": M.recall_at_far(y, p, f)
               for f in model_config()["evaluation"]["fixed_far"]},
            "operating_point": {"chosen_on": "validation", "target_far": far, "threshold": thr,
                                **M.confusion_at(y, p, thr)},
            "hidden_bust": M.hidden_bust_metrics(te, p, thr), "warning_lead": M.warning_lead(te, p, thr),
            "reliability": M.reliability_curve(y, p)}


def evaluate_qgb() -> dict:
    df = pd.read_parquet(TABLE, columns=EVAL_COLS, filters=[("split", "in", ["validation", "test"])])
    df = df.merge(pd.read_parquet(PRED), on="case_id", how="inner")
    va = df[df["split"] == "validation"].reset_index(drop=True)
    te = df[df["split"] == "test"].reset_index(drop=True)
    if te.empty:
        raise RuntimeError("no test rows")
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    far = model_config()["evaluation"]["alert_far"]
    label = "dev-test (2021)" if DEV else ("2022 - V3 first evaluation on this period (previously read by the "
                                           "retired v1/v2 GBT models; not an untouched/pristine test)")
    y_err = te["norm_error"].to_numpy()
    y = te["bust"].to_numpy()
    p = te["calibrated_bust_probability"].to_numpy()
    thr = M.threshold_at_far(va["bust"].to_numpy(), va["calibrated_bust_probability"].to_numpy(), far)

    distribution = {
        "mae_q50": M.mae(y_err, te["q50"].to_numpy()), "rmse_q50": M.rmse(y_err, te["q50"].to_numpy()),
        "pinball": {c: M.pinball_loss(y_err, te[c].to_numpy(), lv) for c, lv in LEVELS.items()},
        "coverage": {c: float(np.mean(y_err <= te[c].to_numpy())) for c in QCOLS},
        "central_range_q25_q75_coverage": float(np.mean((y_err >= te["q25"]) & (y_err <= te["q75"]))),
        "coverage_by_lead_day": {int(d): {c: float(np.mean(g["norm_error"] <= g[c])) for c in QCOLS}
                                 for d, g in te.groupby("lead_day")},
        "validation_quantile_crossing": meta.get("validation_crossing"),
    }
    distribution["pinball_mean"] = float(np.mean(list(distribution["pinball"].values())))

    qgb = {"description": "v3 production model: HistGradientBoostingRegressor(loss='quantile') q10..q95 of "
                          "norm_error -> estimated exceedance probability at the TRAIN-only Q90 threshold -> "
                          "validation-only isotonic calibration", "label": label, "params": meta["params"],
           "distribution": distribution,
           "calibrated_bust_probability": _prob_block(te, va, "calibrated_bust_probability", far),
           "estimated_exceedance_probability_uncalibrated": {
               k: v for k, v in _prob_block(te, va, "estimated_exceedance_probability", far).items()
               if k in ("auprc", "roc_auc", "brier", "ece", "reliability")},
           "q95_sensitivity": {"note": "calibrated bust probability (Q90 target) scored against the Q95 bust "
                               "label; ranking check only", "auprc": M.auprc(te["bust_q95"].to_numpy(), p)},
           "peak_risk_day": M.peak_day_error(te, p), "spatial_overlap": M.spatial_overlap(te, p, thr)}

    refs = {"B0_climatology": {"auprc": M.auprc(y, te["p_B0"].to_numpy()), "brier": M.brier(y, te["p_B0"].to_numpy())}}
    if "p_B2_dev_reference" in te.columns:
        pb2 = te["p_B2_dev_reference"].to_numpy()
        hb = M.hidden_bust_metrics(te, pb2, M.threshold_at_far(va["bust"].to_numpy(),
                                                               va["p_B2_dev_reference"].to_numpy(), far))
        refs["B2_dev_reference"] = {"note": "fresh B2 (spread-only XGBoost, frozen v2 hyper-parameters) on THIS "
                                    "dev split only - archival reference, not part of v3 inference",
                                    "auprc": M.auprc(y, pb2), "brier": M.brier(y, pb2),
                                    "hidden_bust_recall": hb["hidden_bust_recall"]}
        refs["QGB_minus_B2_block_bootstrap"] = M.block_bootstrap_diff(te, p, pb2)
    if not DEV:
        refs["v2_historical"] = _v2_historical_reference()

    by_lead = [{"lead_day": int(d), "n": int(len(g)), "bust_rate": float(g["bust"].mean()),
                "mae_q50": M.mae(g["norm_error"].to_numpy(), g["q50"].to_numpy()),
                "pinball_q90": M.pinball_loss(g["norm_error"].to_numpy(), g["q90"].to_numpy(), 0.9),
                "auprc": M.auprc(g["bust"].to_numpy(), g["calibrated_bust_probability"].to_numpy()),
                "auprc_B0": M.auprc(g["bust"].to_numpy(), g["p_B0"].to_numpy()),
                "brier": M.brier(g["bust"].to_numpy(), g["calibrated_bust_probability"].to_numpy())}
               for d, g in te.groupby("lead_day")]
    by_region = [{"region_id": rid, "n": int(len(g)), "bust_rate": float(g["bust"].mean()),
                  "auprc": M.auprc(g["bust"].to_numpy(), g["calibrated_bust_probability"].to_numpy()),
                  "mae_q50": M.mae(g["norm_error"].to_numpy(), g["q50"].to_numpy())}
                 for rid, g in te.groupby("region_id")]
    reg_auprc = np.array([r["auprc"] for r in by_region], dtype=float)

    out = {"run": RUN or "v1", "dev_split": DEV, "label": label, "model_type": meta["model_type"],
           "experiment_id": meta["experiment_id"], "commit_hash": meta["commit_hash"],
           "rows": {"train": meta["splits"]["train"]["rows"], "validation": int(len(va)), "test": int(len(te))},
           "test_base_rate": float(y.mean()), "quantiles": QUANTILES,
           "primary_metric": "AUPRC of calibrated_bust_probability; MAE/RMSE/pinball/coverage/Brier/calibration "
                             "are reported alongside as guardrails",
           "qgb": qgb, "references": refs, "by_lead_day": by_lead, "by_region": by_region,
           "regional_stability": {"n_regions": int(len(reg_auprc)), "auprc_median": float(np.nanmedian(reg_auprc)),
                                  "auprc_p10": float(np.nanquantile(reg_auprc, 0.1)),
                                  "auprc_p90": float(np.nanquantile(reg_auprc, 0.9))}}
    out_path = ARTIFACT_DIR / "metrics.json"
    out_path.write_text(json.dumps(clean_json(out), indent=1, default=float))
    log.info("qgb eval [%s]: AUPRC %.4f (B0 %.4f), Brier %.4f, MAE(q50) %.4f, base rate %.3f", label,
             qgb["calibrated_bust_probability"]["auprc"], refs["B0_climatology"]["auprc"],
             qgb["calibrated_bust_probability"]["brier"], distribution["mae_q50"], out["test_base_rate"])
    return out
