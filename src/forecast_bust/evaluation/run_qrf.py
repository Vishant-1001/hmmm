"""Single evaluation of the frozen v3 QRF model on whichever split `df["split"]=="test"`
currently means (dev-test 2021 under FBS_SPLIT=dev - the go/no-go gate; 2022 otherwise - the
final holdout). Writes artifacts/<run>/metrics_qrf.json.

Labeling discipline (spec §18): 2022 was already read twice by the retired v1/v2 GBT models.
It has never been read by QRF before this function's first non-dev run. The result is
reported as "first QRF evaluation on the 2022 period", never as "untouched"/"pristine".
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from forecast_bust.config import DEV, RUN, clean_json, ARTIFACT_DIR, REPO_ROOT, model_config
from forecast_bust.evaluation import metrics as M
from forecast_bust.models.qrf import QUANTILES
from forecast_bust.pipeline_qrf import PRED, TABLE

log = logging.getLogger(__name__)
LEVELS = {f"q{int(round(ql * 100))}": ql for ql in QUANTILES}


def _reconcile_v2_b2_reference() -> dict | None:
    """Read-only pull of the already-frozen v2 test (2022) B2 numbers for the historical
    comparison table. Never recomputed, never refit - just reconciled and quoted (spec §20/§21)."""
    p = REPO_ROOT / "artifacts" / "v2" / "metrics.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text())
    b2 = m["models"]["B2"]
    return {"source": str(p.relative_to(REPO_ROOT)), "commit_lineage": "v2 locked at 8b196ee, "
            "scored once at a30bff2 (see AGENT_STATE.md)", "split": "2022 (v2's second look)",
            "auprc": b2["auprc"], "brier": b2["brier"], "ece": b2["ece"],
            "hidden_bust_recall": b2["hidden_bust"]["hidden_bust_recall"],
            "recall_at_far_10": b2["recall_at_far_10"],
            "note": "V2 HISTORICAL B2 RESULT - not recomputed here; distinct from the V3 QRF "
                    "result below. Do not read a QRF-vs-this-number difference as a controlled "
                    "comparison (different model generation, different commit, different "
                    "number of prior test-set exposures)."}


def evaluate_qrf() -> dict:
    cols = ["case_id", "init_time", "valid_time", "region_id", "row", "col", "lat", "lon", "lead_day",
            "season", "split", "error_m", "norm_error", "q_primary", "q_sensitivity", "bust", "bust_q95",
            "hidden_bust", "low_spread", "spread_m", "spread_pct", "support_level", "support_distance",
            "an_n_within", "an_bust_rate", "an_n_eligible"]
    df = pd.read_parquet(TABLE, columns=cols)
    pr = pd.read_parquet(PRED)
    df = df.merge(pr, on="case_id", how="inner")
    va = df[df["split"] == "validation"].reset_index(drop=True)
    te = df[df["split"] == "test"].reset_index(drop=True)
    if te.empty:
        raise RuntimeError("no test rows")
    far = model_config()["evaluation"]["alert_far"]
    label = "dev-test (2021)" if DEV else "2022 - first QRF evaluation on this period (previously " \
            "read twice by the retired v1/v2 GBT models; not an untouched/pristine test)"

    p = te["calibrated_bust_probability"].to_numpy()
    y = te["bust"].to_numpy()
    thr = M.threshold_at_far(va["bust"].to_numpy(), va["calibrated_bust_probability"].to_numpy(), far)
    qrf_result = {
        "description": "QRF (v3 production model): conditional norm_error quantiles -> estimated "
                       "exceedance probability at the training-only Q90 threshold -> validation-only "
                       "isotonic calibration", "label": label,
        "mae_q50": M.mae(te["norm_error"].to_numpy(), te["q50"].to_numpy()),
        "rmse_q50": M.rmse(te["norm_error"].to_numpy(), te["q50"].to_numpy()),
        "quantile_coverage": M.quantile_coverage(te["norm_error"].to_numpy(),
                                                 {k: te[k].to_numpy() for k in LEVELS}, LEVELS),
        "auprc": M.auprc(y, p), "roc_auc": M.roc_auc(y, p), "brier": M.brier(y, p), "ece": M.ece(y, p),
        **{f"recall_at_far_{int(f * 100)}": M.recall_at_far(y, p, f) for f in model_config()["evaluation"]["fixed_far"]},
        "operating_point": {"chosen_on": "validation", "target_far": far, **M.confusion_at(y, p, thr)},
        "hidden_bust": M.hidden_bust_metrics(te, p, thr),
        "warning_lead": M.warning_lead(te, p, thr),
        "peak_risk_day": M.peak_day_error(te, p),
        "spatial_overlap": M.spatial_overlap(te, p, thr),
    }
    b0_result = {"auprc": M.auprc(y, te["p_B0"].to_numpy()), "brier": M.brier(y, te["p_B0"].to_numpy())}
    dev_b2 = None
    if "p_B2_dev_reference" in te.columns:
        pb2 = te["p_B2_dev_reference"].to_numpy()
        dev_b2 = {"note": "fresh B2 refit on THIS dev split only - not the frozen v2/2022 number",
                  "auprc": M.auprc(y, pb2), "brier": M.brier(y, pb2)}

    by_lead = [{"lead_day": int(d), "n": int(len(g)), "bust_rate": float(g["bust"].mean()),
               "mae_q50": M.mae(g["norm_error"].to_numpy(), g["q50"].to_numpy()),
               "auprc": M.auprc(g["bust"].to_numpy(), g["calibrated_bust_probability"].to_numpy())}
              for d, g in te.groupby("lead_day")]
    by_region = [{"region_id": rid, "n": int(len(g)),
                 "auprc": M.auprc(g["bust"].to_numpy(), g["calibrated_bust_probability"].to_numpy())}
                for rid, g in te.groupby("region_id")]

    out = {
        "run": RUN or "v1", "dev_split": DEV, "label": label, "n_test_rows": int(len(te)),
        "test_base_rate": float(y.mean()), "primary_metric": "AUPRC of calibrated_bust_probability "
        "(product-facing); MAE/RMSE/pinball/coverage are mandatory distributional guardrails, not "
        "substitutes (spec §11)",
        "qrf": qrf_result, "b0_reference": b0_result, "b2_dev_reference": dev_b2,
        "b2_v2_historical_reference": _reconcile_v2_b2_reference() if not DEV else None,
        "by_lead_day": by_lead, "by_region": by_region,
    }
    out_path = ARTIFACT_DIR / "metrics_qrf.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(clean_json(out), indent=1, default=float))
    log.info("qrf eval [%s]: AUPRC %.4f, MAE(q50) %.4f, test base rate %.3f", label, qrf_result["auprc"],
             qrf_result["mae_q50"], out["test_base_rate"])
    return out
