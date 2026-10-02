"""BMA experiment evaluation against climatology (B0), B2, V3 on IDENTICAL rows, and the pre-registered
go/no-go gate of config/model_bma.yaml. Writes artifacts/bma[/dev]/metrics.json.

    FBS_RUN=v3 FBS_SPLIT=dev python -m forecast_bust.evaluation.run_bma
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from forecast_bust.config import DEV, clean_json, model_config
from forecast_bust.evaluation import metrics as M
from forecast_bust.models.bma import QCOLS, QUANTILES
from forecast_bust.pipeline import PRED as V3_PRED, TABLE
from forecast_bust.pipeline_bma import ART, PRED as BMA_PRED, bma_config

log = logging.getLogger(__name__)
COLS = ["case_id", "init_time", "region_id", "lead_day", "split", "norm_error", "bust", "hidden_bust", "low_spread"]
V3_COLS = ["case_id", "p_B0", "p_B2_dev_reference", "calibrated_bust_probability", *QCOLS]


def _block(df: pd.DataFrame, va: pd.DataFrame, col: str, far: float) -> dict:
    y, p = df["bust"].to_numpy(), df[col].to_numpy(float)
    thr = M.threshold_at_far(va["bust"].to_numpy(), va[col].to_numpy(float), far)
    b_clim = M.brier(y, np.full(len(y), va["bust"].mean()))
    return {"auprc": M.auprc(y, p), "roc_auc": M.roc_auc(y, p), "brier": M.brier(y, p),
            "brier_skill_vs_validation_climatology": 1 - M.brier(y, p) / b_clim, "ece": M.ece(y, p),
            "recall_at_far_10": M.recall_at_far(y, p, 0.10),
            "hidden_bust_recall_at_validation_10pct_far": M.hidden_bust_metrics(df, p, thr)["hidden_bust_recall"],
            "reliability": M.reliability_curve(y, p)}


def _dist(df: pd.DataFrame, prefix: str, central: str) -> dict:
    y = df["norm_error"].to_numpy()
    return {"mae_central": M.mae(y, df[central].to_numpy()), "rmse_central": M.rmse(y, df[central].to_numpy()),
            "coverage": {c: float(np.mean(y <= df[prefix + c])) for c in QCOLS},
            "pinball": {c: M.pinball_loss(y, df[prefix + c].to_numpy(), q) for c, q in zip(QCOLS, QUANTILES)}}


def evaluate_bma() -> dict:
    far = model_config()["evaluation"]["alert_far"]
    gate = bma_config()["gate"]
    df = pd.read_parquet(TABLE, columns=COLS, filters=[("split", "in", ["validation", "test"])])
    v3 = pd.read_parquet(V3_PRED, columns=V3_COLS).rename(columns={"calibrated_bust_probability": "p_V3",
                                                                   **{c: "v3_" + c for c in QCOLS}})
    bma = pd.read_parquet(BMA_PRED).rename(columns={"bma_exceedance_probability": "p_BMA_raw",
                                                    "calibrated_bust_probability": "p_BMA_cal",
                                                    **{c: "bma_" + c for c in QCOLS}})
    df = df.merge(v3, on="case_id").merge(bma, on="case_id")
    missing = {s: float(g["p_BMA_raw"].isna().mean()) for s, g in df.groupby("split")}
    df = df[df["p_BMA_raw"].notna()].reset_index(drop=True)
    va, te = df[df["split"] == "validation"], df[df["split"] == "test"]
    if "p_B2_dev_reference" not in df or df["p_B2_dev_reference"].isna().all():
        raise RuntimeError("B2 dev reference predictions missing (run the v3 dev pipeline)")
    models = {"B0_climatology": "p_B0", "B2": "p_B2_dev_reference", "V3_QGB": "p_V3",
              "BMA_raw": "p_BMA_raw", "BMA_calibrated": "p_BMA_cal"}
    out = {"label": "dev split: validation 2020, dev-test 2021" if DEV else "final", "rows_without_bma": missing,
           "n_rows": {"validation": int(len(va)), "test": int(len(te))}, "splits": {}}
    for name, part in (("validation", va), ("test", te)):
        res = {m: _block(part, va, c, far) for m, c in models.items()}
        res["distribution"] = {"BMA": {**_dist(part, "bma_", "bma_q50"),
                                       "mae_mean": M.mae(part["norm_error"].to_numpy(), part["bma_mean"].to_numpy())},
                               "V3_QGB": _dist(part, "v3_", "v3_q50")}
        res["bootstrap_auprc_diff"] = {
            "BMA_raw_minus_B2": M.block_bootstrap_diff(part, part["p_BMA_raw"].to_numpy(), part["p_B2_dev_reference"].to_numpy()),
            "BMA_raw_minus_V3": M.block_bootstrap_diff(part, part["p_BMA_raw"].to_numpy(), part["p_V3"].to_numpy())}
        res["by_lead_day"] = [{"lead_day": int(d), "n": int(len(g)), "bust_rate": float(g["bust"].mean()),
                               **{m: M.auprc(g["bust"].to_numpy(), g[c].to_numpy(float)) for m, c in models.items()}}
                              for d, g in part.groupby("lead_day")]
        reg = pd.DataFrame([{"region_id": r, **{m: M.auprc(g["bust"].to_numpy(), g[c].to_numpy(float))
                                                 for m, c in models.items()}} for r, g in part.groupby("region_id")])
        res["by_region"] = {"n_regions": int(len(reg)), "share_regions_BMA_ge_B2": float((reg["BMA_raw"] >= reg["B2"]).mean()),
                            "median_auprc": {m: float(reg[m].median()) for m in models}}
        out["splits"][name] = res

    t, v = out["splits"]["test"], out["splits"]["validation"]
    bb = t["bootstrap_auprc_diff"]["BMA_raw_minus_B2"]
    checks = {
        "devtest_auprc_gain_vs_B2": t["BMA_raw"]["auprc"] - t["B2"]["auprc"] >= gate["devtest_min_auprc_gain"],
        "devtest_ci_lower_above_0": bb["ci95"][0] > gate["devtest_ci_lower_gt"],
        "validation_auprc_gain_vs_B2": v["BMA_raw"]["auprc"] - v["B2"]["auprc"] >= gate["validation_min_auprc_gain"],
        "brier_not_worse_than_B2": t["BMA_calibrated"]["brier"] <= t["B2"]["brier"] + gate["max_brier_excess_vs_b2"],
        "ece_ok": t["BMA_calibrated"]["ece"] <= gate["max_ece"],
        "lead_days_beating_B2": sum(r["BMA_raw"] >= r["B2"] for r in t["by_lead_day"]) >= gate["min_lead_days_beating_b2"],
        "coverage_of_rows": missing.get("test", 1.0) <= gate["max_share_rows_without_prediction"],
    }
    out["gate"] = {"preregistered": gate, "checks": checks, "decision": "GO" if all(checks.values()) else "NO-GO",
                   "lead_days_BMA_ge_B2": int(sum(r["BMA_raw"] >= r["B2"] for r in t["by_lead_day"]))}
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "metrics.json").write_text(json.dumps(clean_json(out), indent=1, default=float))
    log.info("BMA gate: %s %s", out["gate"]["decision"], checks)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    o = evaluate_bma()
    for s in ("validation", "test"):
        r = o["splits"][s]
        print(s, {m: round(r[m]["auprc"], 4) for m in ("B0_climatology", "B2", "V3_QGB", "BMA_raw", "BMA_calibrated")})
    print(o["gate"]["decision"], o["gate"]["checks"])
