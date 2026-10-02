"""V4 evaluation + the pre-registered development gate of config/model_v4.yaml.

    FBS_RUN=v3 FBS_SPLIT=dev python -m forecast_bust.evaluation.run_v4     # dev gate (validation 2020, dev-test 2021)
    FBS_RUN=v3                python -m forecast_bust.evaluation.run_v4     # single 2022 evaluation after a freeze
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from forecast_bust.config import DEV, clean_json, model_config
from forecast_bust.evaluation import metrics as M
from forecast_bust.pipeline_v4 import ART, CASE_COLS, LABELS, PRED, v4_config
from forecast_bust.pipeline import TABLE

log = logging.getLogger(__name__)


def auc_diff_bootstrap(df: pd.DataFrame, y_a, p_a, y_b, p_b, n: int = 200, seed: int = 0) -> dict:
    """Paired init-day block bootstrap of ROC AUC(y_a, p_a) - ROC AUC(y_b, p_b) (labels may differ)."""
    rng = np.random.default_rng(seed)
    day = pd.to_datetime(df["init_time"]).dt.floor("D").to_numpy()
    uniq, inv = np.unique(day, return_inverse=True)
    rows = [np.where(inv == i)[0] for i in range(len(uniq))]
    d = []
    for _ in range(n):
        idx = np.concatenate([rows[i] for i in rng.integers(0, len(uniq), len(uniq))])
        d.append(roc_auc_score(y_a[idx], p_a[idx]) - roc_auc_score(y_b[idx], p_b[idx]))
    d = np.array(d)
    return {"point": float(roc_auc_score(y_a, p_a) - roc_auc_score(y_b, p_b)), "mean": float(d.mean()),
            "ci95": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))], "n_boot": n}


def _block(part: pd.DataFrame, va: pd.DataFrame, ycol: str, pcol: str, far: float) -> dict:
    y, p = part[ycol].to_numpy().astype(int), part[pcol].to_numpy(float)
    thr = M.threshold_at_far(va[ycol].to_numpy().astype(int), va[pcol].to_numpy(float), far)
    prev = float(y.mean())
    view = part.assign(bust=y)                      # metric helpers read the event from "bust"
    if ycol == "pattern_bust":
        view = view.assign(hidden_bust=(part["hidden_pattern_bust"] == 1).astype(int))
    auprc = M.auprc(y, p)
    return {"prevalence": prev, "auprc": auprc, "lift": auprc / prev, "roc_auc": M.roc_auc(y, p),
            "brier": M.brier(y, p), "brier_skill_vs_validation_climatology":
                1 - M.brier(y, p) / M.brier(y, np.full(len(y), va[ycol].mean())),
            "ece": M.ece(y, p), "recall_at_far_10": M.recall_at_far(y, p, 0.10),
            "operating_point": {"chosen_on": "validation", "target_far": far, **M.confusion_at(y, p, thr)},
            "hidden_bust": M.hidden_bust_metrics(view, p, thr), "warning_lead": M.warning_lead(view, p, thr),
            "peak_risk_day": M.peak_day_error(view.assign(norm_error=part["norm_error"]), p),
            "reliability": M.reliability_curve(y, p)}


def evaluate_v4() -> dict:
    cfg = v4_config()
    gate = cfg["gate"]
    far = model_config()["evaluation"]["alert_far"]
    df = pd.read_parquet(TABLE, columns=CASE_COLS + ["q_primary"], filters=[("split", "in", ["validation", "test"])])
    df = df.merge(pd.read_parquet(LABELS), on="case_id").merge(pd.read_parquet(PRED), on="case_id")
    df["p_B1_spread"] = df["spread_pct"].fillna(df["spread_pct"].median())
    va, te = df[df["split"] == "validation"], df[df["split"] == "test"]
    out = {"label": "dev split: validation 2020, dev-test 2021" if DEV else
           "2022 - V4 final evaluation after freeze; 2022 was previously observed by earlier project generations, "
           "so this is not a pristine project-wide unseen test", "splits": {}}
    for name, part in (("validation", va), ("test", te)):
        y_pat, y_mag = part["pattern_bust"].to_numpy().astype(int), part["bust"].to_numpy().astype(int)
        r = {"pattern_target": {"V4": _block(part, va, "pattern_bust", "calibrated_bust_probability", far),
                                "V4_raw": {k: v for k, v in _block(part, va, "pattern_bust", "raw_probability", far).items()
                                           if k in ("auprc", "roc_auc", "brier", "ece")},
                                "B0_climatology": _block(part, va, "pattern_bust", "p_B0_pattern", far),
                                "B1_spread": _block(part, va, "pattern_bust", "p_B1_spread", far)},
             "magnitude_target": {"same_model_on_magnitude": _block(part, va, "bust", "p_magnitude_reference", far),
                                  "V4_scored_on_magnitude": _block(part, va, "bust", "calibrated_bust_probability", far)}}
        r["adequacy_rocauc_gain"] = auc_diff_bootstrap(part, y_pat, part["calibrated_bust_probability"].to_numpy(),
                                                       y_mag, part["p_magnitude_reference"].to_numpy())
        r["v4_minus_b1_auprc"] = M.block_bootstrap_diff(part.assign(bust=y_pat), part["calibrated_bust_probability"].to_numpy(),
                                                        part["p_B1_spread"].to_numpy())
        thr = r["pattern_target"]["V4"]["operating_point"]["threshold"]
        alert = part["calibrated_bust_probability"].to_numpy() >= thr
        r["pattern_specific"] = {
            "alerts": int(alert.sum()),
            "alert_share_magnitude_busts": float(part.loc[alert, "bust"].mean()) if alert.any() else None,
            "alert_share_pattern_failures": float(part.loc[alert, "pattern_failure"].mean()) if alert.any() else None,
            "median_norm_error_alert_vs_not": [float(part.loc[alert, "norm_error"].median()), float(part.loc[~alert, "norm_error"].median())],
            "median_local_acc_alert_vs_not": [float(part.loc[alert, "local_acc"].median()), float(part.loc[~alert, "local_acc"].median())],
            "rocauc_V4_for_magnitude_failure": M.roc_auc(part["magnitude_failure"].to_numpy(), part["calibrated_bust_probability"].to_numpy()),
            "rocauc_V4_for_pattern_failure": M.roc_auc(part["pattern_failure"].to_numpy().astype(int), part["calibrated_bust_probability"].to_numpy())}
        r["by_lead_day"] = [{"lead_day": int(d), "n": int(len(g)), "positives": int(g["pattern_bust"].sum()),
                             "prevalence": float(g["pattern_bust"].mean()),
                             "auprc": M.auprc(g["pattern_bust"].to_numpy().astype(int), g["calibrated_bust_probability"].to_numpy()),
                             "roc_auc": M.roc_auc(g["pattern_bust"].to_numpy().astype(int), g["calibrated_bust_probability"].to_numpy()),
                             "auprc_B1": M.auprc(g["pattern_bust"].to_numpy().astype(int), g["p_B1_spread"].to_numpy()),
                             "roc_auc_magnitude_model": M.roc_auc(g["bust"].to_numpy(), g["p_magnitude_reference"].to_numpy())}
                            for d, g in part.groupby("lead_day")]
        for x in r["by_lead_day"]:
            x["lift"] = x["auprc"] / x["prevalence"]
        reg = [{"region_id": k, "positives": int(g["pattern_bust"].sum()),
                "roc_auc": M.roc_auc(g["pattern_bust"].to_numpy().astype(int), g["calibrated_bust_probability"].to_numpy())}
               for k, g in part.groupby("region_id")]
        ok = [x for x in reg if x["positives"] >= 20]
        r["by_region"] = {"n_regions": len(reg), "n_regions_ge_20_positives": len(ok),
                          "share_auc_gt_0_5": float(np.mean([x["roc_auc"] > 0.5 for x in ok])) if ok else None,
                          "auc_median": float(np.median([x["roc_auc"] for x in ok])) if ok else None,
                          "auc_p10_p90": [float(np.quantile([x["roc_auc"] for x in ok], q)) for q in (0.1, 0.9)] if ok else None}
        out["splits"][name] = r

    if DEV:
        audit = json.loads((ART / "target_audit.json").read_text())
        t, v = out["splits"]["test"], out["splits"]["validation"]
        a = t["adequacy_rocauc_gain"]
        checks = {
            "acc_coverage": min(audit["acc_coverage"].values()) >= gate["min_acc_coverage"],
            "positive_rows": min(audit["splits"][s]["positive_rows"] for s in ("validation", "test")) >= gate["min_positive_rows"],
            "positive_inits": min(audit["splits"][s]["positive_inits"] for s in ("validation", "test")) >= gate["min_positive_inits"],
            "positives_per_lead": min(audit["splits"]["test"]["positives_by_lead"].values()) >= gate["min_positive_rows_per_lead"],
            "rocauc_gain_vs_magnitude": a["point"] >= gate["min_rocauc_gain_vs_magnitude"],
            "rocauc_gain_ci_lower_above_0": a["ci95"][0] > gate["rocauc_gain_ci_lower_gt"],
            "validation_rocauc_gain_positive": v["adequacy_rocauc_gain"]["point"] > gate["validation_rocauc_gain_gt"],
            "beats_spread_score": t["v4_minus_b1_auprc"]["ci95"][0] > gate["beat_spread_score_ci_lower_gt"],
            "positive_brier_skill": t["pattern_target"]["V4"]["brier_skill_vs_validation_climatology"] > gate["min_brier_skill"],
            "lift_every_lead": min(x["lift"] for x in t["by_lead_day"]) >= gate["min_lift_every_lead"],
            "regional_auc": (t["by_region"]["share_auc_gt_0_5"] or 0) >= gate["min_share_regions_auc_gt_0_5"],
        }
        out["gate"] = {"preregistered": gate, "checks": checks, "decision": "PROMOTE" if all(checks.values()) else "STOP"}
        log.info("V4 gate: %s %s", out["gate"]["decision"], checks)
    (ART / "metrics.json").write_text(json.dumps(clean_json(out), indent=1, default=float))
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    o = evaluate_v4()
    for s, r in o["splits"].items():
        p, m = r["pattern_target"], r["magnitude_target"]["same_model_on_magnitude"]
        print(s, {k: (round(p[k]["auprc"], 4), round(p[k]["roc_auc"], 4)) for k in ("V4", "B0_climatology", "B1_spread")},
              "| magnitude model AUPRC/AUC", round(m["auprc"], 4), round(m["roc_auc"], 4),
              "| AUC gain", round(r["adequacy_rocauc_gain"]["point"], 4), [round(c, 4) for c in r["adequacy_rocauc_gain"]["ci95"]])
    print(o.get("gate"))
