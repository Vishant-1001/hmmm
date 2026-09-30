"""Independent reproduction + failure diagnosis of the frozen v1 evaluation.

    .venv/bin/python scripts/audit_v1.py            -> artifacts/diagnosis/v1_audit.json

Part 1 (REPRODUCTION, uses test only to re-derive already-published numbers, decides nothing):
  predictions are regenerated from models/models.joblib on data/interim/table.parquet and scored with
  scikit-learn directly, then compared with data/interim/predictions.parquet and artifacts/metrics.json.
Part 2 (INTEGRITY): target = norm_error > TRAIN Q90 per (region, lead, season); valid = init + lead;
  split boundaries/embargo; thresholds fitted on TRAIN rows only; calibration fitted on validation.
Part 3 (DIAGNOSIS, TRAIN + VALIDATION ONLY): why the spread-driven score cannot flag low-spread
  ("hidden") busts, and whether ANY available forecast-time feature separates busts WITHIN the
  low-spread regime on validation. Test rows are never used in Part 3.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from forecast_bust.config import REPO_ROOT, clean_json, data_config, model_config  # noqa: E402
from forecast_bust.models.sentinel import FEATURE_SETS  # noqa: E402

OUT = REPO_ROOT / "artifacts" / "diagnosis" / "v1_audit.json"


def far_threshold(y, p, far):
    """Smallest threshold whose false-alarm rate on (y, p) is <= far (same rule as evaluation)."""
    neg = np.sort(p[y == 0])[::-1]
    k = int(np.floor(far * len(neg)))
    return float(neg[k]) if k < len(neg) else float(neg[-1])


def main() -> None:
    table = pd.read_parquet(REPO_ROOT / "data/interim/table.parquet")
    models = joblib.load(REPO_ROOT / "models/models.joblib")
    published = json.loads((REPO_ROOT / "artifacts/metrics.json").read_text())
    stored = pd.read_parquet(REPO_ROOT / "data/interim/predictions.parquet")
    out: dict = {}

    # ---------------- Part 1: reproduction ----------------
    te = table[table.split == "test"].reset_index(drop=True)
    va = table[table.split == "validation"].reset_index(drop=True)
    rep = {}
    st = stored.set_index("case_id").loc[te.case_id]
    for name in ["B0", "B2", "FULL", "M1", "M2", "M3", "M4", "M5", "M6", "ALL"]:
        p = models[name].predict(te)
        y = te.bust.values
        pub = published["models"][name]
        rep[name] = {"auprc": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p),
                     "brier": brier_score_loss(y, p),
                     "published_auprc": pub["auprc"], "published_roc_auc": pub["roc_auc"],
                     "max_abs_diff_vs_stored_predictions": float(np.max(np.abs(p - st[f"p_{name}"].values)))}
    rep["FULL_minus_B2_max_abs_prob_diff"] = float(np.max(np.abs(models["FULL"].predict(te) - models["B2"].predict(te))))
    out["reproduction"] = rep

    # ---------------- Part 2: integrity ----------------
    tr = table[table.split == "train"]
    q = tr.groupby(["region_id", "lead_day", "season"])["norm_error"].quantile(model_config()["labels"]["primary_quantile"])
    qj = table.join(q.rename("q_recomputed"), on=["region_id", "lead_day", "season"])
    vt = pd.to_datetime(table.init_time) + pd.to_timedelta(table.lead_day, unit="D")
    spl = {s: (str(g.init_time.min()), str(g.init_time.max()), int(g.init_time.nunique()))
           for s, g in table.groupby("split")}
    out["integrity"] = {
        "rows": int(len(table)), "splits_init_range": spl,
        "bust_equals_norm_error_gt_q_primary": bool(((table.norm_error > table.q_primary).astype(int) == table.bust).all()),
        "q_primary_equals_train_quantile_max_abs_diff": float(np.nanmax(np.abs(qj.q_primary - qj.q_recomputed))),
        "valid_time_equals_init_plus_lead": bool((pd.to_datetime(table.valid_time) == vt).all()),
        "lead_days": sorted(int(d) for d in table.lead_day.unique()),
        "max_train_valid_time": str(pd.to_datetime(tr.valid_time).max()),
        "min_validation_init_time": str(va.init_time.min()),
        "hidden_bust_definition_holds": bool((table.hidden_bust == ((table.bust == 1) & (table.low_spread == 1)).astype(int)).all()),
        "base_rate": {s: float(g.bust.mean()) for s, g in table.groupby("split")},
    }

    # ---------------- Part 3: diagnosis on TRAIN + VALIDATION only ----------------
    far = model_config()["evaluation"]["alert_far"]
    pv = models["B2"].predict(va)
    thr = far_threshold(va.bust.values, pv, far)
    low = va.low_spread == 1
    d = {"threshold_b2_at_val_far": thr, "target_far": far,
         "validation": {
             "share_of_rows_low_spread": float(low.mean()),
             "bust_rate_low_spread": float(va.bust[low].mean()), "bust_rate_other": float(va.bust[~low].mean()),
             "share_of_busts_that_are_hidden": float(va.hidden_bust.sum() / va.bust.sum()),
             "b2_p_low_spread_max": float(pv[low].max()), "b2_p_low_spread_q99": float(np.quantile(pv[low], 0.99)),
             "b2_rows_low_spread_above_threshold": int((pv[low] >= thr).sum()),
         }}
    # within-regime discrimination on validation: every model and every single available feature
    yl = va.bust[low].values
    d["within_low_spread_validation"] = {"base_rate": float(yl.mean()), "n": int(low.sum()), "n_bust": int(yl.sum())}
    wl = {}
    for name in ["B2", "M1", "M2", "M3", "M4", "M5", "M6", "ALL"]:
        p = models[name].predict(va)[low.values]
        wl[name] = {"auprc": average_precision_score(yl, p), "roc_auc": roc_auc_score(yl, p),
                    "lift_over_base": average_precision_score(yl, p) / yl.mean()}
    d["within_low_spread_validation"]["models"] = wl
    feats = sorted(set(sum(FEATURE_SETS.values(), [])))
    fs = []
    for f in feats:
        x = va.loc[low, f].to_numpy(float)
        ok = np.isfinite(x)
        if ok.sum() < 1000 or len(np.unique(yl[ok])) < 2:
            continue
        auc = roc_auc_score(yl[ok], x[ok])
        fs.append({"feature": f, "roc_auc": auc, "abs_auc_minus_half": abs(auc - 0.5), "coverage": float(ok.mean())})
    d["within_low_spread_validation"]["single_feature_auc_top10"] = sorted(fs, key=lambda r: -r["abs_auc_minus_half"])[:10]
    # what a regime-specific operating point would give on VALIDATION (the global threshold cannot)
    best = max(wl, key=lambda k: wl[k]["auprc"])
    p = models[best].predict(va)[low.values]
    t_low = far_threshold(yl, p, 0.10)
    d["validation_regime_threshold_example"] = {
        "model": best, "far_within_low_spread": 0.10, "threshold": t_low,
        "hidden_recall": float((p[yl == 1] >= t_low).mean()),
        "precision": float(yl[p >= t_low].mean()) if (p >= t_low).any() else None}
    out["diagnosis_train_validation_only"] = d
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(clean_json(out), indent=1))
    print(json.dumps(clean_json(out), indent=1))


if __name__ == "__main__":
    main()
