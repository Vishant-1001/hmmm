"""Single test-set evaluation of frozen models + diagnostics. Writes:

artifacts/metrics.json, ablation_results.json, calibration.json, spread_skill.json,
feature_importance.json, predictions_test.parquet
"""
from __future__ import annotations

import json
import logging

import joblib
import numpy as np
import pandas as pd

from forecast_bust.config import RUN, clean_json, ARTIFACT_DIR, MODEL_DIR, data_config, model_config
from forecast_bust.data.assemble import load_states
from forecast_bust.evaluation import metrics as M
from forecast_bust.models.sentinel import DESCRIPTIONS, FEATURE_SETS
from forecast_bust.pipeline import PRED, TABLE

log = logging.getLogger(__name__)
MODELS = ["B0", "B1"] + list(FEATURE_SETS) + ["FULL"]
EXPERIMENTAL = ["EXP_RESIDUAL_B2"]  # reported only if present in predictions; never the Sentinel


def _score_col(name: str) -> str:
    return "s_B1" if name == "B1" else f"p_{name}"


def evaluate() -> dict:
    mcfg = model_config()
    cols = ["case_id", "init_time", "valid_time", "region_id", "row", "col", "lat", "lon", "lead_day", "season",
            "split", "error_m", "norm_error", "q_primary", "q_sensitivity", "bust", "bust_q95", "hidden_bust",
            "low_spread", "spread_m", "spread_pct", "sig_class", "support_level", "support_distance",
            "an_n_within", "an_bust_rate", "an_n_eligible"]
    df = pd.read_parquet(TABLE, columns=cols)
    pr = pd.read_parquet(PRED)
    df = df.merge(pr, on="case_id", how="inner")
    va = df[df["split"] == "validation"].reset_index(drop=True)
    te = df[df["split"] == "test"].reset_index(drop=True)
    if te.empty:
        raise RuntimeError("no test rows")
    far = mcfg["evaluation"]["alert_far"]
    results, ablation = {}, []
    for name in MODELS + [e for e in EXPERIMENTAL if f"p_{e}" in te.columns]:
        c = _score_col(name)
        thr = M.threshold_at_far(va["bust"].values, va[c].values, far)  # chosen on VALIDATION
        y, p = te["bust"].values, te[c].values
        r = {
            "description": DESCRIPTIONS[name], "is_probability": name != "B1",
            "auprc": M.auprc(y, p), "roc_auc": M.roc_auc(y, p),
            "brier": M.brier(y, p) if name != "B1" else None,
            "ece": M.ece(y, p) if name != "B1" else None,
            **{f"recall_at_far_{int(f * 100)}": M.recall_at_far(y, p, f) for f in mcfg["evaluation"]["fixed_far"]},
            "operating_point": {"chosen_on": "validation", "target_far": far, **M.confusion_at(y, p, thr)},
            "hidden_bust": M.hidden_bust_metrics(te, p, thr),
            # hidden-bust recall at fixed overall false-alarm rates (thresholds chosen on VALIDATION)
            "hidden_bust_at_far": {f"far_{int(f * 100)}": M.hidden_bust_metrics(
                te, p, M.threshold_at_far(va["bust"].values, va[c].values, f))["hidden_bust_recall"]
                for f in mcfg["evaluation"]["fixed_far"]},
            "warning_lead": M.warning_lead(te, p, thr),
            "peak_risk_day": M.peak_day_error(te, p),
            "spatial_overlap": M.spatial_overlap(te, p, thr),
        }
        results[name] = r
        ablation.append({"model": name, "description": DESCRIPTIONS[name], "auprc": r["auprc"],
                         "roc_auc": r["roc_auc"], "brier": r["brier"], "ece": r["ece"],
                         "hidden_bust_recall": r["hidden_bust"]["hidden_bust_recall"],
                         "recall_at_far_10": r["recall_at_far_10"],
                         "mean_lead_day_of_detected_busts": r["warning_lead"]["mean_lead_day_of_detected_busts"]})
    y = te["bust"].values
    base_rate = float(y.mean())
    boot = M.block_bootstrap_diff(te, te["p_FULL"].values, te["p_B2"].values, n=300, seed=mcfg["seed"])
    experimental = {}
    for e in EXPERIMENTAL:
        if f"p_{e}" in te.columns:
            b_e = M.block_bootstrap_diff(te, te[f"p_{e}"].values, te["p_B2"].values, n=300, seed=mcfg["seed"])
            g_e = results[e]["auprc"] - results["B2"]["auprc"]
            experimental[e] = {"status": "EXPERIMENTAL comparison only - not the Sentinel, not promoted",
                               "auprc_gain_vs_b2": g_e, "relative_gain": g_e / results["B2"]["auprc"],
                               "bootstrap": b_e,
                               "disagreement_vs_b2": M.disagreement_behaviour(te["bust"].values, te[f"p_{e}"].values,
                                                                              te["p_B2"].values)}
    gain = results["FULL"]["auprc"] - results["B2"]["auprc"]
    # verdict rule fixed in advance: CI of the AUPRC difference must exclude 0 AND relative gain >= 5%
    material = boot["ci95"][0] > 0 and gain / results["B2"]["auprc"] >= 0.05
    by_lead = []
    for d, g in te.groupby("lead_day"):
        by_lead.append({"lead_day": int(d), "n": int(len(g)), "bust_rate": float(g["bust"].mean()),
                        "rmse_m": float(np.sqrt((g["error_m"] ** 2).mean())),
                        "brier_B2": M.brier(g["bust"].values, g["p_B2"].values),
                        "brier_FULL": M.brier(g["bust"].values, g["p_FULL"].values),
                        "auprc_B2": M.auprc(g["bust"].values, g["p_B2"].values),
                        "auprc_FULL": M.auprc(g["bust"].values, g["p_FULL"].values),
                        "auprc_B0": M.auprc(g["bust"].values, g["p_B0"].values)})
    by_region = []
    for rid, g in te.groupby("region_id"):
        by_region.append({"region_id": rid, "lat": float(g["lat"].iloc[0]), "lon": float(g["lon"].iloc[0]),
                          "n": int(len(g)), "bust_rate": float(g["bust"].mean()),
                          "auprc_B2": M.auprc(g["bust"].values, g["p_B2"].values),
                          "auprc_FULL": M.auprc(g["bust"].values, g["p_FULL"].values)})
    boot_models = {"B2": te["p_B2"].values, "FULL": te["p_FULL"].values,
                   **{e: te[f"p_{e}"].values for e in EXPERIMENTAL if f"p_{e}" in te.columns}}
    thr_boot = {k: M.threshold_at_far(va["bust"].values, va["p_B2" if k == "B2" else f"p_{k}"].values, far)
                for k in boot_models}
    boot_all = M.block_bootstrap_metrics(te, boot_models, thr_boot, n=300, seed=mcfg["seed"])
    by_season = []
    for s, g in te.groupby("season"):
        by_season.append({"season": s, "n": int(len(g)), "bust_rate": float(g["bust"].mean()),
                          "auprc_B2": M.auprc(g["bust"].values, g["p_B2"].values),
                          "auprc_FULL": M.auprc(g["bust"].values, g["p_FULL"].values)})
    y95 = te["bust_q95"].values
    q95 = {n: {"auprc": M.auprc(y95, te[f"p_{n}_q95"].values), "brier": M.brier(y95, te[f"p_{n}_q95"].values)}
           for n in ("B0", "B2", "FULL")}
    frozen = {"auprc": M.auprc(y, te["p_FULL_frozen_memory"].values),
              "brier": M.brier(y, te["p_FULL_frozen_memory"].values)}
    ev_names = ["STRONG EVIDENCE", "MODERATE EVIDENCE", "WEAK EVIDENCE", "INSUFFICIENT HISTORICAL SUPPORT"]
    by_evidence = []
    for lv, g in te.groupby("evidence_level"):
        by_evidence.append({"evidence": ev_names[int(lv)], "n": int(len(g)), "bust_rate": float(g["bust"].mean()),
                            "mean_p_FULL": float(g["p_FULL"].mean()),
                            "auprc_FULL": M.auprc(g["bust"].values, g["p_FULL"].values)})
    manifest = json.loads((ARTIFACT_DIR / "experiment_manifest.json").read_text())
    full = pd.read_parquet(TABLE, columns=["init_time", "split", "bust"])
    split_info = {s: {"inits": int(g["init_time"].nunique()), "rows": int(len(g)),
                      "first_init": str(g["init_time"].min()), "last_init": str(g["init_time"].max()),
                      "bust_rate": float(g["bust"].mean())} for s, g in full.groupby("split")}
    out = {
        "dataset": {"source": data_config()["source"]["name"], "forecast_store": data_config()["source"]["forecast_store"],
                    "reference": "ERA5 (WeatherBench 2) - verification reference analysis, not perfect truth",
                    "target": "Z500 ensemble-mean regional error", "splits": split_info},
        "test_base_rate": base_rate, "n_test_rows": int(len(te)),
        "primary_metric": "AUPRC (bust = normalized error > TRAIN Q90 per region x lead x season)",
        "models": results,
        "full_vs_b2": {"auprc_gain": gain, "relative_gain": gain / results["B2"]["auprc"], "bootstrap": boot,
                       "verdict_rule": "material improvement iff 95% block-bootstrap CI of AUPRC(FULL)-AUPRC(B2) "
                                       "excludes 0 AND relative gain >= 5% (rule fixed before test evaluation)",
                       "material_improvement": bool(material)},
        "by_lead_day": by_lead, "by_season": by_season, "by_region": by_region,
        "bootstrap_ci": boot_all,
        "selected_groups": manifest["selected_groups"],
        "q95_sensitivity": q95, "frozen_memory_sensitivity": frozen, "by_evidence_level": by_evidence,
        "disagreement_full_vs_b2": M.disagreement_behaviour(y, te["p_FULL"].values, te["p_B2"].values),
        "experimental": experimental,
        "run": RUN or "v1", "sentinel_learner": manifest.get("sentinel_learner", "standard"),
        "test_history": mcfg.get("v2", {}).get("test_history"),
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(clean_json(out), indent=1, default=float))
    (ARTIFACT_DIR / "ablation_results.json").write_text(json.dumps(clean_json(ablation), indent=1, default=float))
    cal = {n: {"test": M.reliability_curve(y, te[f"p_{n}"].values),
               "validation": M.reliability_curve(va["bust"].values, va[f"p_{n}"].values)}
           for n in ("B0", "B2", "FULL")}
    cal["method"] = "isotonic regression fitted on validation predictions only, then frozen"
    cal["by_lead_day"] = {n: {int(d): {"ece": M.ece(g["bust"].values, g[f"p_{n}"].values),
                                       "mean_pred": float(g[f"p_{n}"].mean()), "obs_freq": float(g["bust"].mean()),
                                       "curve": M.reliability_curve(g["bust"].values, g[f"p_{n}"].values)}
                                  for d, g in te.groupby("lead_day")} for n in ("B2", "FULL")}
    (ARTIFACT_DIR / "pr_curves.json").write_text(json.dumps(clean_json(
        {n: M.pr_curve(y, te[_score_col(n)].values) for n in ["B0", "B1", "B2", "FULL"] +
         [e for e in EXPERIMENTAL if f"p_{e}" in te.columns]}), indent=1))
    (ARTIFACT_DIR / "calibration.json").write_text(json.dumps(clean_json(cal), indent=1))
    (ARTIFACT_DIR / "spread_skill.json").write_text(json.dumps(clean_json(spread_skill(te)), indent=1))
    models = joblib.load(MODEL_DIR / "models.joblib")
    (ARTIFACT_DIR / "feature_importance.json").write_text(json.dumps(clean_json(
        {n: models[n].importance() for n in list(FEATURE_SETS) + ["FULL"]}), indent=1))
    (ARTIFACT_DIR / "feature_diagnostics.json").write_text(json.dumps(clean_json(feature_diagnostics()), indent=1))
    keep = [f"p_{e}" for e in EXPERIMENTAL if f"p_{e}" in te.columns] + ["case_id", "init_time", "valid_time", "region_id", "lead_day", "season", "error_m", "norm_error",
            "bust", "hidden_bust", "spread_m", "p_B0", "p_B2", "p_FULL", "evidence_level", "support_level"]
    te[keep].to_parquet(ARTIFACT_DIR / "predictions_test.parquet", index=False)
    log.info("AUPRC B2 %.4f FULL %.4f (base rate %.3f) material=%s", results["B2"]["auprc"],
             results["FULL"]["auprc"], base_rate, material)
    return out


def spread_skill(te: pd.DataFrame) -> dict:
    """Spread-skill diagnostics on TEST rows (target regions) + rank histogram."""
    ds = load_states()
    n_mem = int(ds["n_members"].max())
    rows = []
    for d, g in te.groupby("lead_day"):
        rows.append({"lead_day": int(d), "rmse_m": float(np.sqrt((g["error_m"] ** 2).mean())),
                     "spread_m": float(np.sqrt((g["spread_m"] ** 2).mean())),
                     "spread_skill_ratio": float(np.sqrt((n_mem + 1) / n_mem * (g["spread_m"] ** 2).mean())
                                                 / np.sqrt((g["error_m"] ** 2).mean())),
                     "corr_spread_error": float(np.corrcoef(g["spread_m"], g["error_m"])[0, 1])})
    test_inits = pd.to_datetime(te["init_time"].unique())
    rk = ds["era5_rank"].sel(init=ds.init.isin(np.asarray(test_inits))).values
    hist = {}
    for j, h in enumerate(ds.lead.values):
        counts = np.bincount(rk[:, j].ravel().astype(int), minlength=n_mem + 1)
        hist[int(h) // 24] = counts.tolist()
    return {"split": "test", "n_members": n_mem, "per_lead": rows, "rank_histogram_by_lead_day": hist,
            "note": "spread-skill ratio uses the (M+1)/M finite-ensemble correction; diagnostics only"}


def feature_diagnostics() -> dict:
    """Univariate ROC AUC of each model input vs the bust label on TRAIN and VALIDATION rows
    (never test). Values < 0.5 mean higher values go with fewer busts. Used to diagnose
    whether information beyond spread exists and whether relationships transfer across periods."""
    from forecast_bust.models.sentinel import FEATURE_SETS as FS
    feats = FS["ALL"]
    df = pd.read_parquet(TABLE, columns=feats + ["split", "bust"])
    out = {}
    for f in feats:
        r = {}
        for s in ("train", "validation"):
            g = df[(df["split"] == s) & df[f].notna()]
            r[s] = M.roc_auc(g["bust"].values, g[f].values) if len(g) > 100 else None
            r[f"n_{s}"] = int(len(g))
        out[f] = r
    return {"note": "univariate ROC AUC vs bust; train and validation only", "features": out}
