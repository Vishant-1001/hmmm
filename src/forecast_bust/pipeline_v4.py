"""V4 pipeline: pattern-aware bust labels + target audit, then ONE shared CalibratedGBM (config/model_v4.yaml).

    FBS_RUN=v3 FBS_SPLIT=dev python -m forecast_bust.pipeline_v4 audit   # labels + prevalence audit (no model)
    FBS_RUN=v3 FBS_SPLIT=dev python -m forecast_bust.pipeline_v4 train   # A/B adequacy fit on the dev split
    FBS_RUN=v3                python -m forecast_bust.pipeline_v4 train   # final fit (only after a passed gate)

Rows, features, magnitude labels and splits come from the v3 namespace table (unchanged). Outputs go to
artifacts/v4[/dev], models/v4[/dev], data/interim/v4[/dev].
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import sys
import time

import joblib
import numpy as np
import pandas as pd
import xarray as xr
import yaml

from forecast_bust.config import DEV, REPO_ROOT, RUN, clean_json
from forecast_bust.labels.pattern import PATTERN_COLS, local_acc_table, pattern_labels
from forecast_bust.models.sentinel import FEATURE_SETS, CalibratedGBM
from forecast_bust.pipeline import TABLE, _git_rev, _versions

log = logging.getLogger(__name__)
SUB = "dev" if DEV else ""
ART = REPO_ROOT / "artifacts" / "v4" / SUB
MODEL_DIR = REPO_ROOT / "models" / "v4" / SUB
INTERIM = REPO_ROOT / "data" / "interim" / "v4" / SUB
ACC = REPO_ROOT / "data" / "interim" / "v4" / "local_acc.parquet"   # split-independent (verification only)
LABELS = INTERIM / "pattern_labels.parquet"
PRED = INTERIM / "predictions.parquet"
CASE_COLS = ["case_id", "init_time", "region_id", "lead_day", "season", "split", "bust", "low_spread",
             "hidden_bust", "norm_error", "spread_pct"]


def v4_config() -> dict:
    return yaml.safe_load((REPO_ROOT / "config" / "model_v4.yaml").read_text())


def build_labels() -> pd.DataFrame:
    if RUN != "v3":
        raise RuntimeError("run with FBS_RUN=v3 (V4 reuses the v3 rows, features and magnitude labels)")
    if not ACC.exists():
        ACC.parent.mkdir(parents=True, exist_ok=True)
        local_acc_table(xr.open_dataset(REPO_ROOT / "data" / "interim" / "states.nc")).to_parquet(ACC, index=False)
    cases = pd.read_parquet(TABLE, columns=CASE_COLS, filters=[("split", "in", ["train", "validation", "test"])])
    labels, th = pattern_labels(cases, pd.read_parquet(ACC), v4_config()["acc_threshold"]["quantile"])
    INTERIM.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(LABELS, index=False)
    (ART / "acc_thresholds.json").write_text(json.dumps(clean_json({
        "definition": "acc_q10 = TRAIN-only Q10 of local 3x3 ACC per region x lead day x season",
        "fitted_on": "train", "thresholds": th.to_dict(orient="records")}), indent=1, default=str))
    return cases.merge(labels, on="case_id")


def audit(d: pd.DataFrame) -> dict:
    out = {"acc_coverage": {s: float(g["local_acc"].notna().mean()) for s, g in d.groupby("split")},
           "invalid_acc_rows": {s: int(g["local_acc"].isna().sum()) for s, g in d.groupby("split")}, "splits": {}}
    for s, g in d.groupby("split"):
        pb = g["pattern_bust"] == 1
        out["splits"][s] = {
            "rows": int(len(g)), "pattern_bust_prevalence": float(g["pattern_bust"].mean()),
            "magnitude_bust_prevalence": float(g["bust"].mean()),
            "pattern_failure_prevalence": float(g["pattern_failure"].mean()),
            "share_of_magnitude_busts_that_are_pattern_busts": float(g.loc[g["bust"] == 1, "pattern_bust"].mean()),
            "share_of_pattern_failures_that_are_magnitude_busts": float(g.loc[g["pattern_failure"] == 1, "bust"].mean()),
            "joint_vs_independence_ratio": float(g["pattern_bust"].mean() / (g["bust"].mean() * g["pattern_failure"].mean())),
            "positive_rows": int(pb.sum()), "positive_inits": int(g.loc[pb, "init_time"].nunique()),
            "n_inits": int(g["init_time"].nunique()),
            "hidden_pattern_busts": int((g["hidden_pattern_bust"] == 1).sum()),
            "positives_by_lead": {int(k): int(v) for k, v in g[pb].groupby("lead_day").size().items()},
            "positives_by_region": {"min": int(g[pb].groupby("region_id").size().reindex(g["region_id"].unique(), fill_value=0).min()),
                                    "median": float(g[pb].groupby("region_id").size().reindex(g["region_id"].unique(), fill_value=0).median()),
                                    "max": int(g[pb].groupby("region_id").size().max())},
            "local_acc_quantiles": {str(q): float(g["local_acc"].quantile(q)) for q in (0.01, 0.1, 0.5, 0.9)},
        }
    (ART / "target_audit.json").write_text(json.dumps(clean_json(out), indent=1, default=str))
    return out


def train(d: pd.DataFrame) -> None:
    """A/B: the SAME classifier, features and params fitted to the pattern target (V4) and to the magnitude
    target (reference for the adequacy comparison). Only V4 is a product candidate."""
    cfg = v4_config()
    feats = FEATURE_SETS[cfg["model"]["features"]]
    t = pd.read_parquet(TABLE, columns=["case_id", *[f for f in feats if f not in d.columns]],
                        filters=[("split", "in", ["train", "validation", "test"])])
    d = d.merge(t, on="case_id")
    d = d[d["pattern_bust"].notna()]
    tr, va = d[d["split"] == "train"], d[d["split"] == "validation"]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    v4 = CalibratedGBM(feats, "V4_pattern", params=cfg["model"]["params"]).fit(tr, va, target="pattern_bust")
    t_v4 = time.time() - t0
    mag = CalibratedGBM(feats, "V4_magnitude_reference", params=cfg["model"]["params"]).fit(tr, va, target="bust")
    joblib.dump({"V4": v4, "magnitude_reference": mag}, MODEL_DIR / "models.joblib")
    rows = d[d["split"].isin(["validation", "test"])]
    pred = pd.DataFrame({"case_id": rows["case_id"].to_numpy(),
                         "raw_probability": v4.predict_raw(rows), "calibrated_bust_probability": v4.predict(rows),
                         "p_magnitude_reference": mag.predict(rows)})
    b0 = tr.groupby(["region_id", "lead_day", "season"])["pattern_bust"].mean().rename("p_B0_pattern").reset_index()
    pred = pred.merge(rows[["case_id", "region_id", "lead_day", "season"]], on="case_id").merge(
        b0, on=["region_id", "lead_day", "season"], how="left").drop(columns=["region_id", "lead_day", "season"])
    pred["p_B0_pattern"] = pred["p_B0_pattern"].fillna(tr["pattern_bust"].mean())
    pred.to_parquet(PRED, index=False)
    meta = {"experiment_id": f"v4-{'dev' if DEV else 'final'}-{_git_rev()}", "commit_hash": _git_rev(),
            "created": dt.datetime.now(dt.timezone.utc).isoformat(), "dev_split": DEV, "model_type": "pattern_aware_xgboost",
            "target": "pattern_bust = (norm_error > TRAIN Q90) AND (local 3x3 ACC < TRAIN Q10)", "features": feats,
            "params": cfg["model"]["params"], "best_iteration": {"V4": v4.best_iteration, "magnitude_reference": mag.best_iteration},
            "fit_seconds_v4": round(t_v4, 1), "importance_top15": dict(list(v4.importance().items())[:15]),
            "rows": {s: int((d["split"] == s).sum()) for s in ("train", "validation", "test")},
            "calibration": "isotonic on validation raw probabilities only", "software": _versions()}
    (MODEL_DIR / "metadata.json").write_text(json.dumps(clean_json(meta), indent=1, default=str))
    (ART / "experiment_manifest.json").write_text(json.dumps(clean_json(meta), indent=1, default=str))
    log.info("V4 fit %.0fs (best it %d); magnitude reference best it %d", t_v4, v4.best_iteration, mag.best_iteration)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    step = sys.argv[1] if len(sys.argv) > 1 else "audit"
    d = build_labels()
    a = audit(d)
    print(json.dumps({"acc_coverage": a["acc_coverage"], **{s: {k: v for k, v in x.items() if k not in
          ("positives_by_lead", "local_acc_quantiles")} for s, x in a["splits"].items()}}, indent=1, default=str))
    if step == "train":
        train(d)
