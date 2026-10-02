"""Build the V4 demo bundle (python -m forecast_bust.demo.build_v4) from the v3 bundle + the frozen V4 model.

Same documented cases, forecast states (no verification column), memory store and reference grids as the
v3 bundle (artifacts/v3/demo, built from the same v3 table). Changes:

  artifacts/v4/demo/model/v4/model.joblib   frozen V4 CalibratedGBM (XGBoost hist + validation isotonic)
  artifacts/v4/demo/model/v4/metadata.json  features, params, target, thresholds, experiment id
  artifacts/v4/demo/memory.parquet          + magnitude_failure, pattern_failure, pattern_bust of each VERIFIED
                                            historical case (evidence only; eligibility stays causal)
  artifacts/v4/demo/cases/<id>/pattern_verification.json   local ACC / criteria per region-day: served ONLY
                                            by reveal(), like verification.json
"""
from __future__ import annotations

import json
import shutil

import pandas as pd

from forecast_bust.config import REPO_ROOT, clean_json, data_config

SRC = REPO_ROOT / "artifacts" / "v3" / "demo"
OUT = REPO_ROOT / "artifacts" / "v4" / "demo"
MODEL = REPO_ROOT / "models" / "v4"
LABELS = REPO_ROOT / "data" / "interim" / "v4" / "pattern_labels.parquet"


def build() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(SRC, OUT, ignore=shutil.ignore_patterns("v3", "model_meta.json"))
    (OUT / "model" / "v4").mkdir(parents=True)
    shutil.copy(MODEL / "models.joblib", OUT / "model" / "v4" / "models.joblib")
    md = json.loads((MODEL / "metadata.json").read_text())
    shutil.copy(MODEL / "metadata.json", OUT / "model" / "v4" / "metadata.json")
    lab = pd.read_parquet(LABELS)
    mem = pd.read_parquet(OUT / "memory.parquet")
    mem = mem.merge(lab[["case_id", "magnitude_failure", "pattern_failure", "pattern_bust"]], on="case_id", how="left")
    mem.to_parquet(OUT / "memory.parquet", index=False, compression="zstd")
    reg = json.loads((OUT / "registry.json").read_text())
    for c in reg["cases"]:
        st = pd.read_parquet(OUT / "cases" / c["case_id"] / "forecast_state.parquet", columns=["case_id", "region_id", "lead_day"])
        p = st.merge(lab, on="case_id", how="left")
        (OUT / "cases" / c["case_id"] / "pattern_verification.json").write_text(json.dumps(clean_json({
            "rows": [{"region_id": r.region_id, "lead_day": int(r.lead_day), "local_acc": r.local_acc,
                      "acc_q10": r.acc_q10, "magnitude_failure": int(r.magnitude_failure),
                      "pattern_failure": int(r.pattern_failure), "pattern_bust": int(r.pattern_bust)}
                     for r in p.itertuples()]}), default=float))
    met = json.loads((REPO_ROOT / "artifacts" / "v4" / "metrics.json").read_text())
    op = met["splits"]["validation"]["pattern_target"]["V4"]["operating_point"]
    meta = {
        "model_version": f"v4-final ({md['commit_hash']})", "experiment_id": md["experiment_id"],
        "model_artifact": "models/v4/models.joblib (V4 CalibratedGBM)",
        "exported_from": "models/v4 -> artifacts/v4/demo/model/v4/ (file copy)", "retrained_for_demo": False,
        "training_window": "2018-01-01 .. 2020-12-29", "calibration_window": "2021-01-01 .. 2021-12-28",
        "calibration": "isotonic regression of the raw XGBoost probability, fitted on validation (2021) only",
        "alert_threshold": float(op["threshold"]),
        "alert_threshold_definition": "calibrated pattern-bust probability giving a 10% false-alarm rate on VALIDATION",
        "confidence_definition": "reliability confidence = 1 - P(pattern bust); label LOW if P >= alert threshold, "
                                 "MODERATE if P >= the training pattern-bust rate, otherwise HIGH",
        "target": data_config()["variable"] + " 500 hPa (Z500): pattern-aware bust = normalized regional error > TRAIN Q90 "
                  "AND local 3x3 anomaly correlation < TRAIN Q10",
        "base_rate_pattern_bust": float(lab.loc[lab["case_id"].str[:4] <= "2020", "pattern_bust"].mean()),
        "base_rate_pattern_failure": float(lab.loc[lab["case_id"].str[:4] <= "2020", "pattern_failure"].mean()),
    }
    (OUT / "model_meta.json").write_text(json.dumps(clean_json(meta), indent=1))
    print(f"v4 demo bundle: {len(reg['cases'])} cases -> {OUT}")


if __name__ == "__main__":
    build()
