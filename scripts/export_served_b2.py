"""Write the served-model card of the MVP: artifacts/v2/demo/model/served_b2.json.

    .venv/bin/python scripts/export_served_b2.py

Everything is copied from existing frozen artifacts (no fitting): the exported B2 booster/isotonic calibrator
(artifacts/v2/demo/model/models.json), the locked hyper-parameters (config/model_v2.yaml), the training /
calibration windows (artifacts/v2/demo/model_meta.json) and the validation-chosen alert threshold and test
metrics of B2 (artifacts/v2/metrics.json).
"""
import hashlib
import json

import xgboost as xgb
import yaml

from forecast_bust.config import REPO_ROOT

RUN = REPO_ROOT / "artifacts" / "v2"
MODEL = RUN / "demo" / "model"


def main() -> None:
    spec = json.loads((MODEL / "models.json").read_text())["B2"]
    meta = json.loads((RUN / "demo" / "model_meta.json").read_text())
    met = json.loads((RUN / "metrics.json").read_text())["models"]["B2"]
    params = yaml.safe_load((REPO_ROOT / "config" / "model_v2.yaml").read_text())["v2"]["b2_xgboost"]
    bst = xgb.Booster()
    bst.load_model(MODEL / spec["booster"].split("/")[-1])
    gain = bst.get_score(importance_type="total_gain")
    tot = sum(gain.values()) or 1.0
    op = met["operating_point"]
    if op["chosen_on"] != "validation":
        raise ValueError("alert threshold must be chosen on validation")
    card = {
        "model_id": "b2_spread_calibrated",
        "model_type": "b2_spread_xgboost",
        "model_version": "B2 v2-final (locked 8b196ee)",
        "provider": "ecmwf_research",
        "dataset_mode": "historical_replay_real_ecmwf_ifs_ens_era5",
        "calibration_version": "isotonic_validation_2021_v2",
        "estimator": "xgboost gradient-boosted trees (CPU hist) + isotonic calibration fitted on validation only",
        "model_artifact": "artifacts/v2/demo/model/b2_booster.json + models.json (B2.isotonic)",
        "booster_sha256": hashlib.sha256((MODEL / "b2_booster.json").read_bytes()).hexdigest(),
        "exported_from": meta["exported_from"],
        "retrained_for_demo": False,
        "training_window": meta["training_window"],
        "calibration_window": meta["calibration_window"],
        "calibration": "isotonic regression fitted on VALIDATION (2021) predictions only, then frozen",
        "params": {k: float(v) for k, v in params.items()},
        "best_iteration": int(spec["best_iteration"]),
        "features": list(spec["features"]),
        "importance": {k: v / tot for k, v in sorted(gain.items(), key=lambda kv: -kv[1])},
        "alert_threshold": float(op["threshold"]),
        "alert_threshold_definition": "calibrated B2 bust probability giving a 10% false-alarm rate on VALIDATION (2021)",
        "target": "bust = regional Z500 ensemble-mean RMSE vs ERA5, normalized by the TRAIN anomaly scale, exceeds the "
                  "TRAIN Q90 for (region, lead day, season)",
        "probability_meaning": "estimated probability that this region-day's Z500 forecast error exceeds the project-defined "
                               "large-error (bust) threshold. Not the probability of a weather event, and not a global "
                               "forecast-quality score.",
        "test_metrics_2022": {k: met[k] for k in ("auprc", "roc_auc", "brier", "ece", "recall_at_far_10")},
    }
    (MODEL / "served_b2.json").write_text(json.dumps(card, indent=1))
    print(json.dumps({k: card[k] for k in ("model_id", "model_version", "alert_threshold", "test_metrics_2022")}, indent=1))


if __name__ == "__main__":
    main()
