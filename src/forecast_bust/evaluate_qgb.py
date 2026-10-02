"""python -m forecast_bust.evaluate_qgb  (single evaluation of the v3 quantile-gradient-boosting model)"""
import json
import logging

from forecast_bust.evaluation.run_qgb import evaluate_qgb

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    out = evaluate_qgb()
    q = out["qgb"]
    print(json.dumps({k: out[k] for k in ("run", "dev_split", "label", "test_base_rate")} |
                     {"auprc": round(q["calibrated_bust_probability"]["auprc"], 4),
                      "brier": round(q["calibrated_bust_probability"]["brier"], 4),
                      "mae_q50": round(q["distribution"]["mae_q50"], 4)}, indent=1, default=str))
