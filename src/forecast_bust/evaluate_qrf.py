"""python -m forecast_bust.evaluate_qrf  (single evaluation of the frozen v3 QRF model)"""
import json
import logging

from forecast_bust.evaluation.run_qrf import evaluate_qrf

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    out = evaluate_qrf()
    print(json.dumps({k: out[k] for k in ("run", "dev_split", "label", "test_base_rate")} |
                     {"qrf_auprc": round(out["qrf"]["auprc"], 4), "qrf_mae_q50": round(out["qrf"]["mae_q50"], 4)},
                     indent=1, default=str))
