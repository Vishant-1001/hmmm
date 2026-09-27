"""python -m forecast_bust.evaluate  (single evaluation of frozen models on the test split)"""
import json
import logging

from forecast_bust.evaluation.run import evaluate

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    out = evaluate()
    print(json.dumps({k: {m: round(v["auprc"], 4) for m, v in out["models"].items()} if k == "models" else v
                      for k, v in out.items() if k in ("models", "test_base_rate", "full_vs_b2")}, indent=1, default=str))
