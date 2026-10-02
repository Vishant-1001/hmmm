"""V4 production loader: ONE shared XGBoost classifier (models.sentinel.CalibratedGBM) for the pattern-aware
bust target (labels/pattern.py, config/model_v4.yaml). raw_probability (XGBoost) and calibrated_bust_probability
(validation-only isotonic) are kept distinct."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_TYPE = "pattern_aware_xgboost"


class V4Model:
    def __init__(self, model_dir: Path):
        model_dir = Path(model_dir)
        self.metadata = json.loads((model_dir / "metadata.json").read_text())
        self.gbm = joblib.load(model_dir / "models.joblib")["V4"]
        self.features = list(self.gbm.features)
        self.model_type = MODEL_TYPE

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        raw = self.gbm.predict_raw(rows)
        return pd.DataFrame({"raw_probability": raw, "calibrated_bust_probability": self.gbm.calibrator.predict(raw)},
                            index=rows.index)

    def contributions(self, rows: pd.DataFrame) -> np.ndarray:
        """TreeSHAP (log-odds, before calibration; last column = bias)."""
        return self.gbm.contributions(rows)
