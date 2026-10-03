"""Served MVP model: the frozen B2 spread baseline (v2, locked 8b196ee), loaded from its exported artifacts.

    artifacts/v2/demo/model/b2_booster.json   XGBoost booster (7 spread / lead / region / season inputs)
    artifacts/v2/demo/model/models.json       B2.isotonic: validation-2021 isotonic calibrator (x, y knots)
    artifacts/v2/demo/model/served_b2.json    model card (ids, windows, alert threshold), scripts/export_served_b2.py

Identical to models.sentinel.CalibratedGBM.predict for the trained B2 (parity-tested); raw_probability and the
calibrated bust probability are kept distinct.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

MODEL_TYPE = "b2_spread_xgboost"
B2_FEATURES = ["spread_m", "spread_pct", "spread_thr_ratio", "lead_day", "region_code", "season_code", "init_hour"]


class B2Served:
    def __init__(self, model_dir: Path):
        model_dir = Path(model_dir)
        spec = json.loads((model_dir / "models.json").read_text())["B2"]
        self.card = json.loads((model_dir / "served_b2.json").read_text())
        self.features = list(spec["features"])
        if self.features != B2_FEATURES:
            raise ValueError(f"served B2 features changed: {self.features}")
        self.best_iteration = int(spec["best_iteration"])
        self.booster = xgb.Booster()
        self.booster.load_model(model_dir / Path(spec["booster"]).name)
        self.iso_x = np.asarray(spec["isotonic"]["x"], float)
        self.iso_y = np.asarray(spec["isotonic"]["y"], float)
        self.model_type = MODEL_TYPE

    def _dm(self, rows: pd.DataFrame) -> xgb.DMatrix:
        return xgb.DMatrix(rows[self.features])

    def predict(self, rows: pd.DataFrame) -> pd.DataFrame:
        raw = self.booster.predict(self._dm(rows), iteration_range=(0, self.best_iteration + 1))
        return pd.DataFrame({"raw_probability": raw, "calibrated_bust_probability": np.interp(raw, self.iso_x, self.iso_y)},
                            index=rows.index)

    def contributions(self, rows: pd.DataFrame) -> np.ndarray:
        """TreeSHAP (log-odds, before calibration; last column = bias)."""
        return self.booster.predict(self._dm(rows), pred_contribs=True, iteration_range=(0, self.best_iteration + 1))
