"""Baselines (B0, B1, B2) and the Sentinel gradient-boosted classifier.

All learned models share one protocol so the ablation is fair:
  1. fit on TRAIN rows (early stopping monitored on VALIDATION),
  2. isotonic calibration fitted on VALIDATION predictions only, then frozen,
  3. single evaluation on TEST.
B2 uses exactly the same learner and protocol as Sentinel; it is not weakened.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

from forecast_bust.config import model_config
from forecast_bust.features.build import GROUPS

GROUP = ["region_id", "lead_day", "season"]

# Monotone-in-time counts (memory size) act as hidden timestamps; they are shown as evidence
# but never used as model inputs (decided on the dev split, see docs/leakage_controls.md).
NON_MODEL_FEATURES = {"an_n_eligible", "an_n_within", "rec_n"}


def _g(name: str) -> list[str]:
    return [f for f in GROUPS[name] if f not in NON_MODEL_FEATURES]


CANDIDATE_GROUPS = {"M1": "ATM", "M2": "ENS", "M3": "PAT", "M4": "EVO", "M5": "MEM", "M6": "REC"}
FEATURE_SETS = {"B2": _g("SPREAD")}
FEATURE_SETS.update({m: _g("SPREAD") + _g(g) for m, g in CANDIDATE_GROUPS.items()})
FEATURE_SETS["ALL"] = _g("SPREAD") + sum((_g(g) for g in CANDIDATE_GROUPS.values()), [])


def full_feature_set(selected_groups: list[str]) -> list[str]:
    """FULL = B2 inputs + feature groups validated on the VALIDATION split."""
    return _g("SPREAD") + sum((_g(g) for g in selected_groups), [])


DESCRIPTIONS = {
    "B0": "Climatological bust frequency (TRAIN, per region x lead x season)",
    "B1": "Raw ensemble-spread score (spread percentile; ranking score, not a probability)",
    "B2": "Calibrated spread-only gradient-boosted model (spread, spread percentile, lead, region, season, init hour)",
    "M1": "B2 + atmospheric state", "M2": "B2 + ensemble behaviour", "M3": "B2 + large-scale pattern (PCA)",
    "M4": "B2 + forecast evolution", "M5": "B2 + historical forecast-state memory (analogues)",
    "M6": "B2 + recent verified forecast-error behaviour",
    "ALL": "B2 + every feature group (no validation-based selection)",
    "FULL": "Sentinel: B2 inputs + feature groups that improved VALIDATION AUPRC over B2",
}


class B0Climatology:
    def fit(self, train: pd.DataFrame, target: str = "bust"):
        self.table = train.groupby(GROUP)[target].mean().rename("p").reset_index()
        self.global_rate = float(train[target].mean())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        m = df[GROUP].merge(self.table, on=GROUP, how="left")["p"].to_numpy()
        return np.where(np.isnan(m), self.global_rate, m)


def b1_score(df: pd.DataFrame) -> np.ndarray:
    return df["spread_pct"].to_numpy(dtype=float)


class CalibratedGBM:
    """XGBoost (CPU hist) + validation-only isotonic calibration."""

    def __init__(self, features: list[str], name: str):
        self.features = list(features)
        self.name = name
        self.calibrator: IsotonicRegression | None = None

    def fit(self, train: pd.DataFrame, val: pd.DataFrame, target: str = "bust"):
        cfg = dict(model_config()["xgboost"])
        es = cfg.pop("early_stopping_rounds")
        pos = train[target].mean()
        self.model = xgb.XGBClassifier(**cfg, early_stopping_rounds=es, random_state=model_config()["seed"],
                                       n_jobs=4, base_score=float(pos))
        self.model.fit(train[self.features], train[target], eval_set=[(val[self.features], val[target])],
                       verbose=False)
        raw_val = self.predict_raw(val)
        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw_val, val[target])
        self.best_iteration = int(self.model.best_iteration)
        return self

    def predict_raw(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(df[self.features])[:, 1]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.calibrator is None:
            raise RuntimeError("model is not calibrated")
        return self.calibrator.predict(self.predict_raw(df))

    def importance(self) -> dict:
        imp = self.model.get_booster().get_score(importance_type="total_gain")
        tot = sum(imp.values()) or 1.0
        return {k: v / tot for k, v in sorted(imp.items(), key=lambda kv: -kv[1])}

    def contributions(self, df: pd.DataFrame) -> np.ndarray:
        """TreeSHAP contributions (log-odds scale, last column = bias)."""
        d = xgb.DMatrix(df[self.features])
        return self.model.get_booster().predict(d, pred_contribs=True,
                                                iteration_range=(0, self.best_iteration + 1))
