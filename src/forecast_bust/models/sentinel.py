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
if "DYN" in GROUPS:
    CANDIDATE_GROUPS["M7"] = "DYN"
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
    "M7": "B2 + wind / vorticity / divergence / MSLP state (v2 Group B completion)",
    "ALL": "B2 + every feature group (no validation-based selection)",
    "FULL": "Sentinel: B2 inputs + feature groups that improved VALIDATION AUPRC over B2",
    "EXP_RESIDUAL_B2": "EXPERIMENTAL (not the Sentinel): stacked model boosting from cross-fitted B2 log-odds, "
                       "validation-selected groups",
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


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class B2Margin:
    """Log-odds margin of the (uncalibrated) B2 spread-only model, used as the starting point of
    the residual Sentinel learner. TRAIN rows receive cross-fitted margins (leave-one-year-out B2
    fits with B2's early-stopped size) so the residual trees see B2 as it behaves out of sample;
    all other rows receive the margin of the full B2 model."""

    def __init__(self, b2: "CalibratedGBM", train: pd.DataFrame, target: str = "bust"):
        self.b2 = b2
        years = pd.to_datetime(train["init_time"]).dt.year.to_numpy()
        cf = np.full(len(train), np.nan)
        cfg = dict(model_config()["xgboost"])
        cfg.pop("early_stopping_rounds")
        cfg.pop("eval_metric")
        cfg["n_estimators"] = b2.best_iteration + 1
        for y in np.unique(years):
            fit_rows = years != y
            m = xgb.XGBClassifier(**cfg, random_state=model_config()["seed"], n_jobs=4,
                                  base_score=float(train[target].to_numpy()[fit_rows].mean()))
            m.fit(train[b2.features][fit_rows], train[target][fit_rows], verbose=False)
            cf[~fit_rows] = m.predict_proba(train[b2.features][~fit_rows])[:, 1]
        self.crossfit = pd.Series(_logit(cf), index=train["case_id"].to_numpy())
        self.crossfit_years = [int(y) for y in np.unique(years)]

    def margin(self, df: pd.DataFrame) -> np.ndarray:
        out = _logit(self.b2.predict_raw(df))
        cf = self.crossfit.reindex(df["case_id"].to_numpy()).to_numpy()
        return np.where(np.isfinite(cf), cf, out)


class CalibratedGBM:
    """XGBoost (CPU hist) + validation-only isotonic calibration.

    With `base` (a B2Margin) the booster starts from B2's log-odds instead of the base rate
    (residual learner); early stopping then decides how much beyond-spread structure the
    validation data support (0 extra trees = B2 itself)."""

    def __init__(self, features: list[str], name: str, base: B2Margin | None = None):
        self.features = list(features)
        self.name = name
        self.base = base
        self.calibrator: IsotonicRegression | None = None

    @property
    def learner(self) -> str:
        return "residual_b2" if getattr(self, "base", None) is not None else "standard"

    def _margin(self, df: pd.DataFrame):
        return self.base.margin(df) if getattr(self, "base", None) is not None else None

    def fit(self, train: pd.DataFrame, val: pd.DataFrame, target: str = "bust"):
        cfg = dict(model_config()["xgboost"])
        es = cfg.pop("early_stopping_rounds")
        pos = train[target].mean()
        kw = {}
        if self.base is not None:
            kw = {"base_margin": self._margin(train), "base_margin_eval_set": [self._margin(val)]}
        self.model = xgb.XGBClassifier(**cfg, early_stopping_rounds=es, random_state=model_config()["seed"],
                                       n_jobs=4, base_score=float(pos) if self.base is None else 0.5)
        self.model.fit(train[self.features], train[target], eval_set=[(val[self.features], val[target])],
                       verbose=False, **kw)
        raw_val = self.predict_raw(val)
        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw_val, val[target])
        self.best_iteration = int(self.model.best_iteration)
        return self

    def predict_raw(self, df: pd.DataFrame) -> np.ndarray:
        m = self._margin(df)
        kw = {"base_margin": m} if m is not None else {}
        return self.model.predict_proba(df[self.features], **kw)[:, 1]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.calibrator is None:
            raise RuntimeError("model is not calibrated")
        return self.calibrator.predict(self.predict_raw(df))

    def importance(self) -> dict:
        imp = self.model.get_booster().get_score(importance_type="total_gain")
        tot = sum(imp.values()) or 1.0
        return {k: v / tot for k, v in sorted(imp.items(), key=lambda kv: -kv[1])}

    def contributions(self, df: pd.DataFrame) -> np.ndarray:
        """TreeSHAP contributions (log-odds scale, last column = bias). For the residual learner
        the contributions describe the trees added on top of B2; the B2 margin is returned
        separately by `base_margin_of`."""
        d = xgb.DMatrix(df[self.features])
        return self.model.get_booster().predict(d, pred_contribs=True,
                                                iteration_range=(0, self.best_iteration + 1))

    def base_margin_of(self, df: pd.DataFrame) -> np.ndarray | None:
        return self._margin(df)
