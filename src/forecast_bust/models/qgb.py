"""V3 predictive core: quantile gradient boosting (QGB) conditional error-distribution model.

One model family: scikit-learn `HistGradientBoostingRegressor(loss="quantile", quantile=q)`, one
estimator per level in QUANTILES, all sharing ONE hyper-parameter configuration. Replaces the
retired B2/Sentinel XGBoost classifiers in the production path (`models/sentinel.py`) and the
abandoned QRF attempt (repeated OOM on the 8 GB dev machine). Target: the existing `norm_error`
continuous label (`labels/build.py`, unchanged). Established method (quantile regression /
gradient-boosted quantile post-processing of NWP output), project-specific application:

    forecast-state features
            -> HistGradientBoostingRegressor(loss="quantile"), one fit per level, sequential
            -> conditional quantiles q10..q95 of norm_error
            -> rearrangement if (and only if) the independently fitted quantiles cross
            -> ESTIMATED exceedance probability at the existing TRAIN-only bust threshold
               (q_primary) from the predicted quantile function (see `exceedance_from_quantiles`)
            -> validation-only isotonic calibration
            -> CALIBRATED bust probability

`estimated_exceedance_probability` is an interpolation over six predicted quantiles with an
exponential tail beyond q10/q95 - an ESTIMATE, not an exact CDF. `calibrated_bust_probability` is
its isotonic recalibration fitted on VALIDATION rows only. The two are kept distinct everywhere.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

from forecast_bust.config import model_config

QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95]  # fixed, one model family (spec)
QCOLS = [f"q{int(round(ql * 100))}" for ql in QUANTILES]
DEFAULT_PARAMS = {"learning_rate": 0.05, "max_iter": 200, "max_leaf_nodes": 31, "min_samples_leaf": 50,
                  "l2_regularization": 1.0}
FIXED_PARAMS = {"loss": "quantile", "early_stopping": False}  # never searched
MODEL_TYPE = "quantile_gradient_boosting"
ESTIMATOR = "sklearn.ensemble.HistGradientBoostingRegressor(loss='quantile')"
EXCEEDANCE_METHOD = ("piecewise-linear CDF between the six predicted quantiles (q10..q95); beyond q95 "
                     "(below q10) an exponential tail through the (q90, q95) [(q10, q25)] knots. An "
                     "ESTIMATE from six predicted quantiles, not an exact CDF")
CROSSING_FIX = ("rearrangement (Chernozhukov, Fernandez-Val & Galichon 2010): each row's six predicted "
                "values are sorted, which never increases the quantile-estimation error; applied at "
                "prediction time only, crossing frequency measured and reported")


def _params(override: dict | None = None) -> dict:
    cfg = dict(model_config().get("qgb", {}).get("locked_params") or DEFAULT_PARAMS)
    cfg.update(override or {})
    return cfg


def crossing_stats(raw: np.ndarray) -> dict:
    """Frequency/size of quantile crossing in a raw (n, 6) prediction matrix (before rearrangement)."""
    d = np.diff(raw, axis=1)
    bad = d < 0
    return {"rows": int(len(raw)), "rows_crossing": int(bad.any(axis=1).sum()),
            "share_rows_crossing": float(bad.any(axis=1).mean()) if len(raw) else 0.0,
            "by_pair": {f"{QCOLS[j]}>{QCOLS[j + 1]}": float(bad[:, j].mean()) for j in range(bad.shape[1])},
            "max_violation": float(max(0.0, -d.min())) if len(raw) else 0.0,
            "median_violation_when_crossing": float(np.median(-d[bad])) if bad.any() else 0.0}


def assert_quantiles_ordered(q: dict[str, np.ndarray]) -> None:
    arr = np.column_stack([q[c] for c in QCOLS])
    if (np.diff(arr, axis=1) < -1e-9).any():
        raise ValueError("quantile ordering violated")


def exceedance_from_quantiles(qv: np.ndarray, thr: np.ndarray) -> np.ndarray:
    """Estimated P(norm_error > thr) from monotone predicted quantiles qv (n, 6) at QUANTILES.

    Inside [q10, q95]: linear interpolation of the CDF level between the bracketing quantiles.
    Above q95: survival decays exponentially, log S(t) linear through (q90, 0.10) and (q95, 0.05),
    i.e. S(t) = 0.05 * 2**(-(t - q95) / (q95 - q90)). Below q10: log F(t) linear through
    (q10, 0.10) and (q25, 0.25). The exponential tails keep a strictly ordered (rankable) value for
    rows whose threshold lies outside the predicted range instead of a flat 0/1. Ties between
    quantile values (zero spacing) are handled with a small floor on the spacing."""
    lv = np.asarray(QUANTILES)
    qv = np.asarray(qv, dtype=np.float64)
    thr = np.asarray(thr, dtype=np.float64)
    eps = 1e-6
    n, k = qv.shape
    idx = np.sum(qv <= thr[:, None], axis=1)  # number of quantile values <= thr, 0..k
    cdf = np.empty(n)
    inner = (idx > 0) & (idx < k)
    if inner.any():
        lo, r = idx[inner] - 1, np.nonzero(inner)[0]
        x0, x1 = qv[r, lo], qv[r, lo + 1]
        cdf[inner] = lv[lo] + (thr[inner] - x0) / np.maximum(x1 - x0, eps) * (lv[lo + 1] - lv[lo])
    top = idx == k
    if top.any():
        scale = np.maximum(qv[top, -1] - qv[top, -2], eps) / np.log((1 - lv[-2]) / (1 - lv[-1]))
        cdf[top] = 1.0 - (1 - lv[-1]) * np.exp(-(thr[top] - qv[top, -1]) / scale)
    bot = idx == 0
    if bot.any():
        scale = np.maximum(qv[bot, 1] - qv[bot, 0], eps) / np.log(lv[1] / lv[0])
        cdf[bot] = lv[0] * np.exp(-(qv[bot, 0] - thr[bot]) / scale)
    return np.clip(1.0 - cdf, 0.0, 1.0)  # clip removes floating-point noise only


class QGBErrorModel:
    """Fit on TRAIN rows (target `norm_error`); calibrate exceedance probability on VALIDATION.

    The six estimators are fitted one after another on a single float32 feature matrix; when
    `model_dir` is given each is written to `qgb_q<NN>.joblib` as soon as it is fitted. A fitted
    HistGradientBoostingRegressor holds only its trees (no training rows), so the family stays a
    few MB in memory. HistGradientBoosting handles NaN natively (MEM/REC/EVO are legitimately NaN
    early in the record), so no imputation is applied."""

    def __init__(self, features: list[str], params: dict | None = None, seed: int | None = None):
        self.features = list(features)
        self.params = _params(params)
        self.seed = model_config()["seed"] if seed is None else seed
        self.estimators: dict[str, HistGradientBoostingRegressor] = {}
        self.calibrator: IsotonicRegression | None = None
        self.val_crossing: dict | None = None

    def _X(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.features].to_numpy(dtype=np.float32)

    def fit(self, train: pd.DataFrame, val: pd.DataFrame, target: str = "norm_error",
            model_dir: Path | None = None) -> "QGBErrorModel":
        X = self._X(train)
        y = train[target].to_numpy(dtype=np.float32)
        for ql, col in zip(QUANTILES, QCOLS):
            est = HistGradientBoostingRegressor(quantile=ql, random_state=self.seed, **FIXED_PARAMS, **self.params)
            est.fit(X, y)
            self.estimators[col] = est
            if model_dir is not None:
                Path(model_dir).mkdir(parents=True, exist_ok=True)
                joblib.dump(est, Path(model_dir) / f"qgb_{col}.joblib")
            gc.collect()
        del X, y
        gc.collect()
        raw = self.raw_quantiles(val)
        self.val_crossing = crossing_stats(raw)
        p = exceedance_from_quantiles(np.sort(raw, axis=1), val["q_primary"].to_numpy())
        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, val["bust"].to_numpy())
        if model_dir is not None:
            joblib.dump(self.calibrator, Path(model_dir) / "calibration.joblib")
        return self

    def raw_quantiles(self, df: pd.DataFrame) -> np.ndarray:
        """(n, 6) predictions exactly as the independent estimators produce them (may cross)."""
        X = self._X(df)
        return np.column_stack([self.estimators[c].predict(X) for c in QCOLS])

    def predict_quantiles(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        q = np.sort(self.raw_quantiles(df), axis=1)  # rearrangement - see CROSSING_FIX
        return {c: q[:, j].astype(np.float32) for j, c in enumerate(QCOLS)}

    def estimated_exceedance_probability(self, df: pd.DataFrame, q: dict | None = None,
                                         threshold_col: str = "q_primary") -> np.ndarray:
        q = self.predict_quantiles(df) if q is None else q
        return exceedance_from_quantiles(np.column_stack([q[c] for c in QCOLS]), df[threshold_col].to_numpy())

    def calibrate(self, raw_p: np.ndarray) -> np.ndarray:
        if self.calibrator is None:
            raise RuntimeError("model is not calibrated")
        return self.calibrator.predict(raw_p)

    def calibrated_bust_probability(self, df: pd.DataFrame) -> np.ndarray:
        return self.calibrate(self.estimated_exceedance_probability(df))

    def predict_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Quantiles, estimated exceedance and calibrated bust probability in one pass."""
        q = self.predict_quantiles(df)
        out = pd.DataFrame(q, index=df.index)
        out["estimated_exceedance_probability"] = self.estimated_exceedance_probability(df, q)
        out["calibrated_bust_probability"] = self.calibrate(out["estimated_exceedance_probability"].to_numpy())
        return out

    def save(self, model_dir: Path, metadata: dict) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        for c in QCOLS:
            joblib.dump(self.estimators[c], model_dir / f"qgb_{c}.joblib")
        joblib.dump(self.calibrator, model_dir / "calibration.joblib")
        (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=1, default=str))


class V3PredictiveModel:
    """Production loader: the six quantile estimators + the calibrator as ONE model. The rest of
    the application calls `predict(df)`; it never touches individual estimators."""

    def __init__(self, model_dir: Path):
        model_dir = Path(model_dir)
        self.metadata = json.loads((model_dir / "metadata.json").read_text())
        m = QGBErrorModel(self.metadata["features"], params=self.metadata["params"], seed=self.metadata["seed"])
        m.estimators = {c: joblib.load(model_dir / f"qgb_{c}.joblib") for c in QCOLS}
        m.calibrator = joblib.load(model_dir / "calibration.joblib")
        self.model = m
        self.features = m.features
        self.model_type = MODEL_TYPE

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.model.predict_all(df)
