"""V3 predictive core: Quantile Regression Forest (QRF) conditional error-distribution model.

Replaces the retired B2/Sentinel XGBoost classifiers in the production prediction path
(see `models/sentinel.py` module docstring). Target: the existing `norm_error` continuous
label (`labels/build.py`, unchanged). Method, per the frozen v3 spec:

    forecast-state features
            -> RandomForestQuantileRegressor (quantile_forest package)
            -> conditional quantiles q10..q95 of norm_error
            -> ESTIMATED exceedance probability at the existing TRAIN-only bust threshold
               (q_primary), by interpolating the threshold into a fine quantile grid
            -> validation-only isotonic calibration
            -> CALIBRATED bust probability

Two distinct quantities are kept separate everywhere, in code and artifacts:

  * `estimated_exceedance_probability` - an interpolation over a finite predicted quantile
    grid. It is an ESTIMATE, not an exact CDF evaluation (the quantile_forest package has no
    native arbitrary-point CDF call; `RandomForestQuantileRegressor.predict(X,
    quantiles=[...])` returns requested quantile VALUES, confirmed against the installed
    1.4.2 API).
  * `calibrated_bust_probability` - the isotonic-calibrated version of the above, fitted on
    VALIDATION rows only, used for the product alert threshold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from quantile_forest import RandomForestQuantileRegressor
from sklearn.isotonic import IsotonicRegression

from forecast_bust.config import model_config

QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95]  # no duplicates, fixed, small (spec)
FINE_GRID_LEVELS = np.linspace(0.005, 0.995, 199)  # exceedance interpolation grid: FINAL evaluation
SEARCH_GRID_LEVELS = np.linspace(0.005, 0.995, 31)  # coarse grid: hyperparameter SEARCH only, over
# the SAME 0.005..0.995 range as the fine grid - never used for the frozen/final model or evaluation
DEFAULT_PARAMS = {"n_estimators": 600, "max_depth": 6, "min_samples_leaf": 100}


def _qrf_params(override: dict | None = None) -> dict:
    cfg = dict(model_config().get("qrf", {}).get("locked_params") or DEFAULT_PARAMS)
    cfg.update(override or {})
    return cfg


def assert_quantiles_ordered(q: dict[str, np.ndarray]) -> None:
    """Raise if the predicted quantile grid is not non-decreasing row-wise. Never silently
    sorted/clipped: a violation would indicate an implementation or estimator defect."""
    cols = [f"q{int(round(ql * 100))}" for ql in QUANTILES]
    arr = np.column_stack([q[c] for c in cols])
    bad = np.diff(arr, axis=1) < -1e-9
    if bad.any():
        raise ValueError(f"quantile ordering violated on {int(bad.any(axis=1).sum())} rows")


class QRFErrorModel:
    """Fit on TRAIN rows (target `norm_error`); calibrate exceedance probability on VALIDATION.

    `grid_levels`: the quantile grid used for the exceedance-probability interpolation (both
    during calibration in `fit()` and in `estimated_exceedance_probability()`). Defaults to the
    fine 199-point `FINE_GRID_LEVELS` - the grid the FROZEN/final model and final evaluation
    always use. Hyperparameter search may pass the coarser `SEARCH_GRID_LEVELS` (same 0.005..0.995
    range, fewer points) purely to make the search loop cheaper; the winning configuration is
    always refit with the default fine grid before anything is frozen or finally evaluated."""

    def __init__(self, features: list[str], params: dict | None = None, seed: int | None = None,
                grid_levels: np.ndarray | None = None):
        self.features = list(features)
        self.params = _qrf_params(params)
        self.seed = model_config()["seed"] if seed is None else seed
        self.grid_levels = FINE_GRID_LEVELS if grid_levels is None else np.asarray(grid_levels)
        self.calibrator: IsotonicRegression | None = None
        self.fill_: pd.Series | None = None  # TRAIN per-feature median, for NaN-safe inputs

    def _X(self, df: pd.DataFrame) -> np.ndarray:
        """RandomForestQuantileRegressor (unlike the retired XGBoost models) has no native
        missing-value handling; several feature groups are legitimately NaN before enough
        history exists (MEM/REC/EVO early in the record - see analogues/memory.py,
        analogues/recent.py). Impute with the TRAIN median, fit once in `fit()`, applied
        identically everywhere - not a target/leakage change, just making NaN finite."""
        x = df[self.features]
        if self.fill_ is not None:
            x = x.fillna(self.fill_)
        return x.to_numpy(dtype=np.float32)

    def fit(self, train: pd.DataFrame, val: pd.DataFrame, target: str = "norm_error") -> "QRFErrorModel":
        self.fill_ = train[self.features].median(numeric_only=True)
        self.model = RandomForestQuantileRegressor(**self.params, random_state=self.seed, n_jobs=4)
        self.model.fit(self._X(train), train[target].to_numpy(dtype=np.float32))
        raw_val = self.estimated_exceedance_probability(val)
        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
            raw_val, val["bust"].to_numpy())
        return self

    def predict_quantiles(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        q = self.model.predict(self._X(df), quantiles=QUANTILES)  # (n, len(QUANTILES))
        out = {f"q{int(round(ql * 100))}": q[:, i].astype(np.float32) for i, ql in enumerate(QUANTILES)}
        assert_quantiles_ordered(out)
        return out

    def estimated_exceedance_probability(self, df: pd.DataFrame, threshold_col: str = "q_primary") -> np.ndarray:
        """P(norm_error > threshold | features): interpolate each row's threshold into its own
        predicted (value -> level) quantile curve on `self.grid_levels`, 1 - level. Vectorised:
        for each row, count grid points <= threshold, then linearly interpolate the level
        between the bracketing grid points. An ESTIMATE from a finite grid, not an exact CDF."""
        qvals = self.model.predict(self._X(df), quantiles=self.grid_levels.tolist()).astype(np.float64)  # (n, G)
        thr = df[threshold_col].to_numpy(dtype=np.float64)
        g = len(self.grid_levels)
        idx = np.sum(qvals <= thr[:, None], axis=1)  # 0..g
        lo = np.clip(idx - 1, 0, g - 1)
        hi = np.clip(idx, 0, g - 1)
        x0 = np.take_along_axis(qvals, lo[:, None], axis=1)[:, 0]
        x1 = np.take_along_axis(qvals, hi[:, None], axis=1)[:, 0]
        y0, y1 = self.grid_levels[lo], self.grid_levels[hi]
        denom = np.where(x1 > x0, x1 - x0, 1.0)
        level = np.where(x1 > x0, y0 + (thr - x0) / denom * (y1 - y0), y0)
        level = np.where(idx == 0, 0.0, level)
        level = np.where(idx == g, 1.0, level)
        return np.clip(1.0 - level, 0.0, 1.0)

    def calibrated_bust_probability(self, df: pd.DataFrame) -> np.ndarray:
        if self.calibrator is None:
            raise RuntimeError("model is not calibrated")
        return self.calibrator.predict(self.estimated_exceedance_probability(df))

    def importance(self) -> dict:
        """Model-level feature importance only (native RF impurity-based importance). Never
        presented as a per-row/causal explanation - see explainability/explain.py."""
        imp = np.asarray(self.model.feature_importances_, dtype=float)
        tot = imp.sum() or 1.0
        return {f: float(v / tot) for f, v in sorted(zip(self.features, imp), key=lambda kv: -kv[1])}
