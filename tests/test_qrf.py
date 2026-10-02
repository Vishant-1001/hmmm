"""v3 QRF predictive core: quantile ordering, determinism, exceedance probability, calibration,
persistence, and feature-provenance (no B2/forbidden inputs)."""
import numpy as np
import pandas as pd
import pytest

from forecast_bust.models.qrf import QUANTILES, QRFErrorModel, assert_quantiles_ordered


def _toy(n_per_year=2000, years=(2018, 2019), val_year=2020, seed=0):
    rng = np.random.default_rng(seed)

    def _gen(n, y):
        x = rng.normal(size=n)
        s = rng.gamma(2, 1.0, n)
        err = np.abs(0.5 + 0.4 * x + 0.3 * s) * rng.gamma(2, 1.0, n) / 2
        return pd.DataFrame({"init_time": pd.Timestamp(f"{y}-06-01"), "x": x, "spread_m": s,
                             "norm_error": err, "q_primary": 1.2})

    train = pd.concat([_gen(n_per_year, y) for y in years], ignore_index=True)
    val = _gen(n_per_year, val_year)
    for df in (train, val):
        df["bust"] = (df["norm_error"] > df["q_primary"]).astype(int)
        df["case_id"] = [f"c{i}" for i in range(len(df))]
    return train, val


FEATURES = ["x", "spread_m"]
PARAMS = {"n_estimators": 50, "max_depth": 4, "min_samples_leaf": 20}


def test_quantiles_ordered_and_deterministic():
    train, val = _toy()
    m1 = QRFErrorModel(FEATURES, params=PARAMS, seed=0).fit(train, val)
    q1 = m1.predict_quantiles(val)
    assert_quantiles_ordered(q1)  # no exception = ordered
    cols = [f"q{int(round(ql * 100))}" for ql in QUANTILES]
    assert cols == ["q10", "q25", "q50", "q75", "q90", "q95"]

    m2 = QRFErrorModel(FEATURES, params=PARAMS, seed=0).fit(train, val)
    q2 = m2.predict_quantiles(val)
    for c in cols:
        np.testing.assert_allclose(q1[c], q2[c])


def test_quantile_ordering_violation_is_raised_not_hidden():
    bad = {"q10": np.array([5.0]), "q25": np.array([1.0]), "q50": np.array([2.0]),
           "q75": np.array([3.0]), "q90": np.array([4.0]), "q95": np.array([6.0])}
    with pytest.raises(ValueError):
        assert_quantiles_ordered(bad)


def test_estimated_exceedance_probability_monotone_in_threshold():
    train, val = _toy()
    m = QRFErrorModel(FEATURES, params=PARAMS, seed=0).fit(train, val)
    row = val.iloc[:50].copy()
    low = row.assign(q_primary=0.1)
    high = row.assign(q_primary=10.0)
    p_low = m.estimated_exceedance_probability(low)
    p_high = m.estimated_exceedance_probability(high)
    assert np.all(p_low >= p_high - 1e-9)
    assert np.all((p_low >= 0) & (p_low <= 1))
    assert np.all((p_high >= 0) & (p_high <= 1))


def test_calibrated_bust_probability_in_unit_interval_and_fit_only_on_validation():
    train, val = _toy()
    m = QRFErrorModel(FEATURES, params=PARAMS, seed=0).fit(train, val)
    p = m.calibrated_bust_probability(val)
    assert np.all((p >= 0) & (p <= 1))
    assert np.isfinite(p).all()
    # calibrator knots come only from validation predictions/labels, not train
    assert len(m.calibrator.X_thresholds_) <= len(val) + 1


def test_uncalibrated_model_raises():
    m = QRFErrorModel(FEATURES, params=PARAMS, seed=0)
    with pytest.raises(RuntimeError):
        m.calibrated_bust_probability(_toy()[1])


def test_persistence_round_trip(tmp_path):
    import joblib
    train, val = _toy()
    m = QRFErrorModel(FEATURES, params=PARAMS, seed=0).fit(train, val)
    before = m.calibrated_bust_probability(val)
    p = tmp_path / "qrf.joblib"
    joblib.dump(m, p)
    m2 = joblib.load(p)
    after = m2.calibrated_bust_probability(val)
    np.testing.assert_allclose(before, after)


def test_feature_set_excludes_forbidden_and_b2_inputs():
    from forecast_bust.features.build import FORBIDDEN_INPUTS
    from forecast_bust.models.sentinel import FEATURE_SETS
    qrf_features = FEATURE_SETS["ALL"]  # the production QRF feature set reuses this group union
    assert not any(f.startswith(x) for f in qrf_features for x in FORBIDDEN_INPUTS)
    assert all("b2" not in f.lower() for f in qrf_features)


def test_fits_and_predicts_with_nan_features():
    """Regression test: RandomForestQuantileRegressor has no native NaN support (unlike the
    retired XGBoost models) and used to crash with ValueError: Input X contains NaN - several
    real feature groups (MEM/REC/EVO) are legitimately NaN before enough history exists."""
    train, val = _toy()
    rng = np.random.default_rng(1)
    train.loc[rng.random(len(train)) < 0.1, "x"] = np.nan
    val.loc[rng.random(len(val)) < 0.1, "spread_m"] = np.nan
    m = QRFErrorModel(FEATURES, params=PARAMS, seed=0).fit(train, val)
    p = m.calibrated_bust_probability(val)
    assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))


def test_importance_is_model_level_only():
    train, val = _toy()
    m = QRFErrorModel(FEATURES, params=PARAMS, seed=0).fit(train, val)
    imp = m.importance()
    assert set(imp) == set(FEATURES)
    assert abs(sum(imp.values()) - 1.0) < 1e-6
