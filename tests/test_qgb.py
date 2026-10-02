"""v3 quantile-gradient-boosting core: quantile shape/order/determinism, crossing measurement,
estimated exceedance probability, calibration, persistence, NaN inputs, feature provenance."""
import numpy as np
import pandas as pd
import pytest

from forecast_bust.models.qgb import (MODEL_TYPE, QCOLS, QUANTILES, QGBErrorModel, V3PredictiveModel,
                                      assert_quantiles_ordered, crossing_stats, exceedance_from_quantiles)


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
    return train, val


FEATURES = ["x", "spread_m"]
PARAMS = {"learning_rate": 0.1, "max_iter": 40, "max_leaf_nodes": 15, "min_samples_leaf": 20,
          "l2_regularization": 1.0}


@pytest.fixture(scope="module")
def fitted():
    train, val = _toy()
    return QGBErrorModel(FEATURES, params=PARAMS, seed=0).fit(train, val), train, val


def test_quantile_set_and_shape(fitted):
    m, _, val = fitted
    assert QUANTILES == [0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    assert QCOLS == ["q10", "q25", "q50", "q75", "q90", "q95"]
    assert set(m.estimators) == set(QCOLS)
    assert all(est.loss == "quantile" for est in m.estimators.values())
    assert [m.estimators[c].quantile for c in QCOLS] == QUANTILES
    q = m.predict_quantiles(val)
    assert all(q[c].shape == (len(val),) for c in QCOLS)


def test_quantiles_ordered_and_deterministic(fitted):
    m, train, val = fitted
    q1 = m.predict_quantiles(val)
    assert_quantiles_ordered(q1)
    q2 = QGBErrorModel(FEATURES, params=PARAMS, seed=0).fit(train, val).predict_quantiles(val)
    for c in QCOLS:
        np.testing.assert_array_equal(q1[c], q2[c])


def test_quantiles_track_their_level(fitted):
    """Empirical coverage on the training-like validation sample increases with the level."""
    m, _, val = fitted
    q = m.predict_quantiles(val)
    cov = [float((val["norm_error"].to_numpy() <= q[c]).mean()) for c in QCOLS]
    assert all(b >= a for a, b in zip(cov, cov[1:]))
    assert abs(cov[2] - 0.5) < 0.1


def test_crossing_is_measured_not_hidden():
    raw = np.array([[1, 2, 3, 4, 5, 6], [1, 3, 2, 4, 5, 6]], dtype=float)
    s = crossing_stats(raw)
    assert s["rows_crossing"] == 1 and s["by_pair"]["q25>q50"] == 0.5 and s["max_violation"] == 1.0
    with pytest.raises(ValueError):
        assert_quantiles_ordered({c: raw[1:, j] for j, c in enumerate(QCOLS)})


def test_exceedance_exact_at_knots_and_monotone():
    qv = np.array([[0.2, 0.4, 0.6, 0.8, 1.0, 1.2]])
    for lvl, t in zip(QUANTILES, qv[0]):
        np.testing.assert_allclose(exceedance_from_quantiles(qv, np.array([t])), 1 - lvl, atol=1e-9)
    # exponential tail: one q90->q95 spacing beyond q95 halves the survival 0.05 -> 0.025
    np.testing.assert_allclose(exceedance_from_quantiles(qv, np.array([1.4])), 0.025, atol=1e-9)
    thr = np.linspace(-1, 4, 400)
    p = exceedance_from_quantiles(np.repeat(qv, len(thr), 0), thr)
    assert np.all(np.diff(p) <= 1e-12) and np.all((p >= 0) & (p <= 1))
    assert np.all(p[thr > 1.2] > 0)  # tail stays rankable instead of a flat 0
    # tied quantiles do not produce NaN
    assert np.isfinite(exceedance_from_quantiles(np.ones((1, 6)), np.array([1.0]))).all()


def test_probabilities_in_unit_interval(fitted):
    m, _, val = fitted
    out = m.predict_all(val)
    for c in ("estimated_exceedance_probability", "calibrated_bust_probability"):
        assert np.isfinite(out[c]).all() and out[c].between(0, 1).all()
    assert len(m.calibrator.X_thresholds_) <= len(val) + 1  # fitted on validation rows only


def test_uncalibrated_model_raises():
    m = QGBErrorModel(FEATURES, params=PARAMS, seed=0)
    with pytest.raises(RuntimeError):
        m.calibrate(np.array([0.5]))


def test_persistence_round_trip(fitted, tmp_path):
    m, _, val = fitted
    m.save(tmp_path, {"features": FEATURES, "params": m.params, "seed": 0})
    for c in QCOLS:
        assert (tmp_path / f"qgb_{c}.joblib").is_file()
    loaded = V3PredictiveModel(tmp_path)
    assert loaded.model_type == MODEL_TYPE == "quantile_gradient_boosting"
    pd.testing.assert_frame_equal(loaded.predict(val), m.predict_all(val))


def test_fits_and_predicts_with_nan_features():
    train, val = _toy()
    rng = np.random.default_rng(1)
    train.loc[rng.random(len(train)) < 0.1, "x"] = np.nan
    val.loc[rng.random(len(val)) < 0.1, "spread_m"] = np.nan
    out = QGBErrorModel(FEATURES, params=PARAMS, seed=0).fit(train, val).predict_all(val)
    assert np.isfinite(out.to_numpy()).all()


def test_production_feature_set_has_no_b2_forbidden_or_future_inputs():
    from forecast_bust.features.build import FORBIDDEN_INPUTS
    from forecast_bust.pipeline_qgb import qgb_features
    feats = qgb_features()
    assert not any(f.startswith(x) for f in feats for x in FORBIDDEN_INPUTS)
    assert all("b2" not in f.lower() and not f.startswith("p_") for f in feats)
    for leak in ("norm_error", "error_m", "bust", "hidden_bust", "q_primary", "sig_class", "split"):
        assert leak not in feats
