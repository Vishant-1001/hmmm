"""BMA experiment: fitting, exact mixture CDF/quantiles/exceedance, causality, persistence."""
import numpy as np
import pandas as pd
import pytest

from forecast_bust.models.bma import (QCOLS, BMAModel, error_cdf, error_moments, error_quantiles, exceedance,
                                      fit_exchangeable, fit_free_weights)
from forecast_bust.pipeline_bma import rolling_fit

RNG = np.random.default_rng(0)


def _synthetic(n=400, K=50, a=5.0, b=0.8, sigma=20.0):
    truth = RNG.normal(0, 60, n)
    f = truth[:, None] + RNG.normal(0, 30, (n, K))           # exchangeable members
    pick = RNG.integers(0, K, n)
    y = a + b * f[np.arange(n), pick] + RNG.normal(0, sigma, n)  # generated from the BMA model itself
    return f, y


def test_fit_recovers_parameters():
    f, y = _synthetic(n=4000)
    a, b, s, it = fit_exchangeable(f[None], y[None], np.ones((1, len(y)), bool))
    # Raftery et al. (2005) estimate (a, b) by least squares of y on the member forecasts; for y generated
    # from one random member that slope is b * var(signal) / var(member) = 0.8 * 3600 / 4500 = 0.64
    assert abs(b[0] - 0.64) < 0.03 and abs(a[0] - 5) < 3 and np.isfinite(s[0]) and s[0] > 0 and it[0] > 0


def test_free_weights_nonnegative_and_sum_to_one():
    f, y = _synthetic(n=300, K=10)
    fw = fit_free_weights(f, y)
    assert (fw["weights"] >= 0).all() and abs(fw["weights"].sum() - 1) < 1e-9


def _mix(n=200, K=50):
    mu = RNG.normal(0, 40, (n, K))
    return mu, RNG.uniform(5, 40, n), RNG.normal(0, 30, n), RNG.uniform(20, 80, n)


def test_cdf_monotone_finite_quantiles_ordered():
    mu, sg, m, s = _mix()
    ys = np.linspace(0, 6, 60)
    cdf = np.stack([error_cdf(np.full(len(m), y), mu, sg, m, s) for y in ys], 1)
    assert np.isfinite(cdf).all() and (np.diff(cdf, axis=1) >= -1e-12).all()
    assert (cdf[:, 0] >= 0).all() and (cdf[:, -1] <= 1 + 1e-12).all()
    q = error_quantiles(mu, sg, m, s)
    assert np.isfinite(q).all() and (np.diff(q, axis=1) >= 0).all()
    np.testing.assert_allclose(error_cdf(q[:, 2], mu, sg, m, s), 0.5, atol=1e-6)
    mean, std = error_moments(mu, sg, m, s)
    assert np.isfinite(mean).all() and (std >= 0).all()


def test_exceedance_bounds_monotone_and_matches_monte_carlo():
    mu, sg, m, s = _mix(n=5)
    p_lo, p_hi = exceedance(np.full(5, 0.5), mu, sg, m, s), exceedance(np.full(5, 1.5), mu, sg, m, s)
    assert ((p_lo >= 0) & (p_lo <= 1)).all() and (p_hi <= p_lo + 1e-12).all()
    k = RNG.integers(0, mu.shape[1], (5, 200_000))
    z = np.take_along_axis(mu, k, 1) + sg[:, None] * RNG.normal(size=k.shape)
    mc = (np.abs(m[:, None] - z) / s[:, None] > 0.5).mean(1)
    np.testing.assert_allclose(p_lo, mc, atol=0.005)


def test_rolling_fit_is_causal():
    I, L, R, C, K = 120, 2, 1, 2, 50
    inits = np.datetime64("2020-01-01T00", "ns") + np.arange(I) * np.timedelta64(1, "D")
    members = RNG.normal(0, 50, (I, L, R, C, K)).astype(np.float32)
    obs = (members.mean(-1) + RNG.normal(0, 10, (I, L, R, C))).astype(np.float32)
    leads = np.array([24, 48])
    em = {"max_iter": 200, "tol": 1e-8}
    p = rolling_fit(members, obs, inits, leads, 30, 10, em)
    assert (p["last_verified"] <= p["fit_day"]).all() and (p["n_cases"] >= 10).all()
    day = np.datetime64("2020-03-01", "D")
    future = obs.copy()
    future[(inits + np.timedelta64(24, "h")) > day.astype("datetime64[ns]")] += 1000.0  # change only the future
    p2 = rolling_fit(members, future, inits, leads, 30, 10, em, days=np.array([day]))
    p1 = p[p["fit_day"] == day.astype("datetime64[ns]")].reset_index(drop=True)
    pd.testing.assert_frame_equal(p1, p2, check_dtype=False)


def _rows_and_model():
    rows = pd.DataFrame({"init_time": pd.to_datetime(["2021-01-05 00:00", "2021-01-05 12:00", "2021-01-09 00:00"]),
                         "region_id": "R0000", "lead_day": 1, "ens_mean_anom": [10.0, -5.0, 3.0],
                         "scale_m": [50.0, 50.0, 50.0], "q_primary": [0.6, 0.6, 0.6]})
    params = pd.DataFrame({"fit_day": pd.to_datetime(["2021-01-05"]), "region_id": "R0000", "lead_day": 1,
                           "a": [1.0], "b": [0.9], "sigma": [15.0], "n_cases": [60]})
    return rows, RNG.normal(0, 30, (3, 50)), BMAModel(params, {"window_days": 60})


def test_rows_without_parameters_are_missing_not_imputed():
    rows, mem, model = _rows_and_model()
    out = model.predict(rows, mem)
    assert out.iloc[:2].notna().all().all() and out.iloc[2].isna().all()
    assert out.loc[:1, "bma_exceedance_probability"].between(0, 1).all()
    assert (out.loc[:1, QCOLS].diff(axis=1).iloc[:, 1:] >= 0).all().all()


def test_save_load_round_trip(tmp_path):
    from sklearn.isotonic import IsotonicRegression
    rows, mem, model = _rows_and_model()
    model.calibrator = IsotonicRegression(out_of_bounds="clip").fit([0.0, 0.2, 0.5, 1.0], [0, 0, 1, 1])
    model.save(tmp_path, {"note": "test"})
    pd.testing.assert_frame_equal(BMAModel.load(tmp_path).predict(rows, mem), model.predict(rows, mem))
