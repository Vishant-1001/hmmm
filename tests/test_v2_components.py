"""v2 components: DYN kinematics, residual (B2-margin) learner, disagreement metric."""
import numpy as np
import pandas as pd

from forecast_bust.evaluation.metrics import disagreement_behaviour
from forecast_bust.features.dynamics import A_EARTH, kinematics


def _grid():
    lat = np.arange(-30.9375, 65, 5.625)
    lon = np.arange(22.5, 147, 5.625)
    return lat, lon


def test_solid_body_rotation_vorticity():
    # u = U cos(lat), v = 0  ->  vorticity = 2 U sin(lat) / a, divergence = 0
    lat, lon = _grid()
    U = 20.0
    u = np.broadcast_to(U * np.cos(np.deg2rad(lat))[None, :], (len(lon), len(lat))).copy()
    v = np.zeros_like(u)
    vort, div = kinematics(u, v, lat, lon)
    inner = (slice(1, -1), slice(1, -1))
    expect = 2 * U * np.sin(np.deg2rad(lat)) / A_EARTH * 1e5
    np.testing.assert_allclose(vort[inner], np.broadcast_to(expect[None, 1:-1], vort[inner].shape), atol=0.02)
    np.testing.assert_allclose(div[inner], 0, atol=1e-9)
    assert np.isnan(vort[0]).all() and np.isnan(vort[:, -1]).all()


def test_pure_divergence():
    # v = V (uniform) on a sphere gives divergence -V tan(lat)/a and zero zonal gradients
    lat, lon = _grid()
    v = np.full((len(lon), len(lat)), 5.0)
    u = np.zeros_like(v)
    _, div = kinematics(u, v, lat, lon)
    expect = -5.0 * np.tan(np.deg2rad(lat)) / A_EARTH * 1e5
    np.testing.assert_allclose(div[1:-1, 1:-1], np.broadcast_to(expect[None, 1:-1], div[1:-1, 1:-1].shape), atol=0.02)


def _toy(n_per_year=3000, years=(2018, 2019, 2020), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for y in years:
        s = rng.gamma(2, 20, n_per_year)
        x = rng.normal(size=n_per_year)
        p = 1 / (1 + np.exp(-(-3 + 0.04 * s + 0.8 * x)))
        rows.append(pd.DataFrame({"init_time": pd.Timestamp(f"{y}-06-01"), "spread_m": s, "x": x,
                                  "bust": (rng.random(n_per_year) < p).astype(int)}))
    df = pd.concat(rows, ignore_index=True)
    df["case_id"] = [f"c{i}" for i in range(len(df))]
    return df


def test_residual_learner_crossfit_and_gain():
    from forecast_bust.models.sentinel import B2Margin, CalibratedGBM
    from forecast_bust.evaluation.metrics import auprc

    df = _toy()
    tr = df[pd.to_datetime(df["init_time"]).dt.year < 2020]
    va = df[pd.to_datetime(df["init_time"]).dt.year == 2020]
    b2 = CalibratedGBM(["spread_m"], "B2").fit(tr, va)
    base = B2Margin(b2, tr)
    # every TRAIN row gets an out-of-year (cross-fitted) margin; other rows use the full B2 model
    assert set(base.crossfit.index) == set(tr["case_id"])
    assert base.crossfit_years == [2018, 2019]
    full_va = np.log(b2.predict_raw(va) / (1 - b2.predict_raw(va)))
    np.testing.assert_allclose(base.margin(va), full_va, rtol=1e-5)
    assert not np.allclose(base.margin(tr), np.log(b2.predict_raw(tr) / (1 - b2.predict_raw(tr))))
    # the residual model learns the extra signal x on top of B2
    res = CalibratedGBM(["spread_m", "x"], "FULL", base=base).fit(tr, va)
    assert res.learner == "residual_b2"
    assert auprc(va["bust"].values, res.predict_raw(va)) > auprc(va["bust"].values, b2.predict_raw(va))


def test_disagreement_behaviour_bins():
    y = np.array([1, 1, 0, 0, 0, 0])
    pm = np.array([0.6, 0.5, 0.1, 0.1, 0.1, 0.1])
    pb = np.array([0.1, 0.1, 0.1, 0.1, 0.3, 0.3])
    out = disagreement_behaviour(y, pm, pb)
    b = {x["bin"]: x for x in out["bins"]}
    assert b["model_higher"]["n"] == 2 and b["model_higher"]["observed_bust_rate"] == 1.0
    assert b["agree"]["n"] == 2 and b["model_lower"]["n"] == 2
    assert b["model_higher"]["brier_model"] < b["model_higher"]["brier_base"]
    assert abs(out["share_abs_gt_5pp"] - 4 / 6) < 1e-12


def test_sentinel_is_frozen_standard_learner():
    """Frozen spec §23: the Sentinel is one shared GBT; residual/stacked learners are experimental only."""
    import yaml
    from forecast_bust.config import CONFIG_DIR
    for name in ("model_v2", "model_v2a"):
        v2 = yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text())["v2"]
        assert v2["learners"] == ["standard"]
        assert "standard" not in v2.get("experimental_learners", [])


def test_tendency_fields_are_within_forecast_rates():
    """|d/dt| along the lead axis only: a field growing 10 m per day has tendency 10 at every lead."""
    import xarray as xr
    from forecast_bust.features.dynamics import tendency_fields
    lat, lon = _grid()
    leads = np.arange(24, 241, 24)
    z = np.zeros((2, len(leads), 3, len(lon), len(lat)), np.float32)
    z[:, :, 0] = 5500 + 10 * np.arange(len(leads))[None, :, None, None]
    ds = xr.Dataset({"ens_mean": (("init", "lead", "level", "longitude", "latitude"), z)},
                    coords={"lead": leads, "level": [500, 700, 850], "longitude": lon, "latitude": lat})
    mslp = np.full((2, len(leads), len(lon), len(lat)), 1010.0)
    f = tendency_fields(ds, mslp, np.zeros_like(mslp))
    assert np.allclose(f["z500_tend"], 10.0)
    assert np.allclose(f["mslp_tend"], 0.0)
    # changing the first init does not change the second (no cross-cycle information)
    z2 = z.copy()
    z2[0] += 100 * np.arange(len(leads))[:, None, None, None]
    f2 = tendency_fields(ds.assign(ens_mean=(ds["ens_mean"].dims, z2)), mslp, np.zeros_like(mslp))
    assert np.allclose(f2["z500_tend"][1], f["z500_tend"][1])


def test_spread_threshold_ratio_uses_no_verification():
    from forecast_bust.labels.build import spread_threshold_ratio
    df = pd.DataFrame({"spread_m": [10.0, 20.0], "q_primary": [1.0, 2.0], "scale_m": [50.0, 50.0],
                       "error_m": [0.0, 999.0], "norm_error": [0.0, 99.0], "bust": [0, 1]})
    r = spread_threshold_ratio(df)
    assert np.allclose(r, np.array([10, 20]) * np.sqrt(1 + 1 / 50) / np.array([50, 100]), rtol=1e-6)
    df2 = df.assign(error_m=[5.0, 1.0], norm_error=[1.0, 0.0], bust=[1, 0])
    assert np.array_equal(spread_threshold_ratio(df2), r)


def test_block_bootstrap_metrics_identical_models_have_zero_difference():
    from forecast_bust.evaluation.metrics import block_bootstrap_metrics
    rng = np.random.default_rng(0)
    n = 4000
    df = pd.DataFrame({"init_time": pd.date_range("2021-01-01", periods=n // 10, freq="12h").repeat(10),
                       "bust": rng.random(n) < 0.1})
    df["hidden_bust"] = df["bust"] & (rng.random(n) < 0.2)
    p = np.clip(0.1 + 0.2 * df["bust"] + rng.normal(0, 0.1, n), 0, 1)
    res = block_bootstrap_metrics(df, {"B2": p, "FULL": p.copy()}, {"B2": 0.2, "FULL": 0.2}, n=30)
    d = res["difference_vs_reference"]["FULL"]["auprc"]
    assert d["point"] == 0 and d["ci95"] == [0.0, 0.0]
    lo, hi = res["models"]["B2"]["auprc"]["ci95"]
    assert lo <= res["models"]["B2"]["auprc"]["point"] <= hi


def test_xgb_params_override_keeps_defaults():
    from forecast_bust.models.sentinel import xgb_params
    p = xgb_params({"max_depth": 7})
    assert p["max_depth"] == 7 and "early_stopping_rounds" in p and "eval_metric" in p
