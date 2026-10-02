"""V4 pattern-aware bust: local ACC, TRAIN-only thresholds, AND logic, leakage."""
import numpy as np
import pandas as pd

from forecast_bust.labels.pattern import fit_acc_thresholds, pattern_labels, weighted_acc


def test_weighted_acc_matches_pearson_and_bounds():
    rng = np.random.default_rng(0)
    f, o = rng.normal(size=(50, 3, 3)), rng.normal(size=(50, 3, 3))
    r = weighted_acc(f, o, np.ones((3, 3)))
    ref = [np.corrcoef(a.ravel(), b.ravel())[0, 1] for a, b in zip(f, o)]
    np.testing.assert_allclose(r, ref, atol=1e-12)
    assert np.all(np.abs(r) <= 1 + 1e-12)
    np.testing.assert_allclose(weighted_acc(f, 3 * f + 7, np.ones((3, 3))), 1.0)   # centred: offset-invariant
    np.testing.assert_allclose(weighted_acc(f, -f, np.ones((3, 3))), -1.0)
    assert np.isnan(weighted_acc(np.ones((1, 3, 3)), o[:1], np.ones((3, 3))))[0]       # flat field -> invalid


def test_weighted_acc_uses_latitude_weights():
    rng = np.random.default_rng(1)
    f, o = rng.normal(size=(3, 3)), rng.normal(size=(3, 3))
    w = np.array([[0.2, 1.0, 5.0]])
    fw, ow = np.repeat(f, [1, 1, 1], 0), o
    assert not np.isclose(weighted_acc(fw, ow, w), weighted_acc(fw, ow, np.ones((3, 3))))


def _cases():
    rng = np.random.default_rng(2)
    n = 4000
    d = pd.DataFrame({"case_id": [f"c{i}" for i in range(n)], "init_time": pd.Timestamp("2018-01-01"),
                      "region_id": "R0000", "lead_day": 1, "season": "JF",
                      "split": np.where(np.arange(n) < 2000, "train", "test"),
                      "bust": (rng.random(n) < 0.1).astype(np.int8), "low_spread": (rng.random(n) < 0.25).astype(np.int8)})
    d["init_time"] = d["init_time"] + pd.to_timedelta(np.arange(n), "h")
    acc = d[["init_time", "region_id", "lead_day"]].assign(local_acc=rng.uniform(-1, 1, n).astype(np.float32))
    return d, acc


def test_acc_threshold_train_only_and_and_logic():
    d, acc = _cases()
    labels, th = pattern_labels(d, acc, 0.10)
    tr = acc.iloc[:2000]["local_acc"]
    np.testing.assert_allclose(th["acc_q10"].iloc[0], tr.quantile(0.10))
    acc2 = acc.copy()
    acc2.loc[2000:, "local_acc"] = -1.0                                 # change only non-train rows
    _, th2 = pattern_labels(d, acc2, 0.10)
    pd.testing.assert_frame_equal(th, th2)
    m = labels.merge(d[["case_id", "bust"]], on="case_id")
    expect = (m["bust"] == 1) & (m["local_acc"] < m["acc_q10"])
    assert (m["pattern_bust"].astype(bool) == expect).all()
    assert (m["pattern_bust"] <= m["magnitude_failure"]).all() and (m["pattern_bust"] <= m["pattern_failure"]).all()
    labels2, _ = pattern_labels(d, acc, 0.10)
    pd.testing.assert_frame_equal(labels, labels2)                      # deterministic


def test_invalid_acc_is_missing_not_fabricated():
    d, acc = _cases()
    acc.loc[5, "local_acc"] = np.nan
    labels, _ = pattern_labels(d, acc, 0.10)
    assert np.isnan(labels.loc[5, "pattern_bust"]) and np.isnan(labels.loc[5, "pattern_failure"])


def test_local_acc_and_verification_never_model_inputs():
    from forecast_bust.features.build import FORBIDDEN_INPUTS
    from forecast_bust.labels.pattern import PATTERN_COLS
    from forecast_bust.models.sentinel import FEATURE_SETS
    feats = FEATURE_SETS["ALL"]
    for c in PATTERN_COLS + ["norm_error", "error_m", "q_primary", "era5_z500", "sig_class"]:
        assert c not in feats
    assert not any(f.startswith(FORBIDDEN_INPUTS) for f in feats)
    assert not any("acc" in f.lower() for f in feats)
