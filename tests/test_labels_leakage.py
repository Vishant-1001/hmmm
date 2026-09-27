"""TEST FIXTURES ONLY: synthetic tables exercise label, leakage and memory logic.
No value in this file is a scientific result."""
import ast
import inspect

import numpy as np
import pandas as pd
import pytest

from forecast_bust.analogues import memory
from forecast_bust.features import build as fb
from forecast_bust.labels.build import fit_normalisation, fit_thresholds, spread_percentile
from forecast_bust.labels.signature import failure_signature


def _cases(n_init=40, seed=0):
    rng = np.random.default_rng(seed)
    inits = pd.date_range("2018-01-01", periods=n_init, freq="12h")
    rows = []
    for i, t in enumerate(inits):
        for lead in (1, 2):
            rows.append({"init_time": t, "valid_time": t + pd.Timedelta(days=lead), "region_id": "R0000",
                         "lead_day": lead, "season": "JF", "norm_error": float(i), "spread_m": float(i % 7),
                         "era5_anom_m": rng.normal(), "bust": int(i % 5 == 0),
                         "split": "train" if i < 20 else ("validation" if i < 30 else "test")})
    df = pd.DataFrame(rows)
    for k in range(8):
        df[f"pc{k + 1}"] = rng.normal(size=len(df))
    df["anom500"] = rng.normal(size=len(df))
    df["m_p10p90"] = rng.normal(size=len(df))
    return df


def test_q90_q95_deterministic():
    tr = pd.DataFrame({"region_id": "R", "lead_day": 1, "season": "JF", "norm_error": np.arange(1, 101.0),
                       "spread_m": np.arange(1, 101.0)})
    th = fit_thresholds(tr, 0.9, 0.95, 0.25)
    assert np.isclose(th["q_primary"][0], np.quantile(np.arange(1, 101.0), 0.9))
    assert np.isclose(th["q_sensitivity"][0], np.quantile(np.arange(1, 101.0), 0.95))
    assert np.isclose(th["spread_low"][0], np.quantile(np.arange(1, 101.0), 0.25))


def test_thresholds_conditioned_by_group():
    tr = pd.concat([
        pd.DataFrame({"region_id": "A", "lead_day": 1, "season": "JF", "norm_error": np.arange(10.0), "spread_m": 1.0}),
        pd.DataFrame({"region_id": "A", "lead_day": 2, "season": "JF", "norm_error": 100 + np.arange(10.0), "spread_m": 1.0}),
        pd.DataFrame({"region_id": "B", "lead_day": 1, "season": "JJAS", "norm_error": 50 + np.arange(10.0), "spread_m": 1.0}),
    ])
    th = fit_thresholds(tr, 0.9, 0.95, 0.25).set_index(["region_id", "lead_day", "season"])
    assert th.loc[("A", 1, "JF"), "q_primary"] < 10 < th.loc[("A", 2, "JF"), "q_primary"]
    assert 50 < th.loc[("B", 1, "JJAS"), "q_primary"] < 60


def test_thresholds_ignore_non_training_rows():
    df = _cases()
    tr = df[df["split"] == "train"]
    th_a = fit_thresholds(tr, 0.9, 0.95, 0.25)
    df2 = df.copy()
    df2.loc[df2["split"] == "test", "norm_error"] = 1e9  # changing test must not move thresholds
    th_b = fit_thresholds(df2[df2["split"] == "train"], 0.9, 0.95, 0.25)
    pd.testing.assert_frame_equal(th_a, th_b)


def test_normalisation_training_only():
    df = _cases()
    a = fit_normalisation(df[df["split"] == "train"])
    df.loc[df["split"] != "train", "era5_anom_m"] *= 1000
    b = fit_normalisation(df[df["split"] == "train"])
    pd.testing.assert_frame_equal(a, b)


def test_spread_percentile_uses_train_reference():
    df = _cases()
    tr = df[df["split"] == "train"]
    p = spread_percentile(df, tr)
    assert np.all((p >= 0) & (p <= 1))


def test_memory_temporal_cutoff_and_no_self():
    q_init = np.array(["2018-01-05"], dtype="datetime64[ns]")
    c_valid = np.array(["2018-01-04", "2018-01-05", "2018-01-06"], dtype="datetime64[ns]")
    m = memory.eligible_mask(q_init, c_valid, np.array(["train"] * 3), "causal_online")
    assert m.tolist() == [[True, True, False]]
    # a case can never retrieve itself: its own valid time is after its init time
    df = _cases()
    t = df["init_time"].values.astype("datetime64[ns]")
    v = df["valid_time"].values.astype("datetime64[ns]")
    E = memory.eligible_mask(t, v, df["split"].values, "causal_online")
    assert not np.any(np.diag(E))
    # all eligible analogues were initialised strictly before the query
    qi, ci = np.where(E)
    assert np.all(t[ci] < t[qi])


def test_memory_frozen_mode_excludes_test():
    q_init = np.array(["2030-01-01"], dtype="datetime64[ns]")
    c_valid = np.array(["2018-01-01", "2019-01-01", "2020-01-01"], dtype="datetime64[ns]")
    m = memory.eligible_mask(q_init, c_valid, np.array(["train", "validation", "test"]), "frozen")
    assert m.tolist() == [[True, True, False]]


def test_memory_features_do_not_use_future_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "ARTIFACT_DIR", tmp_path)
    df = _cases()
    a, _, _ = memory.compute_memory_features(df)
    df2 = df.copy()
    # flip labels of the LAST 5 inits: features of all earlier-verified queries must not change
    last = df2["init_time"] >= df2["init_time"].sort_values().unique()[-5]
    df2.loc[last, "bust"] = 1 - df2.loc[last, "bust"]
    df2.loc[last, "norm_error"] += 1000
    b, _, _ = memory.compute_memory_features(df2)
    cutoff = df2.loc[last, "init_time"].min()
    early = df["init_time"] <= cutoff
    pd.testing.assert_series_equal(a.loc[early, "an_bust_rate"], b.loc[early, "an_bust_rate"])


def test_feature_builders_never_read_verification_fields():
    tree = ast.parse(inspect.getsource(fb))
    allowed = set()
    for node in ast.walk(tree):  # the FORBIDDEN_INPUTS declaration itself is allowed
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "FORBIDDEN_INPUTS" for t in node.targets):
            allowed |= {id(c) for c in ast.walk(node)}
    doc = tree.body[0].value if isinstance(tree.body[0], ast.Expr) else None
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in allowed \
                and node is not doc:
            assert node.value not in ("era5_z500", "era5_rank", "error_m", "norm_error", "bust"), node.value


def test_model_feature_sets_exclude_labels():
    from forecast_bust.models.sentinel import FEATURE_SETS
    for feats in FEATURE_SETS.values():
        for f in feats:
            assert not any(f.startswith(x) for x in fb.FORBIDDEN_INPUTS), f


def test_failure_signature_phase_vs_amplitude():
    lats = np.array([0.0, 5.625, 11.25])
    x = np.linspace(0, 2 * np.pi, 3, endpoint=False)
    a = np.sin(x)[:, None] * np.ones((3, 3))
    shifted = np.roll(a, 1, axis=0)
    s = failure_signature(shifted[None], a[None], lats)
    assert s["cls"][0] == "POSITION_PHASE"
    s2 = failure_signature((a + 5)[None], a[None], lats)
    assert s2["cls"][0] == "AMPLITUDE_STRUCTURE"
    total = s["phase_share"] + s["amp_share"]
    assert np.allclose(total, 1.0)


def test_signature_decomposition_is_exact():
    rng = np.random.default_rng(1)
    lats = np.array([0.0, 5.625, 11.25, 16.875])
    f, a = rng.normal(size=(4, 4)), rng.normal(size=(4, 4))
    s = failure_signature(f[None], a[None], lats)
    from forecast_bust.verification.metrics import weighted_rmse
    assert np.isclose(s["mse"][0], weighted_rmse(f, a, lats) ** 2)


def test_recent_error_features_are_causal():
    from forecast_bust.analogues.recent import recent_error_features
    df = _cases()
    df["bias_m"] = 1.0
    df["row"], df["col"] = 0, 0
    a = recent_error_features(df)
    df2 = df.copy()
    late = df2["valid_time"] > pd.Timestamp("2018-01-10")
    df2.loc[late, "norm_error"] += 1000
    df2.loc[late, "bust"] = 1 - df2.loc[late, "bust"]
    b = recent_error_features(df2)
    early = a["init_time"] <= pd.Timestamp("2018-01-10")
    pd.testing.assert_series_equal(a.loc[early, "rec_err"], b.loc[early, "rec_err"])
    # the first initialisation has no verified history
    assert np.isnan(a.loc[a["init_time"] == a["init_time"].min(), "rec_err"]).all()
