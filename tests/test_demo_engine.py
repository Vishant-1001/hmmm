"""Live-inference demo: the engine must reproduce the frozen research pipeline and keep verification blind."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from forecast_bust.config import REPO_ROOT, SERVED_RUN
from forecast_bust.demo.build import SERVED_DEMO_DIR as DEMO_DIR
from forecast_bust.features.build import FORBIDDEN_INPUTS

MEMORY_PATH = DEMO_DIR / "memory.parquet"
# research outputs of the SAME run the engine serves (v2 -> models/v2, data/interim/v2)
MODEL_DIR = REPO_ROOT / "models" / SERVED_RUN
INTERIM_DIR = REPO_ROOT / "data" / "interim" / SERVED_RUN
TABLE = INTERIM_DIR / "table.parquet"


def _state_with_memory(eng, case_id):
    """Stored forecast state + MEM features from the engine's causal memory lookup (as in Engine.run)."""
    st = pd.read_parquet(DEMO_DIR / "cases" / case_id / "forecast_state.parquet")
    return eng.memory_lookup(st.sort_values(["region_id", "lead_day"]).reset_index(drop=True))[0]

pytestmark = pytest.mark.skipif(not (DEMO_DIR / "registry.json").exists() or not MEMORY_PATH.exists(),
                                reason="demo bundle not built (python -m forecast_bust.demo.build)")


@pytest.fixture(scope="module")
def eng():
    from forecast_bust.demo.engine import Engine
    return Engine()


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from forecast_bust.api.app import app
    return TestClient(app)


def test_registry_is_small_and_real(eng):
    cases = eng.registry["cases"]
    assert 3 <= len(cases) <= 5
    for c in cases:
        st = pd.read_parquet(DEMO_DIR / "cases" / c["case_id"] / "forecast_state.parquet")
        assert len(st) == c["n_regions"] * 10
        assert not [col for col in st.columns if col.startswith(FORBIDDEN_INPUTS)], "verification leaked into inputs"


@pytest.mark.skipif(not (INTERIM_DIR / "predictions.parquet").exists() or not TABLE.exists(),
                    reason="research predictions not present")
def test_live_inference_reproduces_frozen_benchmark(eng):
    """Served B2 probabilities == the benchmark's p_B2 for every region-day of every demo case."""
    table = pd.read_parquet(TABLE, columns=["case_id", "an_n_within", "an_dist1", "an_bust_rate", "an_err_med",
                                           "an_n_eligible", "support_distance", "support_level"]).set_index("case_id")
    preds = pd.read_parquet(INTERIM_DIR / "predictions.parquet", columns=["case_id", "p_B2", "p_B0"]).set_index("case_id")
    for c in eng.registry["cases"]:
        run = eng.run(c["case_id"], fresh=True)
        ids = run.rows["case_id"]
        np.testing.assert_allclose(run.rows["calibrated_bust_probability"], preds.loc[ids, "p_B2"], atol=1e-6)
        np.testing.assert_allclose(run.rows["p_b0"], preds.loc[ids, "p_B0"], atol=1e-9)
        ref = table.loc[ids]
        for f in ["an_n_within", "an_dist1", "an_bust_rate", "an_err_med", "an_n_eligible", "support_distance"]:
            np.testing.assert_allclose(run.rows[f].astype(float), ref[f].astype(float), rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not (MODEL_DIR / "models.joblib").exists(), reason="v2 model files not present")
def test_exported_model_equals_trained_files(eng):
    import joblib
    st = pd.read_parquet(DEMO_DIR / "cases" / eng.registry["cases"][0]["case_id"] / "forecast_state.parquet")
    np.testing.assert_allclose(eng.model.predict(st)["calibrated_bust_probability"],
                               joblib.load(MODEL_DIR / "models.joblib")["B2"].predict(st), atol=1e-6)


def test_served_model_is_b2_only(eng):
    from forecast_bust.b2_provider import B2_FEATURES
    assert SERVED_RUN == "v2"
    assert eng.model.model_type == "b2_spread_xgboost" and eng.model.card["model_id"] == "b2_spread_calibrated"
    assert eng.model.features == B2_FEATURES
    for f in eng.model.features:
        assert not f.startswith(FORBIDDEN_INPUTS), f
    card = eng.model.card
    assert card["calibration_version"].startswith("isotonic_validation") and "VALIDATION" in card["alert_threshold_definition"]
    met = json.loads((REPO_ROOT / "artifacts" / "v2" / "metrics.json").read_text())["models"]["B2"]
    assert card["alert_threshold"] == met["operating_point"]["threshold"] and met["operating_point"]["chosen_on"] == "validation"
    assert card["test_metrics_2022"]["auprc"] == met["auprc"]


def test_inference_depends_on_inputs(eng):
    """The probabilities are computed from the inputs, not looked up: perturbing the state changes them."""
    st = _state_with_memory(eng, eng.registry["cases"][0]["case_id"])
    p0 = eng.model.predict(st)["raw_probability"].to_numpy()
    p1 = eng.model.predict(st.assign(spread_m=st["spread_m"] * 3, spread_thr_ratio=st["spread_thr_ratio"] * 3))["raw_probability"].to_numpy()
    assert np.abs(p1 - p0).max() > 1e-3


def test_real_inference_probabilities_valid_and_deterministic(eng):
    for c in eng.registry["cases"]:
        a = eng.run(c["case_id"], fresh=True).rows["calibrated_bust_probability"].to_numpy()
        b = eng.run(c["case_id"], fresh=True).rows["calibrated_bust_probability"].to_numpy()
        assert ((a >= 0) & (a <= 1)).all() and np.array_equal(a, b)


def test_memory_is_causal(eng):
    run = eng.run(eng.registry["cases"][0]["case_id"])
    init = run.rows["init_time"].iloc[0]
    for cand, _ in run.neighbours[:50]:
        assert (eng.mem.iloc[cand]["valid_time"] <= init).all()


def test_api_run_is_blind_and_complete(client, eng):
    cid = eng.registry["cases"][0]["case_id"]
    r = client.post(f"/api/demo/cases/{cid}/run")
    assert r.status_code == 200
    j = r.json()
    assert len(j["regions"]) == 64 and all(len(x["days"]) == 10 for x in j["regions"])
    assert j["timings_ms"]["model_inference"] > 0
    txt = json.dumps(j)
    # no verification DATA before reveal (provenance labels such as "ERA5 reanalysis" are allowed)
    for k in ("actual_bust", "normalized_error", "failure_fingerprint", "era5_z500", "era5_rank", "error_m", "\"bust\""):
        assert k not in txt, k
    cell = j["regions"][0]["days"][0]
    assert cell["model_type"] == "b2_spread_xgboost" and cell["model_id"] == j["model"]["model_id"] == "b2_spread_calibrated"
    assert cell["bust_probability"] == cell["calibrated_bust_probability"]
    assert cell["risk_level"] in ("ALERT", "ABOVE CLIMATOLOGY", "AT OR BELOW CLIMATOLOGY")
    assert cell["alert"] == (cell["bust_probability"] >= j["alert_threshold"])
    for k in ("confidence", "p_sentinel", "upper_tail_error", "q95", "bma_", "local_acc", "pattern_", "acc_q10", "V4"):
        assert k not in txt, k


def test_api_explain_and_reveal(client, eng):
    cid = eng.registry["cases"][0]["case_id"]
    client.post(f"/api/demo/cases/{cid}/run")
    rid = eng.run(cid).rows["region_id"].iloc[0]
    ex = client.get(f"/api/demo/cases/{cid}/regions/{rid}/explain?lead_day=4").json()
    assert ex["attribution"]["drivers"] and ex["analogue_summary"]["nearest"] and ex["interpretation"]
    assert {d["feature"] for d in ex["attribution"]["drivers"]} <= set(eng.model.features)
    assert "actual_bust" not in json.dumps(ex) and "normalized_error\": " not in json.dumps({k: v for k, v in ex.items()
                                                                                        if k != "analogue_summary"})
    rv = client.post(f"/api/demo/cases/{cid}/regions/{rid}/reveal?lead_day=4").json()
    v = json.loads((DEMO_DIR / "cases" / cid / "verification.json").read_text())
    day = next(d for x in v["regions"] if x["region_id"] == rid for d in x["days"] if d["lead_day"] == 4)
    assert rv["verification"]["normalized_error"] == day["normalized_error"]
    assert rv["verification"]["failure_fingerprint"]["signature"] == day["signature"]
    assert rv["verification"]["actual_bust"] == bool(day["bust"]) and rv["post_verification"] is True
    assert rv["memory_update"]["verified_cases_after"] == rv["memory_update"]["verified_cases_before"] + 1


def test_api_errors(client):
    assert client.post("/api/demo/cases/2099010100/run").status_code == 404
    assert client.get("/api/demo/cases/2099010100/regions/R0000/explain?lead_day=11").status_code == 400
