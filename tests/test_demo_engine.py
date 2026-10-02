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
# research outputs of the SAME run the engine serves (v3 -> models/v3, data/interim/v3)
MODEL_DIR = REPO_ROOT / "models" / SERVED_RUN
INTERIM_DIR = REPO_ROOT / "data" / "interim" / SERVED_RUN


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


@pytest.mark.skipif(not (INTERIM_DIR / "predictions.parquet").exists(), reason="research predictions not present")
def test_live_inference_reproduces_frozen_pipeline(eng):
    table = pd.read_parquet(INTERIM_DIR / "table.parquet")
    preds = pd.read_parquet(INTERIM_DIR / "predictions.parquet").set_index("case_id")
    for c in eng.registry["cases"]:
        run = eng.run(c["case_id"], fresh=True)
        ids = run.rows["case_id"]
        p = preds.loc[ids]
        ref = table.set_index("case_id").loc[ids]
        for col in ("q10", "q25", "q50", "q75", "q90", "q95", "estimated_exceedance_probability",
                    "calibrated_bust_probability"):
            np.testing.assert_allclose(run.rows[col], p[col], atol=1e-5)
        np.testing.assert_allclose(run.rows["p_b0"], p["p_B0"], atol=1e-12)
        for f in ["an_n_within", "an_dist1", "an_bust_rate", "an_err_med", "an_n_eligible", "support_distance"]:
            np.testing.assert_allclose(run.rows[f].astype(float), ref[f].astype(float), rtol=1e-5, atol=1e-5)
        assert (run.rows["support_level"].to_numpy() == ref["support_level"].to_numpy()).all()


@pytest.mark.skipif(not (MODEL_DIR / "metadata.json").exists(), reason="v3 model files not present")
def test_exported_model_equals_trained_files(eng):
    from forecast_bust.models.qgb import V3PredictiveModel
    st = _state_with_memory(eng, eng.registry["cases"][0]["case_id"])
    pd.testing.assert_frame_equal(eng.model.predict(st), V3PredictiveModel(MODEL_DIR).predict(st))


def test_production_model_is_v3_and_needs_no_b2_or_sentinel(eng):
    assert eng.model.model_type == "quantile_gradient_boosting"
    assert not hasattr(eng, "b2") and not hasattr(eng, "sentinel")
    assert not list((DEMO_DIR / "model").glob("*booster*")), "B2/Sentinel boosters must not be in the v3 bundle"
    assert all("b2" not in f.lower() for f in eng.model.features)


def test_inference_depends_on_inputs(eng):
    """The probabilities are computed from the inputs, not looked up: perturbing the state changes them."""
    st = _state_with_memory(eng, eng.registry["cases"][0]["case_id"])
    p0 = eng.model.predict(st)["calibrated_bust_probability"].to_numpy()
    p1 = eng.model.predict(st.assign(spread_pct=1.0, spread_m=st["spread_m"] * 3))["calibrated_bust_probability"].to_numpy()
    assert np.abs(p1 - p0).max() > 0.01


def test_real_inference_quantiles_ordered_and_probabilities_valid(eng):
    for c in eng.registry["cases"]:
        rows = eng.run(c["case_id"]).rows
        q = rows[["q10", "q25", "q50", "q75", "q90", "q95"]].to_numpy()
        assert (np.diff(q, axis=1) >= 0).all()
        for col in ("estimated_exceedance_probability", "calibrated_bust_probability"):
            assert rows[col].between(0, 1).all()


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
    for k in ("actual_bust", "normalized_error", "failure_fingerprint", "era5"):
        assert k not in txt
    cell = j["regions"][0]["days"][0]
    assert abs(cell["reliability_confidence"] - (1 - cell["bust_probability"])) < 1e-12
    assert cell["model_type"] == "quantile_gradient_boosting"
    assert cell["expected_error"] == cell["q50"] and cell["upper_tail_error"] == cell["q95"]
    assert (cell["uncertainty_low"], cell["uncertainty_high"]) == (cell["q25"], cell["q75"])
    assert cell["bust_probability"] == cell["calibrated_bust_probability"]
    for k in ("b2_probability", "disagreement_pp", "p_sentinel"):
        assert k not in txt


def test_api_explain_and_reveal(client, eng):
    cid = eng.registry["cases"][0]["case_id"]
    client.post(f"/api/demo/cases/{cid}/run")
    rid = eng.run(cid).rows["region_id"].iloc[0]
    ex = client.get(f"/api/demo/cases/{cid}/regions/{rid}/explain?lead_day=4").json()
    assert ex["attribution"]["drivers"] and ex["analogue_summary"]["nearest"] and ex["interpretation"]
    assert all("cause" not in t.lower() or "not shown to cause" in t.lower() for t in ex["interpretation"])
    assert "actual_bust" not in json.dumps(ex)
    rv = client.post(f"/api/demo/cases/{cid}/regions/{rid}/reveal?lead_day=4").json()
    v = json.loads((DEMO_DIR / "cases" / cid / "verification.json").read_text())
    day = next(d for x in v["regions"] if x["region_id"] == rid for d in x["days"] if d["lead_day"] == 4)
    assert rv["verification"]["actual_bust"] == bool(day["bust"])
    assert rv["verification"]["normalized_error"] == day["normalized_error"]
    assert rv["verification"]["failure_fingerprint"]["signature"] == day["signature"]
    assert rv["forecast"]["upper_tail_error"] >= rv["forecast"]["expected_error"]
    assert rv["memory_update"]["verified_cases_after"] == rv["memory_update"]["verified_cases_before"] + 1


def test_api_errors(client):
    assert client.post("/api/demo/cases/2099010100/run").status_code == 404
    assert client.get("/api/demo/cases/2099010100/regions/R0000/explain?lead_day=11").status_code == 400
