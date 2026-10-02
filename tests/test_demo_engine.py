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
# research outputs of the SAME run the engine serves (v4 -> models/v4, data/interim/v4; rows/features from v3)
MODEL_DIR = REPO_ROOT / "models" / SERVED_RUN
INTERIM_DIR = REPO_ROOT / "data" / "interim" / SERVED_RUN
TABLE = REPO_ROOT / "data" / "interim" / "v3" / "table.parquet"


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
def test_live_inference_reproduces_frozen_pipeline(eng):
    table = pd.read_parquet(TABLE)
    preds = pd.read_parquet(INTERIM_DIR / "predictions.parquet").set_index("case_id")
    for c in eng.registry["cases"]:
        run = eng.run(c["case_id"], fresh=True)
        ids = run.rows["case_id"]
        p = preds.loc[ids]
        ref = table.set_index("case_id").loc[ids]
        for col in ("raw_probability", "calibrated_bust_probability"):
            np.testing.assert_allclose(run.rows[col], p[col], atol=1e-5)
        for f in ["an_n_within", "an_dist1", "an_bust_rate", "an_err_med", "an_n_eligible", "support_distance"]:
            np.testing.assert_allclose(run.rows[f].astype(float), ref[f].astype(float), rtol=1e-5, atol=1e-5)
        assert (run.rows["support_level"].to_numpy() == ref["support_level"].to_numpy()).all()


@pytest.mark.skipif(not (MODEL_DIR / "models.joblib").exists(), reason="v4 model files not present")
def test_exported_model_equals_trained_files(eng):
    from forecast_bust.models.pattern_bust import V4Model
    st = _state_with_memory(eng, eng.registry["cases"][0]["case_id"])
    pd.testing.assert_frame_equal(eng.model.predict(st), V4Model(MODEL_DIR).predict(st))


def test_production_model_is_v4_and_needs_no_old_models(eng):
    assert eng.model.model_type == "pattern_aware_xgboost"
    for old in ("b2", "sentinel"):
        assert not hasattr(eng, old)
    assert not (DEMO_DIR / "model" / "v3").exists() and not list((DEMO_DIR / "model").glob("*booster*"))
    for f in eng.model.features:
        assert not f.startswith(FORBIDDEN_INPUTS), f


def test_inference_depends_on_inputs(eng):
    """The probabilities are computed from the inputs, not looked up: perturbing the state changes them."""
    st = _state_with_memory(eng, eng.registry["cases"][0]["case_id"])
    p0 = eng.model.predict(st)["raw_probability"].to_numpy()
    p1 = eng.model.predict(st.assign(u500_anom=st["u500_anom"] * 3))["raw_probability"].to_numpy()
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
    for k in ("actual_bust", "normalized_error", "failure_fingerprint", "era5"):
        assert k not in txt
    cell = j["regions"][0]["days"][0]
    assert abs(cell["reliability_confidence"] - (1 - cell["bust_probability"])) < 1e-12
    assert cell["model_type"] == "pattern_aware_xgboost"
    assert cell["bust_probability"] == cell["calibrated_bust_probability"]
    assert cell["confidence"] in ("LOW", "MODERATE", "HIGH")
    for k in ("b2_probability", "disagreement_pp", "p_sentinel", "upper_tail_error", "q95", "bma_", "local_acc",
              "pattern_failure\"", "acc_q10"):
        assert k not in txt, k


def test_api_explain_and_reveal(client, eng):
    cid = eng.registry["cases"][0]["case_id"]
    client.post(f"/api/demo/cases/{cid}/run")
    rid = eng.run(cid).rows["region_id"].iloc[0]
    ex = client.get(f"/api/demo/cases/{cid}/regions/{rid}/explain?lead_day=4").json()
    assert ex["attribution"]["drivers"] and ex["analogue_summary"]["nearest"] and ex["interpretation"]
    assert set(ex["criteria"]) == {"magnitude", "pattern", "support"}
    assert "local_acc" not in json.dumps(ex) and "acc_q10" not in json.dumps(ex)
    assert "actual_bust" not in json.dumps(ex)
    rv = client.post(f"/api/demo/cases/{cid}/regions/{rid}/reveal?lead_day=4").json()
    v = json.loads((DEMO_DIR / "cases" / cid / "verification.json").read_text())
    day = next(d for x in v["regions"] if x["region_id"] == rid for d in x["days"] if d["lead_day"] == 4)
    assert rv["verification"]["normalized_error"] == day["normalized_error"]
    assert rv["verification"]["failure_fingerprint"]["signature"] == day["signature"]
    pv = json.loads((DEMO_DIR / "cases" / cid / "pattern_verification.json").read_text())["rows"]
    pr = next(x for x in pv if x["region_id"] == rid and x["lead_day"] == 4)
    assert rv["verification"]["local_acc"] == pr["local_acc"] and rv["verification"]["actual_bust"] == bool(pr["pattern_bust"])
    assert rv["verification"]["actual_magnitude_bust"] == bool(day["bust"])
    assert rv["memory_update"]["verified_cases_after"] == rv["memory_update"]["verified_cases_before"] + 1


def test_api_errors(client):
    assert client.post("/api/demo/cases/2099010100/run").status_code == 404
    assert client.get("/api/demo/cases/2099010100/regions/R0000/explain?lead_day=11").status_code == 400
