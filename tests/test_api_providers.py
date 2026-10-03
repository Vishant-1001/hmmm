"""API provenance: provider metadata, synthetic flag and ensemble size are exposed, never inferred by the UI."""
import json

import pytest
from fastapi.testclient import TestClient

from forecast_bust.api.app import app
from forecast_bust.config import REPO_ROOT
from forecast_bust.demo.build import SERVED_DEMO_DIR

client = TestClient(app)


def test_providers_endpoint_flags():
    r = client.get("/api/providers").json()
    by = {p["provider"]: p for p in r["providers"]}
    assert set(by) == {"ecmwf_research", "ncmrwf_tigge", "synthetic"}
    assert by["synthetic"]["synthetic"] is True and by["synthetic"]["demo_only"] is True
    assert by["ecmwf_research"]["synthetic"] is False and by["ncmrwf_tigge"]["synthetic"] is False
    assert set(r["b2_modes"]) == {"b2_ecmwf", "b2_ncmrwf", "synthetic_demo"}
    av = REPO_ROOT / "artifacts" / "ncmrwf_tigge_availability.json"
    if av.exists():   # NCMRWF status is whatever the retrieval check recorded - never a hard-coded claim
        assert by["ncmrwf_tigge"]["status"] == json.loads(av.read_text())["status"]


@pytest.mark.skipif(not (SERVED_DEMO_DIR / "registry.json").exists(), reason="demo bundle not built")
def test_run_response_carries_source():
    cid = client.get("/api/demo/cases").json()["cases"][0]["case_id"]
    src = client.post(f"/api/demo/cases/{cid}/run").json()["source"]
    assert src["provider"] == "ecmwf_research" and src["synthetic"] is False and src["demo_only"] is False
    assert src["ensemble_member_count"] == 50 and src["lead_days"] == list(range(1, 11))
    assert len(src["valid_times"]) == 10 and src["initialization_time"].startswith(cid[:4])
    assert src["model_version"]
