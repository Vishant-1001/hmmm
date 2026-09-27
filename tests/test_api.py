"""API tests against a TEST FIXTURE artifact directory (synthetic, clearly not results)."""
import importlib
import json

import pytest
from fastapi.testclient import TestClient


def _day(d, p):
    return {"lead_day": d, "valid_time": "2022-01-02", "p_bust": p, "confidence": 1 - p, "p_spread_baseline": 0.1,
            "p_climatology": 0.1, "disagreement_pp": 100 * (p - 0.1), "spread_m": 5.0, "spread_pct": 0.5, "alert": p > 0.3,
            "support": "NORMAL SUPPORT", "support_distance": 1.0, "evidence": "WEAK EVIDENCE",
            "analogues_within_radius": 3, "verified_cases_available": 50, "analogue_bust_rate": 0.1,
            "priority_score": p, "explanation": {"drivers": [], "groups": {}, "evidence": []}, "analogues": [],
            "historical_failure_signature": {"basis": "none", "n": 0, "distribution": None}}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    art = tmp_path / "artifacts"
    case = art / "replay" / "2022010100"
    case.mkdir(parents=True)
    region = {"region_id": "R0000", "name": "TEST FIXTURE", "lat": 0.0, "lon": 62.0, "lat_bounds": [-2.8, 2.8],
              "lon_bounds": [59.1, 64.7], "trajectory": [_day(d, 0.05 * d) for d in range(1, 11)],
              "peak_risk_day": 10, "peak_p_bust": 0.5, "first_alert_day": 7, "alert_days": [7, 8, 9, 10]}
    (case / "forecast.json").write_text(json.dumps({
        "case_id": "2022010100", "init_time": "2022-01-01", "data_source": "TEST FIXTURE", "selection": "random",
        "selection_note": "fixture", "alert_threshold": {"p_bust": 0.3}, "priority_formula": {}, "priority_queue": [],
        "regions": [region], "fields": {}, "attribution_note": "fixture"}))
    (case / "verification.json").write_text(json.dumps({"case_id": "2022010100", "regions": [], "summary": {}}))
    (art / "replay" / "index.json").write_text(json.dumps({"cases": [{"case_id": "2022010100"}], "selection_rule": "x"}))
    monkeypatch.setenv("FBS_ARTIFACT_DIR", str(art))
    import forecast_bust.api.app as appmod
    appmod = importlib.reload(appmod)
    return TestClient(appmod.app)


def test_health_and_cases(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and "not a live" in h["mode"]
    assert client.get("/api/forecast/cases").json()["cases"][0]["case_id"] == "2022010100"


def test_region_endpoints(client):
    t = client.get("/api/forecast/2022010100/regions/R0000/trajectory").json()
    assert len(t["days"]) == 10 and t["peak_risk_day"] == 10
    ov = client.get("/api/forecast/2022010100/overview").json()
    assert ov["days"][0]["lead_day"] == 1 and ov["blind"] is True
    ev = client.get("/api/forecast/2022010100/regions/R0000/evidence?lead_day=3").json()
    assert ev["lead_day"] == 3 and ev["evidence_strength"] == "WEAK EVIDENCE"
    assert client.get("/api/forecast/2022010100/regions/R0000/history").json()["lead_day"] == 10


def test_blind_case_has_no_verification_fields(client):
    full = client.get("/api/replay/2022010100/full").json()
    text = json.dumps(full)
    for k in ("era5_z500", "normalized_error", "\"bust\"", "error_m"):
        assert k not in text
    assert client.get("/api/replay/2022010100/verification").status_code == 200


def test_invalid_inputs(client):
    assert client.get("/api/forecast/abc/overview").status_code == 400
    assert client.get("/api/forecast/1999010100/overview").status_code == 404
    assert client.get("/api/forecast/2022010100/regions/R9999/trajectory").status_code == 404
    assert client.get("/api/forecast/2022010100/regions/R0000/evidence?lead_day=11").status_code == 400


def test_missing_metrics_is_explicit(client):
    r = client.get("/api/metrics")
    assert r.status_code == 503 and "NOT YET COMPUTED" in r.json()["detail"]
