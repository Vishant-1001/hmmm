"""Deployment-facing behaviour: liveness probe, CORS policy, project-root resolution, and the
numerical contract of a live demo run (probabilities in [0, 1], Day 1-10, valid = init + lead)."""
from __future__ import annotations

import math
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from forecast_bust.api.app import app
from forecast_bust.config import REPO_ROOT

client = TestClient(app)


def test_health_is_lightweight():
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


@pytest.mark.parametrize("origin,allowed", [
    ("https://forecast-bust-sentinel.onrender.com", True),
    ("https://fbs-demo-git-main.vercel.app", True),
    ("http://localhost:5173", True),
    ("https://evil.example.com", False),
    ("https://onrender.com.evil.example.com", False),
])
def test_cors_policy(origin, allowed):
    r = client.get("/health", headers={"Origin": origin})
    assert (r.headers.get("access-control-allow-origin") == origin) is allowed
    assert "access-control-allow-credentials" not in r.headers


def _resolve_root(env: dict, cwd: Path) -> subprocess.CompletedProcess:
    code = "import forecast_bust.config as c; print(c.REPO_ROOT)"
    return subprocess.run([sys.executable, "-c", code], cwd=cwd, capture_output=True, text=True,
                          env={**os.environ, **env})


def test_project_root_from_env(tmp_path):
    ok = _resolve_root({"FBS_PROJECT_ROOT": str(REPO_ROOT)}, tmp_path)
    assert ok.returncode == 0 and Path(ok.stdout.strip()) == REPO_ROOT
    bad = _resolve_root({"FBS_PROJECT_ROOT": str(tmp_path)}, tmp_path)
    assert bad.returncode != 0 and "does not contain config/model.yaml" in bad.stderr


def test_project_root_without_env_is_the_checkout(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "FBS_PROJECT_ROOT"}
    r = subprocess.run([sys.executable, "-c", "import forecast_bust.config as c; print(c.CONFIG_DIR)"],
                       cwd=tmp_path, capture_output=True, text=True, env=env)
    assert r.returncode == 0 and Path(r.stdout.strip()) == REPO_ROOT / "config"


def test_demo_run_probabilities_and_day_indexing():
    cases = client.get("/api/demo/cases").json()["cases"]
    cid = cases[0]["case_id"]
    run = client.post(f"/api/demo/cases/{cid}/run").json()
    init = datetime.strptime(cid, "%Y%m%d%H")
    rid = run["priority_queue"][0]["region_id"]
    reg = client.get(f"/api/demo/cases/{cid}/regions/{rid}").json()
    assert [t["lead_day"] for t in reg["trajectory"]] == list(range(1, 11))
    for t in reg["trajectory"]:
        assert datetime.fromisoformat(str(t["valid_time"]).replace("Z", "")) == init + timedelta(days=t["lead_day"])
        for k in ("bust_probability", "raw_probability", "b0_probability"):
            v = t[k]
            assert v is not None and math.isfinite(v) and 0.0 <= v <= 1.0, (k, v)
