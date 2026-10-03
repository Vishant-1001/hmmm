"""Dump frontend/src/test/demo_fixture.json from the REAL FastAPI app (TestClient); nothing is hand-written.

    FBS_DEMO_WARM=0 .venv/bin/python scripts/dump_ui_fixture.py
"""
import json

from fastapi.testclient import TestClient

from forecast_bust.api.app import app
from forecast_bust.config import REPO_ROOT


def main() -> None:
    c = TestClient(app)
    cases = c.get("/api/demo/cases").json()
    cid = next(x for x in cases["cases"] if x["selection"] == "random")["case_id"]
    run = c.post(f"/api/demo/cases/{cid}/run").json()
    rid, day = run["priority_queue"][0]["region_id"], run["priority_queue"][0]["lead_day"]
    fx = {"case_id": cid, "region_id": rid, "lead_day": day, "cases": cases, "run": run,
          "region": c.get(f"/api/demo/cases/{cid}/regions/{rid}").json(),
          "explain": c.get(f"/api/demo/cases/{cid}/regions/{rid}/explain?lead_day={day}").json(),
          "reveal": c.post(f"/api/demo/cases/{cid}/regions/{rid}/reveal?lead_day={day}").json()}
    (REPO_ROOT / "frontend" / "src" / "test" / "demo_fixture.json").write_text(json.dumps(fx, indent=1))
    print(cid, rid, day)


if __name__ == "__main__":
    main()
