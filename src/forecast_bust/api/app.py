"""FastAPI backend serving precomputed REAL historical research artifacts.

There is no live NWP feed. Every response carries `mode` so the UI can label it
"Historical research replay". Missing artifacts return explicit 404/503 errors —
never placeholder numbers.

    uvicorn forecast_bust.api.app:app --port 8000
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from forecast_bust.config import REPO_ROOT

ART = Path(os.environ.get("FBS_ARTIFACT_DIR", REPO_ROOT / "artifacts"))
MODE = "Historical research replay - precomputed real ECMWF IFS ENS cases; not a live or NCMRWF feed"

app = FastAPI(title="Forecast Bust Sentinel API", version="0.1.0",
              description="Regional Day 1-10 forecast-bust probability over existing NWP (SIH26079 research prototype)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])


def _read(path: Path):
    if not path.exists():
        raise HTTPException(503, detail=f"NOT YET COMPUTED: artifact {path.relative_to(ART)} is unavailable")
    return json.loads(path.read_text())


@lru_cache(maxsize=32)
def _case(case_id: str) -> dict:
    if not case_id.isdigit() or len(case_id) != 10:
        raise HTTPException(400, detail="case_id must be YYYYMMDDHH")
    p = ART / "replay" / case_id / "forecast.json"
    if not p.exists():
        raise HTTPException(404, detail=f"unknown forecast case {case_id}")
    return json.loads(p.read_text())


def _region(case_id: str, region_id: str) -> dict:
    for r in _case(case_id)["regions"]:
        if r["region_id"] == region_id:
            return r
    raise HTTPException(404, detail=f"unknown region {region_id}")


@app.get("/api/health")
def health():
    idx = ART / "replay" / "index.json"
    return {"status": "ok", "mode": MODE, "artifacts_present": {
        "metrics": (ART / "metrics.json").exists(), "replay_index": idx.exists(),
        "dataset_manifest": (ART / "dataset_manifest.json").exists()}}


@app.get("/api/provenance")
def provenance():
    out = {"mode": MODE, "dataset_manifest": _read(ART / "dataset_manifest.json")}
    em = ART / "experiment_manifest.json"
    if em.exists():
        m = json.loads(em.read_text())
        out["experiment"] = {k: m.get(k) for k in ("created", "git_rev", "seed", "splits", "protocol", "software",
                                                  "descriptions")}
    for name in ("thresholds.json", "preprocessing.json", "pca.json", "support_diagnostics.json"):
        p = ART / name
        out[name.removesuffix(".json")] = "present" if p.exists() else "NOT YET COMPUTED"
    return out


@app.get("/api/metrics")
def metrics():
    out = {"metrics": _read(ART / "metrics.json"), "ablation": _read(ART / "ablation_results.json")}
    for name in ("calibration", "spread_skill", "fingerprint_metrics", "feature_importance"):
        p = ART / f"{name}.json"
        out[name] = json.loads(p.read_text()) if p.exists() else None
    return out


@app.get("/api/forecast/cases")
def cases():
    return {"mode": MODE, **_read(ART / "replay" / "index.json")}


@app.get("/api/forecast/{case_id}/overview")
def overview(case_id: str):
    c = _case(case_id)
    days = []
    for d in range(10):
        ps = [r["trajectory"][d]["p_bust"] for r in c["regions"]]
        days.append({"lead_day": d + 1, "mean_p_bust": sum(ps) / len(ps), "max_p_bust": max(ps),
                     "n_alerts": sum(r["trajectory"][d]["alert"] for r in c["regions"])})
    return {"mode": MODE, "case_id": case_id, "init_time": c["init_time"], "data_source": c["data_source"],
            "selection": c["selection"], "selection_note": c["selection_note"], "alert_threshold": c["alert_threshold"],
            "priority_formula": c["priority_formula"], "priority_queue": c["priority_queue"], "days": days,
            "map": [{k: r[k] for k in ("region_id", "name", "lat", "lon", "lat_bounds", "lon_bounds", "peak_risk_day",
                                       "peak_p_bust", "first_alert_day")}
                    | {"p_bust": [t["p_bust"] for t in r["trajectory"]],
                       "p_spread_baseline": [t["p_spread_baseline"] for t in r["trajectory"]],
                       "evidence": [t["evidence"] for t in r["trajectory"]]} for r in c["regions"]],
            "fields": c["fields"], "blind": True}


@app.get("/api/forecast/{case_id}/regions")
def regions(case_id: str):
    return [{k: r[k] for k in ("region_id", "name", "lat", "lon", "peak_risk_day", "peak_p_bust", "alert_days")}
            for r in _case(case_id)["regions"]]


@app.get("/api/forecast/{case_id}/regions/{region_id}")
def region(case_id: str, region_id: str):
    r = _region(case_id, region_id)
    return {k: v for k, v in r.items() if k != "trajectory"} | {"trajectory": trajectory(case_id, region_id)["days"]}


@app.get("/api/forecast/{case_id}/regions/{region_id}/trajectory")
def trajectory(case_id: str, region_id: str):
    r = _region(case_id, region_id)
    keys = ("lead_day", "valid_time", "p_bust", "confidence", "p_spread_baseline", "p_climatology", "disagreement_pp",
            "spread_m", "spread_pct", "alert", "support", "evidence", "analogues_within_radius", "priority_score")
    return {"region_id": region_id, "peak_risk_day": r["peak_risk_day"], "first_alert_day": r["first_alert_day"],
            "days": [{k: d[k] for k in keys} for d in r["trajectory"]]}


@app.get("/api/forecast/{case_id}/regions/{region_id}/evidence")
def evidence(case_id: str, region_id: str, lead_day: int | None = None):
    r = _region(case_id, region_id)
    day = lead_day or r["peak_risk_day"]
    if not 1 <= day <= 10:
        raise HTTPException(400, detail="lead_day must be 1..10")
    d = r["trajectory"][day - 1]
    return {"region_id": region_id, "lead_day": day, "p_bust": d["p_bust"], "p_spread_baseline": d["p_spread_baseline"],
            "disagreement_pp": d["disagreement_pp"], "evidence_strength": d["evidence"], "support": d["support"],
            "support_distance": d["support_distance"], **d["explanation"]}


@app.get("/api/forecast/{case_id}/regions/{region_id}/history")
def history(case_id: str, region_id: str, lead_day: int | None = None):
    r = _region(case_id, region_id)
    day = lead_day or r["peak_risk_day"]
    if not 1 <= day <= 10:
        raise HTTPException(400, detail="lead_day must be 1..10")
    d = r["trajectory"][day - 1]
    return {"region_id": region_id, "lead_day": day, "verified_cases_available": d["verified_cases_available"],
            "analogues_within_radius": d["analogues_within_radius"], "analogue_bust_rate": d["analogue_bust_rate"],
            "historical_failure_signature": d["historical_failure_signature"], "analogues": d["analogues"],
            "wording": "Historical failure signature among similar forecast states (evidence, not causal proof)"}


@app.get("/api/replay/{case_id}")
def replay(case_id: str):
    return overview(case_id)


@app.get("/api/replay/{case_id}/verification")
def verification(case_id: str):
    _case(case_id)
    return _read(ART / "replay" / case_id / "verification.json")


_dist = REPO_ROOT / "frontend" / "dist"
if _dist.exists():
    app.mount("/assets", StaticFiles(directory=_dist / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith("api/"):
            raise HTTPException(404)
        return FileResponse(_dist / "index.html")
