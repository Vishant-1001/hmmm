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
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from forecast_bust.config import REPO_ROOT

ART = Path(os.environ.get("FBS_ARTIFACT_DIR", REPO_ROOT / "artifacts"))
MODE = "Historical research replay - precomputed real ECMWF IFS ENS cases; not a live or NCMRWF feed"

app = FastAPI(title="Forecast Bust Sentinel API", version="0.1.0",
              description="Regional Day 1-10 forecast-bust probability over existing NWP (SIH26079 research prototype)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1000)


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
            "support_distance": d["support_distance"], "attribution_note": _case(case_id).get("attribution_note"),
            **d["explanation"]}


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


@app.get("/api/replay/{case_id}/full")
def replay_full(case_id: str):
    """Complete blind case (all regions, trajectories, evidence). Contains no verification data."""
    return {"mode": MODE, **_case(case_id)}


@app.get("/api/replay/{case_id}/verification")
def verification(case_id: str):
    _case(case_id)
    return _read(ART / "replay" / case_id / "verification.json")


# ---------------------------------------------------------------------------------------------
# Live inference demo (v1 frozen models executed at request time on stored forecast states)
# ---------------------------------------------------------------------------------------------
from forecast_bust.demo import schemas as S  # noqa: E402

DEMO_MODE = ("Historical replay research prototype - the frozen Sentinel/B2 models execute live on stored real "
             "ECMWF IFS ENS forecast states; not a live or NCMRWF feed")


@lru_cache(maxsize=1)
def _engine():
    from forecast_bust.demo.engine import engine
    try:
        return engine()
    except FileNotFoundError as e:
        raise HTTPException(503, detail=f"NOT YET COMPUTED: {e}")


def _demo(fn, *a):
    from forecast_bust.demo.engine import CaseNotFound
    try:
        return fn(*a)
    except CaseNotFound as e:
        raise HTTPException(404, detail=f"unknown case/region/lead {e}")


def _lead(lead_day: int) -> int:
    if not 1 <= lead_day <= 10:
        raise HTTPException(400, detail="lead_day must be 1..10")
    return lead_day


@app.on_event("startup")
def _warm_engine():
    """Load models + historical memory once at startup (FBS_DEMO_WARM=0 disables)."""
    if os.environ.get("FBS_DEMO_WARM", "1") == "1":
        try:
            _engine()
        except HTTPException:
            pass


@app.get("/api/demo/cases", response_model=S.CaseList)
def demo_cases():
    e = _engine()
    return {"mode": DEMO_MODE, **e.registry}


@app.get("/api/demo/model", response_model=S.ModelSummary)
def demo_model():
    return _engine().model_summary()


@app.get("/api/demo/health")
def demo_health():
    e = _engine()
    return {"status": "ok", "mode": DEMO_MODE, "startup_ms": e.startup_ms, "cases": len(e.registry["cases"]),
            "memory_rows": int(len(e.mem))}


@app.post("/api/demo/cases/{case_id}/run", response_model=S.RunResult)
def demo_run(case_id: str, fresh: bool = True):
    """Executes the models for all regions x Day 1-10 of one initialisation (blind: no verification).
    Every call re-runs inference (fresh=true); region/explain/reveal reuse the latest run of the case."""
    e = _engine()
    _demo(e.run, case_id, fresh)
    return {"mode": DEMO_MODE, **_demo(e.overview, case_id)}


@app.get("/api/demo/cases/{case_id}/regions/{region_id}", response_model=S.RegionDetail)
def demo_region(case_id: str, region_id: str):
    return _demo(_engine().region, case_id, region_id)


@app.get("/api/demo/cases/{case_id}/regions/{region_id}/explain", response_model=S.Explanation)
def demo_explain(case_id: str, region_id: str, lead_day: int):
    return _demo(_engine().explain, case_id, region_id, _lead(lead_day))


@app.get("/api/demo/cases/{case_id}/fields")
def demo_fields(case_id: str):
    """Ensemble-mean and spread Z500 fields of the initialisation (forecast-time information)."""
    return _demo(_engine().fields, case_id)


@app.post("/api/demo/cases/{case_id}/regions/{region_id}/reveal", response_model=S.Reveal)
def demo_reveal(case_id: str, region_id: str, lead_day: int):
    """Explicit user action: returns the ERA5 verification and failure fingerprint."""
    return _demo(_engine().reveal, case_id, region_id, _lead(lead_day))


_dist = REPO_ROOT / "frontend" / "dist"
if _dist.exists():
    app.mount("/assets", StaticFiles(directory=_dist / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith("api/"):
            raise HTTPException(404)
        return FileResponse(_dist / "index.html")
