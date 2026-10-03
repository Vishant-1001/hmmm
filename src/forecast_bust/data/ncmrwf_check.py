"""NCMRWF/TIGGE availability: catalogue probe, actual retrieval, programmatic verification, year coverage.

Three levels of evidence are recorded separately and never merged:
  catalogue   - ECDS constraints endpoint (public, unauthenticated): what the archive CATALOGUE lists
  retrieval   - an actual GRIB download with an ECDS token, checked field by field (`verify_retrieval`)
  coverage    - the same retrieval for one pre-declared date per year

Pre-declared probe dates (fixed before any retrieval; not chosen from any result):
  smoke test : 2021-07-15 00 UTC, Day 1-10 (24..240 h every 24 h)
  coverage   : 15 July 00 UTC of 2018, 2019, 2020, 2021, 2022 at Day 1, 5, 10 (24, 120, 240 h)
"""
from __future__ import annotations

import datetime as dt
import json
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

from forecast_bust.data.providers import (ECDS_PROCESS_URL, NCMRWF_GRIB_CENTRES, G, ForecastBundle,
                                          NCMRWFTIGGEProvider, ensemble_mean_spread)

SMOKE_INIT = "2021-07-15T00:00"
SMOKE_LEADS = list(range(24, 241, 24))
COVERAGE_YEARS = [2018, 2019, 2020, 2021, 2022]
COVERAGE_LEADS = [24, 120, 240]


def _post(url: str, body: dict, timeout: int = 60) -> tuple[int, dict]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, {"detail": str(e)}


def catalogue_probe(years=COVERAGE_YEARS) -> dict:
    """ECDS constraints for origin=ncmrwf, Z500 perturbed members, per year (no authentication needed)."""
    out = {"endpoint": f"{ECDS_PROCESS_URL}/constraints", "queried": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "evidence_level": "catalogue metadata only - NOT a retrieval", "years": {}}
    for y in years:
        body = {"inputs": {"origin": "ncmrwf", "year": str(y), "forecast_type": "perturbed_forecast",
                           "level_type": "pressure", "variable": ["geopotential_height"], "level_value": ["500_hpa"]}}
        try:
            status, c = _post(f"{ECDS_PROCESS_URL}/constraints", body)
        except Exception as e:  # network failure is recorded, not hidden
            out["years"][str(y)] = {"error": f"{type(e).__name__}: {e}"}
            continue
        if status != 200:
            out["years"][str(y)] = {"http_status": status, "error": c}
            continue
        leads = sorted(int(h) for h in c.get("leadtime_hour", []))
        out["years"][str(y)] = {
            "http_status": status, "ncmrwf_listed": "ncmrwf" in c.get("origin", []),
            "months": c.get("month", []), "times": c.get("time", []),
            "geopotential_height_listed": "geopotential_height" in c.get("variable", []),
            "level_500_listed": "500_hpa" in c.get("level_value", []),
            "perturbed_forecast_listed": "perturbed_forecast" in c.get("forecast_type", []),
            "day1_10_leads_listed": all(h in leads for h in SMOKE_LEADS), "max_lead_h": max(leads) if leads else None,
            "note": "constraints are filtered by the selections; day-level gaps are not resolved by this endpoint",
        }
    return out


def unauthenticated_attempt(request: dict) -> dict:
    """What ECDS answers to the smoke request without a token (records the access blocker verbatim)."""
    try:
        status, body = _post(f"{ECDS_PROCESS_URL}/execution", {"inputs": request})
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    return {"http_status": status, "response": body}


def verify_retrieval(bundle: ForecastBundle, facts: dict, requested_init: str, requested_leads: list[int],
                     domain: dict) -> dict:
    """The section-6 checklist, evaluated on the returned data (no assumption taken from the request)."""
    da, md = bundle.data, bundle.metadata
    init = pd.Timestamp(requested_init)
    lat, lon = da["latitude"].values, da["longitude"].values
    leads = [int(h) for h in da["prediction_timedelta"].values]
    z = da.sel(level=500).isel(time=0) / G
    mean, std, n = ensemble_mean_spread(z)
    file_init = pd.Timestamp(facts["init_time"]) if facts.get("init_time") else None
    checks = {
        "retrieval_completed": True,
        "ncmrwf_origin_in_grib": facts.get("grib_centre") in NCMRWF_GRIB_CENTRES,
        "not_synthetic": md.synthetic is False and md.provider == "ncmrwf_tigge",
        "ensemble_members_present": da.sizes["number"] >= 2,
        "member_ids_parsed": len(set(md.member_ids)) == da.sizes["number"],
        "z500_present": 500 in [int(x) for x in da["level"].values] and facts.get("grib_short_name") == "gh",
        "initialization_time_correct": file_init == init,
        "valid_times_correct": md.valid_times == [(init + pd.Timedelta(hours=h)).isoformat() for h in leads],
        "all_requested_leads_present": set(requested_leads) <= set(leads),
        "domain_covers_target": bool(lat.min() <= domain["lat_min"] + 2.82 and lat.max() >= domain["lat_max"] - 2.82
                                     and lon.min() <= domain["lon_min"] + 2.82 and lon.max() >= domain["lon_max"] - 2.82),
        "coordinates_ascending_0_360": bool(np.all(np.diff(lat) > 0) and np.all(np.diff(lon) > 0) and lon.min() >= 0 and lon.max() < 360),
        "units_correct": facts.get("grib_units") in ("gpm", "m") and md.units == "m**2 s**-2",
        "z500_physically_plausible": bool(5000 < float(np.nanmean(z.values)) < 6200),
        "missing_values_handled": bool((np.isnan(mean.values) == (n.values == 0)).all()),
        "ensemble_mean_computed": bool(np.isfinite(mean.values).any()),
        "ensemble_spread_computed": bool(np.isfinite(std.values).any() and np.nanmin(std.values) >= 0),
    }
    return {"checks": checks, "all_passed": all(checks.values()),
            "members_found": int(da.sizes["number"]), "member_ids": md.member_ids,
            "available_leads": leads, "missing_leads": sorted(set(requested_leads) - set(leads)),
            "nan_fraction": float(np.isnan(z.values).mean()),
            "min_valid_members_per_point": int(n.values.min()),
            "grid": md.grid, "native_grid": md.native_grid, "units": {"grib": facts.get("grib_units"), "canonical": md.units},
            "domain": {"lat": [float(lat.min()), float(lat.max())], "lon": [float(lon.min()), float(lon.max())]},
            "z500_mean_m": float(np.nanmean(z.values)), "spread_range_m": [float(np.nanmin(std.values)), float(np.nanmax(std.values))]}


def timed_retrieve(provider: NCMRWFTIGGEProvider, init: str, leads: list[int]) -> dict:
    t0 = time.perf_counter()
    try:
        path = provider.download(init, leads)
        b = provider.retrieve(init, leads, grib_path=path)
        _, facts = provider.read_grib(path)
        return {"ok": True, "bundle": b, "facts": facts, "path": path, "seconds": round(time.perf_counter() - t0, 2),
                "bytes": path.stat().st_size}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "seconds": round(time.perf_counter() - t0, 2)}
