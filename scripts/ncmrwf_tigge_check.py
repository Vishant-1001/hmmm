"""NCMRWF/TIGGE availability check -> artifacts/ncmrwf_tigge_availability.json

    .venv/bin/python scripts/ncmrwf_tigge_check.py

1. catalogue probe (public ECDS constraints, no token)
2. smoke retrieval 2021-07-15 00 UTC, NCMRWF perturbed members, gh 500 hPa, Day 1-10 (needs an ECDS token)
3. programmatic verification of the returned GRIB
4. one pre-declared date per year 2018-2022 (Day 1, 5, 10)
5. B2 smoke inference on the real smoke case (time + peak RAM) and its real ERA5 verification labels
Without a token, steps 2-5 record the exact blocker; nothing is inferred from the catalogue.
"""
import datetime as dt
import json
import resource
import sys
import time

from forecast_bust.config import REPO_ROOT, clean_json, data_config
from forecast_bust.data import ncmrwf_check as C
from forecast_bust.data.providers import ECDS_API_URL, ECDS_DATASET, ECDS_DATASET_URL, NCMRWFTIGGEProvider

OUT = REPO_ROOT / "artifacts" / "ncmrwf_tigge_availability.json"


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def update_dataset_manifest(rep: dict) -> None:
    """Separate provider entries in artifacts/dataset_manifest.json; the existing ECMWF/ERA5 entries are kept."""
    path = REPO_ROOT / "artifacts" / "dataset_manifest.json"
    man = json.loads(path.read_text()) if path.exists() else {}
    cov = rep.get("coverage", {})
    man["providers"] = {
        "ecmwf_era5": {"provider": "ECMWF (IFS ENS) + ERA5 verification", "archive": "WeatherBench 2 (gs://weatherbench2)",
                       "synthetic": False, "role": "benchmark data source of B2/v2/v3/V4 and every served case",
                       "entries": "see `forecast`, `reference`, `climatology` above (unchanged)"},
        "ncmrwf_tigge": {"provider": "NCMRWF", "archive": "TIGGE/ECDS", "origin": "dems", "synthetic": False,
                         "source_url": ECDS_DATASET_URL, "api_url": ECDS_API_URL,
                         "access_type": "authenticated ECDS personal access token (cdsapi); catalogue/constraints endpoint is public",
                         "retrieval_date": rep["generated"] if rep.get("retrieval_success") else None,
                         "catalogue_checked": rep["catalogue"]["queried"],
                         "retrieval_success": rep.get("retrieval_success"),
                         "coverage": {"catalogue_years_listing_z500_pf": sorted(y for y, v in rep["catalogue"]["years"].items()
                                                                                 if v.get("geopotential_height_listed") and v.get("perturbed_forecast_listed")),
                                      "retrieved_years": sorted(y for y, v in cov.items() if v.get("success")),
                                      "measured": bool(rep.get("retrieval_success"))},
                         "status": rep["status"], "details": "artifacts/ncmrwf_tigge_availability.json"},
        "synthetic_demo": {"provider": "SyntheticDemoProvider (in-process, seeded)", "synthetic": True, "demo_only": True,
                           "source_label": "Synthetic demonstration scenario",
                           "role": "UI stress cases, development and tests only; never training, calibration or metrics",
                           "details": "artifacts/demo/synthetic/manifest.json"},
    }
    path.write_text(json.dumps(clean_json(man), indent=1, default=str))


def main() -> int:
    prov = NCMRWFTIGGEProvider()
    req = prov.build_request(C.SMOKE_INIT, C.SMOKE_LEADS)
    smoke_init = dt.datetime.fromisoformat(C.SMOKE_INIT)
    rep = {
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "retrieval_mechanism": {"client": "cdsapi (ECDS / CDS-API engine)", "api_url": ECDS_API_URL,
                                "dataset": ECDS_DATASET, "dataset_url": ECDS_DATASET_URL,
                                "authentication": "ECMWF account + ECDS personal access token; ECDS terms and TIGGE licence accepted"},
        "tested_date": smoke_init.strftime("%Y-%m-%d"), "tested_cycle": smoke_init.strftime("%H UTC"),
        "origin": "ncmrwf (MARS/GRIB origin dems, WMO centre 29)", "variable": "geopotential_height (TIGGE param 156, gpm)",
        "level": "500 hPa", "forecast_type": "perturbed_forecast", "requested_leads": C.SMOKE_LEADS,
        "domain_requested": {"area_NWSE": req["area"], "target_domain": data_config()["target_domain"]},
        "smoke_request": {"dataset": ECDS_DATASET, "request": req},
        "catalogue": C.catalogue_probe(),
        "credentials": prov.credentials_status(),
    }
    if not rep["credentials"]["available"]:
        rep["unauthenticated_attempt"] = C.unauthenticated_attempt(req)
        rep.update(retrieval_success=False, members_found=None, available_leads=None, grid=None, units=None,
                   error_message_if_any=f"Retrieval not attempted with credentials: {rep['credentials']['reason']}. "
                                        f"ECDS answered the smoke request with HTTP "
                                        f"{rep['unauthenticated_attempt'].get('http_status')} "
                                        f"{(rep['unauthenticated_attempt'].get('response') or {}).get('detail', '')!r}.",
                   coverage={str(y): {"attempted": False, "reason": "no ECDS token"} for y in C.COVERAGE_YEARS},
                   b2_smoke={"ran": False, "reason": "no real NCMRWF data retrieved"},
                   status="NCMRWF/TIGGE CATALOGUE CONFIRMED, RETRIEVAL NOT YET VERIFIED")
        OUT.write_text(json.dumps(clean_json(rep), indent=1, default=str))
        update_dataset_manifest(rep)
        print(rep["status"], "-", rep["error_message_if_any"])
        return 2

    r = C.timed_retrieve(prov, C.SMOKE_INIT, C.SMOKE_LEADS)
    rep["retrieval_success"] = r["ok"]
    if not r["ok"]:
        rep.update(error_message_if_any=r["error"], status="NCMRWF/TIGGE CATALOGUE CONFIRMED, RETRIEVAL NOT YET VERIFIED")
        OUT.write_text(json.dumps(clean_json(rep), indent=1, default=str))
        update_dataset_manifest(rep)
        print(rep["status"], "-", r["error"])
        return 1
    v = C.verify_retrieval(r["bundle"], r["facts"], C.SMOKE_INIT, C.SMOKE_LEADS, data_config()["target_domain"])
    rep.update(error_message_if_any=None, verification=v, members_found=v["members_found"],
               available_leads=v["available_leads"], grid=v["grid"], units=v["units"], domain=v["domain"],
               retrieval_size_bytes=r["bytes"], retrieval_seconds=r["seconds"], grib_facts=r["facts"],
               ensemble_note=r["bundle"].metadata.notes[0])
    cov = {}
    for y in C.COVERAGE_YEARS:
        init = f"{y}-07-15T00:00"
        c = C.timed_retrieve(prov, init, C.COVERAGE_LEADS)
        if c["ok"]:
            vv = C.verify_retrieval(c["bundle"], c["facts"], init, C.COVERAGE_LEADS, data_config()["target_domain"])
            cov[str(y)] = {"init": init, "success": vv["all_passed"], "members_found": vv["members_found"],
                           "missing_leads": vv["missing_leads"], "missing_fields": [k for k, ok in vv["checks"].items() if not ok],
                           "nan_fraction": vv["nan_fraction"], "retrieval_size_bytes": c["bytes"], "retrieval_seconds": c["seconds"]}
        else:
            cov[str(y)] = {"init": init, "success": False, "error": c["error"], "retrieval_seconds": c["seconds"]}
    rep["coverage"] = cov
    # B2 smoke inference on the real smoke case (frozen ECMWF-trained B2; calibration labelled as transfer)
    from forecast_bust.b2_provider import run_b2, verify_bundle
    t0 = time.perf_counter()
    out = run_b2(r["bundle"], "b2_ncmrwf")
    wall = time.perf_counter() - t0
    lab = verify_bundle(r["bundle"])
    rows = out["rows"]
    rep["b2_smoke"] = {"ran": True, "rows": int(len(rows)), "wall_seconds": round(wall, 3), "timings_ms": out["timings_ms"],
                       "peak_rss_mb_process": round(peak_rss_mb(), 1),
                       "probability_range": [float(rows["b2_probability"].min()), float(rows["b2_probability"].max())],
                       "spread_m_range": [float(rows["spread_m"].min()), float(rows["spread_m"].max())],
                       "provenance": out["provenance"],
                       "real_verification_labels": {"rows": int(len(lab)), "busts": int(lab["bust"].sum()),
                                                    "note": "ONE initialisation - an alignment demonstration, not a benchmark"}}
    years_ok = [y for y, c in cov.items() if c.get("success")]
    full = v["all_passed"] and len(years_ok) == len(C.COVERAGE_YEARS)
    rep["status"] = ("REAL NCMRWF/TIGGE B2 PATH VERIFIED" if full else
                     "PARTIAL NCMRWF COVERAGE VERIFIED" if v["all_passed"] else
                     "NCMRWF/TIGGE CATALOGUE CONFIRMED, RETRIEVAL NOT YET VERIFIED")
    OUT.write_text(json.dumps(clean_json(rep), indent=1, default=str))
    update_dataset_manifest(rep)
    print(rep["status"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
