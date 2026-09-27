"""GATE 1 real-data smoke test.

Uses REAL WeatherBench 2 data only: one IFS ENS initialisation, one lead day, Z500,
the target domain, ensemble mean and spread, the matching ERA5 reference at the
valid time, and a regional verification error. Prints actual values and writes
artifacts/smoke_test.json.

    python scripts/smoke_test.py [--init 2018-01-01T00] [--lead 72]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json

import numpy as np

from forecast_bust.config import ARTIFACT_DIR, data_config
from forecast_bust.data import wb2
from forecast_bust.data.regions import build_regions, region_members
from forecast_bust.verification.alignment import valid_time
from forecast_bust.verification.metrics import weighted_rmse

G = 9.80665


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="2018-01-01T00")
    ap.add_argument("--lead", type=int, default=72)
    a = ap.parse_args()
    cfg = data_config()
    init = np.datetime64(a.init, "ns")

    ens = wb2.open_forecast_store()
    leads = wb2.lead_to_hours(ens["prediction_timedelta"].values)
    print("forecast store:", cfg["source"]["forecast_store"])
    print("prediction_timedelta attrs:", dict(ens["prediction_timedelta"].attrs), "first values:", leads[:4])
    ti = int(np.where(ens["time"].values == init)[0][0])
    li = int(np.where(leads == a.lead)[0][0])
    z = ens["geopotential"].isel(time=ti, prediction_timedelta=li).sel(level=500)
    z = wb2.subset_domain(z, cfg["target_domain"]).load()
    print("units:", z.attrs.get("units"), "members:", z.sizes["number"], "grid:", z.sizes)
    zm = z.mean("number") / G
    zs = z.std("number", ddof=1) / G

    vt = valid_time(init, a.lead)
    era = wb2.open_reference_store()["geopotential"].sel(time=vt, level=500)
    era = wb2.subset_domain(era, cfg["target_domain"]).load() / G
    assert np.array_equal(era.latitude.values, zm.latitude.values)
    assert np.array_equal(era.longitude.values, zm.longitude.values)

    lats = zm.latitude.values
    lons = zm.longitude.values
    domain_rmse = float(weighted_rmse(zm.values, era.values, lats))
    regions = build_regions(lats, lons, cfg["target_domain"])
    rows = []
    for r in regions:
        li_, oi_ = region_members(r, lats, lons)
        f = zm.values[np.ix_(oi_, li_)]
        o = era.values[np.ix_(oi_, li_)]
        rows.append({
            "region_id": r.region_id, "name": r.name, "lat": r.lat, "lon": r.lon,
            "ens_mean_z500_m": float(f.mean()), "ens_spread_z500_m": float(zs.values[np.ix_(oi_, li_)].mean()),
            "era5_z500_m": float(o.mean()), "rmse_m": float(weighted_rmse(f, o, lats[li_], lat_spacing=5.625)),
        })
    worst = sorted(rows, key=lambda x: -x["rmse_m"])[:5]
    print(f"init={init} lead={a.lead}h valid={vt} (ERA5 time matched: {era.time.values})")
    print(f"domain Z500 ens-mean range {float(zm.min()):.1f}..{float(zm.max()):.1f} m; "
          f"mean spread {float(zs.mean()):.2f} m; domain RMSE vs ERA5 {domain_rmse:.2f} m")
    for w in worst:
        print(f"  {w['region_id']} {w['name']:<28} lat {w['lat']:7.3f} lon {w['lon']:7.3f} "
              f"mean {w['ens_mean_z500_m']:.1f} ERA5 {w['era5_z500_m']:.1f} spread {w['ens_spread_z500_m']:.2f} RMSE {w['rmse_m']:.2f} m")
    out = {
        "gate": "GATE 1 real-data smoke test", "passed": True,
        "run_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "forecast_store": cfg["source"]["forecast_store"], "reference_store": cfg["source"]["reference_store"],
        "init_time": str(init), "lead_hours": a.lead, "valid_time": str(vt), "variable": "Z500 (geopotential/9.80665, m)",
        "n_members": int(z.sizes["number"]), "n_regions": len(rows), "domain_rmse_m": domain_rmse,
        "domain_mean_spread_m": float(zs.mean()), "regions": rows,
    }
    ARTIFACT_DIR.mkdir(exist_ok=True)
    (ARTIFACT_DIR / "smoke_test.json").write_text(json.dumps(out, indent=2))
    print("wrote artifacts/smoke_test.json")


if __name__ == "__main__":
    main()
