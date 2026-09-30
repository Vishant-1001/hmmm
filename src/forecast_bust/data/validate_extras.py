"""Integrity check of the wind/MSLP extras cache before it is used (Gate: never call the data complete
because a download process exited).

    python -m forecast_bust.data.validate_extras      -> artifacts/extras_validation.json, exit 1 on failure

For every cached geopotential block `data/cache/ens/block_k.nc` the extras block `ens_extra/block_k.nc`
must exist, open, and match it exactly in initialisation times, grid and lead hours; hold u/v (500/700/850
hPa) and MSLP ensemble mean/std from 50 members; be finite; and lie in physically plausible ranges.
The ERA5 extras climatology is checked the same way.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

from forecast_bust.config import ARTIFACT_DIR, cache_dir, clean_json, data_config

VARS = [f"{v}_{s}" for v in ("u", "v", "mslp") for s in ("mean", "std")]
# physically plausible bounds (m/s, Pa); std must be non-negative
RANGES = {"u_mean": (-120, 120), "v_mean": (-120, 120), "mslp_mean": (85_000, 110_000),
          "u_std": (0, 60), "v_std": (0, 60), "mslp_std": (0, 6_000)}


def check_block(extra: Path, geo: Path, cfg: dict) -> list[str]:
    """Problems found in one extras block (empty list = valid)."""
    if not extra.exists():
        return ["missing"]
    try:
        e = xr.open_dataset(extra)
        g = xr.open_dataset(geo)
    except Exception as exc:  # unreadable / truncated file
        return [f"unreadable: {exc!r}"]
    errs = []
    with e, g:
        missing = [v for v in VARS if v not in e.data_vars]
        if missing:
            errs.append(f"variables missing: {missing}")
        if not np.array_equal(e["time"].values, g["time"].values):
            errs.append("init times differ from geopotential block")
        for c in ("latitude", "longitude"):
            if not np.allclose(e[c].values, g[c].values):
                errs.append(f"{c} grid differs from geopotential block")
        if list(e["prediction_timedelta"].values) != list(cfg["lead_hours"]):
            errs.append("lead hours differ from config")
        if list(e["level"].values) != list(cfg["levels"]):
            errs.append("levels differ from config")
        if int(e.attrs.get("n_members", -1)) != 50:
            errs.append(f"n_members={e.attrs.get('n_members')}")
        for v in VARS:
            if v not in e.data_vars:
                continue
            x = e[v].values
            if not np.isfinite(x).all():
                errs.append(f"{v}: {int((~np.isfinite(x)).sum())} non-finite values")
                continue
            lo, hi = RANGES[v]
            if x.min() < lo or x.max() > hi:
                errs.append(f"{v}: range [{x.min():.1f}, {x.max():.1f}] outside [{lo}, {hi}]")
    return errs


def validate(root: Path | None = None) -> dict:
    cfg = data_config()
    root = root or cache_dir()
    geo = sorted((root / "ens").glob("block_*.nc"))
    bad, n_ok = {}, 0
    for g in geo:
        errs = check_block(root / "ens_extra" / g.name, g, cfg)
        if errs:
            bad[g.name] = errs
        else:
            n_ok += 1
    clim = root / "era5_clim_1990_2017_extra.nc"
    clim_errs = []
    if not clim.exists():
        clim_errs.append("missing")
    else:
        with xr.open_dataset(clim) as c:
            for v in ("u", "v", "mslp"):
                if v not in c.data_vars:
                    clim_errs.append(f"{v} missing")
                elif not np.isfinite(c[v].values).all():
                    clim_errs.append(f"{v} non-finite")
    orphans = sorted({p.name for p in (root / "ens_extra").glob("block_*.nc")} - {g.name for g in geo})
    report = {"expected_blocks": len(geo), "valid_blocks": n_ok, "invalid_or_missing": bad,
              "orphan_extra_blocks": orphans, "climatology_errors": clim_errs,
              "complete": len(geo) > 0 and n_ok == len(geo) and not clim_errs}
    return report


if __name__ == "__main__":
    rep = validate()
    out = ARTIFACT_DIR / "extras_validation.json"
    out.write_text(json.dumps(clean_json({**rep, "invalid_or_missing": dict(list(rep["invalid_or_missing"].items())[:50]),
                                          "n_invalid_or_missing": len(rep["invalid_or_missing"])}), indent=1))
    print(f"extras: {rep['valid_blocks']}/{rep['expected_blocks']} valid; complete={rep['complete']}; "
          f"climatology errors={rep['climatology_errors']}; report {out}")
    sys.exit(0 if rep["complete"] else 1)
