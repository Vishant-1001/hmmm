"""Integration dry run of the v2 DYN features on whatever extras blocks are already cached.

    .venv/bin/python scripts/dryrun_dyn.py

Builds DYN features exactly as the pipeline does (features.dynamics) for the cached initialisations only,
joins them to the existing table rows, and checks coverage, ranges and physical consistency with the
geopotential features. No model is trained and nothing is evaluated.
Physics checks: 500 hPa wind speed must track the Z500 gradient outside the tropics (geostrophic balance);
the MSLP anomaly must track the Z850 anomaly. A lat/lon transposition or a time misalignment breaks both.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from forecast_bust.config import REPO_ROOT, cache_dir  # noqa: E402
from forecast_bust.data.assemble import load_states  # noqa: E402
from forecast_bust.features import dynamics  # noqa: E402


def main() -> None:
    scratch = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/fbs_dryrun_extras.nc")
    dynamics.extras_path = lambda: scratch  # never overwrite data/interim/extras.nc from a dry run
    have = np.concatenate([xr.open_dataset(f)["time"].values
                           for f in sorted((cache_dir() / "ens_extra").glob("block_*.nc"))])
    ds = load_states()
    inits = np.intersect1d(ds.init.values, have)
    ds = ds.sel(init=inits)
    ex = dynamics.assemble_extras(ds.init.values)
    ex = ex.assign_coords(lead=ex.lead.values.astype(ds.lead.dtype))
    gf = dynamics.dyn_features(ex, ds)
    cols = ["init_time", "lead_hours", "region_id", "lat", "grad_mag", "anom850", "spread_m", "split"]
    tab = pd.read_parquet(REPO_ROOT / "data/interim/table.parquet", columns=cols)
    tab = tab[tab.init_time.isin(pd.to_datetime(inits))]
    m = tab.merge(gf, on=["init_time", "lead_hours", "region_id"], how="left", validate="one_to_one")
    ext = m[m.lat.abs() >= 30]
    rep = {
        "initialisations": int(len(inits)), "rows": int(len(m)),
        "rows_without_dyn": int(m[dynamics.DYN[0]].isna().sum()),
        "nan_share": {f: float(m[f].isna().mean()) for f in dynamics.DYN},
        "ranges": {f: [float(m[f].min()), float(m[f].max())] for f in dynamics.DYN},
        "corr_ws500_vs_z500_gradient_extratropics": float(ext[["ws500", "grad_mag"]].corr().iloc[0, 1]),
        "corr_mslp_anom_vs_z850_anom": float(m[["mslp_anom", "anom850"]].corr().iloc[0, 1]),
        "corr_wspread500_vs_z500_spread": float(m[["wspread500", "spread_m"]].corr().iloc[0, 1]),
    }
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
