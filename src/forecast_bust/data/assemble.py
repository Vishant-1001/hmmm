"""Assemble cached real WB2 blocks into a compact forecast-state dataset.

Output `data/interim/states.nc` (all in metres of geopotential height, Z/g):

  ens_mean(init, lead, level, longitude, latitude)   ensemble mean, context domain
  ens_std (init, lead, level, longitude, latitude)   ensemble std (ddof=1)
  clim    (init, lead, level, longitude, latitude)   ERA5 1990-2017 climatology at valid time
  era5_z500(init, lead, longitude, latitude)         ERA5 verification at valid time (NOT a feature)
  m_iqr, m_p10p90, m_skew, m_sign_agree (init, lead, rlon, rlat)  member statistics, target boxes
  era5_rank(init, lead, rlon, rlat)                  rank of ERA5 among members (diagnostic only)

Verification quantities are stored side by side with forecast quantities for
convenience; the feature builders only read ens_*, clim and m_* (see
docs/leakage_controls.md and tests/test_leakage.py).
"""
from __future__ import annotations

import glob
import logging

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

from forecast_bust.config import REPO_ROOT, cache_dir, data_config
from forecast_bust.data import wb2
from forecast_bust.verification.alignment import align_reference, valid_time

log = logging.getLogger(__name__)
G = 9.80665


def states_path():
    p = REPO_ROOT / "data" / "interim"
    p.mkdir(parents=True, exist_ok=True)
    return p / "states.nc"


def clim_at(clim: xr.DataArray, vt: np.ndarray) -> np.ndarray:
    """Climatology values at valid times vt (any shape) -> vt.shape + (level, lon, lat)."""
    t = pd.DatetimeIndex(np.ravel(vt))
    hi = clim.indexes["hour"].get_indexer(t.hour)
    di = clim.indexes["dayofyear"].get_indexer(t.dayofyear)
    if (hi < 0).any() or (di < 0).any():
        raise KeyError("climatology lookup failed")
    vals = clim.values[hi, di]
    return vals.reshape(np.shape(vt) + vals.shape[1:])


def member_stats(members: np.ndarray, clim_box: np.ndarray, ref: np.ndarray) -> dict:
    """members: (member, ...); clim_box/ref broadcast to (...)."""
    mean = members.mean(0)
    q10, q25, q75, q90 = np.percentile(members, [10, 25, 75, 90], axis=0)
    anom_sign = np.sign(members - clim_box[None])
    mean_sign = np.sign(mean - clim_box)
    return {
        "m_iqr": q75 - q25,
        "m_p10p90": q90 - q10,
        "m_skew": stats.skew(members, axis=0, bias=False),
        "m_sign_agree": (anom_sign == mean_sign[None]).mean(0),
        "era5_rank": (members < ref[None]).sum(0).astype(np.int16),
    }


def assemble() -> xr.Dataset:
    cfg = data_config()
    files = sorted(glob.glob(str(cache_dir() / "ens" / "block_*.nc")))
    if not files:
        raise FileNotFoundError("no cached forecast blocks; run scripts/download_all.sh")
    era = xr.open_dataset(wb2.reference_path())["z"].sel(level=cfg["target_level"]) / G
    clim = xr.open_dataset(wb2.climatology_path())["z"] / G
    dom = cfg["target_domain"]
    parts = []
    for f in files:
        b = xr.open_dataset(f)["z"].load() / G  # (time, number, lead, level, lon, lat)
        b = b.transpose("time", "number", "prediction_timedelta", "level", "longitude", "latitude")
        if not np.array_equal(b.latitude.values, era.latitude.values) or not np.array_equal(
                b.longitude.values, era.longitude.values):
            raise ValueError(f"grid mismatch between {f} and ERA5 cache")
        inits = b.time.values
        leads = b.prediction_timedelta.values
        vt = valid_time(inits[:, None], leads[None, :])  # (init, lead)
        try:
            idx = align_reference(era.time.values, vt)
        except KeyError as e:
            log.warning("skipping %s: %s", f, e)
            continue
        ref = era.values[idx]  # (init, lead, lon, lat)
        cl = clim_at(clim.sel(level=b.level.values), vt)  # (init, lead, level, lon, lat)
        if np.isnan(b.values).any():
            log.warning("%s contains NaN members; they are excluded from statistics", f)
        mean = np.nanmean(b.values, axis=1)
        std = np.nanstd(b.values, axis=1, ddof=1)
        lat = b.latitude.values
        lon = b.longitude.values
        li = np.where((lat >= dom["lat_min"]) & (lat <= dom["lat_max"]))[0]
        oi = np.where((lon >= dom["lon_min"]) & (lon <= dom["lon_max"]))[0]
        lvl500 = int(np.where(b.level.values == cfg["target_level"])[0][0])
        mem_t = b.values[:, :, :, lvl500][..., oi, :][..., li]  # (init, member, lead, rlon, rlat)
        ms = member_stats(np.moveaxis(mem_t, 1, 0), cl[:, :, lvl500][..., oi, :][..., li],
                          ref[..., oi, :][..., li])
        coords = {"init": inits, "lead": leads, "level": b.level.values, "longitude": lon, "latitude": lat,
                  "rlon": lon[oi], "rlat": lat[li]}
        dv = {
            "ens_mean": (("init", "lead", "level", "longitude", "latitude"), mean.astype("float32")),
            "ens_std": (("init", "lead", "level", "longitude", "latitude"), std.astype("float32")),
            "clim": (("init", "lead", "level", "longitude", "latitude"), cl.astype("float32")),
            "era5_z500": (("init", "lead", "longitude", "latitude"), ref.astype("float32")),
            "n_members": (("init",), np.isfinite(b.values[:, :, 0, 0, 0, 0]).sum(1).astype("int16")),
        }
        for k, v in ms.items():
            dv[k] = (("init", "lead", "rlon", "rlat"), v.astype("float32" if k != "era5_rank" else "int16"))
        parts.append(xr.Dataset(dv, coords=coords))
    ds = xr.concat(parts, dim="init").sortby("init")
    ds.attrs.update(
        forecast_source=cfg["source"]["forecast_store"], reference_source=cfg["source"]["reference_store"],
        climatology_source=cfg["source"]["climatology_store"], units="m (geopotential / 9.80665)",
        n_blocks=len(parts),
    )
    out = states_path()
    tmp = out.with_suffix(".tmp.nc")
    ds.to_netcdf(tmp)
    tmp.rename(out)
    log.info("assembled %d inits from %d blocks -> %s", ds.sizes["init"], len(parts), out)
    return ds


def load_states() -> xr.Dataset:
    return xr.open_dataset(states_path())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    d = assemble()
    print(d)
