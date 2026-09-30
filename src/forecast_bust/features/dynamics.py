"""Feature Group B completion (v2): wind, vorticity, divergence, shear and MSLP.

Source: the SAME WeatherBench 2 IFS ENS 64x32 store and initialisations as the geopotential
features (`python -m forecast_bust.data.wb2 extras`), reduced at download time to the
ensemble mean and standard deviation (ddof=1) of u, v (500/700/850 hPa) and mean sea-level
pressure. Anomalies use the ERA5 1990-2017 climatology (pre-dates all experiment years).
Only forecast quantities are read; no verification data is involved.

Kinematics are computed from the ensemble-mean wind on the 5.625 deg grid with centred
differences in spherical coordinates:

    vorticity  = (dv/dlon - d(u cos(lat))/dlat) / (a cos(lat))
    divergence = (du/dlon + d(v cos(lat))/dlat) / (a cos(lat))

Vorticity and divergence are linear in the wind, so the value from the ensemble-mean wind
equals the ensemble mean of member values. At 5.625 deg they describe the resolved
synoptic/planetary scale only (docs/limitations.md).
"""
from __future__ import annotations

import glob
import logging

import numpy as np
import pandas as pd
import xarray as xr

from forecast_bust.config import REPO_ROOT, cache_dir
from forecast_bust.data import wb2
from forecast_bust.data.assemble import clim_at
from forecast_bust.data.regions import region_members
from forecast_bust.labels.build import regions_for
from forecast_bust.verification.alignment import valid_time

log = logging.getLogger(__name__)
A_EARTH = 6.371e6
DYN = ["u500_anom", "v500_anom", "ws500", "ws850", "shear_500_850", "vort500", "vort850", "div500", "div850",
       "mslp_anom", "mslp_grad", "wspread500", "wspread850", "mslp_spread"]
# Flow tendency inside the SAME forecast (rate of change of the predicted state around the target
# lead, np.gradient over lead days: centred in Days 2-9, one-sided at Days 1 and 10). Magnitudes only:
# "rapidly evolving systems" in the SIH statement. Available at initialisation (no other cycle used).
TEND = ["z500_tend", "mslp_tend", "vort500_tend"]
DYN = DYN + TEND


def extras_path():
    p = REPO_ROOT / "data" / "interim"
    p.mkdir(parents=True, exist_ok=True)
    return p / "extras.nc"


def assemble_extras(inits: np.ndarray) -> xr.Dataset:
    """Concatenate cached extra blocks, restricted to (and ordered as) `inits`, with climatology."""
    files = sorted(glob.glob(str(cache_dir() / "ens_extra" / "block_*.nc")))
    if not files:
        raise FileNotFoundError("no extra-variable blocks; run scripts/download_extras.sh")
    ds = xr.concat([xr.open_dataset(f).load() for f in files], dim="time").sortby("time")
    ds = ds.rename(time="init", prediction_timedelta="lead")
    missing = np.setdiff1d(np.asarray(inits, dtype="datetime64[ns]"), ds.init.values.astype("datetime64[ns]"))
    if len(missing):
        raise ValueError(f"{len(missing)} initialisations lack extra variables (first {missing[:3]}); "
                         "finish scripts/download_extras.sh")
    ds = ds.sel(init=inits)
    clim = xr.open_dataset(wb2.extra_climatology_path()).load()
    vt = valid_time(ds.init.values[:, None], ds.lead.values[None, :])
    lv = [int(x) for x in ds.level.values]
    ds["u_clim"] = (("init", "lead", "level", "longitude", "latitude"),
                    clim_at(clim["u"].sel(level=lv).transpose("hour", "dayofyear", "level", "longitude", "latitude"), vt))
    ds["v_clim"] = (("init", "lead", "level", "longitude", "latitude"),
                    clim_at(clim["v"].sel(level=lv).transpose("hour", "dayofyear", "level", "longitude", "latitude"), vt))
    ds["mslp_clim"] = (("init", "lead", "longitude", "latitude"),
                       clim_at(clim["mslp"].transpose("hour", "dayofyear", "longitude", "latitude"), vt))
    for v in ("u_mean", "u_std", "v_mean", "v_std"):
        ds[v] = ds[v].transpose("init", "lead", "level", "longitude", "latitude")
    for v in ("mslp_mean", "mslp_std"):
        ds[v] = ds[v].transpose("init", "lead", "longitude", "latitude")
    ds.to_netcdf(extras_path())
    return ds


def kinematics(u: np.ndarray, v: np.ndarray, lat: np.ndarray, lon: np.ndarray):
    """Relative vorticity and divergence (1e-5 s^-1) of (..., lon, lat) wind fields.
    Edge rows/columns of the cached context domain are NaN (one-sided differences not used)."""
    phi = np.deg2rad(lat)
    cos = np.cos(phi)
    dlam = np.deg2rad(np.median(np.diff(lon)))
    dphi = np.deg2rad(np.median(np.diff(lat)))
    dv_dlam = np.gradient(v, dlam, axis=-2)
    du_dlam = np.gradient(u, dlam, axis=-2)
    duc_dphi = np.gradient(u * cos, dphi, axis=-1)
    dvc_dphi = np.gradient(v * cos, dphi, axis=-1)
    vort = (dv_dlam - duc_dphi) / (A_EARTH * cos) * 1e5
    div = (du_dlam + dvc_dphi) / (A_EARTH * cos) * 1e5
    for f in (vort, div):
        f[..., 0, :] = f[..., -1, :] = f[..., :, 0] = f[..., :, -1] = np.nan
    return vort, div


def tendency_fields(ds: xr.Dataset, mslp_hpa: np.ndarray, vort500: np.ndarray) -> dict:
    """|d/dt| per day of ensemble-mean Z500 (m), MSLP (hPa) and 500 hPa vorticity (1e-5 s^-1), along
    the lead axis of each forecast (axis 1 of (init, lead, lon, lat)); leads must be 24 h apart."""
    if not np.all(np.diff(ds.lead.values.astype(int)) == 24):
        raise ValueError("tendencies need consecutive 24 h leads")
    z500 = ds["ens_mean"].values[:, :, int(np.where(ds.level.values == 500)[0][0])]
    return {"z500_tend": np.abs(np.gradient(z500, axis=1)), "mslp_tend": np.abs(np.gradient(mslp_hpa, axis=1)),
            "vort500_tend": np.abs(np.gradient(vort500, axis=1))}


def dyn_features(ex: xr.Dataset, ds: xr.Dataset) -> pd.DataFrame:
    """Per (init, lead, region) DYN features; `ds` is the geopotential state set (same grid/inits)."""
    if not (np.array_equal(ex.init.values, ds.init.values) and np.array_equal(ex.latitude.values, ds.latitude.values)
            and np.array_equal(ex.longitude.values, ds.longitude.values)
            and np.array_equal(ex.lead.values, ds.lead.values)):
        raise ValueError("extra-variable dataset is not aligned with the geopotential states")
    lat, lon = ds.latitude.values, ds.longitude.values
    lv = {int(x): i for i, x in enumerate(ex.level.values)}
    um, vm = ex["u_mean"].values, ex["v_mean"].values
    us, vs = ex["u_std"].values, ex["v_std"].values
    uc, vc = ex["u_clim"].values, ex["v_clim"].values
    f = {}
    for p in (500, 850):
        i = lv[p]
        f[f"ws{p}"] = np.hypot(um[:, :, i], vm[:, :, i])
        f[f"vort{p}"], f[f"div{p}"] = kinematics(um[:, :, i], vm[:, :, i], lat, lon)
        f[f"wspread{p}"] = np.sqrt(us[:, :, i] ** 2 + vs[:, :, i] ** 2)
    f["u500_anom"] = um[:, :, lv[500]] - uc[:, :, lv[500]]
    f["v500_anom"] = vm[:, :, lv[500]] - vc[:, :, lv[500]]
    f["shear_500_850"] = np.hypot(um[:, :, lv[500]] - um[:, :, lv[850]], vm[:, :, lv[500]] - vm[:, :, lv[850]])
    mslp = ex["mslp_mean"].values / 100.0  # hPa
    f["mslp_anom"] = mslp - ex["mslp_clim"].values / 100.0
    r = 6371.0
    gx = np.gradient(mslp, axis=-2) / (np.deg2rad(np.median(np.diff(lon))) * r * np.cos(np.deg2rad(lat))) * 100
    gy = np.gradient(mslp, axis=-1) / (np.deg2rad(np.median(np.diff(lat))) * r) * 100
    f["mslp_grad"] = np.hypot(gx, gy)  # hPa per 100 km
    f["mslp_spread"] = ex["mslp_std"].values / 100.0
    f.update(tendency_fields(ds, mslp, f["vort500"]))
    inits, leads = ds.init.values, ds.lead.values
    n_i, n_l = len(inits), len(leads)
    frames = []
    for reg in regions_for(ds):
        li, oi = region_members(reg, lat, lon)
        frame = pd.DataFrame({k: v[:, :, oi[:, None], li[None, :]].mean(axis=(-2, -1)).astype(np.float32).ravel()
                              for k, v in f.items()})
        frame["init_time"] = np.repeat(inits, n_l)
        frame["lead_hours"] = np.tile(leads, n_i)
        frame["region_id"] = reg.region_id
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)[DYN + ["init_time", "lead_hours", "region_id"]]


def add_dyn_features(df: pd.DataFrame, ds: xr.Dataset) -> pd.DataFrame:
    ex = assemble_extras(ds.init.values)
    ex = ex.assign_coords(lead=ex.lead.values.astype(ds.lead.dtype))
    gf = dyn_features(ex, ds)
    return df.merge(gf, on=["init_time", "lead_hours", "region_id"], how="left", validate="one_to_one")
