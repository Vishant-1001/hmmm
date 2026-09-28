"""Real-data access to WeatherBench 2 (public GCS bucket, anonymous read).

Downloads are restricted to the context domain and cached locally as NetCDF so that
the ~1 MB/s link is only paid once. Nothing here fabricates data: if a download
fails, the block is simply missing and the pipeline reports it.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import time
from pathlib import Path

import numpy as np
import xarray as xr

from forecast_bust.config import ARTIFACT_DIR, cache_dir, data_config

log = logging.getLogger(__name__)


def _open(store: str) -> xr.Dataset:
    cfg = data_config()
    return xr.open_zarr(store, storage_options=cfg["source"]["storage_options"], chunks=None)


def open_forecast_store() -> xr.Dataset:
    return _open(data_config()["source"]["forecast_store"])


def open_reference_store() -> xr.Dataset:
    return _open(data_config()["source"]["reference_store"])


def open_climatology_store() -> xr.Dataset:
    return _open(data_config()["source"]["climatology_store"])


def lead_to_hours(values) -> np.ndarray:
    """WB2 stores prediction_timedelta either as timedelta64 or as integer hours."""
    v = np.asarray(values)
    if np.issubdtype(v.dtype, np.timedelta64):
        return (v / np.timedelta64(1, "h")).astype(int)
    return v.astype(int)


def subset_domain(ds: xr.Dataset | xr.DataArray, dom: dict):
    lat = ds["latitude"]
    lon = ds["longitude"]
    return ds.isel(
        latitude=np.where((lat >= dom["lat_min"]) & (lat <= dom["lat_max"]))[0],
        longitude=np.where((lon >= dom["lon_min"]) & (lon <= dom["lon_max"]))[0],
    )


def block_order(n_blocks: int, stride: int) -> list[int]:
    """Coarse-to-fine download order so a partial download is evenly spread in time."""
    wanted = list(range(0, n_blocks, stride))
    order, seen = [], set()
    for step in (stride * 8, stride * 4, stride * 2, stride):
        for k in wanted:
            if k % step == 0 and k not in seen:
                order.append(k)
                seen.add(k)
    return order


def block_path(k: int) -> Path:
    p = cache_dir() / "ens"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"block_{k:04d}.nc"


def download_forecast_block(ds: xr.Dataset, k: int) -> Path:
    """Download initialisations [4k, 4k+4) for Day 1-10 leads, all members, 3 levels."""
    cfg = data_config()
    out = block_path(k)
    if out.exists():
        return out
    n = cfg["init_chunk_size"]
    leads = lead_to_hours(ds["prediction_timedelta"].values)
    lead_idx = [int(np.where(leads == h)[0][0]) for h in cfg["lead_hours"]]
    sub = ds[cfg["variable"]].isel(time=slice(k * n, (k + 1) * n), prediction_timedelta=lead_idx)
    sub = sub.sel(level=cfg["levels"])
    sub = subset_domain(sub, cfg["context_domain"]).astype("float32")
    arr = sub.load()
    arr = arr.assign_coords(prediction_timedelta=np.asarray(cfg["lead_hours"], dtype=np.int32))
    arr.attrs.update(units="m**2 s**-2", source=cfg["source"]["forecast_store"],
                     lead_units="hours", downloaded=dt.datetime.utcnow().isoformat())
    tmp = out.with_suffix(".tmp")
    arr.to_dataset(name="z").to_netcdf(tmp)
    tmp.rename(out)
    return out


def download_forecasts(max_blocks: int | None = None) -> None:
    cfg = data_config()
    ds = open_forecast_store()
    n_blocks = ds.sizes["time"] // cfg["init_chunk_size"]
    order = block_order(n_blocks, cfg["block_stride"])
    if max_blocks:
        order = order[:max_blocks]
    for i, k in enumerate(order):
        if block_path(k).exists():
            continue
        for attempt in range(5):
            try:
                t0 = time.time()
                download_forecast_block(ds, k)
                log.info("block %d (%d/%d) in %.0fs", k, i + 1, len(order), time.time() - t0)
                print(f"block {k} ({i + 1}/{len(order)}) {time.time() - t0:.0f}s", flush=True)
                break
            except Exception as e:  # network errors: retry with backoff
                print(f"block {k} attempt {attempt} failed: {e!r}", flush=True)
                time.sleep(10 * (attempt + 1))


def reference_path() -> Path:
    return cache_dir() / "era5_z.nc"


def download_reference() -> Path:
    """ERA5 geopotential (3 levels) for all valid times needed by the 2018-2022 forecasts."""
    cfg = data_config()
    out = reference_path()
    if out.exists():
        return out
    ds = open_reference_store()
    sub = ds[cfg["variable"]].sel(time=slice("2017-12-01", "2023-01-10T18:00"), level=cfg["levels"])
    sub = subset_domain(sub, cfg["context_domain"]).astype("float32").load()
    sub.attrs.update(source=cfg["source"]["reference_store"], downloaded=dt.datetime.utcnow().isoformat())
    sub.to_dataset(name="z").to_netcdf(out)
    return out


def climatology_path() -> Path:
    return cache_dir() / "era5_clim_1990_2017_z.nc"


def download_climatology() -> Path:
    """ERA5 1990-2017 hour-of-day x day-of-year climatology (pre-dates all experiment data)."""
    cfg = data_config()
    out = climatology_path()
    if out.exists():
        return out
    ds = open_climatology_store()
    sub = ds[cfg["variable"]].sel(level=cfg["levels"])
    sub = subset_domain(sub, cfg["context_domain"]).astype("float32").load()
    sub.attrs.update(source=cfg["source"]["climatology_store"], downloaded=dt.datetime.utcnow().isoformat())
    sub.to_dataset(name="z").to_netcdf(out)
    return out


EXTRA_VARS = {"u": "u_component_of_wind", "v": "v_component_of_wind", "mslp": "mean_sea_level_pressure"}


def extra_block_path(k: int) -> Path:
    p = cache_dir() / "ens_extra"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"block_{k:04d}.nc"


def download_extra_block(ds: xr.Dataset, k: int) -> Path:
    """Ensemble mean and std (ddof=1) of u, v (3 levels) and MSLP for the same initialisations,
    leads and context domain as geopotential block k. Members are reduced at download time
    because only these statistics are used (keeps the cache ~25x smaller)."""
    cfg = data_config()
    out = extra_block_path(k)
    if out.exists():
        return out
    n = cfg["init_chunk_size"]
    leads = lead_to_hours(ds["prediction_timedelta"].values)
    lead_idx = [int(np.where(leads == h)[0][0]) for h in cfg["lead_hours"]]
    data = {}
    for short, name in EXTRA_VARS.items():
        a = ds[name].isel(time=slice(k * n, (k + 1) * n), prediction_timedelta=lead_idx)
        if "level" in a.dims:
            a = a.sel(level=cfg["levels"])
        a = subset_domain(a, cfg["context_domain"]).astype("float32").load()
        data[f"{short}_mean"] = a.mean("number")
        data[f"{short}_std"] = a.std("number", ddof=1)
        data[f"{short}_mean"].attrs["units"] = a.attrs.get("units", "")
    out_ds = xr.Dataset(data).assign_coords(prediction_timedelta=np.asarray(cfg["lead_hours"], dtype=np.int32))
    out_ds.attrs.update(source=cfg["source"]["forecast_store"], lead_units="hours", n_members=int(a.sizes["number"]),
                        downloaded=dt.datetime.now(dt.timezone.utc).isoformat())
    tmp = out.with_suffix(".tmp")
    out_ds.to_netcdf(tmp)
    tmp.rename(out)
    return out


def download_extras() -> None:
    """Extra variables for exactly the geopotential blocks already cached (same order, resumable)."""
    cfg = data_config()
    ds = open_forecast_store()
    n_blocks = ds.sizes["time"] // cfg["init_chunk_size"]
    order = [k for k in block_order(n_blocks, cfg["block_stride"]) if block_path(k).exists()]
    for i, k in enumerate(order):
        if extra_block_path(k).exists():
            continue
        for attempt in range(5):
            try:
                t0 = time.time()
                download_extra_block(ds, k)
                print(f"extra block {k} ({i + 1}/{len(order)}) {time.time() - t0:.0f}s", flush=True)
                break
            except Exception as e:  # network errors: retry with backoff
                print(f"extra block {k} attempt {attempt} failed: {e!r}", flush=True)
                time.sleep(10 * (attempt + 1))


def extra_climatology_path() -> Path:
    return cache_dir() / "era5_clim_1990_2017_extra.nc"


def download_extra_climatology() -> Path:
    cfg = data_config()
    out = extra_climatology_path()
    if out.exists():
        return out
    ds = open_climatology_store()
    parts = {}
    for short, name in EXTRA_VARS.items():
        a = ds[name]
        if "level" in a.dims:
            a = a.sel(level=cfg["levels"])
        parts[short] = subset_domain(a, cfg["context_domain"]).astype("float32").load()
    sub = xr.Dataset(parts)
    sub.attrs.update(source=cfg["source"]["climatology_store"], downloaded=dt.datetime.now(dt.timezone.utc).isoformat())
    sub.to_netcdf(out)
    return out


def write_manifest_entry(key: str, entry: dict) -> None:
    path = ARTIFACT_DIR / "dataset_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(path.read_text()) if path.exists() else {}
    manifest[key] = entry
    path.write_text(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Download WB2 subsets to the local cache")
    ap.add_argument("what", choices=["forecasts", "reference", "climatology", "extras", "extra_climatology"])
    ap.add_argument("--max-blocks", type=int, default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    if a.what == "forecasts":
        download_forecasts(a.max_blocks)
    elif a.what == "reference":
        print(download_reference())
    elif a.what == "climatology":
        print(download_climatology())
    elif a.what == "extra_climatology":
        print(download_extra_climatology())
    else:
        download_extras()
