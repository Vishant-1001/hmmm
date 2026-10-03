"""Forecast providers: one normalized ensemble-forecast contract for every source.

    ForecastProvider
      ECMWFResearchProvider   real ECMWF IFS ENS (WeatherBench 2 archive, cached 5.625 deg blocks)
      NCMRWFTIGGEProvider     real NCMRWF NEPS perturbed members from the ECMWF ECDS TIGGE archive
      SyntheticDemoProvider   seeded synthetic fields: synthetic=True, demo_only=True, never a benchmark input

`retrieve()` returns a `ForecastBundle`: a DataArray that satisfies `adapters.validate_forecast`
(dims time, number, prediction_timedelta, level, longitude, latitude; geopotential in m**2 s**-2;
latitude/longitude ascending, longitude in [0, 360)) plus `ProviderMetadata`. Downstream code
(B2 features, verification) reads only that contract and never branches on the provider.

NCMRWF availability is NOT assumed: `NCMRWFTIGGEProvider.retrieve` needs an ECDS personal access
token and every returned file is checked against its own GRIB metadata (origin, members, level,
steps, units). See scripts/ncmrwf_tigge_check.py and docs/ncmrwf_tigge_availability.md.
"""
from __future__ import annotations

import datetime as dt
import glob
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from forecast_bust.config import REPO_ROOT, cache_dir, data_config
from forecast_bust.data.adapters import normalise_coords, validate_forecast

G = 9.80665
ECDS_API_URL = "https://ecds.ecmwf.int/api"
ECDS_DATASET = "tigge-forecasts"
ECDS_DATASET_URL = "https://ecds.ecmwf.int/datasets/tigge-forecasts"
ECDS_PROCESS_URL = f"{ECDS_API_URL}/retrieve/v1/processes/{ECDS_DATASET}"
# GRIB identifiers of NCMRWF (New Delhi) in TIGGE: MARS origin "dems", WMO centre 29
NCMRWF_GRIB_CENTRES = {"dems", "29"}


@dataclass
class ProviderMetadata:
    provider: str                       # ecmwf_research | ncmrwf_tigge | synthetic
    dataset: str
    source_url: str
    retrieval_timestamp: str
    initialization_time: str
    valid_times: list[str]
    forecast_hours: list[int]
    ensemble_member_count: int
    member_ids: list[int]
    variable: str
    level: list[int]
    grid: str
    units: str
    synthetic: bool
    demo_only: bool
    source_label: str
    native_grid: str = ""
    origin: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ForecastBundle:
    data: xr.DataArray                  # canonical contract, ONE initialisation (time dim of length 1)
    metadata: ProviderMetadata


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def reference_grid() -> tuple[np.ndarray, np.ndarray]:
    """(lat, lon) centres of the 5.625 deg WB2 64x32 grid over the cached context domain - the grid on
    which the ECMWF benchmark, ERA5 verification and B2 regions are defined."""
    ref = xr.open_dataset(cache_dir() / "era5_z.nc")
    return ref.latitude.values.astype(float), ref.longitude.values.astype(float)


def conservative_box_mean(da: xr.DataArray, lat_c: np.ndarray, lon_c: np.ndarray) -> xr.DataArray:
    """Aggregate a finer regular lat/lon field onto coarse cell centres (lat_c, lon_c): cos(lat)-weighted
    mean of the fine points whose centres fall in [centre - d/2, centre + d/2) - first-order conservative
    for fine grids that tile the coarse cells. NaN fine points are excluded; a coarse cell with no valid
    fine point is NaN. Input/output: canonical coordinate conventions (ascending, lon in [0, 360))."""
    dlat = float(np.median(np.diff(lat_c)))
    dlon = float(np.median(np.diff(lon_c)))
    flat, flon = da["latitude"].values, da["longitude"].values
    w_lat = np.cos(np.deg2rad(flat))
    vals = da.transpose(..., "longitude", "latitude").values
    out = np.full(vals.shape[:-2] + (len(lon_c), len(lat_c)), np.nan, dtype=np.float64)
    for j, la in enumerate(lat_c):
        li = np.where((flat >= la - dlat / 2) & (flat < la + dlat / 2))[0]
        if not len(li):
            continue
        for i, lo in enumerate(lon_c):
            oi = np.where((flon >= lo - dlon / 2) & (flon < lo + dlon / 2))[0]
            if not len(oi):
                continue
            block = vals[..., oi[:, None], li[None, :]]
            w = np.broadcast_to(w_lat[li][None, :], block.shape)
            ok = np.isfinite(block)
            ws = np.where(ok, w, 0.0).sum(axis=(-2, -1))
            s = np.where(ok, block * w, 0.0).sum(axis=(-2, -1))
            out[..., i, j] = np.where(ws > 0, s / np.where(ws > 0, ws, 1), np.nan)
    dims = [d for d in da.transpose(..., "longitude", "latitude").dims]
    coords = {d: da[d] for d in dims if d not in ("longitude", "latitude") and d in da.coords}
    coords.update(longitude=lon_c, latitude=lat_c)
    return xr.DataArray(out.astype(np.float32), dims=dims, coords=coords, attrs=dict(da.attrs))


def ensemble_mean_spread(da: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """(mean, std ddof=1, valid-member count) over `number`, NaN members excluded per grid point -
    the same estimator as the ECMWF benchmark assembly (data.assemble)."""
    v = da.values
    axis = da.dims.index("number")
    n = np.isfinite(v).sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nanmean(v, axis=axis)
        std = np.where(n >= 2, np.nanstd(v, axis=axis, ddof=1), np.nan)
    dims = [d for d in da.dims if d != "number"]
    coords = {d: da[d] for d in dims}
    return (xr.DataArray(mean, dims=dims, coords=coords), xr.DataArray(std, dims=dims, coords=coords),
            xr.DataArray(n, dims=dims, coords=coords))


class ForecastProvider(ABC):
    provider: str = "abstract"
    dataset: str = ""
    synthetic: bool = False
    demo_only: bool = False
    source_label: str = ""

    @abstractmethod
    def retrieve(self, init_time: str | pd.Timestamp, lead_hours: list[int], levels: list[int] | None = None) -> ForecastBundle:
        ...

    def _meta(self, da: xr.DataArray, init: pd.Timestamp, source_url: str, grid: str, **kw) -> ProviderMetadata:
        leads = [int(h) for h in da["prediction_timedelta"].values]
        return ProviderMetadata(
            provider=self.provider, dataset=self.dataset, source_url=source_url, retrieval_timestamp=_now(),
            initialization_time=init.isoformat(), valid_times=[(init + pd.Timedelta(hours=h)).isoformat() for h in leads],
            forecast_hours=leads, ensemble_member_count=int(da.sizes["number"]),
            member_ids=[int(m) for m in da["number"].values], variable="geopotential",
            level=[int(x) for x in da["level"].values], grid=grid, units=str(da.attrs.get("units", "")),
            synthetic=self.synthetic, demo_only=self.demo_only, source_label=self.source_label, **kw)


class ECMWFResearchProvider(ForecastProvider):
    """Real ECMWF IFS ENS (50 perturbed members) from the cached WeatherBench 2 blocks used by the benchmark."""
    provider = "ecmwf_research"
    dataset = "ECMWF IFS ENS via WeatherBench 2 (64x32 conservative)"
    source_label = "ECMWF IFS ENS / ERA5 (historical research archive)"
    # Every cached block holds members 1..50 (states.nc n_members min = max = 50); read from data in retrieve()
    ensemble_member_count = 50

    def __init__(self, block_dir: Path | None = None):
        self.block_dir = Path(block_dir or cache_dir() / "ens")

    def _find(self, init: pd.Timestamp) -> Path:
        for f in sorted(glob.glob(str(self.block_dir / "block_*.nc"))):
            with xr.open_dataset(f) as b:
                if np.datetime64(init) in b.time.values:
                    return Path(f)
        raise KeyError(f"initialisation {init} is not in the cached ECMWF blocks ({self.block_dir})")

    def retrieve(self, init_time, lead_hours, levels=None) -> ForecastBundle:
        init = pd.Timestamp(init_time)
        levels = levels or [data_config()["target_level"]]
        with xr.open_dataset(self._find(init)) as b:
            da = b["z"].sel(time=[np.datetime64(init)], prediction_timedelta=list(lead_hours), level=levels).load()
        da.attrs["units"] = "m**2 s**-2"
        da = validate_forecast(normalise_coords(da))
        return ForecastBundle(da, self._meta(da, init, data_config()["source"]["forecast_store"],
                                             data_config()["source"]["grid"], origin="ecmf",
                                             native_grid=data_config()["source"]["grid"]))


class NCMRWFTIGGEProvider(ForecastProvider):
    """NCMRWF NEPS perturbed ensemble members (TIGGE origin `dems`) from the ECMWF Data Store (ECDS).

    Access: cdsapi client against https://ecds.ecmwf.int/api with a personal access token (ECMWF account,
    ECDS terms and the TIGGE licence accepted on the dataset page). Credentials: ~/.cdsapirc
    (`url: https://ecds.ecmwf.int/api`, `key: <token>`) or FBS_ECDS_KEY / CDSAPI_KEY in the environment.
    The returned GRIB is aggregated to the 5.625 deg benchmark grid (`conservative_box_mean`)."""
    provider = "ncmrwf_tigge"
    dataset = "TIGGE (ECMWF ECDS tigge-forecasts), origin NCMRWF (dems)"
    source_label = "NCMRWF NEPS via TIGGE/ECDS"
    # Internal operational description (NCMRWF NEPS documentation): 1 control + 22 perturbed = 23 members.
    # Used for LOGGING ONLY; the archived count is always read from the returned GRIB.
    documented_operational_members = 23

    # A request covering the target domain's 5.625 deg cell bounds (-5.625..39.375 N, 59.06..104.06 E) with padding
    DEFAULT_AREA = [42.0, 56.0, -8.0, 108.0]   # N, W, S, E

    def __init__(self, cache: Path | None = None, key: str | None = None, url: str = ECDS_API_URL):
        self.cache = Path(cache or REPO_ROOT / "data" / "cache" / "ncmrwf_tigge")
        self.url = url
        self.key = key or os.environ.get("FBS_ECDS_KEY") or os.environ.get("CDSAPI_KEY")

    # ---------- access ----------
    def credentials_status(self) -> dict:
        rc = Path(os.environ.get("CDSAPI_RC", Path.home() / ".cdsapirc"))
        rc_url = ""
        if rc.is_file():
            for line in rc.read_text().splitlines():
                if line.strip().startswith("url:"):
                    rc_url = line.split(":", 1)[1].strip()
        if self.key:
            return {"available": True, "source": "environment"}
        if rc.is_file() and "ecds.ecmwf.int" in rc_url:
            return {"available": True, "source": str(rc).replace(str(Path.home()), "~")}
        shown = str(rc).replace(str(Path.home()), "~")
        reason = (f"{shown} points to {rc_url!r}, not the ECDS endpoint" if rc.is_file()
                  else f"no ECDS personal access token ({shown} absent; FBS_ECDS_KEY/CDSAPI_KEY unset)")
        return {"available": False, "reason": reason}

    def build_request(self, init_time, lead_hours, levels=None, area=None,
                      forecast_type: str = "perturbed_forecast") -> dict:
        init = pd.Timestamp(init_time)
        levels = levels or [data_config()["target_level"]]
        return {
            "origin": "ncmrwf", "year": [f"{init.year:04d}"], "month": [f"{init.month:02d}"], "day": [f"{init.day:02d}"],
            "time": [f"{init.hour:02d}:00"], "level_type": "pressure", "variable": ["geopotential_height"],
            "level_value": [f"{int(lv)}_hpa" for lv in levels], "forecast_type": forecast_type,
            "leadtime_hour": [str(int(h)) for h in lead_hours], "data_format": "grib",
            "area": list(area or self.DEFAULT_AREA),
        }

    def target_path(self, init_time, lead_hours, levels=None) -> Path:
        init = pd.Timestamp(init_time)
        lv = "-".join(str(x) for x in (levels or [data_config()["target_level"]]))
        tag = f"{min(lead_hours)}-{max(lead_hours)}x{len(lead_hours)}"
        return self.cache / f"dems_pf_gh{lv}_{init:%Y%m%d%H}_{tag}.grib"

    def download(self, init_time, lead_hours, levels=None, area=None) -> Path:
        import cdsapi  # optional dependency (requirements-ncmrwf.txt)
        st = self.credentials_status()
        if not st["available"]:
            raise PermissionError(st["reason"])
        target = self.target_path(init_time, lead_hours, levels)
        if target.is_file() and target.stat().st_size > 0:
            return target
        self.cache.mkdir(parents=True, exist_ok=True)
        kw = {"url": self.url, "key": self.key} if self.key else {}
        client = cdsapi.Client(**kw, quiet=True, progress=False)
        tmp = target.with_suffix(".part")
        client.retrieve(ECDS_DATASET, self.build_request(init_time, lead_hours, levels, area), str(tmp))
        tmp.rename(target)
        return target

    # ---------- parsing ----------
    @staticmethod
    def read_grib(path: Path) -> tuple[xr.DataArray, dict]:
        """GRIB -> raw DataArray (number, step, level, lat, lon) in the file's own units + GRIB header facts."""
        ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
        var = "gh" if "gh" in ds else list(ds.data_vars)[0]
        da = ds[var]
        a = {**ds.attrs, **da.attrs}   # cfgrib puts the originating centre on the dataset attributes
        facts = {"grib_short_name": a.get("GRIB_shortName", var), "grib_param_id": a.get("GRIB_paramId"),
                 "grib_centre": str(a.get("GRIB_centre", "")), "grib_centre_description": a.get("GRIB_centreDescription", ""),
                 "grib_units": a.get("GRIB_units", a.get("units", "")), "grib_data_type": a.get("GRIB_dataType", ""),
                 "grib_grid_type": a.get("GRIB_gridType", ""), "grib_missing_value": a.get("GRIB_missingValue"),
                 "grib_dx": a.get("GRIB_iDirectionIncrementInDegrees"), "grib_dy": a.get("GRIB_jDirectionIncrementInDegrees"),
                 "init_time": str(pd.Timestamp(ds["time"].values)) if "time" in ds.coords else None}
        if "number" not in da.dims:
            num = int(da["number"]) if "number" in da.coords else 0
            da = da.drop_vars("number", errors="ignore").expand_dims(number=[num])
        if "isobaricInhPa" in da.coords and "isobaricInhPa" not in da.dims:
            lv = int(da["isobaricInhPa"])
            da = da.drop_vars("isobaricInhPa").expand_dims(isobaricInhPa=[lv])
        if "step" not in da.dims:
            st = da["step"].values
            da = da.drop_vars(["step", "valid_time"], errors="ignore").expand_dims(step=[st])
        return da, facts

    @staticmethod
    def to_canonical(raw: xr.DataArray, init: pd.Timestamp) -> xr.DataArray:
        """gpm -> m**2 s**-2, canonical dims/order/conventions, single-init time axis."""
        units = str(raw.attrs.get("GRIB_units", raw.attrs.get("units", "")))
        if units in ("gpm", "m"):
            vals, out_units = raw * G, "m**2 s**-2"
        elif units in ("m**2 s**-2", "m2 s-2"):
            vals, out_units = raw, units
        else:
            raise ValueError(f"unexpected geopotential-height units {units!r}")
        drop = [c for c in vals.coords if c not in vals.dims]
        da = vals.drop_vars(drop).expand_dims(time=[np.datetime64(init, "ns")])
        da = normalise_coords(da)
        da = da.assign_coords(level=da["level"].astype(int), number=da["number"].astype(int))
        da.attrs = {"units": out_units}
        return da

    def retrieve(self, init_time, lead_hours, levels=None, grib_path: Path | None = None) -> ForecastBundle:
        init = pd.Timestamp(init_time)
        path = Path(grib_path) if grib_path else self.download(init, lead_hours, levels)
        raw, facts = self.read_grib(path)
        if facts["grib_centre"] not in NCMRWF_GRIB_CENTRES:
            raise ValueError(f"GRIB origin is {facts['grib_centre']!r}, not NCMRWF (dems)")
        native = self.to_canonical(raw, init)
        lat_c, lon_c = reference_grid()
        da = conservative_box_mean(native, lat_c, lon_c)
        da = da.dropna("latitude", how="all").dropna("longitude", how="all")
        da.attrs["units"] = native.attrs["units"]
        da = validate_forecast(da, min_members=2)
        n_doc = self.documented_operational_members
        notes = [f"member count read from GRIB: {da.sizes['number']} perturbed members archived in TIGGE "
                 f"(NCMRWF documentation describes the operational NEPS as {n_doc} members incl. control)",
                 "aggregated to the 5.625 deg benchmark grid (cos-lat weighted box mean of native points)"]
        meta = self._meta(da, init, ECDS_DATASET_URL, "5.625 deg (WB2 64x32 cell centres, box-mean aggregated)",
                          origin=facts["grib_centre"], native_grid=f"{facts['grib_grid_type']} "
                          f"{facts['grib_dx']} x {facts['grib_dy']} deg", notes=notes)
        meta.units = da.attrs["units"]
        return ForecastBundle(da, meta)


class SyntheticDemoProvider(ForecastProvider):
    """Seeded synthetic ensemble fields for UI stress cases, development and tests. NEVER real, never
    used for training, calibration or a reported benchmark."""
    provider = "synthetic"
    dataset = "synthetic demonstration scenario"
    synthetic = True
    demo_only = True
    source_label = "Synthetic demonstration scenario"

    def __init__(self, n_members: int = 20, seed: int = 0, spread_scale_m: float = 30.0):
        self.n_members, self.seed, self.spread_scale_m = n_members, seed, spread_scale_m

    def retrieve(self, init_time, lead_hours, levels=None) -> ForecastBundle:
        init = pd.Timestamp(init_time)
        levels = levels or [data_config()["target_level"]]
        try:
            lat, lon = reference_grid()
        except (FileNotFoundError, OSError):
            lat, lon = np.arange(-30.9375, 65, 5.625), np.arange(22.5, 147, 5.625)
        rng = np.random.default_rng(self.seed)
        leads = np.asarray(lead_hours)
        base = 5700.0 + 120 * np.cos(np.deg2rad(lat))[None, :] * np.ones((len(lon), 1))
        growth = self.spread_scale_m * (0.3 + leads / 240.0)
        shape = (1, self.n_members, len(leads), len(levels), len(lon), len(lat))
        z = base[None, None, None, None] + growth[None, None, :, None, None, None] * rng.standard_normal(shape)
        da = xr.DataArray((z * G).astype(np.float32), dims=("time", "number", "prediction_timedelta", "level", "longitude", "latitude"),
                          coords={"time": [np.datetime64(init, "ns")], "number": np.arange(1, self.n_members + 1),
                                  "prediction_timedelta": leads, "level": levels, "longitude": lon, "latitude": lat},
                          attrs={"units": "m**2 s**-2"})
        da = validate_forecast(da, min_members=2)
        return ForecastBundle(da, self._meta(da, init, "generated in-process (no external source)", "5.625 deg",
                                             notes=["SYNTHETIC - demonstration only; not ECMWF, NCMRWF, ERA5, observed or validated"]))


PROVIDERS = {"ecmwf_research": ECMWFResearchProvider, "ncmrwf_tigge": NCMRWFTIGGEProvider, "synthetic": SyntheticDemoProvider}


def get_provider(name: str, **kw) -> ForecastProvider:
    if name not in PROVIDERS:
        raise KeyError(f"unknown provider {name!r}; available: {sorted(PROVIDERS)}")
    return PROVIDERS[name](**kw)
