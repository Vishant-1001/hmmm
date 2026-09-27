"""NWP-agnostic ingestion interface.

Every forecast provider is converted to one canonical array:

    DataArray  dims (time, number, prediction_timedelta, level, longitude, latitude)
               time = initialisation (datetime64), prediction_timedelta = lead in integer hours,
               longitude in [0, 360) ascending, latitude ascending, geopotential in m**2 s**-2

`validate_forecast` enforces that contract so the rest of the pipeline never sees
provider-specific conventions.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import xarray as xr

CANONICAL_DIMS = ("time", "number", "prediction_timedelta", "level", "longitude", "latitude")


class DataContractError(ValueError):
    """Raised when provider data violates the canonical contract."""


def normalise_coords(da: xr.DataArray) -> xr.DataArray:
    """Map common provider conventions onto the canonical ones."""
    ren = {k: v for k, v in {"lat": "latitude", "lon": "longitude", "member": "number", "step": "prediction_timedelta",
                             "isobaricInhPa": "level", "init_time": "time"}.items() if k in da.dims}
    da = da.rename(ren)
    if "longitude" in da.dims:
        lon = da["longitude"].values
        if lon.min() < 0:
            da = da.assign_coords(longitude=np.mod(lon, 360.0)).sortby("longitude")
    if "latitude" in da.dims and da["latitude"].values[0] > da["latitude"].values[-1]:
        da = da.sortby("latitude")
    if "prediction_timedelta" in da.dims and np.issubdtype(da["prediction_timedelta"].dtype, np.timedelta64):
        da = da.assign_coords(prediction_timedelta=(da["prediction_timedelta"].values / np.timedelta64(1, "h")).astype(int))
    return da


def validate_forecast(da: xr.DataArray, min_members: int = 10) -> xr.DataArray:
    missing = [d for d in CANONICAL_DIMS if d not in da.dims]
    if missing:
        raise DataContractError(f"missing dimensions {missing}")
    da = da.transpose(*CANONICAL_DIMS)
    units = da.attrs.get("units", "")
    if units not in ("m**2 s**-2", "m2 s-2", "m^2/s^2"):
        raise DataContractError(f"geopotential units must be m**2 s**-2, got {units!r}")
    lat = da["latitude"].values
    if np.any(np.diff(lat) <= 0):
        raise DataContractError("latitude must be strictly ascending")
    lon = da["longitude"].values
    if lon.min() < 0 or lon.max() >= 360 or np.any(np.diff(lon) <= 0):
        raise DataContractError("longitude must be ascending in [0, 360)")
    dl = np.diff(lat)
    if not np.allclose(dl, dl[0]):
        raise DataContractError("inconsistent (irregular) latitude grid")
    if da.sizes["number"] < min_members:
        raise DataContractError(f"only {da.sizes['number']} ensemble members (< {min_members})")
    return da


class ForecastSourceAdapter(ABC):
    name: str = "abstract"
    operational: bool = False

    @abstractmethod
    def list_initialisations(self) -> np.ndarray: ...

    @abstractmethod
    def load(self, init_slice: slice, lead_hours: list[int], levels: list[int], domain: dict) -> xr.DataArray: ...


class IFSENSWeatherBenchAdapter(ForecastSourceAdapter):
    """Real ECMWF IFS ENS historical archive (WeatherBench 2, 2018-2022). Research data, not operational."""
    name = "ECMWF IFS ENS via WeatherBench 2"

    def __init__(self):
        from forecast_bust.data import wb2
        self._wb2 = wb2
        self._ds = None

    @property
    def ds(self):
        if self._ds is None:
            self._ds = self._wb2.open_forecast_store()
        return self._ds

    def list_initialisations(self) -> np.ndarray:
        return self.ds["time"].values

    def load(self, init_slice, lead_hours, levels, domain):
        leads = self._wb2.lead_to_hours(self.ds["prediction_timedelta"].values)
        idx = [int(np.where(leads == h)[0][0]) for h in lead_hours]
        da = self.ds["geopotential"].isel(time=init_slice, prediction_timedelta=idx).sel(level=levels)
        da = self._wb2.subset_domain(da, domain).load()
        da = da.assign_coords(prediction_timedelta=np.asarray(lead_hours))
        return validate_forecast(normalise_coords(da))


class NCMRWFAdapter(ForecastSourceAdapter):
    """Placeholder for NCMRWF NEPS ingestion. NOT IMPLEMENTED: no NCMRWF data access was available.

    Integration requires NEPS ensemble geopotential (500/700/850 hPa) for Day 1-10 with
    initialisation metadata; once converted with `normalise_coords` and passing
    `validate_forecast`, the rest of the pipeline is provider-agnostic. Historical error
    statistics would have to be re-learned on NEPS hindcasts (relationships are model-specific).
    """
    name = "NCMRWF NEPS (not available)"
    operational = False

    def list_initialisations(self):
        raise NotImplementedError("NCMRWF data access not available; integration is architected, not claimed")

    def load(self, *a, **k):
        raise NotImplementedError("NCMRWF data access not available; integration is architected, not claimed")


ADAPTERS = {"ifs_ens_wb2": IFSENSWeatherBenchAdapter, "ncmrwf_neps": NCMRWFAdapter}


def get_adapter(name: str) -> ForecastSourceAdapter:
    if name not in ADAPTERS:
        raise KeyError(f"unsupported model source {name!r}; available: {sorted(ADAPTERS)}")
    return ADAPTERS[name]()
