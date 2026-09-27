"""TEST FIXTURES ONLY: synthetic arrays exercise the ingestion contract / failure modes."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from forecast_bust.data.adapters import (DataContractError, NCMRWFAdapter, get_adapter, normalise_coords,
                                         validate_forecast)
from forecast_bust.data.assemble import clim_at, member_stats


def _fixture(n_members=20, lat=np.arange(-10.0, 11, 5), lon=np.arange(60.0, 81, 5), units="m**2 s**-2"):
    shape = (1, n_members, 2, 1, len(lon), len(lat))
    da = xr.DataArray(np.ones(shape, dtype="float32"),
                      dims=("time", "number", "prediction_timedelta", "level", "longitude", "latitude"),
                      coords={"time": [np.datetime64("2020-01-01")], "number": np.arange(n_members),
                              "prediction_timedelta": [24, 48], "level": [500], "longitude": lon, "latitude": lat})
    da.attrs["units"] = units
    return da


def test_valid_fixture_passes():
    assert validate_forecast(_fixture()).dims[0] == "time"


def test_missing_variable_dimension():
    with pytest.raises(DataContractError):
        validate_forecast(_fixture().isel(level=0, drop=True))


def test_wrong_units():
    with pytest.raises(DataContractError):
        validate_forecast(_fixture(units="gpm"))


def test_too_few_members():
    with pytest.raises(DataContractError):
        validate_forecast(_fixture(n_members=3))


def test_inconsistent_grid():
    with pytest.raises(DataContractError):
        validate_forecast(_fixture(lat=np.array([-10.0, -5, 0, 7, 10])))


def test_coordinate_normalisation():
    da = _fixture(lat=np.arange(10.0, -11, -5), lon=np.arange(-20.0, 1, 5)).rename(latitude="lat", longitude="lon")
    out = validate_forecast(normalise_coords(da))
    assert np.all(np.diff(out.latitude.values) > 0)
    assert out.longitude.values.min() >= 0


def test_timedelta_lead_normalised():
    da = _fixture().assign_coords(prediction_timedelta=np.array([24, 48], dtype="timedelta64[h]"))
    assert list(normalise_coords(da).prediction_timedelta.values) == [24, 48]


def test_unsupported_source_and_ncmrwf_not_faked():
    with pytest.raises(KeyError):
        get_adapter("mystery_model")
    with pytest.raises(NotImplementedError):
        NCMRWFAdapter().list_initialisations()


def test_member_stats_and_missing_climatology():
    rng = np.random.default_rng(0)
    m = rng.normal(size=(50, 3))
    s = member_stats(m, np.zeros(3), np.array([10.0, -10.0, 0.0]))
    assert list(s["era5_rank"]) == [50, 0, s["era5_rank"][2]]
    assert np.all((s["m_sign_agree"] >= 0) & (s["m_sign_agree"] <= 1))
    clim = xr.DataArray(np.zeros((4, 366, 1)), dims=("hour", "dayofyear", "x"),
                        coords={"hour": [0, 6, 12, 18], "dayofyear": np.arange(1, 367)})
    assert clim_at(clim, np.array(["2020-03-01T12"], dtype="datetime64[ns]")).shape == (1, 1)
    with pytest.raises(KeyError):
        clim_at(clim, np.array(["2020-03-01T03"], dtype="datetime64[ns]"))  # hour not in climatology
