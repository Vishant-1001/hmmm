"""The extras validator accepts a well-formed block and rejects the failure modes that matter."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from forecast_bust.config import data_config
from forecast_bust.data.validate_extras import check_block, validate

CFG = data_config()
TIME = pd.date_range("2018-01-17", periods=4, freq="12h").values
LAT, LON = np.linspace(-30.94, 64.69, 18), np.linspace(22.5, 146.25, 23)


def _geo(path):
    xr.Dataset({"z": (("time", "longitude", "latitude"), np.zeros((4, 23, 18), "float32"))},
               coords={"time": TIME, "latitude": LAT, "longitude": LON}).to_netcdf(path)


def _extra(path, **mod):
    lev, lead = np.array(CFG["levels"], "int32"), np.array(CFG["lead_hours"], "int32")
    shp2, shp3 = (4, len(lead), 23, 18), (4, len(lead), 3, 23, 18)
    d2, d3 = ("time", "prediction_timedelta", "longitude", "latitude"), ("time", "prediction_timedelta", "level", "longitude", "latitude")
    ds = xr.Dataset({
        "u_mean": (d3, np.full(shp3, 5.0, "float32")), "u_std": (d3, np.full(shp3, 2.0, "float32")),
        "v_mean": (d3, np.full(shp3, -3.0, "float32")), "v_std": (d3, np.full(shp3, 2.0, "float32")),
        "mslp_mean": (d2, np.full(shp2, 101_300.0, "float32")), "mslp_std": (d2, np.full(shp2, 150.0, "float32")),
    }, coords={"time": mod.pop("time", TIME), "prediction_timedelta": lead, "level": lev, "latitude": LAT, "longitude": LON})
    ds.attrs["n_members"] = 50
    for k, v in mod.items():
        if v is None:
            ds = ds.drop_vars(k)
        else:
            ds[k] = v(ds[k])
    ds.to_netcdf(path)


@pytest.fixture
def geo(tmp_path):
    p = tmp_path / "geo.nc"
    _geo(p)
    return p


def test_valid_block_passes(tmp_path, geo):
    _extra(tmp_path / "e.nc")
    assert check_block(tmp_path / "e.nc", geo, CFG) == []


@pytest.mark.parametrize("mod,expect", [
    ({"u_mean": lambda a: a.where(a.level != 500)}, "non-finite"),
    ({"mslp_mean": lambda a: a * 0 + 50_000}, "outside"),
    ({"v_std": lambda a: -a}, "outside"),
    ({"mslp_std": None}, "variables missing"),
    ({"time": TIME + np.timedelta64(6, "h")}, "init times differ"),
])
def test_bad_blocks_are_rejected(tmp_path, geo, mod, expect):
    _extra(tmp_path / "e.nc", **mod)
    errs = check_block(tmp_path / "e.nc", geo, CFG)
    assert any(expect in e for e in errs), errs


def test_missing_and_truncated_files(tmp_path, geo):
    assert check_block(tmp_path / "absent.nc", geo, CFG) == ["missing"]
    _extra(tmp_path / "e.nc")
    data = (tmp_path / "e.nc").read_bytes()
    (tmp_path / "t.nc").write_bytes(data[: len(data) // 3])
    assert check_block(tmp_path / "t.nc", geo, CFG)[0].startswith("unreadable")


def test_incomplete_cache_is_not_complete(tmp_path):
    (tmp_path / "ens").mkdir()
    (tmp_path / "ens_extra").mkdir()
    for k in (0, 8):
        _geo(tmp_path / "ens" / f"block_{k:04d}.nc")
    _extra(tmp_path / "ens_extra" / "block_0000.nc")
    rep = validate(tmp_path)
    assert rep["valid_blocks"] == 1 and rep["expected_blocks"] == 2 and not rep["complete"]
    assert rep["invalid_or_missing"] == {"block_0008.nc": ["missing"]}
