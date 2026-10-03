"""NCMRWF/TIGGE provider, provider-agnostic B2 path and provenance flags.

The GRIB files here are TEST FIXTURES written with eccodes (centre 29 = NCMRWF `dems`, TIGGE gh in gpm,
perturbed members); they exercise the parser only and are never real NCMRWF data.
"""
import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from forecast_bust.config import REPO_ROOT
from forecast_bust.data import providers as P
from forecast_bust.data.providers import (G, NCMRWFTIGGEProvider, SyntheticDemoProvider, conservative_box_mean,
                                          ensemble_mean_spread)

LEADS = list(range(24, 241, 24))
INIT = "2021-07-15T00:00"
COARSE_LAT = -30.9375 + 5.625 * np.arange(18)
COARSE_LON = 22.5 + 5.625 * np.arange(23)
DOMAIN = {"lat_min": -5.0, "lat_max": 40.0, "lon_min": 60.0, "lon_max": 105.0}
HAS_B2 = (REPO_ROOT / "artifacts" / "b2" / "ecmwf" / "spread_reference.parquet").is_file()


def write_grib(path, centre=29, members=(1, 2, 3, 4), leads=LEADS, level=500, missing_member=None):
    """Fine 1.875 deg grid (3x3 points per 5.625 deg cell) over the target cells, N->S scanning like TIGGE."""
    eccodes = pytest.importorskip("eccodes")
    lat = np.arange(44.0625, -10.3125 - 1e-9, -1.875)
    lon = np.arange(54.375, 106.875 + 1e-9, 1.875)
    rng = np.random.default_rng(0)
    with open(path, "wb") as f:
        for m in members:
            for h in leads:
                g = eccodes.codes_grib_new_from_samples("regular_ll_pl_grib2")
                for k, v in [("centre", centre), ("productDefinitionTemplateNumber", 1), ("discipline", 0),
                             ("parameterCategory", 3), ("parameterNumber", 5), ("typeOfFirstFixedSurface", 100),
                             ("level", level), ("dataDate", 20210715), ("dataTime", 0), ("stepUnits", 1),
                             ("forecastTime", h), ("perturbationNumber", m), ("numberOfForecastsInEnsemble", len(members)),
                             ("typeOfEnsembleForecast", 3), ("Ni", len(lon)), ("Nj", len(lat)),
                             ("latitudeOfFirstGridPointInDegrees", lat[0]), ("longitudeOfFirstGridPointInDegrees", lon[0]),
                             ("latitudeOfLastGridPointInDegrees", lat[-1]), ("longitudeOfLastGridPointInDegrees", lon[-1]),
                             ("iDirectionIncrementInDegrees", 1.875), ("jDirectionIncrementInDegrees", 1.875)]:
                    eccodes.codes_set(g, k, v)
                vals = 5800 + 10 * m * (h / 240) + rng.normal(0, 1, len(lat) * len(lon))
                if missing_member == m:
                    eccodes.codes_set(g, "missingValue", 9999.0)
                    eccodes.codes_set(g, "bitmapPresent", 1)
                    vals[:5] = 9999.0
                eccodes.codes_set_values(g, vals)
                eccodes.codes_write(g, f)
                eccodes.codes_release(g)
    return path


@pytest.fixture(autouse=True)
def coarse_grid(monkeypatch):
    monkeypatch.setattr(P, "reference_grid", lambda: (COARSE_LAT, COARSE_LON))


@pytest.fixture
def bundle(tmp_path):
    return NCMRWFTIGGEProvider(cache=tmp_path).retrieve(INIT, LEADS, grib_path=write_grib(tmp_path / "f.grib"))


# ---------------- data ----------------
def test_provider_schema_and_metadata(bundle):
    da, md = bundle.data, bundle.metadata
    assert da.dims == ("time", "number", "prediction_timedelta", "level", "longitude", "latitude")
    assert md.provider == "ncmrwf_tigge" and md.synthetic is False and md.demo_only is False
    assert md.origin == "dems" and md.source_url.startswith("https://ecds.ecmwf.int/")
    for k in ("dataset", "retrieval_timestamp", "initialization_time", "valid_times", "forecast_hours",
              "ensemble_member_count", "variable", "level", "grid", "units"):
        assert getattr(md, k) not in (None, "", [])


def test_member_count_read_from_file_not_assumed(tmp_path):
    b = NCMRWFTIGGEProvider(cache=tmp_path).retrieve(INIT, LEADS, grib_path=write_grib(tmp_path / "g.grib", members=range(1, 8)))
    assert b.metadata.ensemble_member_count == 7 and b.metadata.member_ids == list(range(1, 8))
    assert "23" in b.metadata.notes[0]   # documented operational size is logged, not used


def test_origin_must_be_ncmrwf(tmp_path):
    with pytest.raises(ValueError, match="not NCMRWF"):
        NCMRWFTIGGEProvider(cache=tmp_path).retrieve(INIT, LEADS, grib_path=write_grib(tmp_path / "e.grib", centre=98))


def test_z500_units_and_plausible_values(bundle):
    assert bundle.data.attrs["units"] == "m**2 s**-2"
    assert list(bundle.data["level"].values) == [500]
    z = bundle.data.values / G
    assert 5790 < np.nanmean(z) < 5850          # gpm * g / g round trip


def test_lead_time_and_valid_time(bundle):
    assert list(bundle.data["prediction_timedelta"].values) == LEADS
    assert bundle.metadata.valid_times[0] == "2021-07-16T00:00:00"
    assert bundle.metadata.valid_times[-1] == "2021-07-25T00:00:00"
    assert pd.Timestamp(bundle.data["time"].values[0]) == pd.Timestamp(INIT)


def test_coordinates_normalised_to_benchmark_grid(bundle):
    lat, lon = bundle.data["latitude"].values, bundle.data["longitude"].values
    assert np.all(np.diff(lat) > 0) and np.all(np.diff(lon) > 0) and lon.min() >= 0
    assert set(np.round(lat, 4)) <= set(np.round(COARSE_LAT, 4)) and set(np.round(lon, 4)) <= set(np.round(COARSE_LON, 4))


def test_missing_values_excluded(tmp_path):
    b = NCMRWFTIGGEProvider(cache=tmp_path).retrieve(INIT, LEADS, grib_path=write_grib(tmp_path / "m.grib", missing_member=2))
    assert np.isfinite(b.data.values).all()     # box mean over the remaining valid fine points
    da = xr.DataArray(np.array([[1.0, np.nan], [3.0, 5.0], [5.0, 9.0]]), dims=("number", "x"))
    mean, std, n = ensemble_mean_spread(da)
    assert list(n.values) == [3, 2] and mean.values[1] == 7.0 and std.values[1] == pytest.approx(np.std([5, 9], ddof=1))


def test_spread_matches_ddof1_std():
    rng = np.random.default_rng(1)
    v = rng.normal(size=(9, 4, 3))
    mean, std, n = ensemble_mean_spread(xr.DataArray(v, dims=("number", "a", "b")))
    assert np.allclose(std.values, v.std(axis=0, ddof=1)) and np.allclose(mean.values, v.mean(0)) and (n.values == 9).all()


def test_box_mean_is_area_weighted_average():
    fine_lat = np.array([-1.875, 0.0, 1.875])
    da = xr.DataArray(np.array([[1.0, 2.0, 3.0]]), dims=("longitude", "latitude"),
                      coords={"longitude": [0.0], "latitude": fine_lat})
    out = conservative_box_mean(da, np.array([0.0, 5.625]), np.array([0.0, 5.625]))
    w = np.cos(np.deg2rad(fine_lat))
    assert out.values[0, 0] == pytest.approx((w * [1, 2, 3]).sum() / w.sum())
    assert np.isnan(out.values[0, 1]) and np.isnan(out.values[1, 0])   # no fine point -> NaN, never invented


def test_smoke_request_is_the_minimal_b2_request():
    r = NCMRWFTIGGEProvider().build_request(INIT, LEADS)
    assert r["origin"] == "ncmrwf" and r["forecast_type"] == "perturbed_forecast" and r["data_format"] == "grib"
    assert r["variable"] == ["geopotential_height"] and r["level_value"] == ["500_hpa"]
    assert r["leadtime_hour"] == [str(h) for h in LEADS] and r["time"] == ["00:00"]
    n, w, s, e = r["area"]
    assert n >= 39.375 and s <= -5.625 and w <= 59.0625 and e >= 104.0625


def test_no_token_is_reported_not_faked(tmp_path, monkeypatch):
    for k in ("FBS_ECDS_KEY", "CDSAPI_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("CDSAPI_RC", str(tmp_path / "absent"))
    p = NCMRWFTIGGEProvider(cache=tmp_path)
    assert p.credentials_status()["available"] is False
    with pytest.raises(PermissionError):
        p.download(INIT, LEADS)


def test_availability_checklist_passes_on_fixture(tmp_path):
    from forecast_bust.data.ncmrwf_check import verify_retrieval
    path = write_grib(tmp_path / "c.grib")
    prov = NCMRWFTIGGEProvider(cache=tmp_path)
    b = prov.retrieve(INIT, LEADS, grib_path=path)
    _, facts = prov.read_grib(path)
    v = verify_retrieval(b, facts, INIT, LEADS, DOMAIN)
    assert v["all_passed"], v["checks"]
    assert v["members_found"] == 4 and v["missing_leads"] == []


def test_time_alignment_uses_valid_time(tmp_path, bundle):
    """ERA5 identical to the forecast at VALID time -> zero error; a reference keyed on init time would not be."""
    if not HAS_B2:
        pytest.skip("B2 reference artifacts absent")
    from forecast_bust.b2_provider import verify_bundle
    mean = (bundle.data.isel(time=0).sel(level=500).mean("number")).transpose("prediction_timedelta", "longitude", "latitude")
    vt = pd.DatetimeIndex(bundle.metadata.valid_times)
    times = pd.date_range("2021-07-14", "2021-07-26", freq="6h")
    z = np.full((len(times), 1, len(mean.longitude), len(mean.latitude)), np.nan, np.float32)
    for j, t in enumerate(vt):
        z[times.get_loc(t), 0] = mean.values[j]
    z[times.get_loc(pd.Timestamp(INIT)), 0] = 0.0     # a wrong (init-time) lookup would give a huge error
    era = xr.Dataset({"z": (("time", "level", "longitude", "latitude"), z)},
                     coords={"time": times, "level": [500], "longitude": mean.longitude.values, "latitude": mean.latitude.values})
    era.to_netcdf(tmp_path / "era5.nc")
    lab = verify_bundle(bundle, era5_path=tmp_path / "era5.nc")
    assert np.allclose(lab["error_m"], 0, atol=1e-3) and (lab["bust"] == 0).all()


# ---------------- scientific ----------------
def test_b2_feature_contract_unchanged():
    from forecast_bust.b2_provider import B2_FEATURES
    from forecast_bust.models.sentinel import FEATURE_SETS
    spec = json.loads((REPO_ROOT / "artifacts" / "v2" / "demo" / "model" / "models.json").read_text())["B2"]
    assert B2_FEATURES == FEATURE_SETS["B2"] == list(spec["features"]) == [
        "spread_m", "spread_pct", "spread_thr_ratio", "lead_day", "region_code", "season_code", "init_hour"]


@pytest.mark.skipif(not HAS_B2, reason="B2 reference artifacts absent")
def test_b2_runs_on_any_provider_without_verification(tmp_path, bundle, monkeypatch):
    from forecast_bust import b2_provider as B
    from forecast_bust.features.build import FORBIDDEN_INPUTS
    monkeypatch.setattr(B, "verify_bundle", lambda *a, **k: (_ for _ in ()).throw(AssertionError("verification read")))
    out = B.run_b2(bundle, "b2_ncmrwf")
    X = out["rows"]
    assert len(X) == 64 * 10 and X["b2_probability"].between(0, 1).all()
    assert not any(f.startswith(FORBIDDEN_INPUTS) for f in B.B2_FEATURES)
    assert sorted(X["lead_day"].unique()) == list(range(1, 11))
    pv = out["provenance"]
    assert pv["provider"] == "ncmrwf_tigge" and pv["synthetic"] is False
    assert pv["calibration_validated_for_provider"] is False and "NOT validated" in pv["calibration"]
    assert pv["ensemble_member_count"] == 4
    # finite-ensemble correction uses THIS ensemble's member count
    thr = X["q_primary"] * X["scale_m"]
    assert np.allclose(X["spread_thr_ratio"], X["spread_m"] * np.sqrt(1 + 1 / 4) / thr, rtol=1e-5)


@pytest.mark.skipif(not HAS_B2, reason="B2 reference artifacts absent")
def test_spread_reference_is_train_only():
    m = json.loads((REPO_ROOT / "artifacts" / "b2" / "ecmwf" / "b2_manifest.json").read_text())
    assert m["spread_reference"]["fitted_on"] == "train" and m["spread_reference"]["period"][1].startswith("2020-12-31")
    from forecast_bust.b2_provider import reference
    assert reference().train_period == ["2018-01-01", "2020-12-31T23:59"]
    assert m["provider_path_equivalence"]["passed"] is True


def test_calibration_must_be_validation_only(tmp_path, monkeypatch):
    from forecast_bust import b2_provider as B
    d = tmp_path / "ncmrwf"
    d.mkdir()
    (d / "calibration.json").write_text(json.dumps({"name": "x", "fitted_on": "test", "x": [0, 1], "y": [0, 1]}))
    monkeypatch.setattr(B, "B2_ROOT", tmp_path)
    with pytest.raises(ValueError, match="validation-only"):
        B.provider_calibrator("b2_ncmrwf")
    assert B.provider_calibrator("b2_ecmwf")[2] is True


def test_real_and_synthetic_distinguishable():
    s = SyntheticDemoProvider(n_members=5).retrieve(INIT, LEADS)
    assert s.metadata.synthetic and s.metadata.demo_only and s.metadata.source_label == "Synthetic demonstration scenario"
    from forecast_bust import b2_provider as B
    with pytest.raises(ValueError, match="expects provider"):
        B.run_b2(s, "b2_ncmrwf")          # synthetic can never be scored as NCMRWF
    with pytest.raises(ValueError, match="synthetic"):
        B.verify_bundle(s)                 # and never receives verification labels
