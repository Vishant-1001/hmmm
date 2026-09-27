"""TEST FIXTURES ONLY: synthetic arrays below test software behaviour, never science."""
import numpy as np
import pandas as pd
import pytest

from forecast_bust.data.regions import build_regions, region_members
from forecast_bust.verification.alignment import align_reference, lead_day, season_of, split_of, valid_time
from forecast_bust.verification.metrics import (area_weights, normalized_error, rank_of_reference,
                                                spread_skill_ratio, weighted_rmse)


@pytest.mark.parametrize("lead,expected", [(24, "2021-03-02T12"), (48, "2021-03-03T12"),
                                           (72, "2021-03-04T12"), (240, "2021-03-11T12")])
def test_valid_time(lead, expected):
    assert valid_time(np.datetime64("2021-03-01T12"), lead) == np.datetime64(expected, "ns")


def test_valid_time_crosses_year():
    assert valid_time(np.datetime64("2022-12-31T00"), 240) == np.datetime64("2023-01-10T00", "ns")


def test_lead_day():
    assert [lead_day(h) for h in (24, 48, 240)] == [1, 2, 10]
    with pytest.raises(ValueError):
        lead_day(36)


def test_align_reference_uses_valid_time_not_init():
    ref = pd.date_range("2021-01-01", periods=40, freq="6h").values
    init = np.datetime64("2021-01-01T00")
    idx = align_reference(ref, valid_time(init, np.array([24, 48])))
    assert pd.Timestamp(ref[idx[0]]) == pd.Timestamp("2021-01-02T00")
    assert pd.Timestamp(ref[idx[1]]) == pd.Timestamp("2021-01-03T00")


def test_align_reference_missing_raises():
    ref = pd.date_range("2021-01-01", periods=4, freq="6h").values
    with pytest.raises(KeyError):
        align_reference(ref, valid_time(np.datetime64("2021-01-01"), np.array([240])))


def test_season_and_split():
    t = np.array(["2019-01-15", "2019-04-01", "2021-07-01", "2022-11-30"], dtype="datetime64[ns]")
    assert list(season_of(t)) == ["JF", "MAM", "JJAS", "OND"]
    cfg = {"train": ["2018-01-01", "2020-12-31T23:59"], "validation": ["2021-01-01", "2021-12-31T23:59"],
           "test": ["2022-01-01", "2022-12-31T23:59"]}
    assert list(split_of(t, cfg)) == ["train", "train", "validation", "test"]


def test_area_weights_shrink_poleward():
    w = area_weights(np.array([0.0, 30.0, 60.0]), 5.625)
    assert w[0] > w[1] > w[2] > 0
    np.testing.assert_allclose(w[1] / w[0], np.cos(np.deg2rad(30)), rtol=2e-3)


def test_weighted_rmse_uniform_error():
    lats = np.array([-10.0, 0.0, 10.0, 20.0])
    f = np.full((3, 4), 5.0)
    assert np.isclose(weighted_rmse(f, np.zeros_like(f), lats), 5.0)


def test_weighted_rmse_weighting():
    lats = np.array([0.0, 60.0])
    f = np.array([[0.0, 10.0]])  # error only at 60N (lon, lat)
    r = np.zeros_like(f)
    w = area_weights(lats, 5.625)
    expected = np.sqrt(100 * w[1] / w.sum())
    assert np.isclose(weighted_rmse(f, r, lats, lat_spacing=5.625), expected)
    assert weighted_rmse(f, r, lats, lat_spacing=5.625) < np.sqrt(50)  # less than unweighted


def test_weighted_rmse_nan_and_shape():
    lats = np.array([0.0, 5.0])
    f = np.array([[1.0, np.nan]])
    assert np.isclose(weighted_rmse(f, np.zeros_like(f), lats), 1.0)
    with pytest.raises(ValueError):
        weighted_rmse(np.zeros((2, 2)), np.zeros((2, 3)), lats)


def test_normalized_error():
    np.testing.assert_allclose(normalized_error(np.array([10.0, 20.0]), np.array([5.0, 10.0])), [2.0, 2.0])
    with pytest.raises(ValueError):
        normalized_error(np.array([1.0]), np.array([0.0]))


def test_rank_and_spread_skill():
    members = np.arange(10.0)[:, None]
    assert rank_of_reference(members, np.array([4.5]))[0] == 5
    assert np.isclose(spread_skill_ratio(1.0, 1.0 * 11 / 10, 10), 1.0)


def test_regions_grid():
    lats = np.arange(-87.1875, 90, 5.625)
    lons = np.arange(0, 360, 5.625)
    regs = build_regions(lats, lons, {"lat_min": -5, "lat_max": 40, "lon_min": 60, "lon_max": 105})
    assert len(regs) == 64
    assert regs[0].lat == -2.8125 and regs[0].lon == 61.875
    li, oi = region_members(regs[0], lats, lons)
    assert len(li) == 1 and len(oi) == 1
    # finer grid: several points per region
    li, oi = region_members(regs[0], np.arange(-90, 90.1, 1.5), np.arange(0, 360, 1.5))
    assert len(li) >= 3 and len(oi) >= 3
