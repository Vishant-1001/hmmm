"""V4 pattern-aware bust label (config/model_v4.yaml).

    local_acc    centred, area-weighted spatial anomaly correlation between the ensemble-mean Z500 forecast
                 and ERA5 Z500 over the 3 x 3 native boxes centred on the region box; anomalies against the
                 existing ERA5 1990-2017 climatology at the valid time (data/assemble.py), weights = the
                 existing exact grid-box area weights (verification/metrics.area_weights).
    acc_q10      TRAIN-only Q10 of local_acc per region x lead day x season.
    pattern_bust (norm_error > q_primary) AND (local_acc < acc_q10).

local_acc is a LABEL-CONSTRUCTION variable computed from verification: it is never a model input
(features/build.FORBIDDEN_INPUTS and tests/test_pattern_bust.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from forecast_bust.labels.build import GROUP, regions_for
from forecast_bust.data.regions import region_members
from forecast_bust.verification.alignment import valid_time  # noqa: F401  (alignment is the states.nc one)
from forecast_bust.verification.metrics import area_weights

PATTERN_COLS = ["local_acc", "acc_q10", "magnitude_failure", "pattern_failure", "pattern_bust",
                "hidden_pattern_bust"]


def weighted_acc(f: np.ndarray, o: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Centred weighted Pearson correlation over the last two axes. f, o: (..., a, b) anomalies; w broadcastable
    to (a, b). Returns NaN where either field has zero weighted variance or contains NaN."""
    w = np.broadcast_to(w, f.shape[-2:]).astype(np.float64)
    w = w / w.sum()
    f = np.asarray(f, np.float64)
    o = np.asarray(o, np.float64)
    fc = f - (f * w).sum((-2, -1), keepdims=True)
    oc = o - (o * w).sum((-2, -1), keepdims=True)
    cov = (w * fc * oc).sum((-2, -1))
    vf, vo = (w * fc * fc).sum((-2, -1)), (w * oc * oc).sum((-2, -1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = cov / np.sqrt(vf * vo)
    return np.where((vf > 0) & (vo > 0) & np.isfinite(r), r, np.nan)


def local_acc_table(ds: xr.Dataset, half: int = 1) -> pd.DataFrame:
    """local_acc for every (init, region, lead) from states.nc. Neighbourhoods that leave the cached array are
    invalid (NaN) - no padding, no wrap (the cached context domain is not global)."""
    lat, lon = ds.latitude.values, ds.longitude.values
    dlat = float(np.median(np.diff(lat)))
    fm = ds["ens_mean"].sel(level=500).transpose("init", "lead", "longitude", "latitude").values
    cl = ds["clim"].sel(level=500).transpose("init", "lead", "longitude", "latitude").values
    ob = ds["era5_z500"].transpose("init", "lead", "longitude", "latitude").values
    fa, oa = fm - cl, ob - cl
    inits, leads = ds.init.values, ds.lead.values
    frames = []
    for r in regions_for(ds):
        li, oi = region_members(r, lat, lon)
        lo0, la0 = int(oi.min()) - half, int(li.min()) - half
        lo1, la1 = int(oi.max()) + half + 1, int(li.max()) + half + 1
        if lo0 < 0 or la0 < 0 or lo1 > len(lon) or la1 > len(lat):
            acc = np.full((len(inits), len(leads)), np.nan)
        else:
            w = area_weights(lat[la0:la1], lat_spacing=dlat)[None, :]
            acc = weighted_acc(fa[:, :, lo0:lo1, la0:la1], oa[:, :, lo0:lo1, la0:la1], w)
        frames.append(pd.DataFrame({"init_time": np.repeat(inits, len(leads)), "region_id": r.region_id,
                                    "lead_day": np.tile((leads // 24).astype(np.int16), len(inits)),
                                    "local_acc": acc.ravel().astype(np.float32)}))
    return pd.concat(frames, ignore_index=True)


def fit_acc_thresholds(train: pd.DataFrame, q: float) -> pd.DataFrame:
    return train.groupby(GROUP)["local_acc"].quantile(q).rename("acc_q10").reset_index()


def pattern_labels(cases: pd.DataFrame, acc: pd.DataFrame, q: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """cases: case rows with init_time, region_id, lead_day, season, split, bust, low_spread.
    Returns (case_id + PATTERN_COLS, TRAIN thresholds). Rows with invalid local_acc get pattern_bust = NaN."""
    d = cases.merge(acc, on=["init_time", "region_id", "lead_day"], how="left")
    th = fit_acc_thresholds(d[(d["split"] == "train") & d["local_acc"].notna()], q)
    d = d.merge(th, on=GROUP, how="left")
    valid = d["local_acc"].notna() & d["acc_q10"].notna()
    d["magnitude_failure"] = d["bust"].astype(np.int8)
    d["pattern_failure"] = np.where(valid, (d["local_acc"] < d["acc_q10"]).astype(float), np.nan)
    d["pattern_bust"] = np.where(valid, ((d["bust"] == 1) & (d["local_acc"] < d["acc_q10"])).astype(float), np.nan)
    d["hidden_pattern_bust"] = np.where(valid, (d["pattern_bust"] == 1) & (d["low_spread"] == 1), np.nan)
    return d[["case_id", *PATTERN_COLS]], th
