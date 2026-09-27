"""Forecast / verification time alignment.

valid_time = init_time + lead. Verification is always looked up at valid_time,
never at init_time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEASONS = {"JF": (1, 2), "MAM": (3, 4, 5), "JJAS": (6, 7, 8, 9), "OND": (10, 11, 12)}
SEASON_CODES = {"JF": 0, "MAM": 1, "JJAS": 2, "OND": 3}


def valid_time(init_time, lead_hours) -> np.datetime64 | np.ndarray:
    init = np.asarray(init_time, dtype="datetime64[ns]")
    lead = np.asarray(lead_hours).astype("int64") * np.timedelta64(1, "h")
    return init + lead


def lead_day(lead_hours: int) -> int:
    if lead_hours % 24:
        raise ValueError(f"lead {lead_hours}h is not a whole day")
    return lead_hours // 24


def season_of(times) -> np.ndarray:
    months = pd.DatetimeIndex(np.atleast_1d(times)).month
    out = np.empty(len(months), dtype=object)
    for name, ms in SEASONS.items():
        out[np.isin(months, ms)] = name
    return out


def split_of(init_times, split_cfg: dict) -> np.ndarray:
    t = pd.DatetimeIndex(np.atleast_1d(init_times))
    out = np.full(len(t), "unused", dtype=object)
    for name, (a, b) in split_cfg.items():
        out[(t >= pd.Timestamp(a)) & (t <= pd.Timestamp(b))] = name
    return out


def align_reference(ref_times: np.ndarray, valid_times: np.ndarray) -> np.ndarray:
    """Index into ref_times for each valid time; raises if any valid time is missing."""
    ref = pd.DatetimeIndex(ref_times)
    idx = ref.get_indexer(pd.DatetimeIndex(np.ravel(valid_times)))
    if (idx < 0).any():
        missing = pd.DatetimeIndex(np.ravel(valid_times))[idx < 0]
        raise KeyError(f"verification reference missing for {len(missing)} valid times, e.g. {missing[:3].tolist()}")
    return idx.reshape(np.shape(valid_times))
