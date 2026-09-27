"""Recent verified forecast-error behaviour of the NWP system (causal).

For a query (init T, region r) we use only forecasts whose verification time is at or
before T (valid_time <= T, the same availability rule as the analogue memory) within a
trailing window (default 5 days) and at short leads (Day 1-3, whose verification is
available soonest):

  rec_err        mean normalized error of region r
  rec_bias       mean signed ensemble-mean error (m) of region r
  rec_bust_rate  fraction of those region-r cases that were busts
  rec_n          number of verified cases used
  rec_err_nbhd   mean normalized error of the 3x3 neighbouring regions
  rec_err_domain mean normalized error over all target regions

Operational note: ERA5 has a ~5-day release latency; in operations the NWP centre's own
analysis would play this role for the most recent 5 days (see docs/limitations.md).
Features are identical for all lead days of the same (T, r).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW = pd.Timedelta(days=5)
MAX_LEAD = 3


def _window_stats(q_t: np.ndarray, v_t: np.ndarray, vals: np.ndarray):
    """Mean of vals whose time v_t lies in (q - WINDOW, q]; v_t sorted."""
    cs = np.concatenate([[0.0], np.cumsum(vals)])
    hi = np.searchsorted(v_t, q_t, side="right")
    lo = np.searchsorted(v_t, q_t - WINDOW.to_timedelta64(), side="right")
    n = hi - lo
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)
    return m, n


def recent_error_features(df: pd.DataFrame) -> pd.DataFrame:
    src = df[df["lead_day"] <= MAX_LEAD]
    inits = np.sort(df["init_time"].unique()).astype("datetime64[ns]")
    rows = []
    reg_rc = df.drop_duplicates("region_id").set_index("region_id")[["row", "col"]]
    per_region = {}
    for rid, g in src.groupby("region_id"):
        g = g.sort_values("valid_time")
        v = g["valid_time"].values.astype("datetime64[ns]")
        e, _ = _window_stats(inits, v, g["norm_error"].to_numpy(float))
        b, _ = _window_stats(inits, v, g["bias_m"].to_numpy(float))
        br, n = _window_stats(inits, v, g["bust"].to_numpy(float))
        per_region[rid] = (e, b, br, n)
    dom = src.sort_values("valid_time")
    dom_e, _ = _window_stats(inits, dom["valid_time"].values.astype("datetime64[ns]"), dom["norm_error"].to_numpy(float))
    for rid, (e, b, br, n) in per_region.items():
        r0, c0 = reg_rc.loc[rid]
        nb = [k for k, (r, c) in reg_rc.iterrows() if abs(r - r0) <= 1 and abs(c - c0) <= 1]
        nb_e = np.nanmean(np.stack([per_region[k][0] for k in nb]), 0) if nb else np.full(len(inits), np.nan)
        rows.append(pd.DataFrame({"init_time": inits, "region_id": rid, "rec_err": e, "rec_bias": b,
                                  "rec_bust_rate": br, "rec_n": n.astype(float), "rec_err_nbhd": nb_e,
                                  "rec_err_domain": dom_e}))
    rec = pd.concat(rows, ignore_index=True)
    out = df.drop(columns=[c for c in rec.columns if c.startswith("rec_")], errors="ignore")
    return out.merge(rec, on=["init_time", "region_id"], how="left")
