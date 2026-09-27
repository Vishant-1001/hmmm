"""Historical forecast-state memory (analogue retrieval) with strict temporal causality.

For a query case (init T, region, lead) the candidate analogues are historical cases of the
SAME region and lead day whose verification was complete before the forecast was issued:

    candidate.valid_time <= T     (hence also candidate.init_time < T; no self-retrieval)

In `frozen` memory mode, candidates are additionally restricted to TRAIN + VALIDATION
cases (the memory stops learning at the end of validation). In `causal_online` mode
(the default, and the operational workflow: "learn from verified failures") a test case
may retrieve earlier *already-verified* test cases; no label of the query case or of any
case verified after T is ever used.

Similarity space (standardised with TRAIN statistics only):
    PC1..PCk of the forecast Z500-anomaly pattern at this lead, regional spread,
    regional Z500 anomaly, regional P10-P90 member range.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from forecast_bust.config import clean_json, ARTIFACT_DIR, model_config

log = logging.getLogger(__name__)


def analogue_space(df: pd.DataFrame) -> list[str]:
    k = model_config()["pca"]["n_components"]
    return [f"pc{i + 1}" for i in range(k)] + ["spread_m", "anom500", "m_p10p90"]


def fit_scaler(train: pd.DataFrame, cols: list[str]) -> dict:
    return {"mean": train[cols].mean().to_dict(), "std": train[cols].std().replace(0, 1).to_dict()}


def standardise(df: pd.DataFrame, cols: list[str], scaler: dict) -> np.ndarray:
    mu = np.array([scaler["mean"][c] for c in cols])
    sd = np.array([scaler["std"][c] for c in cols])
    return ((df[cols].to_numpy(dtype=float) - mu) / sd).astype(np.float32)


def eligible_mask(q_init: np.ndarray, c_valid: np.ndarray, c_split: np.ndarray, mode: str) -> np.ndarray:
    """(n_query, n_cand) boolean: candidate verified strictly no later than query init."""
    m = c_valid[None, :] <= q_init[:, None]
    if mode == "frozen":
        m &= np.isin(c_split, ["train", "validation"])[None, :]
    elif mode != "causal_online":
        raise ValueError(mode)
    return m


def _group_distances(X, init, valid, split, idx, mode):
    D = cdist(X[idx], X[idx]).astype(np.float32)
    E = eligible_mask(init[idx], valid[idx], split[idx], mode)
    return np.where(E, D, np.inf)


def compute_memory_features(df: pd.DataFrame, mode: str | None = None, neighbour_rows=None):
    """Adds MEM features to df (copy).

    `neighbour_rows`: optional iterable of row positions whose analogue indices are
    returned (used to build replay evidence)."""
    cfg = model_config()["analogues"]
    mode = mode or cfg["memory_mode"]
    k, kmin = cfg["k"], cfg["min_analogues"]
    cols = analogue_space(df)
    train = df[df["split"] == "train"]
    scaler = fit_scaler(train, cols)
    X = standardise(df, cols, scaler)
    X = np.nan_to_num(X)
    out = {c: np.full(len(df), np.nan, dtype=np.float32) for c in
           ["an_n_within", "an_dist1", "an_dist_mean", "an_bust_rate", "an_err_med", "an_err_q90", "an_n_eligible"]}
    want = set(neighbour_rows) if neighbour_rows is not None else set()
    neigh = {}
    init = df["init_time"].values.astype("datetime64[ns]")
    valid = df["valid_time"].values.astype("datetime64[ns]")
    split = df["split"].values
    bust = df["bust"].values.astype(float)
    err = df["norm_error"].values.astype(float)

    # radius: TRAIN median distance to the k-th eligible neighbour (fitted in a first pass)
    groups = df.groupby(["region_id", "lead_day"]).indices
    kth_train = []
    for key, idx in groups.items():
        idx = np.asarray(idx)
        D = _group_distances(X, init, valid, split, idx, mode)
        is_tr = split[idx] == "train"
        if is_tr.sum() > k:
            srt = np.sort(D[is_tr], axis=1)[:, k - 1]
            kth_train.append(srt[np.isfinite(srt)])
    radius = float(np.median(np.concatenate(kth_train))) if kth_train else np.inf

    for key, idx in groups.items():
        idx = np.asarray(idx)
        D = _group_distances(X, init, valid, split, idx, mode)
        n_elig = np.isfinite(D).sum(1)
        order = np.argsort(D, axis=1)[:, :k]
        dk = np.take_along_axis(D, order, axis=1)
        valid_nb = np.isfinite(dk)
        cnt = valid_nb.sum(1)
        cand = idx[order]
        b = np.where(valid_nb, bust[cand], np.nan)
        e = np.where(valid_nb, err[cand], np.nan)
        has = cnt >= kmin
        with np.errstate(all="ignore"):
            out["an_n_eligible"][idx] = n_elig
            out["an_n_within"][idx] = (D <= radius).sum(1)
            out["an_dist1"][idx] = np.where(cnt > 0, dk[:, 0], np.nan)
            out["an_dist_mean"][idx] = np.where(has, np.nanmean(np.where(valid_nb, dk, np.nan), 1), np.nan)
            out["an_bust_rate"][idx] = np.where(has, np.nanmean(b, 1), np.nan)
            out["an_err_med"][idx] = np.where(has, np.nanmedian(e, 1), np.nan)
            out["an_err_q90"][idx] = np.where(has, np.nanquantile(e, 0.9, axis=1), np.nan)
        if want:
            for row in np.where(np.isin(idx, list(want)))[0]:
                neigh[int(idx[row])] = (cand[row][valid_nb[row]], dk[row][valid_nb[row]])
    res = df.copy()
    for c, v in out.items():
        res[c] = v
    meta = {"mode": mode, "k": k, "min_analogues": kmin, "radius": radius, "space": cols,
            "scaler_fitted_on": "train", "radius_definition": "median TRAIN distance to k-th eligible analogue",
            "temporal_rule": "candidate.valid_time <= query.init_time, same region and lead day"}
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / f"analogue_memory_{mode}.json").write_text(json.dumps(clean_json(
        {**meta, "scaler": scaler}), indent=1, default=float))
    return res, meta, neigh
