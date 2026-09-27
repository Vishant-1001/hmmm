"""Historical-support / out-of-distribution diagnostic and evidence-strength rules.

Support distance: regularised (Ledoit-Wolf) Mahalanobis distance of a case's
analogue-space vector from the TRAIN distribution of the same lead day.
Category thresholds are TRAIN quantiles of that distance (config support.quantiles).
No label of any split is used.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from forecast_bust.analogues.memory import analogue_space
from forecast_bust.config import clean_json, ARTIFACT_DIR, model_config

SUPPORT_LEVELS = ["NORMAL SUPPORT", "MODERATE SUPPORT", "WEAK SUPPORT", "INSUFFICIENT HISTORICAL SUPPORT"]
EVIDENCE_LEVELS = ["STRONG EVIDENCE", "MODERATE EVIDENCE", "WEAK EVIDENCE", "INSUFFICIENT HISTORICAL SUPPORT"]


def fit_support(train: pd.DataFrame) -> dict:
    cols = analogue_space(train)
    qs = model_config()["support"]["quantiles"]
    models = {}
    for lead, g in train.groupby("lead_day"):
        X = g[cols].to_numpy(dtype=float)
        X = X[np.isfinite(X).all(1)]
        lw = LedoitWolf().fit(X)
        d = np.sqrt(lw.mahalanobis(X))
        models[int(lead)] = {"location": lw.location_.tolist(), "precision": lw.precision_.tolist(),
                             "thresholds": np.quantile(d, qs).tolist()}
    obj = {"space": cols, "quantiles": qs, "fitted_on": "train", "per_lead": models,
           "categories": SUPPORT_LEVELS}
    (ARTIFACT_DIR / "support_diagnostics.json").write_text(json.dumps(clean_json(obj), indent=1))
    return obj


def support_distance(df: pd.DataFrame, sup: dict) -> tuple[np.ndarray, np.ndarray]:
    cols = sup["space"]
    dist = np.full(len(df), np.nan)
    level = np.full(len(df), 3, dtype=np.int8)
    for lead, idx in df.groupby("lead_day").indices.items():
        m = sup["per_lead"][str(lead)] if str(lead) in sup["per_lead"] else sup["per_lead"][int(lead)]
        X = df[cols].to_numpy(dtype=float)[idx] - np.asarray(m["location"])
        P = np.asarray(m["precision"])
        d = np.sqrt(np.einsum("ij,jk,ik->i", X, P, X))
        dist[idx] = d
        th = m["thresholds"]
        lv = np.searchsorted(th, d, side="right")  # 0..3
        lv[~np.isfinite(d)] = 3
        level[idx] = lv
    return dist, level


def evidence_strength(support_level: np.ndarray, n_within: np.ndarray, an_bust_rate: np.ndarray,
                      p: np.ndarray, kmin: int) -> np.ndarray:
    """Product rules (documented in docs/scientific_methodology.md):

    INSUFFICIENT : support level INSUFFICIENT, or fewer than `kmin` analogues within radius
    STRONG       : NORMAL support, >= 4*kmin analogues within radius, and the analogue bust
                   rate agrees with the model probability to within 0.25
    MODERATE     : support NORMAL/MODERATE and >= 2*kmin analogues within radius
    WEAK         : everything else
    """
    n = np.nan_to_num(n_within, nan=0)
    agree = np.abs(np.nan_to_num(an_bust_rate, nan=-9) - p) <= 0.25
    out = np.full(len(p), 2, dtype=np.int8)
    out[(support_level <= 1) & (n >= 2 * kmin)] = 1
    out[(support_level == 0) & (n >= 4 * kmin) & agree] = 0
    out[(support_level >= 3) | (n < kmin)] = 3
    return out
