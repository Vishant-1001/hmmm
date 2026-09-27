"""Deterministic failure-signature decomposition (post-verification).

Over a small neighbourhood window around the region, the area-weighted MSE of the
ensemble-mean forecast F against the verification reference A decomposes exactly as
(Murphy 1988):

    MSE = (mean(F) - mean(A))^2 + (sd(F) - sd(A))^2 + 2 sd(F) sd(A) (1 - r)
          \_______ bias _______/  \___ amplitude ___/  \____ pattern/phase ____/

Project rules (NOT universal meteorological categories):
  * POSITION_PHASE       if the pattern term >= `dominance` of MSE
  * AMPLITUDE_STRUCTURE  if bias + amplitude terms >= `dominance` of MSE
  * RESIDUAL_MIXED       otherwise (no single component dominates)
"""
from __future__ import annotations

import numpy as np

from forecast_bust.verification.metrics import area_weights

CLASSES = np.array(["POSITION_PHASE", "AMPLITUDE_STRUCTURE", "RESIDUAL_MIXED"])
LABELS = {"POSITION_PHASE": "Position / phase mismatch", "AMPLITUDE_STRUCTURE": "Amplitude / structure mismatch",
          "RESIDUAL_MIXED": "Residual / mixed mismatch"}


def failure_signature(f: np.ndarray, a: np.ndarray, lats: np.ndarray, dominance: float = 0.6) -> dict:
    """f, a: (..., lon, lat). Returns arrays of shape (...)."""
    w = area_weights(lats, float(np.median(np.diff(lats))) if len(lats) > 1 else 5.625)
    w = np.broadcast_to(w, f.shape)
    ws = w.sum(axis=(-2, -1))

    def wmean(x):
        return (x * w).sum(axis=(-2, -1)) / ws

    mf, ma = wmean(f), wmean(a)
    df = f - mf[..., None, None]
    da = a - ma[..., None, None]
    sf = np.sqrt(wmean(df ** 2))
    sa = np.sqrt(wmean(da ** 2))
    cov = wmean(df * da)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where((sf > 0) & (sa > 0), cov / (sf * sa), 1.0)
    bias2 = (mf - ma) ** 2
    amp2 = (sf - sa) ** 2
    phase = 2 * sf * sa * (1 - r)
    mse = bias2 + amp2 + phase
    with np.errstate(invalid="ignore", divide="ignore"):
        phase_share = np.where(mse > 0, phase / mse, 0.0)
    amp_share = 1.0 - phase_share
    cls = np.where(phase_share >= dominance, 0, np.where(amp_share >= dominance, 1, 2))
    return {"cls": CLASSES[cls], "phase_share": phase_share, "amp_share": amp_share,
            "bias": mf - ma, "corr": r, "mse": mse}
