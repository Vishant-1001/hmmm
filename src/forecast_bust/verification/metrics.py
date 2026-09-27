"""Area-weighted verification metrics."""
from __future__ import annotations

import numpy as np


def area_weights(lats: np.ndarray, lat_spacing: float | None = None) -> np.ndarray:
    """Relative cell area for a regular lat grid.

    Exact for grid boxes: area ~ sin(lat + d/2) - sin(lat - d/2), which is proportional
    to cos(lat) for equal spacing; we use the exact form, clipped at the poles.
    """
    lats = np.asarray(lats, dtype=float)
    if lat_spacing is None:
        lat_spacing = float(np.median(np.abs(np.diff(lats)))) if lats.size > 1 else 1.0
    top = np.deg2rad(np.clip(lats + lat_spacing / 2, -90, 90))
    bot = np.deg2rad(np.clip(lats - lat_spacing / 2, -90, 90))
    return np.sin(top) - np.sin(bot)


def weighted_rmse(forecast: np.ndarray, reference: np.ndarray, lats: np.ndarray,
                  lat_axis: int = -1, lat_spacing: float | None = None) -> np.ndarray:
    """RMSE over the last two (spatial) axes with latitude area weighting.

    `forecast`/`reference` share shape (..., lon, lat) by default (WB2 ordering);
    `lat_axis` selects which axis holds latitude. NaNs are excluded from the mean.
    """
    f = np.asarray(forecast, dtype=float)
    r = np.asarray(reference, dtype=float)
    if f.shape != r.shape:
        raise ValueError(f"shape mismatch {f.shape} vs {r.shape}")
    w1 = area_weights(lats, lat_spacing)
    shape = [1] * f.ndim
    shape[lat_axis] = len(w1)
    w = np.broadcast_to(w1.reshape(shape), f.shape)
    se = (f - r) ** 2
    valid = np.isfinite(se)
    w = np.where(valid, w, 0.0)
    se = np.where(valid, se, 0.0)
    axes = (-2, -1)
    return np.sqrt((se * w).sum(axis=axes) / w.sum(axis=axes))


def normalized_error(rmse: np.ndarray, scale: np.ndarray) -> np.ndarray:
    scale = np.asarray(scale, dtype=float)
    if np.any(scale <= 0):
        raise ValueError("normalisation scale must be positive")
    return np.asarray(rmse, dtype=float) / scale


def spread_skill_ratio(ens_var_mean: float, mse: float, n_members: int) -> float:
    """sqrt((M+1)/M * mean ensemble variance) / RMSE of ensemble mean (Fortin et al. 2014)."""
    return float(np.sqrt((n_members + 1) / n_members * ens_var_mean) / np.sqrt(mse))


def rank_of_reference(members: np.ndarray, reference: np.ndarray, member_axis: int = 0) -> np.ndarray:
    """Rank (0..M) of the reference among ensemble members, for rank histograms."""
    m = np.moveaxis(np.asarray(members), member_axis, 0)
    return (m < np.asarray(reference)[None]).sum(axis=0)
