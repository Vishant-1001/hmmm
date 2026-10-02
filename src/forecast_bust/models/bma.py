"""Bayesian Model Averaging (BMA) of the 50 IFS ENS members (Raftery et al. 2005), adapted to the
project's regional Z500 error target.

Per region r and lead day d, the verifying ERA5 Z500 anomaly z (vs the ERA5 1990-2017 climatology at the
valid time) has the BMA predictive density

    p(z | f_1..f_K) = sum_k w_k N(z; a + b f_k, sigma^2),     w_k = 1/K,

where f_k are the K = 50 member anomalies. The perturbed members are exchangeable by construction, so the
weights are equal and the bias (a, b) and spread (sigma) parameters are common to all members; (a, b) come
from least squares of z on the pooled member forecasts and sigma from EM, as in Raftery et al. (2005).

The project target is the error of the ENSEMBLE MEAN m (labels/build.py): Y = |m - z| / s with the TRAIN
scale s(region, season). Because Y is a deterministic function of z, its distribution follows analytically
from the mixture - no sampling and no interpolation between a few quantiles:

    P(Y <= y) = mean_k [ Phi((m + y s - mu_k) / sigma) - Phi((m - y s - mu_k) / sigma) ],  mu_k = a + b f_k
    P(Y > c)  = the "BMA mixture exceedance probability" of the bust threshold c = q_primary.

This is the exact CDF of the fitted Gaussian mixture model, not an empirical CDF of the target.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import ndtr

QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
QCOLS = [f"q{int(round(q * 100))}" for q in QUANTILES]
MODEL_TYPE = "bayesian_model_averaging"


def fit_exchangeable(f: np.ndarray, y: np.ndarray, valid: np.ndarray, max_iter: int = 200,
                     tol: float = 1e-6) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit G independent exchangeable-member BMA models at once.

    f: (G, n, K) member forecasts, y: (G, n) verifying values, valid: (G, n) bool (padding / missing = False).
    Returns a, b, sigma, n_iter (each (G,)). Weights are fixed at 1/K (exchangeable members)."""
    f = np.asarray(f, np.float64)
    y = np.asarray(y, np.float64)
    v = np.asarray(valid, bool)
    K = f.shape[-1]
    n = v.sum(1).astype(np.float64)
    wv = v[..., None] * np.ones(K)                       # (G, n, K) pooled-OLS weights
    cnt = wv.sum((1, 2))
    fbar = (f * wv).sum((1, 2)) / cnt
    ybar = (y * v).sum(1) / n
    df = (f - fbar[:, None, None]) * wv
    b = (df * (y - ybar[:, None])[..., None]).sum((1, 2)) / (df * (f - fbar[:, None, None])).sum((1, 2))
    a = ybar - b * fbar
    r2 = (y[..., None] - (a[:, None, None] + b[:, None, None] * f)) ** 2   # (G, n, K)
    s2 = (r2 * wv).sum((1, 2)) / cnt
    it = np.zeros(len(a), int)
    active = np.ones(len(a), bool)
    for i in range(max_iter):
        lg = -0.5 * r2 / s2[:, None, None]
        lg -= lg.max(-1, keepdims=True)
        z = np.exp(lg)
        z /= z.sum(-1, keepdims=True)                     # responsibilities (equal prior weights cancel)
        new = (z * r2 * v[..., None]).sum((1, 2)) / n
        new = np.maximum(new, 1e-6)
        done = np.abs(new - s2) <= tol * s2
        s2 = np.where(active, new, s2)
        it += active
        active &= ~done
        if not active.any():
            break
    return a, b, np.sqrt(s2), it


def fit_free_weights(f: np.ndarray, y: np.ndarray, max_iter: int = 500, tol: float = 1e-7) -> dict:
    """Diagnostic only (not used for prediction): the original Raftery et al. (2005) parameterisation with a
    member-specific bias (a_k, b_k) and weight w_k, for ONE group. f: (n, K), y: (n,)."""
    n, K = f.shape
    A = np.stack([np.ones(n), np.zeros(n)], 1)
    ab = np.empty((K, 2))
    for k in range(K):
        A[:, 1] = f[:, k]
        ab[k] = np.linalg.lstsq(A, y, rcond=None)[0]
    r2 = (y[:, None] - (ab[:, 0] + ab[:, 1] * f)) ** 2
    w = np.full(K, 1 / K)
    s2 = float(r2.mean())
    ll_old = -np.inf
    for _ in range(max_iter):
        dens = w * np.exp(-0.5 * r2 / s2) / np.sqrt(2 * np.pi * s2)
        tot = dens.sum(1, keepdims=True)
        ll = float(np.log(tot).sum())
        z = dens / tot
        w = z.mean(0)
        s2 = float((z * r2).sum() / n)
        if ll - ll_old < tol * abs(ll):
            break
        ll_old = ll
    return {"weights": w, "a": ab[:, 0], "b": ab[:, 1], "sigma": np.sqrt(s2), "loglik": ll}


# ----------------------------------------------------------------------------------- mixture for Y
def error_cdf(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, m: np.ndarray, s: np.ndarray) -> np.ndarray:
    """P(Y <= y) for Y = |m - Z| / s, Z ~ mean_k N(mu_k, sigma^2). mu: (n, K); others (n,) ; y (n,) >= 0."""
    y = np.maximum(np.asarray(y, np.float64), 0.0)[:, None]
    sg = np.asarray(sigma, np.float64)[:, None]
    hi = (m[:, None] + y * s[:, None] - mu) / sg
    lo = (m[:, None] - y * s[:, None] - mu) / sg
    return (ndtr(hi) - ndtr(lo)).mean(1)


def exceedance(c: np.ndarray, mu, sigma, m, s) -> np.ndarray:
    """BMA mixture exceedance probability P(Y > c)."""
    return np.clip(1.0 - error_cdf(c, mu, sigma, m, s), 0.0, 1.0)


def error_quantiles(mu, sigma, m, s, levels=QUANTILES, n_iter: int = 50) -> np.ndarray:
    """(n, len(levels)) quantiles of Y by bisection on the exact mixture CDF (monotone in y)."""
    sg = np.asarray(sigma, np.float64)
    upper = (np.abs(m[:, None] - mu).max(1) + 9 * sg) / s
    out = np.empty((len(m), len(levels)))
    for j, p in enumerate(levels):
        lo, hi = np.zeros(len(m)), upper.copy()
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            below = error_cdf(mid, mu, sg, m, s) < p
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        out[:, j] = 0.5 * (lo + hi)
    return out


def error_moments(mu, sigma, m, s) -> tuple[np.ndarray, np.ndarray]:
    """Predictive mean and standard deviation of Y (folded-normal moments, averaged over members)."""
    sg = np.asarray(sigma, np.float64)[:, None]
    d = mu - m[:, None]
    e_abs = sg * np.sqrt(2 / np.pi) * np.exp(-0.5 * (d / sg) ** 2) + d * (1 - 2 * ndtr(-d / sg))
    mean = e_abs.mean(1) / s
    second = (d ** 2 + sg ** 2).mean(1) / s ** 2
    return mean, np.sqrt(np.maximum(second - mean ** 2, 0.0))


class BMAModel:
    """Rolling exchangeable BMA: a table of (fit_day, region_id, lead_day) -> a, b, sigma, n_cases, plus an
    optional validation-fitted isotonic calibrator of the mixture exceedance probability."""

    def __init__(self, params: pd.DataFrame, config: dict, calibrator=None):
        self.params = params
        self.config = config
        self.calibrator = calibrator
        self.model_type = MODEL_TYPE

    def predict(self, rows: pd.DataFrame, members: np.ndarray, chunk: int = 100_000) -> pd.DataFrame:
        """rows: case rows with init_time, region_id, lead_day, ens_mean_anom, scale_m, q_primary;
        members: (len(rows), K) member anomalies aligned with rows. Rows without fitted parameters -> NaN."""
        key = rows[["region_id", "lead_day"]].copy()
        key["fit_day"] = pd.to_datetime(rows["init_time"]).dt.floor("D").to_numpy()
        p = key.merge(self.params, on=["fit_day", "region_id", "lead_day"], how="left")
        out = pd.DataFrame(index=rows.index, columns=["bma_mean", "bma_std", *QCOLS, "bma_exceedance_probability",
                                                      "bma_a", "bma_b", "bma_sigma", "bma_n_cases"], dtype=float)
        ok = p["sigma"].notna().to_numpy()
        idx = np.nonzero(ok)[0]
        m_all = rows["ens_mean_anom"].to_numpy(np.float64)
        s_all = rows["scale_m"].to_numpy(np.float64)
        c_all = rows["q_primary"].to_numpy(np.float64)
        for i0 in range(0, len(idx), chunk):
            ii = idx[i0:i0 + chunk]
            a, b, sg = (p[c].to_numpy(np.float64)[ii] for c in ("a", "b", "sigma"))
            mu = a[:, None] + b[:, None] * members[ii].astype(np.float64)
            m, s = m_all[ii], s_all[ii]
            mean, std = error_moments(mu, sg, m, s)
            q = error_quantiles(mu, sg, m, s)
            pe = exceedance(c_all[ii], mu, sg, m, s)
            block = np.column_stack([mean, std, q, pe, a, b, sg, p["n_cases"].to_numpy()[ii]])
            out.iloc[ii, :] = block
        if self.calibrator is not None:
            raw = out["bma_exceedance_probability"].to_numpy(float)
            cal = np.full(len(raw), np.nan)
            f = np.isfinite(raw)
            cal[f] = self.calibrator.predict(raw[f])
            out["calibrated_bust_probability"] = cal
        return out

    def save(self, d: Path, metadata: dict) -> None:
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        self.params.to_parquet(d / "bma_params.parquet", index=False)
        if self.calibrator is not None:
            joblib.dump(self.calibrator, d / "calibration.joblib")
        (d / "metadata.json").write_text(json.dumps({**metadata, "config": self.config, "model_type": MODEL_TYPE},
                                                    indent=1, default=str))

    @classmethod
    def load(cls, d: Path) -> "BMAModel":
        d = Path(d)
        md = json.loads((d / "metadata.json").read_text())
        cal = joblib.load(d / "calibration.joblib") if (d / "calibration.joblib").exists() else None
        return cls(pd.read_parquet(d / "bma_params.parquet"), md["config"], cal)
