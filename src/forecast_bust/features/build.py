"""Feature groups (all computed from information available at forecast initialisation).

Inputs are ONLY forecast quantities (ens_mean, ens_std, member statistics m_*) and the
1990-2017 ERA5 climatology (which pre-dates every experiment period). The verification
fields era5_z500 / era5_rank are never read here; tests/test_leakage.py enforces this.

Groups
  SPREAD : B2 inputs            spread magnitude, spread percentile, lead, region, season, init hour
  ATM    : atmospheric state    Z500/700/850 anomalies, 500-850 thickness anomaly, gradients, curvature
  ENS    : ensemble behaviour   IQR, P10-P90, skewness, anomaly-sign agreement, neighbourhood spread ...
  PAT    : large-scale pattern  frozen TRAIN-fitted PCA of context-domain Z500 anomaly
  EVO    : forecast evolution   revision vs the cycle 24 h earlier for the same valid time
  MEM    : historical memory    analogue statistics (see forecast_bust.analogues.memory)
  REC    : recent verified error behaviour of this NWP system (see forecast_bust.analogues.recent)
"""
from __future__ import annotations

import json
import logging

import joblib
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.decomposition import PCA

from forecast_bust.config import clean_json, ARTIFACT_DIR, MODEL_DIR, data_config, model_config
from forecast_bust.data.regions import region_members
from forecast_bust.labels.build import regions_for

log = logging.getLogger(__name__)

SPREAD = ["spread_m", "spread_pct", "lead_day", "region_code", "season_code", "init_hour"]
ATM = ["anom500", "abs_anom500", "anom700", "anom850", "thick_anom", "grad_x", "grad_y", "grad_mag",
       "lap500", "nbhd_anom500"]
ENS = ["m_iqr", "m_p10p90", "m_skew", "m_sign_agree", "spread_nbhd", "spread_hetero", "spread700",
       "spread850", "spread_growth", "domain_spread", "spread_rel_domain"]
PAT = [f"pc{i + 1}" for i in range(model_config()["pca"]["n_components"])] + ["pc_norm", "pc_resid"]
EVO = ["rev_region", "rev_nbhd", "rev_pc", "rev_spread", "rev_available"]
MEM = ["an_n_within", "an_dist1", "an_dist_mean", "an_bust_rate", "an_err_med", "an_err_q90", "an_n_eligible"]
REC = ["rec_err", "rec_bias", "rec_bust_rate", "rec_n", "rec_err_nbhd", "rec_err_domain"]
GROUPS = {"SPREAD": SPREAD, "ATM": ATM, "ENS": ENS, "PAT": PAT, "EVO": EVO, "MEM": MEM, "REC": REC}
FORBIDDEN_INPUTS = ("era5_z500", "era5_rank", "error_m", "norm_error", "bust", "sig_", "era5_anom_m", "bias_m")


def _grid_fields(ds: xr.Dataset):
    cfg = data_config()
    lv = {int(v): i for i, v in enumerate(ds.level.values)}
    em = ds["ens_mean"].values
    es = ds["ens_std"].values
    cl = ds["clim"].values
    t = lv[cfg["target_level"]]
    return {
        "anom500": em[:, :, t] - cl[:, :, t],
        "anom700": em[:, :, lv[700]] - cl[:, :, lv[700]],
        "anom850": em[:, :, lv[850]] - cl[:, :, lv[850]],
        "thick_anom": (em[:, :, t] - em[:, :, lv[850]]) - (cl[:, :, t] - cl[:, :, lv[850]]),
        "z500": em[:, :, t],
        "spread500": es[:, :, t], "spread700": es[:, :, lv[700]], "spread850": es[:, :, lv[850]],
    }


def _gradients(z: np.ndarray, lat: np.ndarray, lon: np.ndarray):
    """Centred-difference gradients of (..., lon, lat) fields in m per 100 km."""
    r = 6371.0
    dx = np.deg2rad(np.median(np.diff(lon))) * r * np.cos(np.deg2rad(lat))  # km, per lat
    dy = np.deg2rad(np.median(np.diff(lat))) * r
    gx = np.gradient(z, axis=-2) / dx * 100
    gy = np.gradient(z, axis=-1) / dy * 100
    lap = (np.roll(z, 1, -2) + np.roll(z, -1, -2) + np.roll(z, 1, -1) + np.roll(z, -1, -1) - 4 * z)
    lap[..., 0, :] = lap[..., -1, :] = lap[..., :, 0] = lap[..., :, -1] = np.nan
    return gx, gy, lap


def fit_pca(ds: xr.Dataset, train_mask_init: np.ndarray, seed: int) -> PCA:
    """Fit on TRAIN initialisations only (all leads pooled); persisted and frozen."""
    f = _grid_fields(ds)["anom500"]
    lat = ds.latitude.values
    w = np.sqrt(np.cos(np.deg2rad(lat)))[None, None, None, :]
    X = (f * w)[train_mask_init].reshape(-1, f.shape[-2] * f.shape[-1])
    pca = PCA(n_components=model_config()["pca"]["n_components"], random_state=seed).fit(X)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pca, MODEL_DIR / "pca.joblib")
    (ARTIFACT_DIR / "pca.json").write_text(json.dumps(clean_json({
        "fitted_on": "train initialisations only, all Day 1-10 leads pooled",
        "field": "ensemble-mean Z500 anomaly vs ERA5 1990-2017 climatology, sqrt(cos lat) weighted, context domain",
        "n_components": int(pca.n_components_),
        "explained_variance_ratio": pca.explained_variance_ratio_.round(5).tolist(),
        "n_train_fields": int(X.shape[0]),
    }), indent=1))
    return pca


def pca_project(ds: xr.Dataset, pca: PCA) -> tuple[np.ndarray, np.ndarray]:
    f = _grid_fields(ds)["anom500"]
    lat = ds.latitude.values
    w = np.sqrt(np.cos(np.deg2rad(lat)))[None, None, None, :]
    X = (f * w).reshape(-1, f.shape[-2] * f.shape[-1])
    pcs = pca.transform(X)
    recon = pca.inverse_transform(pcs)
    resid = ((X - recon) ** 2).sum(1) / np.maximum(((X - pca.mean_) ** 2).sum(1), 1e-9)
    n_i, n_l = f.shape[:2]
    return pcs.reshape(n_i, n_l, -1), resid.reshape(n_i, n_l)


def grid_features(ds: xr.Dataset, pca: PCA) -> pd.DataFrame:
    """Per (init, lead, region) forecast-only features (ATM, ENS, PAT, EVO and SPREAD raw)."""
    lat, lon = ds.latitude.values, ds.longitude.values
    g = _grid_fields(ds)
    gx, gy, lap = _gradients(g["z500"], lat, lon)
    pcs, resid = pca_project(ds, pca)
    pc_std = np.sqrt(pca.explained_variance_)
    inits = ds.init.values
    leads = ds.lead.values
    n_i, n_l = len(inits), len(leads)
    # domain-mean spread per (init, lead) over target boxes
    rl = np.isin(lon, ds.rlon.values)
    rt = np.isin(lat, ds.rlat.values)
    domain_spread = np.sqrt(np.mean(g["spread500"][:, :, rl][:, :, :, rt] ** 2, axis=(-2, -1)))
    # evolution: previous cycle (init - 24 h) at lead + 24 h has the same valid time
    init_index = {pd.Timestamp(t): i for i, t in enumerate(inits)}
    prev_idx = np.array([init_index.get(pd.Timestamp(t) - pd.Timedelta(hours=24), -1) for t in inits])
    lead_pos = {int(h): j for j, h in enumerate(leads)}
    prev_lead = np.array([lead_pos.get(int(h) + 24, -1) for h in leads])
    rlon_list = list(ds.rlon.values)
    rlat_list = list(ds.rlat.values)
    frames = []
    for r in regions_for(ds):
        li, oi = region_members(r, lat, lon)
        o, l_ = oi[0], li[0]
        nb = (slice(None), slice(None), slice(max(o - 1, 0), o + 2), slice(max(l_ - 1, 0), l_ + 2))

        def reg(a):
            return a[:, :, oi[:, None], li[None, :]].mean(axis=(-2, -1))

        ro, rl_ = rlon_list.index(lon[o]), rlat_list.index(lat[l_])
        sp = reg(g["spread500"])
        sp_nb = g["spread500"][nb]
        d = {
            "anom500": reg(g["anom500"]), "anom700": reg(g["anom700"]), "anom850": reg(g["anom850"]),
            "thick_anom": reg(g["thick_anom"]), "grad_x": reg(gx), "grad_y": reg(gy),
            "lap500": reg(lap), "nbhd_anom500": g["anom500"][nb].mean(axis=(-2, -1)),
            "m_iqr": ds["m_iqr"].values[:, :, ro, rl_], "m_p10p90": ds["m_p10p90"].values[:, :, ro, rl_],
            "m_skew": ds["m_skew"].values[:, :, ro, rl_], "m_sign_agree": ds["m_sign_agree"].values[:, :, ro, rl_],
            "spread_nbhd": sp_nb.mean(axis=(-2, -1)),
            "spread_hetero": sp_nb.std(axis=(-2, -1)) / np.maximum(sp_nb.mean(axis=(-2, -1)), 1e-6),
            "spread700": reg(g["spread700"]), "spread850": reg(g["spread850"]),
            "spread_growth": sp / np.maximum(sp[:, :1], 1e-6),
            "domain_spread": domain_spread, "spread_rel_domain": sp / np.maximum(domain_spread, 1e-6),
            "pc_resid": resid, "pc_norm": np.sqrt(((pcs / pc_std) ** 2).sum(-1)),
        }
        for k in range(pcs.shape[-1]):
            d[f"pc{k + 1}"] = pcs[:, :, k]
        # evolution features
        zr = reg(g["z500"])
        z_nb = g["z500"][nb]
        rev_region = np.full((n_i, n_l), np.nan)
        rev_nbhd = np.full((n_i, n_l), np.nan)
        rev_pc = np.full((n_i, n_l), np.nan)
        rev_spread = np.full((n_i, n_l), np.nan)
        ok_i = prev_idx >= 0
        ok_l = prev_lead >= 0
        ii = np.where(ok_i)[0]
        jj = np.where(ok_l)[0]
        if len(ii) and len(jj):
            pi = prev_idx[ii][:, None]
            pj = prev_lead[jj][None, :]
            cur = (ii[:, None], jj[None, :])
            rev_region[cur] = np.abs(zr[cur] - zr[pi, pj])
            rev_nbhd[cur] = np.sqrt(((z_nb[cur] - z_nb[pi, pj]) ** 2).mean(axis=(-2, -1)))
            rev_pc[cur] = np.sqrt((((pcs[cur] - pcs[pi, pj]) / pc_std) ** 2).sum(-1))
            rev_spread[cur] = sp[cur] - sp[pi, pj]
        d.update(rev_region=rev_region, rev_nbhd=rev_nbhd, rev_pc=rev_pc, rev_spread=rev_spread,
                 rev_available=np.isfinite(rev_region).astype(np.float32))
        frame = pd.DataFrame({k: np.asarray(v, dtype=np.float32).ravel() for k, v in d.items()})
        frame["init_time"] = np.repeat(inits, n_l)
        frame["lead_hours"] = np.tile(leads, n_i)
        frame["region_id"] = r.region_id
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out["abs_anom500"] = out["anom500"].abs()
    out["grad_mag"] = np.hypot(out["grad_x"], out["grad_y"])
    return out


def add_features(cases: pd.DataFrame, ds: xr.Dataset) -> pd.DataFrame:
    seed = model_config()["seed"]
    train_inits = set(cases.loc[cases["split"] == "train", "init_time"].unique())
    mask = np.array([t in train_inits for t in ds.init.values])
    pca = fit_pca(ds, mask, seed)
    gf = grid_features(ds, pca)
    df = cases.merge(gf, on=["init_time", "lead_hours", "region_id"], how="left", validate="one_to_one")
    df["region_code"] = df["row"] * 100 + df["col"]
    df["init_hour"] = pd.to_datetime(df["init_time"]).dt.hour.astype(np.int8)
    return df
