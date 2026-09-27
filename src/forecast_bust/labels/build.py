"""Case table, regional verification error, training-only normalisation and bust labels.

One row = (init_time, region, lead_day). Everything here that is *learned*
(normalisation scale, Q90/Q95 thresholds, spread percentiles) is fitted on
TRAIN rows only and persisted to artifacts/preprocessing.json and
artifacts/thresholds.json.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import xarray as xr

from forecast_bust.config import clean_json, ARTIFACT_DIR, INTERIM_DIR, data_config, model_config
from forecast_bust.data.regions import Region, build_regions, region_members
from forecast_bust.labels.signature import failure_signature
from forecast_bust.verification.alignment import SEASON_CODES, season_of, split_of, valid_time
from forecast_bust.verification.metrics import weighted_rmse

log = logging.getLogger(__name__)
GROUP = ["region_id", "lead_day", "season"]


def cases_path():
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    return INTERIM_DIR / "cases.parquet"


def regions_for(ds: xr.Dataset) -> list[Region]:
    return build_regions(ds.latitude.values, ds.longitude.values, data_config()["target_domain"])


def base_table(ds: xr.Dataset) -> pd.DataFrame:
    """Verification error, spread and failure signature for every (init, region, lead)."""
    cfg, mcfg = data_config(), model_config()
    lat, lon = ds.latitude.values, ds.longitude.values
    dlat = float(np.median(np.diff(lat)))
    lvl = int(np.where(ds.level.values == cfg["target_level"])[0][0])
    fm = ds["ens_mean"].values[:, :, lvl]  # (init, lead, lon, lat)
    fs = ds["ens_std"].values[:, :, lvl]
    cl = ds["clim"].values[:, :, lvl]
    ob = ds["era5_z500"].values
    inits = ds.init.values
    leads = ds.lead.values
    n_i, n_l = len(inits), len(leads)
    vt = valid_time(inits[:, None], leads[None, :])
    frames = []
    w = mcfg["signature"]["window"]
    for r in regions_for(ds):
        li, oi = region_members(r, lat, lon)
        sl = (slice(None), slice(None), oi[:, None], li[None, :])
        err = weighted_rmse(fm[sl], ob[sl], lat[li], lat_spacing=dlat)
        spread = np.sqrt(np.mean(fs[sl] ** 2, axis=(-2, -1)))
        o_anom = (ob[sl] - cl[sl]).mean(axis=(-2, -1))
        # neighbourhood window for the failure signature (verification-time quantity)
        wo = slice(max(oi.min() - w, 0), oi.max() + w + 1)
        wl = slice(max(li.min() - w, 0), li.max() + w + 1)
        sig = failure_signature(fm[:, :, wo, wl], ob[:, :, wo, wl], lat[wl], mcfg["signature"]["dominance"])
        frames.append(pd.DataFrame({
            "init_time": np.repeat(inits, n_l),
            "lead_hours": np.tile(leads, n_i),
            "valid_time": vt.ravel(),
            "region_id": r.region_id, "row": r.row, "col": r.col, "lat": r.lat, "lon": r.lon,
            "error_m": err.ravel(), "spread_m": spread.ravel(), "era5_anom_m": o_anom.ravel(),
            "sig_class": sig["cls"].ravel(), "sig_phase_share": sig["phase_share"].ravel(),
            "sig_amp_share": sig["amp_share"].ravel(), "sig_bias_m": sig["bias"].ravel(),
            "sig_corr": sig["corr"].ravel(),
        }))
    df = pd.concat(frames, ignore_index=True)
    df["lead_day"] = (df["lead_hours"] // 24).astype(np.int16)
    df["season"] = season_of(df["init_time"].values)
    df["season_code"] = df["season"].map(SEASON_CODES).astype(np.int8)
    df["split"] = split_of(df["init_time"].values, cfg["split"])
    # Purge: a train/validation row whose verification falls after its period ends would
    # carry the next period's reference data into fitting; such rows are embargoed.
    for name in ("train", "validation"):
        end = pd.Timestamp(cfg["split"][name][1])
        df.loc[(df["split"] == name) & (df["valid_time"] > end), "split"] = "embargo"
    df["verified_by"] = df["valid_time"]  # label becomes known at valid time
    return df


def fit_normalisation(train: pd.DataFrame) -> pd.DataFrame:
    """Scale(region, season) = std of ERA5 Z500 anomaly over TRAIN valid times."""
    uniq = train.drop_duplicates(["region_id", "valid_time"])
    sc = uniq.groupby(["region_id", "season"])["era5_anom_m"].std().rename("scale_m").reset_index()
    if (sc["scale_m"] <= 0).any() or sc["scale_m"].isna().any():
        raise ValueError("degenerate normalisation scale")
    return sc


def fit_thresholds(train: pd.DataFrame, q: float, q_sens: float, q_spread: float) -> pd.DataFrame:
    g = train.groupby(GROUP)
    return pd.DataFrame({
        "q_primary": g["norm_error"].quantile(q),
        "q_sensitivity": g["norm_error"].quantile(q_sens),
        "spread_low": g["spread_m"].quantile(q_spread),
        "n_train": g.size(),
    }).reset_index()


def spread_percentile(df: pd.DataFrame, train: pd.DataFrame) -> np.ndarray:
    """Percentile of each row's spread within the TRAIN distribution of its group."""
    out = np.full(len(df), np.nan)
    ref = {k: np.sort(v.values) for k, v in train.groupby(GROUP)["spread_m"]}
    for k, idx in df.groupby(GROUP).indices.items():
        r = ref.get(k)
        if r is None:
            continue
        out[idx] = np.searchsorted(r, df["spread_m"].values[idx], side="right") / len(r)
    return out


def build_cases(ds: xr.Dataset) -> pd.DataFrame:
    mcfg = model_config()["labels"]
    df = base_table(ds)
    train = df[df["split"] == "train"]
    if train.empty:
        raise ValueError("no training rows available")
    sc = fit_normalisation(train)
    df = df.merge(sc, on=["region_id", "season"], how="left")
    df["norm_error"] = df["error_m"] / df["scale_m"]
    train = df[df["split"] == "train"]
    th = fit_thresholds(train, mcfg["primary_quantile"], mcfg["sensitivity_quantile"],
                        mcfg["hidden_bust_spread_quantile"])
    df = df.merge(th.drop(columns="n_train"), on=GROUP, how="left")
    df["bust"] = (df["norm_error"] > df["q_primary"]).astype(np.int8)
    df["bust_q95"] = (df["norm_error"] > df["q_sensitivity"]).astype(np.int8)
    df["low_spread"] = (df["spread_m"] <= df["spread_low"]).astype(np.int8)
    df["hidden_bust"] = (df["bust"].astype(bool) & df["low_spread"].astype(bool)).astype(np.int8)
    df["spread_pct"] = spread_percentile(df, df[df["split"] == "train"])
    df = df.sort_values(["init_time", "region_id", "lead_day"]).reset_index(drop=True)
    df["case_id"] = (pd.to_datetime(df["init_time"]).dt.strftime("%Y%m%d%H") + "_" + df["region_id"]
                     + "_D" + df["lead_day"].astype(str).str.zfill(2))

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "preprocessing.json").write_text(json.dumps(clean_json({
        "normalisation": "normalized_error = regional RMSE / scale(region, season); scale = std of ERA5 Z500 "
                         "anomaly (vs ERA5 1990-2017 climatology) over TRAIN valid times only",
        "fitted_on": "train", "split": data_config()["split"],
        "scales": sc.to_dict(orient="records"),
    }), indent=1, default=str))
    (ARTIFACT_DIR / "thresholds.json").write_text(json.dumps(clean_json({
        "definition": "bust = normalized_error > Q90 of TRAIN normalized_error within (region, lead_day, season). "
                      "The 90th-percentile threshold is a project-defined operational bust criterion.",
        "sensitivity": "Q95 of the same TRAIN distribution",
        "hidden_bust": "bust AND spread <= TRAIN 25th percentile of spread within (region, lead_day, season); "
                       "a documented project diagnostic",
        "fitted_on": "train", "conditioning": GROUP,
        "thresholds": th.to_dict(orient="records"),
    }), indent=1, default=str))
    summary = df.groupby("split").agg(rows=("bust", "size"), bust_rate=("bust", "mean"),
                                      bust_q95_rate=("bust_q95", "mean"), hidden_rate=("hidden_bust", "mean"),
                                      inits=("init_time", "nunique"))
    log.info("case table:\n%s", summary)
    df.to_parquet(cases_path(), index=False)
    return df


def load_cases() -> pd.DataFrame:
    return pd.read_parquet(cases_path())


if __name__ == "__main__":
    from forecast_bust.data.assemble import load_states

    logging.basicConfig(level=logging.INFO)
    build_cases(load_states())
