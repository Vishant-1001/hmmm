"""Provider-agnostic B2 inference: ForecastBundle -> ensemble spread -> frozen B2 -> probability.

B2 is unchanged (the locked v2 B2, artifacts/v2/demo/model/b2_booster.json + its validation-2021
isotonic calibrator). Its seven inputs are built exactly as in labels/build.py:

    spread_m          sqrt(mean over the region's grid boxes of ens_std**2), ens_std ddof=1, 500 hPa, metres
    spread_pct        percentile of spread_m in the TRAIN (2018-2020) spread distribution of (region, lead_day, season)
    spread_thr_ratio  spread_m * sqrt(1 + 1/M) / (TRAIN Q90 normalized error x TRAIN scale), M = members in THIS ensemble
    lead_day, region_code (row*100+col), season_code, init_hour

Nothing in the feature path reads verification data; `verify_bundle` (ERA5 labels) is separate and its
output is never passed to `b2_features`. The TRAIN references (spread distribution, Q90, scale) come from
the ECMWF IFS ENS benchmark: applied to another forecast system they are a documented cross-system
transfer, and the result carries `calibration` = the ECMWF validation calibrator unless a
provider-specific calibrator exists in artifacts/b2/<provider>/ (fitted on that provider's own
validation period, never on test).
"""
from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from forecast_bust.config import REPO_ROOT, data_config
from forecast_bust.data.providers import G, ForecastBundle, ensemble_mean_spread
from forecast_bust.data.regions import build_regions, region_members
from forecast_bust.verification.alignment import SEASON_CODES, season_of, valid_time
from forecast_bust.verification.metrics import weighted_rmse

B2_ROOT = REPO_ROOT / "artifacts" / "b2"
B2_SOURCE_RUN = REPO_ROOT / "artifacts" / "v2"
B2_FEATURES = ["spread_m", "spread_pct", "spread_thr_ratio", "lead_day", "region_code", "season_code", "init_hour"]
GROUP = ["region_id", "lead_day", "season"]

# Model/provider modes. b2_ecmwf is the existing benchmark model; b2_ncmrwf is the same frozen B2 applied
# to NCMRWF inputs; synthetic_demo is never a scientific result.
MODES = {
    "b2_ecmwf": {"provider": "ecmwf_research", "artifact_dir": "ecmwf"},
    "b2_ncmrwf": {"provider": "ncmrwf_tigge", "artifact_dir": "ncmrwf"},
    "synthetic_demo": {"provider": "synthetic", "artifact_dir": "../demo/synthetic"},
}
ECMWF_CALIBRATION = "ECMWF_IFS_v2_validation2021_isotonic"


class B2Reference:
    """Frozen TRAIN-only references of the ECMWF benchmark + the exported B2 booster/calibrator."""

    def __init__(self, ref_dir: Path = B2_ROOT / "ecmwf", run_dir: Path = B2_SOURCE_RUN):
        import xgboost as xgb
        self.ref_dir, self.run_dir = Path(ref_dir), Path(run_dir)
        th = json.loads((run_dir / "thresholds.json").read_text())
        pre = json.loads((run_dir / "preprocessing.json").read_text())
        if th.get("fitted_on") != "train" or pre.get("fitted_on") != "train":
            raise ValueError("B2 references must be fitted on TRAIN only")
        self.train_period = pre["split"]["train"]
        self.thresholds = pd.DataFrame(th["thresholds"])[GROUP + ["q_primary"]]
        self.scales = pd.DataFrame(pre["scales"])[["region_id", "season", "scale_m"]]
        sr = pd.read_parquet(self.ref_dir / "spread_reference.parquet")
        self.spread_ref = {k: np.sort(g["spread_m"].to_numpy(float)) for k, g in sr.groupby(GROUP)}
        spec = json.loads((run_dir / "demo" / "model" / "models.json").read_text())["B2"]
        if list(spec["features"]) != B2_FEATURES:
            raise ValueError(f"exported B2 features changed: {spec['features']}")
        self.booster = xgb.Booster()
        self.booster.load_model(run_dir / "demo" / spec["booster"])
        self.best_iteration = int(spec["best_iteration"])
        self.iso_x = np.asarray(spec["isotonic"]["x"], float)
        self.iso_y = np.asarray(spec["isotonic"]["y"], float)

    def spread_percentile(self, df: pd.DataFrame) -> np.ndarray:
        out = np.full(len(df), np.nan)
        for k, idx in df.groupby(GROUP).indices.items():
            r = self.spread_ref.get(k)
            if r is not None:
                out[idx] = np.searchsorted(r, df["spread_m"].to_numpy(float)[idx], side="right") / len(r)
        return out

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        import xgboost as xgb
        return self.booster.predict(xgb.DMatrix(X[B2_FEATURES]), iteration_range=(0, self.best_iteration + 1))

    def calibrate(self, raw: np.ndarray, calibrator: dict | None = None) -> np.ndarray:
        if calibrator is not None:
            return np.interp(raw, np.asarray(calibrator["x"], float), np.asarray(calibrator["y"], float))
        return np.interp(raw, self.iso_x, self.iso_y)


@lru_cache(maxsize=1)
def reference() -> B2Reference:
    return B2Reference()


def _target_regions(lat: np.ndarray, lon: np.ndarray):
    return build_regions(lat, lon, data_config()["target_domain"])


def regional_ensemble_stats(bundle: ForecastBundle) -> pd.DataFrame:
    """Per (region, lead): spread_m, ensemble-mean Z500 (m) and the minimum valid-member count.
    Forecast-only quantities."""
    lvl = data_config()["target_level"]
    da = bundle.data.sel(level=lvl).isel(time=0) / G                     # (number, lead, lon, lat), metres
    mean, std, n = ensemble_mean_spread(da)
    lat, lon = da["latitude"].values, da["longitude"].values
    leads = da["prediction_timedelta"].values.astype(int)
    frames = []
    for r in _target_regions(lat, lon):
        li, oi = region_members(r, lat, lon)
        if not len(li) or not len(oi):
            continue
        s = std.values[:, oi[:, None], li[None, :]]
        frames.append(pd.DataFrame({
            "region_id": r.region_id, "row": r.row, "col": r.col, "lat": r.lat, "lon": r.lon,
            "lead_hours": leads,
            "spread_m": np.sqrt(np.nanmean(s ** 2, axis=(-2, -1))),
            "ens_mean_m": np.nanmean(mean.values[:, oi[:, None], li[None, :]], axis=(-2, -1)),
            "n_members_min": n.values[:, oi[:, None], li[None, :]].min(axis=(-2, -1)),
        }))
    return pd.concat(frames, ignore_index=True)


def b2_features(stats: pd.DataFrame, init_time, n_members: int, ref: B2Reference | None = None) -> pd.DataFrame:
    """The unchanged B2 feature contract from forecast-only regional statistics."""
    ref = ref or reference()
    init = pd.Timestamp(init_time)
    df = stats.copy()
    df["init_time"] = init
    df["valid_time"] = valid_time(np.datetime64(init), df["lead_hours"].to_numpy())
    if (df["lead_hours"] % 24).any():
        raise ValueError("B2 is defined on whole-day leads (24..240 h)")
    df["lead_day"] = (df["lead_hours"] // 24).astype(np.int16)
    df["season"] = season_of(np.repeat(np.datetime64(init), len(df)))
    df["season_code"] = df["season"].map(SEASON_CODES).astype(np.int8)
    df["init_hour"] = np.int8(init.hour)
    df["region_code"] = df["row"] * 100 + df["col"]
    df = df.merge(ref.thresholds, on=GROUP, how="left").merge(ref.scales, on=["region_id", "season"], how="left")
    df["spread_pct"] = ref.spread_percentile(df)
    thr_m = df["q_primary"].to_numpy(float) * df["scale_m"].to_numpy(float)
    df["spread_thr_ratio"] = (df["spread_m"].to_numpy(float) * np.sqrt(1 + 1 / n_members) / thr_m).astype(np.float32)
    df["case_id"] = (init.strftime("%Y%m%d%H") + "_" + df["region_id"] + "_D" + df["lead_day"].astype(str).str.zfill(2))
    return df


def provider_calibrator(mode: str) -> tuple[dict | None, str, bool]:
    """(calibrator or None, calibration label, validated_for_this_provider)."""
    if mode == "b2_ecmwf":
        return None, ECMWF_CALIBRATION, True
    p = (B2_ROOT / MODES[mode]["artifact_dir"] / "calibration.json").resolve()
    if p.is_file():
        c = json.loads(p.read_text())
        if c.get("fitted_on") != "validation":
            raise ValueError(f"{p} is not a validation-only calibrator")
        return c, c["name"], True
    return None, f"{ECMWF_CALIBRATION} (cross-system transfer; NOT validated for this provider)", False


def run_b2(bundle: ForecastBundle, mode: str) -> dict:
    """Run the frozen B2 on one initialisation from any provider. Returns rows + provenance + timings."""
    if mode not in MODES:
        raise KeyError(f"unknown mode {mode!r}; available {sorted(MODES)}")
    md = bundle.metadata
    if MODES[mode]["provider"] != md.provider:
        raise ValueError(f"mode {mode} expects provider {MODES[mode]['provider']}, got {md.provider}")
    t = {}
    t0 = time.perf_counter()
    ref = reference()
    t["load_reference"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    stats = regional_ensemble_stats(bundle)
    t["ensemble_mean_spread"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    X = b2_features(stats, md.initialization_time, md.ensemble_member_count, ref)
    t["features"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    cal, cal_name, cal_ok = provider_calibrator(mode)
    X["raw_probability"] = ref.predict_raw(X)
    X["b2_probability"] = ref.calibrate(X["raw_probability"].to_numpy(), cal)
    t["b2_inference"] = time.perf_counter() - t0
    return {
        "mode": mode, "rows": X,
        "provenance": {**md.to_dict(), "model_version": "B2 v2 (locked 8b196ee; exported artifacts/v2/demo/model/b2_booster.json)",
                       "calibration": cal_name, "calibration_validated_for_provider": cal_ok,
                       "spread_reference": f"ECMWF IFS ENS TRAIN {ref.train_period[0]}..{ref.train_period[1][:10]}",
                       "bust_threshold_reference": "ECMWF/ERA5 TRAIN Q90(region, lead_day, season) - unchanged definition"},
        "timings_ms": {k: round(1000 * v, 2) for k, v in t.items()} | {"total": round(1000 * sum(t.values()), 2)},
    }


def verify_bundle(bundle: ForecastBundle, era5_path: Path | None = None, ref: B2Reference | None = None) -> pd.DataFrame:
    """REAL verification labels for a real-data bundle: ensemble-mean Z500 vs ERA5 Z500 at valid time on the
    same 5.625 deg grid -> regional area-weighted RMSE -> normalized error -> bust (unchanged definition).
    Never used as a B2 input."""
    if bundle.metadata.synthetic:
        raise ValueError("synthetic forecasts have no verification; they are demo-only")
    ref = ref or reference()
    from forecast_bust.config import cache_dir
    era = xr.open_dataset(era5_path or cache_dir() / "era5_z.nc")["z"].sel(level=data_config()["target_level"]) / G
    da = bundle.data.sel(level=data_config()["target_level"]).isel(time=0) / G
    mean, _, _ = ensemble_mean_spread(da)
    lat, lon = mean["latitude"].values, mean["longitude"].values
    if not (np.allclose(lat, era.latitude.sel(latitude=lat, method="nearest").values) and
            np.allclose(lon, era.longitude.sel(longitude=lon, method="nearest").values)):
        raise ValueError("forecast grid does not coincide with the ERA5 verification grid")
    init = np.datetime64(pd.Timestamp(bundle.metadata.initialization_time))
    leads = mean["prediction_timedelta"].values.astype(int)
    vt = valid_time(init, leads)
    ob = era.sel(time=vt, latitude=lat, longitude=lon).transpose("time", "longitude", "latitude").values
    dlat = float(np.median(np.diff(lat)))
    rows = []
    for r in _target_regions(lat, lon):
        li, oi = region_members(r, lat, lon)
        f = mean.values[:, oi[:, None], li[None, :]]
        o = ob[:, oi[:, None], li[None, :]]
        rows.append(pd.DataFrame({"region_id": r.region_id, "lead_hours": leads, "valid_time": vt,
                                  "error_m": weighted_rmse(f, o, lat[li], lat_spacing=dlat)}))
    df = pd.concat(rows, ignore_index=True)
    df["lead_day"] = (df["lead_hours"] // 24).astype(np.int16)
    df["season"] = season_of(np.repeat(init, len(df)))
    df = df.merge(ref.thresholds, on=GROUP, how="left").merge(ref.scales, on=["region_id", "season"], how="left")
    df["norm_error"] = df["error_m"] / df["scale_m"]
    df["bust"] = (df["norm_error"] > df["q_primary"]).astype(np.int8)
    return df
