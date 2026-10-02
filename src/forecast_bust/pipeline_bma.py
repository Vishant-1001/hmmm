"""BMA experiment pipeline (config/model_bma.yaml). Same rows, target and thresholds as v3:

    FBS_RUN=v3 FBS_SPLIT=dev python -m forecast_bust.pipeline_bma     # dev: calibrate on 2020, score 2021
    FBS_RUN=v3                python -m forecast_bust.pipeline_bma     # final (only after a GO)

1. extract_members(): the 50 genuine IFS ENS member Z500 values at the 64 target boxes, streamed block by block
   from data/cache/ens (the same files and grid that labels/build.py's ensemble mean comes from).
2. rolling_fit(): for every day D and lead d, exchangeable BMA fitted to the cases of that lead VERIFIED in
   (D - window, D] (valid_time <= D 00 UTC); predictions for initialisations on day D use those parameters.
   No case's own outcome or any later outcome is used. The parameters do not depend on the split.
3. predict on the case table rows; isotonic calibration fitted on the validation split only.
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import logging
import time

import joblib
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from sklearn.isotonic import IsotonicRegression

from forecast_bust.config import DEV, REPO_ROOT, RUN, cache_dir, clean_json
from forecast_bust.models.bma import QCOLS, BMAModel, fit_exchangeable, fit_free_weights
from forecast_bust.pipeline import TABLE, _git_rev, _versions

log = logging.getLogger(__name__)
G = 9.80665
MEMBERS = REPO_ROOT / "data" / "interim" / "bma" / "members_z500.npz"
PARAMS = REPO_ROOT / "data" / "interim" / "bma" / "rolling_params.parquet"
ART = REPO_ROOT / "artifacts" / "bma" / ("dev" if DEV else "")
MODEL_DIR = REPO_ROOT / "models" / "bma" / ("dev" if DEV else "")
PRED = REPO_ROOT / "data" / "interim" / "bma" / ("dev" if DEV else "final") / "predictions.parquet"
ROW_COLS = ["case_id", "init_time", "valid_time", "region_id", "row", "col", "lead_day", "split", "scale_m",
            "q_primary", "norm_error", "bust"]


def bma_config() -> dict:
    return yaml.safe_load((REPO_ROOT / "config" / "model_bma.yaml").read_text())


def extract_members() -> dict:
    """members (init, lead, row, col, K) of Z500 anomaly vs the ERA5 climatology at the valid time, plus the
    ERA5 anomaly (verification, used ONLY as the fitting target of past cases) - float32, ~240 MB."""
    if MEMBERS.exists():
        z = np.load(MEMBERS, allow_pickle=False)
        return {k: z[k] for k in z.files}
    st = xr.open_dataset(REPO_ROOT / "data" / "interim" / "states.nc")
    rlon, rlat = st.rlon.values, st.rlat.values
    inits = st.init.values
    pos = {t: i for i, t in enumerate(inits)}
    K = bma_config()["bma"]["members"]
    F = np.full((len(inits), st.sizes["lead"], len(rlat), len(rlon), K), np.nan, np.float32)
    for f in sorted(glob.glob(str(cache_dir() / "ens" / "block_*.nc"))):
        b = xr.open_dataset(f)["z"]
        keep = [t for t in b.time.values if t in pos]
        if not keep:
            continue
        z = (b.sel(time=keep, level=500, longitude=rlon, latitude=rlat).load() / G)
        z = z.transpose("time", "prediction_timedelta", "latitude", "longitude", "number").values
        F[[pos[t] for t in keep]] = z
    sel = dict(level=500, longitude=rlon, latitude=rlat)
    mean = st.ens_mean.sel(**sel).transpose("init", "lead", "latitude", "longitude").values
    clim = st.clim.sel(**sel).transpose("init", "lead", "latitude", "longitude").values
    obs = st.era5_z500.sel(longitude=rlon, latitude=rlat).transpose("init", "lead", "latitude", "longitude").values
    if np.isnan(F).any():
        raise ValueError("missing member values after extraction")
    if np.abs(F.mean(-1) - mean).max() > 0.01:
        raise ValueError("member mean does not reproduce states.nc ens_mean")
    F -= clim[..., None]
    out = {"members": F, "obs_anom": (obs - clim).astype(np.float32), "mean_anom": (mean - clim).astype(np.float32),
           "inits": inits.astype("datetime64[ns]"), "leads": st.lead.values.astype(np.int32)}
    MEMBERS.parent.mkdir(parents=True, exist_ok=True)
    np.savez(MEMBERS, **out)
    return out


def rolling_fit(members: np.ndarray, obs: np.ndarray, inits: np.ndarray, leads_h: np.ndarray,
                window_days: int, min_cases: int, em: dict, days: np.ndarray | None = None) -> pd.DataFrame:
    """members: (I, L, R, C, K) anomalies; obs: (I, L, R, C). Returns one row per (fit_day, lead, row, col)
    with a, b, sigma, n_cases, last_verified (latest valid time used; always <= fit_day)."""
    I, L, R, C, K = members.shape
    inits = inits.astype("datetime64[ns]")
    days = np.unique(inits.astype("datetime64[D]")) if days is None else days
    win = np.timedelta64(window_days, "D")
    out = []
    for li in range(L):
        vt = inits + np.timedelta64(int(leads_h[li]), "h")
        for d in days:
            d0 = d.astype("datetime64[ns]")
            use = np.nonzero((vt <= d0) & (vt > d0 - win))[0]
            if len(use) < min_cases:
                continue
            f = members[use, li].reshape(len(use), R * C, K).transpose(1, 0, 2)    # (R*C, n, K)
            y = obs[use, li].reshape(len(use), R * C).T
            a, b, s, it = fit_exchangeable(f, y, np.ones(y.shape, bool), em["max_iter"], em["tol"])
            out.append(pd.DataFrame({"fit_day": d0, "lead_day": int(leads_h[li] // 24),
                                     "row": np.repeat(np.arange(R), C), "col": np.tile(np.arange(C), R),
                                     "a": a, "b": b, "sigma": s, "n_cases": len(use), "em_iter": it,
                                     "last_verified": vt[use].max()}))
    return pd.concat(out, ignore_index=True)


def member_rows(rows: pd.DataFrame, data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Align (case rows) -> (member anomalies (n, K), ensemble-mean anomaly (n,))."""
    ii = pd.Index(data["inits"]).get_indexer(pd.to_datetime(rows["init_time"]).to_numpy())
    li = pd.Index((data["leads"] // 24).astype(int)).get_indexer(rows["lead_day"].astype(int))
    if (ii < 0).any() or (li < 0).any():
        raise KeyError("case rows not covered by the member archive")
    r, c = rows["row"].to_numpy(), rows["col"].to_numpy()
    return data["members"][ii, li, r, c], data["mean_anom"][ii, li, r, c]


def free_weight_diagnostic(data: dict, train_end: str, n_groups: int = 8, seed: int = 0) -> dict:
    """Section-8 check on TRAIN cases only: fit the original free-weight BMA (member-specific a_k, b_k, w_k)
    for a few random (region, lead) groups and report how far the weights are from 1/K."""
    rng = np.random.default_rng(seed)
    I, L, R, C, K = data["members"].shape
    tr = np.nonzero(data["inits"] + np.timedelta64(240, "h") <= np.datetime64(train_end))[0]
    res = []
    for _ in range(n_groups):
        li, r, c = rng.integers(L), rng.integers(R), rng.integers(C)
        fw = fit_free_weights(data["members"][tr, li, r, c].astype(float), data["obs_anom"][tr, li, r, c].astype(float))
        res.append({"lead_day": int(li + 1), "row": int(r), "col": int(c), "n": int(len(tr)),
                    "w_min": float(fw["weights"].min()), "w_max": float(fw["weights"].max()),
                    "n_weights_below_0.001": int((fw["weights"] < 1e-3).sum()),
                    "top3_members": (np.argsort(-fw["weights"])[:3] + 1).tolist()})
    return {"uniform_weight": 1 / K, "groups": res,
            "note": "free member weights from the original Raftery et al. (2005) EM, TRAIN cases only; members are "
                    "exchangeable perturbations, so which members get large weights is not reproducible meaning"}


def run() -> None:
    if RUN != "v3":
        raise RuntimeError("run with FBS_RUN=v3 (BMA is evaluated on the v3 rows/target/thresholds)")
    t0 = time.time()
    cfg = bma_config()
    bc = cfg["bma"]
    data = extract_members()
    log.info("members %s (%.0f MB) in %.0fs", data["members"].shape, data["members"].nbytes / 1e6, time.time() - t0)
    if PARAMS.exists():
        params = pd.read_parquet(PARAMS)
    else:
        t1 = time.time()
        params = rolling_fit(data["members"], data["obs_anom"], data["inits"], data["leads"], bc["window_days"],
                             bc["min_cases"], bc["em"])
        params.to_parquet(PARAMS, index=False)
        log.info("rolling fit: %d parameter sets in %.0fs", len(params), time.time() - t1)
    if not (params["last_verified"] <= params["fit_day"]).all():
        raise AssertionError("non-causal BMA parameters")

    rows = pd.read_parquet(TABLE, columns=ROW_COLS, filters=[("split", "in", ["train", "validation", "test"])])
    reg = rows.drop_duplicates("region_id")[["region_id", "row", "col"]]
    params = params.merge(reg, on=["row", "col"])
    m, mean_anom = member_rows(rows, data)
    rows["ens_mean_anom"] = mean_anom
    model = BMAModel(params[["fit_day", "region_id", "lead_day", "a", "b", "sigma", "n_cases"]], bc)
    pred = model.predict(rows, m)
    del m
    va = (rows["split"] == "validation").to_numpy() & pred["bma_exceedance_probability"].notna().to_numpy()
    model.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
        pred.loc[va, "bma_exceedance_probability"].to_numpy(float), rows.loc[va, "bust"].to_numpy())
    raw = pred["bma_exceedance_probability"].to_numpy(float)
    ok = np.isfinite(raw)
    pred["calibrated_bust_probability"] = np.nan
    pred.loc[ok, "calibrated_bust_probability"] = model.calibrator.predict(raw[ok])
    pred.insert(0, "case_id", rows["case_id"].to_numpy())
    PRED.parent.mkdir(parents=True, exist_ok=True)
    pred[rows["split"].isin(["validation", "test"]).to_numpy()].to_parquet(PRED, index=False)

    ART.mkdir(parents=True, exist_ok=True)
    sp = rows.groupby("split")["init_time"].agg(["min", "max", "size"])
    diag = free_weight_diagnostic(data, str(sp.loc["train", "max"]))
    stab = params[params["fit_day"] >= np.datetime64("2019-01-01")]
    metadata = {
        "experiment_id": f"bma-{'dev' if DEV else 'final'}-{_git_rev()}", "commit_hash": _git_rev(),
        "created": dt.datetime.now(dt.timezone.utc).isoformat(), "dev_split": DEV,
        "rows_namespace": "v3 (same table, target, thresholds as v2/v3/B2)",
        "members": "50 IFS ENS perturbed members, Z500 at the 64 target boxes (data/cache/ens); member mean "
                   "reproduces the stored ensemble mean exactly",
        "formulation": "exchangeable BMA: w_k = 1/50, common a, b (pooled least squares) and sigma (EM) per "
                       "region x lead, Z500 anomaly vs ERA5 1990-2017 climatology",
        "target_distribution": "Y = |m - Z| / scale with Z ~ BMA mixture: exact mixture CDF, quantiles by bisection",
        "window_days": bc["window_days"], "min_cases": bc["min_cases"],
        "splits": {s: {"rows": int(r["size"]), "first_init": str(r["min"]), "last_init": str(r["max"])}
                   for s, r in sp.iterrows()},
        "calibration": {"method": "isotonic regression of the BMA mixture exceedance probability on the validation "
                                  "split only", "n_rows": int(va.sum())},
        "parameter_stability_since_2019": {c: {q: float(stab[c].quantile(q)) for q in (0.01, 0.5, 0.99)}
                                           for c in ("a", "b", "sigma", "n_cases")},
        "free_weight_diagnostic": diag, "software": _versions(), "runtime_s": round(time.time() - t0, 1),
    }
    model.save(MODEL_DIR, metadata)
    (ART / "experiment_manifest.json").write_text(json.dumps(clean_json(metadata), indent=1, default=str))
    log.info("bma done in %.0fs; rows without prediction: %.4f", time.time() - t0,
             1 - pred["bma_exceedance_probability"].notna().mean())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    run()
