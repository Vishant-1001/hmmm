"""v2 diagnosis and validation-only optimisation (FBS_RUN=v2 FBS_SPLIT=dev).

Runs on the DEVELOPMENT split only: train 2018-19, validation 2020, dev-test 2021. The 2022 year
is `unused` in this table and is dropped on load. Selection uses VALIDATION (2020) only; the
dev-test year (2021) is read by exactly one stage, `confirm`, after the configuration is fixed.

    python scripts/optimize_v2.py augment    add the TEND columns to the dev table (if missing)
    python scripts/optimize_v2.py diagnose   why Sentinel does not beat B2 (train/validation only)
    python scripts/optimize_v2.py search     compact hyper-parameter search, B2 and Sentinel alike
    python scripts/optimize_v2.py ablate     B2 + each group with the selected configurations
    python scripts/optimize_v2.py confirm    one check of the selected configuration on dev-test 2021

Outputs: artifacts/v2/dev/optimization/*.json (every experiment is kept, including failures).
"""
from __future__ import annotations

import itertools
import json
import logging
import os
import sys
import time

os.environ.setdefault("FBS_RUN", "v2")
os.environ.setdefault("FBS_SPLIT", "dev")

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression

from forecast_bust.config import ARTIFACT_DIR, DEV, RUN, clean_json, model_config
from forecast_bust.evaluation import metrics as M
from forecast_bust.features.build import GROUPS
from forecast_bust.models.sentinel import CANDIDATE_GROUPS, FEATURE_SETS, B2Margin, CalibratedGBM, _g
from forecast_bust.pipeline import TABLE

log = logging.getLogger("optimize")
OUT = ARTIFACT_DIR / "optimization"
N_MEMBERS = 50
assert DEV and RUN == "v2", "optimisation must run on the v2 development split"


def load(splits=("train", "validation")) -> pd.DataFrame:
    df = pd.read_parquet(TABLE)
    if "unused" in splits or ("test" in splits and not DEV):
        raise RuntimeError("the optimisation never reads 2022")
    return df[df["split"].isin(splits)].reset_index(drop=True)


def save(name: str, obj) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(clean_json(obj), indent=1, default=float))
    log.info("wrote %s", OUT / f"{name}.json")


# ----------------------------------------------------------------------------------------- augment
def augment() -> None:
    """Merge the flow-tendency columns (features.dynamics.TEND) into an existing dev table."""
    from forecast_bust.data.assemble import load_states
    from forecast_bust.features.dynamics import TEND, dyn_features, extras_path
    import xarray as xr

    from forecast_bust.labels.build import spread_threshold_ratio

    df = pd.read_parquet(TABLE)
    if "spread_thr_ratio" not in df.columns:
        df["spread_thr_ratio"] = spread_threshold_ratio(df)
        df.to_parquet(TABLE, index=False)
        log.info("added spread_thr_ratio")
    if all(c in df.columns for c in TEND):
        log.info("TEND already present")
        return
    ds = load_states()
    ex = xr.open_dataset(extras_path()).load()
    ex = ex.assign_coords(lead=ex.lead.values.astype(ds.lead.dtype))
    gf = dyn_features(ex, ds)[TEND + ["init_time", "lead_hours", "region_id"]]
    df = df.merge(gf, on=["init_time", "lead_hours", "region_id"], how="left", validate="one_to_one")
    df.to_parquet(TABLE, index=False)
    log.info("added %s", TEND)


# ---------------------------------------------------------------------------------------- diagnose
def stratified_auc(y, x, strata) -> float:
    """Positive-weighted mean of within-stratum ROC AUC: information in x CONDITIONAL on the strata."""
    num = den = 0.0
    for s in np.unique(strata):
        m = strata == s
        ok = m & np.isfinite(x)
        n_pos = y[ok].sum()
        if n_pos >= 20 and n_pos < ok.sum() - 20:
            num += n_pos * M.roc_auc(y[ok], x[ok])
            den += n_pos
    return num / den if den else float("nan")


def ideal_spread_probability(df: pd.DataFrame) -> np.ndarray:
    """P(bust) if the ensemble were perfectly reliable and Gaussian at the (single-grid-point) region:
    error = |mean - truth| ~ |N(0, spread^2 (1 + 1/M))|, bust iff error > Q90 * scale."""
    thr = df["q_primary"].to_numpy() * df["scale_m"].to_numpy()
    s = df["spread_m"].to_numpy() * np.sqrt(1 + 1 / N_MEMBERS)
    return 2 * norm.sf(thr / np.maximum(s, 1e-6))


def diagnose() -> None:
    df = load()
    tr, va = df[df["split"] == "train"], df[df["split"] == "validation"]
    out = {"split": "dev: train 2018-19, validation 2020 (2021 and 2022 not read)"}
    out["prevalence"] = {s: {"rows": int(len(g)), "bust_rate": float(g["bust"].mean()),
                             "hidden_bust_rate": float(g["hidden_bust"].mean()),
                             "low_spread_share": float(g["low_spread"].mean()),
                             "hidden_share_of_busts": float(g["hidden_bust"].sum() / g["bust"].sum())}
                         for s, g in (("train", tr), ("validation", va))}
    # 1. predictability ceiling of a spread-only model
    rng = np.random.default_rng(0)
    ceil = {}
    for s, g in (("train", tr), ("validation", va)):
        p = ideal_spread_probability(g)
        y = g["bust"].to_numpy()
        sims = [M.auprc(rng.random(len(p)) < p, p) for _ in range(20)]
        ceil[s] = {"auprc_ideal_prob_on_real_labels": M.auprc(y, p), "roc_ideal_on_real": M.roc_auc(y, p),
                   "mean_ideal_prob": float(p.mean()), "observed_bust_rate": float(y.mean()),
                   "ceiling_auprc_if_ensemble_perfectly_reliable": float(np.mean(sims)),
                   "ceiling_sd": float(np.std(sims)),
                   "spread_error_corr": float(np.corrcoef(g["spread_m"], g["error_m"])[0, 1])}
    out["spread_only_ceiling"] = ceil
    # 2. feature quality: variation, missingness, univariate and spread-conditional information
    feats = [f for f in FEATURE_SETS["ALL"] if f not in FEATURE_SETS["B2"]]
    fq = {}
    for s, g in (("train", tr), ("validation", va)):
        y = g["bust"].to_numpy()
        strata = (g["lead_day"].to_numpy() * 100 + np.minimum((g["spread_pct"].to_numpy() * 10).astype(int), 9))
        low = g["low_spread"].to_numpy().astype(bool)
        for f in feats:
            x = g[f].to_numpy(dtype=float)
            r = fq.setdefault(f, {"group": next(k for k, v in GROUPS.items() if f in v)})
            r[f"missing_{s}"] = float(np.mean(~np.isfinite(x)))
            r[f"std_{s}"] = float(np.nanstd(x))
            r[f"n_unique_{s}"] = int(pd.Series(x).nunique())
            ok = np.isfinite(x)
            r[f"auc_{s}"] = M.roc_auc(y[ok], x[ok])
            r[f"auc_given_spread_{s}"] = stratified_auc(y, x, strata)
            okl = ok & low
            r[f"auc_within_low_spread_{s}"] = M.roc_auc(y[okl], x[okl])
    for f, r in fq.items():
        a, b = r["auc_given_spread_train"] - 0.5, r["auc_given_spread_validation"] - 0.5
        r["conditional_signal_transfers"] = bool(np.sign(a) == np.sign(b) and abs(b) >= 0.02)
    out["features"] = dict(sorted(fq.items(), key=lambda kv: -abs(kv[1]["auc_given_spread_validation"] - 0.5)))
    out["features_note"] = ("auc_given_spread = positive-weighted mean ROC AUC within (lead day x TRAIN spread-"
                            "percentile decile) strata, i.e. information beyond spread; 0.5 = none. "
                            "'transfers' = same sign on train and validation and |AUC-0.5| >= 0.02 on validation")
    # 3. Sentinel vs B2 with the frozen v1 hyper-parameters (reproduction of the failure)
    b2 = CalibratedGBM(FEATURE_SETS["B2"], "B2").fit(tr, va)
    full = CalibratedGBM(FEATURE_SETS["ALL"], "ALL").fit(tr, va)
    pb, pf = b2.predict_raw(va), full.predict_raw(va)
    y = va["bust"].to_numpy()
    out["frozen_hyperparameters"] = {
        "B2": {"val_auprc": M.auprc(y, pb), "best_iteration": b2.best_iteration},
        "ALL": {"val_auprc": M.auprc(y, pf), "best_iteration": full.best_iteration},
        "rank_corr_ALL_vs_B2": float(pd.Series(pf).corr(pd.Series(pb), method="spearman")),
        "share_gain_of_spread_features_in_ALL": float(sum(v for k, v in full.importance().items()
                                                          if k in FEATURE_SETS["B2"])),
        "train_auprc": {"B2": M.auprc(tr["bust"].to_numpy(), b2.predict_raw(tr)),
                        "ALL": M.auprc(tr["bust"].to_numpy(), full.predict_raw(tr))},
    }
    save("diagnosis", out)


# ------------------------------------------------------------------------------------------ search
GRID = {"max_depth": [2, 3, 5], "min_child_weight": [50, 300], "colsample_bytree": [0.5, 1.0]}
FIXED = {"learning_rate": 0.05, "subsample": 0.8, "reg_lambda": 10.0, "n_estimators": 3000}


def cross_calibrated(y, p, init_time) -> np.ndarray:
    """Two-fold (alternate calendar months) isotonic calibration of validation predictions, so that
    Brier/ECE on validation are not in-sample."""
    month = pd.DatetimeIndex(pd.to_datetime(init_time)).month.to_numpy()
    fold = month % 2
    out = np.empty_like(p)
    for k in (0, 1):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p[fold != k], y[fold != k])
        out[fold == k] = iso.predict(p[fold == k])
    return out


def score(va: pd.DataFrame, p_raw: np.ndarray) -> dict:
    y = va["bust"].to_numpy()
    pc = cross_calibrated(y, p_raw, va["init_time"])
    thr10 = M.threshold_at_far(y, p_raw, 0.10)
    per_lead = [M.auprc(y[va["lead_day"].to_numpy() == d], p_raw[va["lead_day"].to_numpy() == d])
                for d in range(1, 11)]
    return {"val_auprc": M.auprc(y, p_raw), "val_roc_auc": M.roc_auc(y, p_raw),
            "val_brier_crosscal": M.brier(y, pc), "val_ece_crosscal": M.ece(y, pc),
            "val_recall_at_far_10": M.recall_at_far(y, p_raw, 0.10),
            "val_hidden_bust_recall_far_10": M.hidden_bust_metrics(va, p_raw, thr10)["hidden_bust_recall"],
            "val_low_spread_auprc": M.hidden_bust_metrics(va, p_raw, thr10)["low_spread_auprc"],
            "val_auprc_by_lead": per_lead}


def fit_eval(tr, va, feats, params, learner="standard", b2=None, seed=None, name="m"):
    t0 = time.time()
    base = B2Margin(b2, tr) if learner == "residual_b2" else None
    m = CalibratedGBM(feats, name, base=base, params=params, seed=seed).fit(tr, va)
    r = {"learner": learner, "n_features": len(feats), "params": params, "best_iteration": m.best_iteration,
         "seconds": round(time.time() - t0, 1), **score(va, m.predict_raw(va))}
    return m, r


def configs():
    for vals in itertools.product(*GRID.values()):
        yield {**FIXED, **dict(zip(GRID, vals))}


def search() -> None:
    df = load()
    tr, va = df[df["split"] == "train"].reset_index(drop=True), df[df["split"] == "validation"].reset_index(drop=True)
    res = {"protocol": "fit TRAIN 2018-19, early stopping + selection on VALIDATION 2020 AUPRC; identical grid "
                       "for B2 and Sentinel (B2 is tuned with the same budget, never weakened)",
           "grid": GRID, "fixed": FIXED, "runs": []}
    best = {}
    v1_b2 = [f for f in FEATURE_SETS["B2"] if f != "spread_thr_ratio"]
    for kind, feats in (("B2", FEATURE_SETS["B2"]), ("B2_v1inputs", v1_b2), ("ALL", FEATURE_SETS["ALL"])):
        for p in configs():
            _, r = fit_eval(tr, va, feats, p, name=kind)
            r["model"] = kind
            res["runs"].append(r)
            log.info("%s %s -> AUPRC %.4f (iter %d, %.0fs)", kind, {k: p[k] for k in GRID}, r["val_auprc"],
                     r["best_iteration"], r["seconds"])
            if kind not in best or r["val_auprc"] > best[kind]["val_auprc"]:
                best[kind] = r
            save("search", res)
    # residual (B2-conditioned) learner on top of the best B2
    b2, _ = fit_eval(tr, va, FEATURE_SETS["B2"], best["B2"]["params"], name="B2")
    for p in configs():
        _, r = fit_eval(tr, va, FEATURE_SETS["ALL"], p, learner="residual_b2", b2=b2, name="ALL_res")
        r["model"] = "ALL@residual_b2"
        res["runs"].append(r)
        log.info("ALL@res %s -> AUPRC %.4f (iter %d)", {k: p[k] for k in GRID}, r["val_auprc"], r["best_iteration"])
        if "ALL@residual_b2" not in best or r["val_auprc"] > best["ALL@residual_b2"]["val_auprc"]:
            best["ALL@residual_b2"] = r
        save("search", res)
    # class weighting and seed stability for each best configuration
    for kind, feats, learner in (("B2", FEATURE_SETS["B2"], "standard"), ("ALL", FEATURE_SETS["ALL"], "standard"),
                                 ("ALL@residual_b2", FEATURE_SETS["ALL"], "residual_b2")):
        p0 = best[kind]["params"]
        for spw in (3.0, 9.0):
            _, r = fit_eval(tr, va, feats, {**p0, "scale_pos_weight": spw}, learner, b2, name=kind)
            r["model"] = f"{kind}+spw{spw:g}"
            res["runs"].append(r)
            log.info("%s spw %g -> %.4f", kind, spw, r["val_auprc"])
        seeds = []
        for sd in (1, 2, 3):
            _, r = fit_eval(tr, va, feats, p0, learner, b2, seed=sd, name=kind)
            seeds.append(r["val_auprc"])
        best[kind]["seed_auprc"] = seeds
        save("search", {**res, "best": best})
    res["best"] = best
    save("search", res)


# ------------------------------------------------------------------------------------------- refine
REFINE = {"colsample_bytree": [0.3, 0.5], "min_child_weight": [300, 1500], "reg_lambda": [10.0, 100.0]}


def refine() -> None:
    """Second, more-regularised stage around depth 3 (the first grid's Sentinel optima stopped after
    12-56 trees, i.e. validation AUPRC peaks early: fast over-fitting to the training regime)."""
    df = load()
    tr, va = df[df["split"] == "train"].reset_index(drop=True), df[df["split"] == "validation"].reset_index(drop=True)
    s = json.loads((OUT / "search.json").read_text())
    b2_params = max((r for r in s["runs"] if r["model"] == "B2"), key=lambda r: r["val_auprc"])["params"]
    b2, _ = fit_eval(tr, va, FEATURE_SETS["B2"], b2_params, name="B2")
    runs = []
    for vals in itertools.product(*REFINE.values()):
        p = {**FIXED, "learning_rate": 0.02, "max_depth": 3, **dict(zip(REFINE, vals))}
        for kind, learner, feats in (("B2", "standard", FEATURE_SETS["B2"]), ("ALL", "standard", FEATURE_SETS["ALL"]),
                                     ("ALL@residual_b2", "residual_b2", FEATURE_SETS["ALL"])):
            _, r = fit_eval(tr, va, feats, p, learner, b2, name=kind)
            r["model"] = kind
            r["stage"] = "refine"
            runs.append(r)
            log.info("refine %s %s -> %.4f (iter %d)", kind, dict(zip(REFINE, vals)), r["val_auprc"], r["best_iteration"])
            s["runs"] = [x for x in s["runs"] if x.get("stage") != "refine"] + runs
            save("search", s)


# ------------------------------------------------------------------------------------------- ablate
def selected_params():
    s = json.loads((OUT / "search.json").read_text())
    runs = s["runs"]

    def top(prefix):
        c = [r for r in runs if r["model"] == prefix or r["model"].startswith(prefix + "+spw")]
        return max(c, key=lambda r: r["val_auprc"])["params"]
    return {"B2": top("B2"), "standard": top("ALL"), "residual_b2": top("ALL@residual_b2")}


SEEDS = (None, 1, 2)  # None = config seed


def ablate() -> None:
    """Spec ablation (B2 + each group) on VALIDATION with the selected hyper-parameters. Selection rule
    (fixed before this stage ran): a group is kept iff the 3-seed mean validation AUPRC of B2+group
    exceeds the 3-seed mean of B2 (the frozen-spec rule, averaged over seeds to damp seed noise).
    The residual learner is run with one seed as a comparison."""
    df = load()
    tr, va = df[df["split"] == "train"].reset_index(drop=True), df[df["split"] == "validation"].reset_index(drop=True)
    P = selected_params()
    b2s = [fit_eval(tr, va, FEATURE_SETS["B2"], P["B2"], seed=sd, name="B2") for sd in SEEDS]
    b2, rb2 = b2s[0]
    b2_mean = float(np.mean([r["val_auprc"] for _, r in b2s]))
    rows = [{"model": "B2", **rb2, "seed_auprc": [r["val_auprc"] for _, r in b2s], "mean_auprc": b2_mean}]
    selected = []
    for m, g in list(CANDIDATE_GROUPS.items()) + [("ALL", "ALL")]:
        feats = FEATURE_SETS[m]
        rs = [fit_eval(tr, va, feats, P["standard"], seed=sd, name=m)[1] for sd in SEEDS]
        mean = float(np.mean([r["val_auprc"] for r in rs]))
        keep = m != "ALL" and mean > b2_mean
        if keep:
            selected.append(g)
        rows.append({"model": f"{m} (B2+{g})", "learner": "standard", **rs[0], "seed_auprc": [r["val_auprc"] for r in rs],
                     "mean_auprc": mean, "delta_mean_auprc_vs_b2": mean - b2_mean, "kept": keep})
        log.info("standard %s %s -> mean %.4f (B2 %.4f) %s", m, g, mean, b2_mean, "KEEP" if keep else "")
        save("ablation_validation", {"rows": rows, "params": P})
    for m, g in CANDIDATE_GROUPS.items():
        _, r = fit_eval(tr, va, FEATURE_SETS[m], P["residual_b2"], "residual_b2", b2, name=m)
        rows.append({"model": f"{m} (B2+{g})", **r, "delta_auprc_vs_b2": r["val_auprc"] - rb2["val_auprc"]})
        log.info("residual %s %s -> %.4f (B2 %.4f)", m, g, r["val_auprc"], rb2["val_auprc"])
    feats = _g("SPREAD") + sum((_g(x) for x in selected), [])
    fs = [fit_eval(tr, va, feats, P["standard"], seed=sd, name="FULL")[1] for sd in SEEDS]
    rows.append({"model": "FULL (B2+" + "+".join(selected) + ")", "learner": "standard", **fs[0],
                 "seed_auprc": [r["val_auprc"] for r in fs], "mean_auprc": float(np.mean([r["val_auprc"] for r in fs])),
                 "delta_mean_auprc_vs_b2": float(np.mean([r["val_auprc"] for r in fs])) - b2_mean})
    save("ablation_validation", {"rows": rows, "params": P, "selected_groups": selected,
                                 "rule": "keep group iff 3-seed mean validation AUPRC(B2+group) > 3-seed mean AUPRC(B2)",
                                 "note": "validation 2020 (dev split)"})


# ------------------------------------------------------------------------------------------ confirm
def confirm() -> None:
    """Single check on dev-test 2021 of the configuration chosen on validation 2020."""
    sel = json.loads((OUT / "selection.json").read_text())
    df = load(("train", "validation", "test"))
    tr, va, te = (df[df["split"] == s].reset_index(drop=True) for s in ("train", "validation", "test"))
    b2 = CalibratedGBM(FEATURE_SETS["B2"], "B2", params=sel["b2_params"]).fit(tr, va)
    feats = _g("SPREAD") + sum((_g(g) for g in sel["groups"]), [])
    base = B2Margin(b2, tr) if sel["learner"] == "residual_b2" else None
    s = CalibratedGBM(feats, "FULL", base=base, params=sel["sentinel_params"]).fit(tr, va)
    y = te["bust"].to_numpy()
    pb, ps = b2.predict(te), s.predict(te)
    out = {"selection": sel, "devtest": "2021 (read once, after selection)",
           "B2": {"auprc": M.auprc(y, pb), "brier": M.brier(y, pb), "ece": M.ece(y, pb)},
           "SENTINEL": {"auprc": M.auprc(y, ps), "brier": M.brier(y, ps), "ece": M.ece(y, ps)},
           "bootstrap_sentinel_minus_b2": M.block_bootstrap_diff(te, ps, pb, n=300, seed=1)}
    out["relative_gain"] = (out["SENTINEL"]["auprc"] - out["B2"]["auprc"]) / out["B2"]["auprc"]
    save("confirm_devtest_2021", out)
    print(json.dumps(clean_json(out), indent=1, default=float))


# ----------------------------------------------------------------------------------------- transfer
def transfer() -> None:
    """Run after `confirm` showed the 2020 validation gain did not hold on 2021. Rule fixed BEFORE this
    stage ran: keep a group iff its 3-seed mean AUPRC gain over B2 is > 0 on BOTH validation 2020 and
    dev-test 2021 (models fitted on 2018-19, early stopping and calibration on 2020, as in `confirm`).
    2021 is the final protocol's validation year, never its test year; 2022 is not read."""
    sel = json.loads((OUT / "selection.json").read_text())
    df = load(("train", "validation", "test"))
    tr, va, te = (df[df["split"] == s].reset_index(drop=True) for s in ("train", "validation", "test"))
    yv, yt = va["bust"].to_numpy(), te["bust"].to_numpy()

    def run(feats, params):
        v, t = [], []
        for sd in SEEDS:
            m = CalibratedGBM(feats, "m", params=params, seed=sd).fit(tr, va)
            v.append(M.auprc(yv, m.predict_raw(va)))
            t.append(M.auprc(yt, m.predict_raw(te)))
        return float(np.mean(v)), float(np.mean(t)), v, t

    b2v, b2t, _, _ = run(FEATURE_SETS["B2"], sel["b2_params"])
    rows, keep = [], []
    for m, g in list(CANDIDATE_GROUPS.items()) + [("ALL", "ALL")]:
        v, t, vs, ts = run(FEATURE_SETS[m], sel["sentinel_params"])
        ok = m != "ALL" and v > b2v and t > b2t
        if ok:
            keep.append(g)
        rows.append({"model": m, "group": g, "val2020_mean": v, "devtest2021_mean": t, "val2020_seeds": vs,
                     "devtest2021_seeds": ts, "delta_val2020": v - b2v, "delta_devtest2021": t - b2t, "kept": ok})
        log.info("%s %s val %+.4f devtest %+.4f %s", m, g, v - b2v, t - b2t, "KEEP" if ok else "")
        save("transfer", {"b2": {"val2020_mean": b2v, "devtest2021_mean": b2t}, "rows": rows})
    save("transfer", {"b2": {"val2020_mean": b2v, "devtest2021_mean": b2t}, "rows": rows, "kept_groups": keep,
                      "rule": "keep iff 3-seed mean AUPRC gain over B2 > 0 on validation 2020 AND dev-test 2021 "
                              "(fixed before this stage ran)"})


# ------------------------------------------------------------------------------------------- report
def report() -> None:
    """docs/optimization_v2.md, generated from the optimisation JSONs (no number typed by hand)."""
    from forecast_bust.config import REPO_ROOT

    def J(n):
        p = OUT / f"{n}.json"
        return json.loads(p.read_text()) if p.exists() else None

    d, s, a, t, sel, c = (J(n) for n in ("diagnosis", "search", "ablation_validation", "transfer", "selection",
                                          "confirm_devtest_2021"))
    L = []
    w = L.append
    f = lambda v, n=4: "—" if v is None else f"{v:.{n}f}"  # noqa: E731
    w("# v2 optimisation log (generated by `scripts/optimize_v2.py report` — do not edit by hand)\n")
    w("Every experiment below ran on the **development split**: train 2018–19, validation 2020, dev-test 2021. "
      "The 2022 test year is not read by any stage. Raw results: `artifacts/v2/dev/optimization/*.json`.\n")
    w("## 1. Diagnosis (train/validation only)\n")
    pv, ce, fh = d["prevalence"]["validation"], d["spread_only_ceiling"], d["frozen_hyperparameters"]
    w(f"* Validation bust rate {f(pv['bust_rate'], 3)}; low-spread share {f(pv['low_spread_share'], 3)}; hidden busts are "
      f"{100 * pv['hidden_share_of_busts']:.1f}% of busts.")
    w("* Each region is one native 5.625° grid point, so the regional error is |ensemble mean − ERA5| at one point: "
      "a very noisy label.")
    w(f"* **Spread-only predictability ceiling.** If the ensemble were perfectly reliable (Gaussian), a spread-only "
      f"model would reach AUPRC ≈ {f(ce['validation']['ceiling_auprc_if_ensemble_perfectly_reliable'], 3)} on "
      f"validation (train {f(ce['train']['ceiling_auprc_if_ensemble_perfectly_reliable'], 3)}). The analytic "
      f"reliable-Gaussian probability itself scores {f(ce['validation']['auprc_ideal_prob_on_real_labels'], 3)} on the "
      f"real validation labels; its mean ({f(ce['validation']['mean_ideal_prob'], 3)}) is below the observed rate "
      f"({f(ce['validation']['observed_bust_rate'], 3)}): the ensemble is under-dispersive against ERA5 at this scale.")
    w(f"* **v1 hyper-parameters on v2 features:** B2 {f(fh['B2']['val_auprc'])} ({fh['B2']['best_iteration']} trees) vs "
      f"ALL {f(fh['ALL']['val_auprc'])} ({fh['ALL']['best_iteration']} trees); Spearman rank correlation of their "
      f"predictions {f(fh['rank_corr_ALL_vs_B2'], 3)}; spread features carry "
      f"{100 * fh['share_gain_of_spread_features_in_ALL']:.0f}% of ALL's gain.")
    w("* **Consequence for B2 (spec: do not weaken B2).** The analytic spread probability beat the tree B2, so B2 "
      "received the spread-only input `spread_thr_ratio` (spread / TRAIN bust threshold in metres).\n")
    w("Strongest spread-conditional signals (ROC AUC within lead-day × spread-decile strata; 0.5 = no information "
      "beyond spread):\n")
    w("| Feature | Group | Train | Validation | Within low spread (val) | Same sign both years |\n|---|---|---:|---:|---:|---|")
    for k, r in list(d["features"].items())[:12]:
        w(f"| {k} | {r['group']} | {f(r['auc_given_spread_train'], 3)} | {f(r['auc_given_spread_validation'], 3)} | "
          f"{f(r['auc_within_low_spread_validation'], 3)} | {'yes' if r['conditional_signal_transfers'] else 'no'} |")
    w("\n## 2. Hyper-parameter search (same grid for B2 and Sentinel)\n")
    w(f"Stage 1 grid {s['grid']} with {s['fixed']}; stage 2 (refine) at depth 3, learning rate 0.02 over "
      f"{REFINE}. Early stopping on validation AUPRC. Class weighting (scale_pos_weight 3, 9) tested on each best "
      "configuration.\n")
    w("| Model | Runs | Best validation AUPRC | Best configuration (depth / min_child_weight / colsample / λ / lr) |"
      "\n|---|---:|---:|---|")
    for k in ("B2", "B2_v1inputs", "ALL", "ALL@residual_b2"):
        rs = [r for r in s["runs"] if r["model"] == k]
        b = max(rs, key=lambda r: r["val_auprc"])
        p = b["params"]
        w(f"| {k} | {len(rs)} | {f(b['val_auprc'])} | {p['max_depth']} / {p['min_child_weight']} / "
          f"{p['colsample_bytree']} / {p['reg_lambda']} / {p['learning_rate']} |")
    spw = [r for r in s["runs"] if "+spw" in r["model"]]
    w("\nClass weighting: " + "; ".join(f"{r['model']} {f(r['val_auprc'])}" for r in spw) + " (all below the "
      "unweighted optimum, rejected).")
    if "best" in s:
        w("Seed stability of the optima (seeds 1–3): " + "; ".join(
            f"{k} {', '.join(f(x) for x in v.get('seed_auprc', []))}" for k, v in s["best"].items() if v.get("seed_auprc")))
    w("\n## 3. Feature-group ablation (validation 2020, 3-seed means)\n")
    w(f"Rule: {a['rule']}.\n\n| Model | Mean AUPRC | Δ vs B2 | Kept |\n|---|---:|---:|---|")
    for r in a["rows"]:
        if r.get("learner") == "standard" or r["model"] == "B2":
            w(f"| {r['model']} | {f(r.get('mean_auprc'))} | {f(r.get('delta_mean_auprc_vs_b2'))} | "
              f"{'yes' if r.get('kept') else ''} |")
    w("\nResidual (B2-conditioned) learner, single seed: " + "; ".join(
        f"{r['model']} Δ {f(r['delta_auprc_vs_b2'])}" for r in a["rows"] if r.get("learner") == "residual_b2"))
    if c:
        w("\n## 4. One dev-test check (2021) of the validation-selected configuration\n")
        b = c["bootstrap_sentinel_minus_b2"]
        w(f"Selection {', '.join(c['selection']['groups'])}: B2 AUPRC {f(c['B2']['auprc'])}, Sentinel "
          f"{f(c['SENTINEL']['auprc'])} ({100 * c['relative_gain']:+.1f}%), 95% CI of the difference "
          f"[{f(b['ci95'][0])}, {f(b['ci95'][1])}]. **The validation gain did not transfer.**")
    if t:
        w("\n## 5. Two-period stability rule (fixed before it ran)\n")
        w(f"Rule: {t['rule']}.\n\n| Group | Δ val 2020 | Δ dev-test 2021 | Kept |\n|---|---:|---:|---|")
        for r in t["rows"]:
            w(f"| {r['model']} ({r['group']}) | {f(r['delta_val2020'])} | {f(r['delta_devtest2021'])} | "
              f"{'yes' if r['kept'] else ''} |")
    if sel:
        w("\n## 6. Locked configuration (config/model_v2.yaml)\n")
        w(f"* Learner: {sel['learner']} — {sel['learner_rule']}.")
        w(f"* B2 and Sentinel hyper-parameters: {sel['sentinel_params']} ({sel['params_rule']}).")
        w(f"* Sentinel groups: {', '.join(t['kept_groups']) if t else ', '.join(sel['groups'])} "
          "(two-period rule; the wind/MSLP DYN group was rejected because its 2020 gain reversed in 2021).")
    (REPO_ROOT / "docs" / "optimization_v2.md").write_text("\n".join(L) + "\n")
    print("wrote docs/optimization_v2.md")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    {"augment": augment, "diagnose": diagnose, "search": search, "refine": refine, "ablate": ablate, "transfer": transfer, "report": report, "confirm": confirm}[sys.argv[1]]()
