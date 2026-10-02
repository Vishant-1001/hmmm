"""Evaluation metrics. Every function takes arrays of real predictions and labels."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


def auprc(y, p) -> float:
    return float(average_precision_score(y, p)) if np.any(y) else float("nan")


def roc_auc(y, p) -> float:
    return float(roc_auc_score(y, p)) if 0 < np.mean(y) < 1 else float("nan")


def brier(y, p) -> float:
    return float(brier_score_loss(y, p))


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y, dtype=float) - np.asarray(p, dtype=float))))


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, dtype=float) - np.asarray(p, dtype=float)) ** 2)))


def pinball_loss(y, q_value, level: float) -> float:
    """Quantile (pinball) loss of a predicted quantile `q_value` at `level` against the
    realised continuous outcome `y`. 0 for a perfect quantile; smaller is better."""
    d = np.asarray(y, dtype=float) - np.asarray(q_value, dtype=float)
    return float(np.mean(np.where(d >= 0, level * d, (level - 1) * d)))


def quantile_coverage(y, quantiles: dict[str, np.ndarray], levels: dict[str, float]) -> dict:
    """Empirical coverage of each predicted quantile: fraction of realised `y` at or below
    the predicted value, which should be close to its nominal level for a well-behaved
    conditional quantile model. `quantiles`/`levels` keyed by the same column names
    (e.g. "q50" -> 0.50)."""
    y = np.asarray(y, dtype=float)
    out = {}
    for name, level in levels.items():
        qv = np.asarray(quantiles[name], dtype=float)
        out[name] = {"nominal_level": level, "empirical_coverage": float(np.mean(y <= qv)),
                     "pinball_loss": pinball_loss(y, qv, level)}
    return out


def reliability_curve(y, p, n_bins: int = 10) -> dict:
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.sum():
            rows.append({"bin_lo": float(bins[b]), "bin_hi": float(bins[b + 1]), "n": int(m.sum()),
                         "mean_pred": float(np.mean(p[m])), "obs_freq": float(np.mean(y[m]))})
    return {"bins": rows}


def ece(y, p, n_bins: int = 10) -> float:
    rc = reliability_curve(y, p, n_bins)["bins"]
    n = len(y)
    return float(sum(r["n"] / n * abs(r["mean_pred"] - r["obs_freq"]) for r in rc))


def threshold_at_far(y, p, far: float) -> float:
    """Smallest threshold whose false-alarm rate (FP / negatives) is <= far."""
    neg = np.sort(np.asarray(p)[np.asarray(y) == 0])[::-1]
    k = int(np.floor(far * len(neg)))
    if k >= len(neg):
        return float(neg[-1])
    return float(np.nextafter(neg[k], np.inf))


def recall_at_far(y, p, far: float) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr[fpr <= far])) if np.any(fpr <= far) else 0.0


def confusion_at(y, p, thr: float) -> dict:
    y = np.asarray(y).astype(bool)
    a = np.asarray(p) >= thr
    tp, fp = int((a & y).sum()), int((a & ~y).sum())
    fn, tn = int((~a & y).sum()), int((~a & ~y).sum())
    return {"threshold": float(thr), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan"),
            "false_alarm_rate": fp / (fp + tn) if fp + tn else float("nan")}


def hidden_bust_metrics(df: pd.DataFrame, p: np.ndarray, thr: float) -> dict:
    """Recall on hidden busts (bust & low spread) and alert precision within low-spread cases."""
    a = p >= thr
    hb = df["hidden_bust"].to_numpy().astype(bool)
    low = df["low_spread"].to_numpy().astype(bool)
    bust = df["bust"].to_numpy().astype(bool)
    return {
        "n_hidden_busts": int(hb.sum()),
        "hidden_bust_recall": float(a[hb].mean()) if hb.any() else float("nan"),
        "low_spread_alert_precision": float(bust[low & a].mean()) if (low & a).any() else float("nan"),
        "low_spread_auprc": auprc(bust[low], p[low]) if low.any() else float("nan"),
    }


def warning_lead(df: pd.DataFrame, p: np.ndarray, thr: float) -> dict:
    a = p >= thr
    b = df["bust"].to_numpy().astype(bool)
    lead = df["lead_day"].to_numpy()
    by_lead = {int(d): float(a[(lead == d) & b].mean()) for d in np.unique(lead) if ((lead == d) & b).any()}
    return {"mean_lead_day_of_detected_busts": float(lead[a & b].mean()) if (a & b).any() else float("nan"),
            "recall_by_lead_day": by_lead}


def peak_day_error(df: pd.DataFrame, p: np.ndarray) -> dict:
    """|argmax_day predicted risk - argmax_day (normalized error / Q90 threshold)| over
    (init, region) trajectories containing at least one bust."""
    d = df[["init_time", "region_id", "lead_day", "bust", "norm_error", "q_primary"]].copy()
    d["p"] = p
    d["exceed"] = d["norm_error"] / d["q_primary"]
    errs = []
    for _, g in d.groupby(["init_time", "region_id"]):
        if g["bust"].any():
            errs.append(abs(int(g.loc[g["p"].idxmax(), "lead_day"]) - int(g.loc[g["exceed"].idxmax(), "lead_day"])))
    e = np.array(errs)
    return {"n_trajectories": int(len(e)), "mean_abs_peak_day_error": float(e.mean()) if len(e) else float("nan"),
            "within_1_day": float((e <= 1).mean()) if len(e) else float("nan")}


def spatial_overlap(df: pd.DataFrame, p: np.ndarray, thr: float) -> dict:
    """Mean Jaccard overlap between alerted and actual-bust regions per (init, lead) map."""
    d = df[["init_time", "lead_day", "bust"]].copy()
    d["a"] = p >= thr
    d["b"] = d["bust"].astype(bool)
    d["inter"] = d["a"] & d["b"]
    d["union"] = d["a"] | d["b"]
    g = d.groupby(["init_time", "lead_day"])[["inter", "union"]].sum()
    g = g[g["union"] > 0]
    j = g["inter"] / g["union"]
    return {"n_maps": int(len(g)), "mean_jaccard": float(j.mean()) if len(j) else float("nan")}


def block_bootstrap_diff(df: pd.DataFrame, p_a: np.ndarray, p_b: np.ndarray, n: int = 200,
                         seed: int = 0) -> dict:
    """Bootstrap over initialisation dates (blocks) of AUPRC(a) - AUPRC(b)."""
    rng = np.random.default_rng(seed)
    y = df["bust"].to_numpy()
    days = pd.to_datetime(df["init_time"]).dt.floor("D").to_numpy()
    uniq, inv = np.unique(days, return_inverse=True)
    rows_by_day = [np.where(inv == i)[0] for i in range(len(uniq))]
    diffs = []
    for _ in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([rows_by_day[i] for i in pick])
        diffs.append(auprc(y[idx], p_a[idx]) - auprc(y[idx], p_b[idx]))
    diffs = np.array(diffs)
    return {"mean": float(diffs.mean()), "ci95": [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))],
            "p_le_0": float((diffs <= 0).mean()), "n_boot": n, "block": "initialisation day"}


def block_bootstrap_metrics(df: pd.DataFrame, probs: dict[str, np.ndarray], thresholds: dict[str, float],
                            n: int = 300, seed: int = 0) -> dict:
    """95% initialisation-day block-bootstrap CIs for AUPRC, Brier, recall@10% FAR and hidden-bust recall
    at the validation-chosen threshold, per model and for each model minus the first one (the reference).
    Region-day rows of one initialisation day are resampled together (they are not independent)."""
    rng = np.random.default_rng(seed)
    y = df["bust"].to_numpy()
    hb = df["hidden_bust"].to_numpy().astype(bool)
    days = pd.to_datetime(df["init_time"]).dt.floor("D").to_numpy()
    uniq, inv = np.unique(days, return_inverse=True)
    rows_by_day = [np.where(inv == i)[0] for i in range(len(uniq))]
    names = list(probs)
    ref = names[0]

    def stats(idx):
        out = {}
        for k in names:
            p = probs[k][idx]
            yy = y[idx]
            h = hb[idx]
            out[k] = (auprc(yy, p), brier(yy, p), recall_at_far(yy, p, 0.10),
                      float((p[h] >= thresholds[k]).mean()) if h.any() else np.nan)
        return out

    draws = []
    for _ in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        draws.append(stats(np.concatenate([rows_by_day[i] for i in pick])))
    labels = ("auprc", "brier", "recall_at_far_10", "hidden_bust_recall_at_alert")
    point = stats(np.arange(len(y)))

    def ci(vals):
        v = np.asarray(vals, dtype=float)
        v = v[np.isfinite(v)]
        return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] if len(v) else [np.nan, np.nan]

    res = {"n_boot": n, "block": "initialisation day", "reference": ref, "models": {}, "difference_vs_reference": {}}
    for k in names:
        res["models"][k] = {lab: {"point": point[k][j], "ci95": ci([d[k][j] for d in draws])}
                            for j, lab in enumerate(labels)}
        if k != ref:
            res["difference_vs_reference"][k] = {
                lab: {"point": point[k][j] - point[ref][j], "ci95": ci([d[k][j] - d[ref][j] for d in draws]),
                      "p_le_0": float(np.mean([d[k][j] - d[ref][j] <= 0 for d in draws]))}
                for j, lab in enumerate(labels)}
    return res


def pr_curve(y, p, n_points: int = 60) -> dict:
    """Precision-recall curve thinned to about n_points points (for plotting)."""
    from sklearn.metrics import precision_recall_curve
    prec, rec, _ = precision_recall_curve(y, p)
    keep = np.unique(np.linspace(0, len(rec) - 1, n_points).astype(int))
    return {"recall": rec[keep].round(4).tolist(), "precision": prec[keep].round(4).tolist(),
            "base_rate": float(np.mean(y))}


def disagreement_behaviour(y, p_model, p_base, margin_pp: float = 5.0) -> dict:
    """How the model's disagreement with the spread baseline relates to what happened.

    Rows are binned by d = p_model - p_base (percentage points): model higher (> margin),
    agree (|d| <= margin), model lower (< -margin). Within each bin the observed bust rate
    is compared with both mean probabilities and both Brier scores, which shows which
    probability was closer to reality when they disagreed."""
    y = np.asarray(y, dtype=float)
    d = 100 * (np.asarray(p_model) - np.asarray(p_base))
    out = {"margin_pp": margin_pp, "mean_abs_pp": float(np.abs(d).mean()),
           "share_abs_gt_5pp": float((np.abs(d) > 5).mean()), "share_abs_gt_10pp": float((np.abs(d) > 10).mean()),
           "quantiles_pp": {str(q): float(np.quantile(d, q)) for q in (0.01, 0.05, 0.5, 0.95, 0.99)}, "bins": []}
    for name, m in (("model_higher", d > margin_pp), ("agree", np.abs(d) <= margin_pp),
                    ("model_lower", d < -margin_pp)):
        if m.any():
            out["bins"].append({"bin": name, "n": int(m.sum()), "share": float(m.mean()),
                                "observed_bust_rate": float(y[m].mean()),
                                "mean_p_model": float(np.mean(np.asarray(p_model)[m])),
                                "mean_p_base": float(np.mean(np.asarray(p_base)[m])),
                                "brier_model": brier(y[m], np.asarray(p_model)[m]),
                                "brier_base": brier(y[m], np.asarray(p_base)[m])})
        else:
            out["bins"].append({"bin": name, "n": 0, "share": 0.0})
    return out
