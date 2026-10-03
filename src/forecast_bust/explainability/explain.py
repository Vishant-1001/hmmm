"""Deterministic, number-backed explanations. No LLM, no free-text causes.

Every sentence is generated from a template filled with values the system actually
computed. Attributions are TreeSHAP contributions of the (pre-calibration) XGBoost
model, summed per feature group; they describe what the model used, not physical cause.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecast_bust.features.build import GROUPS

GROUP_NAMES = {"SPREAD": "Ensemble spread (baseline information)", "ATM": "Atmospheric state",
               "ENS": "Ensemble behaviour", "REC": "Recent verified error behaviour", "PAT": "Large-scale pattern", "EVO": "Forecast evolution",
               "MEM": "Historical forecast-state memory", "DYN": "Wind / vorticity / MSLP state"}
FEATURE_LABELS = {
    "spread_m": "Z500 ensemble spread (m)", "spread_pct": "spread percentile vs training (same region/lead/season)",
    "lead_day": "lead day", "init_hour": "initialisation hour (UTC)", "region_code": "region", "season_code": "season",
    "anom500": "ens-mean Z500 anomaly (m)", "abs_anom500": "|Z500 anomaly| (m)", "anom700": "Z700 anomaly (m)",
    "anom850": "Z850 anomaly (m)", "thick_anom": "500-850 hPa thickness anomaly (m)",
    "grad_x": "zonal Z500 gradient (m/100 km)", "grad_y": "meridional Z500 gradient (m/100 km)",
    "grad_mag": "Z500 gradient magnitude (m/100 km)", "lap500": "Z500 curvature (Laplacian, m)",
    "nbhd_anom500": "neighbourhood Z500 anomaly (m)", "m_iqr": "member IQR (m)", "m_p10p90": "member P10-P90 range (m)",
    "m_skew": "member skewness", "m_sign_agree": "member anomaly-sign agreement (fraction)",
    "spread_nbhd": "neighbourhood spread (m)", "spread_hetero": "spread heterogeneity (CV)",
    "spread700": "Z700 spread (m)", "spread850": "Z850 spread (m)", "spread_growth": "spread relative to Day 1",
    "domain_spread": "domain-mean spread (m)", "spread_rel_domain": "spread relative to domain mean",
    "pc_norm": "pattern distance from training mean (PC space)", "pc_resid": "unexplained pattern variance fraction",
    "rev_region": "cycle-to-cycle revision, region (m)", "rev_nbhd": "cycle-to-cycle revision, neighbourhood RMS (m)",
    "rev_pc": "cycle-to-cycle pattern revision (PC space)", "rev_spread": "cycle-to-cycle spread change (m)",
    "rev_available": "previous cycle available",
    "an_n_within": "analogues within similarity radius", "an_dist1": "nearest analogue distance",
    "an_dist_mean": "mean analogue distance", "an_bust_rate": "analogue bust rate",
    "an_err_med": "analogue median normalized error", "an_err_q90": "analogue 90th pct normalized error",
    "an_n_eligible": "verified historical cases available",
    "rec_err": "recent verified normalized error, this region (5 d)", "rec_bias": "recent verified mean error, this region (m)",
    "rec_bust_rate": "recent verified bust fraction, this region", "rec_n": "recent verified cases (5 d)",
    "rec_err_nbhd": "recent verified normalized error, neighbours", "rec_err_domain": "recent verified normalized error, domain",
}
FEATURE_LABELS.update({
    "u500_anom": "500 hPa zonal wind anomaly (m/s)", "v500_anom": "500 hPa meridional wind anomaly (m/s)",
    "ws500": "500 hPa wind speed (m/s)", "ws850": "850 hPa wind speed (m/s)",
    "shear_500_850": "500-850 hPa vector wind shear (m/s)", "vort500": "500 hPa relative vorticity (1e-5/s)",
    "vort850": "850 hPa relative vorticity (1e-5/s)", "div500": "500 hPa divergence (1e-5/s)",
    "div850": "850 hPa divergence (1e-5/s)", "mslp_anom": "MSLP anomaly (hPa)",
    "mslp_grad": "MSLP gradient (hPa/100 km)", "wspread500": "500 hPa vector-wind spread (m/s)",
    "wspread850": "850 hPa vector-wind spread (m/s)", "mslp_spread": "MSLP ensemble spread (hPa)",
    "spread_thr_ratio": "spread relative to the bust threshold (spread / TRAIN Q90 error, m/m)",
    "z500_tend": "Z500 change per day in this forecast (m/day)", "mslp_tend": "MSLP change per day in this forecast (hPa/day)",
    "vort500_tend": "500 hPa vorticity change per day in this forecast (1e-5/s/day)",
})
for i in range(1, 21):
    FEATURE_LABELS[f"pc{i}"] = f"pattern PC{i} coordinate"


def feature_group(f: str) -> str:
    for g, fs in GROUPS.items():
        if f in fs:
            return g
    return "OTHER"


def group_attributions(contrib: np.ndarray, features: list[str]) -> dict:
    """contrib: (n, n_features + 1) log-odds contributions."""
    out = {}
    for g in GROUPS:
        idx = [i for i, f in enumerate(features) if f in GROUPS[g]]
        if idx:
            out[g] = contrib[:, idx].sum(1)
    return out


def train_percentile(value: float, ref_sorted: np.ndarray) -> float | None:
    if ref_sorted is None or len(ref_sorted) == 0 or not np.isfinite(value):
        return None
    return float(np.searchsorted(ref_sorted, value, side="right") / len(ref_sorted) * 100)


def _state_evidence(row: pd.Series, ref: dict, base_rate: float) -> list[dict]:
    """Evidence items A-D: forecast-state values and causal historical memory (shared by v2/v3)."""
    ev = []
    sp = row.get("spread_pct")
    if sp is not None and np.isfinite(sp):
        ev.append({"kind": "A. Ensemble evidence",
                   "text": f"Z500 ensemble spread {row['spread_m']:.1f} m = {100 * sp:.0f}th percentile of training "
                           f"spread for this region, lead day and season; member anomaly-sign agreement "
                           f"{100 * row.get('m_sign_agree', np.nan):.0f}%."})
    if np.isfinite(row.get("anom500", np.nan)):
        pa = train_percentile(abs(row["anom500"]), ref.get("abs_anom500"))
        ev.append({"kind": "B. Atmospheric-state evidence",
                   "text": f"Forecast Z500 anomaly {row['anom500']:+.0f} m (|anomaly| at the "
                           f"{pa:.0f}th training percentile); pattern distance from the training mean "
                           f"{row.get('pc_norm', np.nan):.2f} (PC units)."})
    n_an = row.get("an_n_eligible", np.nan)
    if np.isfinite(row.get("an_bust_rate", np.nan)):
        ev.append({"kind": "C. Historical evidence",
                   "text": f"Of the 30 most similar verified historical forecast states for this region and lead "
                           f"day ({int(n_an)} verified cases available; {int(row.get('an_n_within', 0))} within the "
                           f"similarity radius), {100 * row['an_bust_rate']:.0f}% exceeded the bust threshold "
                           f"(climatological rate {100 * base_rate:.0f}%)."})
    else:
        ev.append({"kind": "C. Historical evidence",
                   "text": "Historical support unavailable for this forecast state (too few verified analogues)."})
    if np.isfinite(row.get("rev_region", np.nan)):
        pr = train_percentile(row["rev_nbhd"], ref.get("rev_nbhd"))
        ev.append({"kind": "D. Forecast-evolution evidence",
                   "text": f"Versus the cycle 24 h earlier (same valid time) the regional Z500 forecast moved "
                           f"{row['rev_region']:.1f} m (neighbourhood RMS revision {row['rev_nbhd']:.1f} m, "
                           f"{pr:.0f}th training percentile)."})
    else:
        ev.append({"kind": "D. Forecast-evolution evidence",
                   "text": "Previous-cycle forecast for the same valid time not available in the downloaded archive."})
    if np.isfinite(row.get("rec_err", np.nan)):
        ev.append({"kind": "C2. Recent verified error behaviour",
                   "text": f"Over the 5 days before initialisation, verified Day 1-3 forecasts for this region had mean "
                           f"normalized error {row['rec_err']:.2f} (mean error {row['rec_bias']:+.1f} m; "
                           f"{100 * row['rec_bust_rate']:.0f}% busts, n = {int(row['rec_n'])}); neighbouring regions "
                           f"{row['rec_err_nbhd']:.2f}, whole domain {row['rec_err_domain']:.2f}."})
    return ev


def explain_row(row: pd.Series, contrib_row: np.ndarray, features: list[str], ref: dict,
                p_full: float, p_b2: float, base_rate: float, top: int = 6,
                baseline_logodds: float | None = None) -> dict:
    """Structured explanation for one (init, region, lead) case.

    `baseline_logodds`: for the residual learner, the B2 spread-baseline margin the trees start
    from; the feature contributions then describe only the adjustment beyond B2."""
    c = contrib_row[:-1]
    order = np.argsort(-np.abs(c))[:top]
    drivers = []
    for i in order:
        f = features[i]
        v = row.get(f, np.nan)
        v = float(v) if v is not None and np.isfinite(v) else None
        drivers.append({"feature": f, "label": FEATURE_LABELS.get(f, f), "group": feature_group(f),
                        "value": v, "train_percentile": train_percentile(v, ref.get(f)) if v is not None else None,
                        "contribution_logodds": float(c[i]),
                        "direction": "raises risk" if c[i] > 0 else "lowers risk"})
    groups = {}
    for g, fs in GROUPS.items():
        idx = [i for i, f in enumerate(features) if f in fs]
        if idx:
            groups[g] = {"name": GROUP_NAMES[g], "contribution_logodds": float(c[idx].sum())}
    if baseline_logodds is not None:
        groups = {"BASE": {"name": "Spread baseline B2 (starting log-odds)",
                           "contribution_logodds": float(baseline_logodds)}, **groups}
    ev = _state_evidence(row, ref, base_rate)
    ev.append({"kind": "E. Baseline disagreement",
               "text": f"Sentinel {100 * p_full:.0f}% vs calibrated spread-only baseline {100 * p_b2:.0f}% "
                       f"({100 * (p_full - p_b2):+.0f} percentage points)."})
    note = ("TreeSHAP contributions (log-odds, before isotonic calibration). They describe what the model used; "
            "they are associations, not physical causes.")
    if baseline_logodds is not None:
        note = ("Sentinel starts from the B2 spread-baseline log-odds and adds trees for information beyond spread; "
                "the listed TreeSHAP contributions (log-odds, before isotonic calibration) are that adjustment only. "
                "They describe what the model used; they are associations, not physical causes.")
    return {"drivers": drivers, "groups": groups, "evidence": ev, "attribution_note": note}


def explain_row_v3(row: pd.Series, q: dict, threshold: float, p_raw: float, p_cal: float, importance: dict,
                   features: list[str], ref: dict, base_rate: float, top: int = 8) -> dict:
    """v3 explanation without per-row attribution (HistGradientBoosting has no exact per-row
    attribution here and none is fabricated). Three separated parts:
      MODEL OUTPUT  - this row's predicted error quantiles, threshold, exceedance and calibrated probability;
      INPUT EVIDENCE - this row's forecast-state values for the model's most important features (model-level
                       permutation importance) with training percentiles, plus evidence items A-D;
      INTERPRETATION - plain-language statements derived only from the numbers above."""
    drivers = []
    for f in [f for f in importance if f in features][:top]:
        v = row.get(f, np.nan)
        v = float(v) if v is not None and np.isfinite(v) else None
        drivers.append({"feature": f, "label": FEATURE_LABELS.get(f, f), "group": feature_group(f), "value": v,
                        "train_percentile": train_percentile(v, ref.get(f)) if v is not None else None,
                        "model_importance": float(importance[f])})
    groups = {}
    for g, fs in GROUPS.items():
        vals = [importance[f] for f in features if f in fs and f in importance]
        if vals:
            groups[g] = float(sum(vals))
    ev = _state_evidence(row, ref, base_rate)
    ev.append({"kind": "E. Predicted error distribution",
               "text": f"Central predicted error (q50) {q['q50']:.2f}x normal; central predicted range (q25-q75) "
                       f"{q['q25']:.2f}-{q['q75']:.2f}x; upper-tail error (q95) {q['q95']:.2f}x; project bust "
                       f"threshold {threshold:.2f}x. Estimated exceedance probability {100 * p_raw:.0f}% -> "
                       f"calibrated bust probability {100 * p_cal:.0f}%."})
    why = []
    if q["q95"] > threshold:
        why.append(f"The predicted upper-tail error ({q['q95']:.2f}x) is above the project bust threshold "
                   f"({threshold:.2f}x).")
    else:
        why.append(f"The predicted upper-tail error ({q['q95']:.2f}x) stays below the project bust threshold "
                   f"({threshold:.2f}x).")
    if q["q50"] > threshold:
        why.append("Even the central predicted error exceeds the threshold.")
    rate = row.get("an_bust_rate", np.nan)
    if rate is not None and np.isfinite(rate):
        rel = "above" if rate > base_rate else "at or below"
        why.append(f"Similar verified historical forecast states busted {100 * rate:.0f}% of the time, {rel} the "
                   f"climatological {100 * base_rate:.0f}%.")
    hi = [d for d in drivers if d["train_percentile"] is not None and (d["train_percentile"] >= 90
                                                                       or d["train_percentile"] <= 10)]
    if hi:
        why.append("Unusual values among the model's most important inputs (associated with the prediction, not "
                   "shown to cause it): " + ", ".join(f"{d['label']} ({d['train_percentile']:.0f}th pct)"
                                                     for d in hi[:3]) + ".")
    return {"drivers": drivers, "groups": groups, "evidence": ev, "interpretation": why,
            "attribution_note": "Feature ranking is MODEL-LEVEL permutation importance of the q90 estimator "
                                "(validation), the same for every row; values and percentiles are this row's "
                                "inputs. Features are associated with the prediction; they are not causes."}


def criterion_level(rate: float | None, clim: float, n: float | None, kmin: int) -> str:
    """Analogue-based evidence level for one criterion (not a model output): rate among the nearest VERIFIED
    historical analogues relative to its training climatological rate."""
    if rate is None or not np.isfinite(rate) or n is None or not np.isfinite(n) or n < kmin:
        return "INSUFFICIENT"
    return "HIGH" if rate >= 2 * clim else "ELEVATED" if rate > 1.25 * clim else "NORMAL"


def explain_row_v4(row: pd.Series, contrib_row: np.ndarray, features: list[str], ref: dict, p: float,
                   p_raw: float, base_rate: float, crit: dict, top: int = 6) -> dict:
    """V4: TreeSHAP attribution of the pattern-aware XGBoost model (log-odds, before calibration), evidence
    items A-D, and the magnitude / pattern / historical-support evidence from the causal analogue memory.
    `crit` = {magnitude|pattern|support: {"level", "rate", "climatology"}}. Associations, not causes."""
    c = contrib_row[:-1]
    drivers = []
    for i in np.argsort(-np.abs(c))[:top]:
        f = features[i]
        v = row.get(f, np.nan)
        v = float(v) if v is not None and np.isfinite(v) else None
        drivers.append({"feature": f, "label": FEATURE_LABELS.get(f, f), "group": feature_group(f), "value": v,
                        "train_percentile": train_percentile(v, ref.get(f)) if v is not None else None,
                        "contribution_logodds": float(c[i]), "direction": "raises risk" if c[i] > 0 else "lowers risk"})
    groups = {g: float(c[[i for i, f in enumerate(features) if f in fs]].sum())
              for g, fs in GROUPS.items() if any(f in fs for f in features)}
    ev = _state_evidence(row, ref, base_rate)
    names = {"magnitude": "large-error (normalized RMSE > training Q90)",
             "pattern": "pattern-failure (local anomaly correlation < training Q10)",
             "support": "pattern-aware bust (both criteria)"}
    for k in ("magnitude", "pattern", "support"):
        x = crit[k]
        if x["rate"] is not None:
            ev.append({"kind": f"E. Historical {k} evidence",
                       "text": f"Among the most similar verified historical forecast states, {100 * x['rate']:.0f}% had a "
                               f"{names[k]} outcome (training rate {100 * x['climatology']:.1f}%): {x['level']}."})
    why = [f"Calibrated pattern-aware bust probability {100 * p:.1f}% (raw model {100 * p_raw:.1f}%).",
           f"Large-error criterion (historical analogues): {crit['magnitude']['level']}.",
           f"Pattern-failure criterion (historical analogues): {crit['pattern']['level']}.",
           f"Historical support for a pattern-aware bust: {crit['support']['level']}."]
    up = [d for d in drivers if d["contribution_logodds"] > 0][:3]
    if up:
        why.append("Model inputs associated with higher risk for this case (TreeSHAP, not causes): "
                   + ", ".join(d["label"] for d in up) + ".")
    return {"drivers": drivers, "groups": groups, "evidence": ev, "interpretation": why,
            "attribution_note": "TreeSHAP contributions of the V4 XGBoost model (log-odds, before isotonic calibration). "
                                "They describe what the model used; they are associations, not physical causes. "
                                "The criterion levels come from verified historical analogues, not from the model."}


def explain_row_b2(row: pd.Series, contrib_row: np.ndarray, features: list[str], ref: dict, p: float, p_raw: float,
                   p_b0: float, base_rate: float, threshold: float, analogue_level: str, top: int = 7) -> dict:
    """Served B2: TreeSHAP of the 7 B2 inputs (log-odds, before calibration) plus forecast-time evidence (spread,
    atmospheric state, causal verified analogues, forecast evolution, recent errors verified before initialisation).
    No verification of this region-day is read. Associations, not causes."""
    c = contrib_row[:-1]
    drivers = []
    for i in np.argsort(-np.abs(c))[:top]:
        f = features[i]
        v = row.get(f, np.nan)
        v = float(v) if v is not None and np.isfinite(v) else None
        drivers.append({"feature": f, "label": FEATURE_LABELS.get(f, f), "group": feature_group(f), "value": v,
                        "train_percentile": train_percentile(v, ref.get(f)) if v is not None else None,
                        "contribution_logodds": float(c[i]), "direction": "raises risk" if c[i] > 0 else "lowers risk"})
    ev = _state_evidence(row, ref, base_rate)
    ev.append({"kind": "E. Baseline comparison",
               "text": f"B2 bust probability {100 * p:.1f}% vs climatological rate B0 {100 * p_b0:.1f}% for this region, "
                       f"lead day and season ({100 * (p - p_b0):+.1f} percentage points)."})
    why = [f"Bust probability {100 * p:.1f}% (raw model {100 * p_raw:.1f}%): the estimated chance that this region-day's "
           f"Z500 forecast error exceeds the project bust threshold (training Q90).",
           ("At or above" if p >= threshold else "Below") + f" the alert threshold {100 * threshold:.1f}% "
           "(10% false-alarm rate on validation)."]
    sp = row.get("spread_pct")
    if sp is not None and np.isfinite(sp):
        why.append(f"Ensemble spread is at the {100 * sp:.0f}th percentile of training spread for this region, lead day "
                   "and season; B2 uses spread, its percentile and its ratio to the bust threshold, plus lead/region/season.")
    why.append(f"Historical analogue evidence (verified similar forecast states, not a model input): {analogue_level}.")
    up = [d for d in drivers if d["contribution_logodds"] > 0][:3]
    if up:
        why.append("Inputs pushing the risk up for this case (TreeSHAP, associations not causes): "
                   + ", ".join(d["label"] for d in up) + ".")
    return {"drivers": drivers, "evidence": ev, "interpretation": why,
            "attribution_note": "TreeSHAP contributions of the B2 booster (log-odds, before isotonic calibration). B2 sees "
                                "only spread, spread percentile, spread/threshold ratio, lead day, region, season and init "
                                "hour; the other evidence items are context, not model inputs."}
