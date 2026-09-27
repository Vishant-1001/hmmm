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
               "MEM": "Historical forecast-state memory"}
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


def explain_row(row: pd.Series, contrib_row: np.ndarray, features: list[str], ref: dict,
                p_full: float, p_b2: float, base_rate: float, top: int = 6) -> dict:
    """Structured explanation for one (init, region, lead) case."""
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
    ev.append({"kind": "E. Baseline disagreement",
               "text": f"Sentinel {100 * p_full:.0f}% vs calibrated spread-only baseline {100 * p_b2:.0f}% "
                       f"({100 * (p_full - p_b2):+.0f} percentage points)."})
    return {"drivers": drivers, "groups": groups, "evidence": ev,
            "attribution_note": "TreeSHAP contributions (log-odds, before isotonic calibration). They describe "
                                "what the model used; they are associations, not physical causes."}
