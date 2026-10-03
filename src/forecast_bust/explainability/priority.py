"""Transparent forecaster review-priority score (product rule, not a scientific quantity).

    score = p_bust * urgency(lead_day) * evidence_weight(evidence)
            + disagreement_bonus * max(0, p_bust - p_spread_baseline)

    urgency(d) = 0.5 ** ((d - 1) / urgency_halflife_days)

Parameters live in config/model.yaml (priority section).
"""
from __future__ import annotations

import numpy as np

from forecast_bust.config import model_config

EVIDENCE_NAMES = ["STRONG", "MODERATE", "WEAK", "INSUFFICIENT HISTORICAL SUPPORT"]


def priority_score(p: np.ndarray, p_b2: np.ndarray, lead_day: np.ndarray, evidence_level: np.ndarray) -> np.ndarray:
    cfg = model_config()["priority"]
    urg = 0.5 ** ((np.asarray(lead_day) - 1) / cfg["urgency_halflife_days"])
    w = np.array([cfg["evidence_weight"][n] for n in EVIDENCE_NAMES])[np.asarray(evidence_level, dtype=int)]
    return np.asarray(p) * urg * w + cfg["disagreement_bonus"] * np.maximum(0, np.asarray(p) - np.asarray(p_b2))


def formula() -> dict:
    cfg = model_config()["priority"]
    return {"formula": "score = p_bust * 0.5**((lead_day-1)/H) * evidence_weight + B * max(0, p_bust - p_B2)",
            "H_days": cfg["urgency_halflife_days"], "B": cfg["disagreement_bonus"],
            "evidence_weight": cfg["evidence_weight"],
            "note": "Product triage rule; weights are design choices, not fitted or validated constants."}


def priority_score_v3(p: np.ndarray, upper_tail: np.ndarray, threshold: np.ndarray, lead_day: np.ndarray,
                      evidence_level: np.ndarray) -> np.ndarray:
    """v3 (no B2 term): the bonus is the predicted upper tail (q95) above the bust threshold,
    as a fraction of the threshold and capped at 1."""
    cfg = model_config()["priority"]
    urg = 0.5 ** ((np.asarray(lead_day) - 1) / cfg["urgency_halflife_days"])
    w = np.array([cfg["evidence_weight"][n] for n in EVIDENCE_NAMES])[np.asarray(evidence_level, dtype=int)]
    tail = np.clip(np.asarray(upper_tail) / np.asarray(threshold) - 1, 0, 1)
    return np.asarray(p) * urg * w + cfg["disagreement_bonus"] * tail


def formula_v3() -> dict:
    cfg = model_config()["priority"]
    return {"formula": "score = p_bust * 0.5**((lead_day-1)/H) * evidence_weight "
                       "+ B * clip(upper_tail_error / bust_threshold - 1, 0, 1)",
            "H_days": cfg["urgency_halflife_days"], "B": cfg["disagreement_bonus"],
            "evidence_weight": cfg["evidence_weight"],
            "note": "Product triage rule; weights are design choices, not fitted or validated constants."}


def priority_score_v4(p: np.ndarray, lead_day: np.ndarray, evidence_level: np.ndarray) -> np.ndarray:
    """V4: calibrated pattern-aware bust probability x urgency x evidence weight (no baseline/tail bonus)."""
    cfg = model_config()["priority"]
    urg = 0.5 ** ((np.asarray(lead_day) - 1) / cfg["urgency_halflife_days"])
    w = np.array([cfg["evidence_weight"][n] for n in EVIDENCE_NAMES])[np.asarray(evidence_level, dtype=int)]
    return np.asarray(p) * urg * w


def formula_v4() -> dict:
    cfg = model_config()["priority"]
    return {"formula": "score = p_pattern_bust * 0.5**((lead_day-1)/H) * evidence_weight",
            "H_days": cfg["urgency_halflife_days"], "B": 0.0, "evidence_weight": cfg["evidence_weight"],
            "note": "Product triage rule; weights are design choices, not fitted or validated constants."}


def formula_b2() -> dict:
    cfg = model_config()["priority"]
    return {"formula": "score = p_bust(B2) * 0.5**((lead_day-1)/H) * evidence_weight",
            "H_days": cfg["urgency_halflife_days"], "B": 0.0, "evidence_weight": cfg["evidence_weight"],
            "note": "Product triage rule; weights are design choices, not fitted or validated constants."}


# B2 serving uses the same rule (p_bust x urgency x evidence weight, no baseline/tail bonus)
priority_score_b2 = priority_score_v4
