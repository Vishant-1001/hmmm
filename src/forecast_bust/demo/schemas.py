"""Explicit input/output contract of the live demo API (pydantic)."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class CaseInfo(BaseModel):
    case_id: str
    init_time: str
    season: str
    selection: str
    demo_role: str
    selection_note: str
    valid_range: list[str]
    n_regions: int
    lead_days: list[int]
    split: str
    model_input: str
    verification: str


class CaseList(BaseModel):
    mode: str
    selection_rule: str
    cases: list[CaseInfo]


class ModelSummary(BaseModel):
    model_version: str
    model_artifact: str
    exported_from: str
    retrained_for_demo: bool
    training_window: str
    calibration_window: str
    calibration: str
    learner: str
    sentinel_features: list[str]
    b2_features: list[str]
    selected_groups: list[str]
    selection_note: str
    pending: str
    alert_threshold: float
    alert_threshold_definition: str
    confidence_definition: str
    target: str


class Cell(BaseModel):
    """Model output for one region x lead day (forecast-time information only)."""
    lead_day: int
    valid_time: str
    bust_probability: float
    reliability_confidence: float
    b2_probability: float
    b0_probability: float
    disagreement_pp: float
    alert: bool
    spread_m: float
    spread_pct: Optional[float]
    support_level: str
    support_distance: Optional[float]
    evidence_quality: str
    analogues_within_radius: Optional[int]
    verified_cases_available: Optional[int]
    analogue_bust_rate: Optional[float]
    priority_score: float


class QueueItem(Cell):
    region_id: str


class RegionOverview(BaseModel):
    region_id: str
    lat: float
    lon: float
    row: int
    col: int
    peak_lead_day: int
    peak_bust_probability: float
    alert_days: list[int]
    days: list[Cell]


class LeadSummary(BaseModel):
    lead_day: int
    mean_bust_probability: float
    max_bust_probability: float
    n_alerts: int
    mean_disagreement_pp: float


class RunResult(BaseModel):
    mode: str
    case: CaseInfo
    model: ModelSummary
    timings_ms: dict[str, float]
    alert_threshold: float
    priority_formula: dict
    regions: list[RegionOverview]
    priority_queue: list[QueueItem]
    lead_summary: list[LeadSummary]
    blind: bool


class RegionDetail(BaseModel):
    case_id: str
    region_id: str
    lat: float
    lon: float
    alert_threshold: float
    trajectory: list[Cell]


class Driver(BaseModel):
    feature: str
    label: str
    group: str
    value: Optional[float]
    train_percentile: Optional[float]
    contribution_logodds: float
    direction: str


class Attribution(BaseModel):
    method: str
    bias_logodds: float
    drivers: list[Driver]
    groups: dict[str, float]
    note: str


class EvidenceItem(BaseModel):
    kind: str
    text: str


class Analogue(BaseModel):
    case_id: str
    init_time: str
    distance: float
    bust: int
    normalized_error: float
    signature: Optional[str]
    split: str


class AnalogueSummary(BaseModel):
    k: int
    radius: float
    rule: str
    verified_cases_available: Optional[int]
    within_radius: Optional[int]
    bust_rate: Optional[float]
    climatological_bust_rate: float
    median_normalized_error: Optional[float]
    q90_normalized_error: Optional[float]
    nearest: list[Analogue]


class FailureSignature(BaseModel):
    basis: str
    n: int
    distribution: Optional[dict[str, float]]
    top: Optional[str]
    top_label: Optional[str] = None


class ContextFeature(BaseModel):
    feature: str
    label: str
    value: Optional[float]
    train_percentile: Optional[float]


class Explanation(Cell):
    case_id: str
    region_id: str
    attribution: Attribution
    evidence: list[EvidenceItem]
    analogue_summary: AnalogueSummary
    failure_signature: FailureSignature
    context_features: list[ContextFeature]
    model_inputs: dict[str, Optional[float]]


class Fingerprint(BaseModel):
    signature: str
    label: str
    phase_share: Optional[float]
    bias_m: Optional[float]
    pattern_corr: Optional[float]


class VerificationOut(BaseModel):
    actual_bust: bool
    normalized_error: float
    threshold_q90: float
    error_m: float
    hidden_bust: bool
    failure_fingerprint: Fingerprint


class RevealForecast(BaseModel):
    bust_probability: float
    b2_probability: float
    alert: bool
    expected_signature: FailureSignature


class Comparison(BaseModel):
    expected_top: Optional[str]
    p_expected_for_actual: Optional[float]
    match: Optional[bool]
    note: str


class VerifDay(BaseModel):
    lead_day: int
    actual_bust: bool
    normalized_error: float
    threshold_q90: float
    signature: str


class CaseSummary(BaseModel):
    region_days: int
    verified_busts: int
    sentinel_alerts: int
    sentinel_hits: int
    hidden_busts: int


class BustCell(BaseModel):
    region_id: str
    lead_day: int
    bust: int
    signature: str


class MemoryEntry(BaseModel):
    region_id: str
    lead_day: int
    valid_time: str
    bust: bool
    normalized_error: float
    signature: str


class MemoryUpdate(BaseModel):
    entries_added: int
    entry: MemoryEntry
    verified_cases_before: int
    verified_cases_after: int
    available_from: str
    store_size_region_lead: int
    note: str


class Reveal(BaseModel):
    case_id: str
    region_id: str
    lead_day: int
    reference: str
    forecast: RevealForecast
    verification: VerificationOut
    comparison: Comparison
    trajectory: list[VerifDay]
    case_summary: CaseSummary
    bust_map: list[BustCell]
    memory_update: MemoryUpdate
