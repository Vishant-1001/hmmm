// Mirrors src/forecast_bust/demo/schemas.py (the API contract).
export interface CaseInfo {
  case_id: string; init_time: string; season: string; selection: string; demo_role: string;
  selection_note: string; valid_range: string[]; n_regions: number; lead_days: number[]; split: string;
  model_input: string; verification: string;
}
export interface CaseList { mode: string; selection_rule: string; cases: CaseInfo[] }
export interface ModelSummary {
  model_version: string; model_artifact: string; exported_from: string; retrained_for_demo: boolean;
  training_window: string; calibration_window: string; calibration: string; learner: string;
  sentinel_features: string[]; b2_features: string[]; selected_groups: string[]; selection_note: string;
  pending: string; alert_threshold: number; alert_threshold_definition: string; confidence_definition: string; target: string;
}
export interface Cell {
  lead_day: number; valid_time: string; bust_probability: number; reliability_confidence: number;
  b2_probability: number; b0_probability: number; disagreement_pp: number; alert: boolean; spread_m: number;
  spread_pct: number | null; support_level: string; support_distance: number | null; evidence_quality: string;
  analogues_within_radius: number | null; verified_cases_available: number | null; analogue_bust_rate: number | null;
  priority_score: number;
}
export interface QueueItem extends Cell { region_id: string }
export interface RegionOverview {
  region_id: string; lat: number; lon: number; row: number; col: number; peak_lead_day: number;
  peak_bust_probability: number; alert_days: number[]; days: Cell[];
}
export interface LeadSummary { lead_day: number; mean_bust_probability: number; max_bust_probability: number; n_alerts: number; mean_disagreement_pp: number }
export interface RunResult {
  mode: string; case: CaseInfo; model: ModelSummary; timings_ms: Record<string, number>; alert_threshold: number;
  priority_formula: { formula: string; H_days: number; B: number; evidence_weight: Record<string, number>; note: string };
  regions: RegionOverview[]; priority_queue: QueueItem[]; lead_summary: LeadSummary[]; blind: boolean;
}
export interface RegionDetail { case_id: string; region_id: string; lat: number; lon: number; alert_threshold: number; trajectory: Cell[] }
export interface Driver { feature: string; label: string; group: string; value: number | null; train_percentile: number | null; contribution_logodds: number; direction: string }
export interface Analogue { case_id: string; init_time: string; distance: number; bust: number; normalized_error: number; signature: string | null; split: string }
export interface FailureSignature { basis: string; n: number; distribution: Record<string, number> | null; top: string | null; top_label?: string | null }
export interface Explanation extends Cell {
  case_id: string; region_id: string;
  attribution: { method: string; bias_logodds: number; drivers: Driver[]; groups: Record<string, number>; note: string };
  evidence: { kind: string; text: string }[];
  analogue_summary: {
    k: number; radius: number; rule: string; verified_cases_available: number | null; within_radius: number | null;
    bust_rate: number | null; climatological_bust_rate: number; median_normalized_error: number | null;
    q90_normalized_error: number | null; nearest: Analogue[];
  };
  failure_signature: FailureSignature;
  context_features: { feature: string; label: string; value: number | null; train_percentile: number | null }[];
  model_inputs: Record<string, number | null>;
}
export interface Reveal {
  case_id: string; region_id: string; lead_day: number; reference: string;
  forecast: { bust_probability: number; b2_probability: number; alert: boolean; expected_signature: FailureSignature };
  verification: {
    actual_bust: boolean; normalized_error: number; threshold_q90: number; error_m: number; hidden_bust: boolean;
    failure_fingerprint: { signature: string; label: string; phase_share: number | null; bias_m: number | null; pattern_corr: number | null };
  };
  comparison: { expected_top: string | null; p_expected_for_actual: number | null; match: boolean | null; note: string };
  trajectory: { lead_day: number; actual_bust: boolean; normalized_error: number; threshold_q90: number; signature: string }[];
  case_summary: { region_days: number; verified_busts: number; sentinel_alerts: number; sentinel_hits: number; hidden_busts: number };
  bust_map: { region_id: string; lead_day: number; bust: number; signature: string }[];
  memory_update: {
    entries_added: number; verified_cases_before: number; verified_cases_after: number; available_from: string;
    store_size_region_lead: number; note: string;
    entry: { region_id: string; lead_day: number; valid_time: string; bust: boolean; normalized_error: number; signature: string };
  };
}
export interface Fields { lead_days: number[]; lats: number[]; lons: number[]; ens_mean_z500: number[][][]; ens_spread_z500: number[][][] }
