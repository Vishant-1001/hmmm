export interface Driver {
  feature: string;
  label: string;
  group: string;
  value: number | null;
  train_percentile: number | null;
  contribution_logodds: number;
  direction: string;
}

export interface Explanation {
  drivers: Driver[];
  groups: Record<string, { name: string; contribution_logodds: number }>;
  evidence: { kind: string; text: string }[];
  attribution_note: string;
}

export interface Analogue {
  case_id: string;
  init_time: string;
  distance: number | null;
  bust: number;
  normalized_error: number | null;
  signature: string | null;
}

export interface SignatureExpectation {
  basis: string;
  n: number;
  distribution: Record<string, number> | null;
}

export interface Day {
  lead_day: number;
  valid_time: string;
  p_bust: number;
  confidence: number;
  p_spread_baseline: number;
  p_climatology: number;
  disagreement_pp: number;
  spread_m: number;
  spread_pct: number | null;
  alert: boolean;
  support: string;
  support_distance: number | null;
  evidence: string;
  analogues_within_radius: number | null;
  verified_cases_available: number | null;
  analogue_bust_rate: number | null;
  priority_score: number;
  explanation: Explanation;
  analogues: Analogue[];
  historical_failure_signature: SignatureExpectation;
}

export interface Region {
  region_id: string;
  name: string;
  lat: number;
  lon: number;
  lat_bounds: [number, number];
  lon_bounds: [number, number];
  trajectory: Day[];
  peak_risk_day: number;
  peak_p_bust: number;
  first_alert_day: number | null;
  alert_days: number[];
}

export interface QueueItem {
  region_id: string;
  lead_day: number;
  score: number;
  p_bust: number;
  p_spread_baseline: number;
  disagreement_pp: number;
  evidence: string;
  analogues_within_radius: number | null;
  support: string;
}

export interface ForecastCase {
  case_id: string;
  init_time: string;
  mode: string;
  data_source: string;
  selection: string;
  selection_note: string;
  alert_threshold: { p_bust: number; definition: string };
  priority_formula: Record<string, unknown>;
  priority_queue: QueueItem[];
  regions: Region[];
  fields: { lead_days: number[]; lats: number[]; lons: number[]; ens_mean_z500: number[][][]; ens_spread_z500: number[][][] };
}

export interface VerifDay {
  lead_day: number;
  error_m: number;
  normalized_error: number;
  threshold_q90: number;
  bust: number;
  bust_q95: number;
  hidden_bust: number;
  signature: string;
  signature_label: string;
  phase_share: number;
  bias_m: number;
  pattern_corr: number;
}

export interface Verification {
  case_id: string;
  reference: string;
  era5_z500: number[][][];
  regions: { region_id: string; days: VerifDay[] }[];
  summary: Record<string, number>;
}

export interface CaseIndexItem {
  case_id: string;
  init_time: string;
  selection: string;
  selection_note: string;
  n_alerts: number;
}
