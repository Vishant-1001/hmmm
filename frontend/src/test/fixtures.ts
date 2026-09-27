// TEST FIXTURES ONLY — synthetic values for rendering tests. Never scientific results.
import type { ForecastCase, Verification } from "../types";

const day = (d: number, p: number) => ({
  lead_day: d, valid_time: `2022-01-0${Math.min(d, 9)}T00:00:00`, p_bust: p, confidence: 1 - p, p_spread_baseline: 0.1,
  p_climatology: 0.1, disagreement_pp: 100 * (p - 0.1), spread_m: 10, spread_pct: 0.3, alert: p > 0.3,
  support: "NORMAL SUPPORT", support_distance: 1, evidence: "MODERATE EVIDENCE", analogues_within_radius: 20,
  verified_cases_available: 100, analogue_bust_rate: 0.2, priority_score: p,
  explanation: { drivers: [], groups: {}, evidence: [{ kind: "C. Historical evidence", text: "FIXTURE evidence" }], attribution_note: "fixture" },
  analogues: [], historical_failure_signature: { basis: "none", n: 0, distribution: null },
});

export const fixtureCase: ForecastCase = {
  case_id: "2022010100", init_time: "2022-01-01T00:00:00", mode: "TEST FIXTURE", data_source: "TEST FIXTURE",
  selection: "random", selection_note: "TEST FIXTURE", alert_threshold: { p_bust: 0.3, definition: "fixture" },
  priority_formula: { formula: "fixture" },
  priority_queue: [{ region_id: "R0000", lead_day: 3, score: 1, p_bust: 0.6, p_spread_baseline: 0.1, disagreement_pp: 50,
    evidence: "MODERATE EVIDENCE", analogues_within_radius: 20, support: "NORMAL SUPPORT" }],
  regions: [{ region_id: "R0000", name: "Fixture", lat: 0, lon: 62, lat_bounds: [-2.8, 2.8], lon_bounds: [59, 64.7],
    trajectory: Array.from({ length: 10 }, (_, i) => day(i + 1, i === 2 ? 0.6 : 0.05)),
    peak_risk_day: 3, peak_p_bust: 0.6, first_alert_day: 3, alert_days: [3] }],
  fields: { lead_days: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], lats: [0, 5.6], lons: [62, 67.5],
    ens_mean_z500: Array(10).fill([[5800, 5810], [5805, 5815]]), ens_spread_z500: Array(10).fill([[5, 6], [7, 8]]) },
};

export const fixtureVerification: Verification = {
  case_id: "2022010100", reference: "TEST FIXTURE", era5_z500: Array(10).fill([[5790, 5810], [5805, 5815]]),
  regions: [{ region_id: "R0000", days: Array.from({ length: 10 }, (_, i) => ({
    lead_day: i + 1, error_m: 10, normalized_error: i === 2 ? 2 : 0.5, threshold_q90: 1, bust: i === 2 ? 1 : 0, bust_q95: 0,
    hidden_bust: 0, signature: "POSITION_PHASE", signature_label: "Position / phase mismatch", phase_share: 0.8, bias_m: 1, pattern_corr: 0.5 })) }],
  summary: { n_bust_region_days: 1, n_hidden_bust_region_days: 0, sentinel_alerts: 1, sentinel_hits: 1, b2_alerts: 0, b2_hits: 0,
    jaccard_sentinel: 1, jaccard_b2: 0 },
};
