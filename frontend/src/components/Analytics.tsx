import { useEffect, useState } from "react";
import { api } from "../api";
import { BarRow, ReliabilityDiagram } from "./Charts";

// Renders artifacts/<served run>/metrics.json as written by forecast_bust.evaluation.run_v4 (V4).
const f = (v: number | null | undefined, n = 3) => (v == null || Number.isNaN(v) ? "—" : v.toFixed(n));

export function Analytics() {
  const [m, setM] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { api.metrics().then(setM).catch((e) => setErr(String(e.message ?? e))); }, []);
  if (err) return <div className="card err" data-testid="metrics-error">Metrics NOT YET COMPUTED / DATA UNAVAILABLE — {err}</div>;
  if (!m) return <div className="card muted">Loading metrics…</div>;
  const mt = m.metrics;
  if (!mt.splits?.test?.pattern_target) return <div className="card muted" data-testid="metrics-error">Served metrics are not in the V4 format.</div>;
  const t = mt.splits.test;
  const pt = t.pattern_target;
  const v4 = pt.V4;
  const mag = t.magnitude_target.same_model_on_magnitude;
  const ps = t.pattern_specific;
  const maxLead = Math.max(...t.by_lead_day.map((x: any) => x.auprc), 0.001);
  const rows: [string, any][] = [["V4 calibrated", v4], ["B0 pattern-bust climatology", pt.B0_climatology], ["B1 spread percentile", pt.B1_spread]];
  return (
    <>
      <div className="card" style={{ marginBottom: 12 }}>
        <h2>Evaluation — {mt.label}</h2>
        <p className="muted" data-testid="eval-provenance">Event: pattern-aware bust (large error AND poor local pattern agreement), prevalence {f(v4.prevalence, 4)}.
          AUPRC must be read against that prevalence (lift = AUPRC / prevalence). Discrimination, Brier, calibration and alerting are reported separately.</p>
        <table data-testid="headline-table">
          <thead><tr><th></th><th>AUPRC</th><th>Lift</th><th>ROC AUC</th><th>Brier</th><th>ECE</th><th>Recall @10% FAR</th><th>Precision at alert</th><th>Hidden-bust recall</th></tr></thead>
          <tbody>{rows.map(([name, x]) => (
            <tr key={name} style={name.startsWith("V4") ? { fontWeight: 600 } : undefined}>
              <td>{name}</td><td>{f(x.auprc, 4)}</td><td>{f(x.lift, 2)}</td><td>{f(x.roc_auc)}</td><td>{f(x.brier, 5)}</td><td>{f(x.ece, 4)}</td>
              <td>{f(x.recall_at_far_10)}</td><td>{f(x.operating_point.precision)}</td><td>{f(x.hidden_bust.hidden_bust_recall)}</td>
            </tr>
          ))}</tbody>
        </table>
        <p className="muted small" data-testid="adequacy">Same model, same features, trained on the old magnitude-only target: ROC AUC {f(mag.roc_auc)} (AUPRC {f(mag.auprc, 4)} at prevalence {f(mag.prevalence, 3)}).
          Pattern-target ROC AUC gain {f(t.adequacy_rocauc_gain.point, 3)} [{f(t.adequacy_rocauc_gain.ci95[0], 3)}, {f(t.adequacy_rocauc_gain.ci95[1], 3)}].
          Part of that gain reflects that low local correlation is more frequent when the forecast anomaly field is weak (see model card).</p>
      </div>
      <div className="grid2">
        <div className="card">
          <h2>Reliability of the pattern-aware bust probability</h2>
          <ReliabilityDiagram minN={200} curves={[{ name: "V4 calibrated", color: "var(--series-1)", bins: v4.reliability.bins }]} />
        </div>
        <div className="card">
          <h2>AUPRC by lead day</h2>
          {t.by_lead_day.map((r: any) => (
            <div key={r.lead_day} style={{ marginBottom: 6 }}>
              <div className="muted">Day {r.lead_day} · {r.positives} events · lift {f(r.lift, 2)} · ROC AUC {f(r.roc_auc)}</div>
              <BarRow label="V4" value={r.auprc} max={maxLead} fmt={(v) => v.toFixed(3)} />
              <BarRow label="B1 spread" value={r.auprc_B1} max={maxLead} color="var(--neutral)" fmt={(v) => v.toFixed(3)} />
            </div>
          ))}
        </div>
        <div className="card">
          <h2>Is V4 alerting on the intended event?</h2>
          <table><tbody>
            <tr><td>Alerts (validation 10%-FAR threshold)</td><td>{ps.alerts}</td></tr>
            <tr><td>Share of alerts with a large-error bust / a pattern failure</td><td>{f(ps.alert_share_magnitude_busts)} / {f(ps.alert_share_pattern_failures)}</td></tr>
            <tr><td>Median normalized error, alerts vs others</td><td>{f(ps.median_norm_error_alert_vs_not[0])} vs {f(ps.median_norm_error_alert_vs_not[1])}</td></tr>
            <tr><td>Median local ACC, alerts vs others</td><td>{f(ps.median_local_acc_alert_vs_not[0])} vs {f(ps.median_local_acc_alert_vs_not[1])}</td></tr>
            <tr><td>ROC AUC of V4 for the large-error / pattern-failure criterion alone</td><td>{f(ps.rocauc_V4_for_magnitude_failure)} / {f(ps.rocauc_V4_for_pattern_failure)}</td></tr>
            <tr><td>Warning lead (mean lead day of detected busts)</td><td>{f(v4.warning_lead.mean_lead_day_of_detected_busts, 2)}</td></tr>
            <tr><td>Regional ROC AUC (median, 10th–90th pct; regions with ≥ 20 events)</td>
              <td>{f(t.by_region.auc_median)} ({f(t.by_region.auc_p10_p90?.[0])}–{f(t.by_region.auc_p10_p90?.[1])}), {t.by_region.n_regions_ge_20_positives} regions</td></tr>
          </tbody></table>
        </div>
        <div className="card">
          <h2>Archived generations (history, not used by V4)</h2>
          <p className="muted small">Measured on the earlier magnitude-only target, so not directly comparable with V4: v2 B2 / Sentinel, V3 quantile
            gradient boosting and BMA (failed its pre-registered gate). See docs/model_card_v3.md, docs/model_card_bma.md, docs/evaluation_v4.md.</p>
        </div>
      </div>
    </>
  );
}
