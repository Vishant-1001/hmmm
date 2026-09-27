import { useEffect, useState } from "react";
import { api } from "../api";
import { BarRow, ReliabilityDiagram } from "./Charts";

const f = (v: number | null | undefined, n = 3) => (v == null || Number.isNaN(v) ? "—" : v.toFixed(n));
const ORDER = ["B0", "B1", "B2", "M1", "M2", "M3", "M4", "M5", "FULL"];

export function Analytics() {
  const [m, setM] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { api.metrics().then(setM).catch((e) => setErr(String(e.message ?? e))); }, []);
  if (err) return <div className="card err" data-testid="metrics-error">Metrics NOT YET COMPUTED / DATA UNAVAILABLE — {err}</div>;
  if (!m) return <div className="card muted">Loading metrics…</div>;
  const mt = m.metrics;
  const models = mt.models;
  const fv = mt.full_vs_b2;
  const ds = mt.dataset;
  return (
    <>
      <div className="card" style={{ marginBottom: 12 }}>
        <h2>Does Sentinel add predictive value beyond ensemble spread? (TEST split, evaluated once)</h2>
        <p data-testid="verdict" style={{ fontSize: 15 }}>
          {fv.material_improvement
            ? <>Material improvement established under the pre-registered rule: AUPRC {f(models.FULL.auprc)} vs {f(models.B2.auprc)} for the calibrated spread-only baseline (B2).</>
            : <><strong>Incremental predictive value NOT established</strong> under the pre-registered rule: AUPRC {f(models.FULL.auprc)} (Sentinel) vs {f(models.B2.auprc)} (B2).</>}
          {" "}Difference {f(fv.auprc_gain)} ({(100 * fv.relative_gain).toFixed(1)}% relative), 95% block-bootstrap CI [{f(fv.bootstrap.ci95[0])}, {f(fv.bootstrap.ci95[1])}].
        </p>
        <p className="muted">{fv.verdict_rule}. Test base rate {f(mt.test_base_rate)} over {mt.n_test_rows.toLocaleString()} region×day cases.
          Data: {ds.source}; reference: {ds.reference}.</p>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Split</th><th>Initialisations</th><th>First</th><th>Last</th><th>Bust rate</th></tr></thead>
            <tbody>{Object.entries(ds.splits).map(([k, v]: any) => (
              <tr key={k}><td>{k}</td><td>{v.inits}</td><td>{String(v.first_init).slice(0, 13)}</td><td>{String(v.last_init).slice(0, 13)}</td><td>{f(v.bust_rate)}</td></tr>
            ))}</tbody>
          </table>
        </div>
      </div>

      <div className="card" style={{ marginBottom: 12 }}>
        <h2>Model comparison and ablation (TEST)</h2>
        <div className="table-wrap">
          <table data-testid="model-table">
            <thead><tr><th>Model</th><th>Description</th><th>AUPRC</th><th>ROC AUC</th><th>Brier</th><th>ECE</th>
              <th>Recall @ FAR 10%</th><th>Hidden-bust recall</th><th>Mean lead day of detected busts</th></tr></thead>
            <tbody>{ORDER.filter((k) => models[k]).map((k) => {
              const r = models[k];
              return (
                <tr key={k} style={k === "FULL" || k === "B2" ? { fontWeight: 600 } : undefined}>
                  <td>{k}</td><td className="muted">{r.description}</td><td>{f(r.auprc)}</td><td>{f(r.roc_auc)}</td>
                  <td>{f(r.brier, 4)}</td><td>{f(r.ece, 4)}</td><td>{f(r.recall_at_far_10)}</td>
                  <td>{f(r.hidden_bust.hidden_bust_recall)}</td><td>{f(r.warning_lead.mean_lead_day_of_detected_busts, 2)}</td>
                </tr>
              );
            })}</tbody>
          </table>
        </div>
        <p className="muted">B1 is a ranking score (spread percentile), so Brier/ECE do not apply. Hidden-bust recall and operating-point
          metrics use each model's alert threshold chosen on VALIDATION for a 10% false-alarm rate.</p>
      </div>

      <div className="grid2">
        <div className="card">
          <h2>Calibration (TEST reliability diagram)</h2>
          {m.calibration ? (
            <ReliabilityDiagram curves={[
              { name: "Sentinel (FULL)", color: "var(--series-1)", bins: m.calibration.FULL.test.bins },
              { name: "Spread baseline (B2)", color: "var(--series-2)", bins: m.calibration.B2.test.bins },
            ]} />
          ) : <p className="muted">NOT YET COMPUTED</p>}
          <p className="muted">{m.calibration?.method}</p>
        </div>
        <div className="card">
          <h2>AUPRC by lead day (TEST)</h2>
          {mt.by_lead_day.map((r: any) => (
            <div key={r.lead_day}>
              <div className="muted">Day {r.lead_day} (bust rate {f(r.bust_rate, 2)})</div>
              <BarRow label="Sentinel" value={r.auprc_FULL} max={Math.max(...mt.by_lead_day.map((x: any) => x.auprc_FULL), 0.01)} />
              <BarRow label="B2 spread baseline" value={r.auprc_B2} max={Math.max(...mt.by_lead_day.map((x: any) => x.auprc_FULL), 0.01)} color="var(--series-2)" />
            </div>
          ))}
        </div>
      </div>

      <div className="grid2" style={{ marginTop: 12 }}>
        <div className="card">
          <h2>Diagnostics</h2>
          <table><tbody>
            <tr><td>Peak-risk day error (Sentinel, mean |Δday|)</td><td>{f(models.FULL.peak_risk_day.mean_abs_peak_day_error, 2)} (within ±1 day: {f(models.FULL.peak_risk_day.within_1_day, 2)})</td></tr>
            <tr><td>Peak-risk day error (B2)</td><td>{f(models.B2.peak_risk_day.mean_abs_peak_day_error, 2)}</td></tr>
            <tr><td>Spatial overlap, mean Jaccard (Sentinel / B2)</td><td>{f(models.FULL.spatial_overlap.mean_jaccard)} / {f(models.B2.spatial_overlap.mean_jaccard)}</td></tr>
            <tr><td>Low-spread AUPRC (Sentinel / B2)</td><td>{f(models.FULL.hidden_bust.low_spread_auprc)} / {f(models.B2.hidden_bust.low_spread_auprc)}</td></tr>
            <tr><td>Q95 sensitivity AUPRC (FULL / B2 / B0)</td><td>{f(mt.q95_sensitivity.FULL.auprc)} / {f(mt.q95_sensitivity.B2.auprc)} / {f(mt.q95_sensitivity.B0.auprc)}</td></tr>
            <tr><td>Frozen-memory FULL AUPRC</td><td>{f(mt.frozen_memory_sensitivity.auprc)}</td></tr>
            {m.fingerprint_metrics && <>
              <tr><td>Failure-signature top-1 agreement (analogues / climatological reference)</td><td>{f(m.fingerprint_metrics.top1_agreement)} / {f(m.fingerprint_metrics.reference_top1_agreement)}</td></tr>
              <tr><td>Mean probability assigned to actual signature (analogues / reference)</td><td>{f(m.fingerprint_metrics.mean_prob_assigned_to_actual)} / {f(m.fingerprint_metrics.reference_mean_prob_assigned)}</td></tr>
            </>}
          </tbody></table>
          <h3>Evidence strength vs outcome (TEST)</h3>
          <table><thead><tr><th>Evidence</th><th>n</th><th>Mean Sentinel p</th><th>Observed bust rate</th></tr></thead>
            <tbody>{mt.by_evidence_level.map((r: any) => (
              <tr key={r.evidence}><td>{r.evidence}</td><td>{r.n}</td><td>{f(r.mean_p_FULL)}</td><td>{f(r.bust_rate)}</td></tr>
            ))}</tbody></table>
        </div>
        <div className="card">
          <h2>Spread–skill (IFS ENS, TEST, target regions)</h2>
          {m.spread_skill ? (
            <table><thead><tr><th>Day</th><th>RMSE (m)</th><th>Spread (m)</th><th>Spread/skill</th><th>corr(spread, error)</th></tr></thead>
              <tbody>{m.spread_skill.per_lead.map((r: any) => (
                <tr key={r.lead_day}><td>{r.lead_day}</td><td>{f(r.rmse_m, 1)}</td><td>{f(r.spread_m, 1)}</td><td>{f(r.spread_skill_ratio, 2)}</td><td>{f(r.corr_spread_error, 2)}</td></tr>
              ))}</tbody></table>
          ) : <p className="muted">NOT YET COMPUTED</p>}
          {m.spread_skill && <RankHist counts={Object.values(m.spread_skill.rank_histogram_by_lead_day as Record<string, number[]>)
            .reduce((a: number[], c: number[]) => a.map((x, i) => x + c[i]))} />}
          <p className="muted">{m.spread_skill?.note}</p>
        </div>
      </div>
    </>
  );
}

function RankHist({ counts }: { counts: number[] }) {
  const tot = counts.reduce((a, b) => a + b, 0);
  const mx = Math.max(...counts);
  const W = 500, H = 120;
  const bw = W / counts.length;
  return (
    <div>
      <h3>Rank histogram of ERA5 among 50 members (all lead days)</h3>
      <svg viewBox={`0 0 ${W} ${H + 16}`} width="100%" role="img" aria-label="Rank histogram">
        <line x1={0} x2={W} y1={H - (H * (tot / counts.length)) / mx} y2={H - (H * (tot / counts.length)) / mx}
          stroke="var(--text-muted)" strokeDasharray="4 3" />
        {counts.map((c, i) => (
          <rect key={i} x={i * bw + 1} y={H - (H * c) / mx} width={bw - 2} height={(H * c) / mx} rx={1} fill="var(--series-1)">
            <title>{`rank ${i}: ${c} (${((100 * c) / tot).toFixed(1)}%)`}</title>
          </rect>
        ))}
        <text x={0} y={H + 13}>rank 0</text>
        <text x={W} y={H + 13} textAnchor="end">rank 50</text>
      </svg>
      <div className="muted">Dashed line = flat (perfectly dispersed) expectation. U-shape indicates under-dispersion.</div>
    </div>
  );
}
