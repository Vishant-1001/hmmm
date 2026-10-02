import { useEffect, useState } from "react";
import { api } from "../api";
import { BarRow, ReliabilityDiagram } from "./Charts";

// Renders artifacts/<served run>/metrics.json as written by forecast_bust.evaluation.run_qgb (v3).
const f = (v: number | null | undefined, n = 3) => (v == null || Number.isNaN(v) ? "—" : v.toFixed(n));
const QS = ["q10", "q25", "q50", "q75", "q90", "q95"];

export function Analytics() {
  const [m, setM] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { api.metrics().then(setM).catch((e) => setErr(String(e.message ?? e))); }, []);
  if (err) return <div className="card err" data-testid="metrics-error">Metrics NOT YET COMPUTED / DATA UNAVAILABLE — {err}</div>;
  if (!m) return <div className="card muted">Loading metrics…</div>;
  const mt = m.metrics;
  if (!mt.qgb) return <div className="card muted" data-testid="metrics-error">Served metrics are not in the v3 format.</div>;
  const q = mt.qgb;
  const p = q.calibrated_bust_probability;
  const raw = q.estimated_exceedance_probability_uncalibrated;
  const d = q.distribution;
  const refs = mt.references;
  const maxLead = Math.max(...mt.by_lead_day.map((x: any) => Math.max(x.auprc, x.auprc_B0)), 0.01);
  return (
    <>
      <div className="card" style={{ marginBottom: 12 }}>
        <h2>Evaluation — {mt.label}</h2>
        <p className="muted" data-testid="eval-provenance">{mt.model_type} · experiment {mt.experiment_id} · rows train {mt.rows.train.toLocaleString()} /
          validation {mt.rows.validation.toLocaleString()} / evaluated {mt.rows.test.toLocaleString()} · bust base rate {f(mt.test_base_rate)}.
          Discrimination (AUPRC), probabilistic accuracy (Brier), calibration and quantile quality are reported separately: calibration changes
          probability reliability, not ranking.</p>
        <table data-testid="headline-table">
          <thead><tr><th></th><th>AUPRC</th><th>ROC AUC</th><th>Brier</th><th>ECE</th><th>Recall @10% FAR</th><th>Hidden-bust recall</th></tr></thead>
          <tbody>
            <tr style={{ fontWeight: 600 }}><td>v3 calibrated bust probability</td><td>{f(p.auprc)}</td><td>{f(p.roc_auc)}</td><td>{f(p.brier)}</td><td>{f(p.ece)}</td>
              <td>{f(p.recall_at_far_10)}</td><td>{f(p.hidden_bust.hidden_bust_recall)} <span className="muted">(n = {p.hidden_bust.n_hidden_busts})</span></td></tr>
            <tr><td>v3 estimated exceedance (before calibration)</td><td>{f(raw.auprc)}</td><td>{f(raw.roc_auc)}</td><td>{f(raw.brier)}</td><td>{f(raw.ece)}</td><td>—</td><td>—</td></tr>
            <tr><td>B0 climatology (reference)</td><td>{f(refs.B0_climatology.auprc)}</td><td>—</td><td>{f(refs.B0_climatology.brier)}</td><td>—</td><td>—</td><td>—</td></tr>
            {refs.B2_dev_reference && <tr className="muted"><td>B2 spread-only (archival dev reference)</td><td>{f(refs.B2_dev_reference.auprc)}</td><td>—</td>
              <td>{f(refs.B2_dev_reference.brier)}</td><td>—</td><td>—</td><td>{f(refs.B2_dev_reference.hidden_bust_recall)}</td></tr>}
            {refs.v2_historical && <tr className="muted"><td>v2 B2 on 2022 (historical, different generation)</td><td>{f(refs.v2_historical.B2.auprc)}</td><td>—</td>
              <td>{f(refs.v2_historical.B2.brier)}</td><td>—</td><td>—</td><td>{f(refs.v2_historical.B2.hidden_bust_recall)}</td></tr>}
          </tbody>
        </table>
        {refs.QGB_minus_B2_block_bootstrap && <p className="muted small" data-testid="b2-diff">v3 − B2 (dev reference) AUPRC difference: mean {f(refs.QGB_minus_B2_block_bootstrap.mean, 4)}, 95% CI [{f(refs.QGB_minus_B2_block_bootstrap.ci95[0], 4)}, {f(refs.QGB_minus_B2_block_bootstrap.ci95[1], 4)}] (block bootstrap by initialisation day).</p>}
      </div>
      <div className="grid2">
        <div className="card">
          <h2>Predicted error distribution quality</h2>
          <p>Central (q50) error: MAE {f(d.mae_q50)}, RMSE {f(d.rmse_q50)} · mean pinball loss {f(d.pinball_mean, 4)} · q25–q75 coverage {f(d.central_range_q25_q75_coverage)} (nominal 0.50)</p>
          <table data-testid="quantile-table"><thead><tr><th>Quantile</th><th>Nominal</th><th>Empirical coverage</th><th>Pinball loss</th></tr></thead>
            <tbody>{QS.map((k, i) => (
              <tr key={k}><td>{k}</td><td>{f(mt.quantiles[i], 2)}</td><td>{f(d.coverage[k])}</td><td>{f(d.pinball[k], 4)}</td></tr>
            ))}</tbody></table>
          {d.validation_quantile_crossing && <p className="muted small">Quantile crossing before rearrangement (validation): {f(100 * d.validation_quantile_crossing.share_rows_crossing, 2)}% of rows; median size {f(d.validation_quantile_crossing.median_violation_when_crossing)}.</p>}
        </div>
        <div className="card">
          <h2>Reliability of the bust probability</h2>
          <ReliabilityDiagram curves={[
            { name: "calibrated", color: "var(--series-1)", bins: p.reliability.bins },
            { name: "before calibration", color: "var(--series-2)", bins: raw.reliability.bins },
          ]} />
        </div>
        <div className="card">
          <h2>AUPRC by lead day</h2>
          {mt.by_lead_day.map((r: any) => (
            <div key={r.lead_day} style={{ marginBottom: 6 }}>
              <div className="muted">Day {r.lead_day} · bust rate {f(r.bust_rate, 2)} · MAE(q50) {f(r.mae_q50)}</div>
              <BarRow label="v3" value={r.auprc} max={maxLead} />
              <BarRow label="B0 climatology" value={r.auprc_B0} max={maxLead} color="var(--neutral)" />
            </div>
          ))}
        </div>
        <div className="card">
          <h2>Operational and regional checks</h2>
          <table><tbody>
            <tr><td>Operating point (chosen on validation, 10% FAR)</td><td>threshold {f(p.operating_point.threshold)} · precision {f(p.operating_point.precision)} · recall {f(p.operating_point.recall)}</td></tr>
            <tr><td>Mean lead day of detected busts</td><td>{f(p.warning_lead.mean_lead_day_of_detected_busts, 2)}</td></tr>
            <tr><td>Low-spread AUPRC</td><td>{f(p.hidden_bust.low_spread_auprc)}</td></tr>
            <tr><td>Q95 bust label AUPRC (sensitivity)</td><td>{f(q.q95_sensitivity.auprc)}</td></tr>
            <tr><td>Peak-risk day error (mean |Δday|)</td><td>{f(q.peak_risk_day.mean_abs_peak_day_error, 2)}</td></tr>
            <tr><td>Spatial overlap (mean Jaccard)</td><td>{f(q.spatial_overlap.mean_jaccard)}</td></tr>
            <tr><td>Regional AUPRC (median, 10th–90th pct over {mt.regional_stability.n_regions} regions)</td>
              <td>{f(mt.regional_stability.auprc_median)} ({f(mt.regional_stability.auprc_p10)}–{f(mt.regional_stability.auprc_p90)})</td></tr>
          </tbody></table>
        </div>
      </div>
    </>
  );
}
