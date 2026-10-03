import { useEffect, useState } from "react";
import { api } from "../api";
import { BarRow, ReliabilityDiagram } from "./Charts";

// Renders artifacts/v2/metrics.json (the B2 benchmark: real ECMWF IFS ENS + ERA5, 2022 evaluation) unchanged.
const f = (v: number | null | undefined, n = 3) => (v == null || Number.isNaN(v) ? "—" : v.toFixed(n));
const ORDER = ["B0", "B1", "B2", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "ALL", "FULL", "EXP_RESIDUAL_B2"];
const NAME: Record<string, string> = { B0: "B0 climatology", B1: "B1 spread percentile", B2: "B2 (served)", FULL: "Sentinel (FULL)" };

export function Analytics() {
  const [m, setM] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { api.metrics().then(setM).catch((e) => setErr(String(e.message ?? e))); }, []);
  if (err) return <div className="panel err" data-testid="metrics-error">Metrics DATA UNAVAILABLE — {err}</div>;
  if (!m) return <div className="panel muted">Loading metrics…</div>;
  const mt = m.metrics;
  const models = mt?.models;
  if (!models?.B2 || !mt.full_vs_b2) return <div className="panel muted" data-testid="metrics-error">Served metrics are not in the B2 benchmark format.</div>;
  const b2 = models.B2;
  const fv = mt.full_vs_b2;
  const ci = mt.bootstrap_ci?.models?.B2?.auprc?.ci95;
  const maxLead = Math.max(...mt.by_lead_day.map((x: any) => Math.max(x.auprc_B2, x.auprc_FULL ?? 0)), 0.01);
  return (
    <>
      <section className="panel" data-testid="headline">
        <header className="panel-h"><h2>B2 benchmark — real ECMWF IFS ENS + ERA5, 2022 evaluation</h2></header>
        <p data-testid="headline-b2">B2 AUPRC <strong>{f(b2.auprc)}</strong>{ci && <> (95% CI {f(ci[0])}–{f(ci[1])})</>} at a bust base rate of {f(mt.test_base_rate)};
          climatology B0 {f(models.B0.auprc)}. ROC AUC {f(b2.roc_auc)}, Brier {f(b2.brier, 4)}, ECE {f(b2.ece, 4)}, recall at a 10% false-alarm rate {f(b2.recall_at_far_10)}.
          B2 has real but modest skill: it is the strongest validated baseline and the model this MVP serves.</p>
        <p data-testid="verdict">
          {fv.material_improvement
            ? <>The richer Sentinel model showed a material improvement over B2 under the pre-registered rule.</>
            : <><strong>The richer Sentinel model did NOT beat B2</strong>: AUPRC {f(models.FULL.auprc)} vs {f(b2.auprc)}, difference {f(fv.auprc_gain, 4)} (95% block-bootstrap CI [{f(fv.bootstrap.ci95[0], 4)}, {f(fv.bootstrap.ci95[1], 4)}]). That is why B2 is served.</>}
        </p>
        <p className="muted small">{fv.verdict_rule}. {mt.n_test_rows.toLocaleString()} region×day cases; data {mt.dataset.source}; reference {mt.dataset.reference}.</p>
        <p className="muted small" data-testid="test-history"><strong>Test-set history:</strong> {mt.test_history} Since then, 2022 was also read by the V3 and V4 evaluations, so it is not an untouched test set.</p>
      </section>

      <section className="panel">
        <header className="panel-h"><h2>All evaluated models (2022, unchanged)</h2></header>
        <div className="table-wrap">
          <table className="tbl" data-testid="headline-table">
            <thead><tr><th>Model</th><th>Description</th><th>AUPRC</th><th>ROC AUC</th><th>Brier</th><th>ECE</th><th>Recall @ FAR 10%</th></tr></thead>
            <tbody>{ORDER.filter((k) => models[k]).map((k) => {
              const r = models[k];
              return (
                <tr key={k} style={k === "B2" ? { fontWeight: 650 } : undefined}>
                  <td>{NAME[k] ?? k}</td><td className="muted">{r.description}</td><td>{f(r.auprc)}</td><td>{f(r.roc_auc)}</td>
                  <td>{f(r.brier, 4)}</td><td>{f(r.ece, 4)}</td><td>{f(r.recall_at_far_10)}</td>
                </tr>
              );
            })}</tbody>
          </table>
        </div>
        <p className="muted small">B1 is a ranking score, so Brier and ECE do not apply. Operating points use thresholds chosen on validation (2021).
          M1–M7 and ALL add feature groups to B2; none is materially better. Axes and tables are not truncated.</p>
      </section>

      <div className="grid-2">
        <section className="panel">
          <header className="panel-h"><h2>Calibration of B2 (2022 reliability diagram)</h2></header>
          {m.calibration?.B2 ? (
            <ReliabilityDiagram curves={[{ name: "B2 (served)", color: "var(--series-1)", bins: m.calibration.B2.test.bins }]} />
          ) : <p className="muted">NOT YET COMPUTED</p>}
          <p className="muted small">{m.calibration?.method}</p>
        </section>
        <section className="panel">
          <header className="panel-h"><h2>AUPRC by lead day (2022)</h2></header>
          {mt.by_lead_day.map((r: any) => (
            <div key={r.lead_day}>
              <div className="muted small">Day {r.lead_day} · bust rate {f(r.bust_rate, 2)}</div>
              <BarRow label="B2" value={r.auprc_B2} max={maxLead} />
              <BarRow label="B0" value={r.auprc_B0} max={maxLead} color="var(--text-muted)" />
            </div>
          ))}
        </section>
      </div>

      <section className="panel">
        <header className="panel-h"><h2>Archived experiments (not served)</h2></header>
        <p className="muted small">V3 quantile gradient boosting (did not beat B2 on the development split), BMA (failed its pre-registered gate) and the V4
          pattern-aware model (a different, rarer target) are documented in docs/model_card_v3.md, docs/model_card_bma.md and docs/model_card_v4.md.
          Their results are not comparable with this table and are not used by the demo.</p>
      </section>
    </>
  );
}
