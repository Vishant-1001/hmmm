import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import type { CaseIndexItem, ForecastCase, Verification, VerifDay } from "../types";
import { BarRow, ErrorRatioChart, TrajectoryChart } from "./Charts";
import { FieldMap, RiskMap } from "./Maps";

const SIG_LABEL: Record<string, string> = {
  POSITION_PHASE: "Position / phase", AMPLITUDE_STRUCTURE: "Amplitude / structure", RESIDUAL_MIXED: "Residual / mixed",
};
const pct = (v: number | null | undefined) => (v == null ? "—" : `${(100 * v).toFixed(0)}%`);

export function CaseView({ cases }: { cases: CaseIndexItem[] }) {
  const [caseId, setCaseId] = useState(cases[0]?.case_id ?? "");
  const [data, setData] = useState<ForecastCase | null>(null);
  const [verif, setVerif] = useState<Verification | null>(null);
  const [revealed, setRevealed] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [day, setDay] = useState(5);
  const [region, setRegion] = useState<string | null>(null);
  const [field, setField] = useState<"mean" | "spread" | "error">("mean");

  useEffect(() => {
    if (!caseId) return;
    setData(null); setVerif(null); setRevealed(false); setErr(null);
    api.forecastCase(caseId).then((c) => {
      setData(c);
      const top = c.priority_queue[0];
      setRegion(top?.region_id ?? c.regions[0].region_id);
      setDay(top?.lead_day ?? 5);
    }).catch((e) => setErr(String(e.message ?? e)));
  }, [caseId]);

  const reveal = () => {
    api.verification(caseId).then((v) => { setVerif(v); setRevealed(true); }).catch((e) => setErr(String(e.message ?? e)));
  };
  const vmap = useMemo(() => {
    if (!verif || !revealed) return undefined;
    return new Map<string, VerifDay[]>(verif.regions.map((r) => [r.region_id, r.days]));
  }, [verif, revealed]);

  if (err) return <div className="card err">{err}</div>;
  if (!data) return <div className="card muted">Loading historical case…</div>;
  const reg = data.regions.find((r) => r.region_id === region) ?? data.regions[0];
  const d = reg.trajectory[day - 1];
  const vr = vmap?.get(reg.region_id);
  const vd = vr?.[day - 1];
  const sig = d.historical_failure_signature;

  return (
    <>
      <div className="card" style={{ marginBottom: 12 }}>
        <div className="row" style={{ justifyContent: "space-between" }}>
          <div className="row">
            <label htmlFor="case">Forecast initialisation</label>
            <select id="case" value={caseId} onChange={(e) => setCaseId(e.target.value)}>
              {cases.map((c) => (
                <option key={c.case_id} value={c.case_id}>{c.init_time.slice(0, 16)} UTC — {c.selection}</option>
              ))}
            </select>
          </div>
          {!revealed ? (
            <button className="btn primary" onClick={reveal} data-testid="reveal">REVEAL VERIFICATION</button>
          ) : (
            <button className="btn" onClick={() => setRevealed(false)}>Back to blind mode</button>
          )}
        </div>
        <div className="muted" style={{ marginTop: 6 }}>
          {data.data_source} · {data.selection_note} ·{" "}
          <strong data-testid="mode-flag">{revealed ? "VERIFICATION REVEALED (ERA5 reference)" : "BLIND MODE — verification hidden"}</strong>
        </div>
        <div className="row daybar" style={{ marginTop: 8 }}>
          <span className="muted">Lead day:</span>
          {data.fields.lead_days.map((ld) => (
            <button key={ld} className="btn" aria-pressed={ld === day} onClick={() => setDay(ld)}
              style={ld === day ? { background: "var(--text-primary)", color: "var(--surface-1)" } : undefined}>D{ld}</button>
          ))}
        </div>
      </div>

      <div className="grid2">
        <div className="card">
          <h2>WHERE — regional bust risk, Day {day} (+{24 * day} h)</h2>
          <RiskMap regions={data.regions} day={day} selected={reg.region_id} onSelect={setRegion}
            threshold={data.alert_threshold.p_bust} verification={vmap} />
          <p className="muted">Bust risk = calibrated probability that the <em>existing</em> IFS ENS ensemble-mean Z500
            forecast exceeds the project-defined large-error threshold (TRAIN Q90). It is not a probability of any weather event.
            5.625° regions; not a high-resolution local forecast.</p>
          {revealed && verif && (
            <div className="reveal card" style={{ marginTop: 8 }} data-testid="verif-summary">
              <h3>Predicted vs actual (all regions, Day 1–10)</h3>
              <div className="kv">
                <div><div className="k">Verified bust region-days</div><div className="v">{verif.summary.n_bust_region_days}</div></div>
                <div><div className="k">…of which hidden (low spread)</div><div className="v">{verif.summary.n_hidden_bust_region_days}</div></div>
                <div><div className="k">Sentinel alerts / hits</div><div className="v">{verif.summary.sentinel_alerts} / {verif.summary.sentinel_hits}</div></div>
                <div><div className="k">B2 alerts / hits</div><div className="v">{verif.summary.b2_alerts} / {verif.summary.b2_hits}</div></div>
                <div><div className="k">Overlap (Jaccard) Sentinel</div><div className="v">{verif.summary.jaccard_sentinel.toFixed(2)}</div></div>
                <div><div className="k">Overlap (Jaccard) B2</div><div className="v">{verif.summary.jaccard_b2.toFixed(2)}</div></div>
              </div>
              <p className="muted">One case is an anecdote. Overall performance: see Verification &amp; analytics.</p>
            </div>
          )}
        </div>

        <div className="card">
          <h2>Forecaster review priority</h2>
          <div className="table-wrap">
            <table>
              <thead><tr><th>#</th><th>Region</th><th>Day</th><th>Bust risk</th><th>B2</th><th>Δ pp</th><th>Evidence</th></tr></thead>
              <tbody>
                {data.priority_queue.slice(0, 12).map((q, i) => {
                  const r = data.regions.find((x) => x.region_id === q.region_id)!;
                  const hit = vmap?.get(q.region_id)?.[q.lead_day - 1];
                  return (
                    <tr key={i} className={`clickable ${q.region_id === reg.region_id && q.lead_day === day ? "sel" : ""}`}
                      onClick={() => { setRegion(q.region_id); setDay(q.lead_day); }}>
                      <td>{i + 1}</td>
                      <td>{q.region_id} <span className="muted">{r.name}</span></td>
                      <td>D{q.lead_day}</td>
                      <td>{pct(q.p_bust)}</td>
                      <td>{pct(q.p_spread_baseline)}</td>
                      <td>{q.disagreement_pp > 0 ? "+" : ""}{q.disagreement_pp.toFixed(0)}</td>
                      <td><span className={`tag ${q.evidence.split(" ")[0]}`}>{q.evidence}</span>
                        {hit && <strong>{hit.bust ? " ✔ bust" : " ✘ no bust"}</strong>}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="muted">Ranking rule (transparent, not fitted): {String(data.priority_formula.formula)}</p>
        </div>
      </div>

      <div className="grid2" style={{ marginTop: 12 }}>
        <div className="card">
          <h2>WHEN — {reg.region_id} {reg.name} ({reg.lat.toFixed(1)}°, {reg.lon.toFixed(1)}°E)</h2>
          <div className="kv">
            <div><div className="k">Bust risk D{day}</div><div className="v" data-testid="p-bust">{pct(d.p_bust)}</div></div>
            <div><div className="k">Reliability confidence</div><div className="v">{pct(d.confidence)}</div></div>
            <div><div className="k">Spread baseline (B2)</div><div className="v">{pct(d.p_spread_baseline)}</div></div>
            <div><div className="k">Confidence disagreement</div><div className="v">{d.disagreement_pp > 0 ? "+" : ""}{d.disagreement_pp.toFixed(0)} pp</div></div>
            <div><div className="k">Peak-risk day</div><div className="v">D{reg.peak_risk_day}</div></div>
            <div><div className="k">Evidence strength</div><div className="v" style={{ fontSize: 13 }}>{d.evidence}</div></div>
            <div><div className="k">Historical support</div><div className="v" style={{ fontSize: 13 }}>{d.support}</div></div>
            <div><div className="k">Ensemble spread</div><div className="v">{d.spread_m.toFixed(1)} m</div></div>
          </div>
          <h3>Day 1–10 reliability trajectory (click a day)</h3>
          <TrajectoryChart days={reg.trajectory} threshold={data.alert_threshold.p_bust} verif={vr} selectedDay={day} onDay={setDay} />
          {vr && (<>
            <h3>Actual regional error relative to the bust threshold (ERA5 reference)</h3>
            <ErrorRatioChart verif={vr} />
          </>)}
        </div>

        <div className="card">
          <h2>WHY — evidence for {reg.region_id}, Day {day}</h2>
          <ul className="evidence">
            {d.explanation.evidence.map((e, i) => <li key={i}><strong>{e.kind}.</strong> {e.text}</li>)}
          </ul>
          <h3>Largest model attributions</h3>
          <div className="table-wrap">
            <table>
              <thead><tr><th>Feature</th><th>Value</th><th>Train pct.</th><th>Effect</th></tr></thead>
              <tbody>
                {d.explanation.drivers.map((x) => (
                  <tr key={x.feature}>
                    <td>{x.label}</td>
                    <td>{x.value == null ? "n/a" : x.value.toFixed(2)}</td>
                    <td>{x.train_percentile == null ? "—" : `${x.train_percentile.toFixed(0)}`}</td>
                    <td>{x.direction} ({x.contribution_logodds > 0 ? "+" : ""}{x.contribution_logodds.toFixed(2)})</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="muted">{d.explanation.attribution_note}</p>
          <h3>Historical failure signature among similar forecast states</h3>
          {sig.distribution ? (
            <>
              {Object.entries(sig.distribution).map(([k, v]) => (
                <BarRow key={k} label={SIG_LABEL[k]} value={v} fmt={(x) => `${(100 * x).toFixed(0)}%`}
                  color={vd && vd.bust && vd.signature === k ? "var(--verify)" : "var(--series-1)"} />
              ))}
              <p className="muted">Basis: {sig.basis} (n = {sig.n}). Evidence from past verified cases, not causal proof.</p>
            </>
          ) : <p className="muted">Historical support unavailable for this forecast state.</p>}
          {vd && (
            <p data-testid="fingerprint"><strong>Actual failure fingerprint:</strong> {vd.bust ? `${vd.signature_label} ` +
              `(phase share ${(100 * vd.phase_share).toFixed(0)}%, bias ${vd.bias_m.toFixed(1)} m, pattern r ${vd.pattern_corr.toFixed(2)})`
              : "no bust at this lead day"} — RMSE {vd.error_m.toFixed(1)} m, normalized {vd.normalized_error.toFixed(2)} vs Q90 {vd.threshold_q90.toFixed(2)}.</p>
          )}
          <h3>Nearest verified historical analogues</h3>
          <div className="table-wrap">
            <table>
              <thead><tr><th>Initialisation</th><th>Distance</th><th>Bust</th><th>Norm. error</th><th>Signature</th></tr></thead>
              <tbody>
                {d.analogues.map((a) => (
                  <tr key={a.case_id}><td>{a.init_time.slice(0, 13)}</td><td>{a.distance?.toFixed(2)}</td>
                    <td>{a.bust ? "yes" : "no"}</td><td>{a.normalized_error?.toFixed(2)}</td>
                    <td>{a.signature ? SIG_LABEL[a.signature] : "—"}</td></tr>
                ))}
                {d.analogues.length === 0 && <tr><td colSpan={5} className="muted">No verified analogues available.</td></tr>}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div className="card" style={{ marginTop: 12 }}>
        <div className="row" style={{ justifyContent: "space-between" }}>
          <h2>Forecast field, Day {day}</h2>
          <div className="row">
            <button className="btn" aria-pressed={field === "mean"} onClick={() => setField("mean")}>Ensemble-mean Z500</button>
            <button className="btn" aria-pressed={field === "spread"} onClick={() => setField("spread")}>Ensemble spread</button>
            {revealed && verif && <button className="btn" aria-pressed={field === "error"} onClick={() => setField("error")}>Forecast − ERA5</button>}
          </div>
        </div>
        <div style={{ maxWidth: 560 }}>
          {field === "mean" && <FieldMap lats={data.fields.lats} lons={data.fields.lons} values={data.fields.ens_mean_z500[day - 1]} label="Ensemble-mean Z500" />}
          {field === "spread" && <FieldMap lats={data.fields.lats} lons={data.fields.lons} values={data.fields.ens_spread_z500[day - 1]} label="Z500 ensemble spread" />}
          {field === "error" && revealed && verif && (
            <FieldMap lats={data.fields.lats} lons={data.fields.lons} diverging label="Ensemble-mean minus ERA5 Z500"
              values={data.fields.ens_mean_z500[day - 1].map((col, i) => col.map((v, j) => v - verif.era5_z500[day - 1][i][j]))} />
          )}
        </div>
      </div>
    </>
  );
}
