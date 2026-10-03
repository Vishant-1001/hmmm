import { useEffect, useMemo, useState } from "react";
import { Analytics } from "../components/Analytics";
import { FieldMap } from "../components/Maps";
import { demoApi } from "./api";
import type { Route, Selection } from "./DemoApp";
import type { Explanation, Fields, ModelSummary, RegionOverview, Reveal, RunResult } from "./types";
import {
  Badge, Bar, Card, DayControl, DivergingBar, evidenceKind, latlon, num, pct, RegionMap, SIG_LABEL, Stat,
  supportKind, Trajectory,
} from "./ui";

function useLoad<T>(fn: () => Promise<T>, deps: unknown[]) {
  const [data, setData] = useState<T | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    setData(null); setErr(null);
    fn().then((d) => live && setData(d)).catch((e) => live && setErr(String(e.message ?? e)));
    return () => { live = false; };
  }, deps); // eslint-disable-line react-hooks/exhaustive-deps
  return { data, err };
}

function ModelNote({ run }: { run: RunResult }) {
  return (
    <div className="note" data-testid="model-note">
      <strong>What the number means.</strong> Bust probability = the estimated chance that this region-day's Z500 forecast error
      (ensemble mean vs ERA5, normalized) exceeds the project bust threshold, the training 90th percentile for that region, lead day
      and season. It is not the chance of a weather event, and it is not an overall forecast-quality score. About{" "}
      {pct(run.model.base_rate_bust, 0)} of training region-days were busts; ALERT means at or above {pct(run.alert_threshold, 1)} (10%
      false-alarm rate on validation). Served model: <strong>{run.model.model_id}</strong> ({run.model.model_version}, calibration{" "}
      {run.model.calibration_version}). Evidence quality and analogue evidence come from verified historical forecast states and are
      not model inputs.
    </div>
  );
}

const LEVEL_KIND: Record<string, string> = { HIGH: "crit", ELEVATED: "warn", NORMAL: "good", INSUFFICIENT: "proto" };
function Level({ v, testid }: { v: string; testid?: string }) {
  return <Badge kind={LEVEL_KIND[v] ?? "proto"} testid={testid}>{v}</Badge>;
}
const RISK_KIND: Record<string, string> = { ALERT: "crit", "ABOVE CLIMATOLOGY": "warn", "AT OR BELOW CLIMATOLOGY": "good" };
function Risk({ v, testid }: { v: string; testid?: string }) {
  return <Badge kind={RISK_KIND[v] ?? "proto"} testid={testid}>{v}</Badge>;
}

/* ------------------------------------------------------------------ OVERVIEW */
export function OverviewScreen({ run, sel, setDay, focus }: { run: RunResult; sel: Selection; setDay: (d: number) => void; focus: (r: string, d: number, route?: Route) => void }) {
  const day = sel.day;
  const ls = run.lead_summary[day - 1];
  const cells = run.regions.map((r) => r.days[day - 1]);
  const supportCounts = countBy(cells.map((c) => c.support_level));
  const top = [...run.regions].sort((a, b) => b.days[day - 1].priority_score - a.days[day - 1].priority_score).slice(0, 8);
  return (
    <div className="screen">
      <div className="screen-h">
        <div><h2>Overview — where should I look?</h2><p className="muted">Regional bust probability for the {run.regions.length} target regions (5.625° boxes, 5°S–40°N, 60°E–105°E), Day 1–10, from one real initialisation.</p></div>
      </div>
      <DayControl day={day} onDay={setDay} cells={run.lead_summary.map((l) => ({ lead_day: l.lead_day, n: l.n_alerts }))} />
      <div className="grid-main">
        <Card title={`Bust-risk map · Day ${day}`} right={<span className="muted">click a region</span>}>
          <RegionMap regions={run.regions} day={day} selected={sel.regionId} onSelect={(id) => focus(id, day, "reliability")} />
        </Card>
        <div className="stack">
          <Card title={`Reliability summary · Day ${day}`} testid="overview-summary">
            <div className="stats">
              <Stat label="Regions on alert" value={<span data-testid="n-alerts">{ls.n_alerts}</span>} sub={`of ${run.regions.length}`} emphasis />
              <Stat label="Mean bust probability" value={pct(ls.mean_bust_probability, 1)} sub="B2, calibrated" />
              <Stat label="Max bust probability" value={pct(ls.max_bust_probability, 1)} />
            </div>
            <div className="support-row">
              {Object.entries(supportCounts).map(([k, v]) => <Badge key={k} kind={supportKind(k)}>{k}: {v}</Badge>)}
            </div>
          </Card>
          <Card title={`Priority regions · Day ${day}`} right={<span className="muted">transparent priority score</span>}>
            <div className="table-wrap">
            <table className="tbl">
              <thead><tr><th>Region</th><th>Bust probability</th><th>Climatology (B0)</th><th>Risk level</th><th>Evidence quality</th><th>Score</th></tr></thead>
              <tbody>
                {top.map((r) => {
                  const c = r.days[day - 1];
                  return (
                    <tr key={r.region_id} className={`clickable ${sel.regionId === r.region_id ? "sel" : ""}`} data-testid={`prio-${r.region_id}`} onClick={() => focus(r.region_id, day, "reliability")}>
                      <td><strong>{r.region_id}</strong> <span className="muted">{latlon(r.lat, r.lon)}</span></td>
                      <td>{pct(c.bust_probability)}{c.alert && <span className="dot" title="alert" />}</td>
                      <td className="muted">{pct(c.b0_probability)}</td>
                      <td><Risk v={c.risk_level} /></td>
                      <td><Badge kind={evidenceKind(c.evidence_quality)}>{c.evidence_quality === "INSUFFICIENT HISTORICAL SUPPORT" ? "INSUFFICIENT" : c.evidence_quality}</Badge></td>
                      <td>{num(c.priority_score, 3)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            </div>
          </Card>
          <Card title="Domain risk by lead day">
            <div className="leadbars">
              {run.lead_summary.map((l) => (
                <button key={l.lead_day} className={`leadbar ${l.lead_day === day ? "on" : ""}`} onClick={() => setDay(l.lead_day)}>
                  <span>D{l.lead_day}</span><Bar value={l.max_bust_probability} max={Math.max(0.3, ...run.lead_summary.map((q) => q.max_bust_probability))} color="var(--risk-4)" /><span className="muted">{pct(l.max_bust_probability, 1)} max · {l.n_alerts}</span>
                </button>
              ))}
            </div>
          </Card>
        </div>
      </div>
      <ModelNote run={run} />
    </div>
  );
}

/* ------------------------------------------------------------------ RELIABILITY */
export function ReliabilityScreen({ run, sel, region, setDay, setRegion, go }: { run: RunResult; sel: Selection; region: RegionOverview; setDay: (d: number) => void; setRegion: (r: string) => void; go: (r: Route) => void }) {
  const { data, err } = useLoad(() => demoApi.region(run.case.case_id, region.region_id), [run.case.case_id, region.region_id]);
  const sorted = useMemo(() => [...run.regions].sort((a, b) => b.peak_bust_probability - a.peak_bust_probability), [run]);
  const c = data?.trajectory[sel.day - 1];
  return (
    <div className="screen">
      <div className="screen-h">
        <div><h2>Reliability — {region.region_id} <span className="muted">{latlon(region.lat, region.lon)}</span></h2>
          <p className="muted">How reliable is the existing NWP Z500 forecast for this region at each lead day?</p></div>
        <label className="case-sel"><span>Region</span>
          <select data-testid="region-select" value={region.region_id} onChange={(e) => setRegion(e.target.value)}>
            {sorted.map((r) => <option key={r.region_id} value={r.region_id}>{r.region_id} · {latlon(r.lat, r.lon)} · peak {pct(r.peak_bust_probability, 0)} (D{r.peak_lead_day})</option>)}
          </select>
        </label>
      </div>
      <DayControl day={sel.day} onDay={setDay} />
      {err && <div className="panel err">{err}</div>}
      {data && c && (
        <>
          <div className="stats wide" data-testid="reliability-stats">
            <Stat label={`Bust probability · Day ${sel.day}`} value={<span data-testid="rel-p">{pct(c.bust_probability)}</span>} sub={c.alert ? "ALERT (≥ threshold)" : "below alert threshold"} emphasis />
            <Stat label="Baseline: climatology (B0)" value={<span data-testid="rel-b0">{pct(c.b0_probability)}</span>} sub="same region / lead / season" />
            <Stat label="Risk level" value={<Risk v={c.risk_level} testid="rel-risk" />} />
            <Stat label="Ensemble spread" value={<span data-testid="rel-spread">{num(c.spread_m, 1)} m</span>} sub={`${pct(c.spread_pct, 0)} of training spread · ${num(c.spread_thr_ratio, 2)}× threshold`} />
            <Stat label="Analogue evidence" value={<Level v={c.analogue_evidence} testid="rel-analogue" />} sub="similar verified states" />
            <Stat label="Evidence quality" value={<Badge kind={evidenceKind(c.evidence_quality)} testid="rel-evidence">{c.evidence_quality}</Badge>} sub={c.support_level === c.evidence_quality ? "historical support" : c.support_level} />
          </div>
          <div className="grid-main">
            <Card title="Day 1–10 bust probability (B2) vs climatology (B0)">
              <Trajectory days={data.trajectory} threshold={data.alert_threshold} day={sel.day} onDay={setDay} />
            </Card>
            <Card title="Locate region">
              <RegionMap regions={run.regions} day={sel.day} selected={region.region_id} onSelect={setRegion} compact />
              <button className="btn primary block" data-testid="go-evidence" onClick={() => go("evidence")}>Why flagged? Open evidence for Day {sel.day} →</button>
            </Card>
          </div>
          <Card title="Lead-day table">
            <div className="table-wrap">
              <table className="tbl" data-testid="traj-table">
                <thead><tr><th>Day</th><th>Valid (UTC)</th><th>Bust probability</th><th>B0</th><th>Risk level</th><th>Spread</th><th>Spread pct</th><th>Analogue evidence</th><th>Support</th><th>Evidence quality</th><th>Analogues</th></tr></thead>
                <tbody>
                  {data.trajectory.map((d) => (
                    <tr key={d.lead_day} className={`clickable ${d.lead_day === sel.day ? "sel" : ""}`} onClick={() => setDay(d.lead_day)}>
                      <td>D{d.lead_day}{d.alert && <span className="dot" />}</td><td>{d.valid_time.slice(0, 13)}</td>
                      <td><strong>{pct(d.bust_probability, 1)}</strong></td><td className="muted">{pct(d.b0_probability, 1)}</td><td><Risk v={d.risk_level} /></td>
                      <td>{num(d.spread_m, 1)} m</td><td>{pct(d.spread_pct, 0)}</td><td><Level v={d.analogue_evidence} /></td>
                      <td><Badge kind={supportKind(d.support_level)}>{d.support_level.replace(" SUPPORT", "").replace(" HISTORICAL", "")}</Badge></td>
                      <td><Badge kind={evidenceKind(d.evidence_quality)}>{d.evidence_quality.replace(" HISTORICAL SUPPORT", "")}</Badge></td>
                      <td>{d.analogues_within_radius ?? "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
          <ModelNote run={run} />
        </>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ EVIDENCE (hero) */
export function EvidenceScreen({ run, sel, region, setDay, go }: { run: RunResult; sel: Selection; region: RegionOverview; setDay: (d: number) => void; go: (r: Route) => void }) {
  const { data: x, err } = useLoad<Explanation>(() => demoApi.explain(run.case.case_id, region.region_id, sel.day), [run.case.case_id, region.region_id, sel.day]);
  const { data: fields } = useLoad<Fields>(() => demoApi.fields(run.case.case_id), [run.case.case_id]);
  return (
    <div className="screen">
      <div className="screen-h">
        <div><h2>Why flagged? — {region.region_id} · Day {sel.day}</h2>
          <p className="muted">{latlon(region.lat, region.lon)} · every number below is computed by the model run and the historical memory for this forecast state.</p></div>
      </div>
      <DayControl day={sel.day} onDay={setDay} />
      {err && <div className="panel err">{err}</div>}
      {!x && !err && <div className="panel muted">Computing explanation…</div>}
      {x && (
        <>
          <div className="stats wide" data-testid="why-stats">
            <Stat label="Bust probability" value={<span data-testid="why-p">{pct(x.bust_probability)}</span>} sub={x.alert ? "ALERT" : "below alert threshold"} emphasis />
            <Stat label="Baseline: climatology (B0)" value={<span data-testid="why-b0">{pct(x.b0_probability)}</span>} sub={`raw model ${pct(x.raw_probability, 1)}`} />
            <Stat label="Risk level" value={<Risk v={x.risk_level} testid="why-risk" />} />
            <Stat label="Analogue evidence" value={<Level v={x.analogue_evidence} testid="why-analogue" />} sub={x.analogue_bust_rate == null ? "" : `${pct(x.analogue_bust_rate, 0)} of analogues busted (normal ${pct(run.model.base_rate_bust, 0)})`} />
            <Stat label="Support / OOD" value={<Badge kind={supportKind(x.support_level)} testid="why-support">{x.support_level}</Badge>} sub={`distance ${num(x.support_distance)}`} />
            <Stat label="Evidence quality" value={<Badge kind={evidenceKind(x.evidence_quality)} testid="why-evidence">{x.evidence_quality}</Badge>} />
          </div>
          <div className="grid-3">
            <Card title="Why this risk? (forecast-time information only)" testid="interpretation">
              <ul className="evlist" data-testid="why-list">{x.interpretation.map((t) => <li key={t}>{t}</li>)}</ul>
              <h3>Model attribution (TreeSHAP)</h3>
              <p className="muted small">Starting log-odds {num(x.attribution.bias_logodds, 2)}; contributions push the B2 log-odds up (red) or down (blue).</p>
              <div className="drivers" data-testid="attribution">
                {x.attribution.drivers.map((d) => {
                  const mx = Math.max(...x.attribution.drivers.map((q) => Math.abs(q.contribution_logodds)));
                  return (
                    <div key={d.feature} className="driver" data-testid={`driver-${d.feature}`}>
                      <div className="driver-l"><strong>{d.label}</strong><span className="muted"> = {fmtVal(d.feature, d.value)}{d.train_percentile != null ? ` · ${d.train_percentile.toFixed(0)}th train pct` : ""}</span></div>
                      <DivergingBar value={d.contribution_logodds} max={mx} />
                      <div className="driver-v">{d.contribution_logodds > 0 ? "+" : ""}{d.contribution_logodds.toFixed(3)}</div>
                    </div>
                  );
                })}
              </div>
              <p className="muted small">{x.attribution.note}</p>
            </Card>
            <Card title="Historical forecast-state evidence" testid="analogues">
              <div className="stats">
                <Stat label="Verified cases available" value={<span data-testid="an-avail">{x.analogue_summary.verified_cases_available ?? "—"}</span>} />
                <Stat label="Within similarity radius" value={<span data-testid="an-within">{x.analogue_summary.within_radius ?? "—"}</span>} />
                <Stat label={`Bust rate, ${x.analogue_summary.k} nearest`} value={<span data-testid="an-rate">{pct(x.analogue_summary.bust_rate, 0)}</span>} sub={`climatology ${pct(x.analogue_summary.climatological_bust_rate, 0)}`} />
                <Stat label="Median norm. error" value={num(x.analogue_summary.median_normalized_error)} sub={`90th pct ${num(x.analogue_summary.q90_normalized_error)}`} />
              </div>
              <table className="tbl compact">
                <thead><tr><th>Nearest analogue (init)</th><th>Dist.</th><th>Outcome</th><th>Norm. err</th></tr></thead>
                <tbody>{x.analogue_summary.nearest.map((a) => (
                  <tr key={a.case_id}><td>{a.init_time.slice(0, 13)} <span className="muted">{a.split}</span></td><td>{num(a.distance)}</td>
                    <td>{a.bust ? <Badge kind="crit">BUST · {SIG_LABEL[a.signature ?? ""]?.split(" ")[0] ?? ""}</Badge> : <span className="muted">no bust</span>}</td><td>{num(a.normalized_error)}</td></tr>
                ))}</tbody>
              </table>
              <p className="muted small">Rule: {x.analogue_summary.rule}. Standardised PC1–8, spread, Z500 anomaly, P10–P90 range; radius {num(x.analogue_summary.radius)}.</p>
            </Card>
            <Card title="Expected failure signature" testid="expected-signature">
              {x.failure_signature.distribution ? (
                <>
                  <p>If this forecast busts, similar past busts suggest: <strong data-testid="exp-sig">{x.failure_signature.top_label}</strong></p>
                  {Object.entries(x.failure_signature.distribution).map(([k, v]) => (
                    <div key={k} className="sigrow"><span>{SIG_LABEL[k]}</span><Bar value={v} color={k === x.failure_signature.top ? "var(--risk-4)" : "var(--neutral)"} /><span>{pct(v, 0)}</span></div>
                  ))}
                  <p className="muted small">Basis: {x.failure_signature.basis} (n = {x.failure_signature.n}). Historical evidence among similar forecast states, not causal proof.</p>
                </>
              ) : <p className="muted">INSUFFICIENT HISTORICAL SUPPORT — too few verified analogues.</p>}
            </Card>
          </div>
          <div className="grid-main">
            <Card title="Ensemble and atmospheric evidence" testid="evidence-text">
              <ul className="evlist">
                {x.evidence.map((e) => <li key={e.kind}><span className="evk">{e.kind}</span>{e.text}</li>)}
              </ul>
              <h3>Forecast-state context</h3>
              <table className="tbl compact">
                <tbody>{x.context_features.map((f) => (
                  <tr key={f.feature}><td>{f.label}</td><td>{fmtVal(f.feature, f.value)}</td><td className="muted">{f.train_percentile != null ? `${f.train_percentile.toFixed(0)}th train pct` : "—"}</td></tr>
                ))}</tbody>
              </table>
            </Card>
            <Card title={`Ensemble Z500 spread · Day ${sel.day}`}>
              {fields ? <FieldMap lats={fields.lats} lons={fields.lons} values={fields.ens_spread_z500[sel.day - 1]} label="IFS ENS Z500 spread (std of 50 members)" /> : <p className="muted">Loading field…</p>}
              <button className="btn primary block" data-testid="go-verification" onClick={() => go("verification")}>Go to verification →</button>
            </Card>
          </div>
          <ModelNote run={run} />
        </>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ PRIORITY */
export function PriorityScreen({ run, focus }: { run: RunResult; focus: (r: string, d: number, route?: Route) => void }) {
  const f = run.priority_formula;
  return (
    <div className="screen">
      <div className="screen-h"><div><h2>Forecaster review priority</h2><p className="muted">Top 20 region-days across all lead days, ranked by the project's transparent priority rule.</p></div></div>
      <Card title="Priority rule">
        <code className="formula">{f.formula}</code>
        <p className="muted small">H = {f.H_days} days, B = {f.B}, evidence weight {Object.entries(f.evidence_weight).map(([k, v]) => `${k} ${v}`).join(" · ")}. {f.note}</p>
      </Card>
      <Card title="Review queue">
        <div className="table-wrap">
          <table className="tbl" data-testid="priority-table">
            <thead><tr><th>#</th><th>Region</th><th>Day</th><th>Bust probability</th><th>B0</th><th>Risk level</th><th>Support</th><th>Evidence quality</th><th>Analogues</th><th>Score</th><th></th></tr></thead>
            <tbody>{run.priority_queue.map((q, i) => {
              const r = run.regions.find((x) => x.region_id === q.region_id)!;
              return (
                <tr key={`${q.region_id}-${q.lead_day}`} className="clickable" onClick={() => focus(q.region_id, q.lead_day, "evidence")}>
                  <td>{i + 1}</td><td><strong>{q.region_id}</strong> <span className="muted">{latlon(r.lat, r.lon)}</span></td><td>D{q.lead_day}</td>
                  <td><strong>{pct(q.bust_probability)}</strong></td><td className="muted">{pct(q.b0_probability)}</td><td><Risk v={q.risk_level} /></td>
                  <td><Badge kind={supportKind(q.support_level)}>{q.support_level.replace(" SUPPORT", "")}</Badge></td>
                  <td><Badge kind={evidenceKind(q.evidence_quality)}>{q.evidence_quality.replace(" HISTORICAL SUPPORT", "")}</Badge></td>
                  <td>{q.analogues_within_radius ?? "—"}</td><td>{num(q.priority_score, 3)}</td><td className="muted">why →</td>
                </tr>
              );
            })}</tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}

/* ------------------------------------------------------------------ VERIFICATION */
export function VerificationScreen({ run, sel, region, setDay, revealed, onReveal, onReset }: { run: RunResult; sel: Selection; region: RegionOverview; setDay: (d: number) => void; revealed: boolean; onReveal: () => void; onReset: () => void }) {
  const { data: x } = useLoad<Explanation>(() => demoApi.explain(run.case.case_id, region.region_id, sel.day), [run.case.case_id, region.region_id, sel.day]);
  const [v, setV] = useState<Reveal | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [tries, setTries] = useState(0);
  useEffect(() => {
    let live = true;
    setV(null); setErr(null);
    if (!revealed) return;
    demoApi.reveal(run.case.case_id, region.region_id, sel.day).then((d) => live && setV(d)).catch((e) => live && setErr(String(e.message ?? e)));
    return () => { live = false; };
  }, [revealed, run.case.case_id, region.region_id, sel.day, tries]);
  const busts = useMemo(() => v ? new Map(v.bust_map.filter((b) => b.lead_day === sel.day).map((b) => [b.region_id, b.bust])) : undefined, [v, sel.day]);
  const c = region.days[sel.day - 1];
  return (
    <div className="screen">
      <div className="screen-h">
        <div><h2>Blind replay — {region.region_id} · Day {sel.day}</h2><p className="muted">{latlon(region.lat, region.lon)} · valid {c.valid_time.slice(0, 16)} UTC · reference ERA5</p></div>
        <div className={`modeflag ${revealed ? "rev" : ""}`} data-testid="verif-mode">{revealed ? "POST-VERIFICATION — ERA5 REVEALED" : "BLIND — forecast-time information only"}</div>
      </div>
      <DayControl day={sel.day} onDay={setDay} />
      <div className="grid-2">
        <Card title="Prediction (issued at initialisation, without verification)" testid="blind-forecast">
          <div className="stats">
            <Stat label="Bust probability (B2)" value={<span data-testid="ver-p">{pct(c.bust_probability)}</span>} sub={c.alert ? "ALERT" : "no alert"} emphasis />
            <Stat label="Climatology (B0)" value={pct(c.b0_probability)} />
            <Stat label="Risk level" value={<Risk v={c.risk_level} />} />
            <Stat label="Evidence quality" value={<Badge kind={evidenceKind(c.evidence_quality)}>{c.evidence_quality.replace(" HISTORICAL SUPPORT", "")}</Badge>} sub={c.support_level === c.evidence_quality ? "historical support" : c.support_level} />
          </div>
          <h3>Expected failure signature (from verified analogues)</h3>
          {x?.failure_signature.distribution
            ? <p data-testid="ver-expected"><strong>{x.failure_signature.top_label}</strong> <span className="muted">({pct(x.failure_signature.distribution[x.failure_signature.top!], 0)} of {x.failure_signature.basis}, n = {x.failure_signature.n})</span></p>
            : <p className="muted">{x ? "INSUFFICIENT HISTORICAL SUPPORT" : "…"}</p>}
          {x && <p className="muted small">{x.evidence.find((e) => e.kind.startsWith("C."))?.text}</p>}
          {!revealed && <button className="btn reveal-btn" data-testid="reveal" onClick={onReveal}>REVEAL VERIFICATION</button>}
          {!revealed && <p className="muted small">The ERA5 outcome is not requested from the server until you click reveal.</p>}
          {revealed && <button className="btn" data-testid="reset" onClick={onReset}>RESET — hide verification and replay again</button>}
        </Card>
        {revealed ? (
          <Card title="POST-VERIFICATION · Verified outcome (ERA5)" testid="verified">
            {err && <p className="err" data-testid="reveal-error">{err} <button className="btn" data-testid="reveal-retry" onClick={() => setTries((n) => n + 1)}>Retry</button></p>}
            {!v && !err && <p className="muted">Loading verification…</p>}
            {v && (
              <>
                <div className="stats">
                  <Stat label="Actual outcome" value={<span data-testid="actual-bust" className={v.verification.actual_bust ? "crit-t" : ""}>{v.verification.actual_bust ? "BUST" : "NO BUST"}</span>} sub={v.verification.hidden_bust ? "hidden bust (low spread)" : "project bust threshold"} emphasis />
                  <Stat label="Normalized error" value={<span data-testid="actual-err">{num(v.verification.normalized_error, 3)}</span>} sub={`bust threshold (train Q90) ${num(v.verification.threshold_q90, 3)}`} />
                  <Stat label="Regional RMSE vs ERA5" value={<span data-testid="actual-rmse">{num(v.verification.error_m, 1)} m</span>} />
                </div>
                <h3>Failure fingerprint <span className="muted">{v.verification.actual_bust ? "" : "(error decomposition of a non-bust day)"}</span></h3>
                <div className="fp" data-testid="fingerprint">
                  <div className="fp-main">{v.verification.failure_fingerprint.label}</div>
                  <div className="fp-parts">
                    <span>phase/pattern share of MSE <strong>{pct(v.verification.failure_fingerprint.phase_share, 0)}</strong></span>
                    <span>mean bias <strong>{num(v.verification.failure_fingerprint.bias_m, 1)} m</strong></span>
                    <span>pattern correlation <strong>{num(v.verification.failure_fingerprint.pattern_corr, 3)}</strong></span>
                  </div>
                </div>
                <h3>Prediction vs outcome</h3>
                <table className="tbl compact" data-testid="exp-vs-act"><tbody>
                  <tr><td>Predicted</td><td>{pct(v.forecast.bust_probability)} ({v.forecast.alert ? "ALERT" : "no alert"}; B0 {pct(v.forecast.b0_probability)})</td><td>Outcome</td><td>{v.verification.actual_bust ? "BUST" : "no bust"}</td></tr>
                  <tr><td>Expected signature</td><td>{v.comparison.expected_top ? SIG_LABEL[v.comparison.expected_top] : "—"}</td><td>Actual fingerprint</td><td>{v.verification.failure_fingerprint.label}</td></tr>
                  <tr><td colSpan={4} className="muted">{v.comparison.match == null ? v.comparison.note : v.comparison.match ? "Signature expectation MATCHED the verified bust." : `Signature expectation did NOT match (it gave the actual signature ${pct(v.comparison.p_expected_for_actual, 0)}).`}</td></tr>
                </tbody></table>
                <p className="muted small">One region-day is a single outcome; probabilities are judged over many cases (see Trust).</p>
              </>
            )}
          </Card>
        ) : (
          <Card title="Verified outcome" className="locked" testid="verified-locked"><p className="muted">Hidden until reveal: ERA5 error, bust truth and fingerprint are not loaded.</p></Card>
        )}
      </div>
      {revealed && v && (
        <>
          <div className="grid-main">
            <Card title={`POST-VERIFICATION · Verified busts across the domain · Day ${sel.day}`}>
              <RegionMap regions={run.regions} day={sel.day} selected={region.region_id} onSelect={() => undefined} busts={busts} />
              <p className="muted small" data-testid="case-summary">This case: {v.case_summary.verified_busts} verified bust region-days of {v.case_summary.region_days}; B2 issued {v.case_summary.model_alerts} alerts, {v.case_summary.model_hits} of them verified busts; {v.case_summary.hidden_busts} hidden busts.</p>
            </Card>
            <Card title="Forecast vs verification, Day 1–10">
              <Trajectory days={region.days} threshold={run.alert_threshold} day={sel.day} onDay={setDay} busts={v.trajectory.map((t) => t.actual_bust)} />
            </Card>
          </div>
          <Card title="Memory update" testid="memory-update">
            <div className="loopbar">
              {["FORECAST", "RELIABILITY", "EVIDENCE", "VERIFICATION", "MEMORY"].map((s, i) => <span key={s} className={i === 4 ? "on" : ""}>{s}</span>)}
            </div>
            <p>Verified outcome for <strong>{v.memory_update.entry.region_id} Day {v.memory_update.entry.lead_day}</strong> (valid {v.memory_update.entry.valid_time.slice(0, 13)}):
              {" "}{v.memory_update.entry.bust ? "BUST" : "no bust"}, normalized error {num(v.memory_update.entry.normalized_error, 3)}, fingerprint {SIG_LABEL[v.memory_update.entry.signature]}.</p>
            <p>Verified analogue candidates for this region and lead day: <strong data-testid="mem-before">{v.memory_update.verified_cases_before}</strong> at initialisation → <strong data-testid="mem-after">{v.memory_update.verified_cases_after}</strong> for initialisations from {v.memory_update.available_from.slice(0, 13)} UTC. All {v.memory_update.entries_added} region-days of this case join the memory the same way.</p>
            <p className="muted small">{v.memory_update.note}</p>
          </Card>
        </>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ TRUST */
export function TrustScreen({ model }: { model: ModelSummary | null }) {
  return (
    <div className="screen">
      <div className="screen-h"><div><h2>Model trust</h2><p className="muted">The ONE model running in this demo and its genuine evaluation artifacts. Nothing here is recomputed for the demo.</p></div></div>
      {model && (
        <Card title="Model running in this demo" testid="model-card">
          <table className="tbl compact"><tbody>
            <tr><td>Model id</td><td data-testid="model-id">{model.model_id}</td></tr>
            <tr><td>Version</td><td>{model.model_version}</td></tr>
            <tr><td>Estimator</td><td data-testid="model-type">{model.model_type} — {model.estimator}</td></tr>
            <tr><td>Provider / data</td><td>{model.provider} · {model.dataset_mode}</td></tr>
            <tr><td>Inputs</td><td>{model.features.join(", ")} (forecast-time only; no verification)</td></tr>
            <tr><td>Predicts</td><td>{model.probability_meaning}</td></tr>
            <tr><td>Bust definition</td><td>{model.target}</td></tr>
            <tr><td>Artifact</td><td>{model.model_artifact}; {model.exported_from}</td></tr>
            <tr><td>Retrained for demo</td><td>{model.retrained_for_demo ? "yes" : "no"}</td></tr>
            <tr><td>Training</td><td>{model.training_window}</td></tr>
            <tr><td>Calibration</td><td>{model.calibration_version}: {model.calibration_window} — {model.calibration}</td></tr>
            <tr><td>Hyper-parameters</td><td>{Object.entries(model.params).map(([k, v]) => `${k} ${v}`).join(" · ")} (locked v2 configuration)</td></tr>
            <tr><td>Alert threshold</td><td>{pct(model.alert_threshold)} — {model.alert_threshold_definition}</td></tr>
            <tr><td>Risk level</td><td>{model.risk_level_definition}</td></tr>
            <tr><td>Most important inputs (gain)</td><td>{model.model_level_importance.slice(0, 4).map((m) => m.label).join(", ")}</td></tr>
            <tr><td>Not served</td><td>The v2 Sentinel (did not beat B2), V3 quantile boosting, BMA and V4 pattern-aware model are archived experiments</td></tr>
            <tr><td>NCMRWF</td><td data-testid="ncmrwf-status">NCMRWF / TIGGE: provider integration implemented; catalogue availability confirmed; authenticated retrieval pending (no NCMRWF data in this demo)</td></tr>
          </tbody></table>
        </Card>
      )}
      <Analytics />
    </div>
  );
}

function countBy(xs: string[]) {
  const o: Record<string, number> = {};
  for (const x of xs) o[x] = (o[x] ?? 0) + 1;
  return o;
}

function fmtVal(f: string, v: number | null) {
  if (v == null) return "—";
  if (f === "spread_pct") return `${(100 * v).toFixed(0)}%`;
  if (f === "m_sign_agree") return `${(100 * v).toFixed(0)}%`;
  if (["lead_day", "init_hour", "region_code", "season_code"].includes(f)) return v.toFixed(0);
  return Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2);
}
