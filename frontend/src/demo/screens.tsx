import { useEffect, useMemo, useState } from "react";
import { Analytics } from "../components/Analytics";
import { FieldMap } from "../components/Maps";
import { demoApi } from "./api";
import type { Route, Selection } from "./DemoApp";
import type { Explanation, Fields, ModelSummary, RegionOverview, Reveal, RunResult } from "./types";
import {
  Badge, Bar, Card, DayControl, DivergingBar, evidenceKind, latlon, num, pct, pp, RegionMap, SIG_LABEL, Stat,
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

function SentinelB2Note({ run }: { run: RunResult }) {
  const same = run.model.selected_groups.length === 0;
  return (
    <div className="note" data-testid="sentinel-b2-note">
      {same
        ? <><strong>Validated Sentinel = B2 inputs.</strong> No candidate feature group beat the spread-only baseline on 2021 validation, so the
          Sentinel booster uses B2's six inputs and its probabilities equal B2's. Disagreement is therefore 0 — shown as measured, not adjusted.
          The added value demonstrated here is the reliability workflow: support/OOD, historical evidence, priority and verification memory.</>
        : <>Sentinel uses validated groups: {run.model.selected_groups.join(", ")}.</>}
    </div>
  );
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
              <Stat label="Mean P(bust)" value={pct(ls.mean_bust_probability)} sub="Sentinel" />
              <Stat label="Max P(bust)" value={pct(ls.max_bust_probability)} />
              <Stat label="Mean Sentinel − B2" value={pp(ls.mean_disagreement_pp)} />
            </div>
            <div className="support-row">
              {Object.entries(supportCounts).map(([k, v]) => <Badge key={k} kind={supportKind(k)}>{k}: {v}</Badge>)}
            </div>
          </Card>
          <Card title={`Priority regions · Day ${day}`} right={<span className="muted">transparent priority score</span>}>
            <table className="tbl">
              <thead><tr><th>Region</th><th>P(bust)</th><th>B2</th><th>Evidence</th><th>Score</th></tr></thead>
              <tbody>
                {top.map((r) => {
                  const c = r.days[day - 1];
                  return (
                    <tr key={r.region_id} className={`clickable ${sel.regionId === r.region_id ? "sel" : ""}`} data-testid={`prio-${r.region_id}`} onClick={() => focus(r.region_id, day, "reliability")}>
                      <td><strong>{r.region_id}</strong> <span className="muted">{latlon(r.lat, r.lon)}</span></td>
                      <td>{pct(c.bust_probability)}{c.alert && <span className="dot" title="alert" />}</td>
                      <td>{pct(c.b2_probability)}</td>
                      <td><Badge kind={evidenceKind(c.evidence_quality)}>{c.evidence_quality === "INSUFFICIENT HISTORICAL SUPPORT" ? "INSUFFICIENT" : c.evidence_quality}</Badge></td>
                      <td>{num(c.priority_score, 3)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </Card>
          <Card title="Domain risk by lead day">
            <div className="leadbars">
              {run.lead_summary.map((l) => (
                <button key={l.lead_day} className={`leadbar ${l.lead_day === day ? "on" : ""}`} onClick={() => setDay(l.lead_day)}>
                  <span>D{l.lead_day}</span><Bar value={l.max_bust_probability} max={0.5} color="var(--risk-4)" /><span className="muted">{pct(l.max_bust_probability, 0)} max · {l.n_alerts}</span>
                </button>
              ))}
            </div>
          </Card>
        </div>
      </div>
      <SentinelB2Note run={run} />
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
            <Stat label="Reliability confidence" value={<span data-testid="rel-conf">{pct(c.reliability_confidence)}</span>} sub="1 − P(bust)" />
            <Stat label="Sentinel" value={<span data-testid="rel-sentinel">{pct(c.bust_probability)}</span>} />
            <Stat label="B2 spread-only" value={<span data-testid="rel-b2">{pct(c.b2_probability)}</span>} />
            <Stat label="Disagreement" value={<span data-testid="rel-dis">{pp(c.disagreement_pp)}</span>} sub="Sentinel − B2" />
            <Stat label="Support / OOD" value={<Badge kind={supportKind(c.support_level)}>{c.support_level}</Badge>} sub={`Mahalanobis ${num(c.support_distance)}`} />
          </div>
          <div className="grid-main">
            <Card title="Day 1–10 reliability trajectory">
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
                <thead><tr><th>Day</th><th>Valid (UTC)</th><th>Sentinel</th><th>Confidence</th><th>B2</th><th>B0</th><th>Δ</th><th>Spread</th><th>Spread pct</th><th>Support</th><th>Evidence</th><th>Analogues</th></tr></thead>
                <tbody>
                  {data.trajectory.map((d) => (
                    <tr key={d.lead_day} className={`clickable ${d.lead_day === sel.day ? "sel" : ""}`} onClick={() => setDay(d.lead_day)}>
                      <td>D{d.lead_day}{d.alert && <span className="dot" />}</td><td>{d.valid_time.slice(0, 13)}</td>
                      <td><strong>{pct(d.bust_probability)}</strong></td><td>{pct(d.reliability_confidence)}</td><td>{pct(d.b2_probability)}</td><td>{pct(d.b0_probability)}</td>
                      <td>{pp(d.disagreement_pp)}</td><td>{num(d.spread_m, 1)} m</td><td>{pct(d.spread_pct, 0)}</td>
                      <td><Badge kind={supportKind(d.support_level)}>{d.support_level.replace(" SUPPORT", "").replace(" HISTORICAL", "")}</Badge></td>
                      <td><Badge kind={evidenceKind(d.evidence_quality)}>{d.evidence_quality.replace(" HISTORICAL SUPPORT", "")}</Badge></td>
                      <td>{d.analogues_within_radius ?? "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
          <SentinelB2Note run={run} />
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
            <Stat label="Sentinel risk" value={<span data-testid="why-sentinel">{pct(x.bust_probability)}</span>} />
            <Stat label="B2 risk" value={<span data-testid="why-b2">{pct(x.b2_probability)}</span>} />
            <Stat label="Disagreement" value={<span data-testid="why-dis">{pp(x.disagreement_pp)}</span>} />
            <Stat label="Support / OOD" value={<Badge kind={supportKind(x.support_level)} testid="why-support">{x.support_level}</Badge>} sub={`distance ${num(x.support_distance)}`} />
            <Stat label="Evidence quality" value={<Badge kind={evidenceKind(x.evidence_quality)} testid="why-evidence">{x.evidence_quality}</Badge>} />
          </div>
          <div className="grid-3">
            <Card title="Model attribution (TreeSHAP)" testid="attribution">
              <p className="muted small">Starting log-odds (training base) {num(x.attribution.bias_logodds, 2)}; contributions push the Sentinel log-odds up (red) or down (blue).</p>
              <div className="drivers">
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
              <h3>Forecast-state context <span className="muted">(computed features the validated model does not use)</span></h3>
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
          <SentinelB2Note run={run} />
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
            <thead><tr><th>#</th><th>Region</th><th>Day</th><th>Risk</th><th>B2</th><th>Δ</th><th>Support</th><th>Evidence</th><th>Analogues</th><th>Score</th><th></th></tr></thead>
            <tbody>{run.priority_queue.map((q, i) => {
              const r = run.regions.find((x) => x.region_id === q.region_id)!;
              return (
                <tr key={`${q.region_id}-${q.lead_day}`} className="clickable" onClick={() => focus(q.region_id, q.lead_day, "evidence")}>
                  <td>{i + 1}</td><td><strong>{q.region_id}</strong> <span className="muted">{latlon(r.lat, r.lon)}</span></td><td>D{q.lead_day}</td>
                  <td><strong>{pct(q.bust_probability)}</strong></td><td>{pct(q.b2_probability)}</td><td>{pp(q.disagreement_pp)}</td>
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
export function VerificationScreen({ run, sel, region, setDay, revealed, onReveal }: { run: RunResult; sel: Selection; region: RegionOverview; setDay: (d: number) => void; revealed: boolean; onReveal: () => void }) {
  const { data: x } = useLoad<Explanation>(() => demoApi.explain(run.case.case_id, region.region_id, sel.day), [run.case.case_id, region.region_id, sel.day]);
  const [v, setV] = useState<Reveal | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    setV(null);
    if (!revealed) return;
    demoApi.reveal(run.case.case_id, region.region_id, sel.day).then(setV).catch((e) => setErr(String(e.message ?? e)));
  }, [revealed, run.case.case_id, region.region_id, sel.day]);
  const busts = useMemo(() => v ? new Map(v.bust_map.filter((b) => b.lead_day === sel.day).map((b) => [b.region_id, b.bust])) : undefined, [v, sel.day]);
  const c = region.days[sel.day - 1];
  return (
    <div className="screen">
      <div className="screen-h">
        <div><h2>Verification — {region.region_id} · Day {sel.day}</h2><p className="muted">{latlon(region.lat, region.lon)} · valid {c.valid_time.slice(0, 16)} UTC · reference ERA5</p></div>
        <div className={`modeflag ${revealed ? "rev" : ""}`} data-testid="verif-mode">{revealed ? "VERIFICATION REVEALED" : "BLIND — forecast-time information only"}</div>
      </div>
      <DayControl day={sel.day} onDay={setDay} />
      <div className="grid-2">
        <Card title="Forecast (issued at initialisation)" testid="blind-forecast">
          <div className="stats">
            <Stat label="Predicted bust risk" value={<span data-testid="ver-p">{pct(c.bust_probability)}</span>} sub={c.alert ? "ALERT" : "no alert"} emphasis />
            <Stat label="B2 risk" value={pct(c.b2_probability)} />
            <Stat label="Evidence" value={<Badge kind={evidenceKind(c.evidence_quality)}>{c.evidence_quality}</Badge>} />
            <Stat label="Support" value={<Badge kind={supportKind(c.support_level)}>{c.support_level}</Badge>} />
          </div>
          <h3>Predicted failure signature</h3>
          {x?.failure_signature.distribution
            ? <p data-testid="ver-expected"><strong>{x.failure_signature.top_label}</strong> <span className="muted">({pct(x.failure_signature.distribution[x.failure_signature.top!], 0)} of {x.failure_signature.basis}, n = {x.failure_signature.n})</span></p>
            : <p className="muted">{x ? "INSUFFICIENT HISTORICAL SUPPORT" : "…"}</p>}
          {x && <p className="muted small">{x.evidence.find((e) => e.kind.startsWith("C."))?.text}</p>}
          {!revealed && <button className="btn reveal-btn" data-testid="reveal" onClick={onReveal}>REVEAL VERIFICATION</button>}
          {!revealed && <p className="muted small">Verification (ERA5 at the valid time) is not requested from the server until you click reveal.</p>}
        </Card>
        {revealed ? (
          <Card title="Verified outcome (ERA5)" testid="verified">
            {err && <p className="err">{err}</p>}
            {v && (
              <>
                <div className="stats">
                  <Stat label="Actual bust" value={<span data-testid="actual-bust" className={v.verification.actual_bust ? "crit-t" : ""}>{v.verification.actual_bust ? "BUST" : "NO BUST"}</span>} sub={v.verification.hidden_bust ? "hidden bust (low spread)" : undefined} emphasis />
                  <Stat label="Normalized error" value={<span data-testid="actual-err">{num(v.verification.normalized_error, 3)}</span>} sub={`bust threshold (train Q90) ${num(v.verification.threshold_q90, 3)}`} />
                  <Stat label="Regional RMSE" value={`${num(v.verification.error_m, 1)} m`} />
                </div>
                <h3>Failure fingerprint {v.verification.actual_bust ? "" : <span className="muted">(error decomposition of a non-bust day)</span>}</h3>
                <div className="fp" data-testid="fingerprint">
                  <div className="fp-main">{v.verification.failure_fingerprint.label}</div>
                  <div className="fp-parts">
                    <span>phase/pattern share of MSE <strong>{pct(v.verification.failure_fingerprint.phase_share, 0)}</strong></span>
                    <span>mean bias <strong>{num(v.verification.failure_fingerprint.bias_m, 1)} m</strong></span>
                    <span>pattern correlation <strong>{num(v.verification.failure_fingerprint.pattern_corr, 3)}</strong></span>
                  </div>
                </div>
                <h3>Expected vs actual</h3>
                <table className="tbl compact" data-testid="exp-vs-act"><tbody>
                  <tr><td>Predicted risk</td><td>{pct(v.forecast.bust_probability)} ({v.forecast.alert ? "alert" : "no alert"})</td><td>Outcome</td><td>{v.verification.actual_bust ? "bust" : "no bust"}</td></tr>
                  <tr><td>Expected signature</td><td>{v.comparison.expected_top ? SIG_LABEL[v.comparison.expected_top] : "—"}</td><td>Actual fingerprint</td><td>{v.verification.failure_fingerprint.label}</td></tr>
                  <tr><td colSpan={4} className="muted">{v.comparison.match == null ? v.comparison.note : v.comparison.match ? "Signature expectation MATCHED the verified bust." : `Signature expectation did NOT match (it gave the actual signature ${pct(v.comparison.p_expected_for_actual, 0)}).`}</td></tr>
                </tbody></table>
              </>
            )}
          </Card>
        ) : (
          <Card title="Verified outcome" className="locked"><p className="muted">Hidden until reveal.</p></Card>
        )}
      </div>
      {revealed && v && (
        <>
          <div className="grid-main">
            <Card title={`Verified busts across the domain · Day ${sel.day}`}>
              <RegionMap regions={run.regions} day={sel.day} selected={region.region_id} onSelect={() => undefined} busts={busts} />
              <p className="muted small" data-testid="case-summary">This case: {v.case_summary.verified_busts} verified bust region-days of {v.case_summary.region_days}; Sentinel issued {v.case_summary.sentinel_alerts} alerts, {v.case_summary.sentinel_hits} of them verified busts; {v.case_summary.hidden_busts} hidden busts.</p>
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
      <div className="screen-h"><div><h2>Model trust</h2><p className="muted">Genuine evaluation artifacts of the frozen models (2022 test year, scored once). Nothing here is recomputed for the demo.</p></div></div>
      {model && (
        <Card title="Model running in this demo" testid="model-card">
          <table className="tbl compact"><tbody>
            <tr><td>Model</td><td>{model.model_version} — {model.learner}</td></tr>
            <tr><td>Artifact</td><td>{model.model_artifact}; {model.exported_from}</td></tr>
            <tr><td>Retrained for demo</td><td>{model.retrained_for_demo ? "yes" : "no"}</td></tr>
            <tr><td>Training</td><td>{model.training_window}</td></tr>
            <tr><td>Calibration</td><td>{model.calibration_window} — {model.calibration}</td></tr>
            <tr><td>Sentinel inputs</td><td>{model.sentinel_features.join(", ")}</td></tr>
            <tr><td>B2 inputs</td><td>{model.b2_features.join(", ")}</td></tr>
            <tr><td>Validated extra groups</td><td>{model.selected_groups.length ? model.selected_groups.join(", ") : "none"} — {model.selection_note}</td></tr>
            <tr><td>Alert threshold</td><td>{pct(model.alert_threshold)} — {model.alert_threshold_definition}</td></tr>
            <tr><td>Target</td><td>{model.target}; ERA5 verification</td></tr>
            <tr><td>Pending</td><td>{model.pending}</td></tr>
            <tr><td>NCMRWF</td><td>Adapter interface only; no NCMRWF data ingested; no operational integration.</td></tr>
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
