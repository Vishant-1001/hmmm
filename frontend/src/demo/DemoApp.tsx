import { useCallback, useEffect, useMemo, useState } from "react";
import { demoApi } from "./api";
import { Badge, latlon } from "./ui";
import { EvidenceScreen, OverviewScreen, PriorityScreen, ReliabilityScreen, TrustScreen, VerificationScreen } from "./screens";
import type { CaseInfo, RunResult } from "./types";

export const ROUTES = [
  ["overview", "Overview", "Where should I look?"],
  ["reliability", "Reliability", "Region × Day 1–10"],
  ["evidence", "Evidence", "Why flagged"],
  ["priority", "Priority", "Forecaster review queue"],
  ["verification", "Verification", "Reveal & fingerprint"],
  ["trust", "Trust", "Model evaluation"],
] as const;
export type Route = (typeof ROUTES)[number][0];

function routeFromHash(): Route {
  const h = (typeof window !== "undefined" ? window.location.hash : "").replace(/^#\/?/, "").split("?")[0];
  return (ROUTES.find((r) => r[0] === h)?.[0] ?? "overview") as Route;
}

export interface Selection { caseId: string; regionId: string | null; day: number }

export default function DemoApp() {
  const [route, setRoute] = useState<Route>(routeFromHash());
  const [cases, setCases] = useState<CaseInfo[] | null>(null);
  const [mode, setMode] = useState("");
  const [sel, setSel] = useState<Selection | null>(null);
  const [run, setRun] = useState<RunResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [revealed, setRevealed] = useState<Set<string>>(new Set());
  const [attempt, setAttempt] = useState(0);
  const retry = () => { setErr(null); setRun(null); setAttempt((n) => n + 1); };

  useEffect(() => {
    const on = () => setRoute(routeFromHash());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  const go = useCallback((r: Route) => { window.location.hash = `/${r}`; setRoute(r); }, []);

  useEffect(() => {
    demoApi.cases().then((c) => {
      setCases(c.cases); setMode(c.mode);
      const first = c.cases.find((x) => x.selection === "random") ?? c.cases[0];
      if (first) setSel({ caseId: first.case_id, regionId: null, day: 3 });
    }).catch((e) => setErr(String(e.message ?? e)));
  }, [attempt]);

  useEffect(() => {
    if (!sel?.caseId) return;
    if (run?.case.case_id === sel.caseId) return;
    setLoading(true); setErr(null);
    demoApi.run(sel.caseId).then((r) => {
      setRun(r);
      setSel((s) => s && s.caseId === r.case.case_id && !s.regionId
        ? { ...s, regionId: r.priority_queue[0]?.region_id ?? r.regions[0].region_id, day: r.priority_queue[0]?.lead_day ?? s.day } : s);
    }).catch((e) => setErr(String(e.message ?? e))).finally(() => setLoading(false));
  }, [sel?.caseId, attempt]); // eslint-disable-line react-hooks/exhaustive-deps

  // Re-selecting the current case must not clear the run: the run effect only fires on a case change.
  const selectCase = (id: string) => { if (id === sel?.caseId) return; setRun(null); setSel({ caseId: id, regionId: null, day: 3 }); };
  const setRegion = (regionId: string) => setSel((s) => s && { ...s, regionId });
  const setDay = (day: number) => setSel((s) => s && { ...s, day });
  const focus = (regionId: string, day: number, r?: Route) => { setSel((s) => s && { ...s, regionId, day }); if (r) go(r); };
  const region = useMemo(() => run?.regions.find((r) => r.region_id === sel?.regionId) ?? null, [run, sel?.regionId]);
  const caseInfo = cases?.find((c) => c.case_id === sel?.caseId);
  const isRevealed = !!sel && revealed.has(sel.caseId);

  return (
    <div className="shell">
      <aside className="side">
        <div className="brand">
          <div className="logo" aria-hidden>FBS</div>
          <div><div className="brand-name">Forecast Bust Sentinel</div><div className="brand-sub">SIH26079</div></div>
        </div>
        <nav aria-label="Screens">
          {ROUTES.map(([k, label, sub], i) => (
            <a key={k} href={`#/${k}`} data-testid={`nav-${k}`} className={route === k ? "on" : ""} onClick={(e) => { e.preventDefault(); go(k); }}>
              <span className="nav-i">{i + 1}</span><span><span className="nav-l">{label}</span><span className="nav-s">{sub}</span></span>
            </a>
          ))}
        </nav>
        <div className="side-foot">
          <div className="loop">FORECAST → RELIABILITY → EVIDENCE → VERIFICATION → MEMORY</div>
          {run && <div className="muted" data-testid="inference-time">Models executed live: {run.timings_ms.total.toFixed(0)} ms for {run.regions.length} regions × 10 days</div>}
          {run && <div className="muted">Model {run.model.model_version}</div>}
        </div>
      </aside>
      <div className="main-col">
        <header className="topbar">
          <div className="tb-left">
            <h1>Forecast Bust Sentinel</h1>
            <Badge kind="replay" testid="replay-badge">HISTORICAL REPLAY</Badge>
            <Badge kind="proto">RESEARCH PROTOTYPE</Badge>
          </div>
          <div className="tb-right">
            <label className="case-sel">
              <span>Forecast case</span>
              <select data-testid="case-select" value={sel?.caseId ?? ""} onChange={(e) => selectCase(e.target.value)} disabled={!cases}>
                {cases?.map((c) => (
                  <option key={c.case_id} value={c.case_id}>{c.init_time.slice(0, 16)} UTC · {c.season}{c.selection === "stress" ? " · stress case" : ""}</option>
                ))}
              </select>
            </label>
            {caseInfo && <div className="init" data-testid="case-init">Init <strong>{caseInfo.init_time.slice(0, 16)} UTC</strong><br /><span className="muted">valid {caseInfo.valid_range[0].slice(0, 10)} → {caseInfo.valid_range[1].slice(0, 10)}</span></div>}
          </div>
        </header>
        <div className="context-strip">
          {caseInfo && <span><strong>{caseInfo.demo_role}.</strong> ECMWF IFS ENS (50 members, WeatherBench 2, 5.625°) Z500, 2022 test year, verified against ERA5.</span>}
          {region && sel && <span className="sel-pill" data-testid="selection-pill">Selected: <strong>{region.region_id}</strong> ({latlon(region.lat, region.lon)}) · Day {sel.day}</span>}
          {mode && <span className="muted">{mode}</span>}
        </div>
        <main className="content">
          {err && <div className="panel err" data-testid="error">{err} <button className="btn" data-testid="retry" onClick={retry}>Retry</button></div>}
          {!err && (loading || !run || !sel) && route !== "trust" && <div className="panel muted" data-testid="loading">Running Sentinel and B2 on the stored forecast state… <span className="muted">(first request after idle can take up to a minute while the demo server wakes)</span></div>}
          {run && sel && !loading && (
            <>
              {route === "overview" && <OverviewScreen run={run} sel={sel} setDay={setDay} focus={focus} />}
              {route === "reliability" && region && <ReliabilityScreen run={run} sel={sel} region={region} setDay={setDay} setRegion={setRegion} go={go} />}
              {route === "evidence" && region && <EvidenceScreen run={run} sel={sel} region={region} setDay={setDay} go={go} />}
              {route === "priority" && <PriorityScreen run={run} focus={focus} />}
              {route === "verification" && region && (
                <VerificationScreen run={run} sel={sel} region={region} setDay={setDay} revealed={isRevealed}
                  onReveal={() => setRevealed((s) => new Set(s).add(sel.caseId))} />
              )}
            </>
          )}
          {route === "trust" && <TrustScreen model={run?.model ?? null} />}
        </main>
      </div>
    </div>
  );
}
