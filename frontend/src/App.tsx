import { useEffect, useState } from "react";
import { api } from "./api";
import { Analytics } from "./components/Analytics";
import { CaseView } from "./components/CaseView";
import type { CaseIndexItem } from "./types";

type Tab = "case" | "analytics" | "about";

export default function App() {
  const [tab, setTab] = useState<Tab>("case");
  const [cases, setCases] = useState<CaseIndexItem[] | null>(null);
  const [mode, setMode] = useState<string>("");
  const [err, setErr] = useState<string | null>(null);
  const [prov, setProv] = useState<any>(null);

  useEffect(() => {
    api.cases().then((c) => { setCases(c.cases); setMode(c.mode); }).catch((e) => setErr(String(e.message ?? e)));
  }, []);
  useEffect(() => { if (tab === "about" && !prov) api.provenance().then(setProv).catch(() => setProv({ error: true })); }, [tab, prov]);

  return (
    <>
      <header className="top">
        <div>
          <h1>Forecast Bust Sentinel</h1>
          <div className="sub">Where and when is the existing NWP forecast likely to fail? · SIH26079 research prototype</div>
        </div>
        <nav className="tabs" role="tablist">
          {([["case", "Forecast case & blind replay"], ["analytics", "Verification & analytics"], ["about", "Provenance & limitations"]] as [Tab, string][]).map(([k, l]) => (
            <button key={k} role="tab" aria-selected={tab === k} onClick={() => setTab(k)}>{l}</button>
          ))}
        </nav>
      </header>
      <div className="banner" data-testid="mode-banner">
        <strong>Historical research replay</strong> — precomputed real ECMWF IFS ENS forecasts from the 2022 test period
        (WeatherBench 2). Not a live feed; NCMRWF integration is architected but not claimed.
        {mode && <span className="muted" data-testid="api-mode"> Server: {mode}.</span>}
      </div>
      <main>
        {err && <div className="card err">{err}</div>}
        {tab === "case" && cases && cases.length > 0 && <CaseView cases={cases} />}
        {tab === "case" && cases && cases.length === 0 && <div className="card">No replay cases available (NOT YET COMPUTED).</div>}
        {tab === "analytics" && <Analytics />}
        {tab === "about" && <About prov={prov} />}
      </main>
    </>
  );
}

function About({ prov }: { prov: any }) {
  return (
    <div className="card">
      <h2>What this system is — and is not</h2>
      <ul className="evidence">
        <li>It estimates the probability that an <em>existing</em> NWP ensemble-mean Z500 forecast exceeds a project-defined
          large-error threshold (TRAIN Q90 of normalized regional RMSE per region × lead day × season). It does not forecast weather.</li>
        <li>Z500 is the first validated target variable; this does not mean the system handles rainfall, cyclones or heat waves.</li>
        <li>ERA5 is the verification reference analysis, not perfect truth.</li>
        <li>Regions are native 5.625° × 5.625° grid boxes (bandwidth-limited WeatherBench 2 64×32 product), not the 5° × 5° cells of the design specification — and not local forecasts.</li>
        <li>Historical failure signatures are evidence from similar past forecast states, not causal proof.</li>
        <li>Relationships learned on ECMWF IFS ENS 2018–2022 need re-learning for NCMRWF NEPS or after model upgrades.</li>
      </ul>
      <h3>Provenance</h3>
      {!prov && <p className="muted">Loading…</p>}
      {prov?.error && <p className="err">DATA UNAVAILABLE</p>}
      {prov && !prov.error && <pre style={{ whiteSpace: "pre-wrap", fontSize: 11.5, maxHeight: 480, overflow: "auto" }}>{JSON.stringify(prov, null, 1)}</pre>}
    </div>
  );
}
