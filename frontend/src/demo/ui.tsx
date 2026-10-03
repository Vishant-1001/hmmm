import { useState, type ReactNode } from "react";
import { Coast, Graticule, proj, riskClass, RISK_BINS, type Box } from "../components/Maps";
import type { Cell, RegionOverview, ForecastSource } from "./types";

export const pct = (v: number | null | undefined, d = 1) => (v == null || Number.isNaN(v) ? "—" : `${(100 * v).toFixed(d)}%`);
export const num = (v: number | null | undefined, d = 2) => (v == null || Number.isNaN(v) ? "—" : v.toFixed(d));
export const pp = (v: number) => `${v > 0 ? "+" : ""}${v.toFixed(1)} pp`;
export const latlon = (lat: number, lon: number) =>
  `${Math.abs(lat).toFixed(1)}°${lat < 0 ? "S" : "N"} ${lon.toFixed(1)}°E`;
export const SIG_LABEL: Record<string, string> = {
  POSITION_PHASE: "Position / phase mismatch",
  AMPLITUDE_STRUCTURE: "Amplitude / structure mismatch",
  RESIDUAL_MIXED: "Residual / mixed mismatch",
};
const HALF = 5.625 / 2;

export function Badge({ kind, children, testid }: { kind?: string; children: ReactNode; testid?: string }) {
  return <span className={`badge ${kind ?? ""}`} data-testid={testid}>{children}</span>;
}

const PROVIDER_LABELS: Record<string, string> = { ncmrwf_tigge: "NCMRWF TIGGE", ecmwf_research: "ECMWF IFS / ERA5" };

/** Forecast provenance, visible without developer tools. Synthetic data never gets a real-source label. */
export function SourceBadge({ source }: { source?: ForecastSource | null }) {
  if (!source) return null;
  if (source.synthetic || source.demo_only || source.provider === "synthetic") {
    return <span className="badge source synthetic" data-testid="source-badge" title={source.source_label}>DEMO MODE · SYNTHETIC SCENARIO</span>;
  }
  const label = PROVIDER_LABELS[source.provider] ?? source.provider;
  const title = `${source.dataset} · ${source.ensemble_member_count} members · init ${source.initialization_time.slice(0, 16)} UTC`;
  return <span className="badge source real" data-testid="source-badge" title={title}>SOURCE · {label}</span>;
}

export function supportKind(s: string) {
  return s.startsWith("NORMAL") ? "good" : s.startsWith("MODERATE") ? "info" : s.startsWith("WEAK") ? "warn" : "crit";
}
export function evidenceKind(s: string) {
  return s === "STRONG" ? "good" : s === "MODERATE" ? "info" : s === "WEAK" ? "warn" : "crit";
}

export function Stat({ label, value, sub, testid, emphasis }: { label: string; value: ReactNode; sub?: ReactNode; testid?: string; emphasis?: boolean }) {
  return (
    <div className={`stat ${emphasis ? "emph" : ""}`}>
      <div className="stat-k">{label}</div>
      <div className="stat-v" data-testid={testid}>{value}</div>
      {sub && <div className="stat-s">{sub}</div>}
    </div>
  );
}

export function Card({ title, right, children, className, testid }: { title?: ReactNode; right?: ReactNode; children: ReactNode; className?: string; testid?: string }) {
  return (
    <section className={`panel ${className ?? ""}`} data-testid={testid}>
      {(title || right) && <header className="panel-h"><h2>{title}</h2>{right}</header>}
      {children}
    </section>
  );
}

export function DayControl({ day, onDay, cells }: { day: number; onDay: (d: number) => void; cells?: { lead_day: number; n: number }[] }) {
  return (
    <div className="daybar" role="radiogroup" aria-label="Lead day">
      {Array.from({ length: 10 }, (_, i) => i + 1).map((d) => (
        <button key={d} role="radio" aria-checked={d === day} data-testid={`day-${d}`} className={d === day ? "on" : ""} onClick={() => onDay(d)}>
          <span>Day {d}</span>
          {cells && <small>{cells[d - 1]?.n ?? 0} alerts</small>}
        </button>
      ))}
    </div>
  );
}

export type MapMetric = "risk" | "verified";

export function RegionMap(props: {
  regions: RegionOverview[]; day: number; selected: string | null; onSelect: (id: string) => void;
  metric?: MapMetric; busts?: Map<string, number>; compact?: boolean;
}) {
  const b: Box = { lon0: 56, lon1: 107, lat0: -8.5, lat1: 42.5, w: 540 };
  const p = proj(b);
  const [hover, setHover] = useState<{ r: RegionOverview; x: number; y: number } | null>(null);
  const metric = props.metric ?? "risk";
  const fill = (c: Cell, rid: string) => {
    if (metric === "verified") return props.busts?.get(rid) ? "var(--risk-4)" : "var(--surface-2)";
    return `var(--risk-${riskClass(c.bust_probability)})`;
  };
  return (
    <div className="rel map-wrap">
      <svg viewBox={`0 0 ${b.w} ${p.h.toFixed(0)}`} width="100%" role="img" aria-label={`Regional map, Day ${props.day}`}>
        <rect width={b.w} height={p.h} fill="var(--surface-1)" />
        {props.regions.map((r) => {
          const c = r.days[props.day - 1];
          const x0 = p.x(r.lon - HALF), x1 = p.x(r.lon + HALF), y0 = p.y(r.lat + HALF), y1 = p.y(r.lat - HALF);
          return (
            <rect key={r.region_id} data-testid={`cell-${r.region_id}`} x={x0 + 1} y={y0 + 1} width={x1 - x0 - 2} height={y1 - y0 - 2} rx={2}
              fill={fill(c, r.region_id)} stroke={props.selected === r.region_id ? "var(--focus)" : "none"} strokeWidth={3}
              style={{ cursor: "pointer" }} onClick={() => props.onSelect(r.region_id)}
              onMouseMove={(e) => {
                const box = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect();
                setHover({ r, x: e.clientX - box.left + 14, y: e.clientY - box.top + 14 });
              }}
              onMouseLeave={() => setHover(null)} />
          );
        })}
        <Graticule p={p} b={b} />
        <Coast p={p} />
        {props.regions.map((r) => {
          const c = r.days[props.day - 1];
          const bust = props.busts?.get(r.region_id);
          return (
            <g key={`m${r.region_id}`} pointerEvents="none">
              {metric === "risk" && c.alert && <circle cx={p.x(r.lon)} cy={p.y(r.lat)} r={3.2} fill="var(--text-primary)" />}
              {metric !== "verified" && bust === 1 && (
                <rect x={p.x(r.lon - HALF) + 4} y={p.y(r.lat + HALF) + 4} width={p.x(r.lon + HALF) - p.x(r.lon - HALF) - 8}
                  height={p.y(r.lat - HALF) - p.y(r.lat + HALF) - 8} fill="none" stroke="var(--verify)" strokeWidth={2} />
              )}
            </g>
          );
        })}
      </svg>
      {hover && (() => {
        const c = hover.r.days[props.day - 1];
        return (
          <div className="tooltip" style={{ left: hover.x, top: hover.y }}>
            <strong>{hover.r.region_id}</strong> · {latlon(hover.r.lat, hover.r.lon)} · Day {props.day}<br />
            Bust probability {pct(c.bust_probability)} · {c.risk_level}<br />
            Climatology (B0) {pct(c.b0_probability)} · spread {num(c.spread_m, 1)} m ({pct(c.spread_pct, 0)} pct)<br />
            {c.support_level} · evidence quality {c.evidence_quality}
          </div>
        );
      })()}
      {!props.compact && (
        <div className="legend">
          {metric === "risk" && RISK_BINS.slice(0, -1).map((lo, i) => (
            <span key={i}><span className="chip" style={{ background: `var(--risk-${i})` }} />{`${Math.round(lo * 100)}–${Math.min(100, Math.round(RISK_BINS[i + 1] * 100))}%`}</span>
          ))}
          {metric === "risk" && <span>● alert (≥ validation 10%-FAR threshold)</span>}
          {metric === "verified" && <span><span className="chip" style={{ background: "var(--risk-4)" }} />verified bust (ERA5)</span>}
          {metric !== "verified" && props.busts && <span>▢ verified bust (ERA5)</span>}
        </div>
      )}
    </div>
  );
}

const W = 560, H = 230, M = { l: 44, r: 14, t: 18, b: 30 };
const xs = (d: number) => M.l + ((d - 1) / 9) * (W - M.l - M.r);

export function Trajectory(props: { days: Cell[]; threshold: number; day: number; onDay: (d: number) => void; busts?: boolean[] }) {
  const ymax = Math.max(0.03, props.threshold * 1.4, ...props.days.map((d) => Math.max(d.bust_probability, d.b0_probability) * 1.2));
  const ys = (v: number) => H - M.b - (v / ymax) * (H - M.t - M.b);
  const step = ymax > 0.3 ? 0.1 : ymax > 0.1 ? 0.05 : ymax > 0.05 ? 0.01 : 0.005;
  const ticks = Array.from({ length: Math.floor(ymax / step) + 1 }, (_, i) => +(i * step).toFixed(3));
  const series = [
    { key: "b0_probability", label: "B0 climatological bust rate (baseline)", color: "var(--text-muted)", dash: "5 4", w: 1.5 },
    { key: "bust_probability", label: "B2 bust probability (served model)", color: "var(--series-1)", dash: undefined, w: 2.5 },
  ] as const;
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Day 1-10 bust-probability trajectory">
        <rect x={Math.max(M.l, xs(props.day) - 14)} y={M.t} width={28 - Math.max(0, M.l - (xs(props.day) - 14))} height={H - M.t - M.b} fill="var(--surface-2)" />
        {ticks.map((t) => (
          <g key={t}>
            <line x1={M.l} x2={W - M.r} y1={ys(t)} y2={ys(t)} stroke="var(--grid)" />
            <text x={M.l - 6} y={ys(t) + 4} textAnchor="end">{(t * 100).toFixed(step < 0.01 ? 1 : 0)}%</text>
          </g>
        ))}
        <line x1={M.l} x2={W - M.r} y1={ys(props.threshold)} y2={ys(props.threshold)} stroke="var(--text-muted)" strokeDasharray="4 4" />
        <text x={W - M.r} y={ys(props.threshold) - 5} textAnchor="end">alert threshold {pct(props.threshold)}</text>
        {props.days.map((d) => (
          <text key={d.lead_day} x={xs(d.lead_day)} y={H - 10} textAnchor="middle" style={{ fontWeight: d.lead_day === props.day ? 700 : 400, cursor: "pointer" }}
            onClick={() => props.onDay(d.lead_day)}>D{d.lead_day}</text>
        ))}
        {props.busts?.map((b, i) => b ? (
          <rect key={i} x={xs(i + 1) - 5} y={4} width={10} height={10} rx={2} fill="var(--verify)"><title>{`Day ${i + 1}: verified bust (ERA5)`}</title></rect>
        ) : null)}
        {series.map((s) => (
          <g key={s.key}>
            <polyline fill="none" stroke={s.color} strokeWidth={s.w} strokeDasharray={s.dash} strokeLinejoin="round" strokeLinecap="round"
              points={props.days.map((d) => `${xs(d.lead_day)},${ys(d[s.key])}`).join(" ")} />
            {s.key === "bust_probability" && props.days.map((d) => (
              <circle key={d.lead_day} cx={xs(d.lead_day)} cy={ys(d[s.key])} r={d.lead_day === props.day ? 5.5 : 3.5} fill={s.color}
                stroke="var(--surface-1)" strokeWidth={1.5} style={{ cursor: "pointer" }} onClick={() => props.onDay(d.lead_day)}>
                <title>{`Day ${d.lead_day}: bust probability ${pct(d.bust_probability)} · ${d.risk_level}`}</title>
              </circle>
            ))}
          </g>
        ))}
      </svg>
      <div className="legend">
        {series.slice().reverse().map((s) => (
          <span key={s.key}><svg width="22" height="8"><line x1="0" x2="22" y1="4" y2="4" stroke={s.color} strokeWidth={Math.min(s.w, 6)} strokeDasharray={s.dash} /></svg>{s.label}</span>
        ))}
        {props.busts && <span>■ verified bust day (ERA5)</span>}
      </div>
    </div>
  );
}

export function Bar({ value, max = 1, color = "var(--series-1)" }: { value: number; max?: number; color?: string }) {
  return <div className="bar-track"><div className="bar-fill" style={{ width: `${Math.max(0, Math.min(1, value / max)) * 100}%`, background: color }} /></div>;
}

export function DivergingBar({ value, max }: { value: number; max: number }) {
  const w = Math.min(1, Math.abs(value) / (max || 1)) * 50;
  return (
    <div className="dbar">
      <div className="dbar-mid" />
      <div className="dbar-fill" style={value >= 0
        ? { left: "50%", width: `${w}%`, background: "var(--risk-4)" }
        : { right: "50%", width: `${w}%`, background: "var(--series-1)" }} />
    </div>
  );
}
