import { useState } from "react";
import coast from "../data/coastline.json";
import type { Region, VerifDay } from "../types";

// V4 pattern-aware bust is a ~1% event (training rate 1.3%; validation 10%-FAR alert threshold 2.05%)
export const RISK_BINS = [0, 0.01, 0.02, 0.03, 0.05, 0.08, 1.0001];
export function riskClass(p: number): number {
  for (let i = 0; i < RISK_BINS.length - 1; i++) if (p < RISK_BINS[i + 1]) return i;
  return RISK_BINS.length - 2;
}

export interface Box { lon0: number; lon1: number; lat0: number; lat1: number; w: number; }
export function proj(b: Box) {
  const h = (b.w * (b.lat1 - b.lat0)) / (b.lon1 - b.lon0) / Math.cos((((b.lat0 + b.lat1) / 2) * Math.PI) / 180);
  return {
    h,
    x: (lon: number) => ((lon - b.lon0) / (b.lon1 - b.lon0)) * b.w,
    y: (lat: number) => ((b.lat1 - lat) / (b.lat1 - b.lat0)) * h,
  };
}

export function Coast({ p }: { p: ReturnType<typeof proj> }) {
  return (
    <g fill="none" stroke="var(--text-muted)" strokeWidth={0.8} pointerEvents="none">
      {(coast as { lines: number[][][] }).lines.map((l, i) => (
        <polyline key={i} points={l.map(([lo, la]) => `${p.x(lo).toFixed(1)},${p.y(la).toFixed(1)}`).join(" ")} />
      ))}
    </g>
  );
}

export function Graticule({ p, b }: { p: ReturnType<typeof proj>; b: Box }) {
  const lons = [];
  for (let lo = Math.ceil(b.lon0 / 10) * 10; lo <= b.lon1; lo += 10) lons.push(lo);
  const lats = [];
  for (let la = Math.ceil(b.lat0 / 10) * 10; la <= b.lat1; la += 10) lats.push(la);
  return (
    <g>
      {lons.map((lo) => (
        <g key={`x${lo}`}>
          <line x1={p.x(lo)} x2={p.x(lo)} y1={0} y2={p.h} stroke="var(--grid)" strokeWidth={0.5} />
          <text x={p.x(lo) + 2} y={p.h - 3}>{lo}°E</text>
        </g>
      ))}
      {lats.map((la) => (
        <g key={`y${la}`}>
          <line x1={0} x2={b.w} y1={p.y(la)} y2={p.y(la)} stroke="var(--grid)" strokeWidth={0.5} />
          <text x={2} y={p.y(la) - 2}>{la < 0 ? `${-la}°S` : `${la}°N`}</text>
        </g>
      ))}
    </g>
  );
}

export function RiskMap(props: {
  regions: Region[];
  day: number;
  selected: string | null;
  onSelect: (id: string) => void;
  threshold: number;
  verification?: Map<string, VerifDay[]>;
}) {
  const b: Box = { lon0: 55, lon1: 110, lat0: -10, lat1: 45, w: 560 };
  const p = proj(b);
  const [hover, setHover] = useState<{ r: Region; x: number; y: number } | null>(null);
  return (
    <div className="rel">
      <svg viewBox={`0 0 ${b.w} ${p.h.toFixed(0)}`} width="100%" role="img"
        aria-label={`Bust risk map for Day ${props.day}`}>
        <rect width={b.w} height={p.h} fill="var(--surface-1)" />
        {props.regions.map((r) => {
          const d = r.trajectory[props.day - 1];
          const x0 = p.x(r.lon_bounds[0]), x1 = p.x(r.lon_bounds[1]);
          const y0 = p.y(r.lat_bounds[1]), y1 = p.y(r.lat_bounds[0]);
          return (
            <rect key={r.region_id} data-testid={`cell-${r.region_id}`} x={x0 + 1} y={y0 + 1}
              width={x1 - x0 - 2} height={y1 - y0 - 2} rx={2}
              fill={`var(--risk-${riskClass(d.p_bust)})`}
              stroke={props.selected === r.region_id ? "var(--focus)" : "none"} strokeWidth={2.5}
              style={{ cursor: "pointer" }}
              onClick={() => props.onSelect(r.region_id)}
              onMouseMove={(e) => {
                const box = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect();
                setHover({ r, x: e.clientX - box.left + 12, y: e.clientY - box.top + 12 });
              }}
              onMouseLeave={() => setHover(null)}>
              <title>{`${r.region_id} ${r.name}: bust risk ${(100 * d.p_bust).toFixed(0)}%`}</title>
            </rect>
          );
        })}
        <Graticule p={p} b={b} />
        <Coast p={p} />
        {props.regions.map((r) => {
          const d = r.trajectory[props.day - 1];
          const cx = p.x(r.lon), cy = p.y(r.lat);
          const v = props.verification?.get(r.region_id)?.[props.day - 1];
          return (
            <g key={`m${r.region_id}`} pointerEvents="none">
              {d.alert && <circle cx={cx} cy={cy} r={3.5} fill="var(--text-primary)" />}
              {v?.bust === 1 && (
                <rect x={p.x(r.lon_bounds[0]) + 4} y={p.y(r.lat_bounds[1]) + 4}
                  width={p.x(r.lon_bounds[1]) - p.x(r.lon_bounds[0]) - 8}
                  height={p.y(r.lat_bounds[0]) - p.y(r.lat_bounds[1]) - 8}
                  fill="none" stroke="var(--verify)" strokeWidth={2} strokeDasharray={v.hidden_bust ? "4 2" : undefined} />
              )}
            </g>
          );
        })}
      </svg>
      {hover && (() => {
        const d = hover.r.trajectory[props.day - 1];
        const v = props.verification?.get(hover.r.region_id)?.[props.day - 1];
        return (
          <div className="tooltip" style={{ left: hover.x, top: hover.y }}>
            <strong>{hover.r.region_id}</strong> {hover.r.name}<br />
            {hover.r.lat.toFixed(1)}°, {hover.r.lon.toFixed(1)}°E · Day {props.day}<br />
            Bust risk <strong>{(100 * d.p_bust).toFixed(0)}%</strong> · spread baseline {(100 * d.p_spread_baseline).toFixed(0)}%<br />
            Evidence: {d.evidence}
            {v && <><br />Verified: normalized error {v.normalized_error.toFixed(2)} (Q90 {v.threshold_q90.toFixed(2)}) → {v.bust ? "BUST" : "no bust"}</>}
          </div>
        );
      })()}
      <div className="legend" style={{ marginTop: 6 }}>
        {RISK_BINS.slice(0, -1).map((lo, i) => (
          <span key={i}><span className="swatch" style={{ height: 10, background: `var(--risk-${i})` }} />
            {`${Math.round(lo * 100)}–${Math.min(100, Math.round(RISK_BINS[i + 1] * 100))}%`}</span>
        ))}
        <span>● alert (p ≥ {(100 * props.threshold).toFixed(0)}%, product threshold)</span>
        {props.verification && <span>▢ verified bust (dashed = hidden bust)</span>}
      </div>
    </div>
  );
}

export function FieldMap(props: {
  lats: number[]; lons: number[]; values: number[][]; label: string; diverging?: boolean;
}) {
  const b: Box = { lon0: props.lons[0] - 2.8, lon1: props.lons[props.lons.length - 1] + 2.8,
    lat0: props.lats[0] - 2.8, lat1: props.lats[props.lats.length - 1] + 2.8, w: 460 };
  const p = proj(b);
  const flat = props.values.flat();
  const lo = props.diverging ? -Math.max(...flat.map(Math.abs)) : Math.min(...flat);
  const hi = props.diverging ? -lo : Math.max(...flat);
  const color = (v: number) => {
    const t = (v - lo) / (hi - lo || 1);
    if (props.diverging) {
      const a = Math.abs(t - 0.5) * 2;
      return t >= 0.5 ? `color-mix(in oklab, var(--surface-2), #c24f25 ${a * 100}%)`
        : `color-mix(in oklab, var(--surface-2), #2a78d6 ${a * 100}%)`;
    }
    return `color-mix(in oklab, #e8f0fb, #104281 ${t * 100}%)`;
  };
  const dlo = props.lons[1] - props.lons[0], dla = props.lats[1] - props.lats[0];
  return (
    <div>
      <svg viewBox={`0 0 ${b.w} ${p.h.toFixed(0)}`} width="100%" role="img" aria-label={props.label}>
        {props.lons.map((lon, i) => props.lats.map((lat, j) => (
          <rect key={`${i}-${j}`} x={p.x(lon - dlo / 2)} y={p.y(lat + dla / 2)} width={p.x(lon + dlo / 2) - p.x(lon - dlo / 2) + 0.5}
            height={p.y(lat - dla / 2) - p.y(lat + dla / 2) + 0.5} fill={color(props.values[i][j])}>
            <title>{`${lat.toFixed(1)}°, ${lon.toFixed(1)}°E: ${props.values[i][j].toFixed(1)} m`}</title>
          </rect>
        )))}
        <Coast p={p} />
        <rect x={p.x(59.06)} y={p.y(39.375)} width={p.x(104.06) - p.x(59.06)} height={p.y(-5.625) - p.y(39.375)}
          fill="none" stroke="var(--text-primary)" strokeWidth={1} />
      </svg>
      <div className="muted">{props.label}: {lo.toFixed(0)} … {hi.toFixed(0)} m (box = target domain)</div>
    </div>
  );
}
