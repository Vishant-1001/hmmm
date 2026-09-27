import { useState } from "react";
import type { Day, VerifDay } from "../types";

const W = 520, H = 220, M = { l: 40, r: 12, t: 16, b: 28 };
const xs = (d: number) => M.l + ((d - 1) / 9) * (W - M.l - M.r);

export function TrajectoryChart(props: { days: Day[]; threshold: number; verif?: VerifDay[]; selectedDay: number;
  onDay: (d: number) => void; }) {
  const ymax = Math.max(0.5, ...props.days.map((d) => Math.max(d.p_bust, d.p_spread_baseline))) * 1.05;
  const ys = (v: number) => H - M.b - (v / ymax) * (H - M.t - M.b);
  const [hover, setHover] = useState<number | null>(null);
  const series = [
    { key: "p_bust", label: "Sentinel bust risk", color: "var(--series-1)" },
    { key: "p_spread_baseline", label: "Spread baseline (B2)", color: "var(--series-2)" },
    { key: "p_climatology", label: "Climatology (B0)", color: "var(--neutral)" },
  ] as const;
  const ticks = [0, 0.25, 0.5, 0.75, 1].filter((t) => t <= ymax);
  const hd = hover != null ? props.days[hover - 1] : null;
  return (
    <div className="rel">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Day 1-10 reliability trajectory"
        onMouseLeave={() => setHover(null)}
        onMouseMove={(e) => {
          const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
          const x = ((e.clientX - r.left) / r.width) * W;
          setHover(Math.min(10, Math.max(1, Math.round(1 + ((x - M.l) / (W - M.l - M.r)) * 9))));
        }}
        onClick={() => hover && props.onDay(hover)}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={M.l} x2={W - M.r} y1={ys(t)} y2={ys(t)} stroke="var(--grid)" strokeWidth={1} />
            <text x={M.l - 6} y={ys(t) + 4} textAnchor="end">{Math.round(t * 100)}%</text>
          </g>
        ))}
        {props.days.map((d) => (
          <text key={d.lead_day} x={xs(d.lead_day)} y={H - 8} textAnchor="middle"
            style={{ fontWeight: d.lead_day === props.selectedDay ? 700 : 400 }}>D{d.lead_day}</text>
        ))}
        <line x1={M.l} x2={W - M.r} y1={ys(props.threshold)} y2={ys(props.threshold)} stroke="var(--text-muted)"
          strokeDasharray="5 4" strokeWidth={1} />
        <text x={W - M.r} y={ys(props.threshold) - 4} textAnchor="end">alert threshold</text>
        <line x1={xs(props.selectedDay)} x2={xs(props.selectedDay)} y1={M.t} y2={H - M.b} stroke="var(--grid)" strokeWidth={6} opacity={0.6} />
        {props.verif?.map((v) => v.bust ? (
          <rect key={v.lead_day} x={xs(v.lead_day) - 5} y={M.t - 12} width={10} height={10} rx={2}
            fill="var(--verify)"><title>{`Day ${v.lead_day}: verified bust`}</title></rect>
        ) : null)}
        {series.map((s) => (
          <g key={s.key}>
            <polyline fill="none" stroke={s.color} strokeWidth={2} strokeDasharray={s.key === "p_climatology" ? "2 3" : undefined}
              points={props.days.map((d) => `${xs(d.lead_day)},${ys(d[s.key])}`).join(" ")} />
            {s.key !== "p_climatology" && props.days.map((d) => (
              <circle key={d.lead_day} cx={xs(d.lead_day)} cy={ys(d[s.key])} r={4} fill={s.color}
                stroke="var(--surface-1)" strokeWidth={2} />
            ))}
          </g>
        ))}
        {hover && <line x1={xs(hover)} x2={xs(hover)} y1={M.t} y2={H - M.b} stroke="var(--text-muted)" strokeWidth={1} />}
      </svg>
      {hd && (
        <div className="tooltip" style={{ left: `${(xs(hd.lead_day) / W) * 100}%`, top: 10 }}>
          <strong>Day {hd.lead_day}</strong> (valid {hd.valid_time.slice(0, 13)})<br />
          Sentinel {(100 * hd.p_bust).toFixed(0)}% · B2 {(100 * hd.p_spread_baseline).toFixed(0)}% · B0 {(100 * hd.p_climatology).toFixed(0)}%<br />
          Disagreement {hd.disagreement_pp > 0 ? "+" : ""}{hd.disagreement_pp.toFixed(0)} pp · {hd.evidence}
          {props.verif && <><br />Verified: {props.verif[hd.lead_day - 1].bust ? "BUST" : "no bust"} (norm. error {props.verif[hd.lead_day - 1].normalized_error.toFixed(2)})</>}
        </div>
      )}
      <div className="legend">
        {series.map((s) => (
          <span key={s.key}><span className="swatch" style={{ background: s.color }} />{s.label}</span>
        ))}
        {props.verif && <span>■ verified bust day</span>}
      </div>
    </div>
  );
}

export function ErrorRatioChart({ verif }: { verif: VerifDay[] }) {
  const r = verif.map((v) => v.normalized_error / v.threshold_q90);
  const ymax = Math.max(1.5, ...r) * 1.05;
  const h = 150;
  const ys = (v: number) => h - M.b - (v / ymax) * (h - M.t - M.b);
  const bw = ((W - M.l - M.r) / 10) * 0.6;
  return (
    <svg viewBox={`0 0 ${W} ${h}`} width="100%" role="img" aria-label="Actual error relative to bust threshold">
      {[0, 1].map((t) => (
        <g key={t}>
          <line x1={M.l} x2={W - M.r} y1={ys(t)} y2={ys(t)} stroke={t === 1 ? "var(--text-muted)" : "var(--grid)"}
            strokeDasharray={t === 1 ? "5 4" : undefined} />
          <text x={M.l - 6} y={ys(t) + 4} textAnchor="end">{t === 1 ? "Q90" : "0"}</text>
        </g>
      ))}
      {verif.map((v, i) => (
        <g key={v.lead_day}>
          <rect x={xs(v.lead_day) - bw / 2} y={ys(r[i])} width={bw} height={ys(0) - ys(r[i])} rx={3}
            fill={v.bust ? "var(--verify)" : "var(--neutral)"}>
            <title>{`Day ${v.lead_day}: RMSE ${v.error_m.toFixed(1)} m, normalized ${v.normalized_error.toFixed(2)} vs Q90 ${v.threshold_q90.toFixed(2)}`}</title>
          </rect>
          <text x={xs(v.lead_day)} y={h - 8} textAnchor="middle">D{v.lead_day}</text>
        </g>
      ))}
    </svg>
  );
}

export function BarRow({ label, value, max = 1, color = "var(--series-1)", fmt }: {
  label: string; value: number | null; max?: number; color?: string; fmt?: (v: number) => string; }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "150px 1fr 60px", gap: 8, alignItems: "center", marginBottom: 4 }}>
      <span style={{ fontSize: 12 }}>{label}</span>
      <div className="bar-track">
        {value != null && <div className="bar-fill" style={{ width: `${Math.min(100, (100 * value) / max)}%`, background: color }} />}
      </div>
      <span style={{ fontSize: 12, textAlign: "right" }}>{value == null ? "—" : fmt ? fmt(value) : value.toFixed(3)}</span>
    </div>
  );
}

export function ReliabilityDiagram({ curves }: { curves: { name: string; color: string; bins: { mean_pred: number; obs_freq: number; n: number }[] }[] }) {
  const S = 260, m = 34;
  const sc = (v: number) => m + v * (S - m - 10);
  const sy = (v: number) => S - m - v * (S - m - 10);
  return (
    <div>
      <svg viewBox={`0 0 ${S} ${S}`} width="100%" style={{ maxWidth: 320 }} role="img" aria-label="Reliability diagram">
        {[0, 0.5, 1].map((t) => (
          <g key={t}>
            <line x1={sc(0)} x2={sc(1)} y1={sy(t)} y2={sy(t)} stroke="var(--grid)" />
            <text x={m - 4} y={sy(t) + 4} textAnchor="end">{t}</text>
            <text x={sc(t)} y={S - m + 14} textAnchor="middle">{t}</text>
          </g>
        ))}
        <line x1={sc(0)} y1={sy(0)} x2={sc(1)} y2={sy(1)} stroke="var(--text-muted)" strokeDasharray="4 3" />
        {curves.map((c) => (
          <g key={c.name}>
            <polyline fill="none" stroke={c.color} strokeWidth={2} points={c.bins.map((b) => `${sc(b.mean_pred)},${sy(b.obs_freq)}`).join(" ")} />
            {c.bins.map((b, i) => (
              <circle key={i} cx={sc(b.mean_pred)} cy={sy(b.obs_freq)} r={4} fill={c.color} stroke="var(--surface-1)" strokeWidth={2}>
                <title>{`${c.name}: forecast ${b.mean_pred.toFixed(2)}, observed ${b.obs_freq.toFixed(2)} (n=${b.n})`}</title>
              </circle>
            ))}
          </g>
        ))}
        <text x={S / 2} y={S - 4} textAnchor="middle">predicted probability</text>
      </svg>
      <div className="legend">
        {curves.map((c) => <span key={c.name}><span className="swatch" style={{ background: c.color }} />{c.name}</span>)}
        <span>dashed = perfect reliability</span>
      </div>
    </div>
  );
}
