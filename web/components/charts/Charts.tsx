"use client";

import { scaleLinear, scaleLog } from "d3-scale";
import { useState, type ReactNode } from "react";
import {
  Area, CartesianGrid, ComposedChart, Line, LineChart, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { fmtMs, fmtNum } from "@/lib/format";
import { C, S_COLOR, S_LABEL, STRATEGIES } from "@/lib/theme";
import type { Race, Strategy } from "@/lib/types";
import { useSize } from "@/lib/useSize";

/* ---------------------------------------------------------------- tooltip shell */
export function TipBox({ title, rows }: { title?: ReactNode; rows: { color?: string; label: ReactNode; value: ReactNode; dashed?: boolean }[] }) {
  return (
    <div className="min-w-[170px] rounded-md border border-line-strong bg-[#12151d]/95 px-3 py-2 text-[12px] shadow-xl shadow-black/40">
      {title && <div className="mb-1.5 text-[11px] text-muted">{title}</div>}
      {rows.map((r, i) => (
        <div key={i} className="flex items-center justify-between gap-4 py-[1px]">
          <span className="flex items-center gap-1.5 text-ink-2">
            {r.color && <span className="inline-block h-[2px] w-3" style={{ background: r.dashed ? `repeating-linear-gradient(90deg, ${r.color} 0 3px, transparent 3px 5px)` : r.color }} />}
            {r.label}
          </span>
          <span className="num font-semibold text-ink">{r.value}</span>
        </div>
      ))}
    </div>
  );
}

/* ---------------------------------------------------------------- HV race */
export function HvRace({ race, height = 300 }: { race: Race; height?: number }) {
  const strategies = STRATEGIES.filter((s) => race.strategies[s]);
  const n = race.budget;
  const rows = Array.from({ length: n }, (_, i) => {
    const r: Record<string, number> = { trial: i + 1 };
    for (const s of strategies) r[s] = race.strategies[s]!.mean_curve[i] ?? race.strategies[s]!.mean_curve.at(-1)!;
    for (const [id, c] of Object.entries(race.curves)) r[id] = c.hv[i] ?? c.hv.at(-1)!;
    return r;
  });
  const all = rows.flatMap((r) => strategies.map((s) => r[s]));
  const lo = Math.max(0, Math.min(...all.filter((v) => v > 0)) * 0.97);
  return (
    <div className="w-full">
    <div className="mb-2 flex flex-wrap gap-x-4 gap-y-1">
      {strategies.map((s) => (
        <span key={s} className="inline-flex items-center gap-1.5 text-[11.5px] text-ink-2">
          <span className="inline-block h-[2px] w-3.5 rounded" style={{ background: S_COLOR[s] }} />{S_LABEL[s]}
          <span className="num text-ink">{race.strategies[s]!.final_hv_median.toFixed(3)}</span>
        </span>
      ))}
    </div>
    <div style={{ height }} className="w-full">
      <ResponsiveContainer>
        <LineChart data={rows} margin={{ top: 8, right: 16, bottom: 18, left: 4 }}>
          <CartesianGrid vertical={false} />
          <XAxis dataKey="trial" type="number" domain={[1, n]} tickCount={7} label={{ value: "trials (equal budget per strategy)", position: "insideBottom", offset: -10, fill: C.muted, fontSize: 11 }} />
          <YAxis domain={[lo, "auto"]} tickFormatter={(v) => v.toFixed(2)} width={44} />
          <ReferenceLine y={race.target_hv} stroke={C.ink2} strokeOpacity={0.5}
            label={{ value: `target ${Math.round(race.target_frac * 100)}% of best`, position: "insideTopLeft", fill: C.muted, fontSize: 10.5 }} />
          {Object.entries(race.curves).map(([id, c]) => (
            <Line key={id} dataKey={id} dot={false} stroke={S_COLOR[c.strategy]} strokeOpacity={0.18} strokeWidth={1} isAnimationActive={false} type="stepAfter" />
          ))}
          {strategies.map((s) => (
            <Line key={s} dataKey={s} dot={false} stroke={S_COLOR[s]} strokeWidth={s === "agent" ? 2.5 : 2} type="monotone" isAnimationActive={false} />
          ))}
          <Tooltip cursor={{ stroke: C.lineStrong }} content={({ active, payload, label }) => active && payload ? (
            <TipBox title={`after trial ${label} · mean of ${race.seeds.length} seeds`}
              rows={strategies.map((s) => ({ color: S_COLOR[s], label: S_LABEL[s], value: Number(payload[0].payload[s]).toFixed(4) }))
                .sort((a, b) => Number(b.value) - Number(a.value))} />) : null} />
        </LineChart>
      </ResponsiveContainer>
    </div>
    </div>
  );
}

/* ---------------------------------------------------------------- latency histogram */
export function LatencyHist({ hist, marks }: { hist: { edges: number[]; counts: number[] }; marks?: { v: number; label: string; color?: string }[] }) {
  const [ref, { width }] = useSize<HTMLDivElement>();
  const h = 120, m = { l: 8, r: 8, t: 14, b: 20 };
  if (!hist.edges.length) return null;
  const x = scaleLog().domain([hist.edges[0], hist.edges.at(-1)!]).range([m.l, Math.max(200, width) - m.r]);
  const y = scaleLinear().domain([0, Math.max(...hist.counts)]).range([h - m.b, m.t]);
  const ticks = x.ticks(4).filter((v) => [1, 2, 5].includes(Math.round(v / 10 ** Math.floor(Math.log10(v)))));
  return (
    <div ref={ref} className="w-full">
      {width > 0 && (
        <svg width={width} height={h} role="img" aria-label="Latency distribution">
          {hist.counts.map((c, i) => {
            const x0 = x(hist.edges[i]) + 1, x1 = x(hist.edges[i + 1]) - 1;
            return c > 0 ? <rect key={i} x={x0} width={Math.max(1, x1 - x0)} y={y(c)} height={h - m.b - y(c)} rx={1.5} fill={C.info} opacity={0.75} /> : null;
          })}
          <line x1={m.l} x2={width - m.r} y1={h - m.b} y2={h - m.b} stroke={C.lineStrong} />
          {ticks.slice(0, 6).map((v) => <text key={v} x={x(v)} y={h - 5} fontSize={10} textAnchor="middle" className="num" fill={C.muted}>{fmtMs(v)}</text>)}
          {marks?.map((mk) => (
            <g key={mk.label}>
              <line x1={x(mk.v)} x2={x(mk.v)} y1={m.t - 4} y2={h - m.b} stroke={mk.color ?? C.ink} strokeWidth={1} />
              <text x={x(mk.v) + 3} y={m.t + 2} fontSize={10} fill={C.ink2}>{mk.label}</text>
            </g>
          ))}
        </svg>
      )}
    </div>
  );
}

/* ---------------------------------------------------------------- stage breakdown */
const STAGE_COLORS = ["#3987e5", "#a78bfa", "#c98500", "#199e70", "#22d3ee"];
export function StageBar({ stages }: { stages: Record<string, number> }) {
  const entries = Object.entries(stages).filter(([, v]) => v > 0);
  const total = entries.reduce((a, [, v]) => a + v, 0) || 1;
  return (
    <div>
      <div className="flex h-3 w-full gap-[2px] overflow-hidden rounded-[3px]">
        {entries.map(([k, v], i) => <div key={k} style={{ width: `${(100 * v) / total}%`, background: STAGE_COLORS[i % 5] }} title={`${k}: ${fmtMs(v)}`} />)}
      </div>
      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1">
        {entries.map(([k, v], i) => (
          <span key={k} className="inline-flex items-center gap-1.5 text-[11.5px] text-ink-2">
            <span className="h-2.5 w-2.5 rounded-[2px]" style={{ background: STAGE_COLORS[i % 5] }} />{k}<span className="num text-ink">{fmtMs(v)}</span>
          </span>
        ))}
      </div>
    </div>
  );
}

/* ---------------------------------------------------------------- sparkline */
export function Sparkline({ values, color = C.ink2, height = 28, width = 120, fill = true }: { values: number[]; color?: string; height?: number; width?: number; fill?: boolean }) {
  if (values.length < 2) return <svg width={width} height={height} />;
  const lo = Math.min(...values), hi = Math.max(...values);
  const x = (i: number) => (i / (values.length - 1)) * (width - 4) + 2;
  const y = (v: number) => height - 3 - ((v - lo) / (hi - lo || 1)) * (height - 6);
  const d = values.map((v, i) => `${i ? "L" : "M"}${x(i)},${y(v)}`).join(" ");
  return (
    <svg width={width} height={height} aria-hidden="true">
      {fill && <path d={`${d} L${x(values.length - 1)},${height} L${x(0)},${height} Z`} fill={color} opacity={0.1} />}
      <path d={d} fill="none" stroke={color} strokeWidth={1.5} strokeLinejoin="round" />
      <circle cx={x(values.length - 1)} cy={y(values.at(-1)!)} r={2.5} fill={color} />
    </svg>
  );
}

/* ---------------------------------------------------------------- horizontal bars (importance) */
export function HBars({ data, color = "#3987e5", format = (v: number) => fmtNum(v, 2), max }: { data: { label: string; value: number }[]; color?: string; format?: (v: number) => string; max?: number }) {
  const top = max ?? Math.max(...data.map((d) => d.value), 1e-9);
  return (
    <div className="flex flex-col gap-1.5">
      {data.map((d) => (
        <div key={d.label} className="grid grid-cols-[minmax(70px,108px)_1fr_46px] items-center gap-2">
          <span className="num truncate text-[11.5px] text-ink-2" title={d.label}>{d.label}</span>
          <div className="h-[10px] rounded-r-[4px]" style={{ width: `${Math.max(1.5, (100 * d.value) / top)}%`, background: color }} />
          <span className="num text-right text-[11.5px] text-ink">{format(d.value)}</span>
        </div>
      ))}
    </div>
  );
}

/* ---------------------------------------------------------------- diverging bars (ablation) */
export function Diverging({ rows, better, max }: { rows: { label: string; value: number; sig: boolean; text: string }[]; better: "neg" | "pos"; max?: number }) {
  const top = max ?? Math.max(...rows.map((r) => Math.abs(r.value)), 1e-9);
  return (
    <div className="flex flex-col gap-1.5">
      {rows.map((r) => {
        const good = better === "pos" ? r.value > 0 : r.value < 0;
        const col = good ? "#3987e5" : "#f43f5e";
        const wPct = (50 * Math.abs(r.value)) / top;
        return (
          <div key={r.label} className="grid grid-cols-[1fr_62px] items-center gap-2">
            <div className="relative h-[12px]">
              <div className="absolute inset-y-0 left-1/2 w-px bg-line-strong" />
              <div className="absolute inset-y-[1px] rounded-[3px]" title={r.text}
                style={{ left: r.value < 0 ? `${50 - wPct}%` : "50%", width: `${Math.max(0.8, wPct)}%`, background: col, opacity: r.sig ? 1 : 0.35 }} />
            </div>
            <span className="num text-right text-[11px] text-ink-2">{r.text}</span>
          </div>
        );
      })}
    </div>
  );
}

/* ---------------------------------------------------------------- windowed incident metric */
export function WindowChart({ data, metric, label, format, faultFrom, alarm, promoted, recovered, target, height = 170, betterUp, xMax }: {
  data: { w: number; live: number | null; cf: number | null }[]; metric: string; label: string; format: (v: number) => string;
  faultFrom?: number; alarm?: number; promoted?: number; recovered?: number | null; target?: number; height?: number; betterUp: boolean; xMax?: number;
}) {
  const lastW = xMax ?? data.at(-1)?.w ?? 0;
  return (
    <div style={{ height }} className="w-full">
      <ResponsiveContainer>
        <ComposedChart data={data} margin={{ top: 10, right: 12, bottom: 4, left: 0 }}>
          <CartesianGrid vertical={false} />
          {faultFrom !== undefined && <ReferenceArea x1={faultFrom} x2={lastW} fill={C.critical} fillOpacity={0.06} />}
          <XAxis dataKey="w" type="number" domain={[0, lastW]} allowDecimals={false} tickCount={8} tickFormatter={(v) => `${v}m`} />
          <YAxis width={54} domain={["auto", "auto"]} tickFormatter={(v) => format(v).replace(" ", "")} />
          {target !== undefined && <ReferenceLine y={target} stroke={C.ink2} strokeOpacity={0.45} label={{ value: "target", position: "insideTopRight", fill: C.muted, fontSize: 10 }} />}
          {alarm !== undefined && <ReferenceLine x={alarm} stroke={C.critical} label={{ value: "alarm", position: "insideTopLeft", fill: C.critical, fontSize: 10 }} />}
          {promoted !== undefined && <ReferenceLine x={promoted} stroke={C.ok} label={{ value: "promoted", position: "insideTopLeft", fill: C.ok, fontSize: 10 }} />}
          {recovered != null && <ReferenceLine x={recovered} stroke={C.ok} strokeOpacity={0.5} />}
          <Line dataKey="cf" stroke={C.muted} strokeWidth={1.5} strokeDasharray="4 3" dot={false} isAnimationActive={false} connectNulls={false} />
          <Area dataKey="live" stroke="none" fill={C.ink} fillOpacity={0.04} isAnimationActive={false} />
          <Line dataKey="live" stroke={C.ink} strokeWidth={2} dot={false} isAnimationActive={false} />
          <Tooltip cursor={{ stroke: C.lineStrong }} content={({ active, payload }) => active && payload?.length ? (
            <TipBox title={`minute ${payload[0].payload.w} · ${label}`} rows={[
              { color: C.ink, label: "live", value: payload[0].payload.live == null ? "—" : format(payload[0].payload.live) },
              ...(payload[0].payload.cf != null ? [{ color: C.muted, dashed: true, label: "no remediation", value: format(payload[0].payload.cf) }] : []),
            ]} />) : null} />
        </ComposedChart>
      </ResponsiveContainer>
      <span className="sr-only">{metric} {betterUp ? "higher is better" : "lower is better"}</span>
    </div>
  );
}

/* ---------------------------------------------------------------- CUSUM */
export function CusumChart({ series, h, faultFrom, height = 150 }: { series: { metric: string; values: (number | null)[]; color: string }[]; h: number; faultFrom: number; height?: number }) {
  const n = Math.max(...series.map((s) => s.values.length));
  const rows = Array.from({ length: n }, (_, i) => {
    const r: Record<string, number | null> = { w: faultFrom + i };
    for (const s of series) r[s.metric] = s.values[i] ?? null;
    return r;
  });
  return (
    <div style={{ height }} className="w-full">
      <ResponsiveContainer>
        <LineChart data={rows} margin={{ top: 10, right: 12, bottom: 4, left: 0 }}>
          <CartesianGrid vertical={false} />
          <XAxis dataKey="w" tickFormatter={(v) => `${v}m`} />
          <YAxis width={36} />
          <ReferenceLine y={h} stroke={C.critical} strokeOpacity={0.7} label={{ value: `h = ${h}σ`, position: "insideTopRight", fill: C.critical, fontSize: 10 }} />
          {series.map((s) => <Line key={s.metric} dataKey={s.metric} stroke={s.color} strokeWidth={2} dot={{ r: 3, strokeWidth: 0, fill: s.color }} isAnimationActive={false} />)}
          <Tooltip cursor={{ stroke: C.lineStrong }} content={({ active, payload }) => active && payload?.length ? (
            <TipBox title={`minute ${payload[0].payload.w} · CUSUM statistic`} rows={series.map((s) => ({ color: s.color, label: s.metric, value: payload[0].payload[s.metric] == null ? "—" : Number(payload[0].payload[s.metric]).toFixed(2) }))} />) : null} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

/* ---------------------------------------------------------------- twin time series */
export function TwinSeries({ rows, keys, height = 180, yFormat = fmtMs, log }: {
  rows: Record<string, number | null>[]; keys: { key: string; label: string; color: string; area?: boolean; dashed?: boolean }[]; height?: number; yFormat?: (v: number) => string; log?: boolean;
}) {
  return (
    <div style={{ height }} className="w-full">
      <ResponsiveContainer>
        <ComposedChart data={rows} margin={{ top: 10, right: 12, bottom: 4, left: 0 }}>
          <CartesianGrid vertical={false} />
          <XAxis dataKey="t" type="number" domain={["dataMin", "dataMax"]} tickFormatter={(v) => `${v}s`} tickCount={7} />
          <YAxis width={56} scale={log ? "log" : "auto"} domain={log ? ["auto", "auto"] : [0, "auto"]} allowDataOverflow tickFormatter={(v) => yFormat(v).replace(" ", "")} />
          {keys.map((k) => k.area
            ? <Area key={k.key} dataKey={k.key} stroke={k.color} strokeWidth={1.5} fill={k.color} fillOpacity={0.1} isAnimationActive={false} />
            : <Line key={k.key} dataKey={k.key} stroke={k.color} strokeWidth={2} strokeDasharray={k.dashed ? "4 3" : undefined} dot={false} isAnimationActive={false} connectNulls />)}
          <Tooltip cursor={{ stroke: C.lineStrong }} content={({ active, payload }) => active && payload?.length ? (
            <TipBox title={`t = ${payload[0].payload.t}s`} rows={keys.map((k) => ({ color: k.color, dashed: k.dashed, label: k.label, value: payload[0].payload[k.key] == null ? "—" : yFormat(Number(payload[0].payload[k.key])) }))} />) : null} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

export function StrategyLegend({ visible, toggle }: { visible: Set<Strategy>; toggle?: (s: Strategy) => void }) {
  const [, force] = useState(0);
  return (
    <div className="flex flex-wrap gap-1.5">
      {STRATEGIES.map((s) => (
        <button key={s} onClick={() => { toggle?.(s); force((x) => x + 1); }} aria-pressed={visible.has(s)}
          className="inline-flex items-center gap-1.5 rounded-md border border-line px-2 py-1 text-[11.5px] transition-opacity"
          style={{ opacity: visible.has(s) ? 1 : 0.4 }}>
          <svg width="10" height="10" aria-hidden="true">{s === "agent" ? <circle cx="5" cy="5" r="4" fill={S_COLOR[s]} /> : s === "random" ? <rect x="1.5" y="1.5" width="7" height="7" fill={S_COLOR[s]} /> : s === "tpe" ? <path d="M5,0.8 L9.2,8.6 L0.8,8.6Z" fill={S_COLOR[s]} /> : <path d="M5,0.4 L9.6,5 L5,9.6 L0.4,5Z" fill={S_COLOR[s]} />}</svg>
          <span className="text-ink-2">{S_LABEL[s]}</span>
        </button>
      ))}
    </div>
  );
}
