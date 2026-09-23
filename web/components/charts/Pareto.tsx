"use client";

import { scaleLinear, scaleLog } from "d3-scale";
import { useMemo, useState } from "react";
import { fmtObjective, normalizedMin } from "@/lib/format";
import { C, S_COLOR, S_LABEL, S_SHAPE, type Shape } from "@/lib/theme";
import type { Objective, Trial } from "@/lib/types";
import { useSize } from "@/lib/useSize";

export function ShapeMark({ shape, x, y, r, fill, stroke, strokeWidth = 0, opacity = 1 }: {
  shape: Shape; x: number; y: number; r: number; fill: string; stroke?: string; strokeWidth?: number; opacity?: number;
}) {
  const common = { fill, stroke, strokeWidth, opacity };
  if (shape === "square") return <rect x={x - r * 0.85} y={y - r * 0.85} width={r * 1.7} height={r * 1.7} rx={1} {...common} />;
  if (shape === "triangle") return <path d={`M${x},${y - r * 1.05} L${x + r},${y + r * 0.75} L${x - r},${y + r * 0.75} Z`} {...common} />;
  if (shape === "diamond") return <path d={`M${x},${y - r * 1.1} L${x + r * 1.1},${y} L${x},${y + r * 1.1} L${x - r * 1.1},${y} Z`} {...common} />;
  return <circle cx={x} cy={y} r={r} {...common} />;
}

function frontier2d(pts: { x: number; y: number; t: Trial }[], ox: Objective, oy: Objective) {
  // non-dominated set in the projection, as a staircase ordered by x
  const nx = (v: number) => normalizedMin(ox, v), ny = (v: number) => normalizedMin(oy, v);
  const cand = pts.map((p) => ({ ...p, a: nx(p.t.metrics[ox.name]!), b: ny(p.t.metrics[oy.name]!) })).sort((p, q) => p.a - q.a || p.b - q.b);
  const out: typeof cand = [];
  let best = Infinity;
  for (const p of cand) if (p.b < best - 1e-12) { out.push(p); best = p.b; }
  return out;
}

export function ParetoScatter({ trials, ox, oy, knee, selected, onSelect, upTo, height = 420, visible }: {
  trials: Trial[]; ox: Objective; oy: Objective; knee: string | null; selected: string | null; onSelect: (id: string) => void;
  upTo?: number; height?: number; visible: Set<string>;
}) {
  const [ref, { width }] = useSize<HTMLDivElement>();
  const [hover, setHover] = useState<{ t: Trial; x: number; y: number } | null>(null);
  const m = { l: 58, r: 18, t: 16, b: 44 };
  const w = Math.max(280, width), h = height;

  const data = useMemo(() => trials.filter((t) => t.status === "complete" && t.metrics[ox.name] != null && t.metrics[oy.name] != null), [trials, ox.name, oy.name]);
  const shown = useMemo(() => data.filter((t) => visible.has(t.strategy) && (upTo === undefined || order(t) <= upTo)), [data, visible, upTo]);

  const scales = useMemo(() => {
    const ext = (o: Objective) => {
      const vs = data.map((t) => t.metrics[o.name] as number);
      let lo = Math.min(...vs), hi = Math.max(...vs);
      if (!Number.isFinite(lo)) { lo = o.lo; hi = o.hi; }
      if (o.log) { lo = Math.max(lo / 1.3, 1e-9); hi = hi * 1.3; return scaleLog().domain([lo, hi]); }
      const pad = (hi - lo) * 0.08 || Math.abs(hi) * 0.05 || 1;
      return scaleLinear().domain([lo - pad, hi + pad]);
    };
    const x = ext(ox).range([m.l, w - m.r]);
    const y = ext(oy).range([h - m.b, m.t]);
    return { x, y };
  }, [data, ox, oy, w, h, m.l, m.r, m.t, m.b]);

  const pts = shown.map((t) => ({ t, x: scales.x(t.metrics[ox.name] as number), y: scales.y(t.metrics[oy.name] as number) }));
  const front = frontier2d(pts.filter((p) => p.t.feasible), ox, oy);
  const stair = front.map((p, i) => {
    if (i === 0) return `M${p.x},${p.y}`;
    const prev = front[i - 1];
    // step shape respects which direction is better on y
    return `L${p.x},${prev.y} L${p.x},${p.y}`;
  }).join(" ");

  const xt = axisTicks(scales.x as ReturnType<typeof scaleLinear>, ox.log);
  const yt = axisTicks(scales.y as ReturnType<typeof scaleLinear>, oy.log);

  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - r.left, py = e.clientY - r.top;
    let best: (typeof pts)[number] | null = null, bd = 24 * 24;
    for (const p of pts) { const d = (p.x - px) ** 2 + (p.y - py) ** 2; if (d < bd) { bd = d; best = p; } }
    setHover(best ? { t: best.t, x: best.x, y: best.y } : null);
  };

  const kneePt = pts.find((p) => p.t.id === knee);
  const selPt = pts.find((p) => p.t.id === selected);

  return (
    <div ref={ref} className="relative w-full" style={{ height: h }}>
      {width > 0 && (
        <svg width={w} height={h} className="block" onPointerMove={onMove} onPointerLeave={() => setHover(null)}
          onClick={() => hover && onSelect(hover.t.id)} role="img" aria-label={`${oy.label} versus ${ox.label} for ${shown.length} trials`}>
          {xt.map((v) => <line key={`gx${v}`} x1={scales.x(v)} x2={scales.x(v)} y1={m.t} y2={h - m.b} stroke={C.grid} />)}
          {yt.map((v) => <line key={`gy${v}`} y1={scales.y(v)} y2={scales.y(v)} x1={m.l} x2={w - m.r} stroke={C.grid} />)}
          <line x1={m.l} x2={w - m.r} y1={h - m.b} y2={h - m.b} stroke={C.lineStrong} />
          <line x1={m.l} x2={m.l} y1={m.t} y2={h - m.b} stroke={C.lineStrong} />
          {xt.map((v) => <text key={`tx${v}`} x={scales.x(v)} y={h - m.b + 16} textAnchor="middle" className="num" fontSize={10.5} fill={C.muted}>{tick(ox, v)}</text>)}
          {yt.map((v) => <text key={`ty${v}`} x={m.l - 8} y={scales.y(v) + 3.5} textAnchor="end" className="num" fontSize={10.5} fill={C.muted}>{tick(oy, v)}</text>)}
          <text x={(m.l + w - m.r) / 2} y={h - 8} textAnchor="middle" fontSize={11} fill={C.ink2}>{ox.label}{ox.unit ? ` (${ox.unit}${ox.log ? ", log" : ""})` : ""} · {ox.direction === "min" ? "← better" : "better →"}</text>
          <text transform={`translate(14 ${(m.t + h - m.b) / 2}) rotate(-90)`} textAnchor="middle" fontSize={11} fill={C.ink2}>{oy.label}{oy.unit ? ` (${oy.unit}${oy.log ? ", log" : ""})` : ""} · {oy.direction === "max" ? "better ↑" : "↓ better"}</text>

          {pts.filter((p) => !p.t.feasible).map((p) => (
            <ShapeMark key={p.t.id} shape={S_SHAPE[p.t.strategy]} x={p.x} y={p.y} r={3.6} fill="none" stroke={C.muted} strokeWidth={1.2} opacity={0.55} />
          ))}
          {pts.filter((p) => p.t.feasible && !p.t.global_pareto).map((p) => (
            <ShapeMark key={p.t.id} shape={S_SHAPE[p.t.strategy]} x={p.x} y={p.y} r={4} fill={S_COLOR[p.t.strategy]} stroke={C.surface} strokeWidth={2} opacity={0.5} />
          ))}
          {front.length > 1 && <path d={stair} fill="none" stroke={C.ink} strokeOpacity={0.55} strokeWidth={1.5} />}
          {pts.filter((p) => p.t.global_pareto).map((p) => (
            <g key={p.t.id}>
              <ShapeMark shape={S_SHAPE[p.t.strategy]} x={p.x} y={p.y} r={6.5} fill="none" stroke={C.ink} strokeWidth={1.2} opacity={0.9} />
              <ShapeMark shape={S_SHAPE[p.t.strategy]} x={p.x} y={p.y} r={4.4} fill={S_COLOR[p.t.strategy]} stroke={C.surface} strokeWidth={1.5} />
            </g>
          ))}
          {kneePt && (
            <g pointerEvents="none">
              <circle cx={kneePt.x} cy={kneePt.y} r={11} fill="none" stroke={C.crimson} strokeWidth={1.5} />
              <line x1={kneePt.x + 8} y1={kneePt.y - 8} x2={kneePt.x + 22} y2={kneePt.y - 22} stroke={C.crimson} strokeWidth={1} />
              <text x={kneePt.x + 25} y={kneePt.y - 25} fontSize={11} fill={C.ink} fontWeight={600}>knee · deployed</text>
            </g>
          )}
          {selPt && <circle cx={selPt.x} cy={selPt.y} r={9} fill="none" stroke="#fff" strokeWidth={1.5} strokeDasharray="2 2" pointerEvents="none" />}
          {hover && <circle cx={hover.x} cy={hover.y} r={8} fill="none" stroke="#fff" strokeOpacity={0.7} strokeWidth={1} pointerEvents="none" />}
        </svg>
      )}
      {hover && (
        <div className="pointer-events-none absolute z-10 min-w-[190px] rounded-md border border-line-strong bg-[#12151d]/95 px-3 py-2 text-[12px] shadow-xl shadow-black/40"
          style={{ left: Math.min(hover.x + 14, w - 210), top: Math.max(0, hover.y - 70) }}>
          <div className="mb-1 flex items-center gap-2 text-muted">
            <svg width="10" height="10"><ShapeMark shape={S_SHAPE[hover.t.strategy]} x={5} y={5} r={4} fill={S_COLOR[hover.t.strategy]} /></svg>
            {S_LABEL[hover.t.strategy]} · seed {hover.t.seed} · trial {hover.t.number}
          </div>
          <div className="num text-[13px] font-semibold text-ink">{fmtObjective(oy, hover.t.metrics[oy.name])} <span className="font-normal text-muted">{oy.label}</span></div>
          <div className="num text-[13px] font-semibold text-ink">{fmtObjective(ox, hover.t.metrics[ox.name])} <span className="font-normal text-muted">{ox.label}</span></div>
          <div className="mt-1 text-[11px] text-muted">{hover.t.global_pareto ? "on the global frontier" : hover.t.feasible ? "feasible · dominated" : "violates a constraint"} · {hover.t.origin}</div>
        </div>
      )}
    </div>
  );
}

function order(t: Trial): number { return t.number; }

export function axisTicks(s: ReturnType<typeof scaleLinear>, log: boolean): number[] {
  if (!log) return s.ticks(6);
  const all = s.ticks(8);
  const lead = (v: number) => Math.round(v / 10 ** Math.floor(Math.log10(v)));
  let t = all.filter((v) => [1, 2, 5].includes(lead(v)));
  if (t.length > 7) t = all.filter((v) => lead(v) === 1);
  if (t.length < 2) t = all.filter((_, i) => i % Math.ceil(all.length / 5) === 0);
  return t;
}

function tick(o: Objective, v: number): string {
  if (o.unit === "ms") return v >= 1000 ? `${v / 1000}s` : `${+v.toPrecision(3)}`;
  if (o.unit === "$") return v < 0.01 ? v.toExponential(0) : `${+v.toPrecision(2)}`;
  return `${+v.toPrecision(3)}`;
}
