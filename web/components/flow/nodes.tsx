"use client";

import { Handle, Position, type NodeProps } from "@xyflow/react";
import { Bot, Boxes, Database, Filter, GitMerge, Inbox, Server, Sparkles, Split } from "lucide-react";
import type { ComponentType } from "react";
import { fmtMs } from "@/lib/format";
import { KIND_COLOR } from "@/lib/theme";

const KIND_ICON: Record<string, ComponentType<{ size?: number; strokeWidth?: number }>> = {
  io: Inbox, transform: GitMerge, gate: Filter, model: Boxes, llm: Sparkles, store: Database, split: Split, server: Server, agent: Bot,
};

export interface PipeData extends Record<string, unknown> {
  label: string; sub?: string | null; kind: string; ms?: number | null; share?: number | null; status?: "ok" | "degraded" | "critical" | null; accent?: string; vertical?: boolean;
}

export function PipeNode({ data, selected }: NodeProps) {
  const d = data as PipeData;
  const color = d.status === "critical" ? "#ff4d6d" : d.status === "degraded" ? "#fbbf24" : d.accent ?? KIND_COLOR[d.kind] ?? "#a9b0bf";
  const Icon = KIND_ICON[d.kind] ?? Boxes;
  const ring = d.status === "critical" ? "#f43f5e" : d.status === "degraded" ? "#fbbf2499" : null;
  const bg = d.status === "critical" ? "linear-gradient(180deg,#2a0c15,#170a10)" : d.status === "degraded" ? "linear-gradient(180deg,#221a0b,#13120e)" : "linear-gradient(180deg,#141823,#0f121a)";
  return (
    <div className={`node-card relative w-[184px] rounded-lg border px-3 py-2.5 transition-[border-color,box-shadow] duration-300 ${d.status === "critical" ? "pulse-critical" : ""}`}
      style={{ borderColor: ring ?? (selected ? "#e8eaf0" : "#262b3a"), background: bg, boxShadow: d.status ? `0 0 24px -6px ${color}88` : `0 8px 24px -12px #000, 0 0 0 0 ${color}` }}>
      <Handle type="target" position={d.vertical ? Position.Top : Position.Left} />
      <div className="absolute inset-y-2 left-0 w-[2px] rounded-r" style={{ background: color, boxShadow: `0 0 8px ${color}` }} />
      {d.status && (
        <span className="num absolute -top-2 right-2 rounded-full px-1.5 py-[1px] text-[9px] font-semibold tracking-[0.12em]"
          style={{ background: color, color: "#0b0b0e" }}>{d.status === "critical" ? "ALARM" : "DEGRADED"}</span>
      )}
      <div className="flex items-center gap-2">
        <span className="grid h-6 w-6 shrink-0 place-items-center rounded-md" style={{ background: `${color}1c`, color }}>
          <Icon size={13} strokeWidth={1.8} />
        </span>
        <div className="min-w-0 flex-1">
          <div className="truncate text-[12.5px] font-medium text-ink">{d.label}</div>
          {d.sub && <div className="num truncate text-[10.5px] text-muted">{d.sub}</div>}
        </div>
      </div>
      {(d.ms != null || d.share != null) && (
        <div className="mt-2 border-t border-line pt-1.5 text-[10.5px]">
          <div className="flex items-center justify-between">
            {d.share != null && <span className="text-muted">traffic <span className="num text-ink-2">{Math.round(d.share * 100)}%</span></span>}
            {d.ms != null && <span className="text-muted">cost <span key={Math.round(d.ms * 100)} className="num flash-val" style={{ color: d.status ? color : "#a9b0bf" }}>{fmtMs(d.ms)}/req</span></span>}
          </div>
          {d.share != null && (
            <div className="mt-1 h-[3px] overflow-hidden rounded-full bg-[#1a1e2a]">
              <div className="h-full rounded-full transition-[width] duration-500" style={{ width: `${Math.max(2, Math.round(d.share * 100))}%`, background: color, boxShadow: `0 0 6px ${color}` }} />
            </div>
          )}
        </div>
      )}
      <Handle type="source" position={d.vertical ? Position.Bottom : Position.Right} />
    </div>
  );
}

export interface GroupData extends Record<string, unknown> { label: string; color: string }
export function GroupNode({ data }: NodeProps) {
  const d = data as GroupData;
  return (
    <div className="h-full w-full rounded-xl border border-dashed" style={{ borderColor: `${d.color}55`, background: `${d.color}08` }}>
      <div className="eyebrow px-3 pt-2" style={{ color: d.color }}>{d.label}</div>
    </div>
  );
}

export interface DotData extends Record<string, unknown> { color: string; r: number; ring?: string | null; label?: string | null; dim?: boolean }
export function DotNode({ data }: NodeProps) {
  const d = data as DotData;
  return (
    <div className="relative" style={{ width: d.r * 2, height: d.r * 2 }}>
      <Handle type="target" position={Position.Left} />
      <div className="h-full w-full rounded-full" style={{ background: d.color, opacity: d.dim ? 0.35 : 1, boxShadow: d.ring ? `0 0 0 2px #101218, 0 0 0 3.5px ${d.ring}` : "0 0 0 2px #101218" }} />
      {d.label && <div className="num absolute left-1/2 top-full mt-1 -translate-x-1/2 whitespace-nowrap text-[9.5px] text-ink-2">{d.label}</div>}
      <Handle type="source" position={Position.Right} />
    </div>
  );
}

export function TickNode({ data }: NodeProps) {
  return (
    <div className="flex w-[1000px] items-center gap-2">
      <span className="num w-12 text-right text-[10px] text-muted">{String((data as { label: string }).label)}</span>
      <span className="h-px flex-1 bg-[#1a1d27]" />
    </div>
  );
}

export const nodeTypes = { pipe: PipeNode, group: GroupNode, dot: DotNode, tick: TickNode };
