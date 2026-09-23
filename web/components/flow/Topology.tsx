"use client";

import { Background, BackgroundVariant, MarkerType, ReactFlow, type Edge, type Node } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useMemo } from "react";
import { useSize } from "@/lib/useSize";
import { nodeTypes, type PipeData } from "./nodes";
import { edgeTypes, type PacketData } from "./PacketEdge";
import type { Topology } from "@/lib/types";

/** Layered layout of a pipeline DAG: columns by longest path from the ingress, rows in declaration order. */
function layout(t: Topology["graph"], vertical = false) {
  const depth: Record<string, number> = {};
  const inc: Record<string, string[]> = {};
  for (const e of t.edges) (inc[e.target] ??= []).push(e.source);
  const visit = (id: string, seen = new Set<string>()): number => {
    if (depth[id] !== undefined) return depth[id];
    if (seen.has(id)) return 0;
    seen.add(id);
    const ps = inc[id] ?? [];
    depth[id] = ps.length ? Math.max(...ps.map((p) => visit(p, seen) + 1)) : 0;
    return depth[id];
  };
  t.nodes.forEach((n) => visit(n.id));
  const cols: Record<number, string[]> = {};
  t.nodes.forEach((n) => (cols[depth[n.id]] ??= []).push(n.id));
  const pos: Record<string, { x: number; y: number }> = {};
  Object.entries(cols).forEach(([c, ids]) => ids.forEach((id, i) => (pos[id] = vertical
    ? { x: i * 200 - ((ids.length - 1) * 200) / 2, y: Number(c) * 112 }
    : { x: Number(c) * 296, y: i * 118 - ((ids.length - 1) * 118) / 2 })));
  return pos;
}

export function TopologyFlow({ topo, stageMs, status, height = 360 }: {
  topo: Topology; stageMs?: Record<string, number> | null; status?: Record<string, "ok" | "degraded" | "critical">; height?: number;
}) {
  const [ref, { width }] = useSize<HTMLDivElement>();
  const vertical = width > 0 && width < 700;  // phones: top-to-bottom pipeline
  const { nodes, edges } = useMemo(() => {
    const pos = layout(topo.graph, vertical);
    const nodes: Node[] = topo.graph.nodes.map((n) => ({
      id: n.id, type: "pipe", position: pos[n.id] ?? { x: 0, y: 0 }, draggable: true,
      data: {
        label: n.label, sub: n.sub, kind: n.kind, status: status?.[n.id] ?? null, vertical,
        ms: n.stage && stageMs ? stageMs[n.stage] ?? null : null,
        share: n.stage ? inShare(topo, n.id) : null,
      } satisfies PipeData,
    }));
    const edges: Edge[] = topo.graph.edges.map((e, i) => {
      const share = Math.max(0, Math.min(1, e.share));
      const w = 1 + 5 * Math.sqrt(share);
      const hot = share >= 0.02;
      const st = status?.[e.target] ?? status?.[e.source] ?? null;
      const color = st === "critical" ? "#ff4d6d" : st === "degraded" ? "#fbbf24" : hot ? "#22d3ee" : "#3a4156";
      const packets = share <= 0 ? 0 : 1 + Math.round(3 * Math.sqrt(share));
      return {
        id: `e${i}`, source: e.source, target: e.target, type: "packet",
        label: e.label ? `${e.label} ${Math.round(share * 100)}%` : undefined,
        labelStyle: { fill: st ? color : "#a9b0bf", fontSize: 10, fontFamily: "var(--font-mono)" },
        labelBgStyle: { fill: "#0c0e13" }, labelBgPadding: [4, 2] as [number, number], labelBgBorderRadius: 3,
        style: { stroke: color, strokeWidth: w, strokeOpacity: st ? 0.7 : hot ? 0.35 : 0.9 },
        markerEnd: { type: MarkerType.ArrowClosed, color, width: 16 / w, height: 16 / w },
        data: { packets, color, glow: hot || !!st, dur: st === "critical" ? 3.4 : st === "degraded" ? 2.6 : 2.2 - 0.9 * share, jam: st === "critical", r: 2.8 + 1.4 * Math.sqrt(share) } satisfies PacketData,
      };
    });
    return { nodes, edges };
  }, [topo, stageMs, status, vertical]);
  const depth = Math.max(0, ...nodes.map((n) => n.position.y / 112));

  return (
    <div ref={ref} style={{ height: vertical ? Math.min(760, (depth + 1) * 92 + 40) : height }} className="w-full overflow-hidden rounded-lg border border-line bg-sunken">
      <ReactFlow key={vertical ? "v" : "h"} nodes={nodes} edges={edges} nodeTypes={nodeTypes} edgeTypes={edgeTypes} fitView fitViewOptions={{ padding: 0.06 }} proOptions={{ hideAttribution: true }}
        nodesConnectable={false} zoomOnScroll={false} panOnScroll={false} preventScrolling={false} minZoom={0.3} maxZoom={1.6}>
        <Background variant={BackgroundVariant.Dots} gap={18} size={1} color="#1c2030" />
      </ReactFlow>
    </div>
  );
}

function inShare(topo: Topology, id: string): number {
  return topo.graph.edges.filter((e) => e.target === id).reduce((a, e) => a + e.share, 0);
}
