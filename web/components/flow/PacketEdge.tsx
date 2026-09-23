"use client";

import { BaseEdge, getBezierPath, getSmoothStepPath, type EdgeProps } from "@xyflow/react";
import { useReducedMotion } from "framer-motion";

export interface PacketData extends Record<string, unknown> {
  packets: number;        // particles in flight on this edge
  color: string;          // particle + glow colour
  dur: number;            // seconds for one particle to cross the edge
  r?: number;             // particle radius
  smooth?: boolean;       // smooth-step routing (architecture) instead of bezier (pipelines)
  glow?: boolean;         // soft halo under the edge
  jam?: boolean;          // congested: particles bunch up near the target
}

/** An edge that carries live request packets: SVG circles riding the edge path with SMIL animateMotion. */
export function PacketEdge(props: EdgeProps) {
  const { id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, style, markerEnd,
    label, labelStyle, labelBgStyle, labelBgPadding, labelBgBorderRadius } = props;
  const d = (props.data ?? {}) as PacketData;
  const reduce = useReducedMotion();
  const [path, lx, ly] = d.smooth
    ? getSmoothStepPath({ sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, borderRadius: 10 })
    : getBezierPath({ sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition });
  const n = reduce ? 0 : Math.max(0, Math.min(6, Math.round(d.packets ?? 0)));
  const dur = Math.max(0.5, d.dur ?? 1.8);
  const r = d.r ?? 2.6;
  const color = d.color ?? "#22d3ee";

  return (
    <>
      {d.glow && <path d={path} fill="none" stroke={color} strokeOpacity={0.09} strokeWidth={Number(style?.strokeWidth ?? 2) + 7} strokeLinecap="round" />}
      <BaseEdge id={id} path={path} style={style} markerEnd={markerEnd} label={label} labelX={lx} labelY={ly}
        labelStyle={labelStyle} labelShowBg labelBgStyle={labelBgStyle} labelBgPadding={labelBgPadding} labelBgBorderRadius={labelBgBorderRadius} />
      {Array.from({ length: n }).map((_, i) => (
        <g key={i} style={{ filter: `drop-shadow(0 0 3px ${color})` }}>
          <circle r={r} fill={color}>
            <animateMotion dur={`${dur}s`} repeatCount="indefinite" begin={`${-((i * dur) / n).toFixed(2)}s`} path={path}
              {...(d.jam ? { keyPoints: "0;0.72;0.9;1", keyTimes: "0;0.3;0.85;1", calcMode: "linear" } : {})} />
          </circle>
          <circle r={r * 0.45} fill="#fff" opacity={0.85}>
            <animateMotion dur={`${dur}s`} repeatCount="indefinite" begin={`${-((i * dur) / n).toFixed(2)}s`} path={path}
              {...(d.jam ? { keyPoints: "0;0.72;0.9;1", keyTimes: "0;0.3;0.85;1", calcMode: "linear" } : {})} />
          </circle>
        </g>
      ))}
    </>
  );
}

export const edgeTypes = { packet: PacketEdge };
