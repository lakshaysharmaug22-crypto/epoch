import type { Strategy } from "./types";

export const C = {
  bg: "#08090c",
  surface: "#101218",
  raised: "#161923",
  line: "#1f2330",
  lineStrong: "#2b3040",
  grid: "#1a1d27",
  ink: "#e8eaf0",
  ink2: "#a9b0bf",
  muted: "#7a8296",
  crimson: "#e11d48",
  crimsonHi: "#ff4d6d",
  critical: "#f43f5e",
  criticalDeep: "#7f1d1d",
  ok: "#34d399",
  warn: "#fbbf24",
  info: "#22d3ee",
  violet: "#a78bfa",
} as const;

// fixed order: random, tpe, nsga2, agent — validated adjacent-pair palette on the #101218 surface
export const STRATEGIES: Strategy[] = ["random", "tpe", "nsga2", "agent"];
export const S_COLOR: Record<Strategy, string> = { random: "#c98500", tpe: "#199e70", nsga2: "#3987e5", agent: "#ec4a70" };
export const S_LABEL: Record<Strategy, string> = { random: "Random", tpe: "MOTPE", nsga2: "NSGA-II", agent: "EPOCH agent" };
export type Shape = "circle" | "square" | "triangle" | "diamond";
export const S_SHAPE: Record<Strategy, Shape> = { random: "square", tpe: "triangle", nsga2: "diamond", agent: "circle" };

export const KIND_COLOR: Record<string, string> = {
  io: "#a9b0bf", transform: "#22d3ee", gate: "#fbbf24", model: "#3987e5", llm: "#a78bfa", store: "#c98500",
};

export const GROUP_COLOR: Record<string, string> = {
  preprocess: "#22d3ee", tier1: "#fbbf24", tier2: "#3987e5", tier3: "#a78bfa", runtime: "#ec4a70",
  retrieval: "#199e70", model: "#a78bfa", general: "#7a8296",
};

export const VERDICT: Record<string, { color: string; label: string }> = {
  confirmed: { color: "#34d399", label: "Confirmed" },
  refuted: { color: "#f43f5e", label: "Refuted" },
  inconclusive: { color: "#7a8296", label: "Inconclusive" },
  pending: { color: "#fbbf24", label: "Pending" },
};

export const INCIDENT_STATUS: Record<string, { color: string; label: string }> = {
  resolved: { color: "#34d399", label: "Resolved" },
  mitigated: { color: "#22d3ee", label: "Mitigated" },
  awaiting_approval: { color: "#fbbf24", label: "Awaiting approval" },
  rolled_back: { color: "#f43f5e", label: "Rolled back" },
  rejected: { color: "#7a8296", label: "Rejected" },
  detected: { color: "#f43f5e", label: "Detected" },
  no_candidate: { color: "#f43f5e", label: "No safe fix" },
  escalated: { color: "#fbbf24", label: "Escalated" },
  no_effect: { color: "#7a8296", label: "No effect" },
  not_detected: { color: "#7a8296", label: "Not detected" },
};
