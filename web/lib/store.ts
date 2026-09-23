"use client";

import { create } from "zustand";

export type View = "command" | "evolution" | "lineage" | "trials" | "insights" | "twin" | "incidents" | "architecture";
export const VIEW_IDS: View[] = ["command", "evolution", "lineage", "trials", "insights", "twin", "incidents", "architecture"];

export interface LogEvent { ts: number; kind: string; text: string }

interface UI {
  view: View;
  workload: string;
  trialId: string | null;
  runId: string | null;
  incidentId: string | null;
  objX: string | null;
  objY: string | null;
  events: LogEvent[];
  decisions: Record<string, "approved" | "rejected">; // replay-mode gate clicks (local only)
  paletteOpen: boolean;
  helpOpen: boolean;
  story: number | null;      // active tour step, null = tour off
  storyPaused: boolean;
  booted: boolean;
  setView: (v: View) => void;
  setWorkload: (w: string) => void;
  selectTrial: (id: string | null) => void;
  selectRun: (id: string | null) => void;
  selectIncident: (id: string | null) => void;
  setAxes: (x: string, y: string) => void;
  pushEvent: (e: LogEvent) => void;
  decide: (incident: string, d: "approved" | "rejected") => void;
  setPalette: (open: boolean) => void;
  setHelp: (open: boolean) => void;
  startStory: (step?: number) => void;
  setStory: (step: number | null) => void;
  toggleStoryPause: () => void;
  setBooted: () => void;
}

export const useUI = create<UI>((set) => ({
  view: "command",
  workload: "triage",
  trialId: null,
  runId: null,
  incidentId: null,
  objX: null,
  objY: null,
  events: [],
  decisions: {},
  paletteOpen: false,
  helpOpen: false,
  story: null,
  storyPaused: false,
  booted: false,
  setView: (view) => set({ view }),
  setWorkload: (workload) => set((s) => (s.workload === workload ? {} : { workload, trialId: null, runId: null, incidentId: null, objX: null, objY: null })),
  selectTrial: (trialId) => set({ trialId }),
  selectRun: (runId) => set({ runId }),
  selectIncident: (incidentId) => set({ incidentId }),
  setAxes: (objX, objY) => set({ objX, objY }),
  pushEvent: (e) => set((s) => ({ events: [e, ...s.events].slice(0, 200) })),
  decide: (incident, d) => set((s) => ({ decisions: { ...s.decisions, [incident]: d } })),
  setPalette: (paletteOpen) => set({ paletteOpen }),
  setHelp: (helpOpen) => set({ helpOpen }),
  startStory: (step = 0) => set({ story: step, storyPaused: false, paletteOpen: false, helpOpen: false }),
  setStory: (story) => set({ story }),
  toggleStoryPause: () => set((s) => ({ storyPaused: !s.storyPaused })),
  setBooted: () => set({ booted: true }),
}));

/* ---------- deep links: #/view?w=rag&inc=INC-1&trial=abc ---------- */

export function readHash(): Partial<Pick<UI, "view" | "workload" | "incidentId" | "trialId">> & { tour?: boolean } {
  if (typeof window === "undefined") return {};
  const raw = window.location.hash.replace(/^#\/?/, "");
  const [path, qs = ""] = raw.split("?");
  const p = new URLSearchParams(qs);
  const out: ReturnType<typeof readHash> = {};
  if (path && (VIEW_IDS as string[]).includes(path)) out.view = path as View;
  if (p.get("w")) out.workload = p.get("w")!;
  if (p.get("inc")) out.incidentId = p.get("inc");
  if (p.get("trial")) out.trialId = p.get("trial");
  if (p.has("tour")) out.tour = true;
  return out;
}

export function hashFor(s: Pick<UI, "view" | "workload" | "incidentId" | "trialId">): string {
  const p = new URLSearchParams();
  p.set("w", s.workload);
  if (s.view === "incidents" && s.incidentId) p.set("inc", s.incidentId);
  if (s.view === "trials" && s.trialId) p.set("trial", s.trialId);
  return `#/${s.view}?${p.toString()}`;
}
