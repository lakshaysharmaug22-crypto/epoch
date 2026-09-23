"use client";

import { useMemo } from "react";
import { useBundle } from "./data";
import { useUI } from "./store";
import type { Bundle, Objective, Trial, WorkloadBundle } from "./types";

export interface Derived {
  bundle: Bundle;
  wb: WorkloadBundle;
  name: string;
  objectives: Objective[];
  quality: Objective;
  byId: Map<string, Trial>;
  frontier: Trial[];
  knee: Trial | null;
  complete: Trial[];
}

export function useWorkloadData(): Derived | null {
  const { data } = useBundle();
  const workload = useUI((s) => s.workload);
  return useMemo(() => {
    if (!data) return null;
    const name = data.workloads[workload] ? workload : data.meta.workloads[0];
    const wb = data.workloads[name];
    if (!wb) return null;
    const byId = new Map(wb.trials.map((t) => [t.id, t]));
    const objectives = wb.describe.objectives;
    const quality = objectives.find((o) => o.name === wb.describe.quality_metric) ?? objectives[0];
    const frontier = wb.pareto.map((id) => byId.get(id)).filter(Boolean) as Trial[];
    frontier.sort((a, b) => (b.metrics[quality.name] ?? 0) - (a.metrics[quality.name] ?? 0));
    const knee = wb.knee ? byId.get(wb.knee) ?? null : null;
    const complete = wb.trials.filter((t) => t.status === "complete");
    return { bundle: data, wb, name, objectives, quality, byId, frontier, knee, complete };
  }, [data, workload]);
}
