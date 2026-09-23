// Mirrors epoch/export.py — the one data contract for replay (static bundle) and live (/api/bundle).

export type Direction = "min" | "max";
export type Strategy = "random" | "tpe" | "nsga2" | "agent";
export type Genome = Record<string, string | number | boolean>;
export type Metrics = Record<string, number | null>;

export interface Objective { name: string; direction: Direction; label: string; unit: string; lo: number; hi: number; log: boolean }
export interface Constraint { metric: string; op: "<=" | ">="; threshold: number; label: string }
export interface Gene {
  name: string; kind: "cat" | "int" | "float"; choices: (string | number | boolean)[]; low: number | null; high: number | null;
  log: boolean; step: number | null; default: unknown; group: string; doc: string; when: [string, unknown[]] | null;
}

export interface Trial {
  id: string; run_id: string; number: number; strategy: Strategy; seed: number; group_id: string | null;
  genome_id: string; genome: Genome; origin: string; generation: number; status: "complete" | "pruned" | "failed";
  metrics: Metrics; violations: number[]; feasible: boolean; pareto: boolean; global_pareto: boolean; fidelity: number;
  duration_s: number; hv_after: number; parents: string[]; parents_inferred: boolean; hypothesis_id: string | null;
  latency_hist: { edges: number[]; counts: number[] } | null; stage_ms: Record<string, number> | null;
  prune_reason: string | null; error_kind: string | null; error: string | null; created_at: string;
}

export interface Run {
  id: string; strategy: Strategy; seed: number; group_id: string | null; budget: number; created_at: string;
  finished_at: string | null; provenance: Provenance;
  summary: { final_hv: number; hv_curve: number[]; n_trials: number; n_complete: number; n_pruned: number; n_failed: number;
    n_feasible: number; n_pareto: number; knee: string | null; compute_saved_s: number; wall_s: number;
    best: Record<string, { value: number; trial_id: string }> };
}

export interface Provenance {
  git_sha: string | null; env_hash: string; python: string; platform: string; host: string; device: string;
  gpu: string | null; cpu_count: number; versions: Record<string, string>; seed: number; created_at: string;
}

export interface Hypothesis {
  id: string; run_id: string; created_at: string; source: "llm" | "heuristic"; statement: string; mechanism: string;
  target: string; genes: string[]; expected: Record<string, "improve" | "worsen" | "unchanged">;
  proposals: { genome_id: string; parent: string | null; changes: Record<string, [unknown, unknown]>; predicted: number[] | null; ehvi: number | null }[];
  verdict: "pending" | "confirmed" | "refuted" | "inconclusive";
  evidence: { generation?: number; support?: number; joined_frontier?: number; model?: string | null; llm_cost_usd?: number | null;
    tests?: { genome_id: string; parent?: string; status: string; feasible?: boolean; delta?: Record<string, number>; raw?: Record<string, [number, number]> }[] };
}

export interface Failure { id: number; run_id: string; trial_id: string | null; kind: string; message: string; genome: Genome; created_at: string }

export interface TopoNode { id: string; label: string; kind: "io" | "transform" | "gate" | "model" | "llm" | "store"; sub?: string | null; stage?: string | null }
export interface TopoEdge { source: string; target: string; share: number; label?: string }
export interface Topology { trial_id: string; genome_id: string; genome: Genome; metrics: Metrics; graph: { nodes: TopoNode[]; edges: TopoEdge[] } }

export interface RaceStrategy {
  runs: number; final_hv: number[]; final_hv_median: number; trials_to_target: (number | null)[]; trials_to_target_median: number;
  reached_target: number; trials_to_nsga2_final: (number | null)[]; trials_to_nsga2_final_median: number | null; auc: number; mean_curve: number[];
}
export interface Race {
  group_id: string; reference_hv: number; target_hv: number; target_frac: number; nsga2_median_final_hv: number | null; budget: number;
  strategies: Partial<Record<Strategy, RaceStrategy>>; seeds: number[]; wall_s: number;
  headline?: { agent_trials_to_nsga2_final: number; nsga2_trials_to_own_final: number; pct_fewer_trials: number };
  curves: Record<string, { strategy: Strategy; seed: number; hv: number[] }>;
}

export interface Ablation {
  chosen: Genome; reference: Genome; base: Record<string, number>; noise: Record<string, number>; repeats: number;
  effects: { gene: string; from: unknown; to: unknown; status: string; metrics?: Record<string, number>; delta?: Record<string, number>; significant?: Record<string, boolean>; error_kind?: string }[];
}

export interface ServiceProfile { samples_ms: number[]; alpha: number; overhead_ms: number; network_ms: number; label: string; batch_size?: number; trial_id?: string; metrics?: Metrics }
export interface TwinSeriesPoint { t: number; rps_in: number; rps_out: number; p50: number | null; p95: number | null; p99: number | null; queue: number; util: number; drops: number; timeouts: number }
export interface TwinSummary { p50: number | null; p95: number | null; p99: number | null; throughput_rps: number; offered_rps: number; drop_rate: number; timeout_rate: number; utilization: number; mean_batch: number; max_queue: number; slo_attainment: number | null; stable: boolean }
export interface TwinResult { summary: TwinSummary; series: TwinSeriesPoint[]; config: Record<string, unknown>; traffic: Record<string, unknown>; faults: Record<string, unknown>[] }
export interface TwinValidationLevel { utilization_target: number; rate_rps: number; n: number; client_lag_p95_ms?: number; real: { p50: number; p95: number; p99: number }; twin: { p50: number; p95: number; p99: number; utilization: number }; err_p50: number; err_p95: number }
export interface TwinPack {
  profiles: Record<string, ServiceProfile>; knee_profile: ServiceProfile; capacity_rps: number;
  scenarios: (TwinResult & { key: string; title: string })[];
  validation?: { levels: TwinValidationLevel[]; mape_p50: number; mape_p95: number; capacity_measured_rps: number; capacity_rps: number; unloaded_p50_ms: number; duration_s: number; service_scale: number; error?: string };
}

export interface WindowMetrics extends Metrics { }
export interface IncidentWindow {
  w: number; t_min: number; live: Record<string, number | Record<string, number>>; counterfactual: Record<string, number | Record<string, number>> | null;
  drift_frac: number; fault: Record<string, number>; cusum: Record<string, number>; phase: "baseline" | "fault" | "incident" | "shadow" | "remediated" | "rolled_back";
}
export interface Candidate {
  genome_id: string; origin: string; why: string; diff: Record<string, [unknown, unknown]>; genome?: Genome;
  metrics: Record<string, number | Record<string, number>> | null; meets_targets: boolean; shortfall?: number; error?: string;
}
export interface Cause { kind: string; title: string; score: number; evidence: string[]; stage?: string }
export interface Incident {
  id: string; workload: string; created_at: string; status: string; title: string; severity: string;
  data: {
    scenario: { name: string; title: string; kind: string; windows: number; window_size: number; fault_at: number; spike_ms: number; drift_mix: number; stage: string | null; audit_budget: number };
    series: IncidentWindow[]; timeline: { w: number; event: string; detail: string; seconds?: number }[];
    detectors: Record<string, { mu: number; sigma: number; k: number; h: number; direction: string; history: number[] }>;
    deployed: Genome; alarm_w?: number; rca_w?: number; gate_w?: number; promoted_w?: number; recovered_w?: number | null;
    targets?: Record<string, number>; baseline?: Record<string, number>; audited?: number; bank?: string | null;
    evidence?: { stages: Record<string, { baseline_ms: number; incident_ms: number; delta_ms: number; ratio: number | null }>; shifts: Record<string, { baseline: number | null; incident: number | null; delta: number | null }>; similar_incidents: { id: string; title: string; status: string }[] };
    causes?: Cause[]; narrative?: { summary: string; primary_cause: string; source: string; model?: string };
    candidates?: Candidate[];
    twin?: { traffic: { kind: string; rate: number }; slo_ms: number; results: { genome_id: string; p50: number | null; p95: number | null; p99: number | null; utilization: number; stable: boolean; capacity_rps: number; replicas_needed: number | null; p95_at_plan: number | null }[] };
    proposal?: { status: string; genome_id?: string; genome?: Genome; origin?: string; why?: string; diff?: Record<string, [unknown, unknown]>; canary?: Record<string, number>; incumbent_canary?: Record<string, number>; replicas?: number | null; incumbent_replicas_needed?: number | null; meets_targets?: boolean; expected_gain?: Record<string, number> };
    decision?: { approved: boolean; actor: string; note: string };
    outcome?: { status: string; detection_delay_windows: number; time_to_mitigate_windows: number; live_mean: Record<string, number | null>; counterfactual_mean: Record<string, number | null> } | string;
    prod_rps?: number; twin_slo_ms?: number; rca_seconds?: number; archive_size?: number;
  };
}

export interface Deployment { id: number; ts: string; workload: string; genome_id: string; genome: Genome; reason: string; incident_id: string | null; status: string }

export interface WorkloadBundle {
  describe: { name: string; title: string; description: string; quality_metric: string; objectives: Objective[]; constraints: Constraint[]; space: Gene[]; capabilities: Record<string, unknown> };
  runs: Run[]; trials: Trial[]; pareto: string[]; global_hv: number; knee: string | null; topology: Topology | null;
  deployed_topology?: Topology | null;
  hypotheses: Hypothesis[]; failures: Failure[]; incidents: Incident[]; deployments: Deployment[];
  race?: Race; attribution?: { fanova: Record<string, Record<string, number>>; ablation: Ablation; knee: string }; twin?: TwinPack;
}

export interface Bundle {
  meta: { generated_at: string; epoch_version: string; mode: "replay" | "live"; provenance: Provenance; llm: { enabled: boolean; agent_model: string | null; judge_model: string | null }; workloads: string[]; notes: string[] };
  workloads: Record<string, WorkloadBundle>;
}
