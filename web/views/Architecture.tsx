"use client";

import { Background, BackgroundVariant, MarkerType, ReactFlow, type Edge, type Node } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { nodeTypes, type GroupData, type PipeData } from "@/components/flow/nodes";
import { KeyVal, Panel, Segmented } from "@/components/ui";
import { edgeTypes, type PacketData } from "@/components/flow/PacketEdge";
import { useEffect, useMemo, useState } from "react";
import { useBundle } from "@/lib/data";
import { C } from "@/lib/theme";

const G = (id: string, label: string, color: string, x: number, y: number, w: number, h: number): Node => ({
  id, type: "group", position: { x, y }, data: { label, color } satisfies GroupData, style: { width: w, height: h }, draggable: false, selectable: false, zIndex: -1,
});
const N = (id: string, parent: string, x: number, y: number, label: string, sub: string, kind: string, accent?: string): Node => ({
  id, type: "pipe", parentId: parent, extent: "parent", position: { x, y }, data: { label, sub, kind, accent } satisfies PipeData, draggable: false,
});

const nodes: Node[] = [
  G("g-ui", "Experience", C.info, 0, 0, 230, 330),
  N("web", "g-ui", 20, 40, "Mission control", "Next.js 15 · React Flow · Recharts", "io", C.info),
  N("replay", "g-ui", 20, 150, "Replay bundle", "static JSON · Vercel", "store", C.info),
  N("cli", "g-ui", 20, 250, "epoch CLI", "Typer · demo / race / heal", "io", C.info),

  G("g-api", "Control plane", C.crimson, 270, 0, 230, 330),
  N("api", "g-api", 20, 40, "FastAPI", "REST · SSE event stream", "server", C.crimson),
  N("gate", "g-api", 20, 150, "Approval gate", "human-in-the-loop promotion", "gate", C.warn),
  N("prom", "g-api", 20, 250, "Prometheus", "/metrics · OTel spans · NVML", "store", C.crimson),

  G("g-work", "Execution · GPU worker (Celery)", C.violet, 540, 0, 460, 330),
  N("evo", "g-work", 20, 40, "Evolution engine", "Optuna NSGA-II / MOTPE · ask/tell", "model", C.violet),
  N("agent", "g-work", 240, 40, "Hypothesis agent", "LangGraph · surrogate EHVI screen", "llm"),
  N("bench", "g-work", 20, 150, "Benchmark engine", "quality · p50/p95/p99 · VRAM · $", "gate", C.violet),
  N("rca", "g-work", 240, 150, "RCA agent", "LangGraph · canary · twin · gate", "llm"),
  N("runtime", "g-work", 20, 250, "Inference runtimes", "PyTorch · compile · ONNX RT · bnb", "model", C.violet),
  N("twin", "g-work", 240, 250, "Digital twin", "SimPy DES · capacity planner", "transform", C.violet),

  G("g-mem", "Memory", C.ok, 1040, 0, 230, 330),
  N("pg", "g-mem", 20, 40, "Postgres + pgvector", "trials · lineage · failures", "store", C.ok),
  N("redis", "g-mem", 20, 150, "Redis", "Celery broker · event fan-out", "store", C.ok),
  N("mlflow", "g-mem", 20, 250, "MLflow", "runs · params · artifacts", "store", C.ok),

  G("g-ext", "Models", C.warn, 540, 370, 460, 130),
  N("claude", "g-ext", 20, 40, "Claude API", "hypotheses · RCA · judge · tier-3", "llm"),
  N("hf", "g-ext", 240, 40, "Hugging Face", "Qwen2.5 · bge · cross-encoders", "model", C.warn),
];

const SRC_COLOR: Record<string, string> = {
  web: C.info, replay: C.info, cli: C.info, api: C.crimsonHi, gate: C.warn, prom: C.crimsonHi,
  evo: C.violet, agent: C.violet, bench: C.violet, rca: C.crimsonHi, runtime: C.violet, twin: C.violet,
  pg: C.ok, redis: C.ok, mlflow: C.ok, claude: C.violet, hf: C.warn,
};
type Flow = "all" | "search" | "healing" | "data";
const FLOWS: Record<Exclude<Flow, "all">, { label: string; edges: string[]; text: string }> = {
  search: { label: "Search loop", text: "CLI/API → evolution engine asks the hypothesis agent for genomes → benchmark on real runtimes → results + failure memory to Postgres.",
    edges: ["cli-evo", "api-evo", "evo-agent", "agent-evo", "evo-bench", "bench-runtime", "runtime-hf", "bench-pg", "agent-pg", "agent-claude", "evo-redis", "redis-api", "web-api"] },
  healing: { label: "Healing loop", text: "Prometheus metrics feed CUSUM alarms → RCA agent diagnoses, canaries fixes and asks the twin → human gate approves → promote or roll back.",
    edges: ["bench-prom", "prom-rca", "rca-claude", "rca-twin", "rca-pg", "api-gate", "gate-rca", "web-api", "redis-api"] },
  data: { label: "Data path", text: "Every trial lands in Postgres + MLflow with provenance; events fan out over Redis/SSE; the replay bundle is exported from the same store.",
    edges: ["bench-pg", "evo-mlflow", "evo-redis", "redis-api", "web-api", "replay-web", "bench-prom"] },
};

const E = (s: string, t: string, label?: string, hot = false, flow: Flow = "all"): Edge => {
  const id = `${s}-${t}`;
  const on = flow === "all" ? true : FLOWS[flow].edges.includes(id);
  const color = SRC_COLOR[s] ?? C.info;
  const live = flow === "all" ? hot : on;
  return {
    id, source: s, target: t, label: on ? label : undefined, type: "packet",
    labelStyle: { fill: live ? color : "#a9b0bf", fontSize: 10, fontFamily: "var(--font-mono)" }, labelBgStyle: { fill: "#0c0e13" }, labelBgPadding: [4, 2], labelBgBorderRadius: 3,
    style: { stroke: live ? color : "#343a4d", strokeWidth: live ? 1.8 : 1.2, strokeOpacity: on ? (live ? 0.55 : 1) : 0.25, transition: "stroke-opacity .4s, stroke .4s" },
    markerEnd: { type: MarkerType.ArrowClosed, color: live ? color : "#343a4d", width: 12, height: 12 },
    data: { packets: live ? 2 : on ? 1 : 0, color, dur: live ? 1.9 : 3.2, smooth: true, glow: live, r: live ? 2.6 : 1.8 } satisfies PacketData,
  };
};
const EDGE_SPEC: [string, string, string?, boolean?][] = [
  ["web", "api", "REST + SSE", true], ["replay", "web"], ["cli", "evo"], ["api", "evo", "Celery"], ["api", "gate"],
  ["gate", "rca", "approve"], ["evo", "agent", "propose"], ["agent", "evo", "genomes"], ["evo", "bench", "trial", true], ["bench", "runtime"],
  ["rca", "twin", "what-if"], ["bench", "pg", "results", true], ["agent", "pg", "memory"], ["rca", "pg", "incidents"], ["evo", "redis", "events"],
  ["redis", "api", "fan-out"], ["evo", "mlflow"], ["bench", "prom", "metrics"], ["prom", "rca", "alarms"], ["agent", "claude"], ["rca", "claude"], ["runtime", "hf"],
];

const STACK: [string, string, string][] = [
  ["Search", "Optuna NSGA-II · MOTPE · random, pymoo hypervolume, RF surrogate (EHVI pre-screen), multi-fidelity rejection", "epoch/evolution"],
  ["Agents", "LangGraph state machines, Claude via forced tool-use (structured output), offline heuristic fallback", "epoch/agents"],
  ["Genome", "typed conditional search space; one encoding drives Optuna, agent edits, surrogate features and UI diffs", "epoch/genome.py"],
  ["Benchmark", "quality · p50/p95/p99 · throughput · peak memory (CUDA / tracemalloc) · $/1k requests · robustness split", "epoch/benchmark"],
  ["Inference", "PyTorch eager / torch.compile, bitsandbytes int8 / nf4, ONNX Runtime (+ int8 dynamic quantisation)", "epoch/workloads"],
  ["Attribution", "fANOVA importance per objective; interventional one-gene ablations with 2σ noise bands", "epoch/attribution.py"],
  ["Twin", "SimPy discrete-event sim, measured service profiles, capacity planner, open-loop load-test validation", "epoch/twin"],
  ["Healing", "CUSUM detectors, drift signals (JS / OOV), RCA agent, shadow canary, error-bank retrain, rollback", "epoch/healing"],
  ["Memory", "SQLAlchemy · Postgres + pgvector (SQLite fallback) · MLflow mirror · provenance (git, env hash, seed)", "epoch/memory"],
  ["Serving / ops", "FastAPI micro-batcher, SSE, Celery + Redis, Prometheus, OpenTelemetry, NVML, Docker Compose, GitHub Actions", "epoch/api"],
];

export function ArchitectureView() {
  const { data } = useBundle();
  const p = data?.meta.provenance;
  const [flow, setFlow] = useState<Flow>("search");
  const [auto, setAuto] = useState(true);
  useEffect(() => {
    if (!auto) return;
    const order: Flow[] = ["search", "healing", "data"];
    const id = setInterval(() => setFlow((f) => order[(order.indexOf(f) + 1) % order.length]), 6500);
    return () => clearInterval(id);
  }, [auto]);
  const edges = useMemo(() => EDGE_SPEC.map(([s, t, l, h]) => E(s, t, l, h, flow)), [flow]);
  return (
    <div className="grid gap-4">
      <Panel tour="architecture" eyebrow="How EPOCH is built · packets = live traffic on that path" title="Platform architecture" pad={false}
        right={<Segmented size="xs" value={flow} onChange={(v) => { setAuto(false); setFlow(v); }}
          options={[{ value: "search", label: "Search loop" }, { value: "healing", label: "Healing loop" }, { value: "data", label: "Data path" }, { value: "all", label: "All" }]} />}>
        {flow !== "all" && (
          <div className="flex items-center gap-2 border-b border-line bg-sunken/60 px-4 py-2 text-[12px] text-ink-2">
            <span className="live-dot h-1.5 w-1.5 rounded-full bg-crimson-hi" />
            <span className="whitespace-nowrap font-medium text-ink">{FLOWS[flow].label}</span><span className="text-muted">·</span><span className="min-w-0">{FLOWS[flow].text}</span>
            {auto && <span className="num ml-auto hidden shrink-0 text-[10.5px] text-muted sm:inline">auto-cycling · click a tab to pin</span>}
          </div>
        )}
        <div className="h-[560px] w-full">
          <ReactFlow nodes={nodes} edges={edges} nodeTypes={nodeTypes} edgeTypes={edgeTypes} fitView fitViewOptions={{ padding: 0.06 }} proOptions={{ hideAttribution: true }}
            nodesConnectable={false} zoomOnScroll={false} preventScrolling={false} minZoom={0.3}>
            <Background variant={BackgroundVariant.Dots} gap={18} size={1} color="#1c2030" />
          </ReactFlow>
        </div>
      </Panel>
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_380px]">
        <Panel eyebrow="Stack" title="What each layer uses" pad={false}>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[640px] text-[12.5px]">
              <tbody>
                {STACK.map(([k, v, path]) => (
                  <tr key={k} className="border-b border-line/70">
                    <td className="w-[120px] px-4 py-2.5 align-top font-medium text-ink">{k}</td>
                    <td className="px-4 py-2.5 leading-relaxed text-ink-2">{v}</td>
                    <td className="num whitespace-nowrap px-4 py-2.5 text-right align-top text-[11px] text-muted">{path}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
        <Panel eyebrow="Provenance of this data" title={data?.meta.mode === "live" ? "Live experiment memory" : "Replay bundle"}>
          {p && (
            <>
              <KeyVal k="Generated" v={new Date(data!.meta.generated_at).toLocaleString()} />
              <KeyVal k="Device" v={p.gpu ?? `${p.device} · ${p.cpu_count} cores`} />
              <KeyVal k="Platform" v={p.platform.split("-").slice(0, 2).join(" ")} />
              <KeyVal k="Python" v={p.python} />
              {Object.entries(p.versions).map(([k, v]) => <KeyVal key={k} k={k} v={v} />)}
              <KeyVal k="Env hash" v={p.env_hash} />
              <KeyVal k="LLM agents" v={data!.meta.llm.enabled ? data!.meta.llm.agent_model : "heuristic (no API key)"} />
              <ul className="mt-3 space-y-1.5">
                {data!.meta.notes.map((n) => <li key={n} className="text-[11.5px] leading-relaxed text-muted">{n}</li>)}
              </ul>
            </>
          )}
        </Panel>
      </div>
    </div>
  );
}
