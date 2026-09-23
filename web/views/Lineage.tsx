"use client";

import { Background, BackgroundVariant, ReactFlow, type Edge, type Node } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useMemo } from "react";
import { nodeTypes, type DotData } from "@/components/flow/nodes";
import { GenomeStrip, KeyVal, LegendItem, Panel, Pill, Ticks } from "@/components/ui";
import { useWorkloadData } from "@/lib/derive";
import { fmtObjective, fmtValue } from "@/lib/format";
import { useUI } from "@/lib/store";
import { C, S_COLOR, S_LABEL, STRATEGIES, VERDICT } from "@/lib/theme";

const ORIGIN_COLOR: Record<string, string> = {
  random: "#7a8296", sampler: "#3987e5", "tpe-model": "#199e70", crossover: "#3987e5",
  "agent:hypothesis": "#ec4a70", "agent:mutation": "#a78bfa", "agent:crossover": "#c98500", "agent:explore": "#22d3ee",
};
const originColor = (o: string) => ORIGIN_COLOR[o.replace("+cached", "")] ?? "#7a8296";

export function LineageView() {
  const d = useWorkloadData();
  const { runId, selectRun, trialId, selectTrial } = useUI();

  const run = useMemo(() => {
    if (!d) return null;
    return d.wb.runs.find((r) => r.id === runId) ?? d.wb.runs.find((r) => r.strategy === "agent") ?? d.wb.runs[0];
  }, [d, runId]);

  const graph = useMemo(() => {
    if (!d || !run) return { nodes: [] as Node[], edges: [] as Edge[] };
    const q = d.quality;
    const trials = d.wb.trials.filter((t) => t.run_id === run.id);
    const vals = trials.filter((t) => t.metrics[q.name] != null).map((t) => t.metrics[q.name] as number);
    const lo = Math.min(...vals), hi = Math.max(...vals);
    const nodes: Node[] = trials.map((t) => {
      const v = t.metrics[q.name];
      const y = v == null ? 560 : 520 - ((v - lo) / (hi - lo || 1)) * 500;
      return {
        id: t.id, type: "dot", position: { x: t.number * 15, y }, draggable: false,
        data: { color: t.status === "failed" ? C.critical : originColor(t.origin), r: t.global_pareto ? 7 : t.pareto ? 6 : 4.5,
          ring: t.id === trialId ? "#ffffff" : t.pareto ? C.ink : null, dim: t.status !== "complete",
          label: t.pareto ? `#${t.number}` : null } satisfies DotData,
      };
    });
    for (let i = 0; i <= 4; i++) {  // y-axis guides
      const v = lo + ((hi - lo) * i) / 4;
      nodes.push({ id: `tick${i}`, type: "tick", position: { x: -70, y: 520 - (500 * i) / 4 - 7 }, draggable: false, selectable: false,
        data: { label: v.toFixed(3) } });
    }
    const ids = new Set(trials.map((t) => t.id));
    const edges: Edge[] = trials.flatMap((t) => t.parents.filter((p) => ids.has(p)).map((p) => ({
      id: `${p}->${t.id}`, source: p, target: t.id, type: "simplebezier",
      style: { stroke: originColor(t.origin), strokeOpacity: t.parents_inferred ? 0.18 : 0.45, strokeWidth: 1.2, strokeDasharray: t.parents_inferred ? "3 3" : undefined },
    })));
    return { nodes, edges };
  }, [d, run, trialId]);

  if (!d || !run) return null;
  const sel = trialId ? d.byId.get(trialId) : null;
  const selInRun = sel && sel.run_id === run.id ? sel : null;
  const parent = selInRun?.parents[0] ? d.byId.get(selInRun.parents[0]) : null;
  const hyp = selInRun?.hypothesis_id ? d.wb.hypotheses.find((h) => h.id === selInRun.hypothesis_id) : null;
  const origins = [...new Set(d.wb.trials.filter((t) => t.run_id === run.id).map((t) => t.origin.replace("+cached", "")))];

  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_360px]">
      <Panel tour="lineage" eyebrow={`Genome lineage · x = trial order, y = ${d.quality.label}`}
        title={`${S_LABEL[run.strategy]} · seed ${run.seed} · ${run.summary.n_trials} trials`}
        right={
          <select aria-label="Run" value={run.id} onChange={(e) => selectRun(e.target.value)}
            className="rounded-md border border-line bg-sunken px-2 py-1.5 text-[12px] text-ink">
            {STRATEGIES.flatMap((s) => d.wb.runs.filter((r) => r.strategy === s).map((r) => (
              <option key={r.id} value={r.id}>{S_LABEL[s]} · seed {r.seed} · HV {r.summary.final_hv.toFixed(3)}</option>
            )))}
          </select>
        }>
        <div className="mb-3 flex flex-wrap gap-x-4 gap-y-1.5">
          {origins.map((o) => <LegendItem key={o} shape="dot" color={originColor(o)} label={o} />)}
          <LegendItem shape="line" color={C.muted} label="dashed = parent inferred (NSGA-II)" />
        </div>
        <div className="h-[520px] w-full overflow-hidden rounded-lg border border-line bg-sunken">
          <ReactFlow nodes={graph.nodes} edges={graph.edges} nodeTypes={nodeTypes} fitView fitViewOptions={{ padding: 0.06 }}
            proOptions={{ hideAttribution: true }} nodesConnectable={false} onNodeClick={(_, n) => selectTrial(n.id)} minZoom={0.2}
            zoomOnScroll={false} preventScrolling={false}>
            <Background variant={BackgroundVariant.Dots} gap={26} size={1} color="#1c2030" />
          </ReactFlow>
        </div>
      </Panel>

      <div className="grid content-start gap-4">
        <Panel eyebrow="Selected genome" title={selInRun ? `Trial #${selInRun.number} · ${selInRun.origin}` : "Click a node"}>
          {selInRun ? (
            <div className="flex flex-col gap-3">
              <div className="flex flex-wrap gap-1.5">
                <Pill color={selInRun.status === "complete" ? (selInRun.feasible ? C.ok : C.warn) : C.critical}>{selInRun.status}{selInRun.status === "complete" && !selInRun.feasible ? " · infeasible" : ""}</Pill>
                {selInRun.global_pareto && <Pill color={C.ink}>global Pareto</Pill>}
                <Pill color={S_COLOR[selInRun.strategy]}>{S_LABEL[selInRun.strategy]}</Pill>
              </div>
              <GenomeStrip genome={selInRun.genome} space={d.wb.describe.space} height={12}
                highlight={parent ? new Set(Object.keys(selInRun.genome).filter((k) => selInRun.genome[k] !== parent.genome[k])) : undefined} />
              <div>{d.objectives.map((o) => <KeyVal key={o.name} k={o.label} v={
                <span>{fmtObjective(o, selInRun.metrics[o.name])}{parent && parent.metrics[o.name] != null && selInRun.metrics[o.name] != null && (
                  <span className="ml-1.5 text-muted">from {fmtObjective(o, parent.metrics[o.name])}</span>)}</span>} />)}</div>
              {parent && (
                <div>
                  <div className="eyebrow mb-1.5">Changed vs parent #{parent.number}</div>
                  {Object.keys({ ...parent.genome, ...selInRun.genome }).filter((k) => parent.genome[k] !== selInRun.genome[k]).map((k) => (
                    <div key={k} className="num flex justify-between gap-2 py-0.5 text-[12px]">
                      <span className="text-ink-2">{k}</span>
                      <span><span className="text-muted line-through">{fmtValue(parent.genome[k])}</span> → <span className="text-ink">{fmtValue(selInRun.genome[k])}</span></span>
                    </div>
                  ))}
                </div>
              )}
              {hyp && (
                <div className="rounded-md border border-line bg-sunken p-3">
                  <div className="mb-1 flex items-center justify-between"><span className="eyebrow">Hypothesis {hyp.id}</span><Pill color={VERDICT[hyp.verdict].color}>{VERDICT[hyp.verdict].label}</Pill></div>
                  <p className="text-[12px] leading-relaxed text-ink-2"><Ticks text={hyp.statement} /></p>
                </div>
              )}
              {selInRun.prune_reason && <p className="text-[12px] text-muted">Pruned: {selInRun.prune_reason}</p>}
            </div>
          ) : <p className="text-[12.5px] leading-relaxed text-muted">Each dot is one measured genome. Edges connect a child to the frontier genomes it was bred from — explicit for the agent (hypothesis edits, mutations, crossovers), nearest-neighbour inferred for NSGA-II.</p>}
        </Panel>
        <Panel eyebrow="Run summary" title={run.id}>
          <KeyVal k="Final hypervolume" v={run.summary.final_hv.toFixed(4)} />
          <KeyVal k="Complete / pruned / failed" v={`${run.summary.n_complete} / ${run.summary.n_pruned} / ${run.summary.n_failed}`} />
          <KeyVal k="Feasible · on run frontier" v={`${run.summary.n_feasible} · ${run.summary.n_pareto}`} />
          <KeyVal k="Wall time" v={`${(run.summary.wall_s / 60).toFixed(1)} min`} />
          <KeyVal k="Env hash · seed" v={`${run.provenance.env_hash} · ${run.seed}`} />
        </Panel>
      </div>
    </div>
  );
}
