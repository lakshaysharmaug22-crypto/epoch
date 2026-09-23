"use client";

import { Pause, Play, RotateCcw } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { WindowChart } from "@/components/charts/Charts";
import { TopologyFlow } from "@/components/flow/Topology";
import { Button, GenomeStrip, KeyVal, Panel, Pill, Stat, Ticks } from "@/components/ui";
import { useWorkloadData } from "@/lib/derive";
import { fmtDuration, fmtMs, fmtNum, fmtObjective, fmtPct } from "@/lib/format";
import { useUI } from "@/lib/store";
import { C, INCIDENT_STATUS, S_COLOR, S_LABEL, VERDICT } from "@/lib/theme";
import type { Incident, WorkloadBundle } from "@/lib/types";

const num = (v: unknown) => (typeof v === "number" ? v : null);

export function CommandView() {
  const d = useWorkloadData();
  const { events, setView, selectIncident } = useUI();
  const incidents = d?.wb.incidents ?? [];
  const tape = incidents.at(-1);
  const [w, setW] = useState(0);
  const [playing, setPlaying] = useState(true);
  const maxW = tape ? tape.data.series.at(-1)!.w : 0;

  useEffect(() => {
    if (!playing || !tape) return;
    const id = setInterval(() => setW((x) => (x >= maxW ? 0 : x + 1)), 650);
    return () => clearInterval(id);
  }, [playing, tape, maxW]);

  const log = useMemo(() => (d ? buildLog(d.wb) : []), [d]);
  if (!d) return null;
  const { wb, objectives, quality, knee, frontier } = d;
  const runs = wb.runs;
  const n = wb.trials.length;
  const nPruned = wb.trials.filter((t) => t.status === "pruned").length;
  const nFailed = wb.trials.filter((t) => t.status === "failed").length;
  const saved = runs.reduce((a, r) => a + (r.summary.compute_saved_s ?? 0), 0);
  const wall = runs.reduce((a, r) => a + (r.summary.wall_s ?? 0), 0);
  const head = wb.race?.headline;
  const val = wb.twin?.validation;
  const healed = incidents.filter((i) => ["resolved", "mitigated"].includes(i.status)).length;
  const otherWithTape = d.bundle.meta.workloads.find((w) => (d.bundle.workloads[w]?.incidents.length ?? 0) > 0);

  const cur = tape?.data.series.find((s) => s.w === w) ?? tape?.data.series[0];
  const status = tape && cur ? nodeStatus(tape, cur.phase, cur.w) : undefined;
  const liveStage = cur ? (cur.live.stage_ms as Record<string, number> | undefined) : undefined;
  const series = tape ? tape.data.series.filter((s) => s.w <= w) : [];

  return (
    <div className="grid gap-4">
      {/* KPI strip */}
      <div data-tour="command-kpi" className="panel grid grid-cols-2 gap-x-6 gap-y-5 p-5 sm:grid-cols-3 xl:grid-cols-6">
        <Stat big label="Frontier hypervolume" value={fmtNum(wb.global_hv, 3)} sub={`${frontier.length} Pareto-optimal configs`} />
        <Stat label="Configs measured" value={n.toLocaleString()} sub={`${nPruned} pruned early · ${nFailed} failed`} />
        <Stat label="Compute saved" value={fmtDuration(saved)} sub={`multi-fidelity rejection · ${fmtPct(saved / (wall + saved || 1), 0)} of search`} />
        <Stat label="Agent vs NSGA-II" value={head ? `${head.pct_fewer_trials > 0 ? "−" : "+"}${Math.abs(head.pct_fewer_trials)}%` : "—"}
          tone={head ? (head.pct_fewer_trials > 0 ? C.ok : C.critical) : undefined}
          sub={head ? `trials to NSGA-II's final HV (${head.agent_trials_to_nsga2_final} vs ${head.nsga2_trials_to_own_final})` : "run a race"} />
        <Stat label="Twin error · p95" value={val && !val.error ? fmtPct(val.mape_p95, 0) : "—"} sub={val && !val.error ? `p50 ${fmtPct(val.mape_p50, 0)} · vs real HTTP load test` : "no validation yet"} />
        <Stat label="Incidents healed" value={incidents.length ? `${healed}/${incidents.length}` : "—"} sub={incidents.length ? "detect → RCA → canary → gate" : "healing scenarios recorded on triage"} />
      </div>

      <Panel tour="command-topology" eyebrow="Deployed pipeline · live topology · packets = requests, width = traffic share" title={`${wb.describe.title}${tape ? ` · replaying ${tape.id}` : ""}`}
        right={tape && (
          <div className="flex items-center gap-2">
            <span className="num text-[11px] text-muted">ops tape · minute {w}/{maxW}</span>
            <Button tone="ghost" onClick={() => setPlaying((p) => !p)}>{playing ? <Pause size={13} /> : <Play size={13} />}{playing ? "Pause" : "Play"}</Button>
            <Button tone="ghost" onClick={() => setW(0)}><RotateCcw size={13} /></Button>
          </div>
        )}>
        {(wb.deployed_topology ?? wb.topology) ? (
          <TopologyFlow topo={(wb.deployed_topology ?? wb.topology)!} stageMs={liveStage ?? knee?.stage_ms} status={status} height={250} />
        ) : null}
        {tape && cur && (
          <div className="mt-3 flex flex-wrap items-center gap-2 text-[12px]">
            <PhaseChip phase={cur.phase} />
            <span className="text-ink-2"><Ticks text={phaseText(tape, cur.phase)} /></span>
            <button className="ml-auto text-[12px] text-crimson-hi underline-offset-2 hover:underline" onClick={() => { selectIncident(tape.id); setView("incidents"); }}>
              Open {tape.id} →
            </button>
          </div>
        )}
      </Panel>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_340px]">
        {tape ? (
          <>
        <Panel eyebrow="Production · quality · solid = live, dashed = old config (counterfactual)" title={`${quality.label} per minute`}>
          {tape ? <WindowChart data={series.map((s) => ({ w: s.w, live: num(s.live[quality.name]), cf: s.counterfactual ? num(s.counterfactual[quality.name]) : null }))}
            metric={quality.name} label={quality.label} format={(v) => fmtNum(v, 3)} faultFrom={tape.data.scenario.fault_at <= w ? tape.data.scenario.fault_at : undefined}
            alarm={tape.data.alarm_w !== undefined && tape.data.alarm_w <= w ? tape.data.alarm_w : undefined} xMax={maxW}
            promoted={tape.data.promoted_w !== undefined && tape.data.promoted_w <= w ? tape.data.promoted_w : undefined} betterUp height={190} /> : <div className="text-[12px] text-muted">Run a healing scenario.</div>}
        </Panel>
        <Panel eyebrow="Production · latency" title="p95 per minute">
          {tape && <WindowChart data={series.map((s) => ({ w: s.w, live: num(s.live.p95_ms), cf: s.counterfactual ? num(s.counterfactual.p95_ms) : null }))}
            metric="p95_ms" label="p95" format={fmtMs} faultFrom={tape.data.scenario.fault_at <= w ? tape.data.scenario.fault_at : undefined}
            alarm={tape.data.alarm_w !== undefined && tape.data.alarm_w <= w ? tape.data.alarm_w : undefined} xMax={maxW}
            promoted={tape.data.promoted_w !== undefined && tape.data.promoted_w <= w ? tape.data.promoted_w : undefined} betterUp={false} height={190} />}
        </Panel>
          </>
        ) : (
          <Panel eyebrow="Production replay" title="Self-healing incidents were recorded on the triage deployment" className="xl:col-span-2">
            <p className="max-w-[70ch] text-[12.5px] leading-relaxed text-ink-2">
              Drift, resolver-outage and model-server-slowdown faults were injected into the triage cascade and handled end to end:
              CUSUM detection, RCA, 3× shadow canaries, a twin capacity check and a human approval gate.
            </p>
            {otherWithTape && (
              <div className="mt-3"><Button tone="primary" onClick={() => { const s = useUI.getState(); s.setWorkload(otherWithTape); s.setView("incidents"); }}>
                Open triage incidents →</Button></div>
            )}
          </Panel>
        )}
        <Panel eyebrow="Recommended configuration · knee of the frontier" title={knee ? `Trial ${knee.number} · ${S_LABEL[knee.strategy]} · seed ${knee.seed}` : "—"}>
          {knee && (
            <div className="flex flex-col gap-3">
              <GenomeStrip genome={knee.genome} space={wb.describe.space} height={14} />
              <div>
                {objectives.map((o) => <KeyVal key={o.name} k={`${o.label}${o.direction === "max" ? " ↑" : " ↓"}`} v={fmtObjective(o, knee.metrics[o.name])} />)}
                <KeyVal k="Throughput" v={`${fmtNum(knee.metrics.throughput_rps)} req/s`} />
                <KeyVal k="Robustness (perturbed / clean)" v={fmtPct(knee.metrics.robustness)} />
                {knee.metrics.share_t1 != null && <KeyVal k="Resolved at tier 1 / 2 / 3" v={`${fmtPct(knee.metrics.share_t1, 0)} · ${fmtPct(knee.metrics.share_t2, 0)} · ${fmtPct(knee.metrics.share_t3, 0)}`} />}
              </div>
              <p className="text-[11.5px] leading-relaxed text-muted">
                Closest frontier point to the ideal corner in normalised objective space, chosen from{" "}
                <span className="num text-ink-2">{frontier.length}</span> Pareto configs across <span className="num text-ink-2">{runs.length}</span> runs.
              </p>
            </div>
          )}
        </Panel>
      </div>

      <Panel tour="command-log" eyebrow="Event log" title="What EPOCH did" pad={false}>
        <ol className="max-h-[300px] gap-x-8 overflow-y-auto px-4 py-2 md:columns-2">
          {[...events.map((e) => ({ key: `live${e.ts}`, color: C.info, tag: "live", text: e.text })), ...log].slice(0, 80).map((e) => (
            <li key={e.key} className="flex break-inside-avoid gap-2.5 border-b border-line/60 py-1.5">
              <span className="mt-[5px] h-1.5 w-1.5 shrink-0 rounded-full" style={{ background: e.color }} />
              <div className="min-w-0">
                <div className="eyebrow !text-[9.5px]">{e.tag}</div>
                <div className="text-[12px] leading-snug text-ink-2"><Ticks text={e.text} /></div>
              </div>
            </li>
          ))}
        </ol>
      </Panel>
    </div>
  );
}

function PhaseChip({ phase }: { phase: string }) {
  const m: Record<string, [string, string]> = {
    baseline: [C.ok, "Healthy"], fault: [C.critical, "Fault injected"], incident: [C.critical, "Incident · collecting evidence"],
    shadow: [C.warn, "Shadow canary"], remediated: [C.ok, "Remediated"], rolled_back: [C.critical, "Rolled back"],
  };
  const [c, l] = m[phase] ?? [C.muted, phase];
  return <Pill color={c}>{l}</Pill>;
}

function phaseText(inc: Incident, phase: string): string {
  const sc = inc.data.scenario;
  if (phase === "baseline") return "Serving the deployed genome; CUSUM detectors are fitting their baselines.";
  if (phase === "fault") return `${sc.title} starts ramping in.`;
  if (phase === "incident") return `Alarm raised — RCA agent collecting evidence (${inc.data.causes?.[0]?.title ?? "diagnosing"}).`;
  if (phase === "shadow") return `${(inc.data.candidates?.length ?? 1) - 1} candidate fixes shadow-evaluated on live traffic; twin checks production load.`;
  if (phase === "remediated") return `Promoted after approval: ${inc.data.proposal?.why ?? ""}`;
  return "";
}

function nodeStatus(inc: Incident, phase: string, w: number): Record<string, "ok" | "degraded" | "critical"> {
  const out: Record<string, "ok" | "degraded" | "critical"> = {};
  const hot = ["fault", "incident", "shadow"].includes(phase);
  if (!hot) return out;
  const sc = inc.data.scenario;
  if (sc.kind === "latency" && sc.stage) out[sc.stage] = w >= (inc.data.alarm_w ?? 1e9) ? "critical" : "degraded";
  if (sc.kind === "drift") {
    out.normalize = "degraded";
    out.tier1 = w >= (inc.data.alarm_w ?? 1e9) ? "critical" : "degraded";
    out.tier2 = "degraded";
    out.retrieve = "degraded";
  }
  return out;
}

function buildLog(wb: WorkloadBundle) {
  const rows: { key: string; t: string; color: string; tag: string; text: string }[] = [];
  for (const inc of wb.incidents.slice().reverse()) {
    const st = INCIDENT_STATUS[inc.status] ?? { color: C.muted, label: inc.status };
    rows.push({ key: `i${inc.id}`, t: inc.created_at, color: st.color, tag: `incident · ${inc.id}`, text: `${inc.title} — ${st.label.toLowerCase()}` });
    for (const e of inc.data.timeline.slice().reverse().slice(0, 4)) rows.push({ key: `i${inc.id}${e.event}${e.w}`, t: inc.created_at, color: e.event === "alarm" ? C.critical : e.event === "promoted" || e.event === "recovered" ? C.ok : C.warn, tag: `${inc.id} · minute ${e.w}`, text: `${e.event}: ${e.detail}` });
  }
  for (const h of wb.hypotheses.slice(-8).reverse()) rows.push({ key: `h${h.id}`, t: h.created_at, color: VERDICT[h.verdict]?.color ?? C.muted, tag: `hypothesis · ${h.id} · ${h.verdict}`, text: h.statement });
  for (const r of wb.runs.slice().reverse()) rows.push({ key: `r${r.id}`, t: r.finished_at ?? r.created_at, color: S_COLOR[r.strategy], tag: `run · ${S_LABEL[r.strategy]} · seed ${r.seed}`, text: `${r.summary.n_trials} trials → HV ${r.summary.final_hv.toFixed(4)}, ${r.summary.n_pareto} on its frontier, ${r.summary.n_pruned} pruned` });
  return rows;
}
