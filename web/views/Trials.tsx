"use client";

import { ArrowDown, ArrowUp } from "lucide-react";
import { useMemo, useState } from "react";
import { LatencyHist, StageBar } from "@/components/charts/Charts";
import { GenomeStrip, KeyVal, Panel, Pill, Segmented } from "@/components/ui";
import { useWorkloadData } from "@/lib/derive";
import { cn, fmtDuration, fmtMs, fmtNum, fmtObjective, fmtPct, fmtValue } from "@/lib/format";
import { useUI } from "@/lib/store";
import { C, GROUP_COLOR, S_COLOR, S_LABEL, STRATEGIES } from "@/lib/theme";
import type { Strategy, Trial } from "@/lib/types";

type Filter = "all" | "pareto" | "feasible" | "pruned" | "failed";

export function TrialsView() {
  const d = useWorkloadData();
  const { trialId, selectTrial } = useUI();
  const [filter, setFilter] = useState<Filter>("pareto");
  const [strategy, setStrategy] = useState<Strategy | "all">("all");
  const [sort, setSort] = useState<{ key: string; dir: 1 | -1 }>({ key: "quality", dir: -1 });

  const rows = useMemo(() => {
    if (!d) return [];
    let r = d.wb.trials;
    if (strategy !== "all") r = r.filter((t) => t.strategy === strategy);
    if (filter === "pareto") r = r.filter((t) => t.global_pareto);
    if (filter === "feasible") r = r.filter((t) => t.feasible);
    if (filter === "pruned") r = r.filter((t) => t.status === "pruned");
    if (filter === "failed") r = r.filter((t) => t.status === "failed");
    const key = sort.key === "quality" ? d.quality.name : sort.key;
    const val = (t: Trial) => key === "number" ? t.number : key === "duration" ? t.duration_s : (t.metrics[key] ?? -Infinity);
    return [...r].sort((a, b) => (val(a) > val(b) ? 1 : val(a) < val(b) ? -1 : 0) * sort.dir);
  }, [d, filter, strategy, sort]);

  if (!d) return null;
  const { objectives, quality, wb } = d;
  const sel = (trialId && d.byId.get(trialId)) || rows[0] || null;
  const run = sel ? wb.runs.find((r) => r.id === sel.run_id) : null;
  const cols = [{ key: "number", label: "#" }, ...objectives.map((o) => ({ key: o.name === quality.name ? "quality" : o.name, label: o.label })),
    { key: "robustness", label: "Robust." }, { key: "duration", label: "Time" }];

  const Th = ({ k, label }: { k: string; label: string }) => (
    <th className="whitespace-nowrap px-3 py-2 text-right font-normal first:text-left">
      <button onClick={() => setSort((s) => ({ key: k, dir: s.key === k ? (s.dir === 1 ? -1 : 1) : -1 }))} className="inline-flex items-center gap-1 hover:text-ink">
        {label}{sort.key === k && (sort.dir === 1 ? <ArrowUp size={11} /> : <ArrowDown size={11} />)}
      </button>
    </th>
  );

  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_400px]">
      <Panel eyebrow={`${rows.length} of ${wb.trials.length} trials`} title="Measured configurations" pad={false}
        right={
          <div className="flex flex-wrap gap-2">
            <Segmented size="xs" value={strategy} onChange={setStrategy}
              options={[{ value: "all", label: "All" }, ...STRATEGIES.map((s) => ({ value: s, label: S_LABEL[s] }))]} />
            <Segmented size="xs" value={filter} onChange={setFilter}
              options={[{ value: "pareto", label: "Pareto" }, { value: "feasible", label: "Feasible" }, { value: "all", label: "All" }, { value: "pruned", label: "Pruned" }, { value: "failed", label: "Failed" }]} />
          </div>
        }>
        <div className="max-h-[720px] overflow-auto">
          <table className="w-full min-w-[760px] text-[12px]">
            <thead className="sticky top-0 z-10 bg-surface text-muted shadow-[0_1px_0_#1f2330]">
              <tr>
                <th className="px-3 py-2 text-left font-normal">Strategy</th>
                {cols.map((c) => <Th key={c.key} k={c.key} label={c.label} />)}
                <th className="px-3 py-2 text-left font-normal">Genome</th>
              </tr>
            </thead>
            <tbody>
              {rows.slice(0, 400).map((t) => (
                <tr key={t.id} onClick={() => selectTrial(t.id)}
                  className={cn("cursor-pointer border-b border-line/60 hover:bg-raised", sel?.id === t.id && "bg-[#1a0c12]")}>
                  <td className="whitespace-nowrap px-3 py-2">
                    <span className="inline-flex items-center gap-2 text-ink-2">
                      <span className="h-2 w-2 rounded-full" style={{ background: S_COLOR[t.strategy] }} />{S_LABEL[t.strategy]} <span className="text-muted">s{t.seed}</span>
                    </span>
                  </td>
                  <td className="num px-3 py-2 text-right text-muted">{t.number}</td>
                  {objectives.map((o) => (
                    <td key={o.name} className={cn("num whitespace-nowrap px-3 py-2 text-right", t.status === "complete" ? "text-ink" : "text-muted")}>
                      {t.status === "failed" ? "—" : fmtObjective(o, t.metrics[o.name])}
                    </td>
                  ))}
                  <td className="num px-3 py-2 text-right text-ink-2">{t.metrics.robustness != null ? fmtPct(t.metrics.robustness, 0) : "—"}</td>
                  <td className="num px-3 py-2 text-right text-muted">{fmtDuration(t.duration_s)}</td>
                  <td className="px-3 py-2"><GenomeStrip genome={t.genome} space={wb.describe.space} height={9} /></td>
                </tr>
              ))}
            </tbody>
          </table>
          {rows.length > 400 && <div className="px-4 py-3 text-[12px] text-muted">Showing the first 400 — narrow the filter to see the rest.</div>}
        </div>
      </Panel>

      <div className="grid content-start gap-4">
        {sel && (
          <>
            <Panel eyebrow={`${sel.id}`} title={`Trial #${sel.number} · ${S_LABEL[sel.strategy]} · ${sel.origin}`}>
              <div className="mb-3 flex flex-wrap gap-1.5">
                <Pill color={sel.status === "complete" ? C.ok : sel.status === "pruned" ? C.warn : C.critical}>{sel.status}</Pill>
                {sel.status === "complete" && <Pill color={sel.feasible ? C.ok : C.critical}>{sel.feasible ? "feasible" : "violates constraint"}</Pill>}
                {sel.global_pareto && <Pill color={C.ink}>global Pareto</Pill>}
                {sel.id === wb.knee && <Pill color={C.crimsonHi}>knee</Pill>}
                <Pill color={C.muted}>fidelity {Math.round(sel.fidelity * 100)}%</Pill>
              </div>
              {objectives.map((o) => <KeyVal key={o.name} k={o.label} v={fmtObjective(o, sel.metrics[o.name])} />)}
              <KeyVal k="p50 / p99" v={`${fmtMs(sel.metrics.p50_ms)} / ${fmtMs(sel.metrics.p99_ms)}`} />
              <KeyVal k="Throughput" v={`${fmtNum(sel.metrics.throughput_rps)} req/s`} />
              <KeyVal k="Peak memory" v={`${fmtNum(sel.metrics.peak_mem_mb, 2)} MB`} />
              <KeyVal k="Robustness" v={fmtPct(sel.metrics.robustness)} />
              <KeyVal k="Hypervolume after" v={fmtNum(sel.hv_after, 4)} />
              {sel.prune_reason && <p className="mt-2 text-[12px] leading-relaxed text-warn">Pruned — {sel.prune_reason}</p>}
              {sel.error && <pre className="mt-2 max-h-40 overflow-auto rounded-md border border-line bg-sunken p-2 text-[10.5px] leading-snug text-critical">{sel.error_kind}: {sel.error.split("\n").slice(-4).join("\n")}</pre>}
            </Panel>
            {sel.latency_hist && sel.latency_hist.edges.length > 0 && (
              <Panel eyebrow="Per-request latency · log scale" title={`${fmtMs(sel.metrics.p50_ms)} median · ${fmtMs(sel.metrics.p95_ms)} p95`}>
                <LatencyHist hist={sel.latency_hist} marks={sel.metrics.p95_ms ? [{ v: sel.metrics.p95_ms, label: "p95" }] : []} />
              </Panel>
            )}
            {sel.stage_ms && Object.keys(sel.stage_ms).length > 0 && (
              <Panel eyebrow="Where the time goes · amortised per request" title="Stage breakdown">
                <StageBar stages={sel.stage_ms} />
              </Panel>
            )}
            <Panel eyebrow="Genome" title={`${Object.keys(sel.genome).length} active genes · ${sel.genome_id}`} pad={false}>
              <table className="w-full text-[12px]">
                <tbody>
                  {wb.describe.space.filter((g) => g.name in sel.genome).map((g) => (
                    <tr key={g.name} className="border-b border-line/60">
                      <td className="px-4 py-1.5"><span className="inline-flex items-center gap-2"><span className="h-2 w-2 rounded-[2px]" style={{ background: GROUP_COLOR[g.group] ?? C.muted }} /><span className="num text-ink-2">{g.name}</span></span></td>
                      <td className="num px-4 py-1.5 text-right text-ink">{fmtValue(sel.genome[g.name])}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Panel>
            {run && (
              <Panel eyebrow="Provenance" title="Reproduce this trial">
                <KeyVal k="Run" v={run.id} />
                <KeyVal k="Seed" v={run.seed} />
                <KeyVal k="Git" v={run.provenance.git_sha ?? "not a git checkout"} />
                <KeyVal k="Env hash" v={run.provenance.env_hash} />
                <KeyVal k="Device" v={run.provenance.gpu ?? `${run.provenance.device} · ${run.provenance.cpu_count} cores`} />
                <KeyVal k="Python · optuna" v={`${run.provenance.python} · ${run.provenance.versions.optuna ?? "?"}`} />
                <pre className="mt-3 overflow-x-auto rounded-md border border-line bg-sunken p-2.5 text-[11px] text-ink-2">{`epoch run --workload ${wb.describe.name} --strategy ${sel.strategy} --seed ${run.seed} --budget ${run.budget}`}</pre>
              </Panel>
            )}
          </>
        )}
      </div>
    </div>
  );
}
