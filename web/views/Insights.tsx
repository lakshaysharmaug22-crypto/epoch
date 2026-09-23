"use client";

import { useMemo, useState } from "react";
import { Diverging, HBars } from "@/components/charts/Charts";
import { Button, Panel, Pill, Segmented, Ticks } from "@/components/ui";
import { useWorkloadData } from "@/lib/derive";
import { fmtMs, fmtNum, fmtObjective, fmtValue } from "@/lib/format";
import { C, VERDICT } from "@/lib/theme";
import type { Hypothesis } from "@/lib/types";

export function InsightsView() {
  const d = useWorkloadData();
  const [verdict, setVerdict] = useState<"all" | Hypothesis["verdict"]>("all");
  const [showAll, setShowAll] = useState(false);
  const hyps = useMemo(() => (d ? d.wb.hypotheses.filter((h) => verdict === "all" || h.verdict === verdict).slice().reverse() : []), [d, verdict]);
  if (!d) return null;
  const { wb, objectives, quality } = d;
  const fa = wb.attribution?.fanova ?? {};
  const ab = wb.attribution?.ablation;
  const counts = { confirmed: 0, refuted: 0, inconclusive: 0, pending: 0 } as Record<string, number>;
  wb.hypotheses.forEach((h) => (counts[h.verdict] += 1));
  const latency = objectives.find((o) => o.name === "p95_ms");
  const failKinds = wb.failures.reduce<Record<string, number>>((a, f) => ({ ...a, [f.kind]: (a[f.kind] ?? 0) + 1 }), {});

  return (
    <div className="grid gap-4">
      <Panel tour="insights-fanova" eyebrow="fANOVA · share of each objective's variance explained by a gene (genes present in every trial)" title="Which knobs matter">
        <div className="grid gap-6 md:grid-cols-2 xl:grid-cols-4">
          {objectives.map((o) => {
            const imp = Object.entries(fa[o.name] ?? {}).sort((a, b) => b[1] - a[1]).slice(0, 7);
            return (
              <div key={o.name} className="min-w-0">
                <div className="mb-2 text-[12.5px] font-medium text-ink">{o.label}</div>
                {imp.length ? <HBars data={imp.map(([label, value]) => ({ label, value }))} format={(v) => `${Math.round(v * 100)}%`} max={1} /> : <div className="text-[12px] text-muted">Not enough trials.</div>}
              </div>
            );
          })}
        </div>
      </Panel>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.1fr)_minmax(0,0.9fr)]">
        <Panel tour="insights-ablation" eyebrow={ab ? `Interventional ablation · do(gene := default), all other genes fixed · ${ab.repeats} repeats for noise` : "Interventional ablation"}
          title="What each chosen gene is worth">
          {ab && ab.effects.length ? (
            <div className="overflow-x-auto">
              <div className="grid min-w-[520px] grid-cols-[150px_1fr_1fr] gap-x-4 gap-y-0">
                <div className="eyebrow pb-2">Revert gene</div>
                <div className="eyebrow pb-2">Δ {quality.label} (reverting)</div>
                <div className="eyebrow pb-2">Δ {latency?.label ?? "latency"} (reverting)</div>
                {ab.effects.filter((e) => e.status === "complete").map((e, _i, arr) => {
                  const maxQ = Math.max(...arr.map((x) => Math.abs(x.delta?.[quality.name] ?? 0)), 1e-9);
                  const maxL = Math.max(...arr.map((x) => Math.abs(x.delta?.p95_ms ?? 0)), 1e-9);
                  const dq = e.delta?.[quality.name] ?? 0, dl = e.delta?.p95_ms ?? 0;
                  return (
                    <div key={e.gene} className="contents">
                      <div className="border-t border-line py-2">
                        <div className="num text-[12px] text-ink">{e.gene}</div>
                        <div className="num text-[10.5px] text-muted">{fmtValue(e.from)} → {fmtValue(e.to)}</div>
                      </div>
                      <div className="border-t border-line py-2.5">
                        <Diverging better="pos" max={maxQ} rows={[{ label: e.gene, value: dq, sig: !!e.significant?.[quality.name], text: `${dq > 0 ? "+" : ""}${fmtNum(dq, 3)}` }]} />
                      </div>
                      <div className="border-t border-line py-2.5">
                        <Diverging better="neg" max={maxL} rows={[{ label: e.gene, value: dl, sig: !!e.significant?.p95_ms, text: `${dl > 0 ? "+" : ""}${fmtMs(Math.abs(dl)).replace(/^/, dl < 0 ? "−" : "")}` }]} />
                      </div>
                    </div>
                  );
                })}
              </div>
              <p className="mt-3 text-[11.5px] leading-relaxed text-muted">
                Blue = reverting helps, red = reverting hurts (so the chosen value is doing work). Faded bars are inside 2σ of repeated-measurement noise
                (σ {quality.label} {fmtNum(ab.noise[quality.name], 4)}, σ p95 {fmtMs(ab.noise.p95_ms)}).
              </p>
            </div>
          ) : <div className="text-[12px] text-muted">Run `epoch demo` to compute ablations for the knee config.</div>}
        </Panel>

        <Panel eyebrow="Vector memory of failures" title={`${wb.failures.length} failed trials, clustered by kind`}>
          {wb.failures.length ? (
            <div className="flex flex-col gap-3">
              <div className="flex flex-wrap gap-2">{Object.entries(failKinds).map(([k, n]) => <Pill key={k} color={C.critical}>{k} · {n}</Pill>)}</div>
              {wb.failures.slice(-4).reverse().map((f) => (
                <pre key={f.id} className="overflow-x-auto whitespace-pre-wrap rounded-md border border-line bg-sunken p-2.5 text-[11px] leading-snug text-ink-2">{f.message.split("\n").filter(Boolean).slice(-2).join("\n")}</pre>
              ))}
              <p className="text-[11.5px] text-muted">Failures are embedded (pgvector / 384-d) and retrieved by the agent and the RCA agent as “similar past failures”.</p>
            </div>
          ) : (
            <p className="text-[12.5px] leading-relaxed text-ink-2">
              No trial crashed in this run. The memory still records every pruned and infeasible configuration with its reason, and on GPU it
              captures OOMs and compile failures for the agent to avoid.
            </p>
          )}
        </Panel>
      </div>

      <Panel tour="insights-hypotheses" eyebrow={`Hypothesis notebook · ${wb.hypotheses.length} experiments designed by the agent`} title="Hypotheses and verdicts"
        right={<Segmented size="xs" value={verdict} onChange={setVerdict}
          options={[{ value: "all", label: `All ${wb.hypotheses.length}` }, ...(["confirmed", "refuted", "inconclusive"] as const).map((v) => ({ value: v, label: `${VERDICT[v].label} ${counts[v]}` }))]} />}>
        <div className="grid gap-3 lg:grid-cols-2 2xl:grid-cols-3">
          {hyps.slice(0, showAll ? 200 : 9).map((h) => {
            const target = objectives.find((o) => o.name === h.target);
            return (
              <article key={h.id} className="flex flex-col gap-2 rounded-lg border border-line bg-sunken p-3.5">
                <div className="flex items-center justify-between gap-2">
                  <span className="num text-[11px] text-muted">{h.id} · gen {h.evidence.generation ?? "?"} · {h.source === "llm" ? h.evidence.model ?? "LLM" : "heuristic"}</span>
                  <Pill color={VERDICT[h.verdict].color}>{VERDICT[h.verdict].label}</Pill>
                </div>
                <p className="text-[12.5px] leading-relaxed text-ink"><Ticks text={h.statement} /></p>
                <p className="text-[11.5px] leading-relaxed text-muted"><Ticks text={h.mechanism} /></p>
                <div className="flex flex-wrap gap-1.5">
                  {h.genes.map((g) => <span key={g} className="num rounded border border-line px-1.5 py-0.5 text-[10.5px] text-ink-2">{g}</span>)}
                  {Object.entries(h.expected).map(([k, v]) => <span key={k} className="rounded px-1.5 py-0.5 text-[10.5px] text-muted">{objectives.find((o) => o.name === k)?.label ?? k}: {v}</span>)}
                </div>
                {h.evidence.tests && h.evidence.tests.length > 0 && target && (
                  <div className="mt-1 border-t border-line pt-2">
                    {h.evidence.tests.map((t) => (
                      <div key={t.genome_id} className="num flex justify-between text-[11px]">
                        <span className="text-muted">{t.genome_id}</span>
                        <span className="text-ink-2">{t.raw?.[target.name] ? `${fmtObjective(target, t.raw[target.name][0])} → ${fmtObjective(target, t.raw[target.name][1])}` : t.status}</span>
                      </div>
                    ))}
                  </div>
                )}
              </article>
            );
          })}
        </div>
        {hyps.length === 0 && <div className="text-[12px] text-muted">No hypotheses with this verdict.</div>}
        {hyps.length > 9 && (
          <div className="mt-3 flex justify-center">
            <Button tone="ghost" onClick={() => setShowAll((x) => !x)}>{showAll ? "Show fewer" : `Show all ${hyps.length}`}</Button>
          </div>
        )}
      </Panel>
    </div>
  );
}
