"use client";

import { Pause, Play } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { HvRace, StrategyLegend } from "@/components/charts/Charts";
import { ParetoScatter } from "@/components/charts/Pareto";
import { Button, GenomeStrip, Panel, Segmented, Tip } from "@/components/ui";
import { useWorkloadData } from "@/lib/derive";
import { cn, fmtNum, fmtObjective } from "@/lib/format";
import { useUI } from "@/lib/store";
import { C, S_COLOR, S_LABEL, STRATEGIES } from "@/lib/theme";
import type { Strategy } from "@/lib/types";

export function EvolutionView() {
  const d = useWorkloadData();
  const { objX, objY, setAxes, trialId, selectTrial, setView } = useUI();
  const [visible, setVisible] = useState<Set<Strategy>>(new Set(STRATEGIES));
  const [upTo, setUpTo] = useState<number | null>(null);
  const [playing, setPlaying] = useState(false);
  const budget = d?.wb.race?.budget ?? Math.max(...(d?.wb.trials.map((t) => t.number + 1) ?? [1]));

  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => setUpTo((u) => {
      const next = (u ?? 0) + 1;
      if (next >= budget) { setPlaying(false); return null; }
      return next;
    }), 140);
    return () => clearInterval(id);
  }, [playing, budget]);

  const pairs = useMemo(() => {
    if (!d) return [];
    const q = d.quality.name;
    return d.objectives.filter((o) => o.name !== q).map((o) => ({ x: o.name, y: q }));
  }, [d]);

  if (!d) return null;
  const { wb, objectives, quality, frontier, knee } = d;
  const ox = objectives.find((o) => o.name === (objX ?? pairs[0]?.x)) ?? objectives[1];
  const oy = objectives.find((o) => o.name === (objY ?? pairs[0]?.y)) ?? objectives[0];
  const race = wb.race;
  const nComplete = wb.trials.filter((t) => t.status === "complete").length;
  const nPruned = wb.trials.filter((t) => t.status === "pruned").length;

  return (
    <div className="grid gap-4">
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_360px]">
        <Panel tour="evo-pareto" eyebrow={`Pareto frontier · ${nComplete} measured configs across ${wb.runs.length} runs`} title={`${oy.label} vs ${ox.label}`}
          right={
            <div className="flex flex-wrap items-center gap-2">
              <Segmented value={`${ox.name}|${oy.name}`} onChange={(v) => { const [x, y] = v.split("|"); setAxes(x, y); }}
                options={[...pairs.map((p) => ({ value: `${p.x}|${p.y}`, label: objectives.find((o) => o.name === p.x)!.label })),
                  ...(objectives.length > 2 ? [{ value: `${objectives[1].name}|${objectives[2].name}`, label: `${objectives[2].label} vs ${objectives[1].label}` }] : [])]} />
            </div>
          }>
          <div className="mb-3 flex flex-wrap items-center gap-3">
            <StrategyLegend visible={visible} toggle={(s) => setVisible((v) => { const n = new Set(v); if (n.has(s)) n.delete(s); else n.add(s); return n; })} />
            <div className="ml-auto flex items-center gap-2">
              <Button tone="ghost" onClick={() => { if (upTo === null) setUpTo(0); setPlaying((p) => !p); }}>
                {playing ? <Pause size={13} /> : <Play size={13} />} Replay search
              </Button>
              <input aria-label="Trials per run" type="range" min={1} max={budget} value={upTo ?? budget}
                onChange={(e) => { setPlaying(false); const v = Number(e.target.value); setUpTo(v >= budget ? null : v); }}
                className="w-36 accent-[#e11d48]" />
              <span className="num w-20 text-[11px] text-muted">{upTo === null ? `all ${budget}` : `≤ trial ${upTo}`}</span>
            </div>
          </div>
          <ParetoScatter trials={wb.trials} ox={ox} oy={oy} knee={wb.knee} selected={trialId} visible={visible as Set<string>}
            onSelect={(id) => { selectTrial(id); }} upTo={upTo ?? undefined} height={500} />
          <div className="mt-2 flex flex-wrap gap-x-5 gap-y-1 text-[11.5px] text-muted">
            <span><span className="text-ink">Ringed</span> = global Pareto set (all {objectives.length} objectives)</span>
            <span><span className="text-ink">Line</span> = frontier of this projection</span>
            <span><span className="text-ink">Hollow</span> = violates a constraint</span>
            <span>{nPruned} low-fidelity rejections not plotted</span>
          </div>
        </Panel>

        <div className="grid content-start gap-4">
          {race && (
            <Panel tour="evo-race" eyebrow={`Strategy race · ${race.seeds.length} seeds × ${race.budget} trials each`} title="Who finds the frontier fastest">
              {race.headline && (
                <div className="mb-4 rounded-lg border border-line bg-sunken p-3.5">
                  <div className="flex items-baseline gap-2">
                    <span className="text-[30px] font-semibold tracking-[-0.02em]" style={{ color: race.headline.pct_fewer_trials > 0 ? C.ok : C.critical }}>
                      {race.headline.pct_fewer_trials > 0 ? "−" : "+"}{Math.abs(race.headline.pct_fewer_trials)}%
                    </span>
                    <span className="text-[12.5px] text-ink-2">trials for the EPOCH agent to reach NSGA-II&apos;s final hypervolume</span>
                  </div>
                  <div className="num mt-1 text-[11.5px] text-muted">median {race.headline.agent_trials_to_nsga2_final} vs {race.headline.nsga2_trials_to_own_final} trials · {wb.hypotheses.length} hypotheses tested</div>
                </div>
              )}
              <div className="overflow-x-auto">
                <table className="w-full text-[12px]">
                  <thead>
                    <tr className="text-left text-muted">
                      <th className="pb-2 font-normal">Strategy</th>
                      <th className="pb-2 text-right font-normal">Median HV</th>
                      <th className="pb-2 text-right font-normal"><Tip content="Area under the mean hypervolume curve ÷ budget — rewards finding good configs early."><span className="cursor-help underline decoration-dotted underline-offset-2">AUC</span></Tip></th>
                      <th className="pb-2 text-right font-normal"><Tip content={`Median trials until a run's hypervolume reaches NSGA-II's median final value (${fmtNum(race.nsga2_median_final_hv, 4)}). Runs that never get there count as budget + 1; the second number is how many seeds got there.`}><span className="cursor-help underline decoration-dotted underline-offset-2">→ NSGA-II final</span></Tip></th>
                    </tr>
                  </thead>
                  <tbody>
                    {STRATEGIES.filter((s) => race.strategies[s]).map((s) => {
                      const r = race.strategies[s]!;
                      const best = Math.max(...STRATEGIES.filter((x) => race.strategies[x]).map((x) => race.strategies[x]!.final_hv_median));
                      return (
                        <tr key={s} className="border-t border-line">
                          <td className="py-2"><span className="inline-flex items-center gap-2"><span className="h-[2px] w-3.5 rounded" style={{ background: S_COLOR[s] }} />{S_LABEL[s]}</span></td>
                          <td className={cn("num py-2 text-right", r.final_hv_median === best ? "font-semibold text-ink" : "text-ink-2")}>{r.final_hv_median.toFixed(4)}</td>
                          <td className="num py-2 text-right text-ink-2">{r.auc.toFixed(3)}</td>
                          <td className="num py-2 text-right text-ink-2">{r.trials_to_nsga2_final_median == null || r.trials_to_nsga2_final_median > race.budget ? "—" : r.trials_to_nsga2_final_median}
                            <span className="ml-1 text-muted">({r.trials_to_nsga2_final.filter((x) => x != null).length}/{r.runs})</span></td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </Panel>
          )}
          <Panel eyebrow="Global Pareto set" title={`${frontier.length} non-dominated configurations`} pad={false}>
            <ul className="max-h-[300px] overflow-y-auto">
              {frontier.map((t) => (
                <li key={t.id}>
                  <button onClick={() => { selectTrial(t.id); setView("trials"); }}
                    className={cn("grid w-full grid-cols-[1fr_auto] items-center gap-x-3 gap-y-1.5 border-b border-line/70 px-4 py-2.5 text-left hover:bg-raised", t.id === knee?.id && "bg-[#1a0c12]")}>
                    <span className="flex items-center gap-2 text-[12px] text-ink-2">
                      <span className="h-2 w-2 rounded-full" style={{ background: S_COLOR[t.strategy] }} />
                      {S_LABEL[t.strategy]} · s{t.seed} · #{t.number}{t.id === knee?.id && <span className="text-crimson-hi">· knee</span>}
                    </span>
                    <span className="num text-[12px] text-ink">{fmtObjective(quality, t.metrics[quality.name])}</span>
                    <GenomeStrip genome={t.genome} space={wb.describe.space} height={8} />
                    <span className="num text-[11px] text-muted">{objectives.filter((o) => o.name !== quality.name).map((o) => fmtObjective(o, t.metrics[o.name])).join(" · ")}</span>
                  </button>
                </li>
              ))}
            </ul>
          </Panel>
        </div>
      </div>

      {race && (
        <Panel tour="evo-curves" eyebrow="Hypervolume vs trials · thin lines are individual seeds, thick lines the mean" title="Search efficiency under an equal budget">
          <HvRace race={race} height={320} />
        </Panel>
      )}
    </div>
  );
}
