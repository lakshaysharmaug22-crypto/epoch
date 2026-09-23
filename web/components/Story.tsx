"use client";

import { AnimatePresence, motion } from "framer-motion";
import { ChevronLeft, ChevronRight, Pause, Play, X } from "lucide-react";
import { useEffect, useMemo } from "react";
import { Ticks } from "@/components/ui";
import { useBundle } from "@/lib/data";
import { fmtNum, fmtPct } from "@/lib/format";
import { useUI, type View } from "@/lib/store";
import type { Bundle, Incident } from "@/lib/types";

export interface Step { view: View; workload?: string; incident?: string | null; target?: string; title: string; body: string; ms?: number }

const meanOf = (o: Incident["data"]["outcome"], side: "live_mean" | "counterfactual_mean", k: string) =>
  typeof o === "object" && o ? (o[side][k] ?? null) : null;

/** The tour script. Every number is read from the bundle, so the story re-writes itself after a GPU run. */
export function buildSteps(b: Bundle): Step[] {
  const W = b.workloads;
  const first = b.meta.workloads[0];
  const nTrials = Object.values(W).reduce((a, w) => a + w.trials.length, 0);
  const nRuns = Object.values(W).reduce((a, w) => a + w.runs.length, 0);
  const tri = W.triage ?? W[first];
  const triName = W.triage ? "triage" : first;
  const rag = W.rag;
  const agentMode = b.meta.llm.enabled ? `Claude (\`${b.meta.llm.agent_model}\`)` : "the offline heuristic agent";
  const device = b.meta.provenance.gpu ?? `a ${b.meta.provenance.cpu_count}-core CPU`;
  const steps: Step[] = [
    { view: "command", workload: triName, target: "command-kpi", title: "EPOCH evolves AI pipelines to a Pareto frontier",
      body: `${nTrials.toLocaleString()} real configurations measured across ${b.meta.workloads.length} workloads and ${nRuns} search runs on ${device}. Quality, p95 latency, memory and $/1k requests are all measured, none hand-entered.` },
    { view: "command", workload: triName, target: "command-topology", title: "The deployed pipeline, live",
      body: "Packets are requests; edge width is traffic share. The ops tape replays a real incident: watch a stage turn amber, then red when CUSUM fires, then recover after the fix is promoted." },
    { view: "architecture", target: "architecture", title: "How it's built",
      body: "FastAPI control plane, Celery GPU worker running the evolution engine and two LangGraph agents, a SimPy twin, and Postgres + pgvector memory. The diagram cycles through the search loop, the healing loop and the data path.", ms: 11000 },
  ];
  if (tri) {
    const kneeT = tri.knee ? tri.trials.find((t) => t.id === tri.knee) : null;
    steps.push({ view: "evolution", workload: triName, target: "evo-pareto", title: "Quality vs latency: the frontier",
      body: `${tri.pareto.length} non-dominated configs out of ${tri.trials.length}. The ringed point is the knee${kneeT ? ` (trial ${kneeT.number}, macro-F1 ${fmtNum(kneeT.metrics.macro_f1 ?? 0, 3)})` : ""}: the best trade-off in normalised objective space. Change the axes to see memory and cost.` });
  }
  if (rag?.race?.headline) {
    const h = rag.race.headline;
    steps.push({ view: "evolution", workload: "rag", target: "evo-race", title: `Agent vs NSGA-II on RAG: −${h.pct_fewer_trials}% trials`,
      body: `Under an equal budget and ${rag.race.seeds.length} seeds, ${agentMode} reached NSGA-II's final hypervolume in a median of ${h.agent_trials_to_nsga2_final} trials vs ${h.nsga2_trials_to_own_final}. A random-forest surrogate screens every proposal by EHVI × P(feasible) before it costs a real run.` });
  }
  if (tri) {
    const conf = tri.hypotheses.filter((h) => h.verdict === "confirmed").length;
    steps.push(
      { view: "lineage", workload: triName, target: "lineage", title: "Every genome has a family tree",
        body: "Agent children link to their parents and to the hypothesis that created them. NSGA-II links are nearest-neighbour inferred and flagged as such." },
      { view: "insights", workload: triName, target: "insights-ablation", title: "Which knob actually moved the metric",
        body: "Interventional ablation: set one gene back to its default with everything else fixed, re-measure 3×, and count the effect only if it clears 2σ of repeat noise. fANOVA above gives the observational view." },
      { view: "insights", workload: triName, target: "insights-hypotheses", title: `${tri.hypotheses.length} falsifiable hypotheses, ${conf} confirmed`,
        body: "The agent writes a hypothesis plus concrete genome edits, the batch runs, and a verdict (confirmed / refuted / inconclusive) is fed back into the next round." },
    );
    const v = tri.twin?.validation;
    if (v && !v.error) steps.push({ view: "twin", workload: triName, target: "twin-validation", title: `Digital twin: ${fmtPct(v.mape_p95, 0)} p95 error vs a real load test`,
      body: `A SimPy twin driven by measured service times, checked against a real HTTP server under open-loop Poisson load at 30–80% of capacity. p50 error is ${fmtPct(v.mape_p50, 0)}. The scenario panel runs the same simulator in your browser.`, ms: 11000 });
    const inc = (name: string) => tri.incidents.find((i) => i.data.scenario.name === name) ?? null;
    const drift = inc("drift"), resolver = inc("resolver"), lat = inc("latency");
    if (drift) {
      const q = tri.describe.quality_metric;
      const lv = meanOf(drift.data.outcome, "live_mean", q), cf = meanOf(drift.data.outcome, "counterfactual_mean", q);
      steps.push({ view: "incidents", workload: triName, incident: drift.id, target: "inc-summary", title: `Self-healing: phrasing drift → ${drift.status}`,
        body: `CUSUM alarm → RCA finds input drift (JS divergence and OOV rate up) → low-confidence tickets are audited → tier 2 is retrained on the error bank → shadow canary → human approval.${lv != null && cf != null ? ` Macro-F1 ${fmtNum(cf, 2)} (old config, same traffic) → ${fmtNum(lv, 2)}.` : ""}`, ms: 11000 });
    }
    if (resolver) {
      const lv = meanOf(resolver.data.outcome, "live_mean", "mean_ms"), cf = meanOf(resolver.data.outcome, "counterfactual_mean", "mean_ms");
      const p = resolver.data.proposal;
      steps.push({ view: "incidents", workload: triName, incident: resolver.id, target: "inc-gate", title: `Resolver outage → ${resolver.status}`,
        body: `RCA attributes the added latency to \`tier3\`. The twin shows the incumbent needs ${p?.incumbent_replicas_needed ?? "?"} replicas at production load; the promoted fix needs ${p?.replicas ?? "?"}.${lv != null && cf != null ? ` Mean latency −${Math.round((1 - lv / cf) * 100)}% vs the counterfactual.` : ""}` });
    }
    if (lat) steps.push({ view: "incidents", workload: triName, incident: lat.id, target: "inc-gate", title: "And when nothing helps, it says so",
      body: "Every candidate was replayed 3× on shadow traffic and none beat the incumbent beyond noise, so EPOCH escalated to a human instead of shipping a placebo." });
  }
  steps.push({ view: "command", workload: triName, target: "command-kpi", title: "Run it yourself",
    body: "`epoch demo` re-records every number on your hardware. Press `Ctrl K` to jump anywhere, `1`–`8` to switch views, `T` to replay this tour.", ms: 9000 });
  return steps;
}

export function useSteps() {
  const { data } = useBundle();
  return useMemo(() => (data ? buildSteps(data) : []), [data]);
}

/** Tour controller + caption card. Drives the store, spotlights the target panel and auto-advances. */
export function Story() {
  const steps = useSteps();
  const { story, storyPaused, setStory, toggleStoryPause } = useUI();
  const step = story != null ? steps[story] : null;

  // drive the dashboard to the step
  useEffect(() => {
    if (!step) return;
    const s = useUI.getState();
    if (step.workload) s.setWorkload(step.workload);
    s.setView(step.view);
    if (step.incident) s.selectIncident(step.incident);
    let el: Element | null = null;
    const t = setTimeout(() => {
      el = step.target ? document.querySelector(`[data-tour="${step.target}"]`) : null;
      if (el) {
        el.classList.add("tour-focus");
        el.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    }, 420);
    return () => { clearTimeout(t); el?.classList.remove("tour-focus"); document.querySelectorAll(".tour-focus").forEach((n) => n.classList.remove("tour-focus")); };
  }, [step]);

  // keyboard while touring
  useEffect(() => {
    if (story == null) return;
    const on = (e: KeyboardEvent) => {
      if (e.key === "Escape") setStory(null);
      else if (e.key === "ArrowRight") setStory(Math.min(steps.length - 1, story + 1));
      else if (e.key === "ArrowLeft") setStory(Math.max(0, story - 1));
      else if (e.key === " ") { e.preventDefault(); toggleStoryPause(); }
      else return;
      e.stopImmediatePropagation();
    };
    window.addEventListener("keydown", on, true);
    return () => window.removeEventListener("keydown", on, true);
  }, [story, steps.length, setStory, toggleStoryPause]);

  const next = () => (story != null && story < steps.length - 1 ? setStory(story + 1) : setStory(null));

  return (
    <AnimatePresence>
      {step && story != null && (
        <motion.div key="story" initial={{ y: 40, opacity: 0 }} animate={{ y: 0, opacity: 1 }} exit={{ y: 40, opacity: 0 }} transition={{ type: "spring", stiffness: 320, damping: 30 }}
          className="fixed inset-x-3 bottom-3 z-50 mx-auto max-w-[620px] sm:bottom-6" role="dialog" aria-label="Guided tour">
          <div className="story-card overflow-hidden rounded-xl border border-[#e11d4866] bg-[#0d0f15]/95 shadow-[0_20px_60px_-10px_rgba(0,0,0,.8),0_0_40px_-10px_rgba(225,29,72,.45)] backdrop-blur-md">
            <div className="h-[3px] w-full bg-[#1a1e2a]">
              <div key={story} className="tour-progress h-full bg-gradient-to-r from-crimson to-crimson-hi"
                style={{ animationDuration: `${step.ms ?? 9500}ms`, animationPlayState: storyPaused ? "paused" : "running" }} onAnimationEnd={next} />
            </div>
            <div className="p-4 sm:p-5">
              <div className="mb-1.5 flex items-center gap-2">
                <span className="num text-[10.5px] font-semibold tracking-[0.16em] text-crimson-hi">GUIDED TOUR · {String(story + 1).padStart(2, "0")} / {String(steps.length).padStart(2, "0")}</span>
                <div className="ml-auto flex gap-[3px]">
                  {steps.map((_, i) => (
                    <button key={i} aria-label={`Step ${i + 1}`} onClick={() => setStory(i)}
                      className="h-1.5 rounded-full transition-all" style={{ width: i === story ? 16 : 6, background: i < story ? "#e11d48" : i === story ? "#ff4d6d" : "#2b3040" }} />
                  ))}
                </div>
              </div>
              <AnimatePresence mode="wait">
                <motion.div key={story} initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0 }} exit={{ opacity: 0, x: -12 }} transition={{ duration: 0.22 }}>
                  <h3 className="text-[16px] font-semibold tracking-[-0.01em] text-ink [text-wrap:balance]"><Ticks text={step.title} /></h3>
                  <p className="mt-1.5 text-[13px] leading-relaxed text-ink-2"><Ticks text={step.body} /></p>
                </motion.div>
              </AnimatePresence>
              <div className="mt-3.5 flex items-center gap-1.5">
                <button onClick={() => setStory(Math.max(0, story - 1))} disabled={story === 0} className="tour-btn" aria-label="Previous"><ChevronLeft size={15} /></button>
                <button onClick={toggleStoryPause} className="tour-btn" aria-label={storyPaused ? "Play" : "Pause"}>{storyPaused ? <Play size={14} /> : <Pause size={14} />}</button>
                <button onClick={next} className="tour-btn !bg-crimson !text-white hover:!bg-crimson-hi" aria-label="Next"><ChevronRight size={15} /></button>
                <span className="num ml-2 hidden text-[10.5px] text-muted sm:inline">← → step · space pause · esc exit</span>
                <button onClick={() => setStory(null)} className="tour-btn ml-auto" aria-label="Exit tour"><X size={14} /></button>
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
