"use client";

import { Check, RotateCcw, ShieldCheck, X } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { CusumChart, WindowChart } from "@/components/charts/Charts";
import { Button, KeyVal, Panel, Pill, Ticks, Tip } from "@/components/ui";
import { API, postDecision } from "@/lib/data";
import { useWorkloadData } from "@/lib/derive";
import { cn, fmtMs, fmtNum, fmtPct, fmtValue } from "@/lib/format";
import { useUI } from "@/lib/store";
import { C, INCIDENT_STATUS } from "@/lib/theme";
import type { Incident } from "@/lib/types";

const num = (v: unknown) => (typeof v === "number" ? v : null);
const PHASE_COLOR: Record<string, string> = { baseline: "#1f3a2f", fault: "#4a1420", incident: "#6b1728", shadow: "#4a3a10", remediated: "#12402f", rolled_back: "#6b1728" };
const EVENT_COLOR: Record<string, string> = { escalated: C.warn, no_effect: C.muted, fault: C.critical, alarm: C.critical, rca: C.violet, audit: C.violet, canary: C.warn, proposal: C.warn, approved: C.ok, promoted: C.ok, recovered: C.ok, rollback: C.critical, rejected: C.muted, deploy: C.ink2, no_candidate: C.critical };

export function IncidentsView() {
  const d = useWorkloadData();
  const { incidentId, selectIncident } = useUI();
  const incidents = d?.wb.incidents ?? [];
  const inc = incidents.find((i) => i.id === incidentId) ?? incidents.at(-1);

  if (!d) return null;
  if (!inc) return <Panel title="No incidents yet"><p className="text-[12.5px] text-muted">Run `epoch heal --scenario drift` (or `epoch demo`) to inject a fault into the deployed genome.</p></Panel>;

  return (
    <div className="grid gap-4 xl:grid-cols-[280px_minmax(0,1fr)]">
      <div className="grid content-start gap-2">
        {incidents.slice().reverse().map((i) => {
          const st = INCIDENT_STATUS[i.status] ?? { color: C.muted, label: i.status };
          return (
            <button key={i.id} onClick={() => selectIncident(i.id)}
              className={cn("panel flex flex-col gap-2 p-3.5 text-left transition-colors hover:border-line-strong", i.id === inc.id && "!border-crimson/60 bg-[#140a0e]")}>
              <div className="flex items-center justify-between gap-2">
                <span className="num text-[11px] text-muted">{i.id} · {i.severity.toUpperCase()}</span>
                <Pill color={st.color}>{st.label}</Pill>
              </div>
              <div className="text-[12.5px] font-medium leading-snug text-ink">{i.data.scenario.title}</div>
              <div className="text-[11.5px] leading-snug text-muted"><Ticks text={i.title.split(" — ")[0]} /></div>
            </button>
          );
        })}
      </div>
      <IncidentDetail key={inc.id} inc={inc} />
    </div>
  );
}

function IncidentDetail({ inc }: { inc: Incident }) {
  const d = useWorkloadData()!;
  const { decisions, decide } = useUI();
  const q = d.quality;
  const data = inc.data;
  const gate = data.gate_w ?? data.alarm_w ?? 0;
  const recorded = data.decision;
  const local = decisions[inc.id];
  const [mode, setMode] = useState<"recorded" | "gate" | "approved" | "rejected">("recorded");
  const [reveal, setReveal] = useState(Infinity);
  const [busy, setBusy] = useState(false);
  const liveAwaiting = inc.status === "awaiting_approval";

  useEffect(() => {
    if (mode !== "approved" && mode !== "rejected") return;
    setReveal(gate);
    const last = data.series.at(-1)!.w;
    const id = setInterval(() => setReveal((r) => { if (r >= last) { clearInterval(id); return r; } return r + 1; }), 220);
    return () => clearInterval(id);
  }, [mode, gate, data.series]);

  const series = useMemo(() => {
    return data.series.filter((s) => {
      if (mode === "gate") return s.w <= gate;
      if (mode === "approved" || mode === "rejected") return s.w <= reveal;
      return true;
    }).map((s) => {
      const after = s.w > gate && s.counterfactual;
      const live = mode === "rejected" && after ? s.counterfactual! : s.live;
      return { w: s.w, liveQ: num(live[q.name]), liveP: num(live.p95_ms), cfQ: after && mode !== "rejected" ? num(s.counterfactual![q.name]) : null, cfP: after && mode !== "rejected" ? num(s.counterfactual!.p95_ms) : null };
    });
  }, [data.series, mode, gate, reveal, q.name]);

  const st = INCIDENT_STATUS[inc.status] ?? { color: C.muted, label: inc.status };
  const oc = typeof data.outcome === "object" ? data.outcome : null;
  const prop = data.proposal;
  const cands = data.candidates ?? [];
  const twinBy = new Map((data.twin?.results ?? []).map((r) => [r.genome_id, r]));
  const maxW = data.series.at(-1)?.w ?? 1;
  const cusum = Object.entries(data.detectors ?? {}).map(([metric, det], i) => ({ metric, values: det.history, color: [C.crimsonHi, C.info, C.violet][i % 3] }));
  const h = Object.values(data.detectors ?? {})[0]?.h ?? 5;
  const showPromoted = mode === "recorded" ? data.promoted_w : mode === "approved" ? gate + 1 : undefined;

  const onDecision = async (approved: boolean) => {
    if (API && liveAwaiting) {
      setBusy(true);
      try { await postDecision(inc.id, approved, approved ? "approved in dashboard" : "rejected in dashboard"); } finally { setBusy(false); }
      return;
    }
    decide(inc.id, approved ? "approved" : "rejected");
    setMode(approved ? "approved" : "rejected");
  };

  return (
    <div className="grid min-w-0 content-start gap-4">
      <Panel tour="inc-summary" eyebrow={`${inc.id} · ${inc.severity.toUpperCase()} · ${d.wb.describe.title}`} title={<Ticks text={`${data.scenario.title} → ${inc.title.split(" — ")[0]}`} />}
        right={<Pill color={st.color}>{st.label}</Pill>}>
        <p className="max-w-[80ch] text-[13px] leading-relaxed text-ink-2"><Ticks text={data.narrative?.summary ?? ""} /></p>
        <div className="mt-2 text-[11px] text-muted">Narrative: {data.narrative?.source === "llm" ? `Claude (${data.narrative.model})` : "rule-based RCA (no API key)"} · RCA + canary + twin took {fmtNum(data.rca_seconds ?? 0, 1)} s</div>
        {/* timeline: phase track + the ordered sequence of events */}
        <div className="mt-5">
          <div className="relative">
            <div className="flex h-2.5 overflow-hidden rounded-full">
              {data.series.map((s) => <div key={s.w} className="h-full flex-1" style={{ background: PHASE_COLOR[s.phase] ?? C.line }} title={`${s.w}m · ${s.phase}`} />)}
            </div>
            {dedupeEvents(data.timeline).map((e) => (
              <span key={e.event} className="absolute top-[-3px] h-4 w-[2px] -translate-x-1/2 rounded" title={`${e.event} · minute ${e.w}`}
                style={{ left: `${((e.w + 0.5) / (maxW + 1)) * 100}%`, background: EVENT_COLOR[e.event] ?? C.ink2 }} />
            ))}
          </div>
          <div className="num mt-1.5 flex justify-between text-[10px] text-muted"><span>0m</span><span>{maxW}m</span></div>
          <ol className="mt-3 flex flex-wrap gap-1.5">
            {dedupeEvents(data.timeline).filter((e) => e.event !== "deploy").map((e) => (
              <li key={e.event}>
                <Tip content={<Ticks text={e.detail} />}>
                  <span className="num inline-flex cursor-help items-center gap-1.5 rounded-md border border-line bg-sunken px-2 py-1 text-[11px]">
                    <span className="h-1.5 w-1.5 rounded-full" style={{ background: EVENT_COLOR[e.event] ?? C.ink2 }} />
                    <span className="text-ink">{e.event}</span><span className="text-muted">{e.w}m</span>
                  </span>
                </Tip>
              </li>
            ))}
          </ol>
        </div>
        {oc && (
          <div className="grid grid-cols-2 gap-4 border-t border-line pt-4 md:grid-cols-4">
            <KeyVal k="Detection delay" v={`${oc.detection_delay_windows} min`} />
            <KeyVal k="Fault → mitigation" v={`${oc.time_to_mitigate_windows} min`} />
            <KeyVal k={`${q.label} · fixed vs not`} v={`${fmtNum(oc.live_mean[q.name], 3)} vs ${fmtNum(oc.counterfactual_mean[q.name], 3)}`} />
            <KeyVal k="p95 · fixed vs not" v={`${fmtMs(oc.live_mean.p95_ms)} vs ${fmtMs(oc.counterfactual_mean.p95_ms)}`} />
          </div>
        )}
      </Panel>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel eyebrow="Live vs counterfactual (old config on identical traffic)" title={`${q.label} per minute`}>
          <WindowChart data={series.map((s) => ({ w: s.w, live: s.liveQ, cf: s.cfQ }))} metric={q.name} label={q.label} format={(v) => fmtNum(v, 3)}
            faultFrom={data.scenario.fault_at} alarm={data.alarm_w} promoted={showPromoted} target={data.targets?.[q.name]} betterUp height={200} />
        </Panel>
        <Panel eyebrow="Batch latency, per request" title="p95 per minute">
          <WindowChart data={series.map((s) => ({ w: s.w, live: s.liveP, cf: s.cfP }))} metric="p95_ms" label="p95" format={fmtMs}
            faultFrom={data.scenario.fault_at} alarm={data.alarm_w} promoted={showPromoted} target={data.targets?.p95_ms} betterUp={false} height={200} />
        </Panel>
      </div>

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <Panel eyebrow="Detection · Page's CUSUM on standardised windows (k = 0.5σ)" title="When the detectors fired">
          <CusumChart series={cusum} h={h} faultFrom={data.scenario.fault_at} height={170} />
          <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[11.5px]">
            {Object.entries(data.detectors ?? {}).map(([m, det], i) => (
              <span key={m} className="inline-flex items-center gap-1.5 text-ink-2"><span className="h-[2px] w-3.5 rounded" style={{ background: [C.crimsonHi, C.info, C.violet][i % 3] }} />{m} <span className="num text-muted">μ {fmtNum(det.mu, 3)} σ {fmtNum(det.sigma, 3)} {det.direction === "up" ? "↑" : "↓"}</span></span>
            ))}
          </div>
        </Panel>
        <Panel eyebrow="Root cause · evidence-scored" title={<Ticks text={data.causes?.[0]?.title ?? "—"} />}>
          <div className="flex flex-col gap-3">
            {(data.causes ?? []).map((c, i) => (
              <div key={c.kind} className={cn("rounded-md border p-3", i === 0 ? "border-crimson/40 bg-[#140a0e]" : "border-line bg-sunken")}>
                <div className="flex items-center justify-between gap-3">
                  <span className="text-[12.5px] font-medium text-ink"><Ticks text={c.title} /></span>
                  <span className="num text-[11px] text-muted">score {c.score.toFixed(2)}</span>
                </div>
                <div className="mt-1.5 h-1 rounded-full bg-line"><div className="h-1 rounded-full" style={{ width: `${c.score * 100}%`, background: i === 0 ? C.crimson : C.ink2 }} /></div>
                <ul className="mt-2 list-none space-y-0.5">{c.evidence.map((e) => <li key={e} className="num text-[11px] text-ink-2">{e}</li>)}</ul>
              </div>
            ))}
            {data.evidence && (
              <div className="overflow-x-auto">
                <table className="w-full text-[11.5px]">
                  <thead><tr className="text-left text-muted"><th className="pb-1 font-normal">Stage</th><th className="pb-1 text-right font-normal">Baseline</th><th className="pb-1 text-right font-normal">Incident</th><th className="pb-1 text-right font-normal">×</th></tr></thead>
                  <tbody>{Object.entries(data.evidence.stages).map(([k, v]) => (
                    <tr key={k} className="border-t border-line/70"><td className="num py-1 text-ink-2">{k}</td><td className="num py-1 text-right">{fmtMs(v.baseline_ms)}</td><td className="num py-1 text-right">{fmtMs(v.incident_ms)}</td>
                      <td className="num py-1 text-right" style={{ color: (v.ratio ?? 1) > 1.5 ? C.critical : C.ink2 }}>{v.ratio ? v.ratio.toFixed(1) : "—"}</td></tr>
                  ))}</tbody>
                </table>
              </div>
            )}
          </div>
        </Panel>
      </div>

      <Panel tour="inc-candidates" eyebrow={`Remediation candidates · shadow-evaluated on ${data.scenario.window_size * 2} live requests with the fault still active · twin at ${fmtNum(data.prod_rps ?? 0, 0)} req/s bursty`}
        title={`${cands.length - 1} candidates vs the incumbent`} pad={false}>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[820px] text-[12px]">
            <thead><tr className="text-left text-muted">
              <th className="px-4 py-2 font-normal">Source</th><th className="px-4 py-2 font-normal">Change</th>
              <th className="px-4 py-2 text-right font-normal">{q.label}</th><th className="px-4 py-2 text-right font-normal">p95</th>
              <th className="px-4 py-2 text-right font-normal">Targets</th><th className="px-4 py-2 text-right font-normal">Twin p95 @ 1 replica</th><th className="px-4 py-2 text-right font-normal">Replicas for SLO</th>
            </tr></thead>
            <tbody>
              {cands.map((c) => {
                const chosen = c.genome_id === prop?.genome_id;
                const tw = twinBy.get(c.origin === "incumbent" ? "incumbent" : c.genome_id);
                return (
                  <tr key={c.genome_id + c.origin} className={cn("border-t border-line/70", chosen && "bg-[#0f2a20]")}>
                    <td className="whitespace-nowrap px-4 py-2">
                      <span className="inline-flex items-center gap-2">
                        {chosen && <ShieldCheck size={13} color={C.ok} />}
                        <span className={cn(c.origin === "incumbent" ? "text-muted" : "text-ink-2")}>{c.origin}</span>
                      </span>
                    </td>
                    <td className="max-w-[340px] px-4 py-2 text-ink-2">
                      {Object.keys(c.diff).length ? (
                        <>
                          {Object.entries(c.diff).slice(0, 3).map(([k, [a, b]]) => (
                            <span key={k} className="num mr-2 inline-block text-[11px]">{k}: <span className="text-muted">{fmtValue(a)}</span>→<span className="text-ink">{fmtValue(b)}</span></span>
                          ))}
                          {Object.keys(c.diff).length > 3 && (
                            <Tip content={<span className="num">{Object.entries(c.diff).slice(3).map(([k, [a, b]]) => `${k}: ${fmtValue(a)}→${fmtValue(b)}`).join(" · ")}</span>}>
                              <span className="num cursor-help text-[11px] text-muted underline decoration-dotted">+{Object.keys(c.diff).length - 3} more</span>
                            </Tip>
                          )}
                        </>
                      ) : <span className="text-muted">{c.why}</span>}
                    </td>
                    <td className="num px-4 py-2 text-right text-ink">{c.metrics ? fmtNum(num(c.metrics[q.name]), 3) : "—"}</td>
                    <td className="num whitespace-nowrap px-4 py-2 text-right text-ink">{c.metrics ? fmtMs(num(c.metrics.p95_ms)) : "—"}</td>
                    <td className="px-4 py-2 text-right">{c.meets_targets ? <Check size={14} color={C.ok} className="ml-auto" /> : <X size={14} color={C.muted} className="ml-auto" />}</td>
                    <td className="num px-4 py-2 text-right text-ink-2">{tw ? fmtMs(tw.p95) : "—"}</td>
                    <td className="num px-4 py-2 text-right text-ink-2">{tw ? (tw.replicas_needed ?? "> 8") : "—"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Panel>

      {prop && prop.status === "awaiting_approval" || recorded ? (
        <Panel tour="inc-gate" eyebrow="Human-in-the-loop deployment gate" title={prop?.why ?? "No proposal"}
          right={mode === "recorded" && recorded ? <Button tone="ghost" onClick={() => setMode("gate")}><RotateCcw size={13} /> Replay the decision</Button> : undefined}>
          <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_320px]">
            <div className="flex flex-col gap-3">
              <div className="flex flex-wrap gap-2">
                {prop?.diff && Object.entries(prop.diff).map(([k, [a, b]]) => (
                  <span key={k} className="num rounded-md border border-line bg-sunken px-2 py-1 text-[11.5px]">{k} <span className="text-muted">{fmtValue(a)}</span> → <span className="text-ink">{fmtValue(b)}</span></span>
                ))}
              </div>
              <div className="grid grid-cols-2 gap-x-6 sm:grid-cols-4">
                <KeyVal k={`Δ ${q.label}`} v={<span style={{ color: (prop?.expected_gain?.[q.name] ?? 0) >= 0 ? C.ok : C.critical }}>{(prop?.expected_gain?.[q.name] ?? 0) >= 0 ? "+" : ""}{fmtNum(prop?.expected_gain?.[q.name], 3)}</span>} />
                <KeyVal k="Δ p95" v={<span style={{ color: (prop?.expected_gain?.p95_ms ?? 0) <= 0 ? C.ok : C.critical }}>{(prop?.expected_gain?.p95_ms ?? 0) > 0 ? "+" : "−"}{fmtMs(Math.abs(prop?.expected_gain?.p95_ms ?? 0))}</span>} />
                <KeyVal k="Replicas needed" v={`${prop?.replicas ?? "—"} (was ${prop?.incumbent_replicas_needed ?? "> 8"})`} />
                <KeyVal k="Meets targets" v={prop?.meets_targets ? "yes" : "partially"} />
              </div>
              {data.bank && <p className="text-[11.5px] leading-relaxed text-muted">Error bank <span className="num text-ink-2">{data.bank}</span>: {data.audited} low-confidence production tickets were sent to QA for labels; candidates marked “retrain” refit tier 2 on base + audited data — the CSAI self-improvement loop, run as a remediation.</p>}
            </div>
            <div className="rounded-lg border border-line bg-sunken p-4">
              {mode === "gate" || (liveAwaiting && API) ? (
                <>
                  <div className="eyebrow mb-2">Awaiting approval</div>
                  <p className="mb-3 text-[12px] leading-relaxed text-ink-2">Promote this genome to production? The old config keeps running as a shadow counterfactual; EPOCH rolls back automatically if the new one underperforms it for 3 minutes.</p>
                  <div className="flex gap-2">
                    <Button tone="primary" disabled={busy} onClick={() => onDecision(true)}><Check size={14} /> Approve &amp; promote</Button>
                    <Button tone="danger" disabled={busy} onClick={() => onDecision(false)}><X size={14} /> Reject</Button>
                  </div>
                  {!API && <p className="mt-2 text-[11px] text-muted">Replay: plays back the measured outcome of either choice.</p>}
                </>
              ) : (
                <>
                  <div className="eyebrow mb-2">Decision</div>
                  {mode === "rejected" || local === "rejected" && mode !== "recorded" ? (
                    <p className="text-[12px] leading-relaxed text-ink-2">Rejected — the chart now shows the <span className="text-ink">old config on the same traffic</span>: {q.label} stays at {fmtNum(oc?.counterfactual_mean[q.name], 3)} and p95 at {fmtMs(oc?.counterfactual_mean.p95_ms)}.</p>
                  ) : (
                    <>
                      <p className="text-[12px] leading-relaxed text-ink-2">{mode === "approved" ? "Approved (replay)" : `Approved by ${recorded?.actor}`}. Promoted at minute {gate + 1}; outcome <span className="text-ink">{oc?.status ?? inc.status}</span>{data.recovered_w != null ? `, targets met from minute ${data.recovered_w}` : ""}.</p>
                      {recorded?.note && <p className="mt-1 text-[11.5px] text-muted">“{recorded.note}”</p>}
                    </>
                  )}
                </>
              )}
            </div>
          </div>
        </Panel>
      ) : (
        <Panel tour="inc-gate" eyebrow="Human-in-the-loop deployment gate" title="No configuration-level fix — escalated to a human">
          <p className="max-w-[90ch] text-[12.5px] leading-relaxed text-ink-2"><Ticks text={prop?.why ?? ""} /></p>
          <p className="mt-2 text-[11.5px] leading-relaxed text-muted">Every candidate was replayed three times on shadow traffic; none beat the incumbent by more than the replay-to-replay spread on quality, tail latency, mean latency (capacity) or replicas needed. Shipping one anyway would be a placebo change.</p>
        </Panel>
      )}
      <p className="text-[11px] text-muted">Targets: {q.label} ≥ {fmtNum(data.targets?.[q.name], 3)} (baseline − 0.03) and p95 ≤ {fmtMs(data.targets?.p95_ms)} (1.25× baseline). Production twin SLO {fmtMs(data.twin_slo_ms)}. Traffic per window: {data.scenario.window_size} requests{data.scenario.kind === "drift" ? `, drift mix ramps to ${fmtPct(data.scenario.drift_mix, 0)}` : `, +${data.scenario.spike_ms} ms per ${data.scenario.stage} call`}.</p>
    </div>
  );
}

function dedupeEvents(tl: Incident["data"]["timeline"]) {
  const seen = new Set<string>();
  return tl.filter((e) => { const k = e.event; if (seen.has(k)) return false; seen.add(k); return true; });
}
