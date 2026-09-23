"use client";

import { motion } from "framer-motion";
import { Boxes, Command as CmdIcon, Cpu, PlayCircle } from "lucide-react";
import { useEffect, useRef, type ComponentType } from "react";
import { Boot } from "@/components/Boot";
import { HelpSheet, Palette, useShortcuts, VIEW_META } from "@/components/Palette";
import { Story } from "@/components/Story";
import { Logo } from "@/components/Logo";
import { Pill, Segmented } from "@/components/ui";
import { API, useBundle, useLiveEvents } from "@/lib/data";
import { cn, fmtNum, fmtPct } from "@/lib/format";
import { hashFor, readHash, useUI, VIEW_IDS, type View } from "@/lib/store";
import { ArchitectureView } from "@/views/Architecture";
import { CommandView } from "@/views/Command";
import { EvolutionView } from "@/views/Evolution";
import { IncidentsView } from "@/views/Incidents";
import { InsightsView } from "@/views/Insights";
import { LineageView } from "@/views/Lineage";
import { TrialsView } from "@/views/Trials";
import { TwinView } from "@/views/Twin";

const NAV = VIEW_IDS.map((id) => ({ id, ...VIEW_META[id] }));

const VIEWS: Record<View, ComponentType> = {
  command: CommandView, evolution: EvolutionView, lineage: LineageView, trials: TrialsView,
  insights: InsightsView, twin: TwinView, incidents: IncidentsView, architecture: ArchitectureView,
};

export function Shell() {
  const { data, error, isLoading } = useBundle();
  const { view, setView, workload, setWorkload } = useUI();
  useLiveEvents();

  useShortcuts();
  const { incidentId, trialId, story } = useUI();
  const tourAsked = useRef(false);

  // deep links: read on load + on back/forward, write on every navigation
  useEffect(() => {
    const apply = () => {
      const h = readHash();
      const s = useUI.getState();
      if (h.workload) s.setWorkload(h.workload);
      if (h.view) s.setView(h.view);
      if (h.incidentId) s.selectIncident(h.incidentId);
      if (h.trialId) s.selectTrial(h.trialId);
      if (h.tour) tourAsked.current = true;
    };
    apply();
    window.addEventListener("hashchange", apply);
    return () => window.removeEventListener("hashchange", apply);
  }, []);
  useEffect(() => {
    const h = hashFor({ view, workload, incidentId, trialId });
    if (window.location.hash !== h) window.history.replaceState(null, "", h);
  }, [view, workload, incidentId, trialId]);
  useEffect(() => { if (useUI.getState().story == null) window.scrollTo({ top: 0 }); }, [view]);
  useEffect(() => { if (data && tourAsked.current) { tourAsked.current = false; useUI.getState().startStory(0); } }, [data]);

  useEffect(() => {
    if (data && !data.workloads[workload]) setWorkload(data.meta.workloads[0]);
  }, [data, workload, setWorkload]);

  const openIncidents = data ? Object.values(data.workloads).flatMap((w) => w.incidents).filter((i) => i.status === "awaiting_approval").length : 0;
  const View = VIEWS[view];
  const prov = data?.meta.provenance;

  return (
    <div className="blueprint flex min-h-full flex-col lg:flex-row" style={{ backgroundColor: "transparent" }}>
      {/* rail */}
      <aside className="sticky top-0 z-30 flex shrink-0 flex-col border-b border-line bg-[#0a0b10]/95 backdrop-blur lg:h-screen lg:w-[220px] lg:border-b-0 lg:border-r">
        <div className="flex items-center gap-2.5 px-4 py-3 lg:px-5 lg:py-5">
          <Logo />
          <div className="flex flex-col leading-none">
            <span className="font-mono text-[15px] font-semibold tracking-[0.32em] text-ink">EPOCH</span>
            <span className="mt-1 text-[10.5px] text-muted">AI systems evolution</span>
          </div>
        </div>
        <nav className="flex gap-1 overflow-x-auto px-3 pb-2 lg:flex-col lg:gap-0.5 lg:px-3 lg:pb-0" aria-label="Views">
          {NAV.map(({ id, label, icon: Icon, hint }) => {
            const active = id === view;
            return (
              <button key={id} onClick={() => setView(id)} title={hint}
                className={cn("group relative flex shrink-0 items-center gap-2.5 rounded-md px-2.5 py-2 text-left text-[13px] transition-colors",
                  active ? "bg-[#1a0c12] text-ink" : "text-ink-2 hover:bg-raised hover:text-ink")}>
                {active && <motion.span layoutId="nav-active" className="absolute inset-y-1.5 left-0 hidden w-[2px] rounded-full bg-crimson lg:block" />}
                <Icon size={15} strokeWidth={1.75} />
                <span className="whitespace-nowrap">{label}</span>
                <kbd className="num ml-auto hidden text-[10px] text-muted/70 lg:inline">{VIEW_IDS.indexOf(id) + 1}</kbd>
                {id === "incidents" && openIncidents > 0 && (
                  <span className="rounded-full bg-crimson px-1.5 text-[10px] font-semibold text-white">{openIncidents}</span>
                )}
              </button>
            );
          })}
        </nav>
        <div className="mt-auto hidden border-t border-line px-5 py-4 lg:block">
          <div className="eyebrow mb-2">Measured on</div>
          <div className="flex items-center gap-2 text-[12px] text-ink-2"><Cpu size={13} /> {prov ? (prov.gpu ?? `${prov.cpu_count}-core CPU`) : "—"}</div>
          <div className="mt-1.5 flex items-center gap-2 text-[12px] text-ink-2"><Boxes size={13} /> env {prov?.env_hash ?? "—"}</div>
          <div className="mt-3 text-[11px] leading-relaxed text-muted">v{data?.meta.epoch_version ?? "—"} · {data ? new Date(data.meta.generated_at).toLocaleString() : ""}</div>
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="topline flex flex-wrap items-center gap-x-4 gap-y-2 border-b border-line bg-[#0a0b10]/70 px-4 py-3 backdrop-blur sm:px-6">
          <div className="min-w-0">
            <div className="eyebrow">{NAV.find((n) => n.id === view)?.hint}</div>
            <h1 className="text-[18px] font-semibold tracking-[-0.015em] text-ink">{NAV.find((n) => n.id === view)?.label}</h1>
          </div>
          <div className="ml-auto flex flex-wrap items-center gap-2">
            <button onClick={() => useUI.getState().startStory(0)} className="tour-cta inline-flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-[12px] font-medium text-white">
              <PlayCircle size={14} /> {story != null ? "Restart tour" : "Guided tour"}
            </button>
            <button onClick={() => useUI.getState().setPalette(true)} className="inline-flex items-center gap-2 rounded-md border border-line-strong bg-sunken px-2.5 py-1.5 text-[12px] text-muted hover:text-ink-2">
              <CmdIcon size={13} /> <span className="hidden sm:inline">Jump to…</span> <kbd className="kbd">Ctrl K</kbd>
            </button>
            {data && data.meta.workloads.length > 1 && (
              <Segmented value={workload} onChange={setWorkload}
                options={data.meta.workloads.map((w) => ({ value: w, label: data.workloads[w]?.describe.title ?? w }))} />
            )}
            {data && (API
              ? <Pill color="#34d399"><span className="live-dot">LIVE</span> · {API.replace(/^https?:\/\//, "")}</Pill>
              : <Pill color="#a9b0bf">REPLAY · measured run</Pill>)}
            {data && <Pill color={data.meta.llm.enabled ? "#a78bfa" : "#7a8296"}>{data.meta.llm.enabled ? `Agents · ${data.meta.llm.agent_model}` : "Agents · heuristic mode"}</Pill>}
          </div>
        </header>

        <main className="min-w-0 flex-1 px-4 py-5 sm:px-6">
          {isLoading && <div className="py-24 text-center text-[13px] text-muted">Loading experiment memory…</div>}
          {error && (
            <div className="panel mx-auto mt-10 max-w-xl p-6">
              <div className="eyebrow mb-2 text-critical">No data</div>
              <p className="text-[13px] text-ink-2">{String((error as Error).message)}</p>
            </div>
          )}
          {data && (
            <div key={view + workload} className="stagger view-in">
              <View />
            </div>
          )}
        </main>
        {data && <StatusBar />}
      </div>
      <Palette />
      <HelpSheet />
      <Story />
      <Boot />
    </div>
  );
}

function StatusBar() {
  const { data } = useBundle();
  const { workload, story } = useUI();
  if (!data) return null;
  const wb = data.workloads[workload] ?? data.workloads[data.meta.workloads[0]];
  const inc = wb.incidents;
  const healed = inc.filter((i) => ["resolved", "mitigated"].includes(i.status)).length;
  const v = wb.twin?.validation;
  const cells: [string, string, string?][] = [
    ["HV", fmtNum(wb.global_hv, 3)], ["configs", wb.trials.length.toLocaleString()], ["pareto", String(wb.pareto.length)],
    ...(v && !v.error ? [["twin p95 err", fmtPct(v.mape_p95, 0)] as [string, string]] : []),
    ...(inc.length ? [["healed", `${healed}/${inc.length}`, healed === inc.length ? "#34d399" : "#fbbf24"] as [string, string, string]] : []),
  ];
  return (
    <footer className={cn("sticky bottom-0 z-20 hidden items-center gap-5 border-t border-line bg-[#0a0b10]/90 px-6 py-1.5 backdrop-blur md:flex", story != null && "md:hidden")}>
      <span className="num flex items-center gap-1.5 text-[10.5px] tracking-[0.12em] text-ok"><span className="live-dot h-1.5 w-1.5 rounded-full bg-ok" />{API ? "LIVE" : "REPLAY"}</span>
      <span className="num text-[10.5px] text-muted">{wb.describe.title}</span>
      {cells.map(([k, val, c]) => (
        <span key={k} className="num text-[10.5px] text-muted">{k} <span style={{ color: c ?? "#e8eaf0" }}>{val}</span></span>
      ))}
      <span className="num ml-auto text-[10.5px] text-muted"><kbd className="kbd">Ctrl K</kbd> jump · <kbd className="kbd">1–8</kbd> views · <kbd className="kbd">W</kbd> workload · <kbd className="kbd">T</kbd> tour · <kbd className="kbd">?</kbd> keys</span>
    </footer>
  );
}
