"use client";

import * as Dialog from "@radix-ui/react-dialog";
import { Activity, CornerDownLeft, FlaskConical, GitBranch, Keyboard, Link2, Network, PlayCircle, Radar, Rows3, Search, Siren, Sparkles, Waves, type LucideIcon } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { useBundle } from "@/lib/data";
import { cn, fmtMs, fmtNum } from "@/lib/format";
import { hashFor, useUI, VIEW_IDS, type View } from "@/lib/store";
import { C, INCIDENT_STATUS, S_COLOR, S_LABEL } from "@/lib/theme";

export const VIEW_META: Record<View, { label: string; icon: LucideIcon; hint: string }> = {
  command: { label: "Command", icon: Radar, hint: "Deployed pipeline, health and the event log" },
  evolution: { label: "Evolution", icon: Activity, hint: "Pareto frontier and the strategy race" },
  lineage: { label: "Lineage", icon: GitBranch, hint: "Genome ancestry of a run" },
  trials: { label: "Trials", icon: Rows3, hint: "Every measured configuration" },
  insights: { label: "Insights", icon: FlaskConical, hint: "Gene importance, ablations, hypotheses" },
  twin: { label: "Digital twin", icon: Waves, hint: "Simulate load and faults in the browser" },
  incidents: { label: "Incidents", icon: Siren, hint: "Detection, RCA, remediation and the approval gate" },
  architecture: { label: "Architecture", icon: Network, hint: "How EPOCH itself is built" },
};

interface Item { id: string; group: string; label: string; sub?: string; icon: LucideIcon; color?: string; kbd?: string; run: () => void }

function score(q: string, text: string): number {
  if (!q) return 1;
  const t = text.toLowerCase(), s = q.toLowerCase();
  const i = t.indexOf(s);
  if (i >= 0) return 100 - i;
  let k = 0, gaps = 0, last = -1;
  for (let j = 0; j < t.length && k < s.length; j++) if (t[j] === s[k]) { if (last >= 0) gaps += j - last - 1; last = j; k++; }
  return k === s.length ? 50 - Math.min(49, gaps) : 0;
}

export function Palette() {
  const { data } = useBundle();
  const { paletteOpen, setPalette } = useUI();
  const [q, setQ] = useState("");
  const [sel, setSel] = useState(0);
  const listRef = useRef<HTMLDivElement>(null);

  const items = useMemo<Item[]>(() => {
    const ui = useUI.getState;
    const out: Item[] = [];
    out.push({ id: "tour", group: "Actions", label: "Start the guided tour", sub: "90-second walkthrough of the whole platform", icon: PlayCircle, color: C.crimsonHi, kbd: "T", run: () => ui().startStory(0) });
    out.push({ id: "copy", group: "Actions", label: "Copy link to this view", sub: "deep link with workload and selection", icon: Link2, run: () => { const s = ui(); navigator.clipboard?.writeText(window.location.origin + window.location.pathname + hashFor(s)); } });
    out.push({ id: "keys", group: "Actions", label: "Keyboard shortcuts", icon: Keyboard, kbd: "?", run: () => ui().setHelp(true) });
    VIEW_IDS.forEach((v, i) => { const m = VIEW_META[v]; out.push({ id: `v-${v}`, group: "Views", label: m.label, sub: m.hint, icon: m.icon, kbd: String(i + 1), run: () => ui().setView(v) }); });
    if (!data) return out;
    for (const w of data.meta.workloads) {
      const wb = data.workloads[w];
      out.push({ id: `w-${w}`, group: "Workloads", label: `Switch to ${wb.describe.title}`, sub: `${wb.trials.length} trials · ${wb.runs.length} runs`, icon: Sparkles, kbd: "W", run: () => ui().setWorkload(w) });
      for (const inc of wb.incidents) {
        const st = INCIDENT_STATUS[inc.status] ?? { color: C.muted, label: inc.status };
        out.push({ id: `i-${inc.id}`, group: "Incidents", label: `${inc.id} · ${inc.data.scenario.title}`, sub: `${wb.describe.title} · ${st.label}`, icon: Siren, color: st.color,
          run: () => { const s = ui(); s.setWorkload(w); s.setView("incidents"); s.selectIncident(inc.id); } });
      }
      const q = wb.describe.quality_metric;
      const front = wb.pareto.map((id) => wb.trials.find((t) => t.id === id)!).filter(Boolean).slice(0, 14);
      for (const t of front) out.push({ id: `t-${t.id}`, group: "Pareto configs", label: `Trial #${t.number} · ${S_LABEL[t.strategy]}${t.id === wb.knee ? " · knee" : ""}`,
        sub: `${wb.describe.title} · ${q} ${fmtNum(t.metrics[q] ?? 0, 3)} · p95 ${fmtMs(t.metrics.p95_ms)}`, icon: Rows3, color: S_COLOR[t.strategy],
        run: () => { const s = ui(); s.setWorkload(w); s.setView("trials"); s.selectTrial(t.id); } });
    }
    return out;
  }, [data]);

  const shown = useMemo(() => items.map((it) => ({ it, s: score(q, `${it.label} ${it.sub ?? ""} ${it.group}`) })).filter((x) => x.s > 0)
    .sort((a, b) => (q ? b.s - a.s : 0)).map((x) => x.it).slice(0, 40), [items, q]);

  useEffect(() => { setSel(0); }, [q, paletteOpen]);
  useEffect(() => { if (!paletteOpen) setQ(""); }, [paletteOpen]);
  useEffect(() => { listRef.current?.querySelector(`[data-idx="${sel}"]`)?.scrollIntoView({ block: "nearest" }); }, [sel]);

  const run = (it?: Item) => { if (!it) return; setPalette(false); setTimeout(it.run, 10); };
  let lastGroup = "";

  return (
    <Dialog.Root open={paletteOpen} onOpenChange={setPalette}>
      <Dialog.Portal>
        <Dialog.Overlay className="palette-overlay fixed inset-0 z-[60] bg-black/60 backdrop-blur-[2px]" />
        <Dialog.Content aria-describedby={undefined}
          className="palette-in fixed left-1/2 top-[12vh] z-[61] w-[min(640px,calc(100vw-24px))] -translate-x-1/2 overflow-hidden rounded-xl border border-line-strong bg-[#0e1017] shadow-[0_30px_80px_-10px_rgba(0,0,0,.85),0_0_0_1px_rgba(225,29,72,.18)]">
          <Dialog.Title className="sr-only">Command palette</Dialog.Title>
          <div className="flex items-center gap-2.5 border-b border-line px-4 py-3">
            <Search size={16} className="text-crimson-hi" />
            <input autoFocus value={q} onChange={(e) => setQ(e.target.value)} placeholder="Jump to a view, incident, Pareto config or action…"
              className="min-w-0 flex-1 bg-transparent text-[14px] text-ink outline-none placeholder:text-muted"
              onKeyDown={(e) => {
                if (e.key === "ArrowDown") { e.preventDefault(); setSel((s) => Math.min(shown.length - 1, s + 1)); }
                else if (e.key === "ArrowUp") { e.preventDefault(); setSel((s) => Math.max(0, s - 1)); }
                else if (e.key === "Enter") { e.preventDefault(); run(shown[sel]); }
              }} />
            <kbd className="kbd">esc</kbd>
          </div>
          <div ref={listRef} className="max-h-[min(440px,60vh)] overflow-y-auto p-1.5">
            {shown.length === 0 && <div className="px-3 py-8 text-center text-[12.5px] text-muted">Nothing matches “{q}”.</div>}
            {shown.map((it, i) => {
              const head = it.group !== lastGroup ? (lastGroup = it.group) : null;
              const Icon = it.icon;
              return (
                <div key={it.id}>
                  {head && <div className="eyebrow px-2.5 pb-1 pt-2.5 !text-[9.5px]">{head}</div>}
                  <button data-idx={i} onMouseMove={() => setSel(i)} onClick={() => run(it)}
                    className={cn("flex w-full items-center gap-3 rounded-lg px-2.5 py-2 text-left", i === sel ? "bg-[#1a0c12] shadow-[inset_0_0_0_1px_#e11d4855]" : "")}>
                    <span className="grid h-7 w-7 shrink-0 place-items-center rounded-md bg-raised" style={{ color: it.color ?? "#a9b0bf" }}><Icon size={14} /></span>
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-[13px] text-ink">{it.label}</span>
                      {it.sub && <span className="num block truncate text-[11px] text-muted">{it.sub}</span>}
                    </span>
                    {it.kbd && <kbd className="kbd">{it.kbd}</kbd>}
                    {i === sel && <CornerDownLeft size={13} className="text-crimson-hi" />}
                  </button>
                </div>
              );
            })}
          </div>
          <div className="num flex items-center gap-3 border-t border-line px-4 py-2 text-[10.5px] text-muted">
            <span><kbd className="kbd">↑</kbd> <kbd className="kbd">↓</kbd> navigate</span><span><kbd className="kbd">↵</kbd> open</span><span className="ml-auto">{shown.length} results</span>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

const KEYS: [string, string][] = [["Ctrl K  /  ⌘K  /  /", "Command palette"], ["1 – 8", "Switch view"], ["W", "Toggle workload"], ["T", "Start guided tour"],
  ["← →  ·  space  ·  esc", "Tour: step · pause · exit"], ["?", "This sheet"]];

export function HelpSheet() {
  const { helpOpen, setHelp } = useUI();
  return (
    <Dialog.Root open={helpOpen} onOpenChange={setHelp}>
      <Dialog.Portal>
        <Dialog.Overlay className="palette-overlay fixed inset-0 z-[60] bg-black/60" />
        <Dialog.Content aria-describedby={undefined} className="palette-in fixed left-1/2 top-[18vh] z-[61] w-[min(420px,calc(100vw-24px))] -translate-x-1/2 rounded-xl border border-line-strong bg-[#0e1017] p-5">
          <Dialog.Title className="mb-3 text-[14px] font-semibold text-ink">Keyboard shortcuts</Dialog.Title>
          {KEYS.map(([k, v]) => (
            <div key={k} className="flex items-center justify-between border-b border-line/70 py-2 text-[12.5px] last:border-0">
              <span className="text-ink-2">{v}</span><kbd className="kbd">{k}</kbd>
            </div>
          ))}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

/** Global shortcuts. Ignored while typing in inputs or while a dialog is open. */
export function useShortcuts() {
  const { data } = useBundle();
  useEffect(() => {
    const on = (e: KeyboardEvent) => {
      const s = useUI.getState();
      const typing = (e.target as HTMLElement)?.closest?.("input,textarea,select,[contenteditable=true]");
      if ((e.key === "k" || e.key === "K") && (e.metaKey || e.ctrlKey)) { e.preventDefault(); s.setPalette(!s.paletteOpen); return; }
      if (typing || s.paletteOpen || s.helpOpen || e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === "/") { e.preventDefault(); s.setPalette(true); }
      else if (e.key === "?") s.setHelp(true);
      else if (e.key === "t" || e.key === "T") s.startStory(0);
      else if ((e.key === "w" || e.key === "W") && data) {
        const ws = data.meta.workloads; s.setWorkload(ws[(ws.indexOf(s.workload) + 1) % ws.length]);
      } else if (/^[1-8]$/.test(e.key)) s.setView(VIEW_IDS[Number(e.key) - 1]);
    };
    window.addEventListener("keydown", on);
    return () => window.removeEventListener("keydown", on);
  }, [data]);
}
