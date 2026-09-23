"use client";

import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useMemo, useState } from "react";
import { Logo } from "@/components/Logo";
import { useBundle } from "@/lib/data";
import { fmtNum, fmtPct } from "@/lib/format";
import { useUI } from "@/lib/store";

/** Mission-control boot sequence. Lines are real numbers from the bundle; any key or click skips. */
export function Boot() {
  const { data, error } = useBundle();
  const { booted, setBooted } = useUI();
  const [n, setN] = useState(0);

  const lines = useMemo(() => {
    if (!data) return ["mounting experiment memory"];
    const W = Object.values(data.workloads);
    const trials = W.reduce((a, w) => a + w.trials.length, 0);
    const hv = Math.max(...W.map((w) => w.global_hv));
    const inc = W.reduce((a, w) => a + w.incidents.length, 0);
    const tw = W.find((w) => w.twin?.validation && !w.twin.validation.error)?.twin?.validation;
    const p = data.meta.provenance;
    return [
      `mounting experiment memory ··· ${trials.toLocaleString()} trials · ${W.reduce((a, w) => a + w.runs.length, 0)} runs`,
      `hydrating pareto archive ····· HV ${fmtNum(hv, 3)}`,
      `loading genome space ········· ${data.meta.workloads.join(" + ")}`,
      tw ? `calibrating digital twin ····· p95 err ${fmtPct(tw.mape_p95, 0)}` : "calibrating digital twin ····· skipped",
      `arming CUSUM detectors ······· ${inc} incidents on tape`,
      `agents ······················· ${data.meta.llm.enabled ? data.meta.llm.agent_model : "heuristic (offline)"}`,
      `provenance ··················· ${p.gpu ?? `${p.cpu_count}-core CPU`} · env ${p.env_hash}`,
    ];
  }, [data]);

  const skipBoot = typeof window !== "undefined" && (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches || /tour/.test(window.location.hash) || document.visibilityState === "hidden");

  useEffect(() => {
    if (booted) return;
    if (skipBoot && data) { setBooted(); return; }
    if (!data) return;
    if (n < lines.length) { const t = setTimeout(() => setN((x) => x + 1), 150); return () => clearTimeout(t); }
    const t = setTimeout(setBooted, 520);
    return () => clearTimeout(t);
  }, [n, lines.length, data, booted, setBooted, skipBoot]);

  useEffect(() => {
    if (booted || !data) return;
    const skip = () => setBooted();
    window.addEventListener("keydown", skip, { once: true });
    window.addEventListener("pointerdown", skip, { once: true });
    return () => { window.removeEventListener("keydown", skip); window.removeEventListener("pointerdown", skip); };
  }, [booted, data, setBooted]);

  return (
    <AnimatePresence>
      {!booted && !error && (
        <motion.div key="boot" exit={{ opacity: 0, scale: 1.02, filter: "blur(6px)" }} transition={{ duration: 0.45 }}
          className="blueprint fixed inset-0 z-[80] grid place-items-center px-5">
          <div className="boot-glow pointer-events-none absolute inset-0" />
          <div className="relative w-full max-w-[560px]">
            <div className="mb-6 flex items-center gap-3">
              <div className="logo-spin"><Logo size={40} /></div>
              <div>
                <div className="font-mono text-[22px] font-semibold tracking-[0.34em] text-ink">EPOCH</div>
                <div className="num text-[11px] tracking-[0.12em] text-muted">MISSION CONTROL · {data ? (data.meta.mode === "live" ? "LIVE" : "REPLAY") : "BOOTING"}</div>
              </div>
            </div>
            <div className="num min-h-[170px] space-y-1.5 text-[12px]">
              {lines.slice(0, Math.max(1, n)).map((l, i) => (
                <motion.div key={l} initial={{ opacity: 0, x: -6 }} animate={{ opacity: 1, x: 0 }} className="flex gap-2 text-ink-2">
                  <span className="text-crimson-hi">▸</span><span className="truncate">{l}</span>
                  <span className={i < n - 1 || (data && n >= lines.length) ? "ml-auto text-ok" : "ml-auto text-warn live-dot"}>{i < n - 1 || (data && n >= lines.length) ? "ok" : "··"}</span>
                </motion.div>
              ))}
            </div>
            <div className="mt-5 h-[2px] overflow-hidden rounded-full bg-[#1a1e2a]">
              <motion.div className="h-full bg-gradient-to-r from-crimson to-crimson-hi" animate={{ width: `${data ? Math.round((n / lines.length) * 100) : 12}%` }} transition={{ duration: 0.2 }} />
            </div>
            <div className="num mt-3 text-[10.5px] text-muted">press any key to skip</div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
