"use client";

import * as Tooltip from "@radix-ui/react-tooltip";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { GROUP_COLOR } from "@/lib/theme";
import { cn, fmtValue } from "@/lib/format";
import type { Gene, Genome } from "@/lib/types";

export function Panel({ title, eyebrow, right, children, className, pad = true, tour }: {
  title?: ReactNode; eyebrow?: ReactNode; right?: ReactNode; children: ReactNode; className?: string; pad?: boolean; tour?: string;
}) {
  return (
    <section data-tour={tour} className={cn("panel flex min-w-0 flex-col", className)}>
      {(title || eyebrow || right) && (
        <header className="flex flex-wrap items-start justify-between gap-x-4 gap-y-2 border-b border-line px-4 py-3">
          <div className="min-w-0">
            {eyebrow && <div className="eyebrow mb-1">{eyebrow}</div>}
            {title && <h2 className="text-[13.5px] font-medium tracking-[-0.005em] text-ink [text-wrap:balance]">{title}</h2>}
          </div>
          {right && <div className="flex shrink-0 flex-wrap items-center gap-2">{right}</div>}
        </header>
      )}
      <div className={cn("min-w-0 flex-1", pad && "p-4")}>{children}</div>
    </section>
  );
}

export function Stat({ label, value, sub, tone, big }: { label: string; value: ReactNode; sub?: ReactNode; tone?: string; big?: boolean }) {
  return (
    <div className="flex min-w-0 flex-col gap-1">
      <div className="eyebrow truncate">{label}</div>
      <div className={cn("font-semibold tracking-[-0.02em] text-ink", big ? "text-[34px] leading-none" : "text-[22px] leading-tight")}
        style={tone ? { color: tone, textShadow: `0 0 18px ${tone}55` } : undefined}>{typeof value === "string" ? <CountUp text={value} /> : value}</div>
      {sub && <div className="line-clamp-2 text-[12px] leading-snug text-muted">{sub}</div>}
    </div>
  );
}

export function Pill({ color, children, solid, className }: { color: string; children: ReactNode; solid?: boolean; className?: string }) {
  return (
    <span className={cn("inline-flex items-center gap-1.5 whitespace-nowrap rounded-full px-2 py-[3px] text-[11px] font-medium leading-none", className)}
      style={solid ? { background: color, color: "#0b0b0e" } : { background: `${color}1f`, color, boxShadow: `inset 0 0 0 1px ${color}40` }}>
      {!solid && <span className="h-1.5 w-1.5 rounded-full" style={{ background: color }} />}
      {children}
    </span>
  );
}

export function Segmented<T extends string>({ value, options, onChange, size = "sm" }: {
  value: T; options: { value: T; label: ReactNode }[]; onChange: (v: T) => void; size?: "sm" | "xs";
}) {
  return (
    <div role="tablist" className="inline-flex rounded-md border border-line bg-sunken p-0.5">
      {options.map((o) => (
        <button key={o.value} role="tab" aria-selected={o.value === value} onClick={() => onChange(o.value)}
          className={cn("rounded-[5px] font-medium transition-colors", size === "sm" ? "px-2.5 py-1 text-[12px]" : "px-2 py-0.5 text-[11px]",
            o.value === value ? "bg-raised text-ink shadow-[inset_0_0_0_1px_#2b3040]" : "text-muted hover:text-ink-2")}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Button({ children, onClick, tone = "default", disabled, className, type = "button" }: {
  children: ReactNode; onClick?: () => void; tone?: "default" | "primary" | "danger" | "ghost"; disabled?: boolean; className?: string; type?: "button" | "submit";
}) {
  const tones = {
    default: "bg-raised text-ink border border-line-strong hover:bg-[#1d2130]",
    primary: "bg-crimson text-white border border-crimson hover:bg-crimson-hi",
    danger: "bg-transparent text-critical border border-[#f43f5e55] hover:bg-[#f43f5e14]",
    ghost: "bg-transparent text-ink-2 border border-transparent hover:bg-raised",
  };
  return (
    <button type={type} onClick={onClick} disabled={disabled}
      className={cn("inline-flex items-center justify-center gap-1.5 rounded-md px-3 py-1.5 text-[12.5px] font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40", tones[tone], className)}>
      {children}
    </button>
  );
}

export function Tip({ content, children }: { content: ReactNode; children: ReactNode }) {
  return (
    <Tooltip.Root delayDuration={120}>
      <Tooltip.Trigger asChild>{children}</Tooltip.Trigger>
      <Tooltip.Portal>
        <Tooltip.Content sideOffset={6} className="z-50 max-w-[320px] rounded-md border border-line-strong bg-[#12151d] px-2.5 py-2 text-[12px] leading-snug text-ink-2 shadow-xl shadow-black/40">
          {content}
        </Tooltip.Content>
      </Tooltip.Portal>
    </Tooltip.Root>
  );
}

export function LegendItem({ color, label, shape = "line", value }: { color: string; label: ReactNode; shape?: "line" | "dot" | "square" | "rect"; value?: ReactNode }) {
  const mark = shape === "line"
    ? <span className="inline-block h-[2px] w-3.5 rounded" style={{ background: color }} />
    : shape === "rect" ? <span className="inline-block h-2.5 w-2.5 rounded-[2px]" style={{ background: color }} />
    : <span className={cn("inline-block h-2 w-2", shape === "dot" ? "rounded-full" : "rounded-[1px]")} style={{ background: color }} />;
  return (
    <span className="inline-flex items-center gap-1.5 text-[11.5px] text-ink-2">
      {mark}{label}{value !== undefined && <span className="num text-ink">{value}</span>}
    </span>
  );
}

/** A genome as a strip of gene cells: hue = gene group, fill = value position within its domain. */
export function GenomeStrip({ genome, space, height = 10, highlight }: { genome: Genome; space: Gene[]; height?: number; highlight?: Set<string> }) {
  return (
    <div className="flex items-stretch gap-[2px]" style={{ height }}>
      {space.map((g) => {
        const v = genome[g.name];
        const active = v !== undefined;
        let u = 0.5;
        if (active) {
          if (g.kind === "cat") u = (g.choices.findIndex((c) => c === v) + 1) / Math.max(1, g.choices.length);
          else if (g.low !== null && g.high !== null) {
            const x = Number(v);
            u = g.log ? (Math.log(x) - Math.log(g.low)) / (Math.log(g.high) - Math.log(g.low)) : (x - g.low) / (g.high - g.low || 1);
          }
        }
        const col = GROUP_COLOR[g.group] ?? "#7a8296";
        const hi = highlight?.has(g.name);
        return (
          <Tip key={g.name} content={<span><span className="num text-ink">{g.name}</span> = {active ? fmtValue(v) : "inactive"}<br /><span className="text-muted">{g.doc}</span></span>}>
            <span className="block w-[9px] rounded-[2px]"
              style={{ background: active ? col : "transparent", opacity: active ? 0.25 + 0.75 * Math.max(0, Math.min(1, u)) : 1,
                boxShadow: active ? (hi ? `0 0 0 1.5px #fff` : undefined) : "inset 0 0 0 1px #2b3040" }} />
          </Tip>
        );
      })}
    </div>
  );
}

export function KeyVal({ k, v, mono = true }: { k: ReactNode; v: ReactNode; mono?: boolean }) {
  return (
    <div className="flex items-baseline justify-between gap-3 border-b border-line/70 py-1.5 last:border-0">
      <span className="text-[12px] text-muted">{k}</span>
      <span className={cn("text-right text-[12.5px] text-ink", mono && "num")}>{v}</span>
    </div>
  );
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 px-6 py-10 text-center">
      <div className="text-[13px] font-medium text-ink-2">{title}</div>
      {children && <div className="max-w-md text-[12px] text-muted">{children}</div>}
    </div>
  );
}

/** Renders `code` spans in agent / RCA text as mono chips. */
export function Ticks({ text }: { text: string }) {
  const parts = text.split(/(`[^`]+`)/g);
  return (
    <>
      {parts.map((p, i) => p.startsWith("`") && p.endsWith("`")
        ? <code key={i} className="num rounded-[4px] bg-[#1a1e2a] px-1 py-[1px] text-[0.92em] text-ink">{p.slice(1, -1)}</code>
        : <span key={i}>{p}</span>)}
    </>
  );
}

export function Mono({ children, className }: { children: ReactNode; className?: string }) {
  return <span className={cn("num text-[12px]", className)}>{children}</span>;
}

/** Animates the first number inside a formatted string (keeps prefix, suffix, separators and decimals). */
export function CountUp({ text, ms = 1100 }: { text: string; ms?: number }) {
  const m = /-?[\d,]*\.?\d+/.exec(text);
  const [shown, setShown] = useState(text);
  const raf = useRef(0);
  useEffect(() => {
    if (!m || (typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches)) { setShown(text); return; }
    const raw = m[0];
    const target = Number(raw.replace(/,/g, ""));
    const dec = raw.includes(".") ? raw.split(".")[1].length : 0;
    const commas = raw.includes(",");
    const pre = text.slice(0, m.index), post = text.slice(m.index + raw.length);
    const t0 = performance.now();
    const tick = (t: number) => {
      const k = Math.min(1, (t - t0) / ms);
      const e = 1 - Math.pow(1 - k, 3);
      const v = target * e;
      const s = commas ? v.toLocaleString("en-US", { minimumFractionDigits: dec, maximumFractionDigits: dec }) : v.toFixed(dec);
      setShown(pre + s + post);
      if (k < 1) raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [text, ms]);
  return <span className="num">{shown}</span>;
}
