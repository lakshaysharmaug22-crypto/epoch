import type { Objective } from "./types";

export function fmtNum(v: number | null | undefined, digits = 3): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a !== 0 && (a < 0.001 || a >= 1e6)) return v.toExponential(1);
  if (a >= 1000) return v.toLocaleString("en-US", { maximumFractionDigits: 0 });
  if (a >= 100) return v.toFixed(1);
  if (a >= 10) return v.toFixed(2);
  return v.toFixed(digits);
}

export function fmtMs(v: number | null | undefined): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  if (v >= 10000) return `${(v / 1000).toFixed(1)} s`;
  if (v >= 1000) return `${(v / 1000).toFixed(2)} s`;
  if (v >= 100) return `${v.toFixed(0)} ms`;
  if (v >= 10) return `${v.toFixed(1)} ms`;
  return `${v.toFixed(2)} ms`;
}

export function fmtUsd(v: number | null | undefined): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  if (v === 0) return "$0";
  if (v < 0.01) return `$${v.toExponential(1)}`;
  return `$${v.toFixed(v < 1 ? 3 : 2)}`;
}

export function fmtPct(v: number | null | undefined, digits = 1): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  return `${(v * 100).toFixed(digits)}%`;
}

export function fmtObjective(o: Objective | undefined, v: number | null | undefined): string {
  if (!o) return fmtNum(v);
  if (o.unit === "ms") return fmtMs(v);
  if (o.unit === "$") return fmtUsd(v);
  if (o.unit === "MB") return v == null ? "—" : `${fmtNum(v, 2)} MB`;
  return fmtNum(v);
}

export function fmtValue(v: unknown): string {
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : fmtNum(v, 3);
  if (typeof v === "boolean") return v ? "on" : "off";
  if (v === null || v === undefined) return "—";
  return String(v);
}

export function fmtDuration(s: number | null | undefined): string {
  if (s === null || s === undefined || !Number.isFinite(s)) return "—";
  if (s < 60) return `${s.toFixed(1)} s`;
  if (s < 3600) return `${(s / 60).toFixed(1)} min`;
  return `${(s / 3600).toFixed(1)} h`;
}

export function normalizedMin(o: Objective, v: number): number {
  let x = v, lo = o.lo, hi = o.hi;
  if (o.log) { x = Math.log10(Math.max(v, 1e-12)); lo = Math.log10(Math.max(lo, 1e-12)); hi = Math.log10(Math.max(hi, 1e-12)); }
  let u = hi !== lo ? (x - lo) / (hi - lo) : 0;
  if (o.direction === "max") u = 1 - u;
  return Math.min(1.1, Math.max(-0.1, u));
}

export function shortId(id: string): string {
  const parts = id.split(":");
  return parts.length > 1 ? `#${parts[parts.length - 1]}` : id.slice(0, 8);
}

export function titleCase(s: string): string {
  return s.replace(/_/g, " ").replace(/\b\w/g, (m) => m.toUpperCase());
}

export function cn(...xs: (string | false | null | undefined)[]): string {
  return xs.filter(Boolean).join(" ");
}
