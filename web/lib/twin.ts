// In-browser port of epoch/twin/sim.py: event-driven simulation of N replicas with dynamic batching, a FIFO
// queue, open-loop traffic and injected faults. Service times are sampled from the *measured* profile.

import type { ServiceProfile, TwinResult, TwinSeriesPoint } from "./types";

export interface TwinConfig { replicas: number; max_batch: number; max_wait_ms: number; queue_cap: number; slo_ms: number | null }
export interface Traffic { kind: "poisson" | "bursty" | "diurnal" | "step"; rate: number; duration_s: number; burst_factor: number; burst_every_s: number; burst_len_s: number; step_at_s: number; step_factor: number; period_s: number }
export interface Fault { kind: "replica_down" | "slowdown" | "latency_spike"; start_s: number; duration_s: number; magnitude: number; replica: number }

export const defaultTraffic = (rate: number, duration_s = 90): Traffic => ({
  kind: "poisson", rate, duration_s, burst_factor: 3, burst_every_s: 30, burst_len_s: 6, step_at_s: duration_s / 2, step_factor: 2, period_s: duration_s,
});

function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function rateAt(tr: Traffic, t: number): number {
  if (tr.kind === "bursty") return tr.rate * (t % tr.burst_every_s < tr.burst_len_s ? tr.burst_factor : 1);
  if (tr.kind === "diurnal") return tr.rate * (1 + 0.6 * Math.sin((2 * Math.PI * t) / tr.period_s));
  if (tr.kind === "step") return tr.rate * (t >= tr.step_at_s ? tr.step_factor : 1);
  return tr.rate;
}
const peak = (tr: Traffic) => tr.rate * Math.max(1, tr.kind === "bursty" ? tr.burst_factor : tr.kind === "diurnal" ? 1.6 : tr.kind === "step" ? tr.step_factor : 1);

export function capacity(p: ServiceProfile, replicas = 1, batch = 1): number {
  const mean = p.samples_ms.reduce((a, b) => a + b, 0) / p.samples_ms.length + (p.overhead_ms || 0);
  return (replicas * batch) / ((mean * (1 + p.alpha * (batch - 1))) / 1000);
}

// tiny binary heap keyed on time
class Heap<T extends { t: number }> {
  private a: T[] = [];
  push(x: T) { const a = this.a; a.push(x); let i = a.length - 1; while (i > 0) { const p = (i - 1) >> 1; if (a[p].t <= a[i].t) break; [a[p], a[i]] = [a[i], a[p]]; i = p; } }
  pop(): T | undefined {
    const a = this.a; if (!a.length) return undefined; const top = a[0]; const last = a.pop()!;
    if (a.length) { a[0] = last; let i = 0; for (;;) { const l = 2 * i + 1, r = l + 1; let m = i; if (l < a.length && a[l].t < a[m].t) m = l; if (r < a.length && a[r].t < a[m].t) m = r; if (m === i) break; [a[m], a[i]] = [a[i], a[m]]; i = m; } }
    return top;
  }
  get size() { return this.a.length; }
}

type Ev = { t: number; type: "arrival" | "deadline" | "done" | "wake"; rid: number; token?: number };
type Rep = { state: "idle" | "collecting" | "busy" | "down"; batch: number[]; token: number };

function pct(sorted: number[], q: number): number | null {
  if (!sorted.length) return null;
  const pos = (sorted.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

export function simulate(p: ServiceProfile, cfg: TwinConfig, tr: Traffic, faults: Fault[] = [], seed = 0, bucket = 1): TwinResult {
  const r = rng(seed * 7919 + 17);
  const nb = Math.ceil(tr.duration_s / bucket);
  const lat: number[][] = Array.from({ length: nb }, () => []);
  const arrivals = new Float64Array(nb), done = new Float64Array(nb), drops = new Float64Array(nb), busy = new Float64Array(nb), qlen = new Float64Array(nb);
  const all: number[] = [];
  const batchSizes: number[] = [];
  const samples = p.samples_ms;
  const B = (t: number) => Math.min(nb - 1, Math.max(0, Math.floor(t / bucket)));
  const heap = new Heap<Ev>();
  const queue: number[] = [];
  const reps: Rep[] = Array.from({ length: cfg.replicas }, () => ({ state: "idle", batch: [], token: 0 }));
  const downUntil = (t: number, rid: number) => { for (const f of faults) if (f.kind === "replica_down" && f.replica === rid && f.start_s <= t && t < f.start_s + f.duration_s) return f.start_s + f.duration_s; return null; };
  const slow = (t: number) => { let f = 1, add = 0; for (const x of faults) if (x.start_s <= t && t < x.start_s + x.duration_s) { if (x.kind === "slowdown") f *= x.magnitude; else if (x.kind === "latency_spike") add += x.magnitude; } return [f, add] as const; };

  // arrivals by thinning
  const lamMax = peak(tr) * 1.05;
  let t = 0;
  for (;;) {
    t += -Math.log(1 - r()) / lamMax;
    if (t >= tr.duration_s) break;
    if (r() <= rateAt(tr, t) / lamMax) heap.push({ t, type: "arrival", rid: -1 });
  }

  const startService = (rid: number, now: number) => {
    const rep = reps[rid];
    const [f, add] = slow(now);
    const s = samples[Math.floor(r() * samples.length)];
    const svc = ((s * (1 + p.alpha * (rep.batch.length - 1)) + (p.overhead_ms || 0)) * f + add) / 1000;
    rep.state = "busy";
    // busy accounting inside the traffic window
    let t0 = now; const t1 = Math.min(now + svc, tr.duration_s);
    while (t0 < t1) { const b = B(t0); const edge = Math.min(t1, (b + 1) * bucket); busy[b] += edge - t0; t0 = edge; }
    heap.push({ t: now + svc, type: "done", rid });
  };

  const collect = (rid: number, now: number) => {
    const rep = reps[rid];
    const until = downUntil(now, rid);
    if (until !== null) { rep.state = "down"; heap.push({ t: until, type: "wake", rid }); return; }
    if (!queue.length) { rep.state = "idle"; return; }
    rep.batch = [queue.shift()!];
    while (rep.batch.length < cfg.max_batch && queue.length) rep.batch.push(queue.shift()!);
    if (rep.batch.length >= cfg.max_batch || cfg.max_wait_ms <= 0) { startService(rid, now); return; }
    rep.state = "collecting";
    rep.token += 1;
    heap.push({ t: now + cfg.max_wait_ms / 1000, type: "deadline", rid, token: rep.token });
  };

  let sampleAt = 0;
  for (;;) {
    const ev = heap.pop();
    if (!ev) break;
    while (sampleAt < nb && sampleAt * bucket <= ev.t) { qlen[sampleAt] = queue.length; sampleAt++; }
    const now = ev.t;
    if (ev.type === "arrival") {
      arrivals[B(now)] += 1;
      const coll = reps.findIndex((x) => x.state === "collecting");
      if (coll >= 0) {
        const rep = reps[coll];
        rep.batch.push(now);
        if (rep.batch.length >= cfg.max_batch) { rep.token += 1; startService(coll, now); }
        continue;
      }
      if (queue.length >= cfg.queue_cap) { drops[B(now)] += 1; continue; }
      queue.push(now);
      const idle = reps.findIndex((x) => x.state === "idle");
      if (idle >= 0) collect(idle, now);
    } else if (ev.type === "deadline") {
      const rep = reps[ev.rid];
      if (rep.state === "collecting" && rep.token === ev.token) startService(ev.rid, now);
    } else if (ev.type === "done") {
      const rep = reps[ev.rid];
      batchSizes.push(rep.batch.length);
      for (const ta of rep.batch) {
        const l = (now - ta) * 1000 + (p.network_ms || 0);
        const b = B(now);
        if (now <= tr.duration_s) done[b] += 1;
        lat[b].push(l); all.push(l);
      }
      rep.batch = [];
      collect(ev.rid, now);
    } else if (ev.type === "wake") {
      collect(ev.rid, now);
    }
    if (now > tr.duration_s + 30) break;
  }

  const series: TwinSeriesPoint[] = [];
  for (let b = 0; b < nb; b++) {
    const s = [...lat[b]].sort((x, y) => x - y);
    series.push({ t: b * bucket, rps_in: arrivals[b] / bucket, rps_out: done[b] / bucket, p50: pct(s, 0.5), p95: pct(s, 0.95), p99: pct(s, 0.99),
      queue: qlen[b], util: Math.min(1, busy[b] / (bucket * cfg.replicas)), drops: drops[b], timeouts: 0 });
  }
  const sorted = all.sort((x, y) => x - y);
  const nIn = arrivals.reduce((a, b) => a + b, 0);
  const q = Array.from(qlen);
  const k = Math.max(1, Math.floor(nb / 4));
  const q1 = q.slice(0, k).reduce((a, b) => a + b, 0) / k, q4 = q.slice(-k).reduce((a, b) => a + b, 0) / k;
  return {
    summary: {
      p50: pct(sorted, 0.5), p95: pct(sorted, 0.95), p99: pct(sorted, 0.99),
      throughput_rps: done.reduce((a, b) => a + b, 0) / tr.duration_s, offered_rps: nIn / tr.duration_s,
      drop_rate: nIn ? drops.reduce((a, b) => a + b, 0) / nIn : 0, timeout_rate: 0,
      utilization: busy.reduce((a, b) => a + b, 0) / (tr.duration_s * cfg.replicas),
      mean_batch: batchSizes.length ? batchSizes.reduce((a, b) => a + b, 0) / batchSizes.length : 0,
      max_queue: Math.max(0, ...q), slo_attainment: cfg.slo_ms && sorted.length ? sorted.filter((x) => x <= cfg.slo_ms!).length / sorted.length : null,
      stable: q4 <= Math.max(5, 2 * q1 + 5),
    },
    series, config: { ...cfg }, traffic: { ...tr }, faults: faults.map((f) => ({ ...f })),
  };
}
