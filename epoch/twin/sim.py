"""Digital twin: a discrete-event simulation (SimPy) of a serving deployment of one genome.

Service times are *measured*, not assumed: `ServiceProfile` holds the empirical per-request service-time
distribution of the genome at batch 1 plus a fitted batch-scaling factor α, where
    t(batch=b) = t(1) · (1 + α·(b − 1)).
The twin models open-loop arrivals (Poisson, bursty, diurnal, step), a bounded FIFO queue, N replicas with
dynamic batching (max_batch, max_wait), per-request timeouts, and injected faults (replica loss, slowdown,
dependency latency spikes). The same algorithm is ported to TypeScript for the dashboard's in-browser twin.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import simpy


@dataclass
class ServiceProfile:
    samples_ms: list[float]
    alpha: float = 0.1
    overhead_ms: float = 0.0  # fixed per-request serving overhead (HTTP parse/serialise), added to service time
    network_ms: float = 0.0  # fixed client-side delay, added to end-to-end latency only
    label: str = ""

    def quantiles(self, n: int = 64) -> list[float]:
        return [round(float(x), 4) for x in np.quantile(np.asarray(self.samples_ms), np.linspace(0, 1, n))]

    @property
    def mean_ms(self) -> float:
        return float(np.mean(self.samples_ms)) + self.overhead_ms

    def capacity_rps(self, replicas: int = 1, batch: int = 1) -> float:
        t = self.mean_ms * (1 + self.alpha * (batch - 1)) / 1e3
        return replicas * batch / t if t > 0 else math.inf

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["samples_ms"] = self.quantiles()
        return d


@dataclass
class TwinConfig:
    replicas: int = 1
    max_batch: int = 1
    max_wait_ms: float = 0.0
    queue_cap: int = 5000
    timeout_ms: float | None = None
    slo_ms: float | None = None


@dataclass
class Traffic:
    kind: Literal["poisson", "bursty", "diurnal", "step"] = "poisson"
    rate: float = 20.0
    duration_s: float = 60.0
    burst_factor: float = 4.0
    burst_every_s: float = 20.0
    burst_len_s: float = 4.0
    step_at_s: float = 30.0
    step_factor: float = 2.0
    period_s: float = 60.0

    def rate_at(self, t: float) -> float:
        if self.kind == "bursty":
            return self.rate * (self.burst_factor if (t % self.burst_every_s) < self.burst_len_s else 1.0)
        if self.kind == "diurnal":
            return self.rate * (1 + 0.6 * math.sin(2 * math.pi * t / self.period_s))
        if self.kind == "step":
            return self.rate * (self.step_factor if t >= self.step_at_s else 1.0)
        return self.rate

    def peak(self) -> float:
        return self.rate * max(1.0, self.burst_factor if self.kind == "bursty" else 1.6 if self.kind == "diurnal"
                               else self.step_factor if self.kind == "step" else 1.0)


@dataclass
class Fault:
    kind: Literal["replica_down", "slowdown", "latency_spike"]
    start_s: float
    duration_s: float
    magnitude: float = 2.0  # slowdown factor, or added ms for latency_spike
    replica: int = 0

    def active(self, t: float) -> bool:
        return self.start_s <= t < self.start_s + self.duration_s


@dataclass
class _Req:
    t_arr: float


@dataclass
class _Acc:
    lat: list[list[float]] = field(default_factory=list)
    arrivals: np.ndarray = field(default_factory=lambda: np.zeros(0))
    done: np.ndarray = field(default_factory=lambda: np.zeros(0))
    drops: np.ndarray = field(default_factory=lambda: np.zeros(0))
    timeouts: np.ndarray = field(default_factory=lambda: np.zeros(0))
    busy: np.ndarray = field(default_factory=lambda: np.zeros(0))
    queue: np.ndarray = field(default_factory=lambda: np.zeros(0))
    batch_sizes: list[int] = field(default_factory=list)


def simulate(profile: ServiceProfile, cfg: TwinConfig, traffic: Traffic, faults: list[Fault] | tuple = (),
             seed: int = 0, bucket_s: float = 1.0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    env = simpy.Environment()
    q = simpy.Store(env)
    nb = math.ceil(traffic.duration_s / bucket_s)
    acc = _Acc(lat=[[] for _ in range(nb)], arrivals=np.zeros(nb), done=np.zeros(nb), drops=np.zeros(nb),
               timeouts=np.zeros(nb), busy=np.zeros(nb), queue=np.zeros(nb))
    samples = np.asarray(profile.samples_ms, dtype=np.float64)
    all_lat: list[float] = []

    def bucket(t: float) -> int:
        return min(nb - 1, max(0, int(t / bucket_s)))

    def arrivals():
        lam_max = traffic.peak() * 1.05
        t = 0.0
        while True:
            t += rng.exponential(1.0 / lam_max)
            if t >= traffic.duration_s:
                return
            yield env.timeout(t - env.now)
            if rng.random() <= traffic.rate_at(t) / lam_max:
                acc.arrivals[bucket(t)] += 1
                if len(q.items) >= cfg.queue_cap:
                    acc.drops[bucket(t)] += 1
                else:
                    q.put(_Req(t))

    def slowdown(t: float, rid: int) -> tuple[float, float]:
        f, add = 1.0, 0.0
        for flt in faults:
            if flt.active(t):
                if flt.kind == "slowdown":
                    f *= flt.magnitude
                elif flt.kind == "latency_spike":
                    add += flt.magnitude
        return f, add

    def down(t: float, rid: int) -> float | None:
        for flt in faults:
            if flt.kind == "replica_down" and flt.replica == rid and flt.active(t):
                return flt.start_s + flt.duration_s
        return None

    def replica(rid: int):
        while True:
            until = down(env.now, rid)
            if until is not None:
                yield env.timeout(until - env.now)
                continue
            first = yield q.get()
            batch = [first]
            deadline = env.now + cfg.max_wait_ms / 1e3
            while len(batch) < cfg.max_batch:
                if q.items:
                    batch.append((yield q.get()))
                    continue
                remaining = deadline - env.now
                if remaining <= 0:
                    break
                get = q.get()
                res = yield get | env.timeout(remaining)
                if get in res:
                    batch.append(res[get])
                else:
                    get.cancel()
                    break
            f, add = slowdown(env.now, rid)
            svc = (float(rng.choice(samples)) * (1 + profile.alpha * (len(batch) - 1)) + profile.overhead_ms) * f + add
            start = env.now
            yield env.timeout(svc / 1e3)
            _account_busy(start, env.now)
            acc.batch_sizes.append(len(batch))
            for r in batch:
                lat = (env.now - r.t_arr) * 1e3 + profile.network_ms
                b = bucket(env.now)
                if cfg.timeout_ms is not None and lat > cfg.timeout_ms:
                    acc.timeouts[b] += 1
                else:
                    if env.now <= traffic.duration_s:
                        acc.done[b] += 1
                    acc.lat[b].append(lat)
                    all_lat.append(lat)

    def _account_busy(t0: float, t1: float) -> None:
        t1 = min(t1, traffic.duration_s)  # utilisation is measured inside the traffic window only
        while t0 < t1:
            b = bucket(t0)
            edge = min(t1, (b + 1) * bucket_s)
            acc.busy[b] += edge - t0
            t0 = edge

    def monitor():
        k = 0
        while k < nb:
            acc.queue[k] = len(q.items)
            k += 1
            yield env.timeout(bucket_s)

    env.process(arrivals())
    env.process(monitor())
    for rid in range(cfg.replicas):
        env.process(replica(rid))
    env.run(until=traffic.duration_s + 30.0)  # drain

    series = []
    for b in range(nb):
        lat = np.asarray(acc.lat[b])
        series.append({
            "t": round(b * bucket_s, 3),
            "rps_in": acc.arrivals[b] / bucket_s,
            "rps_out": acc.done[b] / bucket_s,
            "p50": float(np.percentile(lat, 50)) if lat.size else None,
            "p95": float(np.percentile(lat, 95)) if lat.size else None,
            "p99": float(np.percentile(lat, 99)) if lat.size else None,
            "queue": float(acc.queue[b]),
            "util": float(min(1.0, acc.busy[b] / (bucket_s * cfg.replicas))),
            "drops": float(acc.drops[b]),
            "timeouts": float(acc.timeouts[b]),
        })
    lat = np.asarray(all_lat)
    n_in = float(acc.arrivals.sum())
    summary = {
        "p50": float(np.percentile(lat, 50)) if lat.size else None,
        "p95": float(np.percentile(lat, 95)) if lat.size else None,
        "p99": float(np.percentile(lat, 99)) if lat.size else None,
        "throughput_rps": float(acc.done.sum() / traffic.duration_s),
        "offered_rps": n_in / traffic.duration_s,
        "drop_rate": float(acc.drops.sum() / n_in) if n_in else 0.0,
        "timeout_rate": float(acc.timeouts.sum() / n_in) if n_in else 0.0,
        "utilization": float(acc.busy.sum() / (traffic.duration_s * cfg.replicas)),
        "mean_batch": float(np.mean(acc.batch_sizes)) if acc.batch_sizes else 0.0,
        "max_queue": float(acc.queue.max()) if nb else 0.0,
        "slo_attainment": float((lat <= cfg.slo_ms).mean()) if (cfg.slo_ms and lat.size) else None,
    }
    q1, q4 = acc.queue[: max(1, nb // 4)].mean(), acc.queue[-max(1, nb // 4):].mean()
    summary["stable"] = bool(q4 <= max(5.0, 2 * q1 + 5))
    return {"summary": summary, "series": series, "config": asdict(cfg), "traffic": asdict(traffic),
            "faults": [asdict(f) for f in faults]}


def plan_capacity(profile: ServiceProfile, traffic: Traffic, slo_p95_ms: float, *, price_per_replica_hour: float,
                  max_replicas: int = 8, batches: tuple[int, ...] = (1, 4, 16), waits: tuple[float, ...] = (0.0, 5.0, 20.0),
                  seed: int = 0) -> dict[str, Any]:
    """Cheapest (replicas, batch, wait) that meets the p95 SLO without drops for this traffic shape."""
    tried = []
    for r in range(1, max_replicas + 1):
        for b in batches:
            for w in waits if b > 1 else (0.0,):
                if profile.capacity_rps(r, b) < traffic.peak() * 0.7:
                    continue
                res = simulate(profile, TwinConfig(replicas=r, max_batch=b, max_wait_ms=w, slo_ms=slo_p95_ms),
                               Traffic(**{**asdict(traffic), "duration_s": min(traffic.duration_s, 60.0)}), seed=seed)
                s = res["summary"]
                ok = s["p95"] is not None and s["p95"] <= slo_p95_ms and s["drop_rate"] < 0.001 and s["stable"]
                tried.append({"replicas": r, "max_batch": b, "max_wait_ms": w, "p95": s["p95"], "ok": ok,
                              "utilization": s["utilization"], "cost_per_hour": r * price_per_replica_hour})
        if any(t["ok"] for t in tried):
            break
    ok = [t for t in tried if t["ok"]]
    best = min(ok, key=lambda t: (t["cost_per_hour"], t["p95"])) if ok else None
    return {"best": best, "tried": tried}
