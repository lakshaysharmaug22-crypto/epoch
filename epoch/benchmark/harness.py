"""Benchmark engine: compiles a genome, drives it over an eval split, measures quality, latency, memory, cost."""

from __future__ import annotations

import contextlib
import time
import traceback
from typing import Any

import numpy as np

from epoch.benchmark.resources import hw_price_per_hour, peak_memory
from epoch.benchmark.timing import latency_summary, log_histogram
from epoch.genome import Genome
from epoch.objectives import TrialResult
from epoch.workloads.base import Prediction, Workload


def classify_error(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "out of memory" in text or "outofmemory" in text or "cuda oom" in text:
        return "oom"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if isinstance(exc, ImportError | ModuleNotFoundError) or "not available" in text or "unsupported" in text:
        return "capability"
    if "nan" in text or "inf" in text:
        return "numeric"
    return "runtime"


def _drive(pipe, xs: list[Any], bs: int) -> tuple[list[Prediction], list[float], float]:
    preds: list[Prediction] = []
    lat: list[float] = []
    t_start = time.perf_counter()
    for i in range(0, len(xs), bs):
        batch = xs[i : i + bs]
        t0 = time.perf_counter()
        out = pipe.predict_batch(batch)
        dt = (time.perf_counter() - t0) * 1e3
        preds.extend(out)
        lat.extend([dt] * len(out))  # every request in a batch waits for the whole batch
    return preds, lat, time.perf_counter() - t_start


def run_benchmark(
    workload: Workload,
    genome: Genome,
    *,
    split: str = "val",
    fidelity: float = 1.0,
    seed: int = 0,
    warmup: int = 4,
    robustness: bool = True,
) -> TrialResult:
    custom = getattr(workload, "evaluate", None)
    if custom is not None:  # analytic / remote workloads measure themselves
        return custom(genome, split=split, fidelity=fidelity, seed=seed)
    pipe = None
    try:
        t0 = time.perf_counter()
        pipe = workload.build(genome)
        build_s = time.perf_counter() - t0
        xs, ys = workload.dataset(split, fidelity, seed)
        bs = max(1, workload.batch_size(genome))
        with peak_memory() as mem:
            if warmup:
                pipe.predict_batch(xs[: min(warmup, len(xs))])
            preds, lat, wall_s = _drive(pipe, xs, bs)
        quality = workload.score(preds, ys)
        n = len(preds)
        stage_tot: dict[str, float] = {}
        for p in preds:
            for k, v in p.stages_ms.items():
                stage_tot[k] = stage_tot.get(k, 0.0) + v
        stage_mean = {k: v / max(n, 1) for k, v in stage_tot.items()}

        compute_s = wall_s / max(n, 1)
        llm_usd = float(np.mean([p.cost_usd for p in preds])) if preds else 0.0
        metrics: dict[str, float] = {
            **quality,
            **latency_summary(lat),
            "throughput_rps": n / wall_s if wall_s > 0 else float("nan"),
            "peak_mem_mb": round(mem.peak_mb, 3),
            "build_s": round(build_s, 4),
            "compute_s_per_req": compute_s,
            "llm_usd_per_req": llm_usd,
            "cost_per_1k": 1000 * (compute_s * hw_price_per_hour() / 3600 + llm_usd),
        }
        extras: dict[str, Any] = {"latency_hist": log_histogram(lat), "memory_probe": mem.kind, "batch_size": bs}

        if robustness and fidelity >= 0.999:
            try:
                xr, yr = workload.dataset("robust", 1.0, seed)
                if xr:
                    pr, _, _ = _drive(pipe, xr, bs)
                    qr = workload.score(pr, yr)[workload.quality_metric]
                    q = quality[workload.quality_metric]
                    metrics["quality_robust"] = qr
                    metrics["robustness"] = qr / q if q > 0 else 0.0
            except NotImplementedError:
                pass
        return TrialResult(
            metrics=metrics,
            fidelity=fidelity,
            n_requests=n,
            latencies_ms=lat,
            stage_ms=stage_mean,
            extras=extras,
        )
    except Exception as exc:  # a failed trial is data, not a crash
        return TrialResult(
            metrics={},
            status="failed",
            fidelity=fidelity,
            error="".join(traceback.format_exception(exc))[-4000:],
            error_kind=classify_error(exc),
        )
    finally:
        if pipe is not None:
            with contextlib.suppress(Exception):
                pipe.close()
