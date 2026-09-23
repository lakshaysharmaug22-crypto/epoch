"""Latency statistics and a tiny per-stage timer."""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager

import numpy as np


def latency_summary(lat_ms: list[float] | np.ndarray) -> dict[str, float]:
    a = np.asarray(lat_ms, dtype=np.float64)
    if a.size == 0:
        return {"p50_ms": float("nan"), "p95_ms": float("nan"), "p99_ms": float("nan"), "mean_ms": float("nan")}
    return {
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "mean_ms": float(a.mean()),
    }


def log_histogram(lat_ms: list[float] | np.ndarray, bins: int = 24) -> dict[str, list[float]]:
    a = np.asarray(lat_ms, dtype=np.float64)
    a = a[a > 0]
    if a.size == 0:
        return {"edges": [], "counts": []}
    lo, hi = np.log10(a.min()), np.log10(a.max())
    if hi - lo < 1e-6:
        hi = lo + 0.1
    edges = np.logspace(lo, hi, bins + 1)
    counts, _ = np.histogram(a, bins=edges)
    return {"edges": [round(float(e), 5) for e in edges], "counts": [int(c) for c in counts]}


class StageTimer:
    """Accumulates wall time per named stage: `with timer.stage("retrieve"): ...`"""

    def __init__(self) -> None:
        self.ms: dict[str, float] = defaultdict(float)

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter_ns()
        try:
            yield
        finally:
            self.ms[name] += (time.perf_counter_ns() - t0) / 1e6

    def as_dict(self) -> dict[str, float]:
        return dict(self.ms)
