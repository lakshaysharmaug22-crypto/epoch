"""Measure a genome's service-time profile for the twin: per-request samples at batch 1 + batch-scaling α."""

from __future__ import annotations

import time

import numpy as np

from epoch.genome import Genome, genome_id
from epoch.twin.sim import ServiceProfile
from epoch.workloads.base import Workload


def measure_profile(workload: Workload, genome: Genome, *, n: int = 300, probe_batch: int = 8, split: str = "val",
                    seed: int = 0, inputs: list | None = None, inject_ms: dict[str, float] | None = None) -> ServiceProfile:
    pipe = workload.build(genome)
    if inject_ms and hasattr(pipe, "inject_ms"):
        pipe.inject_ms = dict(inject_ms)
    try:
        xs = inputs if inputs is not None else workload.dataset(split, 1.0, seed)[0]
        xs = (xs * (n // max(len(xs), 1) + 1))[:n]
        pipe.predict_batch(xs[:8])
        samples = []
        for x in xs:
            t0 = time.perf_counter()
            pipe.predict_batch([x])
            samples.append((time.perf_counter() - t0) * 1e3)
        tb = []
        for i in range(0, min(len(xs), probe_batch * 25), probe_batch):
            t0 = time.perf_counter()
            pipe.predict_batch(xs[i : i + probe_batch])
            tb.append((time.perf_counter() - t0) * 1e3)
        t1 = float(np.mean(samples))  # means, not medians: a batch almost always contains a slow-path request
        alpha = float(np.clip((np.mean(tb) / t1 - 1) / (probe_batch - 1), 0.0, 1.5)) if t1 > 0 else 0.1
        return ServiceProfile(samples_ms=samples, alpha=alpha, label=genome_id(genome))
    finally:
        pipe.close()
