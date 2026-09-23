"""Analytic LLM-serving surrogate. Millisecond-fast and seeded — used by tests and for developing strategies.
Never presented as a measured result (bundles mark it `synthetic`)."""

from __future__ import annotations

from typing import Any

import numpy as np

from epoch.benchmark.resources import hw_price_per_hour
from epoch.benchmark.timing import latency_summary, log_histogram
from epoch.genome import Gene, Genome, SearchSpace, genome_id
from epoch.objectives import Constraint, Objective, TrialResult
from epoch.workloads.base import Pipeline, Prediction, Workload, register

SIZE = {"0.5b": 0.5, "1.5b": 1.5, "3b": 3.0, "7b": 7.0}
QBITS = {"fp16": 16, "int8": 8, "nf4": 4}
VRAM_MB = 16000


class _Noop(Pipeline):
    def predict_batch(self, xs: list[Any]) -> list[Prediction]:
        return [Prediction(None) for _ in xs]


@register
class SyntheticWorkload(Workload):
    name = "synthetic"
    title = "Synthetic RAG serving (analytic)"
    description = "Closed-form quality/latency/memory model of a RAG serving stack. For tests only."
    quality_metric = "quality"

    def __init__(self, noise: float = 0.03, n: int = 64):
        self.noise, self.n = noise, n
        self.objectives = [
            Objective("quality", "max", "Answer F1", lo=0.3, hi=0.9),
            Objective("p95_ms", "min", "p95 latency", "ms", lo=20, hi=20000, log=True),
            Objective("peak_mem_mb", "min", "Peak VRAM", "MB", lo=300, hi=20000, log=True),
            Objective("cost_per_1k", "min", "Cost / 1k req", "$", lo=1e-3, hi=10, log=True),
        ]
        self.constraints = [Constraint("peak_mem_mb", "<=", 12000, "fits 12 GB"), Constraint("quality", ">=", 0.55)]

    def space(self) -> SearchSpace:
        return SearchSpace([
            Gene("model", "cat", tuple(SIZE), default="1.5b", group="model", doc="Generator size."),
            Gene("quant", "cat", tuple(QBITS), default="fp16", group="runtime", doc="Weight quantisation."),
            Gene("context", "int", low=512, high=8192, log=True, default=2048, group="model", doc="Context budget."),
            Gene("top_k", "int", low=1, high=10, default=4, group="retrieval", doc="Retrieved chunks."),
            Gene("reranker", "cat", (False, True), default=False, group="retrieval", doc="Cross-encoder rerank."),
            Gene("batch_size", "cat", (1, 2, 4, 8, 16), default=1, group="runtime", doc="Batch size."),
        ])

    def prepare(self, seed: int = 0) -> None:
        return None

    def build(self, genome: Genome) -> Pipeline:
        return _Noop()

    def dataset(self, split: str, fidelity: float = 1.0, seed: int = 0):
        n = max(8, int(self.n * fidelity))
        return list(range(n)), [None] * n

    def score(self, preds: list[Prediction], golds: list[Any]) -> dict[str, float]:
        return {"quality": float("nan")}

    @staticmethod
    def true_metrics(g: Genome) -> dict[str, float]:
        size, bits = SIZE[g["model"]], QBITS[g["quant"]]
        q = 0.42 + 0.11 * np.log1p(size) - {16: 0.0, 8: 0.006, 4: 0.03}[bits]
        q += 0.05 * (1 - np.exp(-g["top_k"] / 3)) + (0.045 if g["reranker"] else 0.0)
        q += 0.03 * np.log2(g["context"] / 512) / 4 - (0.02 if g["top_k"] > 7 and g["context"] < 1024 else 0.0)
        mem = size * 1000 * bits / 8 * 1.1 + g["context"] * g["batch_size"] * 0.12 * size + (450 if g["reranker"] else 0)
        per_tok = 0.9 * size * (bits / 16) ** 0.6 + 0.15
        batch_ms = (per_tok * 64 + g["context"] * 0.004 * size) * (1 + 0.08 * (g["batch_size"] - 1))
        batch_ms += (18.0 if g["reranker"] else 0.0) + 2.5 * g["top_k"]
        return {"quality": float(q), "peak_mem_mb": float(mem), "batch_ms": float(batch_ms)}

    def evaluate(self, genome: Genome, *, split: str = "val", fidelity: float = 1.0, seed: int = 0) -> TrialResult:
        t = self.true_metrics(genome)
        rng = np.random.default_rng(int(genome_id(genome), 16) % 2**31 + seed)
        if t["peak_mem_mb"] > VRAM_MB:
            return TrialResult(metrics={}, status="failed", fidelity=fidelity, error_kind="oom",
                               error=f"CUDA out of memory: tried to allocate {t['peak_mem_mb'] - VRAM_MB:.0f} MiB "
                                     f"(model={genome['model']} quant={genome['quant']} ctx={genome['context']} "
                                     f"batch={genome['batch_size']})")
        n = max(8, int(self.n * fidelity))
        bs = genome["batch_size"]
        batches = int(np.ceil(n / bs))
        bms = t["batch_ms"] * rng.lognormal(0, self.noise, size=batches)
        lat = np.repeat(bms, bs)[:n]
        wall_s = bms.sum() / 1e3
        q = t["quality"] + rng.normal(0, 0.004 / np.sqrt(fidelity))
        compute_s = wall_s / n
        metrics = {
            "quality": float(q), **latency_summary(lat), "peak_mem_mb": t["peak_mem_mb"],
            "throughput_rps": n / wall_s, "compute_s_per_req": compute_s, "llm_usd_per_req": 0.0,
            "cost_per_1k": 1000 * compute_s * hw_price_per_hour() / 3600, "build_s": 0.0,
        }
        return TrialResult(metrics=metrics, fidelity=fidelity, n_requests=n, latencies_ms=lat.tolist(),
                           stage_ms={"generate": float(lat.mean())},
                           extras={"latency_hist": log_histogram(lat), "synthetic": True})
