"""Objectives, constraints and the trial result record shared by every workload."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

Direction = Literal["min", "max"]


@dataclass(frozen=True)
class Objective:
    name: str
    direction: Direction
    label: str
    unit: str = ""
    # fixed normalisation bounds so hypervolume is comparable across strategies, seeds and runs
    lo: float = 0.0
    hi: float = 1.0
    log: bool = False

    def normalized_min(self, value: float) -> float:
        """Map to [0, 1] where 0 is best (minimisation space). Values outside bounds are clipped to [-0.1, 1.1]."""
        v = float(value)
        lo, hi = self.lo, self.hi
        if self.log:
            v, lo, hi = np.log10(max(v, 1e-12)), np.log10(max(lo, 1e-12)), np.log10(max(hi, 1e-12))
        u = (v - lo) / (hi - lo) if hi != lo else 0.0
        if self.direction == "max":
            u = 1.0 - u
        return float(np.clip(u, -0.1, 1.1))

    def better(self, a: float, b: float) -> bool:
        return a < b if self.direction == "min" else a > b

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Constraint:
    metric: str
    op: Literal["<=", ">="]
    threshold: float
    label: str = ""

    def violation(self, metrics: dict[str, float]) -> float:
        """<= 0 means satisfied; positive values are relative violation magnitude."""
        v = metrics.get(self.metric)
        if v is None or not np.isfinite(v):
            return 1.0
        scale = abs(self.threshold) or 1.0
        return (v - self.threshold) / scale if self.op == "<=" else (self.threshold - v) / scale

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrialResult:
    metrics: dict[str, float]
    status: Literal["complete", "failed", "pruned"] = "complete"
    fidelity: float = 1.0
    n_requests: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    stage_ms: dict[str, float] = field(default_factory=dict)  # mean ms per stage
    extras: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    error_kind: str | None = None

    def objective_vector(self, objectives: list[Objective]) -> list[float]:
        return [float(self.metrics[o.name]) for o in objectives]
