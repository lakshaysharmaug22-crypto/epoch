"""Workload protocol: what EPOCH needs to evolve any AI pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from epoch.genome import Genome, SearchSpace
from epoch.objectives import Constraint, Objective


@dataclass
class Prediction:
    output: Any
    stages_ms: dict[str, float] = field(default_factory=dict)
    cost_usd: float = 0.0  # external spend for this request (LLM tokens)
    meta: dict[str, Any] = field(default_factory=dict)


class Pipeline(ABC):
    """A compiled genome. Must be safe to call repeatedly; batch semantics are the pipeline's own."""

    @abstractmethod
    def predict_batch(self, xs: list[Any]) -> list[Prediction]: ...

    def close(self) -> None:  # release GPU memory, sessions, threads
        return None


class Workload(ABC):
    name: str = "workload"
    title: str = ""
    description: str = ""
    quality_metric: str = "quality"
    objectives: list[Objective]
    constraints: list[Constraint]

    @abstractmethod
    def space(self) -> SearchSpace: ...

    @abstractmethod
    def prepare(self, seed: int = 0) -> None:
        """Load or generate data. Idempotent."""

    @abstractmethod
    def build(self, genome: Genome) -> Pipeline: ...

    @abstractmethod
    def dataset(self, split: str, fidelity: float = 1.0, seed: int = 0) -> tuple[list[Any], list[Any]]: ...

    @abstractmethod
    def score(self, preds: list[Prediction], golds: list[Any]) -> dict[str, float]:
        """Return quality metrics; must include `self.quality_metric`."""

    def batch_size(self, genome: Genome) -> int:
        return int(genome.get("batch_size", 1))

    def topology(self, genome: Genome, metrics: dict[str, float] | None = None) -> dict[str, Any]:
        return {"nodes": [], "edges": []}

    def capabilities(self) -> dict[str, Any]:
        return {}

    def adapt(self, genome: Genome, audited: list[tuple[Any, Any]], tag: str) -> Genome | None:
        """Optional remediation hook: return a genome that uses newly audited production data (e.g. retrain a
        component on an error bank). Workloads that can't adapt return None."""
        return None

    def default_genome(self) -> Genome:
        return self.space().default()

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "quality_metric": self.quality_metric,
            "objectives": [o.to_json() for o in self.objectives],
            "constraints": [c.to_json() for c in self.constraints],
            "space": self.space().to_json(),
            "capabilities": self.capabilities(),
        }


_REGISTRY: dict[str, type[Workload]] = {}


def register(cls: type[Workload]) -> type[Workload]:
    _REGISTRY[cls.name] = cls
    return cls


def get_workload(name: str, **kwargs: Any) -> Workload:
    # import for registration side effects
    from epoch.workloads import rag, synthetic, triage  # noqa: F401

    if name not in _REGISTRY:
        raise KeyError(f"unknown workload {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def available_workloads() -> list[str]:
    from epoch.workloads import rag, synthetic, triage  # noqa: F401

    return sorted(_REGISTRY)
