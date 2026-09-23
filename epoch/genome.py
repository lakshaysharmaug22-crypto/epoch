"""System genomes: a typed, conditional search space that compiles into a runnable pipeline.

A *gene* is one engineering decision (quantization, runtime, chunk size, a confidence gate...).
A *genome* is a full assignment of active genes. Genes can be conditional (`when`), e.g. the
reranker model only exists when `reranker=True`. The same space drives Optuna samplers, the
agent's mutations, the surrogate's feature encoding and the UI's genome diff.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import optuna
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)

GeneKind = Literal["cat", "int", "float"]
Genome = dict[str, Any]


@dataclass(frozen=True)
class Gene:
    name: str
    kind: GeneKind
    choices: tuple[Any, ...] = ()
    low: float | None = None
    high: float | None = None
    log: bool = False
    step: float | None = None
    default: Any = None
    group: str = "general"
    doc: str = ""
    # active only when genome[when[0]] is in when[1]
    when: tuple[str, tuple[Any, ...]] | None = None

    def __post_init__(self) -> None:
        if self.kind == "cat" and not self.choices:
            raise ValueError(f"categorical gene {self.name} needs choices")
        if self.kind in ("int", "float") and (self.low is None or self.high is None):
            raise ValueError(f"numeric gene {self.name} needs low/high")

    # ---- sampling -------------------------------------------------------------------------
    def distribution(self) -> BaseDistribution:
        if self.kind == "cat":
            return CategoricalDistribution(list(self.choices))
        if self.kind == "int":
            return IntDistribution(int(self.low), int(self.high), log=self.log, step=int(self.step or 1))  # type: ignore[arg-type]
        return FloatDistribution(float(self.low), float(self.high), log=self.log, step=self.step)  # type: ignore[arg-type]

    def suggest(self, trial: optuna.Trial) -> Any:
        if self.kind == "cat":
            return trial.suggest_categorical(self.name, list(self.choices))
        if self.kind == "int":
            return trial.suggest_int(self.name, int(self.low), int(self.high), log=self.log, step=int(self.step or 1))  # type: ignore[arg-type]
        return trial.suggest_float(self.name, float(self.low), float(self.high), log=self.log, step=self.step)  # type: ignore[arg-type]

    def sample(self, rng: np.random.Generator) -> Any:
        if self.kind == "cat":
            return self.choices[int(rng.integers(len(self.choices)))]
        u = float(rng.random())
        return self.from_unit(u)

    def to_unit(self, value: Any) -> float:
        lo, hi = float(self.low), float(self.high)  # type: ignore[arg-type]
        if hi == lo:
            return 0.0
        if self.log:
            return (math.log(value) - math.log(lo)) / (math.log(hi) - math.log(lo))
        return (float(value) - lo) / (hi - lo)

    def from_unit(self, u: float) -> Any:
        u = min(max(u, 0.0), 1.0)
        lo, hi = float(self.low), float(self.high)  # type: ignore[arg-type]
        v = math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo))) if self.log else lo + u * (hi - lo)
        return self.clip(v)

    def clip(self, value: Any) -> Any:
        if self.kind == "cat":
            return value if value in self.choices else (self.default if self.default is not None else self.choices[0])
        lo, hi = float(self.low), float(self.high)  # type: ignore[arg-type]
        v = min(max(float(value), lo), hi)
        if self.step:
            v = lo + round((v - lo) / self.step) * self.step
            v = min(max(v, lo), hi)
        if self.kind == "int":
            return round(v)
        return round(v, 6)

    def mutate(self, value: Any, rng: np.random.Generator, scale: float = 0.2) -> Any:
        if self.kind == "cat":
            others = [c for c in self.choices if c != value]
            return others[int(rng.integers(len(others)))] if others else value
        u = self.to_unit(value) + float(rng.normal(0, scale))
        new = self.from_unit(u)
        if new == value and self.kind == "int":  # force a move for small integer ranges
            new = self.clip(value + (1 if rng.random() < 0.5 else -1))
        return new

    def encode(self, value: Any) -> list[float]:
        if self.kind == "cat":
            vec = [0.0] * len(self.choices)
            if value in self.choices:
                vec[self.choices.index(value)] = 1.0
            return vec
        return [self.to_unit(value) if value is not None else -1.0]

    def width(self) -> int:
        return len(self.choices) if self.kind == "cat" else 1

    def contains(self, value: Any) -> bool:
        if self.kind == "cat":
            return value in self.choices
        try:
            return float(self.low) - 1e-9 <= float(value) <= float(self.high) + 1e-9  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["choices"] = list(self.choices)
        d["when"] = [self.when[0], list(self.when[1])] if self.when else None
        return d


class SearchSpace:
    """Ordered, conditional space. Conditions may only reference earlier genes."""

    def __init__(self, genes: Sequence[Gene]):
        self.genes = list(genes)
        self.by_name = {g.name: g for g in self.genes}
        seen: set[str] = set()
        for g in self.genes:
            if g.when and g.when[0] not in seen:
                raise ValueError(f"gene {g.name} depends on {g.when[0]} which must come first")
            seen.add(g.name)

    def __len__(self) -> int:
        return len(self.genes)

    @staticmethod
    def _active(g: Gene, values: Genome) -> bool:
        return g.when is None or values.get(g.when[0]) in g.when[1]

    def active(self, values: Genome) -> list[Gene]:
        return [g for g in self.genes if self._active(g, values)]

    def suggest(self, trial: optuna.Trial) -> Genome:
        out: Genome = {}
        for g in self.genes:
            if self._active(g, out):
                out[g.name] = g.suggest(trial)
        return out

    def sample(self, rng: np.random.Generator) -> Genome:
        out: Genome = {}
        for g in self.genes:
            if self._active(g, out):
                out[g.name] = g.sample(rng)
        return out

    def default(self) -> Genome:
        out: Genome = {}
        for g in self.genes:
            if self._active(g, out):
                if g.default is not None:
                    out[g.name] = g.clip(g.default)
                elif g.kind == "cat":
                    out[g.name] = g.choices[0]
                else:
                    out[g.name] = g.from_unit(0.5)
        return out

    def repair(self, values: Genome, fallback: Genome | None = None) -> Genome:
        """Drop inactive/unknown genes, clip values, fill missing active genes from fallback/default."""
        base = fallback or self.default()
        out: Genome = {}
        for g in self.genes:
            if not self._active(g, out):
                continue
            if (g.name in values and values[g.name] is not None and g.contains(values[g.name])) or (g.name in values and values[g.name] is not None and g.kind != "cat"):
                out[g.name] = g.clip(values[g.name])
            elif g.name in base:
                out[g.name] = g.clip(base[g.name])
            else:
                out[g.name] = g.sample(np.random.default_rng(0)) if g.default is None else g.clip(g.default)
        return out

    def validate(self, values: Genome) -> list[str]:
        problems = []
        for g in self.active(values):
            if g.name not in values:
                problems.append(f"missing {g.name}")
            elif not g.contains(values[g.name]):
                problems.append(f"{g.name}={values[g.name]!r} out of domain")
        for k in values:
            if k not in self.by_name:
                problems.append(f"unknown gene {k}")
        return problems

    def mutate(self, values: Genome, rng: np.random.Generator, n: int = 1, scale: float = 0.2) -> Genome:
        child = dict(values)
        act = self.active(child)
        for g in rng.choice(np.array(act, dtype=object), size=min(n, len(act)), replace=False):
            child[g.name] = g.mutate(child.get(g.name, g.default), rng, scale)
        return self.repair(child, values)

    def crossover(self, a: Genome, b: Genome, rng: np.random.Generator) -> Genome:
        child = {g.name: (a if rng.random() < 0.5 else b).get(g.name) for g in self.genes}
        return self.repair({k: v for k, v in child.items() if v is not None}, a)

    def encode(self, values: Genome) -> np.ndarray:
        vec: list[float] = []
        for g in self.genes:
            v = values.get(g.name)
            if v is None:
                vec.extend([0.0] * g.width() if g.kind == "cat" else [-1.0])
            else:
                vec.extend(g.encode(v))
        return np.asarray(vec, dtype=np.float64)

    def feature_names(self) -> list[str]:
        names: list[str] = []
        for g in self.genes:
            if g.kind == "cat":
                names.extend(f"{g.name}={c}" for c in g.choices)
            else:
                names.append(g.name)
        return names

    def feature_owner(self) -> list[str]:
        owners: list[str] = []
        for g in self.genes:
            owners.extend([g.name] * g.width())
        return owners

    def distance(self, a: Genome, b: Genome) -> float:
        return float(np.abs(self.encode(a) - self.encode(b)).sum())

    def diff(self, parent: Genome, child: Genome) -> dict[str, tuple[Any, Any]]:
        keys = [g.name for g in self.genes if g.name in parent or g.name in child]
        return {k: (parent.get(k), child.get(k)) for k in keys if parent.get(k) != child.get(k)}

    def distributions(self) -> dict[str, BaseDistribution]:
        return {g.name: g.distribution() for g in self.genes}

    def to_json(self) -> list[dict[str, Any]]:
        return [g.to_json() for g in self.genes]


def genome_id(values: Genome) -> str:
    canon = json.dumps(values, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha1(canon.encode()).hexdigest()[:10]


def describe(values: Genome, keys: Iterable[str] | None = None) -> str:
    items = values.items() if keys is None else ((k, values[k]) for k in keys if k in values)
    return ", ".join(f"{k}={v}" for k, v in items)
