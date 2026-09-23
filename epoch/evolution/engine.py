"""Evolution engine: constraint-aware multi-objective search over system genomes.

Strategies share one ask/tell loop so they are compared on equal terms:
  random  Optuna RandomSampler
  tpe     multivariate MOTPE with constraints
  nsga2   NSGA-II with constrained domination
  agent   LangGraph hypothesis agent + surrogate pre-screening (random warm-up, then hypothesis batches)

Every trial goes through multi-fidelity early rejection (cheap subset first; clear constraint violators or
clearly dominated configs are pruned), is written to experiment memory with provenance and lineage, and
updates the running hypervolume.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import optuna

from epoch.benchmark.harness import run_benchmark
from epoch.config import settings
from epoch.evolution.pareto import hypervolume, non_dominated, normalize
from epoch.genome import Genome, genome_id
from epoch.memory.provenance import provenance
from epoch.memory.store import Store
from epoch.objectives import TrialResult
from epoch.workloads.base import Workload

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger("root").setLevel(logging.ERROR)  # onnxruntime.quantization logs a pre-processing hint per model
# Optuna >= 5 stores constraints on the trial (Trial.set_constraint); older versions read them via constraints_func
_NATIVE_CONSTRAINTS = hasattr(optuna.Trial, "set_constraint")

Sink = Callable[[str, dict], None]
STRATEGIES = ("random", "tpe", "nsga2", "agent")


@dataclass
class EngineConfig:
    budget: int = 60
    seed: int = 0
    population: int = 12
    warmup: int = 10
    agent_batch: int = 4
    early_reject: bool = True
    low_fidelity: float = 0.3
    reject_margin: float = 0.08
    split: str = "val"
    robustness: bool = True


@dataclass
class Proposal:
    genome: Genome
    parents: list[str] = field(default_factory=list)
    hypothesis_id: str | None = None
    note: str = ""


@dataclass
class Record:
    number: int
    trial_id: str
    genome: Genome
    gid: str
    origin: str
    generation: int
    result: TrialResult
    violations: list[float]
    feasible: bool
    parents: list[str]
    parents_inferred: bool
    hypothesis_id: str | None
    duration_s: float
    hv_after: float = 0.0


def knee(norm: np.ndarray) -> int:
    """Index of the point closest to the utopia corner in normalised space."""
    return int(np.argmin(np.linalg.norm(norm, axis=1)))


class Evolution:
    def __init__(self, workload: Workload, strategy: str, store: Store, cfg: EngineConfig | None = None, *,
                 run_id: str | None = None, group_id: str | None = None, sink: Sink | None = None):
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        self.w, self.strategy, self.store = workload, strategy, store
        self.cfg = cfg or EngineConfig()
        self.space = workload.space()
        self.objs, self.cons = workload.objectives, workload.constraints
        self.run_id = run_id or f"{workload.name}-{strategy}-s{self.cfg.seed}-{uuid.uuid4().hex[:6]}"
        self.group_id = group_id
        self.sink = sink or (lambda kind, payload: None)
        self.records: list[Record] = []
        self.cache: dict[str, TrialResult] = {}
        self.queue: list[Proposal] = []
        self.hv_curve: list[float] = []
        self.compute_saved_s = 0.0
        self.agent = None
        self._mlflow = _MlflowMirror(self.run_id) if settings().mlflow_uri else None

    # ------------------------------------------------------------------ setup
    def _sampler(self) -> optuna.samplers.BaseSampler:
        seed = self.cfg.seed
        constraints = None if _NATIVE_CONSTRAINTS else self._constraints_func
        if self.strategy == "tpe":
            return optuna.samplers.TPESampler(seed=seed, multivariate=True, n_startup_trials=self.cfg.warmup,
                                              constraints_func=constraints)
        if self.strategy == "nsga2":
            return optuna.samplers.NSGAIISampler(seed=seed, population_size=self.cfg.population,
                                                 constraints_func=constraints)
        return optuna.samplers.RandomSampler(seed=seed)

    def _constraints_func(self, ft: optuna.trial.FrozenTrial) -> list[float]:
        return ft.user_attrs.get("violations", [1.0] * len(self.cons))

    def _set_constraints(self, trial: optuna.Trial, viol: list[float]) -> None:
        trial.set_user_attr("violations", viol)
        if _NATIVE_CONSTRAINTS:  # keyed constraints; samplers treat all(value <= 0) as feasible
            for i, (c, v) in enumerate(zip(self.cons, viol, strict=True)):
                trial.set_constraint(f"{i}:{c.metric}", float(v))  # type: ignore[attr-defined]

    # ------------------------------------------------------------------ helpers
    def violations(self, metrics: dict[str, float]) -> list[float]:
        return [c.violation(metrics) for c in self.cons]

    def archive(self) -> tuple[list[Record], np.ndarray]:
        recs = [r for r in self.records if r.result.status == "complete" and r.feasible]
        pts = np.array([[r.result.metrics[o.name] for o in self.objs] for r in recs]) if recs else np.zeros((0, len(self.objs)))
        return recs, normalize(pts, self.objs)

    def frontier(self) -> tuple[list[Record], np.ndarray]:
        recs, norm = self.archive()
        if not recs:
            return [], norm
        m = non_dominated(norm)
        return [r for r, keep in zip(recs, m, strict=True) if keep], norm[m]

    def _evaluate(self, genome: Genome) -> TrialResult:
        cfg = self.cfg
        if cfg.early_reject and cfg.low_fidelity < 1.0:
            t0 = time.perf_counter()
            lo = run_benchmark(self.w, genome, split=cfg.split, fidelity=cfg.low_fidelity, seed=cfg.seed, robustness=False)
            lo_s = time.perf_counter() - t0
            if lo.status == "failed":
                return lo
            v = self.violations(lo.metrics)
            reason = None
            if max(v) > cfg.reject_margin:
                reason = f"violates `{self.cons[int(np.argmax(v))].metric}` by {max(v):.0%} at {cfg.low_fidelity:.0%} fidelity"
            else:
                _, front = self.frontier()
                if front.size:
                    p = normalize(np.array([[lo.metrics[o.name] for o in self.objs]]), self.objs)[0]
                    if np.any(np.all(front + cfg.reject_margin <= p, axis=1)):
                        reason = f"dominated by the frontier by >{cfg.reject_margin:.0%} on every objective at low fidelity"
            if reason:
                lo.status = "pruned"
                lo.extras["prune_reason"] = reason
                self.compute_saved_s += lo_s * (1 / cfg.low_fidelity - 1)
                return lo
        return run_benchmark(self.w, genome, split=cfg.split, fidelity=1.0, seed=cfg.seed, robustness=cfg.robustness)

    def _infer_parents(self, genome: Genome, number: int) -> tuple[list[str], bool]:
        if self.strategy != "nsga2":
            return [], False
        gen = number // self.cfg.population
        if gen == 0:
            return [], False
        prev = [r for r in self.records if r.generation < gen and r.result.status == "complete"]
        if not prev:
            return [], False
        d = [self.space.distance(genome, r.genome) for r in prev]
        return [prev[i].trial_id for i in np.argsort(d)[:2]], True

    # ------------------------------------------------------------------ main loop
    def run(self) -> dict[str, Any]:
        cfg = self.cfg
        self.w.prepare(cfg.seed)
        study = optuna.create_study(directions=["minimize" if o.direction == "min" else "maximize" for o in self.objs],
                                    sampler=self._sampler(), study_name=self.run_id)
        self.study = study
        self.store.create_run(id=self.run_id, group_id=self.group_id, workload=self.w.name, strategy=self.strategy,
                              seed=cfg.seed, budget=cfg.budget, provenance=provenance(cfg.seed),
                              config={**asdict(cfg), "workload": self.w.describe()})
        self.sink("run.start", {"run_id": self.run_id, "workload": self.w.name, "strategy": self.strategy,
                                "budget": cfg.budget, "group_id": self.group_id})
        if self.strategy == "agent":
            from epoch.agents.hypothesis import HypothesisAgent

            self.agent = HypothesisAgent(self)
        t_run = time.perf_counter()
        generation = 0
        for number in range(cfg.budget):
            origin, parents, inferred, hyp = "sampler", [], False, None
            if self.strategy == "random" or (self.strategy == "agent" and number < cfg.warmup):
                origin = "random"
            if self.strategy == "agent" and number >= cfg.warmup:
                if not self.queue:
                    generation += 1
                    self.queue = self.agent.propose(k=cfg.agent_batch, generation=generation)  # type: ignore[union-attr]
                if self.queue:
                    p = self.queue.pop(0)
                    study.enqueue_trial(p.genome)
                    origin, parents, hyp = f"agent:{p.note or 'hypothesis'}", p.parents, p.hypothesis_id
                else:
                    origin = "random"
            trial = study.ask()
            genome = self.space.repair(self.space.suggest(trial))
            gid = genome_id(genome)
            if self.strategy != "agent":
                generation = number // cfg.population
                parents, inferred = self._infer_parents(genome, number)
                if self.strategy == "tpe" and number >= cfg.warmup:
                    origin = "tpe-model"
                elif self.strategy == "nsga2" and number >= cfg.population:
                    origin = "crossover"
            t0 = time.perf_counter()
            dup = gid in self.cache
            result = self.cache[gid] if dup else self._evaluate(genome)
            dur = 0.0 if dup else time.perf_counter() - t0
            if result.status != "failed":
                self.cache[gid] = result
            viol = self.violations(result.metrics) if result.metrics else [1.0] * len(self.cons)
            feasible = result.status == "complete" and max(viol, default=0) <= 0
            if result.status == "failed":
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
            elif result.status == "pruned":
                self._set_constraints(trial, viol)
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            else:
                self._set_constraints(trial, viol)
                study.tell(trial, [float(result.metrics[o.name]) for o in self.objs])
            rec = Record(number, f"{self.run_id}:{number}", genome, gid, origin + ("+cached" if dup else ""), generation,
                         result, viol, feasible, parents, inferred, hyp, dur)
            self.records.append(rec)
            _, norm = self.archive()
            rec.hv_after = hypervolume(norm)
            self.hv_curve.append(rec.hv_after)
            self._persist(rec)
            if self.agent is not None and hyp:
                self.agent.observe(rec)
        summary = self.summary(time.perf_counter() - t_run)
        front, _ = self.frontier()
        self.store.set_pareto(self.run_id, {r.trial_id for r in front})
        self.store.finish_run(self.run_id, summary)
        self.sink("run.end", {"run_id": self.run_id, **{k: summary[k] for k in ("final_hv", "n_pareto", "n_complete")}})
        if self._mlflow:
            self._mlflow.finish(summary)
        return summary

    def _persist(self, r: Record) -> None:
        res = r.result
        extras = {k: v for k, v in res.extras.items() if k in ("latency_hist", "prune_reason", "memory_probe", "batch_size")}
        extras["stage_ms"] = res.stage_ms
        metrics = {k: (None if isinstance(v, float) and not np.isfinite(v) else float(v)) for k, v in res.metrics.items()}
        self.store.add_trial(
            id=r.trial_id, run_id=self.run_id, number=r.number, genome_id=r.gid, genome=r.genome, origin=r.origin,
            generation=r.generation, status=res.status, metrics=metrics, violations=r.violations, feasible=r.feasible,
            fidelity=res.fidelity, duration_s=r.duration_s, hv_after=r.hv_after, parents=r.parents,
            parents_inferred=r.parents_inferred, hypothesis_id=r.hypothesis_id, extras=extras,
            error=res.error, error_kind=res.error_kind,
        )
        if res.status == "failed":
            self.store.add_failure(run_id=self.run_id, trial_id=r.trial_id, workload=self.w.name,
                                   kind=res.error_kind or "runtime", message=res.error or "", genome=r.genome)
        payload = {"run_id": self.run_id, "number": r.number, "trial_id": r.trial_id, "status": res.status,
                   "origin": r.origin, "feasible": r.feasible, "hv": r.hv_after, "genome_id": r.gid,
                   "metrics": {o.name: metrics.get(o.name) for o in self.objs}}
        self.store.add_event("trial", payload, run_id=self.run_id)
        self.sink("trial", payload)
        if self._mlflow:
            self._mlflow.log_trial(r, metrics)

    def summary(self, wall_s: float = 0.0) -> dict[str, Any]:
        front, fnorm = self.frontier()
        status = [r.result.status for r in self.records]
        best = {}
        for o in self.objs:
            vals = [(r.result.metrics[o.name], r.trial_id) for r in self.archive()[0]]
            if vals:
                best[o.name] = (min if o.direction == "min" else max)(vals)
        k = front[knee(fnorm)].trial_id if front else None
        return {
            "final_hv": self.hv_curve[-1] if self.hv_curve else 0.0,
            "hv_curve": self.hv_curve,
            "n_trials": len(self.records),
            "n_complete": status.count("complete"),
            "n_pruned": status.count("pruned"),
            "n_failed": status.count("failed"),
            "n_feasible": sum(r.feasible for r in self.records),
            "n_pareto": len(front),
            "pareto": [r.trial_id for r in front],
            "knee": k,
            "best": {name: {"value": v, "trial_id": t} for name, (v, t) in best.items()},
            "compute_saved_s": self.compute_saved_s,
            "wall_s": wall_s,
        }


class _MlflowMirror:
    """Mirrors trials into MLflow as nested runs when EPOCH_MLFLOW_URI is set."""

    def __init__(self, run_name: str):
        try:
            import mlflow

            mlflow.set_tracking_uri(settings().mlflow_uri)
            mlflow.set_experiment("epoch")
            self.mlflow = mlflow
            self.parent = mlflow.start_run(run_name=run_name)
        except Exception:
            self.mlflow = None

    def log_trial(self, r: Record, metrics: dict) -> None:
        if not self.mlflow:
            return
        with self.mlflow.start_run(run_name=f"trial-{r.number}", nested=True):
            self.mlflow.log_params({k: str(v) for k, v in r.genome.items()})
            self.mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None})
            self.mlflow.set_tags({"origin": r.origin, "status": r.result.status, "genome_id": r.gid})

    def finish(self, summary: dict) -> None:
        if not self.mlflow:
            return
        self.mlflow.log_metrics({"final_hv": summary["final_hv"], "n_pareto": summary["n_pareto"]})
        self.mlflow.end_run()
