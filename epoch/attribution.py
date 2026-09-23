"""Causal attribution of the frontier.

fANOVA        functional-ANOVA importance per objective (Optuna evaluator, over genes present in every trial)
ablation      interventional one-gene ablation: take the chosen config, do(gene := reference value) holding every
              other gene fixed, re-measure. Effects smaller than 2σ of repeated-measurement noise are marked
              not significant. This is what lets a report say *which knob actually moved p95*.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import optuna

from epoch.benchmark.harness import run_benchmark
from epoch.genome import Genome
from epoch.workloads.base import Workload


def fanova(workload: Workload, trials: list[dict], seed: int = 0) -> dict[str, dict[str, float]]:
    space = workload.space()
    dists = space.distributions()
    done = [t for t in trials if t["status"] == "complete" and all(t["metrics"].get(o.name) is not None
                                                                   for o in workload.objectives)]
    if len(done) < 8:
        return {}
    study = optuna.create_study(directions=["minimize" if o.direction == "min" else "maximize" for o in workload.objectives])
    for t in done:
        params = {k: v for k, v in t["genome"].items() if k in dists}
        try:
            study.add_trial(optuna.trial.create_trial(
                params=params, distributions={k: dists[k] for k in params},
                values=[float(t["metrics"][o.name]) for o in workload.objectives]))
        except ValueError:
            continue
    out: dict[str, dict[str, float]] = {}
    for j, o in enumerate(workload.objectives):
        try:
            imp = optuna.importance.get_param_importances(
                study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=seed), target=lambda ft, j=j: ft.values[j])
            out[o.name] = {k: round(float(v), 4) for k, v in imp.items()}
        except Exception:
            out[o.name] = {}
    return out


def ablation(workload: Workload, chosen: Genome, reference: Genome | None = None, *, repeats: int = 3,
             seed: int = 0) -> dict[str, Any]:
    space = workload.space()
    reference = reference or space.default()
    names = [o.name for o in workload.objectives]
    base_runs = [run_benchmark(workload, chosen, seed=seed + i, robustness=False) for i in range(repeats)]
    base_ok = [r for r in base_runs if r.status == "complete"]
    if not base_ok:
        return {"error": "chosen genome failed to run", "effects": []}
    base = {n: float(np.mean([r.metrics[n] for r in base_ok])) for n in names}
    noise = {n: float(np.std([r.metrics[n] for r in base_ok], ddof=1)) if len(base_ok) > 1 else 0.0 for n in names}
    effects = []
    for gene, (_, ref_val) in space.diff(chosen, reference).items():
        if gene not in chosen:
            continue
        intervened = space.repair({**chosen, gene: ref_val}, chosen) if ref_val is not None else None
        if intervened is None or intervened == chosen:
            continue
        r = run_benchmark(workload, intervened, seed=seed, robustness=False)
        if r.status != "complete":
            effects.append({"gene": gene, "from": chosen[gene], "to": ref_val, "status": r.status, "error_kind": r.error_kind})
            continue
        delta = {n: r.metrics[n] - base[n] for n in names}
        sig = {n: abs(delta[n]) > max(2 * noise[n], 1e-3 * max(abs(base[n]), 1e-9)) for n in names}
        effects.append({"gene": gene, "from": chosen[gene], "to": ref_val, "status": "complete",
                        "metrics": {n: r.metrics[n] for n in names}, "delta": delta, "significant": sig})
    primary = workload.objectives[0].name
    effects.sort(key=lambda e: -abs(e.get("delta", {}).get(primary, 0.0)))
    return {"chosen": chosen, "reference": reference, "base": base, "noise": noise, "repeats": len(base_ok),
            "effects": effects}
