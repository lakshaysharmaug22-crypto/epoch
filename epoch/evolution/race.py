"""Head-to-head strategy race with an equal trial budget over several seeds.

Headline metric: trials needed to reach a hypervolume target. The target is a fraction of the best
hypervolume any run found (pooled reference), and we also report how many trials each strategy needs to
match NSGA-II's median final hypervolume."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

import numpy as np

from epoch.evolution.engine import EngineConfig, Evolution
from epoch.memory.store import Store
from epoch.workloads.base import Workload


def trials_to(curve: list[float], target: float) -> int | None:
    for i, v in enumerate(curve):
        if v >= target - 1e-12:
            return i + 1
    return None


def summarize(curves: dict[str, list[list[float]]], budget: int, target_frac: float = 0.95) -> dict[str, Any]:
    finals = [c[-1] for runs in curves.values() for c in runs if c]
    ref = max(finals) if finals else 0.0
    target = target_frac * ref
    nsga_final = float(np.median([c[-1] for c in curves.get("nsga2", []) if c])) if curves.get("nsga2") else None
    out: dict[str, Any] = {"reference_hv": ref, "target_hv": target, "target_frac": target_frac,
                           "nsga2_median_final_hv": nsga_final, "budget": budget, "strategies": {}}
    for s, runs in curves.items():
        ttt = [trials_to(c, target) for c in runs]
        ttn = [trials_to(c, nsga_final) for c in runs] if nsga_final is not None else []
        censored = [t if t is not None else budget + 1 for t in ttt]
        mean_curve = np.mean(np.array([c + [c[-1]] * (budget - len(c)) for c in runs]), axis=0) if runs else np.array([])
        out["strategies"][s] = {
            "runs": len(runs),
            "final_hv": [c[-1] for c in runs],
            "final_hv_median": float(np.median([c[-1] for c in runs])),
            "trials_to_target": ttt,
            "trials_to_target_median": float(np.median(censored)),
            "reached_target": sum(t is not None for t in ttt),
            "trials_to_nsga2_final": ttn,
            "trials_to_nsga2_final_median": float(np.median([t if t is not None else budget + 1 for t in ttn])) if ttn else None,
            "auc": float(mean_curve.mean()) if mean_curve.size else 0.0,
            "mean_curve": [round(float(x), 5) for x in mean_curve],
        }
    if "agent" in out["strategies"] and "nsga2" in out["strategies"]:
        a = out["strategies"]["agent"]["trials_to_nsga2_final_median"]
        n = out["strategies"]["nsga2"]["trials_to_nsga2_final_median"]
        if a and n:
            out["headline"] = {
                "agent_trials_to_nsga2_final": a,
                "nsga2_trials_to_own_final": n,
                "pct_fewer_trials": round(100 * (1 - a / n), 1),
            }
    return out


def race(workload: Workload, strategies: list[str], budget: int, seeds: list[int], store: Store, *,
         cfg: EngineConfig | None = None, sink: Callable[[str, dict], None] | None = None,
         group_id: str | None = None) -> dict[str, Any]:
    group_id = group_id or f"race-{workload.name}-{uuid.uuid4().hex[:6]}"
    base = cfg or EngineConfig()
    curves: dict[str, list[list[float]]] = {s: [] for s in strategies}
    run_ids: dict[str, list[str]] = {s: [] for s in strategies}
    for seed in seeds:
        for s in strategies:
            c = EngineConfig(**{**base.__dict__, "budget": budget, "seed": seed})
            evo = Evolution(workload, s, store, c, group_id=group_id, sink=sink)
            summary = evo.run()
            curves[s].append(summary["hv_curve"])
            run_ids[s].append(evo.run_id)
    result = summarize(curves, budget)
    result["group_id"] = group_id
    result["run_ids"] = run_ids
    store.add_event("race.end", {"group_id": group_id, "headline": result.get("headline")})
    return result
