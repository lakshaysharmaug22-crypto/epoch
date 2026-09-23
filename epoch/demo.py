"""End-to-end pipeline: race → attribution → twin (validation + scenarios) → self-healing → replay bundle.

    epoch demo --workloads triage,rag --budget 60 --seeds 3

Every number in the bundle comes from this run on this machine; nothing is hand-entered.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from rich.console import Console

from epoch.attribution import ablation, fanova
from epoch.config import settings
from epoch.evolution.engine import EngineConfig
from epoch.evolution.race import race
from epoch.export import export_bundle, workload_bundle
from epoch.healing import Healer, approve, scenario_for
from epoch.memory.store import Store
from epoch.twin.profile import measure_profile
from epoch.twin.sim import Fault, Traffic, TwinConfig, simulate
from epoch.twin.validate import validate_twin
from epoch.workloads import get_workload

console = Console()


def _save(name: str, workload: str, obj: Any) -> None:
    d = settings().home / "artifacts" / workload
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.json").write_text(json.dumps(obj, default=str))


def load_extras(workload: str) -> dict[str, Any]:
    d = settings().home / "artifacts" / workload
    out: dict[str, Any] = {}
    if d.exists():
        for p in d.glob("*.json"):
            out[p.stem] = json.loads(p.read_text())
    return out


def twin_pack(wl, knee_genome: dict, frontier: list[dict]) -> dict[str, Any]:
    profiles = {}
    for t in frontier[:10]:
        g = {**t["genome"], **({"batch_size": 1} if "batch_size" in t["genome"] else {})}
        p = measure_profile(wl, g, n=200)
        profiles[t["genome_id"]] = {**p.to_json(), "batch_size": int(t["genome"].get("batch_size", 1)),
                                    "trial_id": t["id"], "metrics": t["metrics"]}
    kg = {**knee_genome, **({"batch_size": 1} if "batch_size" in knee_genome else {})}
    prof = measure_profile(wl, kg, n=300)
    bs = int(knee_genome.get("batch_size", 1))
    cap = prof.capacity_rps(1, bs)
    wait = 5.0 if bs > 1 else 0.0
    scenarios = []
    specs = [
        ("steady", "Steady Poisson at 50% capacity", TwinConfig(1, bs, wait), Traffic("poisson", 0.5 * cap, 90), []),
        ("bursty", "Bursts of 3× every 30 s", TwinConfig(1, bs, wait), Traffic("bursty", 0.35 * cap, 90, 3.0, 30, 6), []),
        ("replica_loss", "2 replicas, one lost for 20 s", TwinConfig(2, bs, wait), Traffic("poisson", 0.8 * cap, 90),
         [Fault("replica_down", 30, 20, replica=1)]),
        ("slowdown", "Dependency slowdown 2.5× for 15 s", TwinConfig(1, bs, wait), Traffic("poisson", 0.35 * cap, 90),
         [Fault("slowdown", 40, 15, 2.5)]),
        ("diurnal", "Diurnal load (±60%)", TwinConfig(1, bs, wait), Traffic("diurnal", 0.45 * cap, 120, period_s=120), []),
    ]
    for key, title, cfg, tr, faults in specs:
        cfg.slo_ms = None
        res = simulate(prof, cfg, tr, faults, seed=0)
        scenarios.append({"key": key, "title": title, **res})
    return {"profiles": profiles, "knee_profile": {**prof.to_json(), "batch_size": bs}, "scenarios": scenarios,
            "capacity_rps": cap}


def pick_deployed(wl, wb: dict, need_resolver: bool = False) -> dict | None:
    """Knee among feasible configs (optionally only those that use a tier-3 resolver)."""
    import numpy as np

    from epoch.evolution.pareto import normalize

    objs = wl.objectives
    pool = [t for t in wb["trials"] if t["status"] == "complete" and t["feasible"]
            and (not need_resolver or t["genome"].get("t3_policy") not in (None, "off"))]
    if not pool:
        return None
    pts = normalize(np.array([[t["metrics"][o.name] for o in objs] for t in pool]), objs)
    return pool[int(np.argmin(np.linalg.norm(pts, axis=1)))]


def run_healing(store: Store, wl, wb: dict) -> None:
    console.rule(f"[bold red]{wl.name}[/] self-healing")
    frontier = [t for t in wb["trials"] if t["id"] in set(wb["pareto"])]
    archive = [{"trial_id": t["id"], "genome": t["genome"], "metrics": t["metrics"]} for t in frontier]
    knee = next(t for t in wb["trials"] if t["id"] == wb["knee"])
    resolver = pick_deployed(wl, wb, need_resolver=True)
    plan = [("latency", knee), ("drift", knee)] + ([("resolver", resolver)] if resolver else [])
    for sc_name, trial in plan:
        h = Healer(wl, scenario_for(wl, sc_name), store, trial["genome"], archive=archive)
        data = h.run(approval="manual")
        if data.get("proposal", {}).get("status") == "awaiting_approval":
            approve(store, wl, h.incident_id, approved=True, actor="epoch demo (scripted approval)",
                    note="approved after reviewing canary + twin evidence")
        inc = store.incident(h.incident_id)
        console.print(f"  {h.incident_id}: {inc['status']} · {inc['title']}")


def run_demo(workloads: list[str], budget: int, seeds: int, strategies: list[str], out: Path,
             heal: bool = True, validate: bool = True, rag_budget: int | None = None) -> Path:
    store = Store()
    t_all = time.perf_counter()
    for name in workloads:
        wl = get_workload(name)
        b = rag_budget if (name == "rag" and rag_budget) else budget
        console.rule(f"[bold red]{name}[/] race · {strategies} · budget {b} · seeds {seeds}")
        pop = 12 if b >= 48 else 8

        def sink(kind: str, p: dict) -> None:
            if kind == "run.end":
                console.print(f"  [dim]{p['run_id']}[/] hv={p['final_hv']:.4f} pareto={p['n_pareto']}")
            elif kind == "hypothesis.verdict":
                console.print(f"  [magenta]{p['id']}[/] → {p['verdict']}")

        t0 = time.perf_counter()
        res = race(wl, strategies, b, list(range(seeds)), store, cfg=EngineConfig(population=pop, warmup=pop), sink=sink)
        res["wall_s"] = time.perf_counter() - t0
        res["curves"] = {rid: {"strategy": s, "seed": i, "hv": next(r for r in store.runs(workload=name) if r["id"] == rid)["summary"]["hv_curve"]}
                         for s, ids in res["run_ids"].items() for i, rid in enumerate(ids)}
        res["seeds"] = list(range(seeds))
        _save("race", name, res)
        console.print(f"  headline: {res.get('headline')}")

        wb = workload_bundle(store, name)
        knee = next(t for t in wb["trials"] if t["id"] == wb["knee"])
        frontier = [t for t in wb["trials"] if t["id"] in set(wb["pareto"])]
        console.rule(f"[bold red]{name}[/] attribution")
        fa = fanova(wl, [t for t in wb["trials"]])
        ab = ablation(wl, knee["genome"], repeats=3)
        _save("attribution", name, {"fanova": fa, "ablation": ab, "knee": knee["id"]})

        console.rule(f"[bold red]{name}[/] digital twin")
        twin = twin_pack(wl, knee["genome"], frontier)
        if validate:
            try:
                twin["validation"] = validate_twin(name, knee["genome"], duration_s=20)
                console.print(f"  twin MAPE p50={twin['validation']['mape_p50']:.3f} p95={twin['validation']['mape_p95']:.3f}")
            except Exception as exc:
                twin["validation"] = {"error": str(exc)[:300]}
        _save("twin", name, twin)

        if heal and name == "triage":
            run_healing(store, wl, wb)
    notes = [
        "All metrics were measured by `epoch demo` on the machine in meta.provenance.",
        "LLM agents ran in heuristic mode (no API key)." if not settings().llm_enabled else "LLM agents ran on Claude.",
    ]
    path = export_bundle(store, workloads, out, extras={w: load_extras(w) for w in workloads}, notes=notes)
    console.rule(f"[bold green]bundle → {path} ({path.stat().st_size / 1e6:.1f} MB) in {(time.perf_counter() - t_all) / 60:.1f} min")
    return path
