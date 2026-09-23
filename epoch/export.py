"""Replay bundle: one JSON file with everything the dashboard renders. The live API serves the same shape at
/api/bundle, so the UI has a single data contract for hosted replay and local live mode."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from epoch import __version__
from epoch.config import settings
from epoch.evolution.pareto import hypervolume, non_dominated, normalize
from epoch.memory.provenance import provenance
from epoch.memory.store import Store
from epoch.workloads import get_workload

KEEP_METRICS = {
    "macro_f1", "accuracy", "answer_f1", "exact_match", "retrieval_recall", "quality", "p50_ms", "p95_ms", "p99_ms",
    "throughput_rps", "peak_mem_mb", "cost_per_1k", "llm_usd_per_req", "build_s", "robustness", "quality_robust",
    "share_t1", "share_t2", "share_t3", "precision_t1", "precision_t2", "precision_t3", "escalation_rate",
}


def _clean(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_clean(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating | float):
        f = float(o)
        return None if not np.isfinite(f) else round(f, 6)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def workload_bundle(store: Store, name: str, extras: dict[str, Any] | None = None) -> dict[str, Any]:
    wl = get_workload(name)
    runs = [r for r in store.runs(workload=name) if r["status"] == "complete"]
    trials: list[dict] = []
    for r in runs:
        for t in store.trials(r["id"]):
            m = {k: v for k, v in (t["metrics"] or {}).items() if k in KEEP_METRICS}
            trials.append({
                "id": t["id"], "run_id": t["run_id"], "number": t["number"], "strategy": r["strategy"], "seed": r["seed"],
                "group_id": r["group_id"], "genome_id": t["genome_id"], "genome": t["genome"], "origin": t["origin"],
                "generation": t["generation"], "status": t["status"], "metrics": m, "violations": t["violations"],
                "feasible": t["feasible"], "pareto": t["pareto"], "fidelity": t["fidelity"], "duration_s": t["duration_s"],
                "hv_after": t["hv_after"], "parents": t["parents"], "parents_inferred": t["parents_inferred"],
                "hypothesis_id": t["hypothesis_id"], "latency_hist": (t["extras"] or {}).get("latency_hist"),
                "stage_ms": (t["extras"] or {}).get("stage_ms"), "prune_reason": (t["extras"] or {}).get("prune_reason"),
                "error_kind": t["error_kind"], "error": (t["error"] or "")[-600:] or None, "created_at": t["created_at"],
            })
    objs = wl.objectives
    feas = [t for t in trials if t["feasible"] and all(t["metrics"].get(o.name) is not None for o in objs)]
    uniq: dict[str, dict] = {}
    for t in feas:
        uniq.setdefault(t["genome_id"], t)
    pool = list(uniq.values())
    pts = normalize(np.array([[t["metrics"][o.name] for o in objs] for t in pool]), objs) if pool else np.zeros((0, len(objs)))
    mask = non_dominated(pts) if pool else np.zeros(0, dtype=bool)
    front = [t for t, k in zip(pool, mask, strict=True) if k]
    fpts = pts[mask] if pool else pts
    knee = front[int(np.argmin(np.linalg.norm(fpts, axis=1)))]["id"] if front else None
    front_ids = {t["id"] for t in front}
    for t in trials:
        t["global_pareto"] = t["id"] in front_ids
    hyps = [h for r in runs for h in store.hypotheses(r["id"])]
    fails = [f for r in runs for f in store.failures(r["id"])]
    knee_trial = next((t for t in trials if t["id"] == knee), None)
    topo = None
    if knee_trial:
        topo = {"trial_id": knee, "genome_id": knee_trial["genome_id"], "genome": knee_trial["genome"],
                "metrics": knee_trial["metrics"], "graph": wl.topology(knee_trial["genome"], knee_trial["metrics"])}
    incs = [i for i in store.incidents() if i["workload"] == name]
    dep_topo = None
    if incs:  # topology of the config the most recent incident started from (the ops tape replays it)
        from epoch.genome import genome_id as _gid

        g = incs[-1]["data"]["deployed"]
        gid = _gid(g)
        match = next((t for t in trials if t["genome_id"] == gid and t["status"] == "complete"), None)
        mets = match["metrics"] if match else (knee_trial or {}).get("metrics", {})
        dep_topo = {"trial_id": match["id"] if match else None, "genome_id": gid, "genome": g,
                    "metrics": mets, "graph": wl.topology(g, mets)}
    return _clean({
        "describe": wl.describe(),
        "deployed_topology": dep_topo,
        "runs": [{"id": r["id"], "strategy": r["strategy"], "seed": r["seed"], "group_id": r["group_id"],
                  "budget": r["budget"], "created_at": r["created_at"], "finished_at": r["finished_at"],
                  "provenance": r["provenance"], "summary": {k: v for k, v in r["summary"].items() if k != "pareto"}}
                 for r in runs],
        "trials": trials,
        "pareto": [t["id"] for t in front],
        "global_hv": hypervolume(fpts) if pool else 0.0,
        "knee": knee,
        "topology": topo,
        "hypotheses": hyps,
        "failures": [{"id": f["id"], "run_id": f["run_id"], "trial_id": f["trial_id"], "kind": f["kind"],
                      "message": f["message"][-400:], "genome": f["genome"], "created_at": f["created_at"]} for f in fails],
        "incidents": incs,
        "deployments": store.deployments(name),
        **(extras or {}),
    })


def export_bundle(store: Store, workloads: list[str], out: Path, extras: dict[str, dict] | None = None,
                  notes: list[str] | None = None) -> Path:
    s = settings()
    bundle = {
        "meta": {
            "generated_at": datetime.now(UTC).isoformat(),
            "epoch_version": __version__,
            "mode": "replay",
            "provenance": provenance(0),
            "llm": {"enabled": s.llm_enabled, "agent_model": s.agent_model if s.llm_enabled else None,
                    "judge_model": s.judge_model if s.llm_enabled else None},
            "workloads": workloads,
            "notes": notes or [],
        },
        "workloads": {w: workload_bundle(store, w, (extras or {}).get(w)) for w in workloads},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_clean(bundle), separators=(",", ":"), default=str)
    out.write_text(text)
    # same data as a script, for static hosts / sandboxes that block fetch()
    out.with_suffix(".js").write_text(f"window.__EPOCH_BUNDLE__={text};")
    return out
