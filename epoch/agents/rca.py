"""Root-cause + remediation agent (LangGraph).

  evidence ─► diagnose ─► narrate ─► candidates ─► canary ─► twin ─► gate ─► END

evidence    baseline vs incident windows: per-stage latency deltas, routing-share shifts, per-tier precision,
            confidence shift, input drift (char-trigram JS divergence, OOV rate), similar past incidents/failures
diagnose    scored, evidence-backed causes (deterministic rules over the evidence)
narrate     Claude writes the incident summary and may re-rank causes; offline: templated from the top cause
candidates  configs from the Pareto archive + targeted genome patches for the implicated genes
canary      every candidate replays recent production traffic *with the fault still active*
twin        top candidates are profiled under the fault and simulated at production load vs the incumbent
gate        best candidate that restores the targets is proposed; a human must approve before promotion
"""

from __future__ import annotations

import json
from typing import Any, TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph

from epoch.agents import llm
from epoch.config import settings
from epoch.genome import Genome, genome_id
from epoch.twin.profile import measure_profile
from epoch.twin.sim import Traffic, TwinConfig, plan_capacity, simulate

PATCHES: dict[str, list[tuple[str, Any]]] = {
    # cause kind -> (gene, value | callable(genome)->value)
    "input_drift": [("normalize", "hinglish"), ("t2_features", "char"), ("t2_features", "word_char"),
                    ("t1_tau", lambda g: min(1.0, g.get("t1_tau", 0.9) + 0.08)), ("retriever", "hybrid")],
    "tier1_precision": [("t1_tau", lambda g: min(1.0, g.get("t1_tau", 0.9) + 0.08)),
                        ("t1_delta", lambda g: min(0.4, g.get("t1_delta", 0.1) + 0.1)), ("normalize", "hinglish")],
    "stage_latency": [("t3_policy", "off"), ("t2_tau", lambda g: max(0.3, g.get("t2_tau", 0.8) - 0.2)),
                      ("t1_tau", lambda g: max(0.55, g.get("t1_tau", 0.9) - 0.1)),
                      ("batch_size", lambda g: max(1, int(g.get("batch_size", 16)) // 4)), ("reranker", False)],
    "escalation_surge": [("t2_tau", lambda g: max(0.3, g.get("t2_tau", 0.8) - 0.15)), ("t3_policy", "off")],
}


class RCAState(TypedDict, total=False):
    evidence: dict
    causes: list
    narrative: dict
    candidates: list
    canary: list
    twin: dict
    proposal: dict


class RCAAgent:
    """`ctx` supplies: workload, incumbent genome, baseline/incident window metrics, recent traffic, fault
    environment, archive (Pareto configs), targets, store."""

    def __init__(self, ctx: dict[str, Any]):
        self.ctx = ctx
        self.w = ctx["workload"]
        self.space = self.w.space()
        g = StateGraph(RCAState)
        for name in ("evidence", "diagnose", "narrate", "candidates", "canary", "twin", "gate"):
            g.add_node(name, getattr(self, "_" + name))
        g.add_edge(START, "evidence")
        for a, b in [("evidence", "diagnose"), ("diagnose", "narrate"), ("narrate", "candidates"),
                     ("candidates", "canary"), ("canary", "twin"), ("twin", "gate")]:
            g.add_edge(a, b)
        g.add_edge("gate", END)
        self.graph = g.compile()

    def run(self) -> dict[str, Any]:
        return dict(self.graph.invoke({}))

    # ------------------------------------------------------------------ nodes
    def _evidence(self, _: RCAState) -> RCAState:
        base, inc = self.ctx["baseline_windows"], self.ctx["incident_windows"]

        def avg(ws: list[dict], k: str) -> float | None:
            v = [w[k] for w in ws if w.get(k) is not None and np.isfinite(w[k])]
            return float(np.mean(v)) if v else None

        stages = sorted({s for w in base + inc for s in w.get("stage_ms", {})})
        stage_delta = {}
        for s in stages:
            b = float(np.mean([w["stage_ms"].get(s, 0.0) for w in base]))
            i = float(np.mean([w["stage_ms"].get(s, 0.0) for w in inc]))
            stage_delta[s] = {"baseline_ms": b, "incident_ms": i, "delta_ms": i - b, "ratio": (i / b) if b > 0 else None}
        keys = [k for k in inc[-1] if k.startswith(("share_t", "precision_t"))] + [self.w.quality_metric, "p95_ms",
                                                                                   "mean_conf", "drift_js", "oov_rate"]
        shifts = {k: {"baseline": avg(base, k), "incident": avg(inc, k)} for k in keys}
        for v in shifts.values():
            v["delta"] = (v["incident"] - v["baseline"]) if v["incident"] is not None and v["baseline"] is not None else None
        store = self.ctx["store"]
        query = " ".join(f"{k} {v['delta']:+.3f}" for k, v in shifts.items() if v["delta"] is not None)
        similar = [{"id": i["id"], "title": i["title"], "status": i["status"]} for i in store.incidents()
                   if i["id"] != self.ctx["incident_id"]][-3:]
        failures = store.similar_failures(query, k=3, workload=self.w.name)
        ev = {"stages": stage_delta, "shifts": shifts, "similar_incidents": similar,
              "similar_failures": [{"kind": f["kind"], "message": f["message"][-160:], "similarity": f["similarity"]}
                                   for f in failures],
              "alarms": self.ctx["alarms"]}
        return {"evidence": ev}

    def _diagnose(self, s: RCAState) -> RCAState:
        ev = s["evidence"]
        sh, st = ev["shifts"], ev["stages"]
        q = self.w.quality_metric
        causes = []
        js, oov = sh.get("drift_js", {}), sh.get("oov_rate", {})
        if js.get("delta") is not None and (js["delta"] > 0.02 or (oov.get("delta") or 0) > 0.05):
            causes.append({"kind": "input_drift", "title": "Input distribution drift (new phrasing / code-mixing)",
                           "score": float(min(1.0, js["delta"] * 8 + max(0.0, oov.get("delta") or 0) * 3)),
                           "evidence": [f"char-trigram JS divergence {js['baseline']:.3f} → {js['incident']:.3f}",
                                        f"OOV rate {100 * (oov.get('baseline') or 0):.1f}% → {100 * (oov.get('incident') or 0):.1f}%"]})
        p1 = sh.get("precision_t1", {})
        if p1.get("delta") is not None and p1["delta"] < -0.04:
            causes.append({"kind": "tier1_precision", "title": "Tier-1 fuzzy gate accepting wrong queues",
                           "score": float(min(1.0, -p1["delta"] * 4)),
                           "evidence": [f"tier-1 precision {p1['baseline']:.3f} → {p1['incident']:.3f}",
                                        f"tier-1 share {sh.get('share_t1', {}).get('baseline') or 0:.2f} → {sh.get('share_t1', {}).get('incident') or 0:.2f}"]})
        if st:
            worst = max(st.items(), key=lambda kv: kv[1]["delta_ms"])
            total = sum(max(0.0, v["delta_ms"]) for v in st.values()) or 1e-9
            if worst[1]["delta_ms"] > 0.2 and (worst[1]["ratio"] or 0) > 1.5:
                causes.append({"kind": "stage_latency", "stage": worst[0],
                               "title": f"Latency regression in `{worst[0]}` stage",
                               "score": float(min(1.0, 0.4 + 0.6 * worst[1]["delta_ms"] / total)),
                               "evidence": [f"{worst[0]} {worst[1]['baseline_ms']:.2f} → {worst[1]['incident_ms']:.2f} ms/request "
                                            f"({worst[1]['ratio']:.1f}×), {100 * worst[1]['delta_ms'] / total:.0f}% of the added latency"]})
        s3 = sh.get("share_t3", {})
        if s3.get("delta") is not None and s3["delta"] > 0.05:
            causes.append({"kind": "escalation_surge", "title": "Escalation surge to tier 3",
                           "score": float(min(1.0, s3["delta"] * 3)),
                           "evidence": [f"tier-3 share {s3['baseline']:.2f} → {s3['incident']:.2f}"]})
        qd = sh.get(q, {})
        if not causes:
            causes.append({"kind": "unknown", "title": "No single dominant cause", "score": 0.2,
                           "evidence": [f"{q} {qd.get('baseline')} → {qd.get('incident')}"]})
        causes.sort(key=lambda c: -c["score"])
        return {"causes": causes}

    def _narrate(self, s: RCAState) -> RCAState:
        causes, ev = s["causes"], s["evidence"]
        top = causes[0]
        q = self.w.quality_metric
        qd = ev["shifts"].get(q, {})
        p95 = ev["shifts"].get("p95_ms", {})
        fallback = {
            "summary": f"{top['title']}. {q} {qd.get('baseline', 0):.3f} → {qd.get('incident', 0):.3f}; "
                       f"p95 {p95.get('baseline', 0):.1f} → {p95.get('incident', 0):.1f} ms. " + "; ".join(top["evidence"]),
            "primary_cause": top["kind"], "source": "rules",
        }
        if not settings().llm_enabled:
            return {"narrative": fallback}
        schema = {"type": "object", "properties": {
            "summary": {"type": "string", "description": "3-4 sentence incident summary for an on-call engineer"},
            "primary_cause": {"type": "string", "enum": [c["kind"] for c in causes]},
            "reasoning": {"type": "string"}}, "required": ["summary", "primary_cause", "reasoning"]}
        try:
            r = llm.structured(model=settings().agent_model, schema=schema, tool_name="write_rca", max_tokens=700,
                               system="You are the on-call SRE agent for an ML serving system. Explain the incident from the "
                                      "evidence only; name the primary cause; be specific with numbers.",
                               user=json.dumps({"evidence": ev, "candidate_causes": causes}, default=str))
            out = {**r.data, "source": "llm", "model": r.model, "cost_usd": r.cost_usd}
            return {"narrative": out}
        except Exception as exc:
            return {"narrative": {**fallback, "llm_error": str(exc)[:200]}}

    def _candidates(self, s: RCAState) -> RCAState:
        inc = self.ctx["incumbent"]
        causes = s["causes"]
        primary = s["narrative"].get("primary_cause", causes[0]["kind"])
        order = [primary] + [c["kind"] for c in causes if c["kind"] != primary]
        seen = {genome_id(inc)}
        cands: list[dict] = []

        def add(g: Genome, origin: str, why: str) -> None:
            g = self.space.repair(g, inc)
            gid = genome_id(g)
            if gid in seen or self.space.validate(g):
                return
            seen.add(gid)
            cands.append({"genome": g, "genome_id": gid, "origin": origin, "why": why,
                          "diff": {k: list(v) for k, v in self.space.diff(inc, g).items()}})

        for kind in order:
            for gene, val in PATCHES.get(kind, []):
                if gene not in self.space.by_name:
                    continue
                v = val(inc) if callable(val) else val
                if inc.get(gene) != v:
                    add({**inc, gene: v}, "patch", f"{kind}: set {gene}={v}")
        patches = [c for c in cands if c["origin"] == "patch"]
        if len(patches) >= 2:  # one combined patch of the two strongest single edits
            add({**inc, **{k: v[1] for c in patches[:2] for k, v in c["diff"].items()}}, "patch", "combined top-2 patches")
        adapt = self.ctx.get("adapt")
        if adapt is not None:  # CSAI error-bank loop: retrain on audited production tickets
            self.space = self.w.space()  # the new data version is now a valid gene value
            n = self.ctx.get("audited", 0)
            for base, why in [(inc, "incumbent")] + [(c["genome"], c["why"]) for c in patches[:2]]:
                g = adapt(base)
                if g is not None:
                    add(g, "retrain", f"retrain tier-2 on error bank (+{n} audited) · {why}")
        for a in self.ctx.get("archive", [])[:5]:
            add(dict(a["genome"]), "pareto-archive", f"Pareto config {a.get('trial_id', '')}")
        return {"candidates": cands[:12]}

    def _canary(self, s: RCAState) -> RCAState:
        xs, ys = self.ctx["canary_inputs"], self.ctx["canary_labels"]
        env = self.ctx["fault_env"]
        evaluate = self.ctx["evaluate_window"]
        def repeated(g: Genome) -> dict:  # latency is noisy (often bimodal): median of 3 replays + spread
            runs = [evaluate(g, xs, ys, env) for _ in range(3)]
            m = dict(runs[0])
            for k in ("p50_ms", "p95_ms", "p99_ms", "mean_ms"):
                vals = [r[k] for r in runs]
                m[k] = float(np.median(vals))
                m[k.replace("_ms", "_spread_ms")] = float(max(vals) - min(vals))
            return m

        rows = [{"genome_id": genome_id(self.ctx["incumbent"]), "origin": "incumbent", "why": "current deployment",
                 "diff": {}, "metrics": repeated(self.ctx["incumbent"])}]
        for c in s["candidates"]:
            try:
                rows.append({**c, "metrics": repeated(c["genome"])})
            except Exception as exc:
                rows.append({**c, "metrics": None, "error": str(exc)[:200]})
        t = self.ctx["targets"]
        q = self.w.quality_metric
        for r in rows:
            m = r.get("metrics")
            r["meets_targets"] = bool(m and m[q] >= t[q] and m["p95_ms"] <= t["p95_ms"])
        return {"canary": rows}

    def _twin(self, s: RCAState) -> RCAState:
        q = self.w.quality_metric
        rows = s["canary"]
        t = self.ctx["targets"]
        pool = [r for r in rows[1:] if r.get("metrics")]
        pool.sort(key=lambda r: (not r["meets_targets"],
                                 2 * max(0.0, (t[q] - r["metrics"][q]) / max(abs(t[q]), 1e-9))
                                 + max(0.0, (r["metrics"]["p95_ms"] - t["p95_ms"]) / t["p95_ms"]),
                                 -r["metrics"][q]))
        top = pool[:3]
        env = self.ctx["fault_env"]
        inputs = self.ctx["canary_inputs"]
        rate = self.ctx["prod_rps"]
        slo = self.ctx["twin_slo_ms"]
        traffic = Traffic("bursty", rate=rate, duration_s=60, burst_factor=1.8, burst_every_s=20, burst_len_s=5)
        price = self.ctx.get("price_per_replica_hour", 0.05)
        out = {"traffic": {"kind": "bursty", "rate": rate, "burst_factor": 1.8}, "slo_ms": slo, "results": []}
        for label, g in [("incumbent", self.ctx["incumbent"])] + [(r["genome_id"], r["genome"]) for r in top]:
            prof = measure_profile(self.w, {**g, **({"batch_size": 1} if "batch_size" in g else {})}, n=150,
                                   inputs=inputs, inject_ms=env)
            bs = int(g.get("batch_size", 1))
            wait = 5.0 if bs > 1 else 0.0
            sim = simulate(prof, TwinConfig(replicas=1, max_batch=bs, max_wait_ms=wait, slo_ms=slo), traffic, seed=0)["summary"]
            plan = plan_capacity(prof, traffic, slo, price_per_replica_hour=price, max_replicas=8, batches=(bs,), waits=(wait,))
            best = plan["best"]
            out["results"].append({"genome_id": label, "p50": sim["p50"], "p95": sim["p95"], "p99": sim["p99"],
                                   "utilization": sim["utilization"], "slo_attainment": sim["slo_attainment"],
                                   "stable": sim["stable"], "capacity_rps": prof.capacity_rps(1, bs),
                                   "replicas_needed": best["replicas"] if best else None,
                                   "p95_at_plan": best["p95"] if best else None,
                                   "profile": prof.to_json()})
        return {"twin": out}

    def _gate(self, s: RCAState) -> RCAState:
        q = self.w.quality_metric
        t = self.ctx["targets"]
        twin = {r["genome_id"]: r for r in s["twin"]["results"]}

        def shortfall(m: dict) -> float:  # 0 = meets every canary target; quality misses weigh double
            sq = max(0.0, (t[q] - m[q]) / max(abs(t[q]), 1e-9))
            sp = max(0.0, (m["p95_ms"] - t["p95_ms"]) / t["p95_ms"])
            return 2 * sq + sp

        inc = s["canary"][0]["metrics"]
        s["canary"][0]["shortfall"] = shortfall(inc)
        best, best_key = None, None
        for r in s["canary"][1:]:
            m = r.get("metrics")
            if not m:
                continue
            r["shortfall"] = shortfall(m)
            tw = twin.get(r["genome_id"])
            if tw is None:  # only twin-validated candidates can be proposed
                continue
            reps = tw["replicas_needed"]
            # 1) canary targets met AND servable at production load  2) servable at all (twin plan exists)
            # 3) smallest canary shortfall  4) fewest replicas  5) quality  6) latency
            key = (r["meets_targets"] and reps is not None, reps is not None, -round(r["shortfall"], 4),
                   -(reps or 99), -(tw["p95"] if tw["p95"] is not None else 1e9), m[q], -m["p95_ms"])
            if best_key is None or key > best_key:
                best, best_key = r, key
        inc_tw = twin.get("incumbent", {})
        inc_sf = shortfall(inc)

        def significant(r: dict) -> bool:
            """Beats the incumbent by more than measurement noise on quality, tail latency, capacity or replicas."""
            m, tw = r["metrics"], twin.get(r["genome_id"], {})
            noise = max(inc.get("p95_spread_ms", 0.0), m.get("p95_spread_ms", 0.0))
            dq = m[q] - inc[q]
            p95_gain = (inc["p95_ms"] - m["p95_ms"]) - noise
            cap_gain = (inc["mean_ms"] - m["mean_ms"]) / max(inc["mean_ms"], 1e-9)
            reps_gain = (inc_tw.get("replicas_needed") or 9) - (tw.get("replicas_needed") or 9)
            sf_gain = r["shortfall"] <= min(0.8 * inc_sf, inc_sf - 0.02)
            return dq > 0.01 or (dq > -0.01 and ((sf_gain and p95_gain > 0) or p95_gain > 0.1 * inc["p95_ms"]
                                                 or cap_gain > 0.15 or reps_gain >= 1))

        if best is None or not significant(best):
            alt = [r for r in s["canary"][1:] if r.get("metrics") and r["genome_id"] in twin and significant(r)]
            best = max(alt, key=lambda r: (r["meets_targets"], -(twin[r["genome_id"]]["replicas_needed"] or 99),
                                           -(twin[r["genome_id"]]["p95"] or 1e9), r["metrics"][q])) if alt else None
        if best is None:
            stage = next((c.get("stage") for c in s["causes"] if c.get("stage")), None)
            return {"proposal": {
                "status": "no_candidate",
                "why": "no configuration beats the incumbent beyond canary noise — the fault is outside the genome's "
                       "control; escalated to the owner of " + (f"`{stage}`" if stage else "the failing dependency"),
                "best_candidate": best and {k: best[k] for k in ("genome_id", "origin", "why", "shortfall")},
                "incumbent_shortfall": inc_sf, "incumbent_replicas_needed": inc_tw.get("replicas_needed")}}
        tw = twin[best["genome_id"]]
        return {"proposal": {
            "status": "awaiting_approval",
            "genome_id": best["genome_id"], "genome": best["genome"], "origin": best["origin"], "why": best["why"],
            "diff": best["diff"], "canary": best["metrics"], "incumbent_canary": inc, "twin": tw, "incumbent_twin": inc_tw,
            "replicas": tw["replicas_needed"], "incumbent_replicas_needed": inc_tw.get("replicas_needed"),
            "meets_targets": best["meets_targets"], "shortfall": best["shortfall"], "incumbent_shortfall": shortfall(inc),
            "expected_gain": {q: best["metrics"][q] - inc[q], "p95_ms": best["metrics"]["p95_ms"] - inc["p95_ms"]},
        }}
