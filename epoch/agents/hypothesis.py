"""Hypothesis agent (LangGraph).

    analyze ──► hypothesize ──► materialize ──► screen ──► END
                     ▲                             │
                     └──── retry if nothing survives screening (≤2)

analyze      deterministic evidence pack: frontier, surrogate gene importance per objective, weakest objective,
             past hypotheses + verdicts, failure clusters from vector memory, pruning reasons, search space
hypothesize  Claude writes ONE falsifiable hypothesis about 1–3 genes and concrete edits to frontier genomes
             (offline: a surrogate-guided heuristic writes the same structure, marked source="heuristic")
materialize  apply edits, repair to the search space, dedupe, add surrogate-guided mutation/crossover explorers
screen       random-forest surrogate ranks candidates by optimistic hypervolume gain × P(feasible)

After the batch is benchmarked, `observe` issues a verdict: confirmed / refuted / inconclusive, measured against
each proposal's parent genome — the result is written back to memory and fed into the next hypothesis.
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any, Literal, TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from epoch.agents import llm
from epoch.config import settings
from epoch.evolution.pareto import non_dominated
from epoch.evolution.surrogate import Surrogate
from epoch.genome import Genome, genome_id

if TYPE_CHECKING:
    from epoch.evolution.engine import Evolution, Proposal, Record

NOISE = 0.01  # normalised-objective change treated as measurement noise


class Expectation(BaseModel):
    objective: str
    direction: Literal["improve", "worsen", "unchanged"]


class Edit(BaseModel):
    base: str = Field(description="genome_id of a frontier genome to modify")
    changes: dict[str, Any] = Field(description="gene -> new value; must be inside the gene's domain")
    rationale: str = ""


class HypothesisOut(BaseModel):
    statement: str = Field(description="One falsifiable sentence: changing X will move Y because Z")
    mechanism: str = Field(description="Systems-level mechanism behind the expected effect")
    target_objective: str
    genes: list[str]
    expected: list[Expectation]
    edits: list[Edit] = Field(description="2-4 concrete edits of frontier genomes that test the hypothesis")


SYSTEM = """You are the experiment-design agent inside EPOCH, an autonomous AI-systems optimiser.
You receive the current Pareto frontier of a multi-objective search over pipeline configurations ("genomes"),
surrogate-estimated gene importance, past hypotheses with their verdicts, failure clusters and the search space.

Write ONE falsifiable engineering hypothesis about how 1-3 genes causally affect the objectives, then 2-4 edits
of listed frontier genomes that test it. Rules:
- Only use gene names and values from the search space; respect `when` conditions (a conditional gene only
  exists when its parent gene has one of the listed values).
- Target the weakest objective unless a refuted hypothesis already covered it; never repeat a refuted idea.
- Prefer edits that could push the frontier outward, not just re-measure known points.
- Avoid configurations resembling the failure clusters.
- State expected side effects on the other objectives honestly."""


class AgentState(TypedDict, total=False):
    k: int
    generation: int
    attempts: int
    context: dict
    hypothesis: dict
    source: str
    candidates: list[dict]
    screened: list[dict]


class HypothesisAgent:
    def __init__(self, evo: Evolution):
        self.evo = evo
        self.space = evo.space
        self.objs = evo.objs
        self.rng = np.random.default_rng(evo.cfg.seed + 101)
        self.surrogate = Surrogate(self.space, self.objs, seed=evo.cfg.seed)
        self.pending: dict[str, dict] = {}  # hypothesis_id -> {"proposals": {trial genome_id: parent_record}, "results": []}
        self.history: list[dict] = []
        self.n = 0
        g = StateGraph(AgentState)
        g.add_node("analyze", self._analyze)
        g.add_node("hypothesize", self._hypothesize)
        g.add_node("materialize", self._materialize)
        g.add_node("screen", self._screen)
        g.add_edge(START, "analyze")
        g.add_edge("analyze", "hypothesize")
        g.add_edge("hypothesize", "materialize")
        g.add_edge("materialize", "screen")
        g.add_conditional_edges("screen", lambda s: "hypothesize" if not s["screened"] and s["attempts"] < 2 else END)
        self.graph = g.compile()

    # ------------------------------------------------------------------ public
    def propose(self, k: int, generation: int) -> list[Proposal]:
        from epoch.evolution.engine import Proposal

        out = self.graph.invoke({"k": k, "generation": generation, "attempts": 0})
        return [Proposal(c["genome"], c["parents"], c.get("hypothesis_id"), c["note"]) for c in out.get("screened", [])]

    def observe(self, rec: Record) -> None:
        h = self.pending.get(rec.hypothesis_id or "")
        if h is None:
            return
        h["results"].append(rec)
        if len(h["results"]) >= h["expected_n"]:
            self._verdict(rec.hypothesis_id, h)  # type: ignore[arg-type]

    # ------------------------------------------------------------------ graph nodes
    def _complete(self) -> list[Record]:
        return [r for r in self.evo.records if r.result.status == "complete"]

    def _analyze(self, state: AgentState) -> AgentState:
        done = self._complete()
        from epoch.evolution.pareto import normalize

        norm = normalize(np.array([[r.result.metrics[o.name] for o in self.objs] for r in done]), self.objs)
        feas = np.array([r.feasible for r in done])
        self.surrogate.fit([r.genome for r in done], norm, feas)
        front, fnorm = self.evo.frontier()
        if not front:  # nothing feasible yet: treat least-violating completed trials as the frontier
            order = np.argsort([max(r.violations) for r in done])[:6]
            front, fnorm = [done[i] for i in order], norm[order]
        order = np.argsort(np.linalg.norm(fnorm, axis=1))[:8]
        front, fnorm = [front[i] for i in order], fnorm[order]
        gaps = fnorm.min(axis=0)  # best achieved per objective (0 = ideal)
        last = self.history[-1] if self.history else None
        weak = [int(i) for i in np.argsort(-gaps)]
        if last and last["verdict"] == "refuted" and len(weak) > 1 and self.objs[weak[0]].name == last["target"]:
            weak = weak[1:] + weak[:1]
        imp = self.surrogate.gene_importance() if self.surrogate.fitted else {}
        importance = {o.name: dict(sorted(((g, round(v[j], 3)) for g, v in imp.items()), key=lambda t: -t[1])[:6])
                      for j, o in enumerate(self.objs)} if imp else {}
        failures: dict[str, int] = {}
        for r in self.evo.records:
            if r.result.status == "failed":
                failures[r.result.error_kind or "runtime"] = failures.get(r.result.error_kind or "runtime", 0) + 1
        similar = self.evo.store.similar_failures("out of memory oom failed", k=3, workload=self.evo.w.name) if failures else []
        ctx = {
            "objectives": [{"name": o.name, "direction": o.direction, "label": o.label, "unit": o.unit} for o in self.objs],
            "constraints": [c.to_json() for c in self.evo.cons],
            "frontier": [{"genome_id": r.gid, "trial": r.number, "genome": r.genome,
                          "metrics": {o.name: round(r.result.metrics[o.name], 5) for o in self.objs},
                          "normalized_gap": [round(float(x), 3) for x in n]} for r, n in zip(front, fnorm, strict=True)],
            "weakest_objective": self.objs[weak[0]].name,
            "gene_importance": importance,
            "past_hypotheses": [{k: h[k] for k in ("statement", "target", "genes", "verdict")} for h in self.history[-6:]],
            "failure_counts": failures,
            "failure_examples": [f["message"][-200:] for f in similar],
            "pruned": sum(r.result.status == "pruned" for r in self.evo.records),
            "search_space": [{k: v for k, v in g.to_json().items() if k in ("name", "kind", "choices", "low", "high", "when", "doc")}
                             for g in self.space.genes],
        }
        return {"context": ctx, "attempts": state.get("attempts", 0)}

    def _hypothesize(self, state: AgentState) -> AgentState:
        ctx = state["context"]
        attempts = state.get("attempts", 0) + 1
        if settings().llm_enabled:
            try:
                r = llm.structured(model=settings().agent_model, system=SYSTEM,
                                   user="Current evidence (JSON):\n" + json.dumps(ctx, default=str),
                                   schema=llm.schema_of(HypothesisOut), tool_name="propose_hypothesis",
                                   max_tokens=1500, temperature=0.4 if attempts > 1 else 0.2, sample_id=attempts)
                h = HypothesisOut.model_validate(r.data).model_dump()
                h["_cost_usd"], h["_model"] = r.cost_usd, r.model
                return {"hypothesis": h, "source": "llm", "attempts": attempts}
            except Exception as exc:  # degrade gracefully; the search must never stall on the API
                self.evo.sink("agent.error", {"error": str(exc)[:300]})
        return {"hypothesis": self._heuristic(ctx, attempts), "source": "heuristic", "attempts": attempts}

    def _heuristic(self, ctx: dict, attempts: int) -> dict:
        """Surrogate-guided fallback that fills the same schema as the LLM."""
        target_name = ctx["weakest_objective"]
        tj = [o.name for o in self.objs].index(target_name)
        refuted = {g for h in self.history if h["verdict"] == "refuted" for g in h["genes"]}
        ranked = list(ctx["gene_importance"].get(target_name, {}).keys()) or [g.name for g in self.space.genes]
        ranked = [g for g in ranked if g not in refuted] or ranked
        gene = self.space.by_name[ranked[min(attempts - 1, len(ranked) - 1)]]
        edits, predicted = [], []
        for f in ctx["frontier"][:3]:
            base = f["genome"]
            if gene.name not in base:
                continue
            options = list(gene.choices) if gene.kind == "cat" else [gene.from_unit(u) for u in (0.0, 0.25, 0.5, 0.75, 1.0)]
            options = [v for v in options if v != base[gene.name]]
            if not options:
                continue
            cands = [self.space.repair({**base, gene.name: v}, base) for v in options]
            if self.surrogate.fitted:
                mu, _, feas = self.surrogate.predict(cands)
                score = mu[:, tj] + 0.3 * (1 - feas)
                best = int(np.argmin(score))
                predicted.append(mu[best] - self.surrogate.predict([base])[0][0])
            else:
                best = int(self.rng.integers(len(cands)))
            edits.append({"base": f["genome_id"], "changes": {gene.name: cands[best][gene.name]},
                          "rationale": f"surrogate-best value of {gene.name} for this frontier point"})
        side = []
        if predicted:
            d = np.mean(predicted, axis=0)
            for j, o in enumerate(self.objs):
                if j != tj:
                    side.append({"objective": o.name,
                                 "direction": "worsen" if d[j] > NOISE else "improve" if d[j] < -NOISE else "unchanged"})
        label = self.objs[tj].label
        verb = "raise" if self.objs[tj].direction == "max" else "cut"
        return {
            "statement": f"`{gene.name}` is the dominant lever on {label}: re-setting it on frontier configs will {verb} {label}.",
            "mechanism": gene.doc or f"Surrogate importance ranks {gene.name} first for {label}.",
            "target_objective": target_name,
            "genes": [gene.name],
            "expected": [{"objective": target_name, "direction": "improve"}, *side],
            "edits": edits,
        }

    def _materialize(self, state: AgentState) -> AgentState:
        h = state["hypothesis"]
        ctx = state["context"]
        by_gid = {f["genome_id"]: f for f in ctx["frontier"]}
        rec_by_gid = {r.gid: r for r in self.evo.records}
        seen = set(self.evo.cache) | {r.gid for r in self.evo.records}
        cands: list[dict] = []
        for e in h.get("edits", []):
            base = by_gid.get(e.get("base")) or ctx["frontier"][0]
            child = self.space.repair({**base["genome"], **(e.get("changes") or {})}, base["genome"])
            gid = genome_id(child)
            if gid in seen or self.space.validate(child):
                continue
            seen.add(gid)
            parent = rec_by_gid.get(base["genome_id"])
            cands.append({"genome": child, "gid": gid, "parents": [parent.trial_id] if parent else [], "note": "hypothesis",
                          "parent_gid": base["genome_id"]})
        front = [f["genome"] for f in ctx["frontier"]]
        fr_recs = [rec_by_gid[f["genome_id"]] for f in ctx["frontier"] if f["genome_id"] in rec_by_gid]
        for _ in range(state["k"] * 10):  # exploration candidates, surrogate decides which are worth a run
            if self.rng.random() < 0.2:  # global samples keep the surrogate honest outside the frontier's basin
                child = self.space.sample(self.rng)
                parents, note = [], "explore"
            elif len(front) >= 2 and self.rng.random() < 0.35:
                i, j = self.rng.choice(len(front), 2, replace=False)
                child = self.space.crossover(front[i], front[j], self.rng)
                parents, note = [fr_recs[i].trial_id, fr_recs[j].trial_id] if len(fr_recs) > max(i, j) else [], "crossover"
            else:
                i = int(self.rng.integers(len(front)))
                child = self.space.mutate(front[i], self.rng, n=int(self.rng.integers(1, 4)))
                parents, note = ([fr_recs[i].trial_id] if len(fr_recs) > i else []), "mutation"
            gid = genome_id(child)
            if gid in seen:
                continue
            seen.add(gid)
            cands.append({"genome": child, "gid": gid, "parents": parents, "note": note})
        return {"candidates": cands}

    def _screen(self, state: AgentState) -> AgentState:
        k = state["k"]
        cands = state["candidates"]
        _, fnorm = self.evo.frontier()
        hyp = [c for c in cands if c["note"] == "hypothesis"]
        exp = [c for c in cands if c["note"] != "hypothesis"]
        n_h = min(len(hyp), max(2, math.ceil(k / 2)))
        by_gid = {c["gid"]: c for c in cands}
        chosen_h = self.surrogate.screen([c["genome"] for c in hyp], fnorm, n_h)
        chosen_e = [s for s in self.surrogate.screen([c["genome"] for c in exp], fnorm, len(exp))
                    if s.feasible_p >= 0.2][: k - len(chosen_h)]
        screened = []
        for s in [*chosen_h, *chosen_e]:
            c = dict(by_gid[genome_id(s.genome)])
            c["predicted"] = None if np.isnan(s.mean).any() else [round(float(x), 4) for x in s.mean]
            c["ehvi"] = round(s.ehvi, 5)
            screened.append(c)
        if screened and any(c["note"] == "hypothesis" for c in screened):
            self._register(state, screened)
        return {"screened": screened}

    # ------------------------------------------------------------------ bookkeeping
    def _register(self, state: AgentState, screened: list[dict]) -> None:
        self.n += 1
        h = state["hypothesis"]
        hid = f"H{self.evo.run_id[-6:]}-{self.n:02d}"
        tests = [c for c in screened if c["note"] == "hypothesis"]
        for c in tests:
            c["hypothesis_id"] = hid
        self.pending[hid] = {"expected_n": len(tests), "results": [], "parents": {c["gid"]: c.get("parent_gid") for c in tests},
                             "target": h["target_objective"], "expected": h["expected"],
                             "evidence0": {"generation": state.get("generation"), "llm_cost_usd": h.get("_cost_usd"),
                                           "model": h.get("_model")}}
        record = {
            "id": hid, "run_id": self.evo.run_id, "source": state.get("source", "heuristic"),
            "statement": h["statement"], "mechanism": h.get("mechanism", ""), "target": h["target_objective"],
            "genes": h.get("genes", []), "expected": {e["objective"]: e["direction"] for e in h.get("expected", [])},
            "proposals": [{"genome_id": c["gid"], "parent": c.get("parent_gid"), "changes": self.space.diff(
                self._genome_of(c.get("parent_gid")) or {}, c["genome"]), "predicted": c.get("predicted"), "ehvi": c.get("ehvi")}
                for c in tests],
            "verdict": "pending",
            "evidence": {"generation": state.get("generation"), "llm_cost_usd": h.get("_cost_usd"), "model": h.get("_model")},
        }
        for p in record["proposals"]:
            p["changes"] = {k: list(v) for k, v in p["changes"].items()}
        self.evo.store.add_hypothesis(**record)
        self.history.append({"id": hid, "statement": h["statement"], "target": h["target_objective"],
                             "genes": h.get("genes", []), "verdict": "pending"})
        self.evo.sink("hypothesis", {"id": hid, "statement": h["statement"], "source": record["source"]})

    def _genome_of(self, gid: str | None) -> Genome | None:
        for r in self.evo.records:
            if r.gid == gid:
                return r.genome
        return None

    def _verdict(self, hid: str, h: dict) -> None:
        from epoch.evolution.pareto import normalize

        names = [o.name for o in self.objs]
        rec_by_gid = {r.gid: r for r in self.evo.records}
        rows, votes = [], []
        tj = names.index(h["target"]) if h["target"] in names else 0
        for r in h["results"]:
            parent = rec_by_gid.get(h["parents"].get(r.gid))
            if r.result.status != "complete" or parent is None or parent.result.status != "complete":
                rows.append({"genome_id": r.gid, "status": r.result.status})
                votes.append(-1 if r.result.status in ("failed", "pruned") else 0)
                continue
            a = normalize(np.array([[parent.result.metrics[n] for n in names]]), self.objs)[0]
            b = normalize(np.array([[r.result.metrics[n] for n in names]]), self.objs)[0]
            delta = b - a  # negative = better
            rows.append({"genome_id": r.gid, "parent": parent.gid, "status": "complete", "feasible": r.feasible,
                         "delta": {n: round(float(d), 4) for n, d in zip(names, delta, strict=True)},
                         "raw": {n: [parent.result.metrics[n], r.result.metrics[n]] for n in names}})
            votes.append(1 if delta[tj] < -NOISE else -1 if delta[tj] > NOISE else 0)
        score = float(np.mean(votes)) if votes else 0.0
        verdict = "confirmed" if score > 0.34 else "refuted" if score < -0.34 else "inconclusive"
        front, _ = self.evo.frontier()
        joined = sum(1 for r in h["results"] if any(r.trial_id == f.trial_id for f in front))
        evidence = {"tests": rows, "support": score, "joined_frontier": joined}
        self.evo.store.update_hypothesis(hid, verdict=verdict, evidence={**h.get("evidence0", {}), **evidence})
        for x in self.history:
            if x["id"] == hid:
                x["verdict"] = verdict
        self.pending.pop(hid, None)
        self.evo.sink("hypothesis.verdict", {"id": hid, "verdict": verdict, "support": score})


def front_mask(norm: np.ndarray) -> np.ndarray:
    return non_dominated(norm) if norm.size else np.zeros(0, dtype=bool)
