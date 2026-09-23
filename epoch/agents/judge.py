"""Rubric-as-code LLM judge (from the CSAI project, now EPOCH's quality objective for generative workloads).

The rubric is versioned YAML with written anchors per level. The model only fills a schema — one level per
criterion plus the evidence quote; code does the weighting, hard gates and pass/fail. Self-consistency sampling
gives a per-criterion σ (anchor-quality signal) and `calibrate` reports Cohen's κ against human labels, so a
rubric version is only trusted once it agrees with people."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from epoch.agents import llm
from epoch.benchmark.metrics import cohen_kappa
from epoch.config import CONFIG_DIR, settings


@dataclass(frozen=True)
class Rubric:
    id: str
    version: str
    pass_threshold: float
    criteria: tuple[dict, ...]

    @property
    def key(self) -> str:
        return f"{self.id}/{self.version}"


@lru_cache
def load_rubric(name: str = "rag-answer-1.0.0") -> Rubric:
    d = yaml.safe_load(Path(CONFIG_DIR / "rubrics" / f"{name}.yaml").read_text())
    return Rubric(d["id"], d["version"], d["pass_threshold"], tuple(d["criteria"]))


def _schema(r: Rubric) -> dict[str, Any]:
    props = {}
    for c in r.criteria:
        props[c["name"]] = {"type": "object", "properties": {
            "level": {"type": "integer", "minimum": 1, "maximum": 4},
            "evidence": {"type": "string", "description": "short quote from the answer/context justifying the level"}},
            "required": ["level", "evidence"]}
    return {"type": "object", "properties": props, "required": [c["name"] for c in r.criteria]}


def _system(r: Rubric) -> str:
    lines = [f"You grade answers with rubric {r.key}. For each criterion pick the level whose anchor fits best."]
    for c in r.criteria:
        lines.append(f"\n{c['name']}:")
        lines += [f"  {lvl}: {txt}" for lvl, txt in sorted(c["anchors"].items())]
    return "\n".join(lines)


def aggregate(r: Rubric, levels: dict[str, int]) -> dict[str, Any]:
    score = sum(c["weight"] * (levels[c["name"]] - 1) / 3 for c in r.criteria)
    gated = [c["name"] for c in r.criteria if c.get("hard_gate") and levels[c["name"]] <= 2]
    return {"score": 0.0 if gated else score, "raw_score": score, "gated": gated,
            "passed": not gated and score >= r.pass_threshold}


def judge(question: str, context: str, answer: str, *, rubric: Rubric | None = None, samples: int = 1) -> dict[str, Any]:
    r = rubric or load_rubric()
    if not settings().llm_enabled:
        raise llm.LLMUnavailable("judge needs ANTHROPIC_API_KEY")
    runs = []
    for i in range(samples):
        res = llm.structured(model=settings().judge_model, system=_system(r), schema=_schema(r), tool_name="grade",
                             user=f"Question: {question}\n\nContext:\n{context}\n\nAnswer: {answer}",
                             temperature=0.0 if samples == 1 else 0.7, sample_id=i, max_tokens=600)
        runs.append(({k: int(v["level"]) for k, v in res.data.items()}, res))
    levels = {c["name"]: round(np.mean([lv[c["name"]] for lv, _ in runs])) for c in r.criteria}
    sigma = {c["name"]: float(np.std([lv[c["name"]] for lv, _ in runs])) for c in r.criteria}
    return {"rubric": r.key, "levels": levels, "sigma": sigma, **aggregate(r, levels),
            "evidence": {k: v["evidence"] for k, v in runs[0][1].data.items()},
            "cost_usd": sum(res.cost_usd for _, res in runs)}


def calibrate(items: list[dict], rubric: Rubric | None = None) -> dict[str, Any]:
    """items: [{question, context, answer, human: {criterion: level}}] → per-criterion Cohen's κ."""
    r = rubric or load_rubric()
    judged = [judge(i["question"], i["context"], i["answer"], rubric=r) for i in items]
    out = {}
    for c in r.criteria:
        h = [i["human"][c["name"]] for i in items]
        m = [j["levels"][c["name"]] for j in judged]
        out[c["name"]] = {"kappa": cohen_kappa(h, m), "agreement": float(np.mean([a == b for a, b in zip(h, m, strict=True)]))}
    return {"rubric": r.key, "n": len(items), "criteria": out,
            "trusted": all(v["kappa"] >= 0.6 for v in out.values())}
