"""RAG QA pipeline as an EPOCH workload. The search space is hardware-aware: genes that need a GPU,
bitsandbytes, sentence-transformers or ONNX Runtime GPU only appear when those are available."""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Any

import numpy as np

from epoch.benchmark.metrics import contains_answer, exact_match, normalize_answer, token_f1
from epoch.benchmark.resources import device_info
from epoch.genome import Gene, Genome, SearchSpace
from epoch.objectives import Constraint, Objective
from epoch.workloads.base import Pipeline, Prediction, Workload, register
from epoch.workloads.rag import corpus as C
from epoch.workloads.rag.generation import MODELS, Extractive, HFGenerator, build_context
from epoch.workloads.rag.retrieval import EMBEDDERS, RERANKERS, Reranker, Retriever


@lru_cache
def _has(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:
        return False


class RagPipeline(Pipeline):
    def __init__(self, g: Genome, device: str):
        self.g = g
        docs, _ = C.build()
        self.retriever = Retriever(docs, size=g["chunk_size"], overlap=g["chunk_overlap"], kind=g["retriever"],
                                   embedder=g.get("embedder"), device=device)
        self.reranker = Reranker(g["rerank_model"], device) if g.get("reranker") else None
        gen = g["generator"]
        self.extractive = Extractive() if gen == "extractive" else None
        self.hf = None if gen == "extractive" else HFGenerator(
            gen, g.get("quant", "fp16"), g.get("runtime", "eager"), g.get("max_new_tokens", 24), g.get("prompt", "concise"), device)
        self.budget = g.get("context_budget", 400)
        self.inject_ms: dict[str, float] = {}  # fault-injection hook used by healing scenarios

    def predict_batch(self, xs: list[str]) -> list[Prediction]:
        n = len(xs)
        t0 = time.perf_counter()
        if self.inject_ms.get("retrieve"):
            time.sleep(self.inject_ms["retrieve"] / 1e3)
        hits = self.retriever.search(xs, self.g["top_k"])
        t_ret = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        if self.reranker:
            hits = [self.reranker.rerank(q, h, self.g["rerank_keep"]) for q, h in zip(xs, hits, strict=True)]
        t_rr = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        if self.inject_ms.get("generate"):
            time.sleep(self.inject_ms["generate"] / 1e3)
        if self.extractive:
            answers = [self.extractive.answer(q, h, self.budget) for q, h in zip(xs, hits, strict=True)]
        else:
            ctxs = [build_context(h, self.budget) for h in hits]
            answers = self.hf.answer_batch(xs, ctxs)  # type: ignore[union-attr]
        t_gen = (time.perf_counter() - t0) * 1e3
        stages = {"retrieve": t_ret / n, "rerank": t_rr / n, "generate": t_gen / n}
        return [Prediction(a, stages_ms=stages, meta={"question": q, "chunks": [c.text for c in h]})
                for q, a, h in zip(xs, answers, hits, strict=True)]


@register
class RagWorkload(Workload):
    name = "rag"
    title = "RAG QA pipeline"
    description = "Answer home-loan policy questions from a 30-document corpus: chunk → retrieve → rerank → generate."
    quality_metric = "answer_f1"

    def __init__(self) -> None:
        import os

        from epoch.config import settings

        info = device_info()
        self.device = info["device"]
        vram = (info.get("gpu_mem_gb") or 0) * 1024
        # EPOCH_RAG_JUDGE=1 makes the rubric-as-code LLM judge the quality objective (needs ANTHROPIC_API_KEY)
        self.use_judge = os.environ.get("EPOCH_RAG_JUDGE") == "1" and settings().llm_enabled
        if self.use_judge:
            self.quality_metric = "judge_score"
        self.objectives = [
            Objective(self.quality_metric, "max", "Judge score" if self.use_judge else "Answer F1", lo=0.2, hi=1.0),
            Objective("p95_ms", "min", "p95 latency", "ms", lo=1, hi=30000, log=True),
            Objective("peak_mem_mb", "min", "Peak memory", "MB", lo=1, hi=24000, log=True),
            Objective("cost_per_1k", "min", "Cost / 1k req", "$", lo=1e-5, hi=10, log=True),
        ]
        self.constraints = [
            Constraint(self.quality_metric, ">=", 0.6, "quality floor"),
            Constraint("peak_mem_mb", "<=", vram * 0.9 if vram else 4000.0, "fits device memory"),
        ]

    def capabilities(self) -> dict[str, Any]:
        info = device_info()
        return {
            "device": self.device,
            "dense_retrieval": _has("sentence_transformers") and _has("faiss"),
            "hf_generation": _has("transformers") and _has("torch"),
            "bnb": info["capabilities"]["bnb"],
            "torch_compile": info["capabilities"]["torch_compile"],
            "ort_generation": _has("optimum.onnxruntime"),
        }

    def space(self) -> SearchSpace:
        cap = self.capabilities()
        retrievers = ("bm25", "dense", "hybrid") if cap["dense_retrieval"] else ("bm25",)
        gens: tuple = ("extractive",)
        if cap["hf_generation"]:
            models = tuple(MODELS) if self.device == "cuda" else ("qwen0.5b",)
            gens = (*models, "extractive")
        runtimes = ["eager"]
        if cap["torch_compile"]:
            runtimes.append("compile")
        if cap["ort_generation"]:
            runtimes.append("ort")
        hf = tuple(g for g in gens if g != "extractive")
        genes = [
            Gene("chunk_size", "int", low=40, high=320, step=20, default=120, group="retrieval",
                 doc="Words per chunk; small chunks sharpen retrieval, large ones carry more context."),
            Gene("chunk_overlap", "cat", (0.0, 0.15, 0.3), default=0.15, group="retrieval", doc="Chunk overlap fraction."),
            Gene("retriever", "cat", retrievers, default=retrievers[0], group="retrieval", doc="Sparse, dense or RRF hybrid."),
        ]
        if cap["dense_retrieval"]:
            genes.append(Gene("embedder", "cat", tuple(EMBEDDERS), default="bge-small", group="retrieval",
                              when=("retriever", ("dense", "hybrid")), doc="Bi-encoder for dense retrieval."))
        genes.append(Gene("top_k", "int", low=1, high=10, default=4, group="retrieval", doc="Chunks retrieved."))
        if cap["dense_retrieval"]:
            genes += [
                Gene("reranker", "cat", (False, True), default=False, group="retrieval", doc="Cross-encoder rerank."),
                Gene("rerank_model", "cat", tuple(RERANKERS), default="minilm-l6", group="retrieval",
                     when=("reranker", (True,)), doc="Cross-encoder checkpoint."),
                Gene("rerank_keep", "int", low=1, high=5, default=3, group="retrieval", when=("reranker", (True,)),
                     doc="Chunks kept after reranking."),
            ]
        genes += [
            Gene("generator", "cat", gens, default=gens[0], group="model", doc="Answer generator."),
            Gene("context_budget", "int", low=80, high=1200, log=True, default=400, group="model",
                 doc="Max context words stuffed into the prompt."),
        ]
        if hf:
            genes += [
                Gene("runtime", "cat", tuple(runtimes), default="eager", group="runtime", when=("generator", hf),
                     doc="eager PyTorch, torch.compile (static KV cache) or ONNX Runtime."),
                Gene("quant", "cat", ("fp16", "int8", "nf4") if cap["bnb"] else ("fp16",), default="fp16",
                     group="runtime", when=("runtime", ("eager",)),
                     doc="Weight quantisation (bitsandbytes); compile/ORT paths run fp16/fp32."),
                Gene("max_new_tokens", "cat", (12, 24, 48), default=24, group="model", when=("generator", hf),
                     doc="Decode budget."),
                Gene("prompt", "cat", ("concise", "cite"), default="concise", group="model", when=("generator", hf),
                     doc="Answer prompt template."),
            ]
        genes.append(Gene("batch_size", "cat", (1, 4, 8, 16), default=4, group="runtime", doc="Request batch size."))
        return SearchSpace(genes)

    def prepare(self, seed: int = 0) -> None:
        C.build()

    def build(self, genome: Genome) -> Pipeline:
        return RagPipeline(genome, self.device)

    def dataset(self, split: str, fidelity: float = 1.0, seed: int = 0):
        rows = C.build()[1][split]
        if fidelity < 1.0 and rows:
            idx = np.random.default_rng(seed).permutation(len(rows))[: max(16, int(len(rows) * fidelity))]
            rows = [rows[i] for i in sorted(idx)]
        return [r.question for r in rows], [r for r in rows]

    def score(self, preds: list[Prediction], golds: list[C.QA]) -> dict[str, float]:
        f1 = [token_f1(p.output, g.answer) for p, g in zip(preds, golds, strict=True)]
        em = [exact_match(p.output, g.answer) for p, g in zip(preds, golds, strict=True)]
        ca = [contains_answer(p.output, g.answer) for p, g in zip(preds, golds, strict=True)]
        rec = [any(normalize_answer(g.fact) in normalize_answer(c) for c in p.meta.get("chunks", []))
               for p, g in zip(preds, golds, strict=True)]
        out = {"answer_f1": float(np.mean(f1)), "exact_match": float(np.mean(em)),
               "contains": float(np.mean(ca)), "retrieval_recall": float(np.mean(rec))}
        if self.use_judge:
            from epoch.agents.judge import judge

            graded = [judge(p.meta["question"], "\n".join(p.meta["chunks"]), str(p.output)) for p in preds[:24]]
            out["judge_score"] = float(np.mean([g["score"] for g in graded]))
            out["judge_pass_rate"] = float(np.mean([g["passed"] for g in graded]))
        return out

    def topology(self, genome: Genome, metrics: dict[str, float] | None = None) -> dict[str, Any]:
        gen = genome.get("generator", "extractive")
        gsub = gen if gen == "extractive" else f"{gen} · {genome.get('quant', 'fp16')} · {genome.get('runtime', 'eager')}"
        nodes = [
            {"id": "ingress", "label": "Question", "kind": "io", "sub": "API · chat"},
            {"id": "retrieve", "label": "Retriever", "kind": "model", "stage": "retrieve",
             "sub": f"{genome.get('retriever')}{' · ' + genome['embedder'] if genome.get('embedder') else ''} · k={genome.get('top_k')}"},
            {"id": "index", "label": "Chunk index", "kind": "store", "sub": f"{genome.get('chunk_size')}w · {genome.get('chunk_overlap')} overlap"},
        ]
        edges = [{"source": "ingress", "target": "retrieve", "share": 1.0}, {"source": "index", "target": "retrieve", "share": 1.0}]
        last = "retrieve"
        if genome.get("reranker"):
            nodes.append({"id": "rerank", "label": "Reranker", "kind": "model", "stage": "rerank",
                          "sub": f"{genome.get('rerank_model')} keep {genome.get('rerank_keep')}"})
            edges.append({"source": "retrieve", "target": "rerank", "share": 1.0})
            last = "rerank"
        nodes += [
            {"id": "prompt", "label": "Prompt builder", "kind": "transform", "sub": f"budget {genome.get('context_budget')}w"},
            {"id": "generate", "label": "Generator", "kind": "llm" if gen != "extractive" else "model", "stage": "generate", "sub": gsub},
            {"id": "egress", "label": "Answer", "kind": "io", "sub": f"batch {genome.get('batch_size')}"},
        ]
        edges += [{"source": last, "target": "prompt", "share": 1.0}, {"source": "prompt", "target": "generate", "share": 1.0},
                  {"source": "generate", "target": "egress", "share": 1.0}]
        return {"nodes": nodes, "edges": edges}
