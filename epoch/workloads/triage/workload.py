"""CSAI triage cascade as an EPOCH workload.

Tier 1  fuzzy gate      rapidfuzz scores vs queue aliases; accept if confidence >= tau1 and margin >= delta1
Tier 2  text classifier TF-IDF + {logreg | complement NB | LSA->MLP}; LSA->MLP can run on sklearn, ONNX Runtime
                         or int8-quantised ONNX Runtime; accept if p_max >= tau2
Tier 3  resolver        Claude re-ranks the tier-2 shortlist (k queues, error-bank few-shots) — or, when no API key
                         is configured, a local kNN resolver. The search space adapts to what is available.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from rapidfuzz import fuzz, process
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion, make_pipeline

from epoch.benchmark.metrics import macro_f1
from epoch.config import settings
from epoch.genome import Gene, Genome, SearchSpace
from epoch.objectives import Constraint, Objective
from epoch.workloads.base import Pipeline, Prediction, Workload, register
from epoch.workloads.triage import data as D

SCORERS = {"wratio": fuzz.WRatio, "token_set": fuzz.token_set_ratio, "partial": fuzz.partial_ratio}
_FIT_LOCK = threading.Lock()
_T2_CACHE: dict[tuple, Any] = {}
_ORT_CACHE: dict[tuple, Any] = {}
_KNN_CACHE: dict[str, Any] = {}


def _bank_path(tag: str):
    return settings().home / "banks" / f"{tag}.json"


def save_bank(tag: str, rows: list[tuple[str, str]]) -> None:
    import json

    p = _bank_path(tag)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rows))


def load_bank(tag: str) -> list[tuple[str, str]]:
    import json

    if tag == "base":
        return []
    return [tuple(r) for r in json.loads(_bank_path(tag).read_text())]


def bank_tags() -> list[str]:
    d = settings().home / "banks"
    return sorted(p.stem for p in d.glob("*.json")) if d.exists() else []


# ---------------------------------------------------------------- normalisation
def normalize(text: str, mode: str) -> str:
    t = text.lower()
    if mode == "none":
        return t
    import re

    t = re.sub(r"\bhl\d+\b|\bp\d{4}\b|rs \d+ ?(lakh|k)?|\d+", " ", t)
    t = re.sub(r"[^a-z\s-]", " ", t)
    if mode == "hinglish":
        t = " ".join(D.HINGLISH_REVERSE.get(w, w) for w in t.split())
    return " ".join(t.split())


# ---------------------------------------------------------------- tier 1
class FuzzyGate:
    def __init__(self, scorer: str, norm: str):
        self.scorer = SCORERS[scorer]
        self.norm = norm
        self.aliases: list[str] = []
        owner: list[int] = []
        for qi, q in enumerate(D.LABELS):
            for a in D.QUEUES[q]["aliases"]:
                self.aliases.append(normalize(a, "basic"))
                owner.append(qi)
        self.owner = np.asarray(owner)
        self.starts = np.r_[0, np.flatnonzero(np.diff(self.owner)) + 1]

    def scores(self, texts: list[str]) -> np.ndarray:
        q = [normalize(t, self.norm) for t in texts]
        s = process.cdist(q, self.aliases, scorer=self.scorer, dtype=np.float32, workers=1)
        return np.maximum.reduceat(s, self.starts, axis=1) / 100.0


# ---------------------------------------------------------------- tier 2
def _vectorizer(kind: str):
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2, max_features=40000)
    if kind == "word":
        return word
    if kind == "char":
        return char
    return FeatureUnion([("w", word), ("c", char)])


def _fit_tier2(features: str, model: str, c: float | None, svd_dim: int | None, norm: str, data: str = "base"):
    key = (features, model, c, svd_dim, norm, data)
    with _FIT_LOCK:
        if key in _T2_CACHE:
            return _T2_CACHE[key]
        train = D.build_splits()["train"]
        extra = load_bank(data) * 3  # audited production tickets, up-weighted
        x = [normalize(t.text, norm) for t in train] + [normalize(t, norm) for t, _ in extra]
        y = [D.LABELS.index(t.label) for t in train] + [D.LABELS.index(lab) for _, lab in extra]
        vec = _vectorizer(features).fit(x)
        xv = vec.transform(x)
        if model == "logreg":
            clf = LogisticRegression(C=c or 1.0, max_iter=2000).fit(xv, y)
            head = clf
        elif model == "cnb":
            clf = ComplementNB(alpha=0.3).fit(xv, y)
            head = clf
        else:
            svd = TruncatedSVD(svd_dim or 128, random_state=0)
            mlp = MLPClassifier((256,), max_iter=300, random_state=0, early_stopping=True, n_iter_no_change=12)
            head = make_pipeline(svd, mlp).fit(xv, y)
        _T2_CACHE[key] = (vec, head, xv.shape[1])
        return _T2_CACHE[key]


def _ort_session(key: tuple, head, dim: int, quant: bool):
    ck = (*key, quant)
    with _FIT_LOCK:
        if ck in _ORT_CACHE:
            return _ORT_CACHE[ck]
        import onnxruntime as ort
        from skl2onnx import to_onnx
        from skl2onnx.common.data_types import FloatTensorType

        onx = to_onnx(head, initial_types=[("x", FloatTensorType([None, dim]))],
                      options={id(head.steps[-1][1]): {"zipmap": False}}, target_opset=17)
        d = tempfile.mkdtemp(prefix="epoch-ort-")
        path = os.path.join(d, "t2.onnx")
        with open(path, "wb") as f:
            f.write(onx.SerializeToString())
        if quant:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            qpath = os.path.join(d, "t2.int8.onnx")
            quantize_dynamic(path, qpath, weight_type=QuantType.QInt8)
            path = qpath
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        _ORT_CACHE[ck] = (sess, os.path.getsize(path))
        return _ORT_CACHE[ck]


class Tier2:
    def __init__(self, g: Genome):
        self.norm = g["normalize"]
        c = g.get("t2_C")
        svd = g.get("t2_svd_dim")
        key = (g["t2_features"], g["t2_model"], c, svd, self.norm, g.get("t2_data", "base"))
        self.vec, self.head, dim = _fit_tier2(*key)
        self.runtime = g.get("t2_runtime", "sklearn")
        self.sess = None
        self.model_bytes = 0
        if self.runtime in ("onnx", "onnx_int8"):
            self.sess, self.model_bytes = _ort_session(key, self.head, dim, quant=self.runtime == "onnx_int8")

    def proba(self, texts: list[str]) -> np.ndarray:
        xv = self.vec.transform([normalize(t, self.norm) for t in texts])
        if self.sess is not None:
            return np.asarray(self.sess.run(None, {"x": xv.toarray().astype(np.float32)})[1])
        return self.head.predict_proba(xv)


# ---------------------------------------------------------------- tier 3
def _knn_index():
    with _FIT_LOCK:
        if "idx" not in _KNN_CACHE:
            train = D.build_splits()["train"]
            vec = _vectorizer("word_char").fit([normalize(t.text, "hinglish") for t in train])
            xt = vec.transform([normalize(t.text, "hinglish") for t in train])
            _KNN_CACHE["idx"] = (vec, xt, np.asarray([D.LABELS.index(t.label) for t in train]))
        return _KNN_CACHE["idx"]


def _error_bank(n_per_class: int) -> dict[str, list[str]]:
    """Few-shot examples mined from tier-2 mistakes on the training split — the CSAI error bank."""
    if n_per_class <= 0:
        return {}
    key = f"bank{n_per_class}"
    with _FIT_LOCK:
        cached = _KNN_CACHE.get(key)
    if cached is not None:
        return cached
    vec, head, _ = _fit_tier2("word", "logreg", 1.0, None, "basic")
    train = D.build_splits()["train"]
    pred = head.predict(vec.transform([normalize(t.text, "basic") for t in train]))
    bank: dict[str, list[str]] = {q: [] for q in D.LABELS}
    for t, p in zip(train, pred, strict=True):
        if D.LABELS[p] != t.label and len(bank[t.label]) < n_per_class:
            bank[t.label].append(t.text)
    for t in train:  # top up with ordinary examples if a class had few mistakes
        if len(bank[t.label]) < n_per_class:
            bank[t.label].append(t.text)
    with _FIT_LOCK:
        _KNN_CACHE[key] = bank
    return bank


class Tier3:
    def __init__(self, g: Genome):
        self.policy = g["t3_policy"]
        self.k = int(g.get("t3_k", 3))
        if self.policy == "local":
            self.knn = int(g.get("t3_knn", 11))
            self.vec, self.xt, self.yt = _knn_index()
        elif self.policy == "llm":
            s = settings()
            self.model = s.triage_haiku_model if g.get("t3_model") == "haiku" else s.triage_sonnet_model
            self.shots = _error_bank(int(g.get("t3_shots", 0)))
            self.pool = ThreadPoolExecutor(max_workers=8)

    def resolve(self, texts: list[str], shortlists: list[list[int]]) -> list[tuple[int, float, float]]:
        """Returns (label_idx, confidence, cost_usd) per text."""
        if self.policy == "local":
            q = self.vec.transform([normalize(t, "hinglish") for t in texts])
            sims = (q @ self.xt.T).toarray()
            out = []
            for i, cands in enumerate(shortlists):
                top = np.argpartition(-sims[i], self.knn)[: self.knn]
                votes = Counter()
                for j in top:
                    if self.yt[j] in cands:
                        votes[int(self.yt[j])] += float(sims[i, j])
                if votes:
                    lab, w = votes.most_common(1)[0]
                    out.append((lab, w / (sum(votes.values()) or 1), 0.0))
                else:
                    out.append((cands[0], 0.0, 0.0))
            return out
        return list(self.pool.map(lambda a: self._llm(*a), zip(texts, shortlists, strict=True)))

    def _llm(self, text: str, cands: list[int]) -> tuple[int, float, float]:
        from epoch.agents import llm

        names = [D.LABELS[c] for c in cands]
        lines = [f"- {q}: {D.QUEUES[q]['desc']}" for q in names]
        shots = []
        for q in names:
            for ex in self.shots.get(q, []):
                shots.append(f'"{ex}" -> {q}')
        system = (
            "You route partner support tickets for a home-loan platform to exactly one operations queue.\n"
            "Pick the queue that owns the action the partner is asking for; ignore context that only mentions "
            "other stages.\nQueues:\n" + "\n".join(lines)
            + ("\nLabelled examples of past routing mistakes:\n" + "\n".join(shots) if shots else "")
        )
        schema = {
            "type": "object",
            "properties": {
                "queue": {"type": "string", "enum": names},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["queue", "confidence"],
        }
        r = llm.structured(model=self.model, system=system, user=f"Ticket: {text}", schema=schema,
                           tool_name="route_ticket", max_tokens=120)
        q = r.data.get("queue")
        lab = D.LABELS.index(q) if q in D.LABELS else cands[0]
        return lab, float(r.data.get("confidence", 0.5)), r.cost_usd

    def close(self) -> None:
        if self.policy == "llm":
            self.pool.shutdown(wait=False)


# ---------------------------------------------------------------- pipeline
class TriagePipeline(Pipeline):
    def __init__(self, g: Genome):
        self.g = g
        self.t1 = FuzzyGate(g["t1_scorer"], g["normalize"])
        self.t2 = Tier2(g)
        self.t3 = Tier3(g) if g["t3_policy"] != "off" else None
        self.tau1, self.delta1, self.tau2 = g["t1_tau"], g["t1_delta"], g["t2_tau"]
        self.inject_ms: dict[str, float] = {}  # fault-injection hook (stage -> added ms per call), used by healing

    def predict_batch(self, xs: list[str]) -> list[Prediction]:
        n = len(xs)
        preds: list[Prediction | None] = [None] * n
        stage = {"tier1": 0.0, "tier2": 0.0, "tier3": 0.0}

        t0 = time.perf_counter()
        s1 = self.t1.scores(xs)
        order = np.argsort(-s1, axis=1)
        top, sec = s1[np.arange(n), order[:, 0]], s1[np.arange(n), order[:, 1]]
        acc1 = (top >= self.tau1) & ((top - sec) >= self.delta1)
        stage["tier1"] = (time.perf_counter() - t0) * 1e3
        for i in np.flatnonzero(acc1):
            preds[i] = Prediction(D.LABELS[order[i, 0]], meta={"tier": 1, "conf": float(top[i])})

        rest = [i for i in range(n) if preds[i] is None]
        if rest:
            t0 = time.perf_counter()
            if self.inject_ms.get("tier2"):
                time.sleep(self.inject_ms["tier2"] / 1e3)
            p2 = self.t2.proba([xs[i] for i in rest])
            stage["tier2"] = (time.perf_counter() - t0) * 1e3
            esc: list[tuple[int, list[int]]] = []
            for j, i in enumerate(rest):
                pm = float(p2[j].max())
                if pm >= self.tau2 or self.t3 is None:
                    preds[i] = Prediction(D.LABELS[int(p2[j].argmax())], meta={"tier": 2, "conf": pm})
                else:
                    esc.append((i, [int(c) for c in np.argsort(-p2[j])[: self.t3.k]]))
            if esc:
                t0 = time.perf_counter()
                if self.inject_ms.get("tier3"):
                    time.sleep(self.inject_ms["tier3"] / 1e3)
                res = self.t3.resolve([xs[i] for i, _ in esc], [c for _, c in esc])  # type: ignore[union-attr]
                stage["tier3"] = (time.perf_counter() - t0) * 1e3
                for (i, _), (lab, conf, cost) in zip(esc, res, strict=True):
                    preds[i] = Prediction(D.LABELS[lab], cost_usd=cost, meta={"tier": 3, "conf": conf})
        per = {k: v / n for k, v in stage.items()}  # amortised per request
        for p in preds:
            p.stages_ms = per  # type: ignore[union-attr]
        return preds  # type: ignore[return-value]

    def close(self) -> None:
        if self.t3:
            self.t3.close()


@register
class TriageWorkload(Workload):
    name = "triage"
    title = "CSAI triage cascade"
    description = ("Route home-loan partner tickets to 6 ops queues through a 3-tier cost-aware cascade "
                   "(fuzzy gate → classifier → LLM/kNN resolver).")
    quality_metric = "macro_f1"

    def __init__(self, llm: bool | None = None):
        self.llm = settings().llm_enabled if llm is None else llm
        self.objectives = [
            Objective("macro_f1", "max", "Macro-F1", "", lo=0.5, hi=1.0),
            Objective("p95_ms", "min", "p95 latency", "ms", lo=0.05, hi=5000, log=True),
            Objective("cost_per_1k", "min", "Cost / 1k req", "$", lo=1e-5, hi=10, log=True),
        ]
        self.constraints = [
            Constraint("macro_f1", ">=", 0.80, "quality floor"),
            Constraint("p95_ms", "<=", 1500.0, "p95 SLO"),
        ]

    def capabilities(self) -> dict[str, Any]:
        return {"tier3_llm": self.llm, "tier3_backend": "claude" if self.llm else "local-knn", "onnxruntime": True}

    def space(self) -> SearchSpace:
        t3 = ("off", "local", "llm") if self.llm else ("off", "local")
        genes = [
            Gene("normalize", "cat", ("none", "basic", "hinglish"), default="basic", group="preprocess",
                 doc="Text normalisation before matching; `hinglish` maps code-mixed tokens to English."),
            Gene("t1_scorer", "cat", ("wratio", "token_set", "partial"), default="token_set", group="tier1",
                 doc="Fuzzy scorer against queue aliases."),
            Gene("t1_tau", "float", low=0.55, high=1.0, default=0.9, group="tier1",
                 doc="Tier-1 confidence floor; higher sends more tickets to slower, stronger tiers."),
            Gene("t1_delta", "float", low=0.0, high=0.4, default=0.1, group="tier1",
                 doc="Tier-1 margin over runner-up; guards against ambiguous tickets."),
            Gene("t2_features", "cat", ("word", "char", "word_char"), default="word_char", group="tier2",
                 doc="TF-IDF features; char n-grams resist typos and code-mixing."),
            Gene("t2_data", "cat", ("base", *bank_tags()), default="base", group="tier2",
                 doc="Tier-2 training data version: base set, or base + an audited production error bank."),
            Gene("t2_model", "cat", ("logreg", "cnb", "lsa_mlp"), default="logreg", group="tier2",
                 doc="Tier-2 classifier head."),
            Gene("t2_C", "cat", (0.1, 0.3, 1.0, 3.0, 10.0, 30.0), default=3.0, group="tier2",
                 when=("t2_model", ("logreg",)), doc="Inverse regularisation of logistic regression."),
            Gene("t2_svd_dim", "cat", (64, 128, 256), default=128, group="tier2",
                 when=("t2_model", ("lsa_mlp",)), doc="LSA projection width before the MLP."),
            Gene("t2_runtime", "cat", ("sklearn", "onnx", "onnx_int8"), default="sklearn", group="runtime",
                 when=("t2_model", ("lsa_mlp",)),
                 doc="Inference runtime for the dense head: sklearn, ONNX Runtime, or int8 dynamic-quantised ORT."),
            Gene("t2_tau", "float", low=0.3, high=0.99, default=0.8, group="tier2",
                 doc="Tier-2 acceptance threshold; below it tickets escalate to tier 3."),
            Gene("t3_policy", "cat", t3, default="local", group="tier3", doc="Tier-3 resolver."),
            Gene("t3_k", "int", low=2, high=6, default=3, group="tier3", when=("t3_policy", ("local", "llm")),
                 doc="Shortlist size handed to tier 3."),
            Gene("t3_knn", "int", low=3, high=31, step=2, default=11, group="tier3", when=("t3_policy", ("local",)),
                 doc="Neighbours in the local kNN resolver."),
        ]
        if self.llm:
            genes += [
                Gene("t3_model", "cat", ("haiku", "sonnet"), default="haiku", group="tier3",
                     when=("t3_policy", ("llm",)), doc="Claude model used for re-ranking."),
                Gene("t3_shots", "int", low=0, high=2, default=1, group="tier3", when=("t3_policy", ("llm",)),
                     doc="Error-bank few-shot examples per shortlisted queue."),
            ]
        genes.append(Gene("batch_size", "cat", (1, 4, 16, 64), default=16, group="runtime",
                          doc="Micro-batch size; amortises vectorisation at the cost of queueing latency."))
        return SearchSpace(genes)

    def prepare(self, seed: int = 0) -> None:
        D.build_splits()

    def build(self, genome: Genome) -> Pipeline:
        return TriagePipeline(genome)

    def adapt(self, genome: Genome, audited: list[tuple[Any, Any]], tag: str) -> Genome | None:
        save_bank(tag, [(str(x), str(y)) for x, y in audited])
        return {**genome, "t2_data": tag}

    def dataset(self, split: str, fidelity: float = 1.0, seed: int = 0) -> tuple[list[str], list[str]]:
        rows = D.build_splits()[split]
        if fidelity < 1.0:
            rng = np.random.default_rng(seed)
            idx = rng.permutation(len(rows))[: max(24, int(len(rows) * fidelity))]
            rows = [rows[i] for i in sorted(idx)]
        return [r.text for r in rows], [r.label for r in rows]

    def score(self, preds: list[Prediction], golds: list[str]) -> dict[str, float]:
        y = [p.output for p in preds]
        tiers = np.asarray([p.meta.get("tier", 0) for p in preds])
        correct = np.asarray([a == b for a, b in zip(y, golds, strict=True)])
        out = {"macro_f1": macro_f1(golds, y), "accuracy": float(correct.mean())}
        for t in (1, 2, 3):
            m = tiers == t
            out[f"share_t{t}"] = float(m.mean())
            out[f"precision_t{t}"] = float(correct[m].mean()) if m.any() else float("nan")
        out["escalation_rate"] = float((tiers == 3).mean())
        out["mean_conf"] = float(np.mean([p.meta.get("conf", 0.0) for p in preds]))
        return out

    def topology(self, genome: Genome, metrics: dict[str, float] | None = None) -> dict[str, Any]:
        m = metrics or {}
        s1, s2, s3 = m.get("share_t1", 0.4), m.get("share_t2", 0.4), m.get("share_t3", 0.2)
        t3 = genome.get("t3_policy", "off")
        t2label = genome.get("t2_model", "logreg") + (f" · {genome['t2_runtime']}" if genome.get("t2_runtime") else "")
        nodes = [
            {"id": "ingress", "label": "Ticket ingress", "kind": "io", "sub": "partner app · WhatsApp · email"},
            {"id": "normalize", "label": "Normalise", "kind": "transform", "sub": genome.get("normalize"), "stage": None},
            {"id": "tier1", "label": "Tier 1 · fuzzy gate", "kind": "gate", "stage": "tier1",
             "sub": f"{genome.get('t1_scorer')} τ={genome.get('t1_tau', 0):.2f} δ={genome.get('t1_delta', 0):.2f}"},
            {"id": "tier2", "label": "Tier 2 · classifier", "kind": "model", "stage": "tier2",
             "sub": f"{genome.get('t2_features')} · {t2label} τ={genome.get('t2_tau', 0):.2f}"},
        ]
        edges = [
            {"source": "ingress", "target": "normalize", "share": 1.0},
            {"source": "normalize", "target": "tier1", "share": 1.0},
            {"source": "tier1", "target": "router", "share": s1, "label": "accept"},
            {"source": "tier1", "target": "tier2", "share": 1 - s1, "label": "escalate"},
            {"source": "tier2", "target": "router", "share": s2, "label": "accept"},
        ]
        if t3 != "off":
            kind = "llm" if t3 == "llm" else "model"
            sub = f"Claude {genome.get('t3_model')} · k={genome.get('t3_k')}" if t3 == "llm" else f"kNN k={genome.get('t3_knn')} · shortlist {genome.get('t3_k')}"
            nodes.append({"id": "tier3", "label": "Tier 3 · resolver", "kind": kind, "stage": "tier3", "sub": sub})
            edges += [{"source": "tier2", "target": "tier3", "share": s3, "label": "escalate"},
                      {"source": "tier3", "target": "router", "share": s3, "label": "resolve"}]
        nodes += [
            {"id": "router", "label": "Queue router", "kind": "transform", "sub": "6 ops queues"},
            {"id": "memory", "label": "Error bank", "kind": "store", "sub": "misroutes → few-shots"},
            {"id": "egress", "label": "Ops queues", "kind": "io", "sub": "disbursal · docs · sanction · payout · tech · legal"},
        ]
        edges += [{"source": "router", "target": "egress", "share": 1.0},
                  {"source": "router", "target": "memory", "share": 0.05, "label": "flagged"}]
        return {"nodes": nodes, "edges": edges}
