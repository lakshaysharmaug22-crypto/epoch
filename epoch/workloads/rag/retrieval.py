"""Chunking, BM25 (sparse, vectorised), dense (sentence-transformers + FAISS), hybrid RRF, cross-encoder rerank."""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from epoch.workloads.rag.corpus import Doc

EMBEDDERS = {
    "bge-small": ("BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: ", ""),
    "e5-small": ("intfloat/e5-small-v2", "query: ", "passage: "),
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", "", ""),
}
RERANKERS = {
    "minilm-l6": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "bge-reranker-base": "BAAI/bge-reranker-base",
}
_LOCK = threading.Lock()
_INDEX_CACHE: OrderedDict = OrderedDict()


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    title: str
    text: str


def chunk_docs(docs: list[Doc], size: int, overlap: float) -> list[Chunk]:
    out: list[Chunk] = []
    step = max(1, int(size * (1 - overlap)))
    for d in docs:
        words = d.text.split()
        for i, start in enumerate(range(0, max(1, len(words)), step)):
            piece = words[start : start + size]
            if not piece:
                break
            out.append(Chunk(f"{d.id}#{i}", d.id, d.title, f"{d.title}. " + " ".join(piece)))
            if start + size >= len(words):
                break
    return out


class BM25:
    def __init__(self, chunks: list[Chunk], k1: float = 1.4, b: float = 0.75):
        self.vec = CountVectorizer(token_pattern=r"(?u)\b\w+\b", lowercase=True)
        tf = self.vec.fit_transform([c.text for c in chunks]).tocsr().astype(np.float64)
        n = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        idf = np.log((n - df + 0.5) / (df + 0.5) + 1.0)
        dl = np.asarray(tf.sum(axis=1)).ravel()
        norm = k1 * (1 - b + b * dl / dl.mean())
        tf = tf.tocoo()
        w = tf.data * (k1 + 1) / (tf.data + norm[tf.row]) * idf[tf.col]
        self.w = sparse.csr_matrix((w, (tf.row, tf.col)), shape=tf.shape)

    def search(self, queries: list[str], k: int) -> list[list[tuple[int, float]]]:
        q = self.vec.transform(queries)
        q.data[:] = 1.0
        s = (self.w @ q.T).T.toarray()
        return _topk(s, k)


class Dense:
    def __init__(self, chunks: list[Chunk], embedder: str, device: str):
        import faiss
        from sentence_transformers import SentenceTransformer

        mid, self.qp, pp = EMBEDDERS[embedder]
        self.model = SentenceTransformer(mid, device=device)
        emb = self.model.encode([pp + c.text for c in chunks], batch_size=64, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)

    def search(self, queries: list[str], k: int) -> list[list[tuple[int, float]]]:
        q = self.model.encode([self.qp + x for x in queries], normalize_embeddings=True, convert_to_numpy=True,
                              show_progress_bar=False).astype(np.float32)
        d, i = self.index.search(q, k)
        return [[(int(a), float(b)) for a, b in zip(ii, dd, strict=True) if a >= 0] for ii, dd in zip(i, d, strict=True)]


def rrf(*rankings: list[list[tuple[int, float]]], k: int, c: int = 60) -> list[list[tuple[int, float]]]:
    out = []
    for per_query in zip(*rankings, strict=True):
        score: dict[int, float] = {}
        for ranking in per_query:
            for rank, (idx, _) in enumerate(ranking):
                score[idx] = score.get(idx, 0.0) + 1.0 / (c + rank + 1)
        out.append(sorted(score.items(), key=lambda t: -t[1])[:k])
    return out


def _topk(s: np.ndarray, k: int) -> list[list[tuple[int, float]]]:
    k = min(k, s.shape[1])
    idx = np.argpartition(-s, k - 1, axis=1)[:, :k]
    out = []
    for r in range(s.shape[0]):
        row = sorted(((int(j), float(s[r, j])) for j in idx[r]), key=lambda t: -t[1])
        out.append(row)
    return out


class Retriever:
    """Cached per (chunking, retriever, embedder) — index build time is reported as build_s, not latency."""

    def __init__(self, docs: list[Doc], *, size: int, overlap: float, kind: str, embedder: str | None, device: str):
        key = (size, overlap, kind, embedder)
        with _LOCK:
            hit = _INDEX_CACHE.get(key)
        if hit is None:
            chunks = chunk_docs(docs, size, overlap)
            bm25 = BM25(chunks) if kind in ("bm25", "hybrid") else None
            dense = Dense(chunks, embedder or "bge-small", device) if kind in ("dense", "hybrid") else None
            hit = (chunks, bm25, dense)
            with _LOCK:
                _INDEX_CACHE[key] = hit
                while len(_INDEX_CACHE) > 12:
                    _INDEX_CACHE.popitem(last=False)
        self.chunks, self.bm25, self.dense = hit
        self.kind = kind

    def search(self, queries: list[str], k: int) -> list[list[Chunk]]:
        if self.kind == "bm25":
            res = self.bm25.search(queries, k)
        elif self.kind == "dense":
            res = self.dense.search(queries, k)
        else:
            res = rrf(self.bm25.search(queries, k * 2), self.dense.search(queries, k * 2), k=k)
        return [[self.chunks[i] for i, _ in r] for r in res]


_RERANK_CACHE: dict[str, object] = {}


class Reranker:
    def __init__(self, name: str, device: str):
        from sentence_transformers import CrossEncoder

        with _LOCK:
            if name not in _RERANK_CACHE:
                _RERANK_CACHE[name] = CrossEncoder(RERANKERS[name], device=device, max_length=512)
            self.model = _RERANK_CACHE[name]

    def rerank(self, query: str, chunks: list[Chunk], keep: int) -> list[Chunk]:
        if not chunks:
            return chunks
        scores = self.model.predict([(query, c.text) for c in chunks], show_progress_bar=False)
        order = np.argsort(-np.asarray(scores))
        return [chunks[i] for i in order[:keep]]
