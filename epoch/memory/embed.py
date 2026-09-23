"""384-d text embeddings for failure/incident memory. Uses sentence-transformers when available (same width as
bge-small / MiniLM), else a deterministic hashing embedding so memory search works fully offline."""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

DIM = 384


@lru_cache
def _st():
    name = os.environ.get("EPOCH_EMBEDDER")
    if not name:
        return None
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(name)
    except Exception:
        return None


_HASH = HashingVectorizer(n_features=DIM, alternate_sign=False, ngram_range=(1, 2), norm="l2",
                          token_pattern=r"(?u)\b[a-zA-Z_][a-zA-Z0-9_]+\b")


def embed(texts: list[str]) -> np.ndarray:
    st = _st()
    if st is not None:
        return np.asarray(st.encode(texts, normalize_embeddings=True), dtype=np.float32)
    return _HASH.transform(texts).toarray().astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    return a @ b.T
