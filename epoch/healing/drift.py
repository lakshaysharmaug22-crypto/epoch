"""Label-free input drift signals for text traffic."""

from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

_CHAR = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 3), n_features=2**12, alternate_sign=False, norm=None)
_TOK = re.compile(r"[a-z]+")


def char_hist(texts: list[str]) -> np.ndarray:
    h = np.asarray(_CHAR.transform([t.lower() for t in texts]).sum(axis=0)).ravel() + 1e-9
    return h / h.sum()


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a * np.log2(a / b)))  # noqa: E731
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def oov_rate(texts: list[str], vocab: set[str]) -> float:
    toks = [w for t in texts for w in _TOK.findall(t.lower()) if len(w) > 2]
    return float(np.mean([w not in vocab for w in toks])) if toks else 0.0


def vocab_of(texts: list[str]) -> set[str]:
    return {w for t in texts for w in _TOK.findall(t.lower()) if len(w) > 2}
