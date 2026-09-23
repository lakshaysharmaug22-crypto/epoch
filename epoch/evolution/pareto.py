"""Pareto utilities in a normalised minimisation space with fixed bounds (so hypervolume is comparable)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from epoch.objectives import Objective

REF = 1.1  # reference point in normalised space (worst corner + 10%)


def normalize(points: np.ndarray | Sequence[Sequence[float]], objectives: list[Objective]) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    if p.size == 0:
        return p.reshape(0, len(objectives))
    return np.column_stack([[o.normalized_min(v) for v in p[:, j]] for j, o in enumerate(objectives)])


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def non_dominated(norm: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated rows (minimisation)."""
    n = norm.shape[0]
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        dom = np.all(norm <= norm[i], axis=1) & np.any(norm < norm[i], axis=1)
        if dom.any():
            mask[i] = False
    return mask


def hypervolume(norm: np.ndarray, ref: float = REF) -> float:
    if norm.size == 0:
        return 0.0
    pts = norm[np.all(norm < ref, axis=1)]
    if pts.shape[0] == 0:
        return 0.0
    pts = pts[non_dominated(pts)]
    d = pts.shape[1]
    if d == 2:
        pts = pts[np.argsort(pts[:, 0])]
        hv, prev_y = 0.0, ref
        for x, y in pts:
            if y < prev_y:
                hv += (ref - x) * (prev_y - y)
                prev_y = y
        return float(hv)
    from pymoo.indicators.hv import HV

    return float(HV(ref_point=np.full(d, ref))(pts))


def hv_contribution(norm_front: np.ndarray, candidate: np.ndarray, ref: float = REF) -> float:
    base = hypervolume(norm_front, ref)
    return hypervolume(np.vstack([norm_front, candidate[None, :]]) if norm_front.size else candidate[None, :], ref) - base


def crowding(norm: np.ndarray) -> np.ndarray:
    n, d = norm.shape
    if n <= 2:
        return np.full(n, np.inf)
    dist = np.zeros(n)
    for j in range(d):
        order = np.argsort(norm[:, j])
        span = norm[order[-1], j] - norm[order[0], j] or 1.0
        dist[order[0]] = dist[order[-1]] = np.inf
        dist[order[1:-1]] += (norm[order[2:], j] - norm[order[:-2], j]) / span
    return dist
