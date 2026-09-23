"""Surrogate pre-screening: a random-forest ensemble per objective predicts (mean, std) for unseen genomes so only
candidates with the highest optimistic hypervolume improvement get a real (expensive) benchmark run."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

from epoch.evolution.pareto import hv_contribution, non_dominated
from epoch.genome import Genome, SearchSpace
from epoch.objectives import Objective


@dataclass
class Screened:
    genome: Genome
    mean: np.ndarray  # normalised-min space
    std: np.ndarray
    ehvi: float
    feasible_p: float


class Surrogate:
    def __init__(self, space: SearchSpace, objectives: list[Objective], seed: int = 0):
        self.space, self.objectives, self.seed = space, objectives, seed
        self.models: list[RandomForestRegressor] = []
        self.feas: RandomForestRegressor | None = None
        self.r2: list[float] = []
        self.fitted = False

    def fit(self, genomes: list[Genome], norm_values: np.ndarray, feasible: np.ndarray) -> Surrogate:
        if len(genomes) < 6:
            self.fitted = False
            return self
        x = np.vstack([self.space.encode(g) for g in genomes])
        self.models, self.r2 = [], []
        for j in range(norm_values.shape[1]):
            m = RandomForestRegressor(n_estimators=120, min_samples_leaf=2, random_state=self.seed, n_jobs=1)
            m.fit(x, norm_values[:, j])
            self.models.append(m)
            if len(genomes) >= 12:
                cv = cross_val_score(RandomForestRegressor(n_estimators=40, random_state=self.seed), x, norm_values[:, j],
                                     cv=3, scoring="r2")
                self.r2.append(float(np.mean(cv)))
        self.feas = RandomForestRegressor(n_estimators=60, random_state=self.seed).fit(x, feasible.astype(float))
        self.fitted = True
        return self

    def predict(self, genomes: list[Genome]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.vstack([self.space.encode(g) for g in genomes])
        means, stds = [], []
        for m in self.models:
            per_tree = np.stack([t.predict(x) for t in m.estimators_])
            means.append(per_tree.mean(0))
            stds.append(per_tree.std(0))
        feas = self.feas.predict(x) if self.feas is not None else np.ones(len(genomes))
        return np.column_stack(means), np.column_stack(stds), feas

    def screen(self, candidates: list[Genome], front_norm: np.ndarray, k: int, kappa: float = 1.0) -> list[Screened]:
        if not candidates:
            return []
        if not self.fitted:
            return [Screened(g, np.full(len(self.objectives), np.nan), np.full(len(self.objectives), np.nan), 0.0, 1.0)
                    for g in candidates[:k]]
        mu, sd, feas = self.predict(candidates)
        front = front_norm[non_dominated(front_norm)] if front_norm.size else front_norm
        scored = []
        for g, m, s, f in zip(candidates, mu, sd, feas, strict=True):
            optimistic = np.clip(m - kappa * s, -0.1, 1.1)
            gain = hv_contribution(front, optimistic) * max(f, 0.05)
            scored.append(Screened(g, m, s, float(gain), float(f)))
        scored.sort(key=lambda z: -z.ehvi)
        return scored[:k]

    def gene_importance(self) -> dict[str, list[float]]:
        """Impurity importance aggregated per gene, one list per objective."""
        owners = self.space.feature_owner()
        out: dict[str, list[float]] = {g.name: [0.0] * len(self.models) for g in self.space.genes}
        for j, m in enumerate(self.models):
            for f, imp in zip(owners, m.feature_importances_, strict=True):
                out[f][j] += float(imp)
        return out
