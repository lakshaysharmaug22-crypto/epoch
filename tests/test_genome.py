import numpy as np
import optuna

from epoch.genome import Gene, SearchSpace, genome_id


def space() -> SearchSpace:
    return SearchSpace([
        Gene("model", "cat", ("a", "b"), default="a"),
        Gene("reranker", "cat", (False, True), default=False),
        Gene("keep", "int", low=1, high=5, default=3, when=("reranker", (True,))),
        Gene("lr", "float", low=1e-4, high=1e-1, log=True, default=1e-2),
        Gene("chunk", "int", low=40, high=320, step=20, default=120),
    ])


def test_conditional_genes_are_dropped_and_filled():
    s = space()
    g = s.repair({"model": "b", "reranker": False, "keep": 4, "lr": 0.5, "chunk": 133})
    assert "keep" not in g
    assert g["lr"] == 0.1  # clipped
    assert g["chunk"] == 140  # snapped to step
    g2 = s.repair({"model": "b", "reranker": True})
    assert g2["keep"] == 3 and not s.validate(g2)


def test_sample_mutate_crossover_stay_valid():
    s = space()
    rng = np.random.default_rng(0)
    for _ in range(200):
        a, b = s.sample(rng), s.sample(rng)
        assert not s.validate(a)
        assert not s.validate(s.mutate(a, rng, n=2))
        assert not s.validate(s.crossover(a, b, rng))


def test_optuna_suggest_matches_space_and_enqueue():
    s = space()
    study = optuna.create_study()
    study.enqueue_trial({"model": "b", "reranker": True, "keep": 5, "lr": 0.001, "chunk": 200})
    t = study.ask()
    g = s.repair(s.suggest(t))
    assert g == {"model": "b", "reranker": True, "keep": 5, "lr": 0.001, "chunk": 200}


def test_encoding_width_and_ids():
    s = space()
    g = s.default()
    assert s.encode(g).shape[0] == len(s.feature_names()) == len(s.feature_owner())
    assert genome_id(g) == genome_id(dict(reversed(list(g.items()))))
    assert s.diff(g, {**g, "model": "b"}) == {"model": ("a", "b")}
