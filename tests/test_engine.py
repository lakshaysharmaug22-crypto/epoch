import itertools

import numpy as np
import pytest

from epoch.evolution import EngineConfig, Evolution
from epoch.memory import Store
from epoch.workloads import get_workload


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    return Store(f"sqlite:///{tmp_path_factory.mktemp('mem') / 'epoch.db'}")


@pytest.mark.parametrize("strategy", ["random", "tpe", "nsga2", "agent"])
def test_strategies_run_and_persist(store, strategy):
    w = get_workload("synthetic")
    evo = Evolution(w, strategy, store, EngineConfig(budget=18, seed=1, population=6, warmup=6))
    s = evo.run()
    assert s["n_trials"] == 18
    hv = s["hv_curve"]
    assert all(b >= a - 1e-12 for a, b in itertools.pairwise(hv)), "hypervolume must be monotone"
    trials = store.trials(evo.run_id)
    assert len(trials) == 18
    assert sum(t["pareto"] for t in trials) == s["n_pareto"]
    run = store.run(evo.run_id)
    assert run["status"] == "complete" and run["provenance"]["env_hash"]


def test_failures_are_classified_and_searchable(store):
    w = get_workload("synthetic")
    Evolution(w, "random", store, EngineConfig(budget=30, seed=3)).run()
    fails = store.failures()
    assert fails and all(f["kind"] == "oom" for f in fails)
    hit = store.similar_failures("CUDA out of memory 7b fp16", k=1)
    assert hit and hit[0]["kind"] == "oom"


def test_agent_writes_hypotheses_with_verdicts(store):
    w = get_workload("synthetic")
    evo = Evolution(w, "agent", store, EngineConfig(budget=30, seed=2, warmup=8, agent_batch=4))
    evo.run()
    hs = store.hypotheses(evo.run_id)
    assert hs, "agent should register hypotheses"
    assert {h["source"] for h in hs} == {"heuristic"}
    assert any(h["verdict"] in ("confirmed", "refuted", "inconclusive") for h in hs)
    agent_trials = [t for t in store.trials(evo.run_id) if t["origin"].startswith("agent")]
    assert agent_trials and any(t["parents"] for t in agent_trials)


def test_early_rejection_prunes(store):
    w = get_workload("synthetic")
    evo = Evolution(w, "random", store, EngineConfig(budget=40, seed=5, early_reject=True))
    s = evo.run()
    assert s["n_pruned"] > 0
    assert np.isfinite(s["final_hv"])
