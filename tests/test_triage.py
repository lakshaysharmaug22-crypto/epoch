import numpy as np

from epoch.benchmark import run_benchmark
from epoch.workloads import get_workload
from epoch.workloads.triage import data as D
from epoch.workloads.triage.workload import Tier2


def test_splits_are_deterministic_and_heldout():
    s = D.build_splits()
    assert len(s["train"]) == 1500 and len(s["drift"]) == 900
    assert not any(t.heldout for t in s["train"])
    assert any(t.heldout for t in s["val"])
    assert s["val"][0].text == D.build_splits()["val"][0].text


def test_offline_space_has_no_llm_genes():
    w = get_workload("triage", llm=False)
    names = {g.name for g in w.space().genes}
    assert "t3_model" not in names and "t3_knn" in names
    assert "t3_model" in {g.name for g in get_workload("triage", llm=True).space().genes}


def test_onnx_runtime_matches_sklearn():
    base = {"normalize": "basic", "t2_features": "word", "t2_model": "lsa_mlp", "t2_svd_dim": 64}
    texts = [t.text for t in D.build_splits()["val"][:200]]
    p_sk = Tier2({**base, "t2_runtime": "sklearn"}).proba(texts)
    p_ort = Tier2({**base, "t2_runtime": "onnx"}).proba(texts)
    p_q = Tier2({**base, "t2_runtime": "onnx_int8"}).proba(texts)
    assert np.allclose(p_sk, p_ort, atol=1e-4)
    assert (p_sk.argmax(1) == p_q.argmax(1)).mean() > 0.95


def test_benchmark_reports_all_objectives():
    w = get_workload("triage", llm=False)
    r = run_benchmark(w, w.space().default(), fidelity=0.2, robustness=False)
    assert r.status == "complete"
    for o in w.objectives:
        assert o.name in r.metrics
    assert abs(r.metrics["share_t1"] + r.metrics["share_t2"] + r.metrics["share_t3"] - 1) < 1e-9
