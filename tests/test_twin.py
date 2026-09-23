import numpy as np

from epoch.twin.sim import Fault, ServiceProfile, Traffic, TwinConfig, plan_capacity, simulate


def exp_profile(mean_ms: float, n: int = 20000) -> ServiceProfile:
    return ServiceProfile(samples_ms=list(np.random.default_rng(0).exponential(mean_ms, n)), alpha=0.0)


def test_mm1_mean_response_time():
    # M/M/1 with rho = 0.5: E[T] = 1 / (mu - lambda) = 2 * service mean
    prof = exp_profile(10.0)
    res = simulate(prof, TwinConfig(), Traffic("poisson", rate=50.0, duration_s=600), seed=1)
    lat = [x for s in res["series"] for x in ([s["p50"]] if s["p50"] else [])]
    assert lat
    util = res["summary"]["utilization"]
    assert abs(util - 0.5) < 0.05
    # median of an exponential with mean 20 ms is 20*ln2 ≈ 13.9 ms
    assert 11.5 < res["summary"]["p50"] < 16.5


def test_replica_loss_hurts_tail():
    prof = exp_profile(10.0)
    tr = Traffic("poisson", rate=150.0, duration_s=120)
    ok = simulate(prof, TwinConfig(replicas=2), tr, seed=2)["summary"]
    bad = simulate(prof, TwinConfig(replicas=2), tr, [Fault("replica_down", 30, 40, replica=1)], seed=2)["summary"]
    assert bad["p95"] > 2 * ok["p95"]


def test_batching_raises_capacity():
    prof = ServiceProfile(samples_ms=[10.0] * 100, alpha=0.1)
    tr = Traffic("poisson", rate=180.0, duration_s=60)
    single = simulate(prof, TwinConfig(max_batch=1), tr, seed=3)["summary"]
    batched = simulate(prof, TwinConfig(max_batch=8, max_wait_ms=2), tr, seed=3)["summary"]
    assert not single["stable"] or single["p95"] > 1000
    assert batched["stable"] and batched["p95"] < 200 and batched["mean_batch"] > 1.5


def test_capacity_planner_finds_cheapest():
    prof = exp_profile(10.0)
    plan = plan_capacity(prof, Traffic("poisson", rate=160.0, duration_s=40), slo_p95_ms=80, price_per_replica_hour=1.0,
                         batches=(1,), waits=(0.0,))
    assert plan["best"] is not None and plan["best"]["replicas"] >= 2
