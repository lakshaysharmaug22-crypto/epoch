"""Celery worker: long-running studies, races and healing scenarios run here (on the GPU box) while the API stays
responsive. Events are published to Redis so every API replica can stream them to dashboards."""

from __future__ import annotations

import json

from celery import Celery

from epoch.config import settings

_url = settings().redis_url or "redis://localhost:6379/0"
celery = Celery("epoch", broker=_url, backend=_url)
celery.conf.update(task_acks_late=True, worker_prefetch_multiplier=1, task_track_started=True)


def _publisher():
    import redis

    r = redis.Redis.from_url(_url)

    def sink(kind: str, payload: dict) -> None:
        r.publish("epoch:events", json.dumps({"kind": kind, **payload}, default=str))

    return sink


@celery.task(name="epoch.run_study")
def run_study(workload: str, strategy: str, budget: int, seed: int) -> dict:
    from epoch.evolution import EngineConfig, Evolution
    from epoch.memory import Store
    from epoch.workloads import get_workload

    evo = Evolution(get_workload(workload), strategy, Store(), EngineConfig(budget=budget, seed=seed), sink=_publisher())
    return {k: v for k, v in evo.run().items() if k != "hv_curve"}


@celery.task(name="epoch.run_race")
def run_race(workload: str, strategies: list[str], budget: int, seeds: int) -> dict:
    from epoch.evolution.race import race
    from epoch.memory import Store
    from epoch.workloads import get_workload

    res = race(get_workload(workload), strategies, budget, list(range(seeds)), Store(), sink=_publisher())
    return res.get("headline") or {}


@celery.task(name="epoch.run_heal")
def run_heal(workload: str, scenario: str) -> dict:
    from epoch.api.main import _run_heal

    return _run_heal(workload, scenario)
