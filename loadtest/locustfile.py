"""Locust load test for a served genome (closed-loop users). For latency SLOs prefer the open-loop generator in
`epoch/twin/validate.py` — closed-loop load hides queueing delay (coordinated omission).

    EPOCH_SERVE_WORKLOAD=triage EPOCH_SERVE_GENOME='{...}' uvicorn epoch.serving:app_from_env --factory --port 8100
    locust -f loadtest/locustfile.py --host http://localhost:8100
"""

import random

from locust import FastHttpUser, between, task

from epoch.workloads.triage import data as D

TICKETS = [t.text for t in D.build_splits()["test"]]


class Partner(FastHttpUser):
    wait_time = between(0.0, 0.05)

    @task
    def route(self) -> None:
        self.client.post("/predict", json={"x": random.choice(TICKETS)})
