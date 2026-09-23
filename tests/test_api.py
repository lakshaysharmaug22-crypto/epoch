from fastapi.testclient import TestClient

from epoch.api.main import app


def test_health_and_twin_endpoint():
    c = TestClient(app)
    assert c.get("/api/health").json()["ok"]
    r = c.post("/api/twin/simulate", json={
        "profile": {"samples_ms": [5.0, 6.0, 7.0, 30.0], "alpha": 0.1},
        "config": {"replicas": 1, "max_batch": 4, "max_wait_ms": 2},
        "traffic": {"kind": "bursty", "rate": 60, "duration_s": 20},
        "faults": [{"kind": "slowdown", "start_s": 5, "duration_s": 5, "magnitude": 2.0}],
    })
    body = r.json()
    assert r.status_code == 200 and body["summary"]["p95"] > 0 and len(body["series"]) == 20
    assert "epoch_request_latency_seconds" in c.get("/metrics").text
