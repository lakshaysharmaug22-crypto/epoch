"""EPOCH API — the dashboard's live mode. Serves the same bundle contract as the static replay file, plus
control endpoints (start runs/races, approve remediations, run twin what-ifs) and an SSE event stream."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any, Literal

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from epoch import __version__
from epoch.config import settings
from epoch.memory import Store
from epoch.observability import metrics as M

app = FastAPI(title="EPOCH", version=__version__, description="Autonomous AI engineering & evolution platform")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_store: Store | None = None
_bundle_cache: dict[str, tuple[float, dict]] = {}
_jobs: dict[str, dict] = {}


def store() -> Store:
    global _store
    if _store is None:
        _store = Store()
    return _store


def _dispatch(name: str, fn, *args, **kwargs) -> str:
    """Run on Celery when a broker is configured, else on a background thread."""
    job_id = f"{name}-{int(time.time() * 1000)}"
    _jobs[job_id] = {"id": job_id, "name": name, "status": "running", "started": time.time()}
    if settings().redis_url:
        from epoch import worker

        task = getattr(worker, name).delay(*args, **kwargs)
        _jobs[job_id].update(celery_id=task.id, status="queued")
        return job_id

    def target() -> None:
        try:
            _jobs[job_id]["result"] = fn(*args, **kwargs)
            _jobs[job_id]["status"] = "complete"
        except Exception as exc:
            _jobs[job_id].update(status="failed", error=str(exc)[:500])

    threading.Thread(target=target, daemon=True).start()
    return job_id


# ------------------------------------------------------------------ read
@app.get("/api/health")
def health() -> dict[str, Any]:
    from epoch.benchmark.resources import device_info

    s = settings()
    return {"ok": True, "version": __version__, "device": device_info()["device"], "llm": s.llm_enabled,
            "db": s.resolved_db_url.split("://")[0]}


@app.get("/api/bundle")
def bundle(workloads: str = "triage,rag", max_age: float = 5.0) -> dict[str, Any]:
    from epoch.demo import load_extras
    from epoch.export import _clean, workload_bundle
    from epoch.memory.provenance import provenance

    key = workloads
    hit = _bundle_cache.get(key)
    if hit and time.time() - hit[0] < max_age:
        return hit[1]
    s = settings()
    ws = [w for w in workloads.split(",") if w]
    data = _clean({
        "meta": {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "epoch_version": __version__,
                 "mode": "live", "provenance": provenance(0), "workloads": ws, "notes": [],
                 "llm": {"enabled": s.llm_enabled, "agent_model": s.agent_model if s.llm_enabled else None,
                         "judge_model": s.judge_model if s.llm_enabled else None}},
        "workloads": {w: workload_bundle(store(), w, load_extras(w)) for w in ws},
    })
    _bundle_cache[key] = (time.time(), data)
    return data


@app.get("/api/runs")
def runs(workload: str | None = None) -> list[dict]:
    return store().runs(workload=workload)


@app.get("/api/runs/{run_id}/trials")
def trials(run_id: str) -> list[dict]:
    return store().trials(run_id)


@app.get("/api/incidents")
def incidents() -> list[dict]:
    return [{k: v for k, v in i.items() if k != "data"} for i in store().incidents()]


@app.get("/api/incidents/{iid}")
def incident(iid: str) -> dict:
    i = store().incident(iid)
    if not i:
        raise HTTPException(404)
    return i


@app.get("/api/jobs")
def jobs() -> list[dict]:
    return list(_jobs.values())


# ------------------------------------------------------------------ control
class RunIn(BaseModel):
    workload: str = "triage"
    strategy: Literal["random", "tpe", "nsga2", "agent"] = "agent"
    budget: int = Field(40, ge=4, le=500)
    seed: int = 0


class RaceIn(BaseModel):
    workload: str = "triage"
    strategies: list[str] = ["random", "tpe", "nsga2", "agent"]
    budget: int = Field(40, ge=4, le=500)
    seeds: int = Field(2, ge=1, le=10)


class ApproveIn(BaseModel):
    approved: bool = True
    actor: str = "dashboard"
    note: str = ""


class HealIn(BaseModel):
    workload: str = "triage"
    scenario: Literal["drift", "latency"] = "drift"


def _run_study(workload: str, strategy: str, budget: int, seed: int) -> dict:
    from epoch.evolution import EngineConfig, Evolution
    from epoch.workloads import get_workload

    evo = Evolution(get_workload(workload), strategy, store(), EngineConfig(budget=budget, seed=seed), sink=_publish)
    return {k: v for k, v in evo.run().items() if k != "hv_curve"}


def _run_race(workload: str, strategies: list[str], budget: int, seeds: int) -> dict:
    from epoch.evolution.race import race
    from epoch.workloads import get_workload

    return race(get_workload(workload), strategies, budget, list(range(seeds)), store(), sink=_publish).get("headline") or {}


def _run_heal(workload: str, scenario: str) -> dict:
    from epoch.export import workload_bundle
    from epoch.healing import Healer, scenario_for
    from epoch.workloads import get_workload

    wl = get_workload(workload)
    wb = workload_bundle(store(), workload)
    knee = next(t for t in wb["trials"] if t["id"] == wb["knee"])
    archive = [{"trial_id": t["id"], "genome": t["genome"], "metrics": t["metrics"]} for t in wb["trials"] if t["global_pareto"]]
    h = Healer(wl, scenario_for(wl, scenario), store(), knee["genome"], archive=archive, sink=_publish)
    h.run(approval="manual")
    return {"incident": h.incident_id}


@app.post("/api/runs")
def start_run(body: RunIn) -> dict:
    return {"job": _dispatch("run_study", _run_study, body.workload, body.strategy, body.budget, body.seed)}


@app.post("/api/race")
def start_race(body: RaceIn) -> dict:
    return {"job": _dispatch("run_race", _run_race, body.workload, body.strategies, body.budget, body.seeds)}


@app.post("/api/heal")
def start_heal(body: HealIn) -> dict:
    return {"job": _dispatch("run_heal", _run_heal, body.workload, body.scenario)}


@app.post("/api/incidents/{iid}/decision")
def decide(iid: str, body: ApproveIn, bg: BackgroundTasks) -> dict:
    from epoch.healing import approve
    from epoch.workloads import get_workload

    inc = store().incident(iid)
    if not inc:
        raise HTTPException(404)
    if inc["status"] != "awaiting_approval":
        raise HTTPException(409, f"incident is {inc['status']}")
    bg.add_task(approve, store(), get_workload(inc["workload"]), iid, approved=body.approved, actor=body.actor, note=body.note)
    M.INCIDENTS.labels(status="approved" if body.approved else "rejected").inc()
    return {"ok": True, "status": "promoting" if body.approved else "rejecting"}


class TwinIn(BaseModel):
    profile: dict  # ServiceProfile json (samples_ms, alpha, overhead_ms, network_ms)
    config: dict = {}
    traffic: dict = {}
    faults: list[dict] = []
    seed: int = 0


@app.post("/api/twin/simulate")
def twin_simulate(body: TwinIn) -> dict:
    from epoch.twin.sim import Fault, ServiceProfile, Traffic, TwinConfig, simulate

    p = ServiceProfile(**{k: v for k, v in body.profile.items() if k in ("samples_ms", "alpha", "overhead_ms", "network_ms", "label")})
    return simulate(p, TwinConfig(**body.config), Traffic(**body.traffic), [Fault(**f) for f in body.faults], seed=body.seed)


# ------------------------------------------------------------------ events
_subscribers: set[asyncio.Queue] = set()
_bg_tasks: set[asyncio.Task] = set()
_loop: asyncio.AbstractEventLoop | None = None


def _publish(kind: str, payload: dict) -> None:
    if kind == "trial":
        M.TRIALS.labels(status=payload.get("status", "?"), strategy=payload.get("run_id", "").split("-")[1] if "-" in payload.get("run_id", "") else "?").inc()
        M.HYPERVOLUME.labels(run_id=payload.get("run_id", "")).set(payload.get("hv") or 0)
    if _loop is None:
        return
    msg = json.dumps({"kind": kind, **payload}, default=str)
    for q in list(_subscribers):
        _loop.call_soon_threadsafe(q.put_nowait, msg)


@app.on_event("startup")
async def _startup() -> None:
    global _loop
    _loop = asyncio.get_running_loop()
    if settings().redis_url:  # fan-in events published by Celery workers
        _bg_tasks.add(asyncio.create_task(_redis_bridge()))


async def _redis_bridge() -> None:
    import redis.asyncio as aioredis

    r = aioredis.from_url(settings().redis_url)
    ps = r.pubsub()
    await ps.subscribe("epoch:events")
    async for m in ps.listen():
        if m.get("type") == "message":
            for q in list(_subscribers):
                q.put_nowait(m["data"].decode() if isinstance(m["data"], bytes) else m["data"])


@app.get("/api/events/stream")
async def stream() -> EventSourceResponse:
    q: asyncio.Queue = asyncio.Queue(maxsize=1000)
    _subscribers.add(q)

    async def gen():
        try:
            yield {"event": "hello", "data": json.dumps({"version": __version__})}
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=15)
                    yield {"event": "epoch", "data": msg}
                except TimeoutError:
                    yield {"event": "ping", "data": "{}"}
        finally:
            _subscribers.discard(q)

    return EventSourceResponse(gen())


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    return PlainTextResponse(M.render(), media_type=M.CONTENT_TYPE)

