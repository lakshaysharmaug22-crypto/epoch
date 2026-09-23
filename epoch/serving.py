"""Serving runtime for a deployed genome: FIFO micro-batching (max_batch / max_wait) in front of the compiled
pipeline, Prometheus metrics and OpenTelemetry spans. One process = one replica — the exact semantics the
digital twin simulates, which is what makes twin-vs-load-test validation meaningful."""

from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from epoch.observability import metrics as M
from epoch.observability import tracer


class PredictIn(BaseModel):
    x: Any


class MicroBatcher:
    def __init__(self, pipe, max_batch: int = 1, max_wait_ms: float = 0.0):
        self.pipe, self.max_batch, self.max_wait = pipe, max_batch, max_wait_ms / 1e3
        self.q: asyncio.Queue = asyncio.Queue()
        self.task: asyncio.Task | None = None

    async def submit(self, x: Any):
        fut = asyncio.get_running_loop().create_future()
        await self.q.put((x, fut, time.perf_counter()))
        return await fut

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            batch = [await self.q.get()]
            deadline = loop.time() + self.max_wait
            while len(batch) < self.max_batch:
                try:
                    batch.append(self.q.get_nowait())
                    continue
                except asyncio.QueueEmpty:
                    pass
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self.q.get(), remaining))
                except TimeoutError:
                    break
            t0 = time.perf_counter()
            with tracer().start_as_current_span("pipeline.predict_batch") as span:
                span.set_attribute("batch.size", len(batch))
                try:
                    preds = self.pipe.predict_batch([b[0] for b in batch])  # deliberately blocking: one serial server
                    err = None
                except Exception as exc:  # surface per-request errors, keep serving
                    preds, err = [None] * len(batch), exc
            svc = time.perf_counter() - t0
            M.BATCH_SIZE.observe(len(batch))
            for (_, fut, t_in), p in zip(batch, preds, strict=True):
                M.REQUEST_LATENCY.labels(stage="e2e").observe(time.perf_counter() - t_in)
                if err is not None:
                    fut.set_exception(err)
                else:
                    for stage, ms in (p.stages_ms or {}).items():
                        M.REQUEST_LATENCY.labels(stage=stage).observe(ms / 1e3)
                    fut.set_result((p, svc))
            await asyncio.sleep(0)


def create_app(workload_name: str, genome: dict, max_batch: int = 1, max_wait_ms: float = 0.0) -> FastAPI:
    from epoch.workloads import get_workload

    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        wl = get_workload(workload_name)
        wl.prepare()
        pipe = wl.build(genome)
        mb = MicroBatcher(pipe, max_batch, max_wait_ms)
        mb.task = asyncio.create_task(mb.run())
        state.update(pipe=pipe, mb=mb)
        yield
        mb.task.cancel()
        pipe.close()

    app = FastAPI(title="EPOCH serving", lifespan=lifespan)

    @app.post("/predict")
    async def predict(body: PredictIn) -> dict[str, Any]:
        p, svc = await state["mb"].submit(body.x)
        return {"output": p.output, "meta": p.meta, "service_ms": svc * 1e3}

    @app.get("/noop")
    async def noop() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/metrics")
    async def metrics() -> PlainTextResponse:
        return PlainTextResponse(M.render(), media_type=M.CONTENT_TYPE)

    return app


def app_from_env() -> FastAPI:
    return create_app(
        os.environ["EPOCH_SERVE_WORKLOAD"],
        json.loads(os.environ["EPOCH_SERVE_GENOME"]),
        int(os.environ.get("EPOCH_SERVE_MAX_BATCH", "1")),
        float(os.environ.get("EPOCH_SERVE_MAX_WAIT_MS", "0")),
    )
