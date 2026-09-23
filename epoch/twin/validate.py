"""Validate the twin against reality.

1. Serve the genome as a real HTTP service (one replica, FIFO micro-batcher).
2. Calibrate: unloaded latency gives the in-server service-time distribution and fixed client overhead; a short
   closed-loop saturation run gives real capacity, from which the per-request serving overhead is derived.
3. Validate: drive the service with an open-loop Poisson generator at 30/55/80% of measured capacity (latency is
   measured from the *scheduled* send time — no coordinated omission) and compare p50/p95 with the twin.

Only capacity is calibrated; the latency distribution under load (queueing) is a genuine prediction."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from typing import Any

import httpx
import numpy as np

from epoch.genome import Genome
from epoch.twin.profile import measure_profile
from epoch.twin.sim import Traffic, TwinConfig, simulate
from epoch.workloads import get_workload


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _RawHTTP:
    """Minimal keep-alive HTTP/1.1 client on asyncio streams. A load generator must be much cheaper than the
    service it measures; httpx costs ~1 ms of CPU per request, this costs a few µs."""

    def __init__(self, host: str, port: int):
        self.host, self.port = host, port
        self.idle: list[tuple[asyncio.StreamReader, asyncio.StreamWriter]] = []

    async def post(self, path: str, body: bytes) -> bytes:
        for attempt in range(2):
            conn = self.idle.pop() if self.idle and attempt == 0 else await asyncio.open_connection(self.host, self.port)
            try:
                return await self._roundtrip(conn, path, body)
            except (asyncio.IncompleteReadError, ConnectionError):  # server closed an idle keep-alive connection
                conn[1].close()
        raise ConnectionError("request failed twice")

    async def _roundtrip(self, conn, path: str, body: bytes) -> bytes:
        reader, writer = conn
        writer.write(b"POST " + path.encode() + b" HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                     b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
        head = await reader.readuntil(b"\r\n\r\n")
        n = 0
        for line in head.split(b"\r\n"):
            if line.lower().startswith(b"content-length:"):
                n = int(line.split(b":")[1])
        payload = await reader.readexactly(n)
        self.idle.append(conn)
        return payload

    async def close(self) -> None:
        for _, w in self.idle:
            w.close()


def _bodies(payloads: list[str]) -> list[bytes]:
    return [json.dumps({"x": p}).encode() for p in payloads]


async def _open_loop(port: int, payloads: list[str], rate: float, duration: float, seed: int,
                     warmup_s: float) -> tuple[list[float], list[float]]:
    rng = np.random.default_rng(seed)
    arrivals, t = [], 0.0
    while True:
        t += rng.exponential(1.0 / rate)
        if t >= duration:
            break
        arrivals.append(t)
    bodies = _bodies(payloads)
    cli = _RawHTTP("127.0.0.1", port)
    lat: list[float | None] = [None] * len(arrivals)
    lag: list[float] = []
    loop = asyncio.get_running_loop()
    start = loop.time() + 0.3

    async def one(i: int, at: float) -> None:
        await asyncio.sleep(max(0.0, start + at - loop.time()))
        lag.append((loop.time() - (start + at)) * 1e3)
        await cli.post("/predict", bodies[i % len(bodies)])
        lat[i] = (loop.time() - (start + at)) * 1e3

    await asyncio.gather(*(one(i, a) for i, a in enumerate(arrivals)))
    await cli.close()
    return [x for x, a in zip(lat, arrivals, strict=True) if x is not None and a >= warmup_s], lag


async def _saturate(port: int, payloads: list[str], seconds: float, workers: int = 16) -> float:
    """Closed-loop saturation: completed requests per second with `workers` always-busy clients."""
    bodies = _bodies(payloads)
    cli = _RawHTTP("127.0.0.1", port)
    done = 0
    loop = asyncio.get_running_loop()
    end = loop.time() + seconds

    async def worker(i: int) -> None:
        nonlocal done
        k = i
        while loop.time() < end:
            await cli.post("/predict", bodies[k % len(bodies)])
            done += 1
            k += workers

    await asyncio.gather(*(worker(i) for i in range(workers)))
    await cli.close()
    return done / seconds


def _wait(url: str, timeout: float = 60) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if httpx.get(url, timeout=1).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.3)
    raise RuntimeError("serving process did not come up")


def validate_twin(workload_name: str, genome: Genome, *, utilizations: tuple[float, ...] = (0.3, 0.55, 0.8),
                  duration_s: float = 20.0, seed: int = 0) -> dict[str, Any]:
    wl = get_workload(workload_name)
    wl.prepare()
    g = {**genome, "batch_size": 1} if "batch_size" in genome else genome
    profile = measure_profile(wl, g, seed=seed)
    port = _free_port()
    env = {**os.environ, "EPOCH_SERVE_WORKLOAD": workload_name, "EPOCH_SERVE_GENOME": json.dumps(g),
           "EPOCH_SERVE_MAX_BATCH": "1", "EPOCH_SERVE_MAX_WAIT_MS": "0"}
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "epoch.serving:app_from_env", "--factory", "--port",
                             str(port), "--log-level", "warning", "--no-access-log",
                             "--timeout-keep-alive", "120"], env=env)
    base = f"http://127.0.0.1:{port}"
    try:
        _wait(base + "/noop")
        xs, _ = wl.dataset("val", 1.0, seed)
        e2e, svc = [], []
        with httpx.Client() as c:  # warm up, then unloaded latency
            for x in xs[:50]:
                c.post(base + "/predict", json={"x": x})
            for x in xs[:300]:
                t0 = time.perf_counter()
                r = c.post(base + "/predict", json={"x": x}).json()
                e2e.append((time.perf_counter() - t0) * 1e3)
                svc.append(r["service_ms"])
                time.sleep(0.002)
        cap_real = asyncio.run(_saturate(port, xs, seconds=6.0))
        # capacity calibration: rescale the measured service-time distribution so its mean matches the
        # saturation throughput (keeps the shape, fixes the level); trim single-sample outliers above p99.5
        svc_a = np.asarray(svc)
        svc_a = np.minimum(svc_a, np.percentile(svc_a, 99.5))
        scale = (1000.0 / cap_real) / float(svc_a.mean())
        profile.samples_ms = (svc_a * scale).tolist()
        profile.overhead_ms = 0.0
        profile.network_ms = max(0.0, float(np.median(e2e)) - float(np.median(svc_a)))
        r0 = float(np.median(e2e))
        cap = profile.capacity_rps()
        levels = []
        for i, u in enumerate(utilizations):
            rate = u * cap
            real, lag = asyncio.run(_open_loop(port, xs, rate, duration_s, seed + i, warmup_s=2.0))
            sim = simulate(profile, TwinConfig(), Traffic("poisson", rate=rate, duration_s=duration_s), seed=seed + i)["summary"]
            rp50, rp95, rp99 = (float(np.percentile(real, q)) for q in (50, 95, 99))
            levels.append({
                "utilization_target": u, "rate_rps": round(rate, 2), "n": len(real),
                "client_lag_p95_ms": float(np.percentile(lag, 95)),
                "real": {"p50": rp50, "p95": rp95, "p99": rp99},
                "twin": {"p50": sim["p50"], "p95": sim["p95"], "p99": sim["p99"], "utilization": sim["utilization"]},
                "err_p50": abs(sim["p50"] - rp50) / rp50, "err_p95": abs(sim["p95"] - rp95) / rp95,
            })
        return {"workload": workload_name, "genome": g, "profile": profile.to_json(), "unloaded_p50_ms": r0,
                "capacity_rps": cap, "capacity_measured_rps": cap_real, "service_scale": scale, "duration_s": duration_s, "levels": levels,
                "mape_p50": float(np.mean([lv["err_p50"] for lv in levels])),
                "mape_p95": float(np.mean([lv["err_p95"] for lv in levels]))}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
