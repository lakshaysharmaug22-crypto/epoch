"""Prometheus metrics, OpenTelemetry tracing and a GPU telemetry sampler (NVML)."""

from __future__ import annotations

import threading
import time
from functools import lru_cache

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from epoch.config import settings

REGISTRY = CollectorRegistry()
_BUCKETS = (0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)


class metrics:  # namespace
    CONTENT_TYPE = CONTENT_TYPE_LATEST
    REQUEST_LATENCY = Histogram("epoch_request_latency_seconds", "Request latency by stage", ["stage"],
                                buckets=_BUCKETS, registry=REGISTRY)
    BATCH_SIZE = Histogram("epoch_batch_size", "Formed batch sizes", buckets=(1, 2, 4, 8, 16, 32, 64), registry=REGISTRY)
    TRIALS = Counter("epoch_trials_total", "Trials by status and strategy", ["status", "strategy"], registry=REGISTRY)
    HYPERVOLUME = Gauge("epoch_hypervolume", "Running hypervolume", ["run_id"], registry=REGISTRY)
    INCIDENTS = Counter("epoch_incidents_total", "Incidents by status", ["status"], registry=REGISTRY)
    GPU_UTIL = Gauge("epoch_gpu_utilization", "GPU utilisation %", ["gpu"], registry=REGISTRY)
    GPU_MEM = Gauge("epoch_gpu_memory_used_bytes", "GPU memory used", ["gpu"], registry=REGISTRY)

    @staticmethod
    def render() -> bytes:
        return generate_latest(REGISTRY)


@lru_cache
def tracer():
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider

    provider = TracerProvider(resource=Resource.create({"service.name": "epoch"}))
    endpoint = settings().otlp_endpoint
    if endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        except ImportError:
            pass
    trace.set_tracer_provider(provider)
    return trace.get_tracer("epoch")


class GpuSampler(threading.Thread):
    """Samples NVML utilisation/memory every `interval` seconds into Prometheus gauges and a ring buffer."""

    def __init__(self, interval: float = 1.0, keep: int = 600):
        super().__init__(daemon=True)
        self.interval, self.keep = interval, keep
        self.samples: list[dict] = []
        self._stop = threading.Event()

    def run(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
        except Exception:
            return
        while not self._stop.is_set():
            for i, h in enumerate(handles):
                u = pynvml.nvmlDeviceGetUtilizationRates(h)
                m = pynvml.nvmlDeviceGetMemoryInfo(h)
                metrics.GPU_UTIL.labels(gpu=str(i)).set(u.gpu)
                metrics.GPU_MEM.labels(gpu=str(i)).set(m.used)
                self.samples.append({"t": time.time(), "gpu": i, "util": u.gpu, "mem_mb": m.used / 2**20})
            self.samples = self.samples[-self.keep :]
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()
