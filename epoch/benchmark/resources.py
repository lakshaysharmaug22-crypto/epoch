"""Device detection and peak-memory measurement (CUDA via torch, CPU via tracemalloc)."""

from __future__ import annotations

import os
import platform
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache

from epoch.config import pricing, settings


@lru_cache
def device_info() -> dict:
    info = {
        "device": "cpu",
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "gpu": None,
        "gpu_mem_gb": None,
        "cuda": None,
        "torch": None,
        "capabilities": {"bnb": False, "onnx_gpu": False, "torch_compile": False, "triton": False, "hf": False},
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["capabilities"]["hf"] = True
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info.update(device="cuda", gpu=p.name, gpu_mem_gb=round(p.total_memory / 1e9, 1), cuda=torch.version.cuda)
            try:
                import bitsandbytes  # noqa: F401

                info["capabilities"]["bnb"] = True
            except Exception:
                pass
            try:
                import triton  # noqa: F401

                info["capabilities"]["triton"] = True
                # inductor (torch.compile on CUDA) needs Triton; native Windows usually lacks it
                info["capabilities"]["torch_compile"] = True
            except Exception:
                pass
    except Exception:
        pass
    try:
        import onnxruntime as ort

        info["capabilities"]["onnx_gpu"] = "CUDAExecutionProvider" in ort.get_available_providers()
        info["onnxruntime"] = ort.__version__
    except Exception:
        pass
    forced = settings().device
    if forced:
        info["device"] = forced
    return info


def hw_price_per_hour() -> float:
    s = settings()
    if s.hw_price_per_hour is not None:
        return float(s.hw_price_per_hour)
    return float(pricing()["hardware_per_hour"][device_info()["device"]])


@dataclass
class MemoryProbe:
    peak_mb: float = 0.0
    kind: str = "cpu-heap"
    samples: list[float] = field(default_factory=list)


@contextmanager
def peak_memory(device: str | None = None):
    """Yields a MemoryProbe filled with the peak allocated memory inside the block."""
    probe = MemoryProbe()
    device = device or device_info()["device"]
    if device == "cuda":
        import torch

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        probe.kind = "cuda-allocated"
        try:
            yield probe
        finally:
            torch.cuda.synchronize()
            probe.peak_mb = torch.cuda.max_memory_allocated() / 2**20
        return
    started = not tracemalloc.is_tracing()
    if started:
        tracemalloc.start()
    tracemalloc.reset_peak()
    base, _ = tracemalloc.get_traced_memory()
    try:
        yield probe
    finally:
        _, peak = tracemalloc.get_traced_memory()
        probe.peak_mb = max(0.0, (peak - base) / 2**20)
        if started:
            tracemalloc.stop()
