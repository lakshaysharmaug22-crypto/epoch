"""Everything needed to reproduce a trial: code version, environment fingerprint, hardware, seed."""

from __future__ import annotations

import contextlib
import hashlib
import platform
import socket
import subprocess
import sys
from datetime import UTC, datetime
from functools import lru_cache
from importlib import metadata

from epoch.benchmark.resources import device_info


@lru_cache
def git_sha() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=3)
        sha = out.stdout.strip()
        if not sha:
            return None
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=3).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return None


@lru_cache
def env_hash() -> str:
    pkgs = sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions() if d.metadata["Name"])
    return hashlib.sha1("\n".join(pkgs).encode()).hexdigest()[:12]


def key_versions() -> dict[str, str]:
    out = {}
    for p in ("optuna", "scikit-learn", "onnxruntime", "torch", "transformers", "langgraph", "anthropic", "numpy"):
        with contextlib.suppress(metadata.PackageNotFoundError):
            out[p] = metadata.version(p)
    return out


def provenance(seed: int) -> dict:
    info = device_info()
    return {
        "git_sha": git_sha(),
        "env_hash": env_hash(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "host": socket.gethostname(),
        "device": info["device"],
        "gpu": info.get("gpu"),
        "cpu_count": info.get("cpu_count"),
        "versions": key_versions(),
        "seed": seed,
        "created_at": datetime.now(UTC).isoformat(),
    }
