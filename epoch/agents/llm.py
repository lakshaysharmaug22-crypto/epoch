"""Thin Claude client: forced tool-use structured output, retries, token cost accounting and a
record/replay cache so repeated identical calls during search don't re-bill (recorded latency is replayed
so latency measurements stay honest; cost is always reported as if uncached)."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from epoch.config import pricing, settings


class LLMUnavailable(RuntimeError):
    pass


@dataclass
class LLMResult:
    data: dict[str, Any]
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    latency_ms: float
    cost_usd: float
    replayed: bool = False


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    defs = schema.pop("$defs", {})

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(dict(defs[node["$ref"].split("/")[-1]]))
            return {k: walk(v) for k, v in node.items() if k != "title"}
        if isinstance(node, list):
            return [walk(x) for x in node]
        return node

    return walk(schema)


def schema_of(model: type[BaseModel]) -> dict[str, Any]:
    return _inline_refs(model.model_json_schema())


def token_cost(model: str, inp: int, out: int, cache_read: int = 0, cache_write: int = 0) -> float:
    p = pricing()["llm_per_million_tokens"].get(model)
    if not p:
        return 0.0
    return (inp * p["input"] + out * p["output"] + cache_read * p["cache_read"] + cache_write * p["cache_write"]) / 1e6


class _Cache:
    def __init__(self) -> None:
        s = settings()
        s.home.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(s.home / "llm_cache.db", check_same_thread=False)
        self.conn.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")
        self.lock = threading.Lock()

    def get(self, k: str) -> dict | None:
        with self.lock:
            row = self.conn.execute("SELECT v FROM c WHERE k=?", (k,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, k: str, v: dict) -> None:
        with self.lock:
            self.conn.execute("INSERT OR REPLACE INTO c VALUES (?, ?)", (k, json.dumps(v)))
            self.conn.commit()


_cache: _Cache | None = None
_client = None
_lock = threading.Lock()


def _get_client():
    global _client
    with _lock:
        if _client is None:
            s = settings()
            if not s.llm_enabled:
                raise LLMUnavailable("ANTHROPIC_API_KEY not set (or EPOCH_OFFLINE=1)")
            import anthropic

            _client = anthropic.Anthropic(api_key=s.anthropic_api_key, max_retries=3, timeout=60)
        return _client


def structured(
    *,
    model: str,
    system: str,
    user: str,
    schema: dict[str, Any],
    tool_name: str = "respond",
    tool_description: str = "Return the structured answer.",
    max_tokens: int = 1024,
    temperature: float = 0.0,
    use_cache: bool = True,
    replay_latency: bool = True,
    sample_id: int = 0,
) -> LLMResult:
    """Call Claude and force a single tool call whose input matches `schema`."""
    global _cache
    key = hashlib.sha256(
        json.dumps([model, system, user, schema, tool_name, temperature, sample_id], sort_keys=True).encode()
    ).hexdigest()
    if use_cache:
        if _cache is None:
            _cache = _Cache()
        hit = _cache.get(key)
        if hit:
            if replay_latency:
                time.sleep(hit["latency_ms"] / 1e3)
            return LLMResult(**{**hit, "replayed": True})

    client = _get_client()
    t0 = time.perf_counter()
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
        tools=[{"name": tool_name, "description": tool_description, "input_schema": schema}],
        tool_choice={"type": "tool", "name": tool_name},
    )
    latency_ms = (time.perf_counter() - t0) * 1e3
    block = next((b for b in resp.content if getattr(b, "type", "") == "tool_use"), None)
    if block is None:
        raise RuntimeError("model did not return a tool call")
    u = resp.usage
    cr = getattr(u, "cache_read_input_tokens", 0) or 0
    cw = getattr(u, "cache_creation_input_tokens", 0) or 0
    res = LLMResult(
        data=dict(block.input),
        model=model,
        input_tokens=u.input_tokens,
        output_tokens=u.output_tokens,
        cache_read_tokens=cr,
        cache_write_tokens=cw,
        latency_ms=latency_ms,
        cost_usd=token_cost(model, u.input_tokens, u.output_tokens, cr, cw),
    )
    if use_cache and _cache is not None:
        _cache.put(key, {k: v for k, v in res.__dict__.items() if k != "replayed"})
    return res
