# EPOCH — autonomous AI engineering & evolution platform

**[Live dashboard → epoch-omega.vercel.app](https://epoch-omega.vercel.app)** · press **Guided tour** for a 90-second walkthrough, `Ctrl K` to jump anywhere.

![CI](https://github.com/lakshaysharmaug22-crypto/epoch/actions/workflows/ci.yml/badge.svg)

EPOCH takes an AI pipeline, an eval set and constraints, and **evolves the pipeline's configuration to a
multi-objective Pareto frontier** (quality × latency × cost × memory). An agent designs the experiments,
a surrogate model screens them, and every trial is measured, attributed and stored with provenance. A
**digital twin** validated against a real load test predicts production behaviour. A **self-healing loop**
detects regressions, finds the root cause, shadow-tests fixes and promotes one through a human approval gate.

```
genome (typed, conditional search space) ──► evolution engine ──► benchmark engine ──► experiment memory
        ▲                                    random · MOTPE ·       quality · p50/p95/p99 ·  Postgres + pgvector,
        │                                    NSGA-II · agent        memory · $/1k · robust   lineage, provenance
        └── hypothesis agent (LangGraph + Claude) ◄── surrogate EHVI ◄── fANOVA + interventional ablation
                                                                                  │
 serving (micro-batcher) ──► CUSUM detectors ──► RCA agent ──► canary ×3 ──► twin + capacity plan ──► gate ──► promote / rollback
```

Two real workloads ship with it:

| Workload | What evolves | Genes |
|---|---|---|
| **CSAI triage cascade** | fuzzy gate → TF-IDF classifier (sklearn / ONNX Runtime / int8 ORT) → Claude or kNN resolver, routing home-loan partner tickets to 6 ops queues | 16 (conditional) |
| **RAG QA** | chunking → BM25 / dense / hybrid retrieval → cross-encoder rerank → Qwen2.5 generator (fp16 / int8 / nf4 · eager / `torch.compile` / ONNX Runtime) | up to 16, hardware-aware |

The search space adapts to the machine. Genes that need a GPU, bitsandbytes, sentence-transformers or a
Claude key only appear when those are available, so the same code runs on a laptop CPU and on a GPU box.

## Measured results

The replay bundle in `web/public/replay/` was produced by `epoch demo` on a **2-core CPU, heuristic agents (no
API key)**. Every number below comes from that run. Nothing is hand-entered.

**Search, equal budgets, 3 seeds each**

| | Random | MOTPE | NSGA-II | EPOCH agent |
|---|---|---|---|---|
| Triage — median final hypervolume (60 trials) | 0.836 | 0.878 | 0.890 | **0.891** |
| Triage — trials to NSGA-II's final HV (median) | never | never | 60 | **56** (−6.7%) |
| RAG — median final hypervolume (36 trials) | 1.376 | 1.410 | 1.366 | **1.430** |
| RAG — trials to NSGA-II's final HV (median) | 23 | 30 | 31 | **9** (−71%) |

On triage the heuristic agent is only on par: MOTPE has the best area under the curve and one TPE seed found an
outlier (p95 0.37 ms at macro-F1 0.911, a fuzzy-gate-first cascade), which became the knee. The Claude-driven
agent is the version to benchmark next (see *Run it on your GPU*).

**Digital twin vs a real HTTP load test** (open-loop Poisson at 30/55/80% of measured capacity, capacity-only
calibration): triage p50 error 8%, p95 error 13% (MAPE). RAG p50 7%, p95 42%, because sub-millisecond
extractive service times make the tail very sensitive near saturation.

**Self-healing, three injected faults on the triage cascade**

| Fault | What EPOCH did | Outcome vs counterfactual (old config, same traffic) |
|---|---|---|
| Code-mixed phrasing drift (60% unseen templates) | CUSUM alarm → RCA: input drift (char-trigram JS ↑, OOV ↑) → 80 low-confidence tickets to QA → retrain tier 2 on the error bank → shadow canary → approve | **mitigated**: macro-F1 0.74 → 0.88 |
| Tier-3 resolver +40 ms per call | RCA: ~97% of added latency in `tier3` → twin shows the incumbent needs 4 replicas at production load → retrain + lower tier-2 threshold | **resolved**: mean latency −62%, replicas 4 → 1 |
| Tier-2 model server +12 ms per call | every candidate replayed 3× on shadow traffic; none beats the incumbent beyond noise | **escalated**: no placebo change shipped |

Exact per-run values are in the dashboard's Incidents view; `epoch demo` re-records them on your hardware.

## Quickstart

```bash
make setup                 # uv venv + CPU install
make test                  # 28 tests
make demo                  # ~40 min on 2 CPU cores: race → attribution → twin → healing → replay bundle
make web                   # dashboard on :3000 (replay mode, or live if `make serve` is running)
```

Single commands:

```bash
epoch doctor                                   # device, capabilities, LLM, DB
epoch run   --workload triage --strategy agent --budget 60
epoch race  --workload rag --strategies random,tpe,nsga2,agent --budget 36 --seeds 3
epoch heal  --workload triage --scenario drift          # opens an incident, stops at the gate
epoch approve INC-XXXXXX --workload triage              # or --reject
epoch twin-validate --workload triage
epoch export --out web/public/replay/bundle.json
epoch serve                                    # API: REST + SSE + /metrics
```

### Run it on your GPU with Claude

```bash
make setup-gpu                                 # torch, transformers, bitsandbytes, optimum, faiss, sentence-transformers
export ANTHROPIC_API_KEY=sk-ant-...            # hypothesis agent, RCA narrative, rubric judge, triage tier 3
epoch doctor                                   # RAG space now includes Qwen2.5 0.5B/1.5B/3B × fp16/int8/nf4 × eager/compile/ORT
epoch demo --workloads triage,rag --budget 60 --seeds 3
```

`EPOCH_RAG_JUDGE=1` makes the rubric-as-code LLM judge (`configs/rubrics/`) the RAG quality objective.
LLM calls are recorded and replayed with their original latency. Repeated calls during search don't re-bill,
and cost is always reported as if uncached.

### Full stack

```bash
cp .env.example .env && docker compose up --build   # Postgres+pgvector, Redis, MLflow, API, GPU worker, dashboard, Prometheus
```

## How it works

**Genome** (`epoch/genome.py`). A typed, conditional search space. A gene can depend on an earlier one
(`rerank_model` exists only when `reranker=True`). One encoding drives Optuna's samplers, the agent's edits,
the surrogate's features and the UI's genome strips and diffs.

**Evolution engine** (`epoch/evolution/`). One ask/tell loop for all strategies, so the race is fair.
- Constraints use constrained domination.
- **Multi-fidelity rejection**: every candidate first runs on 30% of the eval set, and clear constraint violators or configs dominated by the frontier are pruned. This saved 3.9 CPU-minutes in the demo.
- Hypervolume is computed in a fixed normalised space (log axes for latency and cost), so runs are comparable.
- Lineage is explicit for agent children and nearest-neighbour inferred for NSGA-II (flagged as inferred).

**Hypothesis agent** (`epoch/agents/hypothesis.py`). A LangGraph state machine: `analyze → hypothesize →
materialize → screen`.
- **analyze** builds an evidence pack: frontier, per-objective gene importance, past verdicts, failure clusters from vector memory.
- **hypothesize**: Claude writes one falsifiable hypothesis plus concrete edits of frontier genomes, via forced tool-use and a Pydantic schema. Without a key, a surrogate-guided heuristic fills the same schema.
- **screen**: a random-forest surrogate ranks hypothesis edits and mutation/crossover explorers by optimistic hypervolume gain × P(feasible).
- After the batch runs, each hypothesis gets a verdict (confirmed / refuted / inconclusive) against its parents, and the next round reads it.

**Benchmark engine** (`epoch/benchmark/`).
- Quality: EM/F1, macro-F1, or the judge.
- Latency: p50/p95/p99 per request (a request in a batch waits for the whole batch).
- Throughput and peak memory (CUDA allocator or tracemalloc).
- Cost: `$/1k = 1000·(compute-s × hardware $/h ÷ 3600 + LLM tokens)`, priced from `configs/pricing.yaml`.
- Robustness: the quality ratio on a perturbed split.
- Failed trials are data: classified (OOM, capability, timeout…) and embedded into pgvector.

**Attribution** (`epoch/attribution.py`). fANOVA importance per objective, plus **interventional ablation**:
take the chosen config, apply do(gene := default) with everything else held fixed, and re-measure with 3 repeats.
Effects inside 2σ of repeat noise are flagged. This is what lets the Insights view say that reverting `t1_delta`
buys +0.021 F1 for +3.9 ms of p95.

**Digital twin** (`epoch/twin/`). A SimPy discrete-event simulation of N replicas with dynamic batching, a
bounded FIFO queue, open-loop traffic (Poisson, bursty, diurnal, step) and faults (replica loss, slowdown,
latency spikes).
- Service times are sampled from each genome's **measured** distribution, with a fitted batch-scaling factor α.
- `validate_twin` serves the genome over HTTP, calibrates only capacity (via a saturation run), then compares predicted and measured p50/p95 at three loads.
- The load generator is a raw asyncio HTTP/1.1 client, measured from scheduled send times (no coordinated omission).
- The dashboard runs a TypeScript port of the same simulator in the browser.

**Self-healing** (`epoch/healing/`, `epoch/agents/rca.py`).
- Page's CUSUM on p95, quality and confidence. σ is inflated by 1 + 1/√n for small baselines, which cut the false-alarm rate from 18% to 6%.
- After the alarm, the RCA agent runs `evidence → diagnose → narrate → candidates → canary → twin → gate`:
  - **evidence**: per-stage latency deltas, routing-share shifts, per-tier precision, drift signals (JS divergence, OOV).
  - **candidates**: targeted genome patches, Pareto-archive configs, and an **error-bank retrain** (the CSAI loop). Low-confidence production tickets go to QA, and tier 2 is refit on base + audited data.
  - **canary**: every candidate is replayed 3× on shadow traffic with the fault still active.
  - **twin**: the capacity planner computes replicas needed at production load.
- The gate proposes a fix only when it beats the incumbent beyond measurement noise. After approval, the old config keeps running on identical traffic as a **counterfactual**.
- Outcomes are judged against that counterfactual: resolved, mitigated, no effect, or automatic rollback.

**Serving & ops**. FastAPI micro-batcher (one process = one replica, exactly what the twin models),
Prometheus metrics, OpenTelemetry spans, NVML GPU sampling, Celery + Redis workers, SSE event stream,
MLflow mirror, Docker Compose, GitHub Actions (ruff, mypy, pytest, dashboard build).

**Dashboard** (`web/`). Next.js 15 + TypeScript, React Flow, Recharts and d3-scale. It has eight views:
- Command: live pipeline topology plus an ops-tape replay of the last incident.
- Evolution: Pareto scatter with search replay, the strategy race and the hypervolume curves.
- Lineage.
- Trials, with an inspector.
- Insights: fANOVA, ablations, the hypothesis notebook and failure memory.
- Digital twin: in-browser simulation plus reality validation.
- Incidents: CUSUM, RCA, candidates, and a replayable approval gate.
- Architecture.

It reads one data contract, either the static replay bundle (deploy `web/` to Vercel) or the live API.

## Honest scope

- The CPU demo exercises the triage workload fully (including ONNX Runtime and int8 quantisation) and the RAG
  workload in its CPU form (BM25 + extractive reader). The GPU paths (bitsandbytes, `torch.compile`, ORT
  generation, dense retrieval, cross-encoders) are implemented and capability-gated, but have not been
  benchmarked in this repo's recorded run.
- Triage tickets and the RAG corpus are synthetic and deterministic, with held-out templates, label noise and
  code-mixed text. Point `EPOCH_RAG_CORPUS` / `EPOCH_RAG_QA` at your own JSONL to use real data.
- Healing scenarios inject faults into windows of replayed traffic ("minutes" are virtual windows). The QA
  audit uses ground-truth labels to stand in for human reviewers.
- Approvals in the recorded run were scripted (`actor: epoch demo (scripted approval)`). In live mode the
  dashboard's Approve / Reject buttons call the API.

## Layout

```
epoch/
  genome.py objectives.py config.py cli.py demo.py export.py serving.py observability.py attribution.py
  benchmark/   harness, timing, resources (device, memory), metrics
  evolution/   engine (ask/tell, rejection, lineage), pareto (HV), surrogate (RF EHVI), race
  agents/      llm (Claude, tool-use, record/replay), hypothesis (LangGraph), rca (LangGraph), judge (rubric)
  twin/        sim (SimPy), profile, validate (HTTP load test)
  healing/     detector (CUSUM), drift, loop (scenarios, gate, counterfactual, rollback)
  memory/      store (SQLAlchemy, pgvector), embed, provenance
  workloads/   triage/ (cascade + data), rag/ (corpus, retrieval, generation), synthetic (tests)
  api/ worker.py
web/           Next.js dashboard (views/, components/, lib/twin.ts)
configs/       pricing, rubrics
infra/         Dockerfiles, Prometheus
tests/         genome, pareto/HV, engine, twin (M/M/1 check), detector, triage, API
```

## Author

**Lakshay Sharma** · B.Tech, NSUT Delhi · [GitHub](https://github.com/lakshaysharmaug22-crypto)

## License

MIT
