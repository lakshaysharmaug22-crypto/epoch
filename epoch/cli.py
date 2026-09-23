"""EPOCH command line."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(add_completion=False, no_args_is_help=True, help="EPOCH — evolve, simulate and heal AI pipelines.")
console = Console()


@app.command()
def doctor() -> None:
    """Show hardware, capabilities, LLM and storage configuration."""
    from epoch.benchmark.resources import device_info, hw_price_per_hour
    from epoch.config import settings
    from epoch.workloads import available_workloads, get_workload

    s = settings()
    info = device_info()
    t = Table(title="EPOCH doctor", show_header=False)
    t.add_row("device", f"{info['device']}  {info.get('gpu') or info.get('cpu')}  ({info.get('gpu_mem_gb') or '-'} GB)")
    t.add_row("torch / cuda", f"{info.get('torch')} / {info.get('cuda')}")
    t.add_row("capabilities", json.dumps(info["capabilities"]))
    t.add_row("hardware $/h", f"{hw_price_per_hour():.3f}")
    t.add_row("LLM agents", f"{'Claude · ' + s.agent_model if s.llm_enabled else 'heuristic (set ANTHROPIC_API_KEY)'}")
    t.add_row("database", s.resolved_db_url)
    for w in available_workloads():
        wl = get_workload(w)
        t.add_row(f"workload {w}", f"{len(wl.space())} genes · {json.dumps(wl.capabilities())}")
    console.print(t)


@app.command()
def run(workload: str = "triage", strategy: str = "agent", budget: int = 60, seed: int = 0,
        population: int = 12) -> None:
    """Run one evolution study."""
    from epoch.evolution import EngineConfig, Evolution
    from epoch.memory import Store
    from epoch.workloads import get_workload

    evo = Evolution(get_workload(workload), strategy, Store(), EngineConfig(budget=budget, seed=seed, population=population,
                                                                            warmup=population),
                    sink=lambda k, p: console.print(f"[dim]{k}[/] {p}") if k in ("trial", "hypothesis") else None)
    s = evo.run()
    console.print({k: v for k, v in s.items() if k != "hv_curve"})


@app.command("race")
def race_cmd(workload: str = "triage", strategies: str = "random,tpe,nsga2,agent", budget: int = 60, seeds: int = 3) -> None:
    """Head-to-head strategy race with equal budgets."""
    from epoch.evolution.race import race
    from epoch.memory import Store
    from epoch.workloads import get_workload

    res = race(get_workload(workload), strategies.split(","), budget, list(range(seeds)), Store())
    t = Table(title=f"race · {workload}")
    for c in ("strategy", "median final HV", "trials→target", "reached", "trials→NSGA-II final"):
        t.add_column(c)
    for s, v in res["strategies"].items():
        t.add_row(s, f"{v['final_hv_median']:.4f}", f"{v['trials_to_target_median']:.0f}", f"{v['reached_target']}/{v['runs']}",
                  str(v["trials_to_nsga2_final_median"]))
    console.print(t)
    console.print(res.get("headline"))


@app.command()
def heal(workload: str = "triage", scenario: str = "drift", auto_approve: bool = False) -> None:
    """Run a fault scenario against the current knee config and open an incident."""
    from epoch.export import workload_bundle
    from epoch.healing import Healer, scenario_for
    from epoch.memory import Store
    from epoch.workloads import get_workload

    store = Store()
    wl = get_workload(workload)
    wb = workload_bundle(store, workload)
    if not wb["knee"]:
        raise typer.BadParameter("run `epoch race` first — healing needs a Pareto archive")
    knee = next(t for t in wb["trials"] if t["id"] == wb["knee"])
    archive = [{"trial_id": t["id"], "genome": t["genome"], "metrics": t["metrics"]} for t in wb["trials"] if t["global_pareto"]]
    h = Healer(wl, scenario_for(wl, scenario), store, knee["genome"], archive=archive)
    data = h.run(approval="auto" if auto_approve else "manual", actor="cli --auto-approve")
    console.print(f"[bold]{h.incident_id}[/] → {store.incident(h.incident_id)['status']}")
    console.print(data.get("narrative", {}).get("summary"))
    if data.get("proposal", {}).get("status") == "awaiting_approval" and not auto_approve:
        console.print(f"approve with: [bold]epoch approve {h.incident_id} --workload {workload}[/]")


@app.command("approve")
def approve_cmd(incident_id: str, workload: str = "triage", reject: bool = False, note: str = "", actor: str = "cli") -> None:
    """Approve (or --reject) a remediation waiting at the human gate."""
    from epoch.healing import approve
    from epoch.memory import Store
    from epoch.workloads import get_workload

    data = approve(Store(), get_workload(workload), incident_id, approved=not reject, actor=actor, note=note)
    console.print(data.get("outcome") or data.get("decision"))


@app.command("twin-validate")
def twin_validate(workload: str = "triage", duration: float = 20.0) -> None:
    """Load-test the knee config over HTTP and compare with the twin's prediction."""
    from epoch.export import workload_bundle
    from epoch.memory import Store
    from epoch.twin.validate import validate_twin

    wb = workload_bundle(Store(), workload)
    knee = next(t for t in wb["trials"] if t["id"] == wb["knee"])
    v = validate_twin(workload, knee["genome"], duration_s=duration)
    for lv in v["levels"]:
        console.print(lv)
    console.print({"mape_p50": v["mape_p50"], "mape_p95": v["mape_p95"]})


@app.command()
def export(out: Path = Path("web/public/replay/bundle.json"), workloads: str = "triage,rag") -> None:
    """Write the dashboard replay bundle from experiment memory."""
    from epoch.demo import load_extras
    from epoch.export import export_bundle
    from epoch.memory import Store

    ws = workloads.split(",")
    p = export_bundle(Store(), ws, out, extras={w: load_extras(w) for w in ws})
    console.print(f"wrote {p} ({p.stat().st_size / 1e6:.2f} MB)")


@app.command()
def demo(workloads: str = "triage,rag", budget: int = 60, seeds: int = 3, strategies: str = "random,tpe,nsga2,agent",
         out: Path = Path("web/public/replay/bundle.json"), rag_budget: int = 36, no_heal: bool = False,
         no_validate: bool = False) -> None:
    """Full pipeline: race → attribution → twin → self-healing → replay bundle."""
    from epoch.demo import run_demo

    run_demo(workloads.split(","), budget, seeds, strategies.split(","), out, heal=not no_heal,
             validate=not no_validate, rag_budget=rag_budget)


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start the EPOCH API (dashboard live mode)."""
    import uvicorn

    uvicorn.run("epoch.api.main:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    app()
