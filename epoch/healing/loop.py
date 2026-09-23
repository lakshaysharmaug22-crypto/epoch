"""Self-healing loop: serve a deployed genome on windows of production-like traffic, inject a fault, detect it with
CUSUM, run the RCA agent (evidence → diagnosis → candidates → canary → twin → gate), wait for approval, promote,
and keep measuring both the promoted config and the counterfactual (old config on identical traffic).
Automatic rollback if the promoted config underperforms the counterfactual."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from epoch.agents.rca import RCAAgent
from epoch.genome import Genome, genome_id
from epoch.healing.detector import Cusum
from epoch.healing.drift import char_hist, js_divergence, oov_rate, vocab_of
from epoch.memory.store import Store
from epoch.twin.profile import measure_profile
from epoch.twin.sim import Traffic, TwinConfig, simulate
from epoch.workloads.base import Workload


@dataclass
class Scenario:
    name: str
    title: str
    kind: Literal["drift", "latency"]
    windows: int = 36
    window_size: int = 80
    fault_at: int = 12
    ramp: int = 3
    stage: str | None = None
    spike_ms: float = 0.0
    drift_mix: float = 0.6
    evidence_windows: int = 2  # keep observing after the alarm before diagnosing
    shadow_windows: int = 2  # candidates are canaried in shadow on these live windows before the gate
    audit_budget: int = 80  # low-confidence requests sent to human QA for labels (error bank)
    split_base: str = "test"
    split_drift: str = "drift"
    seed: int = 0
    minutes_per_window: float = 1.0


def scenario_for(workload: Workload, name: str, **overrides: Any) -> Scenario:
    if workload.name == "rag":
        presets = {
            "drift": Scenario("drift", "Question phrasing drift (typos, abbreviations)", "drift", split_drift="robust"),
            "latency": Scenario("latency", "Retriever dependency latency spike", "latency", stage="retrieve", spike_ms=15.0),
        }
    else:
        presets = {
            "drift": Scenario("drift", "Code-mixed phrasing drift in partner tickets", "drift"),
            "latency": Scenario("latency", "Tier-2 model-server latency spike", "latency", stage="tier2", spike_ms=12.0),
            "resolver": Scenario("resolver", "Tier-3 resolver dependency latency spike", "latency", stage="tier3",
                                 spike_ms=40.0),
        }
    base = presets[name]
    return Scenario(**{**asdict(base), **overrides})


class Healer:
    def __init__(self, workload: Workload, scenario: Scenario, store: Store, deployed: Genome, *,
                 archive: list[dict] | None = None, sink=None, incident_id: str | None = None):
        self.w, self.sc, self.store = workload, scenario, store
        self.deployed = deployed
        self.archive = archive or []
        self.sink = sink or (lambda k, p: None)
        self.pipes: dict[str, Any] = {}
        self.q = workload.quality_metric
        self.incident_id = incident_id or f"INC-{uuid.uuid4().hex[:6].upper()}"
        workload.prepare()
        base_x, _ = workload.dataset("train", 1.0, 0)
        if not base_x:
            base_x, _ = workload.dataset("val", 1.0, 0)
        self.vocab = vocab_of(base_x)
        self._base_pool = workload.dataset(scenario.split_base, 1.0, 0)
        self._drift_pool = workload.dataset(scenario.split_drift, 1.0, 0)
        ref_x = [x for w in range(2, scenario.fault_at) for x in self.window_data(w)[0]]
        self.ref_hist = char_hist(ref_x)

    # ------------------------------------------------------------------ traffic
    def window_data(self, w: int) -> tuple[list, list, float, dict[str, float]]:
        sc = self.sc
        rng = np.random.default_rng(sc.seed * 10_000 + w)
        ramp = min(1.0, (w - sc.fault_at + 1) / sc.ramp) if w >= sc.fault_at else 0.0
        frac = sc.drift_mix * ramp if sc.kind == "drift" else 0.0
        n_d = round(frac * sc.window_size)
        bx, by = self._base_pool
        dx, dy = self._drift_pool
        bi = rng.choice(len(bx), sc.window_size - n_d, replace=False)
        di = rng.choice(len(dx), n_d, replace=False) if n_d else np.array([], dtype=int)
        rows = [(bx[i], by[i]) for i in bi] + [(dx[i], dy[i]) for i in di]
        order = rng.permutation(len(rows))
        xs, ys = [rows[i][0] for i in order], [rows[i][1] for i in order]
        env = {sc.stage: sc.spike_ms * ramp} if sc.kind == "latency" and ramp > 0 and sc.stage else {}
        return xs, ys, frac, env

    # ------------------------------------------------------------------ measurement
    def pipe(self, g: Genome):
        gid = genome_id(g)
        if gid not in self.pipes:
            self.pipes[gid] = self.w.build(g)
        return self.pipes[gid]

    def evaluate(self, g: Genome, xs: list, ys: list, env: dict[str, float]) -> dict[str, Any]:
        p = self.pipe(g)
        if hasattr(p, "inject_ms"):
            p.inject_ms = dict(env)
        bs = max(1, self.w.batch_size(g))
        preds, lat = [], []
        for i in range(0, len(xs), bs):
            t0 = time.perf_counter()
            out = p.predict_batch(xs[i : i + bs])
            dt = (time.perf_counter() - t0) * 1e3
            preds.extend(out)
            lat.extend([dt] * len(out))
        m: dict[str, Any] = dict(self.w.score(preds, ys))
        a = np.asarray(lat)
        m.update(p50_ms=float(np.percentile(a, 50)), p95_ms=float(np.percentile(a, 95)), p99_ms=float(np.percentile(a, 99)),
                 mean_ms=float(a.mean()))
        stage: dict[str, float] = {}
        for pr in preds:
            for k, v in pr.stages_ms.items():
                stage[k] = stage.get(k, 0.0) + v / len(preds)
        m["stage_ms"] = stage
        return m

    def drift_features(self, xs: list) -> dict[str, float]:
        return {"drift_js": js_divergence(char_hist(xs), self.ref_hist), "oov_rate": oov_rate(xs, self.vocab)}

    # ------------------------------------------------------------------ main
    def run(self, approval: Literal["auto", "manual"] = "manual", actor: str = "auto") -> dict[str, Any]:
        sc = self.sc
        self.store.deploy(workload=self.w.name, genome_id=genome_id(self.deployed), genome=self.deployed,
                          reason=f"scenario `{sc.name}` start")
        base_prof = measure_profile(self.w, {**self.deployed, **({"batch_size": 1} if "batch_size" in self.deployed else {})},
                                    n=150, inputs=self.window_data(0)[0])
        bs = int(self.deployed.get("batch_size", 1))
        prod_rps = 0.5 * base_prof.capacity_rps(1, bs)  # production runs at ~50% of the incumbent's healthy capacity
        base_twin = simulate(base_prof, TwinConfig(max_batch=bs, max_wait_ms=5.0 if bs > 1 else 0.0),
                             Traffic("bursty", rate=prod_rps, duration_s=60, burst_factor=1.8, burst_every_s=20, burst_len_s=5))
        twin_slo = 2.0 * (base_twin["summary"]["p95"] or 1.0)
        dets = [Cusum("p95_ms", "up"), Cusum(self.q, "down")]
        series: list[dict] = []
        timeline: list[dict] = [{"w": 0, "event": "deploy", "detail": f"genome {genome_id(self.deployed)} serving"}]
        alarm_w = None
        for w in range(sc.windows):
            xs, ys, frac, env = self.window_data(w)
            m = self.evaluate(self.deployed, xs, ys, env)
            m.update(self.drift_features(xs))
            if w == sc.fault_at:
                timeline.append({"w": w, "event": "fault", "detail": sc.title})
            if w == sc.fault_at - 1:
                dets = [d for d in dets if all(s["live"].get(d.metric) is not None for s in series[2:])]
                if "mean_conf" in m:
                    dets.append(Cusum("mean_conf", "down"))
                for d in dets:
                    d.fit([s["live"][d.metric] for s in series[2:]] + [m[d.metric]])
            stats, fired = {}, []
            if w >= sc.fault_at:
                for d in dets:
                    s, alarm = d.update(m[d.metric])
                    stats[d.metric] = s
                    if alarm:
                        fired.append(d.metric)
            series.append({"w": w, "t_min": w * sc.minutes_per_window, "live": m, "counterfactual": None,
                           "drift_frac": frac, "fault": env, "cusum": stats,
                           "phase": "baseline" if w < sc.fault_at else "fault"})
            self.sink("healing.window", {"incident": self.incident_id, "w": w, "p95_ms": m["p95_ms"], self.q: m[self.q]})
            if fired and alarm_w is None:
                alarm_w = w
                timeline.append({"w": w, "event": "alarm", "detail": "CUSUM fired on " + ", ".join(fired)})
                break
        if alarm_w is None:
            detectors = {d.metric: {"mu": d.mu, "sigma": d.sigma, "k": d.k, "h": d.h, "direction": d.direction,
                                    "history": d.history} for d in dets}
            data = {"scenario": asdict(sc), "series": series, "timeline": timeline, "detectors": detectors,
                    "deployed": self.deployed, "outcome": "not_detected"}
            self.store.upsert_incident(self.incident_id, workload=self.w.name, status="not_detected",
                                       title=f"{sc.title} (not detected)", severity="sev3", data=data)
            return data

        # 1) keep serving and collecting evidence for a few windows after the alarm
        for w in range(alarm_w + 1, min(sc.windows, alarm_w + 1 + sc.evidence_windows)):
            xs, ys, frac, env = self.window_data(w)
            m = self.evaluate(self.deployed, xs, ys, env)
            m.update(self.drift_features(xs))
            stats = {d.metric: d.update(m[d.metric])[0] for d in dets}
            series.append({"w": w, "t_min": w * sc.minutes_per_window, "live": m, "counterfactual": None,
                           "drift_frac": frac, "fault": env, "cusum": stats, "phase": "incident"})
        rca_w = series[-1]["w"]
        base_ws = [s["live"] for s in series[2 : sc.fault_at]]
        inc_ws = [s["live"] for s in series if s["w"] >= max(sc.fault_at, alarm_w - 1)]
        bq = float(np.mean([x[self.q] for x in base_ws]))
        bp = float(np.mean([x["p95_ms"] for x in base_ws]))
        floor = next((c.threshold for c in self.w.constraints if c.metric == self.q and c.op == ">="), -np.inf)
        targets = {self.q: max(bq - 0.03, floor), "p95_ms": max(bp * 1.25, bp + 0.5)}

        # 2) error bank: lowest-confidence requests from the incident windows go to human QA for labels
        adapt = None
        audited: list = []
        pool = []
        for ww in range(sc.fault_at, rca_w + 1):
            x, y, _, _env_w = self.window_data(ww)
            preds = self.pipe(self.deployed).predict_batch(x)
            pool += [(p.meta.get("conf", 1.0), xi, yi) for p, xi, yi in zip(preds, x, y, strict=True)]
        pool.sort(key=lambda t: t[0])
        audited = [(xi, yi) for _, xi, yi in pool[: sc.audit_budget]]
        tag = f"bank-{self.incident_id.lower()}"
        if self.w.adapt(self.deployed, audited, tag) is not None:
            adapt = lambda g: self.w.adapt(g, audited, tag)  # noqa: E731

        # 3) shadow canary data = the next live windows (incumbent keeps serving them)
        cx, cy = [], []
        shadow = list(range(rca_w + 1, min(sc.windows, rca_w + 1 + sc.shadow_windows)))
        for ww in shadow:
            x, y, frac, env = self.window_data(ww)
            cx += x
            cy += y
            m = self.evaluate(self.deployed, x, y, env)
            m.update(self.drift_features(x))
            series.append({"w": ww, "t_min": ww * sc.minutes_per_window, "live": m, "counterfactual": None,
                           "drift_frac": frac, "fault": env, "cusum": {d.metric: d.update(m[d.metric])[0] for d in dets},
                           "phase": "shadow"})
        gate_w = series[-1]["w"]
        ctx = {"workload": self.w, "incumbent": self.deployed, "baseline_windows": base_ws, "incident_windows": inc_ws,
               "canary_inputs": cx, "canary_labels": cy, "fault_env": env, "evaluate_window": self.evaluate,
               "archive": self.archive, "targets": targets, "store": self.store, "incident_id": self.incident_id,
               "alarms": {d.metric: d.history[-1] if d.history else None for d in dets}, "adapt": adapt,
               "audited": len(audited), "prod_rps": prod_rps, "twin_slo_ms": twin_slo}
        detectors = {d.metric: {"mu": d.mu, "sigma": d.sigma, "k": d.k, "h": d.h, "direction": d.direction,
                                "history": d.history} for d in dets}
        t0 = time.perf_counter()
        rca = RCAAgent(ctx).run()
        rca_s = time.perf_counter() - t0
        timeline.append({"w": rca_w, "event": "rca", "detail": rca["causes"][0]["title"], "seconds": rca_s})
        if adapt:
            timeline.append({"w": rca_w, "event": "audit", "detail": f"{len(audited)} low-confidence requests labelled by QA → error bank `{tag}`"})
        timeline.append({"w": gate_w, "event": "canary", "detail": f"{len(rca['canary']) - 1} candidates shadow-evaluated on {len(cx)} live requests"})
        prop = rca["proposal"]
        status = prop["status"]
        timeline.append({"w": gate_w, "event": "proposal" if status == "awaiting_approval" else "escalated",
                         "detail": prop.get("why", "no candidate restores targets")})
        if status == "no_candidate":
            status = "escalated"
        title = f"{rca['causes'][0]['title']} — {self.w.title}"
        sev = "sev1" if (bq - inc_ws[-1][self.q]) > 0.08 or inc_ws[-1]["p95_ms"] > 3 * bp else "sev2"
        data = {"scenario": asdict(sc), "series": series, "timeline": timeline, "detectors": detectors,
                "deployed": self.deployed, "alarm_w": alarm_w, "rca_w": rca_w, "gate_w": gate_w, "targets": targets,
                "audited": len(audited), "bank": tag if adapt else None,
                "baseline": {self.q: bq, "p95_ms": bp}, "evidence": rca["evidence"], "causes": rca["causes"],
                "narrative": rca["narrative"], "candidates": rca["canary"], "twin": rca["twin"], "proposal": prop,
                "prod_rps": prod_rps, "twin_slo_ms": twin_slo, "rca_seconds": rca_s, "archive_size": len(self.archive)}
        self.store.upsert_incident(self.incident_id, workload=self.w.name, status=status, title=title, severity=sev, data=data)
        self.sink("incident", {"id": self.incident_id, "status": status, "title": title})
        if status == "awaiting_approval" and approval == "auto":
            return self.resume(approved=True, actor=actor)
        return data

    def resume(self, approved: bool, actor: str = "human", note: str = "") -> dict[str, Any]:
        inc = self.store.incident(self.incident_id)
        data = inc["data"]
        sc = self.sc
        prop = data["proposal"]
        timeline = data["timeline"]
        series = data["series"]
        alarm_w = data["alarm_w"]
        gate_w = data.get("gate_w", alarm_w)
        targets = data["targets"]
        if not approved:
            timeline.append({"w": gate_w, "event": "rejected", "detail": f"by {actor}. {note}".strip()})
            data["decision"] = {"approved": False, "actor": actor, "note": note}
            self.store.upsert_incident(self.incident_id, workload=self.w.name, status="rejected", title=inc["title"],
                                       severity=inc["severity"], data=data)
            return data
        new = prop["genome"]
        old = data["deployed"]
        start = gate_w + 1
        timeline.append({"w": start, "event": "approved", "detail": f"by {actor}. {note}".strip()})
        self.store.deploy(workload=self.w.name, genome_id=genome_id(new), genome=new,
                          reason=f"remediation for {self.incident_id}", incident_id=self.incident_id)
        timeline.append({"w": start, "event": "promoted", "detail": f"genome {genome_id(new)} ({prop['origin']})"})
        live_g, status, ok_run, bad_run, recovered_w = new, "monitoring", 0, 0, None
        for w in range(start, sc.windows):
            xs, ys, frac, env = self.window_data(w)
            live = self.evaluate(live_g, xs, ys, env)
            live.update(self.drift_features(xs))
            cf = self.evaluate(old, xs, ys, env)
            meets = live[self.q] >= targets[self.q] and live["p95_ms"] <= targets["p95_ms"]
            worse = live[self.q] < cf[self.q] - 0.02 and live["p95_ms"] > cf["p95_ms"]
            series.append({"w": w, "t_min": w * sc.minutes_per_window, "live": live, "counterfactual": cf,
                           "drift_frac": frac, "fault": env, "cusum": {}, "phase": "remediated" if live_g == new else "rolled_back"})
            if w >= start + 2 and status == "monitoring":
                ok_run = ok_run + 1 if meets else 0
                bad_run = bad_run + 1 if (not meets and worse) else 0
                if ok_run >= 3:
                    status, recovered_w = "resolved", w
                    timeline.append({"w": w, "event": "recovered", "detail": "targets met for 3 consecutive windows"})
                elif bad_run >= 3:
                    status, live_g = "rolled_back", old
                    timeline.append({"w": w, "event": "rollback", "detail": "promoted config worse than counterfactual"})
                    self.store.deploy(workload=self.w.name, genome_id=genome_id(old), genome=old,
                                      reason=f"rollback of {self.incident_id}", incident_id=self.incident_id)
        post = [s for s in series if s["w"] >= start + 2 and s["counterfactual"]]
        if post and status != "rolled_back":
            lq = float(np.mean([s["live"][self.q] for s in post]))
            cq = float(np.mean([s["counterfactual"][self.q] for s in post]))
            lp = float(np.mean([s["live"]["p95_ms"] for s in post]))
            cp = float(np.mean([s["counterfactual"]["p95_ms"] for s in post]))
            lm = float(np.mean([s["live"]["mean_ms"] for s in post]))
            cm = float(np.mean([s["counterfactual"]["mean_ms"] for s in post]))
            better = ((lq > cq + 0.01 and lp < cp * 1.25) or (lp < 0.9 * cp and lq > cq - 0.02)
                      or (lm < 0.85 * cm and lq > cq - 0.02))  # mean service time = capacity
            if not better:  # matching the counterfactual means the change did nothing — don't claim a fix
                status = "no_effect"
                timeline.append({"w": series[-1]["w"], "event": "no_effect", "detail": "promoted config no better than the counterfactual"})
            elif status != "resolved":
                status = "mitigated"
        elif status == "monitoring":
            status = "mitigated"
        data.update({
            "series": series, "timeline": timeline, "decision": {"approved": True, "actor": actor, "note": note},
            "promoted_w": start, "recovered_w": recovered_w,
            "outcome": {
                "status": status,
                "detection_delay_windows": alarm_w - sc.fault_at,
                "time_to_mitigate_windows": start - sc.fault_at,
                "live_mean": {k: float(np.mean([s["live"][k] for s in post])) if post else None
                              for k in (self.q, "p95_ms", "mean_ms")},
                "counterfactual_mean": {k: float(np.mean([s["counterfactual"][k] for s in post])) if post else None
                                        for k in (self.q, "p95_ms", "mean_ms")},
            },
        })
        self.store.upsert_incident(self.incident_id, workload=self.w.name, status=status, title=inc["title"],
                                   severity=inc["severity"], data=data)
        self.sink("incident", {"id": self.incident_id, "status": status})
        return data


def approve(store: Store, workload: Workload, incident_id: str, *, approved: bool = True, actor: str = "human",
            note: str = "") -> dict[str, Any]:
    inc = store.incident(incident_id)
    if inc is None:
        raise KeyError(incident_id)
    if inc["status"] != "awaiting_approval":
        raise ValueError(f"incident {incident_id} is {inc['status']}, not awaiting approval")
    sc = Scenario(**inc["data"]["scenario"])
    h = Healer(workload, sc, store, inc["data"]["deployed"], incident_id=incident_id)
    return h.resume(approved=approved, actor=actor, note=note)
