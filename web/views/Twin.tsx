"use client";

import { useDeferredValue, useMemo, useState } from "react";
import { TwinSeries } from "@/components/charts/Charts";
import { KeyVal, Panel, Pill, Segmented, Stat, Tip } from "@/components/ui";
import { useWorkloadData } from "@/lib/derive";
import { fmtMs, fmtNum, fmtObjective, fmtPct } from "@/lib/format";
import { C, S_LABEL } from "@/lib/theme";
import { capacity, defaultTraffic, simulate, type Fault, type Traffic } from "@/lib/twin";
import type { ServiceProfile } from "@/lib/types";

export function TwinView() {
  const d = useWorkloadData();
  const twin = d?.wb.twin;
  const profiles = useMemo(() => {
    if (!twin) return [] as { id: string; p: ServiceProfile; label: string }[];
    const list = [{ id: "knee", p: twin.knee_profile, label: "Knee (deployed)" }];
    for (const [id, p] of Object.entries(twin.profiles)) {
      const t = p.trial_id ? d?.byId.get(p.trial_id) : undefined;
      list.push({ id, p, label: t ? `${S_LABEL[t.strategy]} #${t.number} · ${fmtObjective(d!.quality, t.metrics[d!.quality.name])}` : id });
    }
    return list;
  }, [twin, d]);

  const [pid, setPid] = useState("knee");
  const prof = profiles.find((x) => x.id === pid)?.p ?? profiles[0]?.p;
  const [load, setLoad] = useState(0.3);
  const [kind, setKind] = useState<Traffic["kind"]>("bursty");
  const [replicas, setReplicas] = useState(1);
  const [batch, setBatch] = useState<number | null>(null);
  const [wait, setWait] = useState(5);
  const [faults, setFaults] = useState({ replica: false, slow: false, spike: false });
  const [slo, setSlo] = useState(50);

  const bs = batch ?? prof?.batch_size ?? 1;
  const cap1 = prof ? capacity(prof, 1, bs) : 1;
  const rate = load * cap1;
  const inputs = useDeferredValue({ prof, rate, kind, replicas, bs, wait, faults, slo });

  const result = useMemo(() => {
    const { prof, rate, kind, replicas, bs, wait, faults, slo } = inputs;
    if (!prof) return null;
    const tr: Traffic = { ...defaultTraffic(rate, 90), kind };
    const fl: Fault[] = [];
    if (faults.replica) fl.push({ kind: "replica_down", start_s: 30, duration_s: 20, magnitude: 1, replica: Math.min(1, replicas - 1) });
    if (faults.slow) fl.push({ kind: "slowdown", start_s: 45, duration_s: 15, magnitude: 2.5, replica: 0 });
    if (faults.spike) fl.push({ kind: "latency_spike", start_s: 60, duration_s: 10, magnitude: 25, replica: 0 });
    return simulate(prof, { replicas, max_batch: bs, max_wait_ms: bs > 1 ? wait : 0, queue_cap: 5000, slo_ms: slo }, tr, fl, 1);
  }, [inputs]);

  if (!d) return null;
  if (!twin || !prof) return <Panel title="No twin data"><p className="text-[12.5px] text-muted">Run `epoch demo` to measure service profiles.</p></Panel>;
  const s = result?.summary;
  const val = twin.validation;
  const rows = (result?.series ?? []).map((p) => ({ t: p.t, p50: p.p50, p95: p.p95, p99: p.p99, queue: p.queue, util: p.util * 100, rps_in: p.rps_in, rps_out: p.rps_out }));
  const sloOk = s?.p95 != null && s.p95 <= slo;

  return (
    <div className="grid gap-4 xl:grid-cols-[320px_minmax(0,1fr)]">
      <Panel tour="twin-config" eyebrow="Scenario" title="Configure the twin">
        <div className="flex flex-col gap-4 text-[12.5px]">
          <label className="flex flex-col gap-1.5">
            <span className="text-muted">Genome (measured service profile)</span>
            <select id="twin-profile" value={pid} onChange={(e) => { setPid(e.target.value); setBatch(null); }} className="rounded-md border border-line bg-sunken px-2 py-1.5 text-ink">
              {profiles.map((x) => <option key={x.id} value={x.id}>{x.label}</option>)}
            </select>
            <span className="num text-[11px] text-muted">{prof.samples_ms.length} quantiles · α = {prof.alpha.toFixed(3)} · mean {fmtMs(prof.samples_ms.reduce((a, b) => a + b, 0) / prof.samples_ms.length)}</span>
          </label>
          <div className="flex flex-col gap-1.5">
            <span className="text-muted">Traffic shape</span>
            <Segmented size="xs" value={kind} onChange={setKind} options={[{ value: "poisson", label: "Poisson" }, { value: "bursty", label: "Bursty 3×" }, { value: "diurnal", label: "Diurnal" }, { value: "step", label: "Step 2×" }]} />
          </div>
          <label className="flex flex-col gap-1.5">
            <span className="flex justify-between text-muted"><span>Offered load</span><span className="num text-ink">{fmtNum(rate, 0)} req/s · {Math.round(load * 100)}% of 1 replica</span></span>
            <input id="twin-load" type="range" min={0.05} max={1.6} step={0.05} value={load} onChange={(e) => setLoad(Number(e.target.value))} className="accent-[#e11d48]" />
          </label>
          <div className="grid grid-cols-2 gap-3">
            <label className="flex flex-col gap-1.5">
              <span className="text-muted">Replicas</span>
              <Segmented size="xs" value={String(replicas)} onChange={(v) => setReplicas(Number(v))} options={[1, 2, 3, 4].map((n) => ({ value: String(n), label: String(n) }))} />
            </label>
            <label className="flex flex-col gap-1.5">
              <span className="text-muted">Max batch</span>
              <select id="twin-batch" value={bs} onChange={(e) => setBatch(Number(e.target.value))} className="rounded-md border border-line bg-sunken px-2 py-1 text-ink">
                {[1, 4, 8, 16, 32, 64].map((b) => <option key={b} value={b}>{b}</option>)}
              </select>
            </label>
          </div>
          <label className="flex flex-col gap-1.5">
            <span className="flex justify-between text-muted"><span>Batch wait</span><span className="num text-ink">{wait} ms</span></span>
            <input id="twin-wait" type="range" min={0} max={30} step={1} value={wait} disabled={bs <= 1} onChange={(e) => setWait(Number(e.target.value))} className="accent-[#e11d48] disabled:opacity-40" />
          </label>
          <label className="flex flex-col gap-1.5">
            <span className="flex justify-between text-muted"><span>p95 SLO</span><span className="num text-ink">{slo} ms</span></span>
            <input id="twin-slo" type="range" min={5} max={500} step={5} value={slo} onChange={(e) => setSlo(Number(e.target.value))} className="accent-[#e11d48]" />
          </label>
          <fieldset className="flex flex-col gap-2 border-t border-line pt-3">
            <legend className="eyebrow mb-1">Fault injection</legend>
            {([
              ["replica", "Lose a replica at 30 s for 20 s"],
              ["slow", "Dependency slowdown 2.5× at 45 s"],
              ["spike", "+25 ms latency spike at 60 s"],
            ] as const).map(([k, label]) => (
              <label key={k} className="flex cursor-pointer items-center gap-2 text-ink-2">
                <input id={`fault-${k}`} type="checkbox" checked={faults[k]} onChange={(e) => setFaults((f) => ({ ...f, [k]: e.target.checked }))} className="accent-[#e11d48]" />
                {label}
              </label>
            ))}
          </fieldset>
          <p className="text-[11px] leading-relaxed text-muted">Runs entirely in your browser: an event-driven port of the SimPy twin, sampling service times from the genome&apos;s measured distribution.</p>
        </div>
      </Panel>

      <div className="grid content-start gap-4">
        <div className="panel grid grid-cols-2 gap-5 p-5 md:grid-cols-3 2xl:grid-cols-6">
          <Stat label="p95 latency" value={fmtMs(s?.p95)} tone={sloOk ? undefined : C.critical} sub={<span>{sloOk ? "within" : "breaches"} {slo} ms SLO</span>} />
          <Stat label="p50 · p99" value={<span className="text-[18px]">{fmtMs(s?.p50)} · {fmtMs(s?.p99)}</span>} />
          <Stat label="Throughput" value={`${fmtNum(s?.throughput_rps, 0)}/s`} sub={`offered ${fmtNum(s?.offered_rps, 0)}/s`} />
          <Stat label="Utilisation" value={fmtPct(s?.utilization, 0)} sub={`mean batch ${fmtNum(s?.mean_batch, 2)}`} />
          <Stat label="SLO attainment" value={fmtPct(s?.slo_attainment, 1)} />
          <Stat label="Queue" value={s && s.drop_rate > 0 ? `dropping ${fmtPct(s.drop_rate, 1)}` : s?.stable ? "stable" : "growing"}
            tone={s && s.drop_rate === 0 && s.stable ? C.ok : C.critical} sub={`max depth ${fmtNum(s?.max_queue, 0)}`} />
        </div>

        <Panel eyebrow="Simulated 90 s · 1 s buckets" title="Latency percentiles">
          <TwinSeries rows={rows} keys={[{ key: "p99", label: "p99", color: C.violet, dashed: true }, { key: "p95", label: "p95", color: C.crimsonHi }, { key: "p50", label: "p50", color: C.ink2 }]} height={220} log />
        </Panel>
        <div className="grid gap-4 lg:grid-cols-2">
          <Panel eyebrow="Traffic" title="Offered vs served (req/s)">
            <TwinSeries rows={rows} keys={[{ key: "rps_in", label: "offered", color: C.info, area: true }, { key: "rps_out", label: "served", color: C.ink }]} yFormat={(v) => fmtNum(v, 0)} height={170} />
          </Panel>
          <Panel eyebrow="Saturation" title="Queue depth and utilisation">
            <TwinSeries rows={rows} keys={[{ key: "queue", label: "queue depth", color: C.warn, area: true }, { key: "util", label: "utilisation %", color: C.ink2 }]} yFormat={(v) => fmtNum(v, 0)} height={170} />
          </Panel>
        </div>

        {val && !val.error && (
          <Panel tour="twin-validation" eyebrow={`Twin vs reality · real HTTP service + open-loop Poisson load generator · ${val.duration_s}s per level`}
            title={`Prediction error: p50 ${fmtPct(val.mape_p50, 0)} · p95 ${fmtPct(val.mape_p95, 0)} (MAPE)`}
            right={<Tip content={`Only capacity is calibrated (saturation throughput ${fmtNum(val.capacity_measured_rps, 0)} req/s rescales the measured service distribution by ×${val.service_scale.toFixed(2)}). The queueing behaviour under load is a genuine prediction.`}><span><Pill color={C.info}>calibrated on capacity only</Pill></span></Tip>}>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[560px] text-[12px]">
                <thead><tr className="text-left text-muted">
                  <th className="pb-2 font-normal">Load</th><th className="pb-2 text-right font-normal">Rate</th>
                  <th className="pb-2 text-right font-normal">Real p50</th><th className="pb-2 text-right font-normal">Twin p50</th>
                  <th className="pb-2 text-right font-normal">Real p95</th><th className="pb-2 text-right font-normal">Twin p95</th>
                  <th className="pb-2 text-right font-normal">Error p95</th>
                </tr></thead>
                <tbody>
                  {val.levels.map((lv) => (
                    <tr key={lv.utilization_target} className="border-t border-line">
                      <td className="num py-2 text-ink-2">{Math.round(lv.utilization_target * 100)}%</td>
                      <td className="num py-2 text-right text-ink-2">{fmtNum(lv.rate_rps, 0)}/s</td>
                      <td className="num py-2 text-right text-ink">{fmtMs(lv.real.p50)}</td>
                      <td className="num py-2 text-right text-ink-2">{fmtMs(lv.twin.p50)}</td>
                      <td className="num py-2 text-right text-ink">{fmtMs(lv.real.p95)}</td>
                      <td className="num py-2 text-right text-ink-2">{fmtMs(lv.twin.p95)}</td>
                      <td className="num py-2 text-right" style={{ color: lv.err_p95 < 0.15 ? C.ok : lv.err_p95 < 0.35 ? C.warn : C.critical }}>{fmtPct(lv.err_p95, 0)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="mt-2 text-[11.5px] text-muted">Error grows toward saturation, where p95 is most sensitive to small capacity differences — the reason EPOCH plans for ≤60% steady-state utilisation.</p>
          </Panel>
        )}

        <Panel eyebrow="Precomputed with the Python twin (SimPy)" title="Reference scenarios for the knee config" pad={false}>
          <div className="grid divide-y divide-line md:grid-cols-2 md:divide-x xl:grid-cols-5 xl:divide-y-0">
            {twin.scenarios.map((sc) => (
              <div key={sc.key} className="p-4">
                <div className="mb-2 text-[12.5px] font-medium text-ink">{sc.title}</div>
                <KeyVal k="p95" v={fmtMs(sc.summary.p95)} />
                <KeyVal k="utilisation" v={fmtPct(sc.summary.utilization, 0)} />
                <KeyVal k="max queue" v={fmtNum(sc.summary.max_queue, 0)} />
              </div>
            ))}
          </div>
        </Panel>
      </div>
    </div>
  );
}
