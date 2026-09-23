"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef } from "react";
import { useUI } from "./store";
import type { Bundle } from "./types";

export const API = process.env.NEXT_PUBLIC_EPOCH_API || "";

declare global {
  interface Window { __EPOCH_BUNDLE__?: Bundle }
}

async function loadBundle(): Promise<Bundle> {
  if (API) {
    const r = await fetch(`${API}/api/bundle`);
    if (!r.ok) throw new Error(`API ${r.status}`);
    return r.json();
  }
  if (typeof window !== "undefined") {
    if (!window.__EPOCH_BUNDLE__) {
      // same-origin script works on hosts that block fetch (sandboxed previews); JSON fetch is the fallback
      await new Promise<void>((resolve) => {
        const s = document.createElement("script");
        s.src = "replay/bundle.js";
        s.onload = () => resolve();
        s.onerror = () => resolve();
        document.head.appendChild(s);
      });
    }
    if (window.__EPOCH_BUNDLE__) return window.__EPOCH_BUNDLE__;
  }
  const r = await fetch("replay/bundle.json");
  if (!r.ok) throw new Error("Replay bundle not found — run `epoch demo` to generate web/public/replay/bundle.json");
  return r.json();
}

export function useBundle() {
  return useQuery({ queryKey: ["bundle"], queryFn: loadBundle, staleTime: API ? 5_000 : Infinity, refetchOnWindowFocus: false });
}

/** Live mode: subscribe to the API's SSE stream, log events and refresh the bundle (throttled). */
export function useLiveEvents() {
  const qc = useQueryClient();
  const push = useUI((s) => s.pushEvent);
  const last = useRef(0);
  useEffect(() => {
    if (!API) return;
    const es = new EventSource(`${API}/api/events/stream`);
    es.addEventListener("epoch", (e) => {
      try {
        const msg = JSON.parse((e as MessageEvent).data);
        push({ ts: Date.now(), kind: msg.kind, text: summarize(msg) });
        const now = Date.now();
        if (now - last.current > 3000) {
          last.current = now;
          qc.invalidateQueries({ queryKey: ["bundle"] });
        }
      } catch { /* ignore malformed frames */ }
    });
    return () => es.close();
  }, [qc, push]);
}

function summarize(m: Record<string, unknown>): string {
  if (m.kind === "trial") return `trial ${m.number} · ${m.status} · hv ${Number(m.hv).toFixed(4)}`;
  if (m.kind === "hypothesis") return String(m.statement);
  if (m.kind === "hypothesis.verdict") return `${m.id} → ${m.verdict}`;
  if (m.kind === "incident") return `${m.id} · ${m.status}`;
  return String(m.kind);
}

export async function postDecision(incidentId: string, approved: boolean, note: string) {
  const r = await fetch(`${API}/api/incidents/${incidentId}/decision`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ approved, note, actor: "dashboard" }),
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}
