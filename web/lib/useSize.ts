"use client";

import { useEffect, useRef, useState } from "react";

export function useSize<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([e]) => setSize({ width: e.contentRect.width, height: e.contentRect.height }));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, size] as const;
}
