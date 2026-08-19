import { useEffect, useRef, useState } from "react";

/**
 * Throttle a rapidly-changing value to at most one update per `ms`.
 *
 * Drives live markdown rendering during streaming: raw text updates every token
 * (~60/s), but re-parsing markdown that fast wedges the main thread (long
 * answers + concurrent 2fps video frames). Feeding the renderer a value
 * throttled to ~120ms gives a real-time feel while cutting parse frequency ~10×.
 * The trailing edge always fires so the final value is never dropped. Pass ms=0
 * to disable (pass-through) — e.g. once streaming ends.
 */
export function useThrottledValue<T>(value: T, ms = 120): T {
  const [throttled, setThrottled] = useState(value);
  const lastRun = useRef(0);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const latest = useRef(value);
  latest.current = value;

  useEffect(() => {
    if (ms <= 0) {
      setThrottled(value);
      return;
    }
    const now = Date.now();
    const elapsed = now - lastRun.current;
    if (elapsed >= ms) {
      lastRun.current = now;
      setThrottled(value);
      return;
    }
    if (timer.current) return;
    timer.current = setTimeout(() => {
      lastRun.current = Date.now();
      timer.current = null;
      setThrottled(latest.current);
    }, ms - elapsed);
    return () => {
      if (timer.current) {
        clearTimeout(timer.current);
        timer.current = null;
      }
    };
  }, [value, ms]);

  return throttled;
}
