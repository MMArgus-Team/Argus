/**
 * Pure reasoning-effort helpers shared by every Thinking control.
 *
 * Kept DOM-free so the node-environment vitest harness can cover the
 * resolution logic without loading React or the UI kit.
 *
 * Values mirror hermes_constants.VALID_REASONING_EFFORTS plus `none`
 * (thinking-off). An empty/unset config value means the Hermes default,
 * which is `medium`.
 *
 * `label` is the long form (settings/menus); `short` is the compact form used
 * on the composer pill and the slider ticks. Both are kept verbatim in sync
 * with desktop's REASONING_LABELS (apps/desktop/src/lib/model-status-label.ts)
 * so a web screenshot and a desktop screenshot read identically.
 */

export interface EffortOption {
  value: string;
  label: string;
  short: string;
  hint: string;
}

export const EFFORT_OPTIONS: ReadonlyArray<EffortOption> = [
  { value: "none", label: "Off (no thinking)", short: "Off", hint: "no reasoning" },
  { value: "minimal", label: "Minimal", short: "Min", hint: "barely any" },
  { value: "low", label: "Low", short: "Low", hint: "shallow" },
  { value: "medium", label: "Medium", short: "Med", hint: "default" },
  { value: "high", label: "High", short: "High", hint: "deep" },
  { value: "xhigh", label: "Max", short: "Max", hint: "deepest" },
];

export const VALID_EFFORTS: ReadonlySet<string> = new Set(
  EFFORT_OPTIONS.map((o) => o.value),
);

/** Normalize a raw `agent.reasoning_effort` config value to a selectable
 *  option. Empty/unknown → `medium` (Hermes' default when unset). */
export function normalizeEffort(raw: unknown): string {
  const value = String(raw ?? "").trim().toLowerCase();
  if (!value) return "medium";
  return VALID_EFFORTS.has(value) ? value : "medium";
}

/** Slider position (0…EFFORT_OPTIONS.length-1) for an effort value. Unknown
 *  values land on `medium`, matching normalizeEffort. */
export function effortToIndex(raw: unknown): number {
  const value = normalizeEffort(raw);
  const index = EFFORT_OPTIONS.findIndex((o) => o.value === value);
  return index >= 0 ? index : EFFORT_OPTIONS.findIndex((o) => o.value === "medium");
}

/** Inverse of effortToIndex. Out-of-range positions clamp to the ends, so a
 *  stray slider event can never write an invalid effort. */
export function indexToEffort(index: number): string {
  const max = EFFORT_OPTIONS.length - 1;
  const clamped = Math.min(max, Math.max(0, Math.round(Number(index) || 0)));
  return EFFORT_OPTIONS[clamped]!.value;
}

/** Compact label for a raw config value — the composer pill's suffix. */
export function effortShortLabel(raw: unknown): string {
  return EFFORT_OPTIONS[effortToIndex(raw)]!.short;
}
