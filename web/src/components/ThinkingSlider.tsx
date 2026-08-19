/**
 * ThinkingSlider — the reasoning-effort dial, as a real slider.
 *
 * Replaces the hand-rolled 4-button bar that used to be duplicated in both
 * ChatModelPill and ModelPickerDialog. Two things it fixes beyond the visuals:
 *
 *   - Stops per surface re-declaring its own tier table. The stops come from
 *     EFFORT_OPTIONS (lib/reasoning-effort.ts), which is invariant-tested
 *     against hermes_constants.VALID_REASONING_EFFORTS — so `minimal` and
 *     `high` are reachable again, and nothing invents a level the backend
 *     would silently normalize away.
 *   - `disabled` lets callers hide/park the control for models whose
 *     capabilities say they don't do reasoning at all, rather than offering a
 *     knob the backend ignores (desktop already gates on capabilities.reasoning).
 *
 * It's a native <input type="range"> over discrete integer positions, so
 * keyboard (arrows/Home/End), touch drag, and screen readers all come free.
 * The value written to `agent.reasoning_effort` is always one of the stops.
 */

import { useId } from "react";

import {
  EFFORT_OPTIONS,
  effortToIndex,
  indexToEffort,
} from "@/lib/reasoning-effort";
import { cn } from "@/lib/utils";

interface Props {
  className?: string;
  /** Raw effort value (as stored in config); normalized internally. */
  value: string;
  onChange(next: string): void;
  /** In-flight save — the track parks but stays readable. */
  saving?: boolean;
  /** Model doesn't support reasoning: shown greyed with a reason tooltip. */
  disabled?: boolean;
  disabledHint?: string;
  /** Section heading. Omit to render just the track (embedded use). */
  label?: string;
}

const MAX = EFFORT_OPTIONS.length - 1;

export function ThinkingSlider({
  className,
  value,
  onChange,
  saving = false,
  disabled = false,
  disabledHint,
  label = "Thinking",
}: Props) {
  const id = useId();
  const index = effortToIndex(value);
  const current = EFFORT_OPTIONS[index]!;
  const inert = disabled || saving;

  // Percentage of the way along the track, used to paint the filled portion.
  const pct = (index / MAX) * 100;

  return (
    <div
      className={cn("flex flex-col gap-1", disabled && "opacity-50", className)}
      title={disabled ? disabledHint : current.hint}
    >
      {label && (
        <div className="flex items-baseline justify-between gap-2">
          <label
            className="text-display text-[10px] uppercase tracking-wide text-text-tertiary"
            htmlFor={id}
          >
            {label}
          </label>
          <span className="font-mono text-[11px] text-foreground">
            {disabled ? "—" : current.short}
          </span>
        </div>
      )}

      <input
        aria-label={label}
        aria-valuetext={current.label}
        className={cn(
          "h-1.5 w-full cursor-pointer appearance-none rounded-full bg-transparent",
          "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary/50",
          inert && "cursor-not-allowed",
          // The track is painted as a gradient so the filled portion reads as
          // "how much thinking". WebKit and Firefox each need their own
          // pseudo-element for the track/thumb; both are styled via the
          // arbitrary-variant selectors below rather than a global stylesheet,
          // keeping this component self-contained.
          "[&::-webkit-slider-runnable-track]:h-1.5 [&::-webkit-slider-runnable-track]:rounded-full",
          "[&::-moz-range-track]:h-1.5 [&::-moz-range-track]:rounded-full [&::-moz-range-track]:bg-muted",
          "[&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:w-3",
          "[&::-webkit-slider-thumb]:-mt-[3px] [&::-webkit-slider-thumb]:rounded-full",
          "[&::-webkit-slider-thumb]:bg-foreground [&::-webkit-slider-thumb]:transition-transform",
          "[&::-webkit-slider-thumb]:hover:scale-110",
          "[&::-moz-range-thumb]:h-3 [&::-moz-range-thumb]:w-3 [&::-moz-range-thumb]:rounded-full",
          "[&::-moz-range-thumb]:border-0 [&::-moz-range-thumb]:bg-foreground",
        )}
        disabled={inert}
        id={id}
        max={MAX}
        min={0}
        onChange={(e) => onChange(indexToEffort(Number(e.target.value)))}
        step={1}
        style={{
          // Filled up to the thumb, muted after it. Applied inline because the
          // stop position is dynamic; the pseudo-element rules above own the
          // geometry.
          background: `linear-gradient(to right, var(--color-primary, currentColor) ${pct}%, color-mix(in srgb, currentColor 20%, transparent) ${pct}%)`,
        }}
        type="range"
        value={index}
      />

      {/* Tick labels double as click targets: clicking "High" jumps there,
          which is quicker than dragging for a 6-stop scale. */}
      <div className="flex justify-between">
        {EFFORT_OPTIONS.map((opt, i) => (
          <button
            className={cn(
              "-mx-0.5 rounded px-0.5 text-[10px] transition-colors",
              i === index
                ? "font-medium text-foreground"
                : "text-text-tertiary hover:text-foreground",
              inert && "pointer-events-none",
            )}
            disabled={inert}
            key={opt.value}
            onClick={() => onChange(opt.value)}
            title={opt.hint}
            type="button"
          >
            {opt.short}
          </button>
        ))}
      </div>
    </div>
  );
}
