import { describe, it, expect } from "vitest";
import {
  EFFORT_OPTIONS,
  VALID_EFFORTS,
  effortShortLabel,
  effortToIndex,
  indexToEffort,
  normalizeEffort,
} from "./reasoning-effort";

describe("normalizeEffort", () => {
  it("treats empty/unset as the Hermes default (medium)", () => {
    expect(normalizeEffort("")).toBe("medium");
    expect(normalizeEffort(null)).toBe("medium");
    expect(normalizeEffort(undefined)).toBe("medium");
    expect(normalizeEffort("   ")).toBe("medium");
  });

  it("passes through every valid effort level", () => {
    for (const level of ["none", "minimal", "low", "medium", "high", "xhigh"]) {
      expect(normalizeEffort(level)).toBe(level);
    }
  });

  it("is case- and whitespace-insensitive", () => {
    expect(normalizeEffort("HIGH")).toBe("high");
    expect(normalizeEffort("  XHigh  ")).toBe("xhigh");
  });

  it("falls back to medium for unknown values", () => {
    expect(normalizeEffort("turbo")).toBe("medium");
    expect(normalizeEffort("max")).toBe("medium"); // 'max' is a label, not a value
    expect(normalizeEffort(42)).toBe("medium");
  });
});

describe("EFFORT_OPTIONS", () => {
  it("every option value is in VALID_EFFORTS (no orphan labels)", () => {
    for (const opt of EFFORT_OPTIONS) {
      expect(VALID_EFFORTS.has(opt.value)).toBe(true);
    }
  });

  it("covers the real reasoning levels plus thinking-off", () => {
    // Invariant against hermes_constants.VALID_REASONING_EFFORTS + 'none'.
    const values = new Set(EFFORT_OPTIONS.map((o) => o.value));
    for (const level of ["none", "minimal", "low", "medium", "high", "xhigh"]) {
      expect(values.has(level)).toBe(true);
    }
  });

  it("declares no extra levels the backend would reject", () => {
    // The slider maps position → value verbatim, so an invented stop here
    // would write an effort parse_reasoning_effort() silently drops.
    expect(EFFORT_OPTIONS).toHaveLength(6);
  });

  it("orders stops from off to deepest (the slider reads left→right)", () => {
    expect(EFFORT_OPTIONS.map((o) => o.value)).toEqual([
      "none",
      "minimal",
      "low",
      "medium",
      "high",
      "xhigh",
    ]);
  });

  it("uses desktop's short labels verbatim", () => {
    // Mirrors REASONING_LABELS in apps/desktop/src/lib/model-status-label.ts
    // so the two surfaces read identically.
    expect(EFFORT_OPTIONS.map((o) => o.short)).toEqual([
      "Off",
      "Min",
      "Low",
      "Med",
      "High",
      "Max",
    ]);
  });
});

describe("effortToIndex / indexToEffort", () => {
  it("round-trips every valid level", () => {
    for (const opt of EFFORT_OPTIONS) {
      expect(indexToEffort(effortToIndex(opt.value))).toBe(opt.value);
    }
  });

  it("maps positions to the declared order", () => {
    expect(effortToIndex("none")).toBe(0);
    expect(effortToIndex("xhigh")).toBe(EFFORT_OPTIONS.length - 1);
    expect(indexToEffort(0)).toBe("none");
    expect(indexToEffort(3)).toBe("medium");
  });

  it("parks unknown/empty values on medium", () => {
    expect(indexToEffort(effortToIndex(""))).toBe("medium");
    expect(indexToEffort(effortToIndex("turbo"))).toBe("medium");
  });

  it("clamps out-of-range positions instead of writing junk", () => {
    expect(indexToEffort(-5)).toBe("none");
    expect(indexToEffort(99)).toBe("xhigh");
    expect(VALID_EFFORTS.has(indexToEffort(NaN))).toBe(true);
  });
});

describe("effortShortLabel", () => {
  it("renders the compact desktop-matching label", () => {
    expect(effortShortLabel("xhigh")).toBe("Max");
    expect(effortShortLabel("none")).toBe("Off");
    expect(effortShortLabel("minimal")).toBe("Min");
  });

  it("falls back to the default tier's label when unset", () => {
    expect(effortShortLabel("")).toBe("Med");
    expect(effortShortLabel(undefined)).toBe("Med");
  });
});
