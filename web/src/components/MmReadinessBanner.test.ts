import { describe, expect, it } from "vitest";

import { selectMissingRequired, type MmReadinessReport } from "./MmReadinessBanner";

function report(partial: Partial<MmReadinessReport>): MmReadinessReport {
  return { ready: false, capabilities: [], ...partial };
}

describe("selectMissingRequired", () => {
  it("returns [] for a null report (fail safe)", () => {
    expect(selectMissingRequired(null)).toEqual([]);
  });

  it("returns [] when ready, regardless of caps", () => {
    const r = report({
      ready: true,
      capabilities: [
        { key: "memory", label: "记忆", status: "broken", required: true, reason: "x", fix: "y" },
      ],
    });
    expect(selectMissingRequired(r)).toEqual([]);
  });

  it("returns [] when a malformed report has no capabilities array", () => {
    // @ts-expect-error — deliberately malformed to prove the guard
    expect(selectMissingRequired({ ready: false })).toEqual([]);
  });

  it("returns only REQUIRED non-ok caps (optional gaps ignored)", () => {
    const r = report({
      capabilities: [
        { key: "memory", label: "记忆", status: "broken", required: true, reason: "", fix: "" },
        { key: "voice", label: "语音", status: "missing", required: false, reason: "", fix: "" },
        { key: "deep", label: "深研", status: "missing", required: false, reason: "", fix: "" },
      ],
    });
    const out = selectMissingRequired(r);
    expect(out.map((c) => c.key)).toEqual(["memory"]);
  });

  it("returns [] when all required caps are ok even if optional ones aren't", () => {
    const r = report({
      capabilities: [
        { key: "memory", label: "记忆", status: "ok", required: true, reason: "", fix: "" },
        { key: "voice", label: "语音", status: "missing", required: false, reason: "", fix: "" },
      ],
    });
    expect(selectMissingRequired(r)).toEqual([]);
  });

  it("includes both 'missing' and 'broken' required caps", () => {
    const r = report({
      capabilities: [
        { key: "a", label: "A", status: "missing", required: true, reason: "", fix: "" },
        { key: "b", label: "B", status: "broken", required: true, reason: "", fix: "" },
        { key: "c", label: "C", status: "ok", required: true, reason: "", fix: "" },
      ],
    });
    expect(selectMissingRequired(r).map((c) => c.key)).toEqual(["a", "b"]);
  });
});
