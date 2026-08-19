/**
 * Tool presentation during a streaming turn. The 💭 thinking line and the
 * "处理过程" card (BgBlock) are shown TOGETHER and split the work: the card
 * carries the tool detail (spinner + name + args while running, ✓ + duration +
 * summary once done) plus the segment's reasoning; the line carries a one-line
 * status and the elapsed timer that BgBlock's per-tool rows deliberately omit.
 *
 * There used to be a "only one of them may show the tools" arbitration, which
 * caused two defects:
 *
 *   1. The card was hidden while a tool was RUNNING, so the tool box only
 *      appeared once the tool had *finished* — late by the tool's whole
 *      duration. The user wants to see the call the moment it starts.
 *   2. Hiding was gated on "this bg row contains a tool" while the line's
 *      takeover was gated on "a tool is still running". In the window after a
 *      tool completed but before the answer text arrived, the card was hidden
 *      and the line had nothing to show — the tools were invisible on BOTH
 *      sides and the user saw only "Waiting response…".
 *
 * The arbitration is gone: the card is never suppressed, so both defects are
 * structurally impossible. These tests pin that down.
 */
import { describe, expect, it } from "vitest";

import { buildRows, deriveTurnToolPresentation } from "./MultimodalChatPage";

const streamingBubble = (reasoning?: string) => ({
  id: "a1",
  role: "assistant",
  text: "",
  streaming: true,
  ...(reasoning ? { reasoning, hasReasoning: true } : { awaitingFirstDelta: true }),
}) as never;

const toolMsg = (id: string, toolName: string, toolDone: boolean, toolCtx?: string) =>
  ({ id, role: "assistant", text: "", kind: "tool", toolName, toolDone, toolCtx }) as never;

// Mirrors renderRow: the bg row always renders a card; the thinking line's label
// comes from the shared derivation.
function presentTurn(msgs: never[]) {
  const rows = buildRows(msgs);
  const isPureThinking = (m: {
    role?: string; streaming?: boolean; text?: string; isError?: boolean;
  }) => m.role === "assistant" && !!m.streaming && !m.text?.trim() && !m.isError;

  return rows.map((row, i) => {
    if (row.type === "bg") return `card(${row.items.length})`;
    if (!isPureThinking(row.msg)) return "bubble";
    const next = rows[i + 1];
    const { toolActivity } = deriveTurnToolPresentation(
      next?.type === "bg" ? next.items : undefined,
    );
    const msg = row.msg as { hasReasoning?: boolean; reasoningSummary?: string };
    const label = toolActivity
      || (msg.hasReasoning ? (msg.reasoningSummary || "Thinking…") : "Waiting response…");
    return `line[${label}]`;
  });
}

describe("deriveTurnToolPresentation", () => {
  it("reports the running tool for the thinking line's label", () => {
    expect(deriveTurnToolPresentation([toolMsg("x1", "read_file", false, "config.yaml")]))
      .toEqual({ inToolCall: true, toolActivity: "read_file · config.yaml" });
  });

  it("keeps inToolCall set but drops the activity once every tool finished", () => {
    // inToolCall stays true so ChatBubble still suppresses the empty bubble;
    // the empty activity lets the line fall back to the reasoning summary.
    expect(deriveTurnToolPresentation([toolMsg("x1", "read_file", true, "config.yaml")]))
      .toEqual({ inToolCall: true, toolActivity: "" });
  });

  it("reports the newest still-running tool, not a finished earlier one", () => {
    expect(deriveTurnToolPresentation([
      toolMsg("x1", "read_file", true, "a.ts"),
      toolMsg("x2", "grep", false, "needle"),
    ]).toolActivity).toBe("grep · needle");
  });

  it("claims nothing for status-only rows or empty input", () => {
    const status = [{ id: "s1", role: "assistant", text: "", kind: "status" }] as never[];
    for (const items of [undefined, [] as never[], status]) {
      expect(deriveTurnToolPresentation(items)).toEqual({ inToolCall: false, toolActivity: "" });
    }
  });

  it("omits the separator when a tool has no context string", () => {
    expect(deriveTurnToolPresentation([toolMsg("x1", "terminal", false)]).toolActivity)
      .toBe("terminal");
  });
});

describe("streaming turn: the tool card is visible from the first tool.start", () => {
  it("shows the card while the very first tool is still running", () => {
    // The point of this change: previously ["line[…]", "card:hidden"] — the box
    // only appeared after the tool completed.
    expect(presentTurn([streamingBubble(), toolMsg("x1", "read_file", false, "config.yaml")]))
      .toEqual(["line[read_file · config.yaml]", "card(1)"]);
  });

  it("keeps the card once the tool finishes, before the answer arrives", () => {
    expect(presentTurn([streamingBubble(), toolMsg("x1", "read_file", true, "config.yaml")]))
      .toEqual(["line[Waiting response…]", "card(1)"]);
  });

  it("shows reasoning on the line while the card shows the calls", () => {
    expect(presentTurn([streamingBubble("检查配置"), toolMsg("x1", "read_file", false, "a.ts")]))
      .toEqual(["line[read_file · a.ts]", "card(1)"]);
  });

  it("renders the card at every phase of a multi-tool lifecycle", () => {
    const lifecycle: Array<[string, never[]]> = [
      ["first running", [
        streamingBubble("想想"), toolMsg("x1", "read_file", false, "a.ts"),
      ] as never[]],
      ["first done", [
        streamingBubble("想想"), toolMsg("x1", "read_file", true, "a.ts"),
      ] as never[]],
      ["second running", [
        streamingBubble("想想"),
        toolMsg("x1", "read_file", true, "a.ts"),
        toolMsg("x2", "grep", false, "needle"),
      ] as never[]],
      ["all done", [
        streamingBubble("想想"),
        toolMsg("x1", "read_file", true, "a.ts"),
        toolMsg("x2", "grep", true, "needle"),
      ] as never[]],
    ];

    for (const [phase, msgs] of lifecycle) {
      const out = presentTurn(msgs);
      // The invariant that replaces the old arbitration: the tools are on
      // screen at every phase, never suppressed waiting for prose.
      expect(out.some((s) => s.startsWith("card(")), `phase=${phase} out=${out.join(" ")}`)
        .toBe(true);
    }
  });

  it("still yields to the full bubble once prose lands", () => {
    const withProse = [
      { id: "a1", role: "assistant", text: "结论", streaming: true } as never,
      toolMsg("x1", "read_file", true, "a.ts"),
    ] as never[];
    expect(presentTurn(withProse)).toEqual(["bubble", "card(1)"]);
  });
});
