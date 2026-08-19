/**
 * Work-segment grouping: a turn is `think → act → think → act → answer`, and
 * the waterfall must show it in that order.
 *
 * The old grouping merged every consecutive tool row into one block regardless
 * of the prose between them, and reasoning was rendered only while streaming
 * and then discarded — so a finished multi-round turn read as one anonymous
 * "processing" pile with no rationale and no round boundaries.
 */
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { TRANSLATIONS } from "../i18n/catalog";
import { BgBlock, buildRows, summarizeStep } from "./MultimodalChatPage";

// Minimal ChatMsg-shaped fixtures. `buildRows` only reads the fields below, so
// the casts keep the fixtures readable instead of spelling out the full type.
const tool = (id: string, toolName: string) =>
  ({ id, role: "assistant", text: "", kind: "tool", toolName }) as never;
const think = (id: string, reasoning: string) =>
  ({ id, role: "assistant", text: "", reasoning }) as never;
const say = (id: string, text: string) => ({ id, role: "assistant", text }) as never;
const ask = (id: string, text: string) => ({ id, role: "user", text }) as never;

const kinds = (rows: ReturnType<typeof buildRows>) =>
  rows.map((r) => (r.type === "bg" ? `bg(${r.items.length})` : `chat:${r.msg.role}`));

describe("buildRows work segments", () => {
  it("splits segments on assistant prose instead of merging across it", () => {
    // think A → tool → PROSE → think B → tool → tool
    const rows = buildRows([
      think("t1", "想A"),
      tool("x1", "read_file"),
      say("m1", "中间说明"),
      think("t2", "想B"),
      tool("x2", "terminal"),
      tool("x3", "grep"),
    ]);

    const bg = rows.filter((r) => r.type === "bg");

    // Two distinct segments — prose is a barrier, so x1 must NOT be pooled
    // together with x2/x3. That pooling is what destroyed the causal chain.
    expect(bg).toHaveLength(2);
    expect(bg[0].type === "bg" && bg[0].items.map((i) => i.toolName)).toEqual(["read_file"]);
    expect(bg[1].type === "bg" && bg[1].items.map((i) => i.toolName)).toEqual([
      "terminal",
      "grep",
    ]);
  });

  it("attaches each segment's own reasoning, so a finished turn keeps its rationale", () => {
    const rows = buildRows([
      think("t1", "先看配置"),
      tool("x1", "read_file"),
      say("m1", "配置没问题"),
      think("t2", "再跑一下测试"),
      tool("x2", "terminal"),
    ]);

    const bg = rows.filter((r) => r.type === "bg");

    expect(bg[0].type === "bg" && bg[0].thinking).toBe("先看配置");
    expect(bg[1].type === "bg" && bg[1].thinking).toBe("再跑一下测试");
  });

  it("numbers segments so multi-round turns are readable", () => {
    const rows = buildRows([
      tool("x1", "read_file"),
      say("m1", "ok"),
      tool("x2", "terminal"),
      say("m2", "ok"),
      tool("x3", "grep"),
    ]);

    expect(rows.filter((r) => r.type === "bg").map((r) => (r.type === "bg" ? r.seg : null))).toEqual(
      [1, 2, 3],
    );
  });

  it("still merges consecutive calls so ten tools are one block, not ten", () => {
    const rows = buildRows([
      tool("a", "t1"),
      tool("b", "t2"),
      tool("c", "t3"),
      tool("d", "t4"),
    ]);

    expect(kinds(rows)).toEqual(["bg(4)"]);
  });

  it("folds interleaved reasoning into an already-open segment", () => {
    // Anthropic-style interleaved thinking: reasoning can arrive *after* the
    // segment opened. It must land on that segment, not be dropped.
    const rows = buildRows([tool("x1", "read_file"), think("t1", "边做边想"), tool("x2", "grep")]);

    const bg = rows.filter((r) => r.type === "bg");

    expect(bg).toHaveLength(1);
    expect(bg[0].type === "bg" && bg[0].thinking).toBe("边做边想");
  });

  it("does not leak reasoning across a user turn", () => {
    const rows = buildRows([think("t1", "上一轮的想法"), ask("u1", "新问题"), tool("x1", "terminal")]);

    const bg = rows.filter((r) => r.type === "bg");

    expect(bg[0].type === "bg" && bg[0].thinking).toBeUndefined();
  });

  it("keeps the rationale after the answer lands on the same bubble", () => {
    // `ensureBubble` reuses ONE assistant message per turn: reasoning
    // accumulates onto it, then the final answer text lands on that SAME
    // object. So the bubble holding the answer is also the only carrier of the
    // turn's reasoning — if prose unconditionally cleared it, the rationale
    // would vanish the instant the answer arrived (the original bug, relocated).
    const midStream = buildRows([
      { id: "a1", role: "assistant", text: "", reasoning: "想A" } as never,
      tool("x1", "read_file"),
    ]);
    const afterAnswer = buildRows([
      { id: "a1", role: "assistant", text: "最终答案", reasoning: "想A" } as never,
      tool("x1", "read_file"),
    ]);

    const thinkingOf = (rows: ReturnType<typeof buildRows>) => {
      const bg = rows.find((r) => r.type === "bg");

      return bg?.type === "bg" ? bg.thinking : undefined;
    };

    expect(thinkingOf(midStream)).toBe("想A");
    expect(thinkingOf(afterAnswer)).toBe("想A");
  });

  it("renders the segment's rationale in the block header without a round index", () => {
    const markup = renderToStaticMarkup(
      createElement(BgBlock, {
        items: [
          { id: "t1", role: "assistant", text: "", kind: "tool", toolName: "read_file", toolDone: true },
          { id: "t2", role: "assistant", text: "", kind: "tool", toolName: "grep", toolDone: true },
        ] as never,
        thinking: "先确认配置再决定改哪里",
      }),
    );

    // Rationale survives into the finished turn (it used to be discarded).
    expect(markup).toContain("先确认配置再决定改哪里");
    // The round index is deliberately not rendered: one segment per turn is the
    // common case, so a constant "#1" was noise that also read as a request id.
    expect(markup).not.toContain("#");
    // Collapsed by default — the full text is behind the toggle, so a long
    // rationale can't push the tool rows off screen.
    expect(markup).toContain('aria-expanded="false"');
  });

  it("keeps router / monitor / watcher rows out of the centre waterfall", () => {
    const rows = buildRows([
      { id: "r", role: "assistant", text: "deep", subRole: "router" } as never,
      { id: "mo", role: "assistant", text: "alert", subRole: "monitor" } as never,
      { id: "w", role: "assistant", text: "report", subRole: "watcher_report" } as never,
      say("m1", "kept"),
    ]);

    expect(kinds(rows)).toEqual(["chat:assistant"]);
  });
});

describe("summarizeStep", () => {
  // Identity translator: assert on the key + count, not on English wording.
  const tr = (key: string, n: number) => `${key.replace("multimodal.misc.", "")}(${n})`;
  const t = (toolName: string, extra: Record<string, unknown> = {}) =>
    ({ kind: "tool", toolName, toolDone: true, ...extra });

  it("says what the step did, grouped by tool family", () => {
    expect(
      summarizeStep([t("read_file"), t("read_file"), t("terminal")], tr),
    ).toBe("stepRead(2) · stepRun(1)");
  });

  it("uses a stable order regardless of call order", () => {
    const a = summarizeStep([t("terminal"), t("read_file")], tr);
    const b = summarizeStep([t("read_file"), t("terminal")], tr);

    expect(a).toBe(b);
    expect(a).toBe("stepRead(1) · stepRun(1)");
  });

  it("falls back to a generic count for unrecognised tools", () => {
    expect(summarizeStep([t("some_exotic_skill")], tr)).toBe("stepCall(1)");
  });

  it("calls out failures", () => {
    expect(
      summarizeStep([t("read_file"), t("terminal", { toolIsError: true })], tr),
    ).toBe("stepRead(1) · stepRun(1) · stepFailed(1)");
  });

  it("ignores status rows and unnamed entries", () => {
    expect(summarizeStep([{ kind: "status" }, { kind: "tool" }], tr)).toBe("");
  });
});

describe("ordering is preserved as given", () => {
  it("keeps prose and calls in the order they appear in the message list", () => {
    // The UI intentionally uses ONE answer bubble per turn: narration keeps
    // appending to it and the tool rows group below. So this only pins that
    // buildRows never REORDERS what it is handed — it is the history-reload
    // shape (which does interleave, one stored row per segment) that exercises
    // the interleaved path.
    const rows = buildRows([
      ask("u", "查一下北京下周天气"),
      tool("x1", "browser_navigate"),
      say("a1", "Google 搜索被验证码拦截了。"),
      tool("x2", "browser_navigate"),
      say("a2", "weather.com 这个链接 404 了。"),
      tool("x3", "execute_code"),
    ]);

    expect(
      rows.map((r) =>
        r.type === "bg"
          ? `tools:${r.items.map((i) => i.toolName).join("+")}`
          : `text:${r.msg.text.slice(0, 6)}`,
      ),
    ).toEqual([
      "text:查一下北京下",
      "tools:browser_navigate",
      "text:Google",
      "tools:browser_navigate",
      "text:weathe",
      "tools:execute_code",
    ]);
  });
});

describe("step summary i18n keys", () => {
  // The step-verb keys are passed via constants, so they sit outside
  // key-integrity.test.ts's `translateNow("...")` regex. Resolve them here so a
  // key missing from one locale still fails a test instead of rendering the raw
  // key path into the header.
  it("resolves every step key in both shipped locales", () => {
    const keys = [
      "multimodal.misc.stepRead",
      "multimodal.misc.stepEdit",
      "multimodal.misc.stepRun",
      "multimodal.misc.stepSearch",
      "multimodal.misc.stepBrowse",
      "multimodal.misc.stepLook",
      "multimodal.misc.stepCall",
      "multimodal.misc.stepFailed",
    ];

    for (const locale of Object.keys(TRANSLATIONS) as (keyof typeof TRANSLATIONS)[]) {
      for (const key of keys) {
        const value = key
          .split(".")
          .reduce<unknown>(
            (cur, part) =>
              cur && typeof cur === "object" ? (cur as Record<string, unknown>)[part] : undefined,
            TRANSLATIONS[locale],
          );

        expect(typeof value, `${key} in ${locale}`).toBe("function");
      }
    }
  });
});

describe("failed tool rows", () => {
  it("shows the failure reason inline instead of a success checkmark", () => {
    // A failed call used to render with the same green ✓ as a success, with the
    // reason buried in a collapsed disclosure — indistinguishable at a glance.
    const markup = renderToStaticMarkup(
      createElement(BgBlock, {
        items: [{
          id: "t1", role: "assistant", text: "", kind: "tool",
          toolName: "read_file", toolCtx: "/nope.txt", toolDone: true,
          toolIsError: true, toolError: "file not found: /nope.txt",
        }] as never,
      }),
    );

    expect(markup).toContain("file not found: /nope.txt");
    expect(markup).toContain("✕");
    expect(markup).not.toContain("✓");
  });

  it("keeps showing the success summary when nothing failed", () => {
    const markup = renderToStaticMarkup(
      createElement(BgBlock, {
        items: [{
          id: "t1", role: "assistant", text: "", kind: "tool",
          toolName: "read_file", toolDone: true, toolSummary: "142 lines",
        }] as never,
      }),
    );

    expect(markup).toContain("142 lines");
    expect(markup).toContain("✓");
  });
});

