import { describe, expect, it } from "vitest";

import { HISTORY_FIRST_PAINT, historyPaintWindow } from "./MultimodalChatPage";

/**
 * 切换 session 时恢复历史的两段式首屏。列表是非虚拟化的普通 div, 一次交付 400 条会
 * 同步解析 400 份 Markdown → "切换后要等一下才出内容"。这里锁住切分不变量。
 */
describe("historyPaintWindow", () => {
  const MAX = 400;

  it("首屏只取尾部一屏, 剩下的留给第二段", () => {
    const { firstStart, fullStart, needsSecondPaint } = historyPaintWindow(1000, MAX);
    // 最终窗口是尾部 400 条; 首屏只交付尾部 HISTORY_FIRST_PAINT 条。
    expect(fullStart).toBe(600);
    expect(firstStart).toBe(1000 - HISTORY_FIRST_PAINT);
    expect(needsSecondPaint).toBe(true);
    // 两段拼起来正好等于最终窗口, 不重不漏。
    expect(1000 - firstStart).toBe(HISTORY_FIRST_PAINT);
    expect(firstStart - fullStart).toBe(MAX - HISTORY_FIRST_PAINT);
  });

  it("firstStart 永不越过 fullStart (首屏不会渲染出窗口以外的历史)", () => {
    for (const total of [0, 1, 30, 59, 60, 61, 200, 399, 400, 401, 5000]) {
      const { firstStart, fullStart } = historyPaintWindow(total, MAX);
      expect(firstStart).toBeGreaterThanOrEqual(fullStart);
      expect(fullStart).toBeGreaterThanOrEqual(0);
      expect(firstStart).toBeLessThanOrEqual(Math.max(0, total));
    }
  });

  it("历史短于首屏时一次画完, 不做多余的第二次渲染", () => {
    const short = historyPaintWindow(20, MAX);
    expect(short.firstStart).toBe(0);
    expect(short.fullStart).toBe(0);
    expect(short.needsSecondPaint).toBe(false);

    // 刚好等于首屏条数 → 仍然只需要一段。
    const exact = historyPaintWindow(HISTORY_FIRST_PAINT, MAX);
    expect(exact.firstStart).toBe(0);
    expect(exact.needsSecondPaint).toBe(false);
  });

  it("历史不超过 MAX_MESSAGES 时 fullStart=0 (上方没有可翻的历史)", () => {
    const { fullStart, needsSecondPaint } = historyPaintWindow(MAX, MAX);
    expect(fullStart).toBe(0);
    // 400 > 60, 所以仍要补第二段, 但补完就是全部历史。
    expect(needsSecondPaint).toBe(true);
  });

  it("firstPaint 可覆盖, 且 0/负数不会退化成空首屏", () => {
    expect(historyPaintWindow(1000, MAX, 100).firstStart).toBe(900);
    // 防御: firstPaint<=0 时至少画 1 条, 否则首屏空白。
    expect(historyPaintWindow(1000, MAX, 0).firstStart).toBe(999);
    expect(historyPaintWindow(1000, MAX, -5).firstStart).toBe(999);
  });
});
