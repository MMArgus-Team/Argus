import { describe, expect, it } from "vitest";

import { formatElapsed, startedAt } from "@/hooks/useElapsedSeconds";

// node-env (无 DOM) → 只测纯逻辑: 时长格式化 + 起始时刻注册表的幂等性。
// 注册表是"计时器卸载重挂不倒退、换轮才归零"的唯一依据, 所以它就是这个 hook
// 值得锁住的行为。React 订阅部分 (setInterval) 无 DOM 环境不测。

describe("formatElapsed", () => {
  it("60s 以内显示秒", () => {
    expect(formatElapsed(0)).toBe("0s");
    expect(formatElapsed(4)).toBe("4s");
    expect(formatElapsed(59)).toBe("59s");
  });

  it("满 60s 起显示 m:ss (秒补零)", () => {
    expect(formatElapsed(60)).toBe("1:00");
    expect(formatElapsed(65)).toBe("1:05");
    expect(formatElapsed(125)).toBe("2:05");
    expect(formatElapsed(3599)).toBe("59:59");
  });
});

describe("startedAt", () => {
  it("同一 key 幂等 —— 卸载重挂 (虚拟滚动) 后计时不倒退", () => {
    const first = startedAt("activity:msg-1");
    const again = startedAt("activity:msg-1");
    expect(again).toBe(first);
  });

  it("同一 key 反复调用始终返回首次的时刻", () => {
    const first = startedAt("activity:msg-repeat");
    for (let i = 0; i < 5; i++) {
      expect(startedAt("activity:msg-repeat")).toBe(first);
    }
  });

  it("不同 key (新一轮对话) 各自独立记时", () => {
    const a = startedAt("activity:turn-a");
    const b = startedAt("activity:turn-b");
    // 两个 key 互不影响: b 不会复用 a 的起点。
    expect(startedAt("activity:turn-a")).toBe(a);
    expect(startedAt("activity:turn-b")).toBe(b);
  });
});
