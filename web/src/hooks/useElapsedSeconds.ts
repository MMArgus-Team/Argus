import { useEffect, useState } from "react";

// "进行中"计时器 —— 给思考行末尾那个 "4s" 供数。
//
// 与 desktop 的 components/chat/activity-timer.ts 行为一致 (含 module-level
// 注册表), 这样两端同一条状态行的计时语义相同。★ 但实现【不用 useRef】: web
// 的 eslint 开了 react-hooks/refs, 渲染期读写 ref 会报错。这里改成
// "module 注册表 = 起始时刻的唯一真相 + useState 兜住匿名计时器", 无 ref
// 参与渲染, 语义不变。
//
// ★ 为什么要 module-level 注册表: 聊天流用 react-virtuoso 虚拟滚动, 一条消息
//   滚出视口会真的卸载。若把起始时刻只放在组件 state 里, 滚回来时会从 0s 重新
//   数 —— 用户看到计时器倒退。按 timerKey (绑消息 id) 记在模块作用域, 卸载/
//   重挂后仍接着数。
const startedAtByKey = new Map<string, number>();

/**
 * 该 key 的起始时刻 (首次调用时落盘, 之后恒定) —— 对同一 key 幂等。
 * exported for tests: 这是"卸载重挂不倒退 / 换轮才重置"的唯一依据。
 */
export function startedAt(key: string): number {
  const existing = startedAtByKey.get(key);
  if (existing !== undefined) return existing;
  const now = Date.now();
  startedAtByKey.set(key, now);
  return now;
}

/** 60s 以内显示 "45s", 超过显示 "1:05"。 */
export function formatElapsed(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}

function secondsSince(start: number): number {
  return Math.max(0, Math.floor((Date.now() - start) / 1000));
}

/**
 * 自 timerKey 首次出现起经过的整秒数。`active=false` 时停表 (不再 setInterval,
 * 也就不再每秒触发一次 re-render) —— 调用点应在流式结束后传 false。
 * 换 timerKey → start 变化 → effect 重挂, 从新 key 的起点重新计。
 */
export function useElapsedSeconds(active = true, timerKey?: string): number {
  // 匿名计时器 (无 key) 没法进注册表, 用 state 初始化函数固定住本次挂载的起点。
  const [anonStart] = useState(() => Date.now());
  const start = timerKey ? startedAt(timerKey) : anonStart;

  const [elapsed, setElapsed] = useState(() => secondsSince(start));

  useEffect(() => {
    if (!active) return;
    const tick = () => setElapsed(secondsSince(start));
    tick();
    const id = window.setInterval(tick, 1000);
    return () => window.clearInterval(id);
  }, [active, start]);

  return elapsed;
}
