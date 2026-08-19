/**
 * ChatModelPill — 输入框内右下角的一颗胶囊, 点开就是 model + reasoning-effort
 * 二合一面板 (对齐 desktop ModelPill 观感):
 *   - 顶部搜索框
 *   - 中部按 provider 分组的 model 列表, 点击即切
 *   - 底部 Thinking 一栏, 6 档滑块 Off/Min/Low/Med/High/Max —— 档位与文案都取自
 *     lib/reasoning-effort 的 EFFORT_OPTIONS (它有不变量测试守着, 对齐
 *     hermes_constants.VALID_REASONING_EFFORTS), 不再在这里手写第二份表。
 *     后端按当前模型能力做映射: OpenAI reasoning_effort / Claude budget /
 *     Qwen enable_thinking。当前模型 capabilities.supports_reasoning 为 false
 *     时滑块整体置灰 (对齐 desktop 按 capabilities.reasoning 收起该控件)。
 *
 * model 和 effort 现在是同一套【会话级】语义: 有 ChatSessionContext (即挂在一个
 * 真实会话的 composer 上) 时分别走 setSessionModel / setSessionReasoningEffort
 * → config.set{scope 或 --session}, 只改这个会话的活 agent + 会话行, 下一轮立即
 * 生效, 不碰 config.yaml。没有会话时 (独立/无连接) 退回写 config.yaml 的 REST /
 * setReasoningEffort —— 那只是【新会话】的默认值。
 *
 * ★ 为什么不能只写 config.yaml (这块以前一直"点了没反应"的原因):
 *   1) agent 每会话只 build 一次, 之后每轮只读 agent.model / agent.reasoning_config
 *      这些内存属性 (从不回读磁盘) → 写盘对活会话零影响;
 *   2) HERMES_HOME/config.yaml 每次 dashboard 启动都被 git 里的项目 config.yaml
 *      整体覆盖 (sync_project_config), 而项目配置没有 agent: 段 → 写进去的值重启
 *      即消失, 读回来恒为 normalizeEffort 的兜底 "medium"。
 *   ★ model 曾长期是这个 bug 的最后一块: effort 早已改走 session RPC, model 却还
 *     留在 api.setModelAssignment (纯写盘) 上 —— 所以在会话里切模型对当前会话
 *     无效, 只能靠"重启聊天"生效 (ChatSidebar 那个确认框就是为此存在的补丁)。
 *     后端 config.set{key:"model"} 会调 agent.switch_model() 原地换 client 并
 *     广播 session.info, 才是真正的热切换。
 */

import { ChevronDown, Search } from "lucide-react";
import { useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";

import { ThinkingSlider } from "@/components/ThinkingSlider";
import { ChatSessionContext } from "@/contexts/chat-session-context";
import { api } from "@/lib/api";
import type { ModelOptionProvider } from "@/lib/api";
import {
  getReasoningEffort,
  setReasoningEffort,
  setSessionModel,
  setSessionReasoningEffort,
  type SessionModelSwitchResult,
} from "@/lib/config-api";
import {
  effortShortLabel,
  normalizeEffort,
  VALID_EFFORTS,
} from "@/lib/reasoning-effort";
import { cn } from "@/lib/utils";

interface Props {
  className?: string;
}

interface FlatEntry {
  provider: string;
  providerSlug: string;
  model: string;
}

/** "confirm" ⇒ the switch was REJECTED pending the user's OK (expensive-model
 *  guard), so it carries the target to retry. "warning" ⇒ it applied. */
type ModelNotice =
  | { kind: "confirm"; model: string; provider: string; text: string }
  | { kind: "warning"; text: string };

export function ChatModelPill({ className }: Props) {
  const [model, setModel] = useState<string>("");
  const [provider, setProvider] = useState<string>("");
  const [effort, setEffort] = useState<string>("medium");
  // Whether to show the thinking control at all. Needs BOTH:
  //   supports_reasoning     — the model reasons
  //   can_toggle_reasoning   — and this endpoint lets US change that
  // Defaults to true: hiding the dial from a capable-but-uncatalogued model is
  // the worse failure (same rationale as hermes_cli/inventory.py's
  // _apply_capabilities).
  //
  // ★ The second flag is why this isn't just `supports_reasoning`. Gating on
  //   capability alone rendered a live-looking switch for models whose endpoint
  //   accepts no toggle — thinking-only models, pre-toggle generations, and
  //   aggregators serving another vendor's model. A control that silently does
  //   nothing is worse than an absent one, so back ends that can't honour it
  //   say so and we hide it.
  const [canReason, setCanReason] = useState(true);
  const [open, setOpen] = useState(false);
  const [providers, setProviders] = useState<ModelOptionProvider[]>([]);
  const [query, setQuery] = useState("");
  const [saving, setSaving] = useState(false);
  const [modelNotice, setModelNotice] = useState<ModelNotice | null>(null);
  const rootRef = useRef<HTMLDivElement | null>(null);
  // null when this pill isn't inside a chat (dashboard/standalone) → effort
  // writes fall back to the global config default.
  const chatSession = useContext(ChatSessionContext);

  // ★ Deliberately does NOT re-read the effort: this runs again after every
  // model switch, and the global config value is merely the NEW-session
  // baseline — re-reading it would stomp the effort the user picked for this
  // session back to the baseline (looking like the dial "reset itself").
  //
  // ★ `keepIdentity` exists for the same reason, one level up: /api/model/info
  // reports what config.yaml says, which after a SESSION-scoped switch is
  // deliberately still the old global default. Letting it write back would make
  // a successful hot swap flicker to the previous model. After a switch we only
  // want the capability probe (does this model support reasoning?), so the model
  // identity we just applied is preserved. The live value is authoritative and
  // arrives via `session.info` below.
  const refresh = useCallback((keepIdentity = false) => {
    void api
      .getModelInfo()
      .then((r) => {
        if (!keepIdentity) {
          if (r?.model) setModel(String(r.model));
          if (r?.provider) setProvider(String(r.provider));
        }
        setCanReason(
          r?.capabilities?.supports_reasoning !== false &&
            r?.capabilities?.can_toggle_reasoning !== false,
        );
      })
      .catch(() => {
        /* keep last known */
      });
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // The live session is the source of truth for the current model: the backend
  // broadcasts `session.info` after agent.switch_model(), so a switch made
  // anywhere (this pill, a /model slash command, the TUI) shows up here.
  useEffect(() => {
    const session = chatSession?.resolve();

    if (!session) return;

    return session.gw.on<{ model?: string; provider?: string }>(
      "session.info",
      (ev) => {
        if (ev.payload?.model) setModel(String(ev.payload.model));
        if (ev.payload?.provider) setProvider(String(ev.payload.provider));
      },
    );
  }, [chatSession]);

  // Seed the dial ONCE from the config baseline — the same value a brand-new
  // session is built with. From here on this session's effort is owned by the
  // user's clicks (each one applied to the live agent via the session RPC).
  useEffect(() => {
    void getReasoningEffort().then((e) => setEffort(normalizeEffort(e)));
  }, []);

  useEffect(() => {
    if (!open) return;
    void api.getModelOptions().then((r) => {
      setProviders(r?.providers ?? []);
    }).catch(() => {
      setProviders([]);
    });
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (!rootRef.current) return;
      if (!rootRef.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  // ★ 故意不做 `next === effort` 短路。以前有这一支, 而读回来的值又恒为兜底
  //   "medium" —— 于是点 "Med" 这一档【连请求都不发】, 一个持久化 bug 长得跟
  //   "控件坏了"完全一样, 极难排查。现在同档重复点击也真发一次 (幂等), 保留
  //   saving 守卫防并发写。
  const applyEffort = useCallback(
    (next: string) => {
      if (!VALID_EFFORTS.has(next) || saving) return;
      const prev = effort;
      setEffort(next); // optimistic
      setSaving(true);
      const session = chatSession?.resolve() ?? null;
      const write = session
        ? setSessionReasoningEffort(session.gw, session.sessionId, next)
        : setReasoningEffort(next);
      write
        .catch(() => setEffort(prev))
        .finally(() => setSaving(false));
    },
    [chatSession, effort, saving],
  );

  // Same session/global split as applyEffort above: a live session gets a hot
  // swap that keeps the conversation; without one we're just setting the
  // baseline for the next session, which is what REST is for.
  const applyModel = useCallback(
    (nextProvider: string, nextModel: string, confirmExpensive = false) => {
      if (nextModel === model && nextProvider === provider && !confirmExpensive) {
        setOpen(false);
        return;
      }
      const prevModel = model;
      const prevProvider = provider;
      const revert = () => {
        setModel(prevModel);
        setProvider(prevProvider);
      };

      setModel(nextModel);
      setProvider(nextProvider);
      setOpen(false);
      setModelNotice(null);

      const session = chatSession?.resolve() ?? null;
      // Normalized to one shape. The two endpoints differ in their replies —
      // REST has no `warning` and names the applied model `model`, the session
      // RPC calls it `value` — so map REST into the RPC's shape here rather
      // than special-casing every read below.
      const write: Promise<SessionModelSwitchResult> = session
        ? setSessionModel(session.gw, session.sessionId, {
            confirmExpensive,
            model: nextModel,
            provider: nextProvider,
          })
        : api
            .setModelAssignment({
              confirm_expensive_model: confirmExpensive,
              scope: "main",
              provider: nextProvider,
              model: nextModel,
            })
            .then((r) => ({ ...r, value: r.model }));

      void write
        .then((res) => {
          // The expensive-model guard returns before the switch is applied, so
          // the optimistic update must come back out.
          if (res?.confirm_required) {
            revert();
            setModelNotice({
              kind: "confirm",
              model: nextModel,
              provider: nextProvider,
              text: res.confirm_message || res.warning || "",
            });
            return;
          }
          // ★ The backend may have auto-corrected a near-miss name (a typo, or a
          //   variant the endpoint doesn't serve: `glm-5v-turbo` → `glm-5-turbo`).
          //   `value` is what the agent ACTUALLY switched to, so it wins over our
          //   optimistic guess — otherwise the pill advertises a model that was
          //   never selected.
          if (res?.value && res.value !== nextModel) {
            setModel(res.value);
          }
          if (res?.warning) {
            setModelNotice({ kind: "warning", text: res.warning });
          }
          // Keep the identity we just applied on a session switch — see refresh().
          refresh(!!session);
        })
        .catch(revert);
    },
    [chatSession, model, provider, refresh],
  );

  const filtered = useMemo<FlatEntry[]>(() => {
    const q = query.trim().toLowerCase();
    const out: FlatEntry[] = [];
    for (const p of providers) {
      const models = p.models || [];
      for (const m of models) {
        if (!q || m.toLowerCase().includes(q) || p.name.toLowerCase().includes(q)) {
          out.push({ provider: p.name, providerSlug: p.slug, model: m });
        }
      }
    }
    return out;
  }, [providers, query]);

  const grouped = useMemo(() => {
    const g = new Map<string, FlatEntry[]>();
    for (const e of filtered) {
      const arr = g.get(e.provider);
      if (arr) arr.push(e);
      else g.set(e.provider, [e]);
    }
    return Array.from(g.entries());
  }, [filtered]);

  const modelShort = model.split("/").slice(-1)[0] || model || "select model";
  const tierLabel = effortShortLabel(effort);

  return (
    <div ref={rootRef} className={cn("relative flex items-center text-xs", className)}>
      {/* Sits above the pill because the panel closes on select — a notice
          rendered inside it would never be seen. */}
      {modelNotice && !open && (
        <div className="absolute bottom-full right-0 z-30 mb-1 w-60 rounded-md border bg-background p-2 shadow-lg">
          <div className="text-[11px] leading-snug text-foreground">
            {modelNotice.text ||
              (modelNotice.kind === "confirm"
                ? "This model has unusually high known pricing."
                : "")}
          </div>
          <div className="mt-1.5 flex justify-end gap-1.5">
            <button
              type="button"
              className="rounded px-1.5 py-0.5 text-[11px] text-text-tertiary hover:bg-muted"
              onClick={() => setModelNotice(null)}
            >
              {modelNotice.kind === "confirm" ? "Cancel" : "Dismiss"}
            </button>
            {modelNotice.kind === "confirm" && (
              <button
                type="button"
                className="rounded bg-muted px-1.5 py-0.5 text-[11px] font-medium hover:bg-muted/70"
                onClick={() => {
                  const { model: m, provider: p } = modelNotice;
                  setModelNotice(null);
                  applyModel(p, m, true);
                }}
              >
                Switch anyway
              </button>
            )}
          </div>
        </div>
      )}

      <button
        type="button"
        className={cn(
          // h-6: pill 现在活在单行 composer 里, 它是那一行的最高元素 —— 高度直接
          // 决定编辑框高度。h-8(32px) 会把框顶到 43px, 明显比一行文字该有的高度高。
          "flex h-6 items-center gap-1 rounded px-1.5 text-text-tertiary transition-colors",
          "hover:bg-muted hover:text-foreground",
          open && "bg-muted text-foreground",
        )}
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="listbox"
        aria-expanded={open}
        title={canReason ? `${modelShort} · ${tierLabel}` : modelShort}
      >
        {/* 窄屏(<sm)收成"只剩箭头"的紧凑态 —— composer 是单行布局, pill 若始终
            占满 ~137px 会把输入区挤成 0 宽、并把发送按钮顶出边框 (对齐 desktop
            ModelPill 的 compact 模式: 窄容器下同样只留 chevron)。 */}
        <span className="hidden max-w-[10rem] truncate font-normal sm:inline">{modelShort}</span>
        {/* 不支持推理的模型不显示档位 —— 显示一个不生效的 "Med" 是误导。 */}
        {canReason && (
          <>
            <span className="hidden text-text-tertiary/70 sm:inline">·</span>
            <span className="hidden font-normal text-text-tertiary sm:inline">{tierLabel}</span>
          </>
        )}
        <ChevronDown className="h-3 w-3 shrink-0 opacity-50" />
      </button>

      {/* 面板宽度 w-60 (240px): 由最窄的硬约束决定 —— Thinking 那排 6 个刻度
          (Off/Min/Low/Med/High/Max, 10px 字) 约 130px, 加左右 padding 也就
          ~180px。之前的 w-80(320px) 是照抄 dialog 的尺寸, 挂在单行 composer 上方
          时明显过宽、右侧留一大片空白; 模型名本来就靠 truncate 收口, 再宽也换不
          来可读性。 */}
      {open && (
        <div className="absolute bottom-full right-0 z-30 mb-1 w-60 rounded-md border bg-background p-0 shadow-lg">
          <div className="flex items-center gap-1.5 border-b px-2 py-1.5">
            <Search className="h-3.5 w-3.5 text-text-tertiary" />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search model or provider…"
              className="min-w-0 flex-1 border-0 bg-transparent text-xs outline-none placeholder:text-text-tertiary"
              autoFocus
            />
          </div>

          <div className="max-h-64 overflow-y-auto p-1">
            {providers.length === 0 && (
              <div className="px-2 py-3 text-center text-xs text-text-tertiary">
                Loading models…
              </div>
            )}
            {providers.length > 0 && filtered.length === 0 && (
              <div className="px-2 py-3 text-center text-xs text-text-tertiary">
                No matches
              </div>
            )}
            {grouped.map(([providerName, entries]) => (
              <div key={providerName} className="mb-1 last:mb-0">
                <div className="px-2 pb-0.5 pt-1 text-[10px] uppercase tracking-wide text-text-tertiary">
                  {providerName}
                </div>
                {entries.map((entry) => {
                  const active =
                    entry.model === model && entry.providerSlug === (
                      providers.find((p) => p.name === entry.provider)?.slug || ""
                    );
                  return (
                    <button
                      key={`${entry.providerSlug}::${entry.model}`}
                      type="button"
                      className={cn(
                        "flex w-full items-center justify-between rounded px-2 py-1 text-left text-xs hover:bg-muted",
                        active && "bg-muted font-medium",
                      )}
                      onClick={() => applyModel(entry.providerSlug, entry.model)}
                    >
                      <span className="min-w-0 truncate">{entry.model}</span>
                      {active && (
                        <span className="ml-2 shrink-0 text-text-tertiary">✓</span>
                      )}
                    </button>
                  );
                })}
              </div>
            ))}
          </div>

          {/* Thinking —— 6 档滑块, 档位/文案来自 EFFORT_OPTIONS。后端按当前模型
              能力做映射 (OpenAI reasoning_effort / Claude budget / Qwen bool)。 */}
          <div className="border-t px-3 py-2">
            <ThinkingSlider
              disabled={!canReason}
              disabledHint="This model doesn't support reasoning effort."
              onChange={applyEffort}
              saving={saving}
              value={effort}
            />
          </div>
        </div>
      )}
    </div>
  );
}
