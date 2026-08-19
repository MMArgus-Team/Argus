import { ThreadPrimitive, useAuiEvent, useAuiState } from '@assistant-ui/react'
import {
  type ComponentProps,
  type CSSProperties,
  type FC,
  memo,
  type ReactNode,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState
} from 'react'
import { useStickToBottom } from 'use-stick-to-bottom'

import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import {
  onScrollToBottomRequest,
  onThreadEditClose,
  onThreadEditOpen,
  resetThreadScroll,
  setThreadAtBottom
} from '@/store/thread-scroll'
import { isSecondaryWindow } from '@/store/windows'

import { MessageRenderBoundary } from './message-render-boundary'

type ThreadMessageComponents = ComponentProps<typeof ThreadPrimitive.MessageByIndex>['components']

type MessageGroup = { id: string; weight: number } & (
  | { index: number; kind: 'standalone' }
  | { indices: number[]; kind: 'turn' }
)

// DOM is bounded by a rendered-PART budget, not a message/turn count: a single
// assistant message folds every tool call into a part, so heavy sessions are
// ~40 turns / ~100 messages but ~1000 parts — and parts are what drive node
// count. "Show earlier" prepends another page; whole turns stay intact so the
// sticky human bubble never loses its turn. This is the long-session perf lever
// WITHOUT a virtualizer — pure rendering, never touches scrollTop, so it can't
// fight use-stick-to-bottom (the single scroll owner).
// ★ 长会话加载雪崩修: 之前初始渲染 300 → 400+ 轮 session 一进来同步 mount 300 turn
//   (每 turn markdown+katex+highlight+image) 卡数秒。改成:
//   - INITIAL_RENDER_BUDGET=30 (最近 ~15 QA), 秒开;
//   - showEarlier 一次加 RENDER_BUDGET_STEP=100, 用户按需展开更早;
//   - 保底 200ms 后自动升到 100, 让常规滚动可见更多历史 (idle-tail warm-up)。
const INITIAL_RENDER_BUDGET = 30
const RENDER_BUDGET_STEP = 100
const RENDER_BUDGET_WARM = 100
const RENDER_BUDGET_WARMUP_MS = 200

interface ThreadMessageListProps {
  clampToComposer: boolean
  components: ThreadMessageComponents
  emptyPlaceholder?: ReactNode
  loadingIndicator?: ReactNode
  sessionKey?: string | null
  // 常驻的顶部横幅 (如多模态引导气泡): 空态时显示在顶部, 非空态时也置顶,
  // 随消息一起滚动 (不 sticky)。发消息后不消失。
  topBanner?: ReactNode
}

// Group each user message with the assistant turn(s) that follow it so the
// human bubble can `position: sticky` against the scroller across its whole
// turn (see StickyHumanMessageContainer in thread.tsx).
function buildGroups(signature: string): MessageGroup[] {
  if (!signature) {
    return []
  }

  const messages = signature.split('\n').map(row => {
    const [index, id, role, weight] = row.split(':')

    return { id, index: Number(index), role, weight: Number(weight) || 1 }
  })

  const groups: MessageGroup[] = []

  for (let i = 0; i < messages.length; i++) {
    const message = messages[i]

    if (message.role !== 'user') {
      groups.push({ id: message.id, index: message.index, kind: 'standalone', weight: message.weight })

      continue
    }

    const indices = [message.index]
    let weight = message.weight

    while (i + 1 < messages.length && messages[i + 1].role !== 'user') {
      weight += messages[++i].weight
      indices.push(messages[i].index)
    }

    groups.push({ id: message.id, indices, kind: 'turn', weight })
  }

  return groups
}

const ThreadMessageListInner: FC<ThreadMessageListProps> = ({
  clampToComposer,
  components,
  emptyPlaceholder,
  loadingIndicator,
  sessionKey,
  topBanner
}) => {
  const messageSignature = useAuiState(s =>
    s.thread.messages
      .map((message, index) => `${index}:${message.id}:${message.role}:${message.content?.length ?? 1}`)
      .join('\n')
  )

  const { t } = useI18n()
  // ★ 长会话性能: 400+ 轮时 buildGroups 的 split+loop 是 O(N)。用 messageSignature 做
  //   memo key —— signature 是 useAuiState 稳定输出, 变了才重算。省掉每次父渲染都
  //   重跑一遍 O(N)。
  const groups = useMemo(() => buildGroups(messageSignature), [messageSignature])
  // 有 topBanner 时空态也走非空分支 (沿 chat-gutter 左对齐), 避免 empty 容器的
  // mx-auto max-w-composer 把气泡居中封顶。
  const renderEmpty = groups.length === 0 && Boolean(emptyPlaceholder) && !topBanner

  // use-stick-to-bottom owns scrollTop (single writer): follow while locked,
  // escape on user scroll-up, re-lock at bottom. Snap instantly, not spring — a
  // spring can't tell live-token growth from a session-switch bulk relayout, and
  // chasing the latter reads as the view scrolling to random spots before
  // settling. Its refs hang off our own DOM so the sticky human bubbles survive.
  const { scrollRef, contentRef, isAtBottom, scrollToBottom, stopScroll } = useStickToBottom({
    initial: 'instant',
    resize: 'instant'
  })

  const [renderBudget, setRenderBudget] = useState(INITIAL_RENDER_BUDGET)

  // Walk turns newest-first, summing their part weights until the budget is met;
  // everything before that first kept turn is hidden.
  let firstVisible = groups.length

  for (let i = groups.length - 1, weight = 0; i >= 0; i--) {
    weight += groups[i].weight
    firstVisible = i

    if (weight >= renderBudget) {
      break
    }
  }

  const hiddenCount = firstVisible
  const visibleGroups = hiddenCount > 0 ? groups.slice(hiddenCount) : groups
  const restoreFromBottomRef = useRef<number | null>(null)
  // Secondary windows (new-session scratch, subagent watch, cmd-click pop-out)
  // hide the titlebar tool cluster + session header, but the OS traffic lights
  // still sit in the top-left, so reserve the titlebar gap above the transcript.
  const secondaryWindow = isSecondaryWindow()
  // NB: CSS calc() requires whitespace around the +/- operator. This string is
  // assigned verbatim to the --sticky-human-top inline style below (it does not
  // go through Tailwind, which would auto-space it), so the spaces are load-
  // bearing — without them the declaration is invalid, gets dropped, and the
  // sticky user bubble falls back to its ~4px default and slides under the OS
  // traffic lights.
  const secondaryTitlebarGap = 'calc(var(--titlebar-height) + 0.75rem)'

  const threadContentTopPad = secondaryWindow
    ? 'pt-[calc(var(--titlebar-height)+0.75rem)]'
    : 'pt-[calc(var(--titlebar-height)-0.5rem)]'

  useEffect(() => setThreadAtBottom(isAtBottom), [isAtBottom])
  useEffect(() => () => resetThreadScroll(), [])

  // Floating jump button (outside this subtree) → return to the bottom.
  useEffect(() => onScrollToBottomRequest(() => void scrollToBottom()), [scrollToBottom])

  const endEditHold = useCallback(() => {
    scrollRef.current?.removeAttribute('data-editing')
  }, [scrollRef])

  // Inline edit grows a sticky bubble. Escape before focus/layout so the
  // resize-follow can't snap scrollTop; native anchoring holds the viewport.
  const beginEditHold = useCallback(() => {
    const el = scrollRef.current

    if (!el) {
      return
    }

    endEditHold()
    stopScroll()
    el.setAttribute('data-editing', 'true')
  }, [endEditHold, scrollRef, stopScroll])

  useEffect(() => onThreadEditOpen(beginEditHold), [beginEditHold])
  useEffect(() => onThreadEditClose(endEditHold), [endEditHold])
  useEffect(() => () => endEditHold(), [endEditHold])
  // New run → snap to the latest turn.
  useAuiEvent('thread.runStart', () => void scrollToBottom())

  // Reset the cap and pin to bottom on mount + every session switch (messages
  // swap in place on a long-lived runtime, so sessionKey is the only signal).
  // The swap is multi-step and lays out over many frames; letting the library
  // follow re-pins every frame to a moving target — visible as ~10 scroll jumps.
  // Instead: quiet it, glue to the true bottom until the height holds steady,
  // then hand back locked. Live streaming afterward uses the normal resize follow.
  useLayoutEffect(() => {
    // ★ 会话切换/mount: 先秒开一个小窗 (INITIAL_RENDER_BUDGET), 别一上来就 mount
    //   几百 turn 的 markdown/katex 卡死主线程。稍后 idle 时静默升到 WARM (让快速向上
    //   滚也能顺畅看到常规历史范围)。用户按 showEarlier 才继续加。
    setRenderBudget(INITIAL_RENDER_BUDGET)

    const el = scrollRef.current

    if (!el) {
      return
    }

    stopScroll()
    el.scrollTop = el.scrollHeight

    let frame = 0
    let stableFrames = 0
    let lastHeight = el.scrollHeight

    const settle = () => {
      const node = scrollRef.current

      if (!node) {
        return
      }

      const height = node.scrollHeight

      stableFrames = height === lastHeight ? stableFrames + 1 : 0
      lastHeight = height
      node.scrollTop = height

      // ~5 steady frames ≈ layout has settled; the frame cap bounds slow loads.
      if (stableFrames >= 5 || ++frame > 90) {
        void scrollToBottom('instant')

        return
      }

      rafId = requestAnimationFrame(settle)
    }

    let rafId = requestAnimationFrame(settle)

    return () => cancelAnimationFrame(rafId)
  }, [scrollRef, scrollToBottom, sessionKey, stopScroll])

  // ★ 加载完初窗后, 静默升到 WARM (100 turn) 让快速向上滚也能拿到近期历史范围, 免得
  //   用户每次滚上都得点 showEarlier。用 setTimeout 挪到 idle: 初窗已 mount+滚定后再
  //   跑 —— 首屏可交互延迟不受影响, 只是"再加一档"在后台悄悄发生。
  //   会话切换重跑 (sessionKey 变) → 先取消上一次 pending, 免得旧 session 的 warm-up
  //   套到新 session 头上。用户已手动 showEarlier 升过就不覆盖 (取 max)。
  useEffect(() => {
    const timer = setTimeout(() => {
      setRenderBudget(cur => (cur < RENDER_BUDGET_WARM ? RENDER_BUDGET_WARM : cur))
    }, RENDER_BUDGET_WARMUP_MS)
    return () => clearTimeout(timer)
  }, [sessionKey])

  // Prepend an older page while preserving the on-screen position. The user is
  // scrolled up (reading history) so the stick-to-bottom lock is escaped and
  // won't fight this manual restore.
  const showEarlier = useCallback(() => {
    const el = scrollRef.current

    restoreFromBottomRef.current = el ? el.scrollHeight - el.scrollTop : null
    setRenderBudget(budget => budget + RENDER_BUDGET_STEP)
  }, [scrollRef])

  useLayoutEffect(() => {
    const el = scrollRef.current

    if (el && restoreFromBottomRef.current != null) {
      el.scrollTop = el.scrollHeight - restoreFromBottomRef.current
      restoreFromBottomRef.current = null
    }
  }, [scrollRef, renderBudget])

  return (
    <div
      className="relative min-h-0 max-w-full overflow-hidden contain-[layout_paint]"
      style={
        {
          height: clampToComposer ? 'var(--thread-viewport-height)' : '100%',
          ...(secondaryWindow ? { '--sticky-human-top': secondaryTitlebarGap } : {})
        } as CSSProperties
      }
    >
      {secondaryWindow && (
        // Secondary windows hide the titlebar chrome, so the scroller runs to
        // the window's top edge and streamed text slides up under the OS
        // traffic lights. Content padding alone scrolls away with the text — a
        // fixed opaque strip (the titlebar's drag region) masks anything behind
        // it and keeps the window draggable, matching the main window's header.
        <div
          aria-hidden="true"
          className="absolute inset-x-0 top-0 z-10 h-(--titlebar-height) bg-background [-webkit-app-region:drag]"
        />
      )}
      <div
        className="size-full overflow-x-hidden overflow-y-auto overscroll-contain"
        data-following={isAtBottom ? 'true' : 'false'}
        data-slot="aui_thread-viewport"
        ref={scrollRef as React.RefCallback<HTMLDivElement>}
      >
        {renderEmpty ? (
          <div
            className="mx-auto grid h-full w-full max-w-(--composer-width) grid-rows-[minmax(0,1fr)_auto] min-w-0 gap-(--conversation-turn-gap) px-6 py-8"
            data-slot="aui_thread-content"
          >
            {emptyPlaceholder}
          </div>
        ) : (
          <div
            className={cn(
              // 撑满整个列宽, 不再居中封顶; 左留 --chat-gutter, 右留更宽的 --chat-gutter-right。
              'flex w-full min-w-0 flex-col pl-(--chat-gutter) pr-(--chat-gutter-right)',
              threadContentTopPad
            )}
            data-slot="aui_thread-content"
            ref={contentRef as React.RefCallback<HTMLDivElement>}
          >
            {topBanner}
            {hiddenCount > 0 && (
              <button
                className="mx-auto mb-(--conversation-turn-gap) rounded-full border border-border/65 bg-(--composer-fill) px-3 py-1 text-xs text-muted-foreground hover:text-foreground"
                onClick={showEarlier}
                type="button"
              >
                {t.assistant.thread.showEarlier}
              </button>
            )}
            {visibleGroups.map(group => (
              <div
                className="flex min-w-0 flex-col gap-(--conversation-turn-gap) pb-(--conversation-turn-gap)"
                key={group.id}
              >
                <MessageRenderBoundary resetKey={messageSignature}>
                  {group.kind === 'turn' ? (
                    <div
                      className="composer-human-ai-pair-container relative flex min-w-0 flex-col gap-(--conversation-turn-gap)"
                      data-slot="aui_turn-pair"
                    >
                      {group.indices.map(index => (
                        <ThreadPrimitive.MessageByIndex components={components} index={index} key={index} />
                      ))}
                    </div>
                  ) : (
                    <ThreadPrimitive.MessageByIndex components={components} index={group.index} />
                  )}
                </MessageRenderBoundary>
              </div>
            ))}
            {loadingIndicator}
            {clampToComposer && (
              <div
                aria-hidden="true"
                className="shrink-0"
                data-slot="aui_composer-clearance"
                style={{ height: 'var(--thread-last-message-clearance)' }}
              />
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export const ThreadMessageList = memo(ThreadMessageListInner)
