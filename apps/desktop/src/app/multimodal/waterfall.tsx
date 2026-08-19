import { useStore } from '@nanostores/react'
import { memo, type ReactNode, useState } from 'react'

import { CompactMarkdown } from '@/components/chat/compact-markdown'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { useI18n } from '@/i18n'
import { Volume2 } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { $mmMessages, answerMultimodalClarify, fmtClock, type MmMessage } from '@/store/multimodal'
import { $mmTtsEnabled, $mmVoiceDialogEnabled, speakText } from '@/store/multimodal-voice'

import { DisclosureTitle, MmDisclosure } from './disclosure'

/**
 * Standalone chat waterfall for the multimodal page. Renders chat / tool /
 * status / clarify rows in one column. Deliberately NOT assistant-ui — the
 * multimodal stream has sub-agent (monitor / deep-research) messages + inline
 * clarify that don't fit the session runtime. Markdown reuses the shared
 * Streamdown-based CompactMarkdown (katex/code highlighting).
 *
 * Visual language reuses the main desktop chat's primitives (DisclosureRow via
 * MmDisclosure, --conversation-* typography, glass bubbles) so it reads as part
 * of the product, not a bespoke widget set:
 *   • user      → glass bubble (--dt-user-bubble) with avatar
 *   • assistant → framed bubble, main-chat typography
 *   • thinking / tool → collapsible DisclosureRow (hover caret, auto-collapse)
 *   • 深度分析/监控 → calm left-accent card
 */
export function Waterfall() {
  const messages = useStore($mmMessages)

  return (
    <div className="mx-auto flex w-full max-w-3xl flex-col gap-5 px-1 pb-4">
      {messages
        // The in-progress watcher stream renders in the DeepPanel (right
        // column); the central chat shows only its final threaded-back answer.
        // Without this, the streaming router bubble AND the threadback bubble
        // both appeared here → the deep answer rendered twice.
        .filter(m => !(m.subRole === 'router' && !m.threadback))
        .map(m => (
          <Row key={m.id} m={m} />
        ))}
    </div>
  )
}

/**
 * ★ 性能: Row 用 memo 包裹, 按 `m` 引用浅比。runUnifiedFlush 每 80ms flush 时, .map
 *   里未变的消息直接 `return m` (同引用, 见 store/multimodal.ts), 只有正在追加 delta
 *   的活跃 bubble 换新对象。没 memo 时 Waterfall 每次 flush 全量重建所有 Row (含历史
 *   monitor/watcher/assistant 卡片, 每张都重跑 CompactMarkdown 解析) → 长会话 / 密集
 *   深研时掉帧。加 memo 后流式期只重渲染活跃那一条, 历史全部跳过。
 */
const Row = memo(function Row({ m }: { m: MmMessage }) {
  const { t } = useI18n()
  const c = t.multimodal.composer
  // ★ 逐条 ▶ 朗读按钮只在"无自动播报"时显示 (喇叭关 且 对话关)。任一自动播报开着 →
  //   隐藏, 防"自动念 + 手动点"双重播放。hook 必须在早 return 前无条件调用。
  const autoSpeakOn = useStore($mmTtsEnabled) || useStore($mmVoiceDialogEnabled)
  if (m.kind === 'clarify') return <ClarifyRow m={m} />

  if (m.kind === 'tool') {
    const live = !m.toolDone
    const title = (
      <span className="flex min-w-0 items-center gap-1.5">
        <span className={cn('shrink-0', m.toolDone ? 'text-(--ui-green)' : 'animate-pulse text-(--ui-text-tertiary)')}>
          {m.toolDone ? '✓' : '◌'}
        </span>
        <DisclosureTitle live={live}>{m.toolName || 'tool'}</DisclosureTitle>
        {m.toolSummary && (
          <span className="min-w-0 truncate text-(--ui-text-tertiary)">· {m.toolSummary}</span>
        )}
      </span>
    )
    // Two-segment card: header line = the "正在派发 … 事件ID: #req_xxx" label
    // (toolSummary), body = the result note (toolDetail) — one box, the body
    // reads as the newline-separated second segment. No detail → plain line.
    return (
      <MmDisclosure defaultOpen={false} syncOpen={false} title={title}>
        <div data-selectable-text="true" className="whitespace-pre-wrap pl-[1.375rem] text-(--ui-text-secondary)">
          {m.toolDetail || m.toolSummary || c.noMoreInfo}
        </div>
      </MmDisclosure>
    )
  }

  if (m.kind === 'status') {
    return (
      <div
        data-selectable-text="true"
        className="text-[length:var(--conversation-tool-font-size)] italic text-(--ui-text-tertiary)"
      >
        {m.text}
      </div>
    )
  }

  // ── User: left-aligned, avatar + framed bubble (web-style) ────────────────
  if (m.role === 'user') {
    return (
      <div className="flex gap-2">
        <Avatar>U</Avatar>
        <div className="min-w-0 flex-1">
          <div className="mb-1 flex items-center gap-1.5 text-xs text-(--ui-text-tertiary)">
            <span>You</span>
            <Clock m={m} />
            {m.voice && <span title={c.voiceInput}>🎤</span>}
          </div>
          <div data-selectable-text="true" className="whitespace-pre-wrap break-words rounded-md border border-(--dt-user-bubble-border) bg-(--dt-user-bubble) px-3 py-2 text-sm text-(--ui-text-primary)">
            {m.text}
          </div>
        </div>
      </div>
    )
  }

  // ── Watcher per-round report: a monitor-style notification card, but the
  //    (often multi-line) body is FOLDED by default so it doesn't flood the chat.
  //    This is an ephemeral $mmMessages bubble — never written to history.
  if (m.subRole === 'watcher_report') {
    const tag = m.brief || t.multimodal.deepAnalysis.title
    return (
      <div className="rounded-lg border-l-2 border-(--ui-purple) bg-(--ui-chat-surface-background) py-2 pl-3 pr-3">
        <MmDisclosure
          title={
            <span className="flex items-center gap-1.5">
              <span className="rounded bg-(--ui-purple)/15 px-1.5 py-0.5 text-[10px] font-medium text-(--ui-purple)">
                🔬 {tag}
              </span>
              <Clock m={m} />
            </span>
          }
        >
          <div data-selectable-text="true" className="pt-1">
            <CompactMarkdown className="text-sm text-(--ui-text-primary)" text={m.text} />
          </div>
        </MmDisclosure>
      </div>
    )
  }

  // ── Monitor / Deep-research: calm left-accent card ────────────────────────
  if (m.subRole === 'monitor' || m.subRole === 'router') {
    const isMonitor = m.subRole === 'monitor'
    // Router bubbles are labelled with the analysis EVENT NAME (the brief the
    // user asked), replacing the old "已回传主对话" — falls back to "深度分析".
    const tag = isMonitor ? m.monitorLabel || c.monitorTag : m.brief || t.multimodal.deepAnalysis.title
    return (
      <div
        className={cn(
          'rounded-lg border-l-2 bg-(--ui-chat-surface-background) py-2 pl-3 pr-3',
          isMonitor ? 'border-(--ui-yellow)' : 'border-(--ui-purple)'
        )}
      >
        <div className="mb-1 flex items-center gap-1.5">
          <span
            className={cn(
              'rounded px-1.5 py-0.5 text-[10px] font-medium',
              isMonitor ? 'bg-(--ui-yellow)/15 text-(--ui-yellow)' : 'bg-(--ui-purple)/15 text-(--ui-purple)'
            )}
          >
            {isMonitor ? '👁 ' : '🔬 '}
            {tag}
          </span>
          <Clock m={m} />
        </div>
        {m.isError ? (
          <div data-selectable-text="true" className="text-sm text-(--ui-red)">{m.text}</div>
        ) : (
          <div data-selectable-text="true">
            <CompactMarkdown className="text-sm text-(--ui-text-primary)" text={m.text} />
          </div>
        )}
      </div>
    )
  }

  // ── System / error notice ─────────────────────────────────────────────────
  if (m.role === 'system' || m.isError) {
    return (
      <div
        data-selectable-text="true"
        className={cn(
          'mx-auto max-w-[90%] rounded-md px-3 py-1.5 text-center text-xs',
          m.isError ? 'bg-(--ui-red)/10 text-(--ui-red)' : 'text-(--ui-text-tertiary)'
        )}
      >
        {m.text}
      </div>
    )
  }

  // ── Assistant: avatar + reasoning disclosure + framed answer bubble ───────
  return (
    <div className="flex gap-2">
      <Avatar>A</Avatar>
      <div className="min-w-0 flex-1">
        <div className="mb-1 flex items-center gap-1.5 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
          <span>Assistant</span>
          <Clock m={m} />
          {!autoSpeakOn && !m.streaming && m.text.trim() && (
            <button
              className="text-(--ui-text-quaternary) transition-colors hover:text-(--ui-text-secondary)"
              onClick={() => speakText(m.text)}
              title={c.readAloud}
            >
              <Volume2 className="size-3.5" />
            </button>
          )}
        </div>
        {m.reasoning ? (
          // "Thinking" disclosure: auto-open while the answer is still streaming,
          // auto-collapse once done — same behaviour as the main chat.
          // ★ B4: 正文还没开始 (reasoning 到了但 answer 未出) = 正在思考 → 标题带跳动
          //   三点, 避免长思考被误认为卡死 (对齐 web)。
          <div className="mb-1.5">
            <MmDisclosure
              syncOpen={m.streaming}
              title={
                <DisclosureTitle live={m.streaming}>
                  {c.thinkingProcess}
                  {m.streaming && !m.text.trim() && (
                    <span className="ml-1 inline-flex gap-0.5 align-middle">
                      <span className="size-1 animate-bounce rounded-full bg-(--ui-text-tertiary) [animation-delay:-0.3s]" />
                      <span className="size-1 animate-bounce rounded-full bg-(--ui-text-tertiary) [animation-delay:-0.15s]" />
                      <span className="size-1 animate-bounce rounded-full bg-(--ui-text-tertiary)" />
                    </span>
                  )}
                </DisclosureTitle>
              }
            >
              <div data-selectable-text="true" className="whitespace-pre-wrap text-(--ui-text-secondary)">
                {m.reasoning}
              </div>
            </MmDisclosure>
          </div>
        ) : null}
        <div
          data-selectable-text="true"
          className="break-words rounded-lg border border-(--ui-stroke-secondary) bg-(--ui-chat-surface-background) px-3 py-2 text-[length:var(--conversation-text-font-size)] leading-(--conversation-line-height) text-(--ui-text-primary)"
        >
          <AssistantAnswer text={m.text} streaming={!!m.streaming} />
        </div>
      </div>
    </div>
  )
})

/**
 * Assistant answer body with LIVE markdown rendering while streaming.
 * ★ 不再叠加 120ms useThrottledValue: text 现在只在 store 的统一 80ms flush 时才变
 *   (见 multimodal.ts runUnifiedFlush), 唯一节流就是那层。再叠一层与之相位错开会
 *   产生"一段段"拍频 (对齐 web 的修复)。memo: 只有 text/streaming 变才重解析 markdown。
 * parseIncompleteMarkdown handles half-written syntax mid-stream. A blinking
 * caret marks the live tail.
 */
const AssistantAnswer = memo(function AssistantAnswer({ text, streaming }: { text: string; streaming: boolean }) {
  return (
    <>
      <CompactMarkdown className="text-(--ui-text-primary)" streaming={streaming} text={text} />
      {streaming && <span className="animate-pulse text-(--ui-accent)">▍</span>}
    </>
  )
})

/** Absolute HH:MM:SS timestamp shown beside a message's role name. */
function Clock({ m }: { m: MmMessage }) {
  const t = fmtClock(m.createdAt)
  if (!t) return null
  return <span className="tabular-nums text-(--ui-text-quaternary)">{t}</span>
}

/** Small round avatar bubble shared by user / assistant rows (web-style). */
function Avatar({ children }: { children: ReactNode }) {
  return (
    <div className="flex size-7 flex-shrink-0 items-center justify-center rounded-full bg-(--ui-bg-tertiary) text-xs font-semibold text-(--ui-text-secondary)">
      {children}
    </div>
  )
}

/**
 * Inline clarify row. Before answering: question + option buttons (or a text
 * field for open-ended). After answering: collapses to a compact "已选择：…"
 * line (matches the web behavior). The answer already reached the tool via
 * clarify.respond; this is presentation only.
 */
function ClarifyRow({ m }: { m: MmMessage }) {
  const { t } = useI18n()
  const c = t.multimodal.composer
  const reqId = m.clarifyReqId || ''
  const answered = m.clarifyAnswer !== undefined
  const choices = m.clarifyChoices || []
  const openEnded = choices.length === 0
  const [draft, setDraft] = useState('')

  if (answered) {
    return (
      <div className="flex items-center gap-2 text-xs text-(--ui-text-tertiary)">
        <span className="flex size-4 items-center justify-center rounded-full bg-(--ui-bg-tertiary) text-[10px]">✓</span>
        <span>
          {c.selectedLabel('')}<span className="text-(--ui-text-primary)">{m.clarifyAnswer || c.emptyChoice}</span>
        </span>
      </div>
    )
  }

  const submitText = () => {
    const value = draft.trim()
    if (!value) return
    void answerMultimodalClarify(reqId, value)
    setDraft('')
  }

  return (
    <div className="rounded-lg border-l-2 border-(--ui-yellow) bg-(--ui-chat-surface-background) py-2 pl-3 pr-3">
      <div className="mb-1.5 flex items-center gap-1.5">
        <span className="rounded bg-(--ui-yellow)/15 px-1.5 py-0.5 text-[10px] font-medium text-(--ui-yellow)">{c.needConfirm}</span>
      </div>
      <div className="mb-2 whitespace-pre-wrap text-sm text-(--ui-text-primary)">{m.clarifyQuestion}</div>
      {!openEnded ? (
        <div className="flex flex-wrap gap-1.5">
          {choices.map(c => (
            <Button key={c} onClick={() => void answerMultimodalClarify(reqId, c)} size="xs" variant="outline">
              {c}
            </Button>
          ))}
        </div>
      ) : (
        <div className="flex items-center gap-1.5">
          <Textarea
            className="min-h-0 flex-1"
            onChange={e => setDraft(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                submitText()
              }
            }}
            placeholder={c.answerPlaceholderEnter}
            rows={1}
            value={draft}
          />
          <Button onClick={submitText} size="xs">
            {c.submit}
          </Button>
        </div>
      )}
    </div>
  )
}
