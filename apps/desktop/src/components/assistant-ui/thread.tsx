import type { Unstable_TriggerAdapter, Unstable_TriggerItem } from '@assistant-ui/core'
import {
  ActionBarPrimitive,
  BranchPickerPrimitive,
  ComposerPrimitive,
  ErrorPrimitive,
  MessagePrimitive,
  type ToolCallMessagePartProps,
  useAui,
  useAuiState,
  useMessageRuntime
} from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import {
  type ClipboardEvent,
  type ComponentProps,
  type FC,
  type FocusEvent,
  type FormEvent,
  type KeyboardEvent,
  type DragEvent as ReactDragEvent,
  type ReactNode,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState
} from 'react'

import { COMPOSER_DROP_ACTIVE_CLASS, COMPOSER_DROP_FADE_CLASS } from '@/app/chat/composer/drop-affordance'
import {
  type ComposerInsertMode,
  focusComposerInput,
  markActiveComposer,
  onComposerFocusRequest,
  onComposerInsertRequest
} from '@/app/chat/composer/focus'
import { useAtCompletions } from '@/app/chat/composer/hooks/use-at-completions'
import { useSlashCompletions } from '@/app/chat/composer/hooks/use-slash-completions'
import { QueryWorkerTrajectoryPanel } from '@/app/multimodal/query-worker-trajectory-panel'
import { queryWorkerTaskId } from '@/app/multimodal/query-worker-trajectory'
import type { MmTrajectoryEntry } from '@/app/multimodal/trajectory-grouping'
import {
  dragHasAttachments,
  droppedFileInlineRefs,
  type InlineRefInput,
  insertInlineRefsIntoEditor
} from '@/app/chat/composer/inline-refs'
import {
  composerPlainText,
  placeCaretEnd,
  refChipElement,
  renderComposerContents,
  RICH_INPUT_SLOT
} from '@/app/chat/composer/rich-editor'
import { detectTrigger, textBeforeCaret, type TriggerState } from '@/app/chat/composer/text-utils'
import { ComposerTriggerPopover } from '@/app/chat/composer/trigger-popover'
import {
  extractDroppedFiles,
  HERMES_PATHS_MIME,
  isImagePath,
  partitionDroppedFiles
} from '@/app/chat/hooks/use-composer-actions'
import { uploadComposerAttachment } from '@/app/session/hooks/use-prompt-actions'
import { ClarifyTool } from '@/components/assistant-ui/clarify-tool'
import { DirectiveContent, hermesDirectiveFormatter } from '@/components/assistant-ui/directive-text'
import { MarkdownText, MarkdownTextContent } from '@/components/assistant-ui/markdown-text'
import { ThreadMessageList } from '@/components/assistant-ui/thread-list'
import { ThreadTimeline } from '@/components/assistant-ui/thread-timeline'
import { ToolFallback, ToolGroupSlot } from '@/components/assistant-ui/tool-fallback'
import {
  selectMessageHasPendingTool,
  selectMessageHasVisibleText,
  selectMessageRunning,
  summarizeToolSteps,
  toolPartDisclosureId
} from '@/components/assistant-ui/tool-fallback-model'
import { TooltipIconButton } from '@/components/assistant-ui/tooltip-icon-button'
import { UserMessageText } from '@/components/assistant-ui/user-message-text'
import { useElapsedSeconds } from '@/components/chat/activity-timer'
import { ActivityTimerText } from '@/components/chat/activity-timer-text'
import { DisclosureRow } from '@/components/chat/disclosure-row'
import { GeneratedImage } from '@/components/chat/generated-image-result'
import { Intro, type IntroProps } from '@/components/chat/intro'
import { PreviewAttachment } from '@/components/chat/preview-attachment'
import { Codicon } from '@/components/ui/codicon'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import { CopyButton } from '@/components/ui/copy-button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { Loader } from '@/components/ui/loader'
import type { HermesGateway } from '@/hermes'
import { useResizeObserver } from '@/hooks/use-resize-observer'
import { useI18n } from '@/i18n'
import { Badge } from '@/components/ui/badge'
import { attachmentDisplayText, attachmentId, pathLabel } from '@/lib/chat-runtime'
import { DATA_IMAGE_URL_RE } from '@/lib/embedded-images'
import { LinkifiedText } from '@/lib/external-link'
import { triggerHaptic } from '@/lib/haptics'
import { Eye, GitBranchIcon, Loader2Icon, Play, Search, StopFilled, Volume2Icon, VolumeXIcon, XIcon } from '@/lib/icons'
import { extractPreviewTargets } from '@/lib/preview-targets'
import { useEnterAnimation } from '@/lib/use-enter-animation'
import { cn } from '@/lib/utils'
import { playSpeechText, stopVoicePlayback } from '@/lib/voice-playback'
import { $backgroundResume } from '@/store/background-delegation'
import { queryTrajectoryTaskStore } from '@/store/multimodal'
import { $compactionActive } from '@/store/compaction'
import type { ComposerAttachment } from '@/store/composer'
import { notifyError } from '@/store/notifications'
import { $activeSessionAwaitingInput } from '@/store/prompts'
import { fmtClock } from '@/store/multimodal'
import { $connection } from '@/store/session'
import { notifyThreadEditClose } from '@/store/thread-scroll'
import { $voicePlayback } from '@/store/voice-playback'
import { $generatingToolName } from '@/store/tool-generating'
import { $toolDisclosureStates, setToolDisclosureOpenBatch } from '@/store/tool-view'
import { $isWindowResizing } from '@/store/window-resize'

type ThreadLoadingState = 'response' | 'session'
interface RestoreMessageTarget {
  text: string
  userOrdinal: number | null
}

interface MessageActionProps {
  messageId: string
  /** Lazy accessor — reads the live message text at action time. Passing the
   *  text itself as a prop forces the whole footer to re-render on every
   *  streaming delta flush (the text changes ~30×/s), which profiling showed
   *  was a large slice of per-token script time on long transcripts. */
  getMessageText: () => string
  onBranchInNewChat?: (messageId: string) => void
}

let readAloudAudio: HTMLAudioElement | null = null

function partText(part: unknown): string {
  if (typeof part === 'string') {
    return part
  }

  if (!part || typeof part !== 'object') {
    return ''
  }

  const row = part as { text?: unknown; type?: unknown }

  return (!row.type || row.type === 'text') && typeof row.text === 'string' ? row.text : ''
}

function messageContentText(content: unknown): string {
  if (typeof content === 'string') {
    return content.trim()
  }

  return Array.isArray(content) ? content.map(partText).join('').trim() : ''
}

// Cheap streaming-stable "does this message have visible text" check: returns
// on the first non-whitespace text part without concatenating the whole
// message. Used as a useAuiState selector so its boolean output stays stable
// across token flushes (flips false→true once per turn).
function contentHasVisibleText(content: unknown): boolean {
  if (typeof content === 'string') {
    return content.trim().length > 0
  }

  if (!Array.isArray(content)) {
    return false
  }

  for (const part of content) {
    if (partText(part).trim().length > 0) {
      return true
    }
  }

  return false
}

export const Thread: FC<{
  clampToComposer?: boolean
  cwd?: string | null
  gateway?: HermesGateway | null
  intro?: IntroProps
  loading?: ThreadLoadingState
  onBranchInNewChat?: (messageId: string) => void
  onCancel?: () => Promise<void> | void
  onDismissError?: (messageId: string) => void
  onRestoreToMessage?: (messageId: string, target?: RestoreMessageTarget) => Promise<void> | void
  sessionId?: string | null
  sessionKey?: string | null
}> = ({
  clampToComposer = false,
  cwd = null,
  gateway = null,
  intro,
  loading,
  onBranchInNewChat,
  onCancel,
  onDismissError,
  onRestoreToMessage,
  sessionId = null,
  sessionKey
}) => {
  const { t } = useI18n()
  const copy = t.assistant.thread

  const [restoreConfirmTarget, setRestoreConfirmTarget] = useState<
    (RestoreMessageTarget & { messageId: string }) | null
  >(null)

  const closeRestoreConfirm = useCallback(() => setRestoreConfirmTarget(null), [])

  const confirmRestore = useCallback(() => {
    if (!restoreConfirmTarget || !onRestoreToMessage) {
      throw new Error('Restore is unavailable for this message.')
    }

    const { messageId, text, userOrdinal } = restoreConfirmTarget

    closeRestoreConfirm()
    void Promise.resolve(onRestoreToMessage(messageId, { text, userOrdinal })).catch((error: unknown) => {
      notifyError(error, 'Restore failed')
    })
  }, [closeRestoreConfirm, onRestoreToMessage, restoreConfirmTarget])

  const requestRestoreConfirm = useCallback((messageId: string, target: RestoreMessageTarget) => {
    setRestoreConfirmTarget({ messageId, ...target })
  }, [])

  const messageComponents = useMemo(
    () => ({
      AssistantMessage: () => (
        <AssistantMessage onBranchInNewChat={onBranchInNewChat} onDismissError={onDismissError} />
      ),
      SystemMessage,
      UserEditComposer: () => <UserEditComposer cwd={cwd} gateway={gateway} sessionId={sessionId} />,
      UserMessage: () => (
        <UserMessage
          onCancel={onCancel}
          onRequestRestoreConfirm={onRestoreToMessage ? requestRestoreConfirm : undefined}
        />
      )
    }),
    [cwd, gateway, onBranchInNewChat, onCancel, onDismissError, onRestoreToMessage, requestRestoreConfirm, sessionId]
  )

  // ★ 多模态引导气泡: 常驻主 Agent 对话顶部, 空态时也显示 (空态走
  //   emptyPlaceholder, 非空时走 topBanner 与消息一起滚动)。发送消息后不消失。
  const introBubble = intro ? <Intro {...intro} /> : undefined
  const emptyPlaceholder = introBubble ? (
    <div className="flex min-h-0 w-full flex-col items-stretch justify-start px-4 pt-4">
      {introBubble}
    </div>
  ) : undefined

  return (
    <div className="relative grid h-full min-h-0 max-w-full grid-rows-[minmax(0,1fr)] overflow-hidden bg-transparent contain-[layout_paint]">
      <ThreadMessageList
        clampToComposer={clampToComposer}
        components={messageComponents}
        emptyPlaceholder={emptyPlaceholder}
        loadingIndicator={loading === 'response' ? <ResponseLoadingIndicator /> : <BackgroundResumeNotice />}
        sessionKey={sessionKey}
        topBanner={introBubble}
      />
      {loading === 'session' && <CenteredThreadSpinner />}
      <ThreadTimeline />
      <ConfirmDialog
        confirmLabel={copy.restoreConfirm}
        description={copy.restoreBody}
        destructive
        onClose={closeRestoreConfirm}
        onConfirm={confirmRestore}
        open={Boolean(restoreConfirmTarget)}
        title={copy.restoreTitle}
      />
    </div>
  )
}

function pickPrimaryPreviewTarget(targets: string[]): string[] {
  if (targets.length <= 1) {
    return targets
  }

  const localUrl = targets.find(value => /^https?:\/\/(?:localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\])/i.test(value))

  return [localUrl || targets[targets.length - 1]]
}

const CenteredThreadSpinner: FC = () => {
  const { t } = useI18n()

  return (
    <div
      aria-label={t.assistant.thread.loadingSession}
      className="pointer-events-none absolute inset-0 z-1 grid place-items-center"
      role="status"
    >
      <Loader
        aria-hidden="true"
        className="size-12 text-midground/70"
        pathSteps={220}
        role="presentation"
        strokeScale={0.72}
        type="rose-curve"
      />
    </div>
  )
}

// Sub-role metadata carried on an assistant message (monitor SPEAK /
// deep-research threadback) — see toRuntimeMessage's metadata.custom.
interface SubRoleMeta {
  subRole?: 'monitor' | 'router' | 'watcher_report'
  monitorLabel?: string
  brief?: string
  deepReportRid?: string
  deepRange?: string
  deepRound?: number
  model?: string
  voice?: boolean
}

// (SubRoleHeader 已移除: monitor/router 的头部行现在渲染在各自卡片内第一行,
//  watcher_report 用其 <details> summary; 不再有卡片外的独立头部行。)

// ── Web-style 头像 + 头部行 (对齐 web 多模态页): 圆形头像 (U/A) + 角色名 + 时间 +
//    (assistant) 模型徽标 + 播放按钮。配色全用桌面 token, 不引入 web 硬编码色。 ──────
const MessageAvatar: FC<{ role: 'user' | 'assistant' }> = ({ role }) => (
  <div
    className={cn(
      'flex size-7 shrink-0 select-none items-center justify-center rounded-full text-[0.7rem] font-semibold',
      role === 'user'
        ? 'bg-(--ui-accent)/20 text-(--ui-accent)'
        : 'bg-(--ui-purple)/15 text-(--ui-purple)'
    )}
    aria-hidden="true"
  >
    {role === 'user' ? 'U' : 'A'}
  </div>
)

// 可见的播放/停止按钮 (放在 assistant 头部行)。复用主聊天原生的 $voicePlayback /
// playSpeechText / stopVoicePlayback: 它已带 messageId + 单实例语义 (playSpeechText
// 内部先 stopVoicePlayback → 切到别条会自动停旧的, 旧按钮据 messageId 复位为播放)。
const ReadAloudButton: FC<{ getText: () => string; messageId: string }> = ({ getText, messageId }) => {
  const { t } = useI18n()
  const copy = t.assistant.thread
  const voicePlayback = useStore($voicePlayback)
  const status =
    voicePlayback.source === 'read-aloud' && voicePlayback.messageId === messageId
      ? voicePlayback.status
      : 'idle'
  const isPreparing = status === 'preparing'
  const isSpeaking = status === 'speaking'
  // 播放 ▶ (三角) / 停止 ■ (StopFilled) / 准备中转圈。
  const Icon = isPreparing ? Loader2Icon : isSpeaking ? StopFilled : Play
  const onClick = useCallback(async () => {
    if (isSpeaking) {
      void stopVoicePlayback()
      return
    }
    if (isPreparing) return
    const text = getText()
    if (!text) return
    try {
      await playSpeechText(text, { messageId, source: 'read-aloud' })
    } catch (error) {
      notifyError(error, copy.readAloudFailed)
    }
  }, [copy.readAloudFailed, getText, isPreparing, isSpeaking, messageId])
  return (
    <button
      className="inline-flex items-center gap-1 rounded border border-(--ui-stroke-tertiary) px-1.5 py-0.5 text-[0.6rem] text-(--ui-text-tertiary) hover:bg-(--ui-control-hover-background) hover:text-foreground [-webkit-app-region:no-drag]"
      onClick={() => void onClick()}
      title={isSpeaking ? 'Stop' : 'Play'}
      type="button"
    >
      <Icon className={cn('size-3', isPreparing && 'animate-spin')} />
      {isSpeaking ? 'Stop' : 'Play'}
    </button>
  )
}

// 普通 assistant 头部行: A 角色名 + 时间 + 模型徽标 + 播放按钮。
const AssistantHeaderRow: FC<{
  createdAt?: Date
  model?: string
  getText: () => string
  messageId: string
  showActions?: boolean
  onBranchInNewChat?: (messageId: string) => void
}> = ({ createdAt, model, getText, messageId, showActions, onBranchInNewChat }) => {
  const clock = fmtClock(createdAt ? createdAt.getTime() : undefined)
  return (
    <div className="mb-1 flex flex-wrap items-center gap-1.5 text-[0.65rem] text-(--ui-text-tertiary)">
      <span className="font-medium text-(--ui-text-secondary)">Assistant</span>
      {clock && <span className="tabular-nums text-(--ui-text-quaternary)">{clock}</span>}
      {model && <Badge variant="outline">{model}</Badge>}
      <ReadAloudButton getText={getText} messageId={messageId} />
      {/* Copy / Reload / More: 挪到头部行右侧 (ml-auto), 不再占正文下方一整行。
         hover 显隐仍靠 MessagePrimitive.Root 的 group。 */}
      {showActions && (
        <div className="ml-auto">
          <AssistantActionBar getMessageText={getText} messageId={messageId} onBranchInNewChat={onBranchInNewChat} />
        </div>
      )}
    </div>
  )
}

// 深度回传正文卡: 受控折叠 (非 <details>, 因为原生 details 折叠时整块隐藏, 做不到"露一行")。
// 默认折叠 → 三角 ▸ + 正文单行行末省略号 (truncate); 点三角展开 → ▾ + 多行全文。
// 头部 (第N段/时段区间/时间/#id) 在卡片外, 这里只管正文的折叠。
const WatcherReportBody: FC = () => {
  const { t } = useI18n()
  const [open, setOpen] = useState(false)
  // ★ 与 web 对齐: 正文卡头显示「第N段 + 时段区间」。段号走 deepRound(独立字段),
  //   时段走 deepRange —— 之前 desktop 抠出 deepRange 却没渲染 (bug 级遗漏)。
  const subMeta = useAuiState(s => (s.message.metadata?.custom ?? {}) as SubRoleMeta)
  const segLabel = subMeta.deepRound != null ? t.multimodal.deepAnalysis.segment(subMeta.deepRound) : ''
  const rangeLabel = subMeta.deepRange || ''
  return (
    <div className="rounded-lg border-l-2 border-(--ui-purple) bg-(--ui-purple)/8 px-3 py-2">
      <button
        className="flex w-full cursor-pointer select-none items-start gap-1.5 text-left text-[length:var(--conversation-text-font-size)] leading-(--dt-line-height) text-foreground"
        onClick={() => setOpen(o => !o)}
        type="button"
      >
        <span className={cn('mt-0.5 shrink-0 select-none text-(--ui-text-quaternary) transition-transform', open && 'rotate-90')}>
          ▸
        </span>
        {(segLabel || rangeLabel) && (
          <span className="mt-0.5 shrink-0 tabular-nums text-(--ui-text-tertiary)">
            {[segLabel, rangeLabel].filter(Boolean).join(' ')}
          </span>
        )}
        {/* 折叠: 用 max-height 夹成约一行高 + overflow-hidden (line-clamp/truncate 对
            块级 markdown 子元素无效, 这才可靠)。折叠时右下角叠一个 " ..." 表示省略。
            展开: max-height 放开显示全文, 隐藏省略号。 */}
        <div className="relative min-w-0 max-w-full">
          <div
            className="wrap-anywhere overflow-hidden"
            style={{ maxHeight: open ? 'none' : '1.6em' }}
          >
            <MessagePrimitive.Parts components={MESSAGE_PARTS_COMPONENTS} />
          </div>
          {!open && (
            <span className="pointer-events-none absolute bottom-0 right-0 bg-(--ui-purple)/8 pl-1 text-(--ui-text-tertiary)">
              {' ...'}
            </span>
          )}
        </div>
      </button>
    </div>
  )
}

const AssistantMessage: FC<{
  onBranchInNewChat?: (messageId: string) => void
  onDismissError?: (messageId: string) => void
}> = ({ onBranchInNewChat, onDismissError }) => {
  const messageId = useAuiState(s => s.message.id)
  const messageRuntime = useMessageRuntime()
  const { t } = useI18n()

  // Sub-role tag (monitor SPEAK / deep-research threadback). Stable across token
  // flushes, so reading it here doesn't add per-delta re-renders.
  const subMeta = useAuiState(s => (s.message.metadata?.custom ?? {}) as SubRoleMeta)
  const subRole = subMeta.subRole
  const messageCreatedAt = useAuiState(s => s.message.createdAt)

  // PERF: this component must NOT subscribe to the streaming text. Every
  // selector here returns a value that stays referentially stable across
  // token flushes (booleans, status strings, '' while running), so the
  // 30 Hz delta stream only re-renders the markdown part and the tiny
  // StreamStallIndicator leaf — not the footer/preview/root subtree.
  const messageStatus = useAuiState(s => s.message.status?.type)
  const isRunning = messageStatus === 'running'
  const isPlaceholder = useAuiState(s => s.message.status?.type === 'running' && s.message.content.length === 0)
  const hasVisibleText = useAuiState(s => contentHasVisibleText(s.message.content))

  // Preview targets only materialize once the turn completes — while running
  // the selector returns '' (stable), so per-token flushes skip the regex
  // scan and the re-render it would cause.
  const completedText = useAuiState(s =>
    s.message.status?.type === 'running' ? '' : messageContentText(s.message.content)
  )

  const previewTargets = useMemo(() => {
    if (!completedText || !/(https?:\/\/|file:\/\/)/i.test(completedText)) {
      return []
    }

    return pickPrimaryPreviewTarget(extractPreviewTargets(completedText))
  }, [completedText])

  const getMessageText = useCallback(() => messageContentText(messageRuntime.getState().content), [messageRuntime])

  const enterRef = useEnterAnimation(isRunning, `assistant-message:${messageId}`)

  // ★ 占位态 (running 但 content 还完全为空, 一个 part 都没到) → 本组件不渲染, 交给
  //   Thread 的 ResponseLoadingIndicator 独占这一行, 也就是用户看到的
  //   "Waiting response…" → (3s 无 delta) "Thinking…"。
  //   首个 part (reasoning/tool) 落地后 content 非空, 本分支不再命中, 下面的完整结构
  //   (答案框 + 过程框) 接管 —— 这是整轮里唯一一次结构变化。
  if (isPlaceholder) {
    return null
  }

  // ★ 首个 part 落地之后【不再】有第二种形态。此处原先还有一条"纯思考态"分支: 运行中但
  //   正文未到时只渲染一行 💭 状态 (无头像/无头部/无卡片/无工具), 等正文落地才整块换成
  //   完整卡片。那一行确实安静, 但代价是整轮过程【完全不可见】—— reasoning 原文和工具的
  //   参数/结果在那个窗口里根本没有渲染者 (ToolGroupSlot 从未挂载, ToolHistoryPanel 被
  //   hasPendingTool 挡住), 而且正文到达那一帧要把"一行"整体换成"头像+头部+卡片+页脚+
  //   过程框", 下方内容全部位移 —— 就是用户报的"界面又乱了"。
  //
  //   现在: 从第一个 part 起就是最终结构。正文未到时卡片靠 empty:hidden 收成 0 高 (依赖
  //   下方 Parts 的 unstable_showEmptyOnNonTextEnd={false}, 否则库会塞一个空 Text part
  //   进去、卡片就不再 :empty 了), 思考与工具进展实时长在卡片下方的过程框里。整轮零结构跳变。
  return (
    <MessagePrimitive.Root
      className="group flex w-full min-w-0 max-w-full flex-row gap-2 self-start overflow-hidden"
      data-role="assistant"
      data-slot="aui_assistant-message-root"
      data-streaming={isRunning ? 'true' : undefined}
      ref={enterRef}
    >
      {/* Web 风格头像列 (A / subRole emoji), body 列靠它自然缩进。 */}
      <MessageAvatar role="assistant" />
      <div className="flex min-w-0 flex-1 flex-col gap-0">
      {/* 普通 assistant: 头部行 (角色名+时间+模型+播放按钮)。subRole 用各自的 SubRoleHeader。 */}
      {!subRole && (
        <AssistantHeaderRow
          createdAt={messageCreatedAt}
          getText={getMessageText}
          messageId={messageId}
          model={subMeta.model}
          onBranchInNewChat={onBranchInNewChat}
          showActions={hasVisibleText}
        />
      )}
      {/* 监控 / 深度分析(router/watcher_report) 卡片外头部行: [事件tab (含"第N段")] [绝对时间]
         左对齐, #事件id 右对齐 (ml-auto)。 */}
      {(subRole === 'monitor' || subRole === 'router' || subRole === 'watcher_report') && (
        <div className="mb-1 flex items-center gap-1.5 text-[0.65rem]">
          <Badge
            className={cn(
              'bg-transparent px-0',
              subRole === 'monitor' ? 'text-(--ui-yellow)' : 'text-(--ui-purple)'
            )}
            variant="muted"
          >
            {subRole === 'monitor' ? <Eye /> : <Search />}
            {subRole === 'monitor'
              ? subMeta.monitorLabel || t.assistant.thread.monitorAlert
              : subMeta.monitorLabel || subMeta.brief || t.assistant.thread.deepResearch}
          </Badge>
          {fmtClock(messageCreatedAt ? messageCreatedAt.getTime() : undefined) && (
            <span className="tabular-nums text-(--ui-text-quaternary)">
              {fmtClock(messageCreatedAt ? messageCreatedAt.getTime() : undefined)}
            </span>
          )}
          {subMeta.deepReportRid && (
            <span className="ml-auto font-mono text-(--ui-text-quaternary)">#{subMeta.deepReportRid}</span>
          )}
        </div>
      )}
      {/* ★ 停顿兜底行 (卡片外)。此处原先还有 CurrentActivityLine ("💭 Running: …" +
         计时) —— 现在它搬进了下方过程框的标题行: 过程框全程可见, 两处都报当前动作就是
         一模一样的两行并排, 正是当初要消灭的"重复 Thinking 行"。停顿指示器留在卡外,
         因为它报的是"流卡住了"这个跨部件的整体状态, 不属于任何一个 part。 */}
      {isRunning && !subRole && <StreamStallIndicator />}
      {/* ★ 完成态 reasoning 不再单独占一块: 它已经并入正文卡【下方】的过程框
         (ToolHistoryPanel 里的 ☁️ 折叠项)。此前它是正文卡【上方】的独立面板, 且
         每段 reasoning 各出一个 disclosure —— 一轮里因此出现两行 "Thinking"
         (其中一行显示的 reasoning 原文恰好提到某条命令, 看着像多出来的工具行),
         过程信息被答案框劈成上下两半。现在一轮 = 答案框 + 一个过程框。 */}
      {/* 深度分析回传: 卡片内 = 第几段 + 时段区间 + 正文(单行, 行末省略)。 */}
      {subRole === 'watcher_report' ? (
        // 深度回传: 正文左侧三角 (▸/▾), 默认折叠单行行末省略, 点三角展开全文。
        <WatcherReportBody />
      ) : (
      <div
        className={cn(
          'wrap-anywhere min-w-0 max-w-full overflow-hidden text-pretty text-[length:var(--conversation-text-font-size)] leading-(--dt-line-height) text-foreground',
          // 监控 = 偏黄, 深度研究 = 偏紫: 整卡带对应色调的淡底 (不只是左边框),
          // 与主 Assistant 的中性 --ui-bg-elevated 明显区分, 避免撞色。
          subRole === 'monitor' &&
            'rounded-lg border-l-2 border-(--ui-yellow) bg-(--ui-yellow)/8 py-2 pl-3 pr-3',
          subRole === 'router' &&
            'rounded-lg border-l-2 border-(--ui-purple) bg-(--ui-purple)/8 py-2 pl-3 pr-3',
          // ★ 普通主 Assistant 回复 (无 subRole): 浅中性底色卡片 (--ui-bg-elevated),
          //   比 user 气泡浅一档, 一眼区分, 且与紫色深度卡不混淆。
          //   ★ empty:hidden 现在是【承重】的, 不再只是兜底: 本卡从第一个 part 起就挂上,
          //   而正文可能还没到 (先 reasoning / 先工具), 这时它必须收成 0 高、看不出有个空框。
          //   配套依赖下方 Parts 的 unstable_showEmptyOnNonTextEnd={false}。
          !subRole && 'rounded-lg bg-(--ui-bg-elevated) px-3 py-2 empty:hidden'
        )}
        data-slot="aui_assistant-message-content"
      >
        {/* 监控/router 头部行已移到卡片外 (见上); 此处只放正文。
           当前动作/计时在下方过程框的标题行, 停顿兜底 (StreamStallIndicator) 在卡外 ——
           卡内只承载 message.parts 生成的正文节点 (工具行也不在这里, 见 ToolGroupSlot)。 */}
        {/* Todos render in the composer status stack now, not inline. */}
        {/* ★ unstable_showEmptyOnNonTextEnd={false} 是承重的, 不是调优: 该 prop 默认 true,
           当【最后一个 part 不是 text/reasoning】(正文未到、末项是 tool-call, 也就是本轮
           执行中的常态) 时, 库会额外渲染一个空 Text part。那个空节点让卡片不再匹配 :empty,
           于是 empty:hidden 失效 —— 屏幕上多出一个可见的空白卡。置 false 后卡片真正为空、
           收成 0 高, 正文一到再自然长出来。 */}
        <MessagePrimitive.Parts
          components={MESSAGE_PARTS_COMPONENTS}
          unstable_showEmptyOnNonTextEnd={false}
        />
        {previewTargets.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-2">
            {previewTargets.map(target => (
              <PreviewAttachment key={target} source="explicit-link" target={target} />
            ))}
          </div>
        )}
        <MessagePrimitive.Error>
          <ErrorPrimitive.Root
            className="mt-1.5 flex items-start gap-1.5 text-[0.78rem] leading-5 text-[color-mix(in_srgb,var(--dt-destructive)_78%,var(--ui-text-secondary))]"
            role="alert"
          >
            <ErrorPrimitive.Message className="min-w-0 flex-1" />
            {onDismissError && (
              <TooltipIconButton
                className="-my-0.5 shrink-0 text-current opacity-70 hover:opacity-100"
                onClick={() => onDismissError(messageId)}
                side="top"
                tooltip={t.assistant.thread.dismissError}
              >
                <XIcon className="size-3.5" />
              </TooltipIconButton>
            )}
          </ErrorPrimitive.Root>
        </MessagePrimitive.Error>
      </div>
      )}
      {hasVisibleText && subRole !== 'watcher_report' && (
        <AssistantFooter getMessageText={getMessageText} messageId={messageId} onBranchInNewChat={onBranchInNewChat} />
      )}
      {/* ★ 工具历史折叠区: 正文落地后, 工具从卡内消失 (ToolGroupSlot 返回 null),
         转而以独立白框呈现在 footer 下方。在 body 列内, 自然和正文卡左对齐,
         但没有额外的头像/header。默认折叠一行 "N tool calls", 点击展开。 */}
      {!subRole && <ToolHistoryPanel />}
      </div>
    </MessagePrimitive.Root>
  )
}

const StatusRow: FC<{ children: ReactNode; label: string } & React.ComponentPropsWithoutRef<'div'>> = ({
  children,
  label,
  className,
  ...rest
}) => (
  <div
    aria-label={label}
    aria-live="polite"
    className={cn('flex w-full max-w-full items-center gap-1.5 self-start text-xs text-muted-foreground/70', className)}
    role="status"
    {...rest}
  >
    {children}
  </div>
)

// Fixed label while auto-compaction runs — decoupled from backend status text.
const COMPACTION_LABEL = 'Summarizing thread'

const CompactionHint: FC = () => (
  <span className="shimmer min-w-0 truncate text-muted-foreground/55">{COMPACTION_LABEL}</span>
)

const ResponseLoadingIndicator: FC = () => {
  const { t } = useI18n()
  const elapsed = useElapsedSeconds()
  const compacting = useStore($compactionActive)

  // ★ 事件驱动状态机 (与 web ThinkingLine 对齐):
  //   - 流开始 → "Waiting response…"
  //   - 3s 都还没任何 delta → 兜底切 "Thinking…" (覆盖闭源模型不透 reasoning 但
  //     内部真推理的场景, 如 GPT-5.6 Luna)。
  //   一旦第一个 reasoning / tool / message part 到达, AssistantMessage 挂上
  //   (答案框 + 过程框), 由过程框标题行接管当前动作与计时 —— 本组件就消失。
  const [fallbackThinking, setFallbackThinking] = useState(false)
  useEffect(() => {
    const timer = window.setTimeout(() => setFallbackThinking(true), 3000)
    return () => window.clearTimeout(timer)
  }, [])
  const thinkingLabel = fallbackThinking ? t.assistant.thread.thinking : 'Waiting response…'

  // ★ 【不显示头像 A】, 但保留头像列宽度的隐形占位 (size-7 + gap-2), 让这行文字与下方
  //   "You"/正文卡的左缘对齐。一行 💭 状态, 无头部行、无正文卡 —— 这是首个 part 落地
  //   【之前】的形态。首个 part 到达 → 本组件卸载, AssistantMessage 挂上完整结构
  //   (头像 + 头部 + 答案框 + 过程框), 这是整轮唯一一次结构变化。
  return (
    <div
      aria-label={compacting ? COMPACTION_LABEL : t.assistant.thread.loadingResponse}
      aria-live="polite"
      className="group flex w-full min-w-0 max-w-full flex-row items-center gap-2 self-start overflow-hidden"
      data-slot="aui_response-loading"
      role="status"
    >
      {/* 隐形头像占位: 与 AssistantMessage 的 MessageAvatar (size-7 + gap-2) 同宽,
          让 "Thinking/Waiting" 行 body 与 "You"/正文卡左缘对齐。 */}
      <div aria-hidden className="size-7 shrink-0" />
      <StatusRow label="">
        {compacting ? (
          <CompactionHint />
        ) : (
          <>
            <span className="animate-pulse">💭</span>
            <span className="shimmer min-w-0 truncate text-muted-foreground/70">{thinkingLabel}</span>
          </>
        )}
        <ActivityTimerText className="ml-auto" seconds={elapsed} />
      </StatusRow>
    </div>
  )
}

// Parked-background affordance: a top-level delegate_task runs in the
// background, so the parent turn ends and the app goes idle while the subagent
// keeps working and its result re-enters as a fresh turn later. Instead of a
// spinner (reads as "stuck"), reuse the same compact, centered system-note
// chrome as the steer / slash-status lines (SystemMessage above) so it sits in
// the thread like every other meta line. Idle-only (gated upstream). Null when
// nothing is parked.
const BackgroundResumeNotice: FC = () => {
  const { t } = useI18n()
  const resume = useStore($backgroundResume)

  if (!resume) {
    return null
  }

  const label = resume.activity ?? t.assistant.thread.resumeWhenBackgroundDone(resume.count)

  return (
    <div
      aria-live="polite"
      className="flex max-w-[min(86%,44rem)] items-center gap-1.5 self-center px-2 py-0.5 text-[0.6875rem] leading-5 text-muted-foreground/55"
      data-slot="aui_background-resume"
      role="status"
    >
      <Codicon className="text-muted-foreground/55" name="sync" size="0.75rem" />
      <span className="shimmer min-w-0 truncate">{label}</span>
    </div>
  )
}

// Seconds of no visible output (text or part count) before a still-running turn
// is treated as stalled and the thinking indicator returns at the tail.
const STREAM_STALL_S = 2

// Tail "still thinking" indicator: the pre-first-token spinner goes away once
// text flows, but if the stream then goes quiet mid-turn (tool think-time,
// provider stall) nothing signals that work continues. Watch a per-flush
// activity signal; when it hasn't changed for STREAM_STALL_S, re-show the
// dither + a timer counting from the last activity.
//
// Subscribes to the activity signal ITSELF (rather than taking it as a prop)
// so that per-token updates re-render only this leaf, not the whole
// AssistantMessage subtree.
// Exported for tests only (like ToolHistoryPanel): the duplicate-"Thinking"
// regression this component caused had NO coverage, which is why it survived the
// 0fea14e1 cleanup that removed its sibling CurrentActivityLine.
export const StreamStallIndicator: FC = () => {
  const activity = useAuiState(s => {
    let textLength = 0

    for (const part of s.message.content) {
      const text = (part as { text?: unknown }).text

      if (typeof text === 'string') {
        textLength += text.length
      }
    }

    return `${s.message.content.length}:${textLength}`
  })

  // ★ 过程框 (ToolHistoryPanel) 挂载时【本行不出】。它的标题行已经在报同一件事,
  //   而且报得更好: 有具体动作 (activityLabel)、shimmer、计时。两处都报就是当初
  //   0fea14e1 要消灭的"重复 Thinking 行" —— CurrentActivityLine 那时被搬进了
  //   过程框, 但本组件漏了, 于是长工具调用 (stream 静默 >2s 必然触发) 时框外一行
  //   死的灰字、框内一行活的, 且框外这行是 StatusRow (纯 div, 无 disclosure) ——
  //   点不开、无内容、不在思考框里。
  //   条件与 ToolHistoryPanel 的挂载条件严格一致 (见那里的 early return)。
  const panelMounted = useAuiState(s => {
    let hasReasoning = false
    let toolCount = 0

    for (const part of s.message.parts) {
      const p = part as { text?: unknown; type?: string } | null

      if (p?.type === 'tool-call') {
        toolCount += 1
      } else if (p?.type === 'reasoning' && typeof p.text === 'string' && p.text.trim().length > 0) {
        hasReasoning = true
      }
    }

    return hasReasoning || toolCount > 0
  })

  // tool.generating 也算过程框在场: 参数还在写, 一个 part 都没有, 但框已为它挂载。
  const generatingTool = useStore($generatingToolName)

  const [stalled, setStalled] = useState(false)
  const compacting = useStore($compactionActive)
  const { t } = useI18n()
  // A pending clarify / approval / sudo / secret means the turn is paused on the
  // user, not working — so don't resurrect the "thinking" timer while they
  // decide (matches the pet's awaitingInput pose taking priority over busy).
  const awaitingInput = useStore($activeSessionAwaitingInput)

  useEffect(() => {
    setStalled(false)
    const id = window.setTimeout(() => setStalled(true), STREAM_STALL_S * 1000)

    return () => window.clearTimeout(id)
  }, [activity])

  // 压缩中是全局态 (与本轮 part 无关), 所以不受过程框影响; 停顿指示仅在过程框缺席时补位。
  const active = (compacting || (stalled && !panelMounted && !generatingTool)) && !awaitingInput
  const elapsed = useElapsedSeconds(active)

  if (!active) {
    return null
  }

  return (
    <StatusRow
      className="mt-1.5"
      data-slot="aui_stream-stall"
      label={compacting ? COMPACTION_LABEL : t.assistant.thread.thinking}
    >
      {compacting ? (
        <CompactionHint />
      ) : (
        <>
          <span className="animate-pulse">💭</span>
          <span className="shimmer min-w-0 truncate text-muted-foreground/70">{t.assistant.thread.thinking}</span>
        </>
      )}
      <ActivityTimerText className="ml-auto" seconds={elapsed} />
    </StatusRow>
  )
}

// ─────────────────────────────────────────────────────────────────────────
// "当前动作"标签 (Claude Code 风格的心跳): 一行持续告诉用户 agent 现在在干什么。
// 消费者是过程框的标题行 (ToolHistoryPanel 的 headerLabel) —— 跑动中显示动作,
// 完成后换成批次摘要。encrypted reasoning 场景也不会只剩一个空框。
//
// 从【消息 parts 数组的末尾一项】派生成一句人话:
//   * tool-call part -> "Reading /path/to/file"、"Running: git status"…
//   * 其它 (无工具 / 纯文本刚开始) -> "Thinking"
//   * 末项是正文 text -> '' (正文已开始, 不必再报动作)
//
// 返回【小体积派生字符串】而不是对象/数组: useAuiState 用 Object.is 比对, 只有这样
// 才能让每次 token flush 不触发订阅方 re-render (parts 数组身份每次都变, 但这个
// 字符串只在"动作切换"时才变)。计时器由调用方用 ActivityTimerText +
// useElapsedSeconds 提供, timerKey 按消息 id 稳定, 不因动作切换而重置总时长。
const _MAX_ACTIVITY_LEN = 80

function _truncate(s: string, max = _MAX_ACTIVITY_LEN): string {
  const t = s.trim()
  if (t.length <= max) return t
  return t.slice(0, max - 1).trimEnd() + '…'
}

function _pickToolLabel(toolName: string, args: unknown): string {
  // 只提取一个字段作为一句话摘要, 不调 buildToolView (那是完整视图, 每次 flush
  // 都跑正则/JSON 解析太贵)。字段挑选参考 tool-fallback-model.ts 里各工具用的
  // 主参数名, 覆盖读文件/终端/浏览器/搜索这几大类, 其它工具回退到 toolName 本身。
  const a = (args && typeof args === 'object') ? (args as Record<string, unknown>) : {}
  const str = (v: unknown): string => (typeof v === 'string' ? v : '')

  // 路径类
  const p = str(a.path) || str(a.file_path) || str(a.filepath) || str(a.filename)
  if (p) {
    if (toolName === 'read_file' || toolName === 'view_file') return `Reading ${p}`
    if (toolName === 'edit_file' || toolName === 'str_replace_editor'
        || toolName === 'apply_patch' || toolName === 'write_file') return `Editing ${p}`
    return `${toolName} ${p}`
  }
  // Shell / code
  const cmd = str(a.command) || str(a.code)
  if (cmd) return `Running: ${_truncate(cmd, 60)}`
  // 搜索类
  const q = str(a.query) || str(a.pattern) || str(a.q)

  if (q) {
    // query_multimodal may inspect current frames, recall buffered evidence,
    // or hand the answer to QueryWorker. Calling every one of those paths a
    // web-style "Search" made a healthy visual dispatch look like the wrong
    // tool was running.
    if (toolName === 'query_multimodal') {
      return `QueryWorker: ${_truncate(q, 60)}`
    }

    return `Searching: ${_truncate(q, 60)}`
  }
  // 浏览器导航
  const u = str(a.url)
  if (u) return `Opening ${u}`
  // 兜底: 直接用工具名
  return toolName
}

// activitySnapshot: 从 parts 末尾派生【一句显示标签】+ 一个稳定 kind 判据。
// 返回一个 [label, kind] 元组的编码字符串, 让 useAuiState 内部做 identity 比对
// 只在真正变化时触发 re-render。返回 '' 表示"不显示这一行"(正文已开始 / 无 parts)。
//
// 直接返回 string 而不是 object: useAuiState 的默认比较是 Object.is, 每次新对象
// 都会触发 re-render, 用 primitive string 才能保证 token flush 不重渲。
function _computeActivityLabel(parts: readonly unknown[]): string {
  // ★ 最大信息量原则: 一旦本轮出现过 tool-call, 始终显示最近的工具状态,
  //   即使后续出现 reasoning part 也不切回 thinking。工具信息量 >> thinking。

  // 1) 从末尾找第一个有意义的 part: 如果是 text (正文已流出) → 隐藏活动行
  for (let i = parts.length - 1; i >= 0; i--) {
    const p = parts[i] as { type?: string; toolName?: string; args?: unknown; text?: string } | null
    if (!p) continue
    if (p.type === 'text' && typeof p.text === 'string' && p.text.trim().length > 0) return ''
    // 遇到 tool-call 或 reasoning 就停止向前找 text (text 只在末尾才意味着"正文开始")
    if (p.type === 'tool-call' || p.type === 'reasoning') break
  }

  // 2) 找最近的 tool-call (不管后面是否还有 reasoning, 一旦有 tool 就锁定显示工具)
  for (let i = parts.length - 1; i >= 0; i--) {
    const p = parts[i] as { type?: string; toolName?: string; args?: unknown } | null
    if (p?.type === 'tool-call' && typeof p.toolName === 'string') {
      return _truncate(_pickToolLabel(p.toolName, p.args))
    }
  }

  // 3) 没有任何 tool-call → 一律显示稳定的 "Thinking"。
  //
  // ★ 这里【不再】滚动 reasoning 原文的最后一行。这一行报告的是【状态】,不是内容:
  //   原文尾行每来一个 delta 就换一次 (~80ms 一次 flush), 叠加 .shimmer 流光 +
  //   脉冲 emoji + 计时器, 快模型下一秒抖十几次 —— 既读不清内容, 也失去了"还活着"
  //   这个唯一诚实的信号。reasoning 全文没有丢: 完成态由过程框里的 ☁️ 折叠项
  //   (ReasoningInlineItem) 静态呈现, 可以静下来读。与 web 端 ThinkingLine 一致。
  return 'Thinking'
}

// (CurrentActivityLine / ThinkingBubble 已删除。它们是"正文卡上方一行 💭 当前动作 +
//  计时"的旧表达, 与全程可见的过程框标题行报告的是同一件事 —— 两者同屏就是重复的
//  Thinking 行。_computeActivityLabel 仍在用: 现由 ToolHistoryPanel 的标题行消费,
//  见那里的 headerLabel。StreamStallIndicator 仍单独挂在卡外, 因为它报的是"流卡住了"
//  这个跨部件状态, 不属于任何一个 part。)

// Reasoning segments are joined with a visible rule so multi-round thinking
// keeps its inter-segment pause when read in one block (see ReasoningInlineItem).
const _REASONING_SEP = '\n\n───\n\n'

// ToolHistoryPanel — 正文卡【下方】的【过程框】: 本轮的 reasoning (☁️ 折叠项, 见
// ReasoningInlineItem) + 所有 tool-call 聚在同一个区块里。一轮 = 答案框 + 一个过程框。
//
// ★ 全程可见 (不再是"完成态才出现")。它从第一个 part 起就挂上, 执行中实时长出思考与
//   工具行, 完成后原地定稿 —— 同一个组件、同一个 DOM 位置, 没有"先藏着最后一次性出现"
//   造成的位移。此前它被 hasPendingTool + hasVisibleText 双重挡住, 于是整个执行过程
//   在界面上【完全不存在】: reasoning 原文和工具参数/结果在那个窗口里没有任何渲染者。
//
// ⚠ 渲染稳定性: 本组件现在在流式期间也挂着, 每个 delta 都会跑一遍它的 selector。所以
//   每个 useAuiState selector 都【必须】返回 primitive —— 返回新数组/新对象会被
//   Object.is 判定不等 → 每 delta 重渲染 (极端情况无限循环)。完整 content 仅在展开
//   (open) 时通过 runtime.getState() 同步读取, 不走订阅。
export const ToolHistoryPanel: FC = () => {
  const { t } = useI18n()
  const hasVisibleText = useAuiState(selectMessageHasVisibleText)
  const messageRunning = useAuiState(selectMessageRunning)
  // Stable primitive: only changes when tool-call count actually changes.
  const toolCount = useAuiState(s => {
    let count = 0

    for (const part of s.message.parts) {
      if ((part as { type?: string } | null)?.type === 'tool-call') {
        count++
      }
    }

    return count
  })
  // ★ Also a primitive (the joined label), NOT the parts array — this selector's
  //   output is compared with Object.is, so returning a fresh array/object here
  //   would re-render on every streaming delta (see the note above).
  //   "Read 2 files · Ran 1 command" beats a bare "3 tool calls", and it is the
  //   only summary a finished turn has when thinking mode is off.
  const stepSummary = useAuiState(s =>
    summarizeToolSteps(
      s.message.parts as readonly { isError?: boolean; toolName?: string; type?: string }[],
      t.assistant.thread
    )
  )

  // A tool still awaiting its result. No longer a reason to hide this panel (it
  // IS where pending rows live now — see the approval note on `open` below); it
  // only decides whether the header can report final counts.
  const hasPendingTool = useAuiState(selectMessageHasPendingTool)

  // 当前动作 ("Running: curl …" / "Thinking")。同样是 primitive string —— 见组件顶部
  // 关于 selector 必须返回 primitive 的说明。空串表示"正文已开始, 不必再报动作"。
  const activityLabel = useAuiState(s => _computeActivityLabel(s.message.parts as readonly unknown[]))

  const runtime = useMessageRuntime()
  const messageId = useAuiState(s => s.message.id)
  const disclosureStates = useStore($toolDisclosureStates)

  // This block also carries the turn's reasoning now, so a thinking-only turn
  // (no tools at all) must still render it — otherwise the rationale would be
  // silently dropped, which is what happened before reasoning moved in here.
  const hasReasoning = useAuiState(s =>
    s.message.parts.some(part => {
      const p = part as { text?: unknown; type?: string } | null

      return p?.type === 'reasoning' && typeof p.text === 'string' && p.text.trim().length > 0
    })
  )

  // 待用户输入 (clarify / approval / sudo / secret) 时本轮是【停在人身上】, 不是在干活,
  // 所以计时暂停 —— 否则把用户自己犹豫的时间也算成 agent 耗时。这一点在过程框里尤其
  // 重要: 待批准的 Run/Reject 条就挂在本框的工具行上, 那正是最典型的 awaitingInput。
  // (行为承自已删除的 CurrentActivityLine, 与宠物 awaitingInput 姿态优先于 busy 一致。)
  const awaitingInput = useStore($activeSessionAwaitingInput)
  // 模型正在写某个工具调用的参数 (gateway 的 tool.generating)。刻意【不是】一个 part:
  // 它没有 tool_id, 变成 part 就会留下 tool.start 无法归并的孤儿行 (当年的"重复工具行"
  // bug)。所以只在标题行报一句"正在准备", 不出行。见 store/tool-generating.ts。
  const generatingToolRaw = useStore($generatingToolName)
  // 只在本轮真的还在跑时采信: 这个 store 是 session 级的, 历史消息重渲染时不该跟着
  // 显示别人的"正在准备"。
  const generatingTool = messageRunning ? generatingToolRaw : ''
  // ⚠ 必须在任何 early return 之【前】调用 (Hook 规则)。计时 key 绑 messageId, 所以
  //   本轮内思考↔工具切换不会把总时长清零, 换一轮才重新计。
  const elapsed = useElapsedSeconds(messageRunning && !awaitingInput, `activity:${messageId}`)

  const [userOpen, setUserOpen] = useState<boolean | null>(null)
  // Auto-open while the turn is running, auto-collapse once it finishes and there
  // is prose to read. The first explicit user toggle wins from then on (`userOpen`).
  //
  // ⚠ 承重, 不只是观感: 待批准的工具行 (PendingToolApproval / Run·Reject 条) 只存在于
  //   这个框【展开时】渲染出来的那一行里。运行中默认展开, 用户才有东西可点; 否则本轮会
  //   一直卡在等待批准而界面上看不到任何入口。用户手动折叠也是允许的 —— 那一行随之卸载,
  //   registerApprovalInlineAnchor 的计数归零, 于是 PendingApprovalFallback (composer
  //   上方的浮动批准条) 自动接管, 不会真的锁死。
  //
  //   完成态而【没有】正文时也保持展开: 那种轮次 (被中断 / 纯思考回复) 框里就是全部内容,
  //   折叠等于把整轮藏进一个小三角后面。
  const open = userOpen ?? (messageRunning || !hasVisibleText)

  // 只要本轮有过程 (思考或工具) 就渲染, 与是否已有正文、是否还在跑都无关 —— 这正是
  // "执行中也要看得见过程"的要求。两者都没有 (纯正文回答) 才什么都不出。
  //
  // ★ generatingTool 也算"有过程": 模型正在写工具参数, 但此时【一个 part 都还没有】
  //   (tool.start 要等参数写完 + 预检跑完才发, 且一批多个调用是攒齐一起发)。只看 part
  //   的话这段窗口整个是空的 —— 界面看着发呆好几秒, 然后框一出来已经列着 2-3 个工具。
  //   这是本轮唯一能提前知道"要调工具了"的信号。
  if (!hasReasoning && toolCount === 0 && !generatingTool) {
    return null
  }

  // 工具行【只】在这里渲染 (ToolGroupSlot 已恒 return null), 所以不再有"卡内/框内
  // 二选一"的搬家问题, 也就不需要 hasVisibleText 这层条件。
  const showTools = toolCount > 0

  // Extract full tool parts only when expanded — avoids array allocation cost
  // when collapsed, and sidesteps the useAuiState identity problem entirely.
  const toolParts: Array<ToolCallMessagePartProps & { argsFields?: unknown }> = []

  if (open && showTools) {
    const state = runtime.getState()

    for (const part of state.content) {
      if (part.type !== 'tool-call') {
        continue
      }

      const partRuntime = runtime.getMessagePartByToolCallId(part.toolCallId)
      const partState = partRuntime.getState()
      if (partState.type !== 'tool-call') {
        continue
      }

      // Rebuild the exact renderer props from the live part runtime. This keeps
      // status/addResult/resume and the private argsFields sidecar intact, then
      // runs the same specialised dispatcher used while the turn is streaming
      // (QueryWorker trajectory, image generation, clarify, etc.).
      toolParts.push({
        ...partState,
        ...('argsFields' in part ? { argsFields: part.argsFields } : {}),
        addResult: result => partRuntime.addToolResult(result),
        resume: payload => partRuntime.resumeToolCall(payload)
      })
    }
  }

  // Compute disclosure IDs for all tool entries so we can batch toggle them.
  const toolDisclosureIds = open
    ? toolParts.map(part => {
        const toolPart = { args: part.args, isError: part.isError, result: part.result, toolCallId: part.toolCallId, toolName: part.toolName, type: 'tool-call' as const }

        return `tool-entry:${messageId}:${toolPartDisclosureId(toolPart)}`
      })
    : []

  const allToolsExpanded = toolDisclosureIds.length > 0 && toolDisclosureIds.every(id => disclosureStates[id])

  const handleToggleAllTools = () => {
    setToolDisclosureOpenBatch(toolDisclosureIds, !allToolsExpanded)
  }

  // 标题行 = 本轮过程的一句话状态。
  //   未完成 → 当前动作 ("Running: curl …" / "Thinking"), 即原 CurrentActivityLine 的
  //            职责搬到这里 —— 过程框全程可见, 框外再放一行同样的状态就是重复。
  //   完成后 → 定稿的批次摘要 ("执行了 1 条命令")。
  // 未完成时【不】显示计数摘要: 计数还会变, 每来一个工具就抖一次; 更要紧的是, 停在
  // 审批上时说"执行了 1 条命令"是错的 —— 那条命令还没跑。所以"显示什么"只看进度,
  // 不看 awaitingInput。
  const headerInProgress = messageRunning && (hasPendingTool || !hasVisibleText || Boolean(generatingTool))
  // "正在准备 <tool>" 优先于 activityLabel: 后者派生自已有 parts, 在这个窗口里最多只能
  // 说出上一个工具或泛泛的 "Thinking", 而 generatingTool 才是当下真正在发生的事。
  const headerLabel = headerInProgress
    ? generatingTool
      ? t.assistant.thread.preparingTool(generatingTool)
      : activityLabel || t.assistant.thread.thinking
    : stepSummary || t.assistant.thread.toolHistory(toolCount)
  // 而"要不要摆动效 / 走秒"另算: 停在用户身上 (审批/追问) 时 agent 并没有在干活,
  // 继续流光 + 计时就是在说谎, 也会把用户自己犹豫的时间算进耗时。
  const headerRunning = headerInProgress && !awaitingInput
  // 计时 key 绑 messageId: 本轮内 label 在思考↔工具间切换不重置总时长。
  const headerTimer = headerRunning ? <ActivityTimerText seconds={elapsed} /> : undefined

  return (
    <div
      className="w-full min-w-0 max-w-full rounded-lg bg-(--ui-bg-elevated) px-3 py-2 text-[length:var(--conversation-tool-font-size)] text-(--ui-text-tertiary)"
      data-slot="aui_tool-history-panel"
    >
      {/* 还没有任何工具行, 但正在准备工具 → 只出一行状态标题 (不可展开, 因为没有内容),
          下面接 reasoning 折叠项 (若本轮已有思考)。这就是填补 tool.generating →
          tool.start 那段空窗的最小表达: 有反馈, 但不制造一个之后无法归并的孤儿工具行。 */}
      {toolCount === 0 && generatingTool ? (
        <>
          <div className="flex items-center justify-between">
            <span
              className={cn(
                'min-w-0 truncate text-[length:var(--conversation-tool-font-size)] font-medium leading-(--conversation-line-height) text-(--ui-text-secondary)',
                headerRunning && 'shimmer'
              )}
              data-slot="aui_tool-preparing"
            >
              {headerLabel}
            </span>
            {headerTimer}
          </div>
          <ReasoningInlineItem />
        </>
      ) : /* No tools to summarise → the block IS the reasoning disclosure. Wrapping
          it in an outer "Thinking" row would make the user open "Thinking" only
          to find another "Thinking" inside. */
      !showTools || toolCount === 0 ? (
        <ReasoningInlineItem />
      ) : (
        <>
      <div className="flex items-center justify-between">
        <DisclosureRow onToggle={() => setUserOpen(!open)} open={open} trailing={headerTimer}>
          <span
            className={cn(
              'text-[length:var(--conversation-tool-font-size)] font-medium leading-(--conversation-line-height) text-(--ui-text-secondary)',
              // 跑动中走 shimmer 流光 (与工具行未完成态同一套动效语言), 完成后静态实色。
              headerRunning && 'shimmer'
            )}
          >
            {headerLabel}
          </span>
        </DisclosureRow>
        {open && toolDisclosureIds.length > 0 && (
          <button
            className="shrink-0 cursor-pointer select-none text-[length:var(--conversation-tool-font-size)] font-medium text-(--ui-text-tertiary) transition-colors hover:text-(--ui-text-primary)"
            onClick={handleToggleAllTools}
            type="button"
          >
            {allToolsExpanded ? t.assistant.thread.collapseAllTools : t.assistant.thread.expandAllTools}
          </button>
        )}
      </div>
      {open && (
        <div className="mt-1 grid min-w-0 max-w-full gap-(--tool-row-gap) overflow-hidden">
          {/* ☁️ This turn's reasoning, as ONE nested item above the calls it
              produced. Previously reasoning lived in a separate panel ABOVE the
              answer card, and `CompletedReasoningPanel` emitted one disclosure
              per reasoning segment — so a single turn showed two "thinking"
              rows (one of which displayed reasoning prose that merely mentioned
              a command, reading like a stray tool row) on one side of the answer
              and the tool list on the other. One process block, one ☁️ item. */}
          <ReasoningInlineItem />
          {toolParts.map(part => (
            <ChainToolFallback
              key={part.toolCallId || part.toolName}
              {...part}
            />
          ))}
        </div>
      )}
        </>
      )}
    </div>
  )
}

// The ☁️ reasoning row inside the process block. Merges EVERY reasoning segment
// of the turn into a single collapsible item — segments are joined rather than
// each getting their own disclosure, which is what produced the duplicate
// "Thinking" rows. Collapsed by default: raw chain-of-thought is long, often
// English even in a zh UI, and is never the answer.
const ReasoningInlineItem: FC = () => {
  const { t } = useI18n()
  const [open, setOpen] = useState(false)
  const reasoningText = useAuiState(s => {
    const segments: string[] = []

    for (const part of s.message.parts) {
      const p = part as { text?: unknown; type?: string } | null

      if (p?.type === 'reasoning' && typeof p.text === 'string') {
        const trimmed = p.text.trim()

        if (trimmed) {
          segments.push(trimmed)
        }
      }
    }

    return segments.join(_REASONING_SEP)
  })

  if (!reasoningText) {
    return null
  }

  return (
    // Keeps the established `aui_thinking-disclosure` slot: streaming.test.tsx
    // pins "exactly ONE reasoning disclosure per turn, collapsed by default"
    // against this marker, and that contract still holds here.
    <div className="min-w-0 max-w-full" data-slot="aui_thinking-disclosure">
      <DisclosureRow onToggle={() => setOpen(!open)} open={open}>
        <span className="flex min-w-0 items-baseline gap-1.5">
          <span aria-hidden>☁️</span>
          <span className="text-[length:var(--conversation-tool-font-size)] font-medium leading-(--conversation-line-height) text-(--ui-text-secondary)">
            {t.assistant.thread.thinking}
          </span>
        </span>
      </DisclosureRow>
      {open && (
        <div className="mt-0.5 max-h-40 w-full min-w-0 max-w-full overflow-auto wrap-anywhere pb-1">
          <MarkdownTextContent
            containerClassName="text-xs leading-snug text-muted-foreground/85"
            containerProps={{ 'data-slot': 'aui_reasoning-text' } as ComponentProps<'div'>}
            isRunning={false}
            text={reasoningText}
          />
        </div>
      )}
    </div>
  )
}

const ImageGenerateTool: FC<ToolCallMessagePartProps> = ({ args, result }) => {
  const aspectRatio = typeof args?.aspect_ratio === 'string' ? args.aspect_ratio : undefined

  return (
    <div className="mt-1.5">
      <GeneratedImage aspectRatio={aspectRatio} result={result} />
    </div>
  )
}

function queryWorkerResult(value: unknown): Record<string, unknown> {
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    return value as Record<string, unknown>
  }

  if (typeof value !== 'string' || !value.trim()) {
    return {}
  }

  try {
    const parsed = JSON.parse(value) as unknown

    return parsed && typeof parsed === 'object' && !Array.isArray(parsed)
      ? parsed as Record<string, unknown>
      : {}
  } catch {
    return {}
  }
}

/** The ordinary tool card remains the durable receipt. The runtime-only panel
 * below it mirrors Web's QueryWorker progress inspector and is populated from
 * the redacted trajectory sidechannel, never from model-facing history. */
export function selectQueryWorkerToolEntries(
  trajectory: MmTrajectoryEntry[],
  taskId: string,
  toolCallId: string
): MmTrajectoryEntry[] {
  if (!taskId) {
    return []
  }

  const selected = new Map<string, MmTrajectoryEntry>()

  for (const entry of trajectory) {
    const entryTaskId = queryWorkerTaskId(entry)
    const payloadToolId = typeof entry.payload.tool_id === 'string' ? entry.payload.tool_id : ''

    if (entryTaskId === taskId || (payloadToolId && payloadToolId === toolCallId)) {
      selected.set(entry.id, entry)
    }
  }

  return [...selected.values()].sort((a, b) => a.seq - b.seq || a.ts - b.ts)
}

export const QueryMultimodalTool: FC<ToolCallMessagePartProps> = props => {
  const { t } = useI18n()
  const result = queryWorkerResult(props.result)
  const taskId = typeof result.task_id === 'string' ? result.task_id.trim() : ''
  const trajectoryStore = useMemo(() => queryTrajectoryTaskStore(taskId), [taskId])
  const trajectory = useStore(trajectoryStore)

  const entries = useMemo(
    () => selectQueryWorkerToolEntries(trajectory, taskId, props.toolCallId),
    [props.toolCallId, taskId, trajectory]
  )

  return (
    <div className="space-y-2">
      <ToolFallback {...props} />
      {taskId && entries.length > 0 && <QueryWorkerTrajectoryPanel entries={entries} />}
      {taskId && entries.length === 0 && (
        <div
          className="rounded border border-cyan-400/25 bg-cyan-400/5 px-2 py-1.5 text-[0.6875rem] text-cyan-200"
          data-testid="query-worker-trajectory-waiting"
        >
          <span className="mr-1.5 inline-block animate-spin">◌</span>
          {t.multimodal.trajectory.waitingFirstEntry}
        </div>
      )}
    </div>
  )
}

const ChainToolFallback: FC<ToolCallMessagePartProps> = props => {
  // todo parts are hoisted to a dedicated panel above the message content.
  if (props.toolName === 'todo') {
    return null
  }

  if (props.toolName === 'image_generate') {
    return <ImageGenerateTool {...props} />
  }

  if (props.toolName === 'clarify') {
    return <ClarifyTool {...props} />
  }

  if (props.toolName === 'query_multimodal') {
    return <QueryMultimodalTool {...props} />
  }

  return <ToolFallback {...props} />
}

// ★ ReasoningGroup slot 全程 return null: reasoning 的 UI 表达完全交给过程框里的
//   ReasoningInlineItem (ToolHistoryPanel 内的 ☁️ 折叠项) —— 全程只有那一处、一整块,
//   不再随 message.parts 顺序内嵌到正文卡里。这样彻底避开 interleaved thinking
//   (reasoning ↔ tool_call ↔ text 交错) 把多个 "Thinking" 折叠块散布进 tool card /
//   正文之间的显示异常。
const ReasoningAccordionGroup: FC<{ children?: ReactNode; endIndex: number; startIndex: number }> = () => null

const ReasoningTextPart: FC<{ text: string; status?: { type: string } }> = ({ text, status }) => {
  const displayText = text.trimStart()
  const messageRunning = useAuiState(s => s.message.status?.type === 'running')
  const isRunning = status?.type === 'running' || messageRunning

  return (
    <MarkdownTextContent
      containerClassName="text-xs leading-snug text-muted-foreground/85"
      containerProps={{ 'data-slot': 'aui_reasoning-text' } as ComponentProps<'div'>}
      isRunning={isRunning}
      text={displayText}
    />
  )
}

// Module-level constant so the `components` prop on `MessagePrimitive.Parts`
// has a stable identity across renders. Without this every AssistantMessage
// render would create a fresh `components` object, invalidating the memo on
// `MessagePrimitivePartByIndex` and forcing every tool/reasoning child to
// re-render on every streaming delta. Memo invalidation alone doesn't
// remount, but combined with the previous ToolFallback group-swap it was a
// big chunk of the per-delta work.
const MESSAGE_PARTS_COMPONENTS = {
  Reasoning: ReasoningTextPart,
  ReasoningGroup: ReasoningAccordionGroup,
  Text: MarkdownText,
  ToolGroup: ToolGroupSlot,
  tools: { Fallback: ChainToolFallback }
} as const

const TIME_FMT = new Intl.DateTimeFormat(undefined, { hour: 'numeric', minute: '2-digit' })

const SHORT_FMT = new Intl.DateTimeFormat(undefined, {
  day: 'numeric',
  hour: 'numeric',
  minute: '2-digit',
  month: 'short'
})

function startOfDay(d: Date): number {
  return new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime()
}

function formatMessageTimestamp(
  value: Date | string | number | undefined,
  labels: { today: (time: string) => string; yesterday: (time: string) => string }
): string {
  if (!value) {
    return ''
  }

  const date = value instanceof Date ? value : new Date(value)

  if (Number.isNaN(date.getTime())) {
    return ''
  }

  const dayDelta = Math.round((startOfDay(new Date()) - startOfDay(date)) / 86_400_000)

  if (dayDelta === 0) {
    return labels.today(TIME_FMT.format(date))
  }

  if (dayDelta === 1) {
    return labels.yesterday(TIME_FMT.format(date))
  }

  return SHORT_FMT.format(date)
}

const AssistantActionBar: FC<MessageActionProps> = ({ messageId, getMessageText, onBranchInNewChat }) => {
  const { t } = useI18n()
  const copy = t.assistant.thread
  const [menuOpen, setMenuOpen] = useState(false)

  return (
    <div className="relative flex shrink-0 justify-end">
      <ActionBarPrimitive.Root
        className={cn(
          // NOTE: intentionally NOT `hideWhenRunning`. That prop unmounts the
          // bar while the thread streams, which shifts layout when the turn
          // resolves. It's invisible by default (opacity-0 + pointer-events-none,
          // reveals on hover), so keeping it mounted keeps layout stable.
          // Lives inline in the header row now — no vertical padding, sits at
          // text height, hover-reveals in place at the row's right edge.
          'relative flex flex-row items-center justify-end gap-1 opacity-0 pointer-events-none group-hover:pointer-events-auto group-hover:opacity-100 focus-within:pointer-events-auto focus-within:opacity-100',
          menuOpen && 'pointer-events-auto opacity-100 [&_button]:opacity-100'
        )}
        data-slot="aui_msg-actions"
      >
        <CopyButton appearance="icon" buttonSize="icon" label={copy.copy} text={getMessageText} />
        <ActionBarPrimitive.Reload asChild>
          <TooltipIconButton onClick={() => triggerHaptic('submit')} tooltip={copy.refresh}>
            <Codicon name="refresh" />
          </TooltipIconButton>
        </ActionBarPrimitive.Reload>
        <DropdownMenu onOpenChange={setMenuOpen} open={menuOpen}>
          <DropdownMenuTrigger asChild>
            <TooltipIconButton tooltip={copy.moreActions}>
              <Codicon name="ellipsis" />
            </TooltipIconButton>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start" onCloseAutoFocus={e => e.preventDefault()} sideOffset={6}>
            <MessageTimestamp />
            <DropdownMenuItem onSelect={() => onBranchInNewChat?.(messageId)}>
              <GitBranchIcon />
              {copy.branchNewChat}
            </DropdownMenuItem>
            <ReadAloudItem getText={getMessageText} messageId={messageId} />
          </DropdownMenuContent>
        </DropdownMenu>
      </ActionBarPrimitive.Root>
    </div>
  )
}

const ReadAloudItem: FC<{ getText: () => string; messageId: string }> = ({ getText, messageId }) => {
  const { t } = useI18n()
  const copy = t.assistant.thread
  const voicePlayback = useStore($voicePlayback)

  const readAloudStatus =
    voicePlayback.source === 'read-aloud' && voicePlayback.messageId === messageId ? voicePlayback.status : 'idle'

  const isPreparing = readAloudStatus === 'preparing'
  const isSpeaking = readAloudStatus === 'speaking'
  const anyPlaybackActive = voicePlayback.status !== 'idle'
  const Icon = isPreparing ? Loader2Icon : isSpeaking ? VolumeXIcon : Volume2Icon

  const read = useCallback(async () => {
    const text = getText()

    if (!text || $voicePlayback.get().status !== 'idle') {
      return
    }

    try {
      await playSpeechText(text, { messageId, source: 'read-aloud' })
    } catch (error) {
      notifyError(error, copy.readAloudFailed)
    }
  }, [copy.readAloudFailed, getText, messageId])

  return (
    <DropdownMenuItem
      disabled={isPreparing || (!isSpeaking && anyPlaybackActive)}
      onSelect={e => {
        e.preventDefault()
        void (isSpeaking ? stopVoicePlayback() : read())
      }}
    >
      <Icon className={isPreparing ? 'animate-spin' : undefined} />
      {isPreparing ? copy.preparingAudio : isSpeaking ? copy.stopReading : copy.readAloud}
    </DropdownMenuItem>
  )
}

const MessageTimestamp: FC = () => {
  const { t } = useI18n()
  const createdAt = useAuiState(s => s.message.createdAt)
  const label = formatMessageTimestamp(createdAt, t.assistant.thread)

  if (!label) {
    return null
  }

  return <DropdownMenuLabel className="text-xs font-normal text-muted-foreground">{label}</DropdownMenuLabel>
}

// Footer now only carries the branch picker (hidden when single branch). The
// Copy / Reload / More action bar moved up into the header row (right-aligned),
// so a completed reply no longer reserves an extra footer row below the prose.
const AssistantFooter: FC<MessageActionProps> = () => (
  <div className="flex flex-col items-end gap-1 pr-(--message-text-indent) pl-(--message-text-indent) empty:hidden">
    {/* empty:hidden + no min-height: with a single branch the picker renders
        nothing, so this row collapses to 0 instead of reserving space. */}
    <BranchPickerPrimitive.Root
      className="inline-flex h-6 items-center gap-1 text-xs text-muted-foreground"
      hideWhenSingleBranch
    >
      <BranchPickerPrimitive.Previous className="grid size-6 place-items-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-foreground disabled:cursor-default disabled:opacity-35">
        <Codicon name="chevron-left" size="0.875rem" />
      </BranchPickerPrimitive.Previous>
      <span className="tabular-nums">
        <BranchPickerPrimitive.Number /> / <BranchPickerPrimitive.Count />
      </span>
      <BranchPickerPrimitive.Next className="grid size-6 place-items-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-foreground disabled:cursor-default disabled:opacity-35">
        <Codicon name="chevron-right" size="0.875rem" />
      </BranchPickerPrimitive.Next>
    </BranchPickerPrimitive.Root>
  </div>
)

const EMPTY_ATTACHMENT_REFS: string[] = []

function messageAttachmentRefs(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return EMPTY_ATTACHMENT_REFS
  }

  return value.every(ref => typeof ref === 'string') ? value : EMPTY_ATTACHMENT_REFS
}

function StickyHumanMessageContainer({
  attachments,
  children,
  messageId
}: {
  attachments?: ReactNode
  children: ReactNode
  messageId?: string
}) {
  return (
    // Fragment, not a wrapper: a wrapping element becomes the sticky's
    // containing block (it'd stick within its own height = never). The bubble
    // and attachments are flow siblings so the bubble pins against the scroller
    // while attachments below it scroll away.
    <>
      <div
        className="group/user-message sticky z-40 -mx-4 flex w-[calc(100%+2rem)] min-w-0 max-w-none flex-col items-stretch gap-0 self-start overflow-visible bg-(--ui-chat-surface-background) px-4 pb-(--conversation-turn-gap) pt-1"
        data-message-id={messageId}
        data-role="user"
        data-slot="aui_user-message-root"
      >
        {children}
      </div>
      {attachments}
    </>
  )
}

// Shared "user bubble" base. Both the read-only message and the inline
// edit composer render the same bubble surface (rounded glass card);
// they only differ in border weight, cursor, and padding-right (the
// read-only view reserves room for the restore icon).
//
// no-drag: sticky bubbles park at --sticky-human-top (~4px), sliding under the
// titlebar's [-webkit-app-region:drag] strips (app-shell.tsx). Electron resolves
// drag regions at the compositor level — z-index and pointer-events don't help —
// so without the carve-out, clicking a stuck bubble drags the window instead of
// opening the edit composer.
const USER_BUBBLE_BASE_CLASS =
  'composer-human-message standalone-glass relative flex w-full min-w-0 max-w-full flex-col gap-1.5 overflow-y-auto rounded-xl border bg-(--dt-user-bubble) px-3 py-2 text-left [-webkit-app-region:no-drag]'

const USER_ACTION_ICON_BUTTON_CLASS =
  'grid place-items-center rounded-md bg-transparent text-(--ui-text-secondary) transition-colors hover:bg-(--ui-control-active-background) hover:text-foreground disabled:cursor-default disabled:text-(--ui-text-quaternary) disabled:opacity-70'

const USER_ACTION_ICON_SIZE = '0.6875rem'
const StopGlyph = <StopFilled aria-hidden className="size-3.5 -translate-y-px" />

// Background-process notifications are injected into the conversation as user
// messages (the agent must react to them, and message-role alternation forbids
// a synthetic system row mid-loop). They are NOT something the human typed, so
// render them as a compact system-style notice instead of a user bubble.
// Shape: see tools/process_registry.py format_process_notification().
const PROCESS_NOTIFICATION_RE = /^\[IMPORTANT: Background process [\s\S]*\]$/

const ProcessNotificationNote: FC<{ text: string }> = ({ text }) => {
  const body = text.replace(/^\[IMPORTANT:\s*/, '').replace(/\]$/, '')
  const newline = body.indexOf('\n')
  const headline = (newline === -1 ? body : body.slice(0, newline)).trim()
  const detail = newline === -1 ? '' : body.slice(newline + 1).trim()

  return (
    <div className="flex max-w-[min(86%,44rem)] flex-col gap-0.5 self-center px-2 py-0.5 text-[0.6875rem] leading-5 text-muted-foreground/60">
      <span className="flex items-center gap-1.5">
        <Codicon className="shrink-0 text-muted-foreground/55" name="terminal" size="0.75rem" />
        <span className="wrap-anywhere">{headline}</span>
      </span>
      {detail && (
        <details className="pl-[1.3125rem]">
          <summary className="cursor-pointer select-none text-muted-foreground/45 hover:text-muted-foreground/70">
            output
          </summary>
          <pre
            className="mt-0.5 max-h-48 overflow-auto whitespace-pre-wrap font-mono text-[0.625rem] leading-4 text-muted-foreground/55"
            data-selectable-text="true"
          >
            {detail}
          </pre>
        </details>
      )}
    </div>
  )
}

const UserMessage: FC<{
  onCancel?: () => Promise<void> | void
  onRequestRestoreConfirm?: (messageId: string, target: RestoreMessageTarget) => void
}> = ({ onCancel, onRequestRestoreConfirm }) => {
  const { t } = useI18n()
  const copy = t.assistant.thread
  const messageId = useAuiState(s => s.message.id)
  const content = useAuiState(s => s.message.content)
  const messageText = messageContentText(content)
  const threadRunning = useAuiState(s => s.thread.isRunning)
  const messageCreatedAt = useAuiState(s => s.message.createdAt)

  const latestUserId = useAuiState(s => {
    for (let i = s.thread.messages.length - 1; i >= 0; i--) {
      const message = s.thread.messages[i] as { id?: string; role?: string }

      if (message.role === 'user') {
        return message.id ?? null
      }
    }

    return null
  })

  const runtimeUserOrdinal = useAuiState(s => {
    let ordinal = 0

    for (const message of s.thread.messages) {
      if (message.role !== 'user') {
        continue
      }

      if (message.id === s.message.id) {
        return ordinal
      }

      ordinal += 1
    }

    return null
  })

  const attachmentRefs = useAuiState(s => {
    const custom = (s.message.metadata?.custom ?? {}) as { attachmentRefs?: unknown }

    return messageAttachmentRefs(custom.attachmentRefs)
  })

  // Sticky human bubbles clamp to ~2 lines with a soft fade so a long prompt
  // doesn't dominate the viewport while the response streams underneath; the
  // clamp lifts on hover / focus (see styles.css). We measure the *unclamped*
  // inner wrapper so the ResizeObserver only fires on real content / width
  // changes, not on every frame while the outer max-height animates open.
  const clampInnerRef = useRef<HTMLDivElement | null>(null)
  const [bodyClamped, setBodyClamped] = useState(false)
  const lastClampHeightRef = useRef(-1)
  const lineHeightRef = useRef(0)
  // ★ 拖窗口时 resize 门控: 100+ user bubble 各自的 measureClamp 在拖动每 tick 都
  //   fire → 100+ setState 涌上 = 内容跟不上"缓好久"。拖动期只缓存新 height 不
  //   setState; 停手 (windowResize settle) 后一次性 flush 最后一次的值。见
  //   [[../store/window-resize.ts]] 的 SETTLE_MS。
  const pendingClampHeightRef = useRef<number | null>(null)

  const measureClamp = useCallback((entries: readonly ResizeObserverEntry[]) => {
    const inner = clampInnerRef.current
    const outer = inner?.parentElement

    if (!inner || !outer) {
      return
    }

    // Prefer the size the ResizeObserver already computed — reading
    // `scrollHeight` outside RO timing forces a synchronous layout, and with
    // many user bubbles observed at once those reads interleave with the
    // style write below into a read-write-read reflow cascade.
    const entryHeight = entries.find(entry => entry.target === inner)?.borderBoxSize?.[0]?.blockSize
    const fullHeight = Math.ceil(entryHeight ?? inner.scrollHeight)

    if (fullHeight === lastClampHeightRef.current) {
      return
    }

    // ★ 窗口正在 resize: 只缓存, 不 setState/写 style。settle useEffect (见下) 会 flush。
    if ($isWindowResizing.get()) {
      pendingClampHeightRef.current = fullHeight
      return
    }

    lastClampHeightRef.current = fullHeight

    // Line-height is stable for the life of the bubble (font settings don't
    // change under it) — resolve the computed style once.
    if (!lineHeightRef.current) {
      const styles = getComputedStyle(inner)
      lineHeightRef.current = parseFloat(styles.lineHeight) || 1.5 * parseFloat(styles.fontSize) || 20
    }

    outer.style.setProperty('--human-msg-full', `${fullHeight}px`)
    setBodyClamped(fullHeight > lineHeightRef.current * 2 + 1)
  }, [])

  // ★ 窗口 settle 时 flush 最新 pending 尺寸 (每 user bubble 独立跑, 但 settle 只发生
  //   一次 → 一批 flush 集中发生, React 会 batch 掉多个 setState 到同一次渲染)。
  useEffect(() => {
    const unsub = $isWindowResizing.subscribe(resizing => {
      if (resizing) return
      const pending = pendingClampHeightRef.current
      if (pending == null) return
      pendingClampHeightRef.current = null
      const inner = clampInnerRef.current
      const outer = inner?.parentElement
      if (!inner || !outer) return
      lastClampHeightRef.current = pending
      if (!lineHeightRef.current) {
        const styles = getComputedStyle(inner)
        lineHeightRef.current = parseFloat(styles.lineHeight) || 1.5 * parseFloat(styles.fontSize) || 20
      }
      outer.style.setProperty('--human-msg-full', `${pending}px`)
      setBodyClamped(pending > lineHeightRef.current * 2 + 1)
    })
    return unsub
  }, [])

  useResizeObserver(measureClamp, clampInnerRef)

  // Injected background-process notification, not a human prompt — render the
  // compact system-style notice (after all hooks above have run).
  if (PROCESS_NOTIFICATION_RE.test(messageText.trim())) {
    return (
      <MessagePrimitive.Root
        className="flex w-full min-w-0 flex-col items-stretch"
        data-role="user"
        data-slot="aui_user-message-root"
      >
        <ProcessNotificationNote text={messageText.trim()} />
      </MessagePrimitive.Root>
    )
  }

  const hasBody = messageText.trim().length > 0
  const isLatestUser = messageId === latestUserId
  const showStop = isLatestUser && threadRunning && Boolean(onCancel)
  // Restore (re-run this exact prompt) is available everywhere the Stop button
  // isn't — including mid-stream on older prompts, since the action interrupts
  // the live turn before rewinding.
  const showRestore = !showStop && Boolean(onRequestRestoreConfirm) && hasBody

  // user 气泡与 Assistant 正文卡片完全一致: 圆角 + 浅中性底色 (--ui-bg-elevated),
  // 只读, 不可编辑 (无 hover 变色 / 无编辑光标 / 无内联编辑器)。
  const bubbleClassName = cn(
    'composer-human-message relative flex w-full min-w-0 max-w-full flex-col gap-1.5 rounded-lg bg-(--ui-bg-elevated) px-3 py-2 text-left [-webkit-app-region:no-drag]',
    'text-[length:var(--conversation-text-font-size)] leading-(--dt-line-height) text-foreground'
  )

  const bubbleContent = hasBody && (
    // Render the user's text through a minimal markdown pipeline:
    // backtick `code` and ``` fenced ``` blocks, with directive chips
    // (`@file:` etc.) still resolved inside the plain-text spans.
    <div className="sticky-human-clamp" data-clamped={bodyClamped ? 'true' : undefined}>
      {/* Match the edit composer's collapsed line box (min-h-[1.25rem]) so
          clicking to edit can't grow the bubble by a sub-pixel and reflow the
          turn 1px. */}
      <div className="min-h-[1.25rem]" ref={clampInnerRef}>
        <UserMessageText className="wrap-anywhere" text={messageText} />
      </div>
    </div>
  )

  return (
    <MessagePrimitive.Root asChild>
      <StickyHumanMessageContainer
        attachments={
          // Attachments live BELOW the sticky bubble in normal flow, so they
          // scroll away behind the pinned bubble instead of riding along with
          // it. Image refs render as thumbnails, file refs as chips; no border.
          attachmentRefs.length > 0 ? (
            <div className="flex flex-wrap gap-1 -mt-3 mb-2">
              <DirectiveContent text={attachmentRefs.join(' ')} />
            </div>
          ) : null
        }
        messageId={messageId}
      >
        <ActionBarPrimitive.Root className="relative w-full max-w-full" data-slot="aui_user-bubble-actions">
          {/* 与 Assistant 一致的布局: 左侧头像列 + body 列, 使 You 与 Assistant 的
             发言块左缘对齐。 */}
          <div className="human-message-with-todos-wrapper flex w-full flex-row gap-2">
            <MessageAvatar role="user" />
            <div className="flex min-w-0 flex-1 flex-col gap-0">
            {/* Web 风格头部行 (左对齐, 与 Assistant 一致): "You" + 时间。
               Stop / Restore 挪到本行右侧 (ml-auto), 不再叠在气泡右下角。
               hover 显隐仍靠 MessagePrimitive.Root 的 group/user-message。 */}
            <div className="mb-1 flex items-center gap-1.5 text-[0.65rem] text-(--ui-text-tertiary)">
              <span className="font-medium text-(--ui-text-secondary)">You</span>
              {fmtClock(messageCreatedAt ? messageCreatedAt.getTime() : undefined) && (
                <span className="tabular-nums text-(--ui-text-quaternary)">
                  {fmtClock(messageCreatedAt ? messageCreatedAt.getTime() : undefined)}
                </span>
              )}
              {/* ★ Stop 按钮 (showStop) 已从此处移除 —— 流式期间要取消, 用 composer
                  提交按钮 (会翻成 Stop, thread.tsx 底部 submitting 分支)。这里只保留
                  restore checkpoint 按钮 (只在 !showStop 且有正文时出现)。 */}
              {showRestore && (
                <div className="pointer-events-none ml-auto flex items-center justify-center opacity-0 transition-opacity group-hover/user-message:opacity-100 group-focus-within/user-message:opacity-100">
                  <button
                    aria-label={copy.restoreCheckpoint}
                    className={cn('pointer-events-auto size-6', USER_ACTION_ICON_BUTTON_CLASS)}
                    onClick={event => {
                      event.preventDefault()
                      event.stopPropagation()
                      triggerHaptic('selection')
                      onRequestRestoreConfirm?.(messageId, {
                        text: messageText,
                        userOrdinal: runtimeUserOrdinal
                      })
                    }}
                    onPointerDown={event => {
                      event.preventDefault()
                      event.stopPropagation()
                    }}
                    title={copy.restoreFromHere}
                    type="button"
                  >
                    <Codicon name="discard" size="0.875rem" />
                  </button>
                </div>
              )}
            </div>
            <div className="relative w-full">
              {/* Read-only user bubble — styled identically to the Assistant
                  content card. No inline editing, no overlaid controls. */}
              <div className={bubbleClassName} data-slot="aui_user-bubble">
                {bubbleContent}
              </div>
            </div>
            <BranchPickerPrimitive.Root
              className="checkpoint-container flex items-center gap-1 pb-0 pt-1 pl-1.5 text-[0.75rem] leading-none text-(--ui-text-tertiary)"
              hideWhenSingleBranch
            >
              <span aria-hidden className="checkpoint-icon size-1.5 rounded-full border border-current" />
              <BranchPickerPrimitive.Previous
                className="checkpoint-restore-text rounded-sm bg-transparent px-1 opacity-65 hover:opacity-100 disabled:hidden disabled:cursor-default"
                title={copy.restorePrevious}
              >
                {copy.restoreCheckpoint}
              </BranchPickerPrimitive.Previous>
              <span className="checkpoint-divider opacity-55">
                <BranchPickerPrimitive.Number />/<BranchPickerPrimitive.Count />
              </span>
              <BranchPickerPrimitive.Next
                className="checkpoint-restore-text rounded-sm bg-transparent px-1 opacity-65 hover:opacity-100 disabled:hidden disabled:cursor-default"
                title={copy.restoreNext}
              >
                {copy.goForward}
              </BranchPickerPrimitive.Next>
            </BranchPickerPrimitive.Root>
            </div>
          </div>
        </ActionBarPrimitive.Root>
      </StickyHumanMessageContainer>
    </MessagePrimitive.Root>
  )
}

const SLASH_STATUS_RE = /^slash:(?<command>\/[^\n]+)\n(?<output>[\s\S]*)$/
const STEER_NOTE_RE = /^steer:(?<text>[\s\S]+)$/

const SystemMessage: FC = () => {
  const text = useAuiState(s => messageContentText(s.message.content))

  if (!text) {
    return null
  }

  const steerNote = text.match(STEER_NOTE_RE)

  if (steerNote?.groups) {
    return (
      <MessagePrimitive.Root
        className="flex max-w-[min(86%,44rem)] items-center gap-1.5 self-center px-2 py-0.5 text-[0.6875rem] leading-5 text-muted-foreground/60"
        data-role="system"
        data-slot="aui_system-message-root"
      >
        <Codicon className="text-muted-foreground/55" name="compass" size="0.75rem" />
        <span className="text-muted-foreground/55">steered</span>
        <span className="text-muted-foreground/35">·</span>
        <span className="whitespace-pre-wrap">{steerNote.groups.text.trim()}</span>
      </MessagePrimitive.Root>
    )
  }

  const slashStatus = text.match(SLASH_STATUS_RE)

  if (slashStatus?.groups) {
    const output = slashStatus.groups.output.trim()
    // Single-line status (e.g. "model → x") reads best centered inline; padded
    // multiline output (catalogs, usage tables) needs left-aligned, wider room
    // or the column alignment breaks.
    const multiline = output.includes('\n')

    return (
      <MessagePrimitive.Root
        className={cn(
          'w-[60%] max-w-[44rem] self-center px-2 py-0.5 text-[0.6875rem] leading-5 text-muted-foreground/60',
          multiline ? 'text-left' : 'text-center'
        )}
        data-role="system"
        data-slot="aui_system-message-root"
      >
        <span className="font-mono text-muted-foreground/55">{slashStatus.groups.command}</span>
        {multiline ? (
          <LinkifiedText className="mt-0.5 block whitespace-pre-wrap" explicitOnly pretty={false} text={output} />
        ) : (
          <>
            <span className="mx-1.5 text-muted-foreground/35">·</span>
            <LinkifiedText className="whitespace-pre-wrap" explicitOnly pretty={false} text={output} />
          </>
        )}
      </MessagePrimitive.Root>
    )
  }

  const multiline = text.includes('\n')

  return (
    <MessagePrimitive.Root
      className={cn(
        'w-[60%] max-w-[44rem] self-center px-2 py-0.5 text-[0.6875rem] leading-5 text-muted-foreground/55',
        multiline ? 'text-left' : 'text-center'
      )}
      data-role="system"
      data-slot="aui_system-message-root"
    >
      <LinkifiedText className="whitespace-pre-wrap" explicitOnly pretty={false} text={text} />
    </MessagePrimitive.Root>
  )
}

interface UserEditComposerProps {
  cwd: string | null
  gateway: HermesGateway | null
  sessionId: string | null
}

const UserEditComposer: FC<UserEditComposerProps> = ({ cwd, gateway, sessionId }) => {
  const { t } = useI18n()
  const copy = t.assistant.thread
  const aui = useAui()
  const draft = useAuiState(s => s.composer.text)
  const rootRef = useRef<HTMLDivElement | null>(null)
  const editorRef = useRef<HTMLDivElement | null>(null)
  const draftRef = useRef(draft)
  const dragDepthRef = useRef(0)
  const [dragActive, setDragActive] = useState(false)
  const [trigger, setTrigger] = useState<TriggerState | null>(null)
  const [triggerActive, setTriggerActive] = useState(0)
  const [triggerItems, setTriggerItems] = useState<readonly Unstable_TriggerItem[]>([])
  // See index.tsx: set in keydown when the open popover consumes a nav/control
  // key so the matching keyup skips refreshTrigger (timing-immune vs reading
  // `trigger`, which keyup sees as already-null after Escape).
  const triggerKeyConsumedRef = useRef(false)
  const [triggerPlacement, setTriggerPlacement] = useState<'bottom' | 'top'>('top')
  const [focusRequestId, setFocusRequestId] = useState(0)
  const [submitting, setSubmitting] = useState(false)
  // True while OS-drop files are being staged/uploaded into the session. Blocks
  // submit and shows a spinner so confirming the edit can't race the async
  // upload and drop the gateway-side ref before it lands in the draft.
  const [staging, setStaging] = useState(false)
  const expanded = draft.includes('\n')
  const canSubmit = draft.trim().length > 0
  const at = useAtCompletions({ cwd, gateway, sessionId })
  const slash = useSlashCompletions({ gateway })

  useEffect(() => () => notifyThreadEditClose(), [])

  const focusEditor = useCallback(() => {
    const editor = editorRef.current

    focusComposerInput(editor)

    if (editor) {
      placeCaretEnd(editor)
    }

    markActiveComposer('edit')
  }, [])

  const requestEditFocus = useCallback(() => {
    setFocusRequestId(id => id + 1)
  }, [])

  const appendExternalText = useCallback(
    (text: string, mode: ComposerInsertMode) => {
      const value = text.trim()

      if (!value) {
        return
      }

      const base = mode === 'inline' ? draftRef.current.trimEnd() : draftRef.current
      const sep = mode === 'inline' ? (base ? ' ' : '') : base && !base.endsWith('\n') ? '\n\n' : ''
      const next = `${base}${sep}${value}`

      draftRef.current = next
      aui.composer().setText(next)

      const editor = editorRef.current

      if (editor) {
        renderComposerContents(editor, next)
        placeCaretEnd(editor)
      }

      setFocusRequestId(id => id + 1)
    },
    [aui]
  )

  useEffect(() => {
    draftRef.current = draft

    const editor = editorRef.current

    if (
      editor &&
      (editor.childNodes.length === 0 || (document.activeElement !== editor && composerPlainText(editor) !== draft))
    ) {
      renderComposerContents(editor, draft)

      if (document.activeElement === editor) {
        placeCaretEnd(editor)
      }
    }
  }, [draft])

  useEffect(() => {
    focusEditor()
  }, [focusEditor, focusRequestId])

  useEffect(() => {
    const offFocus = onComposerFocusRequest(target => {
      if (target === 'edit') {
        setFocusRequestId(id => id + 1)
      }
    })

    const offInsert = onComposerInsertRequest(({ mode, target, text }) => {
      if (target === 'edit') {
        appendExternalText(text, mode)
      }
    })

    return () => {
      offFocus()
      offInsert()
    }
  }, [appendExternalText])

  const syncDraftFromEditor = useCallback(
    (editor: HTMLDivElement) => {
      const nextDraft = composerPlainText(editor)

      if (nextDraft !== draftRef.current) {
        draftRef.current = nextDraft
        aui.composer().setText(nextDraft)
      }

      return nextDraft
    },
    [aui]
  )

  const refreshTrigger = useCallback(() => {
    const editor = editorRef.current

    if (!editor) {
      return
    }

    const before = textBeforeCaret(editor)
    const detected = detectTrigger(before ?? composerPlainText(editor))

    if (detected) {
      const rect = editor.getBoundingClientRect()
      const spaceAbove = rect.top
      const spaceBelow = window.innerHeight - rect.bottom

      setTriggerPlacement(spaceAbove < 220 && spaceBelow > spaceAbove ? 'bottom' : 'top')
    }

    setTrigger(detected)

    // Only reset the highlight when the trigger actually changed (opened, or
    // the query/kind differs). Re-detecting the *same* trigger — e.g. on a
    // caret move (mouseup) or a stray refresh — must preserve the user's
    // current selection instead of snapping back to the first item.
    if (detected?.kind !== trigger?.kind || detected?.query !== trigger?.query) {
      setTriggerActive(0)
    }
  }, [trigger])

  const closeTrigger = useCallback(() => {
    setTrigger(null)
    setTriggerItems([])
    setTriggerActive(0)
  }, [])

  const triggerAdapter: Unstable_TriggerAdapter | null =
    trigger?.kind === '@' ? at.adapter : trigger?.kind === '/' ? slash.adapter : null

  useEffect(() => {
    if (!trigger || !triggerAdapter?.search) {
      setTriggerItems([])

      return
    }

    setTriggerItems(triggerAdapter.search(trigger.query))
  }, [trigger, triggerAdapter])

  useEffect(() => {
    setTriggerActive(idx => Math.min(idx, Math.max(0, triggerItems.length - 1)))
  }, [triggerItems.length])

  const triggerLoading = trigger?.kind === '@' ? at.loading : trigger?.kind === '/' ? slash.loading : false

  const replaceTriggerWithChip = useCallback(
    (item: Unstable_TriggerItem) => {
      const editor = editorRef.current

      if (!editor || !trigger) {
        return
      }

      const serialized = hermesDirectiveFormatter.serialize(item)
      const starter = serialized.endsWith(':')
      const text = starter || serialized.endsWith(' ') ? serialized : `${serialized} `
      const directive = !starter && serialized.match(/^@([^:]+):(.+)$/)

      const finish = () => {
        draftRef.current = composerPlainText(editor)
        aui.composer().setText(draftRef.current)
        requestEditFocus()
        starter ? window.setTimeout(refreshTrigger, 0) : closeTrigger()
      }

      const sel = window.getSelection()
      const range = sel?.rangeCount ? sel.getRangeAt(0) : null
      const node = range?.startContainer
      const offset = range?.startOffset ?? 0

      if (!sel || !range || node?.nodeType !== Node.TEXT_NODE || offset < trigger.tokenLength) {
        const current = composerPlainText(editor)
        renderComposerContents(editor, `${current.slice(0, Math.max(0, current.length - trigger.tokenLength))}${text}`)
        placeCaretEnd(editor)

        return finish()
      }

      const replaceRange = document.createRange()
      replaceRange.setStart(node, offset - trigger.tokenLength)
      replaceRange.setEnd(node, offset)
      replaceRange.deleteContents()

      if (directive) {
        const chip = refChipElement(directive[1], directive[2])
        const space = document.createTextNode(' ')
        const fragment = document.createDocumentFragment()
        fragment.append(chip, space)
        replaceRange.insertNode(fragment)

        const caret = document.createRange()
        caret.setStart(space, 1)
        caret.collapse(true)
        sel.removeAllRanges()
        sel.addRange(caret)

        return finish()
      }

      document.execCommand('insertText', false, text)
      finish()
    },
    [aui, closeTrigger, refreshTrigger, requestEditFocus, trigger]
  )

  const insertRefStrings = useCallback(
    (refs: InlineRefInput[]) => {
      const editor = editorRef.current

      if (!editor || refs.length === 0) {
        return false
      }

      const nextDraft = insertInlineRefsIntoEditor(editor, refs)

      if (nextDraft === null) {
        return false
      }

      draftRef.current = nextDraft
      aui.composer().setText(nextDraft)
      requestEditFocus()

      return true
    },
    [aui, requestEditFocus]
  )

  const insertDroppedRefs = useCallback(
    (candidates: ReturnType<typeof extractDroppedFiles>) => insertRefStrings(droppedFileInlineRefs(candidates, cwd)),
    [cwd, insertRefStrings]
  )

  // OS/Finder drops carry an absolute path on THIS machine — the gateway can't
  // read it in remote mode, and an image needs its bytes uploaded for vision.
  // Stage each through the same file.attach/image.attach_bytes pipeline the main
  // composer uses, then insert the *gateway-side* ref the agent can resolve —
  // never the raw local path (the MahmoudR remote-attach bug, which the main
  // composer fixes but this edit composer used to reproduce).
  const uploadOsDropRefs = useCallback(
    async (osDrops: ReturnType<typeof extractDroppedFiles>): Promise<InlineRefInput[]> => {
      if (!gateway || !sessionId) {
        // No session to stage into — best-effort inline refs (matches old path).
        return droppedFileInlineRefs(osDrops, cwd)
      }

      const remote = $connection.get()?.mode === 'remote'

      const requestGateway = <T,>(method: string, params?: Record<string, unknown>) =>
        gateway.request<T>(method, params)

      const refs: InlineRefInput[] = []

      for (const candidate of osDrops) {
        const path = candidate.path || ''

        if (!path) {
          continue
        }

        const kind: ComposerAttachment['kind'] =
          candidate.file?.type.startsWith('image/') || isImagePath(candidate.file?.name || path) ? 'image' : 'file'

        try {
          const uploaded = await uploadComposerAttachment(
            { detail: path, id: attachmentId(kind, path), kind, label: pathLabel(path), path },
            { remote, requestGateway, sessionId }
          )

          const ref = attachmentDisplayText(uploaded)

          if (ref) {
            refs.push(ref)
          }
        } catch (err) {
          notifyError(err, t.desktop.dropFiles)
        }
      }

      return refs
    },
    [cwd, gateway, sessionId, t.desktop.dropFiles]
  )

  const resetDragState = useCallback(() => {
    dragDepthRef.current = 0
    setDragActive(false)
  }, [])

  const handleDragEnter = (event: ReactDragEvent<HTMLElement>) => {
    if (!dragHasAttachments(event.dataTransfer, HERMES_PATHS_MIME)) {
      return
    }

    event.preventDefault()
    dragDepthRef.current += 1

    if (!dragActive) {
      setDragActive(true)
    }
  }

  const handleDragOver = (event: ReactDragEvent<HTMLElement>) => {
    if (!dragHasAttachments(event.dataTransfer, HERMES_PATHS_MIME)) {
      return
    }

    event.preventDefault()
    event.dataTransfer.dropEffect = 'copy'
  }

  const handleDragLeave = (event: ReactDragEvent<HTMLElement>) => {
    event.preventDefault()
    dragDepthRef.current = Math.max(0, dragDepthRef.current - 1)

    if (dragDepthRef.current === 0) {
      setDragActive(false)
    }
  }

  const handleDrop = (event: ReactDragEvent<HTMLElement>) => {
    if (!dragHasAttachments(event.dataTransfer, HERMES_PATHS_MIME)) {
      return
    }

    const candidates = extractDroppedFiles(event.dataTransfer)

    if (!candidates.length) {
      return
    }

    event.preventDefault()
    event.stopPropagation()
    resetDragState()

    // In-app drags (project tree / gutter) are workspace-relative paths that
    // resolve on the gateway as-is, so they stay inline refs. OS drops need to
    // be staged + uploaded first, then their gateway-side ref is inserted.
    const { inAppRefs, osDrops } = partitionDroppedFiles(candidates)

    if (insertDroppedRefs(inAppRefs)) {
      triggerHaptic('selection')
    }

    if (osDrops.length) {
      setStaging(true)
      void uploadOsDropRefs(osDrops)
        .then(refs => {
          if (insertRefStrings(refs)) {
            triggerHaptic('selection')
          }
        })
        .finally(() => setStaging(false))
    }
  }

  const handleInput = (event: FormEvent<HTMLDivElement>) => {
    const editor = event.currentTarget

    if (editor.childNodes.length === 1 && editor.firstChild?.nodeName === 'BR') {
      editor.replaceChildren()
    }

    syncDraftFromEditor(editor)
    window.setTimeout(refreshTrigger, 0)
  }

  const handlePaste = (event: ClipboardEvent<HTMLDivElement>) => {
    const pastedText = event.clipboardData.getData('text')

    if (!pastedText || DATA_IMAGE_URL_RE.test(pastedText.trim())) {
      event.preventDefault()

      return
    }

    event.preventDefault()
    document.execCommand('insertText', false, pastedText)
    syncDraftFromEditor(event.currentTarget)
  }

  const submitEdit = (editor: HTMLDivElement) => {
    const nextDraft = syncDraftFromEditor(editor)

    if (submitting || staging || !nextDraft.trim()) {
      return
    }

    setSubmitting(true)
    aui.composer().send()
  }

  const handleEditBlur = useCallback(
    (event: FocusEvent<HTMLDivElement>) => {
      const nextTarget = event.relatedTarget

      if (nextTarget instanceof Node && event.currentTarget.contains(nextTarget)) {
        return
      }

      window.setTimeout(() => {
        const root = rootRef.current
        const active = document.activeElement

        if (submitting || (root && active && root.contains(active))) {
          return
        }

        closeTrigger()
        aui.composer().cancel()
      }, 80)
    },
    [aui, closeTrigger, submitting]
  )

  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (trigger && triggerItems.length > 0) {
      if (event.key === 'ArrowDown') {
        event.preventDefault()
        triggerKeyConsumedRef.current = true
        setTriggerActive(idx => (idx + 1) % triggerItems.length)

        return
      }

      if (event.key === 'ArrowUp') {
        event.preventDefault()
        triggerKeyConsumedRef.current = true
        setTriggerActive(idx => (idx - 1 + triggerItems.length) % triggerItems.length)

        return
      }

      if (event.key === 'Enter' || event.key === 'Tab') {
        event.preventDefault()
        triggerKeyConsumedRef.current = true
        const item = triggerItems[triggerActive]

        if (item) {
          replaceTriggerWithChip(item)
        }

        return
      }

      if (event.key === 'Escape') {
        event.preventDefault()
        triggerKeyConsumedRef.current = true
        closeTrigger()

        return
      }
    }

    if (event.key === 'Escape') {
      event.preventDefault()
      aui.composer().cancel()

      return
    }

    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      submitEdit(event.currentTarget)
    }
  }

  const handleKeyUp = () => {
    // If this keyup belongs to a key the open trigger popover already consumed
    // in keydown (Arrow/Enter/Tab/Escape), skip the refresh. Those keys never
    // edit text, and for Escape the keydown already closed the menu — a refresh
    // here would re-detect the still-present `/` and instantly reopen it. We
    // read a ref set during keydown rather than `trigger`, because by keyup
    // time React has re-rendered and `trigger` may already be null.
    if (triggerKeyConsumedRef.current) {
      triggerKeyConsumedRef.current = false

      return
    }

    window.setTimeout(refreshTrigger, 0)
  }

  return (
    <ComposerPrimitive.Root className="contents" data-slot="aui_edit-composer-root">
      <StickyHumanMessageContainer>
        <div
          className="composer-human-message-container human-execution-message-top relative flex w-full items-start rounded-md bg-(--ui-chat-surface-background)"
          onBlur={handleEditBlur}
          onDragEnter={handleDragEnter}
          onDragLeave={handleDragLeave}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          ref={rootRef}
        >
          {trigger && (
            <ComposerTriggerPopover
              activeIndex={triggerActive}
              items={triggerItems}
              kind={trigger.kind}
              loading={triggerLoading}
              onHover={setTriggerActive}
              onPick={replaceTriggerWithChip}
              placement={triggerPlacement}
            />
          )}
          <div
            className={cn(
              USER_BUBBLE_BASE_CLASS,
              'ui-prompt-input__container relative border-(--ui-stroke-secondary) data-[expanded=true]:min-h-20',
              COMPOSER_DROP_FADE_CLASS,
              dragActive && COMPOSER_DROP_ACTIVE_CLASS
            )}
            data-expanded={expanded ? 'true' : undefined}
          >
            <div
              aria-label={copy.editMessage}
              autoCapitalize="off"
              autoCorrect="off"
              className={cn(
                'ui-prompt-input-editor__input max-h-48 w-full resize-none bg-transparent p-0 pr-7 text-[length:var(--conversation-text-font-size)] text-foreground/95 outline-none',
                'empty:before:content-[attr(data-placeholder)] empty:before:text-muted-foreground/60',
                '**:data-ref-text:cursor-default',
                expanded ? 'min-h-16' : 'min-h-[1.25rem]'
              )}
              contentEditable
              data-placeholder={copy.editMessage}
              data-slot={RICH_INPUT_SLOT}
              onBlur={() => window.setTimeout(closeTrigger, 80)}
              onDragOver={handleDragOver}
              onDrop={handleDrop}
              onFocus={() => markActiveComposer('edit')}
              onInput={handleInput}
              onKeyDown={handleKeyDown}
              onKeyUp={handleKeyUp}
              onMouseUp={refreshTrigger}
              onPaste={handlePaste}
              ref={editorRef}
              role="textbox"
              spellCheck={false}
              suppressContentEditableWarning
            />
            <ComposerPrimitive.Input
              asChild
              className="sr-only"
              submitMode="ctrlEnter"
              tabIndex={-1}
              unstable_focusOnScrollToBottom={false}
            >
              <textarea
                aria-hidden
                autoCapitalize="off"
                autoComplete="off"
                autoCorrect="off"
                className="sr-only"
                spellCheck={false}
                tabIndex={-1}
              />
            </ComposerPrimitive.Input>
            {staging && (
              <span
                className="pointer-events-none absolute bottom-2 left-2 inline-flex items-center gap-1 rounded-full bg-background/80 px-1.5 py-0.5 text-[0.62rem] text-muted-foreground backdrop-blur-[1px]"
                data-slot="aui_edit-staging"
              >
                <Loader2Icon className="size-3 animate-spin" />
                {copy.attachingFile}
              </span>
            )}
            <button
              aria-label={copy.sendEdited}
              className={cn('absolute right-2 bottom-2 size-5', USER_ACTION_ICON_BUTTON_CLASS)}
              disabled={!canSubmit || submitting || staging}
              onClick={() => {
                const editor = editorRef.current

                if (editor) {
                  submitEdit(editor)
                }
              }}
              title={copy.sendEdited}
              type="button"
            >
              {submitting ? StopGlyph : <Codicon name="arrow-up" size={USER_ACTION_ICON_SIZE} />}
            </button>
          </div>
        </div>
      </StickyHumanMessageContainer>
    </ComposerPrimitive.Root>
  )
}
