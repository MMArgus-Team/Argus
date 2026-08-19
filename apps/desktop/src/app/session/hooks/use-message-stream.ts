import type { QueryClient } from '@tanstack/react-query'
import { type MutableRefObject, useCallback, useEffect, useRef } from 'react'

import { readActiveTerminal } from '@/app/right-sidebar/terminal/buffer'
import { translateNow } from '@/i18n'
import {
  appendAssistantTextPart,
  appendReasoningPart,
  assistantTextPart,
  type ChatMessage,
  type ChatMessagePart,
  chatMessageText,
  type ChatSubRole,
  type GatewayEventPayload,
  reasoningPart,
  renderMediaTags,
  textPart,
  upsertToolPart
} from '@/lib/chat-messages'
import { coerceGatewayText, coerceThinkingText, normalizePersonalityValue } from '@/lib/chat-runtime'
import { playCompletionSound } from '@/lib/completion-sound'
import { gatewayEventRequiresSessionId } from '@/lib/gateway-events'
import {
  dedupeGeneratedImageEchoesInParts,
  generatedImageEchoSources,
  stripGeneratedImageEchoes
} from '@/lib/generated-images'
import { triggerHaptic } from '@/lib/haptics'
import { isProviderSetupErrorMessage } from '@/lib/provider-setup-errors'
import { parseTodos } from '@/lib/todos'
import { clearClarifyRequest, setClarifyRequest } from '@/store/clarify'
import { setSessionCompacting } from '@/store/compaction'
import { refreshBackgroundProcesses } from '@/store/composer-status'
import { $gateway } from '@/store/gateway'
import { dispatchNativeNotification } from '@/store/native-notifications'
import { notify } from '@/store/notifications'
import { requestDesktopOnboarding } from '@/store/onboarding'
import { flashPetActivity, markPetUnread, setPetActivity } from '@/store/pet'
import { followActiveSessionCwd } from '@/store/projects'
import { clearAllPrompts, setApprovalRequest, setSecretRequest, setSudoRequest } from '@/store/prompts'
import {
  $currentCwd,
  $currentModel,
  setCurrentBranch,
  setCurrentCwd,
  setCurrentFastMode,
  setCurrentModel,
  setCurrentPersonality,
  setCurrentProvider,
  setCurrentReasoningEffort,
  setCurrentServiceTier,
  setCurrentUsage,
  setSessions,
  setTurnStartedAt,
  setYoloActive
} from '@/store/session'
import { broadcastSessionsChanged } from '@/store/session-sync'
import { clearSessionSubagents, pruneDelegateFallbackSubagents, upsertSubagent } from '@/store/subagents'
import { setSessionTodos } from '@/store/todos'
import { recordToolDiff } from '@/store/tool-diffs'
import { clearSessionGeneratingTool, setSessionGeneratingTool } from '@/store/tool-generating'
import { notifyWorkspaceChanged, toolMayMutateFiles } from '@/store/workspace-events'
import type { RpcEvent } from '@/types/hermes'

import type { ClientSessionState } from '../../types'

interface MessageStreamOptions {
  activeSessionIdRef: MutableRefObject<string | null>
  hydrateFromStoredSession: (
    attempts?: number,
    storedSessionId?: string | null,
    runtimeSessionId?: string | null
  ) => Promise<void>
  queryClient: QueryClient
  refreshHermesConfig: () => Promise<void>
  refreshSessions: () => Promise<void>
  sessionStateByRuntimeIdRef: MutableRefObject<Map<string, ClientSessionState>>
  updateSessionState: (
    sessionId: string,
    updater: (state: ClientSessionState) => ClientSessionState,
    storedSessionId?: string | null
  ) => ClientSessionState
}

interface QueuedStreamDeltas {
  assistant: string
  reasoning: string
}

interface QueuedRoutedStreamDeltas extends QueuedStreamDeltas {
  sessionId: string
  streamId: string
}

interface StreamMutationOptions {
  pending?: (message: ChatMessage) => boolean
  /** Explicit answer bubble selected by the gateway's per-turn request_id. */
  streamId?: string
  /** A detached worker reply must not take ownership of foreground turn state. */
  preserveForeground?: boolean
  /** Stable gateway turn id attached to the created/updated bubble. */
  requestId?: string
}

interface CompleteAssistantMessageOptions {
  ephemeralControl?: boolean
  preserveForeground?: boolean
  requestId?: string
  streamId?: string
  /** Gateway turn outcome ("error" | "interrupted" | "complete"). Authoritative
   *  signal that this completion is a failure — see completionErrorText. */
  status?: string
}

/** Sub-role tag captured from a turn's message.start, attached to the assistant
 *  message it opens (monitor SPEAK / deep-research threadback labeling). */
interface ChatMessageMeta {
  subRole: ChatSubRole
  monitorLabel?: string
  brief?: string
  /** monitor_id / request_id — 事件 id, 气泡上展示 (#mon_xxx / #req_xxx)。 */
  eventId?: string
  model?: string
  voice?: boolean
}

type SessionRuntimeStatePatch = Partial<
  Pick<
    ClientSessionState,
    'branch' | 'cwd' | 'fast' | 'model' | 'personality' | 'provider' | 'reasoningEffort' | 'serviceTier' | 'yolo'
  >
>

function sessionInfoStatePatch(payload: GatewayEventPayload | undefined): SessionRuntimeStatePatch {
  const patch: SessionRuntimeStatePatch = {}

  if (typeof payload?.model === 'string') {
    patch.model = payload.model || ''
  }

  if (typeof payload?.provider === 'string') {
    patch.provider = payload.provider || ''
  }

  if (typeof payload?.cwd === 'string') {
    patch.cwd = payload.cwd
  }

  if (typeof payload?.branch === 'string') {
    patch.branch = payload.branch
  }

  if (typeof payload?.personality === 'string') {
    patch.personality = normalizePersonalityValue(payload.personality)
  }

  if (typeof payload?.reasoning_effort === 'string') {
    patch.reasoningEffort = payload.reasoning_effort
  }

  if (typeof payload?.service_tier === 'string') {
    patch.serviceTier = payload.service_tier
  }

  if (typeof payload?.fast === 'boolean') {
    patch.fast = payload.fast
  }

  if (typeof payload?.yolo === 'boolean') {
    patch.yolo = payload.yolo
  }

  return patch
}

function hasSessionInfoStatePatch(patch: SessionRuntimeStatePatch): boolean {
  return Object.keys(patch).length > 0
}

// Minimum gap between two assistant-text flushes during a stream. Was 16ms
// (rAF only), which at typical LLM token rates of ~30-80 tok/sec meant every
// token got its own React commit + Streamdown markdown re-parse, scaling
// linearly with the growing last-block length. Bumping to 33ms lets ~2 tokens
// batch into one commit at 60 tok/sec without introducing visible lag on the
// streaming text (still 30 fps of visible text growth). Big perceived
// smoothness win on long messages with big trailing paragraphs; see
// `scripts/profile-typing-lag.md` for the measurement work behind this.
const STREAM_DELTA_FLUSH_MS = 33

// Gateway/provider failures arrive as message.complete text rather than an
// explicit error event. They must be marked as inline assistant errors, or the
// post-turn hydrate fallback erases them: a failed turn is deliberately NOT
// persisted as assistant text (gateway/run.py:10580 — the error is a
// gateway-generated hint, not model output), so re-reading the stored session
// returns a turn with no reply and overwrites the visible message.
//
// The gateway already tells us this on every completion via `status`
// ("error" | "interrupted" | "complete"), so trust that first. The text
// patterns below are only a fallback for older gateways that predate the
// status field — they cannot be relied on alone, because the gateway prepends
// "Error: " when the backend produced no visible text, which defeats every
// ^-anchored pattern here (quota, billing, context-overflow, content-policy).
const COMPLETION_ERROR_PATTERNS = [
  /^API call failed after \d+ retries:/i,
  /^HTTP\s+\d{3}\b/i,
  /^(Provider|Gateway)\s+error:/i
]

function completionErrorText(finalText: string, status?: string): string | null {
  const text = finalText.trim()

  // An errored turn often has no usable text at all (the backend bailed before
  // producing any), so fall back to a generic label rather than dropping the
  // error signal and letting hydrate erase the bubble.
  if (status === 'error') {
    return text || translateNow('errors.turnFailed')
  }

  return text && COMPLETION_ERROR_PATTERNS.some(re => re.test(text)) ? text : null
}

function gatewayRequestId(payload: GatewayEventPayload | undefined): string {
  return typeof payload?.request_id === 'string' ? payload.request_id.trim() : ''
}

function isEphemeralControl(payload: GatewayEventPayload | undefined): boolean {
  return (
    payload?.ephemeral === true ||
    payload?.ephemeral_control === true ||
    payload?.history_policy === 'ephemeral_control'
  )
}

function requestStreamRouteKey(sessionId: string, requestId: string): string {
  return `${sessionId}\u0000${requestId}`
}

const SUBAGENT_EVENT_TYPES = new Set([
  'subagent.spawn_requested',
  'subagent.start',
  'subagent.thinking',
  'subagent.tool',
  'subagent.progress',
  'subagent.complete'
])

// Anonymous progress events that carry todos but no name still belong to the
// todo stream; named todo events are obviously routed there too.
function toTodoPayload(payload: GatewayEventPayload | undefined): GatewayEventPayload | undefined {
  if (!payload) {
    return undefined
  }

  const isTodo = payload.name === 'todo' || (!payload.name && Object.hasOwn(payload, 'todos'))

  return isTodo ? { ...payload, name: 'todo', tool_id: payload.tool_id || 'todo-live' } : undefined
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : {}
}

function parseMaybeRecord(value: unknown): Record<string, unknown> {
  if (typeof value === 'string') {
    try {
      return asRecord(JSON.parse(value))
    } catch {
      return {}
    }
  }

  return asRecord(value)
}

const firstString = (...candidates: unknown[]): string => {
  for (const v of candidates) {
    if (typeof v === 'string' && v) {
      return v
    }
  }

  return ''
}

function delegateTaskPayloads(
  payload: GatewayEventPayload | undefined,
  phase: 'running' | 'complete',
  sourceEventType?: string
): Record<string, unknown>[] {
  if (payload?.name !== 'delegate_task') {
    return []
  }

  const args = parseMaybeRecord(payload.args ?? payload.input)
  const result = parseMaybeRecord(payload.result)
  const rawTasks = Array.isArray(args.tasks) ? args.tasks : []
  const tasks = rawTasks.length ? rawTasks.map(parseMaybeRecord) : [args]
  const status = phase === 'complete' ? (payload.error ? 'failed' : 'completed') : 'running'
  const toolId = payload.tool_id || payload.tool_call_id || payload.id || 'delegate_task'
  const progressText = firstString(payload.preview, payload.message, payload.context)

  const eventType =
    phase === 'complete'
      ? 'subagent.complete'
      : sourceEventType === 'tool.start'
        ? 'subagent.start'
        : 'subagent.progress'

  return tasks.map((task, index) => {
    const goal = firstString(task.goal, args.goal, payload.context) || 'Delegated task'
    const summary = firstString(result.summary, payload.summary, payload.message)

    return {
      depth: 0,
      duration_seconds: payload.duration_s,
      goal,
      status,
      subagent_id: `delegate-tool:${toolId}:${index}`,
      summary: summary || undefined,
      task_count: tasks.length,
      task_index: index,
      text: eventType === 'subagent.progress' ? progressText || goal : undefined,
      tool_name: eventType === 'subagent.start' ? 'delegate_task' : undefined,
      tool_preview: eventType === 'subagent.start' ? progressText : undefined,
      toolsets: Array.isArray(task.toolsets) ? task.toolsets : Array.isArray(args.toolsets) ? args.toolsets : [],
      event_type: eventType,
      output_tail:
        phase === 'complete' && summary
          ? [{ is_error: Boolean(payload.error), preview: summary, tool: 'delegate_task' }]
          : undefined
    }
  })
}

export function useMessageStream({
  activeSessionIdRef,
  hydrateFromStoredSession,
  queryClient,
  refreshHermesConfig,
  refreshSessions,
  sessionStateByRuntimeIdRef,
  updateSessionState
}: MessageStreamOptions) {
  const sessionInterrupted = useCallback(
    (sessionId: string) => sessionStateByRuntimeIdRef.current.get(sessionId)?.interrupted ?? false,
    [sessionStateByRuntimeIdRef]
  )

  // A foreground main-agent turn can hand reply ownership to QueryWorker and
  // become idle before that worker returns. Keep its answer slot independently
  // addressable by the gateway's stable request_id; the single foreground
  // state.streamId is intentionally free to move on to the user's next turn.
  const requestStreamIdsRef = useRef<Map<string, string>>(new Map())
  const streamRequestIdsRef = useRef<Map<string, string>>(new Map())
  const requestStreamSequenceRef = useRef(0)

  const requestStreamId = useCallback((sessionId: string, requestId: string, create: boolean) => {
    if (!requestId) {
      return ''
    }

    const key = requestStreamRouteKey(sessionId, requestId)
    const existing = requestStreamIdsRef.current.get(key)

    if (existing || !create) {
      return existing ?? ''
    }

    requestStreamSequenceRef.current += 1
    const streamId = `assistant-stream-${Date.now()}-${requestStreamSequenceRef.current}`
    requestStreamIdsRef.current.set(key, streamId)
    streamRequestIdsRef.current.set(requestStreamRouteKey(sessionId, streamId), requestId)

    return streamId
  }, [])

  const forgetRequestStream = useCallback((sessionId: string, requestId: string) => {
    if (requestId) {
      const key = requestStreamRouteKey(sessionId, requestId)
      const streamId = requestStreamIdsRef.current.get(key)

      requestStreamIdsRef.current.delete(key)
      if (streamId) {
        streamRequestIdsRef.current.delete(requestStreamRouteKey(sessionId, streamId))
      }
    }
  }, [])

  // Patch the in-flight assistant message (or seed it). Centralises the
  // streamId/groupId bookkeeping every event callback would otherwise repeat.
  const mutateStream = useCallback(
    (
      sessionId: string,
      transform: (parts: ChatMessagePart[], message: ChatMessage) => ChatMessagePart[],
      seed: () => ChatMessagePart[],
      opts: StreamMutationOptions = {}
    ) => {
      const apply = () => {
        updateSessionState(sessionId, state => {
          // After a stop, drop any late deltas / tool events for the
          // cancelled turn so they don't keep growing the (now finalized)
          // assistant bubble or, worse, seed a brand-new bubble that
          // appears to belong to the next user message.
          if (state.interrupted && !opts.preserveForeground) {
            return state
          }

          const streamId = opts.streamId ?? state.streamId ?? `assistant-stream-${Date.now()}`
          const requestId =
            opts.requestId ?? streamRequestIdsRef.current.get(requestStreamRouteKey(sessionId, streamId))
          const groupId = opts.preserveForeground ? undefined : state.pendingBranchGroup ?? undefined
          const prev = state.messages
          let nextMessages: ChatMessage[]

          if (!prev.some(m => m.id === streamId)) {
            // Attach the current turn's sub-role tag (monitor / deep-research)
            // captured on message.start, so this fresh assistant message renders
            // with its web-style label + coloring instead of a plain reply.
            const meta = opts.preserveForeground ? undefined : pendingSubRoleRef.current.get(sessionId)
            nextMessages = [
              ...prev,
              {
                id: streamId,
                requestId,
                role: 'assistant',
                parts: seed(),
                pending: true,
                branchGroupId: groupId,
                ...(meta
                  ? {
                      subRole: meta.subRole,
                      monitorLabel: meta.monitorLabel,
                      brief: meta.brief,
                      deepReportRid: meta.eventId,
                      model: meta.model,
                      voice: meta.voice
                    }
                  : {})
              }
            ]
          } else {
            nextMessages = prev.map(m =>
              m.id === streamId
                ? {
                    ...m,
                    ...(requestId && { requestId }),
                    parts: transform(m.parts, m),
                    pending: opts.pending ? opts.pending(m) : true
                  }
                : m
            )
          }

          return opts.preserveForeground
            ? { ...state, messages: nextMessages }
            : {
                ...state,
                messages: nextMessages,
                streamId,
                sawAssistantPayload: true,
                awaitingResponse: false
              }
        })
      }

      apply()
    },
    [updateSessionState]
  )

  const queuedDeltasRef = useRef<Map<string, QueuedStreamDeltas>>(new Map())
  const queuedRoutedDeltasRef = useRef<Map<string, QueuedRoutedStreamDeltas>>(new Map())
  const flushHandleRef = useRef<number | null>(null)
  const lastFlushAtRef = useRef<number>(0)
  const nativeSubagentSessionsRef = useRef<Set<string>>(new Set())
  // Sub-role tag from the current turn's message.start (monitor SPEAK /
  // deep-research threadback), keyed by session. Consumed by mutateStream's
  // seed branch to label + color the freshly-created assistant message, then
  // cleared on message.complete. A plain model reply leaves this unset.
  const pendingSubRoleRef = useRef<Map<string, ChatMessageMeta>>(new Map())
  // Turns that auto-compacted: skip post-turn hydrate so live scrollback survives.
  const compactedTurnRef = useRef<Set<string>>(new Set())
  // Last session we applied a session.info cwd for — lets us tell an agent
  // relocating the SAME session (follow it) from a session switch (don't yank).
  const lastCwdInfoSessionRef = useRef<null | string>(null)

  const flushQueuedDeltas = useCallback(
    (sessionId?: string) => {
      const queue = queuedDeltasRef.current
      const ids = sessionId ? [sessionId] : [...queue.keys()]

      for (const id of ids) {
        const queued = queue.get(id)

        if (!queued) {
          continue
        }

        queue.delete(id)

        if (queued.assistant) {
          mutateStream(
            id,
            parts => dedupeGeneratedImageEchoesInParts(appendAssistantTextPart(parts, queued.assistant)),
            () => [assistantTextPart(queued.assistant)]
          )
        }

        if (queued.reasoning) {
          mutateStream(
            id,
            parts => appendReasoningPart(parts, queued.reasoning),
            () => [reasoningPart(queued.reasoning)]
          )
        }
      }

      // Detached QueryWorkers share the same runtime session but not the same
      // answer bubble. Drain them by explicit stream id so an older worker can
      // never append into the user's newer foreground turn.
      for (const [key, queued] of [...queuedRoutedDeltasRef.current.entries()]) {
        if (sessionId && queued.sessionId !== sessionId) {
          continue
        }

        queuedRoutedDeltasRef.current.delete(key)

        if (queued.assistant) {
          mutateStream(
            queued.sessionId,
            parts => dedupeGeneratedImageEchoesInParts(appendAssistantTextPart(parts, queued.assistant)),
            () => [assistantTextPart(queued.assistant)],
            { preserveForeground: true, streamId: queued.streamId }
          )
        }

        if (queued.reasoning) {
          mutateStream(
            queued.sessionId,
            parts => appendReasoningPart(parts, queued.reasoning),
            () => [reasoningPart(queued.reasoning)],
            { preserveForeground: true, streamId: queued.streamId }
          )
        }
      }
    },
    [mutateStream]
  )

  const scheduleDeltaFlush = useCallback(() => {
    if (flushHandleRef.current !== null) {
      return
    }

    if (typeof window === 'undefined') {
      flushQueuedDeltas()

      return
    }

    // Enforce a floor on the gap between two flushes. Without it, an LLM
    // emitting tokens slower than the rAF cadence (~30-80 tok/sec is typical)
    // forces one React commit + Streamdown re-parse per token, and the
    // last-block markdown re-parse cost is roughly linear in current block
    // length. With this floor, slower streams still coalesce ~2 tokens per
    // commit and the synthetic harness shows longtask counts drop from ~5/5s
    // to ~1/5s on big sessions (see scripts/profile-typing-lag.md).
    const sinceLast = performance.now() - lastFlushAtRef.current

    const runFlush = () => {
      flushHandleRef.current = null
      lastFlushAtRef.current = performance.now()
      flushQueuedDeltas()
    }

    if (sinceLast >= STREAM_DELTA_FLUSH_MS && typeof window.requestAnimationFrame === 'function') {
      flushHandleRef.current = window.requestAnimationFrame(runFlush)

      return
    }

    flushHandleRef.current = window.setTimeout(runFlush, Math.max(0, STREAM_DELTA_FLUSH_MS - sinceLast))
  }, [flushQueuedDeltas])

  const queueDelta = useCallback(
    (sessionId: string, key: keyof QueuedStreamDeltas, delta: string) => {
      if (!delta) {
        return
      }

      const queued = queuedDeltasRef.current.get(sessionId) ?? { assistant: '', reasoning: '' }
      queued[key] += delta
      queuedDeltasRef.current.set(sessionId, queued)
      scheduleDeltaFlush()
    },
    [scheduleDeltaFlush]
  )

  const queueRoutedAssistantDelta = useCallback(
    (sessionId: string, streamId: string, delta: string) => {
      if (!streamId || !delta) {
        return
      }

      const key = `${sessionId}\u0000${streamId}`

      const queued = queuedRoutedDeltasRef.current.get(key) ?? {
        assistant: '',
        reasoning: '',
        sessionId,
        streamId
      }

      queued.assistant += delta
      queuedRoutedDeltasRef.current.set(key, queued)
      scheduleDeltaFlush()
    },
    [scheduleDeltaFlush]
  )

  useEffect(
    () => () => {
      if (flushHandleRef.current !== null && typeof window !== 'undefined') {
        if (typeof window.cancelAnimationFrame === 'function') {
          window.cancelAnimationFrame(flushHandleRef.current)
        } else {
          window.clearTimeout(flushHandleRef.current)
        }
      }

      flushHandleRef.current = null
      flushQueuedDeltas()
    },
    [flushQueuedDeltas]
  )

  const appendAssistantDelta = useCallback(
    (sessionId: string, delta: string) => {
      if (!delta) {
        return
      }

      queueDelta(sessionId, 'assistant', delta)
    },
    [queueDelta]
  )

  const appendReasoningDelta = useCallback(
    (sessionId: string, delta: string, replace = false) => {
      if (!delta) {
        return
      }

      if (!replace) {
        queueDelta(sessionId, 'reasoning', delta)

        return
      }

      flushQueuedDeltas(sessionId)

      mutateStream(
        sessionId,
        (parts, message) => {
          if (replace && chatMessageText(message).trim()) {
            return parts
          }

          if (replace) {
            return [...parts.filter(part => part.type !== 'reasoning'), reasoningPart(delta)]
          }

          return appendReasoningPart(parts, delta)
        },
        () => [reasoningPart(delta)]
      )
    },
    [flushQueuedDeltas, mutateStream, queueDelta]
  )

  const upsertToolCall = useCallback(
    (
      sessionId: string,
      payload: GatewayEventPayload | undefined,
      phase: 'running' | 'complete',
      sourceEventType?: string
    ) => {
      // Text deltas flush on a timer but tool events apply now; flush first so
      // a tool part can't jump ahead of the text that preceded it.
      flushQueuedDeltas(sessionId)

      if (sessionInterrupted(sessionId)) {
        return
      }

      // The composer status stack owns todo display now (no inline panel) —
      // mirror every todo state the tool reports into its session store.
      if (payload?.name === 'todo') {
        const todos = parseTodos(payload.todos) ?? parseTodos(payload.result) ?? parseTodos(payload.args)

        if (todos) {
          setSessionTodos(sessionId, todos)
        }
      }

      if (!nativeSubagentSessionsRef.current.has(sessionId)) {
        for (const subagentPayload of delegateTaskPayloads(payload, phase, sourceEventType)) {
          upsertSubagent(
            sessionId,
            subagentPayload,
            true,
            phase === 'complete' ? 'delegate.complete' : 'delegate.running'
          )
        }
      }

      mutateStream(
        sessionId,
        parts => dedupeGeneratedImageEchoesInParts(upsertToolPart(parts, payload, phase)),
        () => upsertToolPart([], payload, phase),
        { pending: m => phase !== 'complete' || (m.pending ?? false) }
      )
    },
    [flushQueuedDeltas, mutateStream, sessionInterrupted]
  )

  const completeAssistantMessage = useCallback(
    (sessionId: string, text: string, options: CompleteAssistantMessageOptions = {}) => {
      let shouldHydrate = false

      const completedState = updateSessionState(sessionId, state => {
        // Late completion from an already-cancelled turn: cancelRun has
        // already finalized the bubble (kept the partial text, dropped it if
        // empty). Re-running the dedupe below would replace the partial with
        // the just-cancelled full text, so we settle and bail instead.
        if (state.interrupted && !options.preserveForeground) {
          return {
            ...state,
            awaitingResponse: false,
            busy: false,
            needsInput: false,
            pendingBranchGroup: null,
            streamId: null,
            turnStartedAt: null
          }
        }

        const streamId = options.streamId ?? state.streamId
        const finalText = renderMediaTags(text).trim()
        const completionError = completionErrorText(finalText, options.status)
        const normalize = (value: string) => value.replace(/\s+/g, ' ').trim()

        const replaceTextPart = (parts: ChatMessagePart[]) => {
          const visibleFinalText = stripGeneratedImageEchoes(finalText, generatedImageEchoSources(parts)).trim()
          const dedupeReference = normalize(visibleFinalText)

          // ★ 多段 turn (调工具前有说明文字 → 工具卡 → 调工具后文字): payload.text 通常只
          //   是【最后一段】(工具后的确认), 若把所有 text part 都删掉再补它, 会把【工具卡之前
          //   已流式好的说明文字】(如 "好的, 我来设置") 一起吞掉。所以: 保留出现在【首个工具
          //   part 之前】的 text part (它们是前置说明, 顺序/内容都对), 只替换尾部 text。
          const firstToolIdx = parts.findIndex(
            part => part.type !== 'text' && part.type !== 'reasoning'
          )

          const kept = parts.filter((part, idx) => {
            if (part.type === 'text') {
              // 首个工具之前的 text (前置说明) 保留; 其后的尾部 text 交给 finalText 替换。
              return firstToolIdx >= 0 && idx < firstToolIdx
            }

            if (part.type !== 'reasoning' || !dedupeReference) {
              return true
            }

            const r = normalize(part.text)

            return !(r && (dedupeReference.startsWith(r) || r.startsWith(dedupeReference)))
          })

          return visibleFinalText ? [...kept, assistantTextPart(visibleFinalText)] : kept
        }

        const completeMessage = (message: ChatMessage): ChatMessage =>
          completionError
            ? {
                ...message,
                error: completionError,
                parts: message.parts.filter(part => part.type !== 'text'),
                pending: false
              }
            : {
                ...message,
                parts: replaceTextPart(message.parts),
                pending: false
              }

        const completionMeta = options.preserveForeground ? undefined : pendingSubRoleRef.current.get(sessionId)

        const newAssistantFromCompletion = (): ChatMessage => ({
          id: streamId || `assistant-${Date.now()}`,
          requestId: options.requestId,
          role: 'assistant',
          parts: completionError ? [] : [assistantTextPart(finalText)],
          pending: false,
          branchGroupId: options.preserveForeground ? undefined : state.pendingBranchGroup ?? undefined,
          ...(completionError && { error: completionError }),
          // A complete-only monitor/deep-research turn (no prior streamed
          // message) still needs its label + coloring.
          ...(completionMeta
            ? {
                subRole: completionMeta.subRole,
                monitorLabel: completionMeta.monitorLabel,
                brief: completionMeta.brief,
                deepReportRid: completionMeta.eventId,
                model: completionMeta.model,
                voice: completionMeta.voice
              }
            : {})
        })

        const prev = state.messages
        let nextMessages = prev
        // A failed turn frequently carries no text (the backend bailed before
        // producing any), but it still needs a bubble to host the error —
        // otherwise the turn ends with a bare user message and no trace of
        // why nothing came back.
        const hasRenderableCompletion = Boolean(finalText || completionError)

        if (streamId) {
          nextMessages = prev.some(m => m.id === streamId)
            ? prev.map(m => (m.id === streamId ? completeMessage(m) : m))
            : hasRenderableCompletion
              ? [...prev, newAssistantFromCompletion()]
              : prev
        } else {
          const fallbackIndex = [...prev]
            .reverse()
            .findIndex(message => message.role === 'assistant' && !message.hidden)

          if (fallbackIndex >= 0) {
            const index = prev.length - 1 - fallbackIndex
            const existing = prev[index]
            const existingText = chatMessageText(existing).trim()

            if (existing.pending || (finalText && existingText === finalText)) {
              nextMessages = prev.map((message, messageIndex) =>
                messageIndex === index ? completeMessage(message) : message
              )
            } else if (hasRenderableCompletion) {
              nextMessages = [...prev, newAssistantFromCompletion()]
            }
          } else if (hasRenderableCompletion) {
            nextMessages = [...prev, newAssistantFromCompletion()]
          }
        }

        if (options.ephemeralControl) {
          nextMessages = nextMessages.filter(
            message =>
              message.id !== streamId &&
              (!options.requestId || message.requestId !== options.requestId)
          )
        }

        const hasInlineError = nextMessages.some(m => m.role === 'assistant' && m.error && !m.hidden)
        const lastVisible = [...nextMessages].reverse().find(m => !m.hidden)
        const unresolvedUserTail = lastVisible?.role === 'user'
        shouldHydrate = Boolean(
          !options.ephemeralControl &&
          !options.preserveForeground &&
            !completionError &&
            !hasInlineError &&
            !unresolvedUserTail &&
            (!state.sawAssistantPayload || !finalText)
        )

        return options.preserveForeground
          ? { ...state, messages: nextMessages }
          : {
              ...state,
              messages: nextMessages,
              streamId: null,
              pendingBranchGroup: null,
              awaitingResponse: false,
              busy: false,
              needsInput: false,
              turnStartedAt: null
            }
      })

      if (!options.ephemeralControl) {
        void refreshSessions().catch(() => undefined)
      }
      // Sync the freshly-titled row to other windows (e.g. main, when the turn
      // ran in the pop-out).
      if (!options.ephemeralControl) {
        broadcastSessionsChanged()
      }

      if (!options.preserveForeground && compactedTurnRef.current.delete(sessionId)) {
        shouldHydrate = false
      }

      if (shouldHydrate) {
        void hydrateFromStoredSession(3, completedState.storedSessionId, sessionId)
      }

      if (!options.ephemeralControl) {
        dispatchNativeNotification({
          body: text.slice(0, 140) || translateNow('notifications.native.turnDoneBody'),
          kind: 'turnDone',
          sessionId,
          title: translateNow('notifications.native.turnDoneTitle')
        })
      }
    },
    [hydrateFromStoredSession, refreshSessions, updateSessionState]
  )

  const failAssistantMessage = useCallback(
    (sessionId: string, errorMessage: string) => {
      updateSessionState(sessionId, state => {
        const streamId = state.streamId ?? `assistant-error-${Date.now()}`
        const groupId = state.pendingBranchGroup ?? undefined
        const prev = state.messages
        const error = errorMessage.trim() || 'Argus reported an error'

        const nextMessages = prev.some(m => m.id === streamId)
          ? prev.map(message =>
              message.id === streamId
                ? {
                    ...message,
                    error,
                    pending: false
                  }
                : message
            )
          : [
              ...prev,
              {
                id: streamId,
                role: 'assistant' as const,
                parts: [],
                error,
                pending: false,
                branchGroupId: groupId
              }
            ]

        return {
          ...state,
          messages: nextMessages,
          streamId: null,
          pendingBranchGroup: null,
          sawAssistantPayload: true,
          awaitingResponse: false,
          busy: false,
          needsInput: false,
          turnStartedAt: null
        }
      })
    },
    [updateSessionState]
  )

  const handleGatewayEvent = useCallback(
    (event: RpcEvent) => {
      const payload = event.payload as GatewayEventPayload | undefined
      const explicitSid = event.session_id || ''

      if (!explicitSid && gatewayEventRequiresSessionId(event.type)) {
        return
      }

      const sessionId = explicitSid || activeSessionIdRef.current
      const isActiveEvent = !!sessionId && sessionId === activeSessionIdRef.current

      if (event.type === 'gateway.ready') {
        return
      } else if (event.type === 'session.info') {
        // Apply session-scoped fields when the event targets the active
        // session, OR when it's a global broadcast and we have no session.
        const apply = explicitSid ? isActiveEvent : !activeSessionIdRef.current
        const statePatch = sessionInfoStatePatch(payload)
        const hasStatePatch = hasSessionInfoStatePatch(statePatch)
        const modelChanged = typeof payload?.model === 'string'
        const providerChanged = typeof payload?.provider === 'string'
        const runningChanged = typeof payload?.running === 'boolean'

        if (apply) {
          if (modelChanged) {
            setCurrentModel(payload!.model || '')
          }

          if (providerChanged) {
            setCurrentProvider(payload!.provider || '')
          }

          if (typeof payload?.cwd === 'string') {
            // The active session's agent can relocate itself (new repo/worktree
            // via the terminal). When the SAME active session's cwd actually
            // moves, follow it — refresh the project tree + scope so the sidebar
            // tracks the live thread. A fresh selection (different session id)
            // is a switch, not a move, so it refreshes data without yanking scope.
            const cwdMoved = payload.cwd !== $currentCwd.get()
            const sameSession = !!sessionId && sessionId === lastCwdInfoSessionRef.current

            lastCwdInfoSessionRef.current = sessionId
            setCurrentCwd(payload.cwd)

            if (cwdMoved && sameSession) {
              void followActiveSessionCwd(payload.cwd)
            }
          }

          if (typeof payload?.branch === 'string') {
            setCurrentBranch(payload.branch)
          }

          if (typeof payload?.personality === 'string') {
            setCurrentPersonality(normalizePersonalityValue(payload.personality))
          }

          if (typeof payload?.reasoning_effort === 'string') {
            setCurrentReasoningEffort(payload.reasoning_effort)
          }

          if (typeof payload?.service_tier === 'string') {
            setCurrentServiceTier(payload.service_tier)
          }

          if (typeof payload?.fast === 'boolean') {
            setCurrentFastMode(payload.fast)
          }

          if (typeof payload?.yolo === 'boolean') {
            setYoloActive(payload.yolo)
          }
        }

        if (sessionId && hasStatePatch) {
          updateSessionState(sessionId, state => ({
            ...state,
            ...statePatch,
            branch: statePatch.branch ?? state.branch,
            cwd: statePatch.cwd ?? state.cwd
          }))
        }

        if (apply) {
          if (runningChanged && sessionId) {
            updateSessionState(sessionId, state => {
              const busy = Boolean(payload!.running)

              if (state.busy === busy && (busy || !state.awaitingResponse)) {
                return state
              }

              if (busy) {
                return {
                  ...state,
                  busy,
                  turnStartedAt: state.turnStartedAt ?? Date.now()
                }
              }

              if (state.awaitingResponse && !state.sawAssistantPayload) {
                return state
              }

              return {
                ...state,
                awaitingResponse: false,
                busy,
                pendingBranchGroup: null,
                streamId: null,
                turnStartedAt: null
              }
            })
          }
        }

        if (payload?.usage && (!explicitSid || isActiveEvent)) {
          setCurrentUsage(current => ({ ...current, ...payload.usage }))
        }

        if (typeof payload?.credential_warning === 'string' && payload.credential_warning) {
          requestDesktopOnboarding(payload.credential_warning)
        }

        void refreshHermesConfig()

        if (modelChanged || providerChanged) {
          void queryClient.invalidateQueries({
            queryKey: explicitSid && sessionId ? ['model-options', sessionId] : ['model-options']
          })
        }
      } else if (event.type === 'message.start') {
        if (!sessionId) {
          return
        }

        // Monitor SPEAK is a side-channel notification owned by the
        // multimodal rail. Treating it as a foreground assistant turn here
        // duplicates the alert, can splice text into an in-flight main answer,
        // and lets message.complete clear unrelated prompts/busy state.
        if (payload?.source === 'monitor' || payload?.monitor_id) {
          return
        }

        const requestId = gatewayRequestId(payload)
        const routedStreamId = requestId ? requestStreamId(sessionId, requestId, true) : ''

        flushQueuedDeltas(sessionId)
        clearSessionSubagents(sessionId)
        setSessionCompacting(sessionId, false)
        // A new turn starts with nothing in flight. Stale from a previous turn
        // (e.g. one that died between tool.generating and tool.start) would
        // otherwise show a "preparing" line for a tool that will never run.
        clearSessionGeneratingTool(sessionId)
        compactedTurnRef.current.delete(sessionId)
        nativeSubagentSessionsRef.current.delete(sessionId)

        // Capture the turn's sub-role tag (monitor SPEAK / deep-research
        // threadback) so the assistant message this start opens renders with a
        // web-style label + coloring. Plain replies carry no source → cleared.
        const startSource = typeof payload?.source === 'string' ? payload.source : ''

        if (startSource === 'monitor' || startSource === 'watcher' || startSource === 'router') {
          const monitorLabel =
            typeof payload?.monitor_label === 'string' ? payload.monitor_label : undefined

          const brief = typeof payload?.brief === 'string' ? payload.brief : undefined

          // 事件 id: monitor→monitor_id, watcher/router→request_id。气泡上展示。
          const eventId =
            (typeof payload?.monitor_id === 'string' && payload.monitor_id) ||
            (typeof payload?.request_id === 'string' && payload.request_id) ||
            undefined

          pendingSubRoleRef.current.set(sessionId, {
            subRole: startSource === 'monitor' ? 'monitor' : 'router',
            monitorLabel,
            brief,
            eventId,
            model: $currentModel.get() || undefined,
            voice: payload?.voice === true
          })
        } else {
          pendingSubRoleRef.current.delete(sessionId)
        }

        if (isActiveEvent) {
          triggerHaptic('streamStart')
        }

        updateSessionState(sessionId, state => ({
          ...state,
          busy: true,
          awaitingResponse: true,
          // Every gateway turn carries a stable request_id. Pin the current
          // foreground stream to its dedicated bubble now, before a tool event
          // or text delta arrives. If this turn later becomes a deferred
          // QueryWorker reply, the route remains valid after session.info marks
          // the main agent idle and clears this foreground pointer.
          streamId: routedStreamId || state.streamId,
          sawAssistantPayload: false,
          interrupted: false,
          turnStartedAt: Date.now()
        }))

        if (isActiveEvent) {
          setTurnStartedAt(Date.now())
        }
      } else if (event.type === 'message.delta') {
        if (payload?.source === 'monitor' || payload?.monitor_id) {
          return
        }

        if (sessionId) {
          const delta = coerceGatewayText(payload?.text)
          const requestId = gatewayRequestId(payload)

          if (payload?.source === 'query_worker' && requestId) {
            const routedStreamId = requestStreamId(sessionId, requestId, true)
            queueRoutedAssistantDelta(sessionId, routedStreamId, delta)
          } else {
            appendAssistantDelta(sessionId, delta)
          }
        }
      } else if (event.type === 'thinking.delta') {
        // thinking.delta carries the kawaii spinner status (face + verb from
        // KawaiiSpinner), not real reasoning. The bottom-of-thread loading
        // indicator already covers that UX, so we ignore these events to
        // avoid a duplicative "Thinking" disclosure showing spinner text.
      } else if (event.type === 'reasoning.delta') {
        if (sessionId) {
          appendReasoningDelta(sessionId, coerceThinkingText(payload?.text))
        }

        if (isActiveEvent) {
          setPetActivity({ reasoning: true })
        }
      } else if (event.type === 'reasoning.available') {
        if (sessionId) {
          appendReasoningDelta(sessionId, coerceThinkingText(payload?.text), true)
        }

        if (isActiveEvent) {
          setPetActivity({ reasoning: true })
        }
      } else if (event.type === 'moa.reference') {
        // MoA reference-model output — surface as a labelled thinking chunk
        // (tagged with the source model) before the aggregator's response, so
        // the mixture-of-agents process is visible. Reuses the reasoning
        // disclosure rather than introducing a parallel surface.
        if (sessionId) {
          const label = coerceGatewayText(payload?.label) || 'reference'
          const idx = typeof payload?.index === 'number' ? payload.index : undefined
          const cnt = typeof payload?.count === 'number' ? payload.count : undefined
          const header = idx && cnt ? `◇ Reference ${idx}/${cnt} — ${label}` : `◇ Reference — ${label}`
          const body = coerceThinkingText(payload?.text)
          appendReasoningDelta(sessionId, `${header}\n${body}\n\n`, true)
        }

        if (isActiveEvent) {
          setPetActivity({ reasoning: true })
        }
      } else if (event.type === 'moa.aggregating') {
        // Status transition only; the aggregator's reply arrives via the normal
        // message stream. No reasoning/transcript mutation here.
        if (isActiveEvent) {
          setPetActivity({ reasoning: true })
        }
      } else if (event.type === 'message.complete') {
        if (!sessionId) {
          return
        }

        if (payload?.source === 'monitor' || payload?.monitor_id) {
          return
        }

        const requestId = gatewayRequestId(payload)
        const isDeferredQueryWorker = payload?.source === 'query_worker' && Boolean(requestId)
        const ephemeralControl = isEphemeralControl(payload)
        const completionStatus = typeof payload?.status === 'string' ? payload.status : undefined

        if (isDeferredQueryWorker) {
          // QueryWorker owns an older answer slot after the main agent has
          // already released the session. Finalize only that request's bubble:
          // clearing session-wide prompts/busy/streamId here would terminate a
          // newer foreground question that may currently be waiting on input.
          const routedStreamId = requestStreamId(sessionId, requestId, true)
          flushQueuedDeltas(sessionId)
          playCompletionSound()
          const finalText = coerceGatewayText(payload?.text) || coerceGatewayText(payload?.rendered)
          completeAssistantMessage(sessionId, finalText, {
            preserveForeground: true,
            requestId,
            status: completionStatus,
            streamId: routedStreamId
          })
          forgetRequestStream(sessionId, requestId)

          return
        }

        // Turn ended — drop any blocking prompt still open for THIS session
        // (e.g. interrupted, or the approval already resolved). Scoped to the
        // session so a background turn finishing can't wipe the active chat's
        // prompt, and vice versa.
        clearAllPrompts(sessionId)
        clearClarifyRequest(undefined, sessionId)
        setSessionCompacting(sessionId, false)
        // Turn over: nothing is being prepared any more. Load-bearing for the
        // case where the model emitted tool.generating and then answered without
        // ever calling the tool — the line would otherwise never clear.
        clearSessionGeneratingTool(sessionId)

        flushQueuedDeltas(sessionId)

        playCompletionSound()

        const finalText = coerceGatewayText(payload?.text) || coerceGatewayText(payload?.rendered)
        const routedStreamId = requestId ? requestStreamId(sessionId, requestId, false) : ''
        completeAssistantMessage(
          sessionId,
          finalText,
          routedStreamId
            ? { ephemeralControl, requestId, status: completionStatus, streamId: routedStreamId }
            : { ephemeralControl, requestId, status: completionStatus }
        )
        forgetRequestStream(sessionId, requestId)
        // Turn's sub-role tag is consumed — the next start re-sets it (or clears).
        pendingSubRoleRef.current.delete(sessionId)

        if (isActiveEvent) {
          setTurnStartedAt(null)

          // Pet beat: a finished turn always celebrates — go straight to the
          // jump, never linger on the run/reason pose. One atom update (clears
          // toolRunning/reasoning AND sets celebrate together) so no stray "run"
          // frame leaks to the sprite — including the popped-out overlay, which
          // mirrors each activity change. The jump runs ~2 loops, then settles.
          flashPetActivity({ celebrate: true, reasoning: false, toolRunning: false }, 2200)

          // Light up the pet's mail icon if the user wasn't looking when the turn
          // finished — a glanceable "new message" hint on the popped-out overlay.
          // Cleared when they open the app via the mail icon or refocus the window.
          if (typeof document !== 'undefined' && !document.hasFocus()) {
            markPetUnread()
          }
        }

        if (payload?.usage) {
          setCurrentUsage(current => ({ ...current, ...payload.usage }))
        }
      } else if (event.type === 'message.user_echo') {
        // ★ 后端发起的 turn (monitor/watcher hook 触发、通知轮询等) 的 user 指令不是
        //   用户在前端输入的 → 前端没本地加过 user 气泡。后端回显 message.user_echo,
        //   这里把它作为一条正式 user 消息显示 (对齐 web)。否则 hook 指令只进 history
        //   不显示, 用户看不到"是什么指令触发了主 agent 这轮回复"。
        if (!sessionId) {
          return
        }

        if (isEphemeralControl(payload)) {
          return
        }

        const echoText = coerceGatewayText(payload?.text).trim()

        if (echoText) {
          updateSessionState(sessionId, state => ({
            ...state,
            messages: [
              ...state.messages,
              {
                id: `user-echo-${Date.now()}`,
                requestId: gatewayRequestId(payload) || undefined,
                role: 'user',
                parts: [textPart(echoText)]
              } as ChatMessage
            ]
          }))
        }
      } else if (event.type === 'multimodal.asr_final') {
        // Streaming ASR submits the recognized text to the backend itself, so
        // it never travels through the composer's optimistic user-message
        // path. Mirror the server-owned voice turn into the main assistant-ui
        // transcript, keyed by the stable request id. Repeated final events
        // (for example after a reconnect) then update nothing instead of
        // stacking duplicate user bubbles.
        //
        // Unlike ordinary foreground events, ASR finals must be explicitly
        // session-scoped: the shared gateway can carry voice turns from a
        // background chat, and attributing an unscoped final to whichever chat
        // is focused would leak speech across sessions.
        if (!explicitSid || !sessionId) {
          return
        }

        const voiceText = coerceGatewayText(payload?.text).trim()

        if (voiceText) {
          const requestId = gatewayRequestId(payload)
          const messageId = requestId ? `voice-user-${requestId}` : `voice-user-${Date.now()}`

          updateSessionState(sessionId, state => {
            if (state.messages.some(message => message.id === messageId)) {
              return state
            }

            return {
              ...state,
              messages: [
                ...state.messages,
                {
                  id: messageId,
                  requestId: requestId || undefined,
                  role: 'user',
                  parts: [textPart(voiceText)],
                  voice: true
                }
              ]
            }
          })
        }
      } else if (event.type === 'session.title') {
        // Live auto-title push (titler runs async, after the turn's refresh).
        const storedId = typeof payload?.session_id === 'string' ? payload.session_id : ''
        const nextTitle = typeof payload?.title === 'string' ? payload.title.trim() : ''

        if (storedId && nextTitle) {
          setSessions(prev =>
            prev.map(s => (s.id === storedId || s._lineage_root_id === storedId ? { ...s, title: nextTitle } : s))
          )
        }
      } else if (event.type === 'tool.start' || event.type === 'tool.progress' || event.type === 'tool.generating') {
        if (!sessionId) {
          return
        }

        // `tool.generating` is a pre-call signal ("the model is still writing
        // this tool call's arguments") and arrives with tool_id=None. It must NOT
        // become a tool part: an id-less row is one the later id-bearing
        // tool.start can't always merge into — the old "two identical tool rows"
        // bug. So it goes to a transient store instead (see tool-generating.ts),
        // which the process block renders as a header-only "preparing" line with
        // no row.
        //
        // Why surface it at all: the backend fires tool.start only after the
        // arguments are fully written AND the guardrail/plugin/checkpoint
        // preflight has run, and for a multi-call batch it fires for the whole
        // batch at once. Dropping tool.generating outright therefore left the
        // entire pre-call window with no parts to render — the turn looked idle
        // for seconds, then the process block appeared already listing 2-3 tools.
        if (event.type === 'tool.generating') {
          setSessionGeneratingTool(sessionId, String(payload?.name || ''))
        } else {
          // The authoritative row exists now, so the placeholder has served its
          // purpose — clear it or it would sit above the real rows for the rest
          // of the turn.
          clearSessionGeneratingTool(sessionId)
          flushQueuedDeltas(sessionId)
          upsertToolCall(sessionId, toTodoPayload(payload) ?? payload, 'running', event.type)
        }

        if (isActiveEvent) {
          setPetActivity({ reasoning: false, toolRunning: true })
        }
      } else if (event.type === 'tool.complete') {
        if (sessionId) {
          flushQueuedDeltas(sessionId)
          upsertToolCall(sessionId, toTodoPayload(payload) ?? payload, 'complete', event.type)

          if (isActiveEvent) {
            setPetActivity({ toolRunning: false })
          }

          // A pending clarify blocks the turn, so the first tool.complete after
          // one is the clarify resolving — drop the "needs input" flag here so
          // the sidebar indicator clears as soon as it's answered, not only at
          // message.complete.
          updateSessionState(sessionId, state => (state.needsInput ? { ...state, needsInput: false } : state))

          // terminal/process tool calls are the only things that spawn or reap
          // background processes — sync the composer status stack right after.
          if (!sessionInterrupted(sessionId) && (payload?.name === 'terminal' || payload?.name === 'process')) {
            void refreshBackgroundProcesses(sessionId)
          }
        }

        if (typeof payload?.inline_diff === 'string' && payload.inline_diff.trim()) {
          recordToolDiff(payload.tool_id || payload.name || '', payload.inline_diff)
        }

        // A file-mutating tool just finished — nudge the git-mirroring surfaces
        // (coding rail, review pane, file tree) to refresh. Event-driven, not
        // polled: fires exactly when the agent touches the tree.
        if (payload && toolMayMutateFiles(payload)) {
          notifyWorkspaceChanged()
        }
      } else if (SUBAGENT_EVENT_TYPES.has(event.type)) {
        if (sessionId && payload && !sessionInterrupted(sessionId)) {
          if (!nativeSubagentSessionsRef.current.has(sessionId)) {
            pruneDelegateFallbackSubagents(sessionId)
          }

          nativeSubagentSessionsRef.current.add(sessionId)
          upsertSubagent(
            sessionId,
            payload as Record<string, unknown>,
            event.type === 'subagent.spawn_requested' || event.type === 'subagent.start',
            event.type
          )
        }
      } else if (event.type === 'clarify.request') {
        // Surface the clarify tool's overlay. The Python side is blocked on
        // `clarify.respond`, so without this handler the agent would hang
        // forever (see tools/clarify_tool.py + tui_gateway/server.py:_block).
        //
        // Store the request for whichever session raised it — even a background
        // one. clarify.request is a one-shot event; if we dropped it for an
        // unfocused session, that session would block on `clarify.respond`
        // indefinitely and re-focusing it could never recover (the event is
        // gone). Parking it per-session lets the user answer once they switch
        // over; the inline ClarifyTool reads the active session's entry.
        const requestId = typeof payload?.request_id === 'string' ? payload.request_id : ''
        const question = typeof payload?.question === 'string' ? payload.question : ''

        if (requestId && question) {
          setClarifyRequest({
            requestId,
            question,
            choices: Array.isArray(payload?.choices) ? payload!.choices!.filter(c => typeof c === 'string') : null,
            sessionId: sessionId ?? null
          })

          // The transcript only renders the active session, so a background
          // clarify is otherwise invisible (the row just keeps spinning like
          // it's working). Flag the session so the sidebar shows a persistent
          // "needs input" indicator on its row — works for the active session
          // too, and survives alt-tab / window blur (unlike a toast).
          if (sessionId) {
            updateSessionState(sessionId, state => ({ ...state, needsInput: true }))
          }

          dispatchNativeNotification({
            body: question,
            kind: 'input',
            sessionId,
            title: translateNow('notifications.native.inputTitle')
          })
        }
      } else if (event.type === 'approval.request') {
        // Dangerous-command / execute_code approval. The Python side is blocked
        // in _await_gateway_decision() until approval.respond lands; without
        // this the agent stalls until its 5-min timeout and the tool is BLOCKED.
        // Park it per-session (like clarify) so a *background* profile's turn can
        // raise it and wait — the sidebar flags "needs input" and the inline bar
        // surfaces once the user focuses that chat.
        const command = typeof payload?.command === 'string' ? payload.command : ''
        const description = typeof payload?.description === 'string' ? payload.description : 'dangerous command'

        setApprovalRequest({
          // false only when a tirith warning forbids it; backend omits the field otherwise.
          allowPermanent: payload?.allow_permanent !== false,
          command,
          description,
          sessionId: sessionId ?? null
        })

        if (sessionId) {
          updateSessionState(sessionId, state => ({ ...state, needsInput: true }))
        }

        dispatchNativeNotification({
          actions: [
            { id: 'approve', text: translateNow('notifications.native.approveAction') },
            { id: 'reject', text: translateNow('notifications.native.rejectAction') }
          ],
          body: command || description,
          kind: 'approval',
          sessionId,
          title: translateNow('notifications.native.approvalTitle')
        })
      } else if (event.type === 'sudo.request') {
        // Sudo password capture (tools/terminal_tool.py). Blocked on
        // sudo.respond {request_id, password}.
        const requestId = typeof payload?.request_id === 'string' ? payload.request_id : ''

        if (requestId) {
          setSudoRequest({ requestId, sessionId: sessionId ?? null })

          if (sessionId) {
            updateSessionState(sessionId, state => ({ ...state, needsInput: true }))
          }

          dispatchNativeNotification({
            body: translateNow('notifications.native.inputBody'),
            kind: 'input',
            sessionId,
            title: translateNow('notifications.native.inputTitle')
          })
        }
      } else if (event.type === 'secret.request') {
        // Skill credential capture (tools/skills_tool.py). Blocked on
        // secret.respond {request_id, value}.
        const requestId = typeof payload?.request_id === 'string' ? payload.request_id : ''

        if (requestId) {
          const envVar = typeof payload?.env_var === 'string' ? payload.env_var : ''
          const promptText = typeof payload?.prompt === 'string' ? payload.prompt : ''

          setSecretRequest({
            requestId,
            envVar,
            prompt: promptText,
            sessionId: sessionId ?? null
          })

          if (sessionId) {
            updateSessionState(sessionId, state => ({ ...state, needsInput: true }))
          }

          dispatchNativeNotification({
            body: promptText || envVar || translateNow('notifications.native.inputBody'),
            kind: 'input',
            sessionId,
            title: translateNow('notifications.native.inputTitle')
          })
        }
      } else if (event.type === 'terminal.read.request') {
        // read_terminal tool: serialize the renderer's xterm buffer and answer
        // immediately (Python blocks on the respond). Empty text = no live pane.
        const requestId = typeof payload?.request_id === 'string' ? payload.request_id : ''

        if (requestId) {
          const start = typeof payload?.start === 'number' ? payload.start : undefined
          const count = typeof payload?.count === 'number' ? payload.count : undefined
          const result = readActiveTerminal({ start, count })

          void $gateway.get()?.request('terminal.read.respond', {
            request_id: requestId,
            text: result ? JSON.stringify(result) : ''
          })
        }
      } else if (event.type === 'status.update') {
        if (sessionId && payload?.kind === 'compacting') {
          setSessionCompacting(sessionId, true)
          compactedTurnRef.current.add(sessionId)
        } else if (sessionId && payload?.kind === 'process') {
          // The gateway's notification poller announces background process
          // completions / watch matches here — re-sync the status stack.
          void refreshBackgroundProcesses(sessionId)
        }
      } else if (event.type === 'review.summary') {
        // Self-improvement background review saved something to memory/skills
        // and emitted a persistent summary (Python formats it as
        // "💾 Self-improvement review: …"). The CLI prints this via
        // prompt_toolkit and the Ink TUI renders it as a system line; the
        // desktop has neither, so without this handler the skill/memory
        // change happens silently. Surface it as a persistent system message
        // in the transcript so the user is always informed — it must not be a
        // transient toast that can be missed.
        const text = coerceGatewayText(payload?.text).trim()

        if (text && sessionId) {
          flushQueuedDeltas(sessionId)
          updateSessionState(sessionId, state => ({
            ...state,
            messages: [
              ...state.messages,
              {
                id: `review-summary-${Date.now()}`,
                role: 'system',
                parts: [textPart(text)],
                timestamp: Math.floor(Date.now() / 1000)
              }
            ]
          }))
        }
      } else if (event.type === 'error') {
        const errorMessage = payload?.message || 'Argus reported an error'
        const looksLikeProviderSetup = isProviderSetupErrorMessage(errorMessage)

        // A turn that errors out has also ended — drop any open blocking prompt
        // for this session so an approval/sudo/secret overlay can't linger past
        // the failed turn (same intent as the message.complete clear).
        if (sessionId) {
          clearAllPrompts(sessionId)
          clearClarifyRequest(undefined, sessionId)
          setSessionCompacting(sessionId, false)
          clearSessionGeneratingTool(sessionId)
          compactedTurnRef.current.delete(sessionId)
        }

        if (isActiveEvent) {
          setPetActivity({ reasoning: false, toolRunning: false })
          flashPetActivity({ error: true })
        }

        dispatchNativeNotification({
          body: errorMessage,
          kind: 'turnError',
          sessionId,
          title: translateNow('notifications.native.turnErrorTitle')
        })

        if (looksLikeProviderSetup) {
          requestDesktopOnboarding(errorMessage)
        } else {
          // Toast globally, not just when the failing thread is focused: a
          // turn-ending error (e.g. out of funds) blocks every thread, so the
          // inline error alone is too easy to miss. The stable id collapses the
          // same error from multiple blocked threads into one toast.
          notify({
            id: `gateway-error:${errorMessage}`,
            kind: 'error',
            title: 'Argus error',
            message: errorMessage
          })
        }

        if (sessionId) {
          flushQueuedDeltas(sessionId)
          failAssistantMessage(sessionId, errorMessage)
        }

        if (isActiveEvent) {
          setTurnStartedAt(null)
        }
      }
    },
    [
      appendAssistantDelta,
      appendReasoningDelta,
      activeSessionIdRef,
      completeAssistantMessage,
      failAssistantMessage,
      forgetRequestStream,
      flushQueuedDeltas,
      queueRoutedAssistantDelta,
      queryClient,
      refreshHermesConfig,
      requestStreamId,
      sessionInterrupted,
      updateSessionState,
      upsertToolCall
    ]
  )

  return {
    appendAssistantDelta,
    appendReasoningDelta,
    completeAssistantMessage,
    handleGatewayEvent,
    upsertToolCall
  }
}
