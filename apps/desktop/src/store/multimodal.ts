import { atom, computed } from 'nanostores'

import { translateNow } from '@/i18n'
import { compactQueryWorkerTrajectoryImages } from '@/lib/query-worker-trajectory-cache'

import { $gateway } from './gateway'
import { $activeSessionId } from './session'
import {
  ensureCaptureBoundToSession,
  isCapturing,
  pauseFrameLoop,
  resumeFrameLoop,
  stopCaptureAndNotify
} from './multimodal-capture'
import {
  cancelManualMicOnDisconnect,
  hasMicCaptureIntent,
  rearmMicAfterReconnect,
  rearmMicForSessionRebind,
  stopMic
} from './multimodal-voice'
import {
  $mmWatchers,
  applyBgEvent,
  appendMonitorAlert,
  appendMonitorAlertDelta,
  fetchMmRegistries,
  fetchMmSidechannel,
  finalizeMonitorAlert,
  pushMmToast,
  resetDeepUi,
  setMonitors,
  setWatchers,
  setWatcherFinal,
  type MmMonitor,
  type MmWatcher
} from './multimodal-deep'
import { onAsrBuffer, onAsrFinal, onAsrPartial, onTtsChunk, stopAllTts, type TtsChunk } from './multimodal-voice'

/**
 * Multimodal video-page store (desktop port of web MultimodalChatPage state).
 *
 * Self-contained: the multimodal page runs its OWN dedicated gateway session
 * (session.create), separate from the desktop session runtime, and subscribes
 * to message.* / clarify.request directly — it does NOT go through the desktop
 * session/assistant-ui machinery. Phase 1 covers text chat + deep-think +
 * inline clarify; media capture + panels land in later phases.
 */

// ── Message model ───────────────────────────────────────────────────────────
export type MmRole = 'user' | 'assistant' | 'system'
export type MmKind = 'chat' | 'tool' | 'status' | 'clarify'
export type MmSubRole = 'monitor' | 'router' | 'watcher_report' | undefined

export interface MmMessage {
  id: string
  role: MmRole
  text: string
  kind?: MmKind
  streaming?: boolean
  isError?: boolean
  // Extended-thinking / reasoning trace (reasoning.delta + thinking.delta),
  // rendered as a collapsible "💭 思考过程" block on the assistant bubble.
  reasoning?: string
  // This user turn came from streaming ASR (mic) → shows a 🎤 badge.
  voice?: boolean
  // sub-agent routing / coloring
  subRole?: MmSubRole
  // Deep-research FINAL answer threaded back to the main chat. The in-progress
  // watcher stream lives in the DeepPanel (right column); only the
  // threadback belongs in the central chat, so the waterfall shows only these.
  threadback?: boolean
  requestId?: string
  monitorId?: string
  monitorLabel?: string
  // Deep-research event name (the brief the user asked) — shown as a badge on
  // threadback/running bubbles instead of the old "已回传主对话" text.
  brief?: string
  // Client-local creation time (epoch ms), stamped in pushMsg → shown as an
  // absolute HH:MM:SS timestamp beside the role name.
  createdAt?: number
  // tool-entry fields
  toolId?: string
  toolName?: string
  toolDone?: boolean
  toolSummary?: string
  // Second segment of a dispatch tool card (the result note), shown under the
  // "正在派发 … 事件ID: #req_xxx" header line, newline-separated in one box.
  toolDetail?: string
  // clarify-entry fields (kind === 'clarify')
  clarifyReqId?: string
  clarifyQuestion?: string
  clarifyChoices?: string[]
  clarifyAnswer?: string
}

// Cap history so a long vision-heavy session can't grow unbounded.
const MAX_MESSAGES = 5000

// Placeholder-only welcome bubble that always sits at the top of the multimodal
// chat. Purely a frontend cue for capture UX — NEVER persisted to the backend
// session history (never sent through pushMsg → prompt.submit / history export),
// so it doesn't enter the agent's context on any turn.
const MM_WELCOME_ID = '__mm_welcome__'
// Welcome text is resolved at call-time via translateNow so it respects the active locale.
function getMmWelcomeText(): string {
  return translateNow('multimodal.welcome.body')
}
function makeMmWelcome(): MmMessage {
  return {
    id: MM_WELCOME_ID,
    role: 'system',
    text: getMmWelcomeText(),
    createdAt: Date.now(),
  }
}
/** Ensure the welcome bubble is present at the top of $mmMessages (idempotent).
 * Called on page mount + on every session switch so new AND refreshed sessions
 * both show it. */
export function ensureMmWelcome(): void {
  const list = $mmMessages.get()
  if (list.length > 0 && list[0].id === MM_WELCOME_ID) return
  const rest = list.filter(m => m.id !== MM_WELCOME_ID)
  $mmMessages.set([makeMmWelcome(), ...rest])
}

// Three-way connection state for the UI badge (mirrors web MultimodalChatPage):
//   'connecting'   — first connect in flight
//   'open'         — connected
//   'reconnecting' — dropped, gateway is retrying (primary gw auto-reconnects)
//   'closed'       — down / never connected
export type MmConnState = 'connecting' | 'open' | 'reconnecting' | 'closed'

// Observation panel model (multimodal.ctx): 画面观察 / 音频观察 / 搜索事实.
export interface MmObsItem {
  ts?: string
  speaker?: string
  text?: string
}
export interface MmCtx {
  version: number
  obs: MmObsItem[]
  audioObs: MmObsItem[]
  facts: Record<string, string>
}
// Injected-frame debug thumbnails (multimodal.anchor).
export interface MmAnchorFrame {
  ts?: number | null
  jpeg_b64?: string
}

/** Normalized, redacted row from `multimodal.trajectory`. The backend owns
 * redaction and image budgeting; this renderer cache only keeps the active
 * session's bounded QueryWorker subset for the inline tool card. */
export interface MmQueryTrajectoryEntry {
  event: string
  id: string
  payload: Record<string, unknown>
  phase: string
  seq: number
  ts: number
  worker: string
}

const MM_QUERY_TRAJECTORY_LIMIT = 2000
let _queryTrajectoryEpoch = 0
const EMPTY_QUERY_TRAJECTORY: MmQueryTrajectoryEntry[] = []
let _previousQueryTrajectoryIndex = new Map<string, MmQueryTrajectoryEntry[]>()

export const $mmMessages = atom<MmMessage[]>([])
export const $mmConnected = atom<boolean>(false)
export const $mmConnState = atom<MmConnState>('connecting')
export const $mmSessionId = atom<string>('')
export const $mmDeepThinking = atom<boolean>(false)
export const $mmCtx = atom<MmCtx>({ version: 0, obs: [], audioObs: [], facts: {} })
export const $mmAnchor = atom<MmAnchorFrame[]>([])
export const $mmQueryTrajectory = atom<MmQueryTrajectoryEntry[]>([])

function queryTrajectoryTaskId(value: MmQueryTrajectoryEntry): string {
  const result = trajectoryRecord(value.payload.result) || {}
  const taskId = String(value.payload.task_id || result.task_id || '').trim()

  return taskId.startsWith('qry_') ? taskId : ''
}

/** Build one task index per source update and preserve array identity for
 * untouched tasks. Inline tool cards subscribe to their task slice, so an OCR
 * event for Q2 no longer re-renders every historical Q1 card or makes each of
 * them rescan the full 2,000-row trace. */
const $mmQueryTrajectoryByTask = computed($mmQueryTrajectory, entries => {
  const toolOwners = new Map<string, string>()

  for (const entry of entries) {
    const taskId = queryTrajectoryTaskId(entry)
    const toolId = typeof entry.payload.tool_id === 'string' ? entry.payload.tool_id : ''

    if (taskId && toolId) {
      toolOwners.set(toolId, taskId)
    }
  }

  const mutable = new Map<string, MmQueryTrajectoryEntry[]>()

  for (const entry of entries) {
    const toolId = typeof entry.payload.tool_id === 'string' ? entry.payload.tool_id : ''
    const taskId = queryTrajectoryTaskId(entry) || toolOwners.get(toolId) || ''

    if (!taskId) {
      continue
    }

    const group = mutable.get(taskId) || []
    group.push(entry)
    mutable.set(taskId, group)
  }

  const indexed = new Map<string, MmQueryTrajectoryEntry[]>()

  for (const [taskId, group] of mutable) {
    const previous = _previousQueryTrajectoryIndex.get(taskId)
    const stable = previous?.length === group.length && previous.every((entry, index) => entry === group[index])

    indexed.set(taskId, stable ? previous : group)
  }

  _previousQueryTrajectoryIndex = indexed

  return indexed
})

export function queryTrajectoryTaskStore(taskId: string) {
  return computed(
    $mmQueryTrajectoryByTask,
    index => index.get(taskId) || EMPTY_QUERY_TRAJECTORY
  )
}

function trajectoryRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null
}

function validQueryTrajectoryEntry(value: unknown): value is MmQueryTrajectoryEntry {
  const row = trajectoryRecord(value)

  return Boolean(
    row &&
    typeof row.id === 'string' &&
    typeof row.event === 'string' &&
    typeof row.phase === 'string' &&
    typeof row.worker === 'string' &&
    Number.isFinite(Number(row.seq)) &&
    Number.isFinite(Number(row.ts)) &&
    trajectoryRecord(row.payload)
  )
}

/** Keep only rows that can belong to a QueryWorker task. Generic Monitor,
 * Watcher and MemoryWriter trajectory stays in Memory Debug and never bloats
 * every inline chat tool card. */
function queryTrajectoryEntry(value: MmQueryTrajectoryEntry): boolean {
  const payload = value.payload || {}
  const result = trajectoryRecord(payload.result) || {}
  const taskId = String(payload.task_id || result.task_id || '')
  const parentId = String(payload.parent_user_message_id || result.parent_user_message_id || '')
  const name = String(payload.name || payload.tool_name || result.name || '')
  const worker = value.worker.toLowerCase()

  return (
    name === 'query_multimodal' ||
    taskId.startsWith('qry_') ||
    (Boolean(taskId && parentId) &&
      (worker.includes('query') || worker.includes('recall') || worker.includes('search')))
  )
}

export function mergeMmQueryTrajectory(
  current: MmQueryTrajectoryEntry[],
  incoming: unknown[],
  preferIncoming = true
): MmQueryTrajectoryEntry[] {
  const merged = new Map(current.map(entry => [entry.id, entry]))

  for (const value of incoming) {
    if (validQueryTrajectoryEntry(value) && queryTrajectoryEntry(value)) {
      const previous = merged.get(value.id)

      const newer = previous && (
        value.seq > previous.seq ||
        (value.seq === previous.seq && value.ts > previous.ts)
      )
      const sameVersion = previous && value.seq === previous.seq && value.ts === previous.ts

      if (!previous || newer || (preferIncoming && sameVersion)) {
        merged.set(value.id, value)
      }
    }
  }

  return compactQueryWorkerTrajectoryImages(
    [...merged.values()]
      .sort((a, b) => a.seq - b.seq || a.ts - b.ts)
      .slice(-MM_QUERY_TRAJECTORY_LIMIT)
  )
}

/** Hydrate the active runtime's in-memory trace. It deliberately follows the
 * Web contract: trajectory is runtime-only, capped at 2,000 rows, and every
 * late response is rejected by exact gateway + live-session ownership. */
export async function fetchMmQueryTrajectory(): Promise<void> {
  const gw = $gateway.get()
  const sid = $mmSessionId.get()
  const epoch = _queryTrajectoryEpoch

  if (!gw || !sid) {
    return
  }

  try {
    const response = await gw.request<{ entries?: unknown[] }>(
      'multimodal.trajectory.list',
      { limit: MM_QUERY_TRAJECTORY_LIMIT, session_id: sid },
      60_000
    )

    if (
      $gateway.get() !== gw ||
      $mmSessionId.get() !== sid ||
      _queryTrajectoryEpoch !== epoch
    ) {
      return
    }

    $mmQueryTrajectory.set(
      mergeMmQueryTrajectory($mmQueryTrajectory.get(), response?.entries || [], false)
    )
  } catch {
    // Runtime trajectories are a debug enhancement. A cold/reaped runtime may
    // legitimately have none; chat and the QueryWorker answer remain healthy.
  }
}

// 左侧视频流列 (MediaRail) 是否折叠。由共享标题栏上的折叠按钮切换; 会话内内存态, 不持久化。
export const $mmRailCollapsed = atom<boolean>(false)
export function toggleMmRail(): void {
  $mmRailCollapsed.set(!$mmRailCollapsed.get())
}

// 两侧竖栏宽度 (px), 可拖拽调节, 带 min/max 夹取。默认 320 (=原 w-80)。内存态。
export const MM_RAIL_MIN_WIDTH = 240
export const MM_RAIL_MAX_WIDTH = 560
export const $mmMediaRailWidth = atom<number>(320)
export const $mmDeepRailWidth = atom<number>(320)
function clampRail(px: number): number {
  return Math.max(MM_RAIL_MIN_WIDTH, Math.min(MM_RAIL_MAX_WIDTH, Math.round(px)))
}
export function setMediaRailWidth(px: number): void {
  $mmMediaRailWidth.set(clampRail(px))
}
export function setDeepRailWidth(px: number): void {
  $mmDeepRailWidth.set(clampRail(px))
}

// Have we ever reached 'open'? Used to distinguish an INITIAL connect from a
// RECONNECT (non-open → open after having been open): on reconnect the server
// destroyed our session (close_on_disconnect), so we must rebuild it.
let _everOpen = false

// The gateway instance that owns the current $mmSessionId. On multi-profile
// setups $gateway is swapped when the active profile changes; the session id is
// only valid on the gateway that created it. If they diverge we must rebuild
// (else frames/prompts carry a sid the new socket's server doesn't know).
let _sessionGw: unknown = null

// ── Main-chat binding mode ───────────────────────────────────────────────────
// When ChatView hosts the multimodal rail (the merged layout), the multimodal
// side-channel rides the MAIN chat's active session instead of a dedicated
// session.create'd one: $mmSessionId mirrors $activeSessionId. In this mode the
// desktop session runtime (assistant-ui + use-route-resume) owns session
// lifecycle — create, resume-on-reconnect, id rotation on auto-compress — so the
// multimodal store must NOT create/clear the session itself. It only follows the
// active id and drives the frame loop. `ensureMultimodalSession` becomes a no-op
// (returns true iff there's an active session) and the onState reconnect handler
// skips the rebuild/clear it does for the standalone dedicated session.
let _boundToMain = false
let _unbindActive: (() => void) | null = null
let _preserveCaptureOnNextRuntimeRebind = false
let _claimArmedDraftOnNextRuntime = false

/** Claim the next newly-created live runtime for this fresh draft's armed
 * camera/screen. Call immediately before session.create; navigating from an
 * empty draft to an existing conversation deliberately does not claim it. */
export function claimCaptureForNextMainSession(): void {
  _claimArmedDraftOnNextRuntime = Boolean(
    !$mmSessionId.get() && (isCapturing() || hasMicCaptureIntent())
  )
}

export function cancelCaptureForNextMainSessionClaim(): void {
  _claimArmedDraftOnNextRuntime = false
}

export function clearCaptureSessionTransferClaims(): void {
  _claimArmedDraftOnNextRuntime = false
  _preserveCaptureOnNextRuntimeRebind = false
}

/** Preserve armed capture/voice input across a live runtime-id replacement for
 * the same durable conversation (for example session-not-found recovery).
 * User-driven session switches still stop both inputs and clear MM UI state. */
export function preserveCaptureForNextRuntimeRebind(): void {
  if (isCapturing() || hasMicCaptureIntent()) {
    _preserveCaptureOnNextRuntimeRebind = true
  }
}

/**
 * Bind the multimodal side-channel to the MAIN chat's active session
 * ($activeSessionId), instead of a dedicated session.create'd one. Idempotent.
 * Wires event handlers (attachMultimodalGateway) and keeps $mmSessionId synced
 * to $activeSessionId. Returns an unbind (used on ChatView unmount if needed).
 */
export function bindMultimodalToMainSession(): () => void {
  _boundToMain = true
  attachMultimodalGateway()
  // Seed + subscribe: $mmSessionId always mirrors the live main-chat session.
  if (_unbindActive) return _unbindActive
  const sync = (sid: string | null): void => {
    const next = sid || ''
    const prev = $mmSessionId.get()
    if (prev === next) {
      if (next) _preserveCaptureOnNextRuntimeRebind = false
      return
    }
    const micActive = hasMicCaptureIntent()
    const preserveRuntimeRebind = Boolean(
      _preserveCaptureOnNextRuntimeRebind && (isCapturing() || micActive)
    )
    const claimFreshRuntime = Boolean(
      !prev && next && _claimArmedDraftOnNextRuntime && (isCapturing() || micActive)
    )
    if (next) _preserveCaptureOnNextRuntimeRebind = false
    if (next) _claimArmedDraftOnNextRuntime = false
    if (prev !== next) {
      // A real old-session boundary owns and stops its stream. The initial
      // fresh-draft binding (''→B) instead keeps the already-authorized stream
      // armed, and a flagged A→B runtime replacement preserves the same
      // durable conversation's grant. Both are rebound below.
      if (isCapturing() && !preserveRuntimeRebind && !claimFreshRuntime) {
        stopCaptureAndNotify()
      }
      // Mic ownership is session-sticky. A real user switch closes A before
      // publishing B; a flagged same-conversation runtime replacement keeps
      // the user's intent and explicitly rearms against the replacement below.
      if (prev && micActive && !preserveRuntimeRebind) {
        void stopMic()
      }
      // Anything buffered/routed for the previous runtime must be discarded at
      // the same boundary as the visible MM panels. Otherwise an A event that
      // was accepted just before A→B can flush on the next timer tick and
      // repopulate B after reset.
      resetSessionEventRouting()
      // Future A chunks are rejected by mine(ev), but audio that has already
      // reached the player also belongs to A and must not continue into B.
      stopAllTts()
      $mmCtx.set({ version: 0, obs: [], audioObs: [], facts: {} })
      $mmAnchor.set([])
      resetDeepUi()
    }
    $mmSessionId.set(next)
    if (next) {
      $mmConnected.set(true)
      // 每次进入会话 (首次 seed 或 session 切换) 都置顶欢迎气泡; 幂等, 不入 backend history。
      ensureMmWelcome()
      // 进会话主动拉一次注册表: 有未完成 monitor/深度研究 → 面板自动打开 (不依赖后端 push 时机)。
      void fetchMmRegistries()
      // Hydrate monitor alerts + watcher reports from the sidechannel DB.
      // These no longer live in session["history"] so the frontend has to
      // pull them explicitly on session enter.
      void fetchMmSidechannel()
      // QueryWorker's structured trace is runtime-local (same as Web). Load it
      // when entering the session so a completed tool card can still show its
      // three frozen inputs and Recall/Search calls after a renderer remount.
      void fetchMmQueryTrajectory()
      // Camera/screen may have been opened on the fresh draft before a backend
      // session existed, or kept across a same-conversation runtime recovery.
      // Activation is single-flight and waits for backend MM services before
      // the frame loop starts.
      if (isCapturing() && (claimFreshRuntime || preserveRuntimeRebind)) {
        void ensureCaptureBoundToSession(next).catch(() => undefined)
      }
      // A claimed draft mic already owns its local graph. The first non-empty
      // PCM created this runtime, and that same generation starts ASR itself;
      // only a recovery rebind tears the graph down below.
      if (preserveRuntimeRebind && micActive) {
        void rearmMicForSessionRebind()
      }
    }
  }
  sync($activeSessionId.get())
  const off = $activeSessionId.subscribe(sync)
  _unbindActive = () => {
    off()
    _unbindActive = null
  }
  return _unbindActive
}

let _seq = 0
export function mmId(): string {
  _seq += 1
  return `mm${_seq}_${Date.now()}`
}

function cap(list: MmMessage[]): MmMessage[] {
  return list.length > MAX_MESSAGES ? list.slice(list.length - MAX_MESSAGES) : list
}

function pushMsg(m: MmMessage): void {
  // Stamp a client-local creation time once, here (single choke point for every
  // message), so the bubble can show an absolute HH:MM:SS timestamp.
  const stamped = m.createdAt ? m : { ...m, createdAt: Date.now() }
  $mmMessages.set(cap([...$mmMessages.get(), stamped]))
}

/** Format an epoch-ms timestamp as local HH:MM:SS for a message header. */
export function fmtClock(ms?: number): string {
  if (!ms) return ''
  const d = new Date(ms)
  const p = (n: number) => String(n).padStart(2, '0')
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`
}

function patchMsg(id: string, patch: Partial<MmMessage>): void {
  $mmMessages.set($mmMessages.get().map(m => (m.id === id ? { ...m, ...patch } : m)))
}

// ── 统一流式节流器 (对齐 web: ONE timer→rAF, 80ms 一次, 同帧 drain 三条流) ────────
// message.delta / reasoning.delta / multimodal.bg 都是 token 级高频事件。旧代码:
//   (1) 只批 message.delta 且用 rAF(~60fps) → 帧太密; (2) reasoning 每 token 直接
//   patchMsg O(N) map → 不批; (3) bg 每事件同步 applyBgEvent → 深度分析长回答时和主
//   agent 并发就把主线程撑爆 (MEMORY.md 记录的卡死)。
// 现在: text/reasoning 缓进 Map, bg 缓进队列 (answer_delta 按 (rid,seg,ch) 尾合并 →
//   队列不随 token 增长), 统一 80ms timer→rAF 一次性 drain。
const STREAM_FLUSH_MS = 80
const _textBuf = new Map<string, string>()
const _reasonBuf = new Map<string, string>()
let _bgQueue: Array<Record<string, unknown>> = []
let _flushTimer: ReturnType<typeof setTimeout> | null = null

function scheduleFlush(): void {
  if (_flushTimer !== null) return
  _flushTimer = setTimeout(() => {
    _flushTimer = null
    const run = (): void => runUnifiedFlush()
    if (typeof requestAnimationFrame === 'function') requestAnimationFrame(run)
    else run()
  }, STREAM_FLUSH_MS)
}

function runUnifiedFlush(): void {
  // 1) 主 agent 文本 + reasoning: 同一次 $mmMessages.set 打完 (一次重渲)。
  if (_textBuf.size || _reasonBuf.size) {
    const textDrain = new Map(_textBuf); _textBuf.clear()
    const reasonDrain = new Map(_reasonBuf); _reasonBuf.clear()
    $mmMessages.set(
      $mmMessages.get().map(m => {
        const td = textDrain.get(m.id)
        const rd = reasonDrain.get(m.id)
        if (td === undefined && rd === undefined) return m
        return {
          ...m,
          ...(td !== undefined ? { text: m.text + td } : {}),
          ...(rd !== undefined ? { reasoning: (m.reasoning || '') + rd } : {}),
        }
      })
    )
  }
  // 2) 深度分析 bg: 一次性 reduce 队列 (applyBgEvent 内部各自 set, 已 clone-on-write)。
  if (_bgQueue.length) {
    const drain = _bgQueue; _bgQueue = []
    for (const p of drain) {
      try { applyBgEvent(p as never) } catch (e) { console.warn('[mm] applyBgEvent failed:', e) }
    }
  }
}

// 立即 drain (message.complete 收尾时用, 保证末尾 token 落地)。
function flushDeltas(): void {
  if (_flushTimer !== null) { clearTimeout(_flushTimer); _flushTimer = null }
  runUnifiedFlush()
}

function queueDelta(id: string, delta: string): void {
  if (!delta) return
  _textBuf.set(id, (_textBuf.get(id) || '') + delta)
  scheduleFlush()
}

function queueReasoning(id: string, delta: string): void {
  if (!delta) return
  _reasonBuf.set(id, (_reasonBuf.get(id) || '') + delta)
  scheduleFlush()
}

// bg 事件入队, answer_delta 按 (rid,seg,channel) 尾合并 → 队列长度不随 token 增长。
function queueBg(p: Record<string, unknown>): void {
  if (p.type === 'answer_delta' && typeof p.delta === 'string') {
    const tail = _bgQueue[_bgQueue.length - 1]
    if (
      tail && tail.type === 'answer_delta'
      && (tail.request_id || '') === (p.request_id || '')
      && tail.seg === p.seg
      && (tail.channel || 'bg') === (p.channel || 'bg')
    ) {
      tail.delta = String(tail.delta || '') + p.delta
      scheduleFlush()
      return
    }
  }
  _bgQueue.push(p)
  scheduleFlush()
}

// Routing key for message.* events: request_id > monitor_id > __main__.
function keyOf(p: { request_id?: string; monitor_id?: string }): string {
  return p.request_id || p.monitor_id || '__main__'
}

// Per-key open assistant bubble id (concurrent deep-research / monitor streams).
const curAssistantId = new Map<string, string>()
// Per-key monitor alert routing. Keys mirror curAssistantId (via keyOf), but
// point to {monitorId, alertId} so message.delta / message.complete can append
// to $mmMonitorAlerts[monitorId] instead of the center chat.
const curMonitorAlertId = new Map<string, { monitorId: string; alertId: string }>()

/** Drop every pending/routed event owned by the previous session runtime. */
function resetSessionEventRouting(): void {
  if (_flushTimer !== null) {
    clearTimeout(_flushTimer)
    _flushTimer = null
  }
  _textBuf.clear()
  _reasonBuf.clear()
  _bgQueue = []
  curAssistantId.clear()
  curMonitorAlertId.clear()
  _queryTrajectoryEpoch += 1
  $mmQueryTrajectory.set([])
}

// ── Event wiring ────────────────────────────────────────────────────────────
// Registered against the live gateway; returns an unsubscribe that also tears
// down the session-scoped listeners. Idempotent per gateway instance.
type Off = () => void
let _teardown: Off | null = null
// The gateway the current handlers are bound to (so attach can detect a
// profile-driven $gateway swap and rebind to the new instance).
let _attachedGw: unknown = null

interface MessagePayload {
  text?: string
  source?: string
  request_id?: string
  monitor_id?: string
  monitor_label?: string
  status?: string
  brief?: string
  evidence?: unknown
}

// Monitor + watcher are BOTH decoupled from the center chat now. Monitor
// alerts route to $mmMonitorAlerts (right panel). Watcher round reports land
// in $mmBgItems via multimodal.bg + watcher.report_append (right panel too).
// Only main-agent / query-worker / router deep bubbles remain in the center.
function isMonitor(p: MessagePayload): boolean {
  return p.source === 'monitor' || !!p.monitor_id
}

/** Ensure a bubble/alert exists for this event's key. Monitor events land in
 *  the right-panel alerts store; everything else lands in the center chat. */
function ensureBubble(p: MessagePayload): string {
  const key = keyOf(p)
  if (isMonitor(p)) {
    const monitorId = p.monitor_id || key
    const existing = curMonitorAlertId.get(key)
    if (existing) return existing.alertId
    const alertId = appendMonitorAlert(monitorId, '', Date.now(), true)
    curMonitorAlertId.set(key, { monitorId, alertId })
    return alertId
  }
  const existing = curAssistantId.get(key)
  if (existing) return existing
  const id = mmId()
  curAssistantId.set(key, id)
  pushMsg({
    id,
    role: 'assistant',
    text: '',
    streaming: true,
    brief: p.brief,
    requestId: p.request_id,
  })
  return id
}

/** Attach multimodal event handlers to the active gateway. Safe to call once. */
export function attachMultimodalGateway(): Off {
  const gw = $gateway.get()
  if (!gw) return () => undefined
  // Already bound to THIS gateway → reuse. Bound to a DIFFERENT one (profile
  // switch swapped $gateway) → tear the old handlers down and rebind here so
  // events + reconnect logic follow the active socket.
  if (_teardown) {
    if (_attachedGw === gw) return _teardown
    _teardown()
    // Profile/gateway ownership changes before the replacement session atom may
    // update. Fail closed synchronously so the new profile never renders or
    // merges the previous profile's runtime-only trajectory.
    resetSessionEventRouting()
  }
  _attachedGw = gw

  const offs: Off[] = []

  // ★ Session-scoped guard. The multimodal store rides the SHARED primary
  //   gateway socket (same one the main desktop chat uses). Every event carries
  //   ev.session_id (backend _emit stamps the session's sid). If an event is for
  //   a DIFFERENT session (e.g. another chat tab on the same profile), dropping
  //   it here prevents cross-session pollution of our __main__ bubble
  //   (unscoped-key collision → text bleed / premature finalize / stuck streams).
  //   Events with no session_id are treated as ours (legacy/broadcast) only when
  //   we don't yet have a session id.
  const mine = (ev: { session_id?: string }): boolean => {
    const evSid = ev.session_id || ''
    const mySid = $mmSessionId.get()
    // Preserve legacy/bootstrap broadcasts only while no MM runtime is bound.
    // Once bound, an unscoped event cannot be proven to belong to this session
    // and must not be allowed to mutate its UI.
    if (!evSid) {
      return !mySid
    }
    return evSid === mySid
  }

  // Connection lifecycle: drive the 3-way badge AND rebuild the session on
  // reconnect. The primary gateway auto-reconnects (use-gateway-boot backoff);
  // we react to its state transitions here. onState fires immediately with the
  // current state, then on every change.
  offs.push(
    gw.onState(state => {
      if (state === 'open') {
        $mmConnState.set('open')
        $mmConnected.set(true)
        if (_boundToMain) {
          // Main-chat mode: the desktop session runtime owns session lifecycle
          // (create + resume-on-reconnect). We never rebuild here; $mmSessionId
          // is re-synced by the $activeSessionId subscription once the runtime
          // resumes. A short disconnect may reuse the same runtime id, so its
          // atom subscription will not fire again; explicitly rehydrate the
          // runtime-only QueryWorker trace that we cleared while offline.
          void fetchMmQueryTrajectory()
          // Resume the frame loop against the live session. Capture activation
          // keeps its own transport/attempt guards for a stale pre-resume id.
          resumeFrameLoop()
          void rearmMicAfterReconnect()
          _everOpen = true
          return
        }
        if (_everOpen && !$mmSessionId.get()) {
          // Reconnected after a drop: the old server session is gone
          // (close_on_disconnect), so rebuild it, THEN resume frame capture
          // (kept paused-not-released across the drop, so the grant survives).
          void ensureMultimodalSession().then(ok => {
            if (ok) {
              resumeFrameLoop()
              // Re-arm mic ASR against the NEW session (the old ASR session was
              // reaped) so background voice transcription survives a reconnect.
              void rearmMicAfterReconnect()
            }
          })
        } else {
          resumeFrameLoop()
        }
        _everOpen = true
      } else {
        // Non-open: a prior session on the server is being / has been reaped.
        // Drop our stale session id so the next 'open' rebuilds cleanly, close
        // out any streaming bubbles, and PAUSE (don't release) frame capture so
        // we stop encoding into a dead socket without losing the camera grant.
        // In main-chat mode $mmSessionId mirrors $activeSessionId (owned by the
        // desktop runtime) — don't clear it; the runtime handles resume.
        cancelManualMicOnDisconnect()
        if (!_boundToMain) $mmSessionId.set('')
        $mmConnected.set(false)
        $mmConnState.set(state === 'connecting' && !_everOpen ? 'connecting' : 'reconnecting')
        resetSessionEventRouting()
        pauseFrameLoop()
        const list = $mmMessages.get()
        if (list.some(m => m.streaming)) {
          $mmMessages.set(list.map(m => (m.streaming ? { ...m, streaming: false } : m)))
        }
      }
    })
  )

  offs.push(
    gw.on<MessagePayload>('message.start', ev => {
      if (!mine(ev)) return
      const p = ev.payload || {}
      ensureBubble(p)
    })
  )

  offs.push(
    gw.on<MessagePayload>('message.delta', ev => {
      if (!mine(ev)) return
      const p = ev.payload || {}
      if (isMonitor(p)) {
        // Route deltas to the monitor alert list (right panel), NOT the chat.
        ensureBubble(p)
        const rec = curMonitorAlertId.get(keyOf(p))
        if (rec && p.text) {
          appendMonitorAlertDelta(rec.monitorId, rec.alertId, p.text)
        }
        return
      }
      const id = curAssistantId.get(keyOf(p)) ?? ensureBubble(p)
      // Batched (rAF flush) instead of a per-token full-list map.
      queueDelta(id, p.text || '')
    })
  )

  offs.push(
    gw.on<MessagePayload>('message.complete', ev => {
      if (!mine(ev)) return
      const p = ev.payload || {}
      if (isMonitor(p)) {
        // Finalize the monitor alert on the right-panel list. If the whole
        // stream arrived as a single complete, synthesize the alert now so
        // its text isn't lost. Then flip streaming → false.
        ensureBubble(p)
        const key = keyOf(p)
        const rec = curMonitorAlertId.get(key)
        curMonitorAlertId.delete(key)
        if (!rec) return
        finalizeMonitorAlert(rec.monitorId, rec.alertId, p.text || '', p.evidence)
        return
      }
      // (Watcher no longer streams into the center chat via message.* — those
      // dead source=watcher/watcher_running/watcher_threadback branches were
      // removed. Watcher process/report live in the right panel + arrive as
      // folded bubbles via watcher.report_append.)
      const key = keyOf(p)
      const id = curAssistantId.get(key) ?? ensureBubble(p)
      curAssistantId.delete(key)
      // Drain any buffered deltas for this bubble before finalizing its text.
      flushDeltas()
      const cur = $mmMessages.get().find(m => m.id === id)
      const finalText = (cur?.text || '').trim() ? cur!.text : p.text || ''
      patchMsg(id, { text: finalText, streaming: false })
    })
  )

  // Live-watcher is FULLY decoupled from the main agent chat: per-round process
  // arrives via multimodal.bg (DeepPanel segment cards) and the final
  // consolidated report via watcher.final (below). NOTHING is appended to the
  // center chat — no turn-2 mutation, no lock contention with user/agent turns.
  offs.push(
    gw.on<{ request_id?: string; brief?: string; text?: string }>('watcher.final', ev => {
      if (!mine(ev)) {
        return
      }
      const p = ev.payload || {}
      const rid = p.request_id || ''
      const text = (p.text || '').trim()
      if (!rid || !text) return
      setWatcherFinal(rid, text)
    })
  )

  // Per-round watcher reports are now a sidechannel signal only. The live
  // DeepWindow segments stream through multimodal.bg (unchanged path), and the
  // backend persists each round to mm_watcher_reports for reopen restore.
  // Nothing appended to the center chat — no double-render, no history
  // pollution. Kept as a no-op handler so the event stays subscribed for
  // future UI hooks (e.g. "new report" pulse on the panel header).
  offs.push(
    gw.on<{ request_id?: string; round?: number; text?: string }>('watcher.report_append',
      ev => {
        if (!mine(ev)) {
          return
        }
        /* no-op: live=bg, reopen=list_watcher_content */
      })
  )

  offs.push(
    gw.on<{ message?: string }>('error', ev => {
      if (!mine(ev)) return
      // An error aborts in-flight streams: clear open bubbles + surface it.
      _textBuf.clear(); _reasonBuf.clear(); _bgQueue = []
      curAssistantId.clear()
      stopAllTts()
      $mmMessages.set(
        cap([
          ...$mmMessages.get().map(m => (m.streaming ? { ...m, streaming: false } : m)),
          { id: mmId(), role: 'system', text: `${translateNow('multimodal.misc.errorPrefix')}: ${ev.payload?.message || 'unknown'}`, isError: true }
        ])
      )
    })
  )

  // Generic blocking clarify.request — inline in the waterfall (dedup by rid).
  offs.push(
    gw.on<{ request_id?: string; question?: string; choices?: string[] | null }>('clarify.request', ev => {
      if (!mine(ev)) {
        return
      }
      const p = ev.payload || {}
      const reqId = p.request_id || ''
      if (!reqId) return
      if ($mmMessages.get().some(m => m.kind === 'clarify' && m.clarifyReqId === reqId)) return
      const choices = Array.isArray(p.choices) ? p.choices.filter((c): c is string => typeof c === 'string') : []
      pushMsg({
        id: mmId(),
        role: 'assistant',
        text: '',
        kind: 'clarify',
        clarifyReqId: reqId,
        clarifyQuestion: p.question || translateNow('multimodal.misc.pleaseSelect'),
        clarifyChoices: choices
      })
    })
  )

  // Observation panels: 画面观察 / 音频观察 / 搜索事实. Backend pushes the full
  // log each time; cap arrays so a long session doesn't balloon panel renders.
  offs.push(
    gw.on<{ obs?: MmObsItem[]; audio_obs?: MmObsItem[]; facts?: Record<string, string>; version?: number }>(
      'multimodal.ctx',
      ev => {
        if (!mine(ev)) {
          return
        }
        const c = ev.payload || {}
        const rawObs = c.obs || []
        const rawAObs = c.audio_obs || []
        const clip = (a: MmObsItem[]): MmObsItem[] => (a.length > 200 ? a.slice(a.length - 200) : a)
        $mmCtx.set({
          version: c.version || 0,
          obs: clip(rawObs),
          audioObs: clip(rawAObs),
          facts: c.facts || {}
        })
      }
    )
  )

  // Injected-frame debug thumbnails (what the engine sees this turn).
  offs.push(
    gw.on<{ frames?: MmAnchorFrame[] }>('multimodal.anchor', ev => {
      if (!mine(ev)) {
        return
      }
      const frames = ev.payload?.frames
      if (Array.isArray(frames)) $mmAnchor.set(frames)
    })
  )

  // ── Tool call/result cards (dispatch tools: route_multimodal_query /
  //    set_monitor). Rendered as a two-segment tool message: header line =
  //    "[正在派发 … 事件ID: #req_xxx]" (dispatch_label), body = the result note
  //    (dispatch_note), one box, newline-separated (see waterfall Row). ──
  offs.push(
    gw.on<{ tool_id?: string; name?: string }>('tool.start', ev => {
      if (!mine(ev)) return
      const p = ev.payload || {}
      pushMsg({
        id: mmId(), role: 'assistant', text: '', kind: 'tool',
        toolId: p.tool_id, toolName: p.name || 'tool', toolDone: false
      })
    })
  )
  offs.push(
    gw.on<{ tool_id?: string; name?: string; summary?: string;
            dispatch_label?: string; dispatch_note?: string; result_text?: string }>(
      'tool.complete', ev => {
        if (!mine(ev)) return
        const p = ev.payload || {}
        const summary = p.dispatch_label || p.summary || ''
        const detail = p.dispatch_note || p.result_text || ''
        const list = $mmMessages.get()
        // Match the in-flight tool bubble by tool_id (fallback: last open of same name).
        let idx = p.tool_id ? list.findIndex(m => m.kind === 'tool' && m.toolId === p.tool_id && !m.toolDone) : -1
        if (idx < 0 && p.name) {
          for (let i = list.length - 1; i >= 0; i--) {
            const m = list[i]
            if (m.kind === 'tool' && m.toolName === p.name && !m.toolDone) { idx = i; break }
          }
        }
        if (idx >= 0) {
          const next = list.slice()
          next[idx] = { ...next[idx], toolDone: true, toolSummary: summary, toolDetail: detail }
          $mmMessages.set(next)
        } else {
          pushMsg({
            id: mmId(), role: 'assistant', text: '', kind: 'tool',
            toolId: p.tool_id, toolName: p.name || 'tool', toolDone: true,
            toolSummary: summary, toolDetail: detail
          })
        }
      })
  )

  // ── DeepResearch (RouterEngine) progress + monitor registry + follow-ups ──
  offs.push(
    gw.on<Record<string, unknown>>('multimodal.bg', ev => {
      if (!mine(ev)) {
        return
      }
      // ★ 入队走统一 80ms flush (answer_delta 尾合并), 不再每事件同步 applyBgEvent —
      //   深度分析长回答与主 agent 并发时不再撑爆主线程 (MEMORY.md 卡死)。
      queueBg((ev.payload || {}) as Record<string, unknown>)
    })
  )
  offs.push(
    gw.on<{ monitors?: MmMonitor[] }>('multimodal.monitors', ev => {
      if (!mine(ev)) return
      setMonitors(ev.payload?.monitors || [])
    })
  )
  offs.push(
    gw.on<{ watchers?: MmWatcher[] }>('multimodal.watchers', ev => {
      if (!mine(ev)) return
      setWatchers(ev.payload?.watchers || [])
    })
  )
  offs.push(
    gw.on<MmQueryTrajectoryEntry>('multimodal.trajectory', ev => {
      if (!mine(ev) || !validQueryTrajectoryEntry(ev.payload)) {
        return
      }

      $mmQueryTrajectory.set(mergeMmQueryTrajectory($mmQueryTrajectory.get(), [ev.payload]))
    })
  )
  // 监控/深度研究过程失败/停用 → 右侧面板底部 toast (3s 淡出)。不进 history、不发主气泡。
  offs.push(
    gw.on<{ level?: string; text?: string }>('multimodal.toast', ev => {
      if (!mine(ev)) return
      const p = ev.payload || {}
      pushMmToast({ level: p.level, text: p.text || '' })
    })
  )
  // ── reasoning / thinking deltas → the main agent's open bubble ────────────
  // ★ 走统一 flush 的 reasoning 缓冲 (queueReasoning), 不再每 token 直接 patchMsg
  //   O(N) map → 与文本/ bg 同帧一次重渲。
  const appendReasoning = (delta: string): void => {
    if (!delta) return
    // No request_id on these events → attach to the main agent's open bubble.
    const id = curAssistantId.get('__main__') ?? ensureBubble({})
    queueReasoning(id, delta)
  }
  offs.push(gw.on<{ text?: string }>('reasoning.delta', ev => { if (mine(ev)) appendReasoning(ev.payload?.text || '') }))
  offs.push(gw.on<{ text?: string }>('thinking.delta', ev => { if (mine(ev)) appendReasoning(ev.payload?.text || '') }))

  // ── Voice: streaming ASR preview/final + TTS PCM chunks ───────────────────
  offs.push(gw.on<{ text?: string; turn_id?: string }>('multimodal.asr_partial', ev => {
    if (mine(ev)) {
      onAsrPartial(ev.payload?.text || '', ev.payload?.turn_id)
    }
  }))
  offs.push(gw.on<{ segments?: string[]; turn_id?: string }>('multimodal.asr_buffer', ev => {
    if (mine(ev)) {
      onAsrBuffer(ev.payload?.segments || [], ev.payload?.turn_id)
    }
  }))
  offs.push(gw.on<{ text?: string; turn_id?: string }>('multimodal.asr_final', ev => {
    if (mine(ev)) {
      onAsrFinal(ev.payload?.text || '', ev.payload?.turn_id)
    }
  }))
  offs.push(gw.on<TtsChunk>('multimodal.tts', ev => {
    if (mine(ev)) {
      onTtsChunk(ev.payload || {})
    }
  }))

  _teardown = () => {
    offs.forEach(off => off())
    _teardown = null
    _attachedGw = null
  }
  return _teardown
}

// ── Actions ─────────────────────────────────────────────────────────────────

/** Push a user turn that came from streaming ASR (mic). The server already has
 *  the final transcript (asr_final is submitted server-side); this is the local
 *  echo, tagged with `voice` for the 🎤 badge. */
export function addVoiceUserMessage(text: string): void {
  const t = (text || '').trim()
  if (!t) return
  pushMsg({ id: mmId(), role: 'user', text: t, voice: true })
}

/** Create the dedicated multimodal session (idempotent per page mount). */
export async function ensureMultimodalSession(): Promise<boolean> {
  const gw = $gateway.get()
  if (!gw) {
    $mmConnected.set(false)
    return false
  }
  // Main-chat mode: the desktop runtime owns the session; we never create one.
  // Report ready iff there's a live active session id to ride.
  if (_boundToMain) {
    const sid = $activeSessionId.get() || ''
    if (sid) $mmSessionId.set(sid)
    $mmConnected.set(Boolean(sid))
    return Boolean(sid)
  }
  // Reuse an existing session ONLY if it belongs to the current gateway. When
  // the active profile (and thus $gateway) changed, the old sid is invalid on
  // the new socket — drop it and rebuild.
  if ($mmSessionId.get() && _sessionGw === gw) {
    $mmConnected.set(true)
    return true
  }
  if ($mmSessionId.get() && _sessionGw !== gw) {
    $mmSessionId.set('')
  }
  try {
    // close_on_disconnect: server reaps this session when the WS drops (so a
    // reconnect rebuilds cleanly, no orphan sessions). source: "tool" tags it
    // as a multimodal/tool session (hidden from the default `sessions list`).
    const res = await gw.request<{ session_id?: string }>('session.create', {
      close_on_disconnect: true,
      source: 'tool'
    })
    const sid = res?.session_id || ''
    $mmSessionId.set(sid)
    _sessionGw = gw
    $mmConnected.set(true)
    $mmConnState.set('open')
    return true
  } catch {
    $mmConnected.set(false)
    return false
  }
}

/** Submit a user turn to the main agent (with the deep-think flag). */
export async function sendMultimodalPrompt(rawText: string): Promise<void> {
  const text = (rawText || '').trim()
  if (!text) return
  const gw = $gateway.get()
  const sid = $mmSessionId.get()
  if (!gw || !sid) return
  pushMsg({ id: mmId(), role: 'user', text })
  try {
    await gw.request('prompt.submit', {
      session_id: sid,
      text,
      deep_thinking: $mmDeepThinking.get()
    })
  } catch (e) {
    pushMsg({
      id: mmId(),
      role: 'system',
      text: `${translateNow('multimodal.misc.errorPrefix')}: ${e instanceof Error ? e.message : String(e)}`,
      isError: true
    })
  }
}

/** True while a turn is still generating (any assistant/router bubble streaming)
 * → the composer shows a Stop button instead of Send. */
export const $mmGenerating = computed($mmMessages, msgs => msgs.some(m => m.streaming))

/** Interrupt the in-flight turn (the composer's Stop button). Backed by the
 * gateway's session.interrupt — aborts the live turn + clears any queued prompt.
 * Best-effort; also locally clears streaming flags so the UI settles even if the
 * server's final events race the interrupt. */
export async function interruptMultimodal(): Promise<void> {
  const gw = $gateway.get()
  const sid = $mmSessionId.get()
  // Optimistically clear streaming so Stop→Send flips immediately.
  const list = $mmMessages.get()
  if (list.some(m => m.streaming)) {
    $mmMessages.set(list.map(m => (m.streaming ? { ...m, streaming: false } : m)))
  }
  curAssistantId.clear()
  if (!gw || !sid) return
  try {
    await gw.request('session.interrupt', { session_id: sid })
  } catch {
    /* best-effort — the turn may have already finished */
  }
}

/** Answer an inline clarify request; collapses the bubble + unblocks the tool. */
export async function answerMultimodalClarify(reqId: string, answer: string): Promise<void> {
  let already = false
  $mmMessages.set(
    $mmMessages.get().map(m => {
      if (m.kind !== 'clarify' || m.clarifyReqId !== reqId) return m
      if (m.clarifyAnswer !== undefined) {
        already = true
        return m
      }
      return { ...m, clarifyAnswer: answer }
    })
  )
  if (already) return
  const gw = $gateway.get()
  const sid = $mmSessionId.get()
  if (!gw || !sid) return
  try {
    await gw.request('clarify.respond', { session_id: sid, request_id: reqId, answer })
  } catch {
    /* best-effort; the tool times out server-side */
  }
}

export function toggleMultimodalDeepThinking(): void {
  $mmDeepThinking.set(!$mmDeepThinking.get())
}

/** Reset page state on unmount.
 *
 * ★ Background form-factor: the app hides to the tray and KEEPS capturing video
 * in the background (isCapturing) or recording mic (mic recording). While that's
 * happening we MUST NOT tear down the gateway handlers — onState is the only
 * thing that pauses the frame loop on a drop and rebuilds the session +
 * re-arms capture on reconnect, and message/bg/monitor/ctx handlers keep the
 * background session working. Tearing them down on page unmount left background
 * capture pushing frames into a stale/dead session after any reconnect
 * (silent failure of the core feature). So:
 *   - background active  → keep handlers wired; do NOT clear _everOpen; only the
 *     UI needs no reset (the page is gone, but the session/handlers live on).
 *   - nothing active     → full teardown as before.
 */
export function resetMultimodalUi(): void {
  const backgroundActive = isCapturing() || hasMicCaptureIntent()
  if (backgroundActive) {
    // Leave gateway handlers + session + capture/mic running for the tray-hidden
    // background session. Nothing to reset — the page just unmounted.
    return
  }
  if (_teardown) _teardown()
  resetSessionEventRouting()
  _everOpen = false
  $mmMessages.set([])
  ensureMmWelcome()  // 完全 teardown 后 UI 也要留住这条置顶引导 (与 web 对齐)。
  // Clear observation/anchor panels too, so a re-enter doesn't show the prior
  // session's stale 画面观察/音频观察/搜索事实/注入帧 (parity with resetDeepUi).
  $mmCtx.set({ version: 0, obs: [], audioObs: [], facts: {} })
  $mmAnchor.set([])
  resetDeepUi()
  stopAllTts()
}
