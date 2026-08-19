import {
  normalizeMonitorEvidence,
  type MonitorEvidence,
  type MonitorEvidenceFrame
} from '@hermes/shared'
import { atom } from 'nanostores'

import { translateNow } from '@/i18n'

import { $gateway } from './gateway'
import { $mmSessionId } from './multimodal'
import { notifyError } from './notifications'

/**
 * DeepResearch (RouterEngine) sub-window + Monitor registry state (desktop port
 * of web MultimodalChatPage bgItems / monitors).
 *
 * multimodal.bg     → per-(request_id) progress items (search/recall/react/…)
 * multimodal.monitors → the set_monitor registry (list + enable toggle)
 */

// ── DeepResearch progress items ───────────────────────────────────────────
export interface DeepCrop {
  // Backend (crop_images event) sends `jpeg_b64` — must match or every crop
  // renders as a broken image.
  jpeg_b64?: string
  label?: string
}
/** One readable "segment" of a deep-research run — a single analysis round,
 *  rendered as a card: 🎬 第N段 [mm:ss–mm:ss] → 👁 看到 → 🔎/🧩 检索 → 📝 就绪. */
export interface DeepSegment {
  seg: number                 // segment/round index (1-based for display)
  tsRange?: [number, number]  // frame time range for the header
  scene?: string              // 场景标记 (后端从本段 thought 廉价提取, 标题行展示)
  saw?: string                // 👁 what the model saw this round (from `thought`)
  thinking?: string           // 💭 model's raw reasoning trace (thinking models)
  toolCalls?: { name: string; arg?: string }[]  // 🔧 tool calls issued this round
  toolErrors?: { name: string; error: string }[] // ⚠️ tool failures this round
  /** 🔎 search / 🧩 recall lines: query → result summary. */
  lookups: { kind: 'search' | 'recall'; query: string; result?: string; done?: boolean }[]
  ready?: boolean             // 📝 this segment's 解读 is generated
  readyChars?: number
  answer?: string             // 📝 this segment's interpretation text (for folding)
  crops?: DeepCrop[]          // 🖼 crop thumbnails (image search)
}
export interface DeepBgItem {
  id: string                  // one item per request_id
  requestId?: string
  /** UI label (lightweight summary of task_instruction) for the card title. */
  label?: string
  /** Ordered segment cards (one per productive analysis round). */
  segments: DeepSegment[]
  /** Frame-accumulation status (current/target frames + ttl countdown). */
  waiting?: { have: number; need: number; ttlSec?: number; ttlRemaining?: number; seg?: number; paused?: boolean } | null
  done?: boolean
  /** #2/#9: latest incremental deep-research report (accumulated per-batch). */
  report?: string
  reportBatches?: number
  /** Final consolidated report (summarize_watch) pushed once on completion via
   *  watcher.final. This is the watcher's authoritative result — shown in the
   *  panel; the main agent chat is never touched. */
  finalReport?: string
}

export const $mmBgItems = atom<DeepBgItem[]>([])

// ── Monitor registry ───────────────────────────────────────────────────────
export interface MmMonitor {
  monitor_id: string
  brief?: string
  monitor_query?: string
  label?: string
  enabled?: boolean
  // 后端 _push_monitors_event 已带运行态 status ("running"|"interrupted"|"done"|…),
  // 与 web 对齐: 状态副行显示运行态而非配置态。
  status?: string
  trigger_mode?: 'once' | 'continuous'
  silent?: boolean
  report_interval?: number | null
  created_at?: number
}
export const $mmMonitors = atom<MmMonitor[]>([])

// ── Research registry (set_live_watcher on/off) ───────────────────────────
// Mirrors the monitor registry: a resumed session re-registers interrupted
// research jobs (status="interrupted") so the panel can list them with an on/off
// switch. Enabling requires a live video stream (backend guard) — the toggle
// rolls back to off if the stream isn't running.
export interface MmWatcher {
  watcher_id: string
  label?: string
  task_instruction?: string
  status?: string // "running" | "disabled" | "interrupted" | ...
  hook_main_agent?: boolean
  created_at?: number
}
export const $mmWatchers = atom<MmWatcher[]>([])

// ── Monitor alerts (per monitor_id proactive alerts, sidechannel) ─────────
// Monitor SPEAK events no longer stream into the center chat — they land here
// under their monitor_id and render in the right multimodal panel (default:
// newest 2 per monitor, expandable). Hydrated on session.resume via the
// multimodal.list_monitor_alerts RPC (matches web behavior).
export interface MonitorAlert {
  id: string
  text: string
  ts: number
  streaming?: boolean
  evidence?: MonitorEvidence
}

// ★ 2026-08-19: 实现搬到 @hermes/shared/monitor-evidence, web 侧那份逐字副本
// 一并指向同一处 (两个 cap 之前是两边各写一遍的字面量)。这里保留 re-export,
// 所以 `from './multimodal-deep'` 的既有 import 面不变。
export { normalizeMonitorEvidence, type MonitorEvidence, type MonitorEvidenceFrame }

export const $mmMonitorAlerts = atom<Record<string, MonitorAlert[]>>({})
let _alertSeq = 0
function _nextAlertId(): string {
  _alertSeq += 1
  return `alert${_alertSeq}_${Date.now()}`
}
export function appendMonitorAlert(monitorId: string, text: string, ts?: number, streaming = false): string {
  if (!monitorId) return ''
  const id = _nextAlertId()
  const list = $mmMonitorAlerts.get()
  const cur = list[monitorId] || []
  $mmMonitorAlerts.set({ ...list, [monitorId]: [...cur, { id, text, ts: ts || Date.now(), streaming }] })
  return id
}
export function updateMonitorAlert(monitorId: string, alertId: string, patch: Partial<MonitorAlert>): void {
  const list = $mmMonitorAlerts.get()
  const cur = list[monitorId]
  if (!cur) return
  const next = cur.map(a => (a.id === alertId ? { ...a, ...patch } : a))
  $mmMonitorAlerts.set({ ...list, [monitorId]: next })
}

/** Append a delta chunk to an existing streaming alert. Cheap: only rewrites
 *  the targeted alert entry (identity of others unchanged so a memo'd list
 *  view survives a burst of tokens). */
export function appendMonitorAlertDelta(monitorId: string, alertId: string, delta: string): void {
  if (!delta) return
  const list = $mmMonitorAlerts.get()
  const cur = list[monitorId]
  if (!cur) return
  const next = cur.map(a => (a.id === alertId ? { ...a, text: a.text + delta } : a))
  $mmMonitorAlerts.set({ ...list, [monitorId]: next })
}

/** Finalize a streaming alert. If deltas already accumulated we keep them;
 *  otherwise use the payload's full text (single-complete case where no
 *  deltas arrived). Flips streaming → false either way. */
export function finalizeMonitorAlert(
  monitorId: string,
  alertId: string,
  finalText: string,
  evidence?: unknown
): void {
  const list = $mmMonitorAlerts.get()
  const cur = list[monitorId]
  if (!cur) return
  const next = cur.map(a => {
    if (a.id !== alertId) return a
    const text = a.text.trim() ? a.text : finalText
    return { ...a, text, streaming: false, evidence: normalizeMonitorEvidence(evidence) }
  })
  $mmMonitorAlerts.set({ ...list, [monitorId]: next })
}
export function replaceMonitorAlerts(byMonitor: Record<string, MonitorAlert[]>): void {
  $mmMonitorAlerts.set(byMonitor)
}

// ── Watcher reports (per watcher_id per-round reports, sidechannel) ───────
// Watcher round reports no longer inject folded bubbles into the center chat.
// They hydrate the DeepBgItem panel on resume via list_watcher_content, and
// stream live via multimodal.bg (existing path).
export interface WatcherReportRow {
  round_idx: number
  text: string
  label?: string
  ts: number
}
export const $mmWatcherReports = atom<Record<string, WatcherReportRow[]>>({})

export interface WatcherReportSnapshot {
  watcher_id: string
  round_idx: number
  text: string
  label?: string
  wall_ts: number
}

export interface WatcherFinalSnapshot {
  watcher_id: string
  text: string
  wall_ts: number
}

// ── 右侧面板底部 toast (监控/深度研究过程失败/停用通知) ────────────────────────
// 动画小框, 3s 后自动淡出移除。不进 history、不发主 Agent 气泡、无顶部通知。
export interface MmToast {
  id: string
  level: 'error' | 'warning' | 'info'
  text: string
}
export const $mmToasts = atom<MmToast[]>([])
let _toastSeq = 0
export function pushMmToast(t: { level?: string; text: string }): void {
  const text = (t.text || '').trim()
  if (!text) return
  _toastSeq += 1
  const id = `toast${_toastSeq}_${Date.now()}`
  const level = (t.level === 'warning' || t.level === 'info' ? t.level : 'error') as MmToast['level']
  $mmToasts.set([...$mmToasts.get(), { id, level, text }])
  // 6s 后移除 (前端做淡出动画; 这里到点删数据)。3s 用户来不及反应, 改 6s。
  setTimeout(() => {
    $mmToasts.set($mmToasts.get().filter(x => x.id !== id))
  }, 6000)
}

// ── multimodal.bg reducer (mirrors web handler) ────────────────────────────
interface BgPayload {
  type?: string
  channel?: string
  task_id?: string
  request_id?: string
  brief?: string
  label?: string
  phase?: string
  round?: number
  thought?: string
  can_answer?: boolean
  findings_len?: number
  elapsed_sec?: number
  findings?: string
  text_len?: number
  text_preview?: string
  source?: string
  frame_ts?: number
  target?: string
  crops?: DeepCrop[]
  observations?: { name?: string; obs_summary?: string }[]
  delegation_done?: boolean
  have?: number
  need?: number
  report?: string
  batches?: number
  // Search/recall live progress + summary fields (bg_progress / *_done).
  obs_summary?: string
  n_clues?: number
  frame_ts_range?: [number, number]
  // Segment index (1-based round) stamped on every round event by the engine,
  // so the panel groups a round's progress into one segment card.
  seg?: number
  // req: thinking trace (router_thinking) + tool calls (router_react) + text.
  text?: string
  tool_calls?: { name?: string; args?: Record<string, unknown> }[]
  // ttl countdown (waiting) + streaming answer chunk (answer_delta).
  ttl_sec?: number
  ttl_remaining?: number
  delta?: string
  // 场景标记: 后端首次拿到 thought 后补发一条带 scene_label 的 segment_start。
  scene_label?: string
  // 攒帧暂停 (无帧且 ttl 到 → 暂停不倒时, 前端显示"等待新画面…")。
  paused?: boolean
}

const clipStr = (s: unknown, n: number): string =>
  String(s || '').replace(/\s+/g, ' ').trim().slice(0, n)


/** Fold one multimodal.bg event into $mmBgItems (one item per request_id,
 *  holding readable per-round segment cards). Returns the active rid (for
 *  auto-expand) or '' when nothing to expand.
 *
 *  Segment model: each analysis round is a card — 🎬 第N段 [mm:ss–mm:ss] with
 *  👁 看到 (thought), 🔎/🧩 检索 (query→result), and 📝 就绪. Low-level ReAct
 *  plumbing (writer_start / distill / rN_decision / tool_obs) is intentionally
 *  dropped so the panel reads cleanly. */
export function applyBgEvent(p: BgPayload): string {
  const rid = p.request_id || ''
  const t = p.type || ''
  const ch = p.channel || 'bg'

  // Delegation-level done: mark this rid's item done so the window can collapse.
  if (p.delegation_done && rid) {
    $mmBgItems.set($mmBgItems.get().map(b => (b.requestId === rid && !b.done ? { ...b, done: true, waiting: null } : b)))
    return ''
  }

  const itemId = rid || `_:${ch}`
  const prev = $mmBgItems.get()
  const idx = prev.findIndex(b => b.id === itemId)
  // ★ 性能 (clone-on-write): 只浅拷 BgItem + segments 数组本身, 不深拷每个 segment。
  //   旧代码 segments.map(s => ({...s})) 每个事件都给【所有】段换新引用 → SegmentCard
  //   的 memo 全部失效 → 每次 flush 整列段重渲染。改成: 数组浅拷, 只有被本次修改的那个
  //   段 (cowAt 里) 才 clone, 其余段保持原引用 → memo 生效, 只重渲变化的那张卡。
  const cur: DeepBgItem =
    idx >= 0
      ? { ...prev[idx], segments: prev[idx].segments.slice() }
      : { id: itemId, requestId: rid || undefined, segments: [] }
  if (p.label) cur.label = p.label

  // watcher.final is delivered inline while multimodal.bg is throttled through
  // an 80ms queue.  Once the authoritative final report marks this item done,
  // late frame-accumulation/new-round events must not resurrect a terminal run
  // with a fresh "waiting" strip or empty segment.
  if (
    idx >= 0 &&
    prev[idx].done &&
    (t === 'waiting' || t === 'batch_ready' || t === 'segment_start')
  ) {
    return ''
  }

  // clone-on-write: 把 cur.segments[i] 换成新对象 (仅这一个段变 identity), 返回它供修改。
  const cowAt = (i: number): DeepSegment => {
    const copy = { ...cur.segments[i], lookups: cur.segments[i].lookups.slice() }
    cur.segments[i] = copy
    return copy
  }

  // The backend stamps `seg` (round index, 1-based) on every round event. Route
  // this event to the matching segment (create if new), so out-of-order deltas
  // still land in the right card.
  const segNo = typeof p.seg === 'number' && p.seg > 0 ? p.seg : undefined
  const segFor = (): DeepSegment => {
    if (segNo === undefined) {
      if (cur.segments.length === 0) cur.segments.push({ seg: 1, lookups: [] })
      return cowAt(cur.segments.length - 1)
    }
    const at = cur.segments.findIndex(x => x.seg === segNo)
    if (at < 0) {
      const s: DeepSegment = { seg: segNo, lookups: [] }
      cur.segments.push(s); cur.segments.sort((a, b) => a.seg - b.seg)
      return cur.segments.find(x => x.seg === segNo)!
    }
    return cowAt(at)
  }

  if (t === 'waiting') {
    // Frame-accumulation: frames + ttl countdown (before first round AND between).
    cur.waiting = {
      have: p.have ?? 0, need: p.need ?? 0,
      ttlSec: typeof p.ttl_sec === 'number' ? p.ttl_sec : undefined,
      ttlRemaining: typeof p.ttl_remaining === 'number' ? p.ttl_remaining : undefined,
      seg: typeof p.seg === 'number' ? p.seg : undefined,
      paused: !!p.paused,
    }
  } else if (t === 'answer_delta') {
    // Live streaming of this segment's interpretation, token by token.
    const s = segFor()
    if (p.delta) s.answer = (s.answer || '') + String(p.delta)
  } else if (t === 'batch_ready') {
    // ★ A5: 不再置 null (那会让攒帧条整块卸载→下次心跳再挂载, 闪烁)。原位标满额,
    //   保持挂载; 分析期间的心跳 waiting 继续原位更新, 直到 done 才真正清除。
    if (cur.waiting) cur.waiting = { ...cur.waiting, have: cur.waiting.need }
  } else if (t === 'segment_start') {
    // New analysis round → a fresh segment card with its frame time range.
    // ★ A5: 不清 waiting (保持攒帧条原位)。A1: 后端补发带 scene_label 的 segment_start。
    const s = segFor()
    if (p.frame_ts_range && p.frame_ts_range.length === 2) s.tsRange = p.frame_ts_range
    if (p.scene_label) s.scene = String(p.scene_label)
  } else if (t === 'router_react') {
    // The model's per-round reasoning → 👁 看到 + 🔧 tool calls (req ②).
    const s = segFor()
    // ★ 描述行不限字数 (对齐 web): 只折叠空白, 不 clip。
    if (p.thought) s.saw = String(p.thought).replace(/\s+/g, ' ').trim()
    const tc = p.tool_calls || []
    if (tc.length) {
      s.toolCalls = tc.map(c => ({
        name: String(c.name || 'tool'),
        arg: clipStr((c.args && (c.args.query ?? c.args.target ?? JSON.stringify(c.args))) as string, 60),
      }))
    }
  } else if (t === 'router_thinking') {
    // Raw reasoning trace from a thinking model (req ②).
    const s = segFor()
    if (p.text) s.thinking = ((s.thinking || '') + String(p.text)).slice(-2000)
  } else if (t === 'tool_error') {
    // A tool call failed (req ③: surface, don't swallow).
    const s = segFor()
    s.toolErrors = s.toolErrors || []
    s.toolErrors.push({ name: String(p.target || p.brief || 'tool'), error: clipStr(p.findings || p.obs_summary || translateNow('multimodal.misc.callFailed'), 120) })
  } else if (t === 'bg_progress') {
    // A search/recall dispatched — show it "in flight" (query, no result yet).
    const s = segFor()
    const kind = ch === 'recall' ? 'recall' : 'search'
    const query = clipStr(p.brief, 80)
    if (query && !s.lookups.some(l => l.kind === kind && l.query === query)) {
      s.lookups.push({ kind, query })
    }
  } else if (t === 'search_done' || t === 'recall_done') {
    const s = segFor()
    const kind = t === 'recall_done' ? 'recall' : 'search'
    const query = clipStr(p.brief, 80)
    const clues = t === 'recall_done' && p.n_clues ? translateNow('multimodal.misc.clues', p.n_clues) : ''
    const result = translateNow('multimodal.misc.foundChars', p.findings_len || 0, clues) + fmtElapsed(p.elapsed_sec)
    const existing = s.lookups.find(l => l.kind === kind && l.query === query && !l.done)
    if (existing) { existing.result = result; existing.done = true }
    else s.lookups.push({ kind, query, result, done: true })
  } else if (t === 'answer_ready') {
    const s = segFor()
    s.ready = true
    s.readyChars = p.text_len || 0
    // Store this segment's interpretation. Only use the preview if we DIDN'T
    // already stream the full answer via answer_delta (else we'd truncate it).
    if (p.text_preview && !(s.answer && s.answer.length >= (p.text_len || 0)))
      s.answer = clipStr(p.text_preview, 400)
    // Fallback "看到": some rounds jump straight to answering with an empty
    // thought (self-explanatory scene), leaving the card with only "📝就绪".
    // Use the answer preview so the segment card always has readable content.
    if (!s.saw && p.text_preview) s.saw = clipStr(p.text_preview, 140)
  } else if (t === 'progress_report') {
    cur.report = p.report || ''
    cur.reportBatches = p.batches || 0
  } else if (p.phase === 'crop_images') {
    const s = segFor()
    s.crops = (p.crops || []).filter(c => c.jpeg_b64)
  } else if (p.phase === 'done') {
    cur.done = true
    cur.waiting = null   // ★ A5: 整个深度研究结束 → 才真正撤掉攒帧条
  }
  // Note: writer_start / distill / tool_obs / rN_decision / start are
  // intentionally ignored — internal ReAct steps, not user-facing progress.

  const next = idx >= 0 ? prev.slice() : [...prev, cur]
  if (idx >= 0) next[idx] = cur
  // Cap segments per item so a very long run doesn't grow unbounded.
  if (cur.segments.length > 40) cur.segments = cur.segments.slice(-40)
  $mmBgItems.set(next.slice(-8))
  return rid
}

/** Fold the final consolidated watcher report (watcher.final) into its panel
 *  item, creating the item if the run produced no bg events yet. Also marks the
 *  item done. The main agent chat is intentionally NOT touched. */
export function setWatcherFinal(rid: string, text: string): void {
  const t = (text || '').trim()
  if (!rid || !t) return
  const prev = $mmBgItems.get()
  const idx = prev.findIndex(i => i.requestId === rid)
  if (idx >= 0) {
    const item = { ...prev[idx], finalReport: t, done: true, waiting: null }
    const next = prev.slice()
    next[idx] = item
    $mmBgItems.set(next)
  } else {
    $mmBgItems.set(
      [...prev, { id: rid, requestId: rid, segments: [], finalReport: t, done: true }].slice(-8)
    )
  }
}

function fmtElapsed(s?: number): string {
  return s != null ? ` · ${Number(s).toFixed(1)}s` : ''
}

// ── Deep-clarify open/close ─────────────────────────────────────────────────
// (deep-clarify removed — deep-research runs to completion, no follow-up prompt)

// ── Monitor registry + toggle ──────────────────────────────────────────────
export function setMonitors(list: MmMonitor[]): void {
  $mmMonitors.set(Array.isArray(list) ? list : [])
}

/** Toggle a monitor enabled/disabled (optimistic; rollback on failure). */
export async function toggleMonitor(monitorId: string, enabled: boolean): Promise<void> {
  const gw = $gateway.get()
  const sid = $mmSessionId.get()
  if (!gw || !sid) return
  // optimistic update
  const current = $mmMonitors.get().find(m => m.monitor_id === monitorId)
  if (current?.status === 'done' || current?.status === 'complete') return
  let optimistic: MmMonitor | undefined
  $mmMonitors.set($mmMonitors.get().map(m => (
    m.monitor_id === monitorId
      ? (optimistic = { ...m, enabled, status: enabled ? 'running' : 'interrupted' })
      : m
  )))
  try {
    await gw.request('multimodal.monitor_toggle', { session_id: sid, monitor_id: monitorId, enabled })
    // The backend push is best-effort. Pull the authoritative status/mode so a
    // completed one-shot or a resumed continuous task cannot leave the switch
    // and status label disagreeing.
    await fetchMmRegistries()
  } catch (err) {
    // Roll back only while this exact optimistic object still owns the row. A
    // backend push may have completed the one-shot while the RPC was pending;
    // object identity keeps that newer terminal state authoritative.
    $mmMonitors.set(
      $mmMonitors.get().map(m => (m === optimistic && current ? current : m))
    )
    // ★ 顶部提示失败原因 (而不是开关静默弹回, 用户不知为何); 2s 后自动淡出消失
    //   —— 恢复监控但没开视频流是常见的临时状态, 不该常驻顶栏要手动关。
    if (enabled) notifyError(err, translateNow('multimodal.misc.cannotEnableMonitor'), { durationMs: 2_000 })
  }
}

// ── Research registry + toggle ─────────────────────────────────────────────
export function setWatchers(list: MmWatcher[]): void {
  $mmWatchers.set(Array.isArray(list) ? list : [])
}

/** Toggle a deep-research on/off (optimistic; rollback on failure).
 *  A failed enable (no live stream) rolls the switch back to off — the "点 on
 *  没流就自动弹回" behavior the backend guard drives. */
export async function toggleWatcher(watcherId: string, enabled: boolean): Promise<void> {
  const gw = $gateway.get()
  const sid = $mmSessionId.get()
  if (!gw || !sid) {
    return
  }

  const current = $mmWatchers.get().find(r => r.watcher_id === watcherId)
  if (!current || ['done', 'complete', 'stopping', 'deleted'].includes(String(current.status || ''))) {
    return
  }

  // Stopping is an in-flight state: the backend lets the current round finish
  // before publishing interrupted/done. Keeping it distinct prevents a second
  // toggle while that transition is unresolved.
  const nextStatus = enabled ? 'running' : 'stopping'
  let optimistic: MmWatcher | undefined
  $mmWatchers.set(
    $mmWatchers.get().map(r => (
      r.watcher_id === watcherId
        ? (optimistic = { ...r, status: nextStatus })
        : r
    ))
  )
  try {
    await gw.request('multimodal.watcher_toggle', { session_id: sid, watcher_id: watcherId, enabled })
    // Push delivery is best-effort. Reconcile with the authoritative registry
    // so stopping settles to interrupted/done even when that push was missed.
    if ($gateway.get() === gw && $mmSessionId.get() === sid) {
      await fetchMmRegistries()
    }
  } catch (err) {
    // Roll back only if this exact optimistic object still owns the row. A
    // concurrent backend push (notably done) or a newer toggle replaces the
    // object and must remain authoritative.
    $mmWatchers.set(
      $mmWatchers.get().map(r => (r === optimistic ? current : r))
    )
    notifyError(err, enabled ? translateNow('multimodal.misc.cannotEnableResearch') : translateNow('multimodal.misc.cannotPauseResearch'), { durationMs: 2_000 })
  }
}

/** 进会话时主动拉取当前 monitor + watcher 注册表 (不依赖后端 push 时机)。
 *  有未完成任务 → 数组填充 → 右侧深度面板按 gate 自动打开。幂等: 后到的 push 会覆盖。 */
export async function fetchMmRegistries(): Promise<void> {
  const gw = $gateway.get()
  const sid = $mmSessionId.get()
  if (!gw || !sid) return
  try {
    const res = await gw.request<{ ready?: boolean; monitors?: MmMonitor[]; watchers?: MmWatcher[] }>(
      'multimodal.list_registries',
      { session_id: sid }
    )
    // A session switch clears the panel immediately, but the previous
    // session's request can still finish later. Never let that stale response
    // repopulate the new chat with the old monitor/watcher registry.
    if ($gateway.get() !== gw || $mmSessionId.get() !== sid) return
    // ★ 权威快照: 后端返回什么就是什么, 空数组也覆盖前一 session 残留。
    //   之前只在 Array.isArray 时才写入 → 后端不返回该 key 时不清 → 泄露。
    const resolvePull = <T>(current: T[], incoming: T[] | undefined): T[] => {
      if (!Array.isArray(incoming)) return current
      if (res?.ready === true || incoming.length > 0 || current.length === 0) return incoming
      return current
    }
    setMonitors(resolvePull($mmMonitors.get(), res?.monitors))
    setWatchers(resolvePull($mmWatchers.get(), res?.watchers))
  } catch {
    // best-effort; the push path still fills these if the backend emits later.
  }
}

function mergeWatcherReportRows(
  current: Record<string, WatcherReportRow[]>,
  snapshot: Record<string, WatcherReportRow[]>
): Record<string, WatcherReportRow[]> {
  const merged: Record<string, WatcherReportRow[]> = {}
  const watcherIds = new Set([...Object.keys(snapshot), ...Object.keys(current)])

  for (const watcherId of watcherIds) {
    // Persisted watcher rounds are append-only. Seed the snapshot, then let an
    // already-present/live row win on the same round so an older response
    // cannot rewind text or timestamps.
    const byRound = new Map<number, WatcherReportRow>()
    for (const row of snapshot[watcherId] || []) {
      byRound.set(row.round_idx, row)
    }
    for (const row of current[watcherId] || []) {
      byRound.set(row.round_idx, row)
    }
    merged[watcherId] = [...byRound.values()].sort((a, b) => a.round_idx - b.round_idx || a.ts - b.ts)
  }

  return merged
}

function mergeRestoredWatcherItem(restored: DeepBgItem, live: DeepBgItem): DeepBgItem {
  const liveSegmentNumbers = new Set(live.segments.map(segment => segment.seg))
  const historicalSegments = restored.segments.filter(segment => !liveSegmentNumbers.has(segment.seg))

  // Live rows win field-for-field, including rich segment data and terminal
  // state. Restored rows may only fill genuinely missing historical segments,
  // label, or a final report for an already-terminal live item.
  return {
    ...restored,
    ...live,
    done: live.done,
    finalReport: live.finalReport || (live.done ? restored.finalReport : undefined),
    label: live.label || restored.label,
    segments: [...historicalSegments, ...live.segments].sort((a, b) => a.seg - b.seg),
    waiting: live.waiting
  }
}

/** Merge a persisted watcher-content snapshot into state that may have
 *  received live events while the RPC was pending. Reports and finals form a
 *  union, so a watcher that persisted only its final still gets a panel item. */
export function mergeWatcherContentSnapshot(
  currentItems: DeepBgItem[],
  currentReports: Record<string, WatcherReportRow[]>,
  reports: WatcherReportSnapshot[],
  finals: WatcherFinalSnapshot[]
): { items: DeepBgItem[]; reports: Record<string, WatcherReportRow[]> } {
  const snapshotReports: Record<string, WatcherReportRow[]> = {}
  const finalByWatcher = new Map<string, WatcherFinalSnapshot>()

  for (const report of reports) {
    const watcherId = String(report.watcher_id || '')
    const text = String(report.text || '')
    if (!watcherId || !text) {
      continue
    }

    const rows = snapshotReports[watcherId] || (snapshotReports[watcherId] = [])
    rows.push({
      label: report.label,
      round_idx: Number(report.round_idx || 0),
      text,
      ts: typeof report.wall_ts === 'number' ? report.wall_ts * 1000 : Date.now()
    })
  }

  for (const final of finals) {
    const watcherId = String(final.watcher_id || '')
    const text = String(final.text || '')
    if (!watcherId || !text) {
      continue
    }

    finalByWatcher.set(watcherId, final)
    // Keep final-only watcher ids visible in the report registry as an empty
    // round list as well as in the DeepBgItem panel.
    snapshotReports[watcherId] ||= []
  }

  const mergedReports = mergeWatcherReportRows(currentReports, snapshotReports)
  const restoredItems = new Map<string, DeepBgItem>()
  const watcherIds = new Set([...Object.keys(snapshotReports), ...finalByWatcher.keys()])

  for (const watcherId of watcherIds) {
    const rows = mergedReports[watcherId] || []
    const final = finalByWatcher.get(watcherId)
    restoredItems.set(watcherId, {
      done: true,
      finalReport: final ? String(final.text || '') : undefined,
      id: watcherId,
      label: rows.find(row => row.label)?.label,
      requestId: watcherId,
      segments: rows.map(row => ({
        answer: row.text,
        lookups: [],
        ready: true,
        seg: row.round_idx || 1
      }))
    })
  }

  const mergedItems = new Map<string, DeepBgItem>()
  for (const item of currentItems) {
    mergedItems.set(item.requestId || item.id, item)
  }

  for (const [watcherId, restored] of restoredItems) {
    const live = mergedItems.get(watcherId)
    mergedItems.set(watcherId, live ? mergeRestoredWatcherItem(restored, live) : restored)
  }

  return { items: [...mergedItems.values()].slice(-8), reports: mergedReports }
}

/** Fetch persisted monitor alerts + watcher content for the current session
 *  and hydrate the right multimodal panel. Called after session.resume so a
 *  reopened session's proactive alerts and per-round watcher reports come
 *  back on the screen (they no longer live in session["history"]). */
export async function fetchMmSidechannel(): Promise<void> {
  const gw = $gateway.get()
  const sid = $mmSessionId.get()
  if (!gw || !sid) return
  // Monitor alerts → group by monitor_id, oldest-first.
  try {
    const res = await gw.request<{
      alerts?: Array<{
        monitor_id: string
        text: string
        wall_ts: number
        label?: string
        evidence?: unknown
      }>
    }>('multimodal.list_monitor_alerts', { session_id: sid })
    if ($gateway.get() !== gw || $mmSessionId.get() !== sid) return
    const rows = Array.isArray(res?.alerts) ? res!.alerts : []
    const byMon: Record<string, MonitorAlert[]> = {}
    for (const r of rows) {
      const mid = String(r.monitor_id || '')
      if (!mid) continue
      const ts = typeof r.wall_ts === 'number' ? r.wall_ts * 1000 : Date.now()
      const id = _nextAlertId()
      const list = byMon[mid] || (byMon[mid] = [])
      list.push({
        id,
        text: String(r.text || ''),
        ts,
        streaming: false,
        evidence: normalizeMonitorEvidence(r.evidence)
      })
    }
    // The backend persists and emits alerts on separate async paths. A live
    // alert can arrive after this request starts but before its older snapshot
    // returns; merge it instead of replacing it with the stale snapshot.
    const live = $mmMonitorAlerts.get()
    const merged: Record<string, MonitorAlert[]> = { ...byMon }
    for (const [mid, alerts] of Object.entries(live)) {
      const rows = merged[mid] ? [...merged[mid]] : []
      for (const alert of alerts) {
        const duplicate = rows.findIndex(
          row => row.text === alert.text && Math.abs(row.ts - alert.ts) <= 2_000
        )
        if (duplicate >= 0) rows[duplicate] = alert
        else rows.push(alert)
      }
      merged[mid] = rows.sort((a, b) => a.ts - b.ts)
    }
    replaceMonitorAlerts(merged)
  } catch {
    // best-effort — a resumed session with no alerts just has an empty map.
  }
  // Watcher reports → group by watcher_id, feed BgItem segments so the
  // DeepWindow's per-round cards come back (segments.answer = report text).
  try {
    const res = await gw.request<{
      reports?: WatcherReportSnapshot[]
      finals?: WatcherFinalSnapshot[]
    }>('multimodal.list_watcher_content', { session_id: sid })
    if ($gateway.get() !== gw || $mmSessionId.get() !== sid) return
    const rpts = Array.isArray(res?.reports) ? res!.reports : []
    const finals = Array.isArray(res?.finals) ? res!.finals : []
    const merged = mergeWatcherContentSnapshot(
      $mmBgItems.get(),
      $mmWatcherReports.get(),
      rpts,
      finals
    )
    $mmWatcherReports.set(merged.reports)
    $mmBgItems.set(merged.items)
  } catch {
    // best-effort
  }
}

/** Clear deep/monitor UI state (page reset). Capture/monitors server-side are
 *  untouched; this only clears the local view. */
export function resetDeepUi(): void {
  $mmBgItems.set([])
  $mmMonitors.set([])
  $mmWatchers.set([])
  $mmMonitorAlerts.set({})
  $mmWatcherReports.set({})
}
