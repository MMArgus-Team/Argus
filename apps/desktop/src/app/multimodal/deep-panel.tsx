import { useStore } from '@nanostores/react'
import { memo, useEffect, useMemo, useState } from 'react'

import { CompactMarkdown } from '@/components/chat/compact-markdown'
import { Switch } from '@/components/ui/switch'
import { useI18n } from '@/i18n'
import type { Translations } from '@/i18n'
import { isSynthSaw } from '@/lib/mm-sentinels'
import {
  $mmBgItems,
  $mmMonitorAlerts,
  $mmMonitors,
  $mmWatchers,
  type DeepBgItem,
  type DeepSegment,
  type MmWatcher,
  type MonitorAlert,
  toggleMonitor,
  toggleWatcher
} from '@/store/multimodal-deep'

import { DisclosureTitle, MmDisclosure } from './disclosure'

export function visibleWatchers(watchers: MmWatcher[]): MmWatcher[] {
  return watchers.filter(watcher => watcher.status !== 'deleted')
}

type WatcherState = 'interrupted' | 'running' | 'stopping' | 'done'

export function watcherPresentation(watcher: MmWatcher): {
  canToggle: boolean
  running: boolean
  state: WatcherState
} {
  const status = String(watcher.status || 'interrupted')
  const running = status === 'running'
  const done = status === 'done' || status === 'complete'
  const stopping = status === 'stopping'

  return {
    canToggle: !done && !stopping && status !== 'deleted',
    running,
    state: running ? 'running' : done ? 'done' : stopping ? 'stopping' : 'interrupted'
  }
}

function watcherStateLabel(t: Translations, state: WatcherState): string {
  switch (state) {
    case 'running':
      return t.multimodal.watcher.inProgress
    case 'done':
      return t.multimodal.watcher.completed
    case 'stopping':
      return t.multimodal.watcher.stopping
    default:
      return t.multimodal.watcher.interrupted
  }
}

/**
 * Right-column panel for the multimodal page: RouterEngine DeepResearch progress
 * sub-windows (grouped by request_id), and the Monitor registry with enable
 * toggles. Desktop port of web MultimodalChatPage DeepWindow / monitor panel.
 * (Deep-research runs to completion — no follow-up/clarify prompt.)
 */
export function DeepPanel() {
  const bgItems = useStore($mmBgItems)
  const monitors = useStore($mmMonitors)
  const watchers = useStore($mmWatchers)
  const watcherRows = useMemo(() => visibleWatchers(watchers), [watchers])
  const alertsByMonitor = useStore($mmMonitorAlerts)
  const hasAlerts = Object.values(alertsByMonitor).some(a => a && a.length > 0)

  // One item per request_id now (the reducer groups internally), so bgItems IS
  // the window list.
  const hasAny = bgItems.length > 0 || monitors.length > 0
    || watcherRows.length > 0 || hasAlerts

  // Empty → render nothing so the right rail collapses to just the video stage
  // (no placeholder box eating space), matching the web layout.
  if (!hasAny) return null

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto">
      {monitors.length > 0 && <MonitorList />}
      {hasAlerts && <MonitorAlertsPanel />}
      {watcherRows.length > 0 && <WatcherList watchers={watcherRows} />}
      {bgItems.map(item => (
        <DeepWindow key={item.id} item={item} />
      ))}
    </div>
  )
}

// mm:ss formatter for frame time ranges.
function fmtTs(s?: number): string {
  if (s == null || !isFinite(s)) return ''
  const m = Math.floor(s / 60)
  const sec = Math.floor(s % 60)
  return `${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`
}

/** One readable analysis-round card: 🎬 第N段 [mm:ss–mm:ss] → 💭 思考 → 👁 看到 →
 *  🔧 工具 → ⚠️ 失败 → 🔎/🧩 检索 → 🖼 crops → 📝 就绪. Foldable (req ④).
 *  ★ memo: 配合 store 的 clone-on-write (只有变化的段换新引用), 未变的段 memo 命中、
 *    不重渲染 → 长深度分析流式时不再整列段重渲 (对齐 web)。 */
const SegmentCard = memo(function SegmentCard({ s, defaultOpen }: { s: DeepSegment; defaultOpen?: boolean }) {
  const { t } = useI18n()
  const da = t.multimodal.deepAnalysis
  const range = s.tsRange ? ` ${fmtTs(s.tsRange[0])}–${fmtTs(s.tsRange[1])}` : ''
  // 真实描述: 排除后端合成的占位句 (isSynthSaw, 见 lib/mm-sentinels)。
  const desc = s.saw && !isSynthSaw(s.saw) ? s.saw : ''
  const empty = !desc && s.lookups.length === 0 && !s.ready && !(s.crops && s.crops.length)
    && !(s.toolCalls && s.toolCalls.length) && !(s.toolErrors && s.toolErrors.length)
  const [open, setOpen] = useState(!!defaultOpen)
  useEffect(() => { setOpen(!!defaultOpen) }, [defaultOpen])
  const cap = 'text-[length:var(--conversation-caption-font-size)] leading-snug'
  return (
    <div className="flex flex-col gap-1 rounded-md border border-(--ui-stroke-secondary) bg-(--ui-bg-elevated) p-2">
      {/* 标题行 (唯一可点击行): ▸/▾ + 第N段 + 时间戳 + 场景标记 chip。 */}
      <button
        onClick={() => setOpen(o => !o)}
        className="flex w-full items-center gap-1.5 text-left text-[0.6875rem] font-semibold text-(--ui-text-tertiary)"
      >
        <span className="shrink-0">{open ? '▾' : '▸'}</span>
        <span className="shrink-0">{da.segment(s.seg)}</span>
        {range && <span className="shrink-0 font-normal text-(--ui-text-tertiary) tabular-nums">{range}</span>}
        {s.scene && (
          <span className="ml-0.5 truncate rounded bg-(--ui-purple)/15 px-1.5 py-0.5 font-normal text-(--ui-text-secondary)">
            {s.scene}
          </span>
        )}
      </button>
      {/* 固定描述行 (来自 s.saw, 过滤占位句, 不限字数)。 */}
      {desc ? (
        <div className={`${cap} break-words text-(--ui-text-secondary)`}>{desc}</div>
      ) : empty ? (
        <div className={`${cap} text-(--ui-text-tertiary)`}>{da.analyzing}</div>
      ) : null}
      {/* 💭 思考: 折叠, 思考中(未 ready)图标脉冲。始终展示(不锁在 open 里)。 */}
      {s.thinking && (
        <details className={`${cap} text-(--ui-text-tertiary)`}>
          <summary className="flex cursor-pointer list-none select-none items-center gap-1">
            <span className={s.ready ? '' : 'animate-pulse'}>💭</span>
            <span>{s.ready ? da.thinking : da.thinkingInProgress}</span>
          </summary>
          <div className="mt-1 whitespace-pre-wrap break-words rounded bg-(--ui-purple)/5 px-1.5 py-1">{s.thinking}</div>
        </details>
      )}
      {/* 🔧 工具 / ⚠️ 失败 / 🔎 检索: 始终展示 (过程事实, 不锁在 open 里)。 */}
      {s.toolCalls?.map((c, i) => (
        <div key={`tc${i}`} className={`${cap} break-words text-(--ui-blue)`}>
          {da.toolCall(c.name, c.arg)}
        </div>
      ))}
      {s.toolErrors?.map((e, i) => (
        <div key={`te${i}`} className={`${cap} break-words text-(--ui-red)`}>
          {da.toolFailed(e.name, e.error)}
        </div>
      ))}
      {s.lookups.map((l, i) => (
        <div key={i} className={`${cap} break-words text-(--ui-text-secondary)`}>
          {da.lookupLine(l.kind, l.query, l.result)}
        </div>
      ))}
      {/* 折叠区: 占空间的重内容 (crops 缩略图 + 完整 markdown 解读)。 */}
      {open && (<>
        {s.crops && s.crops.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {s.crops.map((c, i) => (
              <img
                key={i}
                alt={c.label || `crop ${i}`}
                className="h-12 rounded border border-(--ui-stroke-secondary) object-cover"
                src={`data:image/jpeg;base64,${c.jpeg_b64}`}
              />
            ))}
          </div>
        )}
        {s.ready && (
          <div className={`${cap} text-(--ui-text-primary)`}>
            {s.answer ? (
              <div className="flex gap-1">
                <span className="shrink-0">📝</span>
                <div className="min-w-0 flex-1"><CompactMarkdown text={s.answer} /></div>
              </div>
            ) : (
              <span className="whitespace-pre-wrap text-(--ui-text-tertiary)">
                {da.noInterpretation}
              </span>
            )}
          </div>
        )}
      </>)}
    </div>
  )
})

function DeepWindow({ item }: { item: DeepBgItem }) {
  const { t } = useI18n()
  const da = t.multimodal.deepAnalysis
  const live = !item.done
  const label = item.label || ''

  // Collapsed preview: newest segment's most-informative line so a folded window
  // is never blank.
  const lastSeg = item.segments[item.segments.length - 1]
  const lastDesc = lastSeg?.saw && !isSynthSaw(lastSeg.saw) ? lastSeg.saw : ''
  const waitingSeg = item.waiting && typeof item.waiting.seg === 'number' ? item.waiting.seg : undefined
  const preview = item.waiting
    ? (item.waiting.paused ? da.previewWaiting(waitingSeg) : da.previewAccumulating(item.waiting.have, item.waiting.need, waitingSeg))
    : lastSeg
      ? (lastDesc ? da.previewSaw(lastDesc) : da.previewSegAnalyzing(lastSeg.seg))
      : ''

  const title = (
    <span className="flex min-w-0 flex-col gap-0.5">
      <span className="flex min-w-0 items-center gap-1.5">
        <DisclosureTitle live={live}>
          {label ? da.titleWithLabel(label) : `🔬 ${da.title}`}
        </DisclosureTitle>
      </span>
      {preview && (
        <span className="min-w-0 truncate text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
          {preview}
        </span>
      )}
    </span>
  )

  return (
    <section className="rounded-lg border border-(--ui-stroke-secondary) border-l-2 border-l-(--ui-purple) bg-(--ui-chat-surface-background) px-2.5 py-2">
      <MmDisclosure
        defaultOpen
        syncOpen={live}
        title={title}
        trailing={
          <span className="text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
            {live ? da.inProgress : da.completed}
          </span>
        }
      >
        <div className="flex flex-col gap-2">
          {/* Frame-accumulation banner — shown WHENEVER accumulating (before the
              first round AND between rounds). req ①: bouncing dots + frame progress
              bar + ttl countdown so the panel always reads as alive. */}
          {item.waiting && (
            item.waiting.paused ? (
              // paused: 无帧且 ttl 到 → 后端暂停攒帧, 不倒计时。显示"等待新画面…", 无进度条。
              <div className="flex items-center gap-1.5 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
                <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-(--ui-purple)/60" />
                <span>
                  {typeof item.waiting.seg === 'number' ? da.waitingFramesSeg(item.waiting.seg) : da.waitingFrames}
                </span>
              </div>
            ) : (
            <div className="flex flex-col gap-1 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
              <div className="flex items-center gap-1.5">
                <span className="inline-flex gap-0.5">
                  <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-(--ui-purple) [animation-delay:-0.3s]" />
                  <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-(--ui-purple) [animation-delay:-0.15s]" />
                  <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-(--ui-purple)" />
                </span>
                <span>
                  {typeof item.waiting.seg === 'number'
                    ? da.accumulatingSeg(item.waiting.seg, item.waiting.have, item.waiting.need)
                    : da.accumulating(item.waiting.have, item.waiting.need)}
                  {typeof item.waiting.ttlRemaining === 'number' && (
                    <span className="ml-1 text-(--ui-text-tertiary)">{da.ttlRemaining(Math.ceil(item.waiting.ttlRemaining))}</span>
                  )}
                </span>
              </div>
              <div className="h-1 w-full overflow-hidden rounded bg-(--ui-purple)/15">
                <div
                  className="h-full rounded bg-(--ui-purple)/70 transition-all duration-300"
                  style={{ width: `${Math.min(100, item.waiting.need ? (item.waiting.have / item.waiting.need) * 100 : 0)}%` }}
                />
              </div>
              {typeof item.waiting.ttlSec === 'number' && typeof item.waiting.ttlRemaining === 'number' && item.waiting.ttlSec > 0 && (
                <div className="h-0.5 w-full overflow-hidden rounded bg-(--ui-yellow)/15">
                  <div
                    className="h-full rounded bg-(--ui-yellow)/60 transition-all duration-300"
                    style={{ width: `${Math.min(100, Math.max(0, item.waiting.ttlRemaining / item.waiting.ttlSec) * 100)}%` }}
                  />
                </div>
              )}
            </div>
            )
          )}
          {/* Readable per-round segment cards. req ④: only the last (current)
              segment is expanded by default; older ones fold. */}
          {item.segments.map((s, i) => (
            <SegmentCard key={s.seg} s={s} defaultOpen={i === item.segments.length - 1} />
          ))}
          {/* Final consolidated report (watcher.final) — the authoritative
              result, shown in-panel; the main agent chat is never touched. */}
          {item.finalReport && (
            <div className="rounded-md border border-(--ui-purple) bg-(--ui-chat-surface-background) px-2.5 py-2">
              <div className="mb-1 text-[length:var(--conversation-caption-font-size)] font-medium text-(--ui-purple)">
                {da.finalReport}
              </div>
              {/* 最终报告走 CompactMarkdown (对齐 web: 表格/latex/代码渲染)。 */}
              <div className="text-[length:var(--conversation-body-font-size)] text-(--ui-text-primary)">
                <CompactMarkdown text={item.finalReport} />
              </div>
            </div>
          )}
        </div>
      </MmDisclosure>
    </section>
  )
}

function MonitorList() {
  const { t } = useI18n()
  const mo = t.multimodal.monitor
  const monitors = useStore($mmMonitors)
  return (
    <section className="rounded-lg border border-(--ui-stroke-secondary) border-l-2 border-l-(--ui-yellow) bg-(--ui-chat-surface-background) p-2.5 text-xs">
      <div className="mb-1.5 flex items-center gap-1.5 font-medium text-(--ui-yellow)">
        👁 {mo.title} <span className="text-[10px] text-(--ui-text-tertiary)">{monitors.length}</span>
      </div>
      <ul className="flex flex-col gap-1.5">
        {monitors.map(m => {
          // Legacy rows predate trigger_mode and behaved continuously.
          const modeLabel = m.trigger_mode === 'once' ? mo.once : mo.continuous
          const done = m.status === 'done' || m.status === 'complete'
          const deleted = m.status === 'deleted'
          const running = !done && !deleted && m.status !== 'interrupted' && m.enabled !== false
          const canToggle = !done && !deleted
          const stateLabel = done ? mo.completed : running ? mo.inProgress : mo.interrupted
          return (
            <li key={m.monitor_id} className="flex items-center justify-between gap-2">
              {/* ★ 与 web 对齐: 单行 label · 状态 · #id (点分隔)。label 可截断,
                  状态/id shrink-0 不被挤掉。 */}
              <div className="flex min-w-0 items-baseline gap-1">
                <span className="truncate text-(--ui-text-primary)">{m.label || m.monitor_query || m.brief || mo.fallbackLabel}</span>
                <span className="shrink-0 rounded border border-current/20 px-1 text-[9px] text-(--ui-text-tertiary)">{modeLabel}</span>
                <span className="shrink-0 text-[10px] text-(--ui-text-tertiary)">· {stateLabel}</span>
                <span className="shrink-0 font-mono text-[10px] text-(--ui-text-tertiary)">· #{m.monitor_id}</span>
              </div>
              <Switch
                checked={running}
                disabled={!canToggle}
                title={done ? mo.completedOnceHint : undefined}
                onCheckedChange={v => canToggle && void toggleMonitor(m.monitor_id, v)}
              />
            </li>
          )
        })}
      </ul>
    </section>
  )
}

function WatcherList({ watchers }: { watchers: MmWatcher[] }) {
  const { t } = useI18n()
  const w = t.multimodal.watcher
  return (
    <section className="rounded-lg border border-(--ui-stroke-secondary) border-l-2 border-l-(--ui-purple) bg-(--ui-chat-surface-background) p-2.5 text-xs">
      <div className="mb-1.5 flex items-center gap-1.5 font-medium text-(--ui-purple)">
        🔬 {w.title} <span className="text-[10px] text-(--ui-text-tertiary)">{watchers.length}</span>
      </div>
      <ul className="flex flex-col gap-1.5">
        {watchers.map(r => {
          const { canToggle, running, state } = watcherPresentation(r)
          const stateLabel = watcherStateLabel(t, state)
          const label = r.label || r.task_instruction || w.fallbackLabel
          return (
            <li className="flex items-center justify-between gap-2" key={r.watcher_id}>
              {/* ★ 与 web 对齐: 单行 label · 状态 · #id (点分隔)。 */}
              <div className="flex min-w-0 items-baseline gap-1">
                <span className="truncate text-(--ui-text-primary)">{label}</span>
                <span className="shrink-0 text-[10px] text-(--ui-text-tertiary)">· {stateLabel}</span>
                <span className="shrink-0 font-mono text-[10px] text-(--ui-text-tertiary)">· #{r.watcher_id}</span>
              </div>
              <Switch
                aria-label={w.switchLabel(label, stateLabel)}
                checked={running}
                disabled={!canToggle}
                onCheckedChange={v => canToggle && void toggleWatcher(r.watcher_id, v)}
                title={
                  state === 'done'
                    ? w.completedHint
                    : state === 'stopping'
                      ? w.stoppingHint
                      : undefined
                }
              />
            </li>
          )
        })}
      </ul>
    </section>
  )
}

// Newest N alerts shown per monitor by default; the rest fold behind "展开更多".
const MONITOR_ALERTS_VISIBLE = 2

function fmtClock(ms?: number): string {
  if (!ms) return ''
  const d = new Date(ms)
  const h = String(d.getHours()).padStart(2, '0')
  const m = String(d.getMinutes()).padStart(2, '0')
  const s = String(d.getSeconds()).padStart(2, '0')
  return `${h}:${m}:${s}`
}

function fmtFrameTs(ts: number): string {
  const sec = Math.max(0, Math.round(ts))
  return `${String(Math.floor(sec / 60)).padStart(2, '0')}:${String(sec % 60).padStart(2, '0')}`
}

/** Per-monitor proactive-alert list. Each monitor gets its own card; alerts
 *  stream in via message.* handlers (curMonitorAlertId → $mmMonitorAlerts) and
 *  hydrate on session enter via list_monitor_alerts. Default: newest 2 alerts
 *  visible; older ones fold behind an expander so a chatty monitor doesn't
 *  swallow the right rail. */
function MonitorAlertsPanel() {
  const { t } = useI18n()
  const mo = t.multimodal.monitor
  const alertsByMonitor = useStore($mmMonitorAlerts)
  const monitors = useStore($mmMonitors)
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set())
  const toggle = (mid: string) => {
    setExpanded(prev => {
      const next = new Set(prev)
      if (next.has(mid)) next.delete(mid)
      else next.add(mid)
      return next
    })
  }
  const monitorLabel = (mid: string): string => {
    const m = monitors.find(x => x.monitor_id === mid)
    return (m?.label || m?.monitor_query || m?.brief || mo.fallbackLabel).slice(0, 40)
  }
  const entries = Object.entries(alertsByMonitor).filter(([, list]) => list && list.length > 0)
  if (entries.length === 0) return null
  return (
    <section className="rounded-lg border border-(--ui-stroke-secondary) border-l-2 border-l-(--ui-yellow) bg-(--ui-chat-surface-background) p-2.5 text-xs">
      <div className="mb-1.5 flex items-center gap-1.5 font-medium text-(--ui-yellow)">
        👁 {mo.alertsTitle} <span className="text-[10px] text-(--ui-text-tertiary)">{entries.length}</span>
      </div>
      <div className="flex flex-col gap-2">
        {entries.map(([mid, list]) => {
          const isExpanded = expanded.has(mid)
          const hiddenCount = Math.max(0, list.length - MONITOR_ALERTS_VISIBLE)
          const shown = isExpanded ? list : list.slice(-MONITOR_ALERTS_VISIBLE)
          return (
            <div key={mid} className="flex flex-col gap-1">
              <div className="flex items-baseline gap-1 text-[10px] text-(--ui-text-tertiary)">
                <span className="truncate text-(--ui-text-secondary)">{monitorLabel(mid)}</span>
                <span className="shrink-0 font-mono">· #{mid}</span>
                {hiddenCount > 0 && (
                  <button
                    onClick={() => toggle(mid)}
                    className="ml-auto shrink-0 rounded px-1.5 py-px hover:text-(--ui-yellow)"
                  >
                    {isExpanded ? mo.collapse : mo.expandMore(hiddenCount)}
                  </button>
                )}
              </div>
              <ul className={`flex flex-col gap-1 ${isExpanded ? 'max-h-56 overflow-y-auto' : ''}`}>
                {shown.map((a: MonitorAlert) => (
                  <li
                    key={a.id}
                    className="rounded border-l-2 border-(--ui-yellow) bg-(--ui-bg-elevated) px-2 py-1"
                  >
                    <div className="mb-0.5 flex items-center gap-1.5 text-[10px] text-(--ui-text-tertiary)">
                      <span className="tabular-nums">{fmtClock(a.ts)}</span>
                      {a.streaming && <span className="animate-pulse">…</span>}
                    </div>
                    <div className="whitespace-pre-wrap break-words text-[11px] leading-relaxed text-(--ui-text-primary)">
                      {a.text || (a.streaming ? '…' : '')}
                    </div>
                    {a.evidence && (
                      <div className="mt-1.5">
                        <div className="mb-1 text-[9px] text-(--ui-text-tertiary)">
                          {mo.evidenceSummary(a.evidence.input_count, a.evidence.frames.length)}
                        </div>
                        <div className="grid grid-cols-3 gap-1">
                          {a.evidence.frames.map((frame, index) => {
                            const src = `data:image/jpeg;base64,${frame.thumb_b64}`
                            return (
                              <button
                                className="group relative overflow-hidden rounded border border-(--ui-stroke-secondary) bg-black"
                                key={`${frame.ts}_${index}`}
                                onClick={() => window.open(src, '_blank')}
                                title={`${fmtFrameTs(frame.ts)}${frame.source_type ? ` · ${frame.source_type}` : ''}`}
                                type="button"
                              >
                                <img
                                  alt={mo.evidenceFrame(index + 1)}
                                  className="aspect-video w-full object-cover"
                                  src={src}
                                />
                                <span className="absolute bottom-0 right-0 bg-black/70 px-1 py-px font-mono text-[8px] text-white">
                                  {fmtFrameTs(frame.ts)}
                                </span>
                              </button>
                            )
                          })}
                        </div>
                      </div>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          )
        })}
      </div>
    </section>
  )
}
