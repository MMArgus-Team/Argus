import { useStore } from '@nanostores/react'
import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from '@/components/ui/sheet'
import { useI18n } from '@/i18n'
import {
  getMultimodalMemoryDebugFrame,
  getMultimodalMemoryDebugSession,
  getMultimodalMemoryDebugSessions,
  getMultimodalMemoryDebugTrace,
  type MmMemoryDebugEvent,
  type MmMemoryDebugFrameResponse,
  type MmMemoryDebugSearchResult,
  type MmMemoryDebugSessionResponse,
  type MmMemoryDebugSessionSummary,
  type MmMemoryDebugTablePayload,
  type MmMemoryDebugTraceResponse,
  searchMultimodalMemoryDebug
} from '@/hermes'
import { Activity, Brain, Bug, FileImage, RefreshCw, Search } from '@/lib/icons'
import { compactQueryWorkerTrajectoryImages } from '@/lib/query-worker-trajectory-cache'
import { cn } from '@/lib/utils'
import { $gateway } from '@/store/gateway'

import {
  publicQueryWorkerTracePayload,
  QueryWorkerTrajectoryPanel
} from './query-worker-trajectory-panel'
import {
  groupTrajectoryByQuestion,
  type MmTrajectoryEntry,
  type MmTrajectoryFrame,
  type MmTrajectoryQuestionGroup,
  type MmTrajectoryWorkerGroup
} from './trajectory-grouping'

export type { MmTrajectoryEntry, MmTrajectoryFrame } from './trajectory-grouping'

type MemoryDebugTab = 'memory' | 'frame' | 'search' | 'debug'
type SearchScope = 'latest' | 'today' | 'all'

interface MemoryDebugPanelProps {
  durableSessionIds: string[]
  liveSessionId: string
  onOpenChange: (open: boolean) => void
  open: boolean
}

interface MemoryDebugPanelScopeProps extends MemoryDebugPanelProps {
  gateway: ReturnType<typeof $gateway.get>
}

interface MemoryDebugSessionIdentity {
  _lineage_root_id?: null | string
  id: string
}

const TRAJECTORY_MAX = 2000
const TRAJECTORY_WORKER_PREVIEW = 30
const gatewayScopeIds = new WeakMap<object, number>()
let nextGatewayScopeId = 1

function gatewayScopeKey(gateway: ReturnType<typeof $gateway.get>): string {
  if ((typeof gateway !== 'object' || gateway === null) && typeof gateway !== 'function') {
    return `primitive:${String(gateway)}`
  }

  const objectGateway = gateway as object
  let scopeId = gatewayScopeIds.get(objectGateway)

  if (scopeId === undefined) {
    scopeId = nextGatewayScopeId
    nextGatewayScopeId += 1
    gatewayScopeIds.set(objectGateway, scopeId)
  }

  return `gateway:${scopeId}`
}

export function resolveMemoryDebugSessionIds(
  selectedStoredSessionId: null | string,
  sessions: MemoryDebugSessionIdentity[],
  liveSessionId: string
): string[] {
  const selectedContinuation = sessions.find(
    session =>
      session.id !== selectedStoredSessionId && session._lineage_root_id === selectedStoredSessionId
  )

  const selectedExact = sessions.find(session => session.id === selectedStoredSessionId)

  const selectedSession = selectedContinuation || selectedExact

  return [
    selectedSession?.id || '',
    selectedStoredSessionId || '',
    selectedSession?._lineage_root_id || '',
    !selectedStoredSessionId ? liveSessionId : ''
  ].filter((value, index, list): value is string => Boolean(value) && list.indexOf(value) === index)
}

function sessionMatches(summary: MmMemoryDebugSessionSummary, candidate: string): boolean {
  const expected = candidate.trim()

  if (!expected) {
    return false
  }

  const actual = String(summary.meta?.hermes_session_id || '').trim()

  return actual === expected
}

function isValidTrajectoryEntry(value: unknown): value is MmTrajectoryEntry {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return false
  }

  const entry = value as Partial<MmTrajectoryEntry>

  return (
    typeof entry.id === 'string' &&
    typeof entry.seq === 'number' &&
    typeof entry.ts === 'number' &&
    typeof entry.event === 'string' &&
    typeof entry.worker === 'string' &&
    typeof entry.phase === 'string' &&
    Boolean(entry.payload) &&
    typeof entry.payload === 'object' &&
    !Array.isArray(entry.payload)
  )
}

/** Resolve only a DB owned by the active durable conversation.
 *
 * Deliberately returns an empty string on a miss. Showing the newest unrelated
 * DB under a "current memory" heading is much worse than an honest empty state.
 */
export function resolveMemoryDebugDb(sessions: MmMemoryDebugSessionSummary[], durableSessionIds: string[]): string {
  for (const candidate of durableSessionIds) {
    const match = sessions.find(session => sessionMatches(session, candidate))

    if (match) {
      return match.name
    }
  }

  return ''
}

export function mergeMemoryDebugTrajectory(
  current: MmTrajectoryEntry[],
  incoming: MmTrajectoryEntry[],
  cap = TRAJECTORY_MAX,
  preferIncoming = true
): MmTrajectoryEntry[] {
  const byId = new Map<string, MmTrajectoryEntry>(
    current.map(entry => [entry.id || `seq:${entry.seq}`, entry])
  )

  for (const entry of incoming) {
    const key = entry.id || `seq:${entry.seq}`
    const previous = byId.get(key)

    const newer = previous && (
      entry.seq > previous.seq ||
      (entry.seq === previous.seq && entry.ts > previous.ts)
    )

    const sameVersion = previous && entry.seq === previous.seq && entry.ts === previous.ts

    if (!previous || newer || (preferIncoming && sameVersion)) {
      byId.set(key, entry)
    }
  }

  return compactQueryWorkerTrajectoryImages(
    [...byId.values()].sort((a, b) => a.seq - b.seq || a.ts - b.ts).slice(-Math.max(1, cap))
  )
}

function fmtRelativeTime(seconds?: number): string {
  if (seconds == null || !Number.isFinite(seconds)) {
    return ''
  }

  const minutes = Math.floor(seconds / 60)
  const rest = Math.floor(seconds % 60)

  return `${String(minutes).padStart(2, '0')}:${String(rest).padStart(2, '0')}`
}

function fmtWallTime(seconds?: number): string {
  if (!seconds) {
    return ''
  }

  return new Date(seconds * 1000).toLocaleString()
}

function fmtBytes(bytes?: number): string {
  const value = Number(bytes || 0)

  if (value < 1024) {
    return `${value} B`
  }

  if (value < 1024 * 1024) {
    return `${(value / 1024).toFixed(1)} KB`
  }

  return `${(value / 1024 / 1024).toFixed(1)} MB`
}

function debugJson(value: unknown): string {
  if (typeof value === 'string') {
    return value
  }

  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return String(value)
  }
}

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function extractBox(block: unknown): null | number[] {
  if (!block || typeof block !== 'object') {
    return null
  }

  const item = block as Record<string, unknown>
  const raw = item.bbox || item.box || item.rect || item.points || item.polygon || item.poly

  if (!Array.isArray(raw)) {
    return null
  }

  if (raw.length >= 4 && raw.every(value => typeof value === 'number')) {
    const values = raw.slice(0, 4) as number[]

    if (values[2] > values[0] && values[3] > values[1]) {
      return values
    }

    return [values[0], values[1], values[0] + Math.max(0, values[2]), values[1] + Math.max(0, values[3])]
  }

  const points = raw.filter(value => Array.isArray(value) && value.length >= 2) as unknown[][]
  const xs = points.map(point => Number(point[0])).filter(Number.isFinite)
  const ys = points.map(point => Number(point[1])).filter(Number.isFinite)

  if (xs.length < 2 || ys.length < 2) {
    return null
  }

  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)]
}

const OcrOverlayImage = memo(function OcrOverlayImage({ blocks, imageB64 }: { blocks: unknown[]; imageB64: string }) {
  const { t } = useI18n()
  const [size, setSize] = useState<null | { height: number; width: number }>(null)

  const boxes = useMemo(() => {
    const rawBoxes = blocks
      .map(block => ({ block, box: extractBox(block) }))
      .filter((item): item is { block: unknown; box: number[] } => Boolean(item.box))

    if (!size || rawBoxes.length === 0) {
      return []
    }

    const maxX = Math.max(...rawBoxes.map(item => Math.max(item.box[0], item.box[2])))
    const maxY = Math.max(...rawBoxes.map(item => Math.max(item.box[1], item.box[3])))
    const normalized = maxX <= 1.5 && maxY <= 1.5

    return rawBoxes.map(({ block, box }) => ({
      block,
      height: normalized ? (box[3] - box[1]) * 100 : ((box[3] - box[1]) / Math.max(maxY, size.height)) * 100,
      left: normalized ? box[0] * 100 : (box[0] / Math.max(maxX, size.width)) * 100,
      top: normalized ? box[1] * 100 : (box[1] / Math.max(maxY, size.height)) * 100,
      width: normalized ? (box[2] - box[0]) * 100 : ((box[2] - box[0]) / Math.max(maxX, size.width)) * 100
    }))
  }, [blocks, size])

  if (!imageB64) {
    return (
      <div className="grid aspect-video place-items-center rounded border border-(--ui-stroke-secondary) bg-black text-xs text-(--ui-text-tertiary)">
        {t.multimodal.memoryDebug.noFrameImage}
      </div>
    )
  }

  return (
    <div className="relative overflow-hidden rounded border border-(--ui-stroke-secondary) bg-black">
      <img
        alt="memory frame"
        className="max-h-[48vh] w-full object-contain"
        onLoad={event =>
          setSize({ height: event.currentTarget.naturalHeight, width: event.currentTarget.naturalWidth })
        }
        src={`data:image/jpeg;base64,${imageB64}`}
      />
      {boxes.map((item, index) => (
        <div
          className="absolute border border-amber-300/90 bg-amber-300/10"
          key={index}
          style={{
            height: `${Math.max(0.4, Math.min(100, item.height))}%`,
            left: `${Math.max(0, Math.min(100, item.left))}%`,
            top: `${Math.max(0, Math.min(100, item.top))}%`,
            width: `${Math.max(0.4, Math.min(100, item.width))}%`
          }}
          title={String((item.block as Record<string, unknown>)?.text || '')}
        />
      ))}
    </div>
  )
})

function MemoryTable({ table }: { table: MmMemoryDebugTablePayload }) {
  const rows = table.rows.slice(0, 30)

  if (table.columns.length === 0) {
    return <pre className="max-h-56 overflow-auto rounded bg-black/20 p-2 text-[0.6875rem]">{debugJson(rows)}</pre>
  }

  return (
    <div className="max-h-64 overflow-auto rounded border border-(--ui-stroke-secondary)">
      <table className="w-full border-collapse text-[0.6875rem]">
        <thead className="sticky top-0 bg-(--ui-bg-elevated)">
          <tr>
            {table.columns.map(column => (
              <th className="border-b border-r border-(--ui-stroke-secondary) px-2 py-1 text-left" key={column}>
                {column}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => {
            const record = row && typeof row === 'object' && !Array.isArray(row) ? (row as Record<string, unknown>) : {}

            return (
              <tr key={index}>
                {table.columns.map(column => (
                  <td
                    className="max-w-56 border-r border-t border-(--ui-stroke-secondary) px-2 py-1 align-top"
                    key={column}
                  >
                    {String(record[column] ?? '')}
                  </td>
                ))}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function EventCard({
  event,
  level,
  onFrame
}: {
  event: MmMemoryDebugEvent
  level: 'macro' | 'micro' | 'super'
  onFrame: (frameId: string) => void
}) {
  const { t } = useI18n()
  const title = event.label || event.action || event.id
  const description = event.summary || event.description || t.multimodal.memoryDebug.noDescription
  const frames = event.frame_ids || []
  const entities = event.entity_names || event.key_entities || []

  return (
    <details
      className="rounded border border-(--ui-stroke-secondary) bg-(--ui-bg-elevated) p-2 text-xs"
      open={level !== 'micro'}
    >
      <summary className="cursor-pointer list-none">
        <span
          className={cn(
            'mr-2 rounded px-1.5 py-0.5 font-mono text-[0.625rem]',
            level === 'super' && 'bg-violet-500/15 text-violet-300',
            level === 'macro' && 'bg-amber-500/15 text-amber-400',
            level === 'micro' && 'bg-cyan-500/15 text-cyan-400'
          )}
        >
          {level}
        </span>
        <span className="font-semibold">{title}</span>
        <span className="ml-2 font-mono text-(--ui-text-tertiary)">
          {fmtRelativeTime(event.t_start)}–{fmtRelativeTime(event.t_end)}
        </span>
      </summary>
      <div className="mt-2 whitespace-pre-wrap text-(--ui-text-secondary)">{description}</div>
      {(entities.length > 0 || frames.length > 0) && (
        <div className="mt-2 flex flex-wrap gap-1">
          {entities.map(entity => (
            <span className="rounded bg-(--ui-bg-quaternary) px-1.5 py-0.5" key={entity}>
              entity: {entity}
            </span>
          ))}
          {frames.map(frameId => (
            <button
              className="rounded border border-emerald-400/30 px-1.5 py-0.5 font-mono text-emerald-400"
              key={frameId}
              onClick={() => onFrame(frameId)}
              type="button"
            >
              {frameId}
            </button>
          ))}
        </div>
      )}
    </details>
  )
}

function EmptyState({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded border border-(--ui-stroke-secondary) p-3 text-xs italic text-(--ui-text-tertiary)">
      {children}
    </div>
  )
}

function TrajectoryEntryCard({ entry }: { entry: MmTrajectoryEntry }) {
  const rawFrames = entry.payload.frames
  const frames = Array.isArray(rawFrames) ? (rawFrames as MmTrajectoryFrame[]) : []

  return (
    <details
      className="rounded border border-(--ui-stroke-secondary) bg-(--ui-bg-elevated) p-2 text-xs"
      open={frames.length > 0}
    >
      <summary className="cursor-pointer list-none">
        <span className="font-mono text-cyan-400">#{entry.seq}</span>
        <span className="ml-2 rounded bg-(--ui-bg-quaternary) px-1.5 py-0.5 font-mono">{entry.phase}</span>
        <span className="ml-2 text-(--ui-text-tertiary)">
          {fmtWallTime(entry.ts)} · {entry.event}
        </span>
      </summary>
      {frames.length > 0 && (
        <div className="mt-2 grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4">
          {frames.map((item, index) => {
            const b64 = item.thumb_b64 || item.jpeg_b64 || ''
            const usable = b64 && !b64.startsWith('<omitted')

            return (
              <figure
                className="overflow-hidden rounded border border-(--ui-stroke-secondary) bg-black/20"
                key={`${item.frame_id || item.ts || index}-${index}`}
              >
                {usable ? (
                  <img
                    alt="trajectory evidence"
                    className="h-28 w-full object-contain"
                    src={`data:image/jpeg;base64,${b64}`}
                  />
                ) : (
                  <div className="grid h-28 place-items-center text-[0.625rem]">thumbnail omitted</div>
                )}
                <figcaption className="px-1.5 py-1 font-mono text-[0.625rem] text-(--ui-text-tertiary)">
                  {item.frame_id || `frame ${index + 1}`} · {fmtRelativeTime(item.ts)}
                </figcaption>
              </figure>
            )
          })}
        </div>
      )}
      <pre className="mt-2 max-h-96 overflow-auto whitespace-pre-wrap rounded bg-black/20 p-2 text-[0.6875rem]">
        {debugJson(publicQueryWorkerTracePayload(entry.payload))}
      </pre>
    </details>
  )
}

function WorkerTrajectory({ group }: { group: MmTrajectoryWorkerGroup }) {
  const { t } = useI18n()
  const [showAll, setShowAll] = useState(false)
  const newestFirst = [...group.entries].reverse()
  const visibleEntries = showAll ? newestFirst : newestFirst.slice(0, TRAJECTORY_WORKER_PREVIEW)

  return (
    <section className="space-y-1.5 rounded border border-(--ui-stroke-secondary) bg-black/10 p-2">
      <div className="flex items-center gap-2 text-xs">
        <span className="font-semibold text-emerald-400">{group.worker}</span>
        <span className="text-(--ui-text-tertiary)">
          {group.entries.length} events · #{group.firstSeq}–#{group.lastSeq}
        </span>
      </div>
      {visibleEntries.map(entry => (
        <TrajectoryEntryCard entry={entry} key={entry.id || `seq:${entry.seq}`} />
      ))}
      {group.entries.length > TRAJECTORY_WORKER_PREVIEW && (
        <Button onClick={() => setShowAll(value => !value)} size="xs" variant="ghost">
          {showAll ? t.multimodal.memoryDebug.showEarlier : t.multimodal.memoryDebug.showAllEvents(group.entries.length)}
        </Button>
      )}
    </section>
  )
}

function TrajectoryQuestion({
  initiallyOpen,
  question
}: {
  initiallyOpen: boolean
  question: MmTrajectoryQuestionGroup
}) {
  const [expanded, setExpanded] = useState(initiallyOpen)

  return (
    <details
      className="rounded border border-cyan-400/25 bg-cyan-400/5 p-2"
      onToggle={event => setExpanded(event.currentTarget.open)}
      open={expanded}
    >
      <summary className="cursor-pointer list-none text-xs">
        <span className="font-semibold text-cyan-300">{question.label}</span>
        <span className="ml-2 font-mono text-[0.625rem] text-(--ui-text-tertiary)">#{question.id}</span>
        <span className="ml-2 text-(--ui-text-tertiary)">
          {question.workers.length} workers ·{' '}
          {question.workers.reduce((total, group) => total + group.entries.length, 0)} events ·{' '}
          {fmtWallTime(question.lastTs)}
        </span>
      </summary>
      {expanded && (
        <div className="mt-2 space-y-2">
          <QueryWorkerTrajectoryPanel entries={question.entries} />
          {question.workers.map(group => (
            <WorkerTrajectory group={group} key={group.worker} />
          ))}
        </div>
      )}
    </details>
  )
}

function BackgroundTrajectory({ groups }: { groups: MmTrajectoryWorkerGroup[] }) {
  const { t } = useI18n()
  const [expanded, setExpanded] = useState(false)

  return (
    <details
      className="rounded border border-amber-400/25 bg-amber-400/5 p-2"
      onToggle={event => setExpanded(event.currentTarget.open)}
      open={expanded}
    >
      <summary className="cursor-pointer list-none text-xs">
        <span className="font-semibold text-amber-300">{t.multimodal.memoryDebug.backgroundTrajectory}</span>
        <span className="ml-2 text-(--ui-text-tertiary)">
          {groups.length} workers · {groups.reduce((total, group) => total + group.entries.length, 0)} events
        </span>
      </summary>
      {expanded && (
        <div className="mt-2 space-y-2">
          {groups.map(group => (
            <WorkerTrajectory group={group} key={group.worker} />
          ))}
        </div>
      )}
    </details>
  )
}

export function MemoryDebugPanel(props: MemoryDebugPanelProps) {
  const gateway = useStore($gateway)
  const scopeKey = `${gatewayScopeKey(gateway)}:${JSON.stringify([props.durableSessionIds, props.liveSessionId])}`

  // Session/profile switches are privacy boundaries. A keyed child is mounted
  // with empty state in the same render that changes the scope, so content from
  // the previous scope cannot survive for one commit while passive effects run.
  return <MemoryDebugPanelScope {...props} gateway={gateway} key={scopeKey} />
}

function MemoryDebugPanelScope({
  durableSessionIds,
  gateway,
  liveSessionId,
  onOpenChange,
  open
}: MemoryDebugPanelScopeProps) {
  const { t } = useI18n()
  const md = t.multimodal.memoryDebug
  const [tab, setTab] = useState<MemoryDebugTab>('memory')
  const [sessions, setSessions] = useState<MmMemoryDebugSessionSummary[]>([])
  const [selectedDb, setSelectedDb] = useState('')
  const [overview, setOverview] = useState<MmMemoryDebugSessionResponse | null>(null)
  const [trace, setTrace] = useState<MmMemoryDebugTraceResponse | null>(null)
  const [frame, setFrame] = useState<MmMemoryDebugFrameResponse | null>(null)
  const [selectedFrameId, setSelectedFrameId] = useState('')
  const [searchQuery, setSearchQuery] = useState('')
  const [searchScope, setSearchScope] = useState<SearchScope>('latest')
  const [searchResults, setSearchResults] = useState<MmMemoryDebugSearchResult[]>([])
  const [trajectory, setTrajectory] = useState<MmTrajectoryEntry[]>([])
  const [workerFilter, setWorkerFilter] = useState('all')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const epochRef = useRef(0)
  const searchRequestRef = useRef(0)
  const manualSelectionRef = useRef(false)
  const durableKey = durableSessionIds.join('|')
  const gatewayKey = gateway
  const selectedDbRef = useRef(selectedDb)
  const selectedFrameIdRef = useRef(selectedFrameId)
  selectedDbRef.current = selectedDb
  selectedFrameIdRef.current = selectedFrameId

  const selectDatabase = useCallback((nextDb: string, manual = false) => {
    if (manual) {
      manualSelectionRef.current = true
    }

    if (nextDb === selectedDbRef.current) {
      return
    }

    // A latest-scope search belongs to the database selected when it started.
    // Invalidate it synchronously with the selection event so neither existing
    // results nor a late A response can render underneath database B.
    selectedDbRef.current = nextDb
    searchRequestRef.current += 1
    setSearchResults([])
    setSelectedDb(nextDb)
  }, [])

  const isCurrent = useCallback((epoch: number) => epochRef.current === epoch, [])

  useEffect(() => {
    epochRef.current += 1
    searchRequestRef.current += 1
    manualSelectionRef.current = false
    selectedDbRef.current = ''
    setSessions([])
    setSelectedDb('')
    setOverview(null)
    setTrace(null)
    setFrame(null)
    setSelectedFrameId('')
    setSearchResults([])
    setTrajectory([])
    setLoading(false)
    setError('')
  }, [durableKey, gateway, liveSessionId, open])

  const loadSessions = useCallback(async () => {
    void gatewayKey

    if (!open) {
      return
    }

    const epoch = epochRef.current
    setLoading(true)
    setError('')

    try {
      const response = await getMultimodalMemoryDebugSessions(80)

      if (!isCurrent(epoch)) {
        return
      }

      setSessions(response.sessions)
      const previous = selectedDbRef.current
      const exists = response.sessions.some(session => session.name === previous)

      const nextDb =
        manualSelectionRef.current && exists
          ? previous
          : resolveMemoryDebugDb(response.sessions, durableSessionIds)

      selectDatabase(nextDb)
    } catch (reason) {
      if (isCurrent(epoch)) {
        setError(errorText(reason))
      }
    } finally {
      if (isCurrent(epoch)) {
        setLoading(false)
      }
    }
  }, [durableSessionIds, gatewayKey, isCurrent, open, selectDatabase])

  useEffect(() => {
    if (!open) {
      return
    }

    void loadSessions()
  }, [loadSessions, open])

  const selectedSummary = useMemo(
    () => sessions.find(session => session.name === selectedDb) || null,
    [selectedDb, sessions]
  )

  const selectedDurableId = String(selectedSummary?.meta?.hermes_session_id || durableSessionIds[0] || '')

  const isOtherSession = Boolean(
    selectedSummary && !durableSessionIds.some(candidate => sessionMatches(selectedSummary, candidate))
  )

  const loadOverview = useCallback(async () => {
    if (!open || !selectedDb) {
      return
    }

    const epoch = epochRef.current
    const db = selectedDb
    setLoading(true)
    setError('')

    try {
      const response = await getMultimodalMemoryDebugSession(db, {
        limit: 260,
        session_id: selectedDurableId
      })

      if (!isCurrent(epoch) || db !== selectedDbRef.current) {
        return
      }

      setOverview(response)
      setSelectedFrameId(previous => previous || response.timeline.at(-1)?.frame_id || '')
    } catch (reason) {
      if (isCurrent(epoch) && db === selectedDbRef.current) {
        setError(errorText(reason))
      }
    } finally {
      if (isCurrent(epoch)) {
        setLoading(false)
      }
    }
  }, [isCurrent, open, selectedDb, selectedDurableId])

  const loadTrace = useCallback(async () => {
    if (!open || tab !== 'debug' || !selectedDb) {
      return
    }

    const epoch = epochRef.current
    const db = selectedDb

    try {
      const response = await getMultimodalMemoryDebugTrace({
        db,
        limit: 160,
        session_id: selectedDurableId
      })

      if (isCurrent(epoch) && db === selectedDbRef.current) {
        setTrace(response)
      }
    } catch (reason) {
      if (isCurrent(epoch) && db === selectedDbRef.current) {
        setError(errorText(reason))
      }
    }
  }, [isCurrent, open, selectedDb, selectedDurableId, tab])

  useEffect(() => {
    setOverview(null)
    setTrace(null)
    setFrame(null)
    setSelectedFrameId('')
    setTrajectory([])
    setWorkerFilter('all')

    if (selectedDb) {
      void loadOverview()
    }
  }, [loadOverview, selectedDb])

  useEffect(() => {
    if (tab === 'debug') {
      void loadTrace()
    }
  }, [loadTrace, tab])

  useEffect(() => {
    if (!open || tab !== 'frame' || !selectedDb || !selectedFrameId) {
      return
    }

    const epoch = epochRef.current
    const db = selectedDb
    const frameId = selectedFrameId
    setFrame(null)
    getMultimodalMemoryDebugFrame(db, frameId)
      .then(response => {
        if (isCurrent(epoch) && db === selectedDbRef.current && frameId === selectedFrameIdRef.current) {
          setFrame(response)
        }
      })
      .catch(reason => {
        if (isCurrent(epoch) && db === selectedDbRef.current && frameId === selectedFrameIdRef.current) {
          setError(errorText(reason))
        }
      })
  }, [isCurrent, open, selectedDb, selectedFrameId, tab])

  const loadTrajectory = useCallback(async () => {
    if (!open || tab !== 'debug' || !gateway || !liveSessionId || isOtherSession) {
      return
    }

    const epoch = epochRef.current
    const sid = liveSessionId

    try {
      const response = await gateway.request<{ count: number; entries: MmTrajectoryEntry[] }>(
        'multimodal.trajectory.list',
        { limit: TRAJECTORY_MAX, session_id: sid },
        60_000
      )

      if (!isCurrent(epoch) || sid !== liveSessionId) {
        return
      }

      setTrajectory(current => mergeMemoryDebugTrajectory(current, response.entries || [], TRAJECTORY_MAX, false))
    } catch (reason) {
      if (isCurrent(epoch) && sid === liveSessionId) {
        setError(errorText(reason))
      }
    }
  }, [gateway, isCurrent, isOtherSession, liveSessionId, open, tab])

  useEffect(() => {
    if (!open || tab !== 'debug' || !gateway || !liveSessionId || isOtherSession) {
      return
    }

    const sid = liveSessionId

    const off = gateway.on<MmTrajectoryEntry>('multimodal.trajectory', event => {
      const entry = event.payload

      if (event.session_id !== sid || !isValidTrajectoryEntry(entry)) {
        return
      }

      setTrajectory(current => mergeMemoryDebugTrajectory(current, [entry]))
    })

    void loadTrajectory()

    return off
  }, [gateway, isOtherSession, liveSessionId, loadTrajectory, open, tab])

  const runSearch = useCallback(async () => {
    const query = searchQuery.trim()

    if (
      query.replace(/\s+/g, '').length < 2 ||
      (searchScope === 'latest' && !selectedDb)
    ) {
      return
    }

    const epoch = epochRef.current
    const request = searchRequestRef.current + 1
    searchRequestRef.current = request
    setLoading(true)
    setError('')

    try {
      const response = await searchMultimodalMemoryDebug(query, {
        limit: 50,
        scope: searchScope,
        session: searchScope === 'latest' ? selectedDb : undefined
      })

      if (isCurrent(epoch) && request === searchRequestRef.current) {
        setSearchResults(response.results)
      }
    } catch (reason) {
      if (isCurrent(epoch) && request === searchRequestRef.current) {
        setError(errorText(reason))
      }
    } finally {
      if (isCurrent(epoch) && request === searchRequestRef.current) {
        setLoading(false)
      }
    }
  }, [isCurrent, searchQuery, searchScope, selectedDb])

  const selectFrame = useCallback((frameId: string) => {
    setSelectedFrameId(frameId)
    setTab('frame')
  }, [])

  const counts = overview?.session.counts || {}
  const memory = overview?.memory
  const entities = memory?.entities || []
  const microEvents = memory?.events.micro || []
  const macroEvents = memory?.events.macro || []
  const superEvents = memory?.events.super || []
  const entityStates = memory?.evolution.entity_states || []
  const revisions = memory?.evolution.revisions || []
  const eventCount = microEvents.length + macroEvents.length + superEvents.length
  const evolutionCount = entityStates.length + revisions.length
  const health = overview?.health || {}
  const activeFrame = frame?.frame_id === selectedFrameId ? frame : null
  const blocks = activeFrame?.screen_text?.ocr_blocks || []

  const workers = useMemo(
    () => [...new Set(trajectory.map(entry => entry.worker).filter(Boolean))].sort(),
    [trajectory]
  )

  const trajectoryGrouping = useMemo(() => groupTrajectoryByQuestion(trajectory), [trajectory])

  const visibleQuestions = useMemo(
    () =>
      trajectoryGrouping.questions
        .map(question => ({
          ...question,
          workers:
            workerFilter === 'all' ? question.workers : question.workers.filter(group => group.worker === workerFilter)
        }))
        .filter(question => question.workers.length > 0),
    [trajectoryGrouping.questions, workerFilter]
  )

  const visibleBackground = useMemo(
    () =>
      workerFilter === 'all'
        ? trajectoryGrouping.background
        : trajectoryGrouping.background.filter(group => group.worker === workerFilter),
    [trajectoryGrouping.background, workerFilter]
  )

  const visibleTrajectoryCount = useMemo(
    () =>
      visibleQuestions.reduce(
        (total, question) =>
          total + question.workers.reduce((questionTotal, group) => questionTotal + group.entries.length, 0),
        0
      ) + visibleBackground.reduce((total, group) => total + group.entries.length, 0),
    [visibleBackground, visibleQuestions]
  )

  const refresh = () => {
    void loadSessions()
    void loadOverview()
    void loadTrace()
    void loadTrajectory()
  }

  return (
    <Sheet onOpenChange={onOpenChange} open={open}>
      <SheetContent className="w-[min(980px,96vw)] gap-0 p-0 sm:max-w-[min(980px,96vw)]" side="right">
        <SheetHeader className="border-b border-(--ui-stroke-secondary) pr-12">
          <div className="flex items-center gap-2">
            <Bug className="size-4 text-emerald-400" />
            <div className="min-w-0 flex-1">
              <SheetTitle>{md.panelTitle}</SheetTitle>
              <SheetDescription className="truncate">
                {md.panelSubtitle}
              </SheetDescription>
            </div>
            <Button
              aria-label={md.refreshLabel}
              disabled={loading}
              onClick={refresh}
              size="icon-xs"
              variant="outline"
            >
              <RefreshCw className={cn('size-3', loading && 'animate-spin')} />
            </Button>
          </div>
        </SheetHeader>

        <div className="flex flex-wrap items-center gap-2 border-b border-(--ui-stroke-secondary) p-3">
          <select
            aria-label={md.dbSelectLabel}
            className="min-w-0 flex-1 rounded border border-(--ui-stroke-secondary) bg-(--ui-bg-elevated) px-2 py-1.5 text-xs"
            onChange={event => {
              selectDatabase(event.target.value, true)
            }}
            value={selectedDb}
          >
            <option value="">{md.noDbOption}</option>
            {sessions.map(session => (
              <option key={session.name} value={session.name}>
                {session.name} · frames {session.counts.memory_frames || 0} · OCR {session.counts.screen_texts || 0}
              </option>
            ))}
          </select>
          {(['memory', 'frame', 'search', 'debug'] as MemoryDebugTab[]).map(value => (
            <Button key={value} onClick={() => setTab(value)} size="xs" variant={tab === value ? 'secondary' : 'ghost'}>
              {value === 'memory'
                ? md.tabMemory
                : value === 'frame'
                  ? md.tabFrame
                  : value === 'search'
                    ? md.tabSearch
                    : md.tabDebug}
            </Button>
          ))}
        </div>

        {isOtherSession && (
          <div className="border-b border-amber-400/30 bg-amber-500/10 px-3 py-2 text-xs text-amber-500">
            {md.otherSessionNotice}
          </div>
        )}
        {error && (
          <div className="border-b border-(--ui-red)/30 bg-(--ui-red)/10 px-3 py-2 text-xs text-(--ui-red)">
            {error}
          </div>
        )}

        <div className="min-h-0 flex-1 overflow-y-auto p-3" data-testid="memory-debug-content">
          {!selectedDb && tab !== 'search' && tab !== 'debug' && (
            <EmptyState>
              {md.noMemoryYet}
            </EmptyState>
          )}

          {tab === 'memory' && selectedDb && (
            <div className="space-y-5">
              <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
                {[
                  [md.statFrames, overview?.timeline.length || 0, counts.memory_frames || 0],
                  [md.statEntities, entities.length, counts.entities || 0],
                  [
                    md.statEvents,
                    eventCount,
                    (counts.micro_events || 0) + (counts.macro_events || 0) + (counts.super_events || 0)
                  ],
                  [md.statEvolution, evolutionCount, (counts.entity_states || 0) + (counts.revision_log || 0)]
                ].map(([label, shown, total]) => (
                  <div
                    className="rounded border border-(--ui-stroke-secondary) bg-(--ui-bg-elevated) p-3"
                    key={String(label)}
                  >
                    <div className="text-[0.6875rem] text-(--ui-text-tertiary)">{label}</div>
                    <div className="mt-1 font-mono text-xl text-emerald-400">{shown}</div>
                    {Number(total) > Number(shown) && (
                      <div className="text-[0.625rem] text-(--ui-text-tertiary)">{md.inDbTotal(total)}</div>
                    )}
                  </div>
                ))}
              </div>

              <section className="space-y-2">
                <div className="flex items-center justify-between gap-2">
                  <div>
                    <h3 className="text-sm font-semibold">{md.section1Title}</h3>
                    <p className="text-xs text-(--ui-text-tertiary)">{md.section1Hint}</p>
                  </div>
                  <Button onClick={() => setTab('frame')} size="xs" variant="outline">
                    <FileImage />
                    {md.viewAllAndOcr}
                  </Button>
                </div>
                {(overview?.timeline.length || 0) === 0 ? (
                  <EmptyState>{md.noFramesYet}</EmptyState>
                ) : (
                  <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6">
                    {(overview?.timeline || [])
                      .slice(-48)
                      .reverse()
                      .map(item => (
                        <button
                          className="overflow-hidden rounded border border-(--ui-stroke-secondary) bg-(--ui-bg-elevated) text-left hover:border-emerald-400/60"
                          key={item.frame_id}
                          onClick={() => selectFrame(item.frame_id)}
                          type="button"
                        >
                          {item.thumb_b64 ? (
                            <img
                              alt={item.frame_id}
                              className="h-24 w-full object-cover"
                              src={`data:image/jpeg;base64,${item.thumb_b64}`}
                            />
                          ) : (
                            <div className="grid h-24 place-items-center bg-black/30 text-[0.625rem]">no image</div>
                          )}
                          <div className="p-1.5">
                            <div className="truncate font-mono text-[0.625rem] text-emerald-400">
                              {fmtRelativeTime(item.t_observed)} · {item.frame_id}
                            </div>
                            <div className="truncate text-[0.625rem] text-(--ui-text-tertiary)">
                              {item.source_type || item.source || 'unknown'} ·{' '}
                              {item.note || item.micro_id || 'key frame'}
                            </div>
                          </div>
                        </button>
                      ))}
                  </div>
                )}
              </section>

              <section className="space-y-2">
                <div>
                  <h3 className="text-sm font-semibold">{md.section2Title}</h3>
                  <p className="text-xs text-(--ui-text-tertiary)">{md.section2Hint}</p>
                </div>
                {entities.length === 0 ? (
                  <EmptyState>{md.noEntities}</EmptyState>
                ) : (
                  <div className="grid grid-cols-1 gap-2 lg:grid-cols-2">
                    {entities.map(entity => (
                      <details
                        className="rounded border border-(--ui-stroke-secondary) bg-(--ui-bg-elevated) p-3 text-xs"
                        key={entity.id}
                        open={entities.length <= 8}
                      >
                        <summary className="cursor-pointer list-none">
                          <span className="rounded bg-emerald-500/15 px-1.5 py-0.5 font-mono text-[0.625rem] text-emerald-400">
                            {entity.type}
                          </span>
                          <span className="ml-2 font-semibold">{entity.name}</span>
                          <span className="ml-2 text-(--ui-text-tertiary)">{md.seenCount(entity.seen_count)}</span>
                        </summary>
                        {entity.aliases.length > 0 && (
                          <div className="mt-2 text-(--ui-text-tertiary)">{md.aliases(entity.aliases.join(' / '))}</div>
                        )}
                        <div className="mt-2 flex flex-wrap gap-1">
                          {Object.entries(entity.attributes || {}).map(([key, value]) => (
                            <span className="rounded bg-(--ui-bg-quaternary) px-1.5 py-0.5" key={key}>
                              <span className="text-(--ui-text-tertiary)">{key}:</span> {debugJson(value)}
                            </span>
                          ))}
                        </div>
                        {entity.frame_ids.length > 0 && (
                          <div className="mt-2 flex flex-wrap gap-1">
                            {entity.frame_ids.map(frameId => (
                              <button
                                className="font-mono text-[0.625rem] text-emerald-400"
                                key={frameId}
                                onClick={() => selectFrame(frameId)}
                                type="button"
                              >
                                {frameId}
                              </button>
                            ))}
                          </div>
                        )}
                      </details>
                    ))}
                  </div>
                )}
              </section>

              <section className="space-y-2">
                <div>
                  <h3 className="text-sm font-semibold">{md.section3Title}</h3>
                  <p className="text-xs text-(--ui-text-tertiary)">
                    {md.section3Hint}
                  </p>
                </div>
                {eventCount === 0 ? (
                  <EmptyState>{md.noEvents}</EmptyState>
                ) : (
                  <div className="space-y-2">
                    {superEvents.map(event => (
                      <EventCard event={event} key={event.id} level="super" onFrame={selectFrame} />
                    ))}
                    {macroEvents.map(event => (
                      <EventCard event={event} key={event.id} level="macro" onFrame={selectFrame} />
                    ))}
                    {microEvents.map(event => (
                      <EventCard event={event} key={event.id} level="micro" onFrame={selectFrame} />
                    ))}
                  </div>
                )}
              </section>

              <section className="space-y-2">
                <div>
                  <h3 className="text-sm font-semibold">{md.section4Title}</h3>
                  <p className="text-xs text-(--ui-text-tertiary)">{md.section4Hint}</p>
                </div>
                {evolutionCount === 0 ? (
                  <EmptyState>{md.noEvolution}</EmptyState>
                ) : (
                  <div className="space-y-2">
                    {entityStates.map(state => (
                      <div
                        className="rounded border border-(--ui-stroke-secondary) border-l-2 border-l-cyan-400 bg-(--ui-bg-elevated) p-2 text-xs"
                        key={`state-${state.id}`}
                      >
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="font-mono text-cyan-400">{fmtRelativeTime(state.t_observed)}</span>
                          <span className="font-semibold">{state.entity_name}</span>
                          <span className="rounded bg-(--ui-bg-quaternary) px-1.5 py-0.5">{state.state_label}</span>
                          <span className="text-(--ui-text-tertiary)">{state.source}</span>
                        </div>
                        <pre className="mt-2 whitespace-pre-wrap text-[0.6875rem]">
                          {debugJson(state.attributes_delta)}
                        </pre>
                      </div>
                    ))}
                    {revisions.map(revision => (
                      <details
                        className={cn(
                          'rounded border border-(--ui-stroke-secondary) border-l-2 bg-(--ui-bg-elevated) p-2 text-xs',
                          revision.success ? 'border-l-amber-400' : 'border-l-(--ui-red)'
                        )}
                        key={`revision-${revision.id}`}
                      >
                        <summary className="cursor-pointer list-none">
                          <span className="font-mono text-amber-400">{fmtWallTime(revision.t_applied)}</span>
                          <span className="ml-2 font-semibold">Reviewer: {revision.op}</span>
                        </summary>
                        <div className="mt-2 whitespace-pre-wrap">
                          {revision.reason || revision.error || md.noExplanation}
                        </div>
                        <pre className="mt-2 max-h-48 overflow-auto rounded bg-black/20 p-2 text-[0.6875rem]">
                          {debugJson(revision.payload)}
                        </pre>
                      </details>
                    ))}
                  </div>
                )}
              </section>
            </div>
          )}

          {tab === 'frame' && selectedDb && (
            <div className="grid min-h-0 grid-cols-1 gap-3 lg:grid-cols-[320px_minmax(0,1fr)]">
              <section className="min-w-0">
                <div className="mb-2 text-xs text-(--ui-text-tertiary)">
                  Memory frames · {overview?.timeline.length || 0}
                </div>
                <div className="max-h-[72vh] space-y-1 overflow-y-auto">
                  {(overview?.timeline || []).map(item => (
                    <button
                      className={cn(
                        'flex w-full gap-2 rounded border p-2 text-left text-xs',
                        selectedFrameId === item.frame_id
                          ? 'border-emerald-400 bg-emerald-400/10'
                          : 'border-(--ui-stroke-secondary) bg-(--ui-bg-elevated)'
                      )}
                      key={item.frame_id}
                      onClick={() => setSelectedFrameId(item.frame_id)}
                      type="button"
                    >
                      {item.thumb_b64 ? (
                        <img
                          alt=""
                          className="h-12 w-16 shrink-0 object-cover"
                          src={`data:image/jpeg;base64,${item.thumb_b64}`}
                        />
                      ) : (
                        <div className="h-12 w-16 shrink-0 bg-black" />
                      )}
                      <span className="min-w-0 flex-1">
                        <span className="block truncate font-mono text-emerald-400">
                          {fmtRelativeTime(item.t_observed)} · {item.frame_id}
                        </span>
                        <span className="block truncate text-(--ui-text-tertiary)">
                          {item.source_type || item.source || 'unknown'} ·{' '}
                          {item.window_title || item.app || item.note || item.micro_id}
                        </span>
                        <span className="line-clamp-2 text-(--ui-text-secondary)">
                          {item.raw_preview || item.observation_preview || '(visual key frame; no OCR text)'}
                        </span>
                      </span>
                    </button>
                  ))}
                </div>
              </section>
              <section className="min-w-0 space-y-3">
                <div className="flex flex-wrap items-center gap-2 text-xs text-(--ui-text-tertiary)">
                  <span className="font-mono text-emerald-400">{selectedFrameId || 'no frame selected'}</span>
                  {activeFrame?.memory_frame?.source_type && (
                    <span className="rounded bg-(--ui-bg-quaternary) px-1.5 py-0.5">
                      {activeFrame.memory_frame.source_type}
                    </span>
                  )}
                </div>
                <OcrOverlayImage blocks={Array.isArray(blocks) ? blocks : []} imageB64={activeFrame?.image_b64 || ''} />
                <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
                  <div>
                    <div className="mb-1 text-xs font-semibold">OCR Raw Text</div>
                    <pre className="max-h-64 overflow-auto whitespace-pre-wrap rounded border border-(--ui-stroke-secondary) bg-(--ui-bg-elevated) p-2 text-[0.6875rem]">
                      {activeFrame?.screen_text?.raw_text || '(empty)'}
                    </pre>
                  </div>
                  <div>
                    <div className="mb-1 text-xs font-semibold">
                      OCR Blocks · {Array.isArray(blocks) ? blocks.length : 0}
                    </div>
                    <pre className="max-h-64 overflow-auto whitespace-pre-wrap rounded border border-(--ui-stroke-secondary) bg-(--ui-bg-elevated) p-2 text-[0.6875rem]">
                      {debugJson((Array.isArray(blocks) ? blocks : []).slice(0, 80))}
                    </pre>
                  </div>
                </div>
                {(activeFrame?.tables || []).map(table => (
                  <div className="space-y-1" key={`${table.table_id}-${table.frame_id}`}>
                    <div className="text-xs font-semibold text-amber-400">
                      {table.table_id} · {table.title || 'table'}
                    </div>
                    <MemoryTable table={table} />
                  </div>
                ))}
                <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
                  <pre className="max-h-52 overflow-auto rounded bg-black/20 p-2 text-[0.6875rem]">
                    {debugJson(activeFrame?.micro_events || [])}
                  </pre>
                  <pre className="max-h-52 overflow-auto rounded bg-black/20 p-2 text-[0.6875rem]">
                    {debugJson(activeFrame?.entities || [])}
                  </pre>
                </div>
              </section>
            </div>
          )}

          {tab === 'search' && (
            <div className="space-y-3">
              <div className="flex gap-2">
                <Input
                  aria-label={md.searchLabel}
                  className="min-w-0 flex-1"
                  onChange={event => setSearchQuery(event.target.value)}
                  onKeyDown={event => {
                    if (event.key === 'Enter') {
                      void runSearch()
                    }
                  }}
                  placeholder={md.searchPlaceholder}
                  value={searchQuery}
                />
                <select
                  className="rounded border border-(--ui-stroke-secondary) bg-(--ui-bg-elevated) px-2 text-xs"
                  onChange={event => setSearchScope(event.target.value as SearchScope)}
                  value={searchScope}
                >
                  <option value="all">{md.scopeAll}</option>
                  <option value="today">{md.scopeToday}</option>
                  <option value="latest">{md.scopeLatest}</option>
                </select>
                <Button
                  disabled={
                    searchQuery.trim().replace(/\s+/g, '').length < 2 ||
                    (searchScope === 'latest' && !selectedDb)
                  }
                  onClick={() => void runSearch()}
                  size="sm"
                >
                  <Search />
                  {md.searchButton}
                </Button>
              </div>
              <div className="space-y-2">
                {searchResults.length === 0 ? (
                  <EmptyState>{md.searchEmpty}</EmptyState>
                ) : (
                  searchResults.map((result, index) => (
                    <details
                      className="rounded border border-(--ui-stroke-secondary) bg-(--ui-bg-elevated) p-2 text-xs"
                      key={`${result.session}-${result.kind}-${result.frame_id}-${index}`}
                      open={index < 3}
                    >
                      <summary className="flex cursor-pointer list-none items-center gap-2">
                        {result.thumb_b64 && (
                          <img
                            alt=""
                            className="h-10 w-14 object-cover"
                            src={`data:image/jpeg;base64,${result.thumb_b64}`}
                          />
                        )}
                        <span className="min-w-0 flex-1">
                          <span className="block truncate font-medium">{result.title}</span>
                          <span className="font-mono text-(--ui-text-tertiary)">
                            {result.session} · {result.kind} · score {result.score}
                          </span>
                        </span>
                      </summary>
                      <pre className="mt-2 max-h-44 overflow-auto whitespace-pre-wrap rounded bg-black/20 p-2">
                        {result.snippet}
                      </pre>
                      {result.table && (
                        <MemoryTable table={{ ...result.table, rows: result.table.row_hits.map(hit => hit.row) }} />
                      )}
                    </details>
                  ))
                )}
              </div>
            </div>
          )}

          {tab === 'debug' && (
            <div className="space-y-4">
              <section className="space-y-2">
                <div className="sticky top-0 z-10 flex flex-wrap items-center gap-2 rounded border border-(--ui-stroke-secondary) bg-(--ui-sidebar-surface-background)/95 p-2 text-xs backdrop-blur">
                  <Activity className="size-3.5 text-cyan-400" />
                  <span className="font-semibold">{md.trajectoryGroupTitle}</span>
                  <span className="text-(--ui-text-tertiary)">
                    {visibleQuestions.length} questions · {visibleTrajectoryCount} events
                  </span>
                  <select
                    className="ml-auto rounded border border-(--ui-stroke-secondary) bg-(--ui-bg-elevated) px-2 py-1 text-xs"
                    onChange={event => setWorkerFilter(event.target.value)}
                    value={workerFilter}
                  >
                    <option value="all">all workers</option>
                    {workers.map(worker => (
                      <option key={worker} value={worker}>
                        {worker}
                      </option>
                    ))}
                  </select>
                </div>
                <p className="text-[0.6875rem] text-(--ui-text-tertiary)">
                  {md.trajectoryGroupHint}
                </p>
                {isOtherSession ? (
                  <EmptyState>
                    {md.trajectoryOtherSession}
                  </EmptyState>
                ) : visibleTrajectoryCount === 0 ? (
                  <EmptyState>{md.trajectoryEmpty}</EmptyState>
                ) : (
                  <div className="space-y-2">
                    {[...visibleQuestions].reverse().map((question, index) => (
                      <TrajectoryQuestion initiallyOpen={index === 0} key={question.id} question={question} />
                    ))}
                    {visibleBackground.length > 0 && <BackgroundTrajectory groups={visibleBackground} />}
                  </div>
                )}
              </section>

              <section className="space-y-2">
                <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
                  {Object.entries(counts).map(([key, value]) => (
                    <div className="rounded border border-(--ui-stroke-secondary) bg-(--ui-bg-elevated) p-2" key={key}>
                      <div className="text-[0.625rem] uppercase text-(--ui-text-tertiary)">{key}</div>
                      <div className="font-mono text-lg text-emerald-400">{String(value)}</div>
                    </div>
                  ))}
                </div>
                {overview && (
                  <div className="grid grid-cols-1 gap-2 lg:grid-cols-2">
                    <div className="rounded border border-(--ui-stroke-secondary) bg-(--ui-bg-elevated) p-2 text-xs">
                      <div className="font-semibold">Session health</div>
                      <div className="mt-1">
                        size: {fmtBytes(overview.session.size)} · frame files: {String(health.frame_files ?? 0)}
                      </div>
                      <div>
                        micro without frames: {String(health.micro_events_without_frames ?? 0)} · empty OCR:{' '}
                        {String(health.screen_texts_without_raw_text ?? 0)}
                      </div>
                    </div>
                    <pre className="max-h-40 overflow-auto rounded border border-(--ui-stroke-secondary) bg-(--ui-bg-elevated) p-2 text-[0.6875rem]">
                      {debugJson(overview.session.meta)}
                    </pre>
                  </div>
                )}
              </section>

              <section className="grid grid-cols-1 gap-3 lg:grid-cols-2">
                <div>
                  <div className="mb-1 flex items-center gap-1 text-xs font-semibold">
                    <Brain className="size-3.5" />
                    Recall / Query messages
                  </div>
                  <div className="space-y-2">
                    {(trace?.messages || []).length === 0 ? (
                      <EmptyState>{md.noRecallMessages}</EmptyState>
                    ) : (
                      trace!.messages.map((message, index) => (
                        <details
                          className="rounded border border-(--ui-stroke-secondary) bg-(--ui-bg-elevated) p-2 text-xs"
                          key={`${message.timestamp}-${index}`}
                        >
                          <summary className="cursor-pointer list-none">
                            <span className="font-mono text-emerald-400">{message.tool_name || message.role}</span>
                            <span className="ml-2 text-(--ui-text-tertiary)">{fmtWallTime(message.timestamp)}</span>
                          </summary>
                          <pre className="mt-2 max-h-64 overflow-auto whitespace-pre-wrap">
                            {message.content || debugJson(message.tool_calls)}
                          </pre>
                        </details>
                      ))
                    )}
                  </div>
                </div>
                <div>
                  <div className="mb-1 text-xs font-semibold">Relevant logs</div>
                  <pre className="max-h-[70vh] overflow-auto rounded border border-(--ui-stroke-secondary) bg-black/30 p-2 text-[0.6875rem] text-(--ui-text-tertiary)">
                    {(trace?.logs || overview?.trace.logs || []).join('\n') || 'No recall/writer/OCR logs found.'}
                  </pre>
                </div>
              </section>
            </div>
          )}
        </div>
      </SheetContent>
    </Sheet>
  )
}
