import { translateNow } from '@/i18n'

export interface MmTrajectoryFrame {
  frame_id?: string
  jpeg_b64?: string
  source_type?: string
  thumb_b64?: string
  ts?: number
}

export interface MmTrajectoryEntry {
  event: string
  id: string
  payload: Record<string, unknown>
  phase: string
  seq: number
  ts: number
  worker: string
}

export interface MmTrajectoryWorkerGroup {
  entries: MmTrajectoryEntry[]
  firstSeq: number
  lastSeq: number
  worker: string
}

export interface MmTrajectoryQuestionGroup {
  entries: MmTrajectoryEntry[]
  firstSeq: number
  firstTs: number
  id: string
  label: string
  lastSeq: number
  lastTs: number
  workers: MmTrajectoryWorkerGroup[]
}

export interface MmTrajectoryGrouping {
  background: MmTrajectoryWorkerGroup[]
  questions: MmTrajectoryQuestionGroup[]
}

interface MutableQuestionGroup {
  entries: MmTrajectoryEntry[]
  firstSeq: number
  firstTs: number
  id: string
  label: string
  labelRank: number
  lastSeq: number
  lastTs: number
}

interface LabelCandidate {
  rank: number
  text: string
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function cleanId(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

function cleanText(value: unknown): string {
  if (typeof value !== 'string') {
    return ''
  }

  return value.replace(/\s+/g, ' ').trim()
}

function resultOf(entry: MmTrajectoryEntry): Record<string, unknown> {
  const result = entry.payload.result

  return isRecord(result) ? result : {}
}

/**
 * Return only identifiers whose backend contract explicitly means "the user
 * message that owns this work". In particular, a bare request_id is excluded:
 * watcher request_ids are watcher task ids, not chat-question ids.
 */
function explicitQuestionId(entry: MmTrajectoryEntry): string {
  const payload = entry.payload || {}
  const result = resultOf(entry)
  const voiceQuestionId = entry.event === 'multimodal.asr_final' ? cleanId(payload.request_id) : ''

  return (
    cleanId(payload.parent_user_message_id) ||
    cleanId(result.parent_user_message_id) ||
    cleanId(payload.client_request_id) ||
    voiceQuestionId
  )
}

function addAlias(aliases: string[], namespace: string, value: unknown): void {
  const id = cleanId(value)

  if (id) {
    aliases.push(`${namespace}:${id}`)
  }
}

/** Correlation handles that are task identities, never assumed question ids. */
function correlationAliases(entry: MmTrajectoryEntry): string[] {
  const payload = entry.payload || {}
  const result = resultOf(entry)
  const aliases: string[] = []

  for (const source of [payload, result]) {
    addAlias(aliases, 'task', source.task_id)
    addAlias(aliases, 'request', source.request_id)
    addAlias(aliases, 'monitor', source.monitor_id)
    addAlias(aliases, 'watcher', source.watcher_id)
    addAlias(aliases, 'tool', source.tool_id)
  }

  if (Array.isArray(payload.task_ids)) {
    for (const taskId of payload.task_ids) {
      addAlias(aliases, 'task', taskId)
    }
  }

  return [...new Set(aliases)]
}

function questionLabel(entry: MmTrajectoryEntry): LabelCandidate | null {
  const payload = entry.payload || {}
  const result = resultOf(entry)

  const candidates: LabelCandidate[] = [
    { rank: 60, text: cleanText(payload.original_user_text) },
    { rank: 60, text: cleanText(result.original_user_text) },
    { rank: 50, text: cleanText(payload.query) },
    { rank: 45, text: cleanText(result.query) },
    {
      rank: 40,
      text:
        entry.event === 'multimodal.asr_final' ||
        /prompt|fifo/i.test(entry.phase) ||
        payload.origin === 'user' ||
        payload.origin === 'voice_asr'
          ? cleanText(payload.text)
          : ''
    },
    { rank: 20, text: cleanText(payload.brief) },
    { rank: 20, text: cleanText(result.brief) }
  ]

  return candidates.find(candidate => candidate.text) || null
}

function groupWorkers(entries: MmTrajectoryEntry[]): MmTrajectoryWorkerGroup[] {
  const grouped = new Map<string, MmTrajectoryEntry[]>()

  for (const entry of entries) {
    const worker = cleanText(entry.worker) || 'Multimodal'
    const current = grouped.get(worker) || []
    current.push(entry)
    grouped.set(worker, current)
  }

  return [...grouped.entries()]
    .map(([worker, workerEntries]) => {
      const sorted = [...workerEntries].sort((a, b) => a.seq - b.seq || a.ts - b.ts)

      return {
        entries: sorted,
        firstSeq: sorted[0]?.seq ?? 0,
        lastSeq: sorted.at(-1)?.seq ?? 0,
        worker
      }
    })
    .sort((a, b) => a.firstSeq - b.firstSeq || a.worker.localeCompare(b.worker))
}

/**
 * Group a flat gateway trajectory into question-owned worker traces.
 *
 * This is deliberately a two-pass reducer. A task may emit progress before its
 * tool-complete handoff publishes task_id -> parent_user_message_id; the later
 * authoritative mapping still groups the earlier rows. Ambiguous alias reuse
 * is failed closed into the background section instead of guessing by time.
 */
export function groupTrajectoryByQuestion(entries: MmTrajectoryEntry[]): MmTrajectoryGrouping {
  const sorted = [...entries].sort((a, b) => a.seq - b.seq || a.ts - b.ts)
  const aliasOwner = new Map<string, null | string>()

  for (const entry of sorted) {
    const questionId = explicitQuestionId(entry)

    if (!questionId) {
      continue
    }

    // A foreground turn's request id is normally identical to its explicit
    // question id. Seed that alias only after the question contract is known;
    // this links sibling rows that carry request_id alone without promoting a
    // watcher/task-only request id into a chat question.
    const aliases = [`request:${questionId}`, ...correlationAliases(entry)]

    for (const alias of new Set(aliases)) {
      const current = aliasOwner.get(alias)

      if (current === undefined || current === questionId) {
        aliasOwner.set(alias, questionId)
      } else {
        aliasOwner.set(alias, null)
      }
    }
  }

  const questionGroups = new Map<string, MutableQuestionGroup>()
  const background: MmTrajectoryEntry[] = []

  for (const entry of sorted) {
    const explicit = explicitQuestionId(entry)

    const correlated = correlationAliases(entry)
      .map(alias => aliasOwner.get(alias))
      .filter((owner): owner is string => Boolean(owner))

    const uniqueOwners = [...new Set(correlated)]
    const questionId = explicit || (uniqueOwners.length === 1 ? uniqueOwners[0] : '')

    if (!questionId) {
      background.push(entry)

      continue
    }

    let group = questionGroups.get(questionId)

    if (!group) {
      group = {
        entries: [],
        firstSeq: entry.seq,
        firstTs: entry.ts,
        id: questionId,
        label: '',
        labelRank: 0,
        lastSeq: entry.seq,
        lastTs: entry.ts
      }
      questionGroups.set(questionId, group)
    }

    group.entries.push(entry)
    group.firstSeq = Math.min(group.firstSeq, entry.seq)
    group.lastSeq = Math.max(group.lastSeq, entry.seq)
    group.firstTs = Math.min(group.firstTs, entry.ts)
    group.lastTs = Math.max(group.lastTs, entry.ts)
    const label = questionLabel(entry)

    if (label && label.rank > group.labelRank) {
      group.label = label.text
      group.labelRank = label.rank
    }
  }

  const questions = [...questionGroups.values()]
    .map(group => ({
      entries: group.entries,
      firstSeq: group.firstSeq,
      firstTs: group.firstTs,
      id: group.id,
      label: group.label || translateNow('multimodal.misc.question', group.id),
      lastSeq: group.lastSeq,
      lastTs: group.lastTs,
      workers: groupWorkers(group.entries)
    }))
    .sort((a, b) => a.firstSeq - b.firstSeq || a.firstTs - b.firstTs)

  return { background: groupWorkers(background), questions }
}
