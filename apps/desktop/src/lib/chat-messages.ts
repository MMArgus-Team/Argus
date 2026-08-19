import type { ThreadMessageLike } from '@assistant-ui/react'

import { dedupeGeneratedImageEchoesInParts } from '@/lib/generated-images'
import { mediaDisplayLabel, mediaMarkdownHref } from '@/lib/media'
import { parseTodos } from '@/lib/todos'
import type { SessionMessage, ToolArgField, UsageStats } from '@/types/hermes'

export type ChatMessagePart = Exclude<ThreadMessageLike['content'], string>[number]

/**
 * A `tool-call` part plus our own `argsFields` sidecar.
 *
 * assistant-ui's part shape is fixed, but `fromThreadMessageLike` converts a
 * tool-call by spreading the whole part (`const { parentId, messages,
 * ...basePart } = part`), so an extra key rides through untouched and reaches
 * `ToolFallback` as a prop. That is the only channel available: the backend
 * ships `args_fields` as a peer of `args`, and it must NOT be folded into
 * `args` itself (see `toolArgFieldsFrom`).
 */
export type ToolCallPartWithArgFields = Extract<ChatMessagePart, { type: 'tool-call' }> & {
  argsFields?: ToolArgField[]
}

/** Read the `argsFields` sidecar off a part, whatever its declared type. */
export function argFieldsOfPart(part: unknown): ToolArgField[] {
  const row = part && typeof part === 'object' ? (part as { argsFields?: unknown }) : null

  return Array.isArray(row?.argsFields) ? (row.argsFields as ToolArgField[]) : []
}

/**
 * Sub-role of an assistant message, from the gateway's message.start tag.
 * A plain model reply has none; monitor SPEAKs and deep-research threadbacks
 * carry one so the chat can label + color them like the web multimodal view.
 */
export type ChatSubRole = 'monitor' | 'router' | 'watcher_report'

export type ChatMessage = {
  id: string
  /** Stable gateway turn id used to correlate optimistic/tool/assistant items. */
  requestId?: string
  role: SessionMessage['role']
  parts: ChatMessagePart[]
  timestamp?: number
  pending?: boolean
  error?: string
  branchGroupId?: string
  hidden?: boolean
  /** Composer attachment ref strings (`@file:...`, `@image:...`) sent with this user message. */
  attachmentRefs?: string[]
  /** Assistant sub-role (monitor SPEAK / deep-research threadback) for labeling + coloring. */
  subRole?: ChatSubRole
  /** monitor: the short event label shown instead of "Assistant". */
  monitorLabel?: string
  /** router (deep research): the analysis event name / brief. */
  brief?: string
  /** watcher_report / router: the deep-research request id (req_xxxx), so the
   *  bubble can reopen the read-only deep window from the on-disk analyse file. */
  deepReportRid?: string
  /** watcher_report: 本段的画面时段区间 (mm:ss–mm:ss), 展示在正文卡头。 */
  deepRange?: string
  /** watcher_report: 段序号 (第N段), 展示在正文卡头 (与 web 对齐)。 */
  deepRound?: number
  /** Model name for the header badge (resolved at capture time). */
  model?: string
  /** This assistant turn was a spoken (voice) reply. */
  voice?: boolean
}

export type GatewayEventPayload = {
  text?: string
  rendered?: string
  status?: string
  message?: string
  id?: string
  name?: string
  tool_id?: string
  tool_call_id?: string
  args?: unknown
  arguments?: unknown
  /** Classified call-side args (see ToolArgField). Present on tool.start. */
  args_fields?: unknown
  context?: string
  input?: unknown
  preview?: string
  result?: unknown
  summary?: string
  error?: string | boolean
  inline_diff?: string
  duration_s?: number
  todos?: unknown
  model?: string
  provider?: string
  reasoning_effort?: string
  service_tier?: string
  fast?: boolean
  yolo?: boolean
  running?: boolean
  cwd?: string
  branch?: string
  credential_warning?: string
  personality?: string
  usage?: Partial<UsageStats>
  // clarify.request
  request_id?: string
  question?: string
  choices?: string[] | null
  // approval.request (dangerous command / execute_code) — session-keyed
  command?: string
  description?: string
  // False when a tirith content-security warning forbids a permanent allow.
  allow_permanent?: boolean
  // secret.request (skill credential capture)
  env_var?: string
  prompt?: string
  // terminal.read.request (GUI agent reading the in-app terminal pane)
  start?: number
  count?: number
  // status.update (kind=process → background process completion/watch-match)
  kind?: string
  // session.title (live auto-title push) — stored session id + generated title
  session_id?: string
  title?: string
  // moa.reference / moa.aggregating (Mixture of Agents per-model relay)
  label?: string
  index?: number
  aggregator?: string
  // message.start/.delta/.complete sub-role tag (monitor SPEAK / deep-research
  // threadback) — drives web-style per-type labeling + coloring in the chat.
  source?: string
  monitor_id?: string
  monitor_label?: string
  brief?: string
  voice?: boolean
  /** Pure control turns live in the multimodal registry, not center chat. */
  history_policy?: unknown
  ephemeral_control?: unknown
  ephemeral?: unknown
}

export function textPart(text: string): ChatMessagePart {
  return { type: 'text', text }
}

export function reasoningPart(text: string): ChatMessagePart {
  return { type: 'reasoning', text }
}

const MEDIA_LINE_RE = /(^|\n)[\t ]*[`"']?MEDIA:\s*(?<line>`[^`\n]+`|"[^"\n]+"|'[^'\n]+'|\S+)[`"']?[\t ]*(\n|$)/g

const MEDIA_TAG_RE = /[`"']?MEDIA:\s*(?<inline>`[^`\n]+`|"[^"\n]+"|'[^'\n]+'|\S+)[`"']?/g

function unquoteMediaPath(value: string): string {
  const trimmed = value.trim()
  const quote = trimmed[0]

  return quote && quote === trimmed.at(-1) && ['"', "'", '`'].includes(quote) ? trimmed.slice(1, -1) : trimmed
}

function mediaLink(value: string): string {
  const path = unquoteMediaPath(value)

  return `[${mediaDisplayLabel(path)}](${mediaMarkdownHref(path)})`
}

export function renderMediaTags(text: string): string {
  return text
    .replace(
      MEDIA_LINE_RE,
      (_match, lead: string, value: string, trailer: string) => `${lead}${mediaLink(value)}${trailer}`
    )
    .replace(MEDIA_TAG_RE, (_match, value: string) => mediaLink(value))
    .replace(/[ \t]+\n/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
}

export function assistantTextPart(text: string): ChatMessagePart {
  return textPart(renderMediaTags(text))
}

export function chatMessageText(message: ChatMessage): string {
  return message.parts
    .filter((part): part is Extract<ChatMessagePart, { type: 'text' }> => part.type === 'text')
    .map(part => part.text)
    .join('')
}

const ATTACHED_CONTEXT_MARKER_RE = /(?:^|\n)--- Attached Context ---\s*\n/
const CONTEXT_WARNINGS_MARKER_RE = /(?:^|\n)--- Context Warnings ---[\s\S]*$/
const CONTEXT_REF_RE = /@(file|folder|url|image|tool|terminal):(?:"[^"\n]+"|'[^'\n]+'|`[^`\n]+`|\S+)/g

// The multimodal vision layer prepends a "[These are a few snapshot frames …]"
// framing marker to the LLM payload and appends the live frames as image parts
// (which serialize to "[screenshot]" / "[image]" placeholders in history text).
// Those are for the model, not the user — web never shows them because it renders
// its own optimistic text. Desktop replays server history, so strip them here so
// a user turn (typed OR a monitor-hook trigger) shows only the real content.
const VISION_SNAPSHOT_MARKER_RE = /\[These are a few snapshot frames[\s\S]*?do not assume you can keep watching yourself\.\]\s*/g
// Match one-or-more consecutive [screenshot]/[image] placeholders (each on its
// own line) as a single run, so back-to-back frames all strip in one pass.
const IMAGE_PLACEHOLDER_RE = /\s*(?:\[(?:screenshot|image)\]\s*)+/gi

function textFromUnknown(value: unknown, depth = 0): string {
  if (typeof value === 'string') {
    return value
  }

  if (value === null || value === undefined) {
    return ''
  }

  if (depth > 2) {
    return ''
  }

  if (Array.isArray(value)) {
    return value.map(item => textFromUnknown(item, depth + 1)).join('')
  }

  if (typeof value === 'object') {
    const row = value as Record<string, unknown>
    const textValue = row.text ?? row.output_text ?? row.content ?? row.message
    const nestedText = textFromUnknown(textValue, depth + 1)

    if (nestedText) {
      return nestedText
    }

    try {
      return JSON.stringify(value)
    } catch {
      return ''
    }
  }

  return String(value)
}

// Drop the multimodal vision framing marker + injected frame placeholders from a
// user turn's text so they never render (matches web, which shows its own text).
function stripVisionInjection(text: string): string {
  return text.replace(VISION_SNAPSHOT_MARKER_RE, '').replace(IMAGE_PLACEHOLDER_RE, ' ').trim()
}

function displayContentForMessage(role: SessionMessage['role'], content: unknown): string {
  const textContent = textFromUnknown(content)

  if (role !== 'user') {
    return textContent
  }

  const cleaned = stripVisionInjection(textContent)

  const marker = cleaned.match(ATTACHED_CONTEXT_MARKER_RE)

  if (!marker || marker.index === undefined) {
    return cleaned.replace(CONTEXT_WARNINGS_MARKER_RE, '').trim()
  }

  const visibleText = cleaned.slice(0, marker.index).replace(CONTEXT_WARNINGS_MARKER_RE, '').trim()
  const attachedContext = cleaned.slice(marker.index + marker[0].length)
  const refs = [...new Set(Array.from(attachedContext.matchAll(CONTEXT_REF_RE)).map(match => match[0]))]

  return [refs.join('\n'), visibleText].filter(Boolean).join('\n\n') || visibleText
}

const STREAM_PART: Record<'reasoning' | 'text', (text: string) => ChatMessagePart> = {
  reasoning: reasoningPart,
  text: textPart
}

// Coalesce a streaming delta into the most recent same-type part within the
// current segment, where a segment is bounded by any non-streaming part (a
// tool call, image, …). The opposite streaming channel (text <-> reasoning) is
// transparent, so a reasoning burst between two content deltas can't shred one
// sentence into text / Thinking / text — the fragmentation models that
// interleave reasoning_content + content otherwise produce. Tool calls still
// open a fresh part, preserving narration order across steps.
function appendStreamPart(
  parts: ChatMessagePart[],
  type: 'reasoning' | 'text',
  delta: string
): { index: number; parts: ChatMessagePart[] } {
  const next = [...parts]

  for (let i = next.length - 1; i >= 0; i--) {
    const part = next[i]

    if (part.type === type) {
      next[i] = { ...part, text: `${(part as { text: string }).text}${delta}` } as ChatMessagePart

      return { index: i, parts: next }
    }

    if (part.type !== 'text' && part.type !== 'reasoning') {
      break
    }
  }

  next.push(STREAM_PART[type](delta))

  return { index: next.length - 1, parts: next }
}

export function appendTextPart(parts: ChatMessagePart[], delta: string): ChatMessagePart[] {
  return appendStreamPart(parts, 'text', delta).parts
}

export function appendReasoningPart(parts: ChatMessagePart[], delta: string): ChatMessagePart[] {
  return appendStreamPart(parts, 'reasoning', delta).parts
}

export function appendAssistantTextPart(parts: ChatMessagePart[], delta: string): ChatMessagePart[] {
  const { index, parts: next } = appendStreamPart(parts, 'text', delta)
  const part = next[index]

  if (part?.type !== 'text') {
    return next
  }

  const mayContainMedia =
    delta.includes('MEDIA:') || delta.includes('DIA:') || delta.includes('EDIA:') || delta.includes('IA:')

  if (mayContainMedia || part.text.includes('MEDIA:')) {
    const rendered = renderMediaTags(part.text)

    if (rendered !== part.text) {
      next[index] = { ...part, text: rendered }
    }
  }

  return next
}

export function hasToolPart(message: ChatMessage): boolean {
  return message.parts.some(part => part.type === 'tool-call')
}

function toolId(payload: GatewayEventPayload | undefined): string {
  return payload?.tool_id || payload?.tool_call_id || payload?.id || ''
}

let liveToolCounter = 0

function nextLiveToolId(name: string): string {
  liveToolCounter += 1

  return `live-tool:${name}:${liveToolCounter}`
}

function firstStringField(record: Record<string, unknown>, keys: readonly string[]): string {
  for (const key of keys) {
    const value = record[key]

    if (typeof value === 'string' && value.trim()) {
      return value.trim()
    }
  }

  return ''
}

function normalizeToolMatchValue(value: string): string {
  return value.trim().toLowerCase()
}

function collectToolMatchValues(query: string, context: string, preview: string): string[] {
  return [...new Set([query, context, preview].map(normalizeToolMatchValue).filter(Boolean))]
}

function toolPayloadMatchValues(payload: GatewayEventPayload | undefined): string[] {
  const payloadArgs = liveToolArgs(payload)
  const query = firstStringField(payloadArgs, ['search_term', 'query'])
  const context = typeof payload?.context === 'string' ? payload.context.trim() : ''
  const preview = typeof payload?.preview === 'string' ? payload.preview.trim() : ''

  return collectToolMatchValues(query, context, preview)
}

function toolPartMatchValues(part: ChatMessagePart): string[] {
  if (part.type !== 'tool-call' || !part.args || typeof part.args !== 'object') {
    return []
  }

  const args = part.args as Record<string, unknown>
  const query = firstStringField(args, ['search_term', 'query'])
  const context = typeof args.context === 'string' ? args.context.trim() : ''
  const preview = typeof args.preview === 'string' ? args.preview.trim() : ''

  return collectToolMatchValues(query, context, preview)
}

function hasToolMatchOverlap(left: string[], right: string[]): boolean {
  if (!left.length || !right.length) {
    return false
  }

  const rightSet = new Set(right)

  return left.some(value => rightSet.has(value))
}

function findToolPartIndex(
  parts: ChatMessagePart[],
  name: string,
  stableId: string,
  payload: GatewayEventPayload | undefined,
  phase: 'running' | 'complete'
): number {
  const matchValues = toolPayloadMatchValues(payload)
  const overlaps = (index: number) => hasToolMatchOverlap(matchValues, toolPartMatchValues(parts[index]))

  if (stableId) {
    const stableIndex = parts.findIndex(part => part.type === 'tool-call' && part.toolCallId === stableId)

    if (stableIndex >= 0) {
      return stableIndex
    }

    // Some live streams start without an id, then complete with one. Fall
    // through to pending same-name/context matching so the completion updates
    // the synthetic live row instead of appending a duplicate completed row.
    if (phase === 'running' && !matchValues.length) {
      return -1
    }
  }

  const pendingIndices = parts
    .map((part, index) => ({ part, index }))
    .filter(({ part }) => part.type === 'tool-call' && part.toolName === name && part.result === undefined)
    .map(({ index }) => index)

  if (pendingIndices.length === 0) {
    return -1
  }

  if (matchValues.length) {
    const contextualIndex = pendingIndices.find(overlaps)

    if (contextualIndex !== undefined) {
      return contextualIndex
    }
  }

  if (pendingIndices.length === 1) {
    const [singlePendingIndex] = pendingIndices

    if (phase === 'running' && matchValues.length && !overlaps(singlePendingIndex)) {
      return stableId ? singlePendingIndex : -1
    }

    return singlePendingIndex
  }

  // Completion events without stable IDs frequently arrive after multiple
  // same-name starts (parallel tool calls). Resolve them oldest-first so we
  // don't collapse an entire burst into a single row.
  if (phase === 'complete') {
    return pendingIndices[0]
  }

  if (stableId) {
    return pendingIndices[0]
  }

  // For progress/running events with no stable id, update the most-recent
  // pending same-name tool instead of creating a phantom extra row.
  return pendingIndices.at(-1) ?? -1
}

// Carry todo state across sparse progress payloads: if this todo event lacks
// a `todos` field, fall back to whatever we previously stored on the part.
function carryTodos(payload: GatewayEventPayload | undefined, ...prev: unknown[]): { todos: unknown } | undefined {
  if (payload && Object.hasOwn(payload, 'todos')) {
    const next = parseTodos(payload.todos)

    return next === null ? undefined : { todos: next }
  }

  if (payload?.name !== 'todo') {
    return undefined
  }

  for (const p of prev) {
    const carried = parseTodos(recordFromUnknown(p)?.todos)

    if (carried !== null) {
      return { todos: carried }
    }
  }

  return undefined
}

function toolArgs(payload: GatewayEventPayload | undefined, prevArgs?: unknown): Record<string, unknown> {
  const prev = parseMaybeJsonObject(prevArgs)
  const eventArgs = liveToolArgs(payload)

  return {
    ...prev,
    ...eventArgs,
    ...(payload?.context ? { context: payload.context } : {}),
    ...(payload?.preview ? { preview: payload.preview } : {}),
    ...carryTodos(payload, prevArgs)
  }
}

function toolResult(
  payload: GatewayEventPayload | undefined,
  prevResult?: unknown,
  prevArgs?: unknown
): Record<string, unknown> {
  const parsedResult = parseMaybeJsonObject(payload?.result)

  return {
    ...parsedResult,
    ...(payload?.inline_diff ? { inline_diff: payload.inline_diff } : {}),
    ...(payload?.summary ? { summary: payload.summary } : {}),
    ...(payload?.message ? { message: payload.message } : {}),
    ...(payload?.preview ? { preview: payload.preview } : {}),
    ...(payload?.duration_s !== undefined ? { duration_s: payload.duration_s } : {}),
    ...carryTodos(payload, prevResult, prevArgs),
    ...(payload?.error ? { error: payload.error } : {})
  }
}

export function upsertToolPart(
  parts: ChatMessagePart[],
  payload: GatewayEventPayload | undefined,
  phase: 'running' | 'complete'
): ChatMessagePart[] {
  const stableId = toolId(payload)
  const name = payload?.name || 'tool'
  const next = [...parts]

  const index = findToolPartIndex(next, name, stableId, payload, phase)

  const prev = index >= 0 ? next[index] : null
  const prevArgs = prev && 'args' in prev ? prev.args : undefined
  const prevResult = prev && 'result' in prev ? prev.result : undefined
  const args = toolArgs(payload, prevArgs)

  const id =
    stableId ||
    (prev && 'toolCallId' in prev && typeof prev.toolCallId === 'string' ? prev.toolCallId : '') ||
    nextLiveToolId(name)

  // Only `tool.start` carries args_fields; the later progress/complete events
  // for the same row don't, so carry the first non-empty list forward rather
  // than letting a bare tool.complete blank the panel mid-turn.
  const argFields = toolArgFieldsFrom(payload?.args_fields)
  const carriedArgFields = argFields.length ? argFields : argFieldsOfPart(prev)

  const base = {
    type: 'tool-call' as const,
    toolCallId: id,
    toolName: name,
    args: args as never,
    argsText: JSON.stringify(args),
    ...(carriedArgFields.length ? { argsFields: carriedArgFields } : {}),
    ...(phase === 'complete' && { result: toolResult(payload, prevResult, prevArgs), isError: Boolean(payload?.error) })
  } satisfies ToolCallPartWithArgFields

  if (index === -1) {
    return [...next, base]
  }

  next[index] = { ...next[index], ...base }

  return next
}

function recordFromUnknown(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' ? (value as Record<string, unknown>) : null
}

function parseMaybeJsonObject(value: unknown): Record<string, unknown> {
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    return value as Record<string, unknown>
  }

  if (typeof value !== 'string' || !value.trim()) {
    return {}
  }

  try {
    const parsed = JSON.parse(value)

    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? (parsed as Record<string, unknown>) : {}
  } catch {
    return {}
  }
}

function firstNonEmptyObject(...values: unknown[]): Record<string, unknown> {
  for (const value of values) {
    const parsed = parseMaybeJsonObject(value)

    if (Object.keys(parsed).length > 0) {
      return parsed
    }
  }

  return {}
}

function liveToolArgs(payload: GatewayEventPayload | undefined): Record<string, unknown> {
  const direct = firstNonEmptyObject(payload?.args, payload?.arguments)
  const input = firstNonEmptyObject(payload?.input)
  const fn = recordFromUnknown(input.function)

  const nested = firstNonEmptyObject(
    input.args,
    input.arguments,
    input.parameters,
    input.input,
    fn?.arguments,
    fn?.args,
    fn?.parameters
  )

  return {
    ...input,
    ...nested,
    ...direct
  }
}

const TOOL_ARG_FIELD_KINDS = new Set<ToolArgField['kind']>([
  'credential',
  'elided',
  'freeform',
  'literal',
  'shape'
])

/**
 * Normalize the backend's `args_fields` into a typed list, dropping anything
 * malformed.
 *
 * Two deliberate choices:
 *
 * 1. An unrecognized `kind` is DROPPED, not passed through as a literal. A
 *    future backend kind is far more likely to be a new privacy class (whose
 *    whole point is withholding something) than a new safe-to-show one, so an
 *    older desktop build must fail closed.
 *
 * 2. These stay a SEPARATE field and are never merged back into `args`. The
 *    values here are display projections — truncated to 40 chars and passed
 *    through the redactor — so feeding them to the card heuristics would make
 *    them lie: `fileEditPath` would report a clipped filename as the real one
 *    and `toolPreviewTarget` would try to open a truncated URL. Real args
 *    (live `tool.complete`) keep flowing through `args` untouched; this list is
 *    render-only, and is the only arg data a RESUMED row has.
 */
export function toolArgFieldsFrom(value: unknown): ToolArgField[] {
  if (!Array.isArray(value)) {
    return []
  }

  const fields: ToolArgField[] = []

  for (const entry of value) {
    const row = recordFromUnknown(entry)
    const kind = row?.kind

    if (!row || typeof kind !== 'string' || !TOOL_ARG_FIELD_KINDS.has(kind as ToolArgField['kind'])) {
      continue
    }

    const key = typeof row.key === 'string' ? row.key : ''

    // Every kind except the synthetic `elided` tail is keyed by an arg name;
    // without one there is nothing to label the row with.
    if (!key && kind !== 'elided') {
      continue
    }

    fields.push({
      key,
      kind: kind as ToolArgField['kind'],
      ...(typeof row.value === 'string' ? { value: row.value } : {}),
      ...(typeof row.chars === 'number' ? { chars: row.chars } : {}),
      ...(typeof row.count === 'number' ? { count: row.count } : {})
    })
  }

  return fields
}

function parseStoredToolResult(content: unknown): unknown {
  if (content && typeof content === 'object') {
    return content
  }

  const textContent = textFromUnknown(content)

  if (!textContent.trim()) {
    return ''
  }

  try {
    return JSON.parse(textContent)
  } catch {
    return textContent
  }
}

function toolPartFromStoredCall(call: unknown, fallbackIndex: number): ChatMessagePart {
  const row = recordFromUnknown(call) ?? {}
  const fn = recordFromUnknown(row.function)
  const id = String(row.id || row.tool_call_id || `stored-tool-${fallbackIndex}`)

  const toolName = String(
    row.name || row.tool_name || fn?.name || (recordFromUnknown(row.input)?.name as string | undefined) || 'tool'
  )

  const args = firstNonEmptyObject(fn?.arguments, row.arguments, row.args, row.input)

  return {
    type: 'tool-call',
    toolCallId: id,
    toolName,
    args: args as never,
    argsText: Object.keys(args).length ? JSON.stringify(args) : ''
  }
}

function applyStoredToolResult(messages: ChatMessage[], toolMessage: SessionMessage): boolean {
  const toolCallId = toolMessage.tool_call_id || undefined
  const toolName = toolMessage.tool_name || toolMessage.name || 'tool'
  // Result = the tool's own output only. `context` (args preview) and `name`
  // must not stand in for it — doing so filled the card body with the command
  // that ran, or literally the tool's name, and made a resumed session look
  // like every tool had returned nothing useful.
  const content = toolMessage.content ?? toolMessage.text ?? ''

  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const message = messages[i]

    if (message.role !== 'assistant') {
      continue
    }

    const partIndex = message.parts.findIndex(
      part =>
        part.type === 'tool-call' &&
        ((toolCallId && part.toolCallId === toolCallId) || (!toolCallId && part.toolName === toolName))
    )

    if (partIndex < 0) {
      continue
    }

    const parts = [...message.parts]
    const existing = parts[partIndex]
    const argsFields = toolArgFieldsFrom(toolMessage.args_fields)
    parts[partIndex] = {
      ...existing,
      // See applyStoredToolResultToParts: fill in only, never overwrite.
      ...(argsFields.length && !argFieldsOfPart(existing).length ? { argsFields } : {}),
      result: parseStoredToolResult(content),
      isError: false
    } as ChatMessagePart
    messages[i] = { ...message, parts }

    return true
  }

  return false
}

function applyStoredToolResultToParts(parts: ChatMessagePart[], toolMessage: SessionMessage): ChatMessagePart[] | null {
  const toolCallId = toolMessage.tool_call_id || undefined
  const toolName = toolMessage.tool_name || toolMessage.name || 'tool'
  // See applyStoredToolResult: output only, never the args preview or the name.
  const content = toolMessage.content ?? toolMessage.text ?? ''

  const partIndex = parts.findIndex(
    part =>
      part.type === 'tool-call' &&
      ((toolCallId && part.toolCallId === toolCallId) || (!toolCallId && part.toolName === toolName))
  )

  if (partIndex < 0) {
    return null
  }

  const next = [...parts]
  const existing = next[partIndex]
  const argsFields = toolArgFieldsFrom(toolMessage.args_fields)
  next[partIndex] = {
    ...existing,
    // The call-side row may predate the classified args (it can come from a
    // stored `tool_calls` entry, which carries raw arguments instead). Only
    // fill the sidecar in — never overwrite a list already on the row.
    ...(argsFields.length && !argFieldsOfPart(existing).length ? { argsFields } : {}),
    result: parseStoredToolResult(content),
    isError: false
  } as ChatMessagePart

  return next
}

function storedToolMessagePart(toolMessage: SessionMessage, fallbackIndex: number): ChatMessagePart {
  const name = toolMessage.tool_name || toolMessage.name || 'tool'
  // `context` is the CALL side (args preview — the command that ran); `content`
  // is the tool's own RETURN value. Collapsing them into one string made
  // args.context === result.context, so fallbackDetailText() saw
  // `resultContext === argContext`, fell through to the arg preview, and the
  // card's body showed the command echoed back instead of its output. Keep them
  // as separate fields so the tool renderer can extract a real detail (and, for
  // terminal/execute_code, split stdout/stderr out of the JSON envelope).
  const context = textFromUnknown(toolMessage.context || '')
  const body = toolMessage.content ?? toolMessage.text ?? ''
  const args = context ? { context } : {}
  const result = parseStoredToolResult(body)
  // A resumed row has NO raw args (history ships only the classified
  // projection), so without this the reopened card's inputs were simply empty
  // while the live stream that produced it had shown them.
  const argsFields = toolArgFieldsFrom(toolMessage.args_fields)

  return {
    type: 'tool-call',
    toolCallId: toolMessage.tool_call_id || `stored-tool-message-${fallbackIndex}`,
    toolName: name,
    args: args as never,
    argsText: Object.keys(args).length ? JSON.stringify(args) : '',
    ...(argsFields.length ? { argsFields } : {}),
    // An absent result means "still running" to the renderer, so fall back to
    // the arg preview only when there is genuinely no stored output.
    result: result === '' ? (context ? { context } : {}) : result,
    isError: false
  }
}

function withUniqueToolCallIds(messages: ChatMessage[]): ChatMessage[] {
  const seen = new Set<string>()

  return messages.map(message => {
    let changed = false

    const parts = message.parts.map((part, index) => {
      if (part.type !== 'tool-call') {
        return part
      }

      const id = part.toolCallId || `${message.id}-tool-${index}`

      if (!seen.has(id)) {
        seen.add(id)

        if (part.toolCallId) {
          return part
        }

        changed = true

        return { ...part, toolCallId: id } as ChatMessagePart
      }

      changed = true
      const uniqueId = `${id}-${message.id}-${index}`
      seen.add(uniqueId)

      return { ...part, toolCallId: uniqueId } as ChatMessagePart
    })

    return changed ? { ...message, parts } : message
  })
}

export function toChatMessages(messages: SessionMessage[]): ChatMessage[] {
  const result: ChatMessage[] = []
  let pendingToolParts: ChatMessagePart[] = []
  let pendingToolTimestamp: number | undefined
  let activeAssistantIndex: null | number = null

  const clearPendingTools = () => {
    pendingToolParts = []
    pendingToolTimestamp = undefined
  }

  const appendPartsToActiveAssistant = (parts: ChatMessagePart[], timestamp?: number): boolean => {
    if (activeAssistantIndex === null) {
      return false
    }

    const active = result[activeAssistantIndex]

    if (!active || active.role !== 'assistant') {
      activeAssistantIndex = null

      return false
    }

    active.parts = [...active.parts, ...parts]
    active.timestamp = timestamp ?? active.timestamp

    return true
  }

  const flushPendingTools = (index: number) => {
    if (!pendingToolParts.length) {
      return
    }

    if (!appendPartsToActiveAssistant(pendingToolParts, pendingToolTimestamp)) {
      result.push({
        id: `${pendingToolTimestamp || Date.now()}-${index}-tools`,
        role: 'assistant',
        parts: pendingToolParts,
        timestamp: pendingToolTimestamp
      })
      activeAssistantIndex = result.length - 1
    }

    clearPendingTools()
  }

  // ★ 功能1: 从恢复的历史里把 monitor/watcher 通知元素 (后端 _history_to_messages
  //   已带上 subRole) 重建成气泡, 形状与实时 _emit 注入的一致 (见 store/multimodal.ts)。
  //   放在 forEach 顶部先处理, 且不参与 pending-tool 合并逻辑。
  const reconstructMmNotice = (message: SessionMessage): ChatMessage | null => {
    // Monitor + watcher_report used to be reconstructed as center-chat bubbles
    // here. They are now sidechannel-only — hydrated into the right multimodal
    // panel via list_monitor_alerts / list_watcher_content on session enter
    // (see fetchMmSidechannel). Detect legacy rows in EITHER payload shape
    //   (1) resume-RPC form: message.subRole == "monitor"/"watcher_report"
    //   (2) raw REST form:   content = {type:"mm_notice", mm_kind:"monitor"|"watcher", ...}
    // and drop them so we don't render duplicates.
    const rawContent = message.content
    const notice = rawContent && typeof rawContent === 'object' && !Array.isArray(rawContent)
      ? (rawContent as Record<string, unknown>) : null
    const rawKind = notice?.type === 'mm_notice' ? String(notice.mm_kind ?? '') : ''
    const sub = message.subRole
      ?? (rawKind === 'watcher' ? 'watcher_report'
          : rawKind ? 'monitor' : undefined)
    if (sub === 'monitor' || sub === 'watcher_report') return null
    return null
  }

  messages.forEach((message, index) => {
    const mmBubble = reconstructMmNotice(message)
    if (mmBubble) {
      flushPendingTools(index)
      result.push(mmBubble)
      // monitor/watcher 气泡不作为可续写的 activeAssistant (它是独立通知)。
      activeAssistantIndex = null
      return
    }
    if (message.role === 'tool') {
      const updatedPendingToolParts = applyStoredToolResultToParts(pendingToolParts, message)

      if (updatedPendingToolParts) {
        pendingToolParts = updatedPendingToolParts

        return
      }

      if (applyStoredToolResult(result, message)) {
        return
      }

      pendingToolParts = [...pendingToolParts, storedToolMessagePart(message, index)]
      pendingToolTimestamp ??= message.timestamp

      return
    }

    const content = message.content || message.text || message.context || message.name
    const displayContent = displayContentForMessage(message.role, content)
    const parts: ChatMessagePart[] = []

    const reasoning =
      message.reasoning ||
      message.reasoning_content ||
      (typeof message.reasoning_details === 'string' ? message.reasoning_details : '')

    if (reasoning && message.role === 'assistant') {
      parts.push(reasoningPart(reasoning))
    }

    if (displayContent) {
      parts.push(message.role === 'assistant' ? assistantTextPart(displayContent) : textPart(displayContent))
    }

    if (message.role === 'assistant' && Array.isArray(message.tool_calls)) {
      parts.push(...message.tool_calls.map((call, callIndex) => toolPartFromStoredCall(call, callIndex)))
    }

    if (!parts.length) {
      if (message.role !== 'assistant') {
        flushPendingTools(index)
        activeAssistantIndex = null
      }

      return
    }

    const isToolOnlyAssistant =
      message.role === 'assistant' && parts.length > 0 && parts.every(part => part.type === 'tool-call')

    if (isToolOnlyAssistant) {
      pendingToolParts = [...pendingToolParts, ...parts]
      pendingToolTimestamp ??= message.timestamp

      return
    }

    if (message.role === 'assistant') {
      if (pendingToolParts.length) {
        if (!appendPartsToActiveAssistant(pendingToolParts, message.timestamp ?? pendingToolTimestamp)) {
          parts.unshift(...pendingToolParts)
        }

        clearPendingTools()
      }

      const activeAssistant =
        activeAssistantIndex !== null && result[activeAssistantIndex]?.role === 'assistant'
          ? result[activeAssistantIndex]
          : null

      const currentHasToolCall = parts.some(part => part.type === 'tool-call')
      const activeHasToolCall = Boolean(activeAssistant?.parts.some(part => part.type === 'tool-call'))

      if (activeAssistant && (currentHasToolCall || activeHasToolCall)) {
        activeAssistant.parts = [...activeAssistant.parts, ...parts]
        activeAssistant.timestamp = message.timestamp ?? activeAssistant.timestamp

        return
      }
    } else {
      flushPendingTools(index)
    }

    result.push({
      id: `${message.timestamp || Date.now()}-${index}-${message.role}`,
      role: message.role,
      parts,
      timestamp: message.timestamp
    })

    activeAssistantIndex = message.role === 'assistant' ? result.length - 1 : null
  })
  flushPendingTools(messages.length)

  const withoutGeneratedImageEchoes = result.map(message =>
    message.role === 'assistant' ? { ...message, parts: dedupeGeneratedImageEchoesInParts(message.parts) } : message
  )

  return withUniqueToolCallIds(
    withoutGeneratedImageEchoes.filter(m => chatMessageText(m).trim() || m.parts.some(part => part.type !== 'text'))
  )
}

export function preserveLocalAssistantErrors(
  nextMessages: ChatMessage[],
  currentMessages: ChatMessage[]
): ChatMessage[] {
  const localById = new Map(currentMessages.map(message => [message.id, message]))

  const mergedNextMessages = nextMessages.map(message => {
    if (message.role !== 'assistant' || message.error || message.hidden) {
      return message
    }

    const local = localById.get(message.id)

    if (!local || local.role !== 'assistant' || !local.error || local.hidden) {
      return message
    }

    return {
      ...message,
      error: local.error,
      pending: false
    }
  })

  const existingIds = new Set(mergedNextMessages.map(message => message.id))
  const preserveIds = new Set<string>()
  const normalize = (value: string) => value.replace(/\s+/g, ' ').trim()
  const tailUserInNext = [...mergedNextMessages].reverse().find(message => message.role === 'user' && !message.hidden)
  const tailUserText = tailUserInNext ? normalize(chatMessageText(tailUserInNext)) : ''
  const tailUserRefs = tailUserInNext ? (tailUserInNext.attachmentRefs ?? []).join('\n') : ''

  const matchesTailUserInNext = (candidate: ChatMessage) =>
    Boolean(tailUserInNext) &&
    normalize(chatMessageText(candidate)) === tailUserText &&
    (candidate.attachmentRefs ?? []).join('\n') === tailUserRefs

  for (let index = 0; index < currentMessages.length; index += 1) {
    const message = currentMessages[index]

    if (message.role !== 'assistant' || !message.error || message.hidden || existingIds.has(message.id)) {
      continue
    }

    preserveIds.add(message.id)

    for (let probe = index - 1; probe >= 0; probe -= 1) {
      const candidate = currentMessages[probe]

      if (candidate.hidden) {
        continue
      }

      if (candidate.role === 'user' && !existingIds.has(candidate.id) && !matchesTailUserInNext(candidate)) {
        preserveIds.add(candidate.id)
      }

      break
    }
  }

  if (preserveIds.size === 0) {
    return mergedNextMessages
  }

  const preserved = currentMessages
    .filter(message => preserveIds.has(message.id))
    .map(message => ({ ...message, pending: false }))

  return [...mergedNextMessages, ...preserved]
}

export function branchGroupForUser(userMessage: ChatMessage): string {
  return `branch:${userMessage.id}`
}
