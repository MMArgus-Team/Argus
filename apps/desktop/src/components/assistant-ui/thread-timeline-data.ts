// Pure timeline helpers — no React/DOM; tested in thread-timeline-data.test.ts.

export interface TimelineSourceMessage {
  id: string
  role: string
  text: string
}

export interface TimelineEntry {
  id: string
  preview: string
}

// Injected as user messages for alternation; not human prompts (thread.tsx).
// Shapes come from tools/process_registry.py format_process_notification():
//   "[IMPORTANT: Background process … Output:\n…]"   brackets wrap the message
//   "[ASYNC DELEGATION BATCH COMPLETE — id]\n…"      bracket closes on line 1
// The delegation header must NOT be anchored at the end: its bracket closes on
// the first line and the report body continues below it, so the original
// `\]$`-anchored pattern never matched and async-delegation completions
// rendered as a full human user bubble.
const PROCESS_NOTIFICATION_RE = /^\[IMPORTANT: Background process [\s\S]*\]$/
const DELEGATION_NOTIFICATION_RE = /^\[ASYNC DELEGATION (?:BATCH )?COMPLETE\b/

/** Whether this user-role message was injected by the runtime, not typed. */
export function isInjectedNotification(text: string): boolean {
  const trimmed = text.trim()
  return (
    PROCESS_NOTIFICATION_RE.test(trimmed)
    || DELEGATION_NOTIFICATION_RE.test(trimmed)
  )
}

const PREVIEW_MAX = 120

export function timelinePreview(text: string, max: number = PREVIEW_MAX): string {
  const collapsed = text.replace(/\s+/g, ' ').trim()

  if (collapsed.length <= max) {
    return collapsed
  }

  return `${collapsed.slice(0, max - 1).trimEnd()}…`
}

export function deriveTimelineEntries(messages: readonly TimelineSourceMessage[]): TimelineEntry[] {
  const entries: TimelineEntry[] = []

  for (const message of messages) {
    if (message.role !== 'user') {
      continue
    }

    const text = message.text.trim()

    if (!text || isInjectedNotification(text)) {
      continue
    }

    entries.push({ id: message.id, preview: timelinePreview(text) })
  }

  return entries
}

/** Last user prompt at/above the viewport top (with slack); else first rendered. */
export function activeTimelineIndex(offsets: readonly (number | null)[], slack: number = 8): number {
  let active = -1
  let firstRendered = -1

  for (let i = 0; i < offsets.length; i++) {
    const offset = offsets[i]

    if (offset == null) {
      continue
    }

    if (firstRendered === -1) {
      firstRendered = i
    }

    if (offset <= slack) {
      active = i
    }
  }

  if (active !== -1) {
    return active
  }

  return firstRendered === -1 ? 0 : firstRendered
}
