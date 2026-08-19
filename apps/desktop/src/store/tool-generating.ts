import { atom, computed } from 'nanostores'

import { $activeSessionId } from './session'

// Per-session "the model is writing a tool call's arguments right now" signal,
// fed by the gateway's `tool.generating` event.
//
// Why this exists as a separate store instead of a message part: `tool.generating`
// arrives with `tool_id=None`, so it cannot be merged into (or later reconciled
// with) the authoritative row that `tool.start` creates. Turning it into a
// tool-call part is what produced the old "two identical tool rows" bug — an
// orphaned id-less row the id-bearing `tool.start` couldn't always match.
//
// So it is deliberately NOT part of the transcript. It is transient UI state that
// only fills the gap between "model began emitting a tool call" and `tool.start`
// (which the backend only fires after the arguments are fully written AND the
// guardrail/plugin/checkpoint preflight has run). Without it, that whole window
// is silent — the process block has no parts yet, so a turn can look idle for
// seconds and then appear with several tools already listed.
//
// Per-session so a background chat can't clobber the foreground view.
const keyFor = (sessionId: string | null | undefined): string => sessionId ?? ''

export const $generatingToolBySession = atom<Record<string, string>>({})

/** Tool name the active session is currently writing arguments for, or ''. */
export const $generatingToolName = computed(
  [$generatingToolBySession, $activeSessionId],
  (sessions, activeId) => sessions[keyFor(activeId)] ?? ''
)

export function setSessionGeneratingTool(sessionId: string | null | undefined, toolName: string): void {
  const key = keyFor(sessionId)
  const name = toolName.trim()
  const sessions = $generatingToolBySession.get()

  if (!name) {
    clearSessionGeneratingTool(sessionId)

    return
  }

  if (sessions[key] === name) {
    return
  }

  $generatingToolBySession.set({ ...sessions, [key]: name })
}

export function clearSessionGeneratingTool(sessionId: string | null | undefined): void {
  const key = keyFor(sessionId)
  const sessions = $generatingToolBySession.get()

  if (!(key in sessions)) {
    return
  }

  const next = { ...sessions }
  delete next[key]
  $generatingToolBySession.set(next)
}

export function resetGeneratingTools(): void {
  if (Object.keys($generatingToolBySession.get()).length === 0) {
    return
  }

  $generatingToolBySession.set({})
}
