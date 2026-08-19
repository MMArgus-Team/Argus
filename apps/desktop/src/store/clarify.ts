import { atom, computed } from 'nanostores'

import { $activeSessionId } from './session'

// Request ids the inline ClarifyTool (a tool-call message part) is currently
// mounted for and actively rendering. A clarify.request that arrives WITHOUT a
// matching tool-call part — e.g. set_monitor's internal clarify_callback, which
// fires a raw clarify.request via the backend `_block` with no tool row — has no
// inline renderer, so it would otherwise sit in $clarifyRequests forever while
// the agent blocks up to 300s. The standalone ClarifyDialog overlay renders any
// active request NOT claimed here, so those raw requests still reach the user.
const $inlineClarifyAnchors = atom<Record<string, number>>({})

/** Mark a request id as owned by an inline ClarifyTool row; returns a releaser. */
export function registerInlineClarifyAnchor(requestId: string): () => void {
  if (!requestId) {
    return () => {}
  }

  const bump = (delta: number) => {
    const all = $inlineClarifyAnchors.get()
    const next = (all[requestId] ?? 0) + delta

    if (next <= 0) {
      const rest = { ...all }
      delete rest[requestId]
      $inlineClarifyAnchors.set(rest)
    } else {
      $inlineClarifyAnchors.set({ ...all, [requestId]: next })
    }
  }

  bump(1)
  let released = false

  return () => {
    if (released) {
      return
    }
    released = true
    bump(-1)
  }
}

export interface ClarifyRequest {
  requestId: string
  question: string
  choices: string[] | null
  sessionId: string | null
}

// Pending clarify requests keyed by the runtime session id that raised them.
// Storing per-session (instead of one shared slot) lets a *background* session
// park its clarify request while the user is looking at a different chat, then
// resolve it once they switch over — without a second concurrent clarify
// clobbering the first. A request with no session id lands under the empty key.
const keyFor = (sessionId: string | null | undefined): string => sessionId ?? ''

export const $clarifyRequests = atom<Record<string, ClarifyRequest>>({})

// The clarify request for the currently-viewed session. The inline ClarifyTool
// only ever mounts inside the active session's transcript, so it reads this
// focus-scoped view rather than reaching into the whole map.
export const $clarifyRequest = computed(
  [$clarifyRequests, $activeSessionId],
  (requests, activeId) => requests[keyFor(activeId)] ?? null
)

// The active-session clarify request that NO inline ClarifyTool row is handling.
// The standalone ClarifyDialog overlay renders this so raw clarify.requests
// (set_monitor et al.) that never produce a tool-call part still reach the user
// instead of blocking the agent until the backend `_block` times out.
export const $unanchoredClarifyRequest = computed(
  [$clarifyRequest, $inlineClarifyAnchors],
  (request, anchors) => (request && !anchors[request.requestId] ? request : null)
)

export function setClarifyRequest(request: ClarifyRequest): void {
  $clarifyRequests.set({ ...$clarifyRequests.get(), [keyFor(request.sessionId)]: request })
}

export function clearClarifyRequest(requestId?: string, sessionId?: string | null): void {
  const requests = $clarifyRequests.get()

  // Targeted clear when the caller knows the session (the common path from the
  // inline ClarifyTool answering its own request).
  if (sessionId !== undefined) {
    const key = keyFor(sessionId)
    const current = requests[key]

    if (!current || (requestId && current.requestId !== requestId)) {
      return
    }

    const next = { ...requests }
    delete next[key]
    $clarifyRequests.set(next)

    return
  }

  // Fallback with no session hint: drop every entry matching the request id
  // (or clear all when none is given).
  const next: Record<string, ClarifyRequest> = {}
  let changed = false

  for (const [key, value] of Object.entries(requests)) {
    if (requestId && value.requestId !== requestId) {
      next[key] = value
    } else {
      changed = true
    }
  }

  if (changed) {
    $clarifyRequests.set(next)
  }
}
