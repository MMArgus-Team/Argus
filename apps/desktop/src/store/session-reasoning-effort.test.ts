/**
 * A persisted thinking-OFF must not become the global default.
 *
 * `$currentReasoningEffort` is GLOBAL sticky state, shipped as
 * `reasoning_effort` on every `session.create`. Per-model memory lives elsewhere
 * (`argus.desktop.model-presets`). So persisting "none" here meant: turning
 * thinking off once, for one model, silently disabled thinking for EVERY model in
 * EVERY later session — across restarts, with no visible cause.
 *
 * The user-visible symptom was an empty, unexpandable Thinking block: the gateway
 * parses "none" → `{enabled: false}`, the provider receives
 * `thinking: {type: "disabled"}`, no `reasoning.delta` is ever emitted, and the
 * ☁️ disclosure renders nothing. The block was a faithful view of a genuinely
 * disabled model, which is why it looked like a UI bug but wasn't.
 *
 * Turning thinking off for the CURRENT chat must still work — that is the whole
 * point of the control. Only the cross-session persistence of "none" is wrong.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest'

const EFFORT_KEY = 'argus.desktop.composer.reasoning-effort'

// The atom is seeded at module load, so each seeding case needs a fresh import.
const freshStore = async () => {
  vi.resetModules()

  return import('./session')
}

beforeEach(() => {
  window.localStorage.clear()
  vi.resetModules()
})

describe('$currentReasoningEffort seeding', () => {
  it('drops a persisted "none" so thinking is not disabled forever', async () => {
    window.localStorage.setItem(EFFORT_KEY, 'none')

    const { $currentReasoningEffort } = await freshStore()

    // '' = "no explicit level" → the model preset / backend default decides.
    expect($currentReasoningEffort.get()).toBe('')
  })

  it.each(['NONE', ' none ', 'None'])('drops "%s" regardless of case/whitespace', async stored => {
    window.localStorage.setItem(EFFORT_KEY, stored)

    const { $currentReasoningEffort } = await freshStore()

    expect($currentReasoningEffort.get()).toBe('')
  })

  it('still restores a real effort level', async () => {
    window.localStorage.setItem(EFFORT_KEY, 'high')

    const { $currentReasoningEffort } = await freshStore()

    expect($currentReasoningEffort.get()).toBe('high')
  })

  it('is empty when nothing was persisted', async () => {
    const { $currentReasoningEffort } = await freshStore()

    expect($currentReasoningEffort.get()).toBe('')
  })
})

describe('setCurrentReasoningEffort', () => {
  it('applies "none" to the live session but does not persist it', async () => {
    const { $currentReasoningEffort, setCurrentReasoningEffort } = await freshStore()

    setCurrentReasoningEffort('none')

    // THIS chat has thinking off — the control must keep working.
    expect($currentReasoningEffort.get()).toBe('none')
    // ...but the next launch must not inherit it.
    expect(window.localStorage.getItem(EFFORT_KEY)).toBeNull()
  })

  it('clears a previously stored level when switching to "none"', async () => {
    window.localStorage.setItem(EFFORT_KEY, 'high')

    const { setCurrentReasoningEffort } = await freshStore()

    setCurrentReasoningEffort('none')

    expect(window.localStorage.getItem(EFFORT_KEY)).toBeNull()
  })

  it('persists a real effort level', async () => {
    const { $currentReasoningEffort, setCurrentReasoningEffort } = await freshStore()

    setCurrentReasoningEffort('low')

    expect($currentReasoningEffort.get()).toBe('low')
    expect(window.localStorage.getItem(EFFORT_KEY)).toBe('low')
  })

  it('round-trips: setting "none" then relaunching yields no explicit level', async () => {
    const first = await freshStore()

    first.setCurrentReasoningEffort('none')

    const { $currentReasoningEffort } = await freshStore()

    expect($currentReasoningEffort.get()).toBe('')
  })
})
