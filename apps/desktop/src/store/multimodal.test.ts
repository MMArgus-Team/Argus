import { beforeEach, describe, expect, it, vi } from 'vitest'

// ── Fake gateway ────────────────────────────────────────────────────────────
// The store reads $gateway from ./gateway; we back it with a controllable fake
// so we can drive onState transitions and count session.create calls.
type StateHandler = (s: string) => void

class FakeGateway {
  stateHandlers = new Set<StateHandler>()
  eventHandlers = new Map<string, Set<(ev: unknown) => void>>()
  createCalls = 0
  private _state = 'connecting'

  onState(h: StateHandler): () => void {
    this.stateHandlers.add(h)
    h(this._state) // fires immediately with current state (matches real client)
    return () => this.stateHandlers.delete(h)
  }

  on<T>(name: string, h: (ev: { payload?: T }) => void): () => void {
    const set = this.eventHandlers.get(name) ?? new Set()
    set.add(h as (ev: unknown) => void)
    this.eventHandlers.set(name, set)
    return () => set.delete(h as (ev: unknown) => void)
  }

  request = vi.fn(async (method: string) => {
    if (method === 'session.create') {
      this.createCalls += 1
      return { session_id: `sid-${this.createCalls}` }
    }
    return {}
  })

  // test helper: transition state and notify subscribers
  setState(s: string): void {
    this._state = s
    for (const h of this.stateHandlers) h(s)
  }
}

let fakeGw: FakeGateway

vi.mock('./gateway', () => ({
  $gateway: { get: () => fakeGw }
}))

// Capture/voice pull in browser media APIs; stub them for the store unit test.
// `_capturing` lets tests flip the "background capture active" signal that
// resetMultimodalUi keys off.
let _capturing = false
export const _setCapturing = (v: boolean) => {
  _capturing = v
}
vi.mock('./multimodal-capture', () => ({
  isCapturing: () => _capturing,
  pauseFrameLoop: vi.fn(),
  resumeFrameLoop: vi.fn()
}))
// The store follows the main chat's $activeSessionId in bound mode. The unit
// test only exercises the standalone (dedicated-session) path, so a minimal
// $activeSessionId atom keeps the heavy session module out of the test graph.
// vi.hoisted runs before the hoisted vi.mock factory + before ./multimodal is
// imported, so the atom is initialized when ./session's mock factory runs.
// Both atoms are created in vi.hoisted so they're initialized before the hoisted
// vi.mock factories + the hoisted `import ./multimodal` run (which import
// ./session and ./multimodal-voice in that order during module evaluation).
const { mockActiveSessionId, _micState } = vi.hoisted(() => {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const { atom } = require('nanostores') as typeof import('nanostores')
  return {
    mockActiveSessionId: atom<string | null>(null),
    _micState: atom<'idle' | 'connecting' | 'recording'>('idle')
  }
})
vi.mock('./session', () => ({ $activeSessionId: mockActiveSessionId }))
vi.mock('./multimodal-voice', () => ({
  $mmMicState: _micState,
  cancelManualMicOnDisconnect: vi.fn(),
  hasMicCaptureIntent: () => _micState.get() !== 'idle',
  rearmMicAfterReconnect: vi.fn(async () => undefined),
  onAsrPartial: vi.fn(),
  onAsrFinal: vi.fn(),
  onTtsChunk: vi.fn(),
  stopAllTts: vi.fn(),
  type: undefined
}))

// Import AFTER the mock is registered.
import {
  $mmConnState,
  $mmSessionId,
  attachMultimodalGateway,
  ensureMultimodalSession,
  resetMultimodalUi
} from './multimodal'

// small async flush so awaited ensureMultimodalSession() settles
const flush = () => new Promise(r => setTimeout(r, 0))

describe('multimodal connection lifecycle (stage 1)', () => {
  beforeEach(() => {
    fakeGw = new FakeGateway()
    resetMultimodalUi()
    $mmSessionId.set('')
    $mmConnState.set('connecting')
  })

  it('session.create passes close_on_disconnect + source=tool', async () => {
    await ensureMultimodalSession()
    expect(fakeGw.request).toHaveBeenCalledWith(
      'session.create',
      expect.objectContaining({ close_on_disconnect: true, source: 'tool' })
    )
    expect($mmSessionId.get()).toBe('sid-1')
  })

  it('badge goes open on connect, reconnecting on drop, open again on recovery', async () => {
    attachMultimodalGateway()
    await ensureMultimodalSession() // first (initial) session
    fakeGw.setState('open')
    expect($mmConnState.get()).toBe('open')

    fakeGw.setState('closed')
    expect($mmConnState.get()).toBe('reconnecting') // dropped after having been open
    expect($mmSessionId.get()).toBe('') // stale session id cleared

    fakeGw.setState('open')
    await flush()
    expect($mmConnState.get()).toBe('open')
  })

  it('rebuilds the session on reconnect (open after a prior open)', async () => {
    attachMultimodalGateway()
    await ensureMultimodalSession()
    fakeGw.setState('open')
    const afterInitial = fakeGw.createCalls

    fakeGw.setState('closed') // drop
    fakeGw.setState('open') // reconnect
    await flush()

    expect(fakeGw.createCalls).toBe(afterInitial + 1) // one rebuild
    expect($mmSessionId.get()).not.toBe('') // fresh session id set
  })

  it('does NOT rebuild on the very first open (initial connect, not a reconnect)', async () => {
    attachMultimodalGateway()
    // first ever 'open' with no prior session — onState should not itself create
    fakeGw.setState('open')
    await flush()
    // ensureMultimodalSession was never called by us here; onState must not
    // fabricate a session on the initial open.
    expect(fakeGw.createCalls).toBe(0)
    expect($mmConnState.get()).toBe('open')
  })

  it('initial connecting state shows connecting, not reconnecting', () => {
    attachMultimodalGateway() // fake starts in 'connecting', never opened
    expect($mmConnState.get()).toBe('connecting')
  })

  // ── Background form-factor: reset must NOT tear down handlers while capture
  //    or mic is running (hide-to-tray keeps the background session alive). ──
  it('resetMultimodalUi keeps gateway handlers wired when capture is active', async () => {
    attachMultimodalGateway()
    await ensureMultimodalSession()
    fakeGw.setState('open')
    _setCapturing(true) // simulate background screen/camera capture running

    resetMultimodalUi() // page unmounts (hidden to tray) but capture continues

    // Handlers still live: a reconnect must still rebuild the session so
    // background frames keep flowing (the goal-critical bug this guards).
    const before = fakeGw.createCalls
    fakeGw.setState('closed')
    fakeGw.setState('open')
    await flush()
    expect(fakeGw.createCalls).toBe(before + 1) // reconnect rebuilt the session
    _setCapturing(false)
  })

  it('resetMultimodalUi fully tears down when nothing is active in background', async () => {
    attachMultimodalGateway()
    await ensureMultimodalSession()
    fakeGw.setState('open')
    _setCapturing(false)
    _micState.set('idle')

    resetMultimodalUi() // nothing running → full teardown

    // Handlers gone: a later reconnect does NOT rebuild (no background work).
    const before = fakeGw.createCalls
    fakeGw.setState('closed')
    fakeGw.setState('open')
    await flush()
    expect(fakeGw.createCalls).toBe(before) // no rebuild after full teardown
  })
})
