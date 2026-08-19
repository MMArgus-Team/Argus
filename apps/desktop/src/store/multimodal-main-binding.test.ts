import type { WritableAtom } from 'nanostores'
import { beforeEach, describe, expect, it, vi } from 'vitest'

type StateHandler = (state: string) => void

class FakeGateway {
  eventHandlers = new Map<string, Set<(event: unknown) => void>>()
  stateHandlers = new Set<StateHandler>()
  connectionState = 'open'
  request = vi.fn<
    (method: string, params?: Record<string, unknown>) => Promise<Record<string, unknown>>
  >(async () => ({}))

  on<T>(name: string, handler: (event: { payload?: T }) => void): () => void {
    const handlers = this.eventHandlers.get(name) ?? new Set()
    handlers.add(handler as (event: unknown) => void)
    this.eventHandlers.set(name, handlers)

    return () => handlers.delete(handler as (event: unknown) => void)
  }

  onState(handler: StateHandler): () => void {
    this.stateHandlers.add(handler)
    handler(this.connectionState)

    return () => this.stateHandlers.delete(handler)
  }
}

let fakeGateway: FakeGateway

const { capture, micState, mockActiveSessionId, voice } = vi.hoisted(() => {
  const { atom } = require('nanostores') as {
    atom: <Value>(initial: Value) => WritableAtom<Value>
  }

  const micState = atom<'idle' | 'connecting' | 'recording'>('idle')

  return {
    capture: {
      active: false,
      ensureBound: vi.fn(async () => undefined),
      pause: vi.fn(),
      resume: vi.fn(),
      stopAndNotify: vi.fn()
    },
    micState,
    mockActiveSessionId: atom<string | null>(null),
    voice: {
      intent: false,
      rearmForRebind: vi.fn(async () => undefined),
      stop: vi.fn(async () => {
        voice.intent = false
        micState.set('idle')
      })
    }
  }
})

vi.mock('./gateway', () => ({
  $gateway: { get: () => fakeGateway }
}))
vi.mock('./session', () => ({ $activeSessionId: mockActiveSessionId }))
vi.mock('./multimodal-capture', () => ({
  ensureCaptureBoundToSession: capture.ensureBound,
  isCapturing: () => capture.active,
  pauseFrameLoop: capture.pause,
  resumeFrameLoop: capture.resume,
  stopCaptureAndNotify: capture.stopAndNotify
}))
vi.mock('./multimodal-voice', () => ({
  $mmMicState: micState,
  cancelManualMicOnDisconnect: vi.fn(),
  hasMicCaptureIntent: () => voice.intent || micState.get() !== 'idle',
  onAsrBuffer: vi.fn(),
  onAsrFinal: vi.fn(),
  onAsrPartial: vi.fn(),
  onTtsChunk: vi.fn(),
  rearmMicAfterReconnect: vi.fn(async () => undefined),
  rearmMicForSessionRebind: voice.rearmForRebind,
  stopMic: voice.stop,
  stopAllTts: vi.fn(),
  type: undefined
}))

import {
  $mmSessionId,
  bindMultimodalToMainSession,
  cancelCaptureForNextMainSessionClaim,
  claimCaptureForNextMainSession,
  clearCaptureSessionTransferClaims,
  preserveCaptureForNextRuntimeRebind
} from './multimodal'
import {
  $mmBgItems,
  $mmMonitorAlerts,
  $mmMonitors,
  $mmWatcherReports,
  $mmWatchers,
  resetDeepUi
} from './multimodal-deep'

interface Deferred<T> {
  promise: Promise<T>
  resolve: (value: T) => void
}

function deferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void

  const promise = new Promise<T>(done => {
    resolve = done
  })

  return { promise, resolve }
}

const flush = async () => {
  await Promise.resolve()
  await new Promise(resolve => setTimeout(resolve, 0))
}

describe('multimodal main-session binding', () => {
  beforeEach(() => {
    capture.active = false
    capture.ensureBound.mockClear()
    capture.pause.mockClear()
    capture.resume.mockClear()
    capture.stopAndNotify.mockClear()
    capture.stopAndNotify.mockImplementation(() => {
      capture.active = false
    })
    voice.rearmForRebind.mockClear()
    voice.stop.mockClear()
    voice.intent = false
    micState.set('idle')
    mockActiveSessionId.set(null)
    $mmSessionId.set('')
    cancelCaptureForNextMainSessionClaim()
    clearCaptureSessionTransferClaims()
    resetDeepUi()
    fakeGateway = new FakeGateway()
    bindMultimodalToMainSession()
    vi.clearAllMocks()
  })

  it("preserves an armed fresh-draft stream across '' -> runtime-B and binds it", async () => {
    capture.active = true
    claimCaptureForNextMainSession()

    mockActiveSessionId.set('runtime-B')
    await flush()

    expect($mmSessionId.get()).toBe('runtime-B')
    expect(capture.stopAndNotify).not.toHaveBeenCalled()
    expect(capture.ensureBound).toHaveBeenCalledTimes(1)
    expect(capture.ensureBound).toHaveBeenCalledWith('runtime-B')
  })

  it('stops an unclaimed fresh-draft preview when the user navigates to an existing session', async () => {
    capture.active = true

    mockActiveSessionId.set('existing-runtime-B')
    await flush()

    expect($mmSessionId.get()).toBe('existing-runtime-B')
    expect(capture.stopAndNotify).toHaveBeenCalledTimes(1)
    expect(capture.active).toBe(false)
    expect(capture.ensureBound).not.toHaveBeenCalled()
  })

  it('stops and clears capture when leaving a real old session for a fresh chat', async () => {
    mockActiveSessionId.set('runtime-A')
    await flush()
    capture.ensureBound.mockClear()
    capture.stopAndNotify.mockClear()
    capture.active = true

    mockActiveSessionId.set(null)
    await flush()

    expect($mmSessionId.get()).toBe('')
    expect(capture.stopAndNotify).toHaveBeenCalledTimes(1)
    expect(capture.active).toBe(false)
    expect($mmMonitors.get()).toEqual([])
    expect($mmWatchers.get()).toEqual([])

    mockActiveSessionId.set('runtime-B')
    await flush()

    expect(capture.stopAndNotify).toHaveBeenCalledTimes(1)
    expect(capture.ensureBound).not.toHaveBeenCalled()
  })

  it('stops capture on a direct user-driven runtime-A -> runtime-B switch', async () => {
    mockActiveSessionId.set('runtime-A')
    await flush()
    capture.stopAndNotify.mockClear()
    capture.active = true

    mockActiveSessionId.set('runtime-B')
    await flush()

    expect(capture.stopAndNotify).toHaveBeenCalledTimes(1)
    expect(capture.active).toBe(false)
    expect(capture.ensureBound).not.toHaveBeenCalledWith('runtime-B')
  })

  it('stops the old session mic on a user-driven runtime-A -> runtime-B switch', async () => {
    mockActiveSessionId.set('runtime-A')
    await flush()
    voice.stop.mockClear()
    micState.set('recording')

    mockActiveSessionId.set('runtime-B')
    await flush()

    expect(voice.stop).toHaveBeenCalledTimes(1)
    expect(voice.rearmForRebind).not.toHaveBeenCalled()
    expect(micState.get()).toBe('idle')
  })

  it('rearms a live mic for one flagged same-conversation runtime replacement', async () => {
    mockActiveSessionId.set('runtime-A')
    await flush()
    micState.set('recording')

    preserveCaptureForNextRuntimeRebind()
    mockActiveSessionId.set('runtime-A2')
    await flush()

    expect(voice.stop).not.toHaveBeenCalled()
    expect(voice.rearmForRebind).toHaveBeenCalledTimes(1)
    expect($mmSessionId.get()).toBe('runtime-A2')
  })

  it('holds a preserved mic across the empty recovery gap and rearms only after the new runtime exists', async () => {
    mockActiveSessionId.set('runtime-A')
    await flush()
    micState.set('recording')

    preserveCaptureForNextRuntimeRebind()
    mockActiveSessionId.set(null)
    await flush()

    expect(voice.stop).not.toHaveBeenCalled()
    expect(voice.rearmForRebind).not.toHaveBeenCalled()

    mockActiveSessionId.set('runtime-A2')
    await flush()

    expect(voice.stop).not.toHaveBeenCalled()
    expect(voice.rearmForRebind).toHaveBeenCalledTimes(1)
  })

  it('rearms a pending mic intent after the stale runtime attempt already returned the UI to idle', async () => {
    mockActiveSessionId.set('runtime-A')
    await flush()
    voice.intent = true
    micState.set('idle')

    preserveCaptureForNextRuntimeRebind()
    mockActiveSessionId.set(null)
    await flush()

    expect(voice.stop).not.toHaveBeenCalled()
    expect(voice.rearmForRebind).not.toHaveBeenCalled()

    mockActiveSessionId.set('runtime-A2')
    await flush()

    expect(voice.stop).not.toHaveBeenCalled()
    expect(voice.rearmForRebind).toHaveBeenCalledTimes(1)
  })

  it('does not transfer a pending A mic intent to stored conversation B after an explicit gap stop', async () => {
    mockActiveSessionId.set('runtime-A')
    await flush()
    voice.intent = true
    micState.set('idle')

    preserveCaptureForNextRuntimeRebind()
    mockActiveSessionId.set(null)
    await flush()

    // The stored-session action clears the same-conversation claim and stops
    // the old owner synchronously before it starts resolving/resuming B.
    clearCaptureSessionTransferClaims()
    await voice.stop()
    mockActiveSessionId.set('runtime-B')
    await flush()

    expect(voice.intent).toBe(false)
    expect(voice.rearmForRebind).not.toHaveBeenCalled()
    expect($mmSessionId.get()).toBe('runtime-B')
  })

  it('preserves one flagged same-conversation runtime replacement, then consumes the flag', async () => {
    mockActiveSessionId.set('runtime-A')
    await flush()
    capture.stopAndNotify.mockClear()
    capture.active = true

    preserveCaptureForNextRuntimeRebind()
    mockActiveSessionId.set('runtime-A2')
    await flush()

    expect(capture.stopAndNotify).not.toHaveBeenCalled()
    expect(capture.ensureBound).toHaveBeenCalledWith('runtime-A2')

    capture.stopAndNotify.mockClear()
    capture.ensureBound.mockClear()
    mockActiveSessionId.set('runtime-B')
    await flush()

    expect(capture.stopAndNotify).toHaveBeenCalledTimes(1)
    expect(capture.ensureBound).not.toHaveBeenCalledWith('runtime-B')
  })

  it('stops the next real switch after a same-SID recovery explicitly consumes preservation', async () => {
    mockActiveSessionId.set('runtime-A')
    await flush()
    capture.active = true

    // A transport recovery can reopen the same live runtime id, so nanostores
    // emits no id change. Its owner explicitly consumes the one-shot transfer
    // claim when that recovery finishes.
    preserveCaptureForNextRuntimeRebind()
    mockActiveSessionId.set('runtime-A')
    await flush()
    clearCaptureSessionTransferClaims()

    capture.stopAndNotify.mockClear()
    capture.ensureBound.mockClear()
    mockActiveSessionId.set('runtime-B')
    await flush()

    expect(capture.stopAndNotify).toHaveBeenCalledTimes(1)
    expect(capture.ensureBound).not.toHaveBeenCalledWith('runtime-B')
  })

  it('preserves a flagged runtime-A -> empty -> runtime-A2 reconnect sequence', async () => {
    mockActiveSessionId.set('runtime-A')
    await flush()
    capture.stopAndNotify.mockClear()
    capture.ensureBound.mockClear()
    capture.active = true

    preserveCaptureForNextRuntimeRebind()
    mockActiveSessionId.set(null)
    await flush()
    expect(capture.stopAndNotify).not.toHaveBeenCalled()

    mockActiveSessionId.set('runtime-A2')
    await flush()

    expect(capture.stopAndNotify).not.toHaveBeenCalled()
    expect(capture.ensureBound).toHaveBeenCalledWith('runtime-A2')
  })

  it('ignores registry and sidechannel responses from the previous session', async () => {
    const pending = new Map<string, Deferred<Record<string, unknown>>>()

    const requestFor = (method: string, sid: string) => {
      const key = `${sid}:${method}`
      const item = deferred<Record<string, unknown>>()
      pending.set(key, item)

      return item.promise
    }

    fakeGateway.request.mockImplementation(
      (method: string, params?: Record<string, unknown>) =>
        requestFor(method, String(params?.session_id || ''))
    )

    mockActiveSessionId.set('runtime-A')
    await Promise.resolve()
    mockActiveSessionId.set('runtime-B')
    await Promise.resolve()

    pending.get('runtime-B:multimodal.list_registries')!.resolve({
      monitors: [{ monitor_id: 'monitor-B' }],
      watchers: [{ watcher_id: 'watcher-B' }]
    })
    pending.get('runtime-B:multimodal.list_monitor_alerts')!.resolve({
      alerts: [{
        monitor_id: 'monitor-B',
        text: 'alert-B',
        wall_ts: 2,
        evidence: {
          input_count: 9,
          frames: [{ ts: 4, source_type: 'screen', thumb_b64: 'dGh1bWI=' }]
        }
      }]
    })
    await flush()
    pending.get('runtime-B:multimodal.list_watcher_content')!.resolve({
      finals: [],
      reports: [{ watcher_id: 'watcher-B', round_idx: 1, text: 'report-B', wall_ts: 2 }]
    })
    await flush()

    expect($mmMonitors.get().map(item => item.monitor_id)).toEqual(['monitor-B'])
    expect($mmWatchers.get().map(item => item.watcher_id)).toEqual(['watcher-B'])
    expect($mmMonitorAlerts.get()['monitor-B']?.map(item => item.text)).toEqual(['alert-B'])
    expect($mmMonitorAlerts.get()['monitor-B']?.[0]?.evidence).toEqual({
      input_count: 9,
      shown_count: 1,
      frames: [{ ts: 4, source_type: 'screen', thumb_b64: 'dGh1bWI=' }]
    })
    expect($mmWatcherReports.get()['watcher-B']?.map(item => item.text)).toEqual(['report-B'])
    expect($mmBgItems.get().map(item => item.requestId)).toEqual(['watcher-B'])

    pending.get('runtime-A:multimodal.list_registries')!.resolve({
      monitors: [{ monitor_id: 'monitor-A-late' }],
      watchers: [{ watcher_id: 'watcher-A-late' }]
    })
    pending.get('runtime-A:multimodal.list_monitor_alerts')!.resolve({
      alerts: [{ monitor_id: 'monitor-A-late', text: 'alert-A-late', wall_ts: 1 }]
    })
    await flush()

    expect($mmMonitors.get().map(item => item.monitor_id)).toEqual(['monitor-B'])
    expect($mmWatchers.get().map(item => item.watcher_id)).toEqual(['watcher-B'])
    expect(Object.keys($mmMonitorAlerts.get())).toEqual(['monitor-B'])
    expect(Object.keys($mmWatcherReports.get())).toEqual(['watcher-B'])
    expect($mmBgItems.get().map(item => item.requestId)).toEqual(['watcher-B'])
    expect(fakeGateway.request).not.toHaveBeenCalledWith(
      'multimodal.list_watcher_content',
      { session_id: 'runtime-A' }
    )
  })

  it('ignores old watcher content that finishes after the new session is hydrated', async () => {
    const oldWatcherContent = deferred<Record<string, unknown>>()
    fakeGateway.request.mockImplementation(
      async (method: string, params?: Record<string, unknown>) => {
        const sid = String(params?.session_id || '')

        if (sid === 'runtime-A') {
          if (method === 'multimodal.list_registries') {return { monitors: [], watchers: [] }}

          if (method === 'multimodal.list_monitor_alerts') {return { alerts: [] }}

          if (method === 'multimodal.list_watcher_content') {return oldWatcherContent.promise}
        }

        if (method === 'multimodal.list_registries') {
          return {
            monitors: [{ monitor_id: 'monitor-B' }],
            watchers: [{ watcher_id: 'watcher-B' }]
          }
        }

        if (method === 'multimodal.list_monitor_alerts') {return { alerts: [] }}

        if (method === 'multimodal.list_watcher_content') {
          return {
            finals: [],
            reports: [{ watcher_id: 'watcher-B', round_idx: 1, text: 'report-B', wall_ts: 2 }]
          }
        }

        return {}
      }
    )

    mockActiveSessionId.set('runtime-A')
    await flush()
    expect(fakeGateway.request).toHaveBeenCalledWith(
      'multimodal.list_watcher_content',
      { session_id: 'runtime-A' }
    )

    mockActiveSessionId.set('runtime-B')
    await flush()
    expect($mmWatcherReports.get()['watcher-B']?.map(item => item.text)).toEqual(['report-B'])

    oldWatcherContent.resolve({
      finals: [],
      reports: [{ watcher_id: 'watcher-A', round_idx: 1, text: 'late-A', wall_ts: 1 }]
    })
    await flush()

    expect(Object.keys($mmWatcherReports.get())).toEqual(['watcher-B'])
    expect($mmBgItems.get().map(item => item.requestId)).toEqual(['watcher-B'])
  })
})
