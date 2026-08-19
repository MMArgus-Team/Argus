import type { WritableAtom } from 'nanostores'
import { beforeEach, describe, expect, it, vi } from 'vitest'

type EventHandler = (event: {
  payload?: unknown
  session_id?: string
}) => void

class FakeGateway {
  eventHandlers = new Map<string, Set<EventHandler>>()
  stateHandlers = new Set<(state: string) => void>()
  connectionState = 'open'

  request = vi.fn<
    (method: string, params?: Record<string, unknown>, timeoutMs?: number) => Promise<Record<string, unknown>>
  >(async (method: string) => {
    if (method === 'multimodal.list_registries') {
      return { ready: true, monitors: [], watchers: [] }
    }

    if (method === 'multimodal.list_monitor_alerts') {
      return { alerts: [] }
    }

    if (method === 'multimodal.list_watcher_content') {
      return { reports: [], finals: [] }
    }

    return {}
  })

  on<T>(name: string, handler: (event: { payload?: T; session_id?: string }) => void): () => void {
    const handlers = this.eventHandlers.get(name) ?? new Set<EventHandler>()

    handlers.add(handler as EventHandler)
    this.eventHandlers.set(name, handlers)

    return () => handlers.delete(handler as EventHandler)
  }

  onState(handler: (state: string) => void): () => void {
    this.stateHandlers.add(handler)
    handler(this.connectionState)

    return () => this.stateHandlers.delete(handler)
  }

  emitState(state: string): void {
    this.connectionState = state
    for (const handler of this.stateHandlers) {
      handler(state)
    }
  }

  emit<T>(name: string, payload: T, sessionId?: string): void {
    const event = sessionId === undefined
      ? { payload }
      : { payload, session_id: sessionId }

    for (const handler of this.eventHandlers.get(name) ?? []) {
      handler(event)
    }
  }
}

let fakeGateway: FakeGateway

const { capture, mockActiveSessionId, micState, voice } = vi.hoisted(() => {
  const { atom } = require('nanostores') as {
    atom: <Value>(initial: Value) => WritableAtom<Value>
  }

  return {
    capture: { active: false },
    mockActiveSessionId: atom<string | null>(null),
    micState: atom<'idle' | 'connecting' | 'recording'>('idle'),
    voice: {
      asrBuffer: vi.fn(),
      asrPartial: vi.fn(),
      asrFinal: vi.fn(),
      tts: vi.fn(),
      stopAll: vi.fn()
    }
  }
})

vi.mock('./gateway', () => ({ $gateway: { get: () => fakeGateway } }))
vi.mock('./session', () => ({ $activeSessionId: mockActiveSessionId }))
vi.mock('./multimodal-capture', () => ({
  ensureCaptureBoundToSession: vi.fn(async () => undefined),
  isCapturing: () => capture.active,
  pauseFrameLoop: vi.fn(),
  resumeFrameLoop: vi.fn(),
  stopCaptureAndNotify: vi.fn()
}))
vi.mock('./multimodal-voice', () => ({
  $mmMicState: micState,
  cancelManualMicOnDisconnect: vi.fn(),
  hasMicCaptureIntent: () => micState.get() !== 'idle',
  onAsrBuffer: voice.asrBuffer,
  onAsrPartial: voice.asrPartial,
  onAsrFinal: voice.asrFinal,
  onTtsChunk: voice.tts,
  rearmMicAfterReconnect: vi.fn(async () => undefined),
  rearmMicForSessionRebind: vi.fn(async () => undefined),
  stopMic: vi.fn(async () => undefined),
  stopAllTts: voice.stopAll,
  type: undefined
}))

import {
  $mmAnchor,
  $mmCtx,
  $mmMessages,
  $mmQueryTrajectory,
  $mmSessionId,
  attachMultimodalGateway,
  bindMultimodalToMainSession,
  queryTrajectoryTaskStore,
  resetMultimodalUi
} from './multimodal'
import {
  $mmBgItems,
  $mmMonitors,
  $mmToasts,
  $mmWatchers,
  resetDeepUi
} from './multimodal-deep'

const waitForEventFlush = () => new Promise(resolve => setTimeout(resolve, 110))

const flushPromises = async () => {
  await Promise.resolve()
  await new Promise(resolve => setTimeout(resolve, 0))
}

function queryTrajectory(id: string, seq: number, taskId: string) {
  return {
    event: 'multimodal.trajectory',
    id,
    payload: { task_id: taskId },
    phase: 'started',
    seq,
    ts: seq,
    worker: 'QueryWorker'
  }
}

function deferred<T>() {
  let resolve!: (value: T) => void

  const promise = new Promise<T>(done => {
    resolve = done
  })

  return { promise, resolve }
}

describe('multimodal gateway event session isolation', () => {
  beforeEach(() => {
    capture.active = false
    micState.set('idle')
    mockActiveSessionId.set(null)
    fakeGateway = new FakeGateway()
    resetMultimodalUi()
    $mmSessionId.set('')
    resetDeepUi()
    $mmToasts.set([])
    voice.asrBuffer.mockClear()
    voice.asrPartial.mockClear()
    voice.asrFinal.mockClear()
    voice.tts.mockClear()
    bindMultimodalToMainSession()
  })

  it('drops delayed A events after binding B across every session-scoped MM surface', async () => {
    mockActiveSessionId.set('runtime-B')
    await flushPromises()

    fakeGateway.emit('multimodal.bg', {
      type: 'segment_start', request_id: 'watcher-A', seg: 1
    }, 'runtime-A')
    fakeGateway.emit('watcher.final', {
      request_id: 'watcher-A', text: 'old final'
    }, 'runtime-A')
    fakeGateway.emit('multimodal.ctx', {
      version: 9, obs: [{ text: 'old observation' }]
    }, 'runtime-A')
    fakeGateway.emit('multimodal.anchor', {
      frames: [{ ts: 1, jpeg_b64: 'old-frame' }]
    }, 'runtime-A')
    fakeGateway.emit('multimodal.monitors', {
      monitors: [{ monitor_id: 'monitor-A', enabled: true }]
    }, 'runtime-A')
    fakeGateway.emit('multimodal.watchers', {
      watchers: [{ watcher_id: 'watcher-A', status: 'running' }]
    }, 'runtime-A')
    fakeGateway.emit('multimodal.toast', { text: 'old toast' }, 'runtime-A')
    fakeGateway.emit('clarify.request', {
      request_id: 'clarify-A', question: 'old question'
    }, 'runtime-A')
    fakeGateway.emit('multimodal.asr_partial', { text: 'old partial' }, 'runtime-A')
    fakeGateway.emit('multimodal.asr_buffer', { segments: ['old segment'] }, 'runtime-A')
    fakeGateway.emit('multimodal.asr_final', { text: 'old final' }, 'runtime-A')
    fakeGateway.emit('multimodal.tts', { pcm_b64: 'old audio' }, 'runtime-A')

    await waitForEventFlush()

    expect($mmBgItems.get()).toEqual([])
    expect($mmCtx.get()).toEqual({ version: 0, obs: [], audioObs: [], facts: {} })
    expect($mmAnchor.get()).toEqual([])
    expect($mmMonitors.get()).toEqual([])
    expect($mmWatchers.get()).toEqual([])
    expect($mmToasts.get()).toEqual([])
    expect($mmMessages.get().some(message => message.clarifyReqId === 'clarify-A')).toBe(false)
    expect(voice.asrBuffer).not.toHaveBeenCalled()
    expect(voice.asrPartial).not.toHaveBeenCalled()
    expect(voice.asrFinal).not.toHaveBeenCalled()
    expect(voice.tts).not.toHaveBeenCalled()

    fakeGateway.emit('multimodal.ctx', {
      version: 1, obs: [{ text: 'current observation' }]
    }, 'runtime-B')
    fakeGateway.emit('multimodal.anchor', {
      frames: [{ ts: 2, jpeg_b64: 'current-frame' }]
    }, 'runtime-B')
    fakeGateway.emit('multimodal.monitors', {
      monitors: [{ monitor_id: 'monitor-B', enabled: true }]
    }, 'runtime-B')
    fakeGateway.emit('multimodal.asr_buffer', {
      segments: ['current segment'],
      turn_id: 'turn-current'
    }, 'runtime-B')

    expect($mmCtx.get().obs[0]?.text).toBe('current observation')
    expect($mmAnchor.get()[0]?.jpeg_b64).toBe('current-frame')
    expect($mmMonitors.get()[0]?.monitor_id).toBe('monitor-B')
    expect(voice.asrBuffer).toHaveBeenCalledWith(['current segment'], 'turn-current')
  })

  it('clears already-accepted A stream buffers at the A to B boundary', async () => {
    mockActiveSessionId.set('runtime-A')
    await flushPromises()
    voice.stopAll.mockClear()

    fakeGateway.emit('multimodal.bg', {
      type: 'segment_start', request_id: 'queued-A', seg: 1
    }, 'runtime-A')
    // The bg event is accepted into the 80ms buffer, but must not flush after
    // the runtime switches and its visible deep state has been reset.
    mockActiveSessionId.set('runtime-B')

    await waitForEventFlush()
    expect($mmBgItems.get()).toEqual([])
    expect(voice.stopAll).toHaveBeenCalledTimes(1)
  })

  it('accepts legacy unscoped bootstrap events only before a session is bound', () => {
    fakeGateway.emit('multimodal.ctx', {
      version: 1, obs: [{ text: 'bootstrap' }]
    })
    expect($mmCtx.get().obs[0]?.text).toBe('bootstrap')

    mockActiveSessionId.set('runtime-B')
    fakeGateway.emit('multimodal.ctx', {
      version: 2, obs: [{ text: 'ambiguous' }]
    })
    expect($mmCtx.get()).toEqual({ version: 0, obs: [], audioObs: [], facts: {} })
  })

  it('accepts live QueryWorker trajectory only for the active runtime', async () => {
    mockActiveSessionId.set('runtime-A')
    await flushPromises()

    fakeGateway.emit('multimodal.trajectory', queryTrajectory('A-1', 1, 'qry_A'), 'runtime-A')
    fakeGateway.emit('multimodal.trajectory', queryTrajectory('B-foreign', 2, 'qry_B'), 'runtime-B')

    expect($mmQueryTrajectory.get().map(row => row.id)).toEqual(['A-1'])

    mockActiveSessionId.set('runtime-B')
    expect($mmQueryTrajectory.get()).toEqual([])

    fakeGateway.emit('multimodal.trajectory', queryTrajectory('A-late', 3, 'qry_A'), 'runtime-A')
    fakeGateway.emit('multimodal.trajectory', queryTrajectory('B-1', 4, 'qry_B'), 'runtime-B')

    expect($mmQueryTrajectory.get().map(row => row.id)).toEqual(['B-1'])
  })

  it('keeps inline QueryWorker subscriptions isolated by task', async () => {
    const taskA = queryTrajectoryTaskStore('qry_A')
    const taskB = queryTrajectoryTaskStore('qry_B')

    mockActiveSessionId.set('runtime-A')
    await flushPromises()
    const emptyB = taskB.get()

    fakeGateway.emit('multimodal.trajectory', queryTrajectory('A-1', 1, 'qry_A'), 'runtime-A')
    expect(taskA.get().map(row => row.id)).toEqual(['A-1'])
    expect(taskB.get()).toBe(emptyB)

    fakeGateway.emit('multimodal.trajectory', queryTrajectory('A-2', 2, 'qry_A'), 'runtime-A')
    expect(taskA.get().map(row => row.id)).toEqual(['A-1', 'A-2'])
    expect(taskB.get()).toBe(emptyB)
  })

  it('rehydrates QueryWorker trajectory when reconnect reuses the same runtime id', async () => {
    fakeGateway.request.mockImplementation(async (method: string) => {
      if (method === 'multimodal.trajectory.list') {
        return { entries: [queryTrajectory('A-restored', 8, 'qry_A')] }
      }

      if (method === 'multimodal.list_registries') {
        return { ready: true, monitors: [], watchers: [] }
      }

      if (method === 'multimodal.list_monitor_alerts') {
        return { alerts: [] }
      }

      if (method === 'multimodal.list_watcher_content') {
        return { reports: [], finals: [] }
      }

      return {}
    })

    mockActiveSessionId.set('runtime-A')
    await flushPromises()
    fakeGateway.emit('multimodal.trajectory', queryTrajectory('A-live', 7, 'qry_A'), 'runtime-A')

    fakeGateway.emitState('reconnecting')
    expect($mmQueryTrajectory.get()).toEqual([])

    fakeGateway.emitState('open')
    await flushPromises()

    expect(fakeGateway.request).toHaveBeenCalledWith(
      'multimodal.trajectory.list',
      { limit: 2000, session_id: 'runtime-A' },
      60_000
    )
    expect($mmQueryTrajectory.get().map(row => row.id)).toEqual(['A-restored'])
  })

  it('rejects a pre-disconnect hydrate after same-sid reconnect starts a new epoch', async () => {
    const beforeDisconnect = deferred<{ entries: unknown[] }>()
    const afterReconnect = deferred<{ entries: unknown[] }>()
    let trajectoryPulls = 0

    fakeGateway.request.mockImplementation(async (method: string) => {
      if (method === 'multimodal.trajectory.list') {
        trajectoryPulls += 1

        return trajectoryPulls === 1 ? beforeDisconnect.promise : afterReconnect.promise
      }

      if (method === 'multimodal.list_registries') {
        return { ready: true, monitors: [], watchers: [] }
      }

      if (method === 'multimodal.list_monitor_alerts') {
        return { alerts: [] }
      }

      if (method === 'multimodal.list_watcher_content') {
        return { reports: [], finals: [] }
      }

      return {}
    })

    mockActiveSessionId.set('runtime-A')
    await flushPromises()
    fakeGateway.emitState('reconnecting')
    fakeGateway.emitState('open')
    await flushPromises()

    afterReconnect.resolve({ entries: [queryTrajectory('new-epoch', 2, 'qry_A')] })
    await flushPromises()
    expect($mmQueryTrajectory.get().map(row => row.id)).toEqual(['new-epoch'])

    beforeDisconnect.resolve({ entries: [queryTrajectory('stale-epoch', 1, 'qry_A')] })
    await flushPromises()
    expect($mmQueryTrajectory.get().map(row => row.id)).toEqual(['new-epoch'])
  })

  it('clears profile-owned trajectory synchronously when the gateway identity changes', async () => {
    mockActiveSessionId.set('runtime-same')
    await flushPromises()
    fakeGateway.emit(
      'multimodal.trajectory',
      queryTrajectory('profile-A', 1, 'qry_A'),
      'runtime-same'
    )
    expect($mmQueryTrajectory.get().map(row => row.id)).toEqual(['profile-A'])

    const oldGateway = fakeGateway
    fakeGateway = new FakeGateway()
    attachMultimodalGateway()

    expect($mmQueryTrajectory.get()).toEqual([])
    oldGateway.emit(
      'multimodal.trajectory',
      queryTrajectory('profile-A-late', 2, 'qry_A'),
      'runtime-same'
    )
    expect($mmQueryTrajectory.get()).toEqual([])
  })

  it('rejects a slow A hydrate after B wins, requests 2000 rows, and deduplicates B list/live entries', async () => {
    const hydrateA = deferred<{ entries: unknown[] }>()
    const hydrateB = deferred<{ entries: unknown[] }>()

    fakeGateway.request.mockImplementation(async (method: string, params?: Record<string, unknown>) => {
      if (method === 'multimodal.trajectory.list') {
        if (params?.session_id === 'runtime-A') {
          return hydrateA.promise
        }

        if (params?.session_id === 'runtime-B') {
          return hydrateB.promise
        }
      }

      if (method === 'multimodal.list_registries') {
        return { ready: true, monitors: [], watchers: [] }
      }

      if (method === 'multimodal.list_monitor_alerts') {
        return { alerts: [] }
      }

      if (method === 'multimodal.list_watcher_content') {
        return { reports: [], finals: [] }
      }

      return {}
    })

    mockActiveSessionId.set('runtime-A')
    await flushPromises()
    mockActiveSessionId.set('runtime-B')
    await flushPromises()

    expect(fakeGateway.request).toHaveBeenCalledWith(
      'multimodal.trajectory.list',
      { limit: 2000, session_id: 'runtime-A' },
      60_000
    )
    expect(fakeGateway.request).toHaveBeenCalledWith(
      'multimodal.trajectory.list',
      { limit: 2000, session_id: 'runtime-B' },
      60_000
    )

    fakeGateway.emit('multimodal.trajectory', queryTrajectory('B-shared', 2, 'qry_B'), 'runtime-B')
    hydrateB.resolve({
      entries: [
        queryTrajectory('B-older', 1, 'qry_B'),
        { ...queryTrajectory('B-shared', 2, 'qry_B'), phase: 'listed-stale-copy' }
      ]
    })
    await flushPromises()

    expect($mmQueryTrajectory.get().map(row => row.id)).toEqual(['B-older', 'B-shared'])
    expect($mmQueryTrajectory.get().find(row => row.id === 'B-shared')?.phase).toBe('started')

    fakeGateway.emit('multimodal.trajectory', {
      ...queryTrajectory('B-shared', 2, 'qry_B'),
      phase: 'live-newer-copy'
    }, 'runtime-B')
    expect($mmQueryTrajectory.get().find(row => row.id === 'B-shared')?.phase).toBe('live-newer-copy')

    hydrateA.resolve({ entries: [queryTrajectory('A-too-late', 9, 'qry_A')] })
    await flushPromises()

    expect($mmQueryTrajectory.get().map(row => row.id)).toEqual(['B-older', 'B-shared'])
  })
})
