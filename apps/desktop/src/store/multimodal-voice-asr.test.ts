import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

interface GatewayLike {
  request: ReturnType<typeof vi.fn>
}

const deps = vi.hoisted(() => ({
  addVoiceUserMessage: vi.fn(),
  captureAnchor: null as null | {
    source: 'camera' | 'screen'
    capture_attempt_id: string
    anchor_ts: number
  },
  gateway: null as GatewayLike | null,
  sessionId: 'runtime-voice'
}))

vi.mock('./gateway', () => ({
  $gateway: { get: () => deps.gateway }
}))

vi.mock('./multimodal', () => ({
  $mmSessionId: { get: () => deps.sessionId },
  addVoiceUserMessage: deps.addVoiceUserMessage
}))
vi.mock('./multimodal-capture', () => ({
  snapshotCaptureAnchor: () => deps.captureAnchor
}))

import {
  $mmAsrBuffer,
  $mmAsrPartial,
  $mmMicError,
  $mmMicState,
  $mmVoiceDialogEnabled,
  cancelManualMicOnDisconnect,
  configureDraftMicSessionEnsurer,
  finishMicTurn,
  hasMicCaptureIntent,
  onAsrBuffer,
  onAsrFinal,
  onAsrPartial,
  rearmMicAfterReconnect,
  rearmMicForSessionRebind,
  startMic,
  stopMic,
  toggleMultimodalVoiceDialog
} from './multimodal-voice'

interface FakeTrack {
  stop: ReturnType<typeof vi.fn>
}

class FakeAudioWorkletNode {
  static instances: FakeAudioWorkletNode[] = []

  connect = vi.fn()
  disconnect = vi.fn()
  flushMode: 'ack' | 'silent' | 'throw' = 'ack'
  flushTail: ArrayBuffer | null = null
  port = {
    close: vi.fn(),
    onmessage: null as ((event: MessageEvent) => void) | null,
    postMessage: vi.fn((message: { type?: string }) => {
      if (message?.type !== 'flush') {
        return
      }
      if (this.flushMode === 'throw') {
        throw new Error('worklet port closed')
      }
      if (this.flushMode === 'silent') {
        return
      }
      if (this.flushTail) {
        const tail = this.flushTail

        this.flushTail = null
        this.port.onmessage?.({ data: tail } as MessageEvent)
      }
      this.port.onmessage?.({ data: { type: 'flushed' } } as MessageEvent)
    })
  }

  constructor() {
    FakeAudioWorkletNode.instances.push(this)
  }

  emit(bytes: number[]): void {
    this.port.onmessage?.({ data: new Uint8Array(bytes).buffer } as MessageEvent)
  }

  emitBuffer(buffer: ArrayBuffer): void {
    this.port.onmessage?.({ data: buffer } as MessageEvent)
  }

  setFlushTail(bytes: number[]): void {
    this.flushTail = new Uint8Array(bytes).buffer
  }
}

class FakeAudioContext {
  static connectBytes: number[] | null = null
  static initialState: AudioContextState = 'running'
  static instances: FakeAudioContext[] = []

  audioWorklet = { addModule: vi.fn(async () => undefined) }
  close = vi.fn(async () => undefined)
  createMediaStreamSource = vi.fn(() => ({
    connect: vi.fn((node: FakeAudioWorkletNode) => {
      if (FakeAudioContext.connectBytes) {
        node.emit(FakeAudioContext.connectBytes)
      }
    }),
    disconnect: vi.fn()
  }))
  destination = {}
  resume = vi.fn(async () => {
    this.state = 'running'
  })
  sampleRate = 48_000
  state = FakeAudioContext.initialState

  constructor() {
    FakeAudioContext.instances.push(this)
  }
}

let originalMediaDevices: PropertyDescriptor | undefined
let tracks: FakeTrack[]

function gatewayCalls(method: string): Array<[string, Record<string, unknown>]> {
  return (deps.gateway?.request.mock.calls || []).filter(
    call => call[0] === method
  ) as Array<[string, Record<string, unknown>]>
}

describe('multimodal voice ASR preview state', () => {
  beforeEach(() => {
    originalMediaDevices = Object.getOwnPropertyDescriptor(navigator, 'mediaDevices')
    tracks = []
    FakeAudioWorkletNode.instances = []
    FakeAudioContext.connectBytes = null
    FakeAudioContext.initialState = 'running'
    FakeAudioContext.instances = []
    deps.captureAnchor = null
    deps.sessionId = 'runtime-voice'
    deps.gateway = {
      request: vi.fn(async (method: string) => method === 'multimodal.asr_start' ? { enabled: true } : { ok: true })
    }

    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: {
        getUserMedia: vi.fn(async () => {
          const track = { stop: vi.fn() }

          tracks.push(track)

          return {
            getTracks: () => [track]
          } as unknown as MediaStream
        })
      }
    })
    vi.stubGlobal('AudioContext', FakeAudioContext)
    vi.stubGlobal('AudioWorkletNode', FakeAudioWorkletNode)
    deps.addVoiceUserMessage.mockClear()
    $mmAsrBuffer.set([])
    $mmAsrPartial.set('')
    $mmMicError.set('')
    $mmVoiceDialogEnabled.set(false)
    configureDraftMicSessionEnsurer(null)
  })

  afterEach(async () => {
    await stopMic()
    configureDraftMicSessionEnsurer(null)
    deps.gateway = null
    vi.unstubAllGlobals()

    if (originalMediaDevices) {
      Object.defineProperty(navigator, 'mediaDevices', originalMediaDevices)
    } else {
      Reflect.deleteProperty(navigator, 'mediaDevices')
    }
  })

  it('keeps stitched EOU segments separate from the live partial', () => {
    onAsrBuffer(['第一段', '', '第二段'])
    onAsrPartial('正在识别')

    expect($mmAsrBuffer.get()).toEqual(['第一段', '第二段'])
    expect($mmAsrPartial.get()).toBe('正在识别')
  })

  it('injects the final voice turn and clears both preview layers', () => {
    onAsrBuffer(['已经说完的前半句'])
    onAsrPartial('后半句')

    onAsrFinal('  这是完整问题  ')

    expect(deps.addVoiceUserMessage).toHaveBeenCalledWith('这是完整问题')
    expect($mmAsrBuffer.get()).toEqual([])
    expect($mmAsrPartial.get()).toBe('')
  })

  it('clears stale preview state even when stop races with an already-idle recorder', async () => {
    onAsrBuffer(['残留段落'])
    onAsrPartial('残留 partial')

    await stopMic()

    expect($mmAsrBuffer.get()).toEqual([])
    expect($mmAsrPartial.get()).toBe('')
  })

  it('pins PCM and stop ownership to the session that opened the mic', async () => {
    await startMic()

    const node = FakeAudioWorkletNode.instances[0]

    node.emit([1, 2, 3])
    expect(gatewayCalls('multimodal.asr_audio')).toHaveLength(1)
    expect(gatewayCalls('multimodal.asr_audio')[0][1]).toEqual(
      expect.objectContaining({ session_id: 'runtime-voice' })
    )

    // Even if the shared session atom changes before its binding cleanup runs,
    // the old worklet must drop PCM instead of dynamically sending it to B.
    deps.sessionId = 'runtime-B'
    node.emit([4, 5, 6])
    expect(gatewayCalls('multimodal.asr_audio')).toHaveLength(1)

    await stopMic()

    expect(gatewayCalls('multimodal.asr_stop')).toContainEqual([
      'multimodal.asr_stop',
      expect.objectContaining({
        session_id: 'runtime-voice',
        disposition: 'cancel',
        turn_id: expect.stringMatching(/^desktop-asr-/)
      })
    ])
    expect(gatewayCalls('multimodal.asr_stop')).not.toContainEqual([
      'multimodal.asr_stop',
      { session_id: 'runtime-B' }
    ])
    expect(tracks[0].stop).toHaveBeenCalledTimes(1)
  })

  it('arms local draft media without creating until first PCM, then starts once with bounded pre-roll', async () => {
    deps.sessionId = ''
    let resolveSession!: (sid: string | null) => void

    const sessionReady = new Promise<string | null>(resolve => {
      resolveSession = resolve
    })

    const ensureSession = vi.fn(() => sessionReady)

    configureDraftMicSessionEnsurer(ensureSession)
    await startMic()

    expect(navigator.mediaDevices.getUserMedia).toHaveBeenCalledTimes(1)
    expect(FakeAudioWorkletNode.instances).toHaveLength(1)
    expect(ensureSession).not.toHaveBeenCalled()
    expect(gatewayCalls('multimodal.asr_start')).toHaveLength(0)

    const node = FakeAudioWorkletNode.instances[0]

    node.emit([1, 2, 3])
    node.emit([4, 5, 6])
    expect(ensureSession).toHaveBeenCalledTimes(1)

    deps.sessionId = 'runtime-draft'
    resolveSession('runtime-draft')
    await vi.waitFor(() => expect(gatewayCalls('multimodal.asr_audio')).toHaveLength(2))

    expect(gatewayCalls('multimodal.asr_start')).toHaveLength(1)
    expect(deps.gateway!.request).toHaveBeenCalledWith(
      'multimodal.asr_start',
      expect.objectContaining({
        session_id: 'runtime-draft',
        mode: 'manual_turn',
        turn_id: expect.stringMatching(/^desktop-asr-/)
      }),
      210_000
    )
    expect(gatewayCalls('multimodal.asr_audio')).toHaveLength(2)
    expect(gatewayCalls('multimodal.asr_audio').every(call => call[1].session_id === 'runtime-draft')).toBe(true)
    const turnId = gatewayCalls('multimodal.asr_start')[0][1].turn_id

    expect(gatewayCalls('multimodal.asr_audio').every(call => call[1].turn_id === turnId)).toBe(true)

    node.emit([7, 8, 9])
    expect(gatewayCalls('multimodal.asr_audio')).toHaveLength(3)
  })

  it('releases an armed draft and ignores a late create after explicit stop', async () => {
    deps.sessionId = ''
    let resolveSession!: (sid: string | null) => void

    const sessionReady = new Promise<string | null>(resolve => {
      resolveSession = resolve
    })

    const ensureSession = vi.fn(() => sessionReady)

    configureDraftMicSessionEnsurer(ensureSession)
    await startMic()
    FakeAudioWorkletNode.instances[0].emit([1, 2, 3])
    expect(ensureSession).toHaveBeenCalledTimes(1)

    await stopMic()
    expect(tracks[0].stop).toHaveBeenCalledTimes(1)
    expect($mmMicState.get()).toBe('idle')

    deps.sessionId = 'runtime-late'
    resolveSession('runtime-late')
    await Promise.resolve()
    await Promise.resolve()

    expect(gatewayCalls('multimodal.asr_start')).toHaveLength(0)
    expect(gatewayCalls('multimodal.asr_audio')).toHaveLength(0)
  })

  it('applies a fresh-draft voice-dialog intent once after ASR binds', async () => {
    deps.sessionId = ''

    const ensureSession = vi.fn(async () => {
      deps.sessionId = 'runtime-dialog'

      return 'runtime-dialog'
    })

    configureDraftMicSessionEnsurer(ensureSession)
    toggleMultimodalVoiceDialog()
    await vi.waitFor(() => expect(FakeAudioWorkletNode.instances).toHaveLength(1))
    expect(ensureSession).not.toHaveBeenCalled()

    FakeAudioWorkletNode.instances[0].emit([1, 2, 3])
    await vi.waitFor(() => expect(deps.gateway!.request.mock.calls.filter(call =>
      call[0] === 'multimodal.voice_dialog_toggle' && call[1]?.enabled === true
    )).toHaveLength(1))

    expect(deps.gateway!.request.mock.calls.filter(call =>
      call[0] === 'multimodal.voice_dialog_toggle' && call[1]?.enabled === true
    )).toEqual([[
      'multimodal.voice_dialog_toggle',
      { session_id: 'runtime-dialog', enabled: true }
    ]])
  })

  it('cancels an in-flight local-first A start without ever stopping B', async () => {
    let resolveStart!: (value: { enabled: boolean }) => void

    const startResponse = new Promise<{ enabled: boolean }>(resolve => {
      resolveStart = resolve
    })

    deps.gateway!.request.mockImplementation(
      async (method: string) => method === 'multimodal.asr_start' ? startResponse : { ok: true }
    )

    const starting = startMic()

    expect($mmMicState.get()).toBe('connecting')
    deps.sessionId = 'runtime-B'
    await stopMic()
    resolveStart({ enabled: true })
    await starting

    expect(navigator.mediaDevices.getUserMedia).toHaveBeenCalledTimes(1)
    expect(gatewayCalls('multimodal.asr_stop')).not.toHaveLength(0)
    expect(gatewayCalls('multimodal.asr_stop').every(call => call[1].session_id === 'runtime-voice')).toBe(true)
    expect($mmMicState.get()).toBe('idle')
  })

  it('cancels immediately while microphone permission is pending and rejects the late stream', async () => {
    let resolveStream!: (stream: MediaStream) => void
    const lateStream = new Promise<MediaStream>(resolve => {
      resolveStream = resolve
    })
    const lateTrack = { stop: vi.fn() }

    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia: vi.fn(() => lateStream) }
    })

    const starting = startMic()

    expect($mmMicState.get()).toBe('connecting')
    await stopMic()
    expect($mmMicState.get()).toBe('idle')

    resolveStream({ getTracks: () => [lateTrack] } as unknown as MediaStream)
    await starting

    expect(lateTrack.stop).toHaveBeenCalledTimes(1)
    expect(gatewayCalls('multimodal.asr_start')).toHaveLength(0)
    expect(gatewayCalls('multimodal.asr_stop').filter(call => call[1].disposition === 'finish')).toHaveLength(0)
    expect(deps.addVoiceUserMessage).not.toHaveBeenCalled()
  })

  it('records locally before a cold backend is ready, then drains every audio ACK before one finish', async () => {
    let resolveStart!: (value: { enabled: boolean }) => void
    const startResponse = new Promise<{ enabled: boolean }>(resolve => {
      resolveStart = resolve
    })
    const audioResolvers: Array<() => void> = []

    deps.gateway!.request.mockImplementation((method: string, params?: Record<string, unknown>) => {
      if (method === 'multimodal.asr_start') {
        return startResponse
      }
      if (method === 'multimodal.asr_audio') {
        return new Promise(resolve => audioResolvers.push(() => resolve({ ok: true })))
      }
      if (method === 'multimodal.asr_stop') {
        return Promise.resolve({ ok: true, turn_id: params?.turn_id, submitted: true })
      }

      return Promise.resolve({ ok: true })
    })

    const starting = startMic()

    await vi.waitFor(() => expect($mmMicState.get()).toBe('recording'))
    const node = FakeAudioWorkletNode.instances[0]

    node.emit([1, 2, 3])
    node.emit([4, 5, 6])
    expect(gatewayCalls('multimodal.asr_audio')).toHaveLength(0)

    const finishing = finishMicTurn()

    expect($mmMicState.get()).toBe('finalizing')
    expect(tracks[0].stop).toHaveBeenCalledTimes(1)
    resolveStart({ enabled: true })
    await starting
    await vi.waitFor(() => expect(gatewayCalls('multimodal.asr_audio')).toHaveLength(2))
    expect(gatewayCalls('multimodal.asr_stop')).toHaveLength(0)

    audioResolvers.forEach(resolve => resolve())
    await finishing

    const startParams = gatewayCalls('multimodal.asr_start')[0][1]
    const audioParams = gatewayCalls('multimodal.asr_audio').map(call => call[1])
    const finishCalls = gatewayCalls('multimodal.asr_stop').filter(call => call[1].disposition === 'finish')

    expect(startParams.mode).toBe('manual_turn')
    expect(audioParams.every(params => params.turn_id === startParams.turn_id)).toBe(true)
    expect(finishCalls).toHaveLength(1)
    expect(finishCalls[0][1]).toEqual(expect.objectContaining({
      session_id: 'runtime-voice',
      turn_id: startParams.turn_id,
      disposition: 'finish'
    }))
    expect($mmMicState.get()).toBe('idle')
  })

  it('installs the pre-roll receiver before connecting the audio graph', async () => {
    let resolveStart!: (value: { enabled: boolean }) => void
    const startResponse = new Promise<{ enabled: boolean }>(resolve => {
      resolveStart = resolve
    })

    FakeAudioContext.connectBytes = [11, 12, 13]
    deps.gateway!.request.mockImplementation(
      (method: string) => method === 'multimodal.asr_start' ? startResponse : Promise.resolve({ ok: true })
    )

    const starting = startMic()

    await vi.waitFor(() => expect($mmMicState.get()).toBe('recording'))
    expect(gatewayCalls('multimodal.asr_audio')).toHaveLength(0)

    resolveStart({ enabled: true })
    await starting

    expect(gatewayCalls('multimodal.asr_audio')).toHaveLength(1)
    expect(atob(String(gatewayCalls('multimodal.asr_audio')[0][1].pcm_b64)).charCodeAt(0)).toBe(11)
  })

  it('deduplicates repeated finish clicks for one stable manual turn', async () => {
    await startMic()
    FakeAudioWorkletNode.instances[0].emit([1, 2, 3])

    await Promise.all([finishMicTurn(), finishMicTurn()])

    expect(gatewayCalls('multimodal.asr_stop').filter(call => call[1].disposition === 'finish')).toHaveLength(1)
  })

  it('sends an independent exact-turn cancel when a session boundary interrupts finalizing', async () => {
    let resolveFinish!: (value: Record<string, unknown>) => void
    const finishResponse = new Promise<Record<string, unknown>>(resolve => {
      resolveFinish = resolve
    })

    deps.gateway!.request.mockImplementation((method: string, params?: Record<string, unknown>) => {
      if (method === 'multimodal.asr_start') {
        return Promise.resolve({ enabled: true, turn_id: params?.turn_id })
      }
      if (method === 'multimodal.asr_stop' && params?.disposition === 'finish') {
        return finishResponse
      }

      return Promise.resolve({ ok: true, turn_id: params?.turn_id, submitted: false })
    })

    await startMic()
    const turnId = String(gatewayCalls('multimodal.asr_start')[0][1].turn_id)
    const finishing = finishMicTurn()

    await vi.waitFor(() => expect(gatewayCalls('multimodal.asr_stop')).toHaveLength(1))
    expect($mmMicState.get()).toBe('finalizing')

    await stopMic()

    expect(gatewayCalls('multimodal.asr_stop').map(call => call[1])).toEqual([
      expect.objectContaining({ turn_id: turnId, disposition: 'finish' }),
      expect.objectContaining({ turn_id: turnId, disposition: 'cancel' })
    ])
    expect($mmMicState.get()).toBe('idle')

    resolveFinish({ ok: true, turn_id: turnId, submitted: false, reason: 'cancelled' })
    await finishing
    expect($mmMicState.get()).toBe('idle')
  })

  it('surfaces an actionable error when a finished turn contains no recognized speech', async () => {
    deps.gateway!.request.mockImplementation(async (method: string, params?: Record<string, unknown>) => {
      if (method === 'multimodal.asr_start') {
        return { enabled: true, turn_id: params?.turn_id }
      }
      if (method === 'multimodal.asr_stop' && params?.disposition === 'finish') {
        return {
          ok: true,
          turn_id: params.turn_id,
          submitted: false,
          reason: 'empty'
        }
      }

      return { ok: true }
    })

    await startMic()

    await expect(finishMicTurn()).rejects.toThrow('No speech recognized, please try again or check your microphone input device')
    expect($mmMicError.get()).toBe('No speech recognized, please try again or check your microphone input device')
    expect($mmMicState.get()).toBe('idle')
  })

  it('maps an upstream finish failure without exposing unbounded backend details', async () => {
    deps.gateway!.request.mockImplementation(async (method: string, params?: Record<string, unknown>) => {
      if (method === 'multimodal.asr_start') {
        return { enabled: true, turn_id: params?.turn_id }
      }
      if (method === 'multimodal.asr_stop' && params?.disposition === 'finish') {
        return {
          ok: false,
          turn_id: params.turn_id,
          submitted: false,
          reason: 'upstream_error',
          error: 'provider secret diagnostic '.repeat(500)
        }
      }

      return { ok: true }
    })

    await startMic()

    await expect(finishMicTurn()).rejects.toThrow('Speech recognition service is temporarily unavailable, please try again')
    expect($mmMicError.get()).toBe('Speech recognition service is temporarily unavailable, please try again')
    expect($mmMicError.get()).not.toContain('provider secret diagnostic')
  })

  it.each(['reject', 'ok-false'] as const)(
    'cancels instead of submitting a partial turn when an audio chunk returns %s',
    async failure => {
      deps.gateway!.request.mockImplementation((method: string, params?: Record<string, unknown>) => {
        if (method === 'multimodal.asr_start') {
          return Promise.resolve({ enabled: true, turn_id: params?.turn_id })
        }
        if (method === 'multimodal.asr_audio') {
          return failure === 'reject'
            ? Promise.reject(new Error('socket interrupted'))
            : Promise.resolve({ ok: false, reason: 'stale_transport' })
        }

        return Promise.resolve({ ok: true, turn_id: params?.turn_id, submitted: false })
      })

      await startMic()
      const turnId = String(gatewayCalls('multimodal.asr_start')[0][1].turn_id)

      FakeAudioWorkletNode.instances[0].emit([1, 2, 3])

      await expect(finishMicTurn()).rejects.toThrow('Audio upload was interrupted, please try again')
      expect($mmMicError.get()).toBe('Audio upload was interrupted, please try again')
      expect(gatewayCalls('multimodal.asr_stop')).toEqual([
        ['multimodal.asr_stop', expect.objectContaining({
          turn_id: turnId,
          disposition: 'cancel'
        })]
      ])
    }
  )

  it('flushes a sub-batch worklet tail before finishing the backend turn', async () => {
    deps.captureAnchor = {
      source: 'camera',
      capture_attempt_id: 'cap-current',
      anchor_ts: 12.5
    }
    await startMic()
    const node = FakeAudioWorkletNode.instances[0]

    node.setFlushTail([9, 8, 7, 6])
    await finishMicTurn()

    expect(node.port.postMessage).toHaveBeenCalledWith({ type: 'flush' })
    const lifecycle = deps.gateway!.request.mock.calls
      .filter(call => call[0] === 'multimodal.asr_audio' || call[0] === 'multimodal.asr_stop')

    expect(lifecycle.map(call => call[0])).toEqual(['multimodal.asr_audio', 'multimodal.asr_stop'])
    expect(lifecycle[1][1]).toEqual(expect.objectContaining({
      disposition: 'finish',
      capture_attempt_id: 'cap-current',
      anchor_ts: 12.5
    }))
  })

  it.each(['timeout', 'throw'] as const)(
    'cancels without a finish submit when worklet tail flush hits %s',
    async failure => {
      if (failure === 'timeout') {
        vi.useFakeTimers()
      }

      try {
        await startMic()
        const node = FakeAudioWorkletNode.instances[0]
        const turnId = String(gatewayCalls('multimodal.asr_start')[0][1].turn_id)

        node.flushMode = failure === 'timeout' ? 'silent' : 'throw'
        const finishing = finishMicTurn()
        const rejection = expect(finishing).rejects.toThrow('Failed to finalize the recording, please try again')

        if (failure === 'timeout') {
          await vi.advanceTimersByTimeAsync(500)
        }

        await rejection
        expect(gatewayCalls('multimodal.asr_stop')).toEqual([
          ['multimodal.asr_stop', expect.objectContaining({
            turn_id: turnId,
            disposition: 'cancel'
          })]
        ])
      } finally {
        vi.useRealTimers()
      }
    }
  )

  it('keeps headroom beyond the 210s activation timeout and drops only the oldest overflow', async () => {
    let resolveStart!: (value: { enabled: boolean }) => void
    const startResponse = new Promise<{ enabled: boolean }>(resolve => {
      resolveStart = resolve
    })

    deps.gateway!.request.mockImplementation(
      (method: string) => method === 'multimodal.asr_start' ? startResponse : Promise.resolve({ ok: true })
    )
    const starting = startMic()

    await vi.waitFor(() => expect($mmMicState.get()).toBe('recording'))
    const node = FakeAudioWorkletNode.instances[0]

    for (let i = 1; i <= 8; i += 1) {
      const chunk = new Uint8Array(1_000_000)

      chunk[0] = i
      node.emitBuffer(chunk.buffer)
    }

    resolveStart({ enabled: true })
    await starting
    await vi.waitFor(() => expect(gatewayCalls('multimodal.asr_audio')).toHaveLength(7))

    const firstRetained = String(gatewayCalls('multimodal.asr_audio')[0][1].pcm_b64 || '')

    expect(atob(firstRetained).charCodeAt(0)).toBe(2)
  })

  it('resumes a suspended AudioContext after the macOS permission bridge allows access', async () => {
    const originalBridge = Object.getOwnPropertyDescriptor(window, 'hermesDesktop')
    const requestMicrophoneAccess = vi.fn(async () => true)

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { requestMicrophoneAccess }
    })
    FakeAudioContext.initialState = 'suspended'

    try {
      await startMic()

      expect(requestMicrophoneAccess).toHaveBeenCalledTimes(1)
      expect(FakeAudioContext.instances[0].resume).toHaveBeenCalledTimes(1)
      expect($mmMicState.get()).toBe('recording')
    } finally {
      if (originalBridge) {
        Object.defineProperty(window, 'hermesDesktop', originalBridge)
      } else {
        Reflect.deleteProperty(window, 'hermesDesktop')
      }
    }
  })

  it('cancels a manual turn instead of rearming it after a gateway disconnect', async () => {
    await startMic()

    cancelManualMicOnDisconnect()
    await vi.waitFor(() => expect($mmMicState.get()).toBe('idle'))

    expect(gatewayCalls('multimodal.asr_start')).toHaveLength(1)
    expect(gatewayCalls('multimodal.asr_stop')).toContainEqual([
      'multimodal.asr_stop',
      expect.objectContaining({ disposition: 'cancel' })
    ])
    expect($mmMicState.get()).toBe('idle')
  })

  it('refuses to relabel an active manual turn as continuous dialog mode', async () => {
    await startMic()
    const startsBefore = gatewayCalls('multimodal.asr_start')

    expect(toggleMultimodalVoiceDialog()).toBe(false)
    expect($mmVoiceDialogEnabled.get()).toBe(false)
    expect($mmMicState.get()).toBe('recording')
    expect(startsBefore).toHaveLength(1)
    expect(startsBefore[0][1].mode).toBe('manual_turn')
    expect(deps.gateway!.request).not.toHaveBeenCalledWith(
      'multimodal.voice_dialog_toggle',
      expect.objectContaining({ enabled: true })
    )
  })

  it('ignores late preview/final events from a superseded turn id', async () => {
    await startMic()
    const turnId = String(gatewayCalls('multimodal.asr_start')[0][1].turn_id)

    onAsrPartial('新轮次', turnId)
    onAsrPartial('旧轮次', 'old-turn')
    onAsrFinal('不该提交', 'old-turn')

    expect($mmAsrPartial.get()).toBe('新轮次')
    expect(deps.addVoiceUserMessage).not.toHaveBeenCalled()
  })

  it('stops the old runtime and rearms PCM on a same-conversation replacement runtime', async () => {
    $mmVoiceDialogEnabled.set(true)
    await startMic()
    deps.sessionId = 'runtime-voice-2'

    await rearmMicForSessionRebind()

    const lifecycle = deps.gateway!.request.mock.calls
      .filter(call => call[0] === 'multimodal.asr_start' || call[0] === 'multimodal.asr_stop')
      .map(call => [call[0], call[1]?.session_id])

    expect(lifecycle).toEqual([
      ['multimodal.asr_start', 'runtime-voice'],
      ['multimodal.asr_stop', 'runtime-voice'],
      ['multimodal.asr_start', 'runtime-voice-2']
    ])
    expect(tracks[0].stop).toHaveBeenCalledTimes(1)
    expect(deps.gateway!.request).toHaveBeenCalledWith(
      'multimodal.voice_dialog_toggle',
      { session_id: 'runtime-voice-2', enabled: true }
    )

    FakeAudioWorkletNode.instances[1].emit([7, 8, 9])
    expect(gatewayCalls('multimodal.asr_audio').at(-1)?.[1]).toEqual(
      expect.objectContaining({ session_id: 'runtime-voice-2' })
    )
  })

  it('keeps reconnect intent when stale A rejects, then rearms only on same-conversation A2', async () => {
    $mmVoiceDialogEnabled.set(true)
    await startMic()
    const oldNode = FakeAudioWorkletNode.instances[0]
    const audioBeforeReconnect = gatewayCalls('multimodal.asr_audio').length
    let rejectStaleA = true

    deps.gateway!.request.mockImplementation(async (method: string, params?: Record<string, unknown>) => {
      if (method === 'multimodal.asr_start') {
        if (params?.session_id === 'runtime-voice' && rejectStaleA) {
          rejectStaleA = false
          throw new Error('session not found: runtime-voice')
        }

        return { enabled: true }
      }

      return { ok: true }
    })

    // Gateway open races use-route-resume and tries the still-published A. A is
    // already gone, so this attempt fails before the replacement id exists.
    await rearmMicAfterReconnect()

    expect($mmMicState.get()).toBe('idle')
    expect(hasMicCaptureIntent()).toBe(true)
    oldNode.emit([4, 5, 6])
    expect(gatewayCalls('multimodal.asr_audio')).toHaveLength(audioBeforeReconnect)

    // The route now enters its empty recovery gap, then the same durable
    // conversation publishes A2. Its one-shot binding consumes the latent mic
    // intent and opens ASR on A2, never on an unrelated B.
    deps.sessionId = ''
    expect(hasMicCaptureIntent()).toBe(true)
    deps.sessionId = 'runtime-voice-2'
    await rearmMicForSessionRebind()

    const starts = gatewayCalls('multimodal.asr_start').map(call => call[1].session_id)

    expect(starts).toEqual(['runtime-voice', 'runtime-voice', 'runtime-voice-2'])
    expect(starts).not.toContain('runtime-B')
    expect(hasMicCaptureIntent()).toBe(true)
    expect($mmMicState.get()).toBe('recording')

    FakeAudioWorkletNode.instances.at(-1)!.emit([7, 8, 9])
    expect(gatewayCalls('multimodal.asr_audio').at(-1)?.[1]).toEqual(
      expect.objectContaining({ session_id: 'runtime-voice-2' })
    )
  })

  it('does not rearm the replacement runtime after an explicit stop in the reconnect gap', async () => {
    $mmVoiceDialogEnabled.set(true)
    await startMic()
    let rejectStaleA = true

    deps.gateway!.request.mockImplementation(async (method: string, params?: Record<string, unknown>) => {
      if (method === 'multimodal.asr_start') {
        if (params?.session_id === 'runtime-voice' && rejectStaleA) {
          rejectStaleA = false
          throw new Error('session not found: runtime-voice')
        }

        return { enabled: true }
      }

      return { ok: true }
    })

    await rearmMicAfterReconnect()
    deps.sessionId = ''
    await stopMic()

    expect(hasMicCaptureIntent()).toBe(false)

    deps.sessionId = 'runtime-voice-2'
    await rearmMicForSessionRebind()

    expect(gatewayCalls('multimodal.asr_start').map(call => call[1].session_id)).toEqual([
      'runtime-voice',
      'runtime-voice'
    ])
    expect($mmMicState.get()).toBe('idle')
  })
})
