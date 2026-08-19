import type { WritableAtom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const { gatewayAtom, pushMmToast, sessionIdAtom } = vi.hoisted(() => {
  const { atom } = require('nanostores') as {
    atom: <Value>(initial: Value) => WritableAtom<Value>
  }

  return {
    gatewayAtom: atom<unknown>(null),
    pushMmToast: vi.fn(),
    sessionIdAtom: atom<string>('')
  }
})

vi.mock('./gateway', () => ({ $gateway: gatewayAtom }))
vi.mock('./multimodal', () => ({
  $mmSessionId: sessionIdAtom,
  addVoiceUserMessage: vi.fn()
}))
vi.mock('./multimodal-deep', () => ({ pushMmToast }))

import { startEnvAudio, startMic, stopEnvAudio, stopMic } from './multimodal-voice'

interface FakeTrack {
  clone: ReturnType<typeof vi.fn>
  stop: ReturnType<typeof vi.fn>
}

class FakeMediaStream {
  constructor(private readonly tracks: FakeTrack[]) {}

  getAudioTracks(): FakeTrack[] {
    return this.tracks
  }

  getTracks(): FakeTrack[] {
    return this.tracks
  }
}

class FakeMediaRecorder {
  static instances: FakeMediaRecorder[] = []
  static failEveryStart = false
  static startErrors: Array<Error | null> = []

  static isTypeSupported(): boolean {
    return true
  }

  ondataavailable: ((event: { data: Blob; timecode: number }) => void) | null = null
  onstop: (() => void | Promise<void>) | null = null
  mimeType: string
  state: 'inactive' | 'recording' = 'inactive'

  constructor(
    readonly stream: FakeMediaStream,
    options?: MediaRecorderOptions
  ) {
    this.mimeType = options?.mimeType || 'audio/webm'
    FakeMediaRecorder.instances.push(this)
  }

  emitData(data: Blob, timecode = 0): void {
    this.ondataavailable?.({ data, timecode })
  }

  finishStop(): void {
    void this.onstop?.()
  }

  start(): void {
    const error = FakeMediaRecorder.startErrors.shift()
    if (FakeMediaRecorder.failEveryStart || error) {
      throw error || new Error('encoder rejected stream')
    }
    this.state = 'recording'
  }

  stop(): void {
    this.state = 'inactive'
  }
}

class FakeAudioWorkletNode {
  static instances: FakeAudioWorkletNode[] = []

  connect = vi.fn()
  disconnect = vi.fn()
  port = {
    close: vi.fn(),
    onmessage: null as ((event: { data: ArrayBuffer }) => void) | null
  }

  constructor() {
    FakeAudioWorkletNode.instances.push(this)
  }

  emit(data: ArrayBuffer): void {
    this.port.onmessage?.({ data })
  }
}

class FakeAudioContext {
  static instances: FakeAudioContext[] = []

  audioWorklet = { addModule: vi.fn(async () => undefined) }
  close = vi.fn(async () => undefined)
  createMediaStreamSource = vi.fn(() => ({
    connect: vi.fn(),
    disconnect: vi.fn()
  }))
  destination = {}
  resume = vi.fn(async () => undefined)
  sampleRate = 48_000

  constructor() {
    FakeAudioContext.instances.push(this)
  }
}

class FakeFileReader {
  error: Error | null = null
  onerror: (() => void) | null = null
  onload: (() => void) | null = null
  result: string | null = null

  readAsDataURL(): void {
    this.result = 'data:audio/webm;base64,ZmFrZS1hdWRpbw=='
    this.onload?.()
  }
}

function audioSource() {
  const clone = { clone: vi.fn(), stop: vi.fn() }

  const original = {
    clone: vi.fn(() => clone),
    stop: vi.fn()
  }

  return {
    clone,
    original,
    stream: { getAudioTracks: vi.fn(() => [original]) } as unknown as MediaStream
  }
}

async function flushPromises(): Promise<void> {
  await Promise.resolve()
  await Promise.resolve()
  await Promise.resolve()
  await Promise.resolve()
}

async function finishCurrentChunk(timecode = 0): Promise<FakeMediaRecorder> {
  const recorder = FakeMediaRecorder.instances.at(-1)!
  recorder.emitData(new Blob([new Uint8Array(1_200)], { type: 'audio/webm;codecs=opus' }), timecode)
  await vi.advanceTimersByTimeAsync(5_000)
  recorder.finishStop()
  await flushPromises()

  return recorder
}

describe('screen-share environment audio ownership', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    FakeMediaRecorder.instances = []
    FakeMediaRecorder.failEveryStart = false
    FakeMediaRecorder.startErrors = []
    FakeAudioContext.instances = []
    FakeAudioWorkletNode.instances = []
    gatewayAtom.set(null)
    sessionIdAtom.set('')
    pushMmToast.mockReset()
    vi.stubGlobal('MediaStream', FakeMediaStream)
    vi.stubGlobal('MediaRecorder', FakeMediaRecorder)
    vi.stubGlobal('AudioContext', FakeAudioContext)
    vi.stubGlobal('AudioWorkletNode', FakeAudioWorkletNode)
    vi.stubGlobal('FileReader', FakeFileReader)
  })

  afterEach(() => {
    stopEnvAudio()
    vi.unstubAllGlobals()
    vi.useRealTimers()
  })

  it('records from owned loopback clones and never stops the screen-share audio track', () => {
    const { clone, original, stream } = audioSource()

    startEnvAudio(stream)

    expect(original.clone).toHaveBeenCalledTimes(1)
    expect(FakeMediaRecorder.instances).toHaveLength(1)
    expect(FakeMediaRecorder.instances[0].stream.getTracks()).toEqual([clone])

    stopEnvAudio()

    expect(clone.stop).toHaveBeenCalledTimes(1)
    expect(original.stop).not.toHaveBeenCalled()
  })

  it('keeps loopback env audio alive while an ordinary mic turn starts and cancels', async () => {
    const request = vi.fn(async (method: string) => {
      if (method === 'multimodal.asr_start') {
        return { enabled: true }
      }

      return { ok: true, ingested: true }
    })

    const { clone, original, stream } = audioSource()
    const micTrack = { stop: vi.fn() }

    const micStream = {
      getTracks: vi.fn(() => [micTrack])
    } as unknown as MediaStream

    gatewayAtom.set({ request })
    sessionIdAtom.set('runtime-audio')
    vi.stubGlobal('navigator', {
      mediaDevices: {
        getUserMedia: vi.fn(async () => micStream)
      }
    })

    startEnvAudio(stream)
    const envRecorder = FakeMediaRecorder.instances[0]

    await startMic()
    await stopMic()

    expect(micTrack.stop).toHaveBeenCalledTimes(1)
    expect(envRecorder.state).toBe('recording')
    expect(clone.stop).not.toHaveBeenCalled()
    expect(original.stop).not.toHaveBeenCalled()

    await finishCurrentChunk()

    expect(request).toHaveBeenCalledWith(
      'multimodal.env_audio',
      expect.objectContaining({ session_id: 'runtime-audio' })
    )
  })

  it('retries with the browser default when an explicit MIME fails at start', () => {
    FakeMediaRecorder.startErrors = [new Error('explicit encoder failed'), null]

    startEnvAudio(audioSource().stream)

    expect(FakeMediaRecorder.instances).toHaveLength(2)
    expect(FakeMediaRecorder.instances[0].mimeType).toBe('audio/webm;codecs=opus')
    expect(FakeMediaRecorder.instances[1].mimeType).toBe('audio/webm')
    expect(FakeMediaRecorder.instances[1].state).toBe('recording')
    expect(pushMmToast).not.toHaveBeenCalled()
  })

  it('falls back to PCM/WAV when every MediaRecorder start attempt fails', async () => {
    const request = vi.fn(async () => ({ ingested: true }))
    gatewayAtom.set({ request })
    sessionIdAtom.set('runtime-audio')
    FakeMediaRecorder.failEveryStart = true

    startEnvAudio(audioSource().stream)
    await flushPromises()

    expect(FakeAudioWorkletNode.instances).toHaveLength(1)
    const pcm = new Int16Array(16_000)
    pcm[8_000] = 1_024
    FakeAudioWorkletNode.instances[0].emit(pcm.buffer)
    await vi.advanceTimersByTimeAsync(5_000)
    await flushPromises()

    expect(request).toHaveBeenCalledTimes(1)
    expect(request).toHaveBeenCalledWith(
      'multimodal.env_audio',
      expect.objectContaining({
        session_id: 'runtime-audio',
        mime: 'audio/wav',
        chunk_seq: 1,
        client_duration_sec: 5
      })
    )
    expect(pushMmToast).not.toHaveBeenCalled()
  })

  it('rejects all-zero PCM fallback instead of uploading silent WAV', async () => {
    const request = vi.fn(async () => ({ ingested: true }))
    gatewayAtom.set({ request })
    sessionIdAtom.set('runtime-audio')
    FakeMediaRecorder.failEveryStart = true

    startEnvAudio(audioSource().stream)
    await flushPromises()

    expect(FakeAudioWorkletNode.instances).toHaveLength(1)
    FakeAudioWorkletNode.instances[0].emit(new ArrayBuffer(16_000 * 2))
    await vi.advanceTimersByTimeAsync(5_000)
    await flushPromises()

    expect(request).not.toHaveBeenCalled()
    expect(pushMmToast).toHaveBeenCalledTimes(1)
    expect(pushMmToast).toHaveBeenCalledWith({
      level: 'error',
      text: 'Shared audio received no valid samples. Check macOS "Screen & System Audio Recording" permission, then stop and re-share the screen.'
    })
  })

  it('drops a trailing recorder slice from an older capture generation after restart', async () => {
    const request = vi.fn(async () => ({ ok: true }))
    gatewayAtom.set({ request })
    sessionIdAtom.set('runtime-audio')
    const first = audioSource()
    const second = audioSource()

    startEnvAudio(first.stream)
    const staleRecorder = FakeMediaRecorder.instances[0]
    staleRecorder.emitData(new Blob([new Uint8Array(1_200)]))

    startEnvAudio(second.stream)
    staleRecorder.finishStop()
    await Promise.resolve()

    expect(request).not.toHaveBeenCalledWith('multimodal.env_audio', expect.anything())
    expect(first.clone.stop).toHaveBeenCalledTimes(1)
    expect(first.original.stop).not.toHaveBeenCalled()
    expect(second.original.stop).not.toHaveBeenCalled()
    expect(pushMmToast).not.toHaveBeenCalled()
  })

  it('sends independently identified chunks with client timeline metadata', async () => {
    const request = vi.fn(async () => ({ ingested: true }))
    gatewayAtom.set({ request })
    sessionIdAtom.set('runtime-audio')

    startEnvAudio(audioSource().stream)
    await finishCurrentChunk(4_321)

    expect(request).toHaveBeenCalledTimes(1)
    const [method, payload] = request.mock.calls[0] as unknown as [string, Record<string, unknown>]
    expect(method).toBe('multimodal.env_audio')
    expect(payload).toMatchObject({
      session_id: 'runtime-audio',
      data_b64: 'ZmFrZS1hdWRpbw==',
      mime: 'audio/webm;codecs=opus',
      chunk_seq: 1,
      client_start_ts: 0,
      client_end_ts: 5,
      client_duration_sec: 5,
      blob_timecode: 4_321,
      window_ts: 0
    })
    expect(payload.capture_id).toMatch(/^cap_/)
    expect(payload.chunk_id).toBe(`${payload.capture_id}:1`)
  })

  it('keeps too_short silent and de-duplicates other ingested:false reasons', async () => {
    const request = vi
      .fn()
      .mockResolvedValueOnce({ ingested: false, reason: 'too_short' })
      .mockResolvedValue({ ingested: false, reason: 'decoder_failed' })

    gatewayAtom.set({ request })
    sessionIdAtom.set('runtime-audio')

    startEnvAudio(audioSource().stream)
    await finishCurrentChunk()
    expect(pushMmToast).not.toHaveBeenCalled()

    await finishCurrentChunk()
    await finishCurrentChunk()

    expect(pushMmToast).toHaveBeenCalledTimes(1)
    expect(pushMmToast).toHaveBeenCalledWith({
      level: 'error',
      text: 'Shared audio ASR not received: decoder_failed'
    })
  })

  it('shows request failures once per reason', async () => {
    const request = vi.fn().mockRejectedValue(new Error('backend offline'))
    gatewayAtom.set({ request })
    sessionIdAtom.set('runtime-audio')

    startEnvAudio(audioSource().stream)
    await finishCurrentChunk()
    await finishCurrentChunk()

    expect(pushMmToast).toHaveBeenCalledTimes(1)
    expect(pushMmToast).toHaveBeenCalledWith({
      level: 'error',
      text: 'Shared audio ASR request failed: backend offline'
    })
  })

  it('does not send or toast when base64 completion belongs to an old capture', async () => {
    const request = vi.fn(async () => ({ ingested: false, reason: 'stale' }))
    gatewayAtom.set({ request })
    sessionIdAtom.set('runtime-audio')
    const first = audioSource()
    const second = audioSource()

    startEnvAudio(first.stream)
    const staleRecorder = FakeMediaRecorder.instances[0]
    staleRecorder.emitData(new Blob([new Uint8Array(1_200)]))
    // onstop starts the async base64 chain. Supersede it before its microtask
    // can dispatch the request.
    staleRecorder.finishStop()
    startEnvAudio(second.stream)
    await flushPromises()

    expect(request).not.toHaveBeenCalled()
    expect(pushMmToast).not.toHaveBeenCalled()
  })

  it('does not toast when an in-flight old-capture request rejects after restart', async () => {
    let rejectRequest: ((reason: Error) => void) | undefined

    const request = vi.fn(
      () =>
        new Promise((_resolve, reject) => {
          rejectRequest = reject
        })
    )

    gatewayAtom.set({ request })
    sessionIdAtom.set('runtime-audio')

    startEnvAudio(audioSource().stream)
    await finishCurrentChunk()
    expect(request).toHaveBeenCalledTimes(1)

    startEnvAudio(audioSource().stream)
    rejectRequest?.(new Error('late failure'))
    await flushPromises()

    expect(pushMmToast).not.toHaveBeenCalled()
  })
})
