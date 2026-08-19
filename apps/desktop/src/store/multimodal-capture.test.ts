import type { WritableAtom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const { mockGatewayAtom, mockSessionId, openSourcePicker, pushMmToast } = vi.hoisted(() => {
  const { atom } = require('nanostores') as {
    atom: <Value>(initial: Value) => WritableAtom<Value>
  }

  return {
    mockGatewayAtom: atom<unknown>(null),
    mockSessionId: atom<string>(''),
    openSourcePicker: vi.fn(),
    pushMmToast: vi.fn()
  }
})

vi.mock('./gateway', () => ({ $gateway: mockGatewayAtom }))
vi.mock('./multimodal', () => ({ $mmSessionId: mockSessionId }))
vi.mock('./multimodal-voice', () => ({
  startEnvAudio: vi.fn(),
  stopEnvAudio: vi.fn()
}))
vi.mock('./multimodal-deep', () => ({ pushMmToast }))
vi.mock('@/components/multimodal/screen-source-picker', () => ({ openSourcePicker }))

import {
  $mmCapStats,
  $mmCaptureDebug,
  $mmSource,
  $mmStream,
  configureDraftCaptureSessionEnsurer,
  ensureCaptureBoundToSession,
  pauseFrameLoop,
  resumeFrameLoop,
  startCameraCapture,
  startScreenCapture,
  stopCapture,
  stopCaptureAndNotify
} from './multimodal-capture'

interface Pending<T> {
  promise: Promise<T>
  reject: (reason?: unknown) => void
  resolve: (value: T) => void
}

function deferred<T>(): Pending<T> {
  let reject!: (reason?: unknown) => void
  let resolve!: (value: T) => void

  const promise = new Promise<T>((done, fail) => {
    reject = fail
    resolve = done
  })

  return { promise, reject, resolve }
}

function fakeStream(options: { audio?: boolean } = {}) {
  const track = {
    addEventListener: vi.fn(),
    stop: vi.fn()
  }

  const audioTrack = {
    addEventListener: vi.fn(),
    clone: vi.fn(),
    stop: vi.fn()
  }

  const stream = {
    getAudioTracks: vi.fn(() => options.audio ? [audioTrack] : []),
    getTracks: vi.fn(() => options.audio ? [track, audioTrack] : [track]),
    getVideoTracks: vi.fn(() => [track])
  } as unknown as MediaStream

  return { audioTrack, stream, track }
}

describe('fresh-draft multimodal capture binding', () => {
  const originalCreateElement = document.createElement.bind(document)
  let createElementSpy: ReturnType<typeof vi.spyOn>
  let fileReader: typeof FileReader

  beforeEach(() => {
    vi.useFakeTimers()
    mockSessionId.set('')
    mockGatewayAtom.set({
      connectionState: 'open',
      notify: vi.fn((_method: string, _params?: Record<string, unknown>) => 0),
      request: vi.fn((method: string) => method === 'multimodal.frame'
        ? Promise.resolve({ buffered: true })
        : Promise.resolve({ ok: true }))
    })
    openSourcePicker.mockReset()
    pushMmToast.mockReset()
    openSourcePicker.mockResolvedValue({
      id: 'screen:1',
      name: 'Display 1',
      shareAudio: false
    })
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { screenShareSystemAudio: false },
      writable: true
    })
    stopCapture()
    configureDraftCaptureSessionEnsurer(async () => {
      mockSessionId.set('runtime-default-draft')

      return 'runtime-default-draft'
    })
    vi.spyOn(console, 'info').mockImplementation(() => undefined)
    vi.spyOn(console, 'warn').mockImplementation(() => undefined)

    fileReader = globalThis.FileReader

    class ImmediateFileReader {
      error: DOMException | null = null
      onerror: ((this: FileReader, event: ProgressEvent<FileReader>) => unknown) | null = null
      onload: ((this: FileReader, event: ProgressEvent<FileReader>) => unknown) | null = null
      result: string | ArrayBuffer | null = null

      readAsDataURL(): void {
        this.result = 'data:image/jpeg;base64,Zmlyc3QtZnJhbWU='
        this.onload?.call(this as unknown as FileReader, {} as ProgressEvent<FileReader>)
      }
    }
    vi.stubGlobal('FileReader', ImmediateFileReader)

    createElementSpy = vi.spyOn(document, 'createElement').mockImplementation((tagName: string) => {
      if (tagName.toLowerCase() === 'video') {
        return {
          muted: false,
          paused: false,
          play: vi.fn(async () => undefined),
          playsInline: false,
          srcObject: null,
          videoHeight: 480,
          videoWidth: 640
        } as unknown as HTMLVideoElement
      }

      if (tagName.toLowerCase() === 'canvas') {
        return {
          getContext: vi.fn(() => ({ drawImage: vi.fn() })),
          height: 0,
          toBlob: vi.fn((done: BlobCallback) => done(new Blob(['jpeg']))),
          width: 0
        } as unknown as HTMLCanvasElement
      }

      return originalCreateElement(tagName)
    })
  })

  afterEach(() => {
    stopCapture()
    configureDraftCaptureSessionEnsurer(null)
    createElementSpy.mockRestore()
    vi.stubGlobal('FileReader', fileReader)
    vi.restoreAllMocks()
    vi.useRealTimers()
  })

  it.each([
    { source: 'camera' as const },
    { source: 'screen' as const }
  ])('starts recording immediately after fresh-draft $source permission succeeds', async ({ source }) => {
    const { stream, track } = fakeStream()
    const notify = vi.fn((_method: string, _params?: Record<string, unknown>) => 0)

    const request = vi.fn((method: string) => method === 'multimodal.frame'
      ? Promise.resolve({ buffered: true })
      : Promise.resolve({ ok: true }))

    const ensureSession = vi.fn(async () => {
      mockSessionId.set(`runtime-fresh-${source}`)

      return `runtime-fresh-${source}`
    })

    configureDraftCaptureSessionEnsurer(ensureSession)
    mockGatewayAtom.set({ connectionState: 'open', notify, request })
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: source === 'camera'
        ? { getUserMedia: vi.fn(async () => stream) }
        : { getDisplayMedia: vi.fn(async () => stream) }
    })

    if (source === 'camera') {
      await startCameraCapture()
    } else {
      await startScreenCapture()
    }

    expect(ensureSession).toHaveBeenCalledTimes(1)
    expect(request).toHaveBeenCalledWith(
      'multimodal.source_stopped',
      expect.objectContaining({
        session_id: `runtime-fresh-${source}`,
        source_type: source,
        started: true
      }),
      expect.any(Number)
    )
    expect(request).toHaveBeenCalledWith(
      'multimodal.frame',
      expect.objectContaining({
        session_id: `runtime-fresh-${source}`,
        source_type: source
      }),
      expect.any(Number)
    )
    expect($mmSource.get()).toBe(source)
    expect($mmCaptureDebug.get().code).toBe('sending')
    expect(track.stop).not.toHaveBeenCalled()
    expect(vi.getTimerCount()).toBe(1)

    await vi.advanceTimersByTimeAsync(500)

    expect(notify).toHaveBeenCalledWith(
      'multimodal.frame',
      expect.objectContaining({ session_id: `runtime-fresh-${source}` })
    )
  })

  it('orders fresh-draft create, source activation, first-frame ACK, then periodic delivery', async () => {
    const { stream } = fakeStream()
    const sourceStarted = deferred<Record<string, never>>()
    const firstFrame = deferred<{ buffered: boolean }>()
    const events: string[] = []

    const notify = vi.fn((method: string) => {
      if (method === 'multimodal.frame') {
        events.push('periodic-frame')
      }

      return 0
    })

    const request = vi.fn((method: string) => {
      if (method === 'multimodal.source_stopped') {
        events.push('source-start')

        return sourceStarted.promise
      }

      if (method === 'multimodal.frame') {
        events.push('first-frame')

        return firstFrame.promise
      }

      return Promise.resolve({})
    })

    const ensureSession = vi.fn(async () => {
      events.push('session')
      mockSessionId.set('runtime-fresh-order')

      return 'runtime-fresh-order'
    })

    configureDraftCaptureSessionEnsurer(ensureSession)
    mockGatewayAtom.set({ connectionState: 'open', notify, request })
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia: vi.fn(async () => stream) }
    })

    const starting = startCameraCapture()

    await vi.waitFor(() => expect(events).toEqual(['session', 'source-start']))
    expect(notify).not.toHaveBeenCalled()
    expect(vi.getTimerCount()).toBe(0)

    sourceStarted.resolve({})
    await vi.waitFor(() => expect(events).toEqual(['session', 'source-start', 'first-frame']))
    expect(notify).not.toHaveBeenCalled()
    expect(vi.getTimerCount()).toBe(0)

    firstFrame.resolve({ buffered: true })
    await starting
    expect(vi.getTimerCount()).toBe(1)

    await vi.advanceTimersByTimeAsync(500)

    expect(events).toEqual(['session', 'source-start', 'first-frame', 'periodic-frame'])
  })

  it('single-flights capture startup and the first-prompt barrier onto one activation', async () => {
    const { stream } = fakeStream()
    const sourceStarted = deferred<Record<string, never>>()
    const firstFrame = deferred<{ buffered: boolean }>()
    const ensureSession = vi.fn(async () => 'must-not-create')

    const request = vi.fn((method: string) => {
      if (method === 'multimodal.source_stopped') {
        return sourceStarted.promise
      }

      if (method === 'multimodal.frame') {
        return firstFrame.promise
      }

      return Promise.resolve({})
    })

    mockSessionId.set('runtime-shared-barrier')
    configureDraftCaptureSessionEnsurer(ensureSession)
    mockGatewayAtom.set({
      connectionState: 'open',
      notify: vi.fn((_method: string, _params?: Record<string, unknown>) => 0),
      request
    })
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia: vi.fn(async () => stream) }
    })

    const captureStartup = startCameraCapture()

    await vi.waitFor(() =>
      expect(request.mock.calls.filter(([method]) => method === 'multimodal.source_stopped'))
        .toHaveLength(1)
    )

    // prompt.submit reaches this same barrier while the media button's
    // source-start RPC is still pending. It must join that exact transaction.
    const firstPromptBarrier = ensureCaptureBoundToSession('runtime-shared-barrier')

    expect(request.mock.calls.filter(([method]) => method === 'multimodal.source_stopped'))
      .toHaveLength(1)

    sourceStarted.resolve({})
    await vi.waitFor(() =>
      expect(request.mock.calls.filter(([method]) => method === 'multimodal.frame'))
        .toHaveLength(1)
    )

    expect(request.mock.calls.filter(([method]) => method === 'multimodal.source_stopped'))
      .toHaveLength(1)

    firstFrame.resolve({ buffered: true })
    await expect(Promise.all([captureStartup, firstPromptBarrier])).resolves.toEqual([
      undefined,
      undefined
    ])

    expect(ensureSession).not.toHaveBeenCalled()
    expect(request.mock.calls.filter(([method]) => method === 'multimodal.source_stopped'))
      .toHaveLength(1)
    expect(request.mock.calls.filter(([method]) => method === 'multimodal.frame'))
      .toHaveLength(1)
    expect(vi.getTimerCount()).toBe(1)
  })

  it.each([
    { source: 'camera' as const },
    { source: 'screen' as const }
  ])('does not create a session when fresh-draft $source permission is rejected', async ({ source }) => {
    const ensureSession = vi.fn(async () => 'must-not-exist')
    const denied = new DOMException('permission denied', 'NotAllowedError')

    configureDraftCaptureSessionEnsurer(ensureSession)
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: source === 'camera'
        ? { getUserMedia: vi.fn(async () => { throw denied }) }
        : { getDisplayMedia: vi.fn(async () => { throw denied }) }
    })

    const starting = source === 'camera' ? startCameraCapture() : startScreenCapture()

    await expect(starting).rejects.toBe(denied)
    expect(ensureSession).not.toHaveBeenCalled()
    expect($mmSource.get()).toBe('none')
  })

  it('does not create a session when the screen-source picker is cancelled', async () => {
    const ensureSession = vi.fn(async () => 'must-not-exist')
    const getDisplayMedia = vi.fn()

    configureDraftCaptureSessionEnsurer(ensureSession)
    openSourcePicker.mockResolvedValueOnce(null)
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getDisplayMedia }
    })

    await startScreenCapture()

    expect(ensureSession).not.toHaveBeenCalled()
    expect(getDisplayMedia).not.toHaveBeenCalled()
    expect($mmSource.get()).toBe('none')
  })

  it('does not attach or create when stop wins over pending camera permission', async () => {
    const permission = deferred<MediaStream>()
    const { stream, track } = fakeStream()
    const ensureSession = vi.fn(async () => 'must-not-exist')
    const getUserMedia = vi.fn(() => permission.promise)
    const request = vi.fn(async () => ({ ok: true }))

    configureDraftCaptureSessionEnsurer(ensureSession)
    mockGatewayAtom.set({
      connectionState: 'open',
      notify: vi.fn((_method: string, _params?: Record<string, unknown>) => 0),
      request
    })
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia }
    })

    const starting = startCameraCapture()

    expect(getUserMedia).toHaveBeenCalledTimes(1)
    stopCaptureAndNotify()
    permission.resolve(stream)
    await starting

    expect(track.stop).toHaveBeenCalledTimes(1)
    expect(ensureSession).not.toHaveBeenCalled()
    expect(request).not.toHaveBeenCalled()
    expect($mmSource.get()).toBe('none')
    expect(vi.getTimerCount()).toBe(0)
  })

  it.each([
    { source: 'camera' as const },
    { source: 'screen' as const }
  ])('does not revive $source after stop/New wins while the preview is starting', async ({ source }) => {
    const playback = deferred<void>()
    const play = vi.fn(() => playback.promise)
    const { stream, track } = fakeStream()
    const ensureSession = vi.fn(async () => {
      mockSessionId.set(`runtime-stale-${source}`)

      return `runtime-stale-${source}`
    })
    const request = vi.fn(async () => ({ ok: true }))

    createElementSpy.mockImplementation((tagName: string) => {
      if (tagName.toLowerCase() === 'video') {
        return {
          muted: false,
          paused: true,
          play,
          playsInline: false,
          srcObject: null,
          videoHeight: 480,
          videoWidth: 640
        } as unknown as HTMLVideoElement
      }

      return originalCreateElement(tagName)
    })
    configureDraftCaptureSessionEnsurer(ensureSession)
    mockGatewayAtom.set({
      connectionState: 'open',
      notify: vi.fn((_method: string, _params?: Record<string, unknown>) => 0),
      request
    })
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: source === 'camera'
        ? { getUserMedia: vi.fn(async () => stream) }
        : { getDisplayMedia: vi.fn(async () => stream) }
    })

    const starting = source === 'camera' ? startCameraCapture() : startScreenCapture()

    await vi.waitFor(() => expect(play).toHaveBeenCalledTimes(1))
    expect($mmSource.get()).toBe('none')
    stopCaptureAndNotify()
    playback.resolve()
    await starting

    expect(track.stop).toHaveBeenCalled()
    expect(ensureSession).not.toHaveBeenCalled()
    expect(request).not.toHaveBeenCalledWith(
      'multimodal.source_stopped',
      expect.objectContaining({ started: true }),
      expect.any(Number)
    )
    expect($mmSource.get()).toBe('none')
    expect($mmStream.get()).toBeNull()
    expect(vi.getTimerCount()).toBe(0)
  })

  it('does not open display permission or create when stop wins over the pending screen picker', async () => {
    const picker = deferred<{ id: string; name: string; shareAudio: boolean } | null>()
    const { stream } = fakeStream()

    const ensureSession = vi.fn(async () => {
      mockSessionId.set('must-not-exist')

      return 'must-not-exist'
    })

    const getDisplayMedia = vi.fn(async () => stream)

    configureDraftCaptureSessionEnsurer(ensureSession)
    openSourcePicker.mockReturnValueOnce(picker.promise)
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getDisplayMedia }
    })

    const starting = startScreenCapture()

    await vi.waitFor(() => expect(openSourcePicker).toHaveBeenCalledTimes(1))
    stopCaptureAndNotify()
    picker.resolve({ id: 'screen:late', name: 'Late display', shareAudio: false })
    await starting

    expect(getDisplayMedia).not.toHaveBeenCalled()
    expect(ensureSession).not.toHaveBeenCalled()
    expect($mmSource.get()).toBe('none')
    expect(vi.getTimerCount()).toBe(0)
  })

  it('does not attach or create when stop wins over pending display permission', async () => {
    const permission = deferred<MediaStream>()
    const { stream, track } = fakeStream()
    const ensureSession = vi.fn(async () => 'must-not-exist')
    const getDisplayMedia = vi.fn(() => permission.promise)
    const request = vi.fn(async () => ({ ok: true }))

    configureDraftCaptureSessionEnsurer(ensureSession)
    mockGatewayAtom.set({
      connectionState: 'open',
      notify: vi.fn((_method: string, _params?: Record<string, unknown>) => 0),
      request
    })
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getDisplayMedia }
    })

    const starting = startScreenCapture()

    await vi.waitFor(() => expect(getDisplayMedia).toHaveBeenCalledTimes(1))
    stopCaptureAndNotify()
    permission.resolve(stream)
    await starting

    expect(track.stop).toHaveBeenCalledTimes(1)
    expect(ensureSession).not.toHaveBeenCalled()
    expect(request).not.toHaveBeenCalled()
    expect($mmSource.get()).toBe('none')
    expect(vi.getTimerCount()).toBe(0)
  })

  it('does not revive capture when stop wins over a late fresh-draft session create', async () => {
    const { stream, track } = fakeStream()
    const session = deferred<string | null>()
    const ensureSession = vi.fn(() => session.promise)
    const request = vi.fn(async () => ({ ok: true }))

    configureDraftCaptureSessionEnsurer(ensureSession)
    mockGatewayAtom.set({
      connectionState: 'open',
      notify: vi.fn((_method: string, _params?: Record<string, unknown>) => 0),
      request
    })
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia: vi.fn(async () => stream) }
    })

    const starting = startCameraCapture()

    await vi.waitFor(() => expect(ensureSession).toHaveBeenCalledTimes(1))
    stopCaptureAndNotify()
    mockSessionId.set('runtime-too-late')
    session.resolve('runtime-too-late')
    await starting

    expect(track.stop).toHaveBeenCalledTimes(1)
    expect($mmSource.get()).toBe('none')
    expect(request).not.toHaveBeenCalledWith(
      'multimodal.source_stopped',
      expect.objectContaining({ started: true }),
      expect.any(Number)
    )
    expect(vi.getTimerCount()).toBe(0)
  })

  it('does not transfer a late fresh-draft create into a user-selected session', async () => {
    const { stream, track } = fakeStream()
    const session = deferred<string | null>()
    const ensureSession = vi.fn(() => session.promise)
    const request = vi.fn(async () => ({ ok: true }))

    configureDraftCaptureSessionEnsurer(ensureSession)
    mockGatewayAtom.set({
      connectionState: 'open',
      notify: vi.fn((_method: string, _params?: Record<string, unknown>) => 0),
      request
    })
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia: vi.fn(async () => stream) }
    })

    const starting = startCameraCapture()

    await vi.waitFor(() => expect(ensureSession).toHaveBeenCalledTimes(1))
    mockSessionId.set('runtime-user-selected-B')
    session.resolve('runtime-created-for-old-draft-A')
    await expect(starting).rejects.toThrow('后端会话创建已取消')

    expect(track.stop).toHaveBeenCalledTimes(1)
    expect($mmSource.get()).toBe('none')
    expect(request).not.toHaveBeenCalledWith(
      'multimodal.source_stopped',
      expect.objectContaining({ started: true }),
      expect.any(Number)
    )
    expect(vi.getTimerCount()).toBe(0)
  })

  it.each([
    { expectedAudio: false, shareAudio: true, supported: false },
    { expectedAudio: true, shareAudio: true, supported: true },
    { expectedAudio: false, shareAudio: false, supported: true }
  ])(
    'requests audio=$expectedAudio when system-audio supported=$supported and picker=$shareAudio',
    async ({ expectedAudio, shareAudio, supported }) => {
      const { stream } = fakeStream({ audio: expectedAudio })
      const getDisplayMedia = vi.fn(async () => stream)

      Object.assign(window.hermesDesktop, { screenShareSystemAudio: supported })
      openSourcePicker.mockResolvedValueOnce({
        id: 'screen:1',
        name: 'Display 1',
        shareAudio
      })
      Object.defineProperty(navigator, 'mediaDevices', {
        configurable: true,
        value: { getDisplayMedia }
      })

      await startScreenCapture()

      expect(getDisplayMedia).toHaveBeenCalledWith(
        expect.objectContaining({ audio: expectedAudio })
      )
    }
  )

  it('reports an actionable error when system audio was requested but no track arrives', async () => {
    const { stream } = fakeStream()
    const getDisplayMedia = vi.fn(async () => stream)

    Object.assign(window.hermesDesktop, { screenShareSystemAudio: true })
    openSourcePicker.mockResolvedValueOnce({
      id: 'screen:1',
      name: 'Display 1',
      shareAudio: true
    })
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getDisplayMedia }
    })

    await startScreenCapture()
    await vi.waitFor(() => expect(pushMmToast).toHaveBeenCalledTimes(1))

    expect(console.warn).toHaveBeenCalledWith(
      expect.stringContaining('没有返回音频轨道')
    )
    expect(pushMmToast).toHaveBeenCalledWith({
      level: 'error',
      text: expect.stringContaining('允许 Argus 录制屏幕与系统音频')
    })
  })

  it('does not report a false error when the requested system-audio track exists', async () => {
    const { stream } = fakeStream({ audio: true })

    Object.assign(window.hermesDesktop, { screenShareSystemAudio: true })
    openSourcePicker.mockResolvedValueOnce({
      id: 'screen:1',
      name: 'Display 1',
      shareAudio: true
    })
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getDisplayMedia: vi.fn(async () => stream) }
    })

    await startScreenCapture()
    await Promise.resolve()

    expect(console.warn).not.toHaveBeenCalledWith(
      expect.stringContaining('没有返回音频轨道')
    )
    expect(pushMmToast).not.toHaveBeenCalled()
  })

  it.each([
    { dpr: 1, platform: 'MacIntel', profile: 'mac DPR1' },
    { dpr: 2, platform: 'MacIntel', profile: 'mac DPR2' },
    { dpr: 2, platform: 'Win32', profile: 'non-mac HiDPI' }
  ])('requests the light 1280x720@4 screen profile on $profile', async ({
    dpr,
    platform
  }) => {
    const { stream } = fakeStream()
    const getDisplayMedia = vi.fn(async () => stream)

    vi.spyOn(navigator, 'platform', 'get').mockReturnValue(platform)
    vi.spyOn(navigator, 'userAgent', 'get').mockReturnValue('Argus Desktop')
    vi.spyOn(window, 'devicePixelRatio', 'get').mockReturnValue(dpr)
    mockGatewayAtom.set({
      connectionState: 'open',
      notify: vi.fn((_method: string, _params?: Record<string, unknown>) => 0),
      request: vi.fn((method: string) => method === 'multimodal.frame'
        ? Promise.resolve({ buffered: true })
        : Promise.resolve({ ok: true }))
    })
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getDisplayMedia }
    })

    await startScreenCapture()

    expect(getDisplayMedia).toHaveBeenCalledWith({
      audio: false,
      video: {
        width: { ideal: 1280, max: 1280 },
        height: { ideal: 720, max: 720 },
        frameRate: { ideal: 4, max: 4 }
      }
    })
  })

  it('requests the normal 1920x1080@4 screen profile on non-mac DPR1', async () => {
    const { stream } = fakeStream()
    const getDisplayMedia = vi.fn(async () => stream)

    vi.spyOn(navigator, 'platform', 'get').mockReturnValue('Win32')
    vi.spyOn(navigator, 'userAgent', 'get').mockReturnValue('Argus Desktop')
    vi.spyOn(window, 'devicePixelRatio', 'get').mockReturnValue(1)
    mockGatewayAtom.set({
      connectionState: 'open',
      notify: vi.fn((_method: string, _params?: Record<string, unknown>) => 0),
      request: vi.fn((method: string) => method === 'multimodal.frame'
        ? Promise.resolve({ buffered: true })
        : Promise.resolve({ ok: true }))
    })
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getDisplayMedia }
    })

    await startScreenCapture()

    expect(getDisplayMedia).toHaveBeenCalledWith({
      audio: false,
      video: {
        width: { ideal: 1920, max: 1920 },
        height: { ideal: 1080, max: 1080 },
        frameRate: { ideal: 4, max: 4 }
      }
    })
  })

  it.each([
    {
      expectedHeight: 720,
      expectedQuality: 0.72,
      expectedWidth: 1280,
      platform: 'Win32',
      profile: 'camera',
      source: 'camera' as const
    },
    {
      expectedHeight: 720,
      expectedQuality: 0.72,
      expectedWidth: 1280,
      platform: 'MacIntel',
      profile: 'light screen',
      source: 'screen' as const
    },
    {
      expectedHeight: 1080,
      expectedQuality: 0.8,
      expectedWidth: 1920,
      platform: 'Win32',
      profile: 'normal screen',
      source: 'screen' as const
    }
  ])('caps and encodes the $profile frame at its transport profile', async ({
    expectedHeight,
    expectedQuality,
    expectedWidth,
    platform,
    source
  }) => {
    const { stream } = fakeStream()
    const drawImage = vi.fn()

    const toBlob = vi.fn((done: BlobCallback, _type?: string, _quality?: number) => {
      done(new Blob(['jpeg']))
    })

    const canvas = {
      getContext: vi.fn(() => ({ drawImage })),
      height: 0,
      toBlob,
      width: 0
    } as unknown as HTMLCanvasElement

    createElementSpy.mockImplementation((tagName: string) => {
      if (tagName.toLowerCase() === 'video') {
        return {
          muted: false,
          paused: false,
          play: vi.fn(async () => undefined),
          playsInline: false,
          srcObject: null,
          videoHeight: 2160,
          videoWidth: 3840
        } as unknown as HTMLVideoElement
      }

      if (tagName.toLowerCase() === 'canvas') {
        return canvas
      }

      return originalCreateElement(tagName)
    })
    vi.spyOn(navigator, 'platform', 'get').mockReturnValue(platform)
    vi.spyOn(navigator, 'userAgent', 'get').mockReturnValue('Argus Desktop')
    vi.spyOn(window, 'devicePixelRatio', 'get').mockReturnValue(1)

    const request = vi.fn((
      method: string,
      _params?: Record<string, unknown>,
      _timeoutMs?: number
    ) => method === 'multimodal.frame'
      ? Promise.resolve({ buffered: true })
      : Promise.resolve({ ok: true }))

    mockSessionId.set(`runtime-${source}-profile`)
    mockGatewayAtom.set({
      connectionState: 'open',
      notify: vi.fn((_method: string, _params?: Record<string, unknown>) => 0),
      request
    })
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: source === 'camera'
        ? { getUserMedia: vi.fn(async () => stream) }
        : { getDisplayMedia: vi.fn(async () => stream) }
    })

    if (source === 'camera') {
      await startCameraCapture()
    } else {
      await startScreenCapture()
    }

    expect(canvas.width).toBe(expectedWidth)
    expect(canvas.height).toBe(expectedHeight)
    expect(drawImage).toHaveBeenCalledWith(
      expect.anything(), 0, 0, expectedWidth, expectedHeight
    )
    expect(toBlob).toHaveBeenCalledWith(
      expect.any(Function), 'image/jpeg', expectedQuality
    )
  })

  it('keeps the latest screen intent when an older camera permission resolves late', async () => {
    const camera = fakeStream()
    const screen = fakeStream()
    const cameraPermission = deferred<MediaStream>()
    const screenPermission = deferred<MediaStream>()
    const getUserMedia = vi.fn(() => cameraPermission.promise)
    const getDisplayMedia = vi.fn(() => screenPermission.promise)

    mockGatewayAtom.set({
      connectionState: 'open',
      notify: vi.fn((_method: string, _params?: Record<string, unknown>) => 0),
      request: vi.fn((method: string) => method === 'multimodal.frame'
        ? Promise.resolve({ buffered: true })
        : Promise.resolve({ ok: true }))
    })
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getDisplayMedia, getUserMedia }
    })

    const cameraStart = startCameraCapture()
    const screenStart = startScreenCapture()

    await vi.waitFor(() => expect(getDisplayMedia).toHaveBeenCalledTimes(1))

    screenPermission.resolve(screen.stream)
    await screenStart
    expect($mmSource.get()).toBe('screen')

    cameraPermission.resolve(camera.stream)
    await cameraStart

    expect($mmSource.get()).toBe('screen')
    expect(camera.track.stop).toHaveBeenCalledTimes(1)
    expect(screen.track.stop).not.toHaveBeenCalled()
  })

  it('discards an older camera permission even when it resolves before the requested screen', async () => {
    const camera = fakeStream()
    const screen = fakeStream()
    const cameraPermission = deferred<MediaStream>()
    const screenPermission = deferred<MediaStream>()
    const getUserMedia = vi.fn(() => cameraPermission.promise)
    const getDisplayMedia = vi.fn(() => screenPermission.promise)

    mockGatewayAtom.set({
      connectionState: 'open',
      notify: vi.fn((_method: string, _params?: Record<string, unknown>) => 0),
      request: vi.fn((method: string) => method === 'multimodal.frame'
        ? Promise.resolve({ buffered: true })
        : Promise.resolve({ ok: true }))
    })
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getDisplayMedia, getUserMedia }
    })

    const cameraStart = startCameraCapture()
    const screenStart = startScreenCapture()

    await vi.waitFor(() => expect(getDisplayMedia).toHaveBeenCalledTimes(1))

    cameraPermission.resolve(camera.stream)
    await cameraStart
    const sourceAfterOldPermission = $mmSource.get()
    const oldTrackStopsBeforeScreen = camera.track.stop.mock.calls.length

    screenPermission.resolve(screen.stream)
    await screenStart

    expect(sourceAfterOldPermission).not.toBe('camera')
    expect(oldTrackStopsBeforeScreen).toBe(1)
    expect($mmSource.get()).toBe('screen')
    expect(screen.track.stop).not.toHaveBeenCalled()
  })

  it('fully tears down an existing-session stream when source activation is rejected', async () => {
    const { stream, track } = fakeStream()
    mockSessionId.set('runtime-existing')

    const request = vi.fn(async (
      method: string,
      params?: Record<string, unknown>,
      _timeoutMs?: number
    ) => {
      if (method === 'multimodal.source_stopped' && params?.started === true) {
        throw new Error('activation rejected')
      }

      return { ok: true }
    })

    mockGatewayAtom.set({
      connectionState: 'open',
      notify: vi.fn((_method: string, _params?: Record<string, unknown>) => 0),
      request
    })
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia: vi.fn(async () => stream) }
    })

    await expect(startCameraCapture()).rejects.toThrow('activation rejected')
    await Promise.resolve()

    expect(track.stop).toHaveBeenCalledTimes(1)
    expect($mmSource.get()).toBe('none')
    expect($mmCaptureDebug.get().code).toBe('idle')
    expect(vi.getTimerCount()).toBe(0)
    expect(request).toHaveBeenCalledWith(
      'multimodal.source_stopped',
      expect.objectContaining({
        session_id: 'runtime-existing',
        started: false
      }),
      expect.any(Number)
    )
  })

  it('does not wait for a hanging rollback before tearing down a timed-out activation', async () => {
    const { stream, track } = fakeStream()
    mockSessionId.set('runtime-timeout')
    const never = new Promise<Record<string, never>>(() => undefined)

    const request = vi.fn((
      method: string,
      params?: Record<string, unknown>,
      timeoutMs?: number
    ) => {
      if (method === 'multimodal.source_stopped' && params?.started === true) {
        return new Promise<Record<string, never>>((_, reject) => {
          setTimeout(() => reject(new Error('source activation timed out')), timeoutMs)
        })
      }

      if (method === 'multimodal.source_stopped' && params?.started === false) {
        return never
      }

      return Promise.resolve({})
    })

    mockGatewayAtom.set({
      connectionState: 'open',
      notify: vi.fn((_method: string, _params?: Record<string, unknown>) => 0),
      request
    })
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia: vi.fn(async () => stream) }
    })

    let rejected = false

    const starting = startCameraCapture().catch(() => {
      rejected = true
    })

    for (let step = 0; step < 10 && request.mock.calls.length === 0; step += 1) {
      await Promise.resolve()
    }

    const activation = request.mock.calls.find(
      ([method, params]) => method === 'multimodal.source_stopped'
        && params?.started === true
    )

    const activationTimeoutMs = Number(activation?.[2])

    expect(activationTimeoutMs).toBeGreaterThanOrEqual(120_000)
    await vi.advanceTimersByTimeAsync(activationTimeoutMs)
    await Promise.resolve()

    expect(request).toHaveBeenCalledWith(
      'multimodal.source_stopped',
      expect.objectContaining({
        session_id: 'runtime-timeout',
        started: false
      }),
      expect.any(Number)
    )

    const rollback = request.mock.calls.find(
      ([method, params]) => method === 'multimodal.source_stopped'
        && params?.started === false
    )

    const rollbackTimeoutMs = Number(rollback?.[2])

    expect(rollbackTimeoutMs).toBeGreaterThan(0)
    expect(rollbackTimeoutMs).toBeLessThan(activationTimeoutMs)
    expect(rejected).toBe(true)
    expect(track.stop).toHaveBeenCalledTimes(1)
    expect($mmSource.get()).toBe('none')
    expect($mmCaptureDebug.get().code).toBe('idle')

    await starting
  })

  it('waits for source_started acknowledgement before starting the frame loop', async () => {
    const { stream } = fakeStream()
    const sourceStarted = deferred<Record<string, never>>()
    const firstFrame = deferred<{ buffered: boolean }>()

    const request = vi.fn((
      method: string,
      _params?: Record<string, unknown>,
      _timeoutMs?: number
    ) => {
      if (method === 'multimodal.source_stopped') {return sourceStarted.promise}

      if (method === 'multimodal.frame') {return firstFrame.promise}

      return Promise.resolve({})
    })

    const gateway = {
      connectionState: 'open',
      notify: vi.fn((_method: string, _params?: Record<string, unknown>) => 0),
      request
    }

    mockGatewayAtom.set(gateway)
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia: vi.fn(async () => stream) }
    })

    mockSessionId.set('runtime-B')

    let bound = false

    const binding = startCameraCapture().then(() => {
      bound = true
    })

    for (let step = 0; step < 10 && request.mock.calls.length === 0; step += 1) {
      await Promise.resolve()
    }

    expect(request).toHaveBeenCalledWith(
      'multimodal.source_stopped',
      expect.objectContaining({
        session_id: 'runtime-B',
        source_type: 'camera',
        started: true
      }),
      expect.any(Number)
    )

    const activationCall = request.mock.calls.find(
      ([method]) => method === 'multimodal.source_stopped'
    )

    expect(activationCall?.[2]).toEqual(expect.any(Number))
    expect(Number(activationCall?.[2])).toBeGreaterThanOrEqual(120_000)
    expect(bound).toBe(false)
    expect($mmCaptureDebug.get().code).toBe('waiting_for_session')
    expect(gateway.notify).not.toHaveBeenCalled()
    expect(vi.getTimerCount()).toBe(0)

    sourceStarted.resolve({})

    for (let step = 0; step < 10 && request.mock.calls.length < 2; step += 1) {
      await Promise.resolve()
    }

    expect(bound).toBe(false)
    expect(request).toHaveBeenCalledWith(
      'multimodal.frame',
      expect.objectContaining({
        jpeg_b64: 'Zmlyc3QtZnJhbWU=',
        session_id: 'runtime-B',
        source_type: 'camera'
      }),
      expect.any(Number)
    )
    const firstFrameCall = request.mock.calls.find(([method]) => method === 'multimodal.frame')
    expect(Number(firstFrameCall?.[2])).toBeGreaterThan(0)
    expect(Number(firstFrameCall?.[2])).toBeLessThanOrEqual(2_000)
    expect(vi.getTimerCount()).toBe(0)

    firstFrame.resolve({ buffered: true })
    await binding

    expect(bound).toBe(true)
    expect($mmCaptureDebug.get().code).toBe('sending')
    // The ACK is the UI's recording boundary, so the first accepted frame must
    // be reflected immediately rather than leaving a truthful REC badge at 0.
    expect($mmCapStats.get().sent).toBe(1)
    expect(vi.getTimerCount()).toBe(1)
  })

  it('rolls back source_started when no first frame is acknowledged', async () => {
    const { stream } = fakeStream()

    const request = vi.fn(async (
      method: string,
      params?: Record<string, unknown>,
      _timeoutMs?: number
    ) => {
      if (method === 'multimodal.source_stopped' && params?.started === true) {return { ok: true }}

      if (method === 'multimodal.frame') {return { buffered: false }}

      return { ok: true }
    })

    mockGatewayAtom.set({
      connectionState: 'open',
      notify: vi.fn((_method: string, _params?: Record<string, unknown>) => 0),
      request
    })
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia: vi.fn(async () => stream) }
    })

    mockSessionId.set('runtime-rollback')

    const binding = startCameraCapture()
    const rejection = expect(binding).rejects.toThrow('首帧')
    await vi.advanceTimersByTimeAsync(2_100)

    await rejection
    expect(request).toHaveBeenCalledWith(
      'multimodal.source_stopped',
      expect.objectContaining({
        session_id: 'runtime-rollback',
        started: false
      }),
      expect.any(Number)
    )
  })

  it('does not let a late failed A binding invalidate the successful B rebind', async () => {
    const { stream } = fakeStream()
    const sourceStartedA = deferred<Record<string, never>>()
    const notify = vi.fn((_method: string, _params?: Record<string, unknown>) => 0)

    const request = vi.fn((
      method: string,
      params?: Record<string, unknown>,
      _timeoutMs?: number
    ) => {
      if (
        method === 'multimodal.source_stopped'
        && params?.started === true
        && params?.session_id === 'runtime-A'
      ) {
        return sourceStartedA.promise
      }

      if (method === 'multimodal.frame') {
        return Promise.resolve({ buffered: true })
      }

      return Promise.resolve({ ok: true })
    })

    mockGatewayAtom.set({ connectionState: 'open', notify, request })
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia: vi.fn(async () => stream) }
    })

    await startCameraCapture()
    mockSessionId.set('runtime-A')
    const bindingA = ensureCaptureBoundToSession('runtime-A')

    for (let step = 0; step < 10; step += 1) {
      if (request.mock.calls.some(
        ([method, params]) => method === 'multimodal.source_stopped'
          && params?.started === true
          && params?.session_id === 'runtime-A'
      )) {
        break
      }

      await Promise.resolve()
    }

    mockSessionId.set('runtime-B')
    await ensureCaptureBoundToSession('runtime-B')

    const startB = request.mock.calls.find(
      ([method, params]) => method === 'multimodal.source_stopped'
        && params?.started === true
        && params?.session_id === 'runtime-B'
    )

    const generationB = startB?.[1]?.capture_generation
    const oldFailure = expect(bindingA).rejects.toThrow('late A activation failure')

    sourceStartedA.reject(new Error('late A activation failure'))
    await oldFailure
    await vi.advanceTimersByTimeAsync(500)

    const periodicFramesForB = notify.mock.calls.filter(
      ([method, params]) => method === 'multimodal.frame'
        && params?.session_id === 'runtime-B'
    )

    expect(periodicFramesForB.length).toBeGreaterThan(0)
    expect(periodicFramesForB.every(
      ([, params]) => params?.capture_generation === generationB
    )).toBe(true)
    expect($mmSource.get()).toBe('camera')
  })

  it('keeps a successful same-session reconnect attempt alive after old A rejects', async () => {
    const { stream } = fakeStream()
    const sourceStartedA = deferred<Record<string, never>>()
    const notifyB = vi.fn((_method: string, _params?: Record<string, unknown>) => 0)

    const requestA = vi.fn((
      method: string,
      params?: Record<string, unknown>,
      _timeoutMs?: number
    ) => {
      if (method === 'multimodal.source_stopped' && params?.started === true) {
        return sourceStartedA.promise
      }

      return Promise.resolve({ ok: true })
    })

    const requestB = vi.fn((
      method: string,
      _params?: Record<string, unknown>,
      _timeoutMs?: number
    ) => {
      if (method === 'multimodal.frame') {
        return Promise.resolve({ buffered: true })
      }

      return Promise.resolve({ ok: true })
    })

    const gatewayA = {
      connectionState: 'open',
      notify: vi.fn((_method: string, _params?: Record<string, unknown>) => 0),
      request: requestA
    }

    const gatewayB = {
      connectionState: 'open',
      notify: notifyB,
      request: requestB
    }

    mockGatewayAtom.set(gatewayA)
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia: vi.fn(async () => stream) }
    })

    mockSessionId.set('runtime-same')
    const bindingA = startCameraCapture()

    for (let step = 0; step < 10 && requestA.mock.calls.length === 0; step += 1) {
      await Promise.resolve()
    }

    mockGatewayAtom.set(gatewayB)
    await ensureCaptureBoundToSession('runtime-same')

    const startA = requestA.mock.calls.find(
      ([method, params]) => method === 'multimodal.source_stopped'
        && params?.started === true
    )

    const startB = requestB.mock.calls.find(
      ([method, params]) => method === 'multimodal.source_stopped'
        && params?.started === true
    )

    const attemptA = startA?.[1]?.capture_attempt_id
    const attemptB = startB?.[1]?.capture_attempt_id

    expect(attemptA).toEqual(expect.any(String))
    expect(attemptB).toEqual(expect.any(String))
    expect(attemptB).not.toBe(attemptA)
    expect(startB?.[1]?.capture_generation).toBe(startA?.[1]?.capture_generation)

    sourceStartedA.reject(new Error('old transport rejected'))
    await bindingA.catch(() => undefined)
    await vi.advanceTimersByTimeAsync(500)

    const rollbackA = requestA.mock.calls.find(
      ([method, params]) => method === 'multimodal.source_stopped'
        && params?.started === false
    )

    const firstFrameB = requestB.mock.calls.find(
      ([method]) => method === 'multimodal.frame'
    )

    const periodicFramesB = notifyB.mock.calls.filter(
      ([method]) => method === 'multimodal.frame'
    )

    expect(rollbackA?.[1]?.capture_attempt_id).toBe(attemptA)
    expect(firstFrameB?.[1]?.capture_attempt_id).toBe(attemptB)
    expect(periodicFramesB.length).toBeGreaterThan(0)
    expect(periodicFramesB.every(
      ([, params]) => params?.capture_attempt_id === attemptB
    )).toBe(true)
    expect($mmSource.get()).toBe('camera')
    expect(vi.getTimerCount()).toBe(1)
  })

  it('rebinds on the same gateway object after reconnect without releasing the stream', async () => {
    const { stream, track } = fakeStream()
    const sourceStartedA = deferred<Record<string, never>>()
    const notify = vi.fn((_method: string, _params?: Record<string, unknown>) => 0)
    let startCount = 0

    const request = vi.fn((
      method: string,
      params?: Record<string, unknown>,
      _timeoutMs?: number
    ) => {
      if (method === 'multimodal.source_stopped' && params?.started === true) {
        startCount += 1

        if (startCount === 1) {
          return sourceStartedA.promise
        }

        return Promise.resolve({ ok: true })
      }

      if (method === 'multimodal.frame') {
        return Promise.resolve({ buffered: true })
      }

      return Promise.resolve({ ok: true })
    })

    const gateway = {
      connectionState: 'open',
      notify,
      request
    }

    mockGatewayAtom.set(gateway)
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia: vi.fn(async () => stream) }
    })

    mockSessionId.set('runtime-same-object')
    const oldBinding = startCameraCapture()
    const oldSettled = oldBinding.catch(() => undefined)

    for (let step = 0; step < 10 && startCount === 0; step += 1) {
      await Promise.resolve()
    }

    gateway.connectionState = 'connecting'
    pauseFrameLoop()
    gateway.connectionState = 'open'
    resumeFrameLoop()

    for (let step = 0; step < 20 && startCount < 2; step += 1) {
      await Promise.resolve()
    }

    const starts = request.mock.calls.filter(
      ([method, params]) => method === 'multimodal.source_stopped'
        && params?.started === true
    )

    if (starts.length < 2) {
      sourceStartedA.reject(new Error('release stale binding after failed assertion'))
      await oldSettled
    }

    expect(starts).toHaveLength(2)
    const attemptA = starts[0]?.[1]?.capture_attempt_id
    const attemptB = starts[1]?.[1]?.capture_attempt_id

    expect(attemptB).not.toBe(attemptA)
    expect(starts[1]?.[1]?.capture_generation).toBe(starts[0]?.[1]?.capture_generation)

    await ensureCaptureBoundToSession('runtime-same-object')
    sourceStartedA.reject(new Error('old socket closed'))
    await oldSettled
    await vi.advanceTimersByTimeAsync(500)

    const rollbackA = request.mock.calls.find(
      ([method, params]) => method === 'multimodal.source_stopped'
        && params?.started === false
        && params?.capture_attempt_id === attemptA
    )

    const periodicFramesB = notify.mock.calls.filter(
      ([method, params]) => method === 'multimodal.frame'
        && params?.capture_attempt_id === attemptB
    )

    expect(rollbackA?.[1]?.capture_generation).toBe(starts[0]?.[1]?.capture_generation)
    expect(periodicFramesB.length).toBeGreaterThan(0)
    expect(track.stop).not.toHaveBeenCalled()
    expect($mmSource.get()).toBe('camera')
    expect(vi.getTimerCount()).toBe(1)
  })

  it('stops advertising REC while an active capture is paused for reconnect', async () => {
    const { stream, track } = fakeStream()

    mockSessionId.set('runtime-paused-recording')
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia: vi.fn(async () => stream) }
    })

    await startCameraCapture()

    expect($mmCaptureDebug.get().code).toBe('sending')
    expect(vi.getTimerCount()).toBe(1)

    pauseFrameLoop()

    expect($mmCaptureDebug.get().code).toBe('gateway_not_open')
    expect($mmSource.get()).toBe('camera')
    expect(track.stop).not.toHaveBeenCalled()
    expect(vi.getTimerCount()).toBe(0)
  })

  it('preserves an existing-session stream when its attach-time attempt loses to reconnect', async () => {
    const { stream, track } = fakeStream()
    const sourceStartedA = deferred<Record<string, never>>()
    const notify = vi.fn((_method: string, _params?: Record<string, unknown>) => 0)
    let startCount = 0

    const request = vi.fn((
      method: string,
      params?: Record<string, unknown>,
      _timeoutMs?: number
    ) => {
      if (method === 'multimodal.source_stopped' && params?.started === true) {
        startCount += 1

        return startCount === 1
          ? sourceStartedA.promise
          : Promise.resolve({ ok: true })
      }

      if (method === 'multimodal.frame') {
        return Promise.resolve({ buffered: true })
      }

      return Promise.resolve({ ok: true })
    })

    const gateway = {
      connectionState: 'open',
      notify,
      request
    }

    mockSessionId.set('runtime-existing-reconnect')
    mockGatewayAtom.set(gateway)
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia: vi.fn(async () => stream) }
    })

    const attachResult = startCameraCapture().catch(error => error)

    for (let step = 0; step < 10 && startCount === 0; step += 1) {
      await Promise.resolve()
    }

    gateway.connectionState = 'connecting'
    pauseFrameLoop()
    gateway.connectionState = 'open'
    resumeFrameLoop()

    for (let step = 0; step < 20 && startCount < 2; step += 1) {
      await Promise.resolve()
    }

    const starts = request.mock.calls.filter(
      ([method, params]) => method === 'multimodal.source_stopped'
        && params?.started === true
    )

    expect(starts).toHaveLength(2)
    const attemptA = starts[0]?.[1]?.capture_attempt_id
    const attemptB = starts[1]?.[1]?.capture_attempt_id

    expect(attemptB).not.toBe(attemptA)

    await ensureCaptureBoundToSession('runtime-existing-reconnect')

    sourceStartedA.reject(new Error('old attach transport closed'))
    await attachResult
    await vi.advanceTimersByTimeAsync(500)

    const rollbackA = request.mock.calls.find(
      ([method, params]) => method === 'multimodal.source_stopped'
        && params?.started === false
        && params?.capture_attempt_id === attemptA
    )

    const periodicFramesB = notify.mock.calls.filter(
      ([method, params]) => method === 'multimodal.frame'
        && params?.capture_attempt_id === attemptB
    )

    expect(rollbackA?.[1]?.capture_generation).toBe(starts[0]?.[1]?.capture_generation)
    expect(periodicFramesB.length).toBeGreaterThan(0)
    expect(track.stop).not.toHaveBeenCalled()
    expect($mmSource.get()).toBe('camera')
    expect(vi.getTimerCount()).toBe(1)
  })

  it('rolls back a late successful A activation without disturbing B', async () => {
    const { stream } = fakeStream()
    const sourceStartedA = deferred<Record<string, never>>()
    const notify = vi.fn((_method: string, _params?: Record<string, unknown>) => 0)

    const request = vi.fn((
      method: string,
      params?: Record<string, unknown>,
      _timeoutMs?: number
    ) => {
      if (
        method === 'multimodal.source_stopped'
        && params?.started === true
        && params?.session_id === 'runtime-A'
      ) {
        return sourceStartedA.promise
      }

      if (method === 'multimodal.frame') {
        return Promise.resolve({ buffered: true })
      }

      return Promise.resolve({ ok: true })
    })

    mockGatewayAtom.set({ connectionState: 'open', notify, request })
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia: vi.fn(async () => stream) }
    })

    await startCameraCapture()
    mockSessionId.set('runtime-A')
    const bindingA = ensureCaptureBoundToSession('runtime-A')

    for (let step = 0; step < 10; step += 1) {
      if (request.mock.calls.some(
        ([method, params]) => method === 'multimodal.source_stopped'
          && params?.started === true
          && params?.session_id === 'runtime-A'
      )) {
        break
      }

      await Promise.resolve()
    }

    const startA = request.mock.calls.find(
      ([method, params]) => method === 'multimodal.source_stopped'
        && params?.started === true
        && params?.session_id === 'runtime-A'
    )

    mockSessionId.set('runtime-B')
    await ensureCaptureBoundToSession('runtime-B')

    const startB = request.mock.calls.find(
      ([method, params]) => method === 'multimodal.source_stopped'
        && params?.started === true
        && params?.session_id === 'runtime-B'
    )

    sourceStartedA.resolve({})
    await bindingA.catch(() => undefined)
    await vi.advanceTimersByTimeAsync(500)

    const rollbackA = request.mock.calls.find(
      ([method, params]) => method === 'multimodal.source_stopped'
        && params?.started === false
        && params?.session_id === 'runtime-A'
    )

    const periodicFramesForB = notify.mock.calls.filter(
      ([method, params]) => method === 'multimodal.frame'
        && params?.session_id === 'runtime-B'
    )

    expect(rollbackA?.[1]?.capture_generation).toBe(startA?.[1]?.capture_generation)
    expect(periodicFramesForB.length).toBeGreaterThan(0)
    expect(periodicFramesForB.every(
      ([, params]) => params?.capture_generation === startB?.[1]?.capture_generation
    )).toBe(true)
    expect($mmSource.get()).toBe('camera')
  })

  it.each([
    { disconnectBeforeStop: true, failure: 'a closed gateway' },
    { disconnectBeforeStop: false, failure: 'a rejected stop RPC' }
  ])('retains exact backend owner cleanup across $failure until reconnect ACK', async ({
    disconnectBeforeStop
  }) => {
    const { stream, track } = fakeStream()
    let rejectOwnerStop = !disconnectBeforeStop

    const request = vi.fn((
      method: string,
      params?: Record<string, unknown>,
      _timeoutMs?: number
    ) => {
      if (method === 'multimodal.frame') {
        return Promise.resolve({ buffered: true })
      }

      if (method === 'multimodal.source_stopped' && params?.started === false) {
        return rejectOwnerStop
          ? Promise.reject(new Error('stop transport unavailable'))
          : Promise.resolve({ ok: true })
      }

      return Promise.resolve({ ok: true })
    })

    const gateway = {
      connectionState: 'open',
      notify: vi.fn((_method: string, _params?: Record<string, unknown>) => 0),
      request
    }

    mockSessionId.set('runtime-owner-cleanup')
    mockGatewayAtom.set(gateway)
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia: vi.fn(async () => stream) }
    })

    await startCameraCapture()

    const ownerStart = request.mock.calls.find(
      ([method, params]) => method === 'multimodal.source_stopped'
        && params?.started === true
    )

    expect(ownerStart).toBeDefined()

    if (disconnectBeforeStop) {
      gateway.connectionState = 'connecting'
    }

    stopCaptureAndNotify()

    for (let step = 0; step < 5; step += 1) {
      await Promise.resolve()
    }

    const ownerStopsBeforeReconnect = request.mock.calls.filter(
      ([method, params]) => method === 'multimodal.source_stopped'
        && params?.started === false
    )

    expect(ownerStopsBeforeReconnect).toHaveLength(disconnectBeforeStop ? 0 : 1)
    expect(track.stop).toHaveBeenCalledTimes(1)
    expect($mmSource.get()).toBe('none')
    expect(vi.getTimerCount()).toBe(0)

    rejectOwnerStop = false
    gateway.connectionState = 'open'
    resumeFrameLoop()

    for (let step = 0; step < 10; step += 1) {
      await Promise.resolve()
    }

    const acknowledgedOwnerStops = request.mock.calls.filter(
      ([method, params]) => method === 'multimodal.source_stopped'
        && params?.started === false
    )

    expect(acknowledgedOwnerStops).toHaveLength(disconnectBeforeStop ? 1 : 2)
    const retriedOwnerStop = acknowledgedOwnerStops.at(-1)

    expect(retriedOwnerStop?.[1]).toEqual(expect.objectContaining({
      session_id: ownerStart?.[1]?.session_id,
      capture_client_id: ownerStart?.[1]?.capture_client_id,
      capture_client_started_at_ms: ownerStart?.[1]?.capture_client_started_at_ms,
      capture_generation: ownerStart?.[1]?.capture_generation,
      capture_attempt_id: ownerStart?.[1]?.capture_attempt_id,
      started: false
    }))
    expect(retriedOwnerStop?.[2]).toEqual(expect.any(Number))

    const acknowledgedStopCount = acknowledgedOwnerStops.length

    resumeFrameLoop()

    for (let step = 0; step < 5; step += 1) {
      await Promise.resolve()
    }

    expect(request.mock.calls.filter(
      ([method, params]) => method === 'multimodal.source_stopped'
        && params?.started === false
    )).toHaveLength(acknowledgedStopCount)
  })

  it('flushes a rejected retired A cleanup after reconnect without stopping healthy B', async () => {
    const { stream, track } = fakeStream()
    const notify = vi.fn((_method: string, _params?: Record<string, unknown>) => 0)
    let rejectRetiredAStop = true

    const request = vi.fn((
      method: string,
      params?: Record<string, unknown>,
      _timeoutMs?: number
    ) => {
      if (method === 'multimodal.frame') {
        return Promise.resolve({ buffered: true })
      }

      if (
        method === 'multimodal.source_stopped'
        && params?.started === false
        && params?.session_id === 'runtime-retired-A'
        && rejectRetiredAStop
      ) {
        return Promise.reject(new Error('old A rollback transport lost'))
      }

      return Promise.resolve({ ok: true })
    })

    const gateway = { connectionState: 'open', notify, request }

    mockGatewayAtom.set(gateway)
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia: vi.fn(async () => stream) }
    })

    await startCameraCapture()
    mockSessionId.set('runtime-retired-A')
    await ensureCaptureBoundToSession('runtime-retired-A')

    const startA = request.mock.calls.find(
      ([method, params]) => method === 'multimodal.source_stopped'
        && params?.started === true
        && params?.session_id === 'runtime-retired-A'
    )

    mockSessionId.set('runtime-live-B')
    await ensureCaptureBoundToSession('runtime-live-B')

    for (let step = 0; step < 10; step += 1) {
      await Promise.resolve()
    }

    const failedStopsA = request.mock.calls.filter(
      ([method, params]) => method === 'multimodal.source_stopped'
        && params?.started === false
        && params?.capture_attempt_id === startA?.[1]?.capture_attempt_id
    )

    expect(failedStopsA).toHaveLength(1)

    gateway.connectionState = 'connecting'
    pauseFrameLoop()
    rejectRetiredAStop = false
    gateway.connectionState = 'open'
    resumeFrameLoop()
    await ensureCaptureBoundToSession('runtime-live-B')
    await vi.advanceTimersByTimeAsync(500)

    const startsB = request.mock.calls.filter(
      ([method, params]) => method === 'multimodal.source_stopped'
        && params?.started === true
        && params?.session_id === 'runtime-live-B'
    )

    const currentStartB = startsB.at(-1)

    const retriedStopsA = request.mock.calls.filter(
      ([method, params]) => method === 'multimodal.source_stopped'
        && params?.started === false
        && params?.capture_attempt_id === startA?.[1]?.capture_attempt_id
    )

    const stopsForCurrentB = request.mock.calls.filter(
      ([method, params]) => method === 'multimodal.source_stopped'
        && params?.started === false
        && params?.capture_attempt_id === currentStartB?.[1]?.capture_attempt_id
    )

    const periodicFramesForCurrentB = notify.mock.calls.filter(
      ([method, params]) => method === 'multimodal.frame'
        && params?.capture_attempt_id === currentStartB?.[1]?.capture_attempt_id
    )

    expect(startsB.length).toBeGreaterThanOrEqual(2)
    expect(retriedStopsA).toHaveLength(2)
    expect(retriedStopsA[1]?.[1]).toEqual(expect.objectContaining({
      session_id: startA?.[1]?.session_id,
      capture_client_id: startA?.[1]?.capture_client_id,
      capture_client_started_at_ms: startA?.[1]?.capture_client_started_at_ms,
      capture_generation: startA?.[1]?.capture_generation,
      capture_attempt_id: startA?.[1]?.capture_attempt_id,
      started: false
    }))
    expect(stopsForCurrentB).toHaveLength(0)
    expect(periodicFramesForCurrentB.length).toBeGreaterThan(0)
    expect(track.stop).not.toHaveBeenCalled()
    expect($mmSource.get()).toBe('camera')
    expect(vi.getTimerCount()).toBe(1)
  })
})
