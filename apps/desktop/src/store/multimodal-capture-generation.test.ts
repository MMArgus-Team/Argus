import type { WritableAtom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const { mockGatewayAtom, mockSessionId } = vi.hoisted(() => {
  const { atom } = require('nanostores') as {
    atom: <Value>(initial: Value) => WritableAtom<Value>
  }

  return {
    mockGatewayAtom: atom<unknown>(null),
    mockSessionId: atom<string>('runtime-generation')
  }
})

vi.mock('./gateway', () => ({ $gateway: mockGatewayAtom }))
vi.mock('./multimodal', () => ({ $mmSessionId: mockSessionId }))
vi.mock('./multimodal-voice', () => ({
  startEnvAudio: vi.fn(),
  stopEnvAudio: vi.fn()
}))

import {
  startCameraCapture,
  stopCapture,
  stopCaptureAndNotify
} from './multimodal-capture'

function fakeStream(): MediaStream {
  const track = {
    addEventListener: vi.fn(),
    stop: vi.fn()
  }

  return {
    getAudioTracks: vi.fn(() => []),
    getTracks: vi.fn(() => [track]),
    getVideoTracks: vi.fn(() => [track])
  } as unknown as MediaStream
}

describe('capture generation lifecycle', () => {
  const originalCreateElement = document.createElement.bind(document)
  let createElementSpy: ReturnType<typeof vi.spyOn>
  let originalFileReader: typeof FileReader

  beforeEach(() => {
    vi.useFakeTimers()
    mockSessionId.set('runtime-generation')
    vi.spyOn(console, 'info').mockImplementation(() => undefined)
    vi.spyOn(console, 'warn').mockImplementation(() => undefined)

    originalFileReader = globalThis.FileReader

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
    createElementSpy.mockRestore()
    vi.stubGlobal('FileReader', originalFileReader)
    vi.restoreAllMocks()
    vi.useRealTimers()
  })

  it('stops the active generation and gives the reopened stream a newer generation', async () => {
    const calls: Array<{ method: string; params?: Record<string, unknown> }> = []

    const request = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      return method === 'multimodal.frame' ? { buffered: true } : { ok: true }
    })

    mockGatewayAtom.set({
      connectionState: 'open',
      notify: vi.fn(() => 0),
      request
    })

    const getUserMedia = vi.fn(async () => fakeStream())
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia }
    })

    await startCameraCapture()
    stopCaptureAndNotify()
    await startCameraCapture()

    const starts = calls.filter(
      call => call.method === 'multimodal.source_stopped' && call.params?.started === true
    )

    const stops = calls.filter(
      call => call.method === 'multimodal.source_stopped' && call.params?.started === false
    )

    const frames = calls.filter(call => call.method === 'multimodal.frame')
    const firstStartGeneration = starts[0]?.params?.capture_generation
    const secondStartGeneration = starts[1]?.params?.capture_generation
    const stopGeneration = stops[0]?.params?.capture_generation

    expect(starts).toHaveLength(2)
    expect(stops).toHaveLength(1)
    expect(frames).toHaveLength(2)
    expect(stopGeneration).toBe(firstStartGeneration)
    expect(Number(secondStartGeneration)).toBeGreaterThan(Number(stopGeneration))
    expect(frames[0]?.params?.capture_generation).toBe(firstStartGeneration)
    expect(frames[1]?.params?.capture_generation).toBe(secondStartGeneration)
    expect(frames.every(call => call.params?.capture_client_id === starts[0]?.params?.capture_client_id)).toBe(true)
    expect(getUserMedia).toHaveBeenCalledTimes(2)
  })

  it('stops the remembered owner when the live session id is temporarily empty', async () => {
    const calls: Array<{ method: string; params?: Record<string, unknown> }> = []

    const request = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      calls.push({ method, params })

      return method === 'multimodal.frame' ? { buffered: true } : { ok: true }
    })

    mockGatewayAtom.set({
      connectionState: 'open',
      notify: vi.fn(() => 0),
      request
    })
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia: vi.fn(async () => fakeStream()) }
    })

    await startCameraCapture()

    const start = calls.find(
      call => call.method === 'multimodal.source_stopped' && call.params?.started === true
    )

    expect(start?.params?.session_id).toBe('runtime-generation')

    mockSessionId.set('')
    stopCaptureAndNotify()

    const stop = calls.find(
      call => call.method === 'multimodal.source_stopped' && call.params?.started === false
    )

    expect(stop?.params?.session_id).toBe('runtime-generation')
    expect(stop?.params?.capture_generation).toBe(start?.params?.capture_generation)
    expect(stop?.params?.capture_client_id).toBe(start?.params?.capture_client_id)
  })
})
