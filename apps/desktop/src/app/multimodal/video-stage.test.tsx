import { act, cleanup, render, screen } from '@testing-library/react'
import type { WritableAtom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const { capture, sessions } = vi.hoisted(() => {
  const { atom } = require('nanostores') as {
    atom: <Value>(initial: Value) => WritableAtom<Value>
  }

  return {
    capture: {
      debug: atom({ code: 'idle', detail: '采集未启动' }),
      source: atom<'camera' | 'none' | 'screen'>('none'),
      stats: atom({ dropped: 0, sent: 0 }),
      stream: atom<MediaStream | null>(null)
    },
    sessions: {
      liveId: atom(''),
      rows: atom<Array<{ id: string }>>([]),
      selectedId: atom<string | null>(null)
    }
  }
})

vi.mock('@/store/multimodal', () => ({ $mmSessionId: sessions.liveId }))
vi.mock('@/store/multimodal-capture', () => ({
  $mmCapStats: capture.stats,
  $mmCaptureDebug: capture.debug,
  $mmSource: capture.source,
  $mmStream: capture.stream,
  startCameraCapture: vi.fn(async () => undefined),
  startScreenCapture: vi.fn(async () => undefined),
  stopCaptureAndNotify: vi.fn()
}))
vi.mock('@/store/session', () => ({
  $selectedStoredSessionId: sessions.selectedId,
  $sessions: sessions.rows
}))
vi.mock('./memory-debug-panel', () => ({
  MemoryDebugPanel: () => null,
  resolveMemoryDebugSessionIds: () => []
}))

import { VideoStage } from './video-stage'

describe('VideoStage recording truthfulness', () => {
  beforeEach(() => {
    capture.debug.set({ code: 'idle', detail: '采集未启动' })
    capture.source.set('none')
    capture.stats.set({ dropped: 0, sent: 0 })
    capture.stream.set(null)
    sessions.liveId.set('')
    sessions.rows.set([])
    sessions.selectedId.set(null)
  })

  afterEach(cleanup)

  it('shows an armed preview before first-frame ACK and turns red REC only after ACK', () => {
    capture.source.set('screen')
    capture.debug.set({
      code: 'waiting_for_session',
      detail: '正在创建会话并初始化多模态后端'
    })

    const { container } = render(<VideoStage />)

    expect(screen.queryByText(/\bREC\b/)).toBeNull()
    expect(container.querySelector('.bg-red-500')).toBeNull()

    act(() => {
      capture.stats.set({ dropped: 0, sent: 1 })
      capture.debug.set({
        code: 'sending',
        detail: 'screen 1920x1080 jpeg_b64_chars=1234 sid=yes'
      })
    })

    expect(screen.getByText('REC · 屏幕 · 1 帧')).toBeTruthy()
    expect(container.querySelector('.bg-red-500')).toBeTruthy()
  })

  it('drops red REC while delivery is paused and restores it only after a resumed ACK', () => {
    capture.source.set('camera')
    capture.stats.set({ dropped: 0, sent: 3 })
    capture.debug.set({
      code: 'sending',
      detail: 'camera 1280x720 jpeg_b64_chars=1234 sid=yes'
    })

    const { container } = render(<VideoStage />)

    expect(screen.getByText('REC · 摄像头 · 3 帧')).toBeTruthy()
    expect(container.querySelector('.bg-red-500')).toBeTruthy()

    act(() => {
      capture.debug.set({
        code: 'gateway_not_open',
        detail: 'Gateway 连接已中断，画面记录已暂停'
      })
    })

    expect(screen.queryByText(/\bREC\b/)).toBeNull()
    expect(container.querySelector('.bg-red-500')).toBeNull()
    expect(container.querySelector('.bg-amber-400')).toBeTruthy()
    expect(screen.getByText(/连接中断，记录已暂停/)).toBeTruthy()

    act(() => {
      capture.debug.set({
        code: 'sending',
        detail: 'camera 1280x720 jpeg_b64_chars=1234 sid=yes'
      })
    })

    expect(screen.getByText('REC · 摄像头 · 3 帧')).toBeTruthy()
    expect(container.querySelector('.bg-red-500')).toBeTruthy()
  })

  it('disables new media grants while a selected stored chat is still resuming', () => {
    sessions.selectedId.set('stored-B')
    sessions.liveId.set('')

    render(<VideoStage />)

    const camera = screen.getByRole('button', { name: /摄像头/ }) as HTMLButtonElement
    const screenShare = screen.getByRole('button', { name: /屏幕共享/ }) as HTMLButtonElement

    expect(camera.disabled).toBe(true)
    expect(screenShare.disabled).toBe(true)
    expect(screen.getByText(/恢复当前会话/)).toBeTruthy()

    act(() => sessions.liveId.set('runtime-B'))

    expect(camera.disabled).toBe(false)
    expect(screenShare.disabled).toBe(false)
    expect(screen.queryByText(/恢复当前会话/)).toBeNull()
  })
})
