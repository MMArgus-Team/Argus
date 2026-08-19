import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import type { WritableAtom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const { actions, stores } = vi.hoisted(() => {
  const { atom } = require('nanostores') as {
    atom: <Value>(initial: Value) => WritableAtom<Value>
  }

  return {
    actions: {
      finishMicTurn: vi.fn(async () => undefined),
      startMic: vi.fn(async () => undefined),
      stopMic: vi.fn(async () => undefined)
    },
    stores: {
      asrBuffer: atom<string[]>([]),
      asrPartial: atom(''),
      generating: atom(false),
      micError: atom(''),
      micState: atom<'idle' | 'connecting' | 'recording' | 'finalizing'>('idle'),
      ttsEnabled: atom(false)
    }
  }
})

vi.mock('@/store/multimodal', () => ({
  $mmGenerating: stores.generating,
  interruptMultimodal: vi.fn(),
  sendMultimodalPrompt: vi.fn()
}))
vi.mock('@/store/multimodal-voice', () => ({
  $mmAsrBuffer: stores.asrBuffer,
  $mmAsrPartial: stores.asrPartial,
  $mmMicError: stores.micError,
  $mmMicState: stores.micState,
  $mmTtsEnabled: stores.ttsEnabled,
  finishMicTurn: actions.finishMicTurn,
  startMic: actions.startMic,
  stopMic: actions.stopMic,
  toggleMultimodalTts: vi.fn()
}))

import { Composer } from './composer'

describe('standalone multimodal Composer mic turn', () => {
  beforeEach(() => {
    stores.asrBuffer.set([])
    stores.asrPartial.set('')
    stores.generating.set(false)
    stores.micError.set('')
    stores.micState.set('idle')
    stores.ttsEnabled.set(false)
    actions.finishMicTurn.mockClear()
    actions.startMic.mockClear()
    actions.stopMic.mockClear()
  })

  afterEach(cleanup)

  it('finishes only after recording has started', () => {
    stores.micState.set('recording')
    render(<Composer />)

    fireEvent.click(screen.getByLabelText('结束录音并发送'))

    expect(actions.finishMicTurn).toHaveBeenCalledTimes(1)
    expect(actions.stopMic).not.toHaveBeenCalled()
  })

  it('cancels while connecting and locks during finalization', () => {
    stores.micState.set('connecting')
    const view = render(<Composer />)

    fireEvent.click(screen.getByLabelText('正在准备麦克风，点击取消'))
    expect(actions.stopMic).toHaveBeenCalledTimes(1)
    expect(actions.finishMicTurn).not.toHaveBeenCalled()

    stores.micState.set('finalizing')
    view.rerender(<Composer />)

    expect((screen.getByLabelText('正在完成识别并发送') as HTMLButtonElement).disabled).toBe(true)
    expect(screen.getByText('正在完成识别…')).toBeTruthy()
  })
})
