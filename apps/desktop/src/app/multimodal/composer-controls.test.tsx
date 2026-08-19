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
      pushToast: vi.fn(),
      startMic: vi.fn(async () => undefined),
      stopMic: vi.fn(async () => undefined),
      toggleVoiceDialog: vi.fn(() => true)
    },
    stores: {
      asrBuffer: atom<string[]>([]),
      asrPartial: atom(''),
      micError: atom(''),
      micState: atom<'idle' | 'connecting' | 'recording' | 'finalizing'>('idle'),
      ttsEnabled: atom(false),
      voiceDialogEnabled: atom(false)
    }
  }
})

vi.mock('@/store/multimodal-deep', () => ({ pushMmToast: actions.pushToast }))
vi.mock('@/store/multimodal-voice', () => ({
  $mmAsrBuffer: stores.asrBuffer,
  $mmAsrPartial: stores.asrPartial,
  $mmMicError: stores.micError,
  $mmMicState: stores.micState,
  $mmTtsEnabled: stores.ttsEnabled,
  $mmVoiceDialogEnabled: stores.voiceDialogEnabled,
  finishMicTurn: actions.finishMicTurn,
  startMic: actions.startMic,
  stopMic: actions.stopMic,
  toggleMultimodalTts: vi.fn(),
  toggleMultimodalVoiceDialog: actions.toggleVoiceDialog
}))

import { MultimodalAsrBar, MultimodalComposerControls } from './composer-controls'

describe('MultimodalAsrBar', () => {
  beforeEach(() => {
    stores.asrBuffer.set([])
    stores.asrPartial.set('')
    stores.micError.set('')
    stores.micState.set('idle')
    stores.voiceDialogEnabled.set(false)
    actions.finishMicTurn.mockClear()
    actions.pushToast.mockClear()
    actions.startMic.mockClear()
    actions.stopMic.mockClear()
    actions.toggleVoiceDialog.mockReset()
    actions.toggleVoiceDialog.mockReturnValue(true)
  })

  afterEach(cleanup)

  it('renders the stitched buffer as a dim prefix before the live partial', () => {
    stores.asrBuffer.set(['第一段', '第二段'])
    stores.asrPartial.set('还在说')
    stores.micState.set('recording')

    render(<MultimodalAsrBar />)

    expect(screen.getByText('第一段 第二段').classList.contains('opacity-60')).toBe(true)
    expect(screen.getByText('还在说')).toBeTruthy()
  })

  it('stays visible for a buffered segment after the current partial clears', () => {
    stores.asrBuffer.set(['已缓冲的语音'])

    render(<MultimodalAsrBar />)

    expect(screen.getByText('已缓冲的语音')).toBeTruthy()
  })

  it('cancels an armed connecting mic before recording starts', () => {
    stores.micState.set('connecting')

    render(<MultimodalComposerControls />)
    const mic = screen.getByTitle('正在准备麦克风，点击取消')

    expect((mic as HTMLButtonElement).disabled).toBe(false)
    fireEvent.click(mic)

    expect(actions.stopMic).toHaveBeenCalledTimes(1)
    expect(actions.finishMicTurn).not.toHaveBeenCalled()
    expect(actions.startMic).not.toHaveBeenCalled()
  })

  it('locks the button and shows completion progress while finalizing', () => {
    stores.micState.set('finalizing')

    render(
      <>
        <MultimodalAsrBar />
        <MultimodalComposerControls />
      </>
    )

    expect(screen.getByText('正在完成识别…')).toBeTruthy()
    expect((screen.getByTitle('正在完成识别并发送…') as HTMLButtonElement).disabled).toBe(true)
  })

  it('labels the ordinary microphone as one-turn dictation', () => {
    render(<MultimodalComposerControls />)

    expect(screen.getByLabelText('开始单次语音输入')).toBeTruthy()
  })

  it('keeps dialog mode off and explains when a manual turn owns the mic', () => {
    stores.micState.set('recording')
    actions.toggleVoiceDialog.mockReturnValue(false)
    render(<MultimodalComposerControls />)

    fireEvent.click(screen.getByTitle('对话模式：关 — 点击进入语音对话交互（自动开麦）'))

    expect(actions.pushToast).toHaveBeenCalledWith({
      level: 'error',
      text: '请先结束当前单次录音，再开启语音对话'
    })
  })
})
