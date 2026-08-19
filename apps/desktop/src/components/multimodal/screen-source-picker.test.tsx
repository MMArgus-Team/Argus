import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  openSourcePicker,
  type PickedSource,
  ScreenSourcePickerHost
} from './screen-source-picker'

const desktopWindow = window as unknown as {
  hermesDesktop?: Window['hermesDesktop']
}

const initialHermesDesktop = desktopWindow.hermesDesktop

const DISPLAY_SOURCE = {
  appIconDataUrl: '',
  displayId: '1',
  id: 'screen:1:0',
  kind: 'screen' as const,
  name: 'Display 1',
  thumbnailDataUrl: ''
}

function installDesktopBridge(systemAudioSupported?: boolean): void {
  const capability =
    systemAudioSupported === undefined
      ? {}
      : { screenShareSystemAudio: systemAudioSupported }

  desktopWindow.hermesDesktop = {
    ...capability,
    multimodalSourcePicker: {
      getSelectedSource: vi.fn(async () => null),
      listSources: vi.fn(async () => ({ ok: true, sources: [DISPLAY_SOURCE] })),
      setSelectedSource: vi.fn(async () => ({ ok: true }))
    }
  } as unknown as Window['hermesDesktop']
}

async function openRenderedPicker(): Promise<{ result: Promise<PickedSource | null> }> {
  let result!: Promise<PickedSource | null>
  act(() => {
    result = openSourcePicker()
  })
  await screen.findByTitle(DISPLAY_SOURCE.name)

  return { result }
}

function selectDisplay(): void {
  fireEvent.click(screen.getByTitle(DISPLAY_SOURCE.name))
}

function toggleAudio(): void {
  fireEvent.click(screen.getByRole('switch'))
}

describe('ScreenSourcePickerHost system audio', () => {
  afterEach(() => {
    const cancel = screen.queryByRole('button', { name: '取消' })

    if (cancel) {
      fireEvent.click(cancel)
    }

    cleanup()
    vi.restoreAllMocks()

    if (initialHermesDesktop) {
      desktopWindow.hermesDesktop = initialHermesDesktop
    } else {
      delete desktopWindow.hermesDesktop
    }
  })

  it.each([
    { capability: false, label: 'explicitly unsupported' },
    { capability: undefined, label: 'missing from an older preload' }
  ])('disables audio and fails closed when $label', async ({ capability }) => {
    installDesktopBridge(capability)
    render(<ScreenSourcePickerHost />)
    const { result } = await openRenderedPicker()

    const audioSwitch = screen.getByRole('switch')
    expect(audioSwitch.hasAttribute('disabled')).toBe(true)
    expect(audioSwitch.getAttribute('aria-checked')).toBe('false')
    expect(screen.getByText('（需 Windows 或 macOS 13+）')).toBeTruthy()

    selectDisplay()
    fireEvent.click(screen.getByRole('button', { name: '开始共享' }))

    await expect(result).resolves.toEqual({
      id: DISPLAY_SOURCE.id,
      name: DISPLAY_SOURCE.name,
      shareAudio: false
    })
  })

  it('returns enabled system audio through the confirm button', async () => {
    installDesktopBridge(true)
    render(<ScreenSourcePickerHost />)
    const { result } = await openRenderedPicker()

    const audioSwitch = screen.getByRole('switch')
    expect(audioSwitch.hasAttribute('disabled')).toBe(false)
    expect(screen.queryByText('（需 Windows 或 macOS 13+）')).toBeNull()

    toggleAudio()
    selectDisplay()
    fireEvent.click(screen.getByRole('button', { name: '开始共享' }))

    await expect(result).resolves.toEqual({
      id: DISPLAY_SOURCE.id,
      name: DISPLAY_SOURCE.name,
      shareAudio: true
    })
  })

  it('returns enabled system audio through source double-click', async () => {
    installDesktopBridge(true)
    render(<ScreenSourcePickerHost />)
    const { result } = await openRenderedPicker()

    toggleAudio()
    fireEvent.doubleClick(screen.getByTitle(DISPLAY_SOURCE.name))

    await expect(result).resolves.toEqual({
      id: DISPLAY_SOURCE.id,
      name: DISPLAY_SOURCE.name,
      shareAudio: true
    })
  })

  it('resets system audio to off after closing and reopening', async () => {
    installDesktopBridge(true)
    render(<ScreenSourcePickerHost />)
    const { result: firstResult } = await openRenderedPicker()

    toggleAudio()
    expect(screen.getByRole('switch').getAttribute('aria-checked')).toBe('true')
    selectDisplay()
    fireEvent.click(screen.getByRole('button', { name: '开始共享' }))
    await expect(firstResult).resolves.toMatchObject({ shareAudio: true })
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())

    const { result: secondResult } = await openRenderedPicker()
    expect(screen.getByRole('switch').getAttribute('aria-checked')).toBe('false')

    fireEvent.click(screen.getByRole('button', { name: '取消' }))
    await expect(secondResult).resolves.toBeNull()
  })
})
