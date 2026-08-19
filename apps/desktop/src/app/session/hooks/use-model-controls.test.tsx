import { QueryClient } from '@tanstack/react-query'
import { cleanup, render, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { getGlobalModelInfo } from '@/hermes'
import { $activeSessionId, $currentModel, $currentProvider, setCurrentModel, setCurrentProvider } from '@/store/session'

import { useModelControls } from './use-model-controls'

const notify = vi.fn()
const notifyError = vi.fn()

vi.mock('@/hermes', () => ({
  getGlobalModelInfo: vi.fn()
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      desktop: {
        modelSwitchConfirmAction: 'Switch anyway',
        modelSwitchConfirmMessage: 'This model has unusually high known pricing.',
        modelSwitchConfirmTitle: 'Confirm model switch',
        modelSwitchFailed: 'Model switch failed',
        modelSwitchWarningTitle: 'Model switched with a warning'
      }
    }
  })
}))

vi.mock('@/store/notifications', () => ({
  notify: (...args: Parameters<typeof notify>) => notify(...args),
  notifyError: (...args: Parameters<typeof notifyError>) => notifyError(...args)
}))

type Controls = ReturnType<typeof useModelControls>

function Harness({
  activeSessionId,
  onReady,
  requestGateway
}: {
  activeSessionId: string | null
  onReady: (controls: Controls) => void
  requestGateway: <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>
}) {
  const controls = useModelControls({
    activeSessionId,
    queryClient: new QueryClient(),
    requestGateway
  })

  onReady(controls)

  return null
}

describe('useModelControls', () => {
  beforeEach(() => {
    $activeSessionId.set(null)
    setCurrentModel('')
    setCurrentProvider('')
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    $activeSessionId.set(null)
    setCurrentModel('')
    setCurrentProvider('')
  })

  it('applies the global model when there is no active runtime session', async () => {
    vi.mocked(getGlobalModelInfo).mockResolvedValue({
      model: 'openai/gpt-5.5',
      provider: 'openai-codex'
    })

    const { result } = renderHook(() =>
      useModelControls({
        activeSessionId: null,
        queryClient: new QueryClient(),
        requestGateway: vi.fn()
      })
    )

    await result.current.refreshCurrentModel()

    expect($currentModel.get()).toBe('openai/gpt-5.5')
    expect($currentProvider.get()).toBe('openai-codex')
  })

  it('does not clobber the active session footer state with global model info', async () => {
    setCurrentModel('deepseek/deepseek-v4-pro')
    setCurrentProvider('deepseek')
    $activeSessionId.set('runtime-1')
    vi.mocked(getGlobalModelInfo).mockResolvedValue({
      model: 'openai/gpt-5.5',
      provider: 'openai-codex'
    })

    const { result } = renderHook(() =>
      useModelControls({
        activeSessionId: 'runtime-1',
        queryClient: new QueryClient(),
        requestGateway: vi.fn()
      })
    )

    await result.current.refreshCurrentModel()

    expect($currentModel.get()).toBe('deepseek/deepseek-v4-pro')
    expect($currentProvider.get()).toBe('deepseek')
  })

  it('routes active-session picker changes through config.set with an explicit provider', async () => {
    const requestGateway = vi.fn(async () => ({ key: 'model', value: 'claude-sonnet-4.6' }) as never)
    let controls!: Controls

    render(
      <Harness activeSessionId="session-1" onReady={value => (controls = value)} requestGateway={requestGateway} />
    )

    await expect(
      controls.selectModel({
        model: 'claude-sonnet-4.6',
        provider: 'anthropic'
      })
    ).resolves.toBe(true)

    // ★ `--session` must be present: without it the backend's
    //   resolve_persist_behavior() defaults to persisting GLOBALLY, so a
    //   composer pick would rewrite the profile default in config.yaml.
    expect(requestGateway).toHaveBeenCalledWith('config.set', {
      confirm_expensive_model: false,
      session_id: 'session-1',
      key: 'model',
      value: 'claude-sonnet-4.6 --provider anthropic --session'
    })
    expect(requestGateway).not.toHaveBeenCalledWith('slash.exec', expect.anything())
  })

  it('rolls back and reports when the backend demands expensive-model confirmation', async () => {
    // The backend returns this BEFORE calling agent.switch_model(), so the
    // switch never happened — reporting success would leave the composer
    // showing a model the agent is not running.
    const requestGateway = vi.fn(
      async () => ({ confirm_message: 'costs a lot', confirm_required: true }) as never
    )
    let controls!: Controls

    setCurrentModel('claude-sonnet-4.6')
    setCurrentProvider('anthropic')

    render(
      <Harness activeSessionId="session-1" onReady={value => (controls = value)} requestGateway={requestGateway} />
    )

    await expect(
      controls.selectModel({ model: 'expensive-opus', provider: 'anthropic' })
    ).resolves.toBe(false)

    // Optimistic update undone.
    expect($currentModel.get()).toBe('claude-sonnet-4.6')
    expect(notify).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'warning', message: 'costs a lot' })
    )
  })

  it('retries with confirm_expensive_model when the user accepts the price', async () => {
    const requestGateway = vi
      .fn()
      .mockResolvedValueOnce({ confirm_message: 'costs a lot', confirm_required: true })
      .mockResolvedValueOnce({ key: 'model', value: 'expensive-opus' })
    let controls!: Controls

    render(
      <Harness activeSessionId="session-1" onReady={value => (controls = value)} requestGateway={requestGateway} />
    )

    await controls.selectModel({ model: 'expensive-opus', provider: 'anthropic' })

    // Fire the notification's action, the way clicking "Switch anyway" would.
    const action = notify.mock.calls.at(-1)?.[0]?.action

    expect(action?.label).toBe('Switch anyway')
    action.onClick()
    await vi.waitFor(() => expect(requestGateway).toHaveBeenCalledTimes(2))

    expect(requestGateway).toHaveBeenLastCalledWith('config.set', {
      confirm_expensive_model: true,
      session_id: 'session-1',
      key: 'model',
      value: 'expensive-opus --provider anthropic --session'
    })
    expect($currentModel.get()).toBe('expensive-opus')
  })

  it('adopts the backend auto-corrected model name over the optimistic pick', async () => {
    // `glm-5v-turbo` isn't served by the endpoint; the backend corrects it to
    // `glm-5-turbo` and switches to THAT. Keeping the optimistic value would
    // leave the composer advertising a model the agent never selected.
    const requestGateway = vi.fn(
      async () =>
        ({
          key: 'model',
          value: 'glm-5-turbo',
          warning: 'Auto-corrected `glm-5v-turbo` → `glm-5-turbo`'
        }) as never
    )
    let controls!: Controls

    render(
      <Harness activeSessionId="session-1" onReady={value => (controls = value)} requestGateway={requestGateway} />
    )

    await expect(
      controls.selectModel({ model: 'glm-5v-turbo', provider: 'custom:open.bigmodel.cn' })
    ).resolves.toBe(true)

    expect($currentModel.get()).toBe('glm-5-turbo')
    expect(notify).toHaveBeenCalledWith(
      expect.objectContaining({ message: 'Auto-corrected `glm-5v-turbo` → `glm-5-turbo`' })
    )
  })

  it('surfaces a non-blocking warning while still reporting success', async () => {
    // `recognized: false` ("couldn't verify this model") arrives here — it is
    // the only early signal that api_mode inference guessed the protocol wrong.
    const requestGateway = vi.fn(
      async () => ({ key: 'model', value: 'glm-5v-turbo', warning: 'could not verify model name' }) as never
    )
    let controls!: Controls

    render(
      <Harness activeSessionId="session-1" onReady={value => (controls = value)} requestGateway={requestGateway} />
    )

    await expect(
      controls.selectModel({ model: 'glm-5v-turbo', provider: 'custom:open.bigmodel.cn' })
    ).resolves.toBe(true)

    expect(notify).toHaveBeenCalledWith(
      expect.objectContaining({ kind: 'warning', message: 'could not verify model name' })
    )
    expect($currentModel.get()).toBe('glm-5v-turbo')
  })

  it('stores a no-session pick as UI state with no gateway or global write', async () => {
    const requestGateway = vi.fn()
    let controls!: Controls

    render(<Harness activeSessionId={null} onReady={value => (controls = value)} requestGateway={requestGateway} />)

    await expect(
      controls.selectModel({
        model: 'claude-sonnet-4.6',
        provider: 'anthropic'
      })
    ).resolves.toBe(true)

    // The pick is plain UI state; session.create ships it later. Nothing touches
    // the gateway or the profile default here.
    expect($currentModel.get()).toBe('claude-sonnet-4.6')
    expect($currentProvider.get()).toBe('anthropic')
    expect(requestGateway).not.toHaveBeenCalled()
  })

  it('seeds an empty composer model from global but never clobbers a pick', async () => {
    vi.mocked(getGlobalModelInfo).mockResolvedValue({ model: 'openai/gpt-5.5', provider: 'openai-codex' })

    const { result } = renderHook(() =>
      useModelControls({
        activeSessionId: null,
        queryClient: new QueryClient(),
        requestGateway: vi.fn()
      })
    )

    // Empty → seeds the default.
    await result.current.refreshCurrentModel()
    expect($currentModel.get()).toBe('openai/gpt-5.5')

    // A user pick must survive the lifecycle refreshes that fire on boot / fresh
    // draft / session events.
    setCurrentModel('anthropic/claude-sonnet-4.6')
    setCurrentProvider('anthropic')
    await result.current.refreshCurrentModel()
    expect($currentModel.get()).toBe('anthropic/claude-sonnet-4.6')

    // A profile swap forces a reseed to the new profile's default.
    await result.current.refreshCurrentModel(true)
    expect($currentModel.get()).toBe('openai/gpt-5.5')
  })
})
