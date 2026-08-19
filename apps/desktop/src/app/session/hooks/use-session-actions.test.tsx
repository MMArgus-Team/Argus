import { cleanup, render, waitFor } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { useEffect } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { getSessionMessages } from '@/hermes'
import { $activeGatewayProfile, $newChatProfile } from '@/store/profile'
import {
  $currentCwd,
  $messages,
  $resumeFailedSessionId,
  $sessions,
  setMessages,
  setResumeFailedSessionId,
  setSessions
} from '@/store/session'

import type { ClientSessionState } from '../../types'

import { useSessionActions } from './use-session-actions'

const captureLifecycle = vi.hoisted(() => ({
  active: false,
  clearTransferClaims: vi.fn(),
  preserveForRebind: vi.fn(),
  stopAndNotify: vi.fn()
}))

const micLifecycle = vi.hoisted(() => ({
  intent: false,
  stop: vi.fn(async () => {
    micLifecycle.intent = false
  })
}))

vi.mock('@/store/multimodal', () => ({
  clearCaptureSessionTransferClaims: captureLifecycle.clearTransferClaims,
  preserveCaptureForNextRuntimeRebind: captureLifecycle.preserveForRebind
}))
vi.mock('@/store/multimodal-capture', () => ({
  isCapturing: () => captureLifecycle.active,
  stopCaptureAndNotify: () => {
    captureLifecycle.stopAndNotify()
    captureLifecycle.active = false
  }
}))
vi.mock('@/store/multimodal-voice', () => ({
  hasMicCaptureIntent: () => micLifecycle.intent,
  stopMic: micLifecycle.stop
}))

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  deleteSession: vi.fn(),
  getSessionMessages: vi.fn(),
  listAllProfileSessions: vi.fn(),
  setApiRequestProfile: vi.fn(),
  setSessionArchived: vi.fn()
}))

beforeEach(() => {
  captureLifecycle.active = false
  captureLifecycle.clearTransferClaims.mockClear()
  captureLifecycle.preserveForRebind.mockClear()
  captureLifecycle.stopAndNotify.mockClear()
  micLifecycle.intent = false
  micLifecycle.stop.mockClear()
})

const RUNTIME_SESSION_ID = 'rt-new-001'

function Harness({
  onReady,
  requestGateway
}: {
  onReady: (create: (preview?: string | null) => Promise<string | null>) => void
  requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
}) {
  const ref = <T,>(value: T): MutableRefObject<T> => ({ current: value })

  const actions = useSessionActions({
    activeSessionId: null,
    activeSessionIdRef: ref<string | null>(null),
    busyRef: ref(false),
    creatingSessionRef: ref(false),
    ensureSessionState: () => ({}) as ClientSessionState,
    getRouteToken: () => 'token',
    navigate: vi.fn() as never,
    requestGateway,
    runtimeIdByStoredSessionIdRef: ref(new Map<string, string>()),
    selectedStoredSessionId: null,
    selectedStoredSessionIdRef: ref<string | null>(null),
    sessionStateByRuntimeIdRef: ref(new Map<string, ClientSessionState>()),
    syncSessionStateToView: vi.fn(),
    updateSessionState: () => ({}) as ClientSessionState
  })

  useEffect(() => {
    onReady(actions.createBackendSessionForSend)
  }, [actions.createBackendSessionForSend, onReady])

  return null
}

async function createWith(profileSetup: () => void): Promise<Record<string, unknown> | undefined> {
  let createParams: Record<string, unknown> | undefined

  const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
    if (method === 'session.create') {
      createParams = params

      return { session_id: RUNTIME_SESSION_ID, stored_session_id: null } as never
    }

    return {} as never
  })

  $currentCwd.set('')
  profileSetup()

  let create: ((preview?: string | null) => Promise<string | null>) | null = null
  render(<Harness onReady={c => (create = c)} requestGateway={requestGateway} />)
  await waitFor(() => expect(create).not.toBeNull())
  await create!()

  return createParams
}

describe('createBackendSessionForSend profile routing', () => {
  afterEach(() => {
    cleanup()
    $newChatProfile.set(null)
    $activeGatewayProfile.set('default')
    vi.restoreAllMocks()
  })

  it('routes a plain new chat (no explicit profile) to the live gateway profile', async () => {
    // The "rubberband to default" bug: the top New Session button clears
    // $newChatProfile to null. In global-remote mode one backend serves every
    // profile, so an omitted `profile` lands the chat on the launch (default)
    // profile. The session must instead carry the active gateway profile.
    const params = await createWith(() => {
      $activeGatewayProfile.set('coder')
      $newChatProfile.set(null)
    })

    expect(params).toMatchObject({ profile: 'coder' })
  })

  it('honours an explicit per-profile "+" selection', async () => {
    const params = await createWith(() => {
      $activeGatewayProfile.set('coder')
      $newChatProfile.set('analyst')
    })

    expect(params).toMatchObject({ profile: 'analyst' })
  })

  it('passes the default profile for single-profile users (backend resolves it to launch)', async () => {
    const params = await createWith(() => {
      $activeGatewayProfile.set('default')
      $newChatProfile.set(null)
    })

    expect(params).toMatchObject({ profile: 'default' })
  })

  it('creates an ordinary desktop session before any on-demand multimodal promotion', async () => {
    const params = await createWith(() => {
      $activeGatewayProfile.set('default')
      $newChatProfile.set(null)
    })

    // Camera/screen activation is deliberately a subsequent, awaited
    // source_started RPC. Marking session.create as multimodal would build a
    // different system prompt and skip the ordinary desktop chat lifecycle.
    expect(params).not.toHaveProperty('source')
  })

  it('single-flights concurrent capture and first-prompt creators onto one runtime', async () => {
    let finishCreate!: (value: Record<string, unknown>) => void

    const pendingCreate = new Promise<Record<string, unknown>>(resolve => {
      finishCreate = resolve
    })

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.create') {
        return pendingCreate as never
      }

      return {} as never
    })

    let create: ((preview?: string | null) => Promise<string | null>) | null = null

    render(<Harness onReady={action => (create = action)} requestGateway={requestGateway} />)
    await waitFor(() => expect(create).not.toBeNull())

    const fromCapture = create!()
    const fromFirstPrompt = create!('hello')

    await waitFor(() =>
      expect(requestGateway.mock.calls.filter(call => call[0] === 'session.create')).toHaveLength(1)
    )

    finishCreate({ session_id: 'runtime-shared', stored_session_id: null })

    await expect(Promise.all([fromCapture, fromFirstPrompt])).resolves.toEqual([
      'runtime-shared',
      'runtime-shared'
    ])
  })
})

function DraftLifecycleHarness({
  onReady,
  requestGateway
}: {
  onReady: (actions: {
    create: (preview?: string | null) => Promise<string | null>
    fresh: () => void
  }) => void
  requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
}) {
  const ref = <T,>(value: T): MutableRefObject<T> => ({ current: value })

  const actions = useSessionActions({
    activeSessionId: null,
    activeSessionIdRef: ref<string | null>(null),
    busyRef: ref(false),
    creatingSessionRef: ref(false),
    ensureSessionState: () => ({}) as ClientSessionState,
    getRouteToken: () => 'new-route',
    navigate: vi.fn() as never,
    requestGateway,
    runtimeIdByStoredSessionIdRef: ref(new Map<string, string>()),
    selectedStoredSessionId: null,
    selectedStoredSessionIdRef: ref<string | null>(null),
    sessionStateByRuntimeIdRef: ref(new Map<string, ClientSessionState>()),
    syncSessionStateToView: vi.fn(),
    updateSessionState: () => ({}) as ClientSessionState
  })

  useEffect(() => {
    onReady({
      create: actions.createBackendSessionForSend,
      fresh: () => actions.startFreshSessionDraft()
    })
  }, [actions, onReady])

  return null
}

describe('fresh draft create ownership', () => {
  afterEach(() => cleanup())

  it.each(['camera permission', 'screen-source picker'])(
    'invalidates a pending %s on New before any source is published',
    async () => {
      const requestGateway = vi.fn(async () => ({} as never))
      let actions: { create: () => Promise<string | null>; fresh: () => void } | null = null

      render(
        <DraftLifecycleHarness
          onReady={ready => (actions = ready)}
          requestGateway={requestGateway}
        />
      )
      await waitFor(() => expect(actions).not.toBeNull())

      // Pending OS work intentionally has no published $mmSource yet. New is an
      // ownership boundary, so it must invalidate the intent unconditionally.
      captureLifecycle.active = false
      actions!.fresh()

      expect(captureLifecycle.clearTransferClaims).toHaveBeenCalledTimes(1)
      expect(captureLifecycle.stopAndNotify).toHaveBeenCalledTimes(1)
      expect(requestGateway).not.toHaveBeenCalledWith('session.create', expect.anything())
    }
  )

  it('closes a late old-draft create while allowing the replacement draft to create', async () => {
    const finishers: Array<(value: Record<string, unknown>) => void> = []

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.create') {
        return new Promise<Record<string, unknown>>(resolve => {
          finishers.push(resolve)
        }) as never
      }

      return {} as never
    })

    let actions: { create: () => Promise<string | null>; fresh: () => void } | null = null

    render(
      <DraftLifecycleHarness
        onReady={ready => (actions = ready)}
        requestGateway={requestGateway}
      />
    )
    await waitFor(() => expect(actions).not.toBeNull())

    const oldCreate = actions!.create()
    actions!.fresh()
    const newCreate = actions!.create()

    await waitFor(() =>
      expect(requestGateway.mock.calls.filter(call => call[0] === 'session.create')).toHaveLength(2)
    )

    finishers[1]({ session_id: 'runtime-new', stored_session_id: null })
    await expect(newCreate).resolves.toBe('runtime-new')
    finishers[0]({ session_id: 'runtime-old', stored_session_id: null })
    await expect(oldCreate).resolves.toBeNull()

    expect(requestGateway).toHaveBeenCalledWith('session.close', {
      session_id: 'runtime-old'
    })
  })
})

// ── Resume failure recovery (the "stuck loading session window" bug) ──────────
// When session.resume rejects AND the REST transcript fallback ALSO fails, the
// hook must (a) not throw out of the fallback (which stranded the loader), and
// (b) arm $resumeFailedSessionId so use-route-resume can retry. A resume that
// succeeds must NOT leave the flag armed.
function ResumeHarness({
  activeSessionIdRef: providedActiveSessionIdRef,
  onReady,
  requestGateway,
  runtimeIdByStoredSessionIdRef: providedRuntimeIdByStoredSessionIdRef,
  sessionStateByRuntimeIdRef: providedSessionStateByRuntimeIdRef,
  selectedStoredSessionId = null
}: {
  activeSessionIdRef?: MutableRefObject<string | null>
  onReady: (resume: (storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) => void
  requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
  runtimeIdByStoredSessionIdRef?: MutableRefObject<Map<string, string>>
  sessionStateByRuntimeIdRef?: MutableRefObject<Map<string, ClientSessionState>>
  selectedStoredSessionId?: string | null
}) {
  const ref = <T,>(value: T): MutableRefObject<T> => ({ current: value })
  const activeSessionIdRef = providedActiveSessionIdRef ?? ref<string | null>(null)
  const runtimeIdByStoredSessionIdRef = providedRuntimeIdByStoredSessionIdRef ?? ref(new Map<string, string>())
  const sessionStateByRuntimeIdRef = providedSessionStateByRuntimeIdRef ?? ref(new Map<string, ClientSessionState>())

  const actions = useSessionActions({
    activeSessionId: activeSessionIdRef.current,
    activeSessionIdRef,
    busyRef: ref(false),
    creatingSessionRef: ref(false),
    ensureSessionState: () => ({}) as ClientSessionState,
    getRouteToken: () => 'token',
    navigate: vi.fn() as never,
    requestGateway,
    runtimeIdByStoredSessionIdRef,
    selectedStoredSessionId,
    selectedStoredSessionIdRef: ref<string | null>(selectedStoredSessionId),
    sessionStateByRuntimeIdRef,
    syncSessionStateToView: vi.fn(),
    updateSessionState: (_sessionId, updater) => updater({} as ClientSessionState)
  })

  useEffect(() => {
    onReady(actions.resumeSession)
  }, [actions.resumeSession, onReady])

  return null
}

describe('resumeSession failure recovery', () => {
  afterEach(() => {
    cleanup()
    setResumeFailedSessionId(null)
    setMessages([])
    setSessions([])
    vi.restoreAllMocks()
  })

  async function runResume(
    requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
  ): Promise<void> {
    let resume: ((storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) | null = null
    render(<ResumeHarness onReady={r => (resume = r)} requestGateway={requestGateway} />)
    await waitFor(() => expect(resume).not.toBeNull())
    await resume!('stored-1', true)
  }

  it('arms $resumeFailedSessionId when resume RPC and REST fallback both fail', async () => {
    // session.resume rejects (e.g. timeout against a wedged backend)...
    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.resume') {
        throw new Error('request timed out: session.resume')
      }

      return {} as never
    })

    // ...and the REST transcript fallback also rejects (backend unreachable).
    vi.mocked(getSessionMessages).mockRejectedValue(new Error('network down'))

    await runResume(requestGateway)

    // The window is no longer silently stranded: the failure latch is armed for
    // the stored session, which use-route-resume consumes to retry.
    expect($resumeFailedSessionId.get()).toBe('stored-1')
  })

  it('does NOT arm the failure latch when the resume RPC fails but the REST fallback paints history', async () => {
    // session.resume rejects, but the REST transcript fallback succeeds and
    // hydrates a readable transcript — the window is NOT stranded.
    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.resume') {
        throw new Error('request timed out: session.resume')
      }

      return {} as never
    })

    vi.mocked(getSessionMessages).mockResolvedValue({
      messages: [
        { content: 'hello', role: 'user', timestamp: 1 },
        { content: 'hi there', role: 'assistant', timestamp: 2 }
      ],
      session_id: 'stored-1'
    } as never)

    await runResume(requestGateway)

    // Arming here would auto-retry a window that already shows history and,
    // on exhaustion, blank that transcript behind the error overlay — a
    // regression vs. plain fallback-success. The latch must stay clear.
    expect($resumeFailedSessionId.get()).toBeNull()
    // The fallback transcript is visible.
    expect($messages.get().length).toBeGreaterThan(0)
  })

  it('does NOT throw out of the fallback when REST also fails (no unhandled rejection)', async () => {
    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.resume') {
        throw new Error('request timed out: session.resume')
      }

      return {} as never
    })

    vi.mocked(getSessionMessages).mockRejectedValue(new Error('network down'))

    // resumeSession must resolve (swallow the fallback failure), not reject.
    await expect(runResume(requestGateway)).resolves.toBeUndefined()
  })

  it('leaves the failure latch clear when resume succeeds', async () => {
    // Pre-arm to prove a successful resume clears it (entry-clear path).
    setResumeFailedSessionId('stored-1')

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === 'session.resume') {
        return { session_id: 'runtime-1', resumed: params?.session_id, messages: [], info: {} } as never
      }

      return {} as never
    })

    vi.mocked(getSessionMessages).mockResolvedValue({ messages: [] } as never)

    await runResume(requestGateway)

    expect($resumeFailedSessionId.get()).toBeNull()
  })

  it('resumes via the gateway default (deferred build) — not lazy, no eager opt-out', async () => {
    // The switch-latency fix lives backend-side: a normal cold resume gets the
    // gateway's default DEFERRED build (transcript returns immediately, agent
    // pre-warms in the background). The client must NOT force the synchronous
    // path (eager_build) and is only `lazy` for subagent watch windows.
    let resumeParams: Record<string, unknown> | undefined

    const requestGateway = vi.fn(async (method: string, params?: Record<string, unknown>) => {
      if (method === 'session.resume') {
        resumeParams = params

        return { session_id: 'runtime-1', resumed: params?.session_id, messages: [], info: {} } as never
      }

      return {} as never
    })

    vi.mocked(getSessionMessages).mockResolvedValue({ messages: [] } as never)

    await runResume(requestGateway)

    expect(resumeParams).not.toHaveProperty('lazy')
    expect(resumeParams).not.toHaveProperty('eager_build')
  })

  it('immediately stops the old capture when switching stored sessions while resume is pending', async () => {
    let finishResume!: (value: Record<string, unknown>) => void

    const pendingResume = new Promise<Record<string, unknown>>(resolve => {
      finishResume = resolve
    })

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.resume') {return pendingResume as never}

      return {} as never
    })

    vi.mocked(getSessionMessages).mockResolvedValue({ messages: [] } as never)
    captureLifecycle.active = true

    let resume: ((storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) | null = null
    render(
      <ResumeHarness
        onReady={action => (resume = action)}
        requestGateway={requestGateway}
        selectedStoredSessionId="stored-A"
      />
    )
    await waitFor(() => expect(resume).not.toBeNull())

    const switching = resume!('stored-B', true)

    // The old live sid is already null in this harness. Capture still owns its
    // remembered backend sid, so the stored-conversation boundary must stop it
    // synchronously, before profile resolution/session.resume can settle.
    expect(captureLifecycle.stopAndNotify).toHaveBeenCalledTimes(1)
    expect(captureLifecycle.active).toBe(false)
    expect(captureLifecycle.clearTransferClaims).toHaveBeenCalledTimes(1)
    expect(captureLifecycle.preserveForRebind).not.toHaveBeenCalled()

    await waitFor(() =>
      expect(requestGateway).toHaveBeenCalledWith(
        'session.resume',
        expect.objectContaining({ session_id: 'stored-B' })
      )
    )
    expect(captureLifecycle.stopAndNotify).toHaveBeenCalledTimes(1)

    finishResume({ session_id: 'runtime-B', messages: [], info: {} })
    await switching
  })

  it('clears warm runtime A synchronously before an A-to-B stored-session switch can grant new capture', async () => {
    const activeSessionIdRef: MutableRefObject<string | null> = { current: 'runtime-A' }
    const runtimeIdByStoredSessionIdRef: MutableRefObject<Map<string, string>> = {
      current: new Map([['stored-B', 'runtime-B']])
    }
    const warmState: ClientSessionState = {
      storedSessionId: 'stored-B',
      messages: [],
      branch: '',
      cwd: '',
      model: '',
      provider: '',
      reasoningEffort: '',
      serviceTier: '',
      fast: false,
      yolo: false,
      personality: '',
      busy: false,
      awaitingResponse: false,
      streamId: null,
      sawAssistantPayload: false,
      pendingBranchGroup: null,
      interrupted: false,
      needsInput: false,
      turnStartedAt: null
    }
    const sessionStateByRuntimeIdRef: MutableRefObject<Map<string, ClientSessionState>> = {
      current: new Map([['runtime-B', warmState]])
    }
    const requestGateway = vi.fn(async () => ({}) as never)

    $sessions.set([{ id: 'stored-B', profile: 'default' }] as never)
    captureLifecycle.active = true

    let resume: ((storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) | null = null
    render(
      <ResumeHarness
        activeSessionIdRef={activeSessionIdRef}
        onReady={action => (resume = action)}
        requestGateway={requestGateway}
        runtimeIdByStoredSessionIdRef={runtimeIdByStoredSessionIdRef}
        selectedStoredSessionId="stored-A"
        sessionStateByRuntimeIdRef={sessionStateByRuntimeIdRef}
      />
    )
    await waitFor(() => expect(resume).not.toBeNull())

    const switching = resume!('stored-B', true)

    // This relationship must hold in the same synchronous call stack as the
    // ownership stop. While profile/route resolution is pending, a media grant
    // must see no live sid at all — never the old runtime A.
    expect(captureLifecycle.stopAndNotify).toHaveBeenCalledTimes(1)
    expect(activeSessionIdRef.current).toBeNull()

    await switching

    expect(activeSessionIdRef.current).toBe('runtime-B')
    expect(requestGateway).toHaveBeenCalledWith('session.usage', {
      session_id: 'runtime-B'
    })
  })

  it('invalidates pending camera permission when switching stored sessions', async () => {
    let finishResume!: (value: Record<string, unknown>) => void

    const pendingResume = new Promise<Record<string, unknown>>(resolve => {
      finishResume = resolve
    })

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.resume') {
        return pendingResume as never
      }

      return {} as never
    })

    vi.mocked(getSessionMessages).mockResolvedValue({ messages: [] } as never)
    // The permission dialog is open, but attachStream has not published a
    // source yet. The stored-chat boundary still has to cancel that old intent.
    captureLifecycle.active = false

    let resume: ((storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) | null = null
    render(
      <ResumeHarness
        onReady={action => (resume = action)}
        requestGateway={requestGateway}
        selectedStoredSessionId="stored-A"
      />
    )
    await waitFor(() => expect(resume).not.toBeNull())

    const switching = resume!('stored-B', true)

    expect(captureLifecycle.clearTransferClaims).toHaveBeenCalledTimes(1)
    expect(captureLifecycle.stopAndNotify).toHaveBeenCalledTimes(1)
    expect(captureLifecycle.preserveForRebind).not.toHaveBeenCalled()

    await waitFor(() =>
      expect(requestGateway).toHaveBeenCalledWith(
        'session.resume',
        expect.objectContaining({ session_id: 'stored-B' })
      )
    )

    finishResume({ session_id: 'runtime-B', messages: [], info: {} })
    await switching

    expect(captureLifecycle.stopAndNotify).toHaveBeenCalledTimes(1)
  })

  it('immediately cancels a pending old-session mic intent across an empty runtime gap', async () => {
    let finishResume!: (value: Record<string, unknown>) => void

    const pendingResume = new Promise<Record<string, unknown>>(resolve => {
      finishResume = resolve
    })

    const requestGateway = vi.fn(async (method: string) => {
      if (method === 'session.resume') {
        return pendingResume as never
      }

      return {} as never
    })

    vi.mocked(getSessionMessages).mockResolvedValue({ messages: [] } as never)
    micLifecycle.intent = true

    let resume: ((storedSessionId: string, replaceRoute?: boolean) => Promise<unknown>) | null = null
    render(
      <ResumeHarness
        onReady={action => (resume = action)}
        requestGateway={requestGateway}
        selectedStoredSessionId="stored-A"
      />
    )
    await waitFor(() => expect(resume).not.toBeNull())

    const switching = resume!('stored-B', true)

    // The live runtime/MM sid is already empty, but the old durable A still
    // owns the reconnect intent. Selecting B is an explicit ownership boundary
    // and must cancel that intent before the slow resume can publish runtime-B.
    expect(micLifecycle.stop).toHaveBeenCalledTimes(1)
    expect(micLifecycle.intent).toBe(false)
    expect(captureLifecycle.preserveForRebind).not.toHaveBeenCalled()

    finishResume({ session_id: 'runtime-B', messages: [], info: {} })
    await switching

    expect(micLifecycle.stop).toHaveBeenCalledTimes(1)
  })
})
