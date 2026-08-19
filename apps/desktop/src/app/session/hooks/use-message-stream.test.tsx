import type { QueryClient } from '@tanstack/react-query'
import { act, cleanup, renderHook } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { ClientSessionState } from '@/app/types'
import { type ChatMessage, chatMessageText, textPart } from '@/lib/chat-messages'
import { $clarifyRequests, clearClarifyRequest, setClarifyRequest } from '@/store/clarify'
import { $busy, setActiveSessionId } from '@/store/session'
import type { RpcEvent } from '@/types/hermes'

import { useMessageStream } from './use-message-stream'

const SESSION_ID = 'runtime-monitor-stream'

function stateWithMessages(messages: ChatMessage[]): ClientSessionState {
  return {
    storedSessionId: 'stored-monitor-stream',
    messages,
    branch: '',
    cwd: '/workspace',
    model: 'model',
    provider: 'provider',
    reasoningEffort: '',
    serviceTier: '',
    fast: false,
    yolo: false,
    personality: '',
    busy: true,
    awaitingResponse: true,
    streamId: null,
    sawAssistantPayload: false,
    pendingBranchGroup: null,
    interrupted: false,
    needsInput: false,
    turnStartedAt: 123
  }
}

function stateWithMainAnswer(): ClientSessionState {
  return {
    storedSessionId: 'stored-monitor-stream',
    messages: [
      {
        id: 'user-main',
        role: 'user',
        parts: [textPart('main question')]
      },
      {
        id: 'assistant-main',
        role: 'assistant',
        parts: [textPart('main answer still streaming')],
        pending: true
      }
    ],

    branch: '',
    cwd: '/workspace',
    model: 'model',
    provider: 'provider',
    reasoningEffort: '',
    serviceTier: '',
    fast: false,
    yolo: false,
    personality: '',
    busy: true,
    awaitingResponse: true,
    streamId: 'assistant-main',
    sawAssistantPayload: true,
    pendingBranchGroup: null,
    interrupted: false,
    needsInput: true,
    turnStartedAt: 123
  }
}

function createMessageStreamHarness(initialState: ClientSessionState) {
  let state = initialState
  const states = new Map([[SESSION_ID, state]])
  const sessionStateByRuntimeIdRef: MutableRefObject<Map<string, ClientSessionState>> = { current: states }

  const updateSessionState = vi.fn(
    (
      sessionId: string,
      updater: (current: ClientSessionState) => ClientSessionState
    ) => {
      state = updater(states.get(sessionId) ?? state)
      states.set(sessionId, state)

      return state
    }
  )

  const activeSessionIdRef: MutableRefObject<string | null> = { current: SESSION_ID }
  setActiveSessionId(SESSION_ID)

  const hydrateFromStoredSession = vi.fn(async () => undefined)

  const hook = renderHook(() =>
    useMessageStream({
      activeSessionIdRef,
      hydrateFromStoredSession,
      queryClient: { invalidateQueries: vi.fn() } as unknown as QueryClient,
      refreshHermesConfig: vi.fn(async () => undefined),
      refreshSessions: vi.fn(async () => undefined),
      sessionStateByRuntimeIdRef,
      updateSessionState
    })
  )

  return {
    ...hook,
    getState: () => state,
    hydrateFromStoredSession,
    update: (updater: (current: ClientSessionState) => ClientSessionState) =>
      updateSessionState(SESSION_ID, updater),
    updateSessionState
  }
}

function queryDispatchEvents(requestId: string, toolId: string, query: string): RpcEvent[] {
  return [
    {
      type: 'message.start',
      session_id: SESSION_ID,
      payload: { request_id: requestId }
    },
    {
      type: 'tool.start',
      session_id: SESSION_ID,
      payload: { tool_id: toolId, name: 'query_multimodal', context: query }
    },
    {
      type: 'tool.complete',
      session_id: SESSION_ID,
      payload: {
        tool_id: toolId,
        name: 'query_multimodal',
        request_id: requestId,
        args: { query },
        result: {
          control: 'handoff',
          handoff_mode: 'deferred_reply',
          parent_user_message_id: requestId,
          query,
          reply_owner: 'query_worker'
        }
      }
    },
    {
      type: 'session.info',
      session_id: SESSION_ID,
      payload: { running: false }
    }
  ]
}

function queryWorkerEvents(requestId: string, partial: string, final: string): RpcEvent[] {
  return [
    {
      type: 'message.delta',
      session_id: SESSION_ID,
      payload: { source: 'query_worker', request_id: requestId, text: partial }
    },
    {
      type: 'message.complete',
      session_id: SESSION_ID,
      payload: { source: 'query_worker', request_id: requestId, text: final }
    }
  ]
}

describe('useMessageStream monitor sidechannel isolation', () => {
  afterEach(() => {
    cleanup()
    clearClarifyRequest(undefined, SESSION_ID)
    setActiveSessionId(null)

    $busy.set(false)
  })

  it('ignores monitor start/delta/complete without changing the main stream, busy state, or prompt', () => {
    let state = stateWithMainAnswer()

    const before = JSON.stringify(state)

    const updateSessionState = vi.fn(
      (
        _sessionId: string,
        updater: (current: ClientSessionState) => ClientSessionState
      ) => {
        state = updater(state)

        return state
      }
    )

    const activeSessionIdRef: MutableRefObject<string | null> = { current: SESSION_ID }
    setActiveSessionId(SESSION_ID)
    $busy.set(true)
    setClarifyRequest({
      choices: ['continue'],
      question: 'main turn is waiting',
      requestId: 'clarify-main',
      sessionId: SESSION_ID
    })

    const { result } = renderHook(() =>
      useMessageStream({
        activeSessionIdRef,
        hydrateFromStoredSession: vi.fn(async () => undefined),
        queryClient: { invalidateQueries: vi.fn() } as unknown as QueryClient,
        refreshHermesConfig: vi.fn(async () => undefined),
        refreshSessions: vi.fn(async () => undefined),
        sessionStateByRuntimeIdRef: { current: new Map([[SESSION_ID, state]]) },
        updateSessionState
      })
    )

    const monitorEvents: RpcEvent[] = [
      {
        type: 'message.start',
        session_id: SESSION_ID,
        payload: { source: 'monitor', monitor_id: 'monitor-1' }
      },
      {
        type: 'message.delta',
        session_id: SESSION_ID,
        payload: { source: 'monitor', monitor_id: 'monitor-1', text: 'alert delta' }
      },
      {
        type: 'message.complete',
        session_id: SESSION_ID,
        payload: { source: 'monitor', monitor_id: 'monitor-1', text: 'alert complete' }
      }
    ]

    act(() => {
      for (const event of monitorEvents) {
        result.current.handleGatewayEvent(event)
      }
    })

    expect(updateSessionState).not.toHaveBeenCalled()
    expect(JSON.stringify(state)).toBe(before)
    expect(state.busy).toBe(true)
    expect(state.messages[1]).toEqual(
      expect.objectContaining({ id: 'assistant-main', pending: true })
    )
    expect($busy.get()).toBe(true)
    expect($clarifyRequests.get()[SESSION_ID]?.requestId).toBe('clarify-main')
  })

  it('removes a finalized ephemeral Monitor-control turn and ignores its echo', () => {
    const harness = createMessageStreamHarness(
      stateWithMessages([
        { id: 'keep-user', role: 'user', parts: [textPart('ordinary question')] },
        {
          id: 'control-user',
          requestId: 'control-turn',
          role: 'user',
          parts: [textPart('set a one-shot monitor')]
        }
      ])
    )

    act(() => {
      harness.result.current.handleGatewayEvent({
        type: 'message.user_echo',
        session_id: SESSION_ID,
        payload: {
          ephemeral_control: true,
          request_id: 'control-turn',
          text: 'must stay hidden'
        }
      })
      harness.result.current.handleGatewayEvent({
        type: 'message.start',
        session_id: SESSION_ID,
        payload: { request_id: 'control-turn' }
      })
      harness.result.current.handleGatewayEvent({
        type: 'tool.start',
        session_id: SESSION_ID,
        payload: {
          tool_id: 'tool-control',
          name: 'set_monitor',
          context: 'watch once'
        }
      })
      harness.result.current.handleGatewayEvent({
        type: 'tool.complete',
        session_id: SESSION_ID,
        payload: {
          tool_id: 'tool-control',
          name: 'set_monitor',
          ephemeral_control: true,
          result: { history_policy: 'ephemeral_control' }
        }
      })
      harness.result.current.handleGatewayEvent({
        type: 'message.complete',
        session_id: SESSION_ID,
        payload: {
          history_policy: 'ephemeral_control',
          request_id: 'control-turn',
          text: 'monitor created'
        }
      })
    })

    expect(harness.getState().messages).toEqual([
      expect.objectContaining({ id: 'keep-user' })
    ])
    expect(harness.getState()).toEqual(
      expect.objectContaining({ busy: false, streamId: null })
    )
  })
})

describe('useMessageStream voice user turns', () => {
  afterEach(() => {
    cleanup()
    setActiveSessionId(null)
    $busy.set(false)
  })

  it('adds one session-scoped voice user bubble per ASR request id', () => {
    const harness = createMessageStreamHarness(stateWithMessages([]))

    const event: RpcEvent = {
      type: 'multimodal.asr_final',
      session_id: SESSION_ID,
      payload: { request_id: 'voice-turn-1', text: '  帮我看下当前画面  ' }
    }

    act(() => {
      harness.result.current.handleGatewayEvent(event)
      // Reconnect/replay may deliver the same final more than once.
      harness.result.current.handleGatewayEvent(event)
    })

    expect(harness.getState().messages).toEqual([
      expect.objectContaining({
        id: 'voice-user-voice-turn-1',
        role: 'user',
        voice: true
      })
    ])

    expect(chatMessageText(harness.getState().messages[0])).toBe('帮我看下当前画面')
  })

  it('does not attribute an unscoped ASR final to the focused chat', () => {
    const harness = createMessageStreamHarness(stateWithMessages([]))

    act(() => {
      harness.result.current.handleGatewayEvent({
        type: 'multimodal.asr_final',
        payload: { request_id: 'voice-without-owner', text: '不应该出现' }
      })
    })

    expect(harness.getState().messages).toEqual([])
  })
})

describe('useMessageStream deferred QueryWorker routing', () => {
  afterEach(() => {
    cleanup()
    clearClarifyRequest(undefined, SESSION_ID)
    setActiveSessionId(null)
    $busy.set(false)
  })

  it('finishes the original query tool bubble after the foreground handoff becomes idle', () => {
    const harness = createMessageStreamHarness(
      stateWithMessages([
        { id: 'user-q1', role: 'user', parts: [textPart('describe the scene')] }
      ])
    )

    act(() => {
      for (const event of queryDispatchEvents('turn-q1', 'tool-q1', 'inspect the camera scene')) {
        harness.result.current.handleGatewayEvent(event)
      }
    })

    expect(harness.getState().streamId).toBeNull()
    expect(harness.getState().messages.filter(message => message.role === 'assistant')).toHaveLength(1)
    expect(harness.getState().messages[1]).toEqual(expect.objectContaining({ pending: true }))

    act(() => {
      for (const event of queryWorkerEvents('turn-q1', 'partial scene', 'final scene answer')) {
        harness.result.current.handleGatewayEvent(event)
      }
    })

    const assistants = harness.getState().messages.filter(message => message.role === 'assistant')

    expect(assistants).toHaveLength(1)
    expect(assistants[0].pending).toBe(false)
    expect(chatMessageText(assistants[0])).toBe('final scene answer')
    expect(assistants[0].parts).toEqual(
      expect.arrayContaining([expect.objectContaining({ toolCallId: 'tool-q1', toolName: 'query_multimodal' })])
    )
  })

  it('keeps two out-of-order QueryWorker replies attached to their originating questions', () => {
    const harness = createMessageStreamHarness(
      stateWithMessages([{ id: 'user-q1', role: 'user', parts: [textPart('question one')] }])
    )

    act(() => {
      for (const event of queryDispatchEvents('turn-q1', 'tool-q1', 'query one')) {
        harness.result.current.handleGatewayEvent(event)
      }

      harness.update(state => ({
        ...state,
        messages: [...state.messages, { id: 'user-q2', role: 'user', parts: [textPart('question two')] }],
        busy: true,
        awaitingResponse: true,
        sawAssistantPayload: false,
        turnStartedAt: 456
      }))

      for (const event of queryDispatchEvents('turn-q2', 'tool-q2', 'query two')) {
        harness.result.current.handleGatewayEvent(event)
      }

      // The later question finishes first.
      for (const event of queryWorkerEvents('turn-q2', 'two partial', 'answer two')) {
        harness.result.current.handleGatewayEvent(event)
      }

      for (const event of queryWorkerEvents('turn-q1', 'one partial', 'answer one')) {
        harness.result.current.handleGatewayEvent(event)
      }
    })

    const visible = harness.getState().messages.map(message =>
      message.role === 'user' ? chatMessageText(message) : chatMessageText(message)
    )

    expect(visible).toEqual(['question one', 'answer one', 'question two', 'answer two'])
    expect(harness.getState().messages.filter(message => message.pending)).toEqual([])
  })

  it('does not let an older QueryWorker completion clear a newer foreground turn or clarify prompt', () => {
    const harness = createMessageStreamHarness(
      stateWithMessages([{ id: 'user-q1', role: 'user', parts: [textPart('question one')] }])
    )

    act(() => {
      for (const event of queryDispatchEvents('turn-q1', 'tool-q1', 'query one')) {
        harness.result.current.handleGatewayEvent(event)
      }

      harness.update(state => ({
        ...state,
        messages: [...state.messages, { id: 'user-q2', role: 'user', parts: [textPart('question two')] }],
        busy: true,
        awaitingResponse: true,
        needsInput: true,
        sawAssistantPayload: false,
        turnStartedAt: 789
      }))
      harness.result.current.handleGatewayEvent({
        type: 'message.start',
        session_id: SESSION_ID,
        payload: { request_id: 'turn-q2' }
      })
      harness.result.current.handleGatewayEvent({
        type: 'tool.start',
        session_id: SESSION_ID,
        payload: { tool_id: 'tool-q2', name: 'clarify', context: 'choose one' }
      })
    })

    setClarifyRequest({
      choices: ['one', 'two'],
      question: 'choose one',
      requestId: 'clarify-q2',
      sessionId: SESSION_ID
    })
    $busy.set(true)
    const foregroundStreamId = harness.getState().streamId

    act(() => {
      for (const event of queryWorkerEvents('turn-q1', 'old partial', 'old answer')) {
        harness.result.current.handleGatewayEvent(event)
      }
    })

    const state = harness.getState()

    expect(state.busy).toBe(true)
    expect(state.awaitingResponse).toBe(false)
    expect(state.streamId).toBe(foregroundStreamId)
    expect(state.needsInput).toBe(true)
    expect(state.messages.find(message => message.id === foregroundStreamId)?.pending).toBe(true)
    expect($busy.get()).toBe(true)
    expect($clarifyRequests.get()[SESSION_ID]?.requestId).toBe('clarify-q2')
  })
})

// A failed turn is deliberately not persisted as assistant text by the backend
// (gateway/run.py:10580), so if the completion isn't marked as an error the
// post-turn hydrate refetches a turn with no reply and erases the message the
// user just saw — it flashes on screen and vanishes. The gateway always sends
// status:"error" for these, so the status field alone must drive the decision;
// matching on the text was unreliable because the gateway prepends "Error: "
// when the backend produced no visible output.
describe('useMessageStream failed-turn completions', () => {
  afterEach(() => {
    cleanup()
    setActiveSessionId(null)
    $busy.set(false)
  })

  const QUOTA_ERRORS = [
    'Billing or credits exhausted: 402 insufficient credits',
    'Error: 429 You exceeded your current quota',
    'Error: Context length exceeded (250,000 tokens). Cannot compress further.',
    'Error: content_policy_blocked: flagged'
  ]

  it.each(QUOTA_ERRORS)('keeps %s as an inline error and skips hydrate', errorText => {
    const harness = createMessageStreamHarness(
      stateWithMessages([{ id: 'user-quota', role: 'user', parts: [textPart('你好')] }])
    )

    act(() => {
      harness.result.current.handleGatewayEvent({
        type: 'message.complete',
        session_id: SESSION_ID,
        payload: { status: 'error', text: errorText }
      })
    })

    const state = harness.getState()
    const assistant = state.messages.find(message => message.role === 'assistant')

    expect(assistant?.error).toBe(errorText)
    expect(assistant?.pending).toBe(false)
    // The decisive assertion: hydrate would overwrite this bubble with a stored
    // turn that has no assistant reply, which is exactly the disappearance bug.
    expect(harness.hydrateFromStoredSession).not.toHaveBeenCalled()
    expect(state.busy).toBe(false)
    expect(state.awaitingResponse).toBe(false)
  })

  it('labels an errored turn that carries no text at all', () => {
    const harness = createMessageStreamHarness(
      stateWithMessages([{ id: 'user-empty', role: 'user', parts: [textPart('你好')] }])
    )

    act(() => {
      harness.result.current.handleGatewayEvent({
        type: 'message.complete',
        session_id: SESSION_ID,
        payload: { status: 'error', text: '' }
      })
    })

    const assistant = harness.getState().messages.find(message => message.role === 'assistant')

    expect(assistant?.error).toBeTruthy()
    expect(harness.hydrateFromStoredSession).not.toHaveBeenCalled()
  })

  it('still hydrates a successful completion', () => {
    const harness = createMessageStreamHarness(
      stateWithMessages([{ id: 'user-ok', role: 'user', parts: [textPart('你好')] }])
    )

    act(() => {
      harness.result.current.handleGatewayEvent({
        type: 'message.complete',
        session_id: SESSION_ID,
        payload: { status: 'complete', text: '嗨' }
      })
    })

    const assistant = harness.getState().messages.find(message => message.role === 'assistant')

    expect(assistant?.error).toBeUndefined()
    expect(chatMessageText(assistant!)).toBe('嗨')
    expect(harness.hydrateFromStoredSession).toHaveBeenCalled()
  })
})
