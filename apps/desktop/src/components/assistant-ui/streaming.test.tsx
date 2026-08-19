import { AssistantRuntimeProvider, type ThreadMessage, useExternalStoreRuntime } from '@assistant-ui/react'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { useEffect, useState } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { Thread } from './thread'

const createdAt = new Date('2026-05-01T00:00:00.000Z')

const resizeObservers = new Set<TestResizeObserver>()

class TestResizeObserver {
  private target: Element | null = null

  constructor(private readonly callback: ResizeObserverCallback) {
    resizeObservers.add(this)
  }

  observe(target: Element) {
    this.target = target
  }

  unobserve() {}

  disconnect() {
    resizeObservers.delete(this)
  }

  trigger(height: number) {
    if (!this.target) {
      return
    }

    this.callback(
      [
        {
          contentRect: { height } as DOMRectReadOnly,
          target: this.target
        } as ResizeObserverEntry
      ],
      this as unknown as ResizeObserver
    )
  }
}

vi.stubGlobal('ResizeObserver', TestResizeObserver)
vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) =>
  window.setTimeout(() => callback(performance.now()), 0)
)
vi.stubGlobal('cancelAnimationFrame', (id: number) => window.clearTimeout(id))

Element.prototype.scrollTo = function scrollTo() {}

Element.prototype.animate = function animate() {
  return {
    cancel: () => {},
    finished: Promise.resolve()
  } as unknown as Animation
}

// jsdom returns 0 for offset*; some layout code reads those to size the
// viewport. Fall through to client* (which tests can override) or a sane
// default so message rows render with non-zero dimensions.
function stubOffsetDimension(
  prop: 'offsetHeight' | 'offsetWidth',
  clientProp: 'clientHeight' | 'clientWidth',
  fallback: number
) {
  const previous = Object.getOwnPropertyDescriptor(HTMLElement.prototype, prop)

  Object.defineProperty(HTMLElement.prototype, prop, {
    configurable: true,
    get() {
      return previous?.get?.call(this) || (this as HTMLElement)[clientProp] || fallback
    }
  })
}

stubOffsetDimension('offsetWidth', 'clientWidth', 800)
stubOffsetDimension('offsetHeight', 'clientHeight', 600)

async function wait(ms: number) {
  await act(async () => {
    await new Promise(resolve => window.setTimeout(resolve, ms))
  })
}

function userMessage(): ThreadMessage {
  return {
    id: 'user-1',
    role: 'user',
    content: [{ type: 'text', text: 'Stream a response' }],
    attachments: [],
    createdAt,
    metadata: { custom: {} }
  } as ThreadMessage
}

function assistantMessage(text: string, running = true): ThreadMessage {
  return {
    id: 'assistant-1',
    role: 'assistant',
    content: [{ type: 'text', text }],
    status: running ? { type: 'running' } : { type: 'complete', reason: 'stop' },
    createdAt,
    metadata: {
      unstable_state: null,
      unstable_annotations: [],
      unstable_data: [],
      steps: [],
      custom: {}
    }
  } as ThreadMessage
}

function assistantErrorMessage(error: string): ThreadMessage {
  return {
    id: 'assistant-error-1',
    role: 'assistant',
    content: [],
    status: { type: 'incomplete', reason: 'error', error },
    createdAt,
    metadata: {
      unstable_state: null,
      unstable_annotations: [],
      unstable_data: [],
      steps: [],
      custom: {}
    }
  } as ThreadMessage
}

function assistantReasoningMessage(text: string, running = false): ThreadMessage {
  return {
    id: 'assistant-reasoning-1',
    role: 'assistant',
    content: [{ type: 'reasoning', text }],
    status: running ? { type: 'running' } : { type: 'complete', reason: 'stop' },
    createdAt,
    metadata: {
      unstable_state: null,
      unstable_annotations: [],
      unstable_data: [],
      steps: [],
      custom: {}
    }
  } as ThreadMessage
}

function assistantMultiReasoningMessage(texts: string[]): ThreadMessage {
  return {
    id: 'assistant-reasoning-multi-1',
    role: 'assistant',
    content: texts.map(text => ({ type: 'reasoning', text })),
    status: { type: 'complete', reason: 'stop' },
    createdAt,
    metadata: {
      unstable_state: null,
      unstable_annotations: [],
      unstable_data: [],
      steps: [],
      custom: {}
    }
  } as ThreadMessage
}

// A running turn that has thought and called a tool but produced no prose yet —
// the phase that used to render as a single status line with no process surface
// at all. `result: undefined` keeps the tool pending.
function assistantThinkingAndCallingMessage(withProse: boolean): ThreadMessage {
  return {
    id: 'assistant-live-1',
    role: 'assistant',
    content: [
      { type: 'reasoning', text: ' Check Beijing first.' },
      {
        type: 'tool-call',
        toolCallId: 'live-call-1',
        toolName: 'terminal',
        args: { command: 'curl wttr.in' },
        argsText: '{}',
        result: { ok: true }
      },
      ...(withProse ? [{ type: 'text', text: 'Beijing is hot tomorrow.' }] : [])
    ],
    status: { type: 'running' },
    createdAt,
    metadata: {
      unstable_state: null,
      unstable_annotations: [],
      unstable_data: [],
      steps: [],
      custom: {}
    }
  } as ThreadMessage
}

// reasoning → text → reasoning: the interleaved shape that used to scatter a
// separate "Thinking" disclosure on either side of the body text.
function assistantSeparatedReasoningMessage(running = true): ThreadMessage {
  return {
    id: 'assistant-reasoning-separated-1',
    role: 'assistant',
    content: [
      { type: 'reasoning', text: ' Complete first thought.', status: { type: 'complete' } },
      { type: 'text', text: 'Interim answer.' },
      {
        type: 'reasoning',
        text: ' Streaming second thought.',
        status: running ? { type: 'running' } : { type: 'complete' }
      }
    ],
    status: running ? { type: 'running' } : { type: 'complete', reason: 'stop' },
    createdAt,
    metadata: {
      unstable_state: null,
      unstable_annotations: [],
      unstable_data: [],
      steps: [],
      custom: {}
    }
  } as ThreadMessage
}

function assistantTodoMessage(
  todos: Array<{ content: string; id: string; status: 'cancelled' | 'completed' | 'in_progress' | 'pending' }>,
  running = true
): ThreadMessage {
  const suffix = todos.map(todo => `${todo.id}:${todo.status}`).join('|') || 'empty'

  return {
    id: `assistant-todo-${running ? 'running' : 'done'}-${suffix}`,
    role: 'assistant',
    content: [
      {
        type: 'tool-call',
        toolCallId: 'todo-1',
        toolName: 'todo',
        args: { todos },
        argsText: JSON.stringify({ todos }),
        ...(running ? {} : { result: { todos } })
      }
    ],
    status: running ? { type: 'running' } : { type: 'complete', reason: 'stop' },
    createdAt,
    metadata: {
      unstable_state: null,
      unstable_annotations: [],
      unstable_data: [],
      steps: [],
      custom: {}
    }
  } as ThreadMessage
}

function assistantImageMessage(running = false): ThreadMessage {
  return {
    id: `assistant-image-${running ? 'running' : 'done'}`,
    role: 'assistant',
    content: [
      {
        type: 'tool-call',
        toolCallId: 'image-1',
        toolName: 'image_generate',
        args: { prompt: 'draw a cat' },
        argsText: JSON.stringify({ prompt: 'draw a cat' }),
        ...(running ? {} : { result: { image: 'https://cdn.example/cat.png', success: true } })
      }
    ],
    status: running ? { type: 'running' } : { type: 'complete', reason: 'stop' },
    createdAt,
    metadata: {
      unstable_state: null,
      unstable_annotations: [],
      unstable_data: [],
      steps: [],
      custom: {}
    }
  } as ThreadMessage
}

function StreamingHarness() {
  const [messages, setMessages] = useState<ThreadMessage[]>([userMessage()])
  const [isRunning, setIsRunning] = useState(true)

  useEffect(() => {
    const first = window.setTimeout(() => {
      setMessages([userMessage(), assistantMessage('first chunk')])
    }, 50)

    const second = window.setTimeout(() => {
      setMessages([userMessage(), assistantMessage('first chunk second chunk')])
    }, 500)

    const complete = window.setTimeout(() => {
      setMessages([userMessage(), assistantMessage('first chunk second chunk', false)])
      setIsRunning(false)
    }, 700)

    return () => {
      window.clearTimeout(first)
      window.clearTimeout(second)
      window.clearTimeout(complete)
    }
  }, [])

  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages,
    isRunning,
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread loading={isRunning && messages.at(-1)?.role !== 'assistant' ? 'response' : undefined} />
    </AssistantRuntimeProvider>
  )
}

function TodoHarness({ message }: { message: ThreadMessage }) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages: [message],
    isRunning: message.status?.type === 'running',
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread />
    </AssistantRuntimeProvider>
  )
}

function MessageHarness({ message }: { message: ThreadMessage }) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages: [message],
    isRunning: false,
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread />
    </AssistantRuntimeProvider>
  )
}

function RunningMessageHarness({ message }: { message: ThreadMessage }) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages: [message],
    isRunning: true,
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread />
    </AssistantRuntimeProvider>
  )
}

function ReasoningHarness() {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages: [assistantReasoningMessage(' The user is asking what this file is.')],
    isRunning: false,
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread />
    </AssistantRuntimeProvider>
  )
}

function RunningReasoningHarness() {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages: [assistantReasoningMessage('```ts\nconst answer = 42\n', true)],
    isRunning: true,
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread />
    </AssistantRuntimeProvider>
  )
}

function GroupedReasoningHarness() {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages: [assistantMultiReasoningMessage([' First thought.', ' Second thought.'])],
    isRunning: false,
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread />
    </AssistantRuntimeProvider>
  )
}

function IntroHarness() {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages: [],
    isRunning: false,
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread intro={{ personality: 'default', seed: 1 }} />
    </AssistantRuntimeProvider>
  )
}

function DismissibleErrorHarness({ onDismissError }: { onDismissError: (messageId: string) => void }) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    messages: [assistantErrorMessage('OpenRouter rejected the request (403).')],
    isRunning: false,
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread onDismissError={onDismissError} />
    </AssistantRuntimeProvider>
  )
}

describe('assistant-ui streaming renderer', () => {
  beforeEach(() => {
    resizeObservers.clear()
  })

  it('renders assistant text incrementally before completion', async () => {
    const { container } = render(<StreamingHarness />)

    expect(screen.getByRole('status', { name: 'Argus is loading a response' })).toBeTruthy()

    await wait(80)

    await waitFor(() => {
      expect(container.textContent).toContain('first chunk')
    })
    expect(container.textContent).not.toContain('second chunk')
    expect(screen.queryByRole('status', { name: 'Argus is loading a response' })).toBeNull()

    await wait(500)

    await waitFor(() => {
      expect(container.textContent).toContain('first chunk second chunk')
    })

    await wait(250)

    await waitFor(() => {
      expect(container.textContent).toContain('first chunk second chunk')
    })
  })

  it('does not render composer clearance for intro-only threads', () => {
    const { container } = render(<IntroHarness />)

    expect(container.querySelector('[data-slot="aui_composer-clearance"]')).toBeNull()
  })

  it('renders assistant provider errors inline', () => {
    render(<MessageHarness message={assistantErrorMessage('OpenRouter rejected the request (403).')} />)

    expect(screen.getByRole('alert').textContent).toContain('OpenRouter rejected the request (403).')
  })

  it('omits the dismiss control when no onDismissError handler is supplied', () => {
    render(<MessageHarness message={assistantErrorMessage('OpenRouter rejected the request (403).')} />)

    expect(screen.queryByRole('button', { name: 'Dismiss error' })).toBeNull()
  })

  it('invokes onDismissError with the errored message id when the dismiss control is clicked', () => {
    const onDismissError = vi.fn()
    render(<DismissibleErrorHarness onDismissError={onDismissError} />)

    const dismiss = screen.getByRole('button', { name: 'Dismiss error' })
    fireEvent.click(dismiss)

    expect(onDismissError).toHaveBeenCalledTimes(1)
    expect(onDismissError).toHaveBeenCalledWith('assistant-error-1')
  })

  // Scroll behavior (follow-at-bottom, escape-on-scroll-up, re-engage) is owned
  // by the use-stick-to-bottom library and covered by its own test suite. We
  // don't re-assert its scrollTop mechanics here — doing so in jsdom (no real
  // layout, spring animation via rAF) only produces brittle change-detector
  // tests. The rendering/streaming-content tests below remain the contract.

  it('renders an incomplete streaming fenced code block as a code card', async () => {
    const { container } = render(<RunningMessageHarness message={assistantMessage('```ts\nconst answer = 42\n')} />)

    await waitFor(() => {
      expect(container.querySelector('[data-slot="code-card"]')).toBeTruthy()
    })

    expect(container.textContent).toContain('const answer = 42')
    expect(container.textContent).not.toContain('```ts')
  })

  // Reasoning is always represented by EXACTLY ONE disclosure per turn, whether
  // the turn is streaming or settled — see the ★ comment on
  // ReasoningAccordionGroup in thread.tsx. The invariant being protected is the
  // count: a per-reasoning-part disclosure scattered several "Thinking" blocks
  // between tool cards and body text on interleaved turns, which is the display
  // bug. A regression toward per-part disclosures is not a feature.
  //
  // This used to additionally assert that mid-stream showed ZERO disclosures and
  // only a one-line `aui_assistant-thinking-row`. That shape has been deliberately
  // reversed: it made the whole think-and-call phase invisible (no reasoning text,
  // no tool rows, nothing expandable) and then swapped the entire subtree in when
  // prose landed. The block is now live for the whole turn; only the *count* of
  // reasoning disclosures is still pinned.
  it('shows exactly one reasoning disclosure while reasoning streams', () => {
    const { container } = render(<RunningReasoningHarness />)

    expect(container.querySelectorAll('[data-slot="aui_thinking-disclosure"]').length).toBe(1)
    // Present and expandable *during* the stream — that is the point.
    expect(within(container).getByRole('button', { name: /thinking/i })).toBeTruthy()
    // The old one-line-only surface must not come back.
    expect(container.querySelector('[data-slot="aui_assistant-thinking-row"]')).toBeNull()
  })

  // "界面又乱了": the turn's shell used to be two different layouts that swapped
  // at the first text delta — a bare status line became avatar + header row +
  // padded card + footer + process block, all in one frame, shoving everything
  // below it. This pins the shell as invariant across that moment: same set of
  // structural slots before and after prose, so nothing mounts, unmounts, or
  // moves. It is a set comparison, not a snapshot — adding a slot to BOTH states
  // is fine, adding one to only one state is the regression.
  it('keeps the same structural slots when prose arrives mid-run', () => {
    const shellSlots = (root: HTMLElement) =>
      [...root.querySelectorAll('[data-slot]')]
        .map(el => el.getAttribute('data-slot'))
        .filter(slot => slot?.startsWith('aui_assistant') || slot === 'aui_tool-history-panel')
        .sort()

    const before = render(<RunningMessageHarness message={assistantThinkingAndCallingMessage(false)} />)
    const slotsBefore = shellSlots(before.container)

    cleanup()

    const after = render(<RunningMessageHarness message={assistantThinkingAndCallingMessage(true)} />)
    const slotsAfter = shellSlots(after.container)

    // The process block and the message shell must exist in BOTH states.
    expect(slotsBefore).toContain('aui_tool-history-panel')
    expect(slotsBefore).toContain('aui_assistant-message-root')
    expect(slotsAfter).toEqual(slotsBefore)
  })

  // The answer card is mounted from the first part, so before prose arrives it
  // must contain NOTHING — it is hidden purely by the `empty:hidden` utility,
  // and `:empty` fails if even one child element is present. jsdom does not
  // apply the CSS, so this asserts the precondition the rule depends on.
  //
  // The trap: `MessagePrimitive.Parts` defaults `unstable_showEmptyOnNonTextEnd`
  // to true, which injects an empty Text part whenever the last part is a
  // tool-call — exactly the mid-execution state. That lone node would keep the
  // card non-empty and paint a stray blank box under every running turn. Hence
  // the explicit `={false}` in thread.tsx; this test is its tripwire.
  it('leaves the answer card truly empty until prose arrives', () => {
    const cardOf = (root: HTMLElement) =>
      root.querySelector('[data-slot="aui_assistant-message-content"]')

    const before = render(<RunningMessageHarness message={assistantThinkingAndCallingMessage(false)} />)
    const emptyCard = cardOf(before.container)

    expect(emptyCard).not.toBeNull()
    expect(emptyCard?.childElementCount).toBe(0)
    expect(emptyCard?.matches(':empty')).toBe(true)

    cleanup()

    // Contrast: once prose lands the same card is non-empty and shows it.
    const after = render(<RunningMessageHarness message={assistantThinkingAndCallingMessage(true)} />)
    const filledCard = cardOf(after.container)

    expect(filledCard?.matches(':empty')).toBe(false)
    expect(filledCard?.textContent).toContain('Beijing is hot tomorrow.')
  })

  it('renders an incomplete reasoning fenced code block as a code card', async () => {
    // Unterminated fence: the model was cut off mid-block, so the transcript
    // keeps `` ```ts `` with no closer. It must still render as a card rather
    // than leaking the backticks as literal text.
    const { container } = render(<MessageHarness message={assistantReasoningMessage('```ts\nconst answer = 42\n')} />)

    fireEvent.click(within(container).getByRole('button', { name: /thinking/i }))

    await waitFor(() => {
      expect(container.querySelector('[data-slot="code-card"]')).toBeTruthy()
    })

    await waitFor(() => {
      expect(container.querySelector('[data-slot="aui_reasoning-text"]')?.textContent).toContain('const answer = 42')
    })
    expect(container.textContent).not.toContain('```ts')
  })

  it('renders reasoning text without a leading token space', () => {
    const { container } = render(<ReasoningHarness />)
    const ui = within(container)

    fireEvent.click(ui.getByRole('button', { name: /thinking/i }))

    expect(container.querySelector('[data-slot="aui_reasoning-text"]')?.textContent).toBe(
      'The user is asking what this file is.'
    )
  })

  it('merges consecutive reasoning parts into one disclosure body', () => {
    const { container } = render(<GroupedReasoningHarness />)

    const disclosures = container.querySelectorAll('[data-slot="aui_thinking-disclosure"]')
    expect(disclosures.length).toBe(1)

    fireEvent.click(disclosures[0].querySelector('button')!)

    // One body, not one per part: the panel joins the segments before handing
    // them to Markdown, so both thoughts live in a single reasoning container.
    const reasoningParts = container.querySelectorAll('[data-slot="aui_reasoning-text"]')
    expect(reasoningParts.length).toBe(1)

    const body = reasoningParts[0]?.textContent ?? ''
    expect(body).toContain('First thought.')
    expect(body).toContain('Second thought.')
    // Order is the transcript's, not reversed or sorted.
    expect(body.indexOf('First thought.')).toBeLessThan(body.indexOf('Second thought.'))
  })

  it('keeps interleaved reasoning in one disclosure instead of one per group', () => {
    // The interleaved shape that motivated the refactor: reasoning, then body
    // text, then more reasoning. Two disclosures here would put a "Thinking"
    // block on either side of the body text — the original display bug.
    const { container } = render(<MessageHarness message={assistantSeparatedReasoningMessage(false)} />)

    const disclosures = container.querySelectorAll('[data-slot="aui_thinking-disclosure"]')
    expect(disclosures.length).toBe(1)

    expect(container.textContent).toContain('Interim answer.')

    fireEvent.click(disclosures[0].querySelector('button')!)

    const body = container.querySelector('[data-slot="aui_reasoning-text"]')?.textContent ?? ''
    expect(body).toContain('Complete first thought.')
    expect(body).toContain('Streaming second thought.')
  })

  it('collapses the thinking disclosure by default once the turn completes', () => {
    const { container } = render(<ReasoningHarness />)

    const toggle = container.querySelector('[data-slot="aui_thinking-disclosure"] button')
    expect(toggle?.getAttribute('aria-expanded')).toBe('false')
    // Collapsed means the reasoning body is absent from the DOM, not just hidden.
    expect(container.querySelector('[data-slot="aui_reasoning-text"]')).toBeNull()
  })

  it('does not render an inline todo panel — todos live in the composer status stack', () => {
    const { container } = render(
      <TodoHarness
        message={assistantTodoMessage([
          { content: 'Gather ingredients', id: 'prep', status: 'completed' },
          { content: 'Boil water', id: 'boil', status: 'in_progress' }
        ])}
      />
    )

    expect(container.querySelector('[data-slot="aui_todo-hoisted"]')).toBeNull()
  })

  it('renders completed image generation results in the tool slot', async () => {
    const { container } = render(<MessageHarness message={assistantImageMessage()} />)

    await waitFor(() => {
      expect(screen.getByRole('img', { name: 'Generated image' }).getAttribute('src')).toBe(
        'https://cdn.example/cat.png'
      )
    })
    expect(container.querySelector('[data-slot="aui_generated-image"]')).toBeTruthy()
    expect(screen.queryByRole('status', { name: /rendering image/i })).toBeNull()
  })
})
