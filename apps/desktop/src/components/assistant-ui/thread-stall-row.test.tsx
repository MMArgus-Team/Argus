/**
 * The stall row must not duplicate the process block's header.
 *
 * `StreamStallIndicator` sits OUTSIDE the answer card and fires after 2s of
 * stream silence — which any slow tool call (a `computer_use` batch) trivially
 * exceeds. The process block's own header already reports the same state, with a
 * better label, a shimmer, and a timer. So both mounting at once produced exactly
 * the duplicate "Thinking" rows that 0fea14e1 set out to remove: it moved
 * `CurrentActivityLine` into the block but left this component behind.
 *
 * Worse, the outer row is a `StatusRow` — a plain `<div role="status">` with no
 * disclosure, no button, no `aria-expanded`. So the visible symptom was a greyed
 * "Thinking" line that had no content, could not be expanded, and sat outside the
 * tool group: three complaints, one stray component.
 *
 * This file is the tripwire that was missing — nothing referenced
 * `aui_stream-stall` before, which is why the regression survived.
 */
import {
  AssistantRuntimeProvider,
  type ThreadMessage,
  ThreadPrimitive,
  useExternalStoreRuntime
} from '@assistant-ui/react'
import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { clearAllPrompts } from '@/store/prompts'
import { $activeSessionId } from '@/store/session'
import { resetGeneratingTools, setSessionGeneratingTool } from '@/store/tool-generating'

vi.mock('@/components/assistant-ui/tool-fallback', () => ({
  ToolFallback: ({ toolCallId }: { toolCallId: string }) => <div>{toolCallId}</div>,
  ToolGroupSlot: () => null
}))

import { StreamStallIndicator } from './thread'

const STALL = '[data-slot="aui_stream-stall"]'

beforeEach(() => {
  vi.useFakeTimers()
})

afterEach(() => {
  vi.useRealTimers()
  cleanup()
  clearAllPrompts()
  resetGeneratingTools()
  $activeSessionId.set(null)
})

function Harness({ message }: { message: ThreadMessage }) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    isRunning: true,
    messages: [message],
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <ThreadPrimitive.Messages components={{ Message: StreamStallIndicator }} />
    </AssistantRuntimeProvider>
  )
}

const message = (content: unknown[]): ThreadMessage =>
  ({
    content,
    createdAt: new Date(0),
    id: 'assistant-1',
    metadata: { custom: {}, steps: [], unstable_annotations: [], unstable_data: [], unstable_state: null },
    role: 'assistant',
    status: { type: 'running' }
  }) as unknown as ThreadMessage

const text = (value: string) => ({ text: value, type: 'text' })
const reasoning = (value: string) => ({ text: value, type: 'reasoning' })

const call = (id: string, toolName: string) => ({
  args: {},
  argsText: '{}',
  result: { ok: true },
  toolCallId: id,
  toolName,
  type: 'tool-call'
})

/** Past the 2s stall threshold. */
const waitForStall = async () => {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(2_500)
  })
}

describe('StreamStallIndicator', () => {
  it('stays silent while a tool call is on screen (the process block owns that)', async () => {
    // The screenshot case: a long computer_use batch. Silence exceeds the
    // threshold, but the block's header is already reporting it.
    const { container } = render(<Harness message={message([call('c1', 'computer_use')])} />)

    await waitForStall()

    expect(container.querySelector(STALL)).toBeNull()
  })

  it('stays silent when the turn has reasoning', async () => {
    const { container } = render(<Harness message={message([reasoning('先想一下')])} />)

    await waitForStall()

    expect(container.querySelector(STALL)).toBeNull()
  })

  it('stays silent while a tool is being prepared', async () => {
    // tool.generating: no part exists yet, but the block IS mounted for it.
    $activeSessionId.set('s1')
    setSessionGeneratingTool('s1', 'terminal')

    const { container } = render(<Harness message={message([text('稍等')])} />)

    await waitForStall()

    expect(container.querySelector(STALL)).toBeNull()
  })

  it('still reports a genuine mid-prose stall (no tools, no reasoning)', async () => {
    // The one case the block cannot cover: prose that stops mid-stream. Keeping
    // this is why the fix gates the row instead of deleting it.
    const { container } = render(<Harness message={message([text('正在回答…')])} />)

    await waitForStall()

    expect(container.querySelector(STALL)).not.toBeNull()
  })

  it('does not fire before the stall threshold', async () => {
    const { container } = render(<Harness message={message([text('正在回答…')])} />)

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1_000)
    })

    expect(container.querySelector(STALL)).toBeNull()
  })

  it('reports via aria-label rather than a hardcoded English string', async () => {
    // Was the literal 'Argus is thinking' — the only un-i18n'd label in the file.
    const { container } = render(<Harness message={message([text('正在回答…')])} />)

    await waitForStall()

    const row = container.querySelector(STALL)

    expect(row?.getAttribute('aria-label')).toBeTruthy()
    expect(row?.getAttribute('aria-label')).not.toBe('Argus is thinking')
  })

  it('is a status row, not a disclosure — it was never clickable', async () => {
    // Pins WHY the user could not expand it, so a future "make it expandable"
    // fix goes to the process block's ☁️ item instead of resurrecting this row.
    const { container } = render(<Harness message={message([text('正在回答…')])} />)

    await waitForStall()

    const row = container.querySelector(STALL)

    expect(row?.getAttribute('role')).toBe('status')
    expect(row?.querySelector('button')).toBeNull()
    expect(row?.getAttribute('aria-expanded')).toBeNull()
  })
})
