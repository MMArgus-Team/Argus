/**
 * The process block: one turn renders an answer card plus ONE block below it
 * holding both the reasoning (☁️) and the tool calls.
 *
 * Two shapes are pinned here.
 *
 * 1. Merged, not split. Reasoning used to live in its own panel ABOVE the answer
 *    card while tools sat in a separate panel BELOW it, and the reasoning panel
 *    emitted one disclosure per reasoning segment. A single two-tool turn
 *    therefore showed *three* process surfaces — two of them labelled
 *    "Thinking", one of which displayed reasoning prose that happened to quote a
 *    shell command and so read like a stray tool row.
 *
 * 2. Live, not retrospective. The block is mounted for the WHOLE turn. It used
 *    to be gated on `hasPendingTool` + `hasVisibleText`, so during the
 *    think-and-call phase it rendered nothing at all and the process was
 *    invisible until the answer arrived — then appeared fully-formed, shoving
 *    the layout. The running-state tests below are that regression's tripwire.
 */
import {
  AssistantRuntimeProvider,
  type ThreadMessage,
  ThreadPrimitive,
  useExternalStoreRuntime
} from '@assistant-ui/react'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { clearAllPrompts, setApprovalRequest } from '@/store/prompts'
import { $activeSessionId } from '@/store/session'
import { resetGeneratingTools, setSessionGeneratingTool } from '@/store/tool-generating'

vi.mock('@/components/assistant-ui/tool-fallback', () => ({
  ToolFallback: ({ toolCallId }: { toolCallId: string }) => (
    <div data-testid="tool-row">{toolCallId}</div>
  ),
  ToolGroupSlot: () => null
}))

import { ToolHistoryPanel } from './thread'

afterEach(() => {
  cleanup()
  clearAllPrompts()
  resetGeneratingTools()
  $activeSessionId.set(null)
})

function Harness({ message, running = false }: { message: ThreadMessage; running?: boolean }) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    isRunning: running,
    messages: [message],
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <ThreadPrimitive.Messages components={{ Message: ToolHistoryPanel }} />
    </AssistantRuntimeProvider>
  )
}

const message = (content: unknown[], running = false): ThreadMessage =>
  ({
    content,
    createdAt: new Date(0),
    id: 'assistant-1',
    metadata: { custom: {}, steps: [], unstable_annotations: [], unstable_data: [], unstable_state: null },
    role: 'assistant',
    status: running ? { type: 'running' } : { reason: 'stop', type: 'complete' }
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

// A call still awaiting its result — `result: undefined` is what marks a tool as
// pending, and a pending tool is what an approval bar hangs off.
const pendingCall = (id: string, toolName: string) => ({
  args: { command: 'curl wttr.in' },
  argsText: '{}',
  toolCallId: id,
  toolName,
  type: 'tool-call'
})

const PANEL = '[data-slot="aui_tool-history-panel"]'

describe('process block', () => {
  it('summarises the batch in the collapsed header', () => {
    render(
      <Harness
        message={message([text('两个城市都查到了。'), call('c1', 'terminal'), call('c2', 'terminal')])}
      />
    )

    // "Ran 2 commands", not a content-free "2 tool calls".
    expect(screen.getByText(/Ran 2 commands/i)).toBeTruthy()
  })

  it('renders exactly ONE thinking row, however many reasoning segments there are', () => {
    render(
      <Harness
        message={message([
          reasoning('先查北京'),
          call('c1', 'terminal'),
          reasoning('再查杭州'),
          call('c2', 'terminal'),
          text('两个城市都查到了。')
        ])}
      />
    )

    fireEvent.click(screen.getByText(/Ran 2 commands/i))

    // Two reasoning segments must merge into a single ☁️ item — one disclosure
    // per segment is what produced the duplicate "Thinking" rows.
    expect(screen.getAllByText('Thinking')).toHaveLength(1)
  })

  it('keeps the reasoning collapsed until asked, then shows every segment', () => {
    render(
      <Harness
        message={message([
          reasoning('先查北京'),
          reasoning('再查杭州'),
          call('c1', 'terminal'),
          text('done')
        ])}
      />
    )

    fireEvent.click(screen.getByText(/Ran 1 command/i))

    // Raw chain-of-thought is long and is not the answer: collapsed by default.
    expect(screen.queryByText(/先查北京/)).toBeNull()

    fireEvent.click(screen.getByText('Thinking'))

    expect(screen.getByText(/先查北京/)).toBeTruthy()
    expect(screen.getByText(/再查杭州/)).toBeTruthy()
  })

  it('still shows the block for a thinking-only turn that called no tools', () => {
    render(<Harness message={message([reasoning('想了想'), text('答案')])} />)

    // No tools → the header names the reasoning rather than printing "0 tool calls".
    expect(screen.getByText('Thinking')).toBeTruthy()
    expect(screen.queryByText(/0 tool call/i)).toBeNull()
  })

  it('renders nothing when a turn has neither reasoning nor tools', () => {
    const { container } = render(<Harness message={message([text('just an answer')])} />)

    expect(container.querySelector(PANEL)).toBeNull()
  })

  it('does not report final counts while a tool is still pending', () => {
    // Counts are not settled mid-flight, so the header reports the current
    // ACTION instead of a tally that would churn on every new call. (It used to
    // hide the whole block here — see the running-state suite below.)
    render(
      <Harness
        message={message([text('working'), pendingCall('c1', 'terminal')], true)}
        running
      />
    )

    expect(screen.queryByText(/Ran 1 command/i)).toBeNull()
  })
})

// ── Running state ───────────────────────────────────────────────────────────
// The user-visible requirement: "一旦思考有内容，应该立刻出现下面这个框，并且持续
// 打出思考和工具调用进展；而不是过程都不显示，最后才出来".
describe('process block while running', () => {
  it('appears as soon as there is reasoning, before any prose', () => {
    const { container } = render(
      <Harness message={message([reasoning('先查北京的天气')], true)} running />
    )

    expect(container.querySelector(PANEL)).not.toBeNull()
  })

  it('appears as soon as a tool is called, before any prose', () => {
    const { container } = render(
      <Harness message={message([pendingCall('c1', 'terminal')], true)} running />
    )

    expect(container.querySelector(PANEL)).not.toBeNull()
  })

  it('renders the pending tool row so its approval bar is reachable', () => {
    // Load-bearing: PendingToolApproval (Run / Reject) only exists inside a
    // rendered pending row. If the block hid or collapsed the row, a turn parked
    // on an approval would have no clickable entry and would hang.
    render(<Harness message={message([reasoning('要跑命令'), pendingCall('c1', 'terminal')], true)} running />)

    expect(screen.getByTestId('tool-row')).toBeTruthy()
  })

  it('keeps the reasoning item collapsed by default even while running', () => {
    render(<Harness message={message([reasoning('先查北京的天气'), pendingCall('c1', 'terminal')], true)} running />)

    expect(screen.getByText('Thinking')).toBeTruthy()
    // Decision: raw chain-of-thought never expands on its own, running or not.
    expect(screen.queryByText(/先查北京的天气/)).toBeNull()
  })

  it('survives prose arriving mid-run without the block disappearing', () => {
    // The old shape swapped surfaces at exactly this moment; the block must be
    // the same panel before and after, never absent.
    const before = render(
      <Harness message={message([reasoning('想'), call('c1', 'terminal')], true)} running />
    )

    expect(before.container.querySelector(PANEL)).not.toBeNull()
    cleanup()

    const after = render(
      <Harness message={message([reasoning('想'), call('c1', 'terminal'), text('答案')], true)} running />
    )

    expect(after.container.querySelector(PANEL)).not.toBeNull()
  })

  it('settles into the count summary once the turn completes', () => {
    render(<Harness message={message([reasoning('想'), call('c1', 'terminal'), text('答案')])} />)

    expect(screen.getByText(/Ran 1 command/i)).toBeTruthy()
  })

  it('does not claim a pending command already ran', () => {
    // A turn parked on an approval must not label itself "Ran 1 command" — the
    // command has not run yet. The header reports the ACTION while in progress
    // and only settles into counts when the turn is actually done.
    render(<Harness message={message([pendingCall('c1', 'terminal')], true)} running />)

    expect(screen.queryByText(/Ran 1 command/i)).toBeNull()
  })

  // ── tool.generating window ────────────────────────────────────────────────
  // The reported symptom: "有时候已经做过工具调用，还没出现思考工具框，而过几秒才
  // 出现。出现时候一般已经调用了 2-3 次工具了". Cause is upstream of rendering — the
  // backend fires tool.start only after the args are fully written AND preflight
  // has run, and for a batch it fires all of them at once. So the pre-call window
  // has ZERO parts and the block had nothing to key off.
  it('appears during the pre-call window, before any tool part exists', () => {
    $activeSessionId.set('sess-1')
    setSessionGeneratingTool('sess-1', 'terminal')

    // No reasoning, no tool parts — the exact state that used to render nothing.
    const { container } = render(<Harness message={message([], true)} running />)

    expect(container.querySelector(PANEL)).not.toBeNull()
    expect(screen.getByText(/Preparing terminal/i)).toBeTruthy()
  })

  it('does not fabricate a tool row for the pre-call signal', () => {
    // Load-bearing: tool.generating carries no tool_id, so a row built from it is
    // one the later id-bearing tool.start cannot merge into — that is exactly the
    // old "two identical tool rows" bug. Status line only, never a row.
    $activeSessionId.set('sess-1')
    setSessionGeneratingTool('sess-1', 'terminal')

    render(<Harness message={message([], true)} running />)

    expect(screen.queryByTestId('tool-row')).toBeNull()
  })

  it('drops the preparing line once the real tool row arrives', () => {
    $activeSessionId.set('sess-1')
    setSessionGeneratingTool('sess-1', 'terminal')

    // The stream clears the store on tool.start; assert the UI follows the store
    // rather than stacking "Preparing" above the row it turned into.
    setSessionGeneratingTool('sess-1', '')

    render(<Harness message={message([pendingCall('c1', 'terminal')], true)} running />)

    expect(screen.queryByText(/Preparing/i)).toBeNull()
    expect(screen.getByTestId('tool-row')).toBeTruthy()
  })

  it('ignores the pre-call signal on a finished turn', () => {
    // The store is session-scoped, so a stale value must not paint "Preparing" on
    // historical messages when they re-render.
    $activeSessionId.set('sess-1')
    setSessionGeneratingTool('sess-1', 'terminal')

    const { container } = render(<Harness message={message([text('答案')])} />)

    expect(container.querySelector(PANEL)).toBeNull()
  })

  it('stops the timer and the shimmer while the turn awaits the user', () => {
    // While a clarify / approval / sudo prompt is outstanding the agent is not
    // working — it is stopped ON the user. Keeping the shimmer sweeping and the
    // seconds ticking would both misreport that and bill the user's own
    // hesitation as agent time. Inherited from the deleted CurrentActivityLine.
    const msg = message([reasoning('要跑命令'), pendingCall('c1', 'terminal')], true)

    const idle = render(<Harness message={msg} running />)

    expect(idle.container.querySelector('.shimmer')).not.toBeNull()

    cleanup()
    // Drive it the real way: a live approval request is what makes
    // $activeSessionAwaitingInput true (it is a computed atom, not settable).
    // Prompts are session-keyed and only surface for the ACTIVE session, so the
    // session id has to be set too or the request stays invisible.
    $activeSessionId.set('sess-1')
    setApprovalRequest({ command: 'curl wttr.in', description: 'network call', sessionId: 'sess-1' })

    const waiting = render(<Harness message={msg} running />)

    expect(waiting.container.querySelector('.shimmer')).toBeNull()
  })
})
