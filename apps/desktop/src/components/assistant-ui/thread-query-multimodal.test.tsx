import {
  AssistantRuntimeProvider,
  type ThreadMessage,
  ThreadPrimitive,
  useExternalStoreRuntime
} from '@assistant-ui/react'
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import type { ComponentProps } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/components/assistant-ui/tool-fallback', () => ({
  ToolFallback: ({ argsFields, toolCallId }: { argsFields?: unknown; toolCallId: string }) => (
    <div data-args-fields={JSON.stringify(argsFields ?? null)} data-testid="durable-tool-receipt">
      {toolCallId}
    </div>
  ),
  ToolGroupSlot: () => null
}))

import type { MmTrajectoryEntry } from '@/app/multimodal/trajectory-grouping'
import { $mmQueryTrajectory } from '@/store/multimodal'

import {
  QueryMultimodalTool,
  selectQueryWorkerToolEntries,
  ToolHistoryPanel
} from './thread'

afterEach(cleanup)

beforeEach(() => {
  $mmQueryTrajectory.set([])
})

function started(taskId: string, toolId: string, seq: number, framePrefix: string): MmTrajectoryEntry {
  return {
    event: 'multimodal.trajectory',
    id: `${taskId}-started`,
    payload: {
      frames: [1, 2, 3].map(index => ({
        frame_id: `${framePrefix}-${index}`,
        jpeg_b64: `${framePrefix}-jpeg-${index}`,
        source_type: 'camera',
        ts: seq + index / 10
      })),
      n_frames: 3,
      parent_user_message_id: `turn-${taskId}`,
      task_id: taskId,
      tool_id: toolId
    },
    phase: 'started',
    seq,
    ts: seq,
    worker: 'QueryWorker'
  }
}

function queryToolProps(
  result: unknown,
  toolCallId: string
): ComponentProps<typeof QueryMultimodalTool> {
  return {
    addResult: vi.fn(),
    args: { instruction: '看当前画面' },
    argsText: '{"instruction":"看当前画面"}',
    result,
    resume: vi.fn(),
    status: { type: 'complete' },
    type: 'tool-call',
    toolCallId,
    toolName: 'query_multimodal'
  }
}

function ToolHistoryHarness({ message }: { message: ThreadMessage }) {
  const runtime = useExternalStoreRuntime<ThreadMessage>({
    isRunning: false,
    messages: [message],
    onNew: async () => {}
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <ThreadPrimitive.Messages components={{ Message: ToolHistoryPanel }} />
    </AssistantRuntimeProvider>
  )
}

describe('QueryMultimodalTool inline trajectory', () => {
  it('binds result.task_id to exactly its three-frame task while preserving the durable receipt', () => {
    $mmQueryTrajectory.set([
      started('qry_A', 'call-A', 1, 'A-frame'),
      started('qry_B', 'call-B', 2, 'B-frame')
    ])

    render(<QueryMultimodalTool {...queryToolProps({ task_id: 'qry_A' }, 'call-A')} />)

    expect(screen.getByTestId('durable-tool-receipt').textContent).toBe('call-A')
    const panel = screen.getByTestId('query-worker-trajectory-panel')

    expect(panel.querySelector('[data-query-worker-task="qry_A"]')).not.toBeNull()
    expect(panel.querySelector('[data-query-worker-task="qry_B"]')).toBeNull()
    expect(within(panel).getAllByRole('img')).toHaveLength(3)
    expect(within(panel).getByRole('img', { name: 'A-frame-1' }).getAttribute('src')).toBe(
      'data:image/jpeg;base64,A-frame-jpeg-1'
    )
    expect(panel.textContent).not.toContain('B-frame')
  })

  it('parses a serialized tool result and never crosses into a concurrent task sharing the store', () => {
    const taskA = started('qry_A', 'call-A', 1, 'A-frame')
    const taskB = started('qry_B', 'call-B', 2, 'B-frame')
    const selected = selectQueryWorkerToolEntries([taskB, taskA], 'qry_B', 'call-B')

    expect(selected.map(entry => entry.id)).toEqual(['qry_B-started'])

    $mmQueryTrajectory.set([taskA, taskB])
    render(<QueryMultimodalTool {...queryToolProps('{"task_id":"qry_B"}', 'call-B')} />)

    const panel = screen.getByTestId('query-worker-trajectory-panel')

    expect(panel.querySelector('[data-query-worker-task="qry_B"]')).not.toBeNull()
    expect(within(panel).getAllByRole('img')).toHaveLength(3)
    expect(panel.textContent).toContain('B-frame-1')
    expect(panel.textContent).not.toContain('A-frame')
  })

  it('shows a task-owned waiting state without borrowing another task trajectory', () => {
    $mmQueryTrajectory.set([started('qry_B', 'call-B', 2, 'B-frame')])

    render(<QueryMultimodalTool {...queryToolProps({ task_id: 'qry_A' }, 'call-A')} />)

    expect(screen.getByTestId('query-worker-trajectory-waiting')).toBeTruthy()
    expect(screen.queryByTestId('query-worker-trajectory-panel')).toBeNull()
    expect(document.body.textContent).not.toContain('B-frame')
  })

  it('keeps the detailed trajectory and args sidecar in completed-text tool history', () => {
    $mmQueryTrajectory.set([started('qry_A', 'call-A', 1, 'A-frame')])

    const message = {
      content: [
        {
          args: { instruction: '看当前画面' },
          argsFields: [{ key: 'instruction', kind: 'freeform', value: '看当前画面' }],
          argsText: '{"instruction":"看当前画面"}',
          result: { task_id: 'qry_A' },
          toolCallId: 'call-A',
          toolName: 'query_multimodal',
          type: 'tool-call'
        },
        { text: '这是 QueryWorker 的最终答案。', type: 'text' }
      ],
      createdAt: new Date('2026-08-14T10:00:00.000Z'),
      id: 'assistant-query-complete',
      metadata: {
        custom: {},
        steps: [],
        unstable_annotations: [],
        unstable_data: [],
        unstable_state: null
      },
      role: 'assistant',
      status: { reason: 'stop', type: 'complete' }
    } as unknown as ThreadMessage

    render(<ToolHistoryHarness message={message} />)

    // The header now summarises what the batch DID (summarizeToolSteps) rather
    // than printing a bare count — `query_multimodal` reads as a frame inspect.
    const history = screen
      .getByText(/Inspected 1 frame/i)
      .closest('[data-slot="aui_tool-history-panel"]')

    expect(history).not.toBeNull()
    fireEvent.click(within(history as HTMLElement).getByRole('button'))

    const receipt = within(history as HTMLElement).getByTestId('durable-tool-receipt')

    expect(receipt.textContent).toBe('call-A')
    expect(receipt.getAttribute('data-args-fields')).toContain('instruction')
    expect(within(history as HTMLElement).getAllByRole('img')).toHaveLength(3)
    expect(within(history as HTMLElement).getByTestId('query-worker-trajectory-panel')).toBeTruthy()
  })
})
