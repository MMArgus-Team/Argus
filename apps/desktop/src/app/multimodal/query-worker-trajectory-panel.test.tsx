import { cleanup, render, screen, within } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import {
  buildQueryWorkerTimelines,
  isQueryWorkerTrajectoryEntry,
  queryWorkerStepFromTrajectory
} from './query-worker-trajectory'
import { QueryWorkerTrajectoryPanel } from './query-worker-trajectory-panel'
import type { MmTrajectoryEntry } from './trajectory-grouping'

afterEach(cleanup)

function entry(
  seq: number,
  worker: string,
  phase: string,
  payload: Record<string, unknown> = {},
  event = 'multimodal.trajectory',
  ts = 1_700_000_000 + seq
): MmTrajectoryEntry {
  return {
    event,
    id: `trace-${seq}`,
    payload,
    phase,
    seq,
    ts,
    worker
  }
}

function queryStarted(seq = 2): MmTrajectoryEntry {
  return entry(seq, 'QueryWorker', 'started', {
    ask_ts: 21.5,
    frames: [
      { frame_id: 'ask-1', jpeg_b64: 'frame-one', source_type: 'camera', ts: 19.5 },
      { frame_id: 'ask-2', source_type: 'screen', thumb_b64: 'frame-two', ts: 20.5 },
      { frame_id: 'ask-3', jpeg_b64: 'frame-three', source_type: 'screen_share', ts: 21.5 }
    ],
    n_frames: 3,
    parent_user_message_id: 'turn-A',
    query: '看看这三帧里的商品信息',
    task_id: 'qry_demo'
  })
}

describe('QueryWorker structured trajectory mapping', () => {
  it('sorts a shuffled task by seq and timestamp and retains the exact three ask-time frames', () => {
    const sameSeqLater = { ...entry(4, 'QueryWorker', 'answer_ready', {
      event: { text_preview: '组织答案' },
      task_id: 'qry_demo'
    }, 'multimodal.trajectory', 40), id: 'same-seq-later' }

    const sameSeqEarlier = { ...entry(4, 'QueryWorker', 'delegate_start', {
      task_id: 'qry_demo'
    }, 'multimodal.trajectory', 30), id: 'same-seq-earlier' }

    const [timeline] = buildQueryWorkerTimelines([sameSeqLater, queryStarted(2), sameSeqEarlier])

    expect(timeline.taskId).toBe('qry_demo')
    expect(timeline.steps.map(step => [step.seq, step.ts])).toEqual([
      [2, 1_700_000_002],
      [4, 30],
      [4, 40]
    ])
    expect(timeline.steps[0].title).toBe('Accepted question and locked ask-time · frozen input frames 3')
    expect(timeline.steps[0].frames.map(frame => [frame.frame_id, frame.ts, frame.source_type])).toEqual([
      ['ask-1', 19.5, 'camera'],
      ['ask-2', 20.5, 'screen'],
      ['ask-3', 21.5, 'screen_share']
    ])
  })

  it('maps query_multimodal tool.start args and tool.complete handoff/error as public execution steps', () => {
    const start = entry(
      1,
      'QueryWorker',
      'tool.start',
      {
        args: { instruction: '识别商品', token: '***' },
        context: '派发视觉问题',
        name: 'query_multimodal',
        tool_id: 'call-query'
      },
      'tool.start'
    )

    const complete = entry(
      3,
      'QueryWorker',
      'tool.complete',
      {
        name: 'query_multimodal',
        result: {
          control: 'handoff',
          parent_user_message_id: 'turn-A',
          query: '识别商品',
          reply_owner: 'query_worker',
          task_id: 'qry_demo'
        },
        tool_id: 'call-query'
      },
      'tool.complete'
    )

    const failed = entry(
      4,
      'QueryWorker',
      'tool.complete',
      {
        name: 'query_multimodal',
        result: { error: 'backend unavailable', task_id: 'qry_demo' },
        tool_id: 'call-query-2'
      },
      'tool.complete'
    )

    const startStep = queryWorkerStepFromTrajectory(start)
    const completeStep = queryWorkerStepFromTrajectory(complete)
    const failedStep = queryWorkerStepFromTrajectory(failed)

    const evidence = entry(
      5,
      'QueryWorker',
      'tool.complete',
      {
        name: 'query_multimodal',
        result: {
          mode: 'evidence',
          query: '定位屏幕中的 PDF',
          visual_evidence: 'Observed: report.pdf'
        },
        tool_id: 'call-query-3'
      },
      'tool.complete'
    )
    const evidenceStep = queryWorkerStepFromTrajectory(evidence)

    expect(startStep).toMatchObject({
      callState: 'called',
      status: 'running',
      title: 'Main Agent called query_multimodal',
      worker: 'Main Agent'
    })
    expect(startStep?.toolCalls).toEqual([
      { args: { instruction: '识别商品', token: '***' }, name: 'query_multimodal' }
    ])
    expect(completeStep).toMatchObject({
      detail: '识别商品',
      status: 'running',
      title: 'Main Agent handed the question to QueryWorker'
    })
    expect(failedStep).toMatchObject({
      detail: 'backend unavailable',
      status: 'error',
      title: 'QueryWorker handoff failed'
    })
    expect(evidenceStep).toMatchObject({
      detail: 'Observed: report.pdf',
      status: 'complete',
      title: 'query_multimodal completed'
    })
    expect(evidenceStep?.toolResults[0]).toMatchObject({
      name: 'query_multimodal',
      summary: 'Observed: report.pdf'
    })
    expect(buildQueryWorkerTimelines([complete])?.[0].status).toBe('running')
    expect(buildQueryWorkerTimelines([failed])?.[0].status).toBe('error')
  })

  it('does not classify an unrelated worker or an ordinary main-agent tool as QueryWorker', () => {
    const memoryWriter = entry(1, 'MemoryWriter', 'writer_start', { n_frames: 3 })

    const ordinaryTool = entry(2, 'MainTool:web_search', 'tool.start', {
      args: { query: 'news' },
      name: 'web_search'
    }, 'tool.start')

    expect(isQueryWorkerTrajectoryEntry(memoryWriter)).toBe(false)
    expect(isQueryWorkerTrajectoryEntry(ordinaryTool)).toBe(false)
    expect(queryWorkerStepFromTrajectory(memoryWriter)).toBeNull()
    expect(queryWorkerStepFromTrajectory(ordinaryTool)).toBeNull()
    expect(buildQueryWorkerTimelines([memoryWriter, ordinaryTool])).toEqual([])
  })
})

describe('QueryWorkerTrajectoryPanel', () => {
  it('renders three frozen inputs, tool arguments/results/errors, and an ascending structured timeline', () => {
    const toolStart = entry(
      1,
      'QueryWorker',
      'tool.start',
      {
        args: { instruction: '识别商品', token: '***' },
        context: '派发视觉问题',
        name: 'query_multimodal',
        tool_id: 'call-query'
      },
      'tool.start'
    )

    const recallResult = entry(3, 'RecallWorker', 'bg_progress', {
      event: {
        channel: 'recall',
        observations: [
          {
            args: { brief: '查找商品标签', token: '***' },
            elapsed_sec: 0.42,
            frame_ids: ['ask-1', 'ask-2', 'ask-3'],
            name: 'search_screen_text',
            obs_len: 18,
            obs_summary: '找到三帧商品证据'
          }
        ],
        phase: 'tool_obs'
      },
      task_id: 'qry_demo'
    })

    const failedChildTool = entry(4, 'SearchWorker', 'tool_error', {
      event: {
        args: { query: '补充搜索' },
        channel: 'search',
        error: 'search timeout',
        tool_name: 'text_search'
      },
      task_id: 'qry_demo'
    })

    const completed = entry(5, 'QueryWorker', 'complete', {
      answer_preview: '已完成回答',
      elapsed_sec: 1.25,
      task_id: 'qry_demo'
    })

    render(
      <QueryWorkerTrajectoryPanel
        entries={[completed, failedChildTool, recallResult, queryStarted(), toolStart]}
      />
    )

    const panel = screen.getByTestId('query-worker-trajectory-panel')
    const task = panel.querySelector('[data-query-worker-task="qry_demo"]')

    expect(task).not.toBeNull()
    expect(within(panel).getByText('QueryWorker 完整轨迹')).toBeTruthy()
    expect(within(panel).getByText('提问时刻冻结输入帧（QueryWorker 实际看到）')).toBeTruthy()
    expect(within(panel).getAllByRole('img')).toHaveLength(3)
    expect(within(panel).getByRole('img', { name: 'ask-1' }).getAttribute('src')).toBe(
      'data:image/jpeg;base64,frame-one'
    )
    expect(within(panel).getByText(/00:19\.5/)).toBeTruthy()
    expect(within(panel).getAllByText(/camera/).length).toBeGreaterThan(0)
    expect(within(panel).getAllByText('实际调用').length).toBeGreaterThan(0)
    expect(within(panel).getByText('工具返回')).toBeTruthy()
    expect(within(panel).getByText('找到三帧商品证据')).toBeTruthy()
    expect(within(panel).getAllByText(/search timeout/).length).toBeGreaterThan(0)
    expect(panel.textContent).toContain('***')
    expect(panel.textContent).not.toContain('raw-secret-value')

    const orderedSeq = [...task!.querySelectorAll('[data-query-worker-seq]')].map(node =>
      Number(node.getAttribute('data-query-worker-seq'))
    )

    expect(orderedSeq).toEqual([1, 2, 3, 4, 5])
  })

  it('renders nothing for non-QueryWorker activity instead of mislabeling it', () => {
    const { container } = render(
      <QueryWorkerTrajectoryPanel
        entries={[
          entry(1, 'MemoryWriter', 'writer_start', { n_frames: 3 }),
          entry(2, 'MainTool:web_search', 'tool.complete', {
            name: 'web_search',
            result: { ok: true }
          }, 'tool.complete')
        ]}
      />
    )

    expect(screen.queryByTestId('query-worker-trajectory-panel')).toBeNull()
    expect(container.innerHTML).toBe('')
  })

  it('shows structured router decisions without rendering hidden thought or router_thinking', () => {
    const structuredDecision = entry(2, 'QueryRouter', 'router_react', {
      event: {
        decision_summary: '需要查询画面文字',
        thought: 'PRIVATE ROUTER THOUGHT MUST NEVER RENDER',
        tool_calls: [{ args: { query: '商品名称' }, name: 'search_screen_text' }],
        type: 'router_react'
      },
      parent_user_message_id: 'turn-A',
      task_id: 'qry_demo'
    })

    const rawThinking = entry(3, 'QueryRouter', 'router_thinking', {
      event: { thought: 'PRIVATE TOKEN STREAM MUST NEVER RENDER' },
      parent_user_message_id: 'turn-A',
      task_id: 'qry_demo'
    })

    render(<QueryWorkerTrajectoryPanel entries={[rawThinking, structuredDecision, queryStarted(1)]} />)

    const panel = screen.getByTestId('query-worker-trajectory-panel')

    expect(within(panel).getByText('Completed a planning round · Search 1')).toBeTruthy()
    expect(within(panel).getByText('需要查询画面文字')).toBeTruthy()
    expect(panel.textContent).toContain('<hidden internal reasoning>')
    expect(panel.textContent).not.toContain('PRIVATE ROUTER THOUGHT MUST NEVER RENDER')
    expect(panel.textContent).not.toContain('PRIVATE TOKEN STREAM MUST NEVER RENDER')
    expect(panel.querySelector('[data-query-worker-seq="3"]')).toBeNull()
  })
})
