import { cleanup, render, screen, within } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import {
  buildQueryWorkerTimelines,
  queryWorkerStepFromTrajectory
} from './query-worker-trajectory'
import { QueryWorkerTrajectoryPanel } from './query-worker-trajectory-panel'
import type { MmTrajectoryEntry } from './trajectory-grouping'

afterEach(cleanup)

function trajectory(
  phase: string,
  event: Record<string, unknown> = {},
  extras: Record<string, unknown> = {},
  seq = 1,
  id = `event-${phase}-${seq}`
): MmTrajectoryEntry {
  return {
    event: 'multimodal.trajectory',
    id,
    payload: {
      event,
      parent_user_message_id: 'turn-test',
      task_id: 'qry_test',
      ...extras
    },
    phase,
    seq,
    ts: 10 + seq,
    worker: phase.includes('search') ? 'SearchWorker' : 'QueryWorker'
  }
}

describe('Desktop QueryWorker Web parity', () => {
  it('keeps an empty frozen snapshot explicit without inventing thumbnails', () => {
    const started = trajectory('started', {}, { ask_ts: 21.5, frames: [], n_frames: 0 })
    const step = queryWorkerStepFromTrajectory(started)

    expect(step?.title).toContain('frozen input frames 0')
    expect(step?.frames).toEqual([])

    render(<QueryWorkerTrajectoryPanel entries={[started]} />)

    expect(screen.getByText(/冻结输入帧 0/)).toBeTruthy()
    expect(screen.queryByRole('img')).toBeNull()
  })

  it('accepts OCR evidence and records aliases while rendering untrusted text as inert text', () => {
    const evidence = trajectory('ocr_evidence', {}, {
      elapsed_sec: 0.317,
      evidence: [{
        evidence_source: 'synchronous_camera_ocr',
        frame_ts: 19.5,
        raw_text: '东方树叶',
        source_type: 'camera'
      }],
      evidence_state: 'available',
      record_count: 1
    }, 1, 'ocr-evidence')

    const recordsAlias = trajectory('ocr_evidence', {}, {
      evidence_state: 'available',
      record_count: 1,
      records: [{
        app: "<script>alert('app')</script>",
        frame_ts: 20.5,
        raw_text: "<img src=x onerror=alert('ocr')>500 mL",
        source_type: 'screen'
      }]
    }, 2, 'ocr-records')

    expect(queryWorkerStepFromTrajectory(evidence)).toMatchObject({
      ocrElapsedSec: 0.317,
      ocrRecordCount: 1,
      ocrState: 'available'
    })
    expect(queryWorkerStepFromTrajectory(recordsAlias)?.ocrRecords).toHaveLength(1)

    const { container } = render(<QueryWorkerTrajectoryPanel entries={[evidence, recordsAlias]} />)

    expect(screen.getByText('东方树叶')).toBeTruthy()
    expect(screen.getAllByText(/<img src=x onerror=alert\('ocr'\)>500 mL/).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/<script>alert\('app'\)<\/script>/).length).toBeGreaterThan(0)
    expect(container.querySelector('script')).toBeNull()
    expect(container.querySelector('img')).toBeNull()
  })

  it('maps the exact Recall and Search plan and explicitly records a no-tool router round', () => {
    const planned = queryWorkerStepFromTrajectory(trajectory('router_react', {
      recall_tasks: [{ brief: '金发男子的攀岩鞋品牌' }],
      source_clip: { n_frames: 12, t_end: 148, t_start: 142 },
      tool_calls: [{
        anchor: 'current',
        args: { query: 'SCARPA 攀岩鞋品牌' },
        name: 'text_search'
      }],
      type: 'router_react'
    }))

    const noTools = queryWorkerStepFromTrajectory(trajectory('router_react', {
      recall_tasks: [],
      tool_calls: [],
      type: 'router_react'
    }, {}, 2))

    expect(planned?.callState).toBe('planned')
    expect(planned?.toolCalls).toEqual([
      { anchor: 'current', args: { query: 'SCARPA 攀岩鞋品牌' }, name: 'text_search' },
      { args: { brief: '金发男子的攀岩鞋品牌' }, name: 'recall_memory' }
    ])
    expect(planned?.metrics).toContain('source 142.0–148.0s · 12 frames')
    expect(noTools?.title).toContain('no Recall / Search this round')
    expect(noTools?.toolCalls).toEqual([])
  })

  it('shows Search call args, returned preview, safe URLs, cache state and source timing', () => {
    const dispatch = trajectory('bg_progress', {
      anchor: 'current',
      anchor_ts: 148,
      args: { query: 'SCARPA 攀岩鞋品牌' },
      channel: 'search',
      task_id: 'r0_s1',
      tool_name: 'text_search',
      type: 'bg_progress'
    }, {}, 1, 'search-dispatch')

    const result = trajectory('search_done', {
      anchor: 'current',
      anchor_ts: 148,
      args: { query: 'SCARPA 攀岩鞋品牌' },
      brief: '原始搜索任务',
      cache_hit: true,
      elapsed_sec: 0.35,
      findings_len: 18,
      findings_preview: 'SCARPA 是意大利户外鞋品牌。',
      source_clip: { n_frames: 12, t_end: 148, t_start: 142 },
      source_urls: [
        'https://example.test/scarpa',
        'https://alice:supersecret@example.test/private',
        'javascript:alert(1)'
      ],
      tool_name: 'text_search',
      type: 'search_done'
    }, {}, 2, 'search-result')

    const dispatchStep = queryWorkerStepFromTrajectory(dispatch)
    const resultStep = queryWorkerStepFromTrajectory(result)

    expect(dispatchStep).toMatchObject({ callState: 'called', taskRef: 'r0_s1' })
    expect(dispatchStep?.toolCalls[0]).toMatchObject({
      anchor: 'current',
      anchorTs: 148,
      args: { query: 'SCARPA 攀岩鞋品牌' },
      name: 'text_search'
    })
    expect(resultStep?.detail).toBe('SCARPA 是意大利户外鞋品牌。')
    expect(resultStep?.detail).not.toBe('原始搜索任务')
    expect(resultStep?.metrics).toEqual(expect.arrayContaining([
      'cache hit',
      '0.35s',
      'source 142.0–148.0s · 12 frames'
    ]))
    expect(resultStep?.toolResults[0]).toMatchObject({
      cacheHit: true,
      elapsedSec: 0.35,
      obsLength: 18,
      sourceUrls: [
        'https://example.test/scarpa',
        'https://alice:supersecret@example.test/private',
        'javascript:alert(1)'
      ],
      summary: 'SCARPA 是意大利户外鞋品牌。'
    })

    render(<QueryWorkerTrajectoryPanel entries={[dispatch, result]} />)

    const panel = screen.getByTestId('query-worker-trajectory-panel')
    const safeLink = within(panel).getByRole('link', { name: 'https://example.test/scarpa' })

    expect(safeLink.getAttribute('href')).toBe('https://example.test/scarpa')
    expect(within(panel).getByRole('link', { name: 'https://alice:***@example.test/private' })).toBeTruthy()
    expect(panel.querySelector('a[href^="javascript:"]')).toBeNull()
    expect(panel.textContent).not.toContain('supersecret')
    expect(within(panel).getByText('cache hit')).toBeTruthy()
  })

  it('preserves Recall frame ids and evidence segments without legacy full observations', () => {
    const entry = trajectory('bg_progress', {
      channel: 'recall',
      new_frame_ids: ['f_84'],
      observations: [{
        args: { query: '探店' },
        elapsed_sec: 0.18,
        evidence_segments: [{
          frame_ids: ['f_84'],
          kind: 'audio',
          preview: '提到联系李维刚',
          t_end: 84.5,
          t_start: 84.5
        }],
        frame_ids: ['f_84'],
        name: 'search_audio',
        obs_full: 'LEGACY FULL OBSERVATION MUST STAY HIDDEN',
        obs_len: 4200,
        obs_summary: '84.5s 提到联系李维刚来探店'
      }],
      parallel_elapsed_sec: 0.42,
      phase: 'tool_obs',
      round: 0
    })

    const step = queryWorkerStepFromTrajectory(entry)

    expect(step?.toolResults[0]).toMatchObject({
      args: { query: '探店' },
      elapsedSec: 0.18,
      evidenceSegments: [{
        frameIds: ['f_84'],
        kind: 'audio',
        preview: '提到联系李维刚',
        tEnd: 84.5,
        tStart: 84.5
      }],
      frameIds: ['f_84'],
      name: 'search_audio',
      obsLength: 4200,
      summary: '84.5s 提到联系李维刚来探店'
    })
    expect(step?.metrics).toEqual(expect.arrayContaining(['并行读取 0.42s', '新证据帧 1']))
    expect(JSON.stringify({
      detail: step?.detail,
      metrics: step?.metrics,
      toolResults: step?.toolResults
    })).not.toContain('LEGACY FULL OBSERVATION')
  })

  it('maps the fast-table Recall result with rows, args, evidence and timing', () => {
    const step = queryWorkerStepFromTrajectory(trajectory('bg_progress', {
      args: { limit: 8, query: '表 2 Argus Score' },
      channel: 'recall',
      elapsed_sec: 0.12,
      evidence_segments: [{
        frame_ids: ['f_table'],
        kind: 'screen',
        t_end: 45,
        t_start: 42
      }],
      findings_len: 120,
      findings_preview: '已命中表 2：Argus | 91.2',
      frame_ids: ['f_table'],
      obs_len: 80,
      obs_summary: '[00:42-00:45] Argus | 91.2',
      phase: 'fast_table',
      task_id: 'r0_r0',
      tool_name: 'search_screen_text'
    }))

    expect(step).toMatchObject({
      status: 'complete',
      taskRef: 'r0_r0'
    })
    expect(step?.title).toContain('search_screen_text')
    expect(step?.detail).toContain('Argus | 91.2')
    expect(step?.metrics).toContain('0.12s')
    expect(step?.toolResults[0]).toMatchObject({
      args: { limit: 8, query: '表 2 Argus Score' },
      frameIds: ['f_table'],
      name: 'search_screen_text',
      summary: '[00:42-00:45] Argus | 91.2'
    })
  })

  it('renders a Recall request error as an error, never as a memory miss', () => {
    const failed = trajectory('bg_progress', {
      channel: 'recall',
      elapsed_sec: 4.2,
      error: 'HTTP 400 Unknown parameter: top_k',
      model: 'GPT-5.6 Luna',
      phase: 'error',
      stage: 'decision'
    })

    const step = queryWorkerStepFromTrajectory(failed)

    expect(step).toMatchObject({
      detail: 'HTTP 400 Unknown parameter: top_k',
      status: 'error'
    })
    expect(step?.title).toContain('Recall request failed')
    expect(step?.title).not.toContain('no reliable clues')
    expect(step?.metrics).toEqual(expect.arrayContaining(['model GPT-5.6 Luna', '4.20s']))

    render(<QueryWorkerTrajectoryPanel entries={[failed]} />)
    expect(screen.getByText('HTTP 400 Unknown parameter: top_k')).toBeTruthy()
  })

  it('keeps child Search/Recall completion non-terminal and only outer completion terminal', () => {
    const childDone = trajectory('search_done', {
      findings_len: 6,
      findings_preview: 'result',
      task_id: 'r0_s0',
      tool_name: 'text_search',
      type: 'search_done'
    }, {}, 1, 'child-done')

    const outerDone = trajectory('complete', {}, {
      answer_preview: 'final answer',
      elapsed_sec: 1.2
    }, 2, 'outer-done')

    expect(queryWorkerStepFromTrajectory(childDone)?.status).toBe('complete')
    expect(buildQueryWorkerTimelines([childDone])?.[0].status).toBe('running')
    expect(buildQueryWorkerTimelines([childDone, outerDone])?.[0].status).toBe('complete')
  })

  it('distinguishes duplicate Recall completion from the two-failure retry ceiling', () => {
    const duplicate = queryWorkerStepFromTrajectory(trajectory('recall_skipped', {
      brief: '视频中店主找谁探店',
      reason: 'duplicate_completed_brief'
    }))

    const exhausted = queryWorkerStepFromTrajectory(trajectory('recall_skipped', {
      brief: '店主找谁探店',
      reason: 'retry_limit_after_two_failures'
    }, {}, 2))

    expect(duplicate?.title).toContain('Skipped duplicate Recall')
    expect(exhausted?.title).toContain('failed twice in a row')
    expect(exhausted?.title).not.toBe(duplicate?.title)
  })

  it('sorts, deduplicates by event id with latest copy winning, and caps each task at 80 steps', () => {
    const rows = Array.from({ length: 90 }, (_, seq) => trajectory(
      'delegate_start',
      {},
      { detail: `step ${seq}` },
      seq,
      `step-${seq}`
    ))

    const latestCopy = {
      ...trajectory('answer_ready', { text_preview: 'latest copy' }, {}, 89, 'step-89'),
      ts: 999
    }

    const [timeline] = buildQueryWorkerTimelines([
      rows[89],
      rows[10],
      ...rows.slice().reverse(),
      latestCopy
    ])

    expect(timeline.steps).toHaveLength(80)
    expect(timeline.steps[0].seq).toBe(10)
    expect(timeline.steps.at(-1)).toMatchObject({
      detail: 'latest copy',
      id: 'step-89',
      seq: 89,
      ts: 999
    })
    expect(new Set(timeline.steps.map(step => step.id)).size).toBe(80)
  })
})
