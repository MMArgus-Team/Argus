import { describe, expect, it } from 'vitest'

import { groupTrajectoryByQuestion, type MmTrajectoryEntry } from './trajectory-grouping'

function entry(
  seq: number,
  worker: string,
  phase: string,
  payload: Record<string, unknown> = {},
  event = 'multimodal.trajectory'
): MmTrajectoryEntry {
  return {
    event,
    id: `tr-${seq}`,
    payload,
    phase,
    seq,
    ts: 1_700_000_000 + seq,
    worker
  }
}

describe('groupTrajectoryByQuestion', () => {
  it('groups one question into separate worker traces using authoritative parent ids', () => {
    const grouped = groupTrajectoryByQuestion([
      entry(1, 'MainScheduler', 'prompt_started', {
        client_request_id: 'turn-A',
        origin: 'user',
        text: '这张图里的第二个商品多少钱？'
      }),
      entry(2, 'QueryWorker', 'started', {
        parent_user_message_id: 'turn-A',
        query: '这张图里的第二个商品多少钱？',
        task_id: 'qry-A'
      }),
      entry(3, 'OCR', 'ocr_evidence', {
        parent_user_message_id: 'turn-A',
        task_id: 'qry-A'
      }),
      entry(4, 'SearchWorker', 'search_done', {
        parent_user_message_id: 'turn-A',
        task_id: 'qry-A'
      })
    ])

    expect(grouped.questions).toHaveLength(1)
    expect(grouped.questions[0].id).toBe('turn-A')
    expect(grouped.questions[0].label).toBe('这张图里的第二个商品多少钱？')
    expect(grouped.questions[0].workers.map(group => group.worker)).toEqual([
      'MainScheduler',
      'QueryWorker',
      'OCR',
      'SearchWorker'
    ])
    expect(grouped.background).toEqual([])
  })

  it('uses a later handoff to backfill earlier tool and task rows without relying on arrival order', () => {
    const grouped = groupTrajectoryByQuestion([
      entry(10, 'QueryWorker', 'tool_start', { tool_id: 'call-A' }, 'tool.start'),
      entry(11, 'QueryWorker', 'started', { task_id: 'qry-A' }),
      entry(
        12,
        'QueryWorker',
        'tool_complete',
        {
          request_id: 'turn-A',
          result: {
            original_user_text: '识别这个银行标志',
            parent_user_message_id: 'turn-A',
            task_id: 'qry-A'
          },
          task_id: 'qry-A',
          tool_id: 'call-A'
        },
        'tool.complete'
      )
    ])

    expect(grouped.questions).toHaveLength(1)
    expect(grouped.questions[0].entries.map(item => item.seq)).toEqual([10, 11, 12])
    expect(grouped.questions[0].label).toBe('识别这个银行标志')
    expect(grouped.background).toEqual([])
  })

  it('keeps interleaved concurrent questions isolated by their task aliases', () => {
    const grouped = groupTrajectoryByQuestion([
      entry(1, 'QueryWorker', 'started', {
        parent_user_message_id: 'turn-A',
        query: '问题 A',
        task_id: 'qry-A'
      }),
      entry(2, 'QueryWorker', 'started', {
        parent_user_message_id: 'turn-B',
        query: '问题 B',
        task_id: 'qry-B'
      }),
      entry(3, 'RecallWorker', 'recall_done', { task_id: 'qry-A' }),
      entry(4, 'SearchWorker', 'search_done', { task_id: 'qry-B' }),
      entry(5, 'SearchWorker', 'search_done', { task_id: 'qry-A' }),
      entry(6, 'RecallWorker', 'recall_done', { task_id: 'qry-B' })
    ])

    expect(grouped.questions.map(group => [group.id, group.entries.map(item => item.seq)])).toEqual([
      ['turn-A', [1, 3, 5]],
      ['turn-B', [2, 4, 6]]
    ])
  })

  it('keeps correlated non-QueryWorker tool events under their real worker labels and in sequence order', () => {
    const grouped = groupTrajectoryByQuestion([
      entry(
        14,
        'MainTool:search_screen_text',
        'tool_complete',
        {
          result: { ok: true },
          task_id: 'qry-A',
          tool_id: 'call-search'
        },
        'tool.complete'
      ),
      entry(10, 'QueryWorker', 'started', {
        parent_user_message_id: 'turn-A',
        query: '看看这三帧里的商品信息',
        task_id: 'qry-A'
      }),
      entry(
        12,
        'MainTool:search_screen_text',
        'tool_start',
        {
          args: { query: '商品名称' },
          task_id: 'qry-A',
          tool_id: 'call-search'
        },
        'tool.start'
      )
    ])

    expect(grouped.questions).toHaveLength(1)
    expect(grouped.questions[0].entries.map(item => item.seq)).toEqual([10, 12, 14])
    expect(grouped.questions[0].workers.map(worker => worker.worker)).toEqual([
      'QueryWorker',
      'MainTool:search_screen_text'
    ])
    expect(grouped.questions[0].workers[1].entries.map(item => item.event)).toEqual([
      'tool.start',
      'tool.complete'
    ])
  })

  it('maps watcher and monitor task ids only when their handoff carries a parent question', () => {
    const grouped = groupTrajectoryByQuestion([
      entry(1, 'WatcherRouter', 'waiting', { request_id: 'watch-1' }, 'multimodal.bg'),
      entry(2, 'MonitorWorker', 'eval_start', { monitor_id: 'mon-1' }),
      entry(
        3,
        'WatcherRouter',
        'tool_complete',
        {
          result: {
            parent_user_message_id: 'turn-A',
            request_id: 'watch-1',
            task_id: 'watch-1',
            task_instruction: '持续观察编译结果'
          },
          tool_id: 'call-watch'
        },
        'tool.complete'
      ),
      entry(
        4,
        'MonitorWorker',
        'tool_complete',
        {
          result: {
            brief: '看到手机提醒我',
            monitor_id: 'mon-1',
            parent_user_message_id: 'turn-B',
            task_id: 'mon-1'
          },
          tool_id: 'call-monitor'
        },
        'tool.complete'
      ),
      entry(5, 'WatcherRouter', 'waiting', { request_id: 'unknown-watch' }, 'multimodal.bg'),
      entry(6, 'MonitorWorker', 'eval_start', { monitor_id: 'unknown-monitor' })
    ])

    expect(grouped.questions.map(group => [group.id, group.entries.map(item => item.seq)])).toEqual([
      ['turn-A', [1, 3]],
      ['turn-B', [2, 4]]
    ])
    expect(grouped.background.flatMap(group => group.entries.map(item => item.seq))).toEqual([5, 6])
  })

  it('never treats a bare request_id as a chat question id', () => {
    const grouped = groupTrajectoryByQuestion([
      entry(
        1,
        'WatcherRouter',
        'segment_start',
        {
          brief: '看看后续发生了什么',
          request_id: 'watcher-task-only'
        },
        'multimodal.bg'
      )
    ])

    expect(grouped.questions).toEqual([])
    expect(grouped.background[0].entries[0].payload.request_id).toBe('watcher-task-only')
  })

  it('links request-only rows after a foreground turn explicitly claims the same id', () => {
    const grouped = groupTrajectoryByQuestion([
      entry(1, 'VoiceAgent', 'accepted', { request_id: 'turn-voice' }, 'multimodal.voice'),
      entry(2, 'MainScheduler', 'prompt_started', {
        client_request_id: 'turn-voice',
        origin: 'voice_asr',
        text: '描述当前画面'
      })
    ])

    expect(grouped.questions).toHaveLength(1)
    expect(grouped.questions[0].id).toBe('turn-voice')
    expect(grouped.questions[0].entries.map(item => item.seq)).toEqual([1, 2])
    expect(grouped.background).toEqual([])
  })

  it('recognizes only final ASR request ids as voice questions and links their worker task', () => {
    const grouped = groupTrajectoryByQuestion([
      entry(
        1,
        'VoiceAgent',
        'multimodal.asr_partial',
        { request_id: 'voice-partial', text: '描述' },
        'multimodal.asr_partial'
      ),
      entry(
        2,
        'VoiceAgent',
        'multimodal.asr_final',
        { request_id: 'voice-final', text: '描述当前画面' },
        'multimodal.asr_final'
      ),
      entry(3, 'QueryWorker', 'started', {
        parent_user_message_id: 'voice-final',
        query: '描述当前画面',
        task_id: 'qry-voice'
      }),
      entry(4, 'RecallWorker', 'recall_done', { task_id: 'qry-voice' })
    ])

    expect(grouped.questions).toHaveLength(1)
    expect(grouped.questions[0].id).toBe('voice-final')
    expect(grouped.questions[0].label).toBe('描述当前画面')
    expect(grouped.questions[0].entries.map(item => item.seq)).toEqual([2, 3, 4])
    expect(grouped.background.flatMap(group => group.entries.map(item => item.seq))).toEqual([1])
  })

  it('keeps continuous MemoryWriter, OCR, reviewer, anchor, and diagnostics in an honest background section', () => {
    const grouped = groupTrajectoryByQuestion([
      entry(1, 'MainScheduler', 'prompt_started', {
        client_request_id: 'turn-A',
        origin: 'user',
        text: '描述当前画面'
      }),
      entry(2, 'MemoryWriter', 'writer_start', { n_frames: 3 }),
      entry(3, 'OCRWorker', 'tick', { written: 1 }),
      entry(4, 'MemoryEntityReviewer', 'review_done', { revisions: 1 }),
      entry(5, 'MainVision', 'multimodal.anchor', { frames: [] }, 'multimodal.anchor'),
      entry(6, 'LLM', 'SEND', { model: 'test' }, 'multimodal.diag')
    ])

    expect(grouped.questions[0].entries.map(item => item.seq)).toEqual([1])
    expect(grouped.background.map(group => group.worker)).toEqual([
      'MemoryWriter',
      'OCRWorker',
      'MemoryEntityReviewer',
      'MainVision',
      'LLM'
    ])
  })

  it('fails closed when one task alias is claimed by two different questions', () => {
    const grouped = groupTrajectoryByQuestion([
      entry(1, 'QueryWorker', 'started', {
        parent_user_message_id: 'turn-A',
        task_id: 'reused-task'
      }),
      entry(2, 'QueryWorker', 'started', {
        parent_user_message_id: 'turn-B',
        task_id: 'reused-task'
      }),
      entry(3, 'QueryWorker', 'progress', { task_id: 'reused-task' })
    ])

    expect(grouped.questions.map(group => group.id)).toEqual(['turn-A', 'turn-B'])
    expect(grouped.background.flatMap(group => group.entries.map(item => item.seq))).toEqual([3])
  })
})
