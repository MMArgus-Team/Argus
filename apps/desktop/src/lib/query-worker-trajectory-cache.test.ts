import { describe, expect, it } from 'vitest'

import { compactQueryWorkerTrajectoryImages } from './query-worker-trajectory-cache'

function row(taskId: string, seq: number, jpegChars: number, phase = 'recall_done') {
  return {
    id: `${taskId}-${phase}`,
    payload: {
      frames: [{ frame_id: `${taskId}-frame`, jpeg_b64: 'x'.repeat(jpegChars), ts: seq }],
      task_id: taskId
    },
    phase,
    seq
  }
}

describe('compactQueryWorkerTrajectoryImages', () => {
  it('bounds live image bytes while preserving every row and frame metadata', () => {
    const rows = Array.from({ length: 5 }, (_, index) => row(`qry_${index + 1}`, index + 1, 1_100_000))
    const compacted = compactQueryWorkerTrajectoryImages(rows)

    expect(compacted).toHaveLength(5)
    expect(compacted.map(entry => entry.payload.frames[0].frame_id)).toEqual([
      'qry_1-frame',
      'qry_2-frame',
      'qry_3-frame',
      'qry_4-frame',
      'qry_5-frame'
    ])
    expect(compacted.map(entry => 'jpeg_b64' in entry.payload.frames[0])).toEqual([
      false,
      false,
      true,
      true,
      true
    ])
  })

  it('always protects the newest task exact started snapshot', () => {
    const old = row('qry_old', 1, 100)
    const newest = row('qry_new', 2, 4_500_000, 'started')
    const compacted = compactQueryWorkerTrajectoryImages([old, newest])

    expect(compacted[0].payload.frames[0]).toEqual({ frame_id: 'qry_old-frame', ts: 1 })
    expect(compacted[1].payload.frames[0].jpeg_b64).toHaveLength(4_500_000)
  })
})
