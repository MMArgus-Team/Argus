import { beforeEach, describe, expect, it } from 'vitest'

import { $mmBgItems, applyBgEvent, setWatcherFinal } from './multimodal-deep'

describe('terminal watcher progress', () => {
  beforeEach(() => {
    $mmBgItems.set([])
  })

  it('keeps a final report terminal when queued waiting and segment events arrive late', () => {
    applyBgEvent({
      have: 2,
      need: 3,
      request_id: 'watch-1',
      seg: 2,
      type: 'waiting'
    })
    setWatcherFinal('watch-1', 'authoritative final report')
    const terminal = $mmBgItems.get()

    expect(terminal[0]).toMatchObject({
      done: true,
      finalReport: 'authoritative final report',
      waiting: null
    })

    expect(applyBgEvent({ have: 3, need: 3, request_id: 'watch-1', type: 'waiting' })).toBe('')
    expect(applyBgEvent({ request_id: 'watch-1', type: 'batch_ready' })).toBe('')
    expect(applyBgEvent({ request_id: 'watch-1', seg: 3, type: 'segment_start' })).toBe('')
    expect($mmBgItems.get()).toEqual(terminal)
  })
})
