import { describe, expect, it } from 'vitest'

import { selectMissingRequired, type ReadinessReport } from './desktop-mm-readiness-card'

function report(partial: Partial<ReadinessReport>): ReadinessReport {
  return { ready: false, capabilities: [], ...partial }
}

describe('selectMissingRequired (desktop)', () => {
  it('returns [] for a null report (fail safe)', () => {
    expect(selectMissingRequired(null)).toEqual([])
  })

  it('returns [] when ready', () => {
    expect(
      selectMissingRequired(report({ ready: true, capabilities: [
        { key: 'memory', label: '记忆', status: 'broken', required: true, reason: '', fix: '' }
      ] }))
    ).toEqual([])
  })

  it('returns only REQUIRED non-ok caps', () => {
    const out = selectMissingRequired(report({ capabilities: [
      { key: 'memory', label: '记忆', status: 'broken', required: true, reason: '', fix: '' },
      { key: 'voice', label: '语音', status: 'missing', required: false, reason: '', fix: '' }
    ] }))
    expect(out.map(c => c.key)).toEqual(['memory'])
  })

  it('includes both missing and broken required caps', () => {
    const out = selectMissingRequired(report({ capabilities: [
      { key: 'a', label: 'A', status: 'missing', required: true, reason: '', fix: '' },
      { key: 'b', label: 'B', status: 'broken', required: true, reason: '', fix: '' },
      { key: 'c', label: 'C', status: 'ok', required: true, reason: '', fix: '' }
    ] }))
    expect(out.map(c => c.key)).toEqual(['a', 'b'])
  })
})
