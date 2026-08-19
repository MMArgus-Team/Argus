import { describe, expect, it } from 'vitest'

import { activeTimelineIndex, deriveTimelineEntries, isInjectedNotification, timelinePreview } from './thread-timeline-data'

describe('timelinePreview', () => {
  it('collapses whitespace to a single line', () => {
    expect(timelinePreview('hello\n\n  world\tagain')).toBe('hello world again')
  })

  it('truncates with an ellipsis past the limit', () => {
    const out = timelinePreview('abcdefghij', 5)
    expect(out).toBe('abcd…')
    expect(out.length).toBe(5)
  })
})

describe('deriveTimelineEntries', () => {
  it('keeps non-empty user prompts in order', () => {
    expect(
      deriveTimelineEntries([
        { id: 'u1', role: 'user', text: 'first' },
        { id: 'a1', role: 'assistant', text: 'answer' },
        { id: 'u2', role: 'user', text: '  second  ' }
      ])
    ).toEqual([
      { id: 'u1', preview: 'first' },
      { id: 'u2', preview: 'second' }
    ])
  })

  it('drops blanks and background-process notifications', () => {
    expect(
      deriveTimelineEntries([
        { id: 'u1', role: 'user', text: '   ' },
        { id: 'u2', role: 'user', text: '[IMPORTANT: Background process 123 finished]' },
        { id: 'u3', role: 'user', text: 'real prompt' }
      ]).map(e => e.id)
    ).toEqual(['u3'])
  })

  it('drops async-delegation completions too', () => {
    // The bracket closes on line 1 and the report continues below it, so an
    // end-anchored pattern never matched these and they leaked into the
    // timeline (and rendered as human user bubbles) as if typed.
    const batch = [
      '[ASYNC DELEGATION BATCH COMPLETE — deleg_43a80d80]',
      'A background fan-out of 1 subagent(s) you dispatched earlier has finished.',
      '--- ✗ TASK 1/1: 深度调研  (status=failed) ---'
    ].join('\n')
    const single = '[ASYNC DELEGATION COMPLETE — deleg_x]\nOriginal goal: something'

    expect(
      deriveTimelineEntries([
        { id: 'u1', role: 'user', text: batch },
        { id: 'u2', role: 'user', text: single },
        { id: 'u3', role: 'user', text: 'real prompt' }
      ]).map(e => e.id)
    ).toEqual(['u3'])
  })
})

describe('isInjectedNotification', () => {
  it('matches both injected shapes', () => {
    expect(isInjectedNotification(
      '[IMPORTANT: Background process 9 completed normally (exit code 0).\nOutput:\nhi]'
    )).toBe(true)
    expect(isInjectedNotification(
      '[ASYNC DELEGATION BATCH COMPLETE — d1]\nbody'
    )).toBe(true)
    expect(isInjectedNotification('[ASYNC DELEGATION COMPLETE — d1]\nbody')).toBe(true)
  })

  it('leaves human prompts alone', () => {
    expect(isInjectedNotification('帮我设置一个深度调研的。')).toBe(false)
    // A human quoting the header is still a human prompt only if it is not the
    // literal injected shape; the leading-bracket form is reserved.
    expect(isInjectedNotification('what does [ASYNC DELEGATION COMPLETE] mean?')).toBe(false)
    expect(isInjectedNotification('[IMPORTANT: read the docs]')).toBe(false)
  })
})

describe('activeTimelineIndex', () => {
  it('returns the last prompt scrolled to or above the top edge', () => {
    expect(activeTimelineIndex([-400, -10, 320])).toBe(1)
  })

  it('falls back to the first rendered entry', () => {
    expect(activeTimelineIndex([null, 120, 480])).toBe(1)
    expect(activeTimelineIndex([null, null])).toBe(0)
  })
})
