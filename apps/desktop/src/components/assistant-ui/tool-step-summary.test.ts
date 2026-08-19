/**
 * Tool-batch summary: "Read 2 files · Ran 1 command" instead of "3 tool calls".
 *
 * The bare count is content-free, and with thinking mode OFF it was the only
 * label a finished turn carried — the tool-history header said nothing about
 * what the turn actually did. Ported from web's `summarizeStep`.
 */
import { describe, expect, it } from 'vitest'

import { en } from '@/i18n/en'
import { zh } from '@/i18n/zh'

import { summarizeToolSteps } from './tool-fallback-model'

// Identity-ish translator: assert on key + count, not on English wording.
const copy = {
  stepBrowse: (n: number) => `browse(${n})`,
  stepCall: (n: number) => `call(${n})`,
  stepEdit: (n: number) => `edit(${n})`,
  stepFailed: (n: number) => `failed(${n})`,
  stepLook: (n: number) => `look(${n})`,
  stepRead: (n: number) => `read(${n})`,
  stepRun: (n: number) => `run(${n})`,
  stepSearch: (n: number) => `search(${n})`
}

const tool = (toolName: string, extra: Record<string, unknown> = {}) => ({
  type: 'tool-call',
  toolName,
  ...extra
})

describe('summarizeToolSteps', () => {
  it('groups calls by tool family', () => {
    expect(summarizeToolSteps([tool('read_file'), tool('read_file'), tool('terminal')], copy)).toBe(
      'read(2) · run(1)'
    )
  })

  it('uses a fixed order regardless of call order', () => {
    const a = summarizeToolSteps([tool('terminal'), tool('read_file')], copy)
    const b = summarizeToolSteps([tool('read_file'), tool('terminal')], copy)

    expect(a).toBe(b)
    expect(a).toBe('read(1) · run(1)')
  })

  it('recognises the browser family', () => {
    expect(summarizeToolSteps([tool('browser_navigate'), tool('browser_navigate')], copy)).toBe(
      'browse(2)'
    )
  })

  it('falls back to a generic count for unrecognised tools', () => {
    expect(summarizeToolSteps([tool('some_exotic_skill')], copy)).toBe('call(1)')
  })

  it('calls out failures so a collapsed header cannot look all-clear', () => {
    expect(summarizeToolSteps([tool('read_file'), tool('terminal', { isError: true })], copy)).toBe(
      'read(1) · run(1) · failed(1)'
    )
  })

  it('returns empty when there are no tool parts, so the caller can fall back', () => {
    expect(summarizeToolSteps([{ type: 'text' }, { type: 'reasoning' }], copy)).toBe('')
    expect(summarizeToolSteps([], copy)).toBe('')
  })

  it('ships every step verb in both locales', () => {
    // These are member-accessed (not string keys), so tsc already guards them —
    // this pins that neither locale silently drops one at runtime.
    for (const [name, catalog] of [
      ['en', en],
      ['zh', zh]
    ] as const) {
      for (const key of [
        'stepRead',
        'stepEdit',
        'stepRun',
        'stepSearch',
        'stepBrowse',
        'stepLook',
        'stepCall',
        'stepFailed'
      ] as const) {
        expect(typeof catalog.assistant.thread[key], `${key} in ${name}`).toBe('function')
      }
    }
  })
})
