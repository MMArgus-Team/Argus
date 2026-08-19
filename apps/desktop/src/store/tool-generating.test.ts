import { beforeEach, describe, expect, it } from 'vitest'

import { $activeSessionId } from './session'
import {
  $generatingToolBySession,
  $generatingToolName,
  clearSessionGeneratingTool,
  resetGeneratingTools,
  setSessionGeneratingTool
} from './tool-generating'

beforeEach(() => {
  resetGeneratingTools()
  $activeSessionId.set(null)
})

describe('generating-tool store', () => {
  it('exposes the active session’s tool name only', () => {
    setSessionGeneratingTool('s1', 'terminal')
    setSessionGeneratingTool('s2', 'read_file')

    $activeSessionId.set('s1')
    expect($generatingToolName.get()).toBe('terminal')

    // A background session writing its own tool must not leak into the view.
    $activeSessionId.set('s2')
    expect($generatingToolName.get()).toBe('read_file')

    $activeSessionId.set('s3')
    expect($generatingToolName.get()).toBe('')
  })

  it('clears on an empty or whitespace name', () => {
    $activeSessionId.set('s1')
    setSessionGeneratingTool('s1', 'terminal')

    // The gateway payload's `name` is optional, so a blank must not pin a
    // permanent "Preparing " line with no tool in it.
    setSessionGeneratingTool('s1', '   ')

    expect($generatingToolName.get()).toBe('')
    expect('s1' in $generatingToolBySession.get()).toBe(false)
  })

  it('keeps the same object identity when the name is unchanged', () => {
    setSessionGeneratingTool('s1', 'terminal')
    const first = $generatingToolBySession.get()

    setSessionGeneratingTool('s1', 'terminal')

    // Repeated tool.generating events for one call are common; re-setting the
    // atom would re-render every subscriber on each one.
    expect($generatingToolBySession.get()).toBe(first)
  })

  it('clearing is idempotent and scoped to one session', () => {
    setSessionGeneratingTool('s1', 'terminal')
    setSessionGeneratingTool('s2', 'grep')

    clearSessionGeneratingTool('s1')
    clearSessionGeneratingTool('s1')

    $activeSessionId.set('s2')
    expect($generatingToolName.get()).toBe('grep')
  })
})
