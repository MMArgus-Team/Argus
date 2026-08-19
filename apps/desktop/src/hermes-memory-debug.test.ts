import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  getMultimodalMemoryDebugFrame,
  getMultimodalMemoryDebugSession,
  getMultimodalMemoryDebugSessions,
  getMultimodalMemoryDebugTrace,
  searchMultimodalMemoryDebug,
  setApiRequestProfile
} from './hermes'

describe('multimodal memory debug REST helpers', () => {
  let api: ReturnType<typeof vi.fn>

  beforeEach(() => {
    api = vi.fn().mockResolvedValue({})
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { api }
    })
    setApiRequestProfile('debug-profile')
  })

  afterEach(() => {
    setApiRequestProfile(null)
    vi.restoreAllMocks()
    Reflect.deleteProperty(window, 'hermesDesktop')
  })

  it('lists databases on the active profile with the debug timeout and limit', async () => {
    await getMultimodalMemoryDebugSessions(80)

    expect(api).toHaveBeenCalledWith({
      path: '/api/multimodal/memory/debug/sessions?limit=80',
      profile: 'debug-profile',
      timeoutMs: 60_000
    })
  })

  it('encodes a database path segment and session overview parameters', async () => {
    await getMultimodalMemoryDebugSession('memory/foo?#.db', {
      session_id: 'session /?#&',
      limit: 0
    })

    expect(api).toHaveBeenCalledWith({
      path: '/api/multimodal/memory/debug/session/memory%2Ffoo%3F%23.db' + '?session_id=session+%2F%3F%23%26&limit=0',
      profile: 'debug-profile',
      timeoutMs: 60_000
    })
  })

  it('encodes both database and frame path segments', async () => {
    await getMultimodalMemoryDebugFrame('memory/a b.db', 'frame/1?#')

    expect(api).toHaveBeenCalledWith({
      path: '/api/multimodal/memory/debug/session/memory%2Fa%20b.db/frame/frame%2F1%3F%23',
      profile: 'debug-profile',
      timeoutMs: 60_000
    })
  })

  it('encodes search text and forwards every search filter', async () => {
    await searchMultimodalMemoryDebug('phone & keys/?# 中文', {
      scope: 'today',
      session: 'session /?#&',
      limit: 25
    })

    expect(api).toHaveBeenCalledWith({
      path:
        '/api/multimodal/memory/debug/search?' +
        'query=phone+%26+keys%2F%3F%23+%E4%B8%AD%E6%96%87' +
        '&scope=today&session=session+%2F%3F%23%26&limit=25',
      profile: 'debug-profile',
      timeoutMs: 60_000
    })
  })

  it('forwards encoded trace parameters on the active profile', async () => {
    await getMultimodalMemoryDebugTrace({
      session_id: 'session /?#&',
      db: 'memory /?#&.db',
      limit: 40
    })

    expect(api).toHaveBeenCalledWith({
      path:
        '/api/multimodal/memory/debug/trace?' + 'session_id=session+%2F%3F%23%26&db=memory+%2F%3F%23%26.db&limit=40',
      profile: 'debug-profile',
      timeoutMs: 60_000
    })
  })
})
