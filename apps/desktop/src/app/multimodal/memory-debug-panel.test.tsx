import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { MmMemoryDebugSessionSummary } from '@/types/hermes'

const memoryDebugApi = vi.hoisted(() => ({
  frame: vi.fn(),
  search: vi.fn(),
  session: vi.fn(),
  sessions: vi.fn(),
  trace: vi.fn()
}))

const { gatewayAtom } = vi.hoisted(() => {
  let value: unknown = null
  const listeners = new Set<(next: unknown) => void>()

  return {
    gatewayAtom: {
      get: () => value,
      listen(listener: (next: unknown) => void) {
        listeners.add(listener)
        listener(value)

        return () => listeners.delete(listener)
      },
      set(next: unknown) {
        value = next
        listeners.forEach(listener => listener(value))
      },
      subscribe(listener: (next: unknown) => void) {
        return this.listen(listener)
      }
    }
  }
})

vi.mock('@/hermes', () => ({
  getMultimodalMemoryDebugFrame: memoryDebugApi.frame,
  getMultimodalMemoryDebugSession: memoryDebugApi.session,
  getMultimodalMemoryDebugSessions: memoryDebugApi.sessions,
  getMultimodalMemoryDebugTrace: memoryDebugApi.trace,
  searchMultimodalMemoryDebug: memoryDebugApi.search
}))

vi.mock('@/store/gateway', () => ({ $gateway: gatewayAtom }))

import {
  MemoryDebugPanel,
  mergeMemoryDebugTrajectory,
  type MmTrajectoryEntry,
  resolveMemoryDebugDb,
  resolveMemoryDebugSessionIds
} from './memory-debug-panel'

afterEach(cleanup)

beforeEach(() => {
  vi.clearAllMocks()

  if (gatewayAtom.get() !== null) {
    gatewayAtom.set(null)
  }

  memoryDebugApi.sessions.mockResolvedValue({ root: '/memory', sessions: [] })
  memoryDebugApi.session.mockResolvedValue({
    health: {},
    memory: {
      entities: [],
      events: { macro: [], micro: [], super: [] },
      evolution: { entity_states: [], revisions: [] }
    },
    session: { counts: {}, meta: {}, mtime: 0, name: '', path: '', size: 0, stem: '' },
    tables: [],
    timeline: [],
    trace: { logs: [] }
  })
  memoryDebugApi.frame.mockResolvedValue({})
  memoryDebugApi.search.mockResolvedValue({ results: [] })
  memoryDebugApi.trace.mockResolvedValue({ logs: [], messages: [] })
})

function summary(name: string, sessionId: string): MmMemoryDebugSessionSummary {
  return {
    counts: {},
    meta: { hermes_session_id: sessionId },
    mtime: 0,
    name,
    path: `/memory/${name}`,
    size: 0,
    stem: name.replace(/\.sqlite3$/, '')
  }
}

function trajectory(id: string, seq: number, event = 'multimodal.trajectory'): MmTrajectoryEntry {
  return {
    event,
    id,
    payload: {},
    phase: event,
    seq,
    ts: seq,
    worker: 'QueryWorker'
  }
}

function emptyOverview(name = '') {
  return {
    health: {},
    memory: {
      entities: [],
      events: { macro: [], micro: [], super: [] },
      evolution: { entity_states: [], revisions: [] }
    },
    session: { counts: {}, meta: {}, mtime: 0, name, path: '', size: 0, stem: '' },
    tables: [],
    timeline: [] as Array<{ frame_id: string }>,
    trace: { logs: [] }
  }
}

function deferred<T>() {
  let resolve!: (value: T) => void

  const promise = new Promise<T>(done => {
    resolve = done
  })

  return { promise, resolve }
}

describe('resolveMemoryDebugDb', () => {
  const sessions = [
    summary('newest-unrelated.sqlite3', 'other-session'),
    summary('root.sqlite3', 'root-A'),
    summary('durable.sqlite3', 'durable-A')
  ]

  it('uses caller identity order so the exact durable tip wins over its lineage root', () => {
    expect(resolveMemoryDebugDb(sessions, ['durable-A', 'root-A'])).toBe('durable.sqlite3')
  })

  it('uses the exact lineage root when the durable continuation has no DB', () => {
    expect(resolveMemoryDebugDb(sessions, ['missing-continuation', 'root-A'])).toBe('root.sqlite3')
  })

  it('fails closed when neither exact identity matches', () => {
    expect(resolveMemoryDebugDb(sessions, ['missing', 'also-missing'])).toBe('')
    expect(resolveMemoryDebugDb(sessions, ['', ''])).toBe('')
    expect(resolveMemoryDebugDb([], ['durable-A', 'root-A'])).toBe('')
  })

  it('does not use prefix or substring matches or fall back to another session', () => {
    expect(
      resolveMemoryDebugDb(
        [
          summary('child.sqlite3', 'durable-A_child'),
          summary('parent.sqlite3', 'durable'),
          summary('someone-else.sqlite3', 'other-session')
        ],
        ['durable-A', 'root-A']
      )
    ).toBe('')
  })
})

describe('resolveMemoryDebugSessionIds', () => {
  it('resolves a selected compression root to both the current tip and root identities', () => {
    expect(
      resolveMemoryDebugSessionIds('root-A', [{ id: 'tip-A', _lineage_root_id: 'root-A' }], 'live-runtime')
    ).toEqual(['tip-A', 'root-A'])
  })

  it('prefers the continuation even when the root row appears first', () => {
    expect(
      resolveMemoryDebugSessionIds(
        'root-A',
        [
          { id: 'root-A', _lineage_root_id: 'root-A' },
          { id: 'tip-A', _lineage_root_id: 'root-A' }
        ],
        'live-runtime'
      )
    ).toEqual(['tip-A', 'root-A'])
  })

  it('prefers the active continuation DB when both the tip and lineage root exist', () => {
    const ids = resolveMemoryDebugSessionIds(
      'root-A',
      [{ id: 'tip-A', _lineage_root_id: 'root-A' }],
      'live-runtime'
    )

    expect(resolveMemoryDebugDb([summary('root.sqlite3', 'root-A'), summary('tip.sqlite3', 'tip-A')], ids)).toBe(
      'tip.sqlite3'
    )
  })

  it('uses the live id only for a fresh session without a durable selection', () => {
    expect(resolveMemoryDebugSessionIds(null, [], 'live-runtime')).toEqual(['live-runtime'])
  })
})

describe('MemoryDebugPanel loading', () => {
  it('keeps the first sessions response current and loads the matching overview', async () => {
    const current = summary('current.sqlite3', 'durable-A')
    memoryDebugApi.sessions.mockResolvedValue({ root: '/memory', sessions: [current] })
    memoryDebugApi.session.mockResolvedValue({
      health: {},
      memory: {
        entities: [],
        events: { macro: [], micro: [], super: [] },
        evolution: { entity_states: [], revisions: [] }
      },
      session: current,
      tables: [],
      timeline: [],
      trace: { logs: [] }
    })

    render(<MemoryDebugPanel durableSessionIds={['durable-A']} liveSessionId="live-A" onOpenChange={vi.fn()} open />)

    expect(await screen.findByRole('option', { name: /current\.sqlite3/ })).toBeTruthy()
    await waitFor(() =>
      expect(memoryDebugApi.session).toHaveBeenCalledWith('current.sqlite3', {
        limit: 260,
        session_id: 'durable-A'
      })
    )
  })

  it('does no REST or trajectory work until the panel is opened', async () => {
    const request = vi.fn()
    const on = vi.fn()

    gatewayAtom.set({ on, request })

    const { rerender } = render(
      <MemoryDebugPanel durableSessionIds={['durable-A']} liveSessionId="live-A" onOpenChange={vi.fn()} open={false} />
    )

    await Promise.resolve()
    expect(memoryDebugApi.sessions).not.toHaveBeenCalled()
    expect(memoryDebugApi.session).not.toHaveBeenCalled()
    expect(memoryDebugApi.trace).not.toHaveBeenCalled()
    expect(memoryDebugApi.frame).not.toHaveBeenCalled()
    expect(memoryDebugApi.search).not.toHaveBeenCalled()
    expect(request).not.toHaveBeenCalled()
    expect(on).not.toHaveBeenCalled()

    rerender(
      <MemoryDebugPanel durableSessionIds={['durable-A']} liveSessionId="live-A" onOpenChange={vi.fn()} open />
    )
    await waitFor(() => expect(memoryDebugApi.sessions).toHaveBeenCalledWith(80))
  })

  it('does not mix the current live trajectory into a manually selected historical DB', async () => {
    memoryDebugApi.sessions.mockResolvedValue({
      root: '/memory',
      sessions: [summary('current.sqlite3', 'durable-A'), summary('historical.sqlite3', 'durable-B')]
    })

    render(
      <MemoryDebugPanel
        durableSessionIds={['durable-A']}
        liveSessionId="live-A"
        onOpenChange={vi.fn()}
        open
      />
    )

    const database = await screen.findByLabelText('Memory 数据库')
    fireEvent.change(database, { target: { value: 'historical.sqlite3' } })
    fireEvent.click(screen.getByRole('button', { name: '高级 Debug' }))

    expect(await screen.findByText(/历史 Memory 数据库没有持久化 live trajectory/)).toBeTruthy()
  })

  it('clears existing search results and rejects a pending latest search when the database changes', async () => {
    const pendingSearch = deferred<{
      results: Array<{ kind: string; score: number; session: string; snippet: string; title: string }>
    }>()

    memoryDebugApi.sessions.mockResolvedValue({
      root: '/memory',
      sessions: [summary('A.sqlite3', 'durable-A'), summary('B.sqlite3', 'durable-B')]
    })
    memoryDebugApi.search
      .mockResolvedValueOnce({
        results: [{ kind: 'ocr', score: 1, session: 'A.sqlite3', snippet: 'RESULT FROM A', title: 'A result' }]
      })
      .mockReturnValueOnce(pendingSearch.promise)

    render(<MemoryDebugPanel durableSessionIds={['durable-A']} liveSessionId="live-A" onOpenChange={vi.fn()} open />)

    const database = await screen.findByLabelText('Memory 数据库')
    fireEvent.click(screen.getByRole('button', { name: '搜索' }))
    const query = screen.getByLabelText('搜索 Memory')
    const submit = screen.getAllByRole('button', { name: '搜索' }).at(-1)!

    fireEvent.change(query, { target: { value: 'alpha' } })
    fireEvent.click(submit)
    expect(await screen.findByText('RESULT FROM A')).toBeTruthy()

    fireEvent.change(query, { target: { value: 'beta' } })
    fireEvent.click(submit)
    await waitFor(() => expect(memoryDebugApi.search).toHaveBeenCalledTimes(2))

    fireEvent.change(database, { target: { value: 'B.sqlite3' } })
    expect(screen.queryByText('RESULT FROM A')).toBeNull()

    pendingSearch.resolve({
      results: [{ kind: 'ocr', score: 1, session: 'A.sqlite3', snippet: 'LATE RESULT FROM A', title: 'late A' }]
    })
    await pendingSearch.promise
    await Promise.resolve()
    expect(screen.queryByText(/RESULT FROM A/)).toBeNull()
  })

  it('remounts fail-closed on both profile and conversation scope changes', async () => {
    memoryDebugApi.sessions
      .mockResolvedValueOnce({ root: '/memory-A', sessions: [summary('A.sqlite3', 'durable-A')] })
      .mockResolvedValueOnce({ root: '/memory-B', sessions: [summary('B.sqlite3', 'durable-A')] })
      .mockResolvedValueOnce({ root: '/memory-C', sessions: [summary('C.sqlite3', 'durable-C')] })
    gatewayAtom.set({ profile: 'A' })

    const { rerender } = render(
      <MemoryDebugPanel durableSessionIds={['durable-A']} liveSessionId="live-A" onOpenChange={vi.fn()} open />
    )

    expect(await screen.findByRole('option', { name: /A\.sqlite3/ })).toBeTruthy()
    const profileAContent = screen.getByTestId('memory-debug-content')

    act(() => gatewayAtom.set({ profile: 'B' }))
    expect(screen.getByTestId('memory-debug-content')).not.toBe(profileAContent)
    expect(screen.queryByRole('option', { name: /A\.sqlite3/ })).toBeNull()

    expect(await screen.findByRole('option', { name: /B\.sqlite3/ })).toBeTruthy()
    const profileBContent = screen.getByTestId('memory-debug-content')

    rerender(
      <MemoryDebugPanel durableSessionIds={['durable-C']} liveSessionId="live-C" onOpenChange={vi.fn()} open />
    )
    expect(screen.getByTestId('memory-debug-content')).not.toBe(profileBContent)
    expect(screen.queryByRole('option', { name: /B\.sqlite3/ })).toBeNull()
    expect(await screen.findByRole('option', { name: /C\.sqlite3/ })).toBeTruthy()
  })

  it('drops a late sessions response after the active conversation changes', async () => {
    const responseA = deferred<{ root: string; sessions: MmMemoryDebugSessionSummary[] }>()
    const responseB = deferred<{ root: string; sessions: MmMemoryDebugSessionSummary[] }>()
    memoryDebugApi.sessions.mockReturnValueOnce(responseA.promise).mockReturnValueOnce(responseB.promise)

    const { rerender } = render(
      <MemoryDebugPanel
        durableSessionIds={['durable-A']}
        liveSessionId="live-A"
        onOpenChange={vi.fn()}
        open
      />
    )

    rerender(
      <MemoryDebugPanel
        durableSessionIds={['durable-B']}
        liveSessionId="live-B"
        onOpenChange={vi.fn()}
        open
      />
    )
    responseB.resolve({ root: '/memory', sessions: [summary('B.sqlite3', 'durable-B')] })

    expect(await screen.findByRole('option', { name: /B\.sqlite3/ })).toBeTruthy()
    responseA.resolve({ root: '/memory', sessions: [summary('A.sqlite3', 'durable-A')] })

    await waitFor(() => expect(screen.queryByRole('option', { name: /A\.sqlite3/ })).toBeNull())
    expect(memoryDebugApi.session).not.toHaveBeenCalledWith(
      'A.sqlite3',
      expect.objectContaining({ session_id: 'durable-A' })
    )
  })

  it('reloads after an active profile gateway switch and drops the old profile response', async () => {
    const profileA = deferred<{ root: string; sessions: MmMemoryDebugSessionSummary[] }>()
    const profileB = deferred<{ root: string; sessions: MmMemoryDebugSessionSummary[] }>()
    memoryDebugApi.sessions.mockReturnValueOnce(profileA.promise).mockReturnValueOnce(profileB.promise)
    gatewayAtom.set({ profile: 'A' })

    render(<MemoryDebugPanel durableSessionIds={['durable-A']} liveSessionId="live-A" onOpenChange={vi.fn()} open />)

    await waitFor(() => expect(memoryDebugApi.sessions).toHaveBeenCalledTimes(1))
    gatewayAtom.set({ profile: 'B' })
    await waitFor(() => expect(memoryDebugApi.sessions).toHaveBeenCalledTimes(2))

    profileB.resolve({ root: '/memory-B', sessions: [summary('B.sqlite3', 'durable-A')] })
    expect(await screen.findByRole('option', { name: /B\.sqlite3/ })).toBeTruthy()

    profileA.resolve({ root: '/memory-A', sessions: [summary('A.sqlite3', 'durable-A')] })
    await waitFor(() => expect(screen.queryByRole('option', { name: /A\.sqlite3/ })).toBeNull())
    expect(memoryDebugApi.session).not.toHaveBeenCalledWith(
      'A.sqlite3',
      expect.objectContaining({ session_id: 'durable-A' })
    )
  })

  it('drops late overview, search, trace, and trajectory responses after A→B', async () => {
    const overviewA = deferred<ReturnType<typeof emptyOverview>>()
    const searchA = deferred<{ results: Array<{ kind: string; score: number; session: string; snippet: string; title: string }> }>()
    const traceA = deferred<{ logs: string[]; messages: Array<Record<string, unknown>> }>()
    const trajectoryA = deferred<{ count: number; entries: MmTrajectoryEntry[] }>()

    const request = vi
      .fn()
      .mockReturnValueOnce(trajectoryA.promise)
      .mockResolvedValue({ count: 0, entries: [] })

    const on = vi.fn(() => vi.fn())

    gatewayAtom.set({ on, request })
    memoryDebugApi.sessions
      .mockResolvedValueOnce({ root: '/memory', sessions: [summary('A.sqlite3', 'durable-A')] })
      .mockResolvedValueOnce({ root: '/memory', sessions: [summary('B.sqlite3', 'durable-B')] })
    memoryDebugApi.session.mockReturnValueOnce(overviewA.promise).mockResolvedValueOnce(emptyOverview('B.sqlite3'))
    memoryDebugApi.search.mockReturnValueOnce(searchA.promise)
    memoryDebugApi.trace.mockReturnValueOnce(traceA.promise)

    const { rerender } = render(
      <MemoryDebugPanel durableSessionIds={['durable-A']} liveSessionId="live-A" onOpenChange={vi.fn()} open />
    )

    expect(await screen.findByRole('option', { name: /A\.sqlite3/ })).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: '高级 Debug' }))
    await waitFor(() => expect(memoryDebugApi.trace).toHaveBeenCalled())
    await waitFor(() => expect(request).toHaveBeenCalled())
    fireEvent.click(screen.getByRole('button', { name: '搜索' }))
    fireEvent.change(screen.getByLabelText('搜索 Memory'), { target: { value: 'alpha' } })
    fireEvent.click(screen.getAllByRole('button', { name: '搜索' }).at(-1)!)
    await waitFor(() => expect(memoryDebugApi.search).toHaveBeenCalled())

    rerender(
      <MemoryDebugPanel durableSessionIds={['durable-B']} liveSessionId="live-B" onOpenChange={vi.fn()} open />
    )
    expect(await screen.findByRole('option', { name: /B\.sqlite3/ })).toBeTruthy()

    overviewA.resolve({ ...emptyOverview('A.sqlite3'), timeline: [{ frame_id: 'frame-A' }] } as ReturnType<
      typeof emptyOverview
    >)
    searchA.resolve({
      results: [{ kind: 'ocr', score: 1, session: 'A.sqlite3', snippet: 'STALE SEARCH A', title: 'STALE A' }]
    })
    traceA.resolve({ logs: ['STALE TRACE A'], messages: [] })
    trajectoryA.resolve({ count: 1, entries: [trajectory('stale-A', 99, 'STALE TRAJECTORY A')] })

    await waitFor(() => expect(screen.queryByText(/STALE SEARCH A/)).toBeNull())
    fireEvent.click(screen.getByRole('button', { name: '高级 Debug' }))
    await waitFor(() => expect(memoryDebugApi.trace).toHaveBeenCalledTimes(2))
    expect(screen.queryByText(/STALE (TRACE|TRAJECTORY) A/)).toBeNull()
    expect(screen.queryByRole('option', { name: /A\.sqlite3/ })).toBeNull()
  })

  it('does not paint an A frame response after the conversation switches to B', async () => {
    const frameA = deferred<Record<string, unknown>>()
    memoryDebugApi.sessions
      .mockResolvedValueOnce({ root: '/memory', sessions: [summary('A.sqlite3', 'durable-A')] })
      .mockResolvedValueOnce({ root: '/memory', sessions: [summary('B.sqlite3', 'durable-B')] })
    memoryDebugApi.session
      .mockResolvedValueOnce({ ...emptyOverview('A.sqlite3'), timeline: [{ frame_id: 'frame-A' }] })
      .mockResolvedValueOnce(emptyOverview('B.sqlite3'))
    memoryDebugApi.frame.mockReturnValueOnce(frameA.promise)

    const { rerender } = render(
      <MemoryDebugPanel durableSessionIds={['durable-A']} liveSessionId="live-A" onOpenChange={vi.fn()} open />
    )

    expect(await screen.findByRole('option', { name: /A\.sqlite3/ })).toBeTruthy()
    await waitFor(() => expect(memoryDebugApi.session).toHaveBeenCalledWith('A.sqlite3', expect.any(Object)))
    fireEvent.click(screen.getByRole('button', { name: '帧详情' }))
    await waitFor(() => expect(memoryDebugApi.frame).toHaveBeenCalledWith('A.sqlite3', 'frame-A'))

    rerender(
      <MemoryDebugPanel durableSessionIds={['durable-B']} liveSessionId="live-B" onOpenChange={vi.fn()} open />
    )
    expect(await screen.findByRole('option', { name: /B\.sqlite3/ })).toBeTruthy()
    frameA.resolve({ frame_id: 'frame-A', screen_text: { raw_text: 'STALE FRAME A', ocr_blocks: [] } })

    await waitFor(() => expect(screen.queryByText('STALE FRAME A')).toBeNull())
  })

  it('unsubscribes trajectory events and ignores pending work when closed', async () => {
    const pendingTrajectory = deferred<{ count: number; entries: MmTrajectoryEntry[] }>()

    const off = vi.fn()
    const on = vi.fn(() => off)
    const request = vi.fn(() => pendingTrajectory.promise)

    gatewayAtom.set({ on, request })
    memoryDebugApi.sessions.mockResolvedValue({
      root: '/memory',
      sessions: [summary('current.sqlite3', 'durable-A')]
    })

    const { rerender } = render(
      <MemoryDebugPanel durableSessionIds={['durable-A']} liveSessionId="live-A" onOpenChange={vi.fn()} open />
    )

    expect(await screen.findByRole('option', { name: /current\.sqlite3/ })).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: '高级 Debug' }))
    await waitFor(() => expect(on).toHaveBeenCalledWith('multimodal.trajectory', expect.any(Function)))
    await waitFor(() => expect(request).toHaveBeenCalled())

    rerender(
      <MemoryDebugPanel durableSessionIds={['durable-A']} liveSessionId="live-A" onOpenChange={vi.fn()} open={false} />
    )
    expect(off).toHaveBeenCalledTimes(1)
    pendingTrajectory.resolve({ count: 1, entries: [trajectory('late-after-close', 9)] })
    await Promise.resolve()
    expect(memoryDebugApi.sessions).toHaveBeenCalledTimes(1)
  })

  it('merges a same-session live trajectory received before the list response and rejects other sessions', async () => {
    const pendingList = deferred<{ count: number; entries: MmTrajectoryEntry[] }>()
    let liveHandler: ((event: { payload: MmTrajectoryEntry; session_id: string }) => void) | undefined

    const on = vi.fn((_name: string, handler: typeof liveHandler) => {
      liveHandler = handler

      return vi.fn()
    })

    gatewayAtom.set({ on, request: vi.fn(() => pendingList.promise) })
    memoryDebugApi.sessions.mockResolvedValue({
      root: '/memory',
      sessions: [summary('current.sqlite3', 'durable-A')]
    })

    render(<MemoryDebugPanel durableSessionIds={['durable-A']} liveSessionId="live-A" onOpenChange={vi.fn()} open />)
    expect(await screen.findByRole('option', { name: /current\.sqlite3/ })).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: '高级 Debug' }))
    await waitFor(() => expect(liveHandler).toBeTypeOf('function'))

    liveHandler?.({ payload: trajectory('live-2', 2, 'live-only'), session_id: 'live-A' })
    liveHandler?.({ payload: trajectory('foreign-9', 9, 'foreign-only'), session_id: 'live-B' })
    pendingList.resolve({ count: 2, entries: [trajectory('listed-1', 1), trajectory('live-2', 2, 'listed-old')] })

    await waitFor(() => expect(screen.getAllByText(/2 events/).length).toBeGreaterThan(0))
    expect(screen.queryByText(/foreign-only/)).toBeNull()
  })

  it('shows a load error and retries successfully from the refresh action', async () => {
    memoryDebugApi.sessions
      .mockRejectedValueOnce(new Error('debug backend unavailable'))
      .mockResolvedValueOnce({ root: '/memory', sessions: [summary('recovered.sqlite3', 'durable-A')] })

    render(<MemoryDebugPanel durableSessionIds={['durable-A']} liveSessionId="live-A" onOpenChange={vi.fn()} open />)

    expect(await screen.findByText('debug backend unavailable')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: '刷新 Memory Debug' }))
    expect(await screen.findByRole('option', { name: /recovered\.sqlite3/ })).toBeTruthy()
    await waitFor(() => expect(screen.queryByText('debug backend unavailable')).toBeNull())
  })
})

describe('mergeMemoryDebugTrajectory', () => {
  it('merges list/live rows by stable id, with the incoming live row winning', () => {
    const duplicateFromList = trajectory('tr-2', 2, 'listed-version')
    const duplicateFromLive = trajectory('tr-2', 2, 'live-version')

    const merged = mergeMemoryDebugTrajectory(
      [trajectory('tr-1', 1), duplicateFromList],
      [duplicateFromLive, trajectory('tr-3', 3)]
    )

    expect(merged.map(item => item.id)).toEqual(['tr-1', 'tr-2', 'tr-3'])
    expect(merged.find(item => item.id === 'tr-2')?.event).toBe('live-version')
  })

  it('keeps a live duplicate over a late hydrate but lets a later live row replace hydrate', () => {
    const live = trajectory('tr-shared', 2, 'live')
    const listed = trajectory('tr-shared', 2, 'listed')

    expect(mergeMemoryDebugTrajectory([live], [listed], 2000, false)[0].event).toBe('live')
    expect(mergeMemoryDebugTrajectory([listed], [live])[0].event).toBe('live')
  })

  it('sorts by sequence then timestamp before retaining only the newest cap', () => {
    const sameSeqEarly = { ...trajectory('tr-3-early', 3), ts: 30 }
    const sameSeqLate = { ...trajectory('tr-3-late', 3), ts: 31 }

    const merged = mergeMemoryDebugTrajectory(
      [trajectory('tr-4', 4), trajectory('tr-2', 2)],
      [sameSeqLate, trajectory('tr-5', 5), sameSeqEarly],
      3
    )

    expect(merged.map(item => item.id)).toEqual(['tr-3-late', 'tr-4', 'tr-5'])
  })

  it('keeps id-less legacy rows distinct by sequence while still deduplicating them', () => {
    const old = { ...trajectory('', 6, 'listed'), id: '' }
    const replacement = { ...trajectory('', 6, 'live'), id: '' }

    const merged = mergeMemoryDebugTrajectory([old], [replacement])

    expect(merged).toHaveLength(1)
    expect(merged[0].event).toBe('live')
  })
})
