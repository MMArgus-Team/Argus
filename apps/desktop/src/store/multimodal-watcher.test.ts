import type { WritableAtom } from 'nanostores'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const { gatewayAtom, sessionIdAtom } = vi.hoisted(() => {
  const { atom } = require('nanostores') as {
    atom: <Value>(initial: Value) => WritableAtom<Value>
  }

  return {
    gatewayAtom: atom<unknown>(null),
    sessionIdAtom: atom<string>('')
  }
})

vi.mock('./gateway', () => ({ $gateway: gatewayAtom }))
vi.mock('./multimodal', () => ({ $mmSessionId: sessionIdAtom }))
vi.mock('./notifications', () => ({ notifyError: vi.fn() }))

import {
  $mmBgItems,
  $mmWatcherReports,
  $mmWatchers,
  fetchMmSidechannel,
  resetDeepUi,
  setWatcherFinal,
  setWatchers,
  toggleWatcher
} from './multimodal-deep'

interface Deferred<T> {
  promise: Promise<T>
  reject: (reason?: unknown) => void
  resolve: (value: T) => void
}

function deferred<T>(): Deferred<T> {
  let reject!: (reason?: unknown) => void
  let resolve!: (value: T) => void

  const promise = new Promise<T>((done, fail) => {
    reject = fail
    resolve = done
  })

  return { promise, reject, resolve }
}

describe('desktop watcher registry lifecycle', () => {
  beforeEach(() => {
    resetDeepUi()
    sessionIdAtom.set('runtime-watcher')
    gatewayAtom.set(null)
  })

  it('uses stopping while disabling and reconciles with an authoritative registry pull', async () => {
    const toggle = deferred<Record<string, never>>()

    const request = vi.fn((method: string) => {
      if (method === 'multimodal.watcher_toggle') {
        return toggle.promise
      }

      return Promise.resolve({
        ready: true,
        monitors: [],
        watchers: [{ watcher_id: 'watch-1', status: 'interrupted' }]
      })
    })

    gatewayAtom.set({ request })
    setWatchers([{ watcher_id: 'watch-1', status: 'running' }])

    const disabling = toggleWatcher('watch-1', false)
    expect($mmWatchers.get()[0]?.status).toBe('stopping')

    toggle.resolve({})
    await disabling

    expect(request).toHaveBeenNthCalledWith(1, 'multimodal.watcher_toggle', {
      enabled: false,
      session_id: 'runtime-watcher',
      watcher_id: 'watch-1'
    })
    expect(request).toHaveBeenNthCalledWith(2, 'multimodal.list_registries', {
      session_id: 'runtime-watcher'
    })
    expect($mmWatchers.get()[0]?.status).toBe('interrupted')
  })

  it('does not roll a concurrent done push back when a disable request fails', async () => {
    const toggle = deferred<Record<string, never>>()
    gatewayAtom.set({ request: vi.fn(() => toggle.promise) })
    setWatchers([{ watcher_id: 'watch-race', status: 'running' }])

    const disabling = toggleWatcher('watch-race', false)
    expect($mmWatchers.get()[0]?.status).toBe('stopping')

    setWatchers([{ watcher_id: 'watch-race', status: 'done' }])
    toggle.reject(new Error('already completed'))
    await disabling

    expect($mmWatchers.get()[0]?.status).toBe('done')
  })

  it('rolls back a failed disable while its own optimistic row still owns the state', async () => {
    const toggle = deferred<Record<string, never>>()
    gatewayAtom.set({ request: vi.fn(() => toggle.promise) })
    setWatchers([{ watcher_id: 'watch-rollback', status: 'running' }])

    const disabling = toggleWatcher('watch-rollback', false)
    expect($mmWatchers.get()[0]?.status).toBe('stopping')

    toggle.reject(new Error('stop failed'))
    await disabling

    expect($mmWatchers.get()[0]?.status).toBe('running')
  })

  it.each(['done', 'complete', 'stopping', 'deleted'])(
    'does not send a toggle for non-toggleable status %s',
    async status => {
      const request = vi.fn(async () => ({}))
      gatewayAtom.set({ request })
      setWatchers([{ watcher_id: `watch-${status}`, status }])

      await toggleWatcher(`watch-${status}`, true)

      expect(request).not.toHaveBeenCalled()
      expect($mmWatchers.get()[0]?.status).toBe(status)
    }
  )
})

describe('desktop watcher sidechannel hydration', () => {
  beforeEach(() => {
    resetDeepUi()
    sessionIdAtom.set('runtime-watcher')
    gatewayAtom.set(null)
  })

  it('hydrates a final-only watcher into both the report union and the panel', async () => {
    gatewayAtom.set({
      request: vi.fn(async (method: string) => {
        if (method === 'multimodal.list_monitor_alerts') {
          return { alerts: [] }
        }

        return {
          finals: [{ watcher_id: 'final-only', text: 'final result', wall_ts: 10 }],
          reports: []
        }
      })
    })

    await fetchMmSidechannel()

    expect($mmWatcherReports.get()).toEqual({ 'final-only': [] })
    expect($mmBgItems.get()).toEqual([
      expect.objectContaining({
        done: true,
        finalReport: 'final result',
        requestId: 'final-only',
        segments: []
      })
    ])
  })

  it('merges a pending snapshot without overwriting richer or terminal live state', async () => {
    const watcherContent = deferred<{
      finals: Array<{ watcher_id: string; text: string; wall_ts: number }>
      reports: Array<{ watcher_id: string; round_idx: number; text: string; wall_ts: number }>
    }>()

    const request = vi.fn((method: string) => {
      if (method === 'multimodal.list_monitor_alerts') {
        return Promise.resolve({ alerts: [] })
      }

      return watcherContent.promise
    })

    gatewayAtom.set({ request })

    const hydration = fetchMmSidechannel()
    await vi.waitFor(() => expect(request).toHaveBeenCalledWith(
      'multimodal.list_watcher_content',
      { session_id: 'runtime-watcher' }
    ))

    $mmBgItems.set([
      {
        id: 'live-rich',
        requestId: 'live-rich',
        segments: [{
          crops: [{ jpeg_b64: 'live-image' }],
          lookups: [{ kind: 'search', query: 'live query' }],
          seg: 1
        }],
        waiting: { have: 2, need: 3 }
      },
      { id: 'live-terminal', requestId: 'live-terminal', segments: [] }
    ])
    setWatcherFinal('live-terminal', 'new live final')

    watcherContent.resolve({
      finals: [
        { watcher_id: 'live-rich', text: 'stale rich final', wall_ts: 1 },
        { watcher_id: 'live-terminal', text: 'stale terminal final', wall_ts: 1 }
      ],
      reports: [
        { watcher_id: 'live-rich', round_idx: 1, text: 'stale segment', wall_ts: 1 },
        { watcher_id: 'live-terminal', round_idx: 1, text: 'old segment', wall_ts: 1 }
      ]
    })
    await hydration

    const rich = $mmBgItems.get().find(item => item.requestId === 'live-rich')
    expect(rich).toMatchObject({
      done: undefined,
      finalReport: undefined,
      waiting: { have: 2, need: 3 }
    })
    expect(rich?.segments[0]).toMatchObject({
      crops: [{ jpeg_b64: 'live-image' }],
      lookups: [{ kind: 'search', query: 'live query' }]
    })

    const terminal = $mmBgItems.get().find(item => item.requestId === 'live-terminal')
    expect(terminal).toMatchObject({ done: true, finalReport: 'new live final' })
  })
})
