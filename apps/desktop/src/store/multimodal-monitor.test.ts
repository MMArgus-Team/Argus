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
  $mmMonitors,
  $mmWatchers,
  fetchMmRegistries,
  resetDeepUi,
  setMonitors,
  setWatchers,
  toggleMonitor
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

describe('desktop monitor registry lifecycle', () => {
  beforeEach(() => {
    resetDeepUi()
    sessionIdAtom.set('runtime-monitor')
    gatewayAtom.set(null)
  })

  it('does not let an unready empty pull overwrite a newer nonempty registry push', async () => {
    const pull = deferred<{ ready: boolean; monitors: never[]; watchers: never[] }>()
    const request = vi.fn(() => pull.promise)
    gatewayAtom.set({ request })

    const fetching = fetchMmRegistries()
    setMonitors([{ monitor_id: 'monitor-pushed', enabled: true, status: 'running' }])
    setWatchers([{ watcher_id: 'watcher-pushed', status: 'running' }])
    pull.resolve({ ready: false, monitors: [], watchers: [] })
    await fetching

    expect($mmMonitors.get().map(item => item.monitor_id)).toEqual(['monitor-pushed'])
    expect($mmWatchers.get().map(item => item.watcher_id)).toEqual(['watcher-pushed'])
  })

  it('accepts a ready empty pull as an authoritative registry clear', async () => {
    setMonitors([{ monitor_id: 'monitor-old', enabled: true, status: 'running' }])
    setWatchers([{ watcher_id: 'watcher-old', status: 'running' }])
    gatewayAtom.set({
      request: vi.fn(async () => ({ ready: true, monitors: [], watchers: [] }))
    })

    await fetchMmRegistries()

    expect($mmMonitors.get()).toEqual([])
    expect($mmWatchers.get()).toEqual([])
  })

  it.each(['done', 'complete'])('never toggles a completed once monitor with status %s', async status => {
    const request = vi.fn(async () => ({}))
    gatewayAtom.set({ request })
    setMonitors([
      {
        monitor_id: `once-${status}`,
        enabled: false,
        status,
        trigger_mode: 'once'
      }
    ])

    await toggleMonitor(`once-${status}`, true)

    expect(request).not.toHaveBeenCalled()
    expect($mmMonitors.get()[0]).toEqual(
      expect.objectContaining({ enabled: false, status, trigger_mode: 'once' })
    )
  })

  it('does not resurrect a monitor when a completion push wins a rejected toggle race', async () => {
    const toggle = deferred<Record<string, never>>()
    const request = vi.fn(() => toggle.promise)
    gatewayAtom.set({ request })
    setMonitors([
      {
        monitor_id: 'once-racing',
        enabled: false,
        status: 'interrupted',
        trigger_mode: 'once'
      }
    ])

    const enabling = toggleMonitor('once-racing', true)
    expect($mmMonitors.get()[0]).toEqual(
      expect.objectContaining({ enabled: true, status: 'running' })
    )

    // Backend completion push arrives while the optimistic RPC is pending.
    setMonitors([
      {
        monitor_id: 'once-racing',
        enabled: false,
        status: 'done',
        trigger_mode: 'once'
      }
    ])
    toggle.reject(new Error('already complete'))
    await enabling

    expect($mmMonitors.get()[0]).toEqual(
      expect.objectContaining({ enabled: false, status: 'done', trigger_mode: 'once' })
    )
  })
})
