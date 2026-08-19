import {
  AssistantRuntimeImpl,
  BaseAssistantRuntimeCore,
  ExternalStoreThreadListRuntimeCore,
  ExternalStoreThreadRuntimeCore,
  hasUpcomingMessage
} from '@assistant-ui/core/internal'
import {
  type AssistantRuntime,
  type ExternalStoreAdapter,
  type ThreadMessage,
  useRuntimeAdapters
} from '@assistant-ui/react'
import { useEffect, useMemo, useState } from 'react'

const EMPTY_ARRAY = Object.freeze([])

const shallowEqual = (a: object, b: object): boolean => {
  const aKeys = Object.keys(a)

  if (aKeys.length !== Object.keys(b).length) {
    return false
  }

  for (const key of aKeys) {
    if (a[key as keyof typeof a] !== b[key as keyof typeof b]) {
      return false
    }
  }

  return true
}

const getThreadListAdapter = (store: ExternalStoreAdapter) => store.adapters?.threadList ?? {}

// Per-runtime memo of what we last applied to the repository, so a streaming
// flush (which changes only the LAST assistant message but hands us a fresh
// full array) doesn't re-`addOrUpdateMessage` all N messages every ~100ms.
// ★ Perf: at 400+ turns (~800 msgs) the old "re-apply everything each flush"
//   was O(N) per flush × ~10 flushes/s = the long-session typing/stream lag.
//   We now only touch messages whose (message object identity, parentId) changed
//   since the last sync — during streaming that's just the one growing bubble.
interface _SyncMemo {
  // id → the exact (ThreadMessage, parentId) we last pushed into the repository.
  //   Identity compare: the upstream WeakMap in ChatViewInner returns the SAME
  //   ThreadMessage object for an unchanged ChatMessage, so `===` is a valid
  //   "did this message change" test.
  applied: Map<string, { message: ThreadMessage; parentId: string | null }>
  // The message-id order we last applied. If the order is unchanged we can skip
  // the full export() reconcile scan entirely.
  orderKey: string
}

function syncRepositoryIncrementally(
  runtime: ExternalStoreThreadRuntimeCore,
  messageRepository: NonNullable<ExternalStoreAdapter['messageRepository']>
): readonly ThreadMessage[] {
  const repository = (runtime as unknown as { repository: ExternalStoreThreadRuntimeCore['repository'] }).repository
  const holder = runtime as unknown as { _syncMemo?: _SyncMemo }
  const memo: _SyncMemo = holder._syncMemo ?? { applied: new Map(), orderKey: '' }

  const incoming = messageRepository.messages
  const nextApplied = new Map<string, { message: ThreadMessage; parentId: string | null }>()
  const incomingIds = new Set<string>()
  let orderKey = ''

  for (const item of incoming) {
    const id = item.message.id
    incomingIds.add(id)
    orderKey += id + '\n'
    nextApplied.set(id, item)
    const prev = memo.applied.get(id)
    // Only push into the repository when this message is new OR its object
    // identity / parent actually changed. Unchanged idle messages are skipped —
    // they already live in the persistent repository from a prior sync.
    if (!prev || prev.message !== item.message || prev.parentId !== item.parentId) {
      repository.addOrUpdateMessage(item.parentId, item.message)
    }
  }

  // Deletions: only scan when the id set/order changed (streaming keeps it stable
  // → skip the O(N) export() scan entirely on the hot path).
  if (orderKey !== memo.orderKey) {
    for (const { message } of repository.export().messages) {
      if (!incomingIds.has(message.id)) {
        repository.deleteMessage(message.id)
      }
    }
  }

  holder._syncMemo = { applied: nextApplied, orderKey }

  const headId = messageRepository.headId ?? incoming.at(-1)?.message.id ?? null

  repository.resetHead(headId)

  return repository.getMessages()
}

class IncrementalExternalStoreThreadRuntimeCore extends ExternalStoreThreadRuntimeCore {
  override __internal_setAdapter(store: ExternalStoreAdapter): void {
    if (!store.messageRepository) {
      super.__internal_setAdapter(store)

      return
    }

    const self = this as unknown as {
      _assistantOptimisticId: null | string
      _capabilities: object
      _messages: readonly ThreadMessage[]
      _notifyEventSubscribers: (event: string, payload: object) => void
      _notifySubscribers: () => void
      _store?: ExternalStoreAdapter
    }

    if (self._store === store) {
      return
    }

    const isRunning = store.isRunning ?? false
    this.isDisabled = store.isDisabled ?? false

    const oldStore = self._store
    self._store = store

    if (this.extras !== store.extras) {
      this.extras = store.extras
    }

    const newSuggestions = store.suggestions ?? EMPTY_ARRAY

    if (!shallowEqual(this.suggestions, newSuggestions)) {
      this.suggestions = newSuggestions
    }

    const newCapabilities = {
      switchToBranch: store.setMessages !== undefined,
      switchBranchDuringRun: false,
      edit: store.onEdit !== undefined,
      reload: store.onReload !== undefined,
      cancel: store.onCancel !== undefined,
      speech: store.adapters?.speech !== undefined,
      dictation: store.adapters?.dictation !== undefined,
      voice: store.adapters?.voice !== undefined,
      unstable_copy: store.unstable_capabilities?.copy !== false,
      attachments: !!store.adapters?.attachments,
      feedback: !!store.adapters?.feedback,
      queue: false
    }

    if (!shallowEqual(self._capabilities, newCapabilities)) {
      self._capabilities = newCapabilities
    }

    if (oldStore && oldStore.isRunning === store.isRunning && oldStore.messageRepository === store.messageRepository) {
      self._notifySubscribers()

      return
    }

    if (self._assistantOptimisticId) {
      this.repository.deleteMessage(self._assistantOptimisticId)
      self._assistantOptimisticId = null
    }

    const messages = syncRepositoryIncrementally(this, store.messageRepository)

    if (messages.length > 0) {
      this.ensureInitialized()
    }

    if ((oldStore?.isRunning ?? false) !== (store.isRunning ?? false)) {
      self._notifyEventSubscribers(store.isRunning ? 'runStart' : 'runEnd', {})
    }

    if (hasUpcomingMessage(isRunning, messages)) {
      self._assistantOptimisticId = this.repository.appendOptimisticMessage(messages.at(-1)?.id ?? null, {
        role: 'assistant',
        content: []
      })
    }

    this.repository.resetHead(self._assistantOptimisticId ?? messages.at(-1)?.id ?? null)
    self._messages = this.repository.getMessages()
    self._notifySubscribers()
  }
}

class IncrementalExternalStoreRuntimeCore extends BaseAssistantRuntimeCore {
  threads: ExternalStoreThreadListRuntimeCore

  constructor(adapter: ExternalStoreAdapter) {
    super()

    this.threads = new ExternalStoreThreadListRuntimeCore(
      getThreadListAdapter(adapter),
      () => new IncrementalExternalStoreThreadRuntimeCore(this._contextProvider, adapter)
    )
  }

  setAdapter(adapter: ExternalStoreAdapter): void {
    this.threads.__internal_setAdapter(getThreadListAdapter(adapter))
    this.threads.getMainThreadRuntimeCore().__internal_setAdapter(adapter)
  }
}

export function useIncrementalExternalStoreRuntime<T extends ThreadMessage>(
  store: ExternalStoreAdapter<T>
): AssistantRuntime {
  const [runtime] = useState(() => new IncrementalExternalStoreRuntimeCore(store as ExternalStoreAdapter))

  useEffect(() => {
    runtime.setAdapter(store as ExternalStoreAdapter)
  })

  const { modelContext } = useRuntimeAdapters() ?? {}

  useEffect(() => {
    if (!modelContext) {
      return undefined
    }

    return runtime.registerModelContextProvider(modelContext)
  }, [modelContext, runtime])

  return useMemo(() => new AssistantRuntimeImpl(runtime), [runtime])
}
