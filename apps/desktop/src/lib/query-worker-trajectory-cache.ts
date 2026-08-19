interface QueryWorkerTrajectoryCacheEntry {
  payload: Record<string, unknown>
  phase: string
}

const IMAGE_TASK_LIMIT = 4
const IMAGE_CHAR_BUDGET = 4_000_000
const IMAGE_KEYS = ['image_b64', 'jpeg_b64', 'thumb_b64'] as const

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function taskId(entry: QueryWorkerTrajectoryCacheEntry): string {
  const result = isRecord(entry.payload.result) ? entry.payload.result : {}
  const value = entry.payload.task_id || result.task_id

  return typeof value === 'string' ? value.trim() : ''
}

function imageChars(frame: Record<string, unknown>): number {
  return IMAGE_KEYS.reduce(
    (total, key) => total + (typeof frame[key] === 'string' ? frame[key].length : 0),
    0
  )
}

function withoutImageBytes(frame: Record<string, unknown>): Record<string, unknown> {
  if (!IMAGE_KEYS.some(key => key in frame)) {
    return frame
  }

  const metadata = { ...frame }

  for (const key of IMAGE_KEYS) {
    delete metadata[key]
  }

  return metadata
}

/** Bound live QueryWorker image retention exactly like the Web client.
 *
 * Gateway hydration already applies a server-side image budget, but a renderer
 * can receive many complete live JPEG events before the next hydrate. Keep all
 * textual steps and frame metadata while retaining image bytes for at most the
 * four newest tasks and 4M characters. The newest task's exact ask-time input
 * is protected even when it alone exceeds that normal budget.
 */
export function compactQueryWorkerTrajectoryImages<
  Entry extends QueryWorkerTrajectoryCacheEntry
>(entries: Entry[]): Entry[] {
  const taskOrder: string[] = []

  for (const entry of entries) {
    const id = taskId(entry)

    if (!id) {
      continue
    }

    const previous = taskOrder.indexOf(id)

    if (previous >= 0) {
      taskOrder.splice(previous, 1)
    }

    taskOrder.push(id)
  }

  const newestTask = taskOrder.at(-1) || ''
  const imageTasks = new Set(taskOrder.slice(-IMAGE_TASK_LIMIT))
  let protectedChars = 0

  for (const entry of entries) {
    const payload = entry.payload

    if (taskId(entry) !== newestTask || String(entry.phase || payload.phase) !== 'started') {
      continue
    }

    for (const frame of Array.isArray(payload.frames) ? payload.frames : []) {
      if (isRecord(frame)) {
        protectedChars += imageChars(frame)
      }
    }
  }

  let remainingChars = Math.max(0, IMAGE_CHAR_BUDGET - protectedChars)

  return entries.slice().reverse().map(entry => {
    const payload = entry.payload
    const id = taskId(entry)
    const rawFrames = Array.isArray(payload.frames) ? payload.frames : null

    if (!id || !rawFrames?.length) {
      return entry
    }

    const protectedInput = id === newestTask && String(entry.phase || payload.phase) === 'started'
    let changed = false

    const frames = rawFrames.map(frame => {
      if (!isRecord(frame) || protectedInput) {
        return frame
      }

      const chars = imageChars(frame)

      if (chars === 0) {
        return frame
      }

      if (imageTasks.has(id) && chars <= remainingChars) {
        remainingChars -= chars

        return frame
      }

      changed = true

      return withoutImageBytes(frame)
    })

    return changed
      ? { ...entry, payload: { ...payload, frames } }
      : entry
  }).reverse()
}
