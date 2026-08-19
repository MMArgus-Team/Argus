/**
 * Monitor SPEAK evidence strip — shared contract between the two frontends.
 *
 * A monitor hit carries a small strip of thumbnails showing the exact frame
 * batch the model evaluated. The producer is Python
 * (`agent/multimodal/monitor_engine.py: build_monitor_evidence`); it arrives
 * two ways, and BOTH are untrusted enough to need normalizing:
 *   - live, on `message.complete` (payload `evidence`);
 *   - on session reopen, from the `mm_monitor_alerts.evidence` TEXT column
 *     (`list_mm_monitor_alerts`), i.e. JSON persisted by an older build.
 *
 * ★ 2026-08-19: this used to be copy-pasted verbatim into BOTH
 * `web/src/lib/monitor-evidence.ts` and
 * `apps/desktop/src/store/multimodal-deep.ts`, with the two caps inline as
 * magic numbers — the same two numbers in 3 places across 2 languages. Both
 * frontends now re-export this module; only the Python constants below still
 * need manual syncing.
 */

/**
 * Keep in sync with `monitor_engine.py: _MM_MONITOR_EVIDENCE_MAX_FRAMES`.
 * The producer already samples down to this, so hitting it here means the
 * payload came from a different (newer/older) build — clamp rather than trust.
 */
export const MONITOR_EVIDENCE_MAX_FRAMES = 6

/**
 * Keep in sync with `monitor_engine.py: _MM_MONITOR_EVIDENCE_MAX_B64_CHARS`.
 * A hydration guard, not a target: 6 frames at 320px/q58 measure ~5–8 KB of
 * base64 in total, so this bounds a malformed/hostile payload rather than
 * normal traffic.
 */
export const MONITOR_EVIDENCE_MAX_B64_CHARS = 600_000

/** Longest `source_type` we will echo into a tooltip. */
const SOURCE_TYPE_MAX_CHARS = 32

/** Upper bound for the "N frames evaluated" counter (display only). */
const INPUT_COUNT_MAX = 100_000

export interface MonitorEvidenceFrame {
  ts: number
  source_type?: string
  thumb_b64: string
}

export interface MonitorEvidence {
  input_count: number
  shown_count: number
  frames: MonitorEvidenceFrame[]
}

/**
 * Coerce an untrusted `evidence` blob into a bounded, renderable strip.
 * Returns `undefined` when there is nothing displayable, so callers can use it
 * directly as an optional field.
 */
export function normalizeMonitorEvidence(value: unknown): MonitorEvidence | undefined {
  if (!value || typeof value !== 'object') {
    return undefined
  }

  const raw = value as { input_count?: unknown; frames?: unknown }

  if (!Array.isArray(raw.frames)) {
    return undefined
  }

  const frames: MonitorEvidenceFrame[] = []
  let imageChars = 0

  for (const item of raw.frames.slice(0, MONITOR_EVIDENCE_MAX_FRAMES)) {
    if (!item || typeof item !== 'object') {
      continue
    }

    const row = item as { ts?: unknown; source_type?: unknown; thumb_b64?: unknown }

    if (typeof row.thumb_b64 !== 'string' || !row.thumb_b64) {
      continue
    }

    if (imageChars + row.thumb_b64.length > MONITOR_EVIDENCE_MAX_B64_CHARS) {
      break
    }

    imageChars += row.thumb_b64.length
    frames.push({
      ts: typeof row.ts === 'number' && Number.isFinite(row.ts) ? row.ts : 0,
      source_type:
        typeof row.source_type === 'string' ? row.source_type.slice(0, SOURCE_TYPE_MAX_CHARS) : '',
      thumb_b64: row.thumb_b64
    })
  }

  if (frames.length === 0) {
    return undefined
  }

  // `input_count` is what the model actually saw; never let it read as fewer
  // than what we're showing, even if the payload disagrees.
  const count =
    typeof raw.input_count === 'number' && Number.isFinite(raw.input_count)
      ? Math.max(frames.length, Math.min(INPUT_COUNT_MAX, Math.round(raw.input_count)))
      : frames.length

  return { input_count: count, shown_count: frames.length, frames }
}
