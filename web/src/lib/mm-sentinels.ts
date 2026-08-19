/** Language-neutral sentinel strings shared with the multimodal backend.
 *
 * These mirror `agent/multimodal/_sentinels.py` verbatim. They are PROTOCOL
 * VALUES, not UI copy — never translate them and never route them through i18n.
 * The backend emits these exact strings; comparing against anything else makes
 * the filter silently no-op (which is exactly what happened when these were
 * hard-coded Chinese and the backend moved to English).
 *
 * Any change here must be made in `_sentinels.py` at the same time.
 */

/** `thought` placeholders the backend synthesizes when the model's own thought
 *  is empty. Not real scene descriptions → filtered out of segment description
 *  rows so a card doesn't show a content-free line. */
export const SYNTH_THOUGHT_DIRECT = "This segment can be answered directly from the frames.";
export const SYNTH_THOUGHT_CONTINUE = "Continue inspecting this segment.";
export const SYNTH_THOUGHTS: ReadonlySet<string> = new Set([
  SYNTH_THOUGHT_DIRECT,
  SYNTH_THOUGHT_CONTINUE,
]);

/** `findings` fallback when Recall finds nothing. Used to derive found/not-found
 *  styling from a findings preview. */
export const RECALL_NO_CLUES = "(no relevant clues found in memory)";

/** True when `saw` is a backend-synthesized placeholder rather than a real
 *  description. Trims first — the backend collapses whitespace but callers may
 *  pass raw text. */
export function isSynthSaw(saw: string | undefined | null): boolean {
  return !!saw && SYNTH_THOUGHTS.has(saw.trim());
}
