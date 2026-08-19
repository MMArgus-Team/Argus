// Shims for browser APIs jsdom does not implement.
//
// These are NOT production polyfills — the renderer only ever runs in Electron's
// Chromium, where all of this exists natively. They exist so a missing jsdom
// feature surfaces as nothing at all instead of a `TypeError` thrown from deep
// inside a React effect, where it reads like an application bug.

// jsdom ships no `CSS` object at all (not just a missing `escape`), so this is
// an assignment rather than a method patch. Used by thread-timeline.tsx to build
// `[data-message-id="…"]` selectors from message ids.
//
// Mirrors the spec's serialization rules for the subset that can appear in an
// attribute-value selector; it is not a full CSS.escape implementation.
if (typeof globalThis.CSS === 'undefined') {
  ;(globalThis as { CSS?: unknown }).CSS = {
    escape: (value: string) =>
      String(value).replace(/[^\w-]/g, ch => {
        const code = ch.codePointAt(0) ?? 0

        // Null is not representable in a selector; the spec replaces it.
        return code === 0 ? '�' : `\\${ch}`
      })
  }
}
