/**
 * Pure geometry helpers for window-state.json — restoring the main window's
 * size, position, and maximized flag across launches. Side-effect-free so the
 * part that actually matters (rejecting garbage + off-screen bounds) is
 * unit-testable without booting Electron; main.cjs owns the file I/O and the
 * live `screen` displays.
 */

// A fresh install starts maximized (see shouldStartMaximized), so DEFAULT_* is
// the *unmaximized* restore size — what the window becomes when the user clicks
// zoom off. Keep it wide enough for the two-column shell (preview rail + chat);
// at 600 the chat column collapses to ~200px and wraps to one word per line.
const DEFAULT_WIDTH = 1220
const DEFAULT_HEIGHT = 800
// MIN_* mirror the live window's minWidth/minHeight so a restored size never
// undershoots what the window allows. MIN_WIDTH must clear the shell's own
// minimums: the preview rail (PREVIEW_RAIL_MIN_WIDTH, 18rem = 288px) plus the
// sidebar (237px) already eat 525px, so anything near 400 leaves chat unusable.
const MIN_WIDTH = 900
const MIN_HEIGHT = 620

// Keep at least this much of the window over a display work area before we trust
// a saved position, so the title bar stays grabbable after a monitor unplugs.
const MIN_VISIBLE = 48

const finite = v => typeof v === 'number' && Number.isFinite(v)
const clamp = (v, lo, hi) => Math.max(lo, Math.min(v, hi))

// Parse raw JSON → clean state, or null if garbage. width/height are required
// and floored; x/y survive only as a finite pair; isMaximized is strict.
//
// `wasTooNarrow` records that the *saved* width was below MIN_WIDTH, i.e. the
// geometry predates MIN_WIDTH covering the two-column shell. Flooring the width
// alone would leave such a window un-maximized at an arbitrary size, so
// shouldStartMaximized() uses this to treat the state as unusable rather than as
// a deliberate user choice.
function sanitizeWindowState(raw) {
  if (!raw || typeof raw !== 'object' || !finite(raw.width) || !finite(raw.height)) return null

  const state = {
    width: Math.max(MIN_WIDTH, Math.round(raw.width)),
    height: Math.max(MIN_HEIGHT, Math.round(raw.height)),
    isMaximized: raw.isMaximized === true,
    wasTooNarrow: Math.round(raw.width) < MIN_WIDTH
  }
  if (finite(raw.x) && finite(raw.y)) {
    state.x = Math.round(raw.x)
    state.y = Math.round(raw.y)
  }
  return state
}

// True when `bounds` overlaps some display's work area by ≥ MIN_VISIBLE on both
// axes. `displays` is Electron's screen.getAllDisplays() shape.
function onScreen(bounds, displays) {
  if (!Array.isArray(displays)) return false
  return displays.some(({ workArea: a } = {}) => {
    if (!a) return false
    const x = Math.min(bounds.x + bounds.width, a.x + a.width) - Math.max(bounds.x, a.x)
    const y = Math.min(bounds.y + bounds.height, a.y + a.height) - Math.max(bounds.y, a.y)
    return x >= MIN_VISIBLE && y >= MIN_VISIBLE
  })
}

// Sanitized state (or null) → BrowserWindow size/position options. Always sets
// width/height, capped to the largest current display so a size saved on a
// since-disconnected bigger monitor can't exceed any screen the user now has.
// Sets x/y only when still on-screen; otherwise Electron centers the window.
function computeWindowOptions(state, displays) {
  const opts = {
    width: finite(state?.width) ? state.width : DEFAULT_WIDTH,
    height: finite(state?.height) ? state.height : DEFAULT_HEIGHT
  }

  const cap = (Array.isArray(displays) ? displays : []).reduce(
    (m, { workArea: a } = {}) =>
      a && finite(a.width) && finite(a.height)
        ? { width: Math.max(m.width, a.width), height: Math.max(m.height, a.height) }
        : m,
    { width: 0, height: 0 }
  )
  if (cap.width && cap.height) {
    opts.width = clamp(opts.width, MIN_WIDTH, cap.width)
    opts.height = clamp(opts.height, MIN_HEIGHT, cap.height)
  }

  if (
    state &&
    finite(state.x) &&
    finite(state.y) &&
    onScreen({ x: state.x, y: state.y, width: opts.width, height: opts.height }, displays)
  ) {
    opts.x = state.x
    opts.y = state.y
  }
  return opts
}

// Should the main window open maximized? A fresh install (no saved state) does,
// so the two-column shell gets the full work area instead of the DEFAULT_WIDTH
// box. Once the user has a saved state we respect their choice exactly — an
// explicit un-maximize must survive the next launch, so absent/false both mean
// "don't maximize" here and only a literal `true` opts back in.
//
// Exception: a state whose saved width was narrower than MIN_WIDTH can't be a
// deliberate choice, because the window could never have been dragged that
// narrow — it was written while DEFAULT_WIDTH was briefly 600 (too narrow for
// the shell; chat wrapped to one word per line). Maximize those instead of
// restoring a size the layout can't use.
function shouldStartMaximized(state) {
  if (!state) return true
  if (state.wasTooNarrow) return true
  return state.isMaximized === true
}

// Trailing debounce: collapse a burst of resize/move events (Linux fires many
// mid-drag) into a single run `delayMs` after the last. `.flush()` runs now and
// cancels the pending timer — used on close, before the window is gone.
function debounce(fn, delayMs) {
  let timer = null
  const debounced = () => {
    clearTimeout(timer)
    timer = setTimeout(() => {
      timer = null
      fn()
    }, delayMs)
  }
  debounced.flush = () => {
    clearTimeout(timer)
    timer = null
    fn()
  }
  return debounced
}

module.exports = {
  DEFAULT_WIDTH,
  DEFAULT_HEIGHT,
  MIN_WIDTH,
  MIN_HEIGHT,
  MIN_VISIBLE,
  sanitizeWindowState,
  onScreen,
  computeWindowOptions,
  shouldStartMaximized,
  debounce
}
