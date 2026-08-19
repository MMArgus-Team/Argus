/**
 * Lightweight runtime hints for adapting expensive UI paths on weaker GPUs /
 * platform-specific compositor quirks (Mac Chrome + Retina is the main case).
 */

/** macOS desktop — includes Intel and Apple Silicon MacBooks. */
export function isMacOS(): boolean {
  if (typeof navigator === "undefined") return false;
  const p = navigator.platform || "";
  const ua = navigator.userAgent || "";
  return /Mac/i.test(p) || /\bMac OS X\b/.test(ua);
}

/** devicePixelRatio > 1 (Retina / HiDPI). Screen capture + canvas blits scale badly. */
export function isHiDPI(): boolean {
  if (typeof window === "undefined") return false;
  return (window.devicePixelRatio || 1) > 1.25;
}

/**
 * Prefer lower capture resolution / fewer compositor effects.
 * Mac Chrome on Retina is the primary target; HiDPI Windows gets the same caps.
 */
export function preferLightCapture(): boolean {
  return isMacOS() || isHiDPI();
}

/** Full-viewport mix-blend / SVG noise is disproportionately costly on Mac Chrome. */
export function preferReducedBackdropEffects(): boolean {
  if (typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    return true;
  }
  return isMacOS();
}
