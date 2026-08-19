import { atom } from 'nanostores'

/**
 * Window-resize settling gate.
 *
 * 大 session (100+ turns 已 mount) 时, 拖动窗口每一 tick 都触发所有 message 的
 * ResizeObserver 回调 (每 turn 一个 measureClamp), 导致 100+ 次 setState 同 tick 涌上,
 * React 重排风暴 = 你看到的"窗口外框动了但内容跟不上, 缓好久"。
 *
 * 修法: 一个全局 flag —— 拖动期间 = true, 停手 150ms 后 = false。measureClamp 读它,
 * true 时只缓存新尺寸不 setState, false (settle) 时最后一次真的 flush。
 *
 * 单一 window resize listener 驱动, 全 desktop 共享 (nanostores 原子, 订阅者自动同步)。
 */
export const $isWindowResizing = atom(false)

let settleTimer: ReturnType<typeof setTimeout> | null = null
let installed = false

const SETTLE_MS = 150

function onResize(): void {
  if (!$isWindowResizing.get()) $isWindowResizing.set(true)
  if (settleTimer) clearTimeout(settleTimer)
  settleTimer = setTimeout(() => {
    settleTimer = null
    $isWindowResizing.set(false)
  }, SETTLE_MS)
}

/**
 * Install the single window-resize listener. Idempotent; call from app bootstrap
 * (or on first import — this module auto-installs on import in browser envs).
 */
export function installWindowResizeGate(): void {
  if (installed) return
  if (typeof window === 'undefined') return
  installed = true
  window.addEventListener('resize', onResize, { passive: true })
}

// Auto-install on import in browser/Electron renderer (harmless no-op in SSR/tests).
installWindowResizeGate()
