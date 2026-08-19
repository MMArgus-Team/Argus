/**
 * DesktopMmReadinessCard — a non-blocking advisory shown after provider
 * onboarding when the multimodal subsystem isn't fully ready.
 *
 * Desktop merges the multimodal capabilities into the main chat, so this does
 * NOT block the app — it surfaces a dismissable card listing missing REQUIRED
 * capabilities and how to fix them (via `argus setup multimodal` or the printed
 * command).
 *
 * Dismissal is scoped to the SESSION (sessionStorage), matching the web banner:
 * dismiss stops the nag for this run, but a genuinely-broken install reminds
 * again next launch — so the advisory can't be permanently lost (the earlier
 * localStorage version meant one stray click hid it forever).
 *
 * Readiness comes from the backend `mm.readiness` RPC — the same single source
 * of truth as `hermes mm doctor` and the web banner. Fails silent (renders
 * nothing) on any probe/transport error: an advisory must never itself break
 * the app.
 */

import { useEffect, useState } from 'react'

import { useI18n } from '@/i18n'

export type CapStatus = 'ok' | 'missing' | 'broken' | 'unknown'

export interface Capability {
  key: string
  label: string
  status: CapStatus
  required: boolean
  reason: string
  fix: string
}

export interface ReadinessReport {
  ready: boolean
  capabilities: Capability[]
}

type Requester = <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>

/**
 * Pure decision logic (view-independent, unit-tested): the REQUIRED capabilities
 * that aren't ok. Empty → render nothing. Null/malformed → empty (fail safe).
 */
export function selectMissingRequired(report: ReadinessReport | null): Capability[] {
  if (!report || report.ready || !Array.isArray(report.capabilities)) {
    return []
  }
  return report.capabilities.filter(c => c.required && c.status !== 'ok')
}

const DISMISS_KEY = 'argus-desktop-mm-readiness-dismissed-session'

function wasDismissed(): boolean {
  try {
    // Session-scoped: reappears next launch so a broken install keeps reminding.
    return window.sessionStorage.getItem(DISMISS_KEY) === '1'
  } catch {
    return false
  }
}

function rememberDismissed() {
  try {
    window.sessionStorage.setItem(DISMISS_KEY, '1')
  } catch {
    // ignore — advisory is best-effort
  }
}

export function DesktopMmReadinessCard({
  requestGateway,
  active
}: {
  requestGateway: Requester
  /** Only probe once onboarding is done and the app is usable. */
  active: boolean
}) {
  const [report, setReport] = useState<ReadinessReport | null>(null)
  const [dismissed, setDismissed] = useState<boolean>(wasDismissed)

  useEffect(() => {
    if (!active || dismissed) {
      return
    }
    let alive = true
    // probe_endpoints=true opts in to the deep readiness probes:
    //   * LLM endpoint TCP reachability (main / monitor / watcher / memory / …)
    //   * auxiliary.text local weights + remote_backend endpoint
    //   * auxiliary.ocr rapidocr package + remote_backend endpoint
    //   * Qwen2.5-0.5B background preload (fire-and-forget)
    // so the card warns "your endpoint is unreachable — requests will hang"
    // instead of the user hitting the mysterious "agent initialization timed
    // out" wall on the first prompt.
    requestGateway<ReadinessReport>('mm.readiness', { probe_endpoints: true })
      .then(r => {
        if (alive && r && typeof r.ready === 'boolean') {
          setReport(r)
        }
      })
      .catch(() => {
        // fail silent
      })
    return () => {
      alive = false
    }
  }, [active, dismissed, requestGateway])

  const missingRequired = selectMissingRequired(report)
  if (dismissed || missingRequired.length === 0) {
    return null
  }

  const dismiss = () => {
    rememberDismissed()
    setDismissed(true)
  }

  return (
    <ReadinessFloating
      count={missingRequired.length}
      items={missingRequired}
      onDismiss={dismiss}
    />
  )
}

/** Floating readiness card, styled with the desktop UI variables so it follows
 *  the app's theme (no hardcoded fallback that clashed with the resolved theme).
 *  Pinned top-right. Collapsed by default; "详情" expands the missing-cap list. */
function ReadinessFloating({
  count, items, onDismiss,
}: {
  count: number
  items: Capability[]
  onDismiss: () => void
}) {
  const { t } = useI18n()
  const [expanded, setExpanded] = useState(false)
  return (
    <div
      className="pointer-events-auto fixed right-4 top-4 z-[80] w-[380px] max-w-[calc(100vw-2rem)] rounded-lg border shadow-xl backdrop-blur"
      style={{
        borderColor: 'var(--ui-yellow)',
        background: 'color-mix(in srgb, var(--ui-bg-elevated) 92%, transparent)',
        color: 'var(--ui-text-primary)',
      }}
    >
      <div className="flex items-start gap-2 px-3.5 py-2.5 text-[13px]">
        <span aria-hidden className="mt-0.5 select-none" style={{ color: 'var(--ui-yellow)' }}>⚠</span>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="font-medium">{t.multimodal.readiness.notReady}</span>
            <span
              className="rounded-full px-1.5 py-0.5 text-[10px] font-medium"
              style={{
                background: 'color-mix(in srgb, var(--ui-yellow) 20%, transparent)',
                color: 'var(--ui-yellow)',
                boxShadow: '0 0 0 1px color-mix(in srgb, var(--ui-yellow) 45%, transparent)',
              }}
            >{count}</span>
            <button
              onClick={() => setExpanded(v => !v)}
              className="ml-auto rounded px-1.5 py-0.5 text-[11px] hover:bg-(--ui-bg-elevated)"
              style={{ color: 'var(--ui-text-tertiary)' }}
              aria-label={expanded ? t.multimodal.readiness.collapse : t.multimodal.readiness.expand}
            >{expanded ? t.multimodal.readiness.collapse : t.multimodal.readiness.details}</button>
            <button
              onClick={onDismiss}
              aria-label={t.multimodal.readiness.close}
              className="rounded p-0.5 hover:bg-(--ui-bg-elevated)"
              style={{ color: 'var(--ui-text-tertiary)' }}
            >×</button>
          </div>
          {expanded && (
            <div
              className="mt-2 space-y-2 border-t pt-2 text-[12px] leading-relaxed"
              style={{ borderColor: 'var(--ui-stroke-secondary)' }}
            >
              <p style={{ color: 'var(--ui-text-tertiary)' }}>
                {t.multimodal.readiness.capsMissing}{' '}
                <code
                  className="rounded px-1 py-0.5 font-mono"
                  style={{
                    background: 'color-mix(in srgb, var(--ui-text-primary) 8%, transparent)',
                    color: 'var(--ui-text-primary)',
                  }}
                >{t.multimodal.readiness.runSetup}</code>{' '}
                {t.multimodal.readiness.toFix}
              </p>
              <ul className="space-y-1.5">
                {items.map(c => (
                  <li
                    key={c.key}
                    className="rounded border px-2 py-1.5"
                    style={{
                      borderColor: 'var(--ui-stroke-secondary)',
                      background: 'color-mix(in srgb, var(--ui-text-primary) 3%, transparent)',
                    }}
                  >
                    <div className="font-medium">{c.label}</div>
                    {c.reason && (
                      <div className="mt-0.5" style={{ color: 'var(--ui-text-tertiary)' }}>{c.reason}</div>
                    )}
                    {c.fix && (
                      <code
                        className="mt-1 block break-all rounded px-1.5 py-1 font-mono text-[11px]"
                        style={{
                          background: 'color-mix(in srgb, var(--ui-text-primary) 6%, transparent)',
                          color: 'var(--ui-text-primary)',
                        }}
                      >{c.fix}</code>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
