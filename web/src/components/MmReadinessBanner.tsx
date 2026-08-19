/**
 * MmReadinessBanner — a non-blocking advisory shown at the top of the
 * multimodal page when a REQUIRED multimodal capability is missing.
 *
 * Design (see decisions in the onboarding review):
 *   * SOFT, not a hard gate — the user still gets into the page (semantics unified
 *     with the desktop card). Missing capabilities just mean the corresponding
 *     features won't work, and we say so.
 *   * Does NOT open its own gateway connection. The host page already has a
 *     GatewayClient; it fetches `mm.readiness` once on connect and passes the
 *     report in as a prop. This avoids the previous "connect twice" cost (a
 *     throwaway WS for the gate + the page's own WS).
 *   * Dismissable for the session (sessionStorage) — reappears next launch so a
 *     genuinely-broken install keeps reminding, but doesn't nag within a session.
 *   * Renders nothing when ready, when there are no missing REQUIRED caps, or
 *     before the report has loaded — so a probe hiccup never blocks the page.
 */

import { useState } from "react";
import { useI18n } from "@/i18n";

export type MmCapStatus = "ok" | "missing" | "broken" | "unknown";

export interface MmCapability {
  key: string;
  label: string;
  status: MmCapStatus;
  required: boolean;
  reason: string;
  fix: string;
}

export interface MmReadinessReport {
  ready: boolean;
  capabilities: MmCapability[];
}

const DISMISS_KEY = "argus-mm-readiness-dismissed-session";

/**
 * Pure decision logic (view-independent, unit-tested): given a readiness report,
 * return the REQUIRED capabilities that aren't ok. An empty array means the
 * banner should render nothing (ready, or only optional gaps). Null/malformed
 * report → empty (fail safe: never show a broken banner).
 */
export function selectMissingRequired(
  report: MmReadinessReport | null,
): MmCapability[] {
  if (!report || report.ready || !Array.isArray(report.capabilities)) {
    return [];
  }
  return report.capabilities.filter((c) => c.required && c.status !== "ok");
}

function wasDismissed(): boolean {
  try {
    return window.sessionStorage.getItem(DISMISS_KEY) === "1";
  } catch {
    return false;
  }
}

export function MmReadinessBanner({ report }: { report: MmReadinessReport | null }) {
  const { t } = useI18n();
  const [dismissed, setDismissed] = useState<boolean>(wasDismissed);
  const [expanded, setExpanded] = useState(false);

  const missingRequired = selectMissingRequired(report);
  if (dismissed || missingRequired.length === 0) {
    return null;
  }

  const dismiss = () => {
    try {
      window.sessionStorage.setItem(DISMISS_KEY, "1");
    } catch {
      /* ignore — advisory is best-effort */
    }
    setDismissed(true);
  };

  // Floating, theme-aware card pinned to the top-right so it never blocks
  // page content. Uses tailwind theme classes (text-foreground / bg-background
  // / border-border) so it follows the app's light/dark palette instead of
  // hard-coded --ui-* fallbacks that clashed with the resolved theme.
  return (
    <div
      role="status"
      className="pointer-events-auto fixed right-4 top-4 z-[80] w-[380px] max-w-[calc(100vw-2rem)] rounded-lg border border-amber-400/40 bg-background/95 text-foreground shadow-xl shadow-black/20 backdrop-blur"
    >
      <div className="flex items-start gap-2 px-3.5 py-2.5 text-sm">
        <span aria-hidden className="mt-0.5 select-none text-amber-400">⚠</span>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="font-medium">{t.multimodal.readiness.notReady}</span>
            <span className="rounded-full bg-amber-400/15 px-1.5 py-0.5 text-[10px] font-medium text-amber-500 ring-1 ring-amber-400/40">
              {missingRequired.length}
            </span>
            <button
              onClick={() => setExpanded((v) => !v)}
              className="ml-auto rounded px-1.5 py-0.5 text-[11px] text-muted-foreground hover:bg-muted/50 hover:text-foreground"
              aria-label={expanded ? t.common.collapse : t.common.expand}
            >
              {expanded ? t.common.collapse : t.common.expand}
            </button>
            <button
              onClick={dismiss}
              aria-label={t.common.close}
              className="rounded p-0.5 text-muted-foreground hover:bg-muted/50 hover:text-foreground"
            >
              ×
            </button>
          </div>
          {expanded && (
            <div className="mt-2 space-y-2 border-t border-border/60 pt-2 text-[12px] leading-relaxed">
              <p className="text-muted-foreground">
                {t.multimodal.readiness.capsMissing}{" "}
                <code className="rounded bg-muted/60 px-1 py-0.5 font-mono text-foreground">
                  argus setup multimodal
                </code>{" "}
                {t.multimodal.readiness.toFix}
              </p>
              <ul className="space-y-1.5">
                {missingRequired.map((c) => (
                  <li key={c.key} className="rounded border border-border/50 bg-muted/20 px-2 py-1.5">
                    <div className="font-medium">{c.label}</div>
                    {c.reason && (
                      <div className="mt-0.5 text-muted-foreground">{c.reason}</div>
                    )}
                    {c.fix && (
                      <code className="mt-1 block break-all rounded bg-muted/40 px-1.5 py-1 font-mono text-[11px] text-foreground/90">
                        {c.fix}
                      </code>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
