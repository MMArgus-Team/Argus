import { type ReactNode, useState } from 'react'

import { DisclosureRow } from '@/components/chat/disclosure-row'
import { cn } from '@/lib/utils'

/**
 * MmDisclosure — a collapsible block for the multimodal page (thinking, tool
 * call, deep-research window). Reuses the SAME shared DisclosureRow primitive as
 * the main chat thread (hover-caret, content-shaped hit-target, trailing slot)
 * so it reads as part of the product, not a bespoke widget.
 *
 * Behaviour mirrors the main chat's ThinkingDisclosure:
 *   • `defaultOpen` seeds the initial state (e.g. open while live).
 *   • `syncOpen` re-drives open/closed while the caller's live state changes
 *     UNTIL the user makes an explicit toggle — after that the user wins. This
 *     is the "auto-open while streaming, auto-collapse when done" feel without
 *     yanking the panel out from under a user who opened it manually.
 */
export function MmDisclosure({
  action,
  children,
  defaultOpen = false,
  syncOpen,
  title,
  trailing
}: {
  action?: ReactNode
  children: ReactNode
  defaultOpen?: boolean
  /** Live-driven open state; ignored once the user toggles manually. */
  syncOpen?: boolean
  title: ReactNode
  trailing?: ReactNode
}) {
  // null = no explicit user toggle yet → follow syncOpen (or defaultOpen).
  const [userOpen, setUserOpen] = useState<boolean | null>(null)
  const open = userOpen ?? syncOpen ?? defaultOpen

  return (
    <div className="text-[length:var(--conversation-tool-font-size)] text-(--ui-text-tertiary)">
      <DisclosureRow action={action} onToggle={() => setUserOpen(!open)} open={open} trailing={trailing}>
        {title}
      </DisclosureRow>
      {open && (
        <div className="mt-1 w-full min-w-0 max-w-full overflow-hidden wrap-anywhere pb-1 leading-(--conversation-line-height)">
          {children}
        </div>
      )}
    </div>
  )
}

/** Shimmering title text for a live (in-progress) disclosure, matching the main
 * chat's "Thinking…" treatment. */
export function DisclosureTitle({ live, children }: { live?: boolean; children: ReactNode }) {
  return (
    <span
      className={cn(
        'text-[length:var(--conversation-tool-font-size)] font-medium leading-(--conversation-line-height)',
        live ? 'shimmer text-foreground/60' : 'text-(--ui-text-secondary)'
      )}
    >
      {children}
    </span>
  )
}
