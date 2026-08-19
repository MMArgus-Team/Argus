import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { KbdCombo } from '@/components/ui/kbd'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { Layers3, SteeringWheel } from '@/lib/icons'
import { formatCombo } from '@/lib/keybinds/combo'
import { cn } from '@/lib/utils'

import { ModelPill } from './model-pill'
import type { ChatBarState } from './types'

export const ICON_BTN = 'size-(--composer-control-size) shrink-0 rounded-md'
export const GHOST_ICON_BTN = cn(
  ICON_BTN,
  'text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground'
)
// Send/voice-conversation primary: solid foreground-on-background circle
// (reads as black-on-white in light mode, white-on-black in dark mode) to
// match the reference composer's high-contrast CTA. Keeps the pill itself
// neutral and lets the action visually dominate the row.
export const PRIMARY_ICON_BTN = cn(
  'size-(--composer-control-primary-size,var(--composer-control-size)) shrink-0 rounded-full p-0',
  'bg-foreground text-background hover:bg-foreground/90',
  'disabled:bg-foreground/30 disabled:text-background disabled:opacity-100'
)

export function ComposerControls({
  busy,
  busyAction,
  canSteer,
  canSubmit,
  compactModelPill = false,
  disabled,
  hasComposerPayload,
  state,
  onSteer
}: {
  busy: boolean
  busyAction: 'queue' | 'stop'
  canSteer: boolean
  canSubmit: boolean
  compactModelPill?: boolean
  disabled: boolean
  hasComposerPayload: boolean
  state: ChatBarState
  onSteer: () => void
}) {
  const { t } = useI18n()
  const c = t.composer
  const steerCombo = formatCombo('mod+enter')
  const steerLabel = `${c.steer} (${steerCombo})`

  const steerTip = (
    <span className="inline-flex items-center gap-1.5">
      {c.steer}
      <KbdCombo combo="mod+enter" size="sm" variant="inverted" />
    </span>
  )

  return (
    <div className="ml-auto flex shrink-0 items-center gap-(--composer-control-gap)">
      <ModelPill compact={compactModelPill} disabled={disabled} model={state.model} />
      {/* Steer takes the slot while the agent runs and the user is typing.
          (The native dictation + voice-conversation controls were removed — the
          multimodal streaming mic on the composer's LEFT is the only voice
          affordance now.) */}
      {canSteer && (
        <Tip label={steerTip}>
          <Button
            aria-label={steerLabel}
            className={GHOST_ICON_BTN}
            disabled={disabled}
            onClick={onSteer}
            size="icon"
            type="button"
            variant="ghost"
          >
            <SteeringWheel size={14} />
          </Button>
        </Tip>
      )}
      {/* Primary is always Send / Stop / Queue now (no voice-primary toggle). */}
      <Tip label={busy ? (busyAction === 'queue' ? c.queueMessage : c.stop) : c.send}>
        <Button
          aria-label={busy ? (busyAction === 'queue' ? c.queueMessage : c.stop) : c.send}
          className={PRIMARY_ICON_BTN}
          disabled={disabled || !canSubmit}
          type="submit"
        >
          {busy ? (
            busyAction === 'queue' ? (
              <Layers3 size={14} />
            ) : (
              <span className="block size-2.5 rounded-[0.1875rem] bg-current" />
            )
          ) : (
            <Codicon name="arrow-up" size="0.875rem" />
          )}
        </Button>
      </Tip>
    </div>
  )
}

