import { useStore } from '@nanostores/react'
import { useEffect, useRef } from 'react'

import { useI18n } from '@/i18n'
import { Square } from '@/lib/icons'
import { $gateway } from '@/store/gateway'
import {
  $mmMessages,
  attachMultimodalGateway,
  ensureMultimodalSession
} from '@/store/multimodal'
import { $mmTtsPlaying, stopAllTts } from '@/store/multimodal-voice'

import { PAGE_INSET_X } from '../layout-constants'
import { Composer } from './composer'
import { DeepPanel } from './deep-panel'
import { ObservationPanels } from './observation-panels'
import { VideoStage } from './video-stage'
import { Waterfall } from './waterfall'

/**
 * MultimodalView — the desktop port of the web multimodal video page.
 *
 * Staged build (see plan):
 *   Phase 0: route + page skeleton.               [done]
 *   Phase 1 (this): text chat over the multimodal protocol — dedicated
 *            session.create, prompt.submit + deep-think, message.* rendering,
 *            inline clarify. Media/panels land in later phases.
 *   Phase 2: camera / screen capture + multimodal.frame push.
 */
export function MultimodalView() {
  const { t } = useI18n()
  const gateway = useStore($gateway)
  const messages = useStore($mmMessages)
  const ttsPlaying = useStore($mmTtsPlaying)
  const scrollRef = useRef<HTMLDivElement | null>(null)

  // Wire gateway events + create/refresh the dedicated session on mount and
  // whenever the active gateway changes (profile switch swaps $gateway).
  // attachMultimodalGateway is idempotent and rebinds itself on a gateway
  // change; ensureMultimodalSession rebuilds the session if it belonged to a
  // different gateway.
  //
  // We deliberately do NOT reset on unmount. The chat log, session, capture and
  // mic all live at MODULE scope (this page is just an observer), so navigating
  // away to another view — a plain in-app route change — must NOT wipe them.
  // Otherwise switching to a new session and back loses the whole conversation.
  // A true reset (gateway/profile swap, reconnect) is handled inside the store.
  useEffect(() => {
    if (!gateway) return
    attachMultimodalGateway()
    void ensureMultimodalSession()
  }, [gateway])

  // Auto-scroll to newest.
  useEffect(() => {
    const el = scrollRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [messages])

  return (
    // ★ Make the whole multimodal page text-selectable/copyable. The desktop
    // app sets `user-select: none` globally on <body> (native feel), which also
    // killed copy in chat/observation/deep panels. This opts the page back in;
    // the global CSS still keeps buttons/icons/media non-selectable.
    <div
      data-selectable-text="true"
      className={`flex h-full min-h-0 flex-col gap-3 py-6 ${PAGE_INSET_X}`}
    >
      <header className="flex items-center gap-2">
        <h1 className="text-lg font-semibold text-(--ui-text-primary)">{t.multimodal.heading}</h1>
        {ttsPlaying && (
          <button
            className="ml-auto inline-flex items-center gap-1 rounded-full bg-sky-500/15 px-2 py-0.5 text-xs text-sky-400 hover:bg-sky-500/25"
            onClick={() => stopAllTts()}
            title={t.multimodal.chat.stopReading}
          >
            <Square className="size-3" /> {t.multimodal.chat.stopReading}
          </button>
        )}
      </header>

      <div className="flex min-h-0 flex-1 gap-4">
        <div className="flex min-h-0 flex-1 flex-col">
          <div className="min-h-0 flex-1 overflow-y-auto" ref={scrollRef}>
            {messages.length === 0 ? (
              <div className="grid h-full place-items-center px-6 text-center text-sm text-(--ui-text-tertiary)">
                <div className="max-w-sm">
                  <div className="mb-1 text-base font-medium text-(--ui-text-secondary)">
                    {t.multimodal.welcome.title}
                  </div>
                  {t.multimodal.welcome.body}
                </div>
              </div>
            ) : (
              <Waterfall />
            )}
          </div>
          <div className="mx-auto w-full max-w-3xl">
            <Composer />
          </div>
        </div>
        {/* Right rail: video preview + controls on top; monitor / deep-analysis
            windows below (self-hide until populated). Mirrors the web layout. */}
        <aside className="hidden w-80 shrink-0 flex-col gap-3 overflow-y-auto border-l border-(--ui-stroke-secondary) pl-4 lg:flex">
          <VideoStage />
          <ObservationPanels />
          <DeepPanel />
        </aside>
      </div>
    </div>
  )
}
