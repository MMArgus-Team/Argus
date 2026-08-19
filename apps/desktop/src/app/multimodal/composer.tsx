import { useStore } from '@nanostores/react'
import { type KeyboardEvent, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { useI18n } from '@/i18n'
import { Loader2, Mic, Send, Square, Volume2 } from '@/lib/icons'
import {
  $mmGenerating,
  interruptMultimodal,
  sendMultimodalPrompt
} from '@/store/multimodal'
import {
  $mmAsrBuffer,
  $mmAsrPartial,
  $mmMicError,
  $mmMicState,
  $mmTtsEnabled,
  finishMicTurn,
  startMic,
  stopMic,
  toggleMultimodalTts
} from '@/store/multimodal-voice'

/**
 * Slim composer for the multimodal page: a text field + mic / TTS toggles.
 * Intentionally NOT the heavy session-bound desktop composer — this page owns
 * its own session.
 *
 * Camera / screen capture controls live in VideoStage (right rail), not here.
 */
export function Composer() {
  const { t: tr } = useI18n()
  const c = tr.multimodal.composer
  const ttsEnabled = useStore($mmTtsEnabled)
  const micState = useStore($mmMicState)
  const micError = useStore($mmMicError)
  const asrPartial = useStore($mmAsrPartial)
  const asrBuffer = useStore($mmAsrBuffer)
  const generating = useStore($mmGenerating)
  const [capError, setCapError] = useState('')
  const [text, setText] = useState('')

  const toggleMic = () => {
    if (micState === 'finalizing') {
      return
    }

    if (micState === 'connecting') {
      void stopMic()

      return
    }

    if (micState === 'recording') {
      setCapError('')
      void finishMicTurn().catch(error => {
        setCapError(c.asrFailed(error instanceof Error ? error.message : String(error)))
      })
    } else {
      setCapError('')
      void startMic().catch(e => setCapError(c.micError(e instanceof Error ? e.message : String(e))))
    }
  }

  const submit = () => {
    if (generating) {return} // don't stack a new turn while one is streaming
    const t = text.trim()

    if (!t) {return}
    void sendMultimodalPrompt(t)
    setText('')
  }

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  const recording = micState === 'recording'
  const connecting = micState === 'connecting'
  const finalizing = micState === 'finalizing'

  const micLabel = recording
    ? c.micLabelEndSend
    : finalizing
      ? c.micLabelFinalizing
      : connecting
        ? c.micLabelConnecting
        : c.micLabelStart

  const bufferedAsr = asrBuffer.join(' ').trim()

  return (
    <div className="flex flex-col gap-2 rounded-xl border border-(--ui-stroke-secondary) bg-(--ui-bg-elevated) p-2 shadow-sm">
      {/* Live ASR partial preview (above the field, like a caption). */}
      {(recording || finalizing || asrPartial || bufferedAsr) && (
        <div className="flex items-center gap-2 px-1 text-xs text-(--ui-text-tertiary)">
          {recording && <span className="size-1.5 shrink-0 animate-pulse rounded-full bg-(--ui-red)" />}
          {finalizing && <Loader2 className="size-3 shrink-0 animate-spin" />}
          <span className="truncate">
            {bufferedAsr ? (
              <>
                <span className="opacity-60">{bufferedAsr}</span>
                {asrPartial ? <span className="ml-1">{asrPartial}</span> : null}
              </>
            ) : (
              asrPartial || (finalizing ? c.finalizingRecognition : c.listening)
            )}
          </span>
        </div>
      )}

      {/* One-row control bar (web-aligned):
            LEFT  toggles: 语音(Mic) — solid when on, outline when off.
            MIDDLE: text field. RIGHT: Send ↔ Stop. */}
      <div className="flex items-center gap-1.5">
        {/* 语音: outline idle · destructive recording · spinner connecting. Runs
            module-scoped in the store, so it survives the window being hidden. */}
        <Button
          aria-label={micLabel}
          aria-pressed={recording}
          className="shrink-0"
          disabled={finalizing}
          onClick={toggleMic}
          size="icon-sm"
          title={`${micLabel}${finalizing ? '…' : ''}`}
          variant={recording ? 'destructive' : 'outline'}
        >
          {connecting || finalizing ? <Loader2 className="size-4 animate-spin" /> : <Mic className="size-4" />}
        </Button>
        {/* 语音播报: solid when ON — 自动朗读主 agent / 深度分析的分析内容。 */}
        <Button
          aria-label={ttsEnabled ? c.ttsAriaOn : c.ttsAriaOff}
          aria-pressed={ttsEnabled}
          className="shrink-0"
          onClick={toggleMultimodalTts}
          size="icon-sm"
          title={ttsEnabled ? c.ttsOnTitle : c.ttsOffTitle}
          variant={ttsEnabled ? 'default' : 'outline'}
        >
          <Volume2 className="size-4" />
        </Button>

        <Textarea
          className="max-h-24 min-h-8 flex-1 resize-none self-center border-0 bg-transparent px-1 shadow-none focus-visible:ring-0"
          onChange={e => setText(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder={c.askWithVideoPlaceholder}
          rows={1}
          value={text}
        />

        {/* Send ↔ Stop: while a turn streams, the primary button interrupts it. */}
        {generating ? (
          <Button className="shrink-0" onClick={() => void interruptMultimodal()} size="sm" title={c.stopGenerating} variant="destructive">
            <Square className="mr-1 size-3.5" /> {c.stopShort}
          </Button>
        ) : (
          <Button className="shrink-0" disabled={!text.trim()} onClick={submit} size="sm">
            <Send className="mr-1 size-3.5" /> {c.send}
          </Button>
        )}
      </div>

      {(capError || micError) && <div className="px-1 text-xs text-(--ui-red)">{capError || micError}</div>}
    </div>
  )
}
