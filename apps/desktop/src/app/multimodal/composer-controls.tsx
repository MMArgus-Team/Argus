import { useStore } from '@nanostores/react'
import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { Loader2, MessageSquareText, Mic, Volume2 } from '@/lib/icons'
import { pushMmToast } from '@/store/multimodal-deep'
import {
  $mmAsrBuffer,
  $mmAsrPartial,
  $mmMicError,
  $mmMicState,
  $mmTtsEnabled,
  $mmVoiceDialogEnabled,
  finishMicTurn,
  startMic,
  stopMic,
  toggleMultimodalTts,
  toggleMultimodalVoiceDialog,
} from '@/store/multimodal-voice'

/**
 * ASR live preview — "语音识别中 …" 条, 出现在输入框上方。对齐 web 的 AsrBar:
 * 麦克风录音中 (含对话模式自动开麦) 或有 partial 文本时显示。因为对话模式 ON 会
 * 自动开麦 → micState='recording', 所以对话模式下同样会显示识别预览。
 */
export function MultimodalAsrBar() {
  const { t } = useI18n()
  const c = t.multimodal.composer
  const micState = useStore($mmMicState)
  const partial = useStore($mmAsrPartial)
  const buffer = useStore($mmAsrBuffer)
  const recording = micState === 'recording'
  const finalizing = micState === 'finalizing'
  const buffered = buffer.join(' ').trim()

  if (!recording && !finalizing && !partial && !buffered) {
    return null
  }

  return (
    <div className="flex items-center gap-2 px-1 pb-1 text-xs text-muted-foreground">
      {recording && <span className="size-1.5 shrink-0 animate-pulse rounded-full bg-red-500" />}
      {finalizing && <Loader2 className="size-3 shrink-0 animate-spin" />}
      <span className="truncate">
        {buffered ? (
          <>
            <span className="opacity-60">{buffered}</span>
            {partial ? <span className="ml-1">{partial}</span> : null}
          </>
        ) : (
          partial || (finalizing ? c.finalizingRecognition : c.listening)
        )}
      </span>
    </div>
  )
}

/**
 * The multimodal toggles injected to the LEFT (outer) of the main ChatBar input,
 * alongside the add-context menu:
 *   - 语音: streaming ASR (startMic/stopMic). Outline idle · destructive
 *     recording · spinner connecting — mirrors the multimodal composer.
 *   - 语音播报 / 对话模式.
 *
 * These are additive; add-context / model / send stay ChatBar-native.
 */
export function MultimodalComposerControls() {
  const { t } = useI18n()
  const c = t.multimodal.composer
  const ttsEnabled = useStore($mmTtsEnabled)
  const voiceDialogEnabled = useStore($mmVoiceDialogEnabled)
  const micState = useStore($mmMicState)
  const asyncMicError = useStore($mmMicError)
  const [micError, setMicError] = useState('')

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

  const toggleMic = () => {
    // ★ 对话模式开时麦由对话托管, 单独点无效 → 拦截 + 小提示 (按钮态不变)。
    if (voiceDialogEnabled) {
      pushMmToast({ level: 'info', text: c.dialogToastMicBlocked })

      return
    }

    if (finalizing) {
      return
    }

    if (connecting) {
      void stopMic()

      return
    }

    if (recording) {
      setMicError('')
      void finishMicTurn().catch(error => {
        const message = error instanceof Error ? error.message : String(error)

        setMicError(message)
        pushMmToast({ level: 'error', text: c.asrFailed(message) })
      })

      return
    }

    setMicError('')
    void startMic().catch(error => {
      const message = error instanceof Error ? error.message : String(error)

      setMicError(message)
      pushMmToast({ level: 'error', text: c.micError(message) })
    })
  }

  // ★ 对话模式开时喇叭由对话托管 (后端强制 TTS), 单独点无效 → 拦截 + 小提示。
  const toggleTtsGuarded = () => {
    if (voiceDialogEnabled) {
      pushMmToast({ level: 'info', text: c.dialogToastTtsBlocked })

      return
    }

    toggleMultimodalTts()
  }

  const toggleVoiceDialogGuarded = () => {
    if (!toggleMultimodalVoiceDialog()) {
      pushMmToast({ level: 'error', text: c.dialogNeedFinishMic })
    }
  }

  return (
    <div className="flex items-center gap-1">
      <Button
        aria-label={micLabel}
        aria-pressed={recording}
        className="size-7 shrink-0"
        disabled={finalizing}
        // ★ 对话模式开时麦由对话托管, 单独点无效 —— 拦截+提示在 toggleMic 里 (按钮态不变)。
        onClick={toggleMic}
        size="icon-sm"
        title={
          micError || asyncMicError
            ? c.micFailedTitle(micError || asyncMicError)
            : `${micLabel}${finalizing ? '…' : ''}`
        }
        type="button"
        variant={recording ? 'destructive' : 'outline'}
      >
        {connecting || finalizing ? <Loader2 className="size-4 animate-spin" /> : <Mic className="size-4" />}
      </Button>
      <Button
        aria-label={ttsEnabled ? c.ttsAriaOn : c.ttsAriaOff}
        aria-pressed={ttsEnabled}
        className="size-7 shrink-0"
        // ★ 对话模式开时喇叭由对话托管 (后端强制 TTS), 单独点无效 —— 拦截+提示在
        //   toggleTtsGuarded 里 (按钮态不变)。
        onClick={toggleTtsGuarded}
        size="icon-sm"
        title={ttsEnabled ? c.ttsOnTitle : c.ttsOffTitle}
        type="button"
        variant={ttsEnabled ? 'default' : 'outline'}
      >
        <Volume2 className="size-4" />
      </Button>
      {/* 对话模式 (Honey Amber #fbbf24) — 开 = VoiceAgent 分诊+秒回+防误识别;
          与 web MultimodalChatPage 同款 (Nous DS button.tsx 的实心 vs outlined,
          desktop shadcn 用 variant='default' vs 'outline' + className 覆盖色).
          cn = tailwind-merge, bg-amber-400 会覆盖默认 bg-primary。 */}
      <Button
        aria-pressed={voiceDialogEnabled}
        className={
          voiceDialogEnabled
            ? 'size-7 shrink-0 bg-amber-400 text-neutral-900 hover:bg-amber-500'
            : 'size-7 shrink-0 hover:text-amber-300'
        }
        onClick={toggleVoiceDialogGuarded}
        size="icon-sm"
        title={
          voiceDialogEnabled
            ? c.dialogOnTitle
            : c.dialogOffTitle
        }
        type="button"
        variant={voiceDialogEnabled ? 'default' : 'outline'}
      >
        <MessageSquareText className="size-4" />
      </Button>
    </div>
  )
}
