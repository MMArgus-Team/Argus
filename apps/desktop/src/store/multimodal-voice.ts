import { atom } from 'nanostores'

import { translateNow } from '@/i18n'

import { $gateway } from './gateway'
import { $mmSessionId, addVoiceUserMessage } from './multimodal'

/**
 * Voice I/O for the multimodal page (desktop port of web startMic/stopMic +
 * onTtsChunk + env-audio):
 *   - Mic streaming ASR: getUserMedia → AudioWorklet(pcm-worklet.js) → 16k PCM
 *     batches → multimodal.asr_audio. Ordinary mic input is one explicit
 *     manual turn: partial/buffer events are preview-only, and the second click
 *     flushes + finishes before exactly one final is submitted. Voice Dialog
 *     keeps the existing continuous/VAD behavior.
 *   - TTS playback: multimodal.tts PCM16 chunks → WebAudio gapless scheduling.
 *   - Env audio: screen-share audio track → MediaRecorder 5s slices →
 *     multimodal.env_audio.
 *
 * Cross-platform: standard WebAudio / MediaRecorder / AudioWorklet (Electron
 * Chromium, identical on macOS/Windows/Linux). Module-scoped state so playback
 * keeps working when the page/window is hidden.
 */

export type MicState = 'idle' | 'connecting' | 'recording' | 'finalizing'
export type MicMode = 'manual_turn' | 'continuous'

export const $mmMicState = atom<MicState>('idle')
export const $mmMicError = atom<string>('')
export const $mmAsrPartial = atom<string>('')
// EOU listening mode can stitch several finalized speech segments before it
// submits the complete user turn. Keep those segments separate from the live
// partial so the composer can render the stable prefix dimmed, matching Web.
export const $mmAsrBuffer = atom<string[]>([])
export const $mmTtsPlaying = atom<boolean>(false)
// ★ 语音自动播报开关 (对齐 web): 开启后后端 VoiceAgent 旁路会自动把主 agent / watcher /
//   monitor 的完成气泡改写口语化 → 播报。与麦克风解耦, 默认关。
export const $mmTtsEnabled = atom<boolean>(false)
// ★ 对话模式开关 (对齐 web): 开启 → ASR final 进 VoiceAgent v2 主线程分诊
//   (self 直答 / 委派主 Agent 时回一句承接语 + 层2/层3防误识别); 关闭 → ASR final
//   走传统路径 (_run_prompt_submit)。与麦克风联动 (见 toggleMultimodalVoiceDialog):
//   开对话自动开麦; 关麦强制关对话 (无麦相当于哑火)。
//   注: 分诊/播报/承接语等全部逻辑在共用的 Python 后端 (agent/multimodal/voice_agent_v2*),
//   desktop 端只有这几个 UI 开关 —— 后端改动 web/desktop 自动共享, 无需在此同步。
export const $mmVoiceDialogEnabled = atom<boolean>(false)

const WORKLET_URL = `${import.meta.env.BASE_URL || './'}pcm-worklet.js`

// ── base64 helpers (chunked to avoid stack blowups on large buffers) ────────
function bytesToBase64(bytes: Uint8Array): string {
  let bin = ''
  const CHUNK = 0x8000
  for (let i = 0; i < bytes.length; i += CHUNK) {
    bin += String.fromCharCode(...bytes.subarray(i, i + CHUNK))
  }
  return btoa(bin)
}

function base64ToBytes(b64: string): Uint8Array {
  const bin = atob(b64)
  const bytes = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i)
  return bytes
}

// ── Mic streaming ASR ───────────────────────────────────────────────────────
interface MicRefs {
  stream: MediaStream | null
  ctx: AudioContext | null
  source: MediaStreamAudioSourceNode | null
  node: AudioWorkletNode | null
  recording: boolean
  generation: number
  gateway: ReturnType<typeof $gateway.get>
  sessionId: string
  /** User intent kept across a reconnect attempt that races runtime resume. */
  rearmPending: boolean
  draftEnsureSession: (() => Promise<string | null>) | null
  preRoll: ArrayBuffer[]
  preRollBytes: number
  cancelPendingDraft: (() => void) | null
  /** Stable logical turn id. It is allocated before permission and follows
   * start/audio/stop so late PCM can never leak into a replacement turn. */
  turnId: string
  mode: MicMode
  backendReady: boolean
  startPromise: Promise<void> | null
  finishPromise: Promise<void> | null
  pendingAudio: Set<Promise<void>>
  audioFailed: boolean
  flushResolver: ((flushed: boolean) => void) | null
}

const mic: MicRefs = {
  stream: null,
  ctx: null,
  source: null,
  node: null,
  recording: false,
  generation: 0,
  gateway: null,
  sessionId: '',
  rearmPending: false,
  draftEnsureSession: null,
  preRoll: [],
  preRollBytes: 0,
  cancelPendingDraft: null,
  turnId: '',
  mode: 'manual_turn',
  backendReady: false,
  startPromise: null,
  finishPromise: null,
  pendingAudio: new Set(),
  audioFailed: false,
  flushResolver: null
}

// Cover the legal 210s cold-activation budget plus 30s of scheduling/ACK
// headroom. Mono PCM16 is ~7.68 MB: bounded, without rolling off the opening
// words at the edge of an otherwise-valid activation.
const MIC_PRE_ROLL_MAX_BYTES = 16_000 * 2 * 240
let micTurnSequence = 0

function nextMicTurnId(): string {
  micTurnSequence += 1
  const random = globalThis.crypto?.randomUUID?.() || Math.random().toString(36).slice(2)

  return `desktop-asr-${Date.now()}-${micTurnSequence}-${random}`
}

async function requestMicrophoneStream(): Promise<MediaStream> {
  const requestAccess = window.hermesDesktop?.requestMicrophoneAccess

  if (requestAccess) {
    const permitted = await requestAccess()

    if (!permitted) {
      throw new Error(translateNow('multimodal.voiceErrors.micPermissionDenied'))
    }
  }

  return navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
      channelCount: 1
    }
  })
}

async function prepareMicContext(stream: MediaStream): Promise<{
  ctx: AudioContext
  source: MediaStreamAudioSourceNode
  node: AudioWorkletNode
}> {
  const Ctx =
    window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
  const ctx = new Ctx()

  try {
    await ctx.audioWorklet.addModule(WORKLET_URL)
    if (ctx.state === 'suspended') {
      await ctx.resume()
    }

    const source = ctx.createMediaStreamSource(stream)
    const node = new AudioWorkletNode(ctx, 'pcm-downsample-processor', {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      processorOptions: { inRate: ctx.sampleRate, batchMs: 200 }
    })

    return { ctx, source, node }
  } catch (error) {
    await ctx.close().catch(() => undefined)

    throw error
  }
}

/** Desktop main-chat injection point. It deliberately creates nothing until
 * the locally-armed draft mic produces non-empty PCM. */
export function configureDraftMicSessionEnsurer(ensureSession: (() => Promise<string | null>) | null): void {
  mic.draftEnsureSession = ensureSession
}

function clearMicPreRoll(): void {
  mic.preRoll = []
  mic.preRollBytes = 0
}

function appendMicPreRoll(buf: ArrayBuffer): void {
  // Worklet packets are normally ~6.4 KB, but keep the invariant strict even
  // if a malformed/alternate producer emits one packet larger than the cap.
  const copy = buf.byteLength > MIC_PRE_ROLL_MAX_BYTES
    ? buf.slice(buf.byteLength - MIC_PRE_ROLL_MAX_BYTES)
    : buf.slice(0)

  mic.preRoll.push(copy)
  mic.preRollBytes += copy.byteLength

  while (mic.preRollBytes > MIC_PRE_ROLL_MAX_BYTES && mic.preRoll.length > 1) {
    const dropped = mic.preRoll.shift()

    mic.preRollBytes -= dropped?.byteLength || 0
  }
}

function clearAsrPreview(): void {
  $mmAsrPartial.set('')
  $mmAsrBuffer.set([])
}

function consumeMicControlMessage(data: unknown): boolean {
  if (!data || typeof data !== 'object' || (data as { type?: unknown }).type !== 'flushed') {
    return false
  }

  const resolve = mic.flushResolver

  mic.flushResolver = null
  resolve?.(true)

  return true
}

async function flushMicTail(): Promise<boolean> {
  const node = mic.node

  if (!node) {
    return false
  }

  return new Promise<boolean>(resolve => {
    let settled = false
    let handleFlush: ((flushed: boolean) => void) | null = null
    const finish = (flushed: boolean) => {
      if (settled) {
        return
      }
      settled = true
      if (mic.flushResolver === handleFlush) {
        mic.flushResolver = null
      }
      resolve(flushed)
    }
    const timer = setTimeout(() => finish(false), 500)

    handleFlush = flushed => {
      clearTimeout(timer)
      finish(flushed)
    }
    mic.flushResolver = handleFlush
    try {
      node.port.postMessage({ type: 'flush' })
    } catch {
      clearTimeout(timer)
      finish(false)
    }
  })
}

/** True when the mic either owns resources or is waiting for the replacement
 * runtime of the same durable conversation. The latter intentionally does not
 * rely on the presentation atom: a failed start against the stale runtime may
 * return the UI to idle before session.resume publishes its replacement id. */
export function hasMicCaptureIntent(): boolean {
  return Boolean(mic.rearmPending || mic.recording || mic.sessionId || $mmMicState.get() !== 'idle')
}

function queueMicAudio(
  gw: NonNullable<ReturnType<typeof $gateway.get>>,
  sid: string,
  turnId: string,
  generation: number,
  buf: ArrayBuffer
): void {
  const pcm_b64 = bytesToBase64(new Uint8Array(buf))

  if (
    mic.generation !== generation ||
    mic.turnId !== turnId ||
    mic.sessionId !== sid ||
    $mmSessionId.get() !== sid
  ) {
    return
  }

  const pending = gw.request<{ ok?: boolean }>('multimodal.asr_audio', {
    session_id: sid,
    turn_id: turnId,
    pcm_b64
  }).then(result => {
    if (result?.ok === false) {
      throw new Error('asr_audio rejected')
    }
  }).catch(() => {
    if (
      mic.mode === 'manual_turn' &&
      mic.generation === generation &&
      mic.turnId === turnId &&
      mic.sessionId === sid
    ) {
      mic.audioFailed = true
    }
  })

  mic.pendingAudio.add(pending)
  void pending.finally(() => mic.pendingAudio.delete(pending))
}

async function startMicOwned(keepRearmIntentOnFailure: boolean): Promise<void> {
  if (mic.recording || $mmMicState.get() === 'connecting') {
    return
  }

  const gw = $gateway.get()
  const sid = $mmSessionId.get()

  if (!gw || !sid) {
    return
  }

  const generation = mic.generation + 1
  const turnId = mic.turnId || nextMicTurnId()
  const mode = mic.mode || ($mmVoiceDialogEnabled.get() ? 'continuous' : 'manual_turn')

  mic.generation = generation
  mic.gateway = gw
  mic.sessionId = sid
  mic.turnId = turnId
  mic.mode = mode
  mic.backendReady = false
  mic.pendingAudio.clear()
  mic.audioFailed = false
  clearAsrPreview()
  $mmMicState.set('connecting')

  let pendingStream: MediaStream | null = null
  let pendingCtx: AudioContext | null = null

  const stillOwnsStart = () => (
    mic.generation === generation &&
    mic.turnId === turnId &&
    mic.sessionId === sid &&
    $mmSessionId.get() === sid
  )
  const releasePending = () => {
    if (pendingCtx) {
      void pendingCtx.close().catch(() => undefined)
      pendingCtx = null
    }
    if (pendingStream) {
      pendingStream.getTracks().forEach(track => track.stop())
      pendingStream = null
    }
  }
  mic.cancelPendingDraft = releasePending
  const cancelOwnedBackend = () => {
    void gw.request('multimodal.asr_stop', {
      session_id: sid,
      turn_id: turnId,
      disposition: 'cancel'
    }).catch(() => undefined)
  }

  try {
    // Acquire first, then open ASR. The worklet buffers immediately so a cold
    // backend cannot eat the first words of the user's explicit recording.
    pendingStream = await requestMicrophoneStream()
    if (!stillOwnsStart()) {
      releasePending()

      return
    }

    const prepared = await prepareMicContext(pendingStream)

    pendingCtx = prepared.ctx
    if (!stillOwnsStart()) {
      prepared.node.port.close()
      prepared.node.disconnect()
      prepared.source.disconnect()
      releasePending()

      return
    }

    mic.stream = pendingStream
    mic.ctx = pendingCtx
    mic.source = prepared.source
    mic.node = prepared.node
    mic.cancelPendingDraft = null
    pendingStream = null
    pendingCtx = null

    prepared.node.port.onmessage = (ev: MessageEvent) => {
      if (consumeMicControlMessage(ev.data)) {
        return
      }
      if (
        !mic.recording ||
        mic.generation !== generation ||
        mic.turnId !== turnId ||
        mic.sessionId !== sid ||
        $mmSessionId.get() !== sid ||
        micGatedForTts()
      ) {
        return
      }

      const buf = ev.data as ArrayBuffer

      if (!buf?.byteLength) {
        return
      }

      if (mic.backendReady) {
        queueMicAudio(gw, sid, turnId, generation, buf)
      } else {
        appendMicPreRoll(buf)
      }
    }
    // Install the receiver before connecting the graph. Chromium may schedule
    // the first audio quantum immediately after source.connect().
    mic.recording = true
    prepared.source.connect(prepared.node)
    prepared.node.connect(prepared.ctx.destination)
    mic.rearmPending = false

    // Permission + local capture are the truthful red-recording boundary.
    if ($mmMicState.get() !== 'finalizing') {
      $mmMicState.set('recording')
    } else {
      mic.recording = false
      _releaseMicResources()
    }

    const res = await gw.request<{ enabled?: boolean; turn_id?: string }>(
      'multimodal.asr_start',
      {
        session_id: sid,
        turn_id: turnId,
        mode
      },
      210_000
    )

    if (!stillOwnsStart()) {
      cancelOwnedBackend()

      return
    }
    if (!res?.enabled) {
      throw new Error(translateNow('multimodal.voiceErrors.streamingNotEnabled'))
    }
    if (res.turn_id && res.turn_id !== turnId) {
      throw new Error(translateNow('multimodal.voiceErrors.turnMismatch'))
    }

    mic.backendReady = true
    const queued = mic.preRoll

    clearMicPreRoll()
    for (const chunk of queued) {
      if (!stillOwnsStart()) {
        break
      }
      queueMicAudio(gw, sid, turnId, generation, chunk)
    }
  } catch (error) {
    releasePending()
    cancelOwnedBackend()

    if (!stillOwnsStart()) {
      return
    }

    mic.recording = false
    mic.backendReady = false
    mic.gateway = null
    mic.sessionId = ''
    mic.turnId = ''
    mic.cancelPendingDraft = null
    _releaseMicResources()
    clearMicPreRoll()

    mic.rearmPending = keepRearmIntentOnFailure

    $mmMicState.set('idle')
    clearAsrPreview()
    $mmMicError.set(error instanceof Error ? error.message : String(error))

    throw error
  }
}

async function armDraftMic(): Promise<void> {
  if (hasMicCaptureIntent()) {
    return
  }

  const ensureSession = mic.draftEnsureSession

  if (!ensureSession) {
    return
  }

  const generation = mic.generation + 1
  const turnId = nextMicTurnId()
  const mode: MicMode = $mmVoiceDialogEnabled.get() ? 'continuous' : 'manual_turn'

  mic.generation = generation
  mic.rearmPending = true
  mic.turnId = turnId
  mic.mode = mode
  mic.backendReady = false
  mic.pendingAudio.clear()
  mic.audioFailed = false
  clearAsrPreview()
  clearMicPreRoll()
  $mmMicState.set('connecting')

  let pendingStream: MediaStream | null = null
  let pendingCtx: AudioContext | null = null
  const stillOwnsDraft = () => mic.generation === generation && mic.turnId === turnId
  const releasePending = () => {
    if (pendingCtx) {
      void pendingCtx.close().catch(() => undefined)
      pendingCtx = null
    }
    if (pendingStream) {
      pendingStream.getTracks().forEach(track => track.stop())
      pendingStream = null
    }
  }
  mic.cancelPendingDraft = releasePending
  const failDraft = (error?: unknown) => {
    if (!stillOwnsDraft()) {
      return
    }

    mic.rearmPending = false
    mic.cancelPendingDraft = null
    mic.recording = false
    mic.backendReady = false
    mic.gateway = null
    mic.sessionId = ''
    mic.turnId = ''
    $mmMicState.set('idle')
    clearMicPreRoll()
    clearAsrPreview()
    _releaseMicResources()
    releasePending()
    if (error) {
      $mmMicError.set(error instanceof Error ? error.message : String(error))
    }

    if ($mmVoiceDialogEnabled.get()) {
      $mmVoiceDialogEnabled.set(false)
    }
  }

  try {
    pendingStream = await requestMicrophoneStream()

    if (!stillOwnsDraft()) {
      releasePending()

      return
    }

    const prepared = await prepareMicContext(pendingStream)

    pendingCtx = prepared.ctx

    if (!stillOwnsDraft()) {
      prepared.node.port.close()
      prepared.node.disconnect()
      prepared.source.disconnect()
      releasePending()

      return
    }

    mic.stream = pendingStream
    mic.ctx = pendingCtx
    mic.source = prepared.source
    mic.node = prepared.node
    mic.cancelPendingDraft = null
    pendingStream = null
    pendingCtx = null

    prepared.node.port.onmessage = (ev: MessageEvent) => {
      if (consumeMicControlMessage(ev.data)) {
        return
      }
      if (!stillOwnsDraft() || !mic.recording || micGatedForTts()) {
        return
      }

      const buf = ev.data as ArrayBuffer

      if (!buf?.byteLength) {
        return
      }

      if (mic.backendReady && mic.gateway && mic.sessionId) {
        queueMicAudio(mic.gateway, mic.sessionId, turnId, generation, buf)

        return
      }

      appendMicPreRoll(buf)

      if (!mic.startPromise) {
        const createInFlight = (async () => {
        const sid = await ensureSession()

        if (!stillOwnsDraft()) {
          return
        }
        if (!sid || $mmSessionId.get() !== sid) {
          failDraft(new Error(translateNow('multimodal.voiceErrors.sessionCreateFailed')))

          return
        }

        const gw = $gateway.get()

        if (!gw) {
          failDraft(new Error(translateNow('multimodal.voiceErrors.connectionUnavailable')))

          return
        }

        mic.gateway = gw
        mic.sessionId = sid
        const res = await gw.request<{ enabled?: boolean; turn_id?: string }>(
          'multimodal.asr_start',
          {
            session_id: sid,
            turn_id: turnId,
            mode
          },
          210_000
        )

        if (!res?.enabled) {
          throw new Error(translateNow('multimodal.voiceErrors.streamingNotEnabled'))
        }
        if (!stillOwnsDraft() || $gateway.get() !== gw || $mmSessionId.get() !== sid) {
          void gw.request('multimodal.asr_stop', {
            session_id: sid,
            turn_id: turnId,
            disposition: 'cancel'
          }).catch(() => undefined)

          failDraft()

          return
        }
        if (res.turn_id && res.turn_id !== turnId) {
          throw new Error(translateNow('multimodal.voiceErrors.turnMismatch'))
        }

        mic.backendReady = true
        mic.rearmPending = false

        const queued = mic.preRoll

        clearMicPreRoll()
        for (const chunk of queued) {
          if (!stillOwnsDraft() || mic.sessionId !== sid || $mmSessionId.get() !== sid) {
            break
          }
          queueMicAudio(gw, sid, turnId, generation, chunk)
        }

        if ($mmVoiceDialogEnabled.get() && mic.generation === generation) {
          void gw
            .request('multimodal.voice_dialog_toggle', {
              session_id: sid,
              enabled: true
            })
            .catch(() => undefined)
        }
        })()

        mic.startPromise = createInFlight
        void createInFlight.catch(failDraft)
      }
    }
    // No opening audio quantum can beat the pre-roll handler.
    mic.recording = true
    prepared.source.connect(prepared.node)
    prepared.node.connect(prepared.ctx.destination)
    $mmMicState.set('recording')
  } catch (error) {
    releasePending()

    if (stillOwnsDraft()) {
      mic.rearmPending = false
      mic.cancelPendingDraft = null
      mic.recording = false
      mic.backendReady = false
      mic.turnId = ''
      $mmMicState.set('idle')
      clearMicPreRoll()
      clearAsrPreview()
      _releaseMicResources()
      $mmMicError.set(error instanceof Error ? error.message : String(error))

      if ($mmVoiceDialogEnabled.get()) {
        $mmVoiceDialogEnabled.set(false)
      }
    }

    throw error
  }
}

export async function startMic(): Promise<void> {
  if (hasMicCaptureIntent()) {
    return
  }

  mic.turnId = nextMicTurnId()
  mic.mode = $mmVoiceDialogEnabled.get() ? 'continuous' : 'manual_turn'
  mic.finishPromise = null
  mic.startPromise = null
  mic.backendReady = false
  mic.audioFailed = false
  $mmMicError.set('')

  if (!$mmSessionId.get()) {
    await armDraftMic()

    return
  }

  // A direct user start is a fresh attempt. Only reconnect/rebind paths keep a
  // latent intent when the old runtime rejects before its replacement exists.
  mic.rearmPending = false
  clearMicPreRoll()
  const starting = startMicOwned(false)

  mic.startPromise = starting
  try {
    await starting
  } finally {
    if (mic.startPromise === starting && $mmMicState.get() !== 'finalizing') {
      mic.startPromise = null
    }
  }
}

/** Tear down mic AudioContext/worklet/stream (no server call). Shared by
 *  stopMic, the startMic failure path, and reconnect re-arm. Idempotent. */
function _releaseMicResources(): void {
  try {
    if (mic.node) {
      try {
        mic.node.port.onmessage = null
        mic.node.port.close()
        mic.node.disconnect()
      } catch {
        /* noop */
      }
    }
    if (mic.source) {
      try {
        mic.source.disconnect()
      } catch {
        /* noop */
      }
    }
    if (mic.ctx) void mic.ctx.close().catch(() => undefined)
    if (mic.stream) mic.stream.getTracks().forEach(t => t.stop())
  } finally {
    const resolveFlush = mic.flushResolver

    mic.flushResolver = null
    resolveFlush?.(false)
    mic.node = null
    mic.source = null
    mic.ctx = null
    mic.stream = null
  }
}

/** ★ Reconnect re-arm (background-lifecycle): after a gateway drop+reconnect the
 *  server's ASR session was reaped (close_on_disconnect), so a still-"recording"
 *  mic would stream PCM into a dead ASR session — transcription silently dies.
 *  Called from the connection onState handler on reconnect: if the user had the
 *  mic on, tear the local audio graph down and start it fresh against the new
 *  session id. No-op if the mic wasn't recording. */
async function rearmMic(stopPreviousBackend: boolean, keepIntentForReplacement: boolean): Promise<void> {
  if (!hasMicCaptureIntent()) {
    return
  }

  const previousGateway = mic.gateway
  const previousSessionId = mic.sessionId
  const previousTurnId = mic.turnId
  const cancelPendingDraft = mic.cancelPendingDraft

  mic.rearmPending = true
  mic.generation += 1
  mic.cancelPendingDraft = null
  mic.recording = false
  mic.backendReady = false
  mic.gateway = null
  mic.sessionId = ''
  mic.startPromise = null
  mic.pendingAudio.clear()
  mic.audioFailed = false
  _releaseMicResources()
  cancelPendingDraft?.()
  clearMicPreRoll()
  clearAsrPreview()

  $mmMicState.set('connecting')

  if (stopPreviousBackend && previousGateway && previousSessionId) {
    void previousGateway
      .request('multimodal.asr_stop', {
        session_id: previousSessionId,
        turn_id: previousTurnId,
        disposition: 'cancel'
      })
      .catch(() => undefined)
  }

  if (!$mmSessionId.get()) {
    if (!keepIntentForReplacement) {
      mic.rearmPending = false
    }

    $mmMicState.set('idle')

    return
  }

  $mmMicState.set('idle')

  try {
    const starting = startMicOwned(keepIntentForReplacement)

    mic.startPromise = starting
    await starting
    if (mic.startPromise === starting) {
      mic.startPromise = null
    }

    // A replacement runtime has a fresh session dictionary. Restore the
    // conversation-mode bit after ASR comes up so the UI's still-enabled
    // dialog mode continues routing finals through VoiceAgent on the new sid.
    const currentGateway = $gateway.get()
    const currentSessionId = $mmSessionId.get()

    if ($mmVoiceDialogEnabled.get() && $mmMicState.get() === 'recording' && currentGateway && currentSessionId) {
      void currentGateway
        .request('multimodal.voice_dialog_toggle', {
          session_id: currentSessionId,
          enabled: true
        })
        .catch(() => undefined)
    }
  } catch {
    // The stale runtime can reject before session.resume publishes A2. The
    // transport-reconnect caller keeps that one-shot intent separately from
    // the idle presentation state; the concrete A2 rebind is the final attempt
    // and consumes it even if ASR is now disabled.
    $mmMicState.set('idle')
  }
}

/** Recreate ASR after a transport reconnect. The old backend-side ASR session
 * was already reaped with the socket, so only the local graph needs rearming. */
export async function rearmMicAfterReconnect(): Promise<void> {
  if (mic.mode === 'manual_turn') {
    await stopMic()

    return
  }

  await rearmMic(false, true)
}

/** A manual push-to-talk turn cannot silently span a transport outage. Release
 * it at the disconnect boundary; continuous Voice Dialog is rearmed instead. */
export function cancelManualMicOnDisconnect(): void {
  if (mic.mode === 'manual_turn' && hasMicCaptureIntent()) {
    void stopMic()
  }
}

/** Move a live mic from an obsolete runtime id to the replacement runtime for
 * the same durable conversation. The old runtime is explicitly stopped before
 * the new ASR session starts; voice-dialog state remains enabled. */
export async function rearmMicForSessionRebind(): Promise<void> {
  if (mic.mode === 'manual_turn') {
    await stopMic()

    return
  }

  await rearmMic(true, false)
}

export interface FinishMicTurnResult {
  error?: string
  ok?: boolean
  turn_id?: string
  transcript?: string
  submitted?: boolean
  reason?: string
}

function finishFailureMessage(result: FinishMicTurnResult): string {
  // A successfully submitted turn may still carry a non-fatal upstream warning
  // (for example a recovered timeout with usable partial text).
  if (result.submitted === true) {
    return ''
  }

  const reason = typeof result.reason === 'string' ? result.reason.trim().slice(0, 120) : ''

  switch (reason) {
    case 'empty':
      return translateNow('multimodal.voiceErrors.asrEmpty')
    case 'finish_timeout':
      return translateNow('multimodal.voiceErrors.asrFinishTimeout')
    case 'dispatch_failed':
      return translateNow('multimodal.voiceErrors.asrDispatchFailed')
    case 'no_engine':
    case 'upstream_error':
    case 'finish_failed':
      return translateNow('multimodal.voiceErrors.asrServiceUnavailable')
    case 'no_active_turn':
    case 'stale_transport':
    case 'stale_turn':
      return translateNow('multimodal.voiceErrors.asrTurnStale')
    default:
      break
  }

  if (result.ok !== false && result.submitted !== false) {
    return ''
  }

  return reason
    ? translateNow('multimodal.voiceErrors.submitFailedWithReason', reason)
    : translateNow('multimodal.voiceErrors.submitFailedGeneric')
}

/** Commit one explicit push-to-talk turn. Local capture stops immediately, but
 * its owner remains valid until startup and every already-dispatched audio ACK
 * settle; only then can finish overtake neither pre-roll nor the last chunk. */
export async function finishMicTurn(): Promise<void> {
  if (mic.finishPromise) {
    return mic.finishPromise
  }

  if ($mmMicState.get() !== 'recording' || !hasMicCaptureIntent() || mic.mode !== 'manual_turn') {
    return
  }

  const generation = mic.generation
  const turnId = mic.turnId
  const anchorPromise = import('./multimodal-capture')
    .then(capture => capture.snapshotCaptureAnchor())
    .catch(() => null)

  $mmMicState.set('finalizing')
  // Stop the physical input synchronously, but keep the worklet graph alive
  // long enough to flush its sub-200ms tail into the same logical turn.
  if (mic.stream) {
    mic.stream.getTracks().forEach(track => track.stop())
    mic.stream = null
  }

  const finish = (async () => {
    try {
      const tailFlushed = await flushMicTail()
      if (mic.generation !== generation || mic.turnId !== turnId) {
        return
      }
      if (!tailFlushed) {
        throw new Error(translateNow('multimodal.voiceErrors.recordingFinalizeFailed'))
      }
      mic.recording = false
      _releaseMicResources()

      const starting = mic.startPromise

      if (starting) {
        await starting
      }

      if (mic.generation !== generation || mic.turnId !== turnId) {
        return
      }

      // A draft with no PCM never created a runtime. Treat the second click as
      // an empty turn and leave no chat/session artifact behind.
      if (!mic.gateway || !mic.sessionId || !mic.backendReady) {
        return
      }

      await Promise.allSettled(Array.from(mic.pendingAudio))
      if (mic.generation !== generation || mic.turnId !== turnId) {
        return
      }
      if (mic.audioFailed) {
        throw new Error(translateNow('multimodal.voiceErrors.audioUploadInterrupted'))
      }

      const anchor = await anchorPromise
      const result = await mic.gateway.request<FinishMicTurnResult>(
        'multimodal.asr_stop',
        {
          session_id: mic.sessionId,
          turn_id: turnId,
          disposition: 'finish',
          ...(anchor || {})
        },
        210_000
      )

      // A session/profile/disconnect boundary may cancel while the finish RPC
      // is in flight. Its response belongs to the retired owner and must not
      // revive an error/preview in the replacement conversation.
      if (mic.generation !== generation || mic.turnId !== turnId) {
        return
      }

      if (result?.turn_id && result.turn_id !== turnId) {
        throw new Error(translateNow('multimodal.voiceErrors.turnMismatch'))
      }
      const failureMessage = finishFailureMessage(result || {})

      if (failureMessage) {
        throw new Error(failureMessage)
      }
    } catch (error) {
      if (mic.generation !== generation || mic.turnId !== turnId) {
        return
      }

      const ownerGateway = mic.gateway
      const ownerSessionId = mic.sessionId

      if (ownerGateway && ownerSessionId && mic.generation === generation && mic.turnId === turnId) {
        void ownerGateway.request('multimodal.asr_stop', {
          session_id: ownerSessionId,
          turn_id: turnId,
          disposition: 'cancel'
        }).catch(() => undefined)
      }

      $mmMicError.set(error instanceof Error ? error.message : String(error))
      throw error
    } finally {
      if (mic.generation === generation && mic.turnId === turnId) {
        mic.generation += 1
        mic.rearmPending = false
        mic.cancelPendingDraft = null
        mic.recording = false
        mic.backendReady = false
        mic.gateway = null
        mic.sessionId = ''
        mic.turnId = ''
        mic.startPromise = null
        mic.pendingAudio.clear()
        mic.audioFailed = false
        clearMicPreRoll()
        clearAsrPreview()
        $mmMicState.set('idle')
      }
    }
  })()

  mic.finishPromise = finish
  try {
    await finish
  } finally {
    if (mic.finishPromise === finish) {
      mic.finishPromise = null
    }
  }
}

/** Cancel recording without submitting a turn. Session/profile/New/disconnect
 * boundaries use this path; the ordinary mic button uses finishMicTurn(). */
export async function stopMic(): Promise<void> {
  // Always clear the preview, even if a late stop races with an ASR/server
  // disconnect and the local recorder is already idle.
  const wasActive = hasMicCaptureIntent()
  const ownerGateway = mic.gateway
  const ownerSessionId = mic.sessionId
  const ownerTurnId = mic.turnId
  const starting = mic.startPromise
  const cancelPendingDraft = mic.cancelPendingDraft
  const disableVoiceDialog = $mmVoiceDialogEnabled.get()

  mic.generation += 1
  mic.rearmPending = false
  mic.cancelPendingDraft = null
  mic.recording = false
  mic.backendReady = false
  mic.gateway = null
  mic.sessionId = ''
  mic.turnId = ''
  mic.startPromise = null
  mic.finishPromise = null
  mic.pendingAudio.clear()
  mic.audioFailed = false
  $mmMicState.set('idle')
  _releaseMicResources()
  cancelPendingDraft?.()
  clearAsrPreview()
  clearMicPreRoll()

  // This presentation/ownership bit must fall synchronously with local mic
  // ownership, before any slow backend stop can overlap a new profile/session.
  if (disableVoiceDialog) {
    $mmVoiceDialogEnabled.set(false)
  }

  if (!wasActive) {
    mic.mode = 'manual_turn'

    return
  }

  if (ownerGateway && ownerSessionId) {
    await ownerGateway
      .request('multimodal.asr_stop', {
        session_id: ownerSessionId,
        turn_id: ownerTurnId,
        disposition: 'cancel'
      })
      .catch(() => undefined)
  }

  // If cancel raced a cold asr_start, its owner detects the bumped generation
  // and issues the same exact-id cancel after the backend start ACK.
  if (starting) {
    void starting.catch(() => undefined)
  }

  // ★ 麦关 → 强制关对话模式 (对话模式必须有活麦, 否则相当于哑火, 对齐 web)。
  //   只在真的处于开态时下发 RPC + set atom, 不做多余调用。不动 $mmTtsEnabled
  //   (喇叭按钮态由用户自己控制; 对话态清掉后后端 is_speaker_on 自然回落到 _mm_tts_on)。
  if (disableVoiceDialog) {
    if (ownerGateway && ownerSessionId) {
      void ownerGateway
        .request('multimodal.voice_dialog_toggle', {
          session_id: ownerSessionId,
          enabled: false
        })
        .catch(() => undefined)
    }
  }

  mic.mode = 'manual_turn'
  $mmMicError.set('')
}

function ownsAsrEvent(turnId?: string): boolean {
  // Modern desktop ASR is exact-turn only. Untagged events are accepted solely
  // when no modern owner exists (legacy compatibility), never into a new turn.
  return mic.turnId ? Boolean(turnId && mic.turnId === turnId) : !turnId
}

export function onAsrPartial(text: string, turnId?: string): void {
  if (!ownsAsrEvent(turnId)) {
    return
  }

  $mmAsrPartial.set(text || '')
}

export function onAsrBuffer(segments: unknown, turnId?: string): void {
  if (!ownsAsrEvent(turnId)) {
    return
  }

  const next = Array.isArray(segments)
    ? segments.filter((segment): segment is string => typeof segment === 'string' && Boolean(segment.trim()))
    : []

  $mmAsrBuffer.set(next)
}
export function onAsrFinal(text: string, turnId?: string): void {
  if (!ownsAsrEvent(turnId)) {
    return
  }

  const t = (text || '').trim()

  if (t) {
    addVoiceUserMessage(t)
  }

  clearAsrPreview()
}

// ── TTS playback (WebAudio gapless) ─────────────────────────────────────────
interface TtsRefs {
  ctx: AudioContext | null
  currentRid: string
  nextStart: number
  active: AudioBufferSourceNode[]
  cancelled: Set<string>
}
const tts: TtsRefs = { ctx: null, currentRid: '', nextStart: 0, active: [], cancelled: new Set() }

// ── Barge-in / self-hear guard ──────────────────────────────────────────────
// On a laptop with SPEAKER output, the mic re-captures the TTS the assistant is
// playing and feeds it back into ASR (echo/loop). Browser echoCancellation
// alone doesn't fully suppress loud speaker playback. So while TTS is audible we
// DROP the mic's PCM instead of sending it to ASR. `ttsMuteUntil` is a monotone
// deadline (epoch ms): each scheduled chunk pushes it to the chunk's playback
// end + a short tail (AEC/speaker decay lingers a bit past the last sample).
const TTS_MIC_TAIL_MS = 300
let ttsMuteUntil = 0
/** True while TTS is playing (or within the post-playback tail) → mute the mic. */
function micGatedForTts(): boolean {
  return Date.now() < ttsMuteUntil
}

function ensureTtsCtx(): AudioContext {
  if (!tts.ctx) {
    const Ctx =
      window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
    tts.ctx = new Ctx()
  }
  // Autoplay policy can leave the context 'suspended' until a user gesture; the
  // first TTS chunk would then play silently. Resume best-effort (mirrors web).
  if (tts.ctx.state === 'suspended') void tts.ctx.resume().catch(() => undefined)
  return tts.ctx
}

export interface TtsChunk {
  response_id?: string
  pcm_b64?: string
  sample_rate?: number
  is_final?: boolean
}

export function onTtsChunk(msg: TtsChunk): void {
  const rid = msg.response_id || ''
  // ★ Barge-in sentinel: 后端 interrupt_tts 发 rid="__interrupt__" + is_final=true 通知
  //   前端立即停播。之前只按 rid 匹配, 这个 sentinel 匹配不上任何当前 rid → 忽略, 用户
  //   已收到的 PCM 继续在 WebAudio 里播完 = "打断没效果"。识别它 → 全停。
  if (rid === '__interrupt__') {
    stopAllTts()
    return
  }
  if (tts.cancelled.has(rid)) return
  if (msg.is_final) {
    if (tts.currentRid === rid) $mmTtsPlaying.set(false)
    return
  }
  if (!msg.pcm_b64) return
  const ctx = ensureTtsCtx()
  if (tts.currentRid !== rid) {
    for (const s of tts.active) {
      try {
        s.stop()
      } catch {
        /* noop */
      }
    }
    tts.active = []
    tts.currentRid = rid
    tts.nextStart = ctx.currentTime
    $mmTtsPlaying.set(true)
  }
  try {
    const bytes = base64ToBytes(msg.pcm_b64)
    // PCM16 = 2 bytes/sample; drop a trailing odd byte so a truncated chunk
    // degrades to a tiny gap instead of a RangeError.
    const evenLen = bytes.byteLength & ~1
    const i16 = new Int16Array(bytes.buffer, 0, evenLen >> 1)
    const f32 = new Float32Array(i16.length)
    for (let i = 0; i < i16.length; i++) f32[i] = i16[i] / 32768.0
    const sr = msg.sample_rate || 24000
    const buf = ctx.createBuffer(1, f32.length, sr)
    buf.copyToChannel(f32, 0)
    const src = ctx.createBufferSource()
    src.buffer = buf
    src.connect(ctx.destination)
    const startAt = Math.max(ctx.currentTime, tts.nextStart)
    src.start(startAt)
    tts.active.push(src)
    tts.nextStart = startAt + buf.duration
    // Mute the mic through this chunk's playback (converting AudioContext time to
    // wall-clock) + a short tail, so speaker output isn't re-captured into ASR.
    const playoutMs = Math.max(0, tts.nextStart - ctx.currentTime) * 1000
    ttsMuteUntil = Math.max(ttsMuteUntil, Date.now() + playoutMs + TTS_MIC_TAIL_MS)
    src.onended = () => {
      const i = tts.active.indexOf(src)
      if (i >= 0) tts.active.splice(i, 1)
    }
  } catch {
    /* drop chunk */
  }
}

/** Stop all TTS playback; cancel the current rid so late chunks are ignored. */
export function stopAllTts(): void {
  if (tts.currentRid) {
    tts.cancelled.add(tts.currentRid)
    // Cap the cancelled-rid set: a long background session (monitors / deep
    // analysis produce many rids) would otherwise grow it unbounded.
    if (tts.cancelled.size > 64) {
      tts.cancelled = new Set(Array.from(tts.cancelled).slice(-32))
    }
  }
  for (const s of tts.active) {
    try {
      s.stop()
    } catch {
      /* noop */
    }
  }
  tts.active = []
  tts.currentRid = ''
  if (tts.ctx) tts.nextStart = tts.ctx.currentTime
  // Playback stopped early → lift the mic mute now (keep only a short tail for
  // the speaker/AEC decay) so the user can talk again immediately.
  ttsMuteUntil = Math.min(ttsMuteUntil, Date.now() + TTS_MIC_TAIL_MS)
  $mmTtsPlaying.set(false)
}

/** Ask the server to speak `text` (multimodal.tts_speak → streams tts chunks).
 *
 * ★ Manual play PREEMPTS any in-flight auto/streaming TTS: stopAllTts() first
 *   stops the currently-audible sources AND cancels the old rid so its remaining
 *   server chunks (already in flight) are dropped by onTtsChunk and never
 *   resume — no double audio, and the preempted auto-speech does not continue
 *   after the manual one finishes. */
export function speakText(text: string): void {
  const t = (text || '').trim()
  const gw = $gateway.get()
  const sid = $mmSessionId.get()
  if (!t || !gw || !sid) return
  stopAllTts()
  void gw.request('multimodal.tts_speak', { session_id: sid, text: t }).catch(() => undefined)
}

/** 切换自动播报开关 (对齐 web): 通知后端 VoiceAgent 旁路开/关。关闭时顺带停掉在播的 TTS。 */
export function toggleMultimodalTts(): void {
  const next = !$mmTtsEnabled.get()
  $mmTtsEnabled.set(next)
  if (!next) stopAllTts()
  const gw = $gateway.get()
  const sid = $mmSessionId.get()
  if (gw && sid) {
    void gw.request('multimodal.tts_toggle', { session_id: sid, enabled: next }).catch(() => undefined)
  }
}

/** ★ 切换对话模式 (对齐 web MultimodalChatPage.toggleVoiceDialog) = 后台统一接管麦/喇叭:
 *    用户方案: UI 麦/喇叭按钮态保持不变, 仅后台联动。
 *    - ON  → ①通知后端 voice_dialog_toggle (后端 is_speaker_on OR 对话态 → 强制 TTS;
 *            ASR final 走 v2 分诊) ②物理开麦 (idle 时 startMic; getUserMedia 采集是唯一
 *            能真正识别的途径, 后端无法凭空开麦; 麦按钮随之自然变红反映真实录音态)。
 *    - OFF → ①通知后端 ②物理关麦 → 一切恢复各自 _mm_asr_on/_mm_tts_on 真实态。
 *    不动 $mmTtsEnabled atom (喇叭按钮态不变; TTS 由后端强制)。stopMic 里有"关麦→
 *    强制关对话"的反向联动, 幂等安全。
 */
export function toggleMultimodalVoiceDialog(): boolean {
  const next = !$mmVoiceDialogEnabled.get()

  // Do not relabel an already-owned manual turn as continuous. Its callbacks
  // and backend mode were frozen at asr_start, so that would make the UI claim
  // dialog takeover while the turn still waits for the ordinary finish click.
  if (next && mic.mode === 'manual_turn' && hasMicCaptureIntent()) {
    return false
  }

  $mmVoiceDialogEnabled.set(next)
  const gw = $gateway.get()
  const sid = $mmSessionId.get()
  if (gw && sid) {
    void gw.request('multimodal.voice_dialog_toggle', { session_id: sid, enabled: next }).catch(() => undefined)
  }
  // 麦克风物理联动 (TTS 由后端强制, 不动 UI atom)。
  if (next) {
    if ($mmMicState.get() === 'idle') void startMic().catch(() => undefined)
  } else if (hasMicCaptureIntent()) {
    void stopMic().catch(() => undefined)
  }

  return true
}

// ── Env audio (screen-share audio → 5s MediaRecorder slices) ────────────────
interface EnvRefs {
  stream: MediaStream | null
  recorder: MediaRecorder | null
  ctx: AudioContext | null
  source: MediaStreamAudioSourceNode | null
  node: AudioWorkletNode | null
  mime: string
  stop: boolean
  timer: ReturnType<typeof setTimeout> | null
  windowSec: number
  startTs: number
  generation: number
  ownerGateway: ReturnType<typeof $gateway.get>
  ownerSessionId: string
  captureId: string
  chunkSeq: number
  lastError: string
  mode: 'idle' | 'media_recorder' | 'pcm_starting' | 'pcm_worklet'
  pcmChunks: ArrayBuffer[]
  pcmBytes: number
  pcmWindowStartedAt: number
}
const env: EnvRefs = {
  stream: null,
  recorder: null,
  ctx: null,
  source: null,
  node: null,
  mime: 'audio/webm',
  stop: false,
  timer: null,
  windowSec: 5,
  startTs: 0,
  generation: 0,
  ownerGateway: null,
  ownerSessionId: '',
  captureId: '',
  chunkSeq: 0,
  lastError: '',
  mode: 'idle',
  pcmChunks: [],
  pcmBytes: 0,
  pcmWindowStartedAt: 0
}

function envRecorderMimeCandidates(): Array<string | null> {
  const candidates = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus']
  if (typeof MediaRecorder === 'undefined') {
    return [null]
  }
  const supported = candidates.filter(m => MediaRecorder.isTypeSupported?.(m))

  return supported.length > 0
    ? [
        supported[0],
        // Chromium occasionally accepts a MIME at construction time but its
        // encoder rejects it in start(). Let Chromium choose its native default
        // before falling back to raw PCM.
        null,
        ...supported.slice(1)
      ]
    : [null]
}

function newEnvCaptureId(generation: number): string {
  const randomId = globalThis.crypto?.randomUUID?.()

  return `cap_${randomId || `${Date.now().toString(36)}_${generation}`}`
}

function ownsEnvCapture(
  generation: number,
  captureId: string,
  ownerGateway: ReturnType<typeof $gateway.get>,
  ownerSessionId: string
): boolean {
  return Boolean(
    !env.stop &&
    env.generation === generation &&
    env.captureId === captureId &&
    env.ownerGateway === ownerGateway &&
    env.ownerSessionId === ownerSessionId &&
    ownerGateway &&
    ownerSessionId &&
    $gateway.get() === ownerGateway &&
    $mmSessionId.get() === ownerSessionId
  )
}

function reportEnvError(
  generation: number,
  captureId: string,
  ownerGateway: ReturnType<typeof $gateway.get>,
  ownerSessionId: string,
  key: string,
  text: string
): void {
  if (!ownsEnvCapture(generation, captureId, ownerGateway, ownerSessionId)) {
    return
  }
  if (env.lastError === key) {
    return
  }

  env.lastError = key
  // Keep the voice module's eager dependency graph small: multimodal-deep
  // already imports the main multimodal store, which in turn imports this
  // module. Resolve the toast action only on an actual error and re-check
  // ownership after the async module boundary so an old capture cannot flash
  // an error in the replacement session.
  void import('./multimodal-deep')
    .then(({ pushMmToast }) => {
      if (!ownsEnvCapture(generation, captureId, ownerGateway, ownerSessionId)) {
        return
      }
      if (env.lastError !== key) {
        return
      }

      pushMmToast({ level: 'error', text })
    })
    .catch(() => undefined)
}

/** Start env-audio capture from a screen-share stream's audio tracks (if any). */
export function startEnvAudio(stream: MediaStream): void {
  const tracks = stream.getAudioTracks()
  if (tracks.length === 0) {
    return
  }

  stopEnvAudio()
  env.generation += 1
  // Record from owned clones, matching the known-good macOS/Electron path.
  // Keeping the native loopback tracks out of the recorder lifecycle prevents
  // MediaRecorder retries and stop/start slicing from disturbing the screen
  // share itself. stopEnvAudio() owns and stops only these clones; stopCapture()
  // remains the sole owner of the original display-media tracks.
  env.stream = new MediaStream(tracks.map(track => track.clone()))
  env.mime = 'audio/webm'
  env.stop = false
  env.startTs = performance.now()
  env.ownerGateway = $gateway.get()
  env.ownerSessionId = $mmSessionId.get()
  env.captureId = newEnvCaptureId(env.generation)
  env.chunkSeq = 0
  env.lastError = ''
  env.mode = 'idle'
  env.pcmChunks = []
  env.pcmBytes = 0
  env.pcmWindowStartedAt = 0
  cycleEnvRecorder()
}

function cycleEnvRecorder(): void {
  if (env.stop || !env.stream) {
    return
  }

  const generation = env.generation
  const ownerGateway = env.ownerGateway
  const ownerSessionId = env.ownerSessionId
  const captureId = env.captureId
  const captureStartTs = env.startTs
  const failures: string[] = []

  for (const requestedMime of envRecorderMimeCandidates()) {
    let rec: MediaRecorder
    try {
      rec = requestedMime ? new MediaRecorder(env.stream, { mimeType: requestedMime }) : new MediaRecorder(env.stream)
    } catch (error) {
      const reason = error instanceof Error ? error.message : String(error)
      failures.push(`${requestedMime || 'browser-default'} construct: ${reason}`)
      continue
    }

    if (
      startEnvMediaRecorder(
        rec,
        requestedMime || rec.mimeType || 'audio/webm',
        generation,
        captureId,
        ownerGateway,
        ownerSessionId,
        captureStartTs
      )
    ) {
      return
    }

    const reason = env.lastError || 'MediaRecorder.start() failed'
    failures.push(`${requestedMime || 'browser-default'} start: ${reason}`)
    // A failed candidate is diagnostic-only. Do not expose it as the active
    // error when the next codec or PCM fallback can keep ASR working.
    env.lastError = ''
  }

  startEnvPcmFallback(generation, captureId, ownerGateway, ownerSessionId, failures)
}

function startEnvMediaRecorder(
  rec: MediaRecorder,
  requestedMime: string,
  generation: number,
  captureId: string,
  ownerGateway: ReturnType<typeof $gateway.get>,
  ownerSessionId: string,
  captureStartTs: number
): boolean {
  env.recorder = rec
  let chunkSeq = 0
  let chunkId = ''
  const chunkStartedAt = performance.now()
  const chunks: Blob[] = []
  let blobTimecode = 0
  let chunkStoppedAt: number | null = null
  let recorderFailed = false
  rec.ondataavailable = ev => {
    if (ev.data && ev.data.size > 0) {
      chunks.push(ev.data)
    }
    if (Number.isFinite(ev.timecode)) {
      blobTimecode = ev.timecode
    }
  }
  rec.onstop = () => {
    if (env.recorder === rec) {
      env.recorder = null
    }

    if (recorderFailed) {
      return
    }

    const chunkEndedAt = chunkStoppedAt ?? performance.now()
    if (ownsEnvCapture(generation, captureId, ownerGateway, ownerSessionId)) {
      cycleEnvRecorder()
    }
    // stopEnvAudio is a hard session/source ownership boundary. Discard the
    // recorder's trailing slice rather than letting its async base64 encode
    // finish after the UI has already rebound to another conversation.
    if (!ownsEnvCapture(generation, captureId, ownerGateway, ownerSessionId)) {
      return
    }
    if (chunks.length === 0) {
      return
    }

    const payloadMime = chunks[0]?.type || rec.mimeType || requestedMime
    const blob = chunks.length === 1 ? chunks[0] : new Blob(chunks, { type: payloadMime })
    const clientStartTs = Math.max(0, (chunkStartedAt - captureStartTs) / 1000)
    const clientEndTs = Math.max(clientStartTs, (chunkEndedAt - captureStartTs) / 1000)
    const clientDurationSec = Math.max(0, (chunkEndedAt - chunkStartedAt) / 1000)

    submitEnvAudioBlob(blob, {
      generation,
      captureId,
      ownerGateway,
      ownerSessionId,
      payloadMime,
      chunkId,
      chunkSeq,
      clientStartTs,
      clientEndTs,
      clientDurationSec,
      blobTimecode
    })
  }
  rec.onerror = event => {
    if (!ownsEnvCapture(generation, captureId, ownerGateway, ownerSessionId)) {
      return
    }

    recorderFailed = true
    if (env.timer) {
      clearTimeout(env.timer)
      env.timer = null
    }
    if (env.recorder === rec) {
      env.recorder = null
    }
    const reason = event.error?.message || 'MediaRecorder runtime error'

    startEnvPcmFallback(generation, captureId, ownerGateway, ownerSessionId, [`${requestedMime} runtime: ${reason}`])
  }
  try {
    rec.start()
  } catch (error) {
    recorderFailed = true
    if (env.recorder === rec) {
      env.recorder = null
    }

    const reason = error instanceof Error ? error.message : String(error)
    env.lastError = reason
    rec.ondataavailable = null
    rec.onstop = null
    rec.onerror = null

    return false
  }
  chunkSeq = ++env.chunkSeq
  chunkId = `${captureId}:${chunkSeq}`
  env.mime = rec.mimeType || requestedMime
  env.mode = 'media_recorder'
  env.lastError = ''
  env.timer = setTimeout(
    () => {
      env.timer = null
      if (rec.state === 'recording') {
        chunkStoppedAt = performance.now()
        try {
          rec.stop()
        } catch (error) {
          const reason = error instanceof Error ? error.message : String(error)
          reportEnvError(
            generation,
            captureId,
            ownerGateway,
            ownerSessionId,
            `stop:${reason}`,
            translateNow('multimodal.voiceErrors.sharedAudioSliceFailed', reason)
          )
        }
      }
    },
    Math.max(1000, Math.round(env.windowSec * 1000))
  )

  return true
}

interface EnvAudioUpload {
  generation: number
  captureId: string
  ownerGateway: ReturnType<typeof $gateway.get>
  ownerSessionId: string
  payloadMime: string
  chunkId: string
  chunkSeq: number
  clientStartTs: number
  clientEndTs: number
  clientDurationSec: number
  blobTimecode: number
}

function submitEnvAudioBlob(blob: Blob, upload: EnvAudioUpload): void {
  if (blob.size < 1000) {
    return
  }

  void blobToBase64(blob)
    .then(b64 => {
      if (!ownsEnvCapture(upload.generation, upload.captureId, upload.ownerGateway, upload.ownerSessionId)) {
        return undefined
      }

      return upload.ownerGateway!.request<{ ingested?: boolean; reason?: string }>('multimodal.env_audio', {
        session_id: upload.ownerSessionId,
        data_b64: b64,
        mime: upload.payloadMime,
        window_ts: upload.clientStartTs,
        capture_id: upload.captureId,
        chunk_id: upload.chunkId,
        chunk_seq: upload.chunkSeq,
        client_start_ts: upload.clientStartTs,
        client_end_ts: upload.clientEndTs,
        client_duration_sec: upload.clientDurationSec,
        blob_timecode: upload.blobTimecode
      })
    })
    .then(result => {
      if (result === undefined) {
        return
      }
      if (!ownsEnvCapture(upload.generation, upload.captureId, upload.ownerGateway, upload.ownerSessionId)) {
        return
      }
      if (result?.ingested !== false) {
        env.lastError = ''

        return
      }

      const reason = result.reason || 'unknown'
      if (reason === 'too_short') {
        return
      }

      reportEnvError(
        upload.generation,
        upload.captureId,
        upload.ownerGateway,
        upload.ownerSessionId,
        reason,
        translateNow('multimodal.voiceErrors.sharedAudioNotReceived', reason)
      )
    })
    .catch(error => {
      const reason = error instanceof Error ? error.message : String(error)
      reportEnvError(
        upload.generation,
        upload.captureId,
        upload.ownerGateway,
        upload.ownerSessionId,
        reason,
        translateNow('multimodal.voiceErrors.sharedAudioAsrFailed', reason)
      )
    })
}

function pcm16WavBlob(chunks: ArrayBuffer[], byteLength: number): Blob {
  const header = new ArrayBuffer(44)
  const view = new DataView(header)
  const writeAscii = (offset: number, value: string) => {
    for (let i = 0; i < value.length; i += 1) {
      view.setUint8(offset + i, value.charCodeAt(i))
    }
  }

  writeAscii(0, 'RIFF')
  view.setUint32(4, 36 + byteLength, true)
  writeAscii(8, 'WAVE')
  writeAscii(12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true)
  view.setUint16(22, 1, true)
  view.setUint32(24, 16_000, true)
  view.setUint32(28, 16_000 * 2, true)
  view.setUint16(32, 2, true)
  view.setUint16(34, 16, true)
  writeAscii(36, 'data')
  view.setUint32(40, byteLength, true)

  return new Blob([header, ...chunks], { type: 'audio/wav' })
}

function pcm16HasSignal(chunks: ArrayBuffer[]): boolean {
  for (const chunk of chunks) {
    const samples = new Int16Array(chunk)

    for (let i = 0; i < samples.length; i += 1) {
      if (samples[i] !== 0) {
        return true
      }
    }
  }

  return false
}

function scheduleEnvPcmWindow(
  upload: Omit<
    EnvAudioUpload,
    'payloadMime' | 'chunkId' | 'chunkSeq' | 'clientStartTs' | 'clientEndTs' | 'clientDurationSec' | 'blobTimecode'
  >
): void {
  env.timer = setTimeout(
    () => {
      env.timer = null
      if (
        env.mode !== 'pcm_worklet' ||
        !ownsEnvCapture(upload.generation, upload.captureId, upload.ownerGateway, upload.ownerSessionId)
      ) {
        return
      }

      const endedAt = performance.now()
      const startedAt = env.pcmWindowStartedAt
      const chunks = env.pcmChunks
      const byteLength = env.pcmBytes

      env.pcmChunks = []
      env.pcmBytes = 0
      env.pcmWindowStartedAt = endedAt
      scheduleEnvPcmWindow(upload)

      if (byteLength < 1000) {
        return
      }

      // A live MediaStreamTrack can still be a dead CoreAudio tap whose PCM is
      // entirely zero. Uploading that valid-looking WAV makes ASR hallucinate
      // fillers such as "嗯。" and hides the real capture failure. Reject exact
      // digital silence locally and surface a stable, actionable diagnostic.
      if (!pcm16HasSignal(chunks)) {
        reportEnvError(
          upload.generation,
          upload.captureId,
          upload.ownerGateway,
          upload.ownerSessionId,
          'capture:silent_pcm',
          translateNow('multimodal.voiceErrors.sharedAudioNoSamples')
        )

        return
      }

      const chunkSeq = ++env.chunkSeq
      submitEnvAudioBlob(pcm16WavBlob(chunks, byteLength), {
        ...upload,
        payloadMime: 'audio/wav',
        chunkId: `${upload.captureId}:${chunkSeq}`,
        chunkSeq,
        clientStartTs: Math.max(0, (startedAt - env.startTs) / 1000),
        clientEndTs: Math.max(0, (endedAt - env.startTs) / 1000),
        clientDurationSec: Math.max(0, (endedAt - startedAt) / 1000),
        blobTimecode: 0
      })
    },
    Math.max(1000, Math.round(env.windowSec * 1000))
  )
}

function startEnvPcmFallback(
  generation: number,
  captureId: string,
  ownerGateway: ReturnType<typeof $gateway.get>,
  ownerSessionId: string,
  recorderFailures: string[]
): void {
  if (
    env.mode === 'pcm_starting' ||
    env.mode === 'pcm_worklet' ||
    !env.stream ||
    !ownsEnvCapture(generation, captureId, ownerGateway, ownerSessionId)
  ) {
    return
  }

  env.mode = 'pcm_starting'
  console.warn(`[multimodal] MediaRecorder unavailable; falling back to PCM/WAV: ${recorderFailures.join(' | ')}`)

  const fallbackStream = env.stream
  void (async () => {
    const Ctx =
      window.AudioContext || (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
    if (!Ctx) {
      throw new Error('AudioContext unavailable')
    }

    const ctx = new Ctx()
    let source: MediaStreamAudioSourceNode | null = null
    let node: AudioWorkletNode | null = null
    const releaseLocal = () => {
      if (node) {
        node.port.onmessage = null
        node.port.close()
        node.disconnect()
      }
      source?.disconnect()
      void ctx.close().catch(() => undefined)
    }

    try {
      await ctx.audioWorklet.addModule(WORKLET_URL)
      if (env.stream !== fallbackStream || !ownsEnvCapture(generation, captureId, ownerGateway, ownerSessionId)) {
        releaseLocal()
        return
      }

      source = ctx.createMediaStreamSource(fallbackStream)
      node = new AudioWorkletNode(ctx, 'pcm-downsample-processor', {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        processorOptions: { inRate: ctx.sampleRate, batchMs: 200 }
      })
      node.port.onmessage = (event: MessageEvent) => {
        if (env.mode !== 'pcm_worklet' || !ownsEnvCapture(generation, captureId, ownerGateway, ownerSessionId)) {
          return
        }
        const pcm = event.data as ArrayBuffer
        if (!pcm?.byteLength) {
          return
        }
        env.pcmChunks.push(pcm)
        env.pcmBytes += pcm.byteLength
      }
      source.connect(node)
      node.connect(ctx.destination)
      await ctx.resume()

      if (env.stream !== fallbackStream || !ownsEnvCapture(generation, captureId, ownerGateway, ownerSessionId)) {
        releaseLocal()
        return
      }

      env.ctx = ctx
      env.source = source
      env.node = node
      env.mime = 'audio/wav'
      env.mode = 'pcm_worklet'
      env.pcmChunks = []
      env.pcmBytes = 0
      env.pcmWindowStartedAt = performance.now()
      env.lastError = ''
      scheduleEnvPcmWindow({
        generation,
        captureId,
        ownerGateway,
        ownerSessionId
      })
    } catch (error) {
      releaseLocal()
      throw error
    }
  })().catch(error => {
    if (!ownsEnvCapture(generation, captureId, ownerGateway, ownerSessionId)) {
      return
    }
    env.mode = 'idle'
    const fallbackReason = error instanceof Error ? error.message : String(error)
    const detail = [...recorderFailures, `PCM fallback: ${fallbackReason}`].join(' | ')
    reportEnvError(
      generation,
      captureId,
      ownerGateway,
      ownerSessionId,
      `capture:${detail}`,
      translateNow('multimodal.voiceErrors.sharedAudioStartFailed', detail)
    )
  })
}

export function stopEnvAudio(): void {
  env.generation += 1
  env.stop = true
  if (env.timer) {
    clearTimeout(env.timer)
    env.timer = null
  }
  if (env.recorder && env.recorder.state === 'recording') {
    try {
      env.recorder.stop()
    } catch {
      /* noop */
    }
  }
  if (env.node) {
    try {
      env.node.port.onmessage = null
      env.node.port.close()
      env.node.disconnect()
    } catch {
      /* noop */
    }
  }

  if (env.source) {
    try {
      env.source.disconnect()
    } catch {
      /* noop */
    }
  }

  if (env.ctx) {
    void env.ctx.close().catch(() => undefined)
  }

  // env.stream contains recorder-owned clones, never the original screen-share
  // tracks. Release them here so repeated sharing cannot leak native captures.
  if (env.stream) {
    env.stream.getTracks().forEach(track => {
      try {
        track.stop()
      } catch {
        /* noop */
      }
    })
  }
  env.recorder = null
  env.stream = null
  env.ctx = null
  env.source = null
  env.node = null
  env.ownerGateway = null
  env.ownerSessionId = ''
  env.captureId = ''
  env.chunkSeq = 0
  env.lastError = ''
  env.mode = 'idle'
  env.pcmChunks = []
  env.pcmBytes = 0
  env.pcmWindowStartedAt = 0
}

function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const fr = new FileReader()
    fr.onload = () => {
      const s = String(fr.result || '')
      const comma = s.indexOf(',')
      resolve(comma >= 0 ? s.slice(comma + 1) : s)
    }
    fr.onerror = () => reject(fr.error)
    fr.readAsDataURL(blob)
  })
}
