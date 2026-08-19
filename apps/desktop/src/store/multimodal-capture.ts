import { atom } from 'nanostores'

import { translateNow } from '@/i18n'

import { $gateway } from './gateway'
import { $mmSessionId } from './multimodal'

/**
 * Multimodal video capture engine (desktop port of web MultimodalChatPage
 * startCamera/startScreen/captureFrame/startCapture).
 *
 * ★ Background-resident by design: the MediaStream, the offscreen <video>, and
 * the 2fps frame-push timer all live at MODULE scope — NOT inside a React
 * component. So capture keeps running after the multimodal page unmounts or the
 * window is hidden to the tray (the terminal product form). The page is only an
 * observer of $mmSource / $mmCapStats.
 *
 * Cross-platform: pure DOM APIs (getUserMedia / getDisplayMedia / <canvas>),
 * identical across Electron's Chromium on macOS / Windows / Linux. Screen share
 * relies on the main process's setDisplayMediaRequestHandler (main.cjs).
 */

export type MmSource = 'none' | 'camera' | 'screen'

export const $mmSource = atom<MmSource>('none')
export const $mmCapStats = atom<{ sent: number; dropped: number }>({ sent: 0, dropped: 0 })
export interface MmCaptureDebugState {
  code:
    | 'idle'
    | 'waiting_for_session'
    | 'gateway_unavailable'
    | 'gateway_not_open'
    | 'video_not_ready'
    | 'encode_failed'
    | 'notify_rejected'
    | 'backpressure'
    | 'sending'
  detail: string
}

// Deliberately contains metadata only: never put JPEG/base64 content here.
// The preview used to stay at "0 帧" while every early-return/catch in the
// capture loop was silent, which made session-binding and encoding failures
// indistinguishable. Exposing the current gate makes dev diagnostics visible in
// both the card and Electron's renderer console without leaking the image.
export const $mmCaptureDebug = atom<MmCaptureDebugState>({
  code: 'idle',
  detail: translateNow('multimodal.capture.notStarted')
})
// The live MediaStream (or null when capture is off), so the VideoStage UI can
// render a <video> preview mirror. The capture pipeline itself uses its own
// offscreen <video>; this atom just lets the page observe the stream.
export const $mmStream = atom<MediaStream | null>(null)

type DraftCaptureSessionEnsurer = () => Promise<string | null>

// Desktop main-chat injection point. A successful media permission grant is
// the ownership boundary: only then may capture create the draft's backend
// session and begin recording. Keeping the creator outside this store avoids a
// capture -> session-hooks dependency cycle and lets text/mic/capture share the
// same single-flight session.create path.
let draftCaptureSessionEnsurer: DraftCaptureSessionEnsurer | null = null

export function configureDraftCaptureSessionEnsurer(
  ensureSession: DraftCaptureSessionEnsurer | null
): void {
  draftCaptureSessionEnsurer = ensureSession
}

// Tunables (match web defaults).
const CAP_FPS = 2
const BUF_LIMIT = 512 * 1024 // skip a tick if the WS send buffer exceeds this
const SCREEN_MAX_SIDE = 1920
const CAMERA_MAX_SIDE = 1280
const JPEG_QUALITY_SCREEN = 0.8
const JPEG_QUALITY_LIGHT = 0.72
const JPEG_QUALITY_CAMERA = 0.72
// A normal agent build plus optional memory/watcher/monitor startup can exceed
// the gateway's generic 30s request budget on a cold machine.
// The backend may spend up to 120s waiting for the desktop agent and then up
// to ~70s starting memory/watcher/monitor services. Keep the end-to-end client
// budget larger than the sum so a legitimate cold start is not rolled back a
// few milliseconds before the backend becomes ready.
const SOURCE_ACTIVATION_TIMEOUT_MS = 210_000
// Rollback is best-effort cleanup. Never make local camera teardown wait for a
// slow activation handler holding the same backend capture lock.
const SOURCE_ROLLBACK_TIMEOUT_MS = 5_000

// ★ HiDPI/Retina light capture (parity with web preferLightCapture): on
// high-DPR displays (Mac Retina, HiDPI Windows) the captured track is physical
// pixels — often 2–4× logical — so drawImage's synchronous downscale from a huge
// source is the main-thread cost behind "屏幕共享卡 + 打字不流畅". Cap smaller and
// offload the resize to createImageBitmap when available.
function preferLightCapture(): boolean {
  try {
    const platform = typeof navigator !== 'undefined' ? navigator.platform || '' : ''
    const userAgent = typeof navigator !== 'undefined' ? navigator.userAgent || '' : ''
    const isMac = /Mac/i.test(platform) || /\bMac OS X\b/.test(userAgent)
    return isMac || (typeof devicePixelRatio === 'number' && devicePixelRatio > 1.25)
  } catch {
    return false
  }
}
const SCREEN_MAX_SIDE_LIGHT = 1280
const CAMERA_MAX_SIDE_LIGHT = 1280

// ── Module-scoped capture state (survives page unmount / window hide) ─────────
interface CapState {
  stream: MediaStream | null
  video: HTMLVideoElement | null
  canvas: HTMLCanvasElement | null
  timer: ReturnType<typeof setInterval> | null
  startTs: number
  inFlight: boolean
  sent: number
  dropped: number
}

interface CaptureBindingOwner {
  attemptId: string
  gateway: NonNullable<ReturnType<typeof $gateway.get>>
  generation: number
  sessionId: string
}

class CaptureBindingError extends Error {
  readonly ownsCurrentCapture: boolean

  constructor(cause: unknown, ownsCurrentCapture: boolean) {
    super(cause instanceof Error ? cause.message : String(cause))
    this.name = 'CaptureBindingError'
    this.cause = cause
    this.ownsCurrentCapture = ownsCurrentCapture
  }
}

const cap: CapState = {
  stream: null,
  video: null,
  canvas: null,
  timer: null,
  startTs: 0,
  inFlight: false,
  sent: 0,
  dropped: 0
}

let lastDebugCode: MmCaptureDebugState['code'] | '' = ''
let captureGeneration = 0
let boundSessionId = ''
let bindingGateway: NonNullable<ReturnType<typeof $gateway.get>> | null = null
let bindingKey = ''
let bindingPromise: Promise<void> | null = null
let bindingAttemptSeq = 0
let currentBindingAttempt = 0
let currentCaptureAttemptId = ''
let boundCaptureAttemptId = ''
const captureBindingOwners = new Map<string, CaptureBindingOwner>()
// Cleanup is an exact (session, client, generation, attempt) transaction.  A
// socket can disappear between the local stop and the backend ACK, so owners
// whose stop request failed must outlive the active capture state and be
// retried when that Gateway reconnects.  Otherwise a Monitor/Watcher can stay
// resident in the backend even though the renderer has released every track.
const pendingCaptureCleanupOwners = new Map<string, CaptureBindingOwner>()
const captureCleanupInFlight = new Set<string>()
let captureStartIntent = 0
const captureClientId = globalThis.crypto?.randomUUID?.() || `renderer-${Date.now()}-${Math.random()}`
const captureClientStartedAtMs = Date.now()

function setCaptureDebug(code: MmCaptureDebugState['code'], detail: string): void {
  const next = { code, detail }
  $mmCaptureDebug.set(next)
  if (code === lastDebugCode) return
  lastDebugCode = code
  const log = code === 'sending' || code === 'idle' ? console.info : console.warn
  log(`[mm-capture-debug] code=${code} detail=${detail}`)
}

async function announceSourceStarted(
  source: Exclude<MmSource, 'none'>,
  sid: string,
  gw: NonNullable<ReturnType<typeof $gateway.get>>,
  generation: number,
  captureAttemptId: string
): Promise<void> {
  setCaptureDebug('waiting_for_session', translateNow('multimodal.capture.initBackend'))
  const response = await gw.request<{ stale?: boolean }>(
    'multimodal.source_stopped',
    {
      session_id: sid,
      started: true,
      source_type: source,
      capture_client_id: captureClientId,
      capture_client_started_at_ms: captureClientStartedAtMs,
      capture_generation: generation,
      capture_attempt_id: captureAttemptId
    },
    SOURCE_ACTIVATION_TIMEOUT_MS
  )
  if (response?.stale) throw new Error(translateNow('multimodal.capture.staleRejected'))
}

function captureCleanupKey(owner: CaptureBindingOwner): string {
  return owner.attemptId || `${owner.sessionId}:${owner.generation}`
}

function flushPendingCaptureOwnerCleanups(): void {
  for (const [key, owner] of pendingCaptureCleanupOwners) {
    if (captureCleanupInFlight.has(key) || owner.gateway.connectionState !== 'open') {
      continue
    }

    captureCleanupInFlight.add(key)
    void owner.gateway.request('multimodal.source_stopped', {
        session_id: owner.sessionId,
        started: false,
        capture_client_id: captureClientId,
        capture_client_started_at_ms: captureClientStartedAtMs,
        capture_generation: owner.generation,
        capture_attempt_id: owner.attemptId || undefined
      }, SOURCE_ROLLBACK_TIMEOUT_MS)
      .then(() => {
        // A newer retry may have replaced the same compact legacy key. Delete
        // only the exact owner whose ACK we just observed.
        if (pendingCaptureCleanupOwners.get(key) === owner) {
          pendingCaptureCleanupOwners.delete(key)
        }
      })
      .catch(() => {
        // Keep the exact owner for the next Gateway-open transition. Backend
        // generation/attempt guards make repeated cleanup idempotent.
      })
      .finally(() => {
        captureCleanupInFlight.delete(key)
      })
  }
}

function queueCaptureOwnerCleanup(owner: CaptureBindingOwner): void {
  pendingCaptureCleanupOwners.set(captureCleanupKey(owner), owner)
  flushPendingCaptureOwnerCleanups()
}

function rollbackAnnouncedSource(
  sid: string,
  gw: NonNullable<ReturnType<typeof $gateway.get>>,
  generation: number,
  captureAttemptId: string
): void {
  // Source activation and the first buffered frame form one transaction. If
  // the latter fails, tell the backend that this exact generation ended so a
  // Watcher/Monitor cannot remain alive against a stream that never started.
  queueCaptureOwnerCleanup({
    attemptId: captureAttemptId,
    gateway: gw,
    generation,
    sessionId: sid
  })
}

function retireSupersededCaptureOwners(currentAttemptId: string): void {
  for (const [attemptId, owner] of captureBindingOwners) {
    if (attemptId === currentAttemptId) continue
    rollbackAnnouncedSource(
      owner.sessionId,
      owner.gateway,
      owner.generation,
      owner.attemptId
    )
    captureBindingOwners.delete(attemptId)
  }
}

/**
 * Bind the already-authorized media stream to the current main-chat session.
 *
 * A fresh desktop draft creates its backend session only after camera/screen
 * permission succeeds. The media button and the first prompt share the same
 * single-flight session creator, so whichever arrives first owns one ordinary
 * main-chat runtime. The prompt path still awaits this function as a first-frame
 * barrier before prompt.submit.
 */
export async function ensureCaptureBoundToSession(expectedSessionId?: string): Promise<void> {
  const source = $mmSource.get()
  if (source === 'none' || !cap.stream) return

  const sid = expectedSessionId || $mmSessionId.get()
  if (!sid) {
    setCaptureDebug('waiting_for_session', translateNow('multimodal.capture.authorized'))
    return
  }
  if ($mmSessionId.get() !== sid) {
    throw new Error(translateNow('multimodal.capture.sessionChanged'))
  }
  const gw = $gateway.get()
  if (!gw) throw new Error(translateNow('multimodal.capture.gatewayUnavailable'))
  if (gw.connectionState !== 'open') {
    setCaptureDebug('gateway_not_open', translateNow('multimodal.capture.gatewayState', gw.connectionState))
    throw new Error(translateNow('multimodal.capture.gatewayState', gw.connectionState))
  }

  const generation = captureGeneration
  const key = `${generation}:${sid}:${source}`
  if (boundSessionId === sid && bindingGateway === gw && cap.timer && cap.sent > 0) return
  if (bindingPromise && bindingKey === key && bindingGateway === gw) {
    await bindingPromise
    return
  }

  pauseFrameLoop(false)
  const attemptId = ++bindingAttemptSeq
  const captureAttemptId = `${captureClientId}:${attemptId}`
  currentBindingAttempt = attemptId
  currentCaptureAttemptId = captureAttemptId
  bindingKey = key
  bindingGateway = gw
  captureBindingOwners.set(captureAttemptId, {
    attemptId: captureAttemptId,
    gateway: gw,
    generation,
    sessionId: sid
  })
  const activeStream = cap.stream
  const stillOwnsBinding = (): boolean => (
    currentBindingAttempt === attemptId
    && currentCaptureAttemptId === captureAttemptId
    && captureGeneration === generation
    && $mmSource.get() === source
    && cap.stream === activeStream
    && $mmSessionId.get() === sid
    && $gateway.get() === gw
  )
  const assertStillOwnsBinding = (): void => {
    if (!stillOwnsBinding()) {
      throw new Error(translateNow('multimodal.capture.sessionChanged'))
    }
  }
  const pending = (async () => {
    // Once the request is dispatched, the backend may have committed this
    // generation even if the response times out/rejects. Treat every failure as
    // transactional and invalidate it before a retry.
    let activationDispatched = false
    try {
      activationDispatched = true
      await announceSourceStarted(source, sid, gw, generation, captureAttemptId)
      // Source/session/gateway may have changed while the backend initialized.
      // A stale acknowledgement must never restart delivery into the old chat.
      assertStillOwnsBinding()
      boundSessionId = sid
      boundCaptureAttemptId = captureAttemptId
      cap.startTs = performance.now()
      // A notification only proves socket enqueue. The frame handler runs in
      // the gateway worker pool, so the first frame uses request/ACK and waits
      // for FrameBuffer.push before prompt.submit may stamp its ask-time anchor.
      const firstFrameDeadline = performance.now() + 2_000
      let sent = false
      while (!sent && performance.now() < firstFrameDeadline) {
        const remainingMs = Math.max(1, firstFrameDeadline - performance.now())
        sent = await sendCapturedFrame(
          source, sid, gw, true, generation, captureAttemptId, remainingMs
        )
        if (!sent) await new Promise(resolve => setTimeout(resolve, 50))
      }
      if (!sent) throw new Error(translateNow('multimodal.capture.firstFrameTimeout'))
      assertStillOwnsBinding()
      // The new attempt is now both active and buffered. Retire any previous
      // runtime/transport owner so old Monitor/Watcher jobs cannot linger, and
      // keep the owner set bounded across reconnects.
      retireSupersededCaptureOwners(captureAttemptId)
      const boundStream = cap.stream
      if (!boundStream) throw new Error(translateNow('multimodal.capture.streamStopped'))
      if (source === 'screen' && boundStream.getAudioTracks().length > 0) {
        const voice = await import('./multimodal-voice')
        assertStillOwnsBinding()
        if (cap.stream !== boundStream) throw new Error(translateNow('multimodal.capture.streamReplaced'))
        voice.startEnvAudio(boundStream)
      }
      assertStillOwnsBinding()
      if (cap.stream !== boundStream) throw new Error(translateNow('multimodal.capture.streamReplaced'))
      startFrameLoop(source, sid, gw, generation, captureAttemptId)
    } catch (error) {
      if (activationDispatched) {
        rollbackAnnouncedSource(sid, gw, generation, captureAttemptId)
      }
      captureBindingOwners.delete(captureAttemptId)
      // Only the currently-owned binding attempt may invalidate local capture
      // state. A delayed failure from runtime A still rolls A back above, but
      // must not bump the generation or clear a successful newer B binding.
      const ownsCurrentCapture = stillOwnsBinding()
      if (ownsCurrentCapture) {
        captureGeneration += 1
        currentCaptureAttemptId = ''
        if (boundSessionId === sid && bindingGateway === gw) {
          boundSessionId = ''
          boundCaptureAttemptId = ''
          bindingGateway = null
        }
        setCaptureDebug(
          'notify_rejected',
          error instanceof Error ? error.message : String(error)
        )
      }
      throw new CaptureBindingError(error, ownsCurrentCapture)
    }
  })()
  bindingPromise = pending
  try {
    await pending
  } finally {
    if (bindingPromise === pending) {
      bindingPromise = null
      bindingKey = ''
    }
  }
}

// Picked screen source (from ScreenSourcePickerHost) — cached per capture start
// so we can stamp source_id/source_name onto every multimodal.frame push.
// Cleared on stopCapture; refreshed on each startScreenCapture.
let pickedSource: { id: string; name: string } | null = null

function pushStats(): void {
  $mmCapStats.set({ sent: cap.sent, dropped: cap.dropped })
}

async function attachStream(
  stream: MediaStream,
  source: MmSource,
  startIntent: number,
  sourceMeta: { id: string; name: string } | null = null
): Promise<void> {
  if (startIntent !== captureStartIntent) {
    stream.getTracks().forEach(track => track.stop())
    return
  }
  // Stop any existing capture first.
  stopCapture(false)
  pickedSource = sourceMeta
  cap.stream = stream
  const attachedGeneration = captureGeneration
  const video = document.createElement('video')
  video.muted = true
  video.playsInline = true
  video.srcObject = stream
  cap.video = video
  try {
    await video.play()
  } catch {
    /* autoplay of a muted stream is allowed; ignore transient errors */
  }
  // video.play() is asynchronous in Chromium. A New/profile/session boundary
  // or a newer capture may have stopped/replaced this stream while autoplay
  // was pending. Re-check ownership before publishing atoms or creating a
  // backend session; a stale permission grant must stay fully invisible.
  if (startIntent !== captureStartIntent || cap.stream !== stream) {
    stream.getTracks().forEach(track => track.stop())
    return
  }
  // A track ending (user clicks the OS "Stop sharing") tears everything down.
  const onEnded = (): void => {
    if (cap.stream === stream && captureGeneration === attachedGeneration) {
      stopCaptureAndNotify()
    }
  }
  stream.getVideoTracks().forEach(t => t.addEventListener('ended', onEnded, { once: true }))

  cap.startTs = performance.now()
  cap.sent = 0
  cap.dropped = 0
  pushStats()
  setCaptureDebug('video_not_ready', translateNow('multimodal.capture.videoNotReady'))
  // Publish the live stream so the VideoStage UI can mirror it in a <video>.
  $mmStream.set(stream)
  $mmSource.set(source)

  // Web parity: once media permission succeeds, sharing means recording — not
  // merely a local preview. A fresh draft creates the same main session that a
  // first text/mic input would create; session.create itself remains DB-lazy, so
  // a user who shares and closes without asking leaves no persisted empty chat.
  let attachedSessionId: string | null = $mmSessionId.get()
  let attachedGateway = $gateway.get()

  if (!attachedSessionId) {
    const ensureSession = draftCaptureSessionEnsurer

    if (!ensureSession) {
      if (cap.stream === stream && $mmSource.get() === source) {
        stopCapture()
      }
      throw new Error(translateNow('multimodal.capture.creatorNotReady'))
    }

    setCaptureDebug('waiting_for_session', translateNow('multimodal.capture.creatingSession'))

    try {
      attachedSessionId = await ensureSession()
    } catch (error) {
      if (cap.stream === stream && $mmSource.get() === source) {
        stopCapture()
      }
      throw error
    }

    // stop/New/profile-switch may have invalidated this attach while the shared
    // session.create was in flight. Never resurrect its stream or bind the late
    // runtime; the session creator owns its own route-token cleanup.
    if (
      startIntent !== captureStartIntent
      || cap.stream !== stream
      || $mmSource.get() !== source
    ) {
      return
    }

    if (!attachedSessionId || $mmSessionId.get() !== attachedSessionId) {
      stopCapture()
      throw new Error(translateNow('multimodal.capture.cancelled'))
    }

    attachedGateway = $gateway.get()
  }

  try {
    await ensureCaptureBoundToSession(attachedSessionId)
  } catch (error) {
    // The binding transaction already rolls back its exact backend source.
    // Tear down locally only when this attach still owns the same session and
    // gateway. A late failure from A must not stop the same MediaStream after
    // it has been successfully rebound to B.
    if (
      error instanceof CaptureBindingError
      && error.ownsCurrentCapture
      && cap.stream === stream
      && $mmSource.get() === source
      && $mmSessionId.get() === attachedSessionId
      && $gateway.get() === attachedGateway
    ) {
      stopCapture()
    }
    throw error
  }
}

async function sendCapturedFrame(
  source: MmSource,
  sid: string,
  gw: NonNullable<ReturnType<typeof $gateway.get>>,
  requireAck = false,
  expectedGeneration = captureGeneration,
  expectedAttemptId = boundCaptureAttemptId || currentCaptureAttemptId,
  ackTimeoutMs?: number
): Promise<boolean> {
  if (cap.inFlight) return false
  if (!sid || $mmSessionId.get() !== sid) {
    setCaptureDebug('waiting_for_session', translateNow('multimodal.capture.noSessionId'))
    return false
  }
  if (gw.connectionState !== 'open' || $gateway.get() !== gw) {
    setCaptureDebug('gateway_not_open', translateNow('multimodal.capture.gatewayState', gw.connectionState))
    return false
  }
  cap.inFlight = true
  try {
    const video = cap.video
    if (!video?.videoWidth || !video.videoHeight) {
      setCaptureDebug('video_not_ready', translateNow('multimodal.capture.videoNotReady'))
      return false
    }
    const data = await captureFrame(source)
    if (!data) {
      setCaptureDebug('encode_failed', translateNow('multimodal.capture.encodeFailed'))
      return false
    }
    // Encoding is asynchronous. A stop/session switch while toBlob/FileReader
    // runs must not let the completed JPEG escape into the old conversation.
    if (
      captureGeneration !== expectedGeneration ||
      currentCaptureAttemptId !== expectedAttemptId ||
      $mmSource.get() !== source ||
      !cap.stream ||
      $mmSessionId.get() !== sid ||
      $gateway.get() !== gw
    ) {
      return false
    }
    const params: Record<string, unknown> = {
      session_id: sid,
      ts: (performance.now() - cap.startTs) / 1000,
      jpeg_b64: data,
      source_type: source,
      capture_client_id: captureClientId,
      capture_client_started_at_ms: captureClientStartedAtMs,
      capture_generation: expectedGeneration
    }
    if (expectedAttemptId) params.capture_attempt_id = expectedAttemptId
    if (pickedSource) {
      params.source_id = pickedSource.id
      params.source_name = pickedSource.name
    }
    let buffered = 0
    if (requireAck) {
      const response = await gw.request<{ buffered?: boolean }>(
        'multimodal.frame',
        params,
        ackTimeoutMs
      )
      if (!response?.buffered) {
        setCaptureDebug('notify_rejected', translateNow('multimodal.capture.notifyRejected'))
        return false
      }
    } else {
      buffered = gw.notify('multimodal.frame', params)
      if (buffered < 0) {
        setCaptureDebug('notify_rejected', 'multimodal.frame not written to WebSocket')
        return false
      }
    }
    if (
      captureGeneration !== expectedGeneration ||
      currentCaptureAttemptId !== expectedAttemptId ||
      $mmSource.get() !== source ||
      !cap.stream ||
      $mmSessionId.get() !== sid ||
      $gateway.get() !== gw
    ) {
      return false
    }
    if (buffered > BUF_LIMIT) {
      cap.dropped += 1
      pushStats()
      setCaptureDebug('backpressure', `WebSocket bufferedAmount=${buffered}`)
      return true // this frame was written; skip only the next expensive tick
    }
    cap.sent += 1
    setCaptureDebug(
      'sending',
      `${source} ${video.videoWidth}x${video.videoHeight} jpeg_b64_chars=${data.length} sid=yes`
    )
    if (cap.sent === 1 || cap.sent % CAP_FPS === 0) pushStats()
    return true
  } catch (error) {
    setCaptureDebug('encode_failed', error instanceof Error ? error.message : String(error))
    return false
  } finally {
    cap.inFlight = false
  }
}

function startFrameLoop(
  source: MmSource,
  sid: string,
  gw: NonNullable<ReturnType<typeof $gateway.get>>,
  generation: number,
  captureAttemptId: string
): void {
  const period = Math.max(50, Math.round(1000 / CAP_FPS))
  if (cap.timer) clearInterval(cap.timer)
  cap.timer = setInterval(() => {
    if ($gateway.get() !== gw) {
      setCaptureDebug('gateway_unavailable', translateNow('multimodal.capture.gatewayUnavailable'))
      return
    }
    void sendCapturedFrame(source, sid, gw, false, generation, captureAttemptId)
  }, period)
}

async function captureFrame(source: MmSource): Promise<string | null> {
  const v = cap.video
  if (!v || !v.videoWidth) return null
  let w = v.videoWidth
  let h = v.videoHeight
  const isScreen = source === 'screen'
  const light = preferLightCapture()
  const maxSide = isScreen
    ? (light ? SCREEN_MAX_SIDE_LIGHT : SCREEN_MAX_SIDE)
    : (light ? Math.min(CAMERA_MAX_SIDE, CAMERA_MAX_SIDE_LIGHT) : CAMERA_MAX_SIDE)
  if (maxSide > 0 && Math.max(w, h) > maxSide) {
    const scale = maxSide / Math.max(w, h)
    w = Math.round(w * scale)
    h = Math.round(h * scale)
  }
  let cvs = cap.canvas
  if (!cvs) {
    cvs = document.createElement('canvas')
    cap.canvas = cvs
  }
  if (cvs.width !== w) cvs.width = w
  if (cvs.height !== h) cvs.height = h
  const ctx = cvs.getContext('2d')
  if (!ctx) return null
  // Prefer createImageBitmap(resize*) — it offloads the downscale off the
  // synchronous drawImage path (big win on Retina where the source is huge).
  // Fall back to a plain drawImage when unavailable or on any failure.
  let drew = false
  if (typeof createImageBitmap === 'function') {
    try {
      const bmp = await createImageBitmap(v, {
        resizeWidth: w,
        resizeHeight: h,
        resizeQuality: isScreen && !light ? 'medium' : 'low'
      } as ImageBitmapOptions)
      ctx.drawImage(bmp, 0, 0, w, h)
      bmp.close?.()
      drew = true
    } catch {
      /* fall back to sync path below */
    }
  }
  if (!drew) ctx.drawImage(v, 0, 0, w, h)
  const quality = isScreen
    ? (light ? JPEG_QUALITY_LIGHT : JPEG_QUALITY_SCREEN)
    : JPEG_QUALITY_CAMERA
  // Off-thread JPEG encode via toBlob, then base64 via FileReader (both async).
  const blob = await new Promise<Blob | null>(resolve => {
    cvs!.toBlob(resolve, 'image/jpeg', quality)
  })
  if (!blob) return null
  const dataUrl = await new Promise<string>((resolve, reject) => {
    const fr = new FileReader()
    fr.onload = () => resolve(String(fr.result || ''))
    fr.onerror = () => reject(fr.error)
    fr.readAsDataURL(blob)
  })
  // Strip the "data:image/jpeg;base64," prefix — server wants raw base64.
  const comma = dataUrl.indexOf(',')
  return comma >= 0 ? dataUrl.slice(comma + 1) : dataUrl
}

// ── Public actions ────────────────────────────────────────────────────────
export async function startCameraCapture(): Promise<void> {
  const startIntent = ++captureStartIntent
  let stream: MediaStream

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280, max: 1280 },
        height: { ideal: 720, max: 720 },
        frameRate: { ideal: 24 },
        facingMode: 'user'
      },
      audio: false
    })
  } catch (error) {
    // A conversation/profile boundary may have invalidated the permission
    // request while the OS prompt was open. Do not surface that stale prompt's
    // eventual rejection in the newly-selected chat.
    if (startIntent !== captureStartIntent) {
      return
    }
    throw error
  }
  if (startIntent !== captureStartIntent) {
    stream.getTracks().forEach(track => track.stop())
    return
  }
  await attachStream(stream, 'camera', startIntent)
}

export async function startScreenCapture(): Promise<void> {
  // Allocate ownership before opening the picker. New/profile/session switch
  // and a newer media-button click can now invalidate the picker itself, not
  // merely the later getDisplayMedia permission promise.
  const startIntent = ++captureStartIntent
  // Part 2: let the user pick which screen/window to share *before* we call
  // getDisplayMedia, then persist the choice into the main process so
  // setDisplayMediaRequestHandler can match against this exact source id.
  const bridge = window.hermesDesktop?.multimodalSourcePicker
  let picked: { id: string; name: string; shareAudio: boolean } | null = null
  try {
    const mod = await import('@/components/multimodal/screen-source-picker')
    picked = await mod.openSourcePicker()
  } catch {
    picked = null
  }
  if (startIntent !== captureStartIntent) {
    return
  }
  if (!picked) {
    // User cancelled the picker — abort cleanly, don't touch getDisplayMedia.
    return
  }
  try {
    await bridge?.setSelectedSource({ id: picked.id, name: picked.name })
  } catch {
    /* main process may reject; fall through to getDisplayMedia's fallback */
  }
  if (startIntent !== captureStartIntent) {
    return
  }
  // Treat the preload capability as authoritative. A stale/modified picker
  // must not request an unsupported system-audio path; main.cjs independently
  // enforces the Windows + macOS 13+ boundary.
  const shareAudio =
    window.hermesDesktop?.screenShareSystemAudio === true && picked.shareAudio
  const light = preferLightCapture()
  const width = light ? 1280 : 1920
  const height = light ? 720 : 1080
  let stream: MediaStream

  try {
    stream = await navigator.mediaDevices.getDisplayMedia({
      video: {
        width: { ideal: width, max: width },
        height: { ideal: height, max: height },
        frameRate: { ideal: 4, max: 4 }
      },
      audio: shareAudio
    })
  } catch (error) {
    if (startIntent !== captureStartIntent) {
      return
    }
    throw error
  }
  if (startIntent !== captureStartIntent) {
    stream.getTracks().forEach(track => track.stop())
    return
  }
  if (shareAudio && stream.getAudioTracks().length === 0) {
    const text = 'Screen share started but the system returned no audio track. On macOS, grant Argus “Screen & System Audio Recording” in System Settings → Privacy & Security, then restart the app.'
    console.warn(`[multimodal] ${text}`)
    void import('./multimodal-deep')
      .then(({ pushMmToast }) => {
        if (startIntent === captureStartIntent) {
          pushMmToast({ level: 'error', text })
        }
      })
      .catch(() => undefined)
  }
  await attachStream(
    stream,
    'screen',
    startIntent,
    { id: picked.id, name: picked.name }
  )
}

/** Stop capture + release the media stream. Does NOT notify the server. */
export function stopCapture(invalidatePendingStart = true): void {
  if (invalidatePendingStart) captureStartIntent += 1
  currentBindingAttempt = ++bindingAttemptSeq
  currentCaptureAttemptId = ''
  captureGeneration += 1
  boundSessionId = ''
  boundCaptureAttemptId = ''
  bindingGateway = null
  bindingKey = ''
  bindingPromise = null
  captureBindingOwners.clear()
  if (cap.timer) {
    clearInterval(cap.timer)
    cap.timer = null
  }
  // Tear down any env-audio recorder tied to a screen share. Lazy import to
  // avoid a static import cycle with the voice module.
  void import('./multimodal-voice').then(v => v.stopEnvAudio()).catch(() => undefined)
  if (cap.stream) {
    cap.stream.getTracks().forEach(t => {
      try {
        t.stop()
      } catch {
        /* noop */
      }
    })
    cap.stream = null
  }
  if (cap.video) {
    cap.video.srcObject = null
    cap.video = null
  }
  // A previous generation may still be inside createImageBitmap/toBlob.
  // Detach its canvas so a newly-started stream cannot share pixels with that
  // in-flight encode; generation checks still discard the old result.
  cap.canvas = null
  cap.inFlight = false
  pickedSource = null
  $mmStream.set(null)
  $mmSource.set('none')
  setCaptureDebug('idle', translateNow('multimodal.capture.notStarted'))
}

/** Stop capture AND tell the server the source is gone (multimodal.source_stopped). */
export function stopCaptureAndNotify(): void {
  const wasActive = $mmSource.get() !== 'none'
  const owners = [...captureBindingOwners.values()]
  if (owners.length === 0) {
    const ownerSid = boundSessionId || $mmSessionId.get()
    const ownerGateway = bindingGateway || $gateway.get()
    const attemptId = boundCaptureAttemptId || currentCaptureAttemptId
    if (ownerGateway && ownerSid) {
      owners.push({
        attemptId,
        gateway: ownerGateway,
        generation: captureGeneration,
        sessionId: ownerSid
      })
    }
  }
  stopCapture()
  if (!wasActive) return
  for (const owner of owners) {
    queueCaptureOwnerCleanup(owner)
  }
}

/** Whether capture is currently running (for tray / UI sync). */
export function isCapturing(): boolean {
  return $mmSource.get() !== 'none'
}

export interface MmCaptureAnchorSnapshot {
  source: Exclude<MmSource, 'none'>
  capture_attempt_id: string
  anchor_ts: number
}

/** Snapshot the capture's client-relative clock at an explicit user commit
 * gesture. The backend validates capture_attempt_id and maps this clock onto
 * its server-authoritative FrameBuffer timeline; it must never use this raw
 * value as a Frame.ts directly. Voice recognition can take seconds to finish,
 * so freezing at the second click excludes frames arriving during ASR flush. */
export function snapshotCaptureAnchor(): MmCaptureAnchorSnapshot | null {
  const source = $mmSource.get()

  if (!cap.stream || source === 'none' || cap.startTs <= 0) {
    return null
  }

  return {
    source,
    capture_attempt_id: boundCaptureAttemptId || currentCaptureAttemptId,
    anchor_ts: Math.max(0, (performance.now() - cap.startTs) / 1000)
  }
}

/** Pause the frame-push loop WITHOUT releasing the media stream/grant. Used on
 * gateway disconnect so we don't encode frames into a dead socket, while
 * keeping the camera/screen grant so a reconnect can resume seamlessly. */
export function pauseFrameLoop(markDisconnected = true): void {
  // A reconnect can reuse both the same Gateway object and runtime id. Make a
  // pending activation lose local ownership so resume starts a fresh attempt
  // instead of deduplicating onto a request tied to the dead transport.
  currentBindingAttempt = ++bindingAttemptSeq
  currentCaptureAttemptId = ''
  bindingPromise = null
  bindingKey = ''
  if (cap.timer) {
    clearInterval(cap.timer)
    cap.timer = null
  }
  if (markDisconnected && $mmSource.get() !== 'none' && cap.stream) {
    setCaptureDebug('gateway_not_open', translateNow('multimodal.video.connectionPaused'))
  }
}

/** Resume the frame-push loop after a reconnect, if a source is still active. */
export function resumeFrameLoop(): void {
  // A user may have stopped capture while the socket was down.  There is no
  // active source in that case, but the backend owner still needs its exact
  // stop transaction as soon as the Gateway reconnects.
  flushPendingCaptureOwnerCleanups()
  const source = $mmSource.get()
  if (source === 'none' || !cap.stream) return
  if (cap.timer) return // already running
  void ensureCaptureBoundToSession().catch(error => {
    setCaptureDebug(
      'notify_rejected',
      error instanceof Error ? error.message : String(error)
    )
  })
}
