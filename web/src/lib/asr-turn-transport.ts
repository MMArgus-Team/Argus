/**
 * Ordered transport for one browser microphone turn.
 *
 * Local capture is intentionally allowed to start before ``asr_start`` has
 * completed. PCM produced during that window is retained in a bounded rolling
 * pre-roll and flushed, in order, once the backend is ready. ``finish`` is
 * serialized behind accepted audio writes; ``cancel`` deliberately overtakes
 * a cold start so the backend can tombstone it.
 */

import { translateNow } from "@/i18n/runtime";

export type AsrTurnMode = "manual_turn" | "continuous";
export type AsrStopDisposition = "finish" | "cancel";

export interface AsrRpcClient {
  request(
    method: string,
    params: Record<string, unknown>,
    timeoutMs?: number,
  ): Promise<unknown>;
}

export interface AsrTurnHandle {
  sessionId: string;
  turnId: string;
  mode: AsrTurnMode;
  ready: Promise<unknown>;
}

export interface AsrStopResult {
  ok?: boolean;
  submitted?: boolean;
  transcript?: string;
  reason?: string;
  error?: string;
}

/** Convert a manual finish result into an actionable, user-facing diagnostic. */
export function asrFinishFailureMessage(raw: unknown): string {
  if (!raw || typeof raw !== "object") return "";
  const result = raw as AsrStopResult;
  if (result.ok !== false && result.submitted !== false) return "";
  if (result.reason === "empty") {
    return translateNow("multimodal.errors.asrEmpty");
  }
  if (result.reason === "upstream_error") {
    return result.error
      ? translateNow("multimodal.errors.asrServiceError", result.error)
      : translateNow("multimodal.errors.asrServiceUnavailable");
  }
  if (result.reason === "audio_delivery_failed") {
    return translateNow("multimodal.errors.asrAudioDeliveryFailed");
  }
  return result.reason
    ? translateNow("multimodal.errors.asrSubmitFailedWithReason", result.reason)
    : translateNow("multimodal.errors.asrSubmitFailed");
}

interface QueuedPcm {
  pcmB64: string;
  bytes: number;
}

interface ActiveTurn extends AsrTurnHandle {
  backendReady: boolean;
  closeDisposition: AsrStopDisposition | null;
  queue: QueuedPcm[];
  queueBytes: number;
  pendingAudio: Set<Promise<unknown>>;
  audioFailure: string | null;
  finishPromise: Promise<unknown> | null;
  cancelPromise: Promise<unknown> | null;
}

interface EventOwner {
  sessionId: string;
  turnId: string;
  mode: AsrTurnMode;
}

// A legal cold activation can include agent init + runtime promotion. Match
// the explicit 210s control-RPC timeout with headroom while still bounding a
// single renderer turn (~7.7 MB raw PCM / ~10.2 MB base64 at 240 seconds).
export const ASR_PRE_ROLL_SECONDS = 240;
export const ASR_PRE_ROLL_MAX_BYTES = 16_000 * 2 * ASR_PRE_ROLL_SECONDS;
export const ASR_CONTROL_TIMEOUT_MS = 210_000;

export function decodedBase64Bytes(value: string): number {
  if (!value) return 0;
  const padding = value.endsWith("==") ? 2 : value.endsWith("=") ? 1 : 0;
  return Math.max(0, Math.floor(value.length * 3 / 4) - padding);
}

export function createAsrTurnId(): string {
  const random = typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
  return `asr_${random}`;
}

/**
 * A slow stop result may settle after a session boundary and a newer capture.
 * Only the exact UI generation on the same live sid may surface its error or
 * clear mic/preview state.
 */
export function ownsAsrStopUi(
  currentGeneration: number,
  stopGeneration: number,
  currentSessionId: string,
  turnSessionId: string,
): boolean {
  return currentGeneration === stopGeneration
    && Boolean(currentSessionId)
    && currentSessionId === turnSessionId;
}

/**
 * Owns at most one ASR turn. A new instance should be kept for the lifetime of
 * one page connection; event ownership remains valid through a graceful
 * ``finish`` so the final event racing the stop response can still render.
 */
export class AsrTurnTransport {
  private active: ActiveTurn | null = null;
  private eventOwner: EventOwner | null = null;
  private lastStop: { key: string; promise: Promise<unknown> } | null = null;
  private readonly rpc: AsrRpcClient;
  private readonly maxPreRollBytes: number;

  constructor(rpc: AsrRpcClient, maxPreRollBytes = ASR_PRE_ROLL_MAX_BYTES) {
    this.rpc = rpc;
    this.maxPreRollBytes = maxPreRollBytes;
  }

  begin(sessionId: string, mode: AsrTurnMode, turnId = createAsrTurnId()): AsrTurnHandle {
    if (this.active) throw new Error("an ASR turn is already active");
    if (!sessionId || !turnId) throw new Error("session_id and turn_id are required");

    let resolveReady!: (value: unknown) => void;
    let rejectReady!: (reason?: unknown) => void;
    const ready = new Promise<unknown>((resolve, reject) => {
      resolveReady = resolve;
      rejectReady = reject;
    });
    // The page observes ``ready``. This extra handler prevents a transient
    // unhandled-rejection report if permission setup takes longer to settle.
    void ready.catch(() => undefined);

    const turn: ActiveTurn = {
      sessionId,
      turnId,
      mode,
      ready,
      backendReady: false,
      closeDisposition: null,
      queue: [],
      queueBytes: 0,
      pendingAudio: new Set(),
      audioFailure: null,
      finishPromise: null,
      cancelPromise: null,
    };
    this.active = turn;
    this.eventOwner = { sessionId, turnId, mode };
    this.lastStop = null;

    void this.rpc.request("multimodal.asr_start", {
      session_id: sessionId,
      turn_id: turnId,
      mode,
    }, ASR_CONTROL_TIMEOUT_MS).then((rawResult) => {
      const result = rawResult && typeof rawResult === "object"
        ? rawResult as Record<string, unknown>
        : null;
      if (!result?.enabled) {
        throw new Error("streaming ASR is disabled");
      }
      if (typeof result.turn_id === "string" && result.turn_id !== turnId) {
        throw new Error("backend returned a mismatched ASR turn_id");
      }
      turn.backendReady = true;
      if (turn.closeDisposition !== "cancel") this.flushQueued(turn);
      else this.clearQueue(turn);
      resolveReady(result);
    }).catch((error) => {
      this.clearQueue(turn);
      if (this.eventOwner?.turnId === turnId) this.eventOwner = null;
      rejectReady(error);
    });

    return { sessionId, turnId, mode, ready };
  }

  /** Accept PCM from the worklet. Returns false after close or for stale turns. */
  pushPcm(sessionId: string, turnId: string, pcmB64: string): boolean {
    const turn = this.active;
    if (!turn || turn.sessionId !== sessionId || turn.turnId !== turnId
      || turn.closeDisposition || !pcmB64) return false;

    if (turn.backendReady) {
      this.scheduleAudio(turn, pcmB64);
      return true;
    }

    const entry = { pcmB64, bytes: decodedBase64Bytes(pcmB64) };
    turn.queue.push(entry);
    turn.queueBytes += entry.bytes;
    // Rolling retention protects memory if a backend handshake stalls. Under
    // normal operation no packet is dropped; the bound holds four full minutes.
    while (turn.queueBytes > this.maxPreRollBytes && turn.queue.length > 1) {
      const dropped = turn.queue.shift();
      turn.queueBytes -= dropped?.bytes || 0;
    }
    return true;
  }

  /**
   * Close the active turn. Repeated calls for the same turn share one promise,
   * so a double-click cannot submit twice.
   */
  stop(
    sessionId: string,
    turnId: string,
    disposition: AsrStopDisposition,
    extraParams: Record<string, unknown> = {},
  ): Promise<unknown> {
    const key = `${sessionId}\n${turnId}`;
    const turn = this.active;
    if (!turn || turn.sessionId !== sessionId || turn.turnId !== turnId) {
      if (this.lastStop?.key === key) return this.lastStop.promise;
      return Promise.resolve({ ok: false, reason: "stale_turn" });
    }

    if (disposition === "cancel") {
      if (turn.cancelPromise) return turn.cancelPromise;
      // A conversation/profile boundary is stronger than a user-requested
      // finish already in progress. Send a distinct exact-turn cancellation;
      // the backend atomically downgrades the stopping turn to cancel.
      turn.closeDisposition = "cancel";
      this.clearQueue(turn);
      if (this.eventOwner?.turnId === turnId) this.eventOwner = null;
      // Cancel must overtake a cold/slow start. The backend records an exact
      // turn-id tombstone even if promotion/start has not published the ASR
      // session yet, preventing a late start from resurrecting the mic.
      const cancelled = this.rpc.request("multimodal.asr_stop", {
        session_id: sessionId,
        turn_id: turnId,
        disposition: "cancel",
      }).finally(() => {
        if (this.active === turn) this.active = null;
      });
      turn.cancelPromise = cancelled;
      this.lastStop = { key, promise: cancelled };
      return cancelled;
    }

    if (turn.closeDisposition === "cancel") {
      return turn.cancelPromise || Promise.resolve({ ok: false, reason: "cancelled" });
    }
    if (turn.finishPromise) return turn.finishPromise;
    turn.closeDisposition = "finish";
    // Another stop("cancel") may run while this async finish is awaiting the
    // backend/audio acknowledgements. Keep the check behind a function so it
    // reads the live field each time instead of TypeScript narrowing the value
    // permanently to "finish" across awaits.
    const wasCancelled = () => turn.closeDisposition === "cancel";

    const stopPromise = (async () => {
      try {
        try {
          await turn.ready;
        } catch {
          // No upstream session was opened, therefore there is nothing to stop.
          return { ok: false, reason: "start_failed" };
        }
        if (wasCancelled()) {
          return await (turn.cancelPromise
            || Promise.resolve({ ok: false, reason: "cancelled" }));
        }
        this.flushQueued(turn);
        // Requests are dispatched immediately (the WebSocket preserves call
        // order), so 5 Hz PCM never waits one network RTT per packet. Finish
        // still waits until every accepted write has settled before stopping.
        await Promise.allSettled(Array.from(turn.pendingAudio));
        if (wasCancelled()) {
          return await (turn.cancelPromise
            || Promise.resolve({ ok: false, reason: "cancelled" }));
        }
        if (turn.audioFailure) {
          // A manual push-to-talk turn is atomic from the user's perspective.
          // Never submit a transcript assembled from only a subset of its PCM.
          turn.closeDisposition = "cancel";
          if (this.eventOwner?.turnId === turnId) this.eventOwner = null;
          turn.cancelPromise = this.rpc.request("multimodal.asr_stop", {
            session_id: sessionId,
            turn_id: turnId,
            disposition: "cancel",
          });
          await turn.cancelPromise.catch(() => undefined);
          return {
            ok: false,
            submitted: false,
            reason: "audio_delivery_failed",
          };
        }
        return await this.rpc.request("multimodal.asr_stop", {
          ...extraParams,
          session_id: sessionId,
          turn_id: turnId,
          disposition,
        }, ASR_CONTROL_TIMEOUT_MS);
      } finally {
        if (this.active === turn) this.active = null;
      }
    })();
    turn.finishPromise = stopPromise;
    this.lastStop = { key, promise: stopPromise };
    return stopPromise;
  }

  /** Reject untagged and stale events; legacy broadcasts must not cross turns. */
  ownsEvent(sessionId: unknown, turnId: unknown): boolean {
    const owner = this.eventOwner;
    return Boolean(
      owner
      && typeof sessionId === "string"
      && typeof turnId === "string"
      && sessionId === owner.sessionId
      && turnId === owner.turnId,
    );
  }

  /** A manual final is terminal. Continuous mode keeps accepting later finals. */
  noteFinal(sessionId: string, turnId: string): void {
    if (!this.ownsEvent(sessionId, turnId)) return;
    if (this.eventOwner?.mode === "manual_turn") this.eventOwner = null;
  }

  current(): Pick<AsrTurnHandle, "sessionId" | "turnId" | "mode"> | null {
    const turn = this.active;
    return turn ? { sessionId: turn.sessionId, turnId: turn.turnId, mode: turn.mode } : null;
  }

  private scheduleAudio(turn: ActiveTurn, pcmB64: string): void {
    const pending = this.rpc.request("multimodal.asr_audio", {
      session_id: turn.sessionId,
      turn_id: turn.turnId,
      pcm_b64: pcmB64,
    }).then((raw) => {
      if (raw && typeof raw === "object"
        && (raw as Record<string, unknown>).ok === false) {
        throw new Error(String((raw as Record<string, unknown>).reason
          || "audio upload rejected"));
      }
      return raw;
    }).catch((error) => {
      turn.audioFailure = error instanceof Error ? error.message : String(error);
      throw error;
    });
    turn.pendingAudio.add(pending);
    void pending.finally(() => turn.pendingAudio.delete(pending)).catch(() => undefined);
  }

  private flushQueued(turn: ActiveTurn): void {
    const queued = turn.queue.splice(0);
    turn.queueBytes = 0;
    for (const entry of queued) this.scheduleAudio(turn, entry.pcmB64);
  }

  private clearQueue(turn: ActiveTurn): void {
    turn.queue = [];
    turn.queueBytes = 0;
  }
}
