/**
 * Small, deterministic state machine for Web Voice Dialog ownership.
 *
 * The UI flag represents user intent, while an ASR turn belongs to one live
 * gateway session. A reconnect/session rebind invalidates that live turn but
 * must not erase the continuous-mode intent. The next live session therefore
 * receives exactly one activation attempt. Manual ASR turns never enter this
 * state machine and are only cancelled at the boundary.
 */

export type VoiceDialogRecoveryPhase =
  | "off"
  | "waiting_for_session"
  | "starting"
  | "active";

export interface VoiceDialogActivation {
  attempt: number;
  sessionId: string;
}

export interface VoiceDialogRecoverySnapshot {
  desired: boolean;
  phase: VoiceDialogRecoveryPhase;
  sessionId: string;
  attempt: number;
}

export class VoiceDialogRecovery {
  private desired = false;
  private phase: VoiceDialogRecoveryPhase = "off";
  private sessionId = "";
  private attempt = 0;

  wantsVoiceDialog(): boolean {
    return this.desired;
  }

  snapshot(): VoiceDialogRecoverySnapshot {
    return {
      desired: this.desired,
      phase: this.phase,
      sessionId: this.sessionId,
      attempt: this.attempt,
    };
  }

  /** Record an ON intent and start now when a live session already exists. */
  enable(liveSessionId: string): VoiceDialogActivation | null {
    if (!this.desired) {
      this.desired = true;
      this.phase = "waiting_for_session";
      this.sessionId = "";
      // Invalidate any late completion from an earlier ON cycle.
      this.attempt += 1;
    }
    return this.sessionAvailable(liveSessionId);
  }

  /** OFF is terminal for every activation already in flight. */
  disable(): void {
    this.desired = false;
    this.phase = "off";
    this.sessionId = "";
    this.attempt += 1;
  }

  /**
   * A transport/session boundary invalidates the old ASR owner. Continuous
   * intent waits for the replacement session; an OFF/manual state stays OFF.
   */
  boundary(): string {
    const previousOwner = this.desired
      && (this.phase === "starting" || this.phase === "active")
      ? this.sessionId
      : "";
    this.sessionId = "";
    this.attempt += 1;
    this.phase = this.desired ? "waiting_for_session" : "off";
    return previousOwner;
  }

  /** Offer a newly created/resumed live session. Duplicate offers are no-ops. */
  sessionAvailable(liveSessionId: string): VoiceDialogActivation | null {
    const sessionId = liveSessionId.trim();
    if (!this.desired || !sessionId) return null;
    if (this.sessionId === sessionId
      && (this.phase === "starting" || this.phase === "active")) return null;

    const activation = { sessionId, attempt: ++this.attempt };
    this.sessionId = sessionId;
    this.phase = "starting";
    return activation;
  }

  /** Apply success only to the exact, still-current activation attempt. */
  activationSucceeded(activation: VoiceDialogActivation): boolean {
    if (!this.owns(activation)) return false;
    this.phase = "active";
    return true;
  }

  /** A current activation failure fails closed; stale failures do nothing. */
  activationFailed(activation: VoiceDialogActivation): boolean {
    if (!this.owns(activation)) return false;
    this.disable();
    return true;
  }

  /**
   * Session establishment finished without a live id. Do not leave the third
   * button claiming ON forever; the user can retry after reconnecting.
   */
  sessionUnavailable(): boolean {
    if (!this.desired || this.phase !== "waiting_for_session") return false;
    this.disable();
    return true;
  }

  owns(activation: VoiceDialogActivation): boolean {
    return this.desired
      && this.phase === "starting"
      && this.sessionId === activation.sessionId
      && this.attempt === activation.attempt;
  }
}
