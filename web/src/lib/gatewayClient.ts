/**
 * Browser WebSocket client for the tui_gateway JSON-RPC protocol.
 *
 * Speaks the exact same newline-delimited JSON-RPC dialect that the Ink TUI
 * drives over stdio. The server-side transport abstraction
 * (tui_gateway/transport.py + ws.py) routes the same dispatcher's writes
 * onto either stdout or a WebSocket depending on how the client connected.
 *
 *   const gw = new GatewayClient()
 *   await gw.connect()
 *   const { session_id } = await gw.request<{ session_id: string }>("session.create")
 *   gw.on("message.delta", (ev) => console.log(ev.payload?.text))
 *   await gw.request("prompt.submit", { session_id, text: "hi" })
 */

import { HERMES_BASE_PATH, getWsTicket } from "@/lib/api";

export type GatewayEventName =
  | "gateway.ready"
  | "session.info"
  | "message.start"
  | "message.delta"
  | "message.complete"
  | "thinking.delta"
  | "reasoning.delta"
  | "reasoning.available"
  | "status.update"
  | "tool.start"
  | "tool.progress"
  | "tool.complete"
  | "tool.generating"
  | "clarify.request"
  | "approval.request"
  | "sudo.request"
  | "secret.request"
  | "background.complete"
  | "error"
  | "skin.changed"
  | (string & {});

export interface GatewayEvent<P = unknown> {
  type: GatewayEventName;
  session_id?: string;
  payload?: P;
}

export type ConnectionState =
  | "idle"
  | "connecting"
  | "open"
  | "reconnecting"
  | "closed"
  | "error";

interface Pending {
  resolve: (v: unknown) => void;
  reject: (e: Error) => void;
  timer: ReturnType<typeof setTimeout>;
}

const DEFAULT_REQUEST_TIMEOUT_MS = 120_000;

/** Wildcard listener key: subscribe to every event regardless of type. */
const ANY = "*";

// Browser-side WS I/O diagnostic. Every 3s dumps one console.log row so we
// can tell — without asking the user to hunt the Network panel — whether
// "chat feels slow while screen-sharing" is (a) upstream frame notifs
// hogging the WS, (b) downstream events (anchor / ctx) hogging it, or (c)
// something else entirely. Also stashes the running counters on
// window.__gwStats so you can inspect them ad-hoc from Console.
const _GW_DIAG_INTERVAL_MS = 3000;
class _GwDiag {
  in_msgs = 0; in_bytes = 0; big_in = 0;
  out_msgs = 0; out_bytes = 0; big_out = 0;
  max_msg_in = 0; max_msg_out = 0;
  last = performance.now();
  timer: number | null = null;

  start() {
    if (this.timer !== null) return;
    this.timer = window.setInterval(() => this.flush(), _GW_DIAG_INTERVAL_MS);
    try { (window as any).__gwStats = this; } catch { /* noop */ }
  }
  stop() { if (this.timer !== null) { clearInterval(this.timer); this.timer = null; } }
  onIn(nbytes: number) {
    this.in_msgs++; this.in_bytes += nbytes;
    if (nbytes > 50_000) this.big_in++;
    if (nbytes > this.max_msg_in) this.max_msg_in = nbytes;
  }
  onOut(nbytes: number) {
    this.out_msgs++; this.out_bytes += nbytes;
    if (nbytes > 50_000) this.big_out++;
    if (nbytes > this.max_msg_out) this.max_msg_out = nbytes;
  }
  private flush() {
    if (this.in_msgs === 0 && this.out_msgs === 0) return;
    const dt = ((performance.now() - this.last) / 1000).toFixed(1);
    // eslint-disable-next-line no-console
    console.log(
      `%c[gw-io] Δ${dt}s  in=${this.in_msgs}/${(this.in_bytes/1024).toFixed(0)}KB (big=${this.big_in}, max=${(this.max_msg_in/1024).toFixed(0)}KB)  out=${this.out_msgs}/${(this.out_bytes/1024).toFixed(0)}KB (big=${this.big_out}, max=${(this.max_msg_out/1024).toFixed(0)}KB)`,
      "color:#0d6efd",
    );
    this.in_msgs = this.in_bytes = this.big_in = this.max_msg_in = 0;
    this.out_msgs = this.out_bytes = this.big_out = this.max_msg_out = 0;
    this.last = performance.now();
  }
}

export class GatewayClient {
  private ws: WebSocket | null = null;
  private reqId = 0;
  private pending = new Map<string, Pending>();
  private listeners = new Map<string, Set<(ev: GatewayEvent) => void>>();
  private _state: ConnectionState = "idle";
  private stateListeners = new Set<(s: ConnectionState) => void>();
  private diag = new _GwDiag();
  // ── Auto-reconnect ──────────────────────────────────────────────────────
  // After the first successful connect, an UNEXPECTED close triggers an
  // exponential-backoff reconnect so a flaky WiFi/VPN drop self-heals instead
  // of stranding the UI on a stale "connected" state with a black-holed socket.
  // Explicit close() disables it. onReconnect fires after a successful reconnect
  // so callers can re-establish session state (e.g. re-create the mm session).
  private _shouldReconnect = false;
  private _reconnectAttempts = 0;
  private _reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private _authToken?: string;
  private reconnectListeners = new Set<() => void>();
  /** Max backoff between reconnect attempts (ms). */
  private readonly _reconnectMaxMs = 15_000;

  /** Subscribe to "reconnected" events (fired after a successful auto-reconnect). */
  onReconnect(cb: () => void): () => void {
    this.reconnectListeners.add(cb);
    return () => this.reconnectListeners.delete(cb);
  }

  get state(): ConnectionState {
    return this._state;
  }

  private setState(s: ConnectionState) {
    if (this._state === s) return;
    this._state = s;
    for (const cb of this.stateListeners) cb(s);
  }

  onState(cb: (s: ConnectionState) => void): () => void {
    this.stateListeners.add(cb);
    cb(this._state);
    return () => this.stateListeners.delete(cb);
  }

  /** Subscribe to a specific event type. Returns an unsubscribe function. */
  on<P = unknown>(
    type: GatewayEventName,
    cb: (ev: GatewayEvent<P>) => void,
  ): () => void {
    let set = this.listeners.get(type);
    if (!set) {
      set = new Set();
      this.listeners.set(type, set);
    }
    set.add(cb as (ev: GatewayEvent) => void);
    return () => set!.delete(cb as (ev: GatewayEvent) => void);
  }

  /** Subscribe to every event (fires after type-specific listeners). */
  onAny(cb: (ev: GatewayEvent) => void): () => void {
    return this.on(ANY as GatewayEventName, cb);
  }

  async connect(token?: string): Promise<void> {
    if (this._state === "open" || this._state === "connecting") return;
    this._authToken = token;
    // From here on, an unexpected close should auto-reconnect (until close()).
    this._shouldReconnect = true;
    this.setState(this._reconnectAttempts > 0 ? "reconnecting" : "connecting");

    // Gated mode: legacy ``?token=`` is rejected by ``_ws_auth_ok``; the
    // SPA must fetch a single-use ticket via /api/auth/ws-ticket instead.
    // Explicit ``token`` overrides the gate check (test-only path).
    let authParamName: string;
    let authParamValue: string;
    if (token) {
      authParamName = "token";
      authParamValue = token;
    } else if (window.__HERMES_AUTH_REQUIRED__) {
      const { ticket } = await getWsTicket();
      authParamName = "ticket";
      authParamValue = ticket;
    } else {
      authParamName = "token";
      authParamValue = window.__ARGUS_SESSION_TOKEN__ ?? "";
      if (!authParamValue) {
        this.setState("error");
        throw new Error(
          "Session token not available — page must be served by the Argus dashboard",
        );
      }
    }

    const scheme = location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(
      `${scheme}//${location.host}${HERMES_BASE_PATH}/api/ws?${authParamName}=${encodeURIComponent(authParamValue)}`,
    );
    this.ws = ws;

    // Register message + close BEFORE awaiting open — the server emits
    // `gateway.ready` immediately after accept, so a listener attached
    // after the open promise resolves can race past it and drop the
    // initial skin payload.
    this.diag.start();
    ws.addEventListener("message", (ev) => {
      const raw = ev.data;
      this.diag.onIn(typeof raw === "string" ? raw.length : (raw?.byteLength ?? 0));
      let parsed: unknown;
      try {
        parsed = JSON.parse(raw);
      } catch {
        /* malformed frame — ignore */
        return;
      }
      // ★ C15: 解析成功后 dispatch 抛错说明是下游 handler 的 bug, 不是坏帧 —
      //   记录出来而不是和坏帧一起静默吞掉, 以免掩盖业务错误。
      try {
        this.dispatch(parsed as any);
      } catch (e) {
        console.error("[gateway] dispatch handler threw:", e);
      }
    });

    ws.addEventListener("close", () => {
      this.diag.stop();
      if (this.ws === ws) this.ws = null;
      this.rejectAllPending(new Error("WebSocket closed"));
      // Unexpected drop while we still want to be connected → reconnect.
      if (this._shouldReconnect) {
        this.setState("reconnecting");
        this.scheduleReconnect();
      } else {
        this.setState("closed");
      }
    });

    await new Promise<void>((resolve, reject) => {
      const onOpen = () => {
        ws.removeEventListener("error", onError);
        const wasReconnect = this._reconnectAttempts > 0;
        this._reconnectAttempts = 0;
        this.setState("open");
        if (wasReconnect) {
          for (const cb of this.reconnectListeners) {
            try { cb(); } catch { /* listener must not break the socket */ }
          }
        }
        resolve();
      };
      const onError = () => {
        ws.removeEventListener("open", onOpen);
        // Don't hard-fail into "error" if we're in an auto-reconnect cycle —
        // the close handler will schedule the next attempt.
        if (!this._shouldReconnect) this.setState("error");
        reject(new Error("WebSocket connection failed"));
      };
      ws.addEventListener("open", onOpen, { once: true });
      ws.addEventListener("error", onError, { once: true });
    });
  }

  /** Schedule an exponential-backoff reconnect (1s,2s,4s,… capped). */
  private scheduleReconnect(): void {
    if (this._reconnectTimer !== null || !this._shouldReconnect) return;
    const attempt = this._reconnectAttempts++;
    const delay = Math.min(1000 * 2 ** attempt, this._reconnectMaxMs);
    this._reconnectTimer = setTimeout(() => {
      this._reconnectTimer = null;
      if (!this._shouldReconnect) return;
      // connect() is a no-op if already open/connecting; on failure its own
      // close/error path re-schedules the next attempt.
      this.connect(this._authToken).catch(() => {
        if (this._shouldReconnect) this.scheduleReconnect();
      });
    }, delay);
  }

  close() {
    // Explicit close: stop auto-reconnecting.
    this._shouldReconnect = false;
    if (this._reconnectTimer !== null) {
      clearTimeout(this._reconnectTimer);
      this._reconnectTimer = null;
    }
    this._reconnectAttempts = 0;
    this.ws?.close();
    this.ws = null;
  }

  private dispatch(msg: Record<string, unknown>) {
    const id = msg.id as string | undefined;

    if (id !== undefined && this.pending.has(id)) {
      const p = this.pending.get(id)!;
      this.pending.delete(id);
      clearTimeout(p.timer);

      const err = msg.error as { message?: string } | undefined;
      if (err) p.reject(new Error(err.message ?? "request failed"));
      else p.resolve(msg.result);
      return;
    }

    if (msg.method !== "event") return;

    const params = (msg.params ?? {}) as GatewayEvent;
    if (typeof params.type !== "string") return;

    for (const cb of this.listeners.get(params.type) ?? []) cb(params);
    for (const cb of this.listeners.get(ANY) ?? []) cb(params);
  }

  private rejectAllPending(err: Error) {
    for (const p of this.pending.values()) {
      clearTimeout(p.timer);
      p.reject(err);
    }
    this.pending.clear();
  }

  /** Send a JSON-RPC request. Rejects on error response or timeout. */
  request<T = unknown>(
    method: string,
    params: Record<string, unknown> = {},
    timeoutMs = DEFAULT_REQUEST_TIMEOUT_MS,
  ): Promise<T> {
    if (!this.ws || this._state !== "open") {
      return Promise.reject(
        new Error(`gateway not connected (state=${this._state})`),
      );
    }

    const id = `w${++this.reqId}`;

    return new Promise<T>((resolve, reject) => {
      const timer = setTimeout(() => {
        if (this.pending.delete(id)) {
          reject(new Error(`request timed out: ${method}`));
        }
      }, timeoutMs);

      this.pending.set(id, {
        resolve: (v) => resolve(v as T),
        reject,
        timer,
      });

      try {
        this.ws!.send(JSON.stringify({ jsonrpc: "2.0", id, method, params }));
      } catch (e) {
        clearTimeout(timer);
        this.pending.delete(id);
        reject(e instanceof Error ? e : new Error(String(e)));
      }
    });
  }

  /** Fire-and-forget notification (no id, no ACK). Returns the size of the
   * outbound WS buffer AFTER the send — callers on high-frequency channels
   * (e.g. multimodal.frame at 2 fps × ~300KB) can use it as backpressure and
   * drop the next tick when the buffer is over some threshold.
   *
   * Returns -1 if the socket isn't open (silently drops — high-freq callers
   * should not treat this as an error). */
  notify(method: string, params: Record<string, unknown> = {}): number {
    if (!this.ws || this._state !== "open") return -1;
    try {
      this.ws.send(JSON.stringify({ jsonrpc: "2.0", method, params }));
      return this.ws.bufferedAmount;
    } catch {
      return -1;
    }
  }
}

declare global {
  interface Window {
    __ARGUS_SESSION_TOKEN__?: string;
    __HERMES_AUTH_REQUIRED__?: boolean;
  }
}
