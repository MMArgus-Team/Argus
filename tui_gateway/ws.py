"""WebSocket transport for the tui_gateway JSON-RPC server.

Reuses :func:`tui_gateway.server.dispatch` verbatim so every RPC method, every
slash command, every approval/clarify/sudo flow, and every agent event flows
through the same handlers whether the client is Ink over stdio or an iOS /
web client over WebSocket.

Wire protocol
-------------
Identical to stdio: newline-delimited JSON-RPC in both directions. The server
emits a ``gateway.ready`` event immediately after connection accept, then
echoes responses/events for inbound requests. No framing differences.

Mounting
--------
    from fastapi import WebSocket
    from tui_gateway.ws import handle_ws

    @app.websocket("/api/ws")
    async def ws(ws: WebSocket):
        await handle_ws(ws)
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import socket
import time
from typing import Any

from tui_gateway import server

_log = logging.getLogger(__name__)

# Max seconds a pool-dispatched handler will block waiting for the event loop
# to flush a WS frame before we mark the transport dead. Protects handler
# threads from a wedged socket.
_WS_WRITE_TIMEOUT_S = 10.0
_WS_LOG_PAYLOAD_PREVIEW = 240

# ── WS I/O diagnostic ────────────────────────────────────────────────────
# Rolling per-second counters so we can tell whether "chat feels slow while
# screen-sharing" is:
#   (a) upstream — big multimodal.frame notifs blocking receive_text on the
#       loop, or
#   (b) downstream — big multimodal.anchor / .ctx events blocking send_text,
# without needing to run tcpdump. Emits one line every N seconds and only when
# there was any activity, so idle sessions stay silent. Counts + max latency
# are the interesting bits; per-frame log would be far too spammy at 2 fps.
_DIAG_INTERVAL_S = 5.0

class _WsIoDiag:
    __slots__ = (
        "peer", "last_flush", "in_frames", "in_bytes",
        "out_frames", "out_bytes", "big_in", "big_out",
        "max_recv_gap_s", "_last_recv_wall", "max_send_s",
    )
    def __init__(self, peer: str) -> None:
        self.peer = peer
        self.last_flush = time.monotonic()
        self.in_frames = 0
        self.in_bytes = 0
        self.out_frames = 0
        self.out_bytes = 0
        self.big_in = 0
        self.big_out = 0
        self.max_recv_gap_s = 0.0
        self._last_recv_wall = time.monotonic()
        self.max_send_s = 0.0

    def on_recv(self, nbytes: int) -> None:
        now = time.monotonic()
        gap = now - self._last_recv_wall
        if gap > self.max_recv_gap_s:
            self.max_recv_gap_s = gap
        self._last_recv_wall = now
        self.in_frames += 1
        self.in_bytes += nbytes
        if nbytes > 50_000:
            self.big_in += 1
        self._maybe_flush(now)

    def on_send(self, nbytes: int, elapsed_s: float) -> None:
        now = time.monotonic()
        self.out_frames += 1
        self.out_bytes += nbytes
        if nbytes > 50_000:
            self.big_out += 1
        if elapsed_s > self.max_send_s:
            self.max_send_s = elapsed_s
        self._maybe_flush(now)

    def _maybe_flush(self, now: float) -> None:
        if now - self.last_flush < _DIAG_INTERVAL_S:
            return
        if self.in_frames == 0 and self.out_frames == 0:
            self.last_flush = now
            return
        _log.info(
            "[ws-io] peer=%s in=%d/%dKB (big=%d) out=%d/%dKB (big=%d) "
            "max_recv_gap=%.2fs max_send=%.2fs",
            self.peer,
            self.in_frames, self.in_bytes // 1024, self.big_in,
            self.out_frames, self.out_bytes // 1024, self.big_out,
            self.max_recv_gap_s, self.max_send_s,
        )
        self.in_frames = self.in_bytes = self.big_in = 0
        self.out_frames = self.out_bytes = self.big_out = 0
        self.max_recv_gap_s = 0.0
        self.max_send_s = 0.0
        self.last_flush = now

# Keep starlette optional at import time; handle_ws uses the real class when
# it's available and falls back to a generic Exception sentinel otherwise.
try:
    from starlette.websockets import WebSocketDisconnect as _WebSocketDisconnect
except ImportError:  # pragma: no cover - starlette is a required install path
    _WebSocketDisconnect = Exception  # type: ignore[assignment]


class WSTransport:
    """Per-connection WS transport.

    ``write`` is safe to call from any thread *other than* the event loop
    thread that owns the socket. Pool workers (the only real caller) run in
    their own threads, so marshalling onto the loop via
    :func:`asyncio.run_coroutine_threadsafe` + ``future.result()`` is correct
    and deadlock-free there.

    When called from the loop thread itself (e.g. by ``handle_ws`` for an
    inline response) the same call would deadlock: we'd schedule work onto
    the loop we're currently blocking. We detect that case and fire-and-
    forget instead. Callers that need to know when the bytes are on the wire
    should use :meth:`write_async` from the loop thread.
    """

    def __init__(
        self,
        ws: Any,
        loop: asyncio.AbstractEventLoop,
        *,
        peer: str = "unknown",
        diag: "_WsIoDiag | None" = None,
    ) -> None:
        self._ws = ws
        self._loop = loop
        self._peer = peer
        self._closed = False
        self._diag = diag

    def write(self, obj: dict) -> bool:
        if self._closed:
            return False

        line = json.dumps(obj, ensure_ascii=False)

        try:
            on_loop = asyncio.get_running_loop() is self._loop
        except RuntimeError:
            on_loop = False

        if on_loop:
            # Fire-and-forget — don't block the loop waiting on itself.
            self._loop.create_task(self._safe_send(line))
            return True

        try:
            from agent.async_utils import safe_schedule_threadsafe
            fut = safe_schedule_threadsafe(self._safe_send(line), self._loop)
            if fut is None:
                self._closed = True
                return False
            fut.result(timeout=_WS_WRITE_TIMEOUT_S)
            return not self._closed
        except concurrent.futures.TimeoutError:  # builtin TimeoutError on 3.11+
            # The event loop is stalled (GIL-heavy agent turn, delegation
            # running N children), NOT the socket dead. The send coroutine is
            # already scheduled and will flush once the loop breathes — latching
            # _closed here permanently silenced live windows after one slow
            # write (the "subagent window shows zero streaming" bug). Unblock
            # the worker thread and keep the transport alive; _safe_send latches
            # on a real socket error when the frame actually fails.
            _log.warning(
                "ws write slow (loop stalled >%ss) peer=%s — frame left in flight",
                _WS_WRITE_TIMEOUT_S, self._peer,
            )
            return not self._closed
        except Exception as exc:
            self._closed = True
            _log.warning(
                "ws write failed peer=%s error_type=%s error=%s",
                self._peer, type(exc).__name__, exc,
            )
            return False

    async def write_async(self, obj: dict) -> bool:
        """Send from the owning event loop. Awaits until the frame is on the wire."""
        if self._closed:
            return False
        await self._safe_send(json.dumps(obj, ensure_ascii=False))
        return not self._closed

    async def _safe_send(self, line: str) -> None:
        t0 = time.monotonic()
        try:
            await self._ws.send_text(line)
        except Exception as exc:
            self._closed = True
            _log.warning(
                "ws send failed peer=%s error_type=%s error=%s",
                self._peer, type(exc).__name__, exc,
            )
            return
        if self._diag is not None:
            self._diag.on_send(len(line), time.monotonic() - t0)

    def close(self) -> None:
        self._closed = True


def _ws_peer_label(ws: Any) -> str:
    """Return ``host:port`` when available, else a stable placeholder."""
    client = getattr(ws, "client", None)
    if client is None:
        return "unknown"
    host = getattr(client, "host", None) or "unknown"
    port = getattr(client, "port", None)
    return f"{host}:{port}" if port is not None else host


def _disable_nagle(ws: Any) -> None:
    """Disable Nagle so streamed JSON-RPC frames go out individually.

    Without it the kernel coalesces the small per-token frames, so a burst after
    the model's think-pause lands on the client in one tick and no client-side
    smoothing can recover the cadence. GUI/WS only; chat platforms don't hit
    this path. Best-effort — skip silently if the socket isn't reachable.
    """
    try:
        scope = getattr(ws, "scope", None) or {}
        transport = (scope.get("extensions") or {}).get("transport") or getattr(ws, "transport", None)
        sock = transport.get_extra_info("socket") if transport is not None else None
        if sock is not None:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except Exception as exc:  # pragma: no cover - best-effort tuning
        _log.debug("ws TCP_NODELAY skip: %s", exc)


def _parse_and_dispatch(line: str, transport: "WSTransport") -> dict | None:
    """Parse a bulk JSON-RPC line and dispatch it, entirely off the loop thread.

    Runs inside ``asyncio.to_thread`` for large base64-carrying messages
    (multimodal.frame / env_audio / user_audio) so the event loop never spends
    the tens-of-ms of GIL-held ``json.loads`` on a few-hundred-KB string — which
    would otherwise stall the same loop that flushes the agent's SSE tokens.

    Returns the dispatch response dict (``None`` for the fire-and-forget
    notifications these bulk methods always are). A malformed frame is logged
    and swallowed (returns ``None``) rather than raising: a dropped frame is
    harmless, and there is no ``id`` to reply to on a notification anyway.
    """
    try:
        req = json.loads(line)
    except json.JSONDecodeError as exc:
        _log.warning("ws bulk parse error error=%s", exc)
        return None
    return server.dispatch(req, transport)


async def handle_ws(ws: Any) -> None:
    """Run one WebSocket session. Wire-compatible with ``tui_gateway.entry``."""
    peer = _ws_peer_label(ws)
    transport: WSTransport | None = None
    messages = 0
    parse_errors = 0
    dispatch_crashes = 0
    send_failures = 0
    disconnect_reason = "not_connected"

    try:
        await ws.accept()
        disconnect_reason = "connected"
        # Push small streamed frames out immediately instead of letting Nagle
        # batch them — keeps the live token cadence intact for GUI clients.
        _disable_nagle(ws)
        _log.info("ws accepted peer=%s", peer)

        diag = _WsIoDiag(peer=peer)
        transport = WSTransport(
            ws, asyncio.get_running_loop(), peer=peer, diag=diag,
        )

        ready_ok = await transport.write_async(
            {
                "jsonrpc": "2.0",
                "method": "event",
                "params": {
                    "type": "gateway.ready",
                    "payload": {"skin": server.resolve_skin()},
                },
            }
        )
        if not ready_ok:
            disconnect_reason = "ready_send_failed"
            send_failures += 1
            _log.error("ws ready frame send failed peer=%s", peer)
            return

        while True:
            try:
                raw = await ws.receive_text()
            except _WebSocketDisconnect as exc:
                disconnect_reason = (
                    "client_disconnect("
                    f"code={getattr(exc, 'code', None)},"
                    f"reason={getattr(exc, 'reason', None)})"
                )
                break
            except Exception:
                disconnect_reason = "receive_failed"
                _log.exception("ws receive failed peer=%s", peer)
                break

            line = raw.strip()
            if not line:
                continue
            messages += 1
            diag.on_recv(len(raw))

            # ── Bulk-payload fast path (loop-starvation fix) ─────────────────
            # multimodal.frame (screen/camera JPEG @ ~2fps, 150-400KB base64),
            # env_audio and user_audio all carry a large base64 blob. json.loads
            # of a few-hundred-KB string holds the GIL for tens of ms — and it
            # runs on THIS loop thread, the same one that flushes the agent's
            # SSE tokens via WSTransport.write's send coroutine. So a frame
            # parse periodically stalls the loop and token sends hit the
            # "loop stalled" write timeout → the visible "streaming freezes
            # while screen-sharing" bug. These handlers are already routed to
            # the thread pool by dispatch (_LONG_HANDLERS), but the parse
            # happened BEFORE dispatch, defeating the offload for the single
            # most expensive step. Detect them by a cheap prefix scan (method
            # name sits near the front of the JSON; cap the window so we never
            # scan the payload) and offload parse+dispatch together so the loop
            # thread never touches the big string.
            head = line[:120]
            is_bulk = (
                '"multimodal.frame"' in head
                or '"multimodal.env_audio"' in head
                or '"multimodal.user_audio"' in head
            )
            if is_bulk:
                try:
                    resp = await asyncio.to_thread(
                        _parse_and_dispatch, line, transport
                    )
                except Exception:
                    dispatch_crashes += 1
                    _log.exception("ws bulk dispatch crash peer=%s", peer)
                    resp = None
                # Bulk messages are fire-and-forget notifications (no id):
                # dispatch returns None and the pool worker writes any response
                # itself. Nothing to write back from the loop here.
                if resp is not None:
                    await transport.write_async(resp)
                continue

            try:
                req = json.loads(line)
            except json.JSONDecodeError as exc:
                parse_errors += 1
                _log.warning(
                    "ws parse error peer=%s index=%d error=%s payload=%r",
                    peer,
                    messages,
                    exc,
                    line[:_WS_LOG_PAYLOAD_PREVIEW],
                )
                ok = await transport.write_async(
                    {
                        "jsonrpc": "2.0",
                        "error": {"code": -32700, "message": "parse error"},
                        "id": None,
                    }
                )
                if not ok:
                    disconnect_reason = "send_failed_after_parse_error"
                    send_failures += 1
                    _log.warning("ws parse-error reply send failed peer=%s", peer)
                    break
                continue

            # dispatch() may schedule long handlers on the pool; it returns
            # None in that case and the worker writes the response itself via
            # the transport we pass in (a separate thread, so transport.write
            # is the safe path there). For inline handlers it returns the
            # response dict, which we write here from the loop.
            req_id = req.get("id") if isinstance(req, dict) else None
            req_method = req.get("method") if isinstance(req, dict) else None
            try:
                resp = await asyncio.to_thread(server.dispatch, req, transport)
            except Exception:
                dispatch_crashes += 1
                _log.exception(
                    "ws dispatch crash peer=%s id=%s method=%s",
                    peer,
                    req_id,
                    req_method,
                )
                ok = await transport.write_async(
                    {
                        "jsonrpc": "2.0",
                        "error": {"code": -32603, "message": "internal error"},
                        "id": req_id if req_id is not None else None,
                    }
                )
                if not ok:
                    disconnect_reason = "send_failed_after_dispatch_crash"
                    send_failures += 1
                    _log.warning(
                        "ws dispatch-crash reply send failed peer=%s id=%s method=%s",
                        peer,
                        req_id,
                        req_method,
                    )
                    break
                continue
            if resp is not None and not await transport.write_async(resp):
                disconnect_reason = "send_failed_after_response"
                send_failures += 1
                _log.warning(
                    "ws response send failed peer=%s id=%s method=%s",
                    peer,
                    req_id,
                    req_method,
                )
                break
    finally:
        reaped_sessions = 0
        detached_sessions = 0
        if transport is not None:
            transport.close()

            # Reap sessions this transport owned (close_on_disconnect sidecar
            # sessions) or detach the rest to the drop sentinel so later emits
            # don't crash into a closed socket or fall through to desktop stdout
            # logs. Detached sessions are handed to the grace-windowed WS-orphan
            # reaper inside _close_sessions_for_transport (a quick reconnect /
            # session.resume cancels it). This is the single WS-disconnect
            # teardown path.
            #
            # Offloaded: _close_session_by_id does a blocking worker.close()
            # (terminate + waits) plus a synchronous DB write — inline that
            # would freeze the uvicorn event loop for every other live
            # connection.
            try:
                reaped_sessions, detached_sessions = await asyncio.to_thread(
                    server._close_sessions_for_transport,
                    transport,
                    end_reason="ws_disconnect",
                )
            except Exception:
                _log.exception("ws transport teardown failed peer=%s", peer)
        try:
            await ws.close()
        except Exception as exc:
            _log.debug("ws close failed peer=%s error=%s", peer, exc)
        _log.info(
            "ws closed peer=%s reason=%s messages=%d parse_errors=%d "
            "dispatch_crashes=%d send_failures=%d reaped_sessions=%d detached_sessions=%d",
            peer,
            disconnect_reason,
            messages,
            parse_errors,
            dispatch_crashes,
            send_failures,
            reaped_sessions,
            detached_sessions,
        )
