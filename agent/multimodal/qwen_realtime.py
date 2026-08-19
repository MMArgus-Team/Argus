"""DashScope (Qwen) realtime speech clients — async, over ``websockets``.

Ported from streaming_demo/qwen_asr.py + qwen_tts_realtime.py, but rewritten on
the async ``websockets`` library (already a dependency) instead of the sync
``websocket-client`` the demo used — so both fit the multimodal backend's
daemon-thread asyncio loop with no extra package and no sync↔async bridge.

Endpoint (Beijing region):
    wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model=<model>
Auth: ``Authorization: Bearer <dashscope_api_key>``.

Both clients are NO-OPs when the api key is blank (the key is a per-account
DashScope key that must be configured in ~/.argus/config.yaml — it is not
bundled). Callers should check ``bool(api_key)`` before starting a session.

  * :class:`QwenRealtimeASR` — streaming user-speech recognition. Feed PCM16
    (16 kHz mono) via :meth:`append_audio`; server-side VAD segments speech and
    emits partial (``on_partial``) + final (``on_final``) text.
  * :class:`QwenRealtimeTTS` — streaming text→speech. :meth:`synthesize` sends
    text + commit and yields PCM16 (24 kHz mono) chunks as they arrive.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import unicodedata
from typing import Awaitable, Callable, Optional

log = logging.getLogger("hermes.multimodal.qwen_realtime")

_BASE_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"

# TTS input_text_buffer.append 单帧安全字节上限。DashScope WS 帧上限 256KB (262144),
# 超了服务端回 1009 (message too big) 直接关连接。留足 JSON 包裹余量 → 60KB (≈2万汉字,
# 单段口播绝够; 长文本会被切成多个 append)。
_TTS_APPEND_MAX_BYTES = 60000


def _chunk_text_by_bytes(text: str, max_bytes: int):
    """把 text 按 UTF-8 字节上限切成多片, **不切坏多字节字符**。短文本原样单片返回。"""
    if not text:
        return
    data = text.encode("utf-8")
    if len(data) <= max_bytes:
        yield text
        return
    buf = []
    size = 0
    for ch in text:
        b = len(ch.encode("utf-8"))
        if size + b > max_bytes and buf:
            yield "".join(buf)
            buf, size = [], 0
        buf.append(ch)
        size += b
    if buf:
        yield "".join(buf)


class QwenRealtimeASR:
    """Streaming ASR over the DashScope realtime WebSocket (server-VAD).

    Lifecycle (all async, run on the caller's loop):
        asr = QwenRealtimeASR(api_key, on_partial=..., on_final=...)
        await asr.connect()          # opens WS + configures session
        await asr.append_audio(pcm)  # feed 16k PCM16 chunks repeatedly
        ...                          # on_partial / on_final fire from _reader
        await asr.close()            # finish session + close WS
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "qwen3-asr-flash-realtime",
        language: str = "zh",
        sample_rate: int = 16000,
        # VAD defaults tuned for "listen like a human" — the demo's 0.7/500ms
        # cuts speech mid-sentence when the user pauses to think. 0.5/1200ms is
        # more tolerant (roughly 2× wait for silence, less trigger-happy on
        # noise). Override via realtime_asr_vad_* in config.
        vad_threshold: float = 0.5,
        vad_silence_ms: int = 1200,
        on_partial: Optional[Callable[[str], Awaitable[None]]] = None,
        on_final: Optional[Callable[[str], Awaitable[None]]] = None,
        on_speech_started: Optional[Callable[[], Awaitable[None]]] = None,
        on_speech_stopped: Optional[Callable[[], Awaitable[None]]] = None,
    ):
        self.api_key = (api_key or "").strip()
        self.model = model
        self.language = language
        self.sample_rate = sample_rate
        self.vad_threshold = vad_threshold
        self.vad_silence_ms = vad_silence_ms
        self.on_partial = on_partial
        self.on_final = on_final
        self.on_speech_started = on_speech_started
        self.on_speech_stopped = on_speech_stopped

        self._ws = None
        self._reader_task: Optional[asyncio.Task] = None
        self._connected = False
        # Set by _reader when the server acknowledges our session.update
        # (session.updated / session.created). connect() waits on this before
        # returning so the caller doesn't start pumping audio while the server
        # is still configuring VAD — otherwise the FIRST utterance's opening is
        # dropped and only the 2nd+ utterance transcribes.
        self._session_ready: asyncio.Event = asyncio.Event()
        # Graceful manual-turn shutdown is a protocol exchange, not merely a
        # socket close.  ``session.finish`` asks DashScope to flush the final
        # audio buffer; the reader then delivers any trailing transcription
        # completion(s) before ``session.finished``.  Keep explicit latches so
        # callers can wait for that ordering with a bounded timeout instead of
        # cancelling the reader and losing the last words.
        self._session_finished: asyncio.Event = asyncio.Event()
        self._terminal_event: asyncio.Event = asyncio.Event()
        self._completion_received: asyncio.Event = asyncio.Event()
        self._speech_observed: asyncio.Event = asyncio.Event()
        self._partial_since_completion = False
        self._close_lock: asyncio.Lock = asyncio.Lock()
        self._close_result: Optional[dict] = None
        self._delivered_finals: list[str] = []
        self._completed_event_ids: set[str] = set()
        self._canonical_transcript = ""
        self._upstream_error: Optional[str] = None
        # Linearizes a stop against an in-flight reconnect.  connect() clears
        # it before its first await; close() sets it synchronously, so a socket
        # that succeeds after Watcher ownership was popped is discarded before
        # it can publish a reader/callback stream.
        self._close_requested = False

    @property
    def is_connected(self) -> bool:
        """True 当且仅当上游 WS 活着 (reader 未把 _connected 置 False, 且 ws 还在)。
        watcher_engine.asr_audio 用它判"上游死没死"→ 死了触发重连自愈。"""
        return bool(self._connected and self._ws is not None)

    async def connect(self) -> bool:
        if not self.api_key:
            log.warning("[qwen-asr] no api_key; realtime ASR disabled")
            return False
        self._close_requested = False
        # ★ 重连安全 (同对象再次 connect): 先清理上一条死连接的残留 reader + 重置
        #   session_ready, 否则重连会泄漏旧 reader task / 卡在旧的 ready 事件上。
        if self._reader_task is not None:
            try:
                self._reader_task.cancel()
            except Exception:
                pass
            self._reader_task = None
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        self._session_ready.clear()
        self._session_finished.clear()
        self._terminal_event.clear()
        self._close_result = None
        self._upstream_error = None
        try:
            import websockets
        except Exception as exc:  # pragma: no cover
            log.warning("[qwen-asr] websockets unavailable: %s", exc)
            return False
        url = f"{_BASE_URL}?model={self.model}"
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "OpenAI-Beta": "realtime=v1"}
        try:
            # additional_headers (websockets>=13); fall back to extra_headers.
            try:
                candidate_ws = await websockets.connect(
                    url, additional_headers=headers)
            except TypeError:
                candidate_ws = await websockets.connect(
                    url, extra_headers=headers)
        except Exception as exc:
            log.warning("[qwen-asr] connect failed: %s", exc)
            return False
        if self._close_requested:
            try:
                await candidate_ws.close()
            except Exception:
                pass
            return False
        try:
            self._ws = candidate_ws
            self._connected = True
            # Start the reader FIRST so it can observe the server's session
            # acknowledgement, THEN send our config.
            self._reader_task = asyncio.create_task(self._reader())
            await self._send_session_update()
            # Wait until the server has processed our session.update (server VAD
            # is then armed).  A caller-side Watcher timeout may cancel connect
            # in this window; the cancellation branch below owns full cleanup.
            try:
                await asyncio.wait_for(
                    self._session_ready.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                log.warning(
                    "[qwen-asr] session.updated not received within 3s — "
                    "proceeding (first utterance may clip)")
        except asyncio.CancelledError:
            self._connected = False
            reader = self._reader_task
            if reader is not None and not reader.done():
                reader.cancel()
                try:
                    await reader
                except (asyncio.CancelledError, Exception):
                    pass
            try:
                await candidate_ws.close()
            except Exception:
                pass
            if self._ws is candidate_ws:
                self._ws = None
            if self._reader_task is reader:
                self._reader_task = None
            raise
        if (self._close_requested or self._ws is not candidate_ws
                or not self._connected):
            try:
                await candidate_ws.close()
            except Exception:
                pass
            if self._ws is candidate_ws:
                self._ws = None
            return False
        log.info("[qwen-asr] connected model=%s", self.model)
        return True

    async def _send_session_update(self) -> None:
        event = {
            "event_id": "session_update",
            "type": "session.update",
            "session": {
                "modalities": ["text"],
                "input_audio_format": "pcm",
                "sample_rate": self.sample_rate,
                "input_audio_transcription": {"language": self.language},
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": self.vad_threshold,
                    "silence_duration_ms": self.vad_silence_ms,
                },
            },
        }
        await self._ws.send(json.dumps(event))

    async def append_audio(self, pcm: bytes) -> bool:
        """Feed a chunk of 16 kHz mono PCM16 audio (raw bytes)."""
        if not self._connected or not self._ws or not pcm:
            return False
        try:
            event = {"type": "input_audio_buffer.append",
                     "audio": base64.b64encode(pcm).decode("ascii")}
            await self._ws.send(json.dumps(event))
            return True
        except Exception as exc:
            self._connected = False
            log.debug("[qwen-asr] append_audio failed: %s", exc)
            return False

    async def _reader(self) -> None:
        try:
            async for message in self._ws:
                try:
                    data = json.loads(message)
                except Exception:
                    continue
                et = data.get("type")
                if et in ("session.updated", "session.created"):
                    # Server processed our session.update (VAD armed). Release
                    # connect() so the caller may start streaming audio.
                    self._session_ready.set()
                elif et == "conversation.item.input_audio_transcription.text":
                    txt = (data.get("text") or "").strip()
                    if txt:
                        self._speech_observed.set()
                        self._partial_since_completion = True
                    if txt and self.on_partial:
                        await self.on_partial(txt)
                elif et == "conversation.item.input_audio_transcription.completed":
                    txt = (data.get("transcript") or "").strip()
                    completed_id = str(
                        data.get("event_id") or data.get("item_id") or ""
                    ).strip()
                    if completed_id and completed_id in self._completed_event_ids:
                        continue
                    if completed_id:
                        self._completed_event_ids.add(completed_id)
                        if len(self._completed_event_ids) > 256:
                            self._completed_event_ids.pop()
                    # Completed events are VAD item boundaries.  Preserve each
                    # non-empty event even when its text equals the prior item
                    # ("好" ... "好" is a valid two-segment turn).  An empty
                    # event does not complete a visible live partial.
                    if txt:
                        self._completion_received.set()
                        self._partial_since_completion = False
                    await self._deliver_final(txt)
                    self._canonical_transcript = self._join_final_segments(
                        self._delivered_finals)
                elif et == "input_audio_buffer.speech_started":
                    self._speech_observed.set()
                    if self.on_speech_started:
                        await self.on_speech_started()
                elif et == "input_audio_buffer.speech_stopped":
                    if self.on_speech_stopped:
                        await self.on_speech_stopped()
                elif et == "session.finished":
                    txt = (data.get("transcript") or "").strip()
                    # session.finished is the provider's authoritative full
                    # transcript and may rewrite words or punctuation anywhere
                    # in earlier completed items ("turn on" -> "turn off").
                    # Never append that full value as another VAD callback.
                    # The one ambiguous legacy shape is an exact echo of only
                    # the last completed item; preserve the already-joined
                    # transcript in that case.
                    joined_before = self._join_final_segments(
                        self._delivered_finals)
                    last_before = (
                        self._delivered_finals[-1]
                        if self._delivered_finals else ""
                    )
                    if txt and txt != last_before:
                        self._completion_received.set()
                        self._partial_since_completion = False
                        # Clear the gateway-owned live partial without creating
                        # a completed-segment callback.  Manual finish commits
                        # the exact canonical transcript from close_result.
                        if self.on_partial:
                            await self.on_partial("")
                        self._canonical_transcript = txt
                    elif txt and txt == last_before:
                        self._canonical_transcript = joined_before
                    else:
                        self._canonical_transcript = joined_before
                    self._session_finished.set()
                    self._terminal_event.set()
                elif et == "error":
                    raw_error = data.get("error") or {}
                    if isinstance(raw_error, dict):
                        raw_message = raw_error.get("message") or "unknown"
                    else:
                        raw_message = raw_error
                    # This string crosses the gateway response boundary.  Keep
                    # it single-line and bounded rather than reflecting an
                    # arbitrary upstream payload into desktop UI/logs.
                    msg = " ".join(str(raw_message).split())[:300] or "unknown"
                    self._upstream_error = msg
                    log.warning("[qwen-asr] server error: %s", msg)
                    # An upstream error is terminal for this ASR stream.  Wake
                    # a manual finish immediately instead of making the click
                    # appear hung for the whole finish timeout.
                    self._terminal_event.set()
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.debug("[qwen-asr] reader ended: %s", exc)
        finally:
            # ★ C2: WS closed or errored out on its own — the connection is dead.
            # Clear the flag so append_audio() stops sending to a dead socket
            # (the caller can then observe disconnection / trigger a reconnect).
            self._connected = False

    @staticmethod
    def _join_final_segments(segments: list[str]) -> str:
        """Join ASR VAD segments without inserting spaces into CJK text."""
        joined = ""
        for raw in segments:
            text = (raw or "").strip()
            if not text:
                continue
            if (joined and joined[-1].isascii() and joined[-1].isalnum()
                    and text[0].isascii() and text[0].isalnum()):
                joined += " "
            joined += text
        return joined.strip()

    @staticmethod
    def _transcript_comparison_key(text: str) -> str:
        """Normalize provider-only formatting for terminal/full comparison."""
        return "".join(
            ch.casefold()
            for ch in str(text or "")
            if not ch.isspace()
            and not unicodedata.category(ch).startswith("P")
        )

    async def _deliver_final(
        self, text: str, *, full_transcript: bool = False,
    ) -> bool:
        """Deliver one novel final segment to the owner callback.

        ``session.finished`` may echo either the last completed item or the
        whole transcript.  Suppress both shapes so a manual turn cannot acquire
        duplicate words while still accepting a genuinely trailing segment.
        """
        text = (text or "").strip()
        if not text:
            return False
        if not full_transcript:
            self._delivered_finals.append(text)
            if self.on_final:
                await self.on_final(text)
            return True
        joined = self._join_final_segments(self._delivered_finals)
        joined_key = self._transcript_comparison_key(joined)
        text_key = self._transcript_comparison_key(text)
        if joined_key and text_key.startswith(joined_key):
            # session.finished may refine punctuation/spacing anywhere in the
            # full transcript (e.g. "你好世界" -> "你好，世界").  It is the
            # authoritative canonical form, not a new trailing VAD segment.
            if text_key == joined_key:
                return False
            # It can also add a final, not-yet-completed tail while inserting
            # punctuation into the prefix.  Slice after the normalized prefix
            # rather than appending the entire full transcript as a new item.
            consumed = 0
            cut = 0
            for index, ch in enumerate(text):
                if (not ch.isspace()
                        and not unicodedata.category(ch).startswith("P")):
                    consumed += len(ch.casefold())
                if consumed >= len(joined_key):
                    cut = index + 1
                    break
            text = text[cut:].strip()
            if not text:
                return False
        if text == joined or (self._delivered_finals
                              and text == self._delivered_finals[-1]):
            return False
        if joined and joined.startswith(text):
            # A repeated prefix/older cumulative snapshot contributes nothing.
            return False
        if joined and text.startswith(joined):
            # session.finished normally carries the full transcript.  Deliver
            # only its novel suffix; appending the full value would turn
            # completed='\u4f60\u597d', finished='\u4f60\u597d\u4e16\u754c' into
            # '\u4f60\u597d\u4f60\u597d\u4e16\u754c'.  Some completed events are also
            # cumulative, so this reduction is safe for either source.
            text = text[len(joined):].strip()
            if not text:
                return False
        self._delivered_finals.append(text)
        if self.on_final:
            await self.on_final(text)
        return True

    async def close(
        self, *, finish_timeout: float = 5.0, graceful: bool = True,
    ) -> dict:
        """Gracefully finish ASR, then close the WebSocket.

        The old implementation sent ``session.finish`` and immediately closed
        the socket/cancelled the reader.  That races the final transcription and
        is especially visible for push-to-talk: the last (or only) utterance is
        silently lost.  Wait for ``session.finished`` with a strict bound; the
        reader processes preceding ``...transcription.completed`` callbacks
        before setting that latch.  The result is idempotently cached for
        concurrent/repeated owners.
        """
        # Set before awaiting the lock: even a connect() currently suspended in
        # websockets.connect observes cancellation as soon as it gets a socket.
        self._close_requested = True
        # Also release connect() if it is waiting for session.updated after the
        # socket was published; it re-checks _close_requested before success.
        self._session_ready.set()
        async with self._close_lock:
            if self._close_result is not None:
                return dict(self._close_result)

            ws = self._ws
            finish_sent = False
            timed_out = False
            if graceful and ws is not None and not self._upstream_error:
                try:
                    await ws.send(json.dumps({"type": "session.finish"}))
                    finish_sent = True
                except Exception as exc:
                    log.debug("[qwen-asr] session.finish failed: %s", exc)

            if finish_sent:
                try:
                    await asyncio.wait_for(
                        self._terminal_event.wait(),
                        timeout=max(0.05, float(finish_timeout)),
                    )
                except asyncio.TimeoutError:
                    timed_out = True
                    log.warning(
                        "[qwen-asr] session.finished not received within %.1fs",
                        finish_timeout,
                    )
                if (self._session_finished.is_set()
                        and (self._partial_since_completion
                             or (self._speech_observed.is_set()
                                 and not self._completion_received.is_set()))):
                    # session.finished is terminal.  If speech/partials were
                    # observed but no completed transcript preceded it, the
                    # protocol flush was incomplete even though the terminal
                    # frame itself arrived; surface that truth so the gateway
                    # can retain its live partial as a best-effort final turn.
                    timed_out = True
                    log.warning(
                        "[qwen-asr] session.finished arrived without a "
                        "transcription.completed event",
                    )

            # Stop accepting new PCM only after the finish frame was sent.  The
            # Watcher removes this object from its key map before awaiting us,
            # so no new caller audio can enter during the grace period.
            self._connected = False
            reader = self._reader_task
            if not graceful and reader is not None and not reader.done():
                # Cancellation/session teardown is abortive by design: stop the
                # callback reader before closing the transport, so a queued
                # trailing completion cannot materialize a ghost user turn.
                reader.cancel()
                try:
                    await reader
                except (asyncio.CancelledError, Exception):
                    pass
            if ws is not None:
                try:
                    await ws.close()
                except Exception:
                    pass
            if reader is not None and reader is not asyncio.current_task():
                if not reader.done():
                    reader.cancel()
                try:
                    await reader
                except (asyncio.CancelledError, Exception):
                    pass
            self._ws = None
            self._reader_task = None
            self._close_result = {
                "ok": bool(
                    not self._upstream_error
                    and (not graceful or self._session_finished.is_set()
                         or ws is None)
                ),
                "finish_sent": finish_sent,
                "completed": self._completion_received.is_set(),
                "session_finished": self._session_finished.is_set(),
                "timed_out": timed_out,
                "aborted": not graceful,
            }
            if self._canonical_transcript:
                self._close_result["transcript"] = self._canonical_transcript
            if self._upstream_error:
                self._close_result.update({
                    "reason": "upstream_error",
                    "error": self._upstream_error,
                })
            return dict(self._close_result)


class QwenRealtimeTTS:
    """Streaming TTS over the DashScope realtime WebSocket (client_commit).

    Usage — yields PCM16 (24 kHz mono) chunks:
        tts = QwenRealtimeTTS(api_key, voice="Cherry")
        async for pcm, sr in tts.synthesize("你好"):
            ...
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "qwen3-tts-flash-realtime",
        voice: str = "Cherry",
        sample_rate: int = 24000,
        language_type: str = "Auto",
        speech_rate: float = 1.3,
    ):
        self.api_key = (api_key or "").strip()
        self.model = model
        self.voice = voice
        self.sample_rate = sample_rate
        self.language_type = language_type
        # Playback speed. This deployment honors `speech_rate` (NOT `rate` /
        # `speed`, probed 2026-07-02): 1.0 = default (sounds slow/robotic),
        # 1.3 ≈ natural conversational pace, 2.0 = fast. Tunable via config
        # realtime_tts_speech_rate.
        self.speech_rate = float(speech_rate)

    async def synthesize(self, text: str):
        """Async generator yielding ``(pcm_bytes, sample_rate)`` chunks."""
        text = (text or "").strip()
        if not text or not self.api_key:
            return
        try:
            import websockets
        except Exception as exc:  # pragma: no cover
            log.warning("[qwen-tts] websockets unavailable: %s", exc)
            return
        url = f"{_BASE_URL}?model={self.model}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        ws = None
        try:
            try:
                ws = await websockets.connect(url, additional_headers=headers)
            except TypeError:
                ws = await websockets.connect(url, extra_headers=headers)
        except Exception as exc:
            log.warning("[qwen-tts] connect failed: %s", exc)
            return
        try:
            await ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "mode": "client_commit",
                    "voice": self.voice,
                    "language_type": self.language_type,
                    "response_format": "pcm",
                    "sample_rate": self.sample_rate,
                    "speech_rate": self.speech_rate,
                },
            }))
            # ★ 分片 append 防 1009 (frame too big, 上限 256KB): text 过长时一次性 append
            #   的出站帧可能超限被服务端 1009 关连接。按 UTF-8 字节安全上限分多次 append
            #   (不切坏多字节字符), 最后 commit。短文本仍是一次 append (行为不变)。
            for _piece in _chunk_text_by_bytes(text, _TTS_APPEND_MAX_BYTES):
                await ws.send(json.dumps({"type": "input_text_buffer.append",
                                          "text": _piece}))
            await ws.send(json.dumps({"type": "input_text_buffer.commit"}))

            async for message in ws:
                try:
                    event = json.loads(message)
                except Exception:
                    continue
                et = event.get("type")
                if et == "response.audio.delta":
                    b64 = event.get("delta", "")
                    if b64:
                        try:
                            yield base64.b64decode(b64), self.sample_rate
                        except Exception:
                            pass
                elif et == "response.done":
                    break
                elif et == "error":
                    msg = (event.get("error") or {}).get("message", "unknown")
                    log.warning("[qwen-tts] server error: %s", msg)
                    break
        except Exception as exc:
            log.warning("[qwen-tts] stream error: %s", exc)
        finally:
            try:
                await ws.send(json.dumps({"type": "session.finish"}))
            except Exception:
                pass
            try:
                await ws.close()
            except Exception:
                pass
