# -*- coding: utf-8 -*-
"""AUTO-SPLIT from the monolithic engine — see agent/multimodal/__init__.py.

One slice of the ported TML Always-On Video Agent engine. The public surface is
re-exported from agent.multimodal.core for backward compatibility.
"""
from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import logging
import os
import re
import sqlite3
import tempfile
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from io import BytesIO
from typing import (
    Any, AsyncIterator, Awaitable, Callable, Deque, Dict, List,
    Optional, Set, Tuple,
)

try:
    import aiohttp  # optional: only needed by the local HTTP speech/Gemini backends
except ImportError:  # pragma: no cover
    aiohttp = None  # type: ignore
import httpx


def _require_aiohttp():
    """Guard for the optional aiohttp dependency.

    The HTTP speech backends (STT/WhisperX/TTS) build ``aiohttp.ClientTimeout``
    and ``aiohttp.ClientSession`` directly. When aiohttp isn't installed the
    module still imports (it's optional), so raise a clear error at call time
    instead of a bare ``AttributeError: 'NoneType' has no attribute ...``.
    """
    if aiohttp is None:  # pragma: no cover - import-guarded path
        raise RuntimeError(
            "aiohttp is required for the HTTP speech backends "
            "(ASR/WhisperX/TTS) but is not installed. Install it with "
            "`pip install aiohttp`."
        )
    return aiohttp
from openai import AsyncOpenAI

log = logging.getLogger("hermes.multimodal")

from ._config import Config
from ._memory import (
    Frame, FrameBuffer, FrameStore, ConversationLog, MacroEvent, MemoryStore,
    frame_to_image_content, fmt_ts, new_response_id,
)
from ._workers import MemoryLLMClient, OpenAIMemoryClient


# =========================================================================== #
# Voice IO (照搬, 不变)
# =========================================================================== #
_HARD_TERMINATORS = "。！？!?；;"
_SOFT_TERMINATORS = "，,、:：—…"


class SentenceAccumulator:
    def __init__(self, min_chars: int = 6, max_chars: int = 120):
        self.min_chars = max(1, min_chars)
        self.max_chars = max(self.min_chars, max_chars)
        self._buf: str = ""

    def feed(self, token: str) -> List[str]:
        if not token: return []
        out: List[str] = []
        for ch in token:
            self._buf += ch
            if ch in _HARD_TERMINATORS:
                s = self._buf.strip()
                if s: out.append(s)
                self._buf = ""; continue
            if ch in _SOFT_TERMINATORS and len(self._buf.strip()) >= self.min_chars:
                s = self._buf.strip()
                if s: out.append(s)
                self._buf = ""; continue
            if len(self._buf) >= self.max_chars:
                s = self._buf.strip()
                if s: out.append(s)
                self._buf = ""
        return out

    def flush(self) -> List[str]:
        s = self._buf.strip(); self._buf = ""
        return [s] if s else []


def _ext_from_mime(mime: str) -> str:
    m = (mime or "").lower()
    if "webm" in m: return "webm"
    if "mp4" in m or "m4a" in m or "aac" in m: return "m4a"
    if "ogg" in m: return "ogg"
    if "wav" in m: return "wav"
    if "mpeg" in m or "mp3" in m: return "mp3"
    return "bin"


def _parse_asr_text(js: dict) -> str:
    """Pull the transcript out of an ASR JSON response across API shapes.

    Handles the internal server's ``{"text": ...}``, OpenAI-compatible
    ``/v1/audio/transcriptions`` (also ``{"text": ...}``), and a couple of
    common variants (``result``/``transcript``, or a ``segments`` list to
    concatenate) so third-party clouds Just Work without per-vendor code.
    """
    if not isinstance(js, dict):
        return ""
    for k in ("text", "result", "transcript"):
        v = js.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # OpenAI-compatible chat/completions shape, used by DashScope Qwen ASR.
    choices = js.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, list):
                parts = []
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())
                if parts:
                    return " ".join(parts).strip()
    output = js.get("output")
    if isinstance(output, dict):
        nested = _parse_asr_text(output)
        if nested:
            return nested
        for k in ("text", "sentence", "transcription"):
            v = output.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    segs = js.get("segments")
    if isinstance(segs, list):
        # ★ C17: text 为 None 时 str(None)=='None' 会污染字幕; 用 `or ""` 归一并过滤空段。
        parts = [str(s.get("text") or "").strip()
                 for s in segs if isinstance(s, dict)]
        joined = " ".join(p for p in parts if p)
        if joined.strip():
            return joined.strip()
    return ""


def _dashscope_audio_format(mime: str, filename: str) -> str:
    m = (mime or "").lower()
    fn = (filename or "").lower()
    if "webm" in m or fn.endswith(".webm"):
        return "webm"
    if "wav" in m or fn.endswith(".wav"):
        return "wav"
    if "mpeg" in m or "mp3" in m or fn.endswith(".mp3"):
        return "mp3"
    if "ogg" in m or fn.endswith(".ogg"):
        return "ogg"
    if "mp4" in m or "m4a" in m or "aac" in m or fn.endswith((".m4a", ".mp4", ".aac")):
        return "m4a"
    return _ext_from_mime(mime)


def _is_dashscope_chat_asr(endpoint: str, model: str) -> bool:
    ep = (endpoint or "").lower()
    md = (model or "").lower()
    if "dashscope" in ep or "aliyuncs.com" in ep:
        return True
    return "qwen" in md and "asr" in md and (
        "/chat/completions" in ep or ep.rstrip("/").endswith("/v1")
    )


def _dashscope_chat_endpoint(endpoint: str) -> str:
    ep = (endpoint or "").strip().rstrip("/")
    if not ep:
        return "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    if ep.endswith("/chat/completions"):
        return ep
    if ep.endswith("/v1"):
        return ep + "/chat/completions"
    return ep


class STTClient:
    """Single-shot ASR (OpenAI-compatible or the internal server).

    Defaults to the user-speech ``asr_*`` config triple. The env-audio (qwen)
    path constructs one with ``endpoint``/``api_key``/``model`` overrides so it
    can point at the ``env_*`` service while sharing all the request logic.
    """

    def __init__(self, cfg: Config, *, endpoint: Optional[str] = None,
                 api_key: Optional[str] = None, model: Optional[str] = None):
        self.cfg = cfg
        self._endpoint = (endpoint or "").strip() or cfg.asr_url
        self._api_key = (api_key if api_key is not None
                         else getattr(cfg, "asr_api_key", "") or "").strip()
        if not self._api_key and _is_dashscope_chat_asr(self._endpoint, model or ""):
            self._api_key = (getattr(cfg, "dashscope_api_key", "") or "").strip()
        self._model = (model if model is not None
                       else getattr(cfg, "asr_model", "") or "").strip()
        # Structured diagnostics for the caller that owns this single-shot
        # request.  Environment ASR reads it immediately after ``await`` and
        # mirrors it into the worker trajectory; raw audio and credentials are
        # deliberately never included.
        self.last_diagnostics: Dict[str, Any] = {}

    async def transcribe(self, audio_bytes: bytes, *, mime: str = "audio/webm",
                         filename: Optional[str] = None) -> str:
        if not audio_bytes:
            self.last_diagnostics = {
                "ok": False,
                "reason": "empty_input",
                "status": None,
                "model": self._model,
            }
            return ""
        if filename is None:
            filename = f"clip.{_ext_from_mime(mime)}"
        if _is_dashscope_chat_asr(self._endpoint, self._model):
            return await self._transcribe_dashscope_chat(
                audio_bytes, mime=mime, filename=filename)
        params = {}
        if not self.cfg.asr_run_vllm:
            params["run_vllm"] = "false"
        # Cloud OpenAI-compatible ASR: Bearer auth + a `model` form field. When
        # api_key/model are empty we fall back to the internal server's plain
        # multipart contract (no auth, no model).
        headers = {}
        api_key = self._api_key
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        model = self._model
        _require_aiohttp()
        timeout = aiohttp.ClientTimeout(total=self.cfg.asr_timeout)
        t0 = time.time()
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                data = aiohttp.FormData()
                data.add_field("file", audio_bytes, filename=filename, content_type=mime)
                if model:
                    data.add_field("model", model)
                async with session.post(self._endpoint, data=data,
                                        params=params, headers=headers) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        self.last_diagnostics = {
                            "ok": False,
                            "reason": "http_error",
                            "status": resp.status,
                            "model": model,
                            "latency_sec": round(time.time() - t0, 4),
                            "error": body[:300],
                        }
                        log.warning("[asr] status=%d body=%s", resp.status, body[:200])
                        return ""
                    text = _parse_asr_text(await self._read_json_or_text(resp))
                    self.last_diagnostics = {
                        "ok": True,
                        "reason": "completed" if text else "empty_transcript",
                        "status": resp.status,
                        "model": model,
                        "latency_sec": round(time.time() - t0, 4),
                        "raw_transcript": text,
                    }
                    log.info("[asr] %.2fs %d bytes → %r",
                             time.time() - t0, len(audio_bytes), text[:120])
                    return text
        except Exception as e:
            self.last_diagnostics = {
                "ok": False,
                "reason": "request_exception",
                "status": None,
                "model": self._model,
                "latency_sec": round(time.time() - t0, 4),
                "error_type": type(e).__name__,
                "error": str(e),
            }
            log.warning("[asr] %.2fs %s: %s endpoint=%s bytes=%d mime=%s",
                        time.time() - t0, type(e).__name__, e,
                        self._endpoint, len(audio_bytes), mime)
            return ""

    async def _transcribe_dashscope_chat(
        self, audio_bytes: bytes, *, mime: str, filename: str
    ) -> str:
        api_key = self._api_key
        if not api_key:
            self.last_diagnostics = {
                "ok": False,
                "reason": "missing_api_key",
                "status": None,
                "model": self._model,
            }
            log.warning("[asr:dashscope] missing api_key endpoint=%s model=%s",
                        self._endpoint, self._model)
            return ""
        model = self._model or "qwen3-asr-flash"
        endpoint = _dashscope_chat_endpoint(self._endpoint)
        fmt = _dashscope_audio_format(mime, filename)
        mediatype = (mime or "").split(";", 1)[0].strip() or f"audio/{fmt}"
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
        data_uri = f"data:{mediatype};base64,{audio_b64}"
        payload = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": data_uri,
                        },
                    },
                ],
            }],
            "asr_options": {
                "language": "zh",
                "enable_itn": True,
            },
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        _require_aiohttp()
        timeout = aiohttp.ClientTimeout(total=self.cfg.asr_timeout)
        t0 = time.time()
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(endpoint, json=payload, headers=headers) as resp:
                    js = await self._read_json_or_text(resp)
                    request_id = ""
                    finish_reason = ""
                    usage = None
                    if isinstance(js, dict):
                        request_id = str(js.get("request_id") or js.get("id") or "")
                        usage = js.get("usage")
                        choices = js.get("choices")
                        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                            finish_reason = str(choices[0].get("finish_reason") or "")
                    request_id = request_id or str(
                        resp.headers.get("x-request-id")
                        or resp.headers.get("x-dashscope-request-id")
                        or ""
                    )
                    latency_sec = round(time.time() - t0, 4)
                    if resp.status != 200:
                        self.last_diagnostics = {
                            "ok": False,
                            "reason": "http_error",
                            "status": resp.status,
                            "model": model,
                            "latency_sec": latency_sec,
                            "provider_request_id": request_id,
                            "error": str(js)[:500],
                        }
                        log.warning(
                            "[asr:dashscope] status=%d request_id=%s body=%s endpoint=%s model=%s bytes=%d fmt=%s",
                            resp.status, request_id or "-", str(js)[:300], endpoint, model,
                            len(audio_bytes), fmt)
                        return ""
                    text = _parse_asr_text(js)
                    self.last_diagnostics = {
                        "ok": True,
                        "reason": "completed" if text else "empty_transcript",
                        "status": resp.status,
                        "model": model,
                        "latency_sec": latency_sec,
                        "provider_request_id": request_id,
                        "finish_reason": finish_reason,
                        "usage": usage,
                        "raw_transcript": text,
                    }
                    log.info(
                        "[asr:dashscope] %.2fs request_id=%s finish=%s %d bytes "
                        "fmt=%s → %r",
                        latency_sec, request_id or "-", finish_reason or "-",
                        len(audio_bytes), fmt, text[:120])
                    return text
        except Exception as e:
            latency_sec = round(time.time() - t0, 4)
            self.last_diagnostics = {
                "ok": False,
                "reason": "request_exception",
                "status": None,
                "model": model,
                "latency_sec": latency_sec,
                "error_type": type(e).__name__,
                "error": str(e),
            }
            log.warning(
                "[asr:dashscope] %.2fs %s: %s endpoint=%s model=%s bytes=%d fmt=%s",
                latency_sec, type(e).__name__, e, endpoint, model,
                len(audio_bytes), fmt)
            return ""

    @staticmethod
    async def _read_json_or_text(resp):
        """Return parsed JSON, or {'text': <body>} for text/plain responses
        (some ASR endpoints return the raw transcript with response_format=text)."""
        try:
            return await resp.json(content_type=None)
        except Exception:
            body = (await resp.text()).strip()
            return {"text": body}


@dataclass
class WhisperXSegment:
    """One WhisperX transcription segment (matches serve_whisperx.py's wire
    schema)."""
    text: str
    start: float = 0.0      # 段起始时间 (相对音频内偏移, 秒)
    end: float = 0.0
    speaker: Optional[str] = None   # 例 "SPEAKER_00"; None 表示 diarize 未给标签


class WhisperXClient:
    """WhisperX HTTP client. Transcribes video-embedded environment audio and
    returns a list of speaker-tagged segments.

    Relationship to STTClient:
       - STTClient handles "user speaking to the agent": single speaker, must be
         fast (<1s), plain text only.
       - WhisperXClient handles "people speaking in the video": multi-speaker,
         needs word-level timestamps + speaker labels, may be slower (~RTF 0.3).
       Both coexist; the env-audio backend is selected by cfg.env_asr_backend.

    Wire contract (matches serve_whisperx.py /asr):
       POST <env_url> multipart file
       → {"text", "language", "duration",
          "segments": [{start, end, text, speaker}], ...}
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        # Env-audio endpoint/key; both empty → fall back to the user-speech
        # asr_* service so env + user share one ASR when not configured apart.
        self._endpoint = (getattr(cfg, "env_url", "") or "").strip() or cfg.asr_url
        self._api_key = ((getattr(cfg, "env_api_key", "") or "").strip()
                         or (getattr(cfg, "asr_api_key", "") or "").strip())

    async def transcribe(
        self, audio_bytes: bytes, *,
        mime: str = "audio/webm",
        filename: Optional[str] = None,
        language: Optional[str] = None,
        diarize: Optional[bool] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> List[WhisperXSegment]:
        """Return the segment list (chronological). On any failure → []."""
        if not audio_bytes:
            return []
        if filename is None:
            filename = f"clip.{_ext_from_mime(mime)}"
        do_diarize = (diarize if diarize is not None
                      else self.cfg.whisperx_diarize)
        lang = (language or self.cfg.whisperx_language or "zh")
        min_spk = min_speakers if min_speakers is not None else self.cfg.whisperx_min_speakers
        max_spk = max_speakers if max_speakers is not None else self.cfg.whisperx_max_speakers

        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        _require_aiohttp()
        timeout = aiohttp.ClientTimeout(total=self.cfg.whisperx_timeout)
        t0 = time.time()
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                data = aiohttp.FormData()
                data.add_field("file", audio_bytes,
                               filename=filename, content_type=mime)
                data.add_field("language", lang)
                data.add_field("diarize", "true" if do_diarize else "false")
                data.add_field("align", "true")
                if min_spk and min_spk > 0:
                    data.add_field("min_speakers", str(min_spk))
                if max_spk and max_spk > 0:
                    data.add_field("max_speakers", str(max_spk))
                async with session.post(self._endpoint, data=data,
                                        headers=headers) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        log.warning("[whisperx] status=%d body=%s",
                                    resp.status, body[:200])
                        return []
                    js = await resp.json()
        except Exception as e:
            log.warning("[whisperx] %.2fs http %s", time.time() - t0, e)
            return []

        segs_raw = js.get("segments") or []
        out: List[WhisperXSegment] = []
        for s in segs_raw:
            text = str(s.get("text", "") or "").strip()
            if not text:
                continue
            try:
                start = float(s.get("start", 0.0) or 0.0)
                end = float(s.get("end", 0.0) or 0.0)
            except (TypeError, ValueError):
                start = 0.0; end = 0.0
            speaker = s.get("speaker")
            out.append(WhisperXSegment(
                text=text, start=start, end=end,
                speaker=str(speaker) if speaker else None,
            ))
        timing = js.get("timing") or {}
        log.info(
            "[whisperx] %.2fs %d bytes → %d segs, %d speakers, "
            "audio=%.1fs RTF=%.3f (asr=%.2f align=%.2f diar=%.2f)",
            time.time() - t0, len(audio_bytes), len(out),
            len(js.get("speakers") or []),
            float(js.get("duration", 0.0) or 0.0),
            float(timing.get("rtf", 0.0) or 0.0),
            float(timing.get("asr", 0.0) or 0.0),
            float(timing.get("align", 0.0) or 0.0),
            float(timing.get("diarize", 0.0) or 0.0),
        )
        return out


class TTSClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    async def health(self) -> Optional[Dict[str, Any]]:
        url = self.cfg.tts_url.rstrip("/") + "/v1/tts/health"
        _require_aiohttp()
        timeout = aiohttp.ClientTimeout(total=5.0)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as resp:
                    if resp.status != 200: return None
                    return await resp.json()
        except Exception as e:
            log.warning("[tts health] %s", e); return None

    async def stream(self, text: str, *,
                     cancel_check: Optional[Callable[[], bool]] = None,
                     ) -> AsyncIterator[Tuple[bytes, int]]:
        url = self.cfg.tts_url.rstrip("/") + "/v1/tts/stream"
        payload = {"text": text, "language": self.cfg.tts_language,
                   "speaker": self.cfg.tts_voice, "instruct": self.cfg.tts_instruct}
        _require_aiohttp()
        timeout = aiohttp.ClientTimeout(
            sock_connect=self.cfg.tts_connect_timeout, total=None,
            sock_read=self.cfg.tts_read_timeout)
        t0 = time.time(); first_chunk_ts = None; n_chunks = 0
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        log.warning("[tts] status=%d body=%s", resp.status, body[:200])
                        return
                    buf = b""
                    async for raw in resp.content.iter_chunked(8192):
                        if cancel_check is not None and cancel_check():
                            log.info("[tts] 取消 %d chunk %.2fs",
                                     n_chunks, time.time() - t0)
                            return
                        buf += raw
                        while len(buf) >= 8:
                            sr = int.from_bytes(buf[0:4], "big", signed=False)
                            pcm_len = int.from_bytes(buf[4:8], "big", signed=False)
                            if len(buf) < 8 + pcm_len: break
                            pcm_data = buf[8:8 + pcm_len]
                            buf = buf[8 + pcm_len:]
                            n_chunks += 1
                            if first_chunk_ts is None:
                                first_chunk_ts = time.time()
                                log.info("[tts] first chunk %.2fs sr=%d pcm=%d",
                                         first_chunk_ts - t0, sr, pcm_len)
                            yield pcm_data, sr
            log.info("[tts] done %d chunks %.2fs (%d字)",
                     n_chunks, time.time() - t0, len(text))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning("[tts] %.2fs %s", time.time() - t0, e)


TTSChunkSink = Callable[[Dict[str, Any]], Awaitable[None]]


# =========================================================================== #
# 环境音
# =========================================================================== #
async def append_audio_observation(conversation: ConversationLog, text: str, *,
                                   rel_ts: Optional[float] = None,
                                   speaker: Optional[str] = None) -> None:
    text = (text or "").strip()
    if not text: return
    await conversation.append(role="system", content=text,
                              kind="audio_observation", rel_ts=rel_ts,
                              speaker=speaker)


# =========================================================================== #
# Speech factory (ASR / TTS 可插拔接缝)
# =========================================================================== #
class SpeechFactory:
    """Abstraction over the source of ASR (user speech + environment audio) and
    TTS.

    DualAgent obtains three speech components through it:
      * make_stt()     → STT client (user speaking to the agent). transcribe(bytes)->str.
      * make_env_asr() → environment-audio ASR (with speaker diarization).
                         ->List[WhisperXSegment].
      * make_tts()     → TTS client. stream(text)->AsyncIterator[(pcm,sr)] + health().

    Under Hermes the implementations live in agent.multimodal.speech
    (HermesSpeechFactory bridges Hermes' built-in providers; CloudSpeechFactory
    goes straight to the cloud). The default LocalSpeechFactory uses local HTTP.
    """

    def make_stt(self) -> Any:
        raise NotImplementedError

    def make_env_asr(self) -> Any:
        raise NotImplementedError

    def make_tts(self) -> Any:
        raise NotImplementedError


class LocalSpeechFactory(SpeechFactory):
    """Default local-HTTP implementation (reuses the STT/WhisperX/TTS clients
    defined in this module)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def make_stt(self) -> Any:
        return STTClient(self.cfg)

    def make_env_asr(self) -> Any:
        return WhisperXClient(self.cfg)

    def make_tts(self) -> Any:
        return TTSClient(self.cfg)


# =========================================================================== #
# LLM client factory (Hermes 融合接缝)
# =========================================================================== #
class LLMClientFactory:
    """Abstraction over the source of the LLM clients + model names workers need.

    Instead of hardcoding AsyncOpenAI(base_url=cfg.base_url), DualAgent asks the
    factory for (client, model). Hermes' implementation
    (agent.multimodal.hermes_glue.HermesClientFactory) pulls the client from
    Hermes' provider/model resolution, sharing the `argus model` selection.

    Each method returns (AsyncOpenAI-compatible client, model_slug); an empty
    model_slug means "keep the value from cfg".
    """

    def main_client(self) -> Tuple[Any, str]:
        raise NotImplementedError

    def memory_client(self, shared_main: Any) -> "MemoryLLMClient":
        return OpenAIMemoryClient(shared_main, model="")
