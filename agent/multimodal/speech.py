"""Pluggable ASR / TTS backends for the multimodal subsystem.

The multimodal agent needs three speech capabilities, each defined by a small
duck-typed contract (so a cloud implementation is a drop-in with no base-class
import required):

  1. STT (user speech → text)  — single speaker, low latency.
       async transcribe(audio_bytes: bytes, *, mime: str = "audio/webm",
                         filename: str | None = None) -> str

  2. Environment ASR (video-embedded speech → segments) — multi-speaker.
       async transcribe(audio_bytes: bytes, *, mime=..., filename=...,
                         language=..., diarize=..., min_speakers=..., max_speakers=...
                         ) -> list[WhisperXSegment]
       (WhisperXSegment has .text/.start/.end/.speaker; see core.py)

  3. TTS (text → streamed PCM).
       async health() -> dict | None
       async stream(text: str, *, cancel_check: Callable[[], bool] | None = None)
             -> AsyncIterator[tuple[bytes, int]]   # (pcm_bytes, sample_rate)

`DualAgent` obtains these via a :class:`SpeechFactory`. The default
:class:`LocalSpeechFactory` (in core.py) returns the bundled local-HTTP
placeholders. To wire cloud APIs, implement the classes below and have
:func:`build_cloud_speech_factory` return a factory that yields them.

============================================================================
 >>> CLOUD API INTEGRATION POINT (implement these three classes) <<<
============================================================================
The user owns the cloud API wiring. Fill in the three ``# TODO(cloud)`` bodies
with real cloud calls (Hermes already exposes credential helpers; this module
deliberately does NOT hardcode any endpoint or key). Then set the multimodal
speech provider so :func:`build_cloud_speech_factory` activates this factory.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import tempfile
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple

from .core import Config, SpeechFactory, WhisperXSegment

log = logging.getLogger("hermes.multimodal.speech")


# --------------------------------------------------------------------------- #
# Cloud STT (user speech → text)
# --------------------------------------------------------------------------- #
class CloudSTTClient:
    """Cloud ASR for user-to-agent speech (single speaker, low latency)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    async def transcribe(self, audio_bytes: bytes, *, mime: str = "audio/webm",
                         filename: Optional[str] = None) -> str:
        # TODO(cloud): POST audio_bytes to your cloud ASR API and return the
        # transcript string. Return "" on empty/failure.
        raise NotImplementedError(
            "CloudSTTClient.transcribe: implement cloud ASR here")


# --------------------------------------------------------------------------- #
# Cloud environment ASR (video-embedded speech → segments, multi-speaker)
# --------------------------------------------------------------------------- #
class CloudEnvASRClient:
    """Cloud ASR for environment audio (video-embedded speech, multi-speaker).

    Must return a list of :class:`WhisperXSegment` (``text``/``start``/``end``/
    ``speaker``). If your backend has no diarization, return a single segment with
    ``speaker=None``.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    async def transcribe(
        self, audio_bytes: bytes, *, mime: str = "audio/webm",
        filename: Optional[str] = None, language: Optional[str] = None,
        diarize: Optional[bool] = None, min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> List[WhisperXSegment]:
        # TODO(cloud): POST audio_bytes to your cloud diarization ASR API and
        # map the response to a list of WhisperXSegment. Return [] on failure.
        raise NotImplementedError(
            "CloudEnvASRClient.transcribe: implement cloud env ASR here")


# --------------------------------------------------------------------------- #
# Cloud TTS (text → streamed PCM)
# --------------------------------------------------------------------------- #
class CloudTTSClient:
    """Cloud TTS streaming PCM chunks.

    ``stream`` must be an async generator yielding ``(pcm_bytes, sample_rate)``
    tuples (16-bit signed little-endian mono PCM). Honor ``cancel_check`` between
    chunks so interrupts stop playback promptly.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    async def health(self) -> Optional[Dict[str, Any]]:
        # TODO(cloud): optional health probe; return None if unsupported.
        return None

    async def stream(self, text: str, *,
                     cancel_check: Optional[Callable[[], bool]] = None,
                     ) -> AsyncIterator[Tuple[bytes, int]]:
        # TODO(cloud): stream synthesized audio from your cloud TTS API,
        # yielding (pcm_bytes, sample_rate). Check cancel_check() between chunks.
        raise NotImplementedError(
            "CloudTTSClient.stream: implement cloud TTS here")
        # The following makes this an async generator for type purposes;
        # remove once a real implementation yields.
        if False:  # pragma: no cover
            yield b"", 24000


# --------------------------------------------------------------------------- #
# Cloud speech factory
# --------------------------------------------------------------------------- #
class CloudSpeechFactory(SpeechFactory):
    """SpeechFactory wiring the three cloud clients above."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def make_stt(self) -> Any:
        return CloudSTTClient(self.cfg)

    def make_env_asr(self) -> Any:
        return CloudEnvASRClient(self.cfg)

    def make_tts(self) -> Any:
        return CloudTTSClient(self.cfg)


# =========================================================================== #
# Hermes-native bridge: reuse Hermes' built-in ASR/TTS providers
#
# Hermes already ships a full provider system (agent/tts_registry.py +
# tools/transcription_tools.py): TTS = edge/elevenlabs/openai/minimax/xai/
# mistral/gemini/neutts/kittentts/piper; STT = local/groq/openai/mistral/xai.
# These classes bridge the multimodal speech contract onto those dispatchers,
# so the multimodal agent shares the SAME `tts:`/`stt:` config (provider, voice,
# keys) as the rest of Hermes — pick a vendor in config and it just works.
#
# Impedance match:
#   * Hermes STT takes a FILE PATH → we write the incoming bytes to a temp file
#     (browser sends webm, which Hermes STT accepts directly).
#   * Hermes TTS writes a FILE (mp3/ogg/...) → the frontend's WebAudio player
#     needs raw 16-bit PCM, so we ffmpeg-decode the produced file to s16le PCM
#     and chunk it. ffmpeg is already a Hermes audio dependency.
# =========================================================================== #
_PCM_SAMPLE_RATE = 24000          # frontend default; ffmpeg resamples to this
_PCM_CHUNK_BYTES = 24000 * 2 // 5  # ~100ms of 16-bit mono @24k per WS chunk


def _ffmpeg_bin() -> Optional[str]:
    return shutil.which("ffmpeg")


class HermesSTTClient:
    """User-speech ASR via Hermes' configured transcription provider."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    async def transcribe(self, audio_bytes: bytes, *, mime: str = "audio/webm",
                         filename: Optional[str] = None) -> str:
        if not audio_bytes:
            return ""
        ext = "webm"
        if filename and "." in filename:
            ext = filename.rsplit(".", 1)[-1]
        elif "mp4" in mime or "m4a" in mime:
            ext = "m4a"
        elif "ogg" in mime:
            ext = "ogg"
        elif "wav" in mime:
            ext = "wav"

        def _run() -> str:
            tmp = tempfile.NamedTemporaryFile(prefix="mm_stt_", suffix=f".{ext}", delete=False)
            try:
                tmp.write(audio_bytes)
                tmp.flush()
                tmp.close()
                from tools.transcription_tools import transcribe_audio
                res = transcribe_audio(tmp.name)
                if isinstance(res, dict) and res.get("success"):
                    return (res.get("transcript") or "").strip()
                log.warning("[hermes-stt] failed: %s",
                            (res or {}).get("error") if isinstance(res, dict) else res)
                return ""
            finally:
                try:
                    os.remove(tmp.name)
                except OSError:
                    pass

        try:
            return await asyncio.to_thread(_run)
        except Exception as e:
            log.warning("[hermes-stt] %s", e)
            return ""


class HermesEnvASRClient:
    """Environment-audio ASR via Hermes STT (no speaker diarization).

    Hermes' built-in STT providers don't expose diarization, so each clip maps
    to a single segment with ``speaker=None``. (To get real diarization, use the
    local WhisperX backend or a cloud provider instead.)
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._stt = HermesSTTClient(cfg)

    async def transcribe(
        self, audio_bytes: bytes, *, mime: str = "audio/webm",
        filename: Optional[str] = None, language: Optional[str] = None,
        diarize: Optional[bool] = None, min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> List[WhisperXSegment]:
        text = await self._stt.transcribe(audio_bytes, mime=mime, filename=filename)
        if not text:
            return []
        return [WhisperXSegment(text=text, start=0.0, end=0.0, speaker=None)]


class HermesTTSClient:
    """TTS via Hermes' configured provider, decoded to streamed PCM.

    Hermes TTS writes an encoded audio file; we ffmpeg-decode it to 16-bit mono
    PCM @24kHz and yield it in ~100ms chunks so the frontend WebAudio player can
    schedule playback (same wire shape as the local TTS backend).
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    async def health(self) -> Optional[Dict[str, Any]]:
        return {"backend": "hermes", "ffmpeg": bool(_ffmpeg_bin())}

    def _synth_to_file(self, text: str) -> Optional[str]:
        from tools.tts_tool import text_to_speech_tool
        out = tempfile.NamedTemporaryFile(prefix="mm_tts_", suffix=".mp3", delete=False)
        out.close()

        def _cleanup_placeholder() -> None:
            # ★ C23: remove the empty temp file we created if it is not the file
            # the caller will actually use, so it doesn't leak on disk.
            try:
                os.remove(out.name)
            except OSError:
                pass

        try:
            raw = text_to_speech_tool(text, output_path=out.name)
        except Exception as exc:
            log.warning("[hermes-tts] synth failed: %s", exc)
            _cleanup_placeholder()
            return None
        # text_to_speech_tool returns a JSON string; the real path may differ
        # (provider picks the extension). Parse it, fall back to our path.
        path = out.name
        try:
            obj = json.loads(raw) if isinstance(raw, str) else (raw or {})
            if isinstance(obj, dict):
                if not obj.get("success", True):
                    log.warning("[hermes-tts] failed: %s", obj.get("error"))
                    _cleanup_placeholder()
                    return None
                path = obj.get("file_path") or obj.get("path") or path
        except (ValueError, TypeError):
            pass
        # If the provider wrote somewhere else, our placeholder is dead weight.
        if path != out.name:
            _cleanup_placeholder()
        result = path if os.path.exists(path) else (out.name if os.path.exists(out.name) else None)
        if result is None:
            _cleanup_placeholder()
        return result

    async def stream(self, text: str, *,
                     cancel_check: Optional[Callable[[], bool]] = None,
                     ) -> AsyncIterator[Tuple[bytes, int]]:
        text = (text or "").strip()
        if not text:
            return
        ffmpeg = _ffmpeg_bin()
        if not ffmpeg:
            log.error("[hermes-tts] ffmpeg not found on PATH; cannot decode to PCM")
            return

        audio_path = await asyncio.to_thread(self._synth_to_file, text)
        if not audio_path:
            return
        try:
            # Decode whole file → raw s16le mono PCM @ _PCM_SAMPLE_RATE via ffmpeg.
            def _decode() -> bytes:
                proc = subprocess.run(
                    [ffmpeg, "-v", "error", "-i", audio_path,
                     "-f", "s16le", "-acodec", "pcm_s16le",
                     "-ac", "1", "-ar", str(_PCM_SAMPLE_RATE), "pipe:1"],
                    capture_output=True, check=False)
                if proc.returncode != 0:
                    log.warning("[hermes-tts] ffmpeg decode failed: %s",
                                proc.stderr.decode("utf-8", "ignore")[:200])
                    return b""
                return proc.stdout

            pcm = await asyncio.to_thread(_decode)
            for i in range(0, len(pcm), _PCM_CHUNK_BYTES):
                if cancel_check is not None and cancel_check():
                    return
                chunk = pcm[i:i + _PCM_CHUNK_BYTES]
                if chunk:
                    yield chunk, _PCM_SAMPLE_RATE
        finally:
            try:
                os.remove(audio_path)
            except OSError:
                pass


class HermesSpeechFactory(SpeechFactory):
    """SpeechFactory bridging multimodal speech to Hermes' built-in providers."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def make_stt(self) -> Any:
        return HermesSTTClient(self.cfg)

    def make_env_asr(self) -> Any:
        return HermesEnvASRClient(self.cfg)

    def make_tts(self) -> Any:
        return HermesTTSClient(self.cfg)


def build_cloud_speech_factory(
    hermes_cfg: Optional[Dict[str, Any]] = None, *, cfg: Optional[Config] = None,
) -> Optional[SpeechFactory]:
    """Return a non-local SpeechFactory based on ``multimodal.speech_provider``.

    Values:
      * ``"local"`` (default) → returns ``None``; caller uses the bundled local
        HTTP placeholder backend (LocalSpeechFactory).
      * ``"hermes"`` → :class:`HermesSpeechFactory`, reusing Hermes' built-in
        ASR/TTS providers (configured under the top-level ``tts:``/``stt:``
        config). This is the recommended way to use cloud vendors.
      * ``"cloud"`` → :class:`CloudSpeechFactory`, a user-implemented direct
        cloud integration (fill in the TODO bodies above).
    """
    from .hermes_glue import flatten_mm_config
    mm = flatten_mm_config(hermes_cfg or {})
    provider = str(mm.get("speech_provider", "local")).strip().lower()
    if provider in ("", "local"):
        return None
    if cfg is None:
        from .hermes_glue import build_config
        cfg = build_config(hermes_cfg)
    if provider == "hermes":
        log.info("[multimodal] Hermes-native speech factory active (reuses tts:/stt: config)")
        return HermesSpeechFactory(cfg)
    if provider == "cloud":
        log.info("[multimodal] cloud speech factory active")
        return CloudSpeechFactory(cfg)
    log.warning("[multimodal] unknown speech_provider=%r; falling back to local", provider)
    return None
