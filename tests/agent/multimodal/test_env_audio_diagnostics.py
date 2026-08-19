"""Env-audio diagnostics, energy gate, and exact-chunk dedupe regressions."""

from __future__ import annotations

import hashlib
import io
import math
import shutil
import struct
import threading
import types
import unittest
import wave
from collections import deque
from unittest.mock import patch

from agent.multimodal import _dual_agent
from agent.multimodal.memory_backend import (
    MemoryBackend,
    _decode_env_audio_signal,
    _env_audio_metadata,
    _pcm16_signal_metrics,
)


def _signal(*, rms: float = 0.1) -> dict:
    return {
        "decode_ok": True,
        "decode_reason": "ok",
        "sample_rate": 16_000,
        "samples": 80_000,
        "duration_sec": 5.0,
        "pcm_bytes": 160_000,
        "rms": rms,
        "peak": min(1.0, rms * 2),
        "dbfs": 20 * math.log10(rms) if rms else -120.0,
        "decoder_stderr": "",
    }


class _STT:
    def __init__(self, text: str = "太吓人了。"):
        self.text = text
        self.calls = 0
        self.last_diagnostics = {
            "ok": True,
            "reason": "completed",
            "status": 200,
            "provider_request_id": "req-test",
            "model": "qwen-test",
        }

    async def transcribe(self, _audio: bytes, *, mime: str) -> str:
        self.calls += 1
        return self.text


def _backend(*, stt_text: str = "太吓人了。"):
    events = []
    writes = []
    obj = MemoryBackend.__new__(MemoryBackend)
    obj.cfg = types.SimpleNamespace(
        env_audio_window_sec=5.0,
        env_audio_enabled=True,
        env_audio_min_rms=0.005,
        env_asr_backend="qwen",
        env_audio_min_text_chars=2,
        conv_max_audio_obs=0,
    )
    obj._emit_cb = lambda event, payload: events.append((event, payload))
    obj._env_audio_seen_keys = set()
    obj._env_audio_seen_order = deque(maxlen=512)
    obj.audio_buffer = None
    obj.stt = _STT(stt_text)
    obj.whisperx = None
    obj.conversation = types.SimpleNamespace(
        _lock=threading.RLock(), _turns=[])
    obj._push_ctx = lambda: None
    obj._last_env_asr_error_emit = 0.0
    return obj, events, writes


class SignalHelperTests(unittest.TestCase):
    def test_pcm16_metrics_are_normalized_and_deterministic(self):
        pcm = struct.pack("<hhhh", 1000, -1000, 1000, -1000)
        result = _pcm16_signal_metrics(pcm, sample_rate=4)

        self.assertTrue(result["decode_ok"])
        self.assertEqual(result["samples"], 4)
        self.assertAlmostEqual(result["duration_sec"], 1.0)
        self.assertAlmostEqual(result["rms"], 1000 / 32768.0)
        self.assertAlmostEqual(result["peak"], 1000 / 32768.0)
        self.assertAlmostEqual(
            result["dbfs"], 20 * math.log10(1000 / 32768.0))

    def test_metadata_fallback_is_stable_and_legacy_compatible(self):
        audio = b"encoded-audio"
        result = _env_audio_metadata(
            audio,
            mime="audio/webm;codecs=opus",
            rel_ts=12.5,
            window_sec=5.0,
            metadata=None,
        )

        self.assertEqual(result["sha256"], hashlib.sha256(audio).hexdigest())
        self.assertTrue(result["chunk_id"].startswith("aud_"))
        self.assertEqual(result["container"], "webm")
        self.assertAlmostEqual(result["t_start"], 12.5)
        self.assertAlmostEqual(result["t_end"], 17.5)
        self.assertAlmostEqual(result["rel_ts"], 12.5)

    def test_server_timing_wins_and_gateway_diagnostics_survive(self):
        result = _env_audio_metadata(
            b"audio",
            mime="audio/webm",
            rel_ts=99.0,
            window_sec=5.0,
            metadata={
                "server_start_ts": 3.25,
                "server_end_ts": 7.75,
                "client_duration_sec": 4.48,
                "blob_timecode": 1234.0,
                "sha256_short": "abc123",
            },
        )

        self.assertEqual(result["t_start"], 3.25)
        self.assertEqual(result["t_end"], 7.75)
        self.assertEqual(result["client_duration_sec"], 4.48)
        self.assertEqual(result["blob_timecode"], 1234.0)
        self.assertEqual(result["sha256_short"], "abc123")


class EnvAudioAsyncTests(unittest.IsolatedAsyncioTestCase):
    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is not installed")
    async def test_ffmpeg_decodes_wav_and_measures_energy(self):
        samples = [4000 if i % 2 else -4000 for i in range(1600)]
        encoded = io.BytesIO()
        with wave.open(encoded, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16_000)
            wav.writeframes(struct.pack(f"<{len(samples)}h", *samples))

        result = await _decode_env_audio_signal(encoded.getvalue())

        self.assertTrue(result["decode_ok"])
        self.assertAlmostEqual(result["duration_sec"], 0.1, delta=0.01)
        self.assertGreater(result["rms"], 0.1)
        self.assertAlmostEqual(
            result["peak"], 4000 / 32768.0, delta=0.003)

    async def test_low_rms_is_filtered_before_asr(self):
        obj, events, _ = _backend()

        async def quiet(_audio):
            return _signal(rms=0.001)

        with patch(
                "agent.multimodal.memory_backend._decode_env_audio_signal",
                quiet):
            result = await obj._ingest_env_audio(
                b"quiet",
                metadata={"chunk_id": "aud-1", "seq": 1, "sha": "sha-1"},
            )

        self.assertEqual(result, "")
        self.assertEqual(obj.stt.calls, 0)
        filtered = [payload for _, payload in events
                    if payload.get("phase") == "filtered"][-1]
        self.assertEqual(filtered["reason"], "low_rms")
        self.assertFalse(filtered["asr_called"])
        self.assertAlmostEqual(filtered["signal"]["rms"], 0.001)
        self.assertEqual(filtered["write_reason"], "not_written_low_rms")

    async def test_decode_failure_is_visible_and_never_calls_asr(self):
        obj, events, _ = _backend()

        async def broken(_audio):
            return {
                **_pcm16_signal_metrics(b""),
                "decode_reason": "ffmpeg_exit_1",
                "decoder_stderr": "invalid data",
            }

        with patch(
                "agent.multimodal.memory_backend._decode_env_audio_signal",
                broken):
            result = await obj._ingest_env_audio(
                b"broken",
                metadata={"chunk_id": "aud-2", "sha": "sha-2"},
            )

        self.assertEqual(result, "")
        self.assertEqual(obj.stt.calls, 0)
        failed = [payload for _, payload in events
                  if payload.get("phase") == "decode_failed"][-1]
        self.assertEqual(failed["reason"], "ffmpeg_exit_1")
        self.assertEqual(failed["signal"]["decoder_stderr"], "invalid data")
        self.assertEqual(
            failed["write_reason"], "not_written_decode_failed")

    async def test_missing_ffmpeg_keeps_direct_cloud_asr_available(self):
        obj, events, writes = _backend(stt_text="仍然能识别")

        async def no_ffmpeg(_audio):
            return {
                **_pcm16_signal_metrics(b""),
                "decode_reason": "ffmpeg_not_found",
                "decoder_stderr": "ffmpeg is not available on PATH",
            }

        async def append(_conversation, text, *, rel_ts=None, speaker=None):
            writes.append((text, rel_ts, speaker))

        with (
            patch("agent.multimodal.memory_backend._decode_env_audio_signal",
                  no_ffmpeg),
            patch.object(_dual_agent, "append_audio_observation", append),
        ):
            result = await obj._ingest_env_audio(
                b"valid-cloud-audio",
                metadata={"chunk_id": "aud-no-ffmpeg", "sha": "sha-no-ffmpeg",
                          "t_start": 5.0, "t_end": 10.0},
            )

        self.assertEqual(result, "仍然能识别")
        self.assertEqual(obj.stt.calls, 1)
        self.assertEqual(writes, [("仍然能识别", 5.0, None)])
        unavailable = [payload for _, payload in events
                       if payload.get("phase") == "signal_unavailable"][-1]
        self.assertEqual(unavailable["reason"], "ffmpeg_not_found")
        self.assertTrue(unavailable["asr_will_run"])

    async def test_provider_http_error_is_failed_not_silence(self):
        obj, events, _ = _backend(stt_text="")
        obj.stt.last_diagnostics = {
            "ok": False,
            "reason": "http_error",
            "status": 400,
            "provider_request_id": "req-400",
        }

        async def loud(_audio):
            return _signal(rms=0.1)

        with patch(
                "agent.multimodal.memory_backend._decode_env_audio_signal",
                loud):
            result = await obj._ingest_env_audio(
                b"request-fails",
                metadata={"chunk_id": "aud-http", "sha": "sha-http"},
            )

        self.assertEqual(result, "")
        self.assertFalse(any(payload.get("phase") == "silence"
                             for _, payload in events))
        failed = [payload for _, payload in events
                  if payload.get("phase") == "failed"][-1]
        self.assertEqual(failed["reason"], "http_error")
        self.assertEqual(failed["provider"]["status"], 400)
        self.assertEqual(failed["write_reason"], "not_written_asr_failed")

    async def test_chunk_idempotency_does_not_erase_legitimate_replays(self):
        obj, events, writes = _backend()

        async def loud(_audio):
            return _signal(rms=0.1)

        async def append(_conversation, text, *, rel_ts=None, speaker=None):
            writes.append((text, rel_ts, speaker))

        with (
            patch("agent.multimodal.memory_backend._decode_env_audio_signal",
                  loud),
            patch.object(_dual_agent, "append_audio_observation", append),
        ):
            first = await obj._ingest_env_audio(
                b"same-audio",
                metadata={"chunk_id": "aud-3", "seq": 3, "sha": "same-sha",
                          "t_start": 10.0, "t_end": 15.0})
            duplicate = await obj._ingest_env_audio(
                b"same-audio",
                metadata={"chunk_id": "aud-3", "seq": 3, "sha": "same-sha",
                          "t_start": 10.0, "t_end": 15.0})
            replayed_audio = await obj._ingest_env_audio(
                b"same-audio",
                metadata={"chunk_id": "aud-4", "seq": 4, "sha": "same-sha",
                          "t_start": 15.0, "t_end": 20.0})
            repeated_text = await obj._ingest_env_audio(
                b"different-audio",
                metadata={"chunk_id": "aud-5", "seq": 5,
                          "sha": "different-sha",
                          "t_start": 40.0, "t_end": 45.0})

        self.assertEqual(first, "太吓人了。")
        self.assertEqual(duplicate, "")
        self.assertEqual(replayed_audio, "太吓人了。")
        self.assertEqual(repeated_text, "太吓人了。")
        self.assertEqual(obj.stt.calls, 3)
        self.assertEqual(writes, [
            ("太吓人了。", 10.0, None),
            ("太吓人了。", 15.0, None),
            ("太吓人了。", 40.0, None),
        ])
        duplicate_event = [payload for _, payload in events
                           if payload.get("phase") == "duplicate"][-1]
        self.assertEqual(duplicate_event["reason"], "duplicate_chunk")

    async def test_completed_event_has_full_diagnostics(self):
        obj, events, writes = _backend(stt_text="收60服务费")

        async def loud(_audio):
            return _signal(rms=0.2)

        async def append(_conversation, text, *, rel_ts=None, speaker=None):
            writes.append((text, rel_ts, speaker))

        with (
            patch("agent.multimodal.memory_backend._decode_env_audio_signal",
                  loud),
            patch.object(_dual_agent, "append_audio_observation", append),
        ):
            result = await obj._ingest_env_audio(
                b"speech",
                mime="audio/webm;codecs=opus",
                metadata={
                    "chunk_id": "aud-6", "seq": 6, "sha": "sha-6",
                    "container": "webm", "t_start": 0.0, "t_end": 5.0,
                    "standalone_header": True, "header_hex": "1a45dfa3",
                },
            )

        self.assertEqual(result, "收60服务费")
        self.assertEqual(writes, [("收60服务费", 0.0, None)])
        completed = [payload for _, payload in events
                     if payload.get("phase") == "completed"][-1]
        self.assertEqual(completed["chunk_id"], "aud-6")
        self.assertEqual(completed["seq"], 6)
        self.assertEqual(completed["sha"], "sha-6")
        self.assertEqual(completed["t_start"], 0.0)
        self.assertEqual(completed["t_end"], 5.0)
        self.assertTrue(completed["standalone_header"])
        self.assertAlmostEqual(completed["signal"]["rms"], 0.2)
        self.assertEqual(
            completed["provider"]["provider_request_id"], "req-test")
        self.assertEqual(completed["raw_transcript"], "收60服务费")
        self.assertIsNone(completed["filter_reason"])
        self.assertEqual(
            completed["write_reason"], "audio_observation_appended")
        self.assertGreaterEqual(completed["asr_latency_sec"], 0)


if __name__ == "__main__":
    unittest.main()
