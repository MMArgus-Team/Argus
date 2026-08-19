import base64
import types
import unittest
import uuid

from tui_gateway import server as gateway
from tui_gateway.server import (
    _audio_container_signature,
    _optional_finite_float,
    _resolve_env_audio_window,
)


class EnvAudioGatewayHelpersTests(unittest.TestCase):
    def test_webm_cluster_is_not_reported_as_a_standalone_audio_file(self):
        self.assertEqual(
            _audio_container_signature(
                b"\x1f\x43\xb6\x75" + b"payload", "audio/webm;codecs=opus"
            ),
            ("webm_cluster", False),
        )
        self.assertEqual(
            _audio_container_signature(
                b"\x1a\x45\xdf\xa3" + b"payload", "audio/webm;codecs=opus"
            ),
            ("webm_ebml", True),
        )

    def test_env_audio_receipt_timestamp_is_converted_to_window_start(self):
        start, end, duration = _resolve_env_audio_window(
            server_end_ts=14.604,
            client_start_ts=10.0,
            client_end_ts=15.0,
            client_duration_sec=5.0,
            default_duration_sec=5.0,
        )

        self.assertAlmostEqual(start, 9.604)
        self.assertAlmostEqual(end, 14.604)
        self.assertAlmostEqual(duration, 5.0)

    def test_env_audio_window_falls_back_to_client_clock_before_first_frame(self):
        start, end, duration = _resolve_env_audio_window(
            server_end_ts=None,
            client_start_ts=0.0,
            client_end_ts=4.8,
            client_duration_sec=4.8,
            default_duration_sec=5.0,
        )

        self.assertAlmostEqual(start, 0.0)
        self.assertAlmostEqual(end, 4.8)
        self.assertAlmostEqual(duration, 4.8)

    def test_browser_timing_parser_rejects_non_finite_values(self):
        self.assertEqual(_optional_finite_float("4.25"), 4.25)
        self.assertIsNone(_optional_finite_float("nan"))
        self.assertIsNone(_optional_finite_float(float("inf")))
        self.assertIsNone(_optional_finite_float("not-a-number"))

    def test_env_audio_rpc_passes_authoritative_window_and_chunk_diagnostics(self):
        class Backend:
            cfg = types.SimpleNamespace(env_audio_window_sec=5.0)

            def __init__(self):
                self.call = None

            def submit_env_audio(self, audio, *, mime, window_ts, metadata):
                self.call = (audio, mime, window_ts, metadata)
                return True

        backend = Backend()
        sid = f"test-env-audio-{uuid.uuid4().hex}"
        gateway._sessions[sid] = {
            "_mm_memory_backend": backend,
            "agent": types.SimpleNamespace(
                frame_buffer=types.SimpleNamespace(now_ts=lambda: 12.0)
            ),
        }
        wav_like = b"RIFF" + (600).to_bytes(4, "little") + b"WAVE" + b"\0" * 600
        try:
            response = gateway._methods["multimodal.env_audio"]("rpc-1", {
                "session_id": sid,
                "data_b64": base64.b64encode(wav_like).decode("ascii"),
                "mime": "audio/wav",
                "capture_id": "capture-a",
                "chunk_id": "capture-a:2",
                "chunk_seq": 2,
                "client_start_ts": 5.0,
                "client_end_ts": 10.0,
                "client_duration_sec": 5.0,
            })
        finally:
            gateway._sessions.pop(sid, None)

        self.assertTrue(response["result"]["ingested"])
        self.assertIsNotNone(backend.call)
        _, mime, window_start, metadata = backend.call
        self.assertEqual(mime, "audio/wav")
        self.assertAlmostEqual(window_start, 7.0)
        self.assertEqual(metadata["capture_id"], "capture-a")
        self.assertEqual(metadata["chunk_id"], "capture-a:2")
        self.assertEqual(metadata["container"], "wav")
        self.assertTrue(metadata["standalone_header"])
        self.assertAlmostEqual(metadata["server_start_ts"], 7.0)
        self.assertAlmostEqual(metadata["server_end_ts"], 12.0)


if __name__ == "__main__":
    unittest.main()
