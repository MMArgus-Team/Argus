"""Tests for MonitorEngine — the async per-monitor job container.

Drives ``_tick_once`` directly on a test event loop with a fake async LLM client
and fake frame buffer, so the ported per-tick behavior (SPEAK emit, silent
record-only, aggregation flush, circuit breaker, cursor advance) is verified
deterministically without the background thread. A separate test exercises the
real thread + add_monitor/remove_monitor job lifecycle.
"""

import asyncio
import base64
from io import BytesIO
import time
import types
import unittest
from unittest.mock import patch

from agent.multimodal.monitor_engine import (
    MonitorEngine, build_monitor_evidence, merge_monitor_evidence,
    parse_monitor_verdict, should_flush, pick_window, sample_frames,
)
from PIL import Image


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class _Frame:
    def __init__(self, ts, *, jpeg_b64="AAAA", source_type="screen"):
        self.ts = ts
        self.jpeg_b64 = jpeg_b64
        self.source_type = source_type


class _Buf:
    """Mirrors agent.multimodal._memory.FrameBuffer's read contract: ``size`` and
    ``latest_ts`` are PROPERTIES (not methods), ``all_after`` is a method."""
    def __init__(self, frames):
        self._frames = list(frames)
        self._last_push_wall = time.time()

    @property
    def size(self):
        return len(self._frames)

    @property
    def latest_ts(self):
        return self._frames[-1].ts if self._frames else None

    def all_after(self, ts):
        return [f for f in self._frames if f.ts >= ts]


class _RawBuf(_Buf):
    """Production-shaped dual stream: sparse memory frames + raw monitor frames."""
    def __init__(self, dedup_frames, monitor_frames):
        super().__init__(dedup_frames)
        self._monitor_frames = list(monitor_frames)
        self.source_generation = 0

    @property
    def monitor_size(self):
        return len(self._monitor_frames)

    @property
    def monitor_latest_ts(self):
        return self._monitor_frames[-1].ts if self._monitor_frames else None

    def monitor_all_after(self, ts):
        return [f for f in self._monitor_frames if f.ts >= ts]


def _resp(text):
    msg = types.SimpleNamespace(content=text, reasoning_content=None, reasoning=None)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


class _FakeAsyncClient:
    """Minimal AsyncOpenAI shape: client.chat.completions.create(**kw) coroutine."""
    def __init__(self, reply="SILENT", *, raise_exc=None):
        self.reply = reply
        self.raise_exc = raise_exc
        self.calls = 0
        self.last_kwargs = None

        async def _create(**kw):
            self.calls += 1
            self.last_kwargs = kw
            if self.raise_exc is not None:
                raise self.raise_exc
            return _resp(self.reply)

        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=_create))


def _make_engine(buf, monitors, *, client, model="m",
                 speak_cb=None, notify_cb=None,
                 stale=False, busy=False):
    eng = MonitorEngine(
        buf, monitors_ref=monitors, sid="s1",
        speak_cb=speak_cb, notify_cb=notify_cb,
        is_source_off=(lambda: stale), is_session_busy=(lambda: busy))
    eng.client = client
    eng.model = model
    return eng


_MM_ON = {"monitor_enabled": True, "monitor_tick_sec": 1.0,
          "monitor_overload_period": 8, "monitor_fail_disable_after": 3}


class TestTickBehavior(unittest.TestCase):
    def setUp(self):
        # append_event writes to disk — stub it everywhere the engine imports it.
        self._patcher = patch("agent.multimodal.monitor_agent.append_event",
                              return_value=("eid", "path"))
        self._append = self._patcher.start()
        self._check_patcher = patch("agent.multimodal.monitor_agent.append_check")
        self._append_check = self._check_patcher.start()
        # Production monitors always have an initialized event file. Keep the
        # unit fixture focused on engine state transitions by accepting status
        # writes unless a test overrides this patch explicitly.
        self._status_patcher = patch(
            "agent.multimodal.monitor_agent.set_status", return_value=True)
        self._set_status = self._status_patcher.start()

    def tearDown(self):
        self._status_patcher.stop()
        self._check_patcher.stop()
        self._patcher.stop()

    def _tick(self, eng, mid):
        asyncio.run(eng._tick_once(mid, _MM_ON))

    def test_speak_emits_and_advances_cursor(self):
        spoken = []
        buf = _Buf([_Frame(1.0), _Frame(2.0), _Frame(3.0)])
        mons = {"m1": {"brief": "有人进来就说", "enabled": True,
                       "last_seen_ts": 0.0}}
        client = _FakeAsyncClient("SPEAK: 有人进来了")
        eng = _make_engine(buf, mons, client=client,
                           speak_cb=lambda mid, m, t: spoken.append((mid, t)) or True)
        self._tick(eng, "m1")
        self.assertEqual(len(spoken), 1)
        self.assertEqual(spoken[0][1], "有人进来了")
        self.assertTrue(self._append.called)
        self.assertEqual(client.last_kwargs["max_tokens"], 2048)
        self.assertEqual(client.last_kwargs["extra_body"], {
            "chat_template_kwargs": {"enable_thinking": False},
        })
        # cursor advanced to newest frame shown
        self.assertEqual(mons["m1"]["last_seen_ts"], 3.0)

    def test_speak_evidence_is_bounded_and_comes_from_exact_model_batch(self):
        delivered = []
        trajectory = []
        frames = [_Frame(float(i), jpeg_b64=f"raw-{i}") for i in range(1, 11)]
        buf = _Buf(frames)
        mons = {"m1": {"brief": "x", "enabled": True, "last_seen_ts": 0.5}}

        def _speak(_mid, monitor, _text):
            delivered.append(monitor.get("_delivery_evidence"))
            return True

        eng = MonitorEngine(
            buf,
            monitors_ref=mons,
            sid="s1",
            speak_cb=_speak,
            emit_cb=lambda event, payload: trajectory.append((event, payload)),
        )
        eng.client = _FakeAsyncClient("SPEAK: target")
        eng.model = "m"
        with patch(
            "agent.multimodal.monitor_engine._monitor_evidence_thumb",
            side_effect=lambda raw: f"thumb-{raw}",
        ):
            self._tick(eng, "m1")

        evidence = delivered[0]
        self.assertEqual(evidence["input_count"], 10)
        self.assertEqual(evidence["shown_count"], 6)
        self.assertEqual(
            [row["ts"] for row in evidence["frames"]],
            [1.0, 3.0, 5.0, 6.0, 8.0, 10.0],
        )
        verdict = next(payload for event, payload in trajectory
                       if event == "multimodal.trajectory"
                       and payload.get("phase") == "verdict")
        self.assertEqual(verdict["frames"], evidence["frames"])
        self.assertNotIn("_delivery_evidence", mons["m1"])

    def test_silent_does_not_record_or_emit(self):
        spoken = []
        buf = _Buf([_Frame(1.0), _Frame(2.0)])
        mons = {"m1": {"brief": "x", "enabled": True, "last_seen_ts": 0.0}}
        eng = _make_engine(buf, mons, client=_FakeAsyncClient("SILENT"),
                           speak_cb=lambda *a: spoken.append(a) or True)
        self._tick(eng, "m1")
        self.assertEqual(spoken, [])
        self.assertFalse(self._append.called)  # SILENT → nothing observed
        self.assertFalse(self._append_check.called)  # no [未触发] timeline noise

    def test_monitor_prefers_raw_frames_over_deduped_memory_stream(self):
        buf = _RawBuf(
            dedup_frames=[_Frame(1.0)],
            monitor_frames=[_Frame(1.0), _Frame(1.5), _Frame(2.0)],
        )
        mons = {"m1": {"brief": "x", "enabled": True, "last_seen_ts": 0.5}}
        client = _FakeAsyncClient("SILENT")
        eng = _make_engine(buf, mons, client=client)
        self._tick(eng, "m1")
        image_parts = client.last_kwargs["messages"][1]["content"][1:]
        self.assertEqual(len(image_parts), 3)
        self.assertEqual(mons["m1"]["last_seen_ts"], 2.0)

    def test_luna_uses_portable_completion_parameters(self):
        buf = _Buf([_Frame(1.0)])
        mons = {"m1": {"brief": "x", "enabled": True, "last_seen_ts": 0.5}}
        client = _FakeAsyncClient("SILENT")
        # Use the exact spelling returned by the configured doc endpoint; the
        # historical hyphenated alias must not be the only portable spelling.
        eng = _make_engine(buf, mons, client=client, model="GPT-5.6 Luna")
        self._tick(eng, "m1")
        # Luna's completion budget also covers hidden reasoning. A multi-image
        # verdict needs enough headroom to emit the final SPEAK/SILENT line.
        self.assertEqual(client.last_kwargs["max_completion_tokens"], 4096)
        self.assertNotIn("max_tokens", client.last_kwargs)
        self.assertNotIn("temperature", client.last_kwargs)
        self.assertNotIn("extra_body", client.last_kwargs)

    def test_kimi_k26_uses_instant_mode_short_messages_contract(self):
        buf = _Buf([_Frame(1.0)])
        mons = {"m1": {"brief": "x", "enabled": True, "last_seen_ts": 0.5}}
        client = _FakeAsyncClient("SILENT")
        eng = _make_engine(buf, mons, client=client, model="kimi-k2.6")
        self._tick(eng, "m1")
        self.assertEqual(client.last_kwargs["max_tokens"], 128)
        self.assertEqual(client.last_kwargs["timeout"], 30.0)
        self.assertNotIn("max_completion_tokens", client.last_kwargs)
        self.assertNotIn("temperature", client.last_kwargs)
        self.assertEqual(client.last_kwargs["extra_body"], {
            "chat_template_kwargs": {"thinking": False},
        })
        self.assertNotIn("reasoning_effort", client.last_kwargs)

    def test_late_verdict_is_discarded_after_source_switch(self):
        buf = _RawBuf([_Frame(1.0)], [_Frame(1.0)])
        mons = {"m1": {"brief": "x", "enabled": True, "last_seen_ts": 0.5}}
        spoken = []
        client = _FakeAsyncClient("SPEAK: 旧摄像头里的手机")
        original_create = client.chat.completions.create

        async def _switch_source_during_request(**kwargs):
            buf.source_generation += 1
            return await original_create(**kwargs)

        client.chat.completions.create = _switch_source_during_request
        eng = _make_engine(
            buf, mons, client=client,
            speak_cb=lambda *args: spoken.append(args) or True,
        )
        self._tick(eng, "m1")
        self.assertEqual(spoken, [])
        self.assertEqual(mons["m1"]["last_seen_ts"], 0.5)

    def test_silent_flag_records_event_but_no_emit(self):
        # A SPEAK from the model on a monitor whose silent=True: record to file,
        # never emit.
        spoken = []
        buf = _Buf([_Frame(1.0), _Frame(2.0)])
        mons = {"m1": {"brief": "x", "enabled": True, "silent": True,
                       "last_seen_ts": 0.0}}
        eng = _make_engine(buf, mons, client=_FakeAsyncClient("SPEAK: 事件"),
                           speak_cb=lambda *a: spoken.append(a) or True)
        self._tick(eng, "m1")
        self.assertEqual(spoken, [])
        self.assertTrue(self._append.called)  # observed → recorded

    def test_once_completes_after_first_accepted_delivery(self):
        spoken = []
        buf = _Buf([_Frame(1.0), _Frame(2.0)])
        mons = {
            "m1": {
                "brief": "打开第一个视频后告诉我标题",
                "enabled": True,
                "trigger_mode": "once",
                "last_seen_ts": 0.0,
            },
        }
        eng = _make_engine(
            buf,
            mons,
            client=_FakeAsyncClient("SPEAK: 视频标题是《测试》"),
            speak_cb=lambda mid, m, text: spoken.append(text) or True,
        )

        with patch("agent.multimodal.monitor_agent.set_status") as set_status:
            self._tick(eng, "m1")

        self.assertEqual(spoken, ["视频标题是《测试》"])
        self.assertFalse(mons["m1"]["enabled"])
        self.assertEqual(mons["m1"]["status"], "done")
        set_status.assert_called_with("m1", "done")

    def test_once_waits_until_delivery_is_accepted(self):
        buf = _Buf([_Frame(1.0)])
        mons = {
            "m1": {
                "brief": "x",
                "enabled": True,
                "trigger_mode": "once",
                "last_seen_ts": 0.0,
            },
        }
        eng = _make_engine(
            buf,
            mons,
            client=_FakeAsyncClient("SPEAK: 命中"),
            speak_cb=lambda *_args: False,
        )

        self._tick(eng, "m1")

        self.assertTrue(mons["m1"]["enabled"])
        self.assertNotEqual(mons["m1"].get("status"), "done")

    def test_inflight_verdict_is_discarded_after_query_mode_revision(self):
        spoken = []
        buf = _Buf([_Frame(1.0)])
        monitor = {
            "brief": "old cat task",
            "enabled": True,
            "trigger_mode": "continuous",
            "_config_revision": 0,
            "last_seen_ts": 0.0,
        }
        mons = {"m1": monitor}
        client = _FakeAsyncClient("SPEAK: old cat hit")
        original_create = client.chat.completions.create

        async def _update_during_request(**kwargs):
            response = await original_create(**kwargs)
            monitor.update(
                brief="new dog task",
                monitor_query="new dog task",
                trigger_mode="once",
                _config_revision=1,
                _fail_streak=2,
                _err_notified=True,
            )
            return response

        client.chat.completions.create = _update_during_request
        eng = _make_engine(
            buf, mons, client=client,
            speak_cb=lambda _mid, _m, text: spoken.append(text) or True,
        )

        self._tick(eng, "m1")

        self.assertEqual(spoken, [])
        self.assertTrue(monitor["enabled"])
        self.assertNotEqual(monitor.get("status"), "done")
        self.assertEqual(monitor["last_seen_ts"], 0.0)
        self.assertEqual(monitor["_fail_streak"], 2)
        self.assertTrue(monitor["_err_notified"])
        self._append.assert_not_called()

    def test_inflight_failure_is_discarded_after_query_revision(self):
        notes = []
        buf = _Buf([_Frame(1.0)])
        monitor = {
            "brief": "old cat task",
            "enabled": True,
            "trigger_mode": "continuous",
            "_config_revision": 0,
            "_fail_streak": 0,
            "last_seen_ts": 0.0,
        }
        mons = {"m1": monitor}
        client = _FakeAsyncClient()

        async def _update_then_fail(**_kwargs):
            monitor.update(
                brief="new dog task",
                monitor_query="new dog task",
                _config_revision=1,
            )
            raise RuntimeError("old endpoint failure")

        client.chat.completions.create = _update_then_fail
        eng = _make_engine(
            buf,
            mons,
            client=client,
            notify_cb=lambda kind, _mid, _m, text: notes.append((kind, text)),
        )

        self._tick(eng, "m1")

        self.assertEqual(monitor["_fail_streak"], 0)
        self.assertTrue(monitor["enabled"])
        self.assertEqual(notes, [])

    def test_inflight_verdict_is_discarded_after_disable(self):
        spoken = []
        buf = _Buf([_Frame(1.0)])
        monitor = {
            "brief": "cat",
            "enabled": True,
            "trigger_mode": "continuous",
            "_config_revision": 0,
            "last_seen_ts": 0.0,
        }
        mons = {"m1": monitor}
        client = _FakeAsyncClient("SPEAK: late hit")
        original_create = client.chat.completions.create

        async def _disable_during_request(**kwargs):
            response = await original_create(**kwargs)
            monitor["enabled"] = False
            monitor["_config_revision"] = 1
            return response

        client.chat.completions.create = _disable_during_request
        eng = _make_engine(
            buf, mons, client=client,
            speak_cb=lambda _mid, _m, text: spoken.append(text) or True,
        )

        self._tick(eng, "m1")

        self.assertEqual(spoken, [])
        self._append.assert_not_called()

    def test_once_event_write_failure_stays_enabled_and_notifies(self):
        notes = []
        buf = _Buf([_Frame(1.0)])
        monitor = {
            "brief": "record once",
            "enabled": True,
            "trigger_mode": "once",
            "silent": True,
            "last_seen_ts": 0.0,
        }
        eng = _make_engine(
            buf,
            {"m1": monitor},
            client=_FakeAsyncClient("SPEAK: hit"),
            notify_cb=lambda kind, _mid, _m, text: notes.append((kind, text)),
        )
        self._append.side_effect = OSError("disk full")

        self._tick(eng, "m1")

        self.assertTrue(monitor["enabled"])
        self.assertNotEqual(monitor.get("status"), "done")
        self.assertEqual(monitor["last_seen_ts"], 0.0)
        self.assertTrue(any(kind == "error" for kind, _text in notes))

    def test_once_done_write_failure_retries_without_duplicate_delivery(self):
        spoken = []
        notes = []
        buf = _Buf([_Frame(1.0)])
        monitor = {
            "brief": "tell once",
            "enabled": True,
            "trigger_mode": "once",
            "last_seen_ts": 0.0,
        }
        eng = _make_engine(
            buf,
            {"m1": monitor},
            client=_FakeAsyncClient("SPEAK: hit"),
            speak_cb=lambda _mid, _m, text: spoken.append(text) or True,
            notify_cb=lambda kind, _mid, _m, text: notes.append((kind, text)),
        )
        self._set_status.side_effect = [False, True]

        self._tick(eng, "m1")
        self.assertTrue(monitor["enabled"])
        self.assertIn("_once_pending_completion", monitor)
        self.assertEqual(spoken, ["hit"])
        self.assertTrue(any(kind == "error" for kind, _text in notes))

        # No new frame is required: the pending terminal write is retried before
        # the ordinary frame gates and the already accepted alert is not repeated.
        self._tick(eng, "m1")
        self.assertFalse(monitor["enabled"])
        self.assertEqual(monitor["status"], "done")
        self.assertEqual(spoken, ["hit"])
        self.assertNotIn("_once_pending_completion", monitor)

    def test_silent_once_completes_on_first_hit_without_delivery(self):
        spoken = []
        buf = _Buf([_Frame(1.0)])
        mons = {
            "m1": {
                "brief": "x",
                "enabled": True,
                "trigger_mode": "once",
                "silent": True,
                "last_seen_ts": 0.0,
            },
        }
        eng = _make_engine(
            buf,
            mons,
            client=_FakeAsyncClient("SPEAK: 命中"),
            speak_cb=lambda *args: spoken.append(args) or True,
        )

        self._tick(eng, "m1")

        self.assertEqual(spoken, [])
        self.assertFalse(mons["m1"]["enabled"])
        self.assertEqual(mons["m1"]["status"], "done")

    def test_continuous_requires_silent_before_next_delivery(self):
        spoken = []
        buf = _Buf([_Frame(1.0)])
        # Missing trigger_mode intentionally exercises legacy=continuous.
        mons = {
            "m1": {"brief": "x", "enabled": True, "last_seen_ts": 0.0},
        }
        client = _FakeAsyncClient("SPEAK: 第一次命中")
        eng = _make_engine(
            buf,
            mons,
            client=client,
            speak_cb=lambda mid, m, text: spoken.append(text) or True,
        )

        self._tick(eng, "m1")
        buf._frames.append(_Frame(2.0))
        client.reply = "SPEAK: 同一个持续画面"
        self._tick(eng, "m1")
        self.assertEqual(spoken, ["第一次命中"])

        buf._frames.append(_Frame(3.0))
        client.reply = "SILENT"
        self._tick(eng, "m1")
        self.assertTrue(mons["m1"]["_trigger_armed"])

        buf._frames.append(_Frame(4.0))
        client.reply = "SPEAK: 新事件"
        self._tick(eng, "m1")
        self.assertEqual(spoken, ["第一次命中", "新事件"])
        self.assertEqual(self._append.call_count, 2)

    def test_continuous_does_not_rearm_on_invalid_or_empty_verdict(self):
        spoken = []
        buf = _Buf([_Frame(1.0)])
        mons = {
            "m1": {
                "brief": "x",
                "enabled": True,
                "trigger_mode": "continuous",
                "last_seen_ts": 0.0,
            },
        }
        client = _FakeAsyncClient("SPEAK: 第一次命中")
        eng = _make_engine(
            buf,
            mons,
            client=client,
            speak_cb=lambda _mid, _m, text: spoken.append(text) or True,
        )

        self._tick(eng, "m1")
        self.assertFalse(mons["m1"]["_trigger_armed"])

        for ts, invalid in ((2.0, "not a verdict"), (3.0, "")):
            buf._frames.append(_Frame(ts))
            client.reply = invalid
            self._tick(eng, "m1")
            self.assertFalse(mons["m1"]["_trigger_armed"])

        buf._frames.append(_Frame(4.0))
        client.reply = "SPEAK: 同一个持续画面"
        self._tick(eng, "m1")
        self.assertEqual(spoken, ["第一次命中"])

        buf._frames.append(_Frame(5.0))
        client.reply = "SILENT"
        self._tick(eng, "m1")
        self.assertTrue(mons["m1"]["_trigger_armed"])

    def test_aggregation_buffers_then_flush(self):
        spoken = []
        buf = _Buf([_Frame(1.0), _Frame(2.0)])
        mons = {"m1": {"brief": "x", "enabled": True, "report_interval": 10,
                       "last_seen_ts": 0.0, "_agg_buf": [], "_agg_window_start": 0.0}}
        eng = _make_engine(buf, mons, client=_FakeAsyncClient("SPEAK: 事件A"),
                           speak_cb=lambda mid, m, t: spoken.append(t) or True)
        # First tick: SPEAK buffered, not emitted.
        self._tick(eng, "m1")
        self.assertEqual(spoken, [])
        self.assertEqual(len(mons["m1"]["_agg_buf"]), 1)
        # Force the window due and flush (no new frames needed).
        mons["m1"]["_agg_window_start"] = time.time() - 100
        eng2 = _make_engine(buf, mons, client=_FakeAsyncClient("SILENT"),
                            speak_cb=lambda mid, m, t: spoken.append(t) or True)
        # advance cursor so no new-frame SPEAK; only the flush should fire
        mons["m1"]["last_seen_ts"] = 99.0
        self._tick(eng2, "m1")
        self.assertEqual(len(spoken), 1)
        self.assertIn("事件A", spoken[0])
        self.assertEqual(mons["m1"]["_agg_buf"], [])

    def test_circuit_breaker_disables_after_n_failures(self):
        notes = []
        buf = _Buf([_Frame(1.0), _Frame(2.0)])
        mons = {"m1": {"brief": "x", "enabled": True, "last_seen_ts": 0.0}}
        eng = _make_engine(
            buf, mons, client=_FakeAsyncClient(raise_exc=RuntimeError("boom")),
            notify_cb=lambda kind, mid, m, t: notes.append((kind, t)))
        # fail_disable_after=3 → 3 failing ticks disable it
        for _ in range(3):
            mons["m1"]["last_seen_ts"] = 0.0  # keep new frames available
            self._tick(eng, "m1")
        self.assertFalse(mons["m1"]["enabled"])
        self.assertTrue(any(k == "interrupted" for k, _ in notes))

    def test_disabled_monitor_skipped(self):
        buf = _Buf([_Frame(1.0)])
        mons = {"m1": {"brief": "x", "enabled": False, "last_seen_ts": 0.0}}
        c = _FakeAsyncClient('{"status": true, "reason": "y"}')
        eng = _make_engine(buf, mons, client=c, speak_cb=lambda *a: True)
        self._tick(eng, "m1")
        self.assertEqual(c.calls, 0)  # never called the LLM

    def test_stale_stream_skips_new_frames_but_flush_still_runs(self):
        spoken = []
        buf = _Buf([_Frame(1.0), _Frame(2.0)])
        mons = {"m1": {"brief": "x", "enabled": True, "report_interval": 10,
                       "last_seen_ts": 99.0,
                       "_agg_buf": [(time.time() - 100, "旧事件")],
                       "_agg_window_start": time.time() - 100}}
        c = _FakeAsyncClient('{"status": true, "reason": "new"}')
        eng = _make_engine(buf, mons, client=c, stale=True,
                           speak_cb=lambda mid, m, t: spoken.append(t) or True)
        self._tick(eng, "m1")
        # flush fired (due window) despite stale stream; no LLM eval of new frames
        self.assertEqual(c.calls, 0)
        self.assertEqual(len(spoken), 1)
        self.assertIn("旧事件", spoken[0])

    def test_busy_foreground_does_not_blind_monitor_evaluation(self):
        spoken = []
        buf = _Buf([_Frame(1.0), _Frame(2.0)])
        mons = {"m1": {"brief": "x", "enabled": True, "last_seen_ts": 0.0}}
        client = _FakeAsyncClient("SPEAK: 并发命中")
        eng = _make_engine(
            buf, mons, client=client, busy=True,
            speak_cb=lambda mid, m, t: spoken.append(t) or True)

        self._tick(eng, "m1")

        self.assertEqual(client.calls, 1)
        self.assertEqual(spoken, ["并发命中"])
        self.assertEqual(mons["m1"]["last_seen_ts"], 2.0)


class TestJobLifecycle(unittest.TestCase):
    def test_add_and_remove_monitor_thread(self):
        with patch("agent.multimodal.monitor_agent.append_event",
                   return_value=("eid", "path")):
            spoken = []
            buf = _Buf([_Frame(1.0), _Frame(2.0), _Frame(3.0)])
            mons = {}
            eng = MonitorEngine(
                buf, monitors_ref=mons, sid="s1",
                speak_cb=lambda mid, m, t: spoken.append(t) or True)
            # Patch _build to inject a fake client (skip real provider resolve).
            eng._build = lambda: (setattr(eng, "client", _FakeAsyncClient(
                                      "SPEAK: hi")),
                                  setattr(eng, "model", "m"), True)[-1]
            # ★ Pin the multimodal config the job loop reads. The threaded job
            #   calls self._mm() → load_config() internally; relying on ambient
            #   config made this test depend on config.yaml having
            #   monitor_enabled=true (and a 5s tick) — brittle under the full
            #   suite. Patch _mm to the _MM_ON fixture (monitor_enabled=True,
            #   fast 1s tick) so the job actually evaluates and ticks quickly.
            eng._mm = lambda: _MM_ON
            eng.start()
            try:
                mons["m1"] = {"brief": "x", "enabled": True, "last_seen_ts": 0.0}
                eng.add_monitor("m1")
                # tick is 1s (via _MM_ON) + thread/async spin-up; 8s is ample.
                deadline = time.time() + 8.0
                while not spoken and time.time() < deadline:
                    time.sleep(0.1)
                self.assertTrue(spoken, "monitor job never emitted a SPEAK")
                # remove cancels the job
                eng.remove_monitor("m1")
                time.sleep(0.3)
                self.assertNotIn("m1", eng._jobs)
            finally:
                eng.stop()
                time.sleep(0.2)

    def test_once_job_retires_after_delivery(self):
        with (
            patch(
                "agent.multimodal.monitor_agent.append_event",
                return_value=("eid", "path"),
            ),
            patch("agent.multimodal.monitor_agent.set_status"),
            patch(
                "agent.multimodal.monitor_agent.read_status",
                return_value={"status": "done"},
            ),
        ):
            buf = _Buf([_Frame(1.0)])
            mons = {
                "m1": {
                    "brief": "x",
                    "enabled": True,
                    "trigger_mode": "once",
                    "last_seen_ts": 0.0,
                },
            }
            eng = MonitorEngine(
                buf,
                monitors_ref=mons,
                sid="s1",
                speak_cb=lambda *_args: True,
            )
            eng._build = lambda: (
                setattr(eng, "client", _FakeAsyncClient("SPEAK: hi")),
                setattr(eng, "model", "m"),
                True,
            )[-1]
            eng._mm = lambda: _MM_ON
            eng.start()
            try:
                self.assertTrue(eng.add_monitor("m1"))
                deadline = time.time() + 3.0
                while "m1" in eng._jobs and time.time() < deadline:
                    time.sleep(0.05)
                self.assertNotIn("m1", eng._jobs)
                self.assertFalse(mons["m1"]["enabled"])
                self.assertEqual(mons["m1"]["status"], "done")
            finally:
                eng.stop()


class TestHelpers(unittest.TestCase):
    def test_monitor_evidence_reencodes_a_small_thumbnail_not_the_model_image(self):
        original = BytesIO()
        Image.new("RGB", (1280, 720), (20, 80, 160)).save(original, format="JPEG")
        original_b64 = base64.b64encode(original.getvalue()).decode("ascii")

        evidence = build_monitor_evidence([
            _Frame(4.25, jpeg_b64=original_b64, source_type="camera"),
        ])

        assert evidence["input_count"] == 1
        assert evidence["shown_count"] == 1
        row = evidence["frames"][0]
        assert row["thumb_b64"] != original_b64
        assert row["source_type"] == "camera"
        with Image.open(BytesIO(base64.b64decode(row["thumb_b64"]))) as thumb:
            assert thumb.size == (320, 180)

    def test_aggregated_evidence_keeps_six_images_but_counts_all_model_inputs(self):
        left = {
            "input_count": 12,
            "frames": [{"ts": float(i), "thumb_b64": f"a{i}"} for i in range(6)],
        }
        right = {
            "input_count": 8,
            "frames": [{"ts": float(i + 6), "thumb_b64": f"b{i}"} for i in range(6)],
        }

        merged = merge_monitor_evidence(left, right)

        self.assertEqual(merged["input_count"], 20)
        self.assertEqual(merged["shown_count"], 6)
        self.assertEqual(merged["frames"][0]["ts"], 0.0)
        self.assertEqual(merged["frames"][-1]["ts"], 11.0)

    def test_monitor_prompt_requires_direct_evidence_and_exact_delivery(self):
        from agent.multimodal.monitor_agent import MONITOR_AGENT_SYSTEM

        self.assertIn("Inspect the whole frame batch frame by frame",
                      MONITOR_AGENT_SYSTEM)
        self.assertIn("clear, direct, and sufficient visual evidence",
                      MONITOR_AGENT_SYSTEM)
        self.assertIn("generic CBD skyline", MONITOR_AGENT_SYSTEM)
        self.assertIn("seeing the same general kind of object is not enough",
                      MONITOR_AGENT_SYSTEM)
        self.assertIn("A clearly identifiable subtype may satisfy its parent category",
                      MONITOR_AGENT_SYSTEM)
        self.assertIn("use that exact wording after a hit", MONITOR_AGENT_SYSTEM)
        self.assertIn("output only SILENT with no explanation",
                      MONITOR_AGENT_SYSTEM)
        self.assertIn("SPEAK: <fixed notification wording or brief confirmed fact>",
                      MONITOR_AGENT_SYSTEM)
        self.assertNotIn("only if the target never appears in the whole batch",
                         MONITOR_AGENT_SYSTEM)

    def test_default_heartbeat_is_short(self):
        eng = _make_engine(_Buf([]), {}, client=_FakeAsyncClient())
        self.assertEqual(eng._tick_sec({}), 1.0)

    def test_parse_monitor_verdict_protocol_and_legacy_json(self):
        # 新单行协议
        self.assertEqual(parse_monitor_verdict("SPEAK: 出现手机"),
                         (True, "出现手机"))
        self.assertEqual(parse_monitor_verdict("SILENT"), (False, ""))
        # 即使模型违约给 SILENT 加解释, parser 也丢弃解释。
        self.assertEqual(parse_monitor_verdict("SILENT: 没看到，这里有很长的解释"),
                         (False, ""))
        self.assertEqual(parse_monitor_verdict('"SPEAK: 门开了"'),
                         (True, "门开了"))
        self.assertEqual(parse_monitor_verdict("```text\nSPEAK: 灯亮了\n```"),
                         (True, "灯亮了"))

        # 滚动升级期兼容旧 JSON
        self.assertEqual(parse_monitor_verdict('{"status": true, "reason": "出现手机"}'),
                         (True, "出现手机"))
        self.assertEqual(parse_monitor_verdict('{"status": false, "reason": "无"}'),
                         (False, ""))
        # 代码围栏 / 前后杂字包裹的 JSON 仍能抠出
        self.assertTrue(parse_monitor_verdict('```json\n{"status": true, "reason": "x"}\n```')[0])
        self.assertTrue(parse_monitor_verdict('好的：{"status": true, "reason": "y"}')[0])
        # status 写成字符串 / 数字
        self.assertTrue(parse_monitor_verdict('{"status": "true", "reason": "z"}')[0])
        self.assertFalse(parse_monitor_verdict('{"status": 0, "reason": "无"}')[0])
        # 命中理由有硬上限，避免污染事件文件 / 通知。
        hit, reason = parse_monitor_verdict(f"SPEAK: {'x' * 500}")
        self.assertTrue(hit)
        self.assertEqual(len(reason), 160)
        # 无效输出 / 空 → 保守未命中
        self.assertEqual(parse_monitor_verdict("not a verdict"), (False, ""))
        self.assertEqual(parse_monitor_verdict(""), (False, ""))

    def test_should_flush(self):
        self.assertFalse(should_flush(0, [("t", "x")], 0, 100))       # no interval
        self.assertFalse(should_flush(10, [], 0, 100))                # empty buffer
        self.assertFalse(should_flush(10, [("t", "x")], 95, 100))     # not due
        self.assertTrue(should_flush(10, [("t", "x")], 80, 100))      # due

    def test_pick_window_subcap_and_congestion(self):
        from agent.multimodal.monitor_engine import (
            _MM_MONITOR_MAX_WINDOW as CAP,
            _MM_MONITOR_MAX_DOUBLINGS as MAXD,
        )
        DEFAULT_PERIOD = 4  # pick_window(base_period 默认) — 与生产默认对齐
        self.assertEqual(pick_window(10, 0)[0], 16)   # sub-cap → power of 2
        self.assertEqual(pick_window(200, 0), (CAP, 1, False))  # at cap (常量, 不硬编码)
        # 持续拥塞每满 base_period 轮翻一倍 stride; 达 MAXD 次翻倍 → tail_only。
        self.assertTrue(pick_window(200, DEFAULT_PERIOD * MAXD)[2])
        # 刚到上限但拥塞不久 → 仍抽样, 非 tail_only。
        self.assertFalse(pick_window(200, DEFAULT_PERIOD)[2])

    def test_sample_frames(self):
        frames = list(range(20))
        self.assertEqual(sample_frames(frames, 5, 1, False), [0, 4, 8, 12, 16])
        self.assertEqual(sample_frames(frames, 5, 1, True), [18, 19])


if __name__ == "__main__":
    unittest.main()
