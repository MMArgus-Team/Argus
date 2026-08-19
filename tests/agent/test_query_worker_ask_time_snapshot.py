"""QueryWorker ask-time frame snapshot regressions.

An explicit ``ask_ts`` is an anti-dirty-read boundary.  Empty snapshots must
remain empty even if newer frames arrive while the query waits or runs.
"""

from __future__ import annotations

import asyncio
import base64
import json
import threading
import unittest
from types import SimpleNamespace

from agent.multimodal._workers import ReactStep, RecallAgent, WatcherWorker
from agent.multimodal.watcher_engine import WatcherAgent


class _FrameBufferProbe:
    def __init__(self, *, eligible=None, latest=None, latest_ts=0.0):
        self.eligible = list(eligible or [])
        self.latest_frames = list(latest or [])
        self.latest_ts = latest_ts
        self.raw_all_le_calls: list[tuple[float, int]] = []
        self.all_le_calls: list[float] = []
        self.latest_calls: list[int] = []

    def raw_all_le(self, ts, n):
        self.raw_all_le_calls.append((float(ts), int(n)))
        return list(self.eligible)[-int(n):]

    def all_le(self, ts):
        self.all_le_calls.append(float(ts))
        return list(self.eligible)

    def latest(self, n):
        self.latest_calls.append(int(n))
        return list(self.latest_frames)[-n:]


def _prime_query_runtime(engine: WatcherAgent, loop) -> None:
    engine._stop = threading.Event()
    engine._healthy = True
    engine._ready = threading.Event()
    engine._ready.set()
    engine._query_lock = threading.RLock()
    engine._active_queries = {}
    engine._query_by_parent = {}
    engine._query_pending = set()
    engine._query_running = set()
    engine._query_max_concurrency = 1
    engine._query_max_pending = 2
    engine._query_semaphore = asyncio.Semaphore(1)


def _submit_and_capture(
    *, ask_ts, frame_buffer, emit_cb=None, frame_store=None, cfg=None,
    query_ocr_worker=None,
):
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def _run_loop():
        asyncio.set_event_loop(loop)
        ready.set()
        loop.run_forever()

    thread = threading.Thread(target=_run_loop, daemon=True)
    thread.start()
    assert ready.wait(2)

    captured = {}
    completed = threading.Event()

    class _Responder:
        async def _spawn_delegation(self, *, sink, on_event, **kwargs):
            captured.update(kwargs)

            async def _drive():
                await sink("ok")
                await on_event({"type": "answer_ready", "answer_full": "ok"})

            return asyncio.create_task(_drive())

    engine = WatcherAgent.__new__(WatcherAgent)
    engine._loop = loop
    engine.responder = _Responder()
    _prime_query_runtime(engine, loop)
    engine.cfg = cfg or SimpleNamespace(cont_recent_frames=8)
    engine.frame_buffer = frame_buffer
    engine.frame_store = frame_store
    engine.query_ocr_worker = query_ocr_worker
    engine._emit_cb = emit_cb
    engine._on_query_complete = lambda *_args: completed.set()

    try:
        task_id = engine.submit_query_async(
            "what was visible?",
            task_id="qry_snapshot",
            parent_user_message_id="turn_snapshot",
            ask_ts=ask_ts,
        )
        assert task_id == "qry_snapshot"
        assert completed.wait(3)
        return captured
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)
        loop.close()


class QueryWorkerAskTimeSnapshotTests(unittest.TestCase):
    @staticmethod
    def _jpeg_b64(label: str) -> str:
        return base64.b64encode(
            b"\xff\xd8" + label.encode("ascii") + b"\xff\xd9"
        ).decode("ascii")

    def test_explicit_ask_ts_freezes_last_three_eligible_frames(self):
        frames = [SimpleNamespace(ts=float(i)) for i in range(1, 5)]
        future = SimpleNamespace(ts=99.0)
        buf = _FrameBufferProbe(
            eligible=frames,
            latest=[future],
            latest_ts=future.ts,
        )

        captured = _submit_and_capture(ask_ts=4.0, frame_buffer=buf)

        self.assertEqual(captured["ask_ts"], 4.0)
        self.assertEqual(captured["ask_frames_override"], frames[-3:])
        self.assertEqual(buf.raw_all_le_calls, [(4.0, 3)])
        self.assertEqual(buf.all_le_calls, [])
        self.assertEqual(buf.latest_calls, [])

    def test_visual_evidence_mode_is_sync_grounding_without_reply_handoff(self):
        frames = [SimpleNamespace(ts=float(i)) for i in range(1, 5)]
        future = SimpleNamespace(ts=99.0)
        buf = _FrameBufferProbe(
            eligible=frames,
            latest=[future],
            latest_ts=future.ts,
        )
        loop = asyncio.new_event_loop()
        ready = threading.Event()

        def _run_loop():
            asyncio.set_event_loop(loop)
            ready.set()
            loop.run_forever()

        thread = threading.Thread(target=_run_loop, daemon=True)
        thread.start()
        self.assertTrue(ready.wait(2))
        captured = {}

        class _Responder:
            async def _spawn_delegation(self, *, sink, on_event, **kwargs):
                captured.update(kwargs)

                async def _drive():
                    answer = (
                        "Observed: PDF viewer.\n"
                        "Relevant text/identifiers: report.pdf, 3/18.\n"
                        "Grounding: the visible artifact is likely a PDF.\n"
                        "Uncertainty: full contents are not visible.\n"
                        "Recommended next capability: PDF reader."
                    )
                    await sink(answer)
                    await on_event({
                        "type": "answer_ready",
                        "answer_full": answer,
                    })

                return asyncio.create_task(_drive())

        engine = WatcherAgent.__new__(WatcherAgent)
        engine._loop = loop
        engine._thread = thread
        engine.responder = _Responder()
        _prime_query_runtime(engine, loop)
        engine.cfg = SimpleNamespace(
            cont_recent_frames=8,
            ocr_timeout_sec=0.1,
        )
        engine.frame_buffer = buf
        engine.query_ocr_worker = None

        try:
            result = engine.query_visual_evidence(
                "定位画面中的 PDF 文件名和页码",
                original_user_query="这个 PDF 讲了什么？",
                ask_ts=4.0,
                timeout=2.0,
            )
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=2)
            loop.close()

        self.assertTrue(result["ok"])
        self.assertEqual(result["n_frames"], 3)
        self.assertEqual(result["t_start"], 2.0)
        self.assertEqual(result["t_end"], 4.0)
        self.assertIn("report.pdf", result["evidence"])
        self.assertEqual(buf.raw_all_le_calls, [(4.0, 3)])
        self.assertEqual(buf.latest_calls, [])
        self.assertTrue(captured["query_worker_mode"])
        self.assertTrue(captured["evidence_only_mode"])
        self.assertEqual(captured["ask_frames_override"], frames[-3:])
        self.assertFalse(captured["router_enable_thinking"])
        self.assertEqual(engine._query_pending, set())
        self.assertEqual(engine._query_running, set())

    def test_query_ocr_receives_and_returns_evidence_for_exact_snapshot(self):
        frames = [
            SimpleNamespace(
                ts=float(i),
                jpeg_b64=self._jpeg_b64(f"source-{i}"),
                source_type="camera",
            )
            for i in range(1, 5)
        ]

        class _QueryOCR:
            def __init__(self):
                self.calls = []

            async def collect_query_evidence(self, got, *, ask_ts):
                self.calls.append((list(got), float(ask_ts)))
                return [{
                    "frame_ts": got[-1].ts,
                    "raw_text": "东方树叶 茉莉花茶",
                    "evidence_source": "synchronous_camera_ocr",
                }]

        query_ocr = _QueryOCR()
        emitted = []
        captured = _submit_and_capture(
            ask_ts=4.0,
            frame_buffer=_FrameBufferProbe(eligible=frames, latest_ts=4.0),
            query_ocr_worker=query_ocr,
            emit_cb=lambda event, payload: emitted.append((event, payload)),
        )

        self.assertEqual(query_ocr.calls, [(frames[-3:], 4.0)])
        self.assertEqual(captured["ask_frames_override"], frames[-3:])
        self.assertEqual(
            captured["query_ocr_evidence"][0]["raw_text"],
            "东方树叶 茉莉花茶",
        )
        phases = [
            payload.get("phase") for event, payload in emitted
            if event == "multimodal.trajectory"
        ]
        self.assertLess(phases.index("started"), phases.index("ocr_evidence"))
        ocr_event = next(
            payload for event, payload in emitted
            if event == "multimodal.trajectory"
            and payload.get("phase") == "ocr_evidence"
        )
        self.assertEqual(ocr_event["evidence_state"], "available")
        self.assertEqual(ocr_event["record_count"], 1)
        self.assertEqual(ocr_event["evidence"], [{
            "frame_ts": 4.0,
            "source_type": "",
            "evidence_source": "synchronous_camera_ocr",
            "app": "",
            "window_title": "",
            "raw_text": "东方树叶 茉莉花茶",
        }])
        self.assertNotIn("jpeg_b64", str(ocr_event["evidence"]))
        self.assertNotIn("ocr_blocks", str(ocr_event["evidence"]))

    def test_query_ocr_trajectory_has_explicit_skipped_empty_and_error_states(self):
        frame = SimpleNamespace(
            ts=1.0,
            jpeg_b64=self._jpeg_b64("source"),
            source_type="camera",
        )

        def _ocr_event(worker):
            emitted = []
            _submit_and_capture(
                ask_ts=1.0,
                frame_buffer=_FrameBufferProbe(
                    eligible=[frame], latest_ts=1.0),
                query_ocr_worker=worker,
                emit_cb=lambda event, payload: emitted.append((event, payload)),
            )
            return next(
                payload for event, payload in emitted
                if event == "multimodal.trajectory"
                and payload.get("phase") == "ocr_evidence"
            )

        skipped = _ocr_event(None)
        self.assertEqual(skipped["evidence_state"], "skipped")
        self.assertEqual(skipped["reason"], "ocr_unavailable")
        self.assertEqual(skipped["evidence"], [])

        class _EmptyOCR:
            async def collect_query_evidence(self, *_args, **_kwargs):
                return []

        empty = _ocr_event(_EmptyOCR())
        self.assertEqual(empty["evidence_state"], "empty")
        self.assertEqual(empty["reason"], "")
        self.assertEqual(empty["evidence"], [])

        class _BrokenOCR:
            async def collect_query_evidence(self, *_args, **_kwargs):
                raise RuntimeError("secret exception detail must not leak")

        error = _ocr_event(_BrokenOCR())
        self.assertEqual(error["status"], "error")
        self.assertEqual(error["evidence_state"], "error")
        self.assertEqual(error["reason"], "collection_failed")
        self.assertEqual(error["evidence"], [])
        self.assertNotIn("secret exception detail", str(error))

    def test_empty_explicit_snapshot_never_substitutes_later_frame(self):
        later = SimpleNamespace(ts=10.0)
        buf = _FrameBufferProbe(
            eligible=[],
            latest=[later],
            latest_ts=later.ts,
        )

        captured = _submit_and_capture(ask_ts=4.0, frame_buffer=buf)

        self.assertEqual(captured["ask_ts"], 4.0)
        self.assertEqual(captured["ask_frames_override"], [])
        self.assertEqual(buf.raw_all_le_calls, [(4.0, 3)])
        self.assertEqual(buf.all_le_calls, [])
        self.assertEqual(
            buf.latest_calls,
            [],
            "a frame arriving after ask_ts must not repair an empty snapshot",
        )

    def test_started_trajectory_renders_exact_frozen_snapshot_thumbnails(self):
        frames = [
            SimpleNamespace(
                ts=float(i),
                jpeg_b64=self._jpeg_b64(f"source-{i}"),
                source_type="camera" if i % 2 else "screen",
            )
            for i in range(1, 5)
        ]
        future = SimpleNamespace(
            ts=99.0,
            jpeg_b64=self._jpeg_b64("future"),
            source_type="camera",
        )
        buf = _FrameBufferProbe(
            eligible=frames,
            latest=[future],
            latest_ts=future.ts,
        )
        thumbnail_calls = []

        class _FrameStore:
            def thumbnail_b64(self, jpeg_b64, *, max_side, quality):
                thumbnail_calls.append((jpeg_b64, max_side, quality))
                return QueryWorkerAskTimeSnapshotTests._jpeg_b64(
                    f"thumb-{len(thumbnail_calls)}"
                )

        emitted = []
        captured = _submit_and_capture(
            ask_ts=4.0,
            frame_buffer=buf,
            frame_store=_FrameStore(),
            emit_cb=lambda event, payload: emitted.append((event, payload)),
            cfg=SimpleNamespace(
                cont_recent_frames=8,
                ui_event_thumb_max_side=321,
                ui_event_thumb_jpeg_quality=54,
            ),
        )

        started = next(
            payload for event, payload in emitted
            if event == "multimodal.trajectory"
            and payload.get("phase") == "started"
        )
        frozen = frames[-3:]
        self.assertEqual(captured["ask_frames_override"], frozen)
        self.assertEqual(
            [preview["ts"] for preview in started["frames"]],
            [frame.ts for frame in frozen],
        )
        self.assertEqual(
            [preview["source_type"] for preview in started["frames"]],
            [frame.source_type for frame in frozen],
        )
        self.assertEqual(
            [preview["jpeg_b64"] for preview in started["frames"]],
            [self._jpeg_b64(f"thumb-{i}") for i in range(1, 4)],
        )
        self.assertTrue(all(
            "frame_id" not in preview for preview in started["frames"]
        ))
        self.assertEqual(
            thumbnail_calls,
            [(frame.jpeg_b64, 321, 54) for frame in frozen],
        )
        self.assertNotIn(
            future.jpeg_b64,
            [call[0] for call in thumbnail_calls],
        )
        self.assertEqual(buf.raw_all_le_calls, [(4.0, 3)])
        self.assertEqual(buf.all_le_calls, [])
        self.assertEqual(buf.latest_calls, [])

    def test_started_trajectory_keeps_explicit_empty_snapshot_empty(self):
        later = SimpleNamespace(
            ts=10.0,
            jpeg_b64=self._jpeg_b64("later"),
            source_type="camera",
        )
        buf = _FrameBufferProbe(
            eligible=[],
            latest=[later],
            latest_ts=later.ts,
        )
        emitted = []
        captured = _submit_and_capture(
            ask_ts=4.0,
            frame_buffer=buf,
            emit_cb=lambda event, payload: emitted.append((event, payload)),
        )

        started = next(
            payload for event, payload in emitted
            if event == "multimodal.trajectory"
            and payload.get("phase") == "started"
        )
        self.assertEqual(captured["ask_frames_override"], [])
        self.assertEqual(started["n_frames"], 0)
        self.assertEqual(started["frames"], [])
        self.assertEqual(buf.latest_calls, [])

    def test_preview_failure_uses_only_a_complete_bounded_jpeg_fallback(self):
        small = self._jpeg_b64("small-original")
        oversized = base64.b64encode(
            b"\xff\xd8" + (b"x" * 400_000) + b"\xff\xd9"
        ).decode("ascii")
        invalid = base64.b64encode(b"not-a-jpeg").decode("ascii")
        frames = [
            SimpleNamespace(ts=1.0, jpeg_b64=small, source_type="screen"),
            SimpleNamespace(ts=2.0, jpeg_b64=oversized, source_type="screen"),
            SimpleNamespace(ts=3.0, jpeg_b64=invalid, source_type="screen"),
        ]

        class _FailingFrameStore:
            def thumbnail_b64(self, *_args, **_kwargs):
                raise RuntimeError("thumbnail failed")

        emitted = []
        _submit_and_capture(
            ask_ts=3.0,
            frame_buffer=_FrameBufferProbe(eligible=frames, latest_ts=3.0),
            frame_store=_FailingFrameStore(),
            emit_cb=lambda event, payload: emitted.append((event, payload)),
        )
        started = next(
            payload for event, payload in emitted
            if event == "multimodal.trajectory"
            and payload.get("phase") == "started"
        )

        self.assertEqual(started["n_frames"], 3)
        self.assertEqual(started["frames"], [{
            "ts": 1.0,
            "source_type": "screen",
            "jpeg_b64": small,
        }])
        self.assertLess(len(started["frames"][0]["jpeg_b64"]), 500_000)

    def test_no_event_sink_skips_thumbnail_work(self):
        frame = SimpleNamespace(
            ts=1.0,
            jpeg_b64=self._jpeg_b64("source"),
            source_type="camera",
        )

        class _FrameStore:
            def __init__(self):
                self.calls = 0

            def thumbnail_b64(self, *_args, **_kwargs):
                self.calls += 1
                return QueryWorkerAskTimeSnapshotTests._jpeg_b64("thumb")

        store = _FrameStore()
        _submit_and_capture(
            ask_ts=1.0,
            frame_buffer=_FrameBufferProbe(eligible=[frame], latest_ts=1.0),
            frame_store=store,
            emit_cb=None,
        )
        self.assertEqual(store.calls, 0)

    def test_missing_ask_ts_keeps_live_latest_frame_behavior(self):
        latest = [SimpleNamespace(ts=7.0), SimpleNamespace(ts=8.0)]
        buf = _FrameBufferProbe(
            eligible=[SimpleNamespace(ts=1.0)],
            latest=latest,
            latest_ts=8.0,
        )

        captured = _submit_and_capture(ask_ts=None, frame_buffer=buf)

        self.assertEqual(captured["ask_ts"], 8.0)
        self.assertEqual(captured["ask_frames_override"], latest)
        self.assertEqual(buf.all_le_calls, [])
        self.assertEqual(buf.latest_calls, [3])

    def test_missing_ask_ts_and_empty_buffer_keeps_inner_live_fallback(self):
        buf = _FrameBufferProbe(eligible=[], latest=[], latest_ts=0.0)

        captured = _submit_and_capture(ask_ts=None, frame_buffer=buf)

        self.assertIsNone(captured["ask_ts"])
        self.assertIsNone(captured["ask_frames_override"])
        self.assertEqual(buf.all_le_calls, [])
        self.assertEqual(buf.latest_calls, [3])

    def test_explicit_empty_override_does_not_trigger_inner_fallback(self):
        async def _case():
            later = SimpleNamespace(ts=10.0)
            buf = _FrameBufferProbe(
                eligible=[],
                latest=[later],
                latest_ts=later.ts,
            )
            worker = WatcherWorker.__new__(WatcherWorker)
            worker.cfg = SimpleNamespace(
                cont_recent_frames=3,
                search_recent_frames=3,
                react_max_rounds=1,
                cont_now_frames=3,
            )
            worker.buf = buf
            worker.inflight = {}
            worker.frame_store = None
            worker.store = None
            worker.conversation = SimpleNamespace(append=_async_noop)

            seen_batches = []

            async def _react_step(**kwargs):
                seen_batches.append(list(kwargs["batch_frames"]))
                # Force the fallback synthesis path; it must also remain bound
                # to the empty ask-time snapshot.
                return ReactStep()

            answer_calls = []

            async def _answer(**kwargs):
                answer_calls.append(kwargs)
                return "ask-time view unavailable", 0.0, 0

            worker.react_step = _react_step
            worker.answer = _answer

            streamed = []

            async def _sink(text):
                streamed.append(text)

            task = await worker._spawn_delegation(
                task_instruction="what is visible?",
                prelude="",
                sink=_sink,
                on_event=None,
                ask_ts=4.0,
                ask_frames_override=[],
                query_worker_mode=True,
            )
            await task

            self.assertEqual(seen_batches, [[]])
            self.assertEqual(answer_calls[0]["ask_frames"], [])
            self.assertEqual(answer_calls[0]["now_frames"], [])
            self.assertEqual(streamed, ["ask-time view unavailable"])
            self.assertEqual(buf.all_le_calls, [])
            self.assertEqual(buf.latest_calls, [])

        asyncio.run(_case())

    def test_inner_worker_without_ask_ts_still_samples_latest(self):
        async def _case():
            latest = [SimpleNamespace(ts=7.0), SimpleNamespace(ts=8.0)]
            buf = _FrameBufferProbe(
                eligible=[],
                latest=latest,
                latest_ts=8.0,
            )
            worker = WatcherWorker.__new__(WatcherWorker)
            worker.cfg = SimpleNamespace(
                cont_recent_frames=3,
                search_recent_frames=3,
                react_max_rounds=1,
                cont_now_frames=3,
            )
            worker.buf = buf
            worker.inflight = {}
            worker.frame_store = None
            worker.store = None
            worker.conversation = SimpleNamespace(append=_async_noop)

            seen = {}

            async def _react_step(**kwargs):
                seen.update(kwargs)
                return ReactStep(answer="live answer")

            worker.react_step = _react_step

            task = await worker._spawn_delegation(
                task_instruction="what is visible?",
                prelude="",
                sink=_async_noop,
                on_event=None,
                ask_ts=None,
                ask_frames_override=None,
                query_worker_mode=True,
            )
            await task

            self.assertEqual(seen["batch_frames"], latest)
            self.assertEqual(seen["ask_ts"], 8.0)
            self.assertEqual(buf.all_le_calls, [])
            self.assertEqual(buf.latest_calls, [3, 3])

        asyncio.run(_case())

    def test_evidence_result_is_not_persisted_as_queryworker_chat_answer(self):
        async def _case():
            frame = SimpleNamespace(ts=4.0)
            buf = _FrameBufferProbe(
                eligible=[frame], latest=[frame], latest_ts=4.0)
            appended = []

            async def _append(*args, **kwargs):
                appended.append((args, kwargs))

            worker = WatcherWorker.__new__(WatcherWorker)
            worker.cfg = SimpleNamespace(
                cont_recent_frames=3,
                search_recent_frames=3,
                react_max_rounds=1,
                cont_now_frames=3,
            )
            worker.buf = buf
            worker.inflight = {}
            worker.frame_store = None
            worker.store = None
            worker.conversation = SimpleNamespace(append=_append)

            async def _react_step(**kwargs):
                self.assertTrue(kwargs["evidence_only_mode"])
                return ReactStep(answer="Observed: report.pdf")

            worker.react_step = _react_step
            task = await worker._spawn_delegation(
                task_instruction="ground the PDF",
                prelude="",
                sink=_async_noop,
                on_event=None,
                ask_ts=4.0,
                ask_frames_override=[frame],
                query_worker_mode=True,
                evidence_only_mode=True,
            )
            await task

            self.assertEqual(appended, [])

        asyncio.run(_case())

    def test_timestamp_helpers_never_return_future_frames(self):
        later = SimpleNamespace(ts=10.0)
        buf = _FrameBufferProbe(
            eligible=[],
            latest=[later],
            latest_ts=later.ts,
        )
        worker = WatcherWorker.__new__(WatcherWorker)
        worker.buf = buf
        recall = RecallAgent.__new__(RecallAgent)
        recall.buf = buf

        self.assertEqual(worker._get_frames_up_to(3, 4.0), [])
        self.assertEqual(worker._frames_up_to(3, 4.0), [])
        self.assertEqual(recall._frames_up_to(3, 4.0), [])
        self.assertEqual(buf.all_le_calls, [4.0, 4.0, 4.0])
        self.assertEqual(buf.latest_calls, [])

    def test_empty_query_worker_prompt_forbids_newer_frame_substitution(self):
        async def _case():
            captured = {}

            class _Completions:
                async def create(self, **kwargs):
                    captured.update(kwargs)
                    return _EmptyAsyncStream()

            snapshot = SimpleNamespace(
                render_full=lambda *args, **kwargs: "(none)",
            )
            worker = WatcherWorker.__new__(WatcherWorker)
            worker.cfg = SimpleNamespace(
                model="test-model",
                cont_recall_frames_max=3,
                react_max_rounds=1,
                react_round_max_tokens=64,
                react_temperature=0.0,
            )
            worker.client = SimpleNamespace(
                chat=SimpleNamespace(completions=_Completions()))
            worker.mem = SimpleNamespace(
                get_recent_macros=lambda *_args, **_kwargs: [],
                get_recent_entities=lambda *_args, **_kwargs: [],
            )
            worker.store = SimpleNamespace(snapshot=lambda: snapshot)
            worker.frame_store = None
            worker.recorder = None

            await worker.react_step(
                task_instruction="what is visible?",
                ask_ts=4.0,
                round_idx=0,
                search_log=[],
                recall_log=[],
                batch_frames=[],
                query_worker_mode=True,
            )

            user_parts = captured["messages"][1]["content"]
            prompt = user_parts[0]["text"]
            self.assertIn("No frame was captured at or before ask_ts", prompt)
            self.assertIn("Never substitute a newer/latest frame", prompt)
            self.assertIn("ask-time view is unavailable", prompt)
            self.assertFalse(any(
                part.get("type") == "image_url" for part in user_parts
            ))
            lifecycle_tools = {
                "finish_watching", "mark_completion_candidate",
            }
            self.assertTrue(lifecycle_tools.isdisjoint(
                {t["function"]["name"] for t in captured["tools"]}
            ))

        asyncio.run(_case())

    def test_visual_evidence_react_mode_omits_all_worker_tools(self):
        async def _case():
            captured = {}

            class _Completions:
                async def create(self, **kwargs):
                    captured.update(kwargs)
                    return _EmptyAsyncStream()

            snapshot = SimpleNamespace(
                render_full=lambda *args, **kwargs: "(none)",
            )
            worker = WatcherWorker.__new__(WatcherWorker)
            worker.cfg = SimpleNamespace(
                model="test-model",
                cont_recall_frames_max=3,
                react_max_rounds=1,
                react_round_max_tokens=64,
                react_temperature=0.0,
            )
            worker.client = SimpleNamespace(
                chat=SimpleNamespace(completions=_Completions()))
            worker.mem = SimpleNamespace(
                get_recent_macros=lambda *_args, **_kwargs: [],
                get_recent_entities=lambda *_args, **_kwargs: [],
            )
            worker.store = SimpleNamespace(snapshot=lambda: snapshot)
            worker.frame_store = None
            worker.recorder = None

            await worker.react_step(
                task_instruction="定位屏幕里的 PDF，供 Main Agent 后续读取",
                ask_ts=4.0,
                round_idx=0,
                search_log=[],
                recall_log=[],
                batch_frames=[],
                query_worker_mode=True,
                evidence_only_mode=True,
            )

            prompt = captured["messages"][1]["content"][0]["text"]
            system_prompt = captured["messages"][0]["content"]
            self.assertIn("visual-evidence mode", prompt)
            self.assertIn("Do NOT answer the user's full task", prompt)
            self.assertIn("PDF, file, terminal", prompt)
            self.assertIn("no tools", system_prompt)
            self.assertNotIn("Available tools", system_prompt)
            self.assertIn(
                "not provided in visual-evidence mode", prompt)
            self.assertNotIn("tools", captured)
            self.assertNotIn("tool_choice", captured)

        asyncio.run(_case())

    def test_continuous_watcher_parses_visual_completion_tool(self):
        async def _case():
            class _FinishStream:
                def __init__(self):
                    self._done = False

                def __aiter__(self):
                    return self

                async def __anext__(self):
                    if self._done:
                        raise StopAsyncIteration
                    self._done = True
                    tool_call = SimpleNamespace(
                        index=0,
                        id="finish_1",
                        function=SimpleNamespace(
                            name="finish_watching",
                            arguments=json.dumps({
                                "reason": "the player shows its ended state",
                                "final_observation": "The video visibly ended.",
                            }),
                        ),
                    )
                    delta = SimpleNamespace(
                        reasoning_content=None,
                        reasoning=None,
                        content=None,
                        tool_calls=[tool_call],
                    )
                    return SimpleNamespace(
                        choices=[SimpleNamespace(delta=delta)])

            class _Completions:
                async def create(self, **_kwargs):
                    return _FinishStream()

            snapshot = SimpleNamespace(render_full=lambda *args, **kwargs: "(none)")
            worker = WatcherWorker.__new__(WatcherWorker)
            worker.cfg = SimpleNamespace(
                model="test-model",
                cont_recall_frames_max=3,
                react_max_rounds=1,
                react_round_max_tokens=64,
                react_temperature=0.0,
            )
            worker.client = SimpleNamespace(
                chat=SimpleNamespace(completions=_Completions()))
            worker.mem = SimpleNamespace(
                get_recent_macros=lambda *_args, **_kwargs: [],
                get_recent_entities=lambda *_args, **_kwargs: [],
            )
            worker.store = SimpleNamespace(snapshot=lambda: snapshot)
            worker.frame_store = None
            worker.recorder = None

            step = await worker.react_step(
                task_instruction="Watch until the embedded video ends.",
                ask_ts=4.0,
                round_idx=0,
                search_log=[],
                recall_log=[],
                batch_frames=[],
                query_worker_mode=False,
            )

            self.assertTrue(step.task_complete)
            self.assertEqual(
                step.completion_reason,
                "the player shows its ended state",
            )
            self.assertEqual(step.answer, "The video visibly ended.")
            self.assertEqual(step.tool_calls, [])

        asyncio.run(_case())

    def test_continuous_watcher_caps_total_request_images_below_provider_limit(self):
        async def _case():
            captured = {}

            class _Completions:
                async def create(self, **kwargs):
                    captured.update(kwargs)
                    return _EmptyAsyncStream()

            snapshot = SimpleNamespace(
                render_full=lambda *args, **kwargs: "(none)")
            worker = WatcherWorker.__new__(WatcherWorker)
            worker.cfg = SimpleNamespace(
                model="test-model",
                cont_recall_frames_max=4,
                watch_request_max_images=48,
                react_max_rounds=1,
                react_round_max_tokens=64,
                react_temperature=0.0,
            )
            worker.client = SimpleNamespace(
                chat=SimpleNamespace(completions=_Completions()))
            worker.mem = SimpleNamespace(
                get_recent_macros=lambda *_args, **_kwargs: [],
                get_recent_entities=lambda *_args, **_kwargs: [],
            )
            worker.store = SimpleNamespace(snapshot=lambda: snapshot)
            worker.frame_store = None
            worker.recorder = None

            from agent.multimodal._memory import Frame
            frames = [
                Frame(ts=float(i), jpeg_b64=f"frame-{i}")
                for i in range(60)
            ]
            await worker.react_step(
                task_instruction="Watch this video until it ends.",
                ask_ts=59.0,
                round_idx=0,
                search_log=[],
                recall_log=[],
                batch_frames=frames,
                query_worker_mode=False,
                static_tail_check=True,
            )

            user_parts = captured["messages"][1]["content"]
            image_parts = [
                part for part in user_parts
                if part.get("type") == "image_url"
            ]
            self.assertEqual(len(image_parts), 48)
            self.assertIn("frame-59", image_parts[-1]["image_url"]["url"])
            tool_names = {
                tool["function"]["name"] for tool in captured["tools"]
            }
            self.assertIn("finish_watching", tool_names)
            self.assertNotIn("mark_completion_candidate", tool_names)
            prompt = user_parts[0]["text"]
            self.assertIn("End-state check", prompt)

        asyncio.run(_case())

    def test_continuous_watcher_parses_ambiguous_completion_candidate(self):
        async def _case():
            class _CandidateStream:
                def __init__(self):
                    self._done = False

                def __aiter__(self):
                    return self

                async def __anext__(self):
                    if self._done:
                        raise StopAsyncIteration
                    self._done = True
                    tool_call = SimpleNamespace(
                        index=0,
                        id="candidate_1",
                        function=SimpleNamespace(
                            name="mark_completion_candidate",
                            arguments=json.dumps({
                                "reason": (
                                    "conclusion chapter and closing question, "
                                    "but a spinner is still visible"
                                ),
                                "final_observation": (
                                    "The presenter closes by asking for agreement."
                                ),
                            }),
                        ),
                    )
                    delta = SimpleNamespace(
                        reasoning_content=None,
                        reasoning=None,
                        content=None,
                        tool_calls=[tool_call],
                    )
                    return SimpleNamespace(
                        choices=[SimpleNamespace(delta=delta)])

            class _Completions:
                async def create(self, **_kwargs):
                    return _CandidateStream()

            snapshot = SimpleNamespace(render_full=lambda *args, **kwargs: "(none)")
            worker = WatcherWorker.__new__(WatcherWorker)
            worker.cfg = SimpleNamespace(
                model="test-model",
                cont_recall_frames_max=3,
                react_max_rounds=1,
                react_round_max_tokens=64,
                react_temperature=0.0,
            )
            worker.client = SimpleNamespace(
                chat=SimpleNamespace(completions=_Completions()))
            worker.mem = SimpleNamespace(
                get_recent_macros=lambda *_args, **_kwargs: [],
                get_recent_entities=lambda *_args, **_kwargs: [],
            )
            worker.store = SimpleNamespace(snapshot=lambda: snapshot)
            worker.frame_store = None
            worker.recorder = None

            step = await worker.react_step(
                task_instruction="Watch until the embedded video ends.",
                ask_ts=4.0,
                round_idx=0,
                search_log=[],
                recall_log=[],
                batch_frames=[],
                query_worker_mode=False,
            )

            self.assertFalse(step.task_complete)
            self.assertTrue(step.completion_candidate)
            self.assertIn("conclusion chapter", step.completion_candidate_reason)
            self.assertEqual(
                step.answer,
                "The presenter closes by asking for agreement.",
            )
            self.assertEqual(step.tool_calls, [])
            self.assertEqual(step.recall_tasks, [])

        asyncio.run(_case())

    def test_visual_completion_confirmation_uses_raw_frames_and_threshold(self):
        async def _case():
            captured = {}

            class _Completions:
                async def create(self, **kwargs):
                    captured.update(kwargs)
                    message = SimpleNamespace(content=json.dumps({
                        "ended": True,
                        "confidence": 0.91,
                        "reason": "replay control and completed progress bar",
                        "final_observation": "The embedded video has ended.",
                    }))
                    return SimpleNamespace(choices=[SimpleNamespace(
                        message=message, finish_reason="stop")])

            worker = WatcherWorker.__new__(WatcherWorker)
            worker.cfg = SimpleNamespace(
                model="test-model",
                watch_completion_confirm_min_confidence=0.8,
            )
            worker.client = SimpleNamespace(
                chat=SimpleNamespace(completions=_Completions()))
            worker.recorder = None

            from agent.multimodal._memory import Frame
            frames = [
                Frame(ts=8.0, jpeg_b64="frame-a"),
                Frame(ts=12.0, jpeg_b64="frame-b"),
            ]
            confirmed, confidence, reason, observation = (
                await worker.confirm_visual_completion(
                    task_instruction="Watch until the video ends.",
                    candidate_reason="The conclusion ended on a replay-like UI.",
                    last_segment_report="The presenter gave the closing line.",
                    idle_sec=8.0,
                    attempt=2,
                    total_idle_sec=28.0,
                    prior_confirmation_reason="spinner may be buffering",
                    frames=frames,
                )
            )

            self.assertTrue(confirmed)
            self.assertEqual(confidence, 0.91)
            self.assertIn("replay control", reason)
            self.assertEqual(observation, "The embedded video has ended.")
            user_parts = captured["messages"][1]["content"]
            self.assertEqual(
                sum(part.get("type") == "image_url" for part in user_parts), 2)
            prompt_text = user_parts[0]["text"]
            self.assertIn("FOLLOW-UP extended-stall", prompt_text)
            self.assertIn("28.0s", prompt_text)
            self.assertIn("spinner may be buffering", prompt_text)
            self.assertFalse(captured["stream"])

        asyncio.run(_case())

    def test_visual_completion_accepts_likely_ending_on_first_check(self):
        """A suspected ending must finish promptly instead of requiring
        explicit replay UI or an exact elapsed/duration match."""
        async def _case():
            captured = {}

            class _Completions:
                async def create(self, **kwargs):
                    captured.update(kwargs)
                    message = SimpleNamespace(content=json.dumps({
                        "ended": True,
                        "confidence": 0.65,
                        "reason": (
                            "closing speech and a stable terminal-looking frame; "
                            "no playback resume is visible"
                        ),
                        "final_observation": "The likely ending remained static.",
                    }))
                    return SimpleNamespace(choices=[SimpleNamespace(
                        message=message, finish_reason="stop")])

            worker = WatcherWorker.__new__(WatcherWorker)
            worker.cfg = SimpleNamespace(
                model="test-model",
                watch_completion_confirm_min_confidence=0.6,
            )
            worker.client = SimpleNamespace(
                chat=SimpleNamespace(completions=_Completions()))
            worker.recorder = None

            from agent.multimodal._memory import Frame
            result = await worker.confirm_visual_completion(
                task_instruction="Watch until the video ends.",
                candidate_reason="The presenter gave a closing line.",
                last_segment_report="A closing interaction was visible.",
                idle_sec=2.0,
                attempt=1,
                total_idle_sec=2.0,
                frames=[
                    Frame(ts=8.0, jpeg_b64="frame-a"),
                    Frame(ts=10.0, jpeg_b64="frame-b"),
                ],
            )

            self.assertTrue(result[0])
            self.assertEqual(result[1], 0.65)
            system_prompt = " ".join(
                captured["messages"][0]["content"].split())
            self.assertIn(
                "Timely completion is the product priority", system_prompt)
            self.assertIn(
                "more consistent with completion than continuation",
                system_prompt,
            )
            self.assertFalse(captured["stream"])

        asyncio.run(_case())

    def test_query_worker_prompt_completes_mixed_visual_external_questions(self):
        async def _case():
            captured = {}

            class _Completions:
                async def create(self, **kwargs):
                    captured.update(kwargs)
                    return _EmptyAsyncStream()

            snapshot = SimpleNamespace(render_full=lambda *args, **kwargs: "(none)")
            worker = WatcherWorker.__new__(WatcherWorker)
            worker.cfg = SimpleNamespace(
                model="test-model",
                cont_recall_frames_max=3,
                react_max_rounds=1,
                react_round_max_tokens=64,
                react_temperature=0.0,
            )
            worker.client = SimpleNamespace(
                chat=SimpleNamespace(completions=_Completions()))
            worker.mem = SimpleNamespace(
                get_recent_macros=lambda *_args, **_kwargs: [],
                get_recent_entities=lambda *_args, **_kwargs: [],
            )
            worker.store = SimpleNamespace(snapshot=lambda: snapshot)
            worker.frame_store = None
            worker.recorder = None

            await worker.react_step(
                task_instruction="我拿的饮料是啥，多少钱一瓶？",
                ask_ts=13.0,
                round_idx=0,
                search_log=[],
                recall_log=[],
                batch_frames=[],
                query_worker_mode=True,
            )

            prompt = captured["messages"][1]["content"][0]["text"]
            self.assertIn("directly useful prior-QA context", prompt)
            self.assertIn("task planning", prompt)
            self.assertIn("preserving uncertainty", prompt)
            self.assertIn("authoritative original user question", prompt)
            self.assertIn("Account for every requested part", prompt)
            self.assertIn("mixed visual-and-outside-fact questions", prompt)
            self.assertIn("then call text_search", prompt)
            self.assertIn("missing printed/on-screen price is not a reason to stop", prompt)
            self.assertIn("must not silently change the visually bound entity", prompt)

            search_schema = next(
                tool["function"]
                for tool in captured["tools"]
                if tool["function"]["name"] == "text_search"
            )
            self.assertIn("retail price", search_schema["description"])
            self.assertIn("does not answer", search_schema["description"])
            self.assertIn(
                "do not use web search to substitute for missing visual evidence",
                search_schema["description"],
            )

        asyncio.run(_case())


async def _async_noop(*_args, **_kwargs):
    return None


class _EmptyAsyncStream:
    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


if __name__ == "__main__":
    unittest.main()
