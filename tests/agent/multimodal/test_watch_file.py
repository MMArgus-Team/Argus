"""Regression tests for the live-watcher analyse-file + continuous loop.

Covers:
  * watch_file: header/round/dedup/summary-read/round-count.
  * WatcherAgent continuous loop: multi-batch with cross-round sub-query dedup,
    stop_delegation ends a run. (The watcher has ONE mode — no query-type
    classification.)

Pure unittest (no pytest / no network / no real LLM) so it runs in CI and in a
restricted sandbox.
"""
import asyncio
import pathlib
import tempfile
import threading
import unittest

import hermes_constants


def _use_temp_home():
    tmp = pathlib.Path(tempfile.mkdtemp())
    hermes_constants.get_hermes_home = lambda: tmp
    return tmp


class TestDeepsearchFile(unittest.TestCase):
    def setUp(self):
        self._orig = hermes_constants.get_hermes_home
        _use_temp_home()

    def tearDown(self):
        hermes_constants.get_hermes_home = self._orig

    def test_header_round_dedup_summary(self):
        import agent.multimodal.watch_file as D
        D.init_file("r1", query="分析这人表现")
        D.append_round("r1", round_idx=1, frame_range=(0.0, 42.5),
                       sub_queries=["识别动作", "情绪状态"], findings="动作流畅")
        D.append_round("r1", round_idx=2, frame_range=(42.5, 90.0),
                       sub_queries=["识别动作", "新手势"], findings="出现挥手")
        seen = D.read_seen_subqueries("r1")
        self.assertIn("识别动作", seen)
        self.assertIn("情绪状态", seen)
        self.assertIn("新手势", seen)
        self.assertEqual(D.count_rounds("r1"), 2)
        full = D.read_all("r1")
        # User-facing "第 N 次分析", not internal "Round N".
        self.assertIn("第 1 次分析", full)
        self.assertIn("第 2 次分析", full)
        self.assertNotIn("Round", full)
        self.assertIn("分析的视频时段", full)
        D.mark_finished("r1", rounds=2, summary_preview="表现良好")
        self.assertIn("完成", D.read_all("r1"))

    def test_drop_last_incomplete_round(self):
        # req ⑥: reopen cleanup drops the last unfinished segment (no ## 完成).
        import agent.multimodal.watch_file as D
        D.init_file("dr1", query="q")
        D.append_round("dr1", round_idx=1, frame_range=(0.0, 10.0),
                       sub_queries=["a"], findings="第一段完整解读")
        D.append_round("dr1", round_idx=2, frame_range=(10.0, 20.0),
                       sub_queries=["b"], findings="第二段完整解读")
        # Simulate a crash mid-round-3: header written, body missing.
        p = D.file_path("dr1")
        with p.open("a", encoding="utf-8") as f:
            f.write("\n## 第 3 次分析  (2026-07-17T00:00:00+08:00)\n- 分析的视频时段: 20s\n")
        dropped = D.drop_last_incomplete_round("dr1")
        self.assertTrue(dropped)
        full = D.read_all("dr1")
        self.assertIn("第一段完整解读", full)
        self.assertIn("第二段完整解读", full)
        self.assertNotIn("第 3 次分析", full)  # incomplete tail removed
        self.assertIn("已丢弃最后一个未完成", full)

    def test_drop_incomplete_noop_when_finished(self):
        # A finished run (## 完成) is left fully intact.
        import agent.multimodal.watch_file as D
        D.init_file("dr2", query="q")
        D.append_round("dr2", round_idx=1, frame_range=(0.0, 10.0),
                       sub_queries=["a"], findings="完整解读")
        D.mark_finished("dr2", rounds=1, summary_preview="done")
        dropped = D.drop_last_incomplete_round("dr2")
        self.assertFalse(dropped)
        self.assertIn("第 1 次分析", D.read_all("dr2"))

    def test_frame_range_wall_clock(self):
        # With a wall epoch, the frame range renders as absolute HH:MM:SS.
        import time
        import agent.multimodal.watch_file as D
        D.init_file("w1", query="q")
        epoch = time.mktime((2026, 7, 5, 14, 23, 0, 0, 0, -1))
        D.append_round("w1", round_idx=1, frame_range=(5.0, 86.0),
                       sub_queries=["x"], findings="f", wall_epoch=epoch)
        full = D.read_all("w1")
        self.assertIn("14:23:05", full)   # epoch + 5s
        self.assertIn("14:24:26", full)   # epoch + 86s
        self.assertNotIn("5.0s", full)    # not relative seconds anymore
        # Duration of the span is shown too (86 - 5 = 81s → "1分21s").
        self.assertIn("时长 1分21s", full)

# --------------------------------------------------------------------------- #
# Continuous loop (WatcherAgent._run_delegation) with mock buffer + stub responder
# --------------------------------------------------------------------------- #
class _MockBuf:
    def __init__(self):
        self._frames = []
        self._last_push_wall = None

    def push_many(self, n, base_ts):
        import time as _t
        from agent.multimodal._memory import Frame
        for i in range(n):
            self._frames.append(Frame(ts=base_ts + i, jpeg_b64="x"))
        self._last_push_wall = _t.time()

    def latest(self, n):
        return self._frames[-n:] if n > 0 else []

    def all_after(self, ts):
        return [f for f in self._frames if f.ts >= ts]

    @property
    def latest_ts(self):
        return self._frames[-1].ts if self._frames else None

    @property
    def size(self):
        return len(self._frames)


class _StubResponder:
    """Stub. `can_answer` controls the emitted router_react can_answer flag — qa
    ends as soon as a batch reports True (default True so qa terminates in tests;
    set False to make qa keep walking)."""

    def __init__(self, can_answer=True):
        self.batches = []
        self.can_answer = can_answer

    async def _spawn_delegation(self, *, task_instruction, prelude, sink,
                                on_event, ask_ts=None, ask_frames_override=None,
                                seen_search_briefs=None, prev_segment=None,
                                static_tail_check=False):
        n = len(ask_frames_override or [])
        self.batches.append(n)
        seen = seen_search_briefs if seen_search_briefs is not None else set()
        search_tasks = []
        for b in ("识别动作",):
            k = " ".join(b.lower().split())
            if k not in seen:
                seen.add(k)
                search_tasks.append({"brief": b})
        if on_event:
            await on_event({"type": "router_react", "round": 0,
                            "can_answer": self.can_answer,
                            "search_tasks": search_tasks,
                            "recall_tasks": [{"brief": "召回历史"}]})

        async def _task():
            await sink(f"answer frames={n}")
        return asyncio.ensure_future(_task())


class TestContinuousLoop(unittest.TestCase):
    def setUp(self):
        self._orig = hermes_constants.get_hermes_home
        _use_temp_home()

    def tearDown(self):
        hermes_constants.get_hermes_home = self._orig

    def _make_engine(self, loop):
        import agent.multimodal.watcher_engine as RE
        from agent.multimodal._config import Config
        eng = RE.WatcherAgent.__new__(RE.WatcherAgent)
        eng._loop = loop
        eng._emit_cb = None
        eng._on_delegation_complete = None
        eng._on_delegation_start = None
        eng._on_round_report = None
        eng._research_registry_cb = None
        eng._delegation_lock = threading.RLock()
        eng._delegation_pending = set()
        eng._stop_events = {}
        eng._stop_reasons = {}
        eng._clarify_events = {}
        eng._clarify_answers = {}
        eng._active = {}
        eng._source_stopped = False
        eng._source_epoch = 0
        eng._state_lock = threading.RLock()
        eng.cfg = Config()
        eng.cfg.watch_min_batch = 4       # target frames per round (test gate)
        eng.cfg.watch_frame_batch = 4
        eng.cfg.watch_round_ttl_sec = 0.1  # short ttl → rounds fire fast in tests
        eng.cfg.watch_poll_interval = 0.05
        eng.cfg.watch_stream_idle_stop = 5.0  # startup snapshot: source live
        eng.cfg.watch_max_rounds = 50
        eng.client = object()
        eng.model = "m"
        return eng

    def test_submit_then_immediate_delete_cannot_start_ghost_run(self):
        """A delete before the coroutine's first loop turn must be retained."""
        async def run():
            eng = self._make_engine(asyncio.get_running_loop())
            eng.frame_buffer = _MockBuf()
            responder = _StubResponder()
            eng.responder = responder
            registry = {"race1": {"_deleted": True, "status": "stopping"}}
            eng._research_registry_cb = lambda rid: registry.get(rid)
            completed = []
            eng._on_delegation_complete = (
                lambda rid, _task, _summary, reason:
                completed.append((rid, reason)))

            rid = eng.submit_complex_async("watch", request_id="race1")
            self.assertEqual(rid, "race1")
            with eng._delegation_lock:
                future = eng._active[rid]
                self.assertIn(rid, eng._delegation_pending)

            # This call runs before the scheduled coroutine can create its Event.
            # It must still report an active stop so the tool keeps its tombstone.
            self.assertTrue(eng.stop_delegation(rid, reason="deleted"))
            self.assertEqual(eng._stop_reasons[rid], "deleted")

            await asyncio.wait_for(asyncio.wrap_future(future), timeout=2.0)
            await asyncio.sleep(0)  # allow the Future done callback to run

            self.assertEqual(responder.batches, [])
            self.assertEqual(completed, [(rid, "deleted")])
            self.assertNotIn(rid, eng._active)
            self.assertNotIn(rid, eng._delegation_pending)
            self.assertNotIn(rid, eng._stop_events)
            self.assertNotIn(rid, eng._stop_reasons)

        asyncio.run(run())

    def test_analysis_multi_batch_dedup(self):
        import agent.multimodal.watch_file as D

        async def run():
            eng = self._make_engine(asyncio.get_event_loop())
            buf = _MockBuf(); buf.push_many(4, 0)
            eng.frame_buffer = buf
            eng.responder = _StubResponder()
            # The contract under test only needs the request id and stop reason.
            got = {}
            eng._on_delegation_complete = lambda rid, _task, _summary, reason: got.update(
                rid=rid, reason=reason)

            # Feed 2 more batches while it runs, then signal the video source
            # closed → the continuous loop drains + finishes (new termination:
            # waits for frames until source_stopped, no idle-timeout kill).
            async def feeder():
                await asyncio.sleep(0.08); buf.push_many(4, 100)
                await asyncio.sleep(0.08); buf.push_many(4, 200)
                await asyncio.sleep(0.08); eng._source_stopped = True
            asyncio.ensure_future(feeder())
            D.init_file("an1", query="q")
            await asyncio.wait_for(
                eng._run_delegation("an1", "q"), timeout=10)
            self.assertGreaterEqual(len(eng.responder.batches), 3)
            self.assertEqual(got.get("rid"), "an1")
            self.assertEqual(got.get("reason"), "source_end")
            # cross-round sub-query dedup: "识别动作" recorded exactly once.
            self.assertEqual(D.read_all("an1").count("[SQ] 识别动作"), 1)
        asyncio.run(run())

    def test_first_round_anchors_to_recent_tail_not_head(self):
        # ★ First round takes the MOST-RECENT target_frames, NOT the buffer HEAD:
        #   a long-running stream would otherwise make round 1 ingest a huge
        #   backlog (or dilute its timeline). With a 200-frame backlog (ts 0..199)
        #   and target=4, round 1's frames must be the tail (~196..199), not 0..N.
        import agent.multimodal.watch_file as D

        first_range = {}

        class _RangeResponder(_StubResponder):
            async def _spawn_delegation(self, *, ask_frames_override=None, **kw):
                if not first_range and ask_frames_override:
                    first_range["start"] = ask_frames_override[0].ts
                    first_range["end"] = ask_frames_override[-1].ts
                return await super()._spawn_delegation(
                    ask_frames_override=ask_frames_override, **kw)

        async def run():
            eng = self._make_engine(asyncio.get_event_loop())
            eng.cfg.watch_min_batch = 4   # target 4 frames/round
            buf = _MockBuf(); buf.push_many(200, 0)   # 20-min-ish backlog, ts 0..199
            eng.frame_buffer = buf
            eng.responder = _RangeResponder()

            async def stopper():
                await asyncio.sleep(0.15); eng._source_stopped = True
            asyncio.ensure_future(stopper())
            D.init_file("head1", query="q")
            await asyncio.wait_for(
                eng._run_delegation("head1", "q"), timeout=5)
            # Round 1 anchored to the RECENT tail: start is near the end (>=196),
            # NOT 0; end is the latest frame. Old backlog (ts 0..195) is dropped.
            self.assertGreaterEqual(first_range.get("start"), 196)
            self.assertEqual(first_range.get("end"), 199)
        asyncio.run(run())

    def test_stop_delegation(self):
        import agent.multimodal.watch_file as D

        async def run():
            eng = self._make_engine(asyncio.get_event_loop())
            buf = _MockBuf()
            eng.frame_buffer = buf

            # Deterministic stop path (no timing race): keep the buffer topped up
            # so a batch is always available (staleness never preempts), and set
            # the stop event from inside the responder on its 2nd batch. The
            # engine checks stop_ev right after each batch → guaranteed stop exit.
            state = {"n": 0, "base": 0}

            class _StopResponder(_StubResponder):
                async def _spawn_delegation(self, **kw):
                    state["n"] += 1
                    # top up so the NEXT round would have frames (proving the stop
                    # — not staleness — is what ends the loop).
                    state["base"] += 100
                    buf.push_many(8, state["base"])
                    if state["n"] >= 2:
                        ev = eng._stop_events.get("st1")
                        if ev is not None:
                            ev.set()
                    return await super()._spawn_delegation(**kw)

            buf.push_many(8, state["base"])  # seed round 1
            eng.responder = _StopResponder()
            D.init_file("st1", query="q")
            await asyncio.wait_for(
                eng._run_delegation("st1", "q"), timeout=10)
            full = D.read_all("st1")
            self.assertIn("停止指令", full)
            self.assertNotIn("视频源已停止", full)  # ended by user STOP, not source-close
        asyncio.run(run())

    def test_source_already_stopped_at_start_is_oneshot(self):
        # If the video source was EXPLICITLY stopped before analysis launches, a
        # continuous run must NOT wait forever — analyze the buffered frames once
        # and finish. (Authoritative signal is _source_stopped, NOT a frame-idle
        # heuristic — a live stream with a startup gap must still run continuously.)
        import agent.multimodal.watch_file as D

        async def run():
            eng = self._make_engine(asyncio.get_event_loop())
            buf = _MockBuf(); buf.push_many(4, 0)
            eng.frame_buffer = buf
            eng._source_stopped = True  # source explicitly closed at launch
            eng.responder = _StubResponder()
            # The contract under test only needs the request id and stop reason.
            got = {}
            eng._on_delegation_complete = lambda rid, _task, _summary, reason: got.update(
                rid=rid, reason=reason)
            D.init_file("os1", query="q")
            await asyncio.wait_for(
                eng._run_delegation("os1", "q"), timeout=5)
            # One batch only (didn't hang waiting for frames that never come).
            self.assertEqual(len(eng.responder.batches), 1)
            self.assertEqual(got.get("rid"), "os1")
            self.assertIn("启动时视频源已停止", D.read_all("os1"))
        asyncio.run(run())

    def test_live_source_with_stale_last_push_still_continuous(self):
        # Regression: a LIVE research run whose _last_push_wall is stale at launch
        # (>idle_stop gap: tab throttle / slow first analysis) must NOT be forced
        # one-shot. It should run continuously until the source is explicitly
        # stopped. (Bug: "明明有视频流却报启动时无视频流".)
        import time as _t
        import agent.multimodal.watch_file as D

        async def run():
            eng = self._make_engine(asyncio.get_event_loop())
            buf = _MockBuf(); buf.push_many(4, 0)
            buf._last_push_wall = _t.time() - 999  # very stale, but source live
            eng.frame_buffer = buf
            eng._source_stopped = False   # NOT stopped
            eng.responder = _StubResponder()

            async def feeder():
                await asyncio.sleep(0.08); buf.push_many(4, 100)
                await asyncio.sleep(0.08); eng._source_stopped = True  # now stop it
            asyncio.ensure_future(feeder())
            D.init_file("live1", query="q")
            await asyncio.wait_for(
                eng._run_delegation("live1", "q"), timeout=5)
            # Ran continuously (>1 batch) and did NOT log the one-shot note.
            self.assertGreaterEqual(len(eng.responder.batches), 2)
            self.assertNotIn("启动时视频源已停止", D.read_all("live1"))
        asyncio.run(run())

    def test_source_stopped_mid_run_drains_and_ends(self):
        # Source closes mid-run → loop stops waiting, drains remaining frames,
        # finishes with the source-close note.
        import agent.multimodal.watch_file as D

        async def run():
            eng = self._make_engine(asyncio.get_event_loop())
            buf = _MockBuf(); buf.push_many(4, 0)
            eng.frame_buffer = buf
            eng.responder = _StubResponder()

            async def closer():
                await asyncio.sleep(0.1); eng._source_stopped = True
            asyncio.ensure_future(closer())
            D.init_file("sc1", query="q")
            await asyncio.wait_for(
                eng._run_delegation("sc1", "q"), timeout=5)
            self.assertIn("视频源已停止", D.read_all("sc1"))
        asyncio.run(run())

    def test_stop_with_partial_frames_still_analyzes(self):
        # req ⑤: stream stops with FEWER than the first-batch gate → the watcher
        # must NOT wait forever; it immediately analyzes the frames on hand and
        # produces a round (so the user gets a result from those frames).
        import agent.multimodal.watch_file as D

        async def run():
            eng = self._make_engine(asyncio.get_event_loop())
            eng.cfg.watch_min_batch = 50  # gate far above available frames
            buf = _MockBuf(); buf.push_many(3, 0)  # only 3 frames, < gate
            eng.frame_buffer = buf
            eng.responder = _StubResponder()

            async def closer():
                await asyncio.sleep(0.1); eng._source_stopped = True
            asyncio.ensure_future(closer())
            D.init_file("pf1", query="q")
            await asyncio.wait_for(
                eng._run_delegation("pf1", "q"), timeout=5)
            # It analyzed the 3 on-hand frames (one batch ran), not zero.
            self.assertGreaterEqual(len(eng.responder.batches), 1)
            self.assertEqual(eng.responder.batches[0], 3)
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
