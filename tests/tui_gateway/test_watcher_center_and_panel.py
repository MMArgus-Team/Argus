# -*- coding: utf-8 -*-
"""End-to-end integration for the DECOUPLED live_watcher pipeline.

Contract (post-decoupling): the watcher is FULLY independent of the main agent
chat. Its per-round process and final report live ONLY in the watcher panel
(UI emits); session["history"] is NEVER mutated by the watcher.

  • _on_round_report → emits ``watcher.report_append`` (panel), touches NO history
  • _on_delegation_complete → emits ``watcher.final`` with the consolidated
    summary + ``watcher.complete``, touches NO history
  • right-panel bg events still carry seg + frame_ts_range + thought so the panel
    renders readable per-round cards

This wires the REAL engine ``_run_delegation`` to gateway-shaped callbacks that
mirror ``_maybe_start_live_watcher_agent`` so a regression in either the
panel-emit path or the "must not touch history" guarantee fails here —
deterministically, offline (no model, no gateway boot).
"""
import asyncio
import base64
import json
import tempfile
import threading
from io import BytesIO
from types import SimpleNamespace

import pytest


def _grad(seed: int) -> str:
    from PIL import Image
    img = Image.new("L", (32, 32))
    px = img.load()
    for y in range(32):
        for x in range(32):
            px[x, y] = (x * 8 + y * 4 + seed * 37) % 256
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _cfg(**over):
    base = dict(
        watch_frame_batch=4, watch_min_batch=4,
        watch_round_ttl_sec=0.05,   # short ttl → rounds fire fast on partial frames
        watch_poll_interval=0.01, watch_max_rounds=200,
        watch_silent_stop_rounds=2,
        watch_report_every_rounds=99, watch_summary_timeout=1.0,
        watch_summary_max_tokens=64,
    )
    base.update(over)
    return SimpleNamespace(**base)


class _FakeResponder:
    """Each _spawn_delegation call emits a router_react (with thought) then
    streams an answer through the sink — mirroring the real worker's event
    surface so seg metadata + on_round_report are exercised."""

    def __init__(self, thoughts, answers, feed_sink=True):
        self._thoughts = list(thoughts)
        self._answers = list(answers)
        self.calls = 0
        self._feed_sink = feed_sink

    async def _spawn_delegation(self, *, task_instruction, prelude, sink,
                               on_event, ask_frames_override=None,
                               seen_search_briefs=None, **_kwargs):
        i = self.calls
        self.calls += 1
        thought = self._thoughts[i] if i < len(self._thoughts) else ""
        answer = self._answers[i] if i < len(self._answers) else ""

        async def _task():
            await on_event({"type": "router_react", "round": 0,
                            "thought": thought, "can_answer": True,
                            "answer_len": len(answer), "tool_calls": [],
                            "recall_tasks": [], "elapsed_sec": 0.1})
            await on_event({"type": "answer_ready", "text_len": len(answer),
                            "text_preview": answer[:120],
                            "answer_full": answer, "source": "react",
                            "task_complete": False,
                            "completion_reason": ""})
            if answer and self._feed_sink:
                await sink(answer)
        return _task()


@pytest.fixture(autouse=True)
def _home(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        monkeypatch.setenv("ARGUS_HOME", d)
        yield d


def _run(coro, timeout=15.0):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(asyncio.wait_for(coro, timeout=timeout))
    finally:
        loop.close()


def _session_with_turn2(rid):
    """Realistic history: user → assistant(+tool_call) → tool(result,rid) →
    assistant(turn-2 confirmation). Used to prove the watcher NEVER mutates it."""
    tr = json.dumps({"op": "create", "request_id": rid, "status": "running",
                     "note": "已创建成功"}, ensure_ascii=False)
    return {
        "history": [
            {"role": "user", "content": "跟我看电影,记精彩片段,最后整理报告"},
            {"role": "assistant", "content": "好,我来设置观影研究。",
             "tool_calls": [{"id": "c1", "function": {"name": "set_live_watcher"}}]},
            {"role": "tool", "name": "set_live_watcher", "content": tr},
            {"role": "assistant", "content": "任务已创建成功「观影心得」#%s。" % rid},
        ],
        "history_lock": threading.Lock(),  # production shape (non-reentrant)
    }


def _build_engine(fb, responder, cfg, session, sid, emits, *, with_complete=True):
    """Wire the real engine with gateway-shaped callbacks that MIRROR the
    decoupled _maybe_start_live_watcher_agent closures: emit only, no history."""
    from agent.multimodal.watcher_engine import WatcherAgent

    def _emit(ev, payload):
        emits.append((ev, payload))

    def _on_round_report(request_id, round_idx, report_text):
        # Decoupled: UI emit ONLY. NO history mutation.
        text = (report_text or "").strip()
        if not text:
            return
        emits.append(("watcher.report_append",
                      {"request_id": str(request_id), "round": int(round_idx),
                       "text": text}))

    def _on_delegation_complete(
        request_id, brief, full_text, stop_reason="normal"
    ):
        # Decoupled: push final consolidated report to the panel; NO history.
        final = (full_text or "").strip()
        if final:
            emits.append(("watcher.final",
                          {"request_id": str(request_id), "brief": brief,
                           "text": final, "stop_reason": stop_reason}))
        emits.append(("watcher.complete",
                      {"request_id": str(request_id), "brief": brief,
                       "stop_reason": stop_reason}))

    eng = WatcherAgent.__new__(WatcherAgent)
    eng.frame_buffer = fb
    eng.cfg = cfg
    eng.client = object()
    eng.model = "fake"
    eng.responder = responder
    eng._emit_cb = _emit
    eng._on_delegation_complete = _on_delegation_complete if with_complete else None
    eng._on_round_report = _on_round_report
    eng._on_delegation_start = None
    eng._stop_events = {}
    eng._source_stopped = False
    eng._sid = sid
    eng._research_registry_cb = None
    return eng


def test_panel_gets_every_round_and_history_untouched():
    rid = "req_center01"
    session = _session_with_turn2(rid)
    history_before = json.dumps(session["history"], ensure_ascii=False)

    from agent.multimodal._memory import Frame, FrameBuffer
    fb = FrameBuffer(SimpleNamespace(buffer_seconds=1800, buffer_capture_fps=2))
    for i in range(6):
        fb.push(Frame(ts=float(i), jpeg_b64=_grad(i)))   # wave 1 (round 1)

    responder = _FakeResponder(
        thoughts=["洛汗村庄遇袭逃亡", "范贡森林强兽人追杀", ""],
        answers=["第1段:洛汗村庄遇袭的逃亡场景解读…",
                 "第2段:范贡森林逃亡与强兽人追杀…", ""])
    emits = []
    eng = _build_engine(fb, responder, _cfg(), session, "sid-x", emits)
    _orig_emit = eng._emit_cb
    _fed = {"w2": False}
    def _emit_and_maybe_stop(ev, payload):
        _orig_emit(ev, payload)
        # After round 1 ran, feed wave 2 so a 2nd productive round happens.
        if responder.calls == 1 and not _fed["w2"]:
            _fed["w2"] = True
            for i in range(6, 12):
                fb.push(Frame(ts=float(i), jpeg_b64=_grad(i)))
        # After 2 productive rounds, stop the source so the run drains + ends.
        if responder.calls >= 2:
            eng._source_stopped = True
    eng._emit_cb = _emit_and_maybe_stop
    _run(eng._run_delegation(rid, task_instruction="观影心得"), timeout=15.0)

    # ── (A) HISTORY UNTOUCHED: the watcher must not mutate the main chat ──
    assert json.dumps(session["history"], ensure_ascii=False) == history_before, \
        "watcher mutated session['history'] — must be fully decoupled"

    # ── (B) PANEL: watcher.report_append emitted per productive round ──
    appends = [p for (ev, p) in emits if ev == "watcher.report_append"]
    assert len(appends) == 2, f"expected 2 report_append, got {len(appends)}"
    assert all(a["request_id"] == rid for a in appends)
    assert "第1段" in appends[0]["text"] and "第2段" in appends[1]["text"]

    # ── (C) FINAL: watcher.final carries the consolidated summary ──
    finals = [p for (ev, p) in emits if ev == "watcher.final"]
    assert len(finals) == 1, f"expected 1 watcher.final, got {len(finals)}"
    assert finals[0]["request_id"] == rid
    assert finals[0]["text"].strip(), "final report is empty"
    assert "第1段" in finals[0]["text"] and "第2段" in finals[0]["text"], \
        "final report did not use the complete accumulated segment history"
    assert any(ev == "watcher.complete" for ev, _ in emits)

    # ── (D) PANEL METADATA: bg events carry seg + frame_ts_range + thought ──
    bg = [p for (ev, p) in emits if ev == "multimodal.bg"]
    seg_starts = [p for p in bg if p.get("type") == "segment_start"]
    assert seg_starts, "no segment_start emitted"
    assert all(isinstance(s.get("seg"), int) and s["seg"] >= 1 for s in seg_starts)
    assert all(s.get("frame_ts_range") for s in seg_starts), "segment_start missing frame_ts_range"
    reacts = [p for p in bg if p.get("type") == "router_react"]
    assert reacts, "no router_react emitted"
    assert all(isinstance(r.get("seg"), int) for r in reacts), \
        "router_react missing seg stamp — panel can't attach 👁看到 to its segment"
    seg1_reacts = [r for r in reacts if r.get("seg") == 1 and (r.get("thought") or "").strip()]
    assert seg1_reacts, "segment 1's router_react has no thought → right card shows no 👁看到"


def test_final_report_via_answer_full_when_sink_not_fed():
    """Even when the _fake_stream→sink side channel is NOT fed, the round report
    must still reach the panel via answer_full captured from answer_ready — and
    still never touch history."""
    rid = "req_bug5"
    session = _session_with_turn2(rid)
    history_before = json.dumps(session["history"], ensure_ascii=False)
    from agent.multimodal._memory import Frame, FrameBuffer
    fb = FrameBuffer(SimpleNamespace(buffer_seconds=1800, buffer_capture_fps=2))
    for i in range(8):
        fb.push(Frame(ts=float(i), jpeg_b64=_grad(i)))
    responder = _FakeResponder(thoughts=["洛汗村庄"], answers=["第1段:洛汗村庄遇袭解读"],
                              feed_sink=False)  # ← sink NOT fed
    emits = []
    eng = _build_engine(fb, responder, _cfg(), session, "sid-b5", emits)
    _orig = eng._emit_cb
    def _stop(ev, p):
        _orig(ev, p)
        if responder.calls >= 1:
            eng._source_stopped = True
    eng._emit_cb = _stop
    _run(eng._run_delegation(rid, task_instruction="观影"), timeout=10.0)
    appends = [p for (ev, p) in emits if ev == "watcher.report_append"]
    assert appends and "第1段:洛汗村庄遇袭解读" in appends[0]["text"], \
        "report lost when sink not fed — answer_full fallback failed"
    assert json.dumps(session["history"], ensure_ascii=False) == history_before, \
        "watcher mutated history even in the sink-not-fed path"


def test_visual_task_complete_signal_stops_live_source_and_finalizes_once():
    """A conclusively observed embedded-player end must stop the watcher even
    while screen sharing remains live, then produce one consolidated completion."""
    rid = "req_player_end"
    session = _session_with_turn2(rid)
    from agent.multimodal._memory import Frame, FrameBuffer
    fb = FrameBuffer(SimpleNamespace(buffer_seconds=1800, buffer_capture_fps=2))
    for i in range(8):
        fb.push(Frame(ts=float(i), jpeg_b64=_grad(i)))

    class _CompletingResponder(_FakeResponder):
        async def _spawn_delegation(self, *, task_instruction, prelude, sink,
                                   on_event, **_kwargs):
            self.calls += 1

            async def _task():
                answer = "播放器已到达结束画面，视频内容已播放完毕。"
                await on_event({
                    "type": "router_react", "round": 0,
                    "thought": "The player visibly reached its ended state.",
                    "answer_len": len(answer), "tool_calls": [],
                    "recall_tasks": [], "elapsed_sec": 0.1,
                    "task_complete": True,
                    "completion_reason": "player ended state visible",
                })
                await on_event({
                    "type": "answer_ready", "text_len": len(answer),
                    "text_preview": answer, "answer_full": answer,
                    "source": "react", "task_complete": True,
                    "completion_reason": "player ended state visible",
                })
            return _task()

    responder = _CompletingResponder([], [])
    emits = []
    eng = _build_engine(fb, responder, _cfg(), session, "sid-end", emits)
    _run(
        eng._run_delegation(
            rid,
            task_instruction="持续观看，播放器视频结束时生成总结",
        ),
        timeout=10.0,
    )

    assert eng._source_stopped is False, "test must keep screen sharing live"
    completions = [p for ev, p in emits if ev == "watcher.complete"]
    finals = [p for ev, p in emits if ev == "watcher.final"]
    assert len(completions) == 1
    assert completions[0]["stop_reason"] == "task_complete"
    assert len(finals) == 1
    assert "播放器" in finals[0]["text"]


def test_static_tail_flushes_partial_batch_before_long_ttl():
    """Two seconds without visual change sends the before/after raw window to
    the dedicated completion verifier instead of waiting for the 60s gate."""
    rid = "req_static_tail_fast"
    session = _session_with_turn2(rid)
    from agent.multimodal._memory import Frame, FrameBuffer
    fb = FrameBuffer(SimpleNamespace(buffer_seconds=1800, buffer_capture_fps=2))
    for i in range(60):
        fb.push(Frame(ts=float(i), jpeg_b64=_grad(i)))

    class _TailResponder(_FakeResponder):
        def __init__(self):
            super().__init__([], [])
            self.batch_timestamps = []
            self.confirm_timestamps = []

        async def _spawn_delegation(self, *, on_event,
                                   ask_frames_override=None, **_kwargs):
            idx = self.calls
            self.calls += 1
            self.batch_timestamps.append([
                frame.ts for frame in (ask_frames_override or [])])

            async def _task():
                if idx == 0:
                    await on_event({
                        "type": "answer_ready", "answer_full": "视频主体内容。",
                        "task_complete": False,
                        "completion_candidate": False,
                    })
                    # One novel end screen, followed by five seconds of the same
                    # raw screen. Only ts=60 enters the dHash-novel buffer.
                    for ts in range(60, 66):
                        fb.push(Frame(ts=float(ts), jpeg_b64=_grad(9)))
            return _task()

        async def confirm_visual_completion(self, **kwargs):
            self.confirm_timestamps = [
                frame.ts for frame in (kwargs.get("frames") or [])]
            return (
                True, 0.96,
                "progress reached 00:50/00:50 across the static boundary",
                "播放器显示 00:50/00:50，视频结束。",
            )

    responder = _TailResponder()
    emits = []
    eng = _build_engine(
        fb, responder,
        _cfg(
            watch_min_batch=60,
            watch_round_ttl_sec=60.0,
            watch_static_tail_flush_sec=2.0,
            watch_poll_interval=0.01,
        ),
        session, "sid-static-tail", emits,
    )
    _run(
        eng._run_delegation(
            rid, task_instruction="持续观看，视频结束后生成总结"),
        timeout=2.0,
    )

    assert responder.calls == 1
    assert min(responder.confirm_timestamps) <= 60.0
    assert max(responder.confirm_timestamps) >= 62.0
    assert any(
        ev == "multimodal.bg" and p.get("type") == "static_tail_flush"
        for ev, p in emits
    )
    completions = [p for ev, p in emits if ev == "watcher.complete"]
    assert completions[-1]["stop_reason"] == "task_complete"
    finals = [p for ev, p in emits if ev == "watcher.final"]
    assert finals and "视频结束" in finals[-1]["text"]


def test_static_tail_flushes_when_zero_novel_frames_but_raw_capture_advances():
    """A fully deduplicated end screen must still receive one raw-tail check.

    This is the exact 0/N regression: after the first segment, capture keeps
    delivering the same JPEG so the raw monitor cursor advances while the
    dHash-novel buffer remains empty.  The watcher must flush one or two raw
    captures before the long TTL instead of entering its indefinite pause.
    """
    rid = "req_static_tail_zero_novel"
    session = _session_with_turn2(rid)
    from agent.multimodal._memory import Frame, FrameBuffer
    fb = FrameBuffer(SimpleNamespace(buffer_seconds=1800, buffer_capture_fps=2))
    for i in range(60):
        fb.push(Frame(ts=float(i), jpeg_b64=_grad(i)))

    class _ZeroNovelTailResponder(_FakeResponder):
        def __init__(self):
            super().__init__([], [])
            self.batch_timestamps = []
            self.confirm_timestamps = []

        async def _spawn_delegation(self, *, on_event,
                                   ask_frames_override=None, **_kwargs):
            idx = self.calls
            self.calls += 1
            batch = list(ask_frames_override or [])
            self.batch_timestamps.append([frame.ts for frame in batch])

            async def _task():
                if idx == 0:
                    await on_event({
                        "type": "answer_ready",
                        "answer_full": "视频主体内容。",
                        "task_complete": False,
                        "completion_candidate": False,
                    })
                    # Exact copies of the last retained frame advance only the
                    # raw monitor deque.  No frame after ts=59 enters the novel
                    # deque, faithfully reproducing the Web watcher at 0/N.
                    for ts in range(60, 63):
                        fb.push(Frame(ts=float(ts), jpeg_b64=_grad(59)))
            return _task()

        async def confirm_visual_completion(self, **kwargs):
            self.confirm_timestamps = [
                frame.ts for frame in (kwargs.get("frames") or [])]
            return (
                True, 0.95, "persistent raw end screen",
                "静止的播放器结束画面已确认，视频结束。",
            )

    responder = _ZeroNovelTailResponder()
    emits = []
    eng = _build_engine(
        fb, responder,
        _cfg(
            watch_min_batch=60,
            watch_round_ttl_sec=60.0,
            watch_static_tail_flush_sec=2.0,
            watch_poll_interval=0.01,
        ),
        session, "sid-static-tail-zero", emits,
    )
    _run(
        eng._run_delegation(
            rid, task_instruction="持续观看，视频结束后生成总结"),
        timeout=2.0,
    )

    assert responder.calls == 1
    assert min(responder.confirm_timestamps) <= 59.0
    assert max(responder.confirm_timestamps) >= 61.0
    flushes = [
        p for ev, p in emits
        if ev == "multimodal.bg" and p.get("type") == "static_tail_flush"
    ]
    assert flushes and flushes[-1]["have"] == 0
    completions = [p for ev, p in emits if ev == "watcher.complete"]
    assert completions[-1]["stop_reason"] == "task_complete"


def test_rejected_static_boundary_is_not_rechecked_until_scene_changes():
    """A buffering/frozen verdict must latch the boundary instead of calling
    the VLM again every two seconds on identical raw captures."""
    rid = "req_static_tail_rejected_once"
    session = _session_with_turn2(rid)
    from agent.multimodal._memory import Frame, FrameBuffer
    fb = FrameBuffer(SimpleNamespace(buffer_seconds=1800, buffer_capture_fps=2))
    for i in range(60):
        fb.push(Frame(ts=float(i), jpeg_b64=_grad(i)))

    class _BufferingResponder(_FakeResponder):
        def __init__(self):
            super().__init__([], [])
            self.confirm_calls = 0

        async def _spawn_delegation(self, *, on_event, **_kwargs):
            self.calls += 1

            async def _task():
                await on_event({
                    "type": "answer_ready",
                    "answer_full": "视频主体内容。",
                    "task_complete": False,
                    "completion_candidate": False,
                })
                for ts in range(60, 70):
                    fb.push(Frame(ts=float(ts), jpeg_b64=_grad(59)))
            return _task()

        async def confirm_visual_completion(self, **_kwargs):
            self.confirm_calls += 1
            return (
                False, 0.98,
                "Spinner may indicate temporary buffering; no end UI is visible.",
                "",
            )

    responder = _BufferingResponder()
    emits = []
    eng = _build_engine(
        fb, responder,
        _cfg(
            watch_min_batch=60,
            watch_round_ttl_sec=60.0,
            watch_static_tail_flush_sec=2.0,
            watch_poll_interval=0.01,
        ),
        session, "sid-static-tail-rejected", emits,
    )
    original_emit = eng._emit_cb

    def _stop_after_latched_wait(event, payload):
        original_emit(event, payload)
        if payload.get("waiting_for_scene_change"):
            eng._source_stopped = True

    eng._emit_cb = _stop_after_latched_wait
    _run(
        eng._run_delegation(
            rid, task_instruction="持续观看，视频结束后生成总结"),
        timeout=2.0,
    )

    assert responder.calls == 1
    assert responder.confirm_calls == 1
    assert any(
        event == "multimodal.bg"
        and payload.get("waiting_for_scene_change")
        for event, payload in emits
    )


def test_ambiguous_ending_is_confirmed_once_after_no_novel_frames():
    """A conclusion-like ending followed by a static shared-screen tail gets
    one independent raw-frame verification and then finalizes, even though the
    screen-share MediaStream itself remains live."""
    rid = "req_candidate_end"
    session = _session_with_turn2(rid)
    from agent.multimodal._memory import Frame, FrameBuffer
    fb = FrameBuffer(SimpleNamespace(buffer_seconds=1800, buffer_capture_fps=2))
    for i in range(8):
        fb.push(Frame(ts=float(i), jpeg_b64=_grad(i)))

    class _CandidateResponder(_FakeResponder):
        def __init__(self):
            super().__init__([], [])
            self.confirm_calls = 0
            self.confirm_frame_count = 0

        async def _spawn_delegation(self, *, task_instruction, prelude, sink,
                                   on_event, **_kwargs):
            self.calls += 1

            async def _task():
                answer = "主持人进入结论章节，最后询问观众‘你同意吗’。"
                await on_event({
                    "type": "router_react", "round": 0,
                    "thought": "Conclusion-like ending, but spinner is ambiguous.",
                    "answer_len": len(answer), "tool_calls": [],
                    "recall_tasks": [], "elapsed_sec": 0.1,
                    "task_complete": False,
                    "completion_candidate": True,
                    "completion_candidate_reason": (
                        "conclusion chapter and closing question; spinner remains"
                    ),
                })
                await on_event({
                    "type": "answer_ready", "text_len": len(answer),
                    "text_preview": answer, "answer_full": answer,
                    "source": "react", "task_complete": False,
                    "completion_reason": "",
                    "completion_candidate": True,
                    "completion_candidate_reason": (
                        "conclusion chapter and closing question; spinner remains"
                    ),
                })
            return _task()

        async def confirm_visual_completion(self, **kwargs):
            self.confirm_calls += 1
            self.confirm_frame_count = len(kwargs.get("frames") or [])
            return (
                True, 0.93,
                "closing interaction followed by a persistent terminal player state",
                "视频在结论与结尾互动后播放完毕。",
            )

    responder = _CandidateResponder()
    emits = []
    eng = _build_engine(
        fb, responder,
        _cfg(watch_completion_confirm_delay_sec=0.02,
             watch_completion_confirm_frames=6),
        session, "sid-candidate", emits,
    )
    _run(
        eng._run_delegation(
            rid, task_instruction="持续观看，视频结束时生成总结"),
        timeout=10.0,
    )

    assert eng._source_stopped is False, "screen sharing must remain live"
    assert responder.calls == 1, "confirmation must not start another segment"
    assert responder.confirm_calls == 1
    assert responder.confirm_frame_count >= 2
    completions = [p for ev, p in emits if ev == "watcher.complete"]
    finals = [p for ev, p in emits if ev == "watcher.final"]
    assert len(completions) == 1
    assert completions[0]["stop_reason"] == "task_complete"
    assert len(finals) == 1
    assert any(
        ev == "multimodal.bg" and p.get("type") == "completion_confirmed"
        for ev, p in emits
    )


def test_buffering_rejection_keeps_candidate_for_extended_followup():
    """A strict first check may reject a spinner as buffering. The candidate
    must survive and receive a follow-up over later raw captures; otherwise the
    dHash-empty loop waits forever and never produces the final report."""
    rid = "req_candidate_retry"
    session = _session_with_turn2(rid)
    from agent.multimodal._memory import Frame, FrameBuffer
    fb = FrameBuffer(SimpleNamespace(buffer_seconds=1800, buffer_capture_fps=2))
    for i in range(8):
        fb.push(Frame(ts=float(i), jpeg_b64=_grad(i)))

    class _RetryResponder(_FakeResponder):
        def __init__(self):
            super().__init__([], [])
            self.confirm_attempts = []
            self.prior_reasons = []

        async def _spawn_delegation(self, *, on_event, **_kwargs):
            self.calls += 1

            async def _task():
                answer = "总结式结束卡片持续显示，播放器中央出现转圈。"
                await on_event({
                    "type": "answer_ready", "text_len": len(answer),
                    "text_preview": answer, "answer_full": answer,
                    "source": "react", "task_complete": False,
                    "completion_reason": "",
                    "completion_candidate": True,
                    "completion_candidate_reason": (
                        "semantic end card with an ambiguous player spinner"),
                })
            return _task()

        async def confirm_visual_completion(self, **kwargs):
            attempt = int(kwargs.get("attempt") or 1)
            self.confirm_attempts.append(attempt)
            self.prior_reasons.append(
                str(kwargs.get("prior_confirmation_reason") or ""))
            if attempt == 1:
                # Raw capture continues while the first verifier is running.
                # These are dHash-identical, so they prove source liveness
                # without representing a resumed/new scene.
                for offset in range(3):
                    fb.push(Frame(ts=20.0 + offset, jpeg_b64=_grad(7)))
                return (
                    False, 0.98,
                    "The same spinner may still indicate temporary buffering.",
                    "",
                )
            return (
                True, 0.94,
                "The semantic end card and spinner persisted across the "
                "extended raw-capture window with no resumed playback.",
                "视频在总结画面后结束。",
            )

    responder = _RetryResponder()
    emits = []
    eng = _build_engine(
        fb, responder,
        _cfg(
            watch_completion_confirm_delay_sec=0.02,
            watch_completion_confirm_retry_total_sec=0.02,
            watch_completion_confirm_max_attempts=2,
            watch_completion_confirm_frames=6,
        ),
        session, "sid-candidate-retry", emits,
    )
    original_emit = eng._emit_cb
    raw_pushed_for = set()

    def _emit_with_static_raw_tail(ev, payload):
        original_emit(ev, payload)
        if payload.get("type") != "completion_confirm_wait":
            return
        attempt = int(payload.get("attempt") or 1)
        if attempt in raw_pushed_for:
            return
        raw_pushed_for.add(attempt)
        # Same JPEG: enters the raw monitor deque but is removed from the
        # dHash-novel scene buffer, faithfully reproducing a static end screen.
        base_ts = 10.0 * attempt
        for offset in range(3):
            fb.push(Frame(
                ts=base_ts + offset,
                jpeg_b64=_grad(7),
            ))

    eng._emit_cb = _emit_with_static_raw_tail
    _run(
        eng._run_delegation(
            rid, task_instruction="持续观看，视频结束时生成总结"),
        timeout=10.0,
    )

    assert responder.calls == 1
    assert responder.confirm_attempts == [1, 2]
    assert "temporary buffering" in responder.prior_reasons[1]
    assert not any(
        ev == "multimodal.bg"
        and p.get("type") == "completion_confirm_wait"
        and p.get("attempt") == 2
        for ev, p in emits
    ), "the follow-up uses total idle time; it must not add another delay"
    completions = [p for ev, p in emits if ev == "watcher.complete"]
    finals = [p for ev, p in emits if ev == "watcher.final"]
    assert len(completions) == 1
    assert completions[0]["stop_reason"] == "task_complete"
    assert len(finals) == 1


def test_new_scene_cancels_completion_candidate_without_confirmation():
    """A candidate is not sticky: playback resuming during the grace period
    cancels it, processes the new scene, and never calls the verifier."""
    rid = "req_candidate_resume"
    session = _session_with_turn2(rid)
    from agent.multimodal._memory import Frame, FrameBuffer
    fb = FrameBuffer(SimpleNamespace(buffer_seconds=1800, buffer_capture_fps=2))
    for i in range(4):
        fb.push(Frame(ts=float(i), jpeg_b64=_grad(i)))

    class _ResumeResponder(_FakeResponder):
        def __init__(self):
            super().__init__([], [])
            self.confirm_calls = 0

        async def _spawn_delegation(self, *, task_instruction, prelude, sink,
                                   on_event, **_kwargs):
            idx = self.calls
            self.calls += 1

            async def _task():
                is_candidate = idx == 0
                answer = (
                    "画面出现疑似结尾卡片。"
                    if is_candidate else "播放继续，出现了新的道路场景。"
                )
                payload = {
                    "type": "answer_ready", "text_len": len(answer),
                    "text_preview": answer, "answer_full": answer,
                    "source": "react", "task_complete": False,
                    "completion_reason": "",
                    "completion_candidate": is_candidate,
                    "completion_candidate_reason": (
                        "possible end card" if is_candidate else ""),
                }
                await on_event(payload)
            return _task()

        async def confirm_visual_completion(self, **_kwargs):
            self.confirm_calls += 1
            return True, 0.99, "should not run", ""

    responder = _ResumeResponder()
    emits = []
    eng = _build_engine(
        fb, responder,
        _cfg(watch_completion_confirm_delay_sec=0.05,
             watch_completion_confirm_frames=6),
        session, "sid-resume", emits,
    )
    original_emit = eng._emit_cb
    resumed = {"done": False}

    def _emit_and_resume(ev, payload):
        original_emit(ev, payload)
        if (responder.calls == 1
                and payload.get("type") == "completion_confirm_wait"
                and not resumed["done"]):
            resumed["done"] = True
            for i in range(4, 8):
                fb.push(Frame(ts=float(i), jpeg_b64=_grad(i)))
        if responder.calls >= 2:
            eng._source_stopped = True

    eng._emit_cb = _emit_and_resume
    _run(
        eng._run_delegation(
            rid, task_instruction="持续观看，视频结束时生成总结"),
        timeout=10.0,
    )

    assert resumed["done"]
    assert responder.calls == 2
    assert responder.confirm_calls == 0
    reports = [p["text"] for ev, p in emits if ev == "watcher.report_append"]
    assert any("播放继续" in report for report in reports)


def test_delete_stop_reason_wins_over_shared_stop_event():
    """Delete and disable may share the wake-up Event, but deletion must remain
    terminal and must never be persisted as a resumable interruption."""
    rid = "req_delete_reason"
    session = _session_with_turn2(rid)
    from agent.multimodal._memory import Frame, FrameBuffer
    from agent.multimodal import watch_file
    fb = FrameBuffer(SimpleNamespace(buffer_seconds=1800, buffer_capture_fps=2))
    for i in range(4):
        fb.push(Frame(ts=float(i), jpeg_b64=_grad(i)))

    registry = {"id": rid, "_deleted": False}
    responder = _FakeResponder(["分析结束"], ["当前段报告"])
    emits = []
    eng = _build_engine(
        fb, responder, _cfg(), session, "sid-delete-reason", emits)
    eng._research_registry_cb = lambda _rid: registry
    original_spawn = responder._spawn_delegation

    async def _spawn_and_delete(**kwargs):
        task = await original_spawn(**kwargs)

        async def _wrapped():
            await task
            registry["_deleted"] = True
            eng._stop_reasons[rid] = "deleted"
            eng._stop_events[rid].set()
        return _wrapped()

    responder._spawn_delegation = _spawn_and_delete
    _run(eng._run_delegation(rid, task_instruction="持续分析"))

    completions = [p for ev, p in emits if ev == "watcher.complete"]
    assert completions[-1]["stop_reason"] == "deleted"
    assert watch_file.read_status(rid)["status"] == "deleted"
    assert watch_file.read_state(rid)["stop_reason"] == "deleted"


def test_resume_hydrates_segment_base_cursor_and_seen_subqueries():
    rid = "req_resume_state"
    session = _session_with_turn2(rid)
    from agent.multimodal._memory import Frame, FrameBuffer
    from agent.multimodal import watch_file
    watch_file.init_file(rid, query="持续分析", session_id="sid-resume")
    watch_file.append_round(
        rid, round_idx=1, frame_range=(0.0, 1.0),
        sub_queries=["Earlier topic"], findings="旧报告")
    watch_file.update_state(
        rid, seg_base=1, cursor_ts=1.0, runtime_id="same-runtime")

    fb = FrameBuffer(SimpleNamespace(buffer_seconds=1800, buffer_capture_fps=2))
    for i in range(4):
        fb.push(Frame(ts=float(i), jpeg_b64=_grad(i)))

    class _CaptureSeen(_FakeResponder):
        async def _spawn_delegation(self, **kwargs):
            self.seen = set(kwargs.get("seen_search_briefs") or set())
            return await super()._spawn_delegation(**kwargs)

    responder = _CaptureSeen(["继续分析"], ["新报告"])
    emits = []
    eng = _build_engine(fb, responder, _cfg(), session, "sid-resume", emits)
    eng._runtime_id = "same-runtime"
    eng._source_stopped = True
    eng._research_registry_cb = lambda _rid: {"_seg_base": 1}

    _run(eng._run_delegation(rid, task_instruction="持续分析"))

    reports = [p for ev, p in emits if ev == "watcher.report_append"]
    assert reports[-1]["round"] == 2
    assert responder.seen == {"earlier topic"}
    rounds = watch_file.read_structured(rid)["rounds"]
    assert [row["n"] for row in rounds] == [1, 2]


def test_persisted_final_report_preserves_complete_multiline_summary():
    """Reopen/history must see the exact full consolidated report, not a preview."""
    from agent.multimodal import watch_file

    rid = "req_full_final"
    watch_file.init_file(rid, query="持续观看并总结")
    watch_file.append_round(
        rid,
        round_idx=1,
        frame_range=(0.0, 10.0),
        findings="第一段事实：价格 6.58 万元。",
    )
    summary = (
        "完整总结标题\n"
        "- 第一段事实：价格 6.58 万元。\n"
        "- 第二段事实：续航 403 公里。\n"
        "结论：视频播放结束。"
    )
    watch_file.mark_finished(rid, rounds=1, summary_preview=summary)

    parsed = watch_file.read_structured(rid)
    assert parsed is not None
    assert parsed["final_report"] == summary
    assert summary in watch_file.read_all(rid)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
