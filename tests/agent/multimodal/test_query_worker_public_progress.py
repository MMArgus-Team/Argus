"""A deferred QueryWorker answer must not be a silent black box.

query_multimodal hands the reply slot to a background worker and returns in
seconds; the answer itself can be minutes later. The rich ``multimodal.trajectory``
stream is a debug surface (base64 frame previews, per-worker payload shapes,
correlated by parent_user_message_id), so a client that is merely blocked on the
answer got nothing at all between tool.complete and message.complete — making a
slow answer indistinguishable from a hung one. These tests pin the public
``multimodal.bg`` progress line: same request_id the caller already holds, small
payload, one legible label per step.
"""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

from agent.multimodal.watcher_engine import WatcherAgent, query_progress_label


class _FrameBuffer:
    latest_ts = 1.0

    @staticmethod
    def latest(_count):
        return []


class _Responder:
    """Emits one recall-shaped progress event, then answers."""

    def __init__(self):
        self.done = threading.Event()

    async def _spawn_delegation(self, *, task_instruction, sink, on_event,
                                **_kwargs):
        async def _drive():
            await on_event({
                "type": "bg_progress",
                "channel": "recall",
                "task_id": "r0_r0",
                "round": 1,
                "phase": "distill",
                "elapsed_sec": 12.5,
            })
            await sink("答案")
            await on_event({"type": "answer_ready", "answer_full": "答案"})
            self.done.set()

        return asyncio.create_task(_drive())


def _wait_until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


def _engine(events):
    def _emit(event, payload):
        events.append((event, payload))

    responder = _Responder()
    engine = WatcherAgent(_FrameBuffer(), emit_cb=_emit)

    def _build():
        engine.cfg = SimpleNamespace(
            cont_recent_frames=3,
            query_worker_max_concurrency=2,
            query_worker_max_pending=8,
            ocr_timeout_sec=0.05,
        )
        engine.responder = responder
        return True

    engine._build = _build
    return engine, responder


def _run_one_query(events):
    engine, responder = _engine(events)
    assert engine.start(timeout=2.0)
    assert engine.submit_query_async(
        "刚才终端输出了什么",
        task_id="qry_abc",
        parent_user_message_id="dsh-live-query-42",
    )
    assert _wait_until(responder.done.is_set)
    assert _wait_until(lambda: any(
        e == "message.complete" for e, _ in events))
    assert engine.stop(timeout=2.0)


def test_progress_rides_multimodal_bg_with_the_callers_request_id():
    events = []
    _run_one_query(events)
    bg = [payload for event, payload in events if event == "multimodal.bg"]
    assert bg, [e for e, _ in events]
    # The caller correlates by request_id; task_id is extra, not a substitute.
    assert all(p["request_id"] == "dsh-live-query-42" for p in bg)
    assert all(p["task_id"] == "qry_abc" for p in bg)
    assert all(p["channel"] == "query" for p in bg)


def test_handoff_and_recall_steps_are_both_announced():
    events = []
    _run_one_query(events)
    labels = [p["phase"] for e, p in events if e == "multimodal.bg"]
    # The handoff itself matters: it explains why the foreground turn ended
    # without an answer.
    assert any("answering in the background" in label
               for label in labels), labels
    # And each inner step is named, with the round and subtask it belongs to, so
    # a client that collapses repeated labels still shows progress rather than
    # one frozen line.
    assert any("recall 1" in label and "distilling" in label
               for label in labels), labels


def test_public_progress_payload_stays_small_and_image_free():
    events = []
    _run_one_query(events)
    for event, payload in events:
        if event != "multimodal.bg":
            continue
        assert "frames" not in payload
        assert "evidence" not in payload
        for key, value in payload.items():
            assert not isinstance(value, (list, dict)), (key, value)
            if isinstance(value, str):
                assert len(value) <= 200, (key, len(value))


def test_trajectory_carries_request_id_for_correlation():
    events = []
    _run_one_query(events)
    trajectory = [p for e, p in events if e == "multimodal.trajectory"]
    assert trajectory
    assert all(p.get("request_id") == "dsh-live-query-42" for p in trajectory)
    # The historical key stays, so existing debug consumers are untouched.
    assert all(p.get("parent_user_message_id") == "dsh-live-query-42"
               for p in trajectory)


def test_label_folds_in_round_so_repeated_stages_stay_distinguishable():
    first = query_progress_label(
        "RecallWorker", "bg_progress",
        {"event": {"phase": "distill", "round": 1, "task_id": "r0_r0"}})
    second = query_progress_label(
        "RecallWorker", "bg_progress",
        {"event": {"phase": "distill", "round": 2, "task_id": "r0_r0"}})
    assert first != second
    assert "distilling" in first and "step 2" in first
    # Terminal statuses ride message.complete; no duplicate public line.
    assert query_progress_label("QueryWorker", "complete", {}) == ""


def test_label_names_every_number_outermost_first():
    """Nesting is spelled out, not encoded: router round → subtask → its step.

    ``recall [r1_r0] r1 tool obs`` packed three numbers with two meanings into
    one line and named none of them.  Each one now sits in its own labelled
    segment, in nesting order, so the line is readable without a legend.
    """
    dispatched = query_progress_label(
        "RecallWorker", "bg_progress",
        {"event": {"type": "bg_progress", "round": 1, "task_id": "r1_r0",
                   "brief": "什么终端输出"}})
    # The dispatch marker's round IS the router round the id already encodes, so
    # it is named once, and never as a bogus inner step of a subtask.
    assert dispatched == "round 2 · recall 1 · dispatched", dispatched

    observed = query_progress_label(
        "RecallWorker", "bg_progress",
        {"event": {"phase": "tool_obs", "round": 1, "task_id": "r1_r0",
                   "observations": [{"name": "a"}, {"name": "b"}]}})
    assert observed == (
        "round 2 · recall 1 · step 2 · read 2 tool results"), observed

    decided = query_progress_label(
        "RecallWorker", "bg_progress",
        {"event": {"phase": "r0_decision", "round": 0, "task_id": "r1_r0",
                   "can_answer": False, "n_next_calls": 2}})
    assert decided == (
        "round 2 · recall 1 · step 1 · needs 2 more lookups"), decided

    planned = query_progress_label(
        "QueryRouter", "router_react",
        {"event": {"type": "router_react", "round": 0,
                   "tool_calls": [{"name": "text_search"}],
                   "recall_tasks": [{"brief": "x"}, {"brief": "y"}]}})
    assert planned == (
        "round 1 · router · dispatched 1 search and 2 recalls"), planned

    # A subtask with no parseable dispatch id still reads, and an unmapped stage
    # degrades to its name rather than disappearing.
    unknown = query_progress_label(
        "RecallWorker", "bg_progress",
        {"event": {"phase": "some_new_stage", "round": 4, "task_id": "legacy"}})
    assert unknown == "recall legacy · step 5 · some new stage", unknown

    for label in (dispatched, observed, decided, planned):
        assert "_" not in label, label
        assert "[" not in label, label
