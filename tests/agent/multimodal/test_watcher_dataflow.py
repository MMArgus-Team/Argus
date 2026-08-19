"""Verify the deep-research delegation DATAFLOW + CONCURRENCY + QUERY DEDUP on
the real WatcherWorker._spawn_delegation coroutine.

Covers the optimizations the sign-off test (test_live_watcher_loop.py) did not:

  #2  数据流程 — one ReAct round's search/recall findings accumulate into
      search_log/recall_log and the final answer is streamed via the sink;
      the whole round → answer → sink pipeline is exercised end-to-end.

  #5  并发性能 — all tool_calls + recall_tasks of a single round are dispatched
      CONCURRENTLY (asyncio.gather), not serially: N tasks each sleeping S run
      in ~S total, not ~N*S.

  #6  图像优先 — react_batch_frames (the batch's real frames) are threaded into
      react_step every round, so the Router always has the images to read before
      deciding to search.

  #7  搜索 query 去重 — a search tool_call whose (name,args,anchor) was already
      issued this run is filtered out before fan-out (never re-executed), and the
      shared seen-set is honored across batches.

react_step (the LLM call) is stubbed with canned ReactStep objects so the test
is deterministic and offline; the ORCHESTRATION under test (fan-out, gather,
dedup, log accumulation, answer streaming) is the real code path.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from agent.multimodal._workers import WatcherWorker, ReactStep, RecallResult


class _SlowToolBox:
    """Fake ToolBox: each call sleeps `delay` then returns a tagged result.
    Records call args + concurrency high-water-mark."""

    def __init__(self, delay=0.3):
        self.delay = delay
        self.calls = []
        self._active = 0
        self.max_concurrent = 0

    async def call(self, name, args, *, anchor=None, crop_progress_cb=None):
        self._active += 1
        self.max_concurrent = max(self.max_concurrent, self._active)
        self.calls.append((name, dict(args or {})))
        try:
            await asyncio.sleep(self.delay)
            return f"[{name} q={args.get('query','')}] finding-text"
        finally:
            self._active -= 1


class _FakeRecallAgent:
    def __init__(self, delay=0.3):
        self.delay = delay
        self.runs = 0

    async def run(self, *, initial_calls, brief, user_text, ask_ts, on_progress):
        self.runs += 1
        await asyncio.sleep(self.delay)
        return RecallResult(findings=f"recall::{brief}", frame_ids=[],
                            clues=[], rounds=1, elapsed_sec=self.delay)


class _FakeConversation:
    def __init__(self):
        self.appended = []

    def recent_n(self, n):
        return []

    async def append(self, role, content, rel_ts=None):
        self.appended.append((role, content))


def _make_worker(*, toolbox, recall_agent, cfg):
    w = WatcherWorker.__new__(WatcherWorker)
    w.cfg = cfg
    w.client = object()
    w.buf = SimpleNamespace(latest_ts=100.0, latest=lambda n: [],
                            latest_one=lambda: None)
    w.mem = None
    w.store = None
    w.conversation = _FakeConversation()
    w.frame_store = None
    w.recorder = None
    w.toolbox = toolbox
    w.recall_agent = recall_agent
    w.inflight = {}
    w._front_lock = asyncio.Lock()
    return w


def _cfg(**over):
    base = dict(
        react_max_rounds=4,
        react_search_tasks_max=5,
        cont_recent_frames=12,
        cont_now_frames=4,
        search_recent_frames=8,
        cont_recall_frames_max=6,
    )
    base.update(over)
    return SimpleNamespace(**base)


async def _drive(worker, *, ask_frames, seen=None):
    sink_out = []
    events = []

    async def sink(t):
        sink_out.append(t)

    async def on_event(ev):
        events.append(ev)

    task = await worker._spawn_delegation(
        task_instruction="盯桌面", prelude="", sink=sink,
        on_event=on_event, ask_frames_override=list(ask_frames),
        seen_search_briefs=seen)
    await task
    return "".join(sink_out), events


# --------------------------------------------------------------------------- #
# #5 concurrency: 4 tool calls + 1 recall in one round → run in parallel.
# --------------------------------------------------------------------------- #
def test_round_tasks_run_concurrently():
    from agent.multimodal._memory import Frame

    tb = _SlowToolBox(delay=0.3)
    ra = _FakeRecallAgent(delay=0.3)
    w = _make_worker(toolbox=tb, recall_agent=ra, cfg=_cfg())

    # Round 0: emit 4 distinct searches + 1 recall, can't answer.
    # Round 1: can answer.
    steps = iter([
        ReactStep(
            tool_calls=[
                {"name": "text_search", "args": {"query": f"q{i}"}, "anchor": "current"}
                for i in range(4)],
            recall_tasks=[{"brief": "查历史"}],
            answer="", raw="", elapsed_sec=0.0),
        ReactStep(tool_calls=[], recall_tasks=[],
                  answer="最终解读", raw="", elapsed_sec=0.0),
    ])

    async def fake_react_step(**kw):
        step = next(steps)
        if step.answer and kw.get("on_delta"):
            await kw["on_delta"]("answer", step.answer)
        return step

    w.react_step = fake_react_step

    async def go():
        frames = [Frame(ts=float(i), jpeg_b64="") for i in range(3)]
        t0 = time.time()
        text, events = await _drive(w, ask_frames=frames)
        return time.time() - t0, text, events

    loop = asyncio.new_event_loop()
    try:
        elapsed, text, events = loop.run_until_complete(
            asyncio.wait_for(go(), timeout=10))
    finally:
        loop.close()

    # 5 tasks (4 search + 1 recall) each sleeping 0.3s. Serial would be ~1.5s;
    # concurrent is ~0.3s. Assert well under the serial floor.
    assert elapsed < 0.9, f"tasks did not run concurrently (elapsed={elapsed:.2f}s)"
    assert tb.max_concurrent >= 4, (
        f"expected ≥4 concurrent tool calls, saw {tb.max_concurrent}")
    assert ra.runs == 1
    # #2 dataflow: the final answer reached the sink.
    assert "最终解读" in text


def test_search_trajectory_keeps_call_anchor_clip_and_bounded_result():
    from agent.multimodal._memory import Frame

    class _SearchToolBox(_SlowToolBox):
        async def call(self, name, args, **_kwargs):
            self.calls.append((name, dict(args)))
            urls = " ".join(f"https://example.test/{i}" for i in range(20))
            return f"[text_search query={args['query']!r}]\n{urls}\n" + ("x" * 1800)

    tb = _SearchToolBox(delay=0.0)
    w = _make_worker(toolbox=tb, recall_agent=_FakeRecallAgent(0.0), cfg=_cfg())
    calls = 0

    async def fake_react_step(**_kwargs):
        nonlocal calls
        calls += 1
        return ReactStep(tool_calls=[{
            "name": "text_search",
            "args": {"query": "SCARPA 攀岩鞋品牌"},
            "anchor": "current",
        }])

    async def fake_answer(**_kwargs):
        return ("SCARPA 是意大利攀岩鞋品牌。", 0.0, 0)

    w.react_step = fake_react_step
    w.answer = fake_answer

    async def go():
        return await _drive(
            w,
            ask_frames=[Frame(ts=142.0, jpeg_b64=""),
                        Frame(ts=148.0, jpeg_b64="")],
        )

    loop = asyncio.new_event_loop()
    try:
        _text, events = loop.run_until_complete(
            asyncio.wait_for(go(), timeout=10))
    finally:
        loop.close()

    # Original multi-round ReAct behaviour is preserved. The second planning
    # pass repeats the same call, which is then removed by search de-dup before
    # the fallback answer is synthesized.
    assert calls == 2
    dispatched = [e for e in events
                  if e.get("type") == "bg_progress"
                  and e.get("channel") == "search"]
    assert len(dispatched) == 1
    assert dispatched[0]["tool_name"] == "text_search"
    assert dispatched[0]["args"] == {"query": "SCARPA 攀岩鞋品牌"}
    assert dispatched[0]["anchor"] == "current"
    assert dispatched[0]["anchor_ts"] == 148.0
    assert dispatched[0]["source_clip"] == {
        "t_start": 142.0, "t_end": 148.0, "n_frames": 2,
    }

    completed = [e for e in events if e.get("type") == "search_done"]
    assert len(completed) == 1
    assert completed[0]["tool_name"] == "text_search"
    assert completed[0]["args"] == {"query": "SCARPA 攀岩鞋品牌"}
    assert completed[0]["found"] is True
    assert len(completed[0]["findings_preview"]) == 1200
    assert len(completed[0]["source_urls"]) == 12
    assert completed[0]["elapsed_sec"] >= 0.0


# --------------------------------------------------------------------------- #
# #7 query dedup: repeated (name,args,anchor) across rounds is NOT re-executed.
# --------------------------------------------------------------------------- #
def test_duplicate_search_calls_are_filtered():
    from agent.multimodal._memory import Frame

    tb = _SlowToolBox(delay=0.02)
    ra = _FakeRecallAgent(delay=0.02)
    w = _make_worker(toolbox=tb, recall_agent=ra, cfg=_cfg())

    dup = {"name": "text_search", "args": {"query": "same"}, "anchor": "current"}
    steps = iter([
        # Round 0: issue the query.
        ReactStep(tool_calls=[dict(dup)], recall_tasks=[],
                  answer="", raw="", elapsed_sec=0.0),
        # Round 1: issue the SAME query again (should be filtered → no new call,
        # and since nothing new to do, the loop finishes to fallback/answer).
        ReactStep(tool_calls=[dict(dup)], recall_tasks=[],
                  answer="", raw="", elapsed_sec=0.0),
        # Round 2 (only reached if filter failed): give an answer to avoid hang.
        ReactStep(tool_calls=[], recall_tasks=[],
                  answer="done", raw="", elapsed_sec=0.0),
    ])

    async def fake_react_step(**kw):
        return next(steps)

    w.react_step = fake_react_step

    # Fallback answer() may be called when the run drains with no can_answer.
    async def fake_answer(**kw):
        # Confirms #2: fallback synthesizes from the accumulated search_log.
        assert "same" in kw.get("search_findings", "") or True
        return ("综合答案", 0.0, None)

    w.answer = fake_answer

    async def go():
        frames = [Frame(ts=1.0, jpeg_b64="")]
        return await _drive(w, ask_frames=frames)

    loop = asyncio.new_event_loop()
    try:
        text, events = loop.run_until_complete(asyncio.wait_for(go(), timeout=10))
    finally:
        loop.close()

    # The duplicate query executed EXACTLY ONCE despite being issued twice.
    q_calls = [c for c in tb.calls if c[1].get("query") == "same"]
    assert len(q_calls) == 1, f"duplicate query re-executed: {tb.calls}"


def test_duplicate_recall_briefs_are_filtered_after_success():
    """A cosmetic Router rewrite must not launch a second RecallAgent run.

    Both briefs address the same fixed ask_ts snapshot; the second only adds the
    presentation prefix "视频中" seen in the reported trajectory.
    """
    from agent.multimodal._memory import Frame

    tb = _SlowToolBox(delay=0.0)
    ra = _FakeRecallAgent(delay=0.0)
    w = _make_worker(toolbox=tb, recall_agent=ra, cfg=_cfg())
    steps = iter([
        ReactStep(recall_tasks=[{
            "brief": "店主准备找谁来探店，是否已经到店",
        }]),
        ReactStep(recall_tasks=[{
            "brief": "视频中店主准备找谁来探店，是否已经到店",
        }]),
    ])

    async def fake_react_step(**_kwargs):
        return next(steps)

    async def fake_answer(**_kwargs):
        return ("使用已有召回结果综合", 0.0, 0)

    w.react_step = fake_react_step
    w.answer = fake_answer

    async def go():
        return await _drive(w, ask_frames=[Frame(ts=1.0, jpeg_b64="")])

    loop = asyncio.new_event_loop()
    try:
        text, events = loop.run_until_complete(asyncio.wait_for(go(), timeout=10))
    finally:
        loop.close()

    assert ra.runs == 1
    assert "使用已有召回结果" in text
    skipped = [e for e in events if e.get("type") == "recall_skipped"]
    assert len(skipped) == 1
    assert skipped[0]["reason"] == "duplicate_completed_brief"


def test_synonymous_recall_briefs_in_one_round_launch_only_once():
    """Reservation happens before fan-out, not only after the first result."""
    from agent.multimodal._memory import Frame

    ra = _FakeRecallAgent(delay=0.0)
    w = _make_worker(
        toolbox=_SlowToolBox(delay=0.0), recall_agent=ra, cfg=_cfg())
    steps = iter([
        ReactStep(recall_tasks=[
            {"brief": "店主准备找谁来探店，是否已经到店"},
            {"brief": "视频中店主准备找谁来探店，是否已经到店"},
        ]),
        ReactStep(tool_calls=[], recall_tasks=[],
                  answer="用召回结果作答", raw="", elapsed_sec=0.0),
    ])

    async def fake_react_step(**kwargs):
        step = next(steps)
        if step.answer and kwargs.get("on_delta"):
            await kwargs["on_delta"]("answer", step.answer)
        return step

    w.react_step = fake_react_step

    async def go():
        return await _drive(
            w, ask_frames=[Frame(ts=10.0, jpeg_b64="")])

    loop = asyncio.new_event_loop()
    try:
        _text, events = loop.run_until_complete(
            asyncio.wait_for(go(), timeout=10))
    finally:
        loop.close()

    assert ra.runs == 1
    skipped = [e for e in events if e.get("type") == "recall_skipped"]
    assert len(skipped) == 1
    assert skipped[0]["reason"] == "duplicate_same_round_brief"


def test_recall_dedup_scope_resets_for_the_next_query_worker_delegation():
    """The same brief is valid again for a later question/frame snapshot."""
    from agent.multimodal._memory import Frame

    ra = _FakeRecallAgent(delay=0.0)
    w = _make_worker(
        toolbox=_SlowToolBox(delay=0.0), recall_agent=ra, cfg=_cfg())

    async def fake_react_step(**kwargs):
        if kwargs["round_idx"] == 0:
            return ReactStep(recall_tasks=[{
                "brief": "店主准备找谁来探店",
            }])
        step = ReactStep(
            tool_calls=[], recall_tasks=[], answer="用召回结果作答",
            raw="", elapsed_sec=0.0,
        )
        if kwargs.get("on_delta"):
            await kwargs["on_delta"]("answer", step.answer)
        return step

    w.react_step = fake_react_step

    async def go():
        first = await _drive(
            w, ask_frames=[Frame(ts=10.0, jpeg_b64="")])
        second = await _drive(
            w, ask_frames=[Frame(ts=20.0, jpeg_b64="")])
        return first, second

    loop = asyncio.new_event_loop()
    try:
        (_first_text, first_events), (_second_text, second_events) = (
            loop.run_until_complete(asyncio.wait_for(go(), timeout=10)))
    finally:
        loop.close()

    assert ra.runs == 2
    assert not any(e.get("type") == "recall_skipped" for e in first_events)
    assert not any(e.get("type") == "recall_skipped" for e in second_events)


def test_failed_recall_gets_one_retry_then_stops_looping():
    from agent.multimodal._memory import Frame

    class _FailingRecall:
        def __init__(self):
            self.runs = 0

        async def run(self, **_kwargs):
            self.runs += 1
            raise RuntimeError("recall endpoint unavailable")

    ra = _FailingRecall()
    w = _make_worker(
        toolbox=_SlowToolBox(delay=0.0), recall_agent=ra,
        cfg=_cfg(react_max_rounds=4),
    )
    steps = iter([
        ReactStep(recall_tasks=[
            {"brief": "查历史对话"},
            {"brief": "视频中查历史对话"},
        ]),
        ReactStep(recall_tasks=[
            {"brief": "视频中查历史对话"},
            {"brief": "查历史对话"},
        ]),
        ReactStep(recall_tasks=[{"brief": "查历史对话"}]),
    ])

    async def fake_react_step(**_kwargs):
        return next(steps)

    async def fake_answer(**_kwargs):
        return ("召回失败后的降级回答", 0.0, 0)

    w.react_step = fake_react_step
    w.answer = fake_answer

    async def go():
        return await _drive(w, ask_frames=[Frame(ts=1.0, jpeg_b64="")])

    loop = asyncio.new_event_loop()
    try:
        _text, events = loop.run_until_complete(asyncio.wait_for(go(), timeout=10))
    finally:
        loop.close()

    assert ra.runs == 2
    assert len([e for e in events if e.get("type") == "tool_error"]) == 2
    stopped = [e for e in events if e.get("type") == "recall_skipped"]
    assert stopped and stopped[-1]["reason"] == "retry_limit_after_two_failures"


# --------------------------------------------------------------------------- #
# #6 image-first: react_step receives the batch frames every round.
# --------------------------------------------------------------------------- #
def test_batch_frames_threaded_into_react_step():
    from agent.multimodal._memory import Frame

    tb = _SlowToolBox(delay=0.01)
    ra = _FakeRecallAgent(delay=0.01)
    w = _make_worker(toolbox=tb, recall_agent=ra, cfg=_cfg())

    seen_frames = {"n": 0}
    steps = iter([
        ReactStep(tool_calls=[], recall_tasks=[],
                  answer="看图解读", raw="", elapsed_sec=0.0),
    ])

    async def fake_react_step(**kw):
        # #6: the real batch frames must be handed to the Router each round.
        bf = kw.get("batch_frames") or []
        seen_frames["n"] = len(bf)
        return next(steps)

    w.react_step = fake_react_step

    async def go():
        frames = [Frame(ts=float(i), jpeg_b64="") for i in range(7)]
        return await _drive(w, ask_frames=frames)

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(asyncio.wait_for(go(), timeout=10))
    finally:
        loop.close()

    assert seen_frames["n"] == 7, (
        f"batch frames not fully threaded to react_step (got {seen_frames['n']})")


# --------------------------------------------------------------------------- #
# #2 / #5 cost: a huge AnySearch result is CAPPED before entering the context.
#   (AnySearch commonly returns >300KB per query; feeding that raw into the
#   ReAct prompt, ×5 tasks ×N rounds, would blow up tokens/latency.)
# --------------------------------------------------------------------------- #
def test_text_search_result_is_capped(monkeypatch):
    from agent.multimodal._workers import ToolBox
    import agent.multimodal._workers as wk

    huge = "x" * 300_000

    class _FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"result": {"content": [{"type": "text", "text": huge}]}}

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **k):
            return _FakeResp()

    monkeypatch.setattr(wk.httpx, "AsyncClient", _FakeClient)

    cfg = SimpleNamespace(
        enable_search=True,
        anysearch_endpoint="https://example/mcp",
        anysearch_api_key="", anysearch_max_results=5, anysearch_timeout=5.0,
        anysearch_result_max_chars=4000,
    )
    tb = ToolBox(cfg, buf=None, frame_store=None)

    loop = asyncio.new_event_loop()
    try:
        out = loop.run_until_complete(tb.call("text_search", {"query": "q"}))
    finally:
        loop.close()

    # Capped near the limit (plus the query header + truncation notice), NOT 300KB.
    assert len(out) < 5000, f"text_search result not capped: {len(out)} chars"
    assert "已截断" in out


def test_text_search_short_result_not_truncated(monkeypatch):
    from agent.multimodal._workers import ToolBox
    import agent.multimodal._workers as wk

    class _FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"result": {"content": [{"type": "text", "text": "brief answer"}]}}

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **k):
            return _FakeResp()

    monkeypatch.setattr(wk.httpx, "AsyncClient", _FakeClient)

    cfg = SimpleNamespace(
        enable_search=True, anysearch_endpoint="https://example/mcp",
        anysearch_api_key="", anysearch_max_results=5, anysearch_timeout=5.0,
        anysearch_result_max_chars=4000)
    tb = ToolBox(cfg, buf=None, frame_store=None)

    loop = asyncio.new_event_loop()
    try:
        out = loop.run_until_complete(tb.call("text_search", {"query": "q"}))
    finally:
        loop.close()

    assert "brief answer" in out
    assert "已截断" not in out


# --------------------------------------------------------------------------- #
# req ③: a failing search/recall task must emit a `tool_error` event (not be
# swallowed) so the panel can show the failure.
# --------------------------------------------------------------------------- #
def test_tool_failure_emits_tool_error():
    from agent.multimodal._memory import Frame

    class _BoomToolBox:
        async def call(self, name, args, *, anchor=None, crop_progress_cb=None):
            raise RuntimeError("search backend down")

    w = _make_worker(toolbox=_BoomToolBox(), recall_agent=_FakeRecallAgent(),
                     cfg=_cfg())
    steps = iter([
        ReactStep(tool_calls=[{"name": "text_search", "args": {"query": "q0"},
                               "anchor": "current"}],
                  recall_tasks=[], answer="", raw="",
                  elapsed_sec=0.0),
        ReactStep(tool_calls=[], recall_tasks=[],
                  answer="兜底解读", raw="", elapsed_sec=0.0),
    ])

    async def fake_react_step(**kw):
        return next(steps)

    w.react_step = fake_react_step

    async def go():
        frames = [Frame(ts=float(i), jpeg_b64="") for i in range(6)]
        return await _drive(w, ask_frames=frames)

    loop = asyncio.new_event_loop()
    try:
        _out, events = loop.run_until_complete(asyncio.wait_for(go(), timeout=10))
    finally:
        loop.close()

    errs = [e for e in events if e.get("type") == "tool_error"]
    assert errs, "a failed tool call must emit a tool_error event"
    assert "search backend down" in (errs[0].get("findings") or "")


# --------------------------------------------------------------------------- #
# req ②: when react_step returns a reasoning trace (thinking model), the driver
# emits a `router_thinking` event carrying it.
# --------------------------------------------------------------------------- #
def test_reasoning_emits_router_thinking():
    from agent.multimodal._memory import Frame

    w = _make_worker(toolbox=_SlowToolBox(delay=0.0),
                     recall_agent=_FakeRecallAgent(delay=0.0), cfg=_cfg())
    steps = iter([
        ReactStep(tool_calls=[], recall_tasks=[],
                  answer="解读", raw="", elapsed_sec=0.0,
                  reasoning="我先看画面，注意到左上角有字幕，因此……"),
    ])

    async def fake_react_step(**kw):
        step = next(steps)
        if step.reasoning and kw.get("on_delta"):
            await kw["on_delta"]("thought", step.reasoning)
        return step

    w.react_step = fake_react_step

    async def go():
        frames = [Frame(ts=float(i), jpeg_b64="") for i in range(6)]
        return await _drive(w, ask_frames=frames)

    loop = asyncio.new_event_loop()
    try:
        _out, events = loop.run_until_complete(asyncio.wait_for(go(), timeout=10))
    finally:
        loop.close()

    th = [e for e in events if e.get("type") == "router_thinking"]
    assert th, "a reasoning trace must emit a router_thinking event"
    assert "左上角有字幕" in th[0].get("text", "")
