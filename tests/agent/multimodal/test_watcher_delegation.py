"""Verification of the deep-research delegation orchestration (十项优化 #2/#5/#7).

Drives WatcherWorker._spawn_delegation with a fake react_step + fake ToolBox
(no model, no network) to prove:

  #5  并发性能 — all tool_calls in one ReAct round are dispatched CONCURRENTLY
      (asyncio.create_task + gather), not serialized. We prove it by making each
      fake tool sleep and asserting the wall-clock ≈ one tool, not the sum.

  #2  数据流程 — the end-to-end pipe within a delegation: react_step ->
      toolbox.call(name,args) -> findings accumulate -> answer text streams
      through the sink -> the assistant turn is appended to the conversation.

  #7  搜索 query 生成 (cross-batch dedup) — a search tool_call whose (name,args,
      anchor) was already issued in an earlier batch (shared seen_search_briefs
      set) is filtered out and NOT re-executed.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from agent.multimodal._config import Config
from agent.multimodal._memory import Frame, FrameBuffer
from agent.multimodal._workers import WatcherWorker, ReactStep


def _worker(*, toolbox, react_scripts, cfg_over=None):
    """A WatcherWorker with only the surface _spawn_delegation touches."""
    cfg = Config()
    for k, v in (cfg_over or {}).items():
        setattr(cfg, k, v)

    w = WatcherWorker.__new__(WatcherWorker)
    w.cfg = cfg
    w.buf = FrameBuffer(cfg)
    w.toolbox = toolbox
    w.recall_agent = None
    w.inflight = {}
    w.frame_store = None
    # merge(dev_0807): WatcherWorker 新增 SearchFactStore (self.store), delegation
    #   在 answer deliver 后 upsert_many 提交 search fact 候选。stub 一个空实现。
    w.store = SimpleNamespace(
        upsert_many=lambda facts: SimpleNamespace(version=1, keys=[]))
    w.conversation = SimpleNamespace(
        appended=[],
        append=lambda role, text, rel_ts=None: _async_none(
            w.conversation.appended.append((role, text))),
    )

    # Script react_step: return the next ReactStep per call.
    scripts = list(react_scripts)
    calls = {"n": 0}

    async def _fake_react_step(**kw):
        i = calls["n"]
        calls["n"] += 1
        # Expose what the loop passed us (esp. seen_search_briefs) for assertions.
        _fake_react_step.last_kwargs = kw
        step = scripts[i] if i < len(scripts) else ReactStep(answer="兜底")
        # merge(dev_0807): 真 react_step 是【流式】—— answer 正文通过 on_delta("answer",..)
        #   逐 token 吐给 sink, 不是从返回值 step.answer 取。收尾轮(无工具、有 answer)
        #   在此模拟流式吐出, 否则 sink 收不到内容。
        _on_delta = kw.get("on_delta")
        if _on_delta is not None and step.answer and not (
                step.tool_calls or step.recall_tasks):
            await _on_delta("answer", step.answer)
        return step

    w.react_step = _fake_react_step
    w._react_calls = calls

    # answer() fallback should not be needed (we always supply react_answer),
    # but stub it so a miss is obvious rather than an AttributeError.
    async def _fake_answer(**kw):
        return ("[fallback answer]", 0.0, None)

    w.answer = _fake_answer
    return w


def _async_none(_ignored=None):
    async def _c():
        return None
    return _c()


class _FakeToolBox:
    """Records concurrency: each call marks its start/end so we can detect
    overlap, and counts how many times it actually executed."""

    def __init__(self, *, per_call_sleep=0.0):
        self._sleep = per_call_sleep
        self.calls = []          # list of (name, args)
        self._active = 0
        self.max_concurrent = 0

    async def call(self, name, args, *, anchor=None, crop_progress_cb=None):
        self.calls.append((name, dict(args or {})))
        self._active += 1
        self.max_concurrent = max(self.max_concurrent, self._active)
        try:
            if self._sleep:
                await asyncio.sleep(self._sleep)
            q = (args or {}).get("query", "")
            return f"[text_search query={q!r}]\nfindings for {q}"
        finally:
            self._active -= 1


class _Sink:
    def __init__(self):
        self.text = ""

    async def __call__(self, token):
        self.text += token or ""


def _run(coro, timeout=10.0):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(asyncio.wait_for(coro, timeout=timeout))
    finally:
        loop.close()


async def _drive(worker, *, sink, seen=None, frames=None):
    events = []

    async def _on_event(ev):
        events.append(ev)

    task = await worker._spawn_delegation(
        task_instruction="盯桌面", prelude="",
        sink=sink, on_event=_on_event,
        ask_frames_override=frames or [Frame(ts=0.0, jpeg_b64="")],
        seen_search_briefs=seen,
    )
    await task
    return events


# --------------------------------------------------------------------------- #
# #5 concurrency: 3 tools each sleeping 0.3s run in ~0.3s, not ~0.9s.
# --------------------------------------------------------------------------- #
def test_tool_calls_run_concurrently():
    tb = _FakeToolBox(per_call_sleep=0.3)
    # Round 0: three DISTINCT searches; round 1: no tools → wrap up with answer.
    react = [
        ReactStep(tool_calls=[
            {"name": "text_search", "args": {"query": "A"}, "anchor": "current"},
            {"name": "text_search", "args": {"query": "B"}, "anchor": "current"},
            {"name": "text_search", "args": {"query": "C"}, "anchor": "current"},
        ]),
        ReactStep(answer="综合解读: A/B/C"),
    ]
    w = _worker(toolbox=tb, react_scripts=react)
    sink = _Sink()

    t0 = time.time()
    _run(_drive(w, sink=sink))
    elapsed = time.time() - t0

    assert tb.max_concurrent == 3, (
        f"tools did not run concurrently (max_concurrent={tb.max_concurrent})")
    # Concurrent: ~0.3s. Serial would be ~0.9s. Generous ceiling for CI jitter.
    assert elapsed < 0.7, f"looks serialized: {elapsed:.2f}s for 3×0.3s tools"
    assert len(tb.calls) == 3


# --------------------------------------------------------------------------- #
# #2 dataflow: react -> toolbox -> findings -> answer -> sink -> conversation.
# --------------------------------------------------------------------------- #
def test_dataflow_react_to_sink_and_conversation():
    tb = _FakeToolBox()
    react = [
        ReactStep(tool_calls=[
            {"name": "text_search", "args": {"query": "甘道夫"}, "anchor": "current"}]),
        ReactStep(answer="这段画面在讲甘道夫的背景。"),
    ]
    w = _worker(toolbox=tb, react_scripts=react)
    sink = _Sink()

    events = _run(_drive(w, sink=sink))

    # toolbox executed the planned search
    assert ("text_search", {"query": "甘道夫"}) in tb.calls
    # answer streamed through the sink
    assert "甘道夫的背景" in sink.text
    # assistant turn recorded to the conversation log
    assert any(role == "assistant" and "甘道夫的背景" in txt
               for role, txt in w.conversation.appended)
    # a search_done and an answer_ready event were emitted (UI dataflow)
    assert any(e.get("type") == "search_done" for e in events)
    assert any(e.get("type") == "answer_ready" for e in events)


# --------------------------------------------------------------------------- #
# #7 cross-batch dedup: a repeat search across batches is filtered, not re-run.
# --------------------------------------------------------------------------- #
def test_cross_batch_search_dedup():
    shared_seen = set()

    # ---- Batch 1: issues search "X" (distinct → executes) ----
    tb1 = _FakeToolBox()
    react1 = [
        ReactStep(tool_calls=[
            {"name": "text_search", "args": {"query": "X"}, "anchor": "current"}]),
        ReactStep(answer="batch1 解读"),
    ]
    w1 = _worker(toolbox=tb1, react_scripts=react1)
    _run(_drive(w1, sink=_Sink(), seen=shared_seen))
    assert tb1.calls == [("text_search", {"query": "X"})]
    assert shared_seen, "batch1 should have recorded its issued search in the shared set"

    # ---- Batch 2: SAME search "X" first (deduped → skipped), then "Y" (new) ----
    tb2 = _FakeToolBox()
    react2 = [
        ReactStep(tool_calls=[
            {"name": "text_search", "args": {"query": "X"}, "anchor": "current"},  # dup
            {"name": "text_search", "args": {"query": "Y"}, "anchor": "current"},  # new
        ]),
        ReactStep(answer="batch2 解读"),
    ]
    w2 = _worker(toolbox=tb2, react_scripts=react2)
    _run(_drive(w2, sink=_Sink(), seen=shared_seen))

    # "X" must NOT be re-executed in batch 2; only the new "Y" runs.
    assert ("text_search", {"query": "X"}) not in tb2.calls, (
        "duplicate cross-batch search was re-executed (dedup failed)")
    assert ("text_search", {"query": "Y"}) in tb2.calls


# --------------------------------------------------------------------------- #
# #5/#2 belt-and-braces: a tool that raises does NOT crash the delegation; the
# answer still streams (graceful concurrency error handling).
# --------------------------------------------------------------------------- #
def test_tool_exception_does_not_break_delegation():
    class _BoomBox(_FakeToolBox):
        async def call(self, name, args, *, anchor=None, crop_progress_cb=None):
            self.calls.append((name, dict(args or {})))
            raise RuntimeError("tool blew up")

    tb = _BoomBox()
    react = [
        ReactStep(tool_calls=[
            {"name": "text_search", "args": {"query": "Z"}, "anchor": "current"}]),
        ReactStep(answer="即便搜索失败也基于画面给出解读。"),
    ]
    w = _worker(toolbox=tb, react_scripts=react)
    sink = _Sink()
    _run(_drive(w, sink=sink))
    assert "基于画面" in sink.text  # delegation survived the tool error
