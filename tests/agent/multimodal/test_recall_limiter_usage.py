"""How RecallAgent uses its concurrency limiter, and what it reports.

The contract that matters here is a negative one: the limiter gates recall only.
MemoryWriter must not appear in it at all — a writer batch is ~28s and a recall
step is 6-12s, and coupling them is what made a blocking recall pay a full writer
batch at every step (140s of one 217s answer).
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

from agent.multimodal._config import Config
from agent.multimodal._workers import RecallAgent


class _RecordingLimiter:
    """Limiter-shaped stand-in that reports a fixed admission wait."""

    def __init__(self, waited_ms=0.0):
        self.waited_ms = waited_ms
        self.limit = 3
        self.calls = []

    @asynccontextmanager
    async def slot(self, tag=""):
        self.calls.append(tag)
        yield self.waited_ms


def _agent(limiter):
    cfg = Config()
    recall = RecallAgent(cfg, MagicMock(), MagicMock(), MagicMock(),
                         model="test-model")
    recall.recall_limiter = limiter
    return recall


def _run_one_step(recall, *, progress=None):
    from agent.multimodal import _workers

    async def scenario():
        _workers._RECALL_PROGRESS.set(progress)
        async with recall._channel_ctx("decide_r0"):
            pass

    asyncio.run(scenario())


def test_each_step_takes_one_recall_slot():
    limiter = _RecordingLimiter()
    _run_one_step(_agent(limiter))
    # Tagged by stage so a wait can be attributed to a specific step.
    assert limiter.calls == ["recall.decide_r0"]


def test_the_limiter_gates_recall_only_and_takes_no_priority_argument():
    """A slot() that accepted a priority would mean writer/recall share a queue."""
    import inspect

    from agent.multimodal._workers import RecallConcurrencyLimiter

    params = inspect.signature(RecallConcurrencyLimiter.slot).parameters
    assert set(params) == {"self", "tag"}, params
    # And nothing about the writer leaks into the limiter's surface.
    surface = dir(RecallConcurrencyLimiter)
    assert not [name for name in surface if "foreground" in name.lower()]
    assert not [name for name in surface if "writer" in name.lower()]


def test_a_long_admission_wait_is_reported_to_the_caller():
    events = []

    async def progress(event):
        events.append(event)

    _run_one_step(_agent(_RecordingLimiter(waited_ms=8_000.0)),
                  progress=progress)

    assert len(events) == 1
    assert events[0]["phase"] == "channel_wait"
    assert events[0]["stage"] == "decide_r0"
    assert events[0]["waited_ms"] == 8_000


def test_a_negligible_wait_is_not_reported():
    events = []

    async def progress(event):
        events.append(event)

    _run_one_step(_agent(_RecordingLimiter(waited_ms=12.0)), progress=progress)
    assert events == []


def test_a_plain_acquire_release_gate_still_works():
    """MemoryBackend may install any gate object; tests install doubles."""

    class _Gate:
        def __init__(self):
            self.acquires = 0
            self.releases = 0

        async def acquire(self):
            self.acquires += 1

        def release(self):
            self.releases += 1

    gate = _Gate()
    _run_one_step(_agent(gate))
    assert (gate.acquires, gate.releases) == (1, 1)


def test_no_limiter_installed_is_a_passthrough():
    _run_one_step(_agent(None))  # must not raise
