"""MemoryWriter must not be gated on anything recall does.

This is the regression that cost 140s of a 217s answer: the writer and recall
shared one mutex, so each of recall's 5-9 steps waited out a full ~28s writer
batch. The fix is not a better queue — it is that the writer takes no LLM gate at
all, and recall's ceiling is recall's own business.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

from agent.multimodal._workers import RecallConcurrencyLimiter
from agent.multimodal.memory_backend import MemoryBackend


class _Writer:
    def __init__(self):
        self.wakes = 0
        self.watchdogs = 0

    async def wake_once(self, on_progress=None):
        self.wakes += 1
        await asyncio.sleep(0)

    async def try_watchdog_seal(self):
        self.watchdogs += 1
        return ""


def _backend(*, frames: int = 3):
    backend = MemoryBackend(SimpleNamespace(size=frames),
                            session_id="writer-independent")
    backend.cfg = SimpleNamespace(
        writer_wake_interval=0.01,
        memory_max_consecutive_failures=5,
        recall_max_concurrency=1,
    )
    backend.memory_writer = _Writer()
    backend._stop = threading.Event()
    backend._push_ctx = lambda: None
    backend._recall_limiter = RecallConcurrencyLimiter(concurrency=1)
    return backend


async def _drive(backend, *, saturate_recall: bool, cycles: int = 4):
    """Run _writer_loop a few cycles, optionally with recall fully saturated."""
    release = asyncio.Event()
    holder = None
    if saturate_recall:
        async def recall_step():
            # concurrency=1, so this holds the ENTIRE recall budget.
            async with backend._recall_limiter.slot("decide"):
                await release.wait()

        holder = asyncio.create_task(recall_step())
        await asyncio.sleep(0)

    loop_task = asyncio.create_task(backend._writer_loop())
    await asyncio.sleep(0.02 * cycles + 0.05)
    backend._stop.set()
    release.set()
    try:
        await asyncio.wait_for(loop_task, timeout=1.0)
    except asyncio.TimeoutError:
        loop_task.cancel()
    if holder is not None:
        await holder


def test_the_writer_keeps_writing_while_recall_is_saturated():
    backend = _backend()
    asyncio.run(_drive(backend, saturate_recall=True))
    assert backend.memory_writer.wakes > 0, (
        "the writer waited on recall — the shared-gate regression is back")


def test_the_writer_rate_is_not_materially_affected_by_recall():
    """Same wall clock, comparable wake counts, with and without recall running."""
    idle = _backend()
    asyncio.run(_drive(idle, saturate_recall=False))
    busy = _backend()
    asyncio.run(_drive(busy, saturate_recall=True))
    # Deliberately a ratio, not an exact count: the number of wakes in a fixed
    # window is scheduler-dependent, so an exact comparison would be flaky
    # without testing anything stronger. The regression this guards against
    # produced 0 vs N, so half the idle rate is a wide enough margin to stay
    # stable and still fail loudly if the gate comes back.
    assert idle.memory_writer.wakes > 0
    assert busy.memory_writer.wakes >= idle.memory_writer.wakes / 2, (
        idle.memory_writer.wakes, busy.memory_writer.wakes)


def test_an_empty_frame_buffer_still_seals_segments_via_the_watchdog():
    backend = _backend(frames=0)
    asyncio.run(_drive(backend, saturate_recall=False))
    assert backend.memory_writer.wakes == 0
    assert backend.memory_writer.watchdogs > 0
