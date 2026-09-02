"""Recall's own concurrency ceiling.

Writer and recall are independent — neither waits for the other. What is bounded
is recall's own fan-out: one ReAct round can spawn several recall tasks and
several queries can be in flight, with no cap anywhere on that path, so a burst
could otherwise put an unbounded number of requests on one endpoint.
"""

from __future__ import annotations

import asyncio

from agent.multimodal._workers import RecallConcurrencyLimiter


def test_steps_up_to_the_limit_run_concurrently():
    async def scenario():
        limiter = RecallConcurrencyLimiter(concurrency=3)
        peak = {"value": 0}

        async def step():
            async with limiter.slot("decide") as waited_ms:
                peak["value"] = max(peak["value"], limiter.in_flight)
                await asyncio.sleep(0.05)
                return waited_ms

        waits = await asyncio.gather(*[step() for _ in range(3)])
        return peak["value"], waits

    peak, waits = asyncio.run(scenario())
    assert peak == 3
    # Nobody queued: three is within the ceiling.
    assert all(w < 20 for w in waits), waits


def test_a_burst_past_the_limit_queues_instead_of_piling_onto_the_endpoint():
    async def scenario():
        limiter = RecallConcurrencyLimiter(concurrency=2)
        peak = {"value": 0}

        async def step():
            async with limiter.slot("decide") as waited_ms:
                peak["value"] = max(peak["value"], limiter.in_flight)
                await asyncio.sleep(0.05)
                return waited_ms

        waits = await asyncio.gather(*[step() for _ in range(5)])
        return peak["value"], waits

    peak, waits = asyncio.run(scenario())
    assert peak == 2, peak
    # All five ran (nothing dropped), but three of them had to wait for a slot.
    assert len(waits) == 5
    assert sum(1 for w in waits if w >= 40) == 3, waits


def test_the_default_is_the_recommended_ceiling():
    """QueryWorker shows no tendency to batch, so the default does not throttle.

    Field logs: every ReAct round dispatched zero or exactly one recall, always
    in round 0. The limiter is insurance against an unobserved fan-out, so its
    default should not sit below what the observed traffic could ever need.
    """
    assert RecallConcurrencyLimiter().limit == 5
    assert RecallConcurrencyLimiter.RECOMMENDED_MAX_CONCURRENCY == 5


def test_a_value_above_the_recommendation_is_honoured_with_a_warning():
    """The ceiling is advice, not policy: silently clamping an operator's number
    down to one they did not ask for hides the setting instead of respecting it."""
    import logging

    from agent.multimodal import _workers

    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Capture()
    _workers.log.addHandler(handler)
    try:
        limiter = RecallConcurrencyLimiter(concurrency=12)
    finally:
        _workers.log.removeHandler(handler)

    assert limiter.limit == 12, "the configured value must not be clamped"
    warnings = [r for r in records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1, records
    assert "exceeds the recommended max" in warnings[0].getMessage()


def test_a_nonsense_value_degrades_to_serial_not_to_unlimited():
    # 0 permits would deadlock every recall, so this bound stays enforced.
    assert RecallConcurrencyLimiter(concurrency=0).limit == 1
    assert RecallConcurrencyLimiter(concurrency=-4).limit == 1


def test_slots_are_returned_when_a_step_raises():
    async def scenario():
        limiter = RecallConcurrencyLimiter(concurrency=1)
        for _ in range(3):
            try:
                async with limiter.slot("decide"):
                    raise RuntimeError("model failed")
            except RuntimeError:
                pass
        # A leaked slot would deadlock the next acquire, not just miscount.
        async with limiter.slot("decide"):
            pass
        return limiter.in_flight

    assert asyncio.run(asyncio.wait_for(scenario(), timeout=2.0)) == 0


def test_slots_are_returned_when_a_step_is_cancelled():
    async def scenario():
        limiter = RecallConcurrencyLimiter(concurrency=1)
        held = asyncio.Event()
        release = asyncio.Event()

        async def holder():
            async with limiter.slot("decide"):
                held.set()
                await release.wait()

        first = asyncio.create_task(holder())
        await held.wait()

        async def waiter():
            async with limiter.slot("decide"):
                pass

        queued = asyncio.create_task(waiter())
        await asyncio.sleep(0.02)
        queued.cancel()
        try:
            await queued
        except asyncio.CancelledError:
            pass
        release.set()
        await first
        return limiter.in_flight

    assert asyncio.run(asyncio.wait_for(scenario(), timeout=2.0)) == 0
