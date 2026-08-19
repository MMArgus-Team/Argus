"""Regression guards for ConversationLog cross-loop lock safety (finding C5).

ConversationLog is shared between the MemoryBackend event loop and the
WatcherAgent event loop (WatcherAgent reuses ``mb.conversation``). It used an
``asyncio.Lock``, which is only valid within the loop that created it and gives
no protection across two different loops/threads; the read methods were
unlocked entirely. The fix switches to a ``threading.RLock`` and locks every
read/write of ``_turns``.

These tests are pure in-memory (no cloud, no hardware).
"""
import asyncio
import threading

from agent.multimodal._memory import ConversationLog


def test_lock_is_threading_not_asyncio():
    """_lock must be a cross-thread primitive, not an asyncio.Lock."""
    log = ConversationLog()
    # asyncio.Lock instances are not usable across loops; assert we're not one.
    assert not isinstance(log._lock, type(asyncio.Lock()))
    # threading.RLock() returns a private type whose repr mentions RLock.
    assert "RLock" in repr(log._lock)


def test_append_is_still_awaitable_and_records():
    """append keeps its async def signature so `await log.append(...)` works."""
    log = ConversationLog()

    async def _go():
        await log.append("user", "hi")
        await log.append("assistant", "there")

    asyncio.run(_go())
    snap = log.snapshot()
    assert [t.content for t in snap] == ["hi", "there"]
    assert [t.role for t in snap] == ["user", "assistant"]


def test_reset_and_reads_under_lock():
    log = ConversationLog()

    async def _seed():
        for i in range(5):
            await log.append("user", f"m{i}")

    asyncio.run(_seed())
    assert len(log.snapshot()) == 5
    log.reset()
    assert log.snapshot() == []
    assert log.as_dump_text() == "(对话尚未开始)"
    assert log.recent_n(5) == []
    assert log.latest_obs(3) == []
    assert log.latest_audio_obs(3) == []


def test_concurrent_append_and_reads_do_not_crash():
    """Reproduce the original 'two loops share one instance' scenario.

    Thread A hammers append() on its own loop; thread B concurrently calls the
    read methods that iterate _turns. Before the fix, reads could observe
    _turns being wholesale-replaced mid-iteration (RuntimeError / lost data);
    with the RLock every read/write is serialized. Assert no exception escapes
    and every appended turn survives (max_chars/max_bg_obs kept large enough
    that trimming never drops normal turns).
    """
    # Large caps so trimming never fires and the final count is deterministic.
    log = ConversationLog(max_chars=10_000_000, min_turns=1, max_bg_obs=10_000_000)
    N = 500
    errors: list[BaseException] = []

    def writer():
        async def _go():
            for i in range(N):
                await log.append("user", f"w{i}")
        try:
            asyncio.run(_go())
        except BaseException as e:  # noqa: BLE001 - capture for assertion
            errors.append(e)

    def reader():
        try:
            for _ in range(N):
                log.snapshot()
                log.as_dump_text()
                log.latest_audio_obs(50)
                log.recent_n(20)
        except BaseException as e:  # noqa: BLE001 - capture for assertion
            errors.append(e)

    ta = threading.Thread(target=writer)
    tb = threading.Thread(target=reader)
    ta.start(); tb.start()
    ta.join(); tb.join()

    assert not errors, f"concurrent access raised: {errors!r}"
    assert len(log.snapshot()) == N
