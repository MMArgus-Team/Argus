"""MonitorEngine lifecycle contracts: bounded teardown and no ghost spawn."""

import threading

from agent.multimodal.monitor_engine import MonitorEngine


class _FakeClient:
    def __init__(self, *, owned: bool):
        self._hermes_submodule_owned = owned
        self.close_calls = 0

    async def close(self):
        self.close_calls += 1


class _LifecycleEngine(MonitorEngine):
    def __init__(self, client):
        super().__init__(None, monitors_ref={}, sid="sid_lifecycle")
        self._test_client = client

    def _build(self):
        self.client = self._test_client
        self.model = "test-monitor"
        return True


def test_stop_joins_thread_and_closes_owned_client_once():
    client = _FakeClient(owned=True)
    engine = _LifecycleEngine(client)

    engine.start()
    assert engine.is_healthy()
    assert engine.stop(timeout=2.0) is True

    assert engine._thread is not None
    assert not engine._thread.is_alive()
    assert engine._loop is None
    assert engine.is_healthy() is False
    assert client.close_calls == 1
    assert engine.stop(timeout=0.1) is True
    assert client.close_calls == 1


def test_stop_never_closes_shared_client():
    client = _FakeClient(owned=False)
    engine = _LifecycleEngine(client)

    engine.start()
    assert engine.stop(timeout=2.0) is True

    assert client.close_calls == 0


def test_start_does_not_publish_health_before_loop_dispatch(monkeypatch):
    from agent.multimodal import monitor_agent

    entered = threading.Event()
    release = threading.Event()

    def _blocked_reconcile(*_args, **_kwargs):
        entered.set()
        assert release.wait(timeout=2.0)
        return 0

    monkeypatch.setattr(monitor_agent, "reconcile_stale", _blocked_reconcile)
    engine = _LifecycleEngine(_FakeClient(owned=True))
    starter = threading.Thread(target=engine.start)
    starter.start()

    assert entered.wait(timeout=1.0)
    assert engine._loop is not None
    assert engine._loop.is_running() is False
    assert engine.is_healthy() is False

    release.set()
    starter.join(timeout=2.0)
    assert not starter.is_alive()
    assert engine.is_healthy() is True
    assert engine.stop(timeout=2.0) is True


def test_add_monitor_timeout_cancels_queued_spawn():
    class _DeferredLoop:
        def __init__(self):
            self.callbacks = []

        def call_soon_threadsafe(self, callback):
            self.callbacks.append(callback)

    engine = MonitorEngine(None, monitors_ref={})
    loop = _DeferredLoop()
    spawned = []
    engine._loop = loop
    engine.is_healthy = lambda: True
    engine._spawn_job = lambda mid: spawned.append(mid) or True

    assert engine.add_monitor("mon_delayed", timeout=0.0) is False
    assert len(loop.callbacks) == 1
    loop.callbacks[0]()
    assert spawned == []


def test_spawn_rejects_job_after_stop_begins():
    engine = MonitorEngine(None, monitors_ref={"mon_late": {"enabled": True}})
    engine._stop.set()

    assert engine._spawn_job("mon_late") is False
    assert engine._jobs == {}
