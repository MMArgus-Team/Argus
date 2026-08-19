"""Per-profile home overrides survive multimodal owner-thread startup."""

import asyncio

from hermes_constants import (
    reset_hermes_home_override,
    set_hermes_home_override,
)

from agent.multimodal import monitor_agent, watch_file
from agent.multimodal.monitor_engine import MonitorEngine
from agent.multimodal.watcher_engine import WatcherAgent


def test_watcher_thread_keeps_secondary_profile_for_watch_files(tmp_path):
    secondary_home = tmp_path / "profiles" / "secondary"
    observed = []
    engine = WatcherAgent(object(), memory_backend=object())

    def run_probe():
        loop = asyncio.new_event_loop()
        engine._loop = loop
        asyncio.set_event_loop(loop)
        observed.append(watch_file.watch_dir())

        def mark_ready():
            if engine._mark_ready():
                return
            loop.stop()

        loop.call_soon(mark_ready)
        try:
            loop.run_forever()
        finally:
            if not loop.is_closed():
                loop.close()
            engine._loop = None
            with engine._state_lock:
                engine._state = engine.STATE_STOPPED
            engine._publish_stopped()

    engine._run = run_probe
    token = set_hermes_home_override(secondary_home)
    try:
        assert engine.start(timeout=1.0) is True
    finally:
        reset_hermes_home_override(token)

    try:
        assert observed == [secondary_home / "analyse"]
    finally:
        assert engine.stop(timeout=1.0) is True
        assert engine.wait_stopped(timeout=1.0) is True


def test_monitor_thread_keeps_secondary_profile_for_event_files(tmp_path):
    secondary_home = tmp_path / "profiles" / "secondary"
    observed = []
    engine = MonitorEngine(object(), monitors_ref={})

    def run_probe():
        loop = asyncio.new_event_loop()
        engine._loop = loop
        asyncio.set_event_loop(loop)
        observed.append(monitor_agent.monitor_dir())

        def mark_ready():
            if engine._stop.is_set():
                loop.stop()
            else:
                engine._healthy = True
                engine._ready.set()

        loop.call_soon(mark_ready)
        try:
            loop.run_forever()
        finally:
            engine._healthy = False
            engine._ready.set()
            if not loop.is_closed():
                loop.close()
            engine._loop = None
            asyncio.set_event_loop(None)

    engine._run = run_probe
    token = set_hermes_home_override(secondary_home)
    try:
        engine.start()
    finally:
        reset_hermes_home_override(token)

    try:
        assert engine.is_healthy() is True
        assert observed == [secondary_home / "monitor"]
    finally:
        assert engine.stop(timeout=1.0) is True
