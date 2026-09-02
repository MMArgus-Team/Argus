"""Reopen/close reconciliation of stale multimodal jobs.

When the app/tab closes (or crashes), the in-process monitor / deep-research
engine jobs die and the video stream stops — but their role=tool receipts in the
persisted history still say status="running". `_reconcile_stale_mm_jobs` flips
those to "interrupted" (so the reopened transcript is truthful) and re-registers
them into the agent's registries (disabled) so the panel can list them for on/off.
"""

import copy
import json
import threading
import types

import tui_gateway.server as gateway_server
from tui_gateway.server import (
    _interrupt_running_mm_jobs,
    _mark_monitor_tool_result_done,
    _mark_watcher_tool_result_complete,
    _maybe_start_monitor_engine,
    _reconcile_stale_mm_jobs,
)


def _tool_msg(payload: dict) -> dict:
    return {"role": "tool", "content": json.dumps(payload, ensure_ascii=False)}


def _history_bytes(history: list) -> bytes:
    return json.dumps(
        history, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def _multiturn_history(job_receipt: dict) -> list:
    return [
        {"role": "user", "content": "first ordinary turn"},
        {"role": "assistant", "content": "first ordinary answer"},
        {"role": "user", "content": "start the background job"},
        _tool_msg(job_receipt),
        {"role": "assistant", "content": "background job started"},
        {"role": "user", "content": "second ordinary turn"},
        {"role": "assistant", "content": "second ordinary answer"},
    ]


def test_running_research_marked_interrupted():
    hist = [
        {"role": "user", "content": "研究屏幕"},
        _tool_msg({"request_id": "req_ab12", "status": "running",
                   "label": "会议纪要", "task_instruction": "持续研究会议"}),
    ]
    n = _reconcile_stale_mm_jobs(hist)
    assert n == 1
    data = json.loads(hist[1]["content"])
    assert data["status"] == "interrupted"
    assert "interrupted_reason" in data
    assert "note" not in data


def test_running_monitor_marked_interrupted():
    hist = [_tool_msg({"monitor_id": "mon_9c1", "status": "running",
                       "op": "create", "label": "甘道夫检测",
                       "monitor_query": "检测甘道夫", "note": "监控已启动"})]
    n = _reconcile_stale_mm_jobs(hist)
    assert n == 1
    data = json.loads(hist[0]["content"])
    assert data["status"] == "interrupted"
    assert "note" not in data


def test_completed_research_left_untouched():
    hist = [_tool_msg({"request_id": "req_done", "status": "complete",
                       "report": "最终报告内容", "label": "x"})]
    n = _reconcile_stale_mm_jobs(hist)
    assert n == 0
    data = json.loads(hist[0]["content"])
    assert data["status"] == "complete"
    assert data["report"] == "最终报告内容"


def test_watcher_completion_marker_does_not_rewrite_cached_history():
    history = _multiturn_history({
        "request_id": "req_fin",
        "status": "running",
        "label": "观影",
        "task_instruction": "记精彩片段",
    })
    before_depth = len(history)
    before_bytes = _history_bytes(history)
    session = {
        "history": history,
        "history_lock": threading.Lock(),
        "history_version": 7,
    }

    assert _mark_watcher_tool_result_complete(session, "req_fin") is False

    assert len(history) == before_depth
    assert _history_bytes(history) == before_bytes
    assert session["history_version"] == 7


def test_non_mm_tool_message_untouched():
    hist = [_tool_msg({"op": "run", "status": "running", "output": "hi"}),  # no req_/mon_ id
            {"role": "tool", "content": "plain non-json result"}]
    n = _reconcile_stale_mm_jobs(hist)
    assert n == 0
    assert json.loads(hist[0]["content"])["status"] == "running"


def test_idempotent():
    hist = [_tool_msg({"request_id": "req_ab12", "status": "running", "label": "x"})]
    first = _reconcile_stale_mm_jobs(hist)
    second = _reconcile_stale_mm_jobs(hist)
    assert first == 1
    assert second == 0  # already interrupted → no re-count
    assert json.loads(hist[0]["content"])["status"] == "interrupted"


def test_reregisters_into_agent_registries_disabled():
    agent = types.SimpleNamespace()
    hist = [
        _tool_msg({"monitor_id": "mon_1", "status": "running", "label": "M",
                   "monitor_query": "q"}),
        _tool_msg({"request_id": "req_1", "status": "running", "label": "R",
                   "task_instruction": "rt", "hook_main_agent": True,
                   "hook_instruction": "do Y"}),
    ]
    _reconcile_stale_mm_jobs(hist, agent)
    # Monitor re-registered, disabled, marked interrupted.
    assert "mon_1" in agent.mm_monitors
    assert agent.mm_monitors["mon_1"]["enabled"] is False
    assert agent.mm_monitors["mon_1"]["_interrupted"] is True
    assert agent.mm_monitors["mon_1"]["trigger_mode"] == "continuous"
    assert agent.mm_monitor_active is False
    # Research re-registered, interrupted, hook preserved.
    assert "req_1" in agent.mm_watchers
    assert agent.mm_watchers["req_1"]["status"] == "interrupted"
    assert agent.mm_watchers["req_1"]["hook_main_agent"] is True
    assert agent.mm_watchers["req_1"]["hook_instruction"] == "do Y"


def test_reregister_does_not_overwrite_existing_entry():
    agent = types.SimpleNamespace()
    agent.mm_watchers = {"req_1": {"id": "req_1", "status": "running", "label": "LIVE"}}
    hist = [_tool_msg({"request_id": "req_1", "status": "running", "label": "STALE"})]
    _reconcile_stale_mm_jobs(hist, agent)
    # A live in-memory entry (this session created it) is NOT clobbered by the
    # history re-register; only the history JSON is flipped.
    assert agent.mm_watchers["req_1"]["label"] == "LIVE"
    assert agent.mm_watchers["req_1"]["status"] == "running"


def test_disk_only_watcher_restores_structured_state(monkeypatch, tmp_path):
    import hermes_constants
    from agent.multimodal import watch_file

    monkeypatch.setattr(
        hermes_constants, "get_hermes_home", lambda: tmp_path)
    watch_file.init_file(
        "req_restore1",
        query="持续总结会议",
        session_id="sid_restore",
        hook_main_agent=True,
        hook_instruction="输出最终纪要",
        state={
            "label": "会议复盘",
            "pacing_mode": "auto",
            "ttl": "30s",
            "ttl_sec": 30,
            "target_frames": 40,
        },
    )
    watch_file.append_round(
        "req_restore1", round_idx=1, frame_range=(0.0, 10.0),
        sub_queries=["Project Alpha status"], findings="第一段结论")
    watch_file.update_state(
        "req_restore1", seg_base=1, cursor_ts=10.0,
        runtime_id="old-process")
    agent = types.SimpleNamespace()

    _reconcile_stale_mm_jobs(
        [], agent, session_id="sid_restore")

    restored = agent.mm_watchers["req_restore1"]
    assert restored["status"] == "interrupted"
    assert restored["task_instruction"] == "持续总结会议"
    assert restored["label"] == "会议复盘"
    assert restored["pacing_mode"] == "auto"
    assert restored["ttl"] == "30s"
    assert restored["ttl_sec"] == 30
    assert restored["target_frames"] == 40
    assert restored["hook_main_agent"] is True
    assert restored["hook_instruction"] == "输出最终纪要"
    assert restored["_seg_base"] == 1
    assert watch_file.read_seen_subqueries("req_restore1") == {
        "project alpha status"}
    persisted = watch_file.read_state("req_restore1")
    assert persisted["status"] == "interrupted"
    assert persisted["stop_reason"] == "restart"


def test_watcher_update_receipt_overlays_legacy_create_receipt():
    history = [
        _tool_msg({
            "op": "create", "request_id": "req_legacy1",
            "status": "running", "task_instruction": "旧任务",
            "label": "旧标签", "ttl": "1min", "ttl_sec": 60,
            "target_frames": 60,
        }),
        _tool_msg({
            "op": "update", "request_id": "req_legacy1",
            "watcher_id": "req_legacy1", "task_instruction": "新任务",
            "label": "新标签", "ttl": "10s", "ttl_sec": 10,
            "target_frames": 15, "hook_main_agent": True,
            "hook_instruction": "生成报告",
        }),
    ]
    agent = types.SimpleNamespace()

    _reconcile_stale_mm_jobs(history, agent)

    restored = agent.mm_watchers["req_legacy1"]
    assert restored["task_instruction"] == "新任务"
    assert restored["label"] == "新标签"
    assert restored["ttl"] == "10s"
    assert restored["ttl_sec"] == 10
    assert restored["target_frames"] == 15
    assert restored["hook_main_agent"] is True
    assert restored["hook_instruction"] == "生成报告"


def test_disk_only_direct_monitor_reregisters_without_tool_history(
    monkeypatch, tmp_path,
):
    import hermes_constants
    from agent.multimodal import monitor_agent as ma

    monkeypatch.setattr(
        hermes_constants, "get_hermes_home", lambda: tmp_path)
    event_file = ma.init_event_file(
        "mon_direct",
        label="手机提醒",
        monitor_query="看到手机就告诉我",
        silent=False,
        report_interval=30,
        session_id="sid_direct",
        hook_main_agent=True,
        hook_instruction="记录手机型号",
    )
    # Provider-safe direct turns intentionally persist no synthetic tool role.
    history = [
        {"role": "user", "content": "看到手机就告诉我"},
        {"role": "assistant", "content": "监控已启动。"},
    ]
    agent = types.SimpleNamespace()

    _reconcile_stale_mm_jobs(
        history, agent, session_id="sid_direct")

    restored = agent.mm_monitors["mon_direct"]
    assert restored["enabled"] is False
    assert restored["status"] == "interrupted"
    assert restored["trigger_mode"] == "continuous"
    assert restored["monitor_query"] == "看到手机就告诉我"
    assert restored["report_interval"] == 30
    assert restored["hook_main_agent"] is True
    assert restored["hook_instruction"] == "记录手机型号"
    assert restored["event_file"] == event_file
    assert ma.read_status("mon_direct")["status"] == "interrupted"


def test_disk_done_once_monitor_restores_done_without_touching_canonical_history(
    monkeypatch, tmp_path,
):
    import hermes_constants
    from agent.multimodal import monitor_agent as ma

    monkeypatch.setattr(
        hermes_constants, "get_hermes_home", lambda: tmp_path)
    ma.init_event_file(
        "mon_done",
        label="一次提醒",
        monitor_query="看到完成弹窗就提醒",
        trigger_mode="once",
        silent=False,
        report_interval=None,
        session_id="sid_done",
    )
    assert ma.set_status("mon_done", "done") is True
    history = _multiturn_history({
        "op": "create",
        "monitor_id": "mon_done",
        "monitor_query": "看到完成弹窗就提醒",
        "status": "running",
    })
    before_depth = len(history)
    before_bytes = _history_bytes(history)
    agent = types.SimpleNamespace(mm_monitor_active=True)

    # Reopen/promotion reconciles a detached snapshot. The event file is the
    # durable completion authority; provider-cached history remains untouched.
    detached = copy.deepcopy(history)
    _reconcile_stale_mm_jobs(detached, agent, session_id="sid_done")

    restored = agent.mm_monitors["mon_done"]
    assert restored["status"] == "done"
    assert restored["enabled"] is False
    assert restored["trigger_mode"] == "once"
    assert "_interrupted" not in restored
    assert agent.mm_monitor_active is False
    receipt = json.loads(
        next(row for row in detached if row.get("role") == "tool")["content"]
    )
    assert receipt["status"] == "done"
    assert receipt["trigger_mode"] == "once"
    assert "interrupted_reason" not in receipt
    assert ma.read_status("mon_done")["status"] == "done"
    assert len(history) == before_depth
    assert _history_bytes(history) == before_bytes


def test_disk_done_watcher_restores_done_without_interrupting_or_rewriting_history(
    monkeypatch, tmp_path,
):
    import hermes_constants
    from agent.multimodal import watch_file

    monkeypatch.setattr(
        hermes_constants, "get_hermes_home", lambda: tmp_path)
    watch_file.init_file(
        "req_done_watcher",
        query="持续总结会议",
        session_id="sid_done_watcher",
        state={
            "label": "会议复盘",
            "task_instruction": "持续总结会议",
        },
    )
    watch_file.set_status(
        "req_done_watcher", "done", round_idx=2, stop_reason="task_complete"
    )
    history = _multiturn_history({
        "request_id": "req_done_watcher",
        "status": "running",
        "label": "会议复盘",
        "task_instruction": "持续总结会议",
    })
    before_depth = len(history)
    before_bytes = _history_bytes(history)
    agent = types.SimpleNamespace()

    _reconcile_stale_mm_jobs(
        copy.deepcopy(history), agent, session_id="sid_done_watcher"
    )

    restored = agent.mm_watchers["req_done_watcher"]
    assert restored["status"] == "done"
    assert "_interrupted" not in restored
    assert restored["task_instruction"] == "持续总结会议"
    assert watch_file.read_status("req_done_watcher")["status"] == "done"
    assert len(history) == before_depth
    assert _history_bytes(history) == before_bytes


def test_done_monitor_receipt_restores_without_disk_as_disabled_once():
    agent = types.SimpleNamespace()
    history = [_tool_msg({
        "op": "create",
        "monitor_id": "mon_receipt_done",
        "monitor_query": "提醒一次",
        "status": "done",
        "trigger_mode": "once",
    })]

    assert _reconcile_stale_mm_jobs(history, agent) == 0

    restored = agent.mm_monitors["mon_receipt_done"]
    assert restored["status"] == "done"
    assert restored["enabled"] is False
    assert restored["trigger_mode"] == "once"
    assert "_interrupted" not in restored
    assert agent.mm_monitor_active is False


def test_disk_only_monitor_resume_uses_latest_updated_header(
    monkeypatch, tmp_path,
):
    import hermes_constants
    from agent.multimodal import monitor_agent as ma

    monkeypatch.setattr(
        hermes_constants, "get_hermes_home", lambda: tmp_path)
    ma.init_event_file(
        "mon_updated",
        label="old",
        monitor_query="old query",
        silent=True,
        report_interval=None,
        session_id="sid_updated",
    )
    assert ma.update_event_file_config(
        "mon_updated",
        label="new",
        monitor_query="new query",
        silent=False,
        report_interval=45,
        hook_main_agent=True,
        hook_instruction="record details",
    )

    agent = types.SimpleNamespace()
    _reconcile_stale_mm_jobs([], agent, session_id="sid_updated")

    restored = agent.mm_monitors["mon_updated"]
    assert restored["label"] == "new"
    assert restored["monitor_query"] == "new query"
    assert restored["silent"] is False
    assert restored["report_interval"] == 45
    assert restored["hook_main_agent"] is True
    assert restored["hook_instruction"] == "record details"


def test_legacy_update_receipt_overlays_create_time_disk_header(
    monkeypatch, tmp_path,
):
    import hermes_constants
    from agent.multimodal import monitor_agent as ma

    monkeypatch.setattr(
        hermes_constants, "get_hermes_home", lambda: tmp_path)
    ma.init_event_file(
        "mon_legacy",
        label="old",
        monitor_query="old query",
        silent=True,
        report_interval=None,
        session_id="sid_legacy",
    )
    history = [_tool_msg({
        "op": "update",
        "monitor_id": "mon_legacy",
        "monitor_query": "new query",
        "label": "new",
        "trigger_mode": "once",
        "silent": False,
        "report_interval": 30,
        "hook_main_agent": True,
        "hook_instruction": "record details",
    })]
    agent = types.SimpleNamespace()

    _reconcile_stale_mm_jobs(
        history, agent, session_id="sid_legacy")

    restored = agent.mm_monitors["mon_legacy"]
    assert restored["monitor_query"] == "new query"
    assert restored["label"] == "new"
    assert restored["silent"] is False
    assert restored["report_interval"] == 30
    assert restored["trigger_mode"] == "once"
    assert restored["hook_main_agent"] is True
    assert restored["hook_instruction"] == "record details"


def test_monitor_stale_reconcile_is_scoped_to_own_session(
    monkeypatch, tmp_path,
):
    import hermes_constants
    from agent.multimodal import monitor_agent as ma

    monkeypatch.setattr(
        hermes_constants, "get_hermes_home", lambda: tmp_path)
    for mid, sid in (("mon_a", "sid_a"), ("mon_b", "sid_b")):
        ma.init_event_file(
            mid,
            label=mid,
            monitor_query=f"watch {mid}",
            silent=False,
            report_interval=None,
            session_id=sid,
        )

    assert ma.reconcile_stale(["mon_a"], session_id="sid_a") == 0
    assert ma.read_status("mon_a")["status"] == "running"
    assert ma.read_status("mon_b")["status"] == "running"


def test_monitor_completion_marker_does_not_rewrite_cached_history():
    history = _multiturn_history({
        "op": "create",
        "monitor_id": "mon_finish",
        "status": "running",
        "trigger_mode": "once",
    })
    before_depth = len(history)
    before_bytes = _history_bytes(history)
    session = {
        "history": history,
        "history_lock": threading.Lock(),
        "history_version": 4,
    }

    assert _mark_monitor_tool_result_done(
        session, "mon_finish", trigger_mode="once") is False

    assert len(history) == before_depth
    assert _history_bytes(history) == before_bytes
    assert session["history_version"] == 4


def test_deleted_monitor_receipt_is_not_reregistered_or_interrupted():
    history = [_tool_msg({
        "op": "delete",
        "monitor_id": "mon_deleted",
        "status": "deleted",
    })]
    agent = types.SimpleNamespace()

    assert _reconcile_stale_mm_jobs(history, agent) == 0
    assert agent.mm_monitors == {}
    assert json.loads(history[0]["content"])["status"] == "deleted"


def test_stream_interrupt_skips_done_and_deleted_monitors(monkeypatch):
    removed = []
    engine = types.SimpleNamespace(
        remove_monitor=lambda monitor_id: removed.append(monitor_id))
    agent = types.SimpleNamespace(
        mm_monitors={
            "mon_running": {"enabled": True, "status": "running"},
            "mon_done": {"enabled": False, "status": "done"},
            "mon_deleted": {"enabled": False, "status": "deleted"},
        },
        mm_watchers={},
        mm_monitor_active=True,
    )
    session = {
        "agent": agent,
        "_mm_monitor_engine": engine,
        "history": [],
        "history_lock": threading.Lock(),
    }
    import tools.live_watcher_tool as live_watcher_tool
    import tools.monitor_tool as monitor_tool
    monkeypatch.setattr(monitor_tool, "_push_monitors_event", lambda *_: None)
    monkeypatch.setattr(live_watcher_tool, "_push_watchers_event", lambda *_: None)

    assert _interrupt_running_mm_jobs("live", session) == 1

    assert removed == ["mon_running"]
    assert agent.mm_monitors["mon_running"]["status"] == "interrupted"
    assert agent.mm_monitors["mon_done"]["status"] == "done"
    assert agent.mm_monitors["mon_deleted"]["status"] == "deleted"
    assert agent.mm_monitor_active is False


def test_list_registries_carries_trigger_mode_and_disk_status(monkeypatch):
    agent = types.SimpleNamespace(mm_monitors={
        "mon_list": {
            "id": "mon_list",
            "monitor_query": "看到完成弹窗",
            "enabled": False,
            "status": "done",
            "trigger_mode": "once",
        },
    })
    from agent.multimodal import monitor_agent as ma
    monkeypatch.setattr(ma, "read_status", lambda _mid: {"status": "done"})
    monkeypatch.setitem(
        gateway_server._sessions, "live-list", {"agent": agent})

    response = gateway_server._methods["multimodal.list_registries"](
        "rpc-list", {"session_id": "live-list"})

    monitor = response["result"]["monitors"][0]
    assert monitor["monitor_id"] == "mon_list"
    assert monitor["status"] == "done"
    assert monitor["enabled"] is False
    assert monitor["trigger_mode"] == "once"


def test_monitor_engine_job_done_preserves_history_and_refreshes_registry(
    monkeypatch,
):
    import agent.multimodal.monitor_engine as monitor_engine_module

    class FakeMonitorEngine:
        def __init__(self, _frame_buffer, **kwargs):
            self.emit_cb = kwargs["emit_cb"]

        def start(self):
            return None

        def is_healthy(self):
            return True

        def stop(self):
            return None

    monkeypatch.setattr(
        monitor_engine_module, "MonitorEngine", FakeMonitorEngine)
    emitted = []
    pushed = []
    monkeypatch.setattr(
        gateway_server, "_emit",
        lambda event, sid, payload=None: emitted.append((event, sid, payload)))
    monkeypatch.setattr(
        gateway_server, "_push_mm_registries",
        lambda sid, agent: pushed.append((sid, agent)))
    history = _multiturn_history({
        "op": "create",
        "monitor_id": "mon_once",
        "status": "running",
        "trigger_mode": "once",
    })
    before_depth = len(history)
    before_bytes = _history_bytes(history)
    agent = types.SimpleNamespace(
        mm_monitors={
            "mon_once": {
                "id": "mon_once",
                "enabled": True,
                "status": "running",
                "trigger_mode": "once",
            },
        },
        mm_monitor_active=True,
    )
    session = {
        "agent": agent,
        "history": history,
        "history_lock": threading.Lock(),
        "history_version": 0,
    }

    engine = _maybe_start_monitor_engine("live-once", session, object())
    assert engine is not None
    agent.mm_monitors["mon_once"].update(enabled=False, status="done")
    engine.emit_cb("multimodal.trajectory", {
        "worker": "MonitorWorker",
        "monitor_id": "mon_once",
        "phase": "job_done",
        "status": "done",
        "trigger_mode": "once",
    })

    assert len(history) == before_depth
    assert _history_bytes(history) == before_bytes
    assert agent.mm_monitor_active is False
    assert pushed == [("live-once", agent)]
    assert emitted[-1][0] == "multimodal.trajectory"


def test_watcher_completion_callback_preserves_history_and_finishes_registry(
    monkeypatch,
):
    from agent.multimodal import hermes_glue
    import agent.multimodal.watcher_engine as watcher_engine_module
    import tools.live_watcher_tool as live_watcher_tool

    callbacks = {}

    class FakeWatcher:
        healthy = True
        is_ready = True
        is_stopped = False

        def __init__(self, frame_buffer, *, memory_backend=None, **kwargs):
            self.frame_buffer = frame_buffer
            self._memory_backend = memory_backend
            callbacks.update(kwargs)

        def start(self, timeout):
            assert timeout == gateway_server._MM_WATCHER_STARTUP_TIMEOUT_SEC
            return True

        def stop(self, timeout=None):
            return None

    monkeypatch.setattr(
        hermes_glue,
        "flatten_mm_config",
        lambda _cfg: {"enabled": True, "memory_enabled": False},
    )
    monkeypatch.setattr(watcher_engine_module, "WatcherAgent", FakeWatcher)
    monkeypatch.setattr(gateway_server, "_MM_ACTIVE_WATCHERS", {})
    monkeypatch.setattr(gateway_server, "_load_cfg", lambda: {})
    emitted = []
    pushed = []
    monkeypatch.setattr(
        gateway_server,
        "_emit",
        lambda event, sid, payload=None: emitted.append((event, sid, payload)),
    )
    monkeypatch.setattr(
        live_watcher_tool,
        "_push_watchers_event",
        lambda sid, agent: pushed.append((sid, agent)),
    )
    history = _multiturn_history({
        "request_id": "req_callback_done",
        "status": "running",
        "label": "会议总结",
        "task_instruction": "持续总结会议",
    })
    before_depth = len(history)
    before_bytes = _history_bytes(history)
    agent = types.SimpleNamespace(
        session_id="stored-watcher",
        mm_watchers={
            "req_callback_done": {
                "status": "running",
                "hook_main_agent": False,
            }
        },
    )
    session = {
        "agent": agent,
        "history": history,
        "history_lock": threading.Lock(),
        "history_version": 11,
        "session_key": "",
    }

    engine = gateway_server._maybe_start_live_watcher_agent(
        "live-watcher-callback", object(), object(), session
    )
    assert engine is not None
    callbacks["on_delegation_complete"](
        "req_callback_done", "会议总结", "最终会议纪要", "task_complete"
    )

    assert len(history) == before_depth
    assert _history_bytes(history) == before_bytes
    assert session["history_version"] == 11
    assert agent.mm_watchers["req_callback_done"]["status"] == "done"
    assert pushed == [("stored-watcher", agent)]
    assert [event for event, _sid, _payload in emitted] == [
        "watcher.final",
        "watcher.complete",
    ]
