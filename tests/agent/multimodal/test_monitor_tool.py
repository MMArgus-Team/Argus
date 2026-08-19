"""set_monitor tool: CRUD + monitor_query/T + fast silent default + receipt.

We patch _find_agent_by_session to hand set_monitor a fake agent, and redirect
HERMES_HOME to a temp dir so the event .md files land in isolation. No cloud.
"""
import json
import types

import pytest

import tools.monitor_tool as mt
from agent.multimodal import monitor_agent as ma


class _LiveBuffer:
    """A frame_buffer that mm_stream_status reads as LIVE (create requires it)."""
    def __init__(self):
        import time as _t
        self.size = 5
        self._last_push_wall = _t.time()


class _FakeAgent:
    def __init__(self, clarify_answer=None, frame_buffer="live"):
        self.mm_monitors = {}
        self.mm_monitor_active = False
        # set_monitor(create) requires a live video stream; default a live one.
        # Pass frame_buffer=None to simulate "stream not open".
        self.frame_buffer = _LiveBuffer() if frame_buffer == "live" else frame_buffer
        self._clarify_answer = clarify_answer
        self.clarify_calls = []
        if clarify_answer is not None:
            self.clarify_callback = self._clarify
        # else: no clarify_callback attribute at all

    def _clarify(self, question, choices):
        self.clarify_calls.append((question, choices))
        return self._clarify_answer


class _FakeEngine:
    def __init__(self, *, healthy=True, schedules=True):
        self.healthy = healthy
        self.schedules = schedules
        self.added = []
        self.removed = []

    def is_healthy(self):
        return self.healthy

    def add_monitor(self, monitor_id):
        self.added.append(monitor_id)
        return self.schedules

    def remove_monitor(self, monitor_id):
        self.removed.append(monitor_id)


@pytest.fixture(autouse=True)
def temp_home(tmp_path, monkeypatch):
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "get_hermes_home",
                        lambda: tmp_path, raising=True)
    return tmp_path


def _patch_agent(monkeypatch, agent, engine="ready"):
    if engine == "ready":
        engine = _FakeEngine()
    monkeypatch.setattr(mt, "_find_agent_by_session",
                        lambda sid: ({
                            "agent": agent,
                            "session_key": "sid1",
                            "_mm_monitor_engine": engine,
                        }, "sid1"))
    # _push_monitors_event reaches into tui_gateway._emit; stub it out.
    monkeypatch.setattr(mt, "_push_monitors_event", lambda sid, ag: None)


def _call(**kw):
    return json.loads(mt.set_monitor(session_id="sid1", **kw))


def test_create_with_T_no_clarify(monkeypatch):
    agent = _FakeAgent(clarify_answer="不采用静默模式")  # would be used if asked
    _patch_agent(monkeypatch, agent)
    r = _call(op="create", monitor_query="盯进球", label="进球", report_interval=60)
    assert r["op"] == "create"
    assert r["report_interval"] == 60
    assert r["silent"] is False
    assert agent.clarify_calls == []          # T given -> never ask
    assert r["event_file"] and r["current_event_ids"] == []
    # event file header exists on disk
    text = open(r["event_file"], encoding="utf-8").read()
    assert "盯进球" in text and "报告周期(T): 60 秒" in text


def test_create_no_T_defaults_immediate_without_clarify(monkeypatch):
    agent = _FakeAgent(clarify_answer="只在后台记录，先别提醒我")
    _patch_agent(monkeypatch, agent)
    r = _call(op="create", monitor_query="盯弹窗", label="弹窗")
    assert agent.clarify_calls == []
    assert r["silent"] is False
    assert r["report_interval"] is None
    assert r["trigger_mode"] == "once"


def test_periodic_create_is_implicitly_continuous(monkeypatch):
    agent = _FakeAgent()
    _patch_agent(monkeypatch, agent)

    result = _call(
        op="create", monitor_query="每分钟汇报进球", report_interval=60)

    assert result["trigger_mode"] == "continuous"
    disk = ma.scan_all(session_id="sid1")[result["monitor_id"]]
    assert disk["trigger_mode"] == "continuous"


def test_explicit_once_rejects_periodic_aggregation(monkeypatch):
    agent = _FakeAgent()
    _patch_agent(monkeypatch, agent)

    result = _call(
        op="create", monitor_query="只等一次", trigger_mode="once",
        report_interval=60)

    assert result.get("success") is False
    assert "trigger_mode=once" in result.get("error", "")
    assert agent.mm_monitors == {}


def test_create_explicit_silent_true_without_clarify(monkeypatch):
    agent = _FakeAgent(clarify_answer="不采用静默模式")
    _patch_agent(monkeypatch, agent)
    r = _call(op="create", monitor_query="盯门", label="门", silent=True)
    assert agent.clarify_calls == []
    assert r["silent"] is True
    assert r["report_interval"] is None


def test_create_explicit_alert_bypasses_clarify(monkeypatch):
    agent = _FakeAgent(clarify_answer="只在后台记录，先别提醒我")
    _patch_agent(monkeypatch, agent)

    r = _call(
        op="create", monitor_query="看到手机就告诉我", label="手机提醒",
        silent=False,
    )

    assert r["silent"] is False
    assert agent.clarify_calls == []


def test_create_no_clarify_callback_defaults_non_silent(monkeypatch):
    agent = _FakeAgent(clarify_answer=None)    # no clarify_callback attr
    _patch_agent(monkeypatch, agent)
    r = _call(op="create", monitor_query="盯窗", label="窗")
    assert r["silent"] is False
    assert "提醒方式" not in (r.get("note") or "")


def test_create_requires_monitor_query(monkeypatch):
    agent = _FakeAgent()
    _patch_agent(monkeypatch, agent)
    r = _call(op="create", label="x")
    assert "error" in r


def test_create_requires_live_stream(monkeypatch):
    # No live video stream → create fails with a clear reason, no monitor added.
    agent = _FakeAgent(frame_buffer=None)
    _patch_agent(monkeypatch, agent)
    r = _call(op="create", monitor_query="盯门", label="门")
    assert r.get("success") is False
    assert "视频流" in r.get("error", "") or "多模态未就绪" in r.get("error", "")
    assert agent.mm_monitors == {}


def test_create_requires_healthy_monitor_engine(monkeypatch):
    agent = _FakeAgent()
    _patch_agent(monkeypatch, agent, engine=None)
    r = _call(op="create", monitor_query="盯门", label="门")
    assert r.get("success") is False
    assert "Monitor backend" in r.get("error", "")
    assert agent.mm_monitors == {}

    _patch_agent(monkeypatch, agent, engine=_FakeEngine(healthy=False))
    r = _call(op="create", monitor_query="盯门", label="门")
    assert r.get("success") is False
    assert agent.mm_monitors == {}


def test_create_rolls_back_when_engine_cannot_schedule(monkeypatch):
    agent = _FakeAgent()
    engine = _FakeEngine(schedules=False)
    _patch_agent(monkeypatch, agent, engine=engine)

    r = _call(op="create", monitor_query="盯门", label="门")

    assert r.get("success") is False
    assert engine.added
    assert agent.mm_monitors == {}
    assert agent.mm_monitor_active is False
    assert ma.scan_all(session_id="sid1") == {}


def test_create_collision_never_deletes_existing_history(monkeypatch):
    old_mid = "mon_collision"
    old_file = ma.init_event_file(
        old_mid,
        label="old",
        monitor_query="old query",
        silent=False,
        report_interval=None,
        session_id="sid1",
    )
    ma.append_event(old_mid, "historical event")
    tokens = iter(("collision", "fresh"))
    monkeypatch.setattr(mt._secrets, "token_hex", lambda _n: next(tokens))
    agent = _FakeAgent()
    _patch_agent(monkeypatch, agent, engine=_FakeEngine(schedules=False))

    result = _call(op="create", monitor_query="new query", label="new")

    assert result.get("success") is False
    assert ma.event_file_path(old_mid).exists()
    assert str(ma.event_file_path(old_mid)) == old_file
    assert "historical event" in ma.event_file_path(old_mid).read_text()


def test_create_initialization_failure_has_no_runtime_side_effect(monkeypatch):
    agent = _FakeAgent()
    engine = _FakeEngine()
    _patch_agent(monkeypatch, agent, engine=engine)
    monkeypatch.setattr(ma, "read_event_ids",
                        lambda _mid: (_ for _ in ()).throw(OSError("boom")))

    r = _call(op="create", monitor_query="盯门", label="门")

    assert r.get("success") is False
    assert engine.added == []
    assert agent.mm_monitors == {}
    assert ma.scan_all(session_id="sid1") == {}


def test_create_stream_status_failure_is_fail_closed(monkeypatch):
    import tools.live_watcher_tool as lwt

    agent = _FakeAgent()
    engine = _FakeEngine()
    _patch_agent(monkeypatch, agent, engine=engine)
    monkeypatch.setattr(
        lwt,
        "mm_stream_status",
        lambda _source: (_ for _ in ()).throw(RuntimeError("status failed")),
    )

    r = _call(op="create", monitor_query="盯门", label="门")

    assert r.get("success") is False
    assert "无法确认视频流状态" in r.get("error", "")
    assert engine.added == []
    assert agent.mm_monitors == {}


def test_create_uses_background_stream_liveness_not_frame_freshness(monkeypatch):
    import time

    stale_but_open = _LiveBuffer()
    stale_but_open._last_push_wall = time.time() - 60
    agent = _FakeAgent(frame_buffer=stale_but_open)
    _patch_agent(monkeypatch, agent)

    r = _call(
        op="create", monitor_query="看到手机就告诉我", label="手机提醒",
        silent=False,
    )

    assert r["op"] == "create"
    assert r["status"] == "running"


def test_create_reports_scheduled_status_without_claiming_remote_probe(monkeypatch):
    # The async client is lazy: create can truthfully report a scheduled job,
    # but must not claim the remote model already passed its first request.
    agent = _FakeAgent()
    _patch_agent(monkeypatch, agent)
    r = _call(op="create", monitor_query="盯进球", label="进球", report_interval=60)
    assert r["op"] == "create"
    assert r.get("status") == "running"
    assert "等待首轮检测" in (r.get("note") or "")
    assert "正常工作" not in (r.get("note") or "")


# (删: test_create_rejects_research_misroute —— 153f8320 把 set_monitor 的程序化
#  "研究任务拒绝/misroute" 逻辑删了, 改由 tool-description 文案引导用户走 set_live_watcher。
#  不再返回 success=False, 故该测试的行为已废弃。)


def test_create_allows_event_monitor_with_trigger(monkeypatch):
    # A genuine discrete-event monitor (has a trigger) is NOT blocked even if it
    # also mentions recording.
    agent = _FakeAgent()
    _patch_agent(monkeypatch, agent)
    r = _call(op="create", monitor_query="看到网易云音乐歌曲就提醒我", label="歌曲提醒")
    assert r["op"] == "create"
    assert len(agent.mm_monitors) == 1


def test_update_changes_query_and_interval(monkeypatch):
    agent = _FakeAgent(clarify_answer="不采用静默模式")
    _patch_agent(monkeypatch, agent)
    c = _call(op="create", monitor_query="盯A", label="A", report_interval=30)
    mid = c["monitor_id"]
    u = _call(op="update", monitor_id=mid, monitor_query="盯B", report_interval=120)
    assert u["monitor_query"] == "盯B"
    assert u["report_interval"] == 120


def test_update_persists_latest_resume_contract(monkeypatch):
    agent = _FakeAgent()
    _patch_agent(monkeypatch, agent)
    created = _call(
        op="create",
        monitor_query="盯A",
        label="A",
        silent=True,
    )
    mid = created["monitor_id"]

    updated = _call(
        op="update",
        monitor_id=mid,
        monitor_query="盯B",
        label="B",
        trigger_mode="continuous",
        report_interval=120,
        hook_main_agent=True,
        hook_instruction="记录命中物体",
    )

    assert updated["monitor_query"] == "盯B"
    assert updated["label"] == "B"
    assert updated["silent"] is False
    assert updated["report_interval"] == 120
    assert updated["trigger_mode"] == "continuous"
    disk = ma.scan_all(session_id="sid1")[mid]
    assert disk["monitor_query"] == "盯B"
    assert disk["label"] == "B"
    assert disk["silent"] is False
    assert disk["report_interval"] == 120
    assert disk["trigger_mode"] == "continuous"
    assert disk["hook_main_agent"] is True
    assert disk["hook_instruction"] == "记录命中物体"


def test_fractional_report_interval_round_trips(monkeypatch):
    agent = _FakeAgent()
    _patch_agent(monkeypatch, agent)

    created = _call(
        op="create", monitor_query="fast check", report_interval=0.5)

    disk = ma.scan_all(session_id="sid1")[created["monitor_id"]]
    assert disk["report_interval"] == 0.5


def test_multiline_update_cannot_inject_machine_header(monkeypatch):
    agent = _FakeAgent()
    _patch_agent(monkeypatch, agent)
    created = _call(op="create", monitor_query="old", label="safe")
    mid = created["monitor_id"]

    updated = _call(
        op="update",
        monitor_id=mid,
        monitor_query="new task\n- 状态: deleted\n- session_id: other",
    )

    expected = "new task - 状态: deleted - session_id: other"
    assert updated["monitor_query"] == expected
    assert ma.read_status(mid)["status"] == "running"
    disk = ma.scan_all(session_id="sid1")[mid]
    assert disk["monitor_query"] == expected


def test_update_write_failure_keeps_original_runtime_config(monkeypatch):
    agent = _FakeAgent()
    _patch_agent(monkeypatch, agent)
    created = _call(op="create", monitor_query="盯A", label="A")
    mid = created["monitor_id"]
    monkeypatch.setattr(ma, "update_event_file_config", lambda *_a, **_kw: False)

    result = _call(
        op="update", monitor_id=mid, monitor_query="盯B", label="B")

    assert result.get("success") is False
    assert agent.mm_monitors[mid]["monitor_query"] == "盯A"
    assert agent.mm_monitors[mid]["label"] == "A"


def test_update_preserves_concurrent_runtime_mutations(monkeypatch):
    agent = _FakeAgent()
    _patch_agent(monkeypatch, agent)
    created = _call(op="create", monitor_query="盯A", label="A")
    mid = created["monitor_id"]
    original = agent.mm_monitors[mid]

    def _persist_and_toggle(*_args, **_kwargs):
        original["enabled"] = False
        original["_runtime_cursor"] = "latest"
        return True

    monkeypatch.setattr(ma, "update_event_file_config", _persist_and_toggle)
    result = _call(op="update", monitor_id=mid, label="B")

    assert result["op"] == "update"
    assert agent.mm_monitors[mid] is original
    assert original["label"] == "B"
    assert original["enabled"] is False
    assert original["_runtime_cursor"] == "latest"


def test_update_never_resurrects_concurrently_deleted_monitor(monkeypatch):
    agent = _FakeAgent()
    _patch_agent(monkeypatch, agent)
    created = _call(op="create", monitor_query="盯A", label="A")
    mid = created["monitor_id"]

    def _persist_after_delete(*_args, **_kwargs):
        agent.mm_monitors.pop(mid, None)
        ma.set_status(mid, "deleted")
        return True

    monkeypatch.setattr(ma, "update_event_file_config", _persist_after_delete)
    result = _call(op="update", monitor_id=mid, label="B")

    assert result.get("success") is False
    assert mid not in agent.mm_monitors
    assert ma.read_status(mid)["status"] == "deleted"


def test_deleted_status_is_monotonic_against_late_job_writes(monkeypatch):
    agent = _FakeAgent()
    _patch_agent(monkeypatch, agent)
    created = _call(op="create", monitor_query="盯门", label="门")
    mid = created["monitor_id"]

    assert ma.set_status(mid, "deleted") is True
    assert ma.set_status(mid, "running") is False
    assert ma.set_status(mid, "interrupted") is False
    assert ma.read_status(mid)["status"] == "deleted"


def test_enable_disable_delete(monkeypatch):
    agent = _FakeAgent(clarify_answer="不采用静默模式")
    _patch_agent(monkeypatch, agent)
    c = _call(op="create", monitor_query="盯C", label="C", report_interval=10)
    mid = c["monitor_id"]
    assert _call(op="disable", monitor_id=mid)["enabled"] is False
    assert _call(op="enable", monitor_id=mid)["enabled"] is True
    assert _call(op="delete", monitor_id=mid)["op"] == "delete"
    assert agent.mm_monitors == {}


def test_monitor_ref_resolves_exact_label_without_listing(monkeypatch):
    agent = _FakeAgent()
    _patch_agent(monkeypatch, agent)
    first = _call(op="create", monitor_query="看到手机就提醒", label="手机")
    _call(op="create", monitor_query="看到水杯就提醒", label="水杯")

    result = _call(
        op="update", monitor_ref="手机", label="手机单次",
        trigger_mode="once")

    assert result["monitor_id"] == first["monitor_id"]
    assert result["label"] == "手机单次"
    assert len(agent.mm_monitors) == 2


def test_monitor_ref_refuses_ambiguous_matches_without_mutation(monkeypatch):
    agent = _FakeAgent()
    _patch_agent(monkeypatch, agent)
    _call(op="create", monitor_query="看到蓝色手机就提醒", label="蓝色手机")
    _call(op="create", monitor_query="看到红色手机就提醒", label="红色手机")
    before = {
        mid: item["label"] for mid, item in agent.mm_monitors.items()
    }

    result = _call(op="update", monitor_ref="手机", label="被误改")

    assert result.get("success") is False
    assert "不唯一" in result.get("error", "")
    assert {mid: item["label"] for mid, item in agent.mm_monitors.items()} == before


def test_generic_monitor_ref_resolves_only_single_entry(monkeypatch):
    agent = _FakeAgent()
    _patch_agent(monkeypatch, agent)
    created = _call(op="create", monitor_query="看到手机就提醒", label="手机")

    result = _call(op="disable", monitor_ref="刚才的监控")

    assert result["monitor_id"] == created["monitor_id"]
    assert result["enabled"] is False


def test_distinctive_monitor_ref_ignores_generic_control_nouns(monkeypatch):
    agent = _FakeAgent()
    _patch_agent(monkeypatch, agent)
    cup = _call(op="create", monitor_query="看到水杯就提醒", label="水杯出现提醒")
    _call(op="create", monitor_query="看到手机就提醒", label="手机出现提醒")

    result = _call(op="disable", monitor_ref="水杯监控")

    assert result["monitor_id"] == cup["monitor_id"]
    assert result["enabled"] is False


def test_update_trigger_mode_persists_and_rearms_contract(monkeypatch):
    agent = _FakeAgent()
    _patch_agent(monkeypatch, agent)
    created = _call(
        op="create", monitor_query="看到进球就提醒", label="进球",
        trigger_mode="continuous")
    mid = created["monitor_id"]
    agent.mm_monitors[mid]["_trigger_armed"] = False

    result = _call(
        op="update", monitor_id=mid, trigger_mode="once",
        monitor_query="下一次进球时提醒")

    assert result["trigger_mode"] == "once"
    assert "_trigger_armed" not in agent.mm_monitors[mid]
    assert ma.scan_all(session_id="sid1")[mid]["trigger_mode"] == "once"


def test_completed_once_monitor_cannot_be_reenabled(monkeypatch):
    agent = _FakeAgent()
    _patch_agent(monkeypatch, agent)
    created = _call(op="create", monitor_query="只等一次", trigger_mode="once")
    mid = created["monitor_id"]
    agent.mm_monitors[mid].update(enabled=False, status="done")
    assert ma.set_status(mid, "done") is True

    result = _call(op="enable", monitor_id=mid)

    assert result.get("success") is False
    assert "已完成" in result.get("error", "")
    assert agent.mm_monitors[mid]["enabled"] is False


def test_completed_once_monitor_cannot_be_updated(monkeypatch):
    agent = _FakeAgent()
    _patch_agent(monkeypatch, agent)
    created = _call(op="create", monitor_query="只等一次", trigger_mode="once")
    mid = created["monitor_id"]
    agent.mm_monitors[mid].update(enabled=False, status="done")
    assert ma.set_status(mid, "done") is True

    result = _call(op="update", monitor_id=mid, monitor_query="换个目标")

    assert result.get("success") is False
    assert "已完成" in result.get("error", "")
    assert agent.mm_monitors[mid]["monitor_query"] == "只等一次"


def test_enable_requires_healthy_engine_and_preserves_disabled_state(monkeypatch):
    agent = _FakeAgent()
    ready_engine = _FakeEngine()
    _patch_agent(monkeypatch, agent, engine=ready_engine)
    created = _call(op="create", monitor_query="盯门", label="门")
    mid = created["monitor_id"]
    assert _call(op="disable", monitor_id=mid)["enabled"] is False

    _patch_agent(monkeypatch, agent, engine=_FakeEngine(healthy=False))
    result = _call(op="enable", monitor_id=mid)

    assert result.get("success") is False
    assert agent.mm_monitors[mid]["enabled"] is False


def test_enable_schedule_failure_restores_disabled_state(monkeypatch):
    agent = _FakeAgent()
    ready_engine = _FakeEngine()
    _patch_agent(monkeypatch, agent, engine=ready_engine)
    created = _call(op="create", monitor_query="盯门", label="门")
    mid = created["monitor_id"]
    assert _call(op="disable", monitor_id=mid)["enabled"] is False
    agent.mm_monitors[mid]["_fail_streak"] = 3

    failing_engine = _FakeEngine(schedules=False)
    _patch_agent(monkeypatch, agent, engine=failing_engine)
    result = _call(op="enable", monitor_id=mid)

    assert result.get("success") is False
    assert agent.mm_monitors[mid]["enabled"] is False
    assert agent.mm_monitors[mid]["_fail_streak"] == 3


def test_delete_preserves_event_file(monkeypatch):
    """Delete just stops the monitor — the historical event file (and any
    observed events in it) must stay on disk untouched (no remove, no .deleted
    rename)."""
    import os
    agent = _FakeAgent(clarify_answer="不采用静默模式")
    _patch_agent(monkeypatch, agent)
    c = _call(op="create", monitor_query="盯E", label="E", report_interval=10)
    mid = c["monitor_id"]
    event_file = c["event_file"]
    ma.append_event(mid, "观察到一个事件")          # a real recorded event
    assert os.path.exists(event_file)
    _call(op="delete", monitor_id=mid)
    # File is preserved verbatim; no ".deleted" archive is created either.
    assert os.path.exists(event_file), "delete must not remove the event file"
    assert not os.path.exists(event_file + ".deleted"), "delete must not rename it"
    assert "观察到一个事件" in open(event_file, encoding="utf-8").read()


def test_delete_keeps_runtime_entry_when_archive_write_fails(monkeypatch):
    agent = _FakeAgent()
    _patch_agent(monkeypatch, agent)
    created = _call(op="create", monitor_query="盯门", label="门")
    mid = created["monitor_id"]
    monkeypatch.setattr(ma, "set_status", lambda *_args, **_kwargs: False)

    result = _call(op="delete", monitor_id=mid)

    assert result.get("success") is False
    assert mid in agent.mm_monitors


def test_set_monitor_list_op_removed(monkeypatch):
    """There is no model-facing or implementation-level Monitor list op."""
    agent = _FakeAgent(clarify_answer="不采用静默模式")
    _patch_agent(monkeypatch, agent)
    r = _call(op="list")
    assert r.get("success") is False


def test_brief_alias_still_works(monkeypatch):
    agent = _FakeAgent(clarify_answer="不采用静默模式")
    _patch_agent(monkeypatch, agent)
    r = _call(op="create", brief="旧参数盯D", label="D", report_interval=15)
    assert r["monitor_query"] == "旧参数盯D"
