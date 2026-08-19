"""MonitorAgent event-file logging (HERMES_HOME/monitor/monitor_<id>.md).

Every observed event is appended with a leading ISO timestamp + event id,
regardless of silent mode or report period T. Header written once at create.
Pure filesystem (temp HERMES_HOME); no cloud, no hardware.
"""
import re

import pytest

from agent.multimodal import monitor_agent as ma


@pytest.fixture()
def temp_home(tmp_path, monkeypatch):
    # Redirect monitor_dir() to a temp folder by patching get_hermes_home.
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "get_hermes_home",
                        lambda: tmp_path, raising=True)
    return tmp_path


def test_header_written_once(temp_home):
    p = ma.init_event_file("mA", label="弹窗监控", monitor_query="出现红色报错弹窗",
                           silent=False, report_interval=None,
                           trigger_mode="once")
    text = open(p, encoding="utf-8").read()
    assert "monitor_id: mA" in text
    assert "出现红色报错弹窗" in text
    assert "触发模式(trigger_mode): once" in text
    assert "静默模式: 否" in text
    assert "## 事件记录" in text
    # Re-init must NOT wipe an existing file.
    ma.append_event("mA", "第一条事件")
    ma.init_event_file("mA", label="x", monitor_query="y", silent=True,
                       report_interval=99)
    assert "第一条事件" in open(p, encoding="utf-8").read()


def test_event_line_has_iso_timestamp_and_id(temp_home):
    ma.init_event_file("mB", label="l", monitor_query="q", silent=False,
                       report_interval=None)
    eid, ts = ma.append_event("mB", "画面出现红色弹窗\n(带换行应被压平)")
    line = [ln for ln in open(ma.event_file_path("mB"), encoding="utf-8")
            if eid in ln][0].strip()
    m = ma._EVENT_LINE_RE.match(line)
    assert m, f"line not parseable: {line!r}"
    assert m.group("id") == eid
    # ISO-ish timestamp at line head
    assert re.match(r"\d{4}-\d{2}-\d{2}T", m.group("ts"))
    # newline flattened to a single line
    assert "\n" not in m.group("desc") and "带换行应被压平" in m.group("desc")


def test_silent_mode_still_writes(temp_home):
    """Silent monitors record events to the file (only reporting is suppressed)."""
    ma.init_event_file("mC", label="l", monitor_query="q", silent=True,
                       report_interval=None)
    ma.append_event("mC", "静默模式下也要写入")
    ids = ma.read_event_ids("mC")
    assert len(ids) == 1


def test_read_event_ids_empty_on_fresh(temp_home):
    ma.init_event_file("mD", label="l", monitor_query="q", silent=False,
                       report_interval=None)
    assert ma.read_event_ids("mD") == []


def test_scan_legacy_file_defaults_trigger_mode_to_continuous(temp_home):
    path = ma.event_file_path("legacy")
    path.write_text(
        "# Monitor legacy\n"
        "- monitor_id: legacy\n"
        "- 任务(monitor_query): q\n"
        "- 状态: running · 2026-01-01T00:00:00+08:00\n"
        "\n## 事件记录\n",
        encoding="utf-8",
    )

    assert ma.scan_all()["legacy"]["trigger_mode"] == "continuous"


def test_update_event_file_inserts_and_scans_trigger_mode(temp_home):
    ma.init_event_file(
        "mE",
        label="l",
        monitor_query="q",
        silent=False,
        report_interval=None,
    )

    assert ma.update_event_file_config(
        "mE",
        label="l",
        monitor_query="q2",
        silent=False,
        report_interval=None,
        hook_main_agent=False,
        hook_instruction="",
        trigger_mode="once",
    )

    assert ma.scan_all()["mE"]["trigger_mode"] == "once"
    assert "触发模式(trigger_mode): once" in ma.event_file_path(
        "mE"
    ).read_text(encoding="utf-8")
