# -*- coding: utf-8 -*-
"""Deep-analysis analyse file — the on-disk log for background multimodal
live-watcher delegations (set_live_watcher → WatcherAgent).

Every live-watcher run gets a file at
``HERMES_HOME/analyse/watch_<request_id>.md``. The path is available to the main
agent the moment the watcher is set, so the agent can read progress at any time;
each round appends a "## 第 N 次分析" section, and a final LLM summary
("## 完成") is written when the run ends. The watcher is continuous: it walks
the video from the head and runs round after round until the source ends or the
worker conclusively observes the task's explicit completion condition.

This module is the file layer ONLY (mirrors monitor_agent's event-file split):
  * ``init_file``            — write the header once (query / time / hook / sid).
  * ``append_round``         — append one round's {frame_range, sub_queries,
                               findings} section.
  * ``append_note`` / ``mark_finished`` — small status / terminal lines.
  * ``set_status`` / ``read_status`` — rewrite/parse the header status line.
  * ``update_state`` / ``read_state`` — atomically persist/hydrate the mutable
                               task, pacing, cursor, hook, and stop contract.
  * ``scan_all`` / ``reconcile_stale`` — enumerate files (session-filtered) and
                               fix stale running/stopping states on startup.
  * ``read_all`` / ``read_structured`` — full text (fed to the summary LLM) /
                               parsed object for the reopen-history UI.
  * ``read_seen_subqueries`` — sub-queries already searched, for cross-round
                               dedup (a recurring sub-query must not be
                               re-searched).
  * ``count_rounds`` / ``drop_last_incomplete_round`` — round bookkeeping /
                               reopen cleanup.

Status is one of five values written into the header ``- 状态:`` line:
running / done / interrupted / stopping / deleted.
"""
from __future__ import annotations

import logging
import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence

log = logging.getLogger("hermes.multimodal.watch")

# A round section header: "## 第 N 次分析 ..." (user-facing) or legacy
# "## Round N ..."; sub-query lines are "- [SQ] ...".
_ROUND_RE = re.compile(r"^##\s+(?:第\s*(?P<n>\d+)\s*次分析|Round\s+(?P<n2>\d+))\b")
_SUBQUERY_RE = re.compile(r"^-\s*\[SQ\]\s*(?P<q>.+?)\s*$")
# 头部状态行 (机器可解析): "- 状态: <status> · 轮次 <N> · <iso>"。旧文件可能只有
# "- 状态: 进行中" (无 · 分隔), read_status 也能容忍。★ 五态统一(2026-07):
# running(运行中) / done(完成) / interrupted(中断:失败/熔断/手动关/解析兜底) /
# stopping(手动停当前轮收尾) / deleted(已删除:存档保留但一律不展示)。
# 旧态 stopped/disabled/unknown/complete 已并入上述五态。
_STATUS_RE = re.compile(
    r"^- 状态:\s*(?P<status>[^\s·]+)(?:\s*·\s*轮次\s*(?P<round>\d+))?"
    r"(?:\s*·\s*(?P<ts>.+))?\s*$")
_QUERY_RE = re.compile(r"^- 任务\(query\):\s*(?P<q>.+?)\s*$")
# 归属 session (本次新增): "- session_id: <sid>"。旧文件无此行 → session_id=None,
# 视为"无归属", 任何 session 的 list 都不显示 (见 scan_all session 过滤)。
_SESSION_RE = re.compile(r"^- session_id:\s*(?P<sid>.+?)\s*$")
_STATE_RE = re.compile(r"^- watcher_state:\s*(?P<json>\{.*\})\s*$")

# One process-wide lock guarding all analyse files (tiny appends; matches the
# monitor event-file locking model).
_FILE_LOCK = threading.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def watch_dir() -> Path:
    """HERMES_HOME/analyse, created on demand."""
    try:
        from hermes_constants import get_hermes_home
        base = get_hermes_home()
    except Exception:  # pragma: no cover - hermes_constants present in the app
        base = Path(os.path.expanduser("~")) / ".argus"
    d = Path(base) / "analyse"
    d.mkdir(parents=True, exist_ok=True)
    return d


def file_path(request_id: str) -> Path:
    return watch_dir() / f"watch_{request_id}.md"


def init_file(request_id: str, *, query: str, session_id: str = "",
              hook_main_agent: bool = False, hook_instruction: str = "",
              state: Optional[dict] = None) -> str:
    """Write the header block once. Returns the file path (str).

    Idempotent: an existing file is left untouched (re-setting a watcher with the
    same rid must not wipe accumulated rounds). The header records request_id,
    optional session_id, query, the hook_main_agent/hook_instruction pairing,
    creation time, and an initial ``- 状态: running`` line.

    hook_main_agent / hook_instruction are recorded in the header so it's easy to
    audit whether successful watcher completion triggers the main agent once.
    """
    path = file_path(request_id)
    with _FILE_LOCK:
        if path.exists():
            return str(path)
        query1 = " ".join(str(query or "").split())
        _hook_desc = " ".join(str(hook_instruction or "").split())
        hook_line = (
            f"- 主Agent联动(hook): 开 · 指令: {_hook_desc or '(未填)'}\n"
            if hook_main_agent
            else "- 主Agent联动(hook): 关 (分段报告仅进面板, 完成时不触发主 Agent)\n"
        )
        _sid_line = f"- session_id: {session_id}\n" if str(session_id or "").strip() else ""
        watcher_state = {
            "version": 1,
            "request_id": str(request_id),
            "session_id": str(session_id or "").strip(),
            "task_instruction": query1,
            "label": "",
            "status": "running",
            # Empty means a pre-pacing-mode/legacy entry whose stored numeric
            # values stay authoritative.  The live_watcher tool always writes
            # either "auto" or "explicit" for newly created tasks.
            "pacing_mode": "",
            "ttl": "",
            "ttl_sec": None,
            "target_frames": None,
            "seg_base": 0,
            "cursor_ts": None,
            "runtime_id": "",
            "hook_main_agent": bool(hook_main_agent),
            "hook_instruction": str(hook_instruction or "").strip(),
            "stop_reason": "",
        }
        if isinstance(state, dict):
            watcher_state.update(state)
        state_line = json.dumps(
            watcher_state, ensure_ascii=False, separators=(",", ":"))
        header = (
            f"# Deep Analysis {request_id}\n"
            f"- request_id: {request_id}\n"
            f"{_sid_line}"
            f"- 任务(query): {query1}\n"
            f"{hook_line}"
            f"- watcher_state: {state_line}\n"
            f"- 创建时间: {_now_iso()}\n"
            f"- 状态: running · {_now_iso()}\n"
            f"\n## 分析记录\n"
        )
        path.write_text(header, encoding="utf-8")
    log.info("[watch] init file %s", path)
    return str(path)


def _state_from_lines(lines: Sequence[str]) -> dict:
    """Parse the structured watcher state from already-read file lines."""
    for raw in lines:
        match = _STATE_RE.match(raw.strip())
        if not match:
            continue
        try:
            value = json.loads(match.group("json"))
        except (TypeError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}
    return {}


def _replace_state_line(lines: List[str], state: dict) -> None:
    encoded = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
    new_line = f"- watcher_state: {encoded}"
    for idx, raw in enumerate(lines):
        if _STATE_RE.match(raw.strip()):
            lines[idx] = new_line
            return
    insert_at = next(
        (idx for idx, raw in enumerate(lines)
         if raw.startswith("- 创建时间:") or raw.startswith("- 状态:")),
        next((idx for idx, raw in enumerate(lines) if raw.startswith("## ")),
             len(lines)),
    )
    lines.insert(insert_at, new_line)


def read_state(request_id: str) -> dict:
    """Read the structured resumable watcher state, or ``{}`` for legacy files."""
    path = file_path(request_id)
    if not path.exists():
        return {}
    try:
        with _FILE_LOCK:
            lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    return _state_from_lines(lines)


def update_state(request_id: str, **patch) -> dict:
    """Atomically merge mutable watcher fields into the analyse-file header."""
    path = file_path(request_id)
    with _FILE_LOCK:
        if not path.exists():
            return {}
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return {}
        state = _state_from_lines(lines)
        if not state:
            state = {"version": 1, "request_id": str(request_id)}
        state.update({key: value for key, value in patch.items()})
        _replace_state_line(lines, state)
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
            os.replace(tmp, path)
        except OSError:
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
            return {}
    return state


def _fmt_clock(epoch_wall: Optional[float], rel_ts: float) -> str:
    """Render a relative frame ts as an absolute wall-clock HH:MM:SS, using the
    session's first-frame wall epoch. Falls back to relative seconds if the epoch
    is unknown (e.g. no frames ever pushed)."""
    if epoch_wall is not None:
        try:
            return datetime.fromtimestamp(epoch_wall + rel_ts).strftime("%H:%M:%S")
        except Exception:
            pass
    return f"{rel_ts:.0f}s"


def _fmt_duration(seconds: float) -> str:
    """Human-readable duration: '40s' / '2分15s' / '1时03分'."""
    s = int(round(max(0.0, seconds)))
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}分{s:02d}s" if s else f"{m}分"
    h, m = divmod(m, 60)
    return f"{h}时{m:02d}分"


def append_round(request_id: str, *, round_idx: int,
                 frame_range: Optional[tuple] = None,
                 sub_queries: Optional[Sequence[str]] = None,
                 findings: str = "",
                 wall_epoch: Optional[float] = None) -> None:
    """Append one round's section.

    frame_range: (ts_start, ts_end) of the frames analysed this round (relative
                 per-session seconds), or None when no frames were consumed.
    wall_epoch:  the session's first-frame wall-clock time; when given, the frame
                 range is shown as absolute HH:MM:SS (user-readable) instead of
                 relative seconds.
    sub_queries: the sub-queries dispatched this round (search + recall briefs).
    findings:    the round's distilled result text (already synthesized).
    """
    path = file_path(request_id)
    if frame_range and frame_range[0] is not None and frame_range[1] is not None:
        _t0, _t1 = float(frame_range[0]), float(frame_range[1])
        _a = _fmt_clock(wall_epoch, _t0)
        _b = _fmt_clock(wall_epoch, _t1)
        # Duration from the RELATIVE ts delta (accurate regardless of wall-clock
        # rendering) so you can see exactly how long a span each round covered
        # and compare spans across rounds.
        _dur = _fmt_duration(_t1 - _t0)
        fr = (f"{_a} – {_b} (时长 {_dur})" if _a != _b
              else f"{_a} (时长 {_dur})")
    else:
        fr = "(无新增画面 / 仅记忆召回)"
    # "第 N 次分析" instead of the internal "Round N" — the user shouldn't see
    # implementation terms like round/batch.
    lines = [f"\n## 第 {round_idx} 次分析  ({_now_iso()})",
             f"- 分析的视频时段: {fr}",
             "- 本次子查询(sub-queries):"]
    sqs = [q for q in (sub_queries or []) if str(q).strip()]
    if sqs:
        for q in sqs:
            lines.append(f"  - [SQ] {' '.join(str(q).split())}")
    else:
        lines.append("  - (本轮未派发新子查询)")
    lines.append("- 调研结果:")
    body = (findings or "").strip() or "(本轮无有效发现)"
    # Indent findings block by 2 spaces so it reads as part of the section.
    lines.append("\n".join("  " + ln for ln in body.splitlines()))
    section = "\n".join(lines) + "\n"
    with _FILE_LOCK:
        if not path.exists():
            path.write_text(f"# Deep Analysis {request_id}\n\n## 分析记录\n",
                            encoding="utf-8")
        with path.open("a", encoding="utf-8") as f:
            f.write(section)


def append_note(request_id: str, note: str) -> None:
    """Append a small status/diagnostic line (e.g. '等待新帧…', '视频流已停止')."""
    line = f"\n> [{_now_iso()}] {' '.join(str(note or '').split())}\n"
    path = file_path(request_id)
    with _FILE_LOCK:
        if not path.exists():
            return
        with path.open("a", encoding="utf-8") as f:
            f.write(line)


def mark_finished(request_id: str, *, rounds: int, summary_preview: str = "") -> None:
    """Append a terminal marker and the complete consolidated final report.

    ``summary_preview`` is retained as the public keyword for compatibility,
    but its value is no longer flattened/truncated. The watcher panel and
    reopen path need the same complete report that was delivered at runtime.
    """
    summary = str(summary_preview or "").strip()
    if summary:
        summary_lines = summary.splitlines()
        summary_block = (
            f"- 最终汇总(摘要): {summary_lines[0]}\n"
            + "\n".join(summary_lines[1:])
            + ("\n" if len(summary_lines) > 1 else "")
        )
    else:
        summary_block = "- 最终汇总(摘要):\n"
    line = (f"\n## 完成  ({_now_iso()})\n"
            f"- 总轮数: {rounds}\n"
            f"{summary_block}")
    path = file_path(request_id)
    with _FILE_LOCK:
        if not path.exists():
            return
        with path.open("a", encoding="utf-8") as f:
            f.write(line)


def set_status(request_id: str, status: str, *, round_idx: Optional[int] = None,
               stop_reason: Optional[str] = None) -> None:
    """Rewrite the header's ``- 状态:`` line in place as
    ``<status> · 轮次 N · <iso>``.

    Single source of truth: the engine calls this each round / on stop / on
    finish, and the list/get tools read it back. No-op when the file is missing.
    When round_idx is omitted the previously-recorded round is preserved (only
    the status changes). Atomic write via a temp file + os.replace.
    """
    path = file_path(request_id)
    with _FILE_LOCK:
        if not path.exists():
            return
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        # 保留原轮次 (若本次没给)。
        _prev_round = None
        for ln in lines:
            m = _STATUS_RE.match(ln.strip())
            if m and m.group("round"):
                try:
                    _prev_round = int(m.group("round"))
                except ValueError:
                    _prev_round = None
                break
        rnd = round_idx if round_idx is not None else _prev_round
        rnd_part = f" · 轮次 {rnd}" if rnd is not None else ""
        new_line = f"- 状态: {status}{rnd_part} · {_now_iso()}"
        replaced = False
        for i, ln in enumerate(lines):
            if _STATUS_RE.match(ln.strip()):
                lines[i] = new_line
                replaced = True
                break
        if not replaced:
            # 无状态行 (极旧文件): 插到第一个 "## " 之前, 否则追加到头部末。
            ins = next((i for i, ln in enumerate(lines) if ln.startswith("## ")),
                       len(lines))
            lines.insert(ins, new_line)
        state = _state_from_lines(lines)
        if not state:
            state = {"version": 1, "request_id": str(request_id)}
        state["status"] = str(status)
        if rnd is not None:
            state["seg_base"] = int(rnd)
        if stop_reason is not None:
            state["stop_reason"] = str(stop_reason)
        _replace_state_line(lines, state)
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
            os.replace(tmp, path)
        except OSError:
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass


def read_status(request_id: str) -> Optional[dict]:
    """Parse the header status line → {"status", "round_idx", "updated_at"}.
    Returns None when the file or the status line is missing."""
    path = file_path(request_id)
    if not path.exists():
        return None
    try:
        with _FILE_LOCK:
            text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    for raw in text.splitlines():
        m = _STATUS_RE.match(raw.strip())
        if m:
            rnd = None
            if m.group("round"):
                try:
                    rnd = int(m.group("round"))
                except ValueError:
                    rnd = None
            return {"status": m.group("status"), "round_idx": rnd,
                    "updated_at": (m.group("ts") or "").strip()}
    return None


def scan_all(session_id: Optional[str] = None) -> dict:
    """Scan every analyse/watch_*.md → {rid: {status, round_idx, query,
    session_id, updated_at}}. Used by list_live_watcher and engine-startup
    reconciliation.

    session_id filter: a non-empty session_id returns only entries whose header
    session_id matches (legacy files with no session_id line are treated as
    "unowned" and excluded from every session's list). Omit it to return all
    entries (engine / migration use). This is what prevents cross-session leakage
    in the list. Files with status ``deleted`` are always skipped.
    """
    out: dict = {}
    want_sid = str(session_id or "").strip()
    try:
        adir = watch_dir()
    except Exception:
        return out
    for p in adir.glob("watch_*.md"):
        rid = p.stem[len("watch_"):]
        if not rid:
            continue
        try:
            with _FILE_LOCK:
                text = p.read_text(encoding="utf-8")
        except OSError:
            continue
        # ★ 五态统一: 默认 interrupted (解析不到 - 状态: 行 = 视为已中断, 不再用 unknown)。
        entry: dict = {
            "status": "interrupted", "round_idx": None, "query": "",
            "session_id": "", "state": {},
        }
        for raw in text.splitlines():
            s = raw.strip()
            ms = _STATUS_RE.match(s)
            if ms:
                entry["status"] = ms.group("status")
                if ms.group("round"):
                    try:
                        entry["round_idx"] = int(ms.group("round"))
                    except ValueError:
                        pass
                entry["updated_at"] = (ms.group("ts") or "").strip()
                continue
            msid = _SESSION_RE.match(s)
            if msid:
                entry["session_id"] = (msid.group("sid") or "").strip()
                continue
            mq = _QUERY_RE.match(s)
            if mq:
                entry["query"] = mq.group("q")
                continue
            mst = _STATE_RE.match(s)
            if mst:
                try:
                    parsed = json.loads(mst.group("json"))
                    if isinstance(parsed, dict):
                        entry["state"] = parsed
                except (TypeError, ValueError):
                    pass
        state = entry.get("state") or {}
        if not entry.get("session_id"):
            entry["session_id"] = str(state.get("session_id") or "")
        if state.get("task_instruction"):
            entry["query"] = str(state.get("task_instruction") or "")
        # ★ deleted: 已删除的任务文件保留在盘上(存档), 但一律不出现在列表 / 不 reopen 展示。
        if entry.get("status") == "deleted":
            continue
        # session 过滤: 只在调用方要求某个 session 时生效。
        if want_sid and entry.get("session_id", "") != want_sid:
            continue
        out[rid] = entry
    return out


def reconcile_stale(active_ids: "Sequence[str]") -> int:
    """Startup reconciliation: any file whose status is running/stopping but has
    no matching active task in the current process (not in active_ids) is marked
    interrupted (stale state left by a previous process that exited abnormally).
    Also corrects the header's recorded round against the real section count via
    count_rounds (guards against an over-written / inflated round number).
    Returns the number of files corrected."""
    active = set(active_ids or [])
    fixed = 0
    for rid, info in scan_all().items():
        st = info.get("status")
        real_rounds = count_rounds(rid)
        # 轮次校正: 头部记录与实际 round 段数不符 → 以实际为准。
        need_round_fix = (info.get("round_idx") is not None
                          and info["round_idx"] != real_rounds)
        if st in ("running", "stopping") and rid not in active:
            set_status(rid, "interrupted", round_idx=real_rounds)
            fixed += 1
        elif need_round_fix:
            set_status(rid, st or "interrupted", round_idx=real_rounds)
            fixed += 1
    return fixed


def read_all(request_id: str) -> str:
    """Full file text (fed to the final summary LLM). '' when missing."""
    path = file_path(request_id)
    if not path.exists():
        return ""
    try:
        with _FILE_LOCK:
            return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def read_structured(request_id: str) -> Optional[dict]:
    """Parse an analyse file into a structured object for the frontend's
    reopen-history read-only deep-analysis window.

    Returns:
      {
        "request_id": rid,
        "found": True,
        "status": <str>,            # running/done/interrupted/stopping/deleted
        "round_idx": <int|None>,
        "query": <str>,             # the header's 任务(query)
        "rounds": [ {"n": int, "frame_range": str, "sub_queries": [str],
                     "findings": str}, … ],   # one per "## 第 N 次分析" section
        "final_report": <str|None>, # summary from the "## 完成" section, if any
      }
    Missing file → {"request_id": rid, "found": False}. Pure parsing, no model
    call; status defaults to "interrupted" when no status line is present.
    """
    path = file_path(request_id)
    if not path.exists():
        return {"request_id": request_id, "found": False}
    try:
        with _FILE_LOCK:
            text = path.read_text(encoding="utf-8")
    except OSError:
        return {"request_id": request_id, "found": False}

    lines = text.splitlines()
    status_info = read_status(request_id) or {}
    query = ""
    rounds: List[dict] = []
    final_report: Optional[str] = None

    # 段解析: 逐行扫描, 遇 "## 第 N 次分析" 开新段; "## 完成" 收尾汇总。
    # 段内识别: "- 分析的视频时段: …" / "- [SQ] …" / "- 调研结果:" 之后的缩进块。
    cur: Optional[dict] = None
    mode: Optional[str] = None   # None | "findings" | "final"
    findings_buf: List[str] = []
    final_buf: List[str] = []

    def _flush_findings():
        if cur is not None:
            cur["findings"] = "\n".join(findings_buf).strip()

    for raw in lines:
        s = raw.strip()
        mq = _QUERY_RE.match(s)
        if mq and not query:
            query = mq.group("q")
            continue
        mr = _ROUND_RE.match(s)
        if mr:
            # 收尾上一段
            _flush_findings()
            findings_buf = []
            n = mr.group("n") or mr.group("n2")
            cur = {"n": int(n) if n else (len(rounds) + 1),
                   "frame_range": "", "sub_queries": [], "findings": ""}
            rounds.append(cur)
            mode = None
            continue
        if _FINISHED_RE.match(s):
            _flush_findings()
            findings_buf = []
            cur = None
            mode = "final"
            continue
        if mode == "final":
            # "## 完成" 段: 收集 "- 最终汇总(摘要): …" 及其后行。
            m_sum = re.match(r"^-\s*最终汇总\(摘要\):\s*(?P<v>.*)$", s)
            if m_sum:
                final_buf.append(m_sum.group("v"))
            elif s and not s.startswith("- 总轮数:"):
                final_buf.append(s)
            continue
        if cur is not None:
            m_fr = re.match(r"^-\s*分析的视频时段:\s*(?P<v>.+)$", s)
            if m_fr:
                cur["frame_range"] = m_fr.group("v").strip()
                mode = None
                continue
            m_sq = _SUBQUERY_RE.match(s)
            if m_sq:
                cur["sub_queries"].append(m_sq.group("q").strip())
                mode = None
                continue
            if re.match(r"^-\s*调研结果:", s):
                mode = "findings"
                findings_buf = []
                continue
            if re.match(r"^-\s*本次子查询", s):
                mode = None
                continue
            if mode == "findings":
                # 段内缩进 2 空格的调研结果块 (append_round 写入格式)。空行也保留。
                findings_buf.append(raw[2:] if raw.startswith("  ") else raw)
                continue

    _flush_findings()
    if final_buf:
        final_report = "\n".join(final_buf).strip() or None
    watcher_state = _state_from_lines(lines)
    if watcher_state.get("task_instruction"):
        query = str(watcher_state.get("task_instruction") or "")

    return {
        "request_id": request_id,
        "found": True,
        "status": status_info.get("status", "interrupted"),
        "round_idx": status_info.get("round_idx"),
        "query": query,
        "session_id": str(watcher_state.get("session_id") or ""),
        "state": watcher_state,
        "rounds": rounds,
        "final_report": final_report,
    }


def read_seen_subqueries(request_id: str) -> set:
    """Return the normalized set of sub-queries already recorded, for cross-round
    dedup. Normalization = lowercased, whitespace-collapsed (matches the Router
    task dedup key style)."""
    path = file_path(request_id)
    seen: set = set()
    if not path.exists():
        return seen
    try:
        with _FILE_LOCK:
            for raw in path.read_text(encoding="utf-8").splitlines():
                m = _SUBQUERY_RE.match(raw.strip())
                if m:
                    seen.add(" ".join(m.group("q").lower().split()))
    except OSError:
        return seen
    return seen


def count_rounds(request_id: str) -> int:
    """How many round sections are already in the file."""
    path = file_path(request_id)
    if not path.exists():
        return 0
    n = 0
    try:
        with _FILE_LOCK:
            for raw in path.read_text(encoding="utf-8").splitlines():
                if _ROUND_RE.match(raw.strip()):
                    n += 1
    except OSError:
        return 0
    return n


# A finished-run terminal marker, written by mark_finished().
_FINISHED_RE = re.compile(r"^##\s+完成\b")


def drop_last_incomplete_round(request_id: str) -> bool:
    """Reopen cleanup (for reopening after the app/web page was closed): if the
    run never finished (no ``## 完成`` marker) truncate the LAST
    ``## 第 N 次分析`` section, because a
    crash/close mid-round can leave a half-written segment. A finished run
    (``## 完成`` present) is left untouched. Returns True if a section was
    dropped.

    We only ever drop the final section — earlier rounds already have complete
    ``- 调研结果:`` bodies (append_round writes the section atomically)."""
    path = file_path(request_id)
    if not path.exists():
        return False
    with _FILE_LOCK:
        try:
            lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        except OSError:
            return False
        # A finished run keeps everything.
        if any(_FINISHED_RE.match(ln.strip()) for ln in lines):
            return False
        # Find the last round-section header.
        last_hdr = -1
        for i, ln in enumerate(lines):
            if _ROUND_RE.match(ln.strip()):
                last_hdr = i
        if last_hdr < 0:
            return False
        # Drop from that header to EOF, then append a note so the file records why.
        kept = lines[:last_hdr]
        note = (f"\n> [{_now_iso()}] 会话中断:已丢弃最后一个未完成的分析段"
                f" (第 ? 段, 无完成标记)。\n")
        path.write_text("".join(kept).rstrip() + "\n" + note, encoding="utf-8")
    log.info("[watch] dropped last incomplete round for %s", request_id)
    return True
