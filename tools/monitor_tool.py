# -*- coding: utf-8 -*-
"""set_monitor — the MonitorAgent control tool, decoupled from delegate_tool.

Previously this lived in ``tools/delegate_tool.py`` under the ``delegation``
toolset (conceptually tangled with set_live_watcher / WatcherAgent). It
is now its own ``monitor`` toolset and drives the independent MonitorAgent
(agent/multimodal/monitor_agent.py + the per-session daemon in tui_gateway).

Upgrades vs the old version:
  * ``monitor_query`` replaces ``brief`` (brief kept as a back-compat alias):
    the main agent should summarise context into ONE complete monitor task and
    create one independent monitor for each newly requested visual condition.
  * ``report_interval`` (T, seconds) and silent-mode now flow through create AND
    update.
  * Fast default: when T/silent are not given, alert immediately without a
    blocking Clarify round-trip. Explicit ``silent=true`` remains available.
  * On create the tool writes the monitor's event .md file header and returns
    the file path + current event ids, so the main agent can hand the user a
    concrete "monitoring started, recorded here" receipt and re-read it any time.
"""
from __future__ import annotations

import secrets as _secrets
import time as _t

from tools.registry import registry, tool_error, tool_handoff, tool_result


def _find_agent_by_session(session_id: str):
    """Walk gateway _sessions to find the agent + session keyed on hermes id."""
    if not session_id:
        return None, None
    try:
        from tui_gateway.server import _sessions
    except Exception:
        return None, None
    for sid, entry in list(_sessions.items()):
        agent = entry.get("agent")
        if agent is None:
            continue
        if entry.get("session_key") == session_id or sid == session_id:
            return entry, sid
    return None, None


def _monitor_label(m: dict) -> str:
    """User-facing short label; fall back to a truncated query if unset."""
    lbl = str(m.get("label", "") or "").strip()
    if lbl:
        return lbl
    q = str(m.get("monitor_query", "") or m.get("brief", "") or "").strip()
    return (q[:10] + "…") if len(q) > 10 else q


def _single_line(value) -> str:
    """Normalize user-controlled monitor metadata for registry/file parity."""
    return " ".join(str(value or "").split())


def _normalize_trigger_mode(value, *, default: str = "once") -> str:
    """Return the durable Monitor lifecycle mode.

    New creates default to one-shot.  Restore code passes ``default=continuous``
    for legacy event files whose historical behavior was long-lived.
    """
    mode = str(value or "").strip().lower()
    return mode if mode in {"once", "continuous"} else default


def _resolve_monitor_target(mons: dict, monitor_id=None, monitor_ref=None):
    """Resolve CRUD target without exposing the registry to the model.

    Resolution is deliberately conservative: exact id, exact label/query,
    unique substring, or the sole live entry.  Ambiguity is returned to the UI
    as labels; it is never resolved by picking the newest entry or merging
    monitor contracts.
    """
    mid = _single_line(monitor_id)
    if mid:
        return (mid, "") if mid in mons else ("", f"monitor_id {mid!r} not found")

    ref = _single_line(monitor_ref)
    if not ref:
        if len(mons) == 1:
            return next(iter(mons)), ""
        return "", "monitor_id or monitor_ref is required"

    # Successful Monitor controls are intentionally absent from model history.
    # A later command may therefore only say "pause that monitor". Resolving a
    # generic pointer is safe when this session has exactly one entry; with
    # multiple entries we still refuse to guess.
    generic_ref = ref.casefold().replace(" ", "")
    if len(mons) == 1 and generic_ref in {
        "这个监控", "那个监控", "刚才的监控", "刚刚的监控",
        "上一个监控", "当前监控", "我的监控", "监控",
        "thismonitor", "thatmonitor", "themonitor", "currentmonitor",
    }:
        return next(iter(mons)), ""

    folded = ref.casefold()
    if ref in mons:
        return ref, ""

    def _fields(item):
        return (
            _monitor_label(item).casefold(),
            _single_line(item.get("monitor_query") or item.get("brief")).casefold(),
        )

    exact = [key for key, item in mons.items() if folded in _fields(item)]
    candidates = exact
    if not candidates:
        candidates = [
            key for key, item in mons.items()
            if any(folded in field or field in folded for field in _fields(item) if field)
        ]
    if not candidates:
        # Remove only generic control nouns and retry a unique containment
        # match. This lets "水杯监控" identify "水杯出现提醒" without
        # fuzzy scores or newest-item fallbacks that could mutate the wrong job.
        def _distinctive(value: str) -> str:
            clean = value.casefold().replace(" ", "")
            for token in (
                "monitor", "alert", "task", "reminder",
                "监控任务", "监控", "任务", "提醒", "通知", "告诉我",
            ):
                clean = clean.replace(token, "")
            return clean

        needle = _distinctive(ref)
        if len(needle) >= 2:
            candidates = [
                key for key, item in mons.items()
                if any(
                    needle in _distinctive(field)
                    or _distinctive(field) in needle
                    for field in _fields(item)
                    if _distinctive(field)
                )
            ]
    if len(candidates) == 1:
        return candidates[0], ""
    if not candidates:
        return "", f"No monitor matches {ref!r}."
    labels = ", ".join(_monitor_label(mons[key]) or key for key in candidates[:6])
    return "", f"Monitor target is ambiguous. Please select one in the UI: {labels}"


def _push_monitors_event(sid: str, agent) -> None:
    """Push the current monitor registry to the frontend (best-effort)."""
    if not sid or agent is None:
        return
    try:
        from tui_gateway.server import _emit
        from agent.multimodal import monitor_agent as _ma
        mons = list((getattr(agent, "mm_monitors", {}) or {}).values())
        payload = []
        for m in mons:
            # ★ 五态 status: 执行态以文件头为准(running/interrupted/done); 读不到则从
            #   enabled 兜底(True→running, False→interrupted)。供前端显示状态标签。
            _fst = (_ma.read_status(m.get("id")) or {}).get("status")
            _status = _fst or ("running" if m.get("enabled", True) else "interrupted")
            payload.append({
                "monitor_id": m.get("id"),
                "brief": m.get("monitor_query", "") or m.get("brief", ""),
                "monitor_query": m.get("monitor_query", "") or m.get("brief", ""),
                "label": _monitor_label(m),
                "enabled": bool(m.get("enabled", True)),
                "status": _status,
                "trigger_mode": _normalize_trigger_mode(
                    m.get("trigger_mode"), default="continuous"),
                "silent": bool(m.get("silent", False)),
                "report_interval": m.get("report_interval"),
                "created_at": m.get("created_at", 0.0)})
        _emit("multimodal.monitors", sid, {"monitors": payload})
    except Exception:
        pass


def _normalize_report_interval(value) -> "float | None":
    """Coerce report_interval (seconds) to a positive float, else None.

    None / missing / non-positive / non-numeric → None. The main agent converts
    user phrasing ("every minute" → 60) into seconds before calling.
    """
    if value is None:
        return None
    try:
        t = float(value)
    except (TypeError, ValueError):
        return None
    return t if t > 0 else None


def _resolve_silent(agent, *, report_interval, silent_arg) -> "tuple[bool, str]":
    """Decide silent mode. Returns (silent, note).

    Deterministic fast-path rules:
      - T given (report_interval > 0) → never ask; silent=False (periodic report).
      - silent explicitly passed → honour it.
      - T absent AND silent unspecified → silent=False (alert on sight).

    Monitor creation must not block on an interactive Clarify: callers such as
    the multimodal fast-path transfer reply ownership in the same tool turn.
    """
    if report_interval and report_interval > 0:
        return False, ""
    if silent_arg is not None:
        return bool(silent_arg), ""
    _ = agent  # kept in the signature for call-site compatibility
    return False, ""


def _rollback_failed_create_file(monitor_agent, monitor_id: str) -> bool:
    """Remove a file created by this failed create transaction.

    The monitor id is freshly generated and the engine has not acknowledged a
    job, so no historical events can belong to it. If unlink is unavailable,
    a deleted tombstone still keeps scan/list/resume from reviving the file.
    """
    try:
        if monitor_agent.discard_event_file(monitor_id):
            return True
    except Exception:
        pass
    try:
        return bool(monitor_agent.set_status(monitor_id, "deleted"))
    except Exception:
        return False


SET_MONITOR_SCHEMA = {
    "name": "set_monitor",
    "description": (
        "Create or manage an independent live screen/camera Monitor. A new visual "
        "condition is always a separate monitor; NEVER merge it into another monitor. "
        "Output is a per-event alert; it never produces "
        "a summary/report/analysis (use set_live_watcher for those).\n"
        "FAST PATH: for a clear request to start a new monitor, call "
        "set_monitor(op='create') directly as the first and only control tool. Do not "
        "call skill_view or check_video_stream first; create validates "
        "the live source itself and returns an actionable error if unavailable.\n"
        "Operations:\n"
        "  - op='create'  : start a monitor. Pass monitor_query (+ label).\n"
        "  - op='update'  : explicitly change an existing monitor. Pass monitor_id or monitor_ref.\n"
        "  - op='enable'/'disable'/'delete': manage one existing monitor by id/ref.\n"
        "Only use update when the user explicitly asks to modify an existing monitor. "
        "Phrases such as 'also monitor phones' describe another condition and must create "
        "a separate monitor. The backend resolves monitor_ref against the live registry; "
        "never request or synthesize a monitor list."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "op": {
                "type": "string",
                "enum": ["create", "update", "enable", "disable", "delete"],
                "description": "Which operation to perform.",
            },
            "monitor_id": {
                "type": "string",
                "description": "Exact Monitor id for update/enable/disable/delete when already known.",
            },
            "monitor_ref": {
                "type": "string",
                "description": (
                    "Human label or distinctive query fragment for an existing monitor. "
                    "Use for update/enable/disable/delete when monitor_id is unknown. "
                    "The backend resolves it conservatively and refuses ambiguous matches."
                ),
            },
            "monitor_query": {
                "type": "string",
                "description": (
                    "create/update: a complete, self-contained description of the monitor "
                    "task, addressed to the worker as the executor (what to watch for, what "
                    "counts as the trigger). "
                    "The worker never sees the original conversation, so resolve pronouns "
                    "and missing context, but preserve the user's semantic scope exactly. "
                    "Do NOT invent exclusions, confidence gates, narrower object taxonomies, "
                    "minimum durations, or repeat/transition requirements that the user did "
                    "not request. Common subtypes and form variants that belong to the user's "
                    "ordinary-language category remain in scope unless the user explicitly "
                    "excluded them. "
                    "Leave out any reporting-cadence phrasing, as that is captured separately "
                    "in the report_interval parameter. "
                    "Video frames come from the session FrameBuffer; do not require or pass "
                    "image_url. "
                    "Prefer one monitor per request."
                ),
            },
            "label": {
                "type": "string",
                "description": "Short label for the UI (a few words, e.g. 'delivery alert'). Set on create/update.",
            },
            "trigger_mode": {
                "type": "string",
                "enum": ["once", "continuous"],
                "description": (
                    "Lifecycle for create/update. once = finish after the first accepted "
                    "hit. continuous = keep watching and re-arm after the condition clears. "
                    "For every create, infer the lifecycle from the user's full intent and "
                    "pass it explicitly; when wording contains competing cues, resolve them "
                    "from context instead of omitting this field. New creates retain a "
                    "backward-compatible once default only for older callers."
                ),
            },
            "report_interval": {
                "type": "number",
                "description": (
                    "Reporting period T, in SECONDS. create/update.\n"
                    "  - positive T = batch the events seen in each T-window into ONE alert; nothing seen in the window -> no message.\n"
                    "  - omit = no periodic cadence; immediate alerting is the deterministic default.\n"
                    "  - 0 (on update) = cancel periodic mode; silent=true stays "
                    "background-only, otherwise alerts become immediate.\n"
                    "Only set when the user explicitly asks for a fixed cadence ('every 5 minutes'); otherwise omit. "
                    "This is still just periodic ALERTS — for periodic summaries/reports use set_live_watcher."
                ),
            },
            "silent": {
                "type": "boolean",
                "default": False,
                "description": (
                    "Optional. Silent = only record events to file, never interrupt the user. "
                    "Pass silent=false when the user says tell/remind/notify/alert me "
                    "('tell me', 'remind me', 'notify me', 'say something when you see it'); pass silent=true only when "
                    "they explicitly want background-only recording. If omitted with no "
                    "report_interval, the tool deterministically defaults to silent=false "
                    "without a Clarify. "
                    "Ignored when report_interval (periodic reporting) is set."
                ),
            },
            "hook_main_agent": {
                "type": "boolean",
                "description": (
                    "Whether the main agent should perform an action when this monitor fires. Default false.\n"
                    "false: just alert the user (the hit bubble IS the alert). 'remind me / tell me / say something when you see X' are all this case.\n"
                    "true: after a hit the main agent also does something (look things up / record / send a message / change config, etc.); put that action in hook_instruction.\n"
                    "Pass false on op=update to cancel an existing hook."
                ),
            },
            "hook_instruction": {
                "type": "string",
                "description": (
                    "The action handed to the main agent after a hit, when hook_main_agent=true.\n"
                    "Write it as a self-contained imperative: only the action to do; minimize anaphoric references; do NOT restate the monitor condition, do NOT paraphrase the user.\n"
                    "The hit's verdict text is appended automatically after your instruction, so just describe the action (e.g. 'search for what was seen in the frame' / 'log this error into the bug list') — no need to reference the result yourself."
                ),
            },
        },
        "required": ["op"],
    },
}


def set_monitor(op=None, monitor_id=None, monitor_ref=None,
                monitor_query=None, brief=None, label=None,
                trigger_mode=None, report_interval=None, silent=None,
                hook_main_agent=None, hook_instruction=None,
                session_id=None, **_kw) -> str:
    """CRUD over agent.mm_monitors, decoupled MonitorAgent control."""
    from agent.multimodal import monitor_agent as _ma

    op = (op or "").strip().lower()
    # brief is a back-compat alias for monitor_query.
    query = _single_line(monitor_query or brief)

    entry, sid = _find_agent_by_session(session_id or "")
    if entry is None:
        return tool_error(
            "set_monitor cannot reach the session's agent. Multimodal session "
            "may not be active.", success=False)
    agent = entry.get("agent")
    # 持久 session id (跨 resume 稳定; 与前端 ?mm=/localStorage/history 同源)。写入
    # 事件文件头, 供 scan_all 按 session 过滤 (堵跨 session 泄漏)。
    _stored_sid = str(entry.get("session_key") or sid or "")
    if not hasattr(agent, "mm_monitors") or agent.mm_monitors is None:
        agent.mm_monitors = {}
    mons = agent.mm_monitors
    # MonitorEngine (async job container). create → spawn a per-monitor job,
    # delete → cancel it. enable/disable/update just mutate the shared entry,
    # which the running job reads live (no engine call needed).
    _engine = entry.get("_mm_monitor_engine")
    target_mid = ""
    if op in {"update", "enable", "disable", "delete"}:
        target_mid, target_error = _resolve_monitor_target(
            mons,
            monitor_id=monitor_id,
            monitor_ref=monitor_ref,
        )
        if target_error:
            return tool_error(
                f"Cannot {op} monitor: {target_error}", success=False)

    if op == "create":
        if not query:
            return tool_error("monitor_query is required for op=create", success=False)
        if _engine is None:
            return tool_error(
                "Cannot start monitor: the Monitor backend is not ready. Please restart the multimodal session and try again.",
                success=False,
            )
        try:
            _engine_ready = bool(_engine.is_healthy())
        except Exception:
            _engine_ready = False
        if not _engine_ready:
            return tool_error(
                "Cannot start monitor: the Monitor backend is unavailable. Please check the multimodal backend.",
                success=False,
            )
        # Require a live video stream: a monitor watches the stream, so with no
        # active stream (never opened / sharing stopped) it can't work. Fail with
        # the reason so the main agent asks the user to (re)start the share.
        # Uses the SAME authoritative UI-switch verdict as set_live_watcher:
        # prefer the WatcherAgent's is_source_live(); fall back to the raw buffer.
        try:
            from tools.live_watcher_tool import mm_stream_status as _mm_stream_status
            _src = entry.get("_mm_live_watcher_agent") or getattr(agent, "frame_buffer", None)
            _live, _why = _mm_stream_status(_src)
        except Exception as exc:
            return tool_error(
                f"Cannot start monitor: could not confirm video stream status ({exc}).",
                success=False,
            )
        if not _live:
            return tool_error(f"Cannot start monitor: {_why}", success=False)
        lbl = _single_line(label)
        ri = _normalize_report_interval(report_interval)
        # A periodic digest is inherently long-lived. Preserve that established
        # caller contract when no explicit lifecycle was supplied; all other new
        # monitors default to one-shot.
        mode = _normalize_trigger_mode(
            trigger_mode,
            default="continuous" if ri is not None else "once",
        )
        if mode == "once" and ri is not None:
            return tool_error(
                "Cannot start monitor: trigger_mode=once cannot be used with report_interval.",
                success=False,
            )
        is_silent, note = _resolve_silent(agent, report_interval=ri, silent_arg=silent)
        if is_silent and hook_main_agent:
            return tool_error(
                "Cannot start monitor: silent=true conflicts with hook_main_agent=true.",
                success=False,
            )
        event_file = ""
        mid = ""
        for _attempt in range(10):
            candidate = "mon_" + _secrets.token_hex(8)
            if candidate in mons:
                continue
            try:
                event_file = _ma.init_event_file(
                    candidate,
                    label=lbl or _monitor_label({"monitor_query": query}),
                    monitor_query=query,
                    trigger_mode=mode,
                    silent=is_silent,
                    report_interval=ri,
                    session_id=_stored_sid,
                    hook_main_agent=bool(hook_main_agent),
                    hook_instruction=_single_line(hook_instruction),
                    require_new=True,
                )
                mid = candidate
                break
            except FileExistsError:
                continue
            except Exception as exc:
                _rollback_failed_create_file(_ma, candidate)
                return tool_error(
                    f"Cannot start monitor: failed to initialize the event record ({exc}).",
                    success=False,
                )
        if not mid:
            return tool_error(
                "Cannot start monitor: failed to allocate a unique monitor_id.",
                success=False,
            )
        try:
            current_event_ids = _ma.read_event_ids(mid)
        except Exception as exc:
            _rollback_failed_create_file(_ma, mid)
            return tool_error(
                f"Cannot start monitor: failed to initialize the event record ({exc}).",
                success=False,
            )
        mons[mid] = {
            "id": mid, "monitor_query": query, "brief": query, "label": lbl,
            "enabled": True, "status": "running", "trigger_mode": mode,
            "_config_revision": 0,
            "silent": is_silent, "report_interval": ri,
            "event_file": event_file, "created_at": _t.time(),
            "last_speak_ts": 0.0, "_agg_buf": [], "_agg_window_start": 0.0,
            # Main-agent hook: fire this monitor -> send hook_instruction to the
            # main agent as a user message (only when idle). Default off.
            "hook_main_agent": bool(hook_main_agent),
            "hook_instruction": _single_line(hook_instruction),
        }
        agent.mm_monitor_active = True
        try:
            scheduled = bool(_engine.add_monitor(mid))
        except Exception:
            scheduled = False
        if not scheduled:
            mons.pop(mid, None)
            agent.mm_monitor_active = any(
                bool(m.get("enabled", True)) for m in mons.values())
            _rollback_failed_create_file(_ma, mid)
            return tool_error(
                "Cannot start monitor: the Monitor backend could not schedule the background job.",
                success=False,
            )
        _push_monitors_event(sid, agent)
        if is_silent:
            _started_note = (
                f"Monitor scheduled on the live video stream and is waiting for the first check: "
                f"\"{_monitor_label(mons[mid])}\". Hits will be recorded in the background only; no proactive alerts.")
        else:
            _started_note = (
                f"Monitor scheduled on the live video stream and is waiting for the first check: "
                f"\"{_monitor_label(mons[mid])}\". It will proactively alert you when matching frames appear.")
        # If a hook was set at create time, surface it too (parity with update).
        _hook_note = ""
        if mons[mid].get("hook_main_agent") and mons[mid].get("hook_instruction"):
            _hit_scope = "the first accepted hit" if mode == "once" else "each new event hit"
            _hook_note = (
                f"Main-agent hook attached: on {_hit_scope}, and only when the main agent is idle, "
                f"run: \"{mons[mid]['hook_instruction']}\".")
        return tool_result({
            "op": "create", "monitor_id": mid, "monitor_query": query,
            "label": _monitor_label(mons[mid]), "trigger_mode": mode,
            "silent": is_silent,
            "report_interval": ri, "event_file": event_file,
            "current_event_ids": current_event_ids,
            "status": "running",
            "hook_main_agent": bool(mons[mid].get("hook_main_agent", False)),
            "hook_instruction": mons[mid].get("hook_instruction", ""),
            "note": " ".join(x for x in (note, _started_note, _hook_note) if x),
        })

    if op == "update":
        mid = target_mid
        original = mons[mid]
        try:
            current_status = str(
                (_ma.read_status(mid) or {}).get("status")
                or original.get("status")
                or ""
            ).strip().lower()
        except Exception:
            current_status = str(original.get("status") or "").strip().lower()
        if current_status in {"done", "complete"}:
            return tool_error(
                "Cannot update: this one-shot monitor is already complete. Create a new monitor instead.",
                success=False,
            )
        updated = dict(original)
        current_mode = _normalize_trigger_mode(
            original.get("trigger_mode"), default="continuous")
        updated["trigger_mode"] = current_mode
        changed_mode = None
        if trigger_mode is not None:
            next_mode = _normalize_trigger_mode(
                trigger_mode, default=current_mode)
            updated["trigger_mode"] = next_mode
            if next_mode != current_mode:
                changed_mode = next_mode
        if query:
            updated["monitor_query"] = query
            updated["brief"] = query
            # Changing the task is a "give it another chance" signal — clear the
            # circuit-breaker state so a previously auto-disabled monitor isn't
            # instantly re-tripped / left silent after the user fixes its config.
            updated["_fail_streak"] = 0
            updated["_err_notified"] = False
        lbl = _single_line(label)
        if lbl:
            updated["label"] = lbl
        note = ""
        _changed_silent = None
        _changed_ri = None
        if report_interval is not None or silent is not None:
            new_ri = _normalize_report_interval(report_interval)
            is_silent, note = _resolve_silent(
                agent, report_interval=new_ri, silent_arg=silent)
            if new_ri != updated.get("report_interval"):
                updated["report_interval"] = new_ri
                updated["_agg_buf"] = []
                updated["_agg_window_start"] = 0.0
                _changed_ri = new_ri
            updated["silent"] = is_silent
            _changed_silent = is_silent
        # Main-agent hook: attach/update/clear. Passing hook_main_agent updates
        # the flag; hook_instruction updates the task. Setting hook_main_agent=false
        # detaches (the desc is kept but inert).
        _hook_note = ""
        if hook_main_agent is not None:
            updated["hook_main_agent"] = bool(hook_main_agent)
        if hook_instruction is not None:
            updated["hook_instruction"] = _single_line(hook_instruction)
        if (updated.get("trigger_mode") == "once"
                and updated.get("report_interval") is not None):
            return tool_error(
                "Cannot update monitor: trigger_mode=once cannot be used with report_interval.",
                success=False,
            )
        if bool(updated.get("silent")) and bool(updated.get("hook_main_agent")):
            return tool_error(
                "Cannot update monitor: silent=true conflicts with hook_main_agent=true.",
                success=False,
            )
        if updated.get("hook_main_agent") and updated.get("hook_instruction"):
            _hit_scope = (
                "the first accepted hit"
                if updated.get("trigger_mode") == "once"
                else "each new event hit"
            )
            _hook_note = (
                f"Main-agent hook attached: on {_hit_scope}, and only when the main agent is idle, "
                f"run: \"{updated['hook_instruction']}\".")
        elif hook_main_agent is False:
            _hook_note = "Main-agent hook removed for this monitor. It will only alert and will no longer trigger the main agent."
        contract_changed = bool(
            query
            or lbl
            or changed_mode is not None
            or report_interval is not None
            or silent is not None
            or hook_main_agent is not None
            or hook_instruction is not None
        )
        # Invalidate an in-flight verdict before performing disk I/O. During the
        # atomic header rewrite the old live contract remains readable, so the
        # explicit updating gate prevents a new evaluation from capturing it
        # under the freshly bumped revision.
        if contract_changed:
            with _ma.monitor_state_lock(original):
                if mons.get(mid) is not original:
                    return tool_error(
                        "Cannot update monitor: monitor state changed during the update. Please retry.",
                        success=False,
                    )
                if str(original.get("status") or "").lower() in {"done", "complete"}:
                    return tool_error(
                        "Cannot update: this one-shot monitor is already complete. Create a new monitor instead.",
                        success=False,
                    )
                original["_config_revision"] = (
                    int(original.get("_config_revision", 0) or 0) + 1
                )
                original["_config_updating"] = True
                original.pop("_once_pending_completion", None)
        # The event-file header is the provider-safe resume source for direct
        # monitor turns. Persist every mutable field before exposing the update
        # in memory; otherwise a restart would restore the create-time values.
        try:
            persisted = _ma.update_event_file_config(
                mid,
                label=_monitor_label(updated),
                monitor_query=updated.get("monitor_query", ""),
                silent=bool(updated.get("silent", False)),
                report_interval=updated.get("report_interval"),
                hook_main_agent=bool(updated.get("hook_main_agent", False)),
                hook_instruction=updated.get("hook_instruction", ""),
                trigger_mode=updated.get("trigger_mode"),
            )
        except Exception:
            persisted = False
        if not persisted:
            if contract_changed:
                with _ma.monitor_state_lock(original):
                    original.pop("_config_updating", None)
            return tool_error(
                "Cannot update monitor: failed to write the event archive. The original configuration was kept.",
                success=False,
            )
        # Commit the live contract under the same lock used by verdict delivery.
        # Bumping the revision invalidates any model request that captured the
        # previous query/mode/delivery settings before its await.
        with _ma.monitor_state_lock(original):
            if mons.get(mid) is not original:
                return tool_error(
                    "Cannot update monitor: monitor state changed during the update. Please retry.",
                    success=False,
                )
            if str(original.get("status") or "").lower() in {"done", "complete"}:
                original.pop("_config_updating", None)
                return tool_error(
                    "Cannot update: this one-shot monitor is already complete. Create a new monitor instead.",
                    success=False,
                )
            if query:
                original["monitor_query"] = updated["monitor_query"]
                original["brief"] = updated["brief"]
                original["_fail_streak"] = 0
                original["_err_notified"] = False
            if lbl:
                original["label"] = updated["label"]
            if report_interval is not None or silent is not None:
                if updated.get("report_interval") != original.get("report_interval"):
                    original["_agg_buf"] = []
                    original["_agg_window_start"] = 0.0
                original["report_interval"] = updated.get("report_interval")
                original["silent"] = bool(updated.get("silent", False))
            if hook_main_agent is not None:
                original["hook_main_agent"] = bool(
                    updated.get("hook_main_agent", False))
            if hook_instruction is not None:
                original["hook_instruction"] = updated.get("hook_instruction", "")
            original["trigger_mode"] = updated.get("trigger_mode", current_mode)
            if contract_changed:
                original.pop("_config_updating", None)
                original.pop("_trigger_armed", None)
                original.pop("_once_pending_completion", None)
                original.pop("_once_delivery_error_notified", None)
                original.pop("_once_status_error_notified", None)
                original.pop("_once_event_error_notified", None)
        if (query or _changed_silent is not None or _changed_ri is not None
                or changed_mode is not None):
            try:
                _ma.mark_task_change(
                    mid, new_query=query if query else "",
                    silent=_changed_silent, report_interval=_changed_ri,
                    trigger_mode=changed_mode)
            except Exception:
                pass
        _push_monitors_event(sid, agent)
        _final_note = " ".join(x for x in (note, _hook_note) if x) or None
        return tool_result({
            "op": "update", "monitor_id": mid,
            "monitor_query": original.get("monitor_query", ""),
            "label": _monitor_label(original),
            "trigger_mode": original.get("trigger_mode", "continuous"),
            "silent": bool(original.get("silent", False)),
            "report_interval": original.get("report_interval"),
            "event_file": original.get("event_file"),
            "hook_main_agent": bool(original.get("hook_main_agent", False)),
            "hook_instruction": original.get("hook_instruction", ""),
            "note": _final_note,
        })

    if op in ("enable", "disable"):
        mid = target_mid
        target = mons[mid]
        # ★ Enabling a (paused/interrupted) monitor needs a LIVE video stream —
        #   same guard as op=create. After an app/session close the monitor's
        #   engine job is gone and the stream isn't shared; re-enabling without a
        #   stream would flip enabled=True over a dead pipe. Fail so the UI toggle
        #   rolls back to off (the "点 on 没流就自动弹回" behavior) and the user
        #   knows to (re)start sharing first.
        if op == "enable":
            try:
                current_status = (_ma.read_status(mid) or {}).get("status")
            except Exception:
                current_status = mons[mid].get("status")
            if current_status == "done":
                return tool_error(
                    "Cannot resume: this one-shot monitor is already complete. Create a new monitor instead.",
                    success=False,
                )
            if _engine is None:
                return tool_error(
                    "Cannot enable monitor: the Monitor backend is not ready.",
                    success=False,
                )
            try:
                _engine_ready = bool(_engine.is_healthy())
            except Exception:
                _engine_ready = False
            if not _engine_ready:
                return tool_error(
                    "Cannot enable monitor: the Monitor backend is unavailable.",
                    success=False,
                )
            try:
                from tools.live_watcher_tool import mm_stream_status as _mm_stream_status
                _src = entry.get("_mm_live_watcher_agent") or getattr(agent, "frame_buffer", None)
                _live, _why = _mm_stream_status(_src)
            except Exception as exc:
                return tool_error(
                    f"Cannot enable monitor: could not confirm video stream status ({exc}).",
                    success=False,
                )
            if not _live:
                return tool_error(f"Cannot enable monitor: {_why}", success=False)
        # This lock is also held by verdict append/delivery. Whichever operation
        # acquires it first commits first; after disable returns, no older model
        # response can append or emit a late alert.
        with _ma.monitor_state_lock(target):
            if mons.get(mid) is not target:
                return tool_error(
                    "Cannot switch monitor: monitor state changed. Please retry.",
                    success=False,
                )
            if op == "enable":
                previous_enabled = bool(target.get("enabled", False))
                previous_status = str(current_status or "interrupted")
                if not _ma.set_status(mid, "running"):
                    return tool_error(
                        "Cannot enable monitor: failed to write the event archive status.",
                        success=False,
                    )
                target["enabled"] = True
                target["status"] = "running"
                target["_config_revision"] = (
                    int(target.get("_config_revision", 0) or 0) + 1
                )
                target.pop("_trigger_armed", None)
                target.pop("_once_pending_completion", None)
                try:
                    scheduled = bool(_engine.add_monitor(mid))
                except Exception:
                    scheduled = False
                if not scheduled:
                    target["enabled"] = previous_enabled
                    target["status"] = previous_status
                    _ma.set_status(mid, previous_status)
                    return tool_error(
                        "Cannot enable monitor: the Monitor backend could not schedule the background job.",
                        success=False,
                    )
                target["_fail_streak"] = 0
                target["_err_notified"] = False
                target.pop("_interrupted", None)
            else:
                if not _ma.set_status(mid, "interrupted"):
                    return tool_error(
                        "Cannot pause monitor: failed to write the event archive status.",
                        success=False,
                    )
                target["_config_revision"] = (
                    int(target.get("_config_revision", 0) or 0) + 1
                )
                target["enabled"] = False
                target["status"] = "interrupted"
                target.pop("_once_pending_completion", None)
                if _engine is not None:
                    try:
                        _engine.remove_monitor(mid)
                    except Exception:
                        pass
        _push_monitors_event(sid, agent)
        agent.mm_monitor_active = any(
            bool(item.get("enabled", False))
            for item in mons.values()
        )
        return tool_result({"op": op, "monitor_id": mid,
                            "enabled": mons[mid]["enabled"],
                            "status": mons[mid].get("status"),
                            "trigger_mode": _normalize_trigger_mode(
                                mons[mid].get("trigger_mode"),
                                default="continuous"),
                            "label": _monitor_label(mons[mid])})

    if op == "delete":
        mid = target_mid
        target = mons[mid]
        # ★ 五态统一: 删除 = 文件头落 status=deleted (monitor 无 stopping, 瞬时删)。
        #   必须【先】写 deleted, 再 remove_monitor —— 否则 job 的 finally 会把它
        #   落成 interrupted。deleted 的文件 scan_all 一律跳过 → UI 永不展示 (存档保留)。
        with _ma.monitor_state_lock(target):
            if mons.get(mid) is not target:
                return tool_error(
                    "Cannot delete monitor: monitor state changed. Please retry.",
                    success=False,
                )
            try:
                archived = bool(_ma.set_status(mid, "deleted"))
            except Exception:
                archived = False
            if not archived:
                return tool_error(
                    "Cannot delete monitor: failed to write the event archive status. The current monitor was kept.",
                    success=False,
                )
            target["_config_revision"] = (
                int(target.get("_config_revision", 0) or 0) + 1
            )
            target["enabled"] = False
            target["status"] = "deleted"
            target.pop("_once_pending_completion", None)
            mons.pop(mid, None)
            if _engine is not None:
                try:
                    _engine.remove_monitor(mid)
                except Exception:
                    pass
        agent.mm_monitor_active = any(
            bool(item.get("enabled", False))
            for item in mons.values()
        )
        # 事件文件 monitor_<id>.md 保留在盘上(存档观测记录), 仅头部标 deleted。
        _push_monitors_event(sid, agent)
        return tool_result({"op": "delete", "monitor_id": mid})

    return tool_error(
        "unknown op; expected one of: create, update, enable, disable, "
        "delete", success=False)


def _set_monitor_handoff(*, session_id=None, **kwargs) -> str:
    """Model-facing set_monitor path: mutate once, then transfer reply ownership.

    Direct Python callers keep receiving the undecorated CRUD payload from
    :func:`set_monitor`; only the registered model tool is terminal.  This keeps
    the control protocol out of the domain implementation and makes success and
    failure receipts equally one-stage.
    """
    op = str(kwargs.get("op") or "").strip().lower()
    if op == "create":
        # Ambiguous natural-language requests intentionally leave the local
        # fast path so the main model can choose the lifecycle. Do not undo
        # that decision here by silently applying set_monitor()'s legacy
        # Python-caller default when the model omitted or misspelled the field.
        # This is a plain tool error (not a terminal handoff), so the model can
        # correct the arguments on its next tool round without any side effect.
        requested_mode = str(kwargs.get("trigger_mode") or "").strip().lower()
        if requested_mode not in {"once", "continuous"}:
            return tool_error(
                "Creating a monitor requires trigger_mode to be explicitly set "
                "to once or continuous. Infer it from the user's full intent, "
                "then call set_monitor again. No monitor was created.",
                success=False,
                code="monitor_trigger_mode_required",
            )
        kwargs["trigger_mode"] = requested_mode

    raw = set_monitor(session_id=session_id, **kwargs)
    try:
        import json
        data = json.loads(raw)
    except Exception:
        data = {}
    entry, _sid = _find_agent_by_session(str(session_id or ""))
    agent = entry.get("agent") if isinstance(entry, dict) else None
    parent_id = str(
        getattr(agent, "_active_parent_user_message_id", "") or "")
    failed = bool(data.get("error") or data.get("success") is False)
    task_id = "" if failed else str(
        data.get("monitor_id")
        or f"monitor_ctl_{_secrets.token_hex(3)}"
    )
    op = str(data.get("op") or kwargs.get("op") or "monitor")
    label = str(data.get("label") or data.get("monitor_query") or "").strip()
    label_suffix = f' "{label}"' if label else ""
    fallback_ack = {
        "create": f"Started monitor{label_suffix}.",
        "update": f"Updated monitor{label_suffix}.",
        "enable": f"Resumed monitor{label_suffix}.",
        "disable": f"Paused monitor{label_suffix}.",
        "delete": "Deleted this monitor.",
    }.get(op, "Monitor operation handled.")
    ack = str(data.get("note") or data.get("error") or fallback_ack).strip()
    return tool_handoff(
        raw,
        reply_owner="main" if failed else "monitor",
        handoff_mode="receipt",
        history_policy="persist" if failed else "ephemeral_control",
        task_id=task_id,
        parent_user_message_id=parent_id,
        ack=ack,
    )


registry.register(
    name="set_monitor",
    toolset="monitor",
    schema=SET_MONITOR_SCHEMA,
    handler=lambda args, **kw: _set_monitor_handoff(
        op=args.get("op"),
        monitor_id=args.get("monitor_id"),
        monitor_ref=args.get("monitor_ref"),
        monitor_query=args.get("monitor_query"),
        brief=args.get("brief"),
        label=args.get("label"),
        trigger_mode=args.get("trigger_mode"),
        report_interval=args.get("report_interval"),
        silent=args.get("silent"),
        hook_main_agent=args.get("hook_main_agent"),
        hook_instruction=args.get("hook_instruction"),
        session_id=kw.get("session_id"),
    ),
    emoji="👁",
)
