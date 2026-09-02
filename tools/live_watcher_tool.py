#!/usr/bin/env python3
"""
Live-Watcher tools: the main-agent entry point for multimodal background
deep research.

This module exposes:
  * set_live_watcher   - hand a video-stream task to a background WatcherAgent
  * get_live_watcher   - read progress or reports for one research task
  * list_live_watcher  - list research tasks for the current session

Watcher tools resolve per-session engines through the gateway session registry;
they do not use the generic delegate_task sub-agent mechanism.
"""
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_WATCHER_ID_RE = re.compile(r"^req_[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")

from tools.registry import registry, tool_error, tool_handoff, tool_result

# --------------------------------------------------------------------------- #
# NOTE: The phase-3 `route_complex_query` tool was removed (phase 12). It was a
# thin delegate_task wrapper (text-only subagent, no video/memory) that predated
# and was fully superseded by `set_live_watcher` (which drives the real
# WatcherAgent = WatcherWorker + Search + Recall). Keeping both only confused the
# main agent and wasted tool-schema tokens.

# --------------------------------------------------------------------------- #
# set_live_watcher — main agent's entry into the multimodal WatcherAgent.
#
# The MAIN agent uses this tool ONLY for COMPLEX multimodal questions it can't
# answer from its own injected frames — cross-source retrieval / long-term
# memory recall / crop-and-compare. One-shot visual Q&A uses QueryWorker; every
# set_live_watcher create goes straight
# to a background Router-ReAct orchestration: the tool returns immediately with
# ``mode=background`` + a request_id so the main agent gives a one-line
# placeholder, and the final answer arrives later as a proactive assistant
# bubble (``message.start/.delta/.complete`` with ``source=watcher``).
#
# This is purely a routing wrapper — Router/Search/Recall logic itself is the
# legacy worker code reused unchanged inside WatcherAgent.
# --------------------------------------------------------------------------- #
SET_LIVE_WATCHER_SCHEMA = {
    "name": "set_live_watcher",
        "description": (
        "Hand a screen/camera-stream task to a background agent that continuously "
        "re-watches the stream to produce analysis / explanation / summary / "
        "comparison / judgment / report / investigation. Always continuous: it keeps "
        "analysing round after round while the stream runs, and returns one final "
        "consolidated summary when the worker observes the task's explicit completion "
        "condition or the stream source ends. Per-round output shows "
        "in the watcher panel; to feed a final deliverable back to you on completion, "
        "use the hook (see hook_main_agent / hook_instruction).\n"
        "Use this for anything that digests the stream into a summary/report (even "
        "when the user says 'watch/keep an eye on'); to just alert on a discrete "
        "event, use set_monitor instead.\n"
        "Operations:\n"
        "  - op='create' (default): start a watcher task. Pass task_instruction (+ label).\n"
        "  - op='update': change task_instruction/label/hook of a running watcher task. "
        "Pass watcher_id. Takes effect on the next round.\n"
        "  - op='enable' / 'disable': (re)start or pause an existing watcher task. Pass watcher_id.\n"
        "  - op='delete': stop and remove a watcher task. Pass watcher_id.\n"
        "To fix or change an existing watcher task, use op='update' with its watcher_id — "
        "do not create a new one. List with list_live_watcher; read progress with get_live_watcher."
    ),

    "parameters": {
        "type": "object",
        "properties": {
            "op": {
                "type": "string",
                "enum": ["create", "update", "delete", "enable", "disable"],
                "description": (
                    "Which operation to perform. Default 'create'. "
                    "enable/disable pause or (re)start an existing watcher task — "
                    "enable requires a live video stream (fails otherwise)."
                ),
            },
            "watcher_id": {
                "type": "string",
                "description": (
                    "Required for op=update/delete/enable/disable. Use the id from a "
                    "prior create's tool result — never invent one."
                ),
            },
            "task_instruction": {
                "type": "string",
                "description": (
                    "create/update. The watcher task to keep analysing on the live "
                    "stream, handed to the deep-analysis worker."
                    "Rewrite the user's request into a self-contained instruction "
                    "addressed to the worker as the executor (imperative, e.g. "
                    "'Keep watching the stream and …'), rather than quoting the user "
                    "verbatim — the worker never sees the original conversation."
                    "If the user names a finite end condition (for example, when an "
                    "embedded video visibly finishes), preserve that condition "
                    "explicitly so the worker can stop and finalize at the right "
                    "moment. Video frames come from the session FrameBuffer; do not "
                    "require or pass image_url."
                ),
            },
            "label": {
                "type": "string",
                "description": (
                    "A short summary of task_instruction for the UI (a few words, "
                    "e.g. 'Argus Iteration AI Minutes' / 'Screen Rendering Code Review'). Shown on the "
                    "deep-analysis card so many tasks don't get confused. "
                    "Set on create/update; auto-derived from task_instruction if omitted."
                ),
            },
            "ttl": {
                "type": "string",
                "enum": ["200s", "1min", "30s", "10s"],
                "description": (
                    "create/update: how long the watcher accumulates frames before "
                    "running one analysis round — pick by the CURRENT SCENE's pace "
                    "(a fast-changing scene needs shorter rounds):\n"
                    "  • 200s — slow scenes: video meeting, desktop office, "
                    "document/code reading, monitoring dashboard.\n"
                    "  • 1min — medium scenes: movies/TV, people talking, teaching demos.\n"
                    "  • 30s  — fast scenes: sports, game streams, outdoor/travel.\n"
                    "  • 10s  — real-time interaction: video calls, live co-streaming, "
                    "real-time operation demos, anything needing immediate response.\n"
                    "Each ttl maps to a target frame count (200s=100, 1min=60, "
                    "30s=40, 10s=15): a round fires when EITHER the frame count is "
                    "reached OR the ttl elapses. If omitted, the tool picks it from "
                    "the auto-detected current scene."
                ),
            },
            "hook_main_agent": {
                "type": "boolean",
                "description": (
                    "Whether the main agent should perform one action after the watcher task successfully completes. Default false.\n"
                    "false: no follow-up action (the report is pushed to the panel, that's enough).\n"
                    "true: after completion the main agent also does something once (present the final summary / send a message / record, etc.); put only that action in hook_instruction. Per-round reports never invoke the main agent.\n"
                    "Pass false on op=update to cancel an existing hook."
                ),
            },
            "hook_instruction": {
                "type": "string",
                "description": (
                    "The one-time action handed to the main agent after the watcher task successfully completes, when hook_main_agent=true.\n"
                    "Write it as a self-contained imperative — only the action to do; minimize anaphoric references；do NOT restate the watcher task, do NOT paraphrase the user.\n"
                    "The full watcher report is appended automatically after your instruction, so just describe the action (e.g. 'write up the conclusions into a memo') — no need to reference the result yourself."
                ),
            },
        },
        "required": [],
    },
}


def _resolve_mm_engine(session_id=None):
    """Resolve the per-session multimodal WatcherAgent + its agent.

    Returns ``(engine, agent)``; either may be ``None`` when no multimodal
    session is running. Shared by ``set_live_watcher`` (deep background
    analysis) and ``query_multimodal`` (one-shot visual QueryWorker handoff) so
    the session-registry walk lives in exactly one place.

    The ``session_id`` we receive is the HERMES session_id (the agent's internal
    id, e.g. "20260630_174817_6e5ea3"), NOT the gateway sid that keys
    ``_sessions`` (the short 8-char hex sid). Try the fast path (direct sid key),
    then fall back to matching by hermes session_key. There's only ever one
    active multimodal session per gateway, so the walk is cheap.
    """
    sid = (session_id or "").strip()
    if not sid:
        return None, None
    try:
        from tui_gateway.server import _sessions
    except Exception:
        return None, None
    try:
        entry = _sessions.get(sid)
        if entry is not None:
            eng = entry.get("_mm_live_watcher_agent")
            if eng is not None:
                return eng, entry.get("agent")
        # Snapshot to a list: _sessions can be mutated by concurrent session
        # create/finalize → "dict changed size during iteration".
        for entry in list(_sessions.values()):
            if entry.get("session_key") == sid:
                eng = entry.get("_mm_live_watcher_agent")
                if eng is not None:
                    return eng, entry.get("agent")
    except Exception:
        return None, None
    return None, None


def _resolve_session_owner(session_id=None):
    """Return ``(agent, durable_session_id)`` for the caller's session.

    Unlike ``_resolve_mm_engine``, reads do not require a live Watcher engine;
    reopened sessions must still be able to inspect their own persisted tasks.
    """
    sid = str(session_id or "").strip()
    if not sid:
        return None, ""
    try:
        from tui_gateway.server import _sessions
    except Exception:
        return None, ""
    try:
        direct = _sessions.get(sid)
        if isinstance(direct, dict):
            return direct.get("agent"), str(
                direct.get("session_key") or sid).strip()
        for gateway_sid, entry in list(_sessions.items()):
            if str(entry.get("session_key") or "") == sid:
                return entry.get("agent"), str(
                    entry.get("session_key") or gateway_sid).strip()
    except Exception:
        return None, ""
    return None, ""


def _validated_watcher_id(value) -> str:
    rid = str(value or "").strip()
    return rid if _WATCHER_ID_RE.fullmatch(rid) else ""


def mm_stream_status(source):
    """Is a live video stream currently on, for BACKGROUND agents (set_monitor /
    set_live_watcher)? Returns ``(live, reason)``.

    ``source`` may be a WatcherAgent (preferred - has the authoritative
    per-session ``is_source_live()``) or a raw frame_buffer (fallback).

    Verdict = UI switch is ON (frontend never sent source_stopped) AND frames
    were ever captured. We deliberately do NOT use the main agent's 10s freshness
    gate (multimodal.vision_stream_max_age_sec): a background monitor / research
    run must not be declared dead just because the scene was static or the tab
    was backgrounded for a few seconds - the authoritative signal is the explicit
    share start/stop switch, not frame recency. Never raises.
    """
    _NEVER = (
        "No live video stream is active; the camera/screen share has never been "
        "started. Ask the user to start camera or screen sharing and retry."
    )
    _STOPPED = (
        "The video stream has stopped; the camera/screen share is closed. Ask "
        "the user to restart sharing and retry."
    )
    # Preferred: the WatcherAgent's authoritative UI-switch verdict.
    try:
        if source is not None and hasattr(source, "is_source_live"):
            if source.is_source_live():
                return True, ""
            buf = getattr(source, "frame_buffer", None)
            never = (buf is None or (getattr(buf, "size", 0) == 0
                     and getattr(buf, "_last_push_wall", None) is None))
            return False, (_NEVER if never else _STOPPED)
    except Exception:
        pass

    # Fallback: raw frame_buffer - no UI-switch flag, judge by capture only.
    buf = source
    if buf is None:
        return False, "Multimodal video buffer is not ready."
    try:
        size = getattr(buf, "size", 0)
        last_push = getattr(buf, "_last_push_wall", None)
    except Exception:
        return False, "Could not read video stream status."
    if size == 0 and last_push is None:
        return False, _NEVER
    return True, ""


def _watcher_label(entry: dict) -> str:
    """Display label for a research task: explicit label else a truncated text."""
    lbl = (entry.get("label") or "").strip()
    if lbl:
        return lbl
    txt = (entry.get("task_instruction") or "").strip()
    return (txt[:17] + "...") if len(txt) > 20 else (txt or "Deep research")


def _push_watchers_event(session_id, agent) -> None:
    """Push the current research registry to the frontend (best-effort).

    Mirrors monitor_tool._push_monitors_event so the desktop deep-analysis panel
    can render each running watcher task's label (avoids confusion once many tasks
    exist). Resolves the runtime sid via the session registry."""
    if agent is None:
        return
    try:
        from tui_gateway.server import _emit, _sessions
        # Resolve the gateway sid (short hex key) for this agent. session_id may
        # be the hermes session_key, so try the direct key then match by agent.
        sid = ""
        sk = (session_id or "").strip()
        if sk and _sessions.get(sk) is not None:
            sid = sk
        else:
            for _k, _e in list(_sessions.items()):
                if _e.get("agent") is agent or _e.get("session_key") == sk:
                    sid = _k
                    break
        if not sid:
            return
        rs = list((getattr(agent, "mm_watchers", {}) or {}).values())
        # ★ 五态统一: 不再派生 "deleting"; status 直接用 registry 值(running/stopping/
        #   interrupted/done)。已 deleted 的一律不推(前端不展示)。
        payload = [{
            "watcher_id": r.get("id"),
            "label": _watcher_label(r),
            "task_instruction": r.get("task_instruction", ""),
            "status": r.get("status", "running"),
            "hook_main_agent": bool(r.get("hook_main_agent", False)),
            "created_at": r.get("created_at", 0.0),
        } for r in rs if r.get("status") != "deleted"]
        _emit("multimodal.watchers", sid, {"watchers": payload})
    except Exception:
        pass


# ★ pacing 单一数据源: 直接复用 scene_dhash._PACE_TIERS, 不再自己维护一份 (历史上
#   这里写死一份旧值 2min=64帧, 与 scene_dhash 漂移 → watcher 用错帧数)。ttl 标签
#   从各档真实 ttl_sec 动态生成 (如 30 → "30s", 120 → "2min"), 保证与四档永远一致。
def _fmt_ttl_label(sec: float) -> str:
    s = int(round(sec))
    return f"{s // 60}min" if s % 60 == 0 and s >= 60 else f"{s}s"


def _build_ttl_pacing():
    """Build {ttl_label: {ttl_sec, target_frames}} plus a pace→label map from
    scene_dhash._PACE_TIERS (falls back to a hardcoded copy on import failure)."""
    try:
        from agent.multimodal.scene_dhash import _PACE_TIERS
    except Exception:
        _PACE_TIERS = {
            "slow":   {"ttl_sec": 200, "target_frames": 100},
            "medium": {"ttl_sec": 60,  "target_frames": 60},
            "fast":   {"ttl_sec": 30,  "target_frames": 40},
            "live":   {"ttl_sec": 10,  "target_frames": 15},
        }
    ttl_pacing, pace_to_ttl = {}, {}
    for pace, p in _PACE_TIERS.items():
        lbl = _fmt_ttl_label(p["ttl_sec"])
        ttl_pacing[lbl] = {"ttl_sec": p["ttl_sec"], "target_frames": p["target_frames"]}
        pace_to_ttl[pace] = lbl
    return ttl_pacing, pace_to_ttl


_TTL_PACING, _PACE_TO_TTL = _build_ttl_pacing()
_DEFAULT_TTL_LABEL = _PACE_TO_TTL.get("medium", "1min")   # 兜底档 = medium


def _pacing_mode_for_ttl(ttl) -> str:
    """Return ``explicit`` only for a concrete supported user TTL."""
    return (
        "explicit"
        if str(ttl or "").strip().lower() in _TTL_PACING
        else "auto"
    )


def _resolve_ttl_pacing(ttl, engine, agent):
    """Resolve (ttl_label, ttl_sec, target_frames).

    Priority:
      1. Explicit ttl label from the LLM.
      2. ttl_sec + target_frames computed by SceneDhashController.
      3. scene pace fallback through _PACE_TIERS.
      4. medium default.
    """
    label = (str(ttl or "").strip().lower())
    if label in _TTL_PACING:
        p = _TTL_PACING[label]
        return label, p["ttl_sec"], p["target_frames"]
    # Fall back to the current scene (FrameBuffer.current_scene).
    try:
        buf = getattr(agent, "frame_buffer", None)
        scene = buf.current_scene if buf is not None else None
        if isinstance(scene, dict):
            # ★ 优先用 scene 自带的 ttl_sec+target_frames (SceneDhashController 可能已
            #   算出比 pace 档位更精准的值); 二者都在时以它们为准, 不再查 pace 表。
            if scene.get("ttl_sec") and scene.get("target_frames"):
                return "", int(scene["ttl_sec"]), int(scene["target_frames"])
            # scene 只给了 pace (没自带具体数值) → 查 _PACE_TIERS 档位表兜底。
            pace = str(scene.get("pace") or "").strip().lower()
            lbl = _PACE_TO_TTL.get(pace)
            if lbl:
                p = _TTL_PACING[lbl]
                return lbl, p["ttl_sec"], p["target_frames"]
    except Exception:
        pass
    p = _TTL_PACING[_DEFAULT_TTL_LABEL]   # medium default
    return _DEFAULT_TTL_LABEL, p["ttl_sec"], p["target_frames"]


def set_live_watcher(task_instruction=None, session_id=None, op=None,
                      watcher_id=None, label=None,
                      hook_main_agent=None, hook_instruction=None,
                      ttl=None, **_kw) -> str:
    """CRUD over the per-session WatcherAgent's live-watcher tasks.

    The watcher has ONE mode (standard multimodal ReAct, always continuous) —
    there is no qa/analysis/research classification. The task text comes in as
    ``task_instruction``. ``ttl`` (200s/1min/30s/10s) sets the per-round
    accumulation window by scene pace; omitted → derived from the auto-detected
    current scene. Extra legacy kwargs are ignored (**_kw).

    Returns a JSON result the main agent can act on (see schema description).
    """
    op = (op or "create").strip().lower()
    text = (task_instruction or "").strip()

    # Reach the per-session engine via the gateway session registry (shared
    # helper). ``agent`` is captured alongside so we can report the branch back.
    engine, agent = _resolve_mm_engine(session_id)
    if engine is None:
        return tool_error(
            "multimodal WatcherAgent is not running for this session "
            "(multimodal disabled or session not multimodal-capable).",
            success=False)
    # 持久 session id (跨 resume 稳定; 与前端 ?mm=/localStorage/history 同源)。写入
    # watch 文件头, 供 scan_all 按 session 过滤 (堵跨 session 泄漏)。
    _owner, _stored_sid = _resolve_session_owner(session_id)
    if _owner is not agent:
        _stored_sid = ""

    # Per-session research registry (mirrors agent.mm_monitors). The engine's
    # _run_delegation reads this each round for batch-boundary update/delete, and
    # the completion hook reads hook_main_agent/hook_instruction from it.
    if not hasattr(agent, "mm_watchers") or agent.mm_watchers is None:
        agent.mm_watchers = {}
    watchers = agent.mm_watchers

    # Record the dispatch branch on the agent (mode + background rid). This is
    # bookkeeping only — the model-facing wrapper turns the tool result into a
    # terminal handoff receipt, so there is no second main-agent model phase.
    # Kept for tests / possible future use. Used by op=create AND op=enable.
    def _report_mode(mode: str, rid: str = "") -> None:
        if agent is not None:
            try:
                agent._mm_route_last_mode = mode
                if mode == "background":
                    agent._mm_route_background_stop = True
                    agent._mm_route_background_rid = rid or None
                else:
                    agent._mm_route_background_stop = False
                    agent._mm_route_background_rid = None
            except Exception:
                pass

    # ── op=update / op=delete: mutate the registry; take effect at batch edge ──
    if op == "update":
        raw_rid = str(watcher_id or "").strip()
        rid = _validated_watcher_id(raw_rid)
        if not raw_rid:
            return tool_error("watcher_id is required for op=update", success=False)
        if not rid:
            return tool_error("invalid watcher_id format", success=False)
        if rid not in watchers:
            return tool_error(f"watcher_id {rid!r} not found", success=False)
        ent = watchers[rid]
        if text:
            ent["task_instruction"] = text
            ent["_pending_update"] = True   # engine picks up new text next round
        if ttl is not None:
            _lbl, _sec, _tf = _resolve_ttl_pacing(ttl, engine, agent)
            ent["pacing_mode"] = _pacing_mode_for_ttl(ttl)
            ent["ttl"] = _lbl
            ent["ttl_sec"] = _sec
            ent["target_frames"] = _tf
            ent["_pending_update"] = True
        if label is not None and label.strip():
            ent["label"] = label.strip()
        _hook_note = ""
        if hook_main_agent is not None:
            ent["hook_main_agent"] = bool(hook_main_agent)
        if hook_instruction is not None:
            ent["hook_instruction"] = str(hook_instruction or "").strip()
        if ent.get("hook_main_agent") and ent.get("hook_instruction"):
            _hook_note = (
                "Completion hook attached: once the watcher successfully finishes, "
                "when the main agent is idle, it will execute exactly once: "
                f"{ent['hook_instruction']!r}."
            )
        elif hook_main_agent is False:
            _hook_note = "The main-agent hook for this research task was cancelled."
        try:
            from agent.multimodal import watch_file as _wf_state
            _wf_state.update_state(
                rid,
                task_instruction=str(ent.get("task_instruction") or ""),
                label=str(ent.get("label") or ""),
                pacing_mode=str(ent.get("pacing_mode") or ""),
                ttl=str(ent.get("ttl") or ""),
                ttl_sec=ent.get("ttl_sec"),
                target_frames=ent.get("target_frames"),
                hook_main_agent=bool(ent.get("hook_main_agent", False)),
                hook_instruction=str(ent.get("hook_instruction") or ""),
            )
        except Exception:
            pass
        _push_watchers_event(session_id, agent)
        return tool_result({
            "op": "update", "watcher_id": rid, "request_id": rid,
            "task_instruction": ent.get("task_instruction", ""),
            "label": _watcher_label(ent),
            "pacing_mode": ent.get("pacing_mode", ""),
            "ttl": ent.get("ttl", ""),
            "ttl_sec": ent.get("ttl_sec"),
            "target_frames": ent.get("target_frames"),
            "hook_main_agent": bool(ent.get("hook_main_agent", False)),
            "hook_instruction": ent.get("hook_instruction", ""),
            "note": (" ".join(x for x in (
                "Updated; changes take effect on the next research round "
                "(the current round will finish first).", _hook_note) if x) or None),
        })

    if op == "delete":
        raw_rid = str(watcher_id or "").strip()
        rid = _validated_watcher_id(raw_rid)
        if not raw_rid:
            return tool_error("watcher_id is required for op=delete", success=False)
        if not rid:
            return tool_error("invalid watcher_id format", success=False)
        if rid not in watchers:
            return tool_error(f"watcher_id {rid!r} not found", success=False)
        # ★ 五态统一: 删除 = 先 stopping(当前轮收尾, UI 显"正在停止") → 收尾后引擎落
        #   status=deleted (watcher_engine 收尾分支), scan_all/list 一律跳过 → UI 消失。
        #   stop_delegation 已把文件落 stopping; registry 也标 stopping + _deleted, 供
        #   list 期间显示"正在停止", 收尾后本条从注册表移除。
        watchers[rid]["_deleted"] = True
        watchers[rid]["status"] = "stopping"
        try:
            from agent.multimodal import watch_file as _wf_state
            _wf_state.update_state(
                rid, status="stopping", stop_reason="deleted")
        except Exception:
            pass
        active_stop = False
        try:
            active_stop = bool(
                engine.stop_delegation(rid, reason="deleted"))
        except Exception:
            active_stop = False
        if not active_stop:
            # The engine may already have finished between the registry check
            # and this delete. There is no loop left to perform cleanup, so make
            # deletion terminal synchronously instead of leaving a permanent
            # stopping/interrupted ghost in the panel.
            try:
                from agent.multimodal import watch_file as _wf_state
                status_info = _wf_state.read_status(rid) or {}
                _wf_state.set_status(
                    rid, "deleted", round_idx=status_info.get("round_idx"),
                    stop_reason="deleted")
            except Exception:
                pass
            watchers.pop(rid, None)
        _push_watchers_event(session_id, agent)
        return tool_result({
            "op": "delete", "watcher_id": rid, "request_id": rid,
            "note": (
                "Delete accepted: the current round will finish, then the task "
                "will end (status stopping -> deleted) and will no longer trigger "
                "the main agent."
                if active_stop else
                "The inactive deep-research task was deleted immediately."
            ),
        })

    # ── op=disable: pause a running watcher task (keep the registry entry) ──────────
    if op == "disable":
        raw_rid = str(watcher_id or "").strip()
        rid = _validated_watcher_id(raw_rid)
        if not raw_rid:
            return tool_error("watcher_id is required for op=disable", success=False)
        if not rid:
            return tool_error("invalid watcher_id format", success=False)
        if rid not in watchers:
            return tool_error(f"watcher_id {rid!r} not found", success=False)
        # ★ 五态统一: 暂停(用户 off) = 先 stopping(当前轮收尾) → 收尾后引擎落
        #   status=interrupted (watcher_engine: disabled→interrupted)。registry 标
        #   stopping, 收尾后随 push 更新为 interrupted。
        watchers[rid]["status"] = "stopping"
        try:
            from agent.multimodal import watch_file as _wf_state
            _wf_state.update_state(
                rid, status="stopping", stop_reason="disabled")
        except Exception:
            pass
        active_stop = False
        try:
            active_stop = bool(
                engine.stop_delegation(rid, reason="disabled"))
        except Exception:
            active_stop = False
        if not active_stop:
            watchers[rid]["status"] = "interrupted"
            watchers[rid]["_interrupted"] = True
            try:
                from agent.multimodal import watch_file as _wf_state
                status_info = _wf_state.read_status(rid) or {}
                _wf_state.set_status(
                    rid, "interrupted",
                    round_idx=status_info.get("round_idx"),
                    stop_reason="disabled")
            except Exception:
                pass
        _push_watchers_event(session_id, agent)
        return tool_result({
            "op": "disable", "watcher_id": rid, "request_id": rid,
            "label": _watcher_label(watchers[rid]),
            "status": "stopping" if active_stop else "interrupted",
            "note": (
                "Research task paused: the current round will finish, then status "
                "will become interrupted. Re-enabling requires a live video stream."
            ),
        })

    # ── op=enable: (re)start a paused/interrupted research ─────────────────────
    if op == "enable":
        raw_rid = str(watcher_id or "").strip()
        rid = _validated_watcher_id(raw_rid)
        if not raw_rid:
            return tool_error("watcher_id is required for op=enable", success=False)
        if not rid:
            return tool_error("invalid watcher_id format", success=False)
        if rid not in watchers:
            return tool_error(f"watcher_id {rid!r} not found", success=False)
        # Same live-stream guard as create: a research re-watches the stream, so
        # with no active stream there is nothing to analyse. Failing here makes
        # the UI toggle roll back to off (点 on 没流就自动弹回).
        _live, _why = mm_stream_status(engine)
        if not _live:
            return tool_error(
                f"Could not enable live deep research: {_why}", success=False)
        ent = watchers[rid]
        _text = str(ent.get("task_instruction") or "").strip()
        if not _text:
            # ★ 兜底: 旧任务的历史 tool 结果里没存 task_instruction (重建后为空), 但 analyse
            #   文件头部记录了完整 query — 从文件读回, 救活旧任务 (否则永远 toggle 不开)。
            try:
                from agent.multimodal import watch_file as _wf_read
                _st = _wf_read.read_structured(rid) or {}
                _text = str(_st.get("query") or "").strip()
                if _text:
                    ent["task_instruction"] = _text   # 补回 registry, 后续无需再读文件
            except Exception:
                pass
        if not _text:
            return tool_error("watcher has no task_instruction to resume", success=False)
        # Clear the stopped flags and (re)submit the delegation for this SAME rid.
        # After a session reopen the engine was rebuilt with no jobs, so enabling
        # must respawn the delegation (idempotent-ish: a still-live run would just
        # get a second job — but enable is only offered for stopped/interrupted
        # entries in the UI, and submit generates work off the shared registry).
        ent.pop("_deleted", None)
        ent.pop("_interrupted", None)
        ent["status"] = "running"
        from agent.multimodal import watch_file as _df2
        try:
            # 文件已存在则 init_file 直接返回 (幂等); 仅新建时 hook 头部才写入。
            ent["watch_file"] = _df2.init_file(
                rid, query=_text, session_id=_stored_sid,
                hook_main_agent=bool(ent.get("hook_main_agent")),
                hook_instruction=str(ent.get("hook_instruction") or "").strip())
            _df2.update_state(
                rid,
                task_instruction=_text,
                label=str(ent.get("label") or ""),
                status="running",
                pacing_mode=str(ent.get("pacing_mode") or ""),
                ttl=str(ent.get("ttl") or ""),
                ttl_sec=ent.get("ttl_sec"),
                target_frames=ent.get("target_frames"),
                hook_main_agent=bool(ent.get("hook_main_agent", False)),
                hook_instruction=str(ent.get("hook_instruction") or ""),
                stop_reason="",
            )
        except Exception:
            pass
        ret = engine.submit_complex_async(_text, request_id=rid)
        if not ret:
            ent["status"] = "interrupted"
            return tool_error("failed to (re)start background watcher", success=False)
        _report_mode("background", rid)
        _push_watchers_event(session_id, agent)
        return tool_result({
            "op": "enable", "watcher_id": rid, "request_id": rid,
            "label": _watcher_label(ent), "status": "running",
            "note": f"Deep research task {_watcher_label(ent)!r} (#{rid}) was re-enabled.",
        })

    # ── op=create (default) ───────────────────────────────────────────────────
    if not text:
        return tool_error("task_instruction is required for op=create", success=False)

    # ── Guard: is a live video stream actually running? ───────────────────────
    # Deep research re-watches the live stream; with no active stream (never
    # opened, or the user stopped sharing / no fresh frames) there is nothing to
    # analyse. Return a TOOL FAILURE with the reason so the main agent tells the
    # user to (re)start the camera/screen share instead of promising an analysis
    # that never produces anything.
    _live, _why = mm_stream_status(engine)  # engine → authoritative is_source_live()
    if not _live:
        _report_mode("simple")
        return tool_error(
            f"Could not start live deep research: {_why}", success=False)

    # Kick off the background live-watcher orchestration. The final summary
    # arrives later in the watcher panel, tagged with the request_id. The watcher
    # has ONE mode (standard multimodal ReAct, always continuous): it runs
    # round-after-round over the video, writes analyse/watch_<rid>.md, and ends
    # with one LLM summary. There is no query-type classification.
    import secrets as _secrets
    from agent.multimodal import watch_file as _df

    bg_rid = "req_" + _secrets.token_hex(4)
    lbl = (label or "").strip()
    # Resolve the per-round accumulation window (ttl / target frames) from the
    # LLM-supplied ttl, else from the auto-detected current scene.
    _pacing_mode = _pacing_mode_for_ttl(ttl)
    _ttl_lbl, _ttl_sec, _target_frames = _resolve_ttl_pacing(ttl, engine, agent)
    watch_path = ""
    try:
        watch_path = _df.init_file(
            bg_rid,
            query=text,
            session_id=_stored_sid,
            hook_main_agent=bool(hook_main_agent),
            hook_instruction=str(hook_instruction or "").strip(),
            state={
                "task_instruction": text,
                "label": lbl or _watcher_label({"task_instruction": text}),
                "pacing_mode": _pacing_mode,
                "ttl": _ttl_lbl,
                "ttl_sec": _ttl_sec,
                "target_frames": _target_frames,
            },
        )
    except Exception:
        watch_path = ""

    # Register BEFORE submit so the engine's first-round registry read (and any
    # completion hook) can see this task's label / hook config / pacing.
    watchers[bg_rid] = {
        "id": bg_rid, "watcher_id": bg_rid,
        "task_instruction": text,
        "label": lbl or _watcher_label({"task_instruction": text}),
        "hook_main_agent": bool(hook_main_agent),
        "hook_instruction": str(hook_instruction or "").strip(),
        "pacing_mode": _pacing_mode,
        "ttl": _ttl_lbl, "ttl_sec": _ttl_sec, "target_frames": _target_frames,
        "status": "running", "watch_file": watch_path,
        "created_at": time.time(),
    }

    ret_rid = engine.submit_complex_async(
        text, request_id=bg_rid)
    if not ret_rid:
        watchers.pop(bg_rid, None)
        return tool_error("failed to submit background analysis", success=False)
    _report_mode("background", bg_rid)
    _push_watchers_event(session_id, agent)

    disp_label = _watcher_label(watchers[bg_rid])

    _hook_note = ""
    if watchers[bg_rid].get("hook_main_agent") and watchers[bg_rid].get("hook_instruction"):
        _hook_note = (
            "Completion hook attached: once the watcher successfully finishes, "
            "the main agent will execute exactly once when idle: "
            f"{watchers[bg_rid]['hook_instruction']!r}."
        )
    # Dispatch receipt — PURE FACTS ONLY (user-facing; rendered in the dispatch
    # card via server.dispatch_note). Behavioral instructions for the model
    # (reply short / ignore injected frames / don't write the report) live in the
    # system prompt (MM_LIVE_GUIDANCE), NOT here — a tool result must carry data,
    # not instructions to the model.
    _started_note = (
        f"Deep research task {disp_label!r} (id #{bg_rid}) was created and is "
        "running on the live video stream. The background worker will continue "
        "multi-round analysis and append each round's report below."
    )
    result = {
        "op": "create",
        "watcher_id": bg_rid,
        "request_id": bg_rid,   # kept so the later report can backfill this turn
        # ★ 必须把 task_instruction 写进持久化的 tool 结果: 重开时
        #   _reconcile_stale_mm_jobs 从 history 的这条结果重建 watcher, 靠它恢复
        #   task_instruction; 缺了它 → 重建出 has_task=False → enable 报
        #   "no task_instruction to resume" → 深度研究 toggle 打不开。
        "task_instruction": text,
        "label": disp_label,
        "status": "running",
        "watch_file": watch_path,
        "pacing_mode": _pacing_mode,
        "ttl": _ttl_lbl,
        "ttl_sec": _ttl_sec,
        "target_frames": _target_frames,
        "hook_main_agent": bool(watchers[bg_rid].get("hook_main_agent", False)),
        "hook_instruction": watchers[bg_rid].get("hook_instruction", ""),
        "note": " ".join(x for x in (_started_note, _hook_note) if x),
    }
    return tool_result(result)


def _set_live_watcher_handoff(*, session_id=None, **kwargs) -> str:
    """Model-facing watcher CRUD path with terminal reply ownership."""
    raw = set_live_watcher(session_id=session_id, **kwargs)
    try:
        data = json.loads(raw)
    except Exception:
        data = {}
    _engine, agent = _resolve_mm_engine(session_id)
    parent_id = str(
        getattr(agent, "_active_parent_user_message_id", "") or "")
    import secrets as _secrets
    task_id = str(
        data.get("request_id")
        or data.get("watcher_id")
        or f"watcher_ctl_{_secrets.token_hex(3)}"
    )
    op = str(data.get("op") or kwargs.get("op") or "create")
    label = str(data.get("label") or "").strip()
    label_suffix = f' "{label}"' if label else ""
    fallback_ack = {
        "create": f"Started background deep research{label_suffix}.",
        "update": f"Updated background deep research{label_suffix}.",
        "enable": f"Resumed background deep research{label_suffix}.",
        "disable": f"Paused background deep research{label_suffix}.",
        "delete": "Accepted deletion for this background deep research task.",
    }.get(op, "Background deep research operation handled.")
    return tool_handoff(
        raw,
        reply_owner="watcher",
        handoff_mode="receipt",
        task_id=task_id,
        parent_user_message_id=parent_id,
        ack=str(data.get("note") or data.get("error") or fallback_ack).strip(),
    )


registry.register(
    name="set_live_watcher",
    # WatcherAgent deep-VQA/search/recall entry. (set_monitor was split out to
    # its own `monitor` toolset in tools/monitor_tool.py — see MonitorAgent.)
    toolset="live_watcher",
    schema=SET_LIVE_WATCHER_SCHEMA,
    handler=lambda args, **kw: _set_live_watcher_handoff(
        op=args.get("op"),
        watcher_id=args.get("watcher_id"),
        task_instruction=args.get("task_instruction"),
        label=args.get("label"),
        ttl=args.get("ttl"),
        hook_main_agent=args.get("hook_main_agent"),
        hook_instruction=args.get("hook_instruction"),
        session_id=kw.get("session_id"),
    ),
    emoji="🎥",
)


# --------------------------------------------------------------------------- #
# get_live_watcher — read a background deep-research delegation's live
# progress from its on-disk analyse file (HERMES_HOME/analyse/watch_*.md).
# --------------------------------------------------------------------------- #
GET_LIVE_WATCHER_SCHEMA = {
    "name": "get_live_watcher",
    "description": (
        "Read the latest progress for a background multimodal deep-research "
        "task created by set_live_watcher. This reads local progress files and "
        "does not call a model.\n\n"
        "Use when the user asks how the analysis is going, whether there is "
        "progress, or whether it is done; also use before replying if you need "
        "the current background findings.\n"
        "  - Pass watcher_id, such as 'req_ab12cd34', to read one task's latest "
        "progress and recent findings.\n"
        "  - Omit watcher_id to list all background deep-research tasks for "
        "this session, equivalent to list_live_watcher.\n"
        "Note: when deep research completes, its final answer normally replaces "
        "the in-progress bubble automatically. This tool is mainly for explicit "
        "progress checks."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "watcher_id": {
                "type": "string",
                "description": "Deep-research task id, such as req_xxxx. Leave empty to list all tasks. Alias: request_id.",
            },
        },
        "required": [],
    },
}


def get_live_watcher(request_id=None, watcher_id=None, **_kw) -> str:
    """Read background deep-research progress from the on-disk analyse file(s).

    ``watcher_id`` is the preferred param; ``request_id`` kept as a back-compat
    alias (both name the same req_xxxx id)."""
    from agent.multimodal import watch_file as _df

    session_id = _kw.get("session_id")
    agent, stored_sid = _resolve_session_owner(session_id)
    if agent is None or not stored_sid:
        return tool_error(
            "get_live_watcher cannot resolve the current session owner.",
            success=False,
        )

    raw_rid = str(watcher_id or request_id or "").strip()
    rid = _validated_watcher_id(raw_rid) if raw_rid else ""
    if raw_rid and not rid:
        return tool_error("invalid watcher_id format", success=False)

    # Omitted id is exactly the session-scoped list operation; never enumerate
    # the process-wide analyse directory here.
    if not rid:
        return list_live_watcher(session_id=session_id)

    # ── Specific task ─────────────────────────────────────────────────────────
    # Ownership is checked before reading the file. A foreign id and a missing
    # id intentionally have the same result to avoid a cross-session oracle.
    try:
        owned = _df.scan_all(session_id=stored_sid) or {}
    except Exception:
        owned = {}
    if rid not in owned:
        return tool_result({
            "request_id": rid,
            "found": False,
            "note": (
                f"No progress record found for #{rid} in the current session. "
                "The id may be wrong, deleted, or owned by another session."
            ),
        })
    text = _df.read_all(rid)
    if not text:
        return tool_result({"request_id": rid, "found": False})
    max_chars = 6000
    body = text if len(text) <= max_chars else (
        text[:800] + "\n...[middle omitted]...\n" + text[-(max_chars - 800):])
    live = _df.read_status(rid) or {}
    return tool_result({
        "request_id": rid,
        "watcher_id": rid,
        "found": True,
        "task_instruction": str(
            (owned.get(rid, {}).get("state") or {}).get("task_instruction")
            or owned.get(rid, {}).get("query") or ""),
        "status": live.get("status", "unknown"),
        "round_idx": live.get("round_idx"),
        "progress": body,
        "note": "This is the progress and findings written so far for this deep-analysis task. Use it to answer the user.",
    })


registry.register(
    name="get_live_watcher",
    toolset="live_watcher",
    schema=GET_LIVE_WATCHER_SCHEMA,
    handler=lambda args, **kw: get_live_watcher(
        watcher_id=args.get("watcher_id"),
        request_id=args.get("request_id"),   # legacy alias
        session_id=kw.get("session_id"),
    ),
    emoji="📊",
)


# --------------------------------------------------------------------------- #
# list_live_watcher — read-only listing of the session's continuous research
# tasks. Reads the live registry (agent.mm_watchers).
# --------------------------------------------------------------------------- #
LIST_LIVE_WATCHER_SCHEMA = {
    "name": "list_live_watcher",
    "description": (
        "List all multimodal background deep-research tasks for the current "
        "session. Read-only; does not change task state.\n"
        "Use when the user asks what research tasks are active, what is being "
        "analyzed, or asks for the research list; also use before create when "
        "you need to avoid duplicates. Use set_live_watcher to start, update, "
        "or delete a task. Use get_live_watcher to read one task's detailed "
        "progress.\n"
        "Returns watcher_id, label, task_instruction, status, and whether the "
        "main-agent hook is enabled. The UI displays only the label."
    ),
    "parameters": {"type": "object", "properties": {}},
}


def list_live_watcher(session_id=None, **_kw) -> str:
    """Read-only listing of the session's research tasks (from the registry)."""
    agent, _stored_sid = _resolve_session_owner(session_id)
    if agent is None:
        return tool_error(
            "list_live_watcher cannot reach the session's agent. Multimodal "
            "session may not be active.", success=False)
    if not _stored_sid:
        return tool_error(
            "list_live_watcher cannot resolve the current session owner.",
            success=False,
        )
    rs = list((getattr(agent, "mm_watchers", {}) or {}).values())
    # 执行态从各任务【自己的日志文件】头部读 (与引擎写入端同源, 单一数据源):
    # 给每条内存 watcher 补真实 status(running/stopping/stopped/done/interrupted)+轮数,
    # 并补全"文件里有、内存没有"的条目。★ 配置态(deleted)以内存为准, 执行态以文件为准。
    # ★ scan_all 按【本 session】过滤 —— 不再捞全局 (堵跨 session 泄漏)。
    from agent.multimodal import watch_file as _df
    _file_ws = {}
    try:
        _file_ws = _df.scan_all(session_id=_stored_sid) or {}
    except Exception:
        _file_ws = {}
    if not rs and not _file_ws:
        return tool_result({
            "found": False,
            "note": "There are no running multimodal background deep-research tasks right now.",
        })
    _mem_ids = set()
    tasks = []
    for r in rs:
        _mem_ids.add(r.get("id"))
        _fs = _file_ws.get(r.get("id")) or {}
        # ★ 五态统一: 不再派生 "deleting"。执行态以文件为准(running/stopping/done/
        #   interrupted), 退回 registry status。deleted 的文件 scan_all 已过滤掉;
        #   若 registry 里已是 deleted(收尾完), 也跳过不列。
        _st = _fs.get("status") or r.get("status", "running")
        if _st == "deleted":
            continue
        tasks.append({
            "watcher_id": r.get("id"),
            "label": _watcher_label(r),
            "task_instruction": r.get("task_instruction", ""),
            "status": _st,
            "round_idx": _fs.get("round_idx"),
            "hook_main_agent": bool(r.get("hook_main_agent", False)),
            "hook_instruction": r.get("hook_instruction", ""),
            "watch_file": r.get("watch_file", ""),
        })
    for fid, fw in _file_ws.items():
        if fid in _mem_ids:
            continue
        # scan_all 已跳过 deleted, 这里 status 直接用文件值(interrupted/done/...)。
        tasks.append({
            "watcher_id": fid,
            "label": fw.get("query", fid),
            "task_instruction": fw.get("query", ""),
            "status": fw.get("status", "interrupted"),
            "round_idx": fw.get("round_idx"),
            "from_file": True,
        })
    return tool_result({
        "found": True,
        "watchers": tasks,
        "note": "These are this session's deep-research tasks, including running, stopping, and finished tasks. To inspect one task in detail, call get_live_watcher with its watcher_id.",
    })


registry.register(
    name="list_live_watcher",
    toolset="live_watcher",
    schema=LIST_LIVE_WATCHER_SCHEMA,
    handler=lambda args, **kw: list_live_watcher(
        session_id=kw.get("session_id"),
    ),
    emoji="📋",
)
