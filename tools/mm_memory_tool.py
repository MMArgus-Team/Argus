# -*- coding: utf-8 -*-
"""query_multimodal — one-shot multimodal answer or visual grounding.

The main agent is not passively fed live frames. In gateway turns this tool
hands the complete one-shot user question to the multimodal QueryWorker.
QueryWorker is VLM-first: it receives the ask-time recent frames, answers
directly when the current picture is enough, and calls RecallWorker / Search
only when the question needs earlier stream memory or outside facts. Historical
stream content lives in the multimodal memory graph (micro/macro events,
entities + their evolution, ASR subtitles), written continuously by the
MemoryWriter/Reviewer.
Synchronous non-gateway callers of the legacy Python
``recall_multimodal_memory`` entry retain the historical RecallWorker
compatibility path; the model-visible tool never takes that narrower route.

Division of labour (also in the schema description):

  * query_multimodal(response_mode="direct") — dispatch a one-shot VLM
    QueryWorker. It may answer from ask-time frames or combine Recall/Search,
    then replies directly to the original question.
  * query_multimodal(response_mode="evidence") — inspect ask-time frames only
    and return bounded visual evidence to the Main Agent. The Main Agent keeps
    reply ownership and can continue with PDF/file/terminal/browser/skills.
  * set_live_watcher — heavy background job that RE-WATCHES the raw video
    stream frame-by-frame (crop-and-compare, counting, whole-stream scans,
    continuous research/report). Use it when memory is insufficient or the answer
    needs the original pixels.

Rule of thumb: visual user questions should enter QueryWorker; QueryWorker then
decides current-frame VQA vs memory recall vs deep route.
"""

from __future__ import annotations

import logging
import secrets

from tools.registry import registry, tool_error, tool_handoff, tool_result
from tools.live_watcher_tool import _resolve_mm_engine

log = logging.getLogger("hermes.multimodal.query")

# Upper bound on how long the tool blocks the main agent while the RecallWorker
# runs its ReAct loop (recall_max_rounds LLM rounds + optional frame verify).
# ★ 25s → 45s: 实测 decide/distill 每次 LLM 调用 6-9s (qwen3.7-plus 代理端点),
#   2-3 轮就撞 25s 上限。backend 会保留 partial findings; 这里仍给
#   2-3 轮完整 ReAct 留出余量。
_RECALL_TIMEOUT_SEC = 45.0
_EVIDENCE_TIMEOUT_SEC = 45.0
_RESPONSE_MODES = {"direct", "evidence"}


def _resolve_send_anchor_ts(session_id=None, engine=None):
    """Best-effort gateway send-time frame anchor for the current user turn."""
    sid = (session_id or "").strip()
    if not sid:
        return None
    try:
        from tui_gateway.server import _sessions
    except Exception:
        return None
    try:
        entry = _sessions.get(sid)
        if entry is not None:
            return entry.get("_mm_send_anchor_ts")
        for entry in list(_sessions.values()):
            if entry.get("session_key") == sid:
                return entry.get("_mm_send_anchor_ts")
            if engine is not None and entry.get("_mm_live_watcher_agent") is engine:
                return entry.get("_mm_send_anchor_ts")
    except Exception:
        return None
    return None


def _query_multimodal_impl(
    query=None,
    session_id=None,
    *,
    response_mode="direct",
    allow_sync_recall=False,
    **_kw,
) -> str:
    """Implement direct-answer handoff, visual evidence, and legacy recall.

    See the schema description for when to use this vs set_live_watcher.
    Interactive calls are non-blocking QueryWorker handoffs. Only the legacy
    Python wrapper opts into synchronous RecallWorker compatibility. Never
    raises — any error becomes a tool_error.
    """
    q = (query or "").strip()
    if not q:
        return tool_error(
            "query is required — what do you want to ask about the current or "
            "historical camera/screen content?",
            success=False)

    mode = str(response_mode or "direct").strip().lower()
    if mode not in _RESPONSE_MODES:
        return tool_error(
            "response_mode must be one of: direct, evidence.",
            success=False,
            code="invalid_query_multimodal_response_mode",
            response_mode=mode,
        )

    engine, _agent = _resolve_mm_engine(session_id)
    if engine is None:
        return tool_error(
            "multimodal memory is not available for this session "
            "(multimodal disabled or session not multimodal-capable).",
            success=False)

    # Evidence mode is an ordinary, synchronous tool observation. It never
    # transfers reply ownership, so it is intentionally allowed after normal
    # tool work and does not inherit the direct-handoff solo-turn restriction.
    # QueryWorker is restricted to visual grounding in this mode; the Main
    # Agent consumes the returned evidence and continues orchestration.
    if mode == "evidence":
        inspect = getattr(engine, "query_visual_evidence", None)
        if not callable(inspect):
            return tool_error(
                "visual evidence mode is unavailable because this multimodal "
                "runtime does not provide a visual evidence collector.",
                success=False,
                code="query_worker_evidence_unavailable",
            )
        original_user_text = str(
            getattr(_agent, "_active_user_message_text", "")
            or _kw.get("user_text")
            or q
        ).strip()
        ask_ts = _resolve_send_anchor_ts(session_id, engine)
        try:
            res = inspect(
                q,
                original_user_query=original_user_text,
                ask_ts=ask_ts,
                timeout=_EVIDENCE_TIMEOUT_SEC,
            )
        except Exception as exc:
            log.warning(
                "[query-evidence] engine.query_visual_evidence raised: %s",
                exc,
                exc_info=True,
            )
            return tool_error(
                f"visual evidence collection failed: {exc}",
                success=False,
                code="query_worker_evidence_failed",
            )
        if not isinstance(res, dict) or not res.get("ok", False):
            res = res if isinstance(res, dict) else {}
            return tool_error(
                str(res.get("error") or "visual evidence collection failed"),
                success=False,
                code="query_worker_evidence_failed",
                timed_out=bool(res.get("timed_out")),
            )
        evidence = str(res.get("evidence") or "").strip()
        return tool_result({
            "mode": "evidence",
            "query": q,
            "original_user_query": original_user_text,
            "visual_evidence": evidence,
            "evidence_scope": {
                "source": "ask_time_frames",
                "ask_ts": res.get("ask_ts"),
                "n_frames": int(res.get("n_frames") or 0),
                "t_start": res.get("t_start"),
                "t_end": res.get("t_end"),
            },
            "limitations": list(res.get("limitations") or []),
            "elapsed_sec": res.get("elapsed_sec", 0.0),
            "next_action": (
                "Main Agent retains reply ownership. Use this visual grounding "
                "as evidence, invoke the required PDF/file/terminal/browser/"
                "skill tools, then produce the final user-facing answer."
            ),
        })

    # Gateway turns preallocate a stable answer slot and stamp its id on the
    # live agent.  In that context QueryWorker is a delegation boundary, not a
    # synchronous tool observation: schedule the existing Router/Recall/Search
    # worker stack and transfer reply ownership immediately.  This releases the
    # main session after its first model call, so later user questions can enter
    # while this query is still researching.
    parent_id = str(
        getattr(_agent, "_active_parent_user_message_id", "") or "")
    original_user_text = str(
        getattr(_agent, "_active_user_message_text", "")
        or _kw.get("user_text")
        or q
    ).strip()
    handoff_block_reason = str(
        getattr(_agent, "_query_multimodal_handoff_block_reason", "") or ""
    ).strip()
    if handoff_block_reason:
        reason_text = (
            "the same model response also requested other tools"
            if handoff_block_reason == "mixed_tool_batch"
            else "ordinary tool work already ran earlier in this turn"
        )
        return tool_error(
            "QueryWorker was not started because " + reason_text + ". "
            "query_multimodal transfers ownership of the final user reply and "
            "must be the first and only tool call in its turn. Preserve the "
            "ordinary tool results, then retry the user's visual question in a "
            "fresh turn with query_multimodal as the sole tool call.",
            success=False,
            code="query_multimodal_requires_solo_turn",
            reason=handoff_block_reason,
        )
    if parent_id and callable(getattr(engine, "submit_query_async", None)):
        task_id = "qry_" + secrets.token_hex(4)
        # Preserve the user's exact question as the answer contract while also
        # giving the worker the main agent's context-resolved delegation brief.
        # This is what lets “第二个物品价格” become
        # recall(identity) -> search(price), or “离我家最近的店” retain an
        # approximate home-area anchor established by prior QA, inside one
        # worker-owned ReAct job.
        instruction = original_user_text
        if q and q != original_user_text:
            instruction = (
                "### AUTHORITATIVE ORIGINAL USER QUESTION\n"
                f"{original_user_text}\n\n"
                "Answer every requested part of the original question. It is the "
                "sole authoritative answer contract.\n\n"
                "### CONTEXT-RESOLVED MAIN AGENT DELEGATION BRIEF\n"
                f"{q}\n\n"
                "Use this brief for directly useful prior-QA context, referent "
                "binding, and task planning, while preserving uncertainty. The "
                "brief must not narrow, replace, contradict, or reinterpret the "
                "original question. Ignore any part of the brief that does.\n\n"
                "Inspect the ask-time frames first. Answer directly only if the "
                "available evidence answers every requested part. For earlier "
                "content or outside facts, bind the target from the frames or "
                "memory, then use Recall/Search as needed."
            )
        ask_ts = _resolve_send_anchor_ts(session_id, engine)
        submitted = engine.submit_query_async(
            instruction,
            task_id=task_id,
            parent_user_message_id=parent_id,
            original_user_query=original_user_text,
            ask_ts=ask_ts,
        )
        if submitted:
            return tool_handoff(
                tool_result({
                    "query": q,
                    "original_user_text": original_user_text,
                    "status": "running",
                    "note": (
                        f"QueryWorker accepted this query (#{submitted}) and "
                        "will answer the original user message directly."
                    ),
                }),
                reply_owner="query_worker",
                handoff_mode="deferred_reply",
                task_id=submitted,
                parent_user_message_id=parent_id,
            )
        # Engine existed but rejected the schedule: still terminate in one stage
        # with a deterministic failure receipt instead of paying for a second
        # main-model call merely to paraphrase this infrastructure error.
        return tool_handoff(
            tool_error(
                "QueryWorker is not ready; could not dispatch the async "
                "multimodal query.", success=False),
            reply_owner="query_worker",
            handoff_mode="receipt",
            task_id=task_id,
            parent_user_message_id=parent_id,
        )

    # ``query_multimodal`` promises the unified QueryWorker route (ask-time
    # frames + optional Recall/Search) and therefore must not silently degrade
    # to Recall-only when there is no interactive answer slot. The separately
    # named legacy Python wrapper is the sole opt-in to synchronous recall.
    if not allow_sync_recall:
        missing = []
        if not parent_id:
            missing.append("parent_user_message_id")
        if not callable(getattr(engine, "submit_query_async", None)):
            missing.append("query_worker_dispatcher")
        return tool_error(
            "query_multimodal requires an interactive QueryWorker answer slot; "
            "this call cannot deliver the unified current-frame/Recall/Search "
            "answer and will not fall back to Recall-only.",
            success=False,
            code="query_worker_handoff_unavailable",
            missing=missing,
        )

    # Only the legacy synchronous compatibility path intrinsically requires a
    # RecallWorker.  Interactive QueryWorker can answer from ask-time frames or
    # Search without a memory backend, so this check must remain *after* the
    # async handoff above.
    if getattr(engine, "recall_worker", None) is None:
        return tool_error(
            "multimodal recall is not available for this non-interactive call "
            "(memory backend disabled or engine not fully initialized).",
            success=False)

    try:
        res = engine.recall_memory(
            brief=q, user_text=str(_kw.get("user_text") or q),
            timeout=_RECALL_TIMEOUT_SEC)
    except Exception as exc:
        log.warning("[recall] engine.recall_memory raised: %s", exc, exc_info=True)
        return tool_error(f"recall failed: {exc}", success=False)

    if not res.get("ok", False):
        err = res.get("error", "recall failed")
        # Timeout is a soft failure — tell the agent it can retry a narrower
        # query or escalate to set_live_watcher.
        if res.get("timed_out"):
            partial_findings = (
                (res.get("partial_findings") or res.get("findings") or "").strip()
                or "\n".join(str(c).strip() for c in (res.get("clues") or [])
                             if str(c).strip()).strip()
            )
            if partial_findings:
                return tool_result({
                    "found": True,
                    "partial": True,
                    "timed_out": True,
                    "query": q,
                    "findings": partial_findings,
                    "clues": list(res.get("clues") or [])[:8],
                    "frame_ids": list(res.get("frame_ids") or [])[:20],
                    "rounds": res.get("rounds", 0),
                    "elapsed_sec": res.get("elapsed_sec", 0.0),
                    "recall_trace": list(res.get("recall_trace") or [])[:40],
                    "note": (
                        f"Recall hit the outer {_RECALL_TIMEOUT_SEC:.0f}s timeout, "
                        "but partial findings are available. Prefer these findings "
                        "when answering. Do not substitute current frames for a "
                        "historical answer. If the partial findings are insufficient, "
                        "state the gap or retry with a narrower query."
                    ),
                })
            return tool_result({
                "found": False,
                "query": q,
                "timed_out": True,
                "note": (
                    f"Recall timed out after {_RECALL_TIMEOUT_SEC:.0f}s without "
                    "usable partial findings. Retry with a narrower query or use "
                    "set_live_watcher for frame-by-frame investigation. Do not "
                    "substitute current frames for a historical answer."
                ),
            })
        return tool_error(
            err,
            success=False,
            recall_trace=list(res.get("recall_trace") or [])[:40],
            rounds=res.get("rounds", 0),
            elapsed_sec=res.get("elapsed_sec", 0.0),
        )

    findings = (res.get("findings") or "").strip()
    frame_ids = list(res.get("frame_ids") or [])
    clues = list(res.get("clues") or [])
    found = bool(res.get("found"))

    result = {
        "found": found,
        "query": q,
        "findings": findings,
        # frame_ids the recall grounded on — handy if a follow-up needs the real
        # pixels via set_live_watcher.
        "frame_ids": frame_ids[:20],
        "rounds": res.get("rounds", 0),
        "elapsed_sec": res.get("elapsed_sec", 0.0),
        "recall_trace": list(res.get("recall_trace") or [])[:40],
    }
    if res.get("timed_out"):
        result["timed_out"] = True
        result["partial"] = bool(res.get("partial"))
        result["partial_findings"] = res.get("partial_findings") or findings
    if found:
        _fid_hint = ""
        if frame_ids:
            _fid_hint = (
                f" Recall returned {len(frame_ids[:20])} grounding frame_id values "
                f"(see the frame_ids field). If the user wants to see the real "
                f"historical image, call show_memory_frame immediately. Prefer "
                f"show_memory_frame(entity_name='<item name>'), or pass the first "
                f"frame_id: show_memory_frame(frame_id='{frame_ids[0]}'). "
                f"Do not claim that multimodal memory does not store images; "
                f"frames are persisted and there is a dedicated retrieval tool."
            )
        result["note"] = (
            "These are RecallWorker's distilled findings from multimodal memory; "
            "answer the user from this evidence. "
            + ("This recall hit a timeout guard, but findings/partial_findings "
               "are usable evidence and should be preferred. "
               if res.get("timed_out") else "")
            + _fid_hint +
            " If this is still insufficient, or the task requires rewatching raw "
            "frames for counting/crop comparison/full-segment scanning/continuous "
            "research, use set_live_watcher."
        )
    else:
        # Surface clues even when there's no confident finding — better than a
        # bare "not found" for the agent to reason from.
        if clues:
            result["clues"] = clues[:8]
        result["note"] = (
            "Multimodal memory did not return content clearly related to this "
            "query. It may not have been observed yet or may not have settled "
            "into memory. Do not invent an answer; if necessary use "
            "set_live_watcher for frame-by-frame investigation, otherwise tell "
            "the user that there is no reliable memory for it."
        )
    return tool_result(result)


def query_multimodal(
    query=None, session_id=None, response_mode="direct", **kwargs,
) -> str:
    """Run one-shot multimodal QA in direct-answer or evidence mode.

    This model-visible contract never falls back to the narrower synchronous
    Recall-only behavior when no QueryWorker answer slot exists.
    """
    return _query_multimodal_impl(
        query=query,
        session_id=session_id,
        response_mode=response_mode,
        allow_sync_recall=False,
        **kwargs,
    )


def recall_multimodal_memory(query=None, session_id=None, **kwargs) -> str:
    """Backward-compatible Python entry with synchronous Recall support.

    The legacy name is intentionally not registered as a model-visible tool.
    It still uses QueryWorker when an interactive answer slot exists; otherwise
    direct Python callers retain the former synchronous RecallWorker behavior.
    """
    # The legacy alias has no evidence-mode contract; ignore an accidental
    # forwarded mode instead of passing the keyword twice.
    kwargs.pop("response_mode", None)
    return _query_multimodal_impl(
        query=query,
        session_id=session_id,
        response_mode="direct",
        allow_sync_recall=True,
        **kwargs,
    )


QUERY_MULTIMODAL_SCHEMA = {
    "name": "query_multimodal",
    "description": (
        "Handle a one-shot question grounded in the user's current or historical "
        "camera/screen context. "
        "Choose response_mode='direct' for pure visual questions and simple visual "
        "+ Recall/Search questions: QueryWorker receives ask-time frames, may use "
        "multimodal Recall and external Search, and replies directly. Choose "
        "response_mode='evidence' when "
        "the task also needs PDF, local files, terminal, browser, or another complex "
        "Main-Agent skill: QueryWorker inspects ask-time frames only and returns "
        "structured visual evidence; the Main Agent must then continue tool "
        "orchestration and produce the final answer. Build a complete, "
        "self-contained delegation from the current question and any directly useful "
        "prior QA; QueryWorker does not receive the main chat history. Preserve the "
        "uncertainty of reused context, leave current visual referents for QueryWorker "
        "to identify from its ask-time frames, and include every requested fact and "
        "output requirement. Never invent missing details. Never replace an outside-fact request "
        "such as retail price, market value, or listing status with a request to read "
        "only text visible in the image. "
        "In direct mode, after this tool returns a handoff, do not synthesize a "
        "second answer; it must be the first and only tool in that turn. In evidence "
        "mode, do not answer yet: consume the evidence and continue with the needed "
        "tools in subsequent rounds. Use "
        "get_current_frame only when raw current frames must be retrieved or shown; "
        "use set_monitor for a future trigger and set_live_watcher for continuous "
        "or deep stream analysis."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "A complete, self-contained visual/multimodal delegation built from "
                    "the current question and directly useful prior QA. Preserve the "
                    "uncertainty of reused context, leave current visual referents for "
                    "QueryWorker to bind from ask-time frames, and include every requested "
                    "part. Preserve current/past time scope, target ids, dates, "
                    "units, table/figure numbers, OCR text, visual referents, and the "
                    "requested output format; do not invent missing specifics."
                ),
            },
            "response_mode": {
                "type": "string",
                "enum": ["direct", "evidence"],
                "default": "direct",
                "description": (
                    "direct: QueryWorker may use visual/Recall/Search evidence and "
                    "answers the user directly. evidence: inspect ask-time frames "
                    "only, return visual grounding to Main Agent, and let Main Agent "
                    "continue with PDF/file/terminal/browser/complex skills."
                ),
            },
        },
        "required": ["query"],
    },
}


registry.register(
    name="query_multimodal",
    # One-shot QueryWorker entry. The worker owns current-frame VQA, memory
    # Recall, and external Search routing, then replies to the original turn.
    toolset="live_watcher",
    schema=QUERY_MULTIMODAL_SCHEMA,
    handler=lambda args, **kw: query_multimodal(
        query=args.get("query"),
        response_mode=args.get("response_mode", "direct"),
        session_id=kw.get("session_id"),
    ),
    emoji="🧠",
)
