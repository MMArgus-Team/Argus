"""frame_buffer toolset — main-agent access to the live video FrameBuffer.

Peer to the `monitor` toolset (tools/monitor_tool.py). Exposes:
  * ``get_current_frame``  — grab the ~3 most recent frames (send-time anchor)
  * ``check_video_stream`` — is the camera/screen-share on?
  * ``show_memory_frame``  — retrieve historical keyframes from multimodal memory

Visibility: registered on the global tool registry, so only the main agent (which
loads the registry) sees it. MonitorAgent / WatcherAgent run their own engines
and never load this registry, so they can't call it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from tools.registry import registry, tool_error, tool_result

# Reuse the gateway session lookup from the monitor toolset (both need the live
# per-session agent, which owns .frame_buffer).
from tools.monitor_tool import _find_agent_by_session

# ── check_video_stream: 视频流是否开启 → True/False ────────────────────────────
# 视频流新鲜度判据: 最近一次推帧在 vision_stream_max_age_sec
# (默认 10s) 内则视为开启。这里只判断 FrameBuffer 是否仍在接收画面，
# 与主 agent 是否看图无关。该窗口内无新帧即判 False。
_STREAM_ON_MAX_AGE_DEFAULT = 10.0


def check_video_stream(session_id: Optional[str] = None, **_kw):
    """Return whether the video stream (camera / screen share) is currently on:
    {"video_stream_on": bool}.

    Verdict: the session has a FrameBuffer AND the last frame was pushed within
    vision_stream_max_age_sec (default 10s). No multimodal session / no buffer /
    no frame ever pushed / stale frame → False. Never raises."""
    entry, _sid = _find_agent_by_session(session_id or "")
    if entry is None:
        return tool_result({"video_stream_on": False,
                            "reason": "no active multimodal session"})
    agent = entry.get("agent")
    buf = getattr(agent, "frame_buffer", None)
    if buf is None:
        return tool_result({"video_stream_on": False,
                            "reason": "no FrameBuffer (video never started)"})

    # 新鲜度阈值从 config 读取，仅表示当前视频流是否活跃。
    max_age = _STREAM_ON_MAX_AGE_DEFAULT
    try:
        from hermes_cli.config import load_config as _lc
        from agent.multimodal.hermes_glue import flatten_mm_config
        mm = flatten_mm_config(_lc() or {})
        max_age = float(mm.get("vision_stream_max_age_sec", _STREAM_ON_MAX_AGE_DEFAULT)
                        or _STREAM_ON_MAX_AGE_DEFAULT)
    except Exception:
        pass

    last_push = getattr(buf, "_last_push_wall", None)
    if last_push is None:
        return tool_result({"video_stream_on": False,
                            "reason": "no frame pushed yet"})
    import time as _t
    age = _t.time() - float(last_push)
    on = (max_age <= 0) or (age <= max_age)
    return tool_result({
        "video_stream_on": bool(on),
        "reason": ("stream live" if on
                   else f"last frame {age:.1f}s ago (> {max_age:.0f}s) — stream not live"),
    })


CHECK_VIDEO_STREAM_SCHEMA = {
    "name": "check_video_stream",
    "description": (
        "Check whether the user's current video stream (camera or screen share) "
        "is live. Takes no parameters and returns {\"video_stream_on\": true|false}.\n"
        "Use only when the user directly asks whether the stream/camera/share is "
        "on, or when a referring expression is genuinely ambiguous and you must "
        "know whether the user is talking about the current live view. For a pure "
        "status query, answer from this result and stop. Do not call "
        "list_live_watcher or set_* tools afterward.\n"
        "Do not use this as a preflight for set_monitor, set_live_watcher, or "
        "get_current_frame; those tools validate the video source themselves. A "
        "clear new monitor request should call set_monitor(op='create') directly. "
        "Main agent only."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}


# ── get_current_frame: no-arg "看当前画面" (发送时刻帧 + 向前近邻共 3 帧) ──────────
_CURRENT_FRAME_N = 3   # 发送时刻帧 + 前 2 帧近邻


def _main_supports_vision() -> bool:
    """Whether the resolved MAIN agent model is vision-capable, per config.

    Reads model.{provider,default} + the supports_vision override / models.dev
    caps via image_routing._lookup_supports_vision. Conservative: unknown → False
    (so we route through auxiliary.vision VQA rather than blindly sending images
    to a model that can't see)."""
    try:
        from hermes_cli.config import load_config
        from agent.image_routing import _lookup_supports_vision
        cfg = load_config() or {}
        model_cfg = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
        provider = str(model_cfg.get("provider") or "").strip()
        model = str(model_cfg.get("default") or "").strip()
        got = _lookup_supports_vision(provider, model, cfg)
        return bool(got)
    except Exception:
        return False


async def _vqa_over_frames(query: str, frames: List[Any]) -> Optional[str]:
    """Run a single VQA turn over the given frames via auxiliary.vision and return
    the answer text. Used when the MAIN model can't see images itself — the vision
    model answers ``query`` about the frames and we hand the main agent the TEXT.
    Returns None on any failure (caller falls back / reports error)."""
    imgs = [f for f in frames if getattr(f, "jpeg_b64", "")]
    if not imgs:
        return None
    try:
        from agent.auxiliary_client import get_async_text_auxiliary_client
        client, model = get_async_text_auxiliary_client(task="vision")
        if client is None or not model:
            return None
        content: List[Dict[str, Any]] = [{
            "type": "text",
            "text": (
                query.strip()
                or "Describe the content of these frames and preserve any "
                   "readable text or numbers verbatim."
            ),
        }]
        for f in imgs:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{f.jpeg_b64}"},
            })
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=0.2,
            max_tokens=1024,
        )
        return (resp.choices[0].message.content or "").strip() or None
    except Exception:
        return None


async def get_current_frame(session_id: Optional[str] = None,
                            query: Optional[str] = None, **_kw):
    """Look at the user's CURRENT screen/camera picture, anchored on the moment
    THIS user message was sent (anchor frame + its immediate predecessors, 3
    frames total).

    Branches on the MAIN agent's vision capability (config supports_vision):
      • vision-capable → returns the frames as real images (_multimodal result)
        for the main agent to read directly.
      • NOT vision-capable → runs ``query`` (or the user's last message as a
        fallback) + the frames through auxiliary.vision (a VQA turn) and returns
        the ANSWER TEXT as the tool result — the main agent never sees pixels.

    ``query`` is what to ask about the picture; the main agent should pass what it
    actually wants to know, e.g. "what error is shown on screen". Returns tool_error when
    there is no stream / no frames."""
    entry, _sid = _find_agent_by_session(session_id or "")
    if entry is None:
        return tool_error(
            "get_current_frame cannot reach the session's agent "
            "(multimodal session may not be active).", success=False)
    agent = entry.get("agent")
    buf = getattr(agent, "frame_buffer", None)
    if buf is None:
        return tool_error(
            "no live FrameBuffer on this session (no video stream).", success=False)

    # 锚点 = UserMessage 发送时刻的 buffer 最新帧 ts (在 _run_prompt_submit 里 stamp)。
    #   据此取"发送时的帧 + 向前近邻", 而不是工具执行时的最新帧 (那时 buffer 又前进了)。
    anchor_ts = entry.get("_mm_send_anchor_ts")
    frames: List[Any] = []
    if anchor_ts is not None:
        try:
            frames = buf.all_le(float(anchor_ts))[-_CURRENT_FRAME_N:]
        except Exception:
            frames = []
    # 兜底: 没锚点 (或锚点后 buffer 被清过) → 用最新 3 帧。
    if not frames:
        frames = buf.latest(_CURRENT_FRAME_N)
    if not frames:
        return tool_error(
            "no frames in the live buffer yet (stream just started?).", success=False)

    _last_ts = getattr(frames[-1], "ts", None)

    # ── Branch: main model can't see images → auxiliary.vision VQA, return TEXT ──
    if not _main_supports_vision():
        # Prefer the agent-supplied query; fall back to the user's last message
        # text stamped on the session at send time.
        eff_query = (query or "").strip() or str(
            entry.get("_mm_last_user_text") or "").strip()
        answer = await _vqa_over_frames(eff_query, frames)
        if not answer:
            return tool_error(
                "The main model is not vision-capable, and auxiliary.vision "
                "failed or is not configured, so the current frame content could "
                "not be obtained.", success=False)
        text = (
            "[Current-frame VQA via auxiliary.vision]\n"
            f"Question: {eff_query or '(describe the frames)'}\n"
            f"Answer: {answer}"
        )
        return {
            "success": True,
            "result": text,
            "text_summary": text,
            "meta": {"n": len(frames), "anchor_ts": anchor_ts,
                     "via": "auxiliary_vision_vqa",
                     "frame_ts": [getattr(f, "ts", None) for f in frames]},
        }

    # ── Branch: main model IS vision-capable → return real images ──
    summary = (
        f"Current view: {len(frames)} frames anchored to the user-message send time"
        + (f", latest frame ts={_last_ts:.1f}s"
           if isinstance(_last_ts, (int, float)) else "")
    )
    content: List[Dict[str, Any]] = [{"type": "text", "text": summary}]
    for f in frames:
        b64 = getattr(f, "jpeg_b64", "") or ""
        if not b64:
            continue
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    return {
        "_multimodal": True,
        "content": content,
        "text_summary": summary,
        "meta": {"n": len(frames),
                 "anchor_ts": anchor_ts,
                 "frame_ts": [getattr(f, "ts", None) for f in frames]},
    }


GET_CURRENT_FRAME_SCHEMA = {
    "name": "get_current_frame",
    "description": (
        "Inspect the user's current screen/camera view. Returns the frames from "
        "the moment this user message was sent plus nearby predecessor frames "
        "(3 frames total). The anchor frame may be a few frames behind the latest "
        "buffer frame by design, so it matches the moment the user asked.\n"
        "Pass query with the specific visual question you want answered. If the "
        "main model is not vision-capable, the system uses query + frames with "
        "auxiliary.vision and returns a text answer; if the main model is "
        "vision-capable, it returns images directly.\n"
        "Use this when the user explicitly asks to retrieve/show/inspect the raw "
        "current frames. For a one-shot question grounded in current or past "
        "camera/screen context, call query_multimodal instead. If desktop control is needed "
        "(clicking, typing, scrolling, dragging, launching apps), use "
        "computer_use instead."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Natural-language question about the current frames. If the "
                    "main model is text-only, this question is sent with the "
                    "frames to auxiliary.vision and the tool returns a text "
                    "answer. If omitted, the user's most recent message is used."
                ),
            },
        },
        "required": [],
    },
}


registry.register(
    name="get_current_frame",
    toolset="live_watcher",
    schema=GET_CURRENT_FRAME_SCHEMA,
    handler=lambda args, **kw: get_current_frame(
        session_id=kw.get("session_id"),
        query=(args or {}).get("query")),
    is_async=True,
    emoji="📸",
)

registry.register(
    name="check_video_stream",
    toolset="live_watcher",
    schema=CHECK_VIDEO_STREAM_SCHEMA,
    handler=lambda args, **kw: check_video_stream(session_id=kw.get("session_id")),
    emoji="🎥",
)


# ── show_memory_frame: 从记忆里取【历史关键帧真图】给用户看 ────────────────────
# 用途: 用户问"给我看看当时那瓶乌龙茶的样子" / "上次的那本书张什么样" —— 需要把
# 已经存在 entity_rep_frames / FrameStore(内存+磁盘) 的关键帧图片回投给用户。
# 与 get_current_frame 的区别: 那个只看当前 buffer 的 ~3s 内新鲜帧; 本工具查
# 【记忆里已沉淀的】关键帧, 支持按 entity 名字或 frame_id 检索, 跨 session 可用。
_SHOW_FRAME_TOP_K = 3   # 每个 entity 最多返回 3 张代表帧
_SHOW_FRAME_THUMB_SIDE = 0    # show_memory_frame 是按需取图, 保留原图读标签小字


def _resolve_memory_store(entry):
    """Best-effort: extract the MemoryStore instance from a session entry."""
    if not entry:
        return None
    mb = entry.get("_mm_memory_backend")
    if mb is None:
        return None
    return getattr(mb, "mem", None)


def _find_entities_by_name(mem, name: str, limit: int = 5):
    """Fuzzy-match entities by name, then aliases; return up to limit rows."""
    import sqlite3 as _sq
    name = (name or "").strip()
    if not name:
        return []
    try:
        with mem._lock, mem._connect() as c:
            # 1) 精确 name (case-sensitive; entity name 一般是原样保留的中英文)
            rows = c.execute(
                "SELECT id, name, type FROM entities "
                "WHERE name=? AND (merged_into IS NULL OR merged_into='') "
                "LIMIT ?", (name, limit)).fetchall()
            if rows:
                return [(r["id"], r["name"], r["type"]) for r in rows]
            # 2) name LIKE 双向包含 (处理 "乌龙茶" 匹配 "栀栀乌龙茶饮料")
            like = f"%{name}%"
            rows = c.execute(
                "SELECT id, name, type FROM entities "
                "WHERE name LIKE ? AND (merged_into IS NULL OR merged_into='') "
                "ORDER BY LENGTH(name) LIMIT ?", (like, limit)).fetchall()
            if rows:
                return [(r["id"], r["name"], r["type"]) for r in rows]
            # 3) aliases LIKE (aliases 是 JSON 数组的字符串, LIKE '%name%' 够用)
            rows = c.execute(
                "SELECT id, name, type FROM entities "
                "WHERE aliases LIKE ? AND (merged_into IS NULL OR merged_into='') "
                "ORDER BY LENGTH(name) LIMIT ?", (like, limit)).fetchall()
            return [(r["id"], r["name"], r["type"]) for r in rows]
    except _sq.Error as e:
        import logging as _lg
        _lg.getLogger(__name__).warning(
            "[show_memory_frame] entity lookup failed name=%r: %s", name, e)
        return []


def _entity_details(mem, entity_ids: List[str]) -> List[Dict[str, Any]]:
    if not entity_ids:
        return []
    qmarks = ",".join("?" for _ in entity_ids)
    try:
        import json as _json
        with mem._lock, mem._connect() as c:
            rows = c.execute(
                f"""SELECT id, name, type, attributes, aliases
                    FROM entities WHERE id IN ({qmarks})""",
                entity_ids,
            ).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            try:
                attrs = _json.loads(r["attributes"] or "{}")
            except Exception:
                attrs = {}
            try:
                aliases = _json.loads(r["aliases"] or "[]")
            except Exception:
                aliases = []
            out.append({
                "id": r["id"], "name": r["name"], "type": r["type"],
                "attributes": attrs, "aliases": aliases,
            })
        return out
    except Exception:
        return []


def _screen_text_for_frames(mem, frame_ids: List[str]) -> List[Dict[str, Any]]:
    ids = [fid for fid in frame_ids if fid]
    if not ids:
        return []
    qmarks = ",".join("?" for _ in ids)
    try:
        with mem._lock, mem._connect() as c:
            rows = c.execute(
                f"""SELECT frame_id, t_observed, app, window_title, raw_text, source
                    FROM screen_texts WHERE frame_id IN ({qmarks})""",
                ids,
            ).fetchall()
        by_id = {r["frame_id"]: {
            "frame_id": r["frame_id"],
            "t_observed": r["t_observed"],
            "app": r["app"] or "",
            "window_title": r["window_title"] or "",
            "raw_text": r["raw_text"] or "",
            "source": r["source"] or "",
        } for r in rows}
        return [by_id[fid] for fid in ids if fid in by_id]
    except Exception:
        return []


def show_memory_frame(entity_name: Optional[str] = None,
                      frame_id: Optional[str] = None,
                      session_id: Optional[str] = None, **_kw):
    """Return persisted historical key-frame images from multimodal memory.

    Use entity_name for fuzzy entity lookup or frame_id for direct image
    retrieval. Returns a _multimodal result with real images.
    """
    entry, _sid = _find_agent_by_session(session_id or "")
    if entry is None:
        return tool_error("Could not find the current multimodal session.",
                          success=False)
    # 直接从 backend 拿 mem + frame_store, 它们是 backend 的兄弟属性 (memory_backend.py:75-78)。
    mb = entry.get("_mm_memory_backend")
    if mb is None:
        return tool_error(
            "The multimodal memory backend is not ready.", success=False)
    mem = getattr(mb, "mem", None)
    fs = getattr(mb, "frame_store", None)
    if mem is None or fs is None:
        return tool_error(
            f"Memory components are not ready "
            f"(mem={mem is not None}, frame_store={fs is not None}).",
            success=False)

    picked_fids: List[str] = []
    picked_meta: List[Dict[str, Any]] = []
    hit_entities: List[Dict[str, Any]] = []

    # 优先按 frame_id 精确取
    fid = (frame_id or "").strip()
    if fid:
        picked_fids.append(fid)
        picked_meta.append({"frame_id": fid, "source": "frame_id"})

    # 按 entity 名字取
    ent_name = (entity_name or "").strip()
    if ent_name and not picked_fids:
        cands = _find_entities_by_name(mem, ent_name, limit=3)
        if not cands:
            return tool_result({
                "shown": 0, "frames": [],
                "note": (
                    f"No entity whose name contains {ent_name!r} was found in "
                    "memory. Call query_multimodal first if you need to "
                    "identify the exact item name."
                )})
        for eid, name, etype in cands:
            hit_entities.append({"id": eid, "name": name, "type": etype})
            # ── 三源查帧, 直到取到 ─────────────────────────────────────────
            # 源 1: entity_rep_frames 表 (omni + reviewer 写, 多帧+quality 排序)
            reps = mem.get_rep_frames(eid, top_k=_SHOW_FRAME_TOP_K)
            for rf in reps:
                if rf.frame_id and rf.frame_id not in picked_fids:
                    picked_fids.append(rf.frame_id)
                    picked_meta.append({
                        "frame_id": rf.frame_id, "entity_id": eid,
                        "entity_name": name, "quality": rf.quality_score,
                        "source": "rep_frames_table"})
            # 源 2: entities.representative_frame_id (标准 writer 写的单帧字段)
            if not picked_fids:
                try:
                    with mem._lock, mem._connect() as _c:
                        _row = _c.execute(
                            "SELECT representative_frame_id FROM entities WHERE id=?",
                            (eid,)).fetchone()
                        _rid = (_row["representative_frame_id"] if _row else "") or ""
                        if _rid and _rid not in picked_fids:
                            picked_fids.append(_rid)
                            picked_meta.append({
                                "frame_id": _rid, "entity_id": eid,
                                "entity_name": name,
                                "source": "entities.representative_frame_id"})
                except Exception as e:
                    import logging as _lg
                    _lg.getLogger(__name__).warning(
                        "[show_memory_frame] rep_frame_id lookup failed: %s", e)
            # 源 3: entity_frame 表 (帧级绑定) — 拿最近 3 张
            if not picked_fids:
                try:
                    with mem._lock, mem._connect() as _c:
                        _rows = _c.execute(
                            """SELECT frame_id, t_observed FROM entity_frame
                               WHERE entity_id=? ORDER BY t_observed DESC LIMIT ?""",
                            (eid, _SHOW_FRAME_TOP_K)).fetchall()
                        for _r in _rows:
                            _fid = _r["frame_id"]
                            if _fid and _fid not in picked_fids:
                                picked_fids.append(_fid)
                                picked_meta.append({
                                    "frame_id": _fid, "entity_id": eid,
                                    "entity_name": name,
                                    "t_observed": _r["t_observed"],
                                    "source": "entity_frame_table"})
                except Exception as e:
                    import logging as _lg
                    _lg.getLogger(__name__).warning(
                        "[show_memory_frame] entity_frame lookup failed: %s", e)
            if picked_fids:
                # 找到第一个有帧的 entity 就停 (避免同名多 entity 一次拉一大堆)
                break

    if not picked_fids:
        return tool_result({
            "shown": 0, "frames": [], "hit_entities": hit_entities,
            "note": (
                "An entity matched, but it has no representative frame yet "
                "(possibly a pure TOPIC or no selected key frame)."
                if hit_entities else
                "No matching entity or frame_id was found."
            )})

    # 从 FrameStore 拉真图 (内存 miss 会走磁盘 fallback)
    stored = fs.get_many(picked_fids)
    if not stored:
        return tool_result({
            "shown": 0, "frames": [], "hit_entities": hit_entities,
            "picked": picked_meta,
            "note": (
                "Memory indexed these frame_id values, but the actual images are "
                "missing (evicted from the in-memory LRU and not present on disk). "
                "This may be an older session recorded before frame persistence."
            )})

    # 缩略到 512px 边长, 减少 prefill token
    content: List[Dict[str, Any]] = []
    summary_parts = []
    for sf in stored:
        b64 = sf.jpeg_b64 or ""
        if not b64:
            continue
        try:
            b64 = fs.thumbnail_b64(b64, max_side=_SHOW_FRAME_THUMB_SIDE, quality=70)
        except Exception:
            pass
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
        summary_parts.append(sf.frame_id)

    header = f"Memory key frames: {len(content)} image(s)"
    if hit_entities:
        header += f" (entity={hit_entities[0]['name']})"
    header += f" [{', '.join(summary_parts)}]"
    entity_evidence = _entity_details(
        mem, [str(e.get("id") or "") for e in hit_entities])
    text_evidence = _screen_text_for_frames(mem, summary_parts)
    evidence_lines: List[str] = []
    if entity_evidence:
        evidence_lines.append("\n### Stored entity attributes")
        for ent in entity_evidence[:3]:
            attrs = ent.get("attributes") or {}
            attr_txt = ", ".join(
                f"{k}={v}" for k, v in list(attrs.items())[:12])
            aliases = ", ".join((ent.get("aliases") or [])[:6])
            evidence_lines.append(
                f"- id={ent['id']} type={ent['type']} name={ent['name']!r} "
                f"attrs={{{attr_txt}}} aliases=[{aliases}]")
    if text_evidence:
        evidence_lines.append("\n### Stored frame text")
        for row in text_evidence[:5]:
            txt = " ".join((row.get("raw_text") or "").split())[:500]
            app = f" app={row.get('app')}" if row.get("app") else ""
            title = (
                f" title={row.get('window_title')!r}"
                if row.get("window_title") else "")
            evidence_lines.append(
                f"- frame_id={row['frame_id']} t={float(row.get('t_observed') or 0.0):.1f}s"
                f"{app}{title} source={row.get('source')}: {txt}")
    evidence_block = "\n".join(evidence_lines)
    # ★ 强制主 Agent 真的去看这张图, 别偷懒用 recall 文本先验瞎答。
    #   之前 qwen3.7-plus 拿到真图但答"盖子黑色" (实际白色), 就是没启用视觉推理。
    guardrail = (
        "\nCRITICAL — read before replying:\n"
        "1. The attached image_url items are real historical key frames from "
        "memory. The frontend automatically renders them for the user. You only "
        "need a brief natural confirmation.\n"
        "2. Do not write MEDIA paths, file:// URLs, /var/folders paths, "
        "~/.argus paths, or Markdown image paths in your reply. The frontend "
        "does not render those path strings and they are likely hallucinated. "
        "The images are already displayed through image_url.\n"
        "3. If the user asks about label text, nutrition, capacity, or numbers, "
        "prefer Stored entity attributes / Stored frame text as structured "
        "evidence; use the image only to verify. If the evidence lacks the field "
        "and the image is unclear, say memory has no clear record of it.\n"
    )
    content.insert(0, {"type": "text", "text": header + evidence_block + guardrail})

    return {
        "_multimodal": True,
        "content": content,
        "text_summary": header,
        "meta": {
            "shown": len(content) - 1,
            "hit_entities": hit_entities,
            "picked": picked_meta,
            "entity_evidence": entity_evidence,
            "text_evidence": text_evidence,
        },
    }


SHOW_MEMORY_FRAME_SCHEMA = {
    "name": "show_memory_frame",
    "description": (
        "Retrieve real historical key-frame images from multimodal memory for "
        "display to the user. Use this when the user wants to see what something "
        "looked like earlier or asks to bring up a historical frame.\n"
        "Unlike get_current_frame, which only sees the current live buffer, this "
        "tool reads persisted historical key frames from entity_rep_frames and "
        "FrameStore, including disk-backed frames across sessions.\n"
        "Usually pair it with query_multimodal: query first to identify "
        "the entity name, then call this tool with entity_name. If you already "
        "know the exact item name, call this tool directly.\n"
        "For factual historical questions about label text, nutrition, capacity, "
        "or counts, call query_multimodal first; this tool is for showing "
        "or visually verifying the original image.\n"
        "Pass either entity_name (recommended, fuzzy match) or frame_id (exact). "
        "The returned _multimodal images are automatically displayed to the user."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "entity_name": {
                "type": "string",
                "description": (
                    "Item/entity name. Supports fuzzy matching via exact name, "
                    "LIKE matching, and aliases."
                ),
            },
            "frame_id": {
                "type": "string",
                "description": (
                    "Exact frame_id in f_xxxxxxxxxx format, usually obtained "
                    "from recall results."
                ),
            },
        },
        "required": [],
    },
}


registry.register(
    name="show_memory_frame",
    toolset="live_watcher",
    schema=SHOW_MEMORY_FRAME_SCHEMA,
    handler=lambda args, **kw: show_memory_frame(
        entity_name=args.get("entity_name"),
        frame_id=args.get("frame_id"),
        session_id=kw.get("session_id"),
    ),
    emoji="🖼️",
)
