# -*- coding: utf-8 -*-
"""mm-memory-eval 核心编排: 离线视频 → 构记忆(与在线一致)→ 跑 QA。

流程:
  1. 校验 (title vs 视频文件名) —— 见 mm_eval_io。
  2. 建栈: FrameBuffer + MemoryBackend.start(offline=True)(跑真实记忆代码, 仅 wake 时机
     由喂帧节奏驱动, 不涉及 monitor/watcher)。
  3. 快速喂帧: 按屏幕共享帧率 (buffer_capture_fps) 解码 → FrameBuffer.push(入口 dHash 去重,
     与在线同一路径) → 每喂够"一拍窗口"帧 → pump_one_wake()(= 在线那次 wake_once)。
     不 sleep 真实时长 → 快速读完。dHash 去重后帧数不够时不做特殊处理(与在线一致)。
  4. finalize_offline: 尾拍 + 等 L2/L3 聚合 + 一轮 reviewer → 记忆完全建好。
  5. 逐条 QA:
       --mode tool : 直接 backend.recall(query) 取 findings 作 answer_predict。
       --mode agent: 评测器先显式召回记忆, 再让真主 agent 综合 findings 与
                     提问时刻最近帧作答。该无头路径不冒充在线 QueryWorker。
     全程明确 log 每次工具调用的参数与返回。
  6. 写回 JSON (每 qa 追加 answer_predict) —— 见 mm_eval_io。
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

log = logging.getLogger("hermes.mm_memory_eval")


_EVAL_LOGGER_NAMES = (
    "hermes",
    "agent",
    "run_agent",
)


def _safe_log_stem(raw: str, *, default: str = "eval") -> str:
    stem = os.path.splitext(os.path.basename(str(raw or "")))[0]
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return stem[:80] or default


def _setup_eval_logging(video: str, log_dir: Optional[str]) -> str:
    """Install a per-mm-memory-eval log file and keep eval logs out of agent.log.

    Long offline evals are often run concurrently. The normal Hermes root logger
    writes every process to ~/.argus/logs/agent.log, which makes recall/writer
    traces impossible to read. This command-scoped handler captures the relevant
    eval loggers in a unique run directory and stops them from propagating to the
    root handlers.
    """
    from hermes_constants import get_hermes_home

    if log_dir:
        run_dir = Path(os.path.abspath(os.path.expanduser(log_dir)))
    else:
        run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{_safe_log_stem(video)}"
        run_dir = get_hermes_home() / "logs" / "mm-memory-eval" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "eval.log"
    resolved = str(log_path.resolve())

    try:
        from agent.redact import RedactingFormatter
        formatter = RedactingFormatter(
            "%(asctime)s %(levelname)s %(name)s[pid=%(process)d]: %(message)s")
    except Exception:
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s[pid=%(process)d]: %(message)s")

    handler = logging.FileHandler(resolved, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    setattr(handler, "_mm_memory_eval_handler", True)

    for name in _EVAL_LOGGER_NAMES:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        # Avoid duplicate eval handlers if a test invokes run_mm_memory_eval twice
        # in the same interpreter.
        for existing in list(logger.handlers):
            if getattr(existing, "_mm_memory_eval_handler", False):
                logger.removeHandler(existing)
                try:
                    existing.close()
                except Exception:
                    pass
        logger.addHandler(handler)
        logger.propagate = False

    logging.getLogger("hermes.mm_memory_eval").info(
        "[mm-eval] isolated log started path=%s video=%s", resolved, video)
    return resolved


def _normalize_source(source: str) -> str:
    s = (source or "camera").strip().lower()
    if s in {"screen", "screenshare", "screen_share", "desktop", "display", "window", "tab"}:
        return "screen"
    return "camera"


def _default_max_side(cfg: Any, source: str) -> int:
    if source == "screen":
        return int(getattr(cfg, "ocr_max_side", 0) or 1536)
    return 720


def _default_trace_path(result_target: str, title: str) -> str:
    root, ext = os.path.splitext(os.path.abspath(result_target))
    stem = os.path.basename(root)
    if title and title not in stem:
        root = f"{root}.{title}"
    suffix = ".trace.json" if ext.lower() == ".json" else ".json.trace.json"
    return root + suffix


def _parse_query_types(raw: Optional[str]) -> Optional[set]:
    """Parse --query-types into a lowercase set. None means no filtering."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() in {"all", "*"}:
        return None
    vals = {
        part.strip().lower()
        for chunk in s.split(",")
        for part in chunk.split()
        if part.strip()
    }
    return vals or None


def _parse_vtt_ts(raw: str) -> Optional[float]:
    """Parse WebVTT timestamp into seconds."""
    s = (raw or "").strip().replace(",", ".")
    if not s:
        return None
    parts = s.split(":")
    try:
        if len(parts) == 3:
            h = int(parts[0])
            m = int(parts[1])
            sec = float(parts[2])
        elif len(parts) == 2:
            h = 0
            m = int(parts[0])
            sec = float(parts[1])
        else:
            return None
        return h * 3600.0 + m * 60.0 + sec
    except Exception:
        return None


_VTT_TAG_RE = re.compile(r"<[^>]+>")


def _strip_vtt_text(raw: str) -> str:
    text = _VTT_TAG_RE.sub("", raw or "")
    return re.sub(r"\s+", " ", text).strip()


def _load_vtt_cues(path: str) -> List[Dict[str, Any]]:
    """Load WebVTT cues as {start,end,text}. Cue text may span multiple lines."""
    cues: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        lines = f.read().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line == "WEBVTT" or line.startswith(("Kind:", "Language:", "STYLE", "REGION")):
            i += 1
            continue
        if line.startswith("NOTE"):
            i += 1
            while i < len(lines) and lines[i].strip():
                i += 1
            continue
        if "-->" not in line:
            if i + 1 < len(lines) and "-->" in lines[i + 1]:
                i += 1
                line = lines[i].strip()
            else:
                i += 1
                continue
        left, right = line.split("-->", 1)
        start = _parse_vtt_ts(left)
        end_token = (right.strip().split() or [""])[0]
        end = _parse_vtt_ts(end_token)
        i += 1
        text_lines: List[str] = []
        while i < len(lines) and lines[i].strip():
            text_lines.append(lines[i].strip())
            i += 1
        text = _strip_vtt_text(" ".join(text_lines))
        if start is not None and end is not None and text:
            cues.append({"start": start, "end": end, "text": text})
        i += 1
    cues.sort(key=lambda c: (float(c["end"]), float(c["start"])))
    return cues


def _resolve_asr_vtt(video: str, spec: Optional[str]) -> Optional[str]:
    """Resolve --asr-vtt.

    auto: search beside the video for <stem>.auto.asr.vtt, <stem>.asr.vtt,
          then <stem>.vtt.
    none/off/false/0: disabled.
    path: explicit file, or directory containing the sidecar for this video.
    """
    raw = "auto" if spec is None else str(spec).strip()
    if not raw or raw.lower() == "auto":
        raw = "auto"
    if raw.lower() in {"none", "off", "false", "0", "no"}:
        return None

    video_abs = os.path.abspath(os.path.expanduser(video))
    vdir = os.path.dirname(video_abs)
    stem = os.path.splitext(os.path.basename(video_abs))[0]

    def _candidates(root: str) -> List[str]:
        return [
            os.path.join(root, stem + ".auto.asr.vtt"),
            os.path.join(root, stem + ".asr.vtt"),
            os.path.join(root, stem + ".vtt"),
        ]

    if raw == "auto":
        for cand in _candidates(vdir):
            if os.path.isfile(cand):
                return cand
        return None

    path = os.path.abspath(os.path.expanduser(raw))
    if os.path.isdir(path):
        for cand in _candidates(path):
            if os.path.isfile(cand):
                return cand
        raise FileNotFoundError(
            f"--asr-vtt 目录中找不到 {stem}.auto.asr.vtt / {stem}.asr.vtt / {stem}.vtt: {path}")
    if os.path.isfile(path):
        return path
    raise FileNotFoundError(f"--asr-vtt 文件不存在: {path}")


def _summarize_recall_trace(events: List[Dict[str, Any]], res: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    tools: List[Dict[str, Any]] = []
    frame_ids: List[str] = []
    frame_seen = set()
    screen_text_hits: List[str] = []
    table_hits: List[str] = []

    def _add_fids(values: Any) -> None:
        if not isinstance(values, list):
            return
        for fid in values:
            fid_s = str(fid or "").strip()
            if fid_s and fid_s not in frame_seen:
                frame_seen.add(fid_s)
                frame_ids.append(fid_s)

    if isinstance(res, dict):
        _add_fids(res.get("frame_ids"))

    for ev in events:
        if not isinstance(ev, dict):
            continue
        phase = str(ev.get("phase") or "")
        if phase == "fast_table":
            _add_fids(ev.get("frame_ids"))
            table_hits.append(
                f"fast_table findings_len={ev.get('findings_len', 0)} "
                f"frames={len(ev.get('frame_ids') or [])}")
        if phase != "tool_obs":
            continue
        _add_fids(ev.get("new_frame_ids"))
        for obs in ev.get("observations") or []:
            if not isinstance(obs, dict):
                continue
            name = str(obs.get("name") or "")
            obs_full = str(obs.get("obs_full") or "")
            obs_summary = str(obs.get("obs_summary") or "")
            _add_fids(obs.get("frame_ids"))
            item = {
                "round": ev.get("round"),
                "name": name,
                "args": obs.get("args") or {},
                "obs_len": obs.get("obs_len", len(obs_full)),
                "frame_ids": list(obs.get("frame_ids") or []),
                "obs_summary": obs_summary,
            }
            tools.append(item)
            if name == "search_screen_text" or "[search_screen_text " in obs_full:
                screen_text_hits.append(obs_summary[:1000])
            if ("[structured_tables " in obs_full
                    or "LOW_CONFIDENCE_TABLE_REBUILD" in obs_full
                    or "table_id" in obs_full.lower()):
                table_hits.append(obs_summary[:1000])

    return {
        "tools": tools,
        "frame_ids": frame_ids,
        "screen_text_hits": screen_text_hits[:8],
        "table_hits": table_hits[:8],
        "events": events,
    }


def _write_trace_json(trace: Dict[str, Any], *, trace_out: Optional[str],
                      result_target: str, title: str) -> Optional[str]:
    try:
        path = trace_out or _default_trace_path(result_target, title)
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(trace, f, ensure_ascii=False, indent=2)
        return path
    except Exception as exc:  # noqa: BLE001
        log.warning("[mm-eval] trace 写回失败: %s", exc)
        return None


def _recent_frame_image_paths(buf: Any, *, limit: int = 3) -> List[str]:
    """Write recent buffer frames to temporary JPEG files for agent-mode vision."""
    out: List[str] = []
    try:
        frames = buf.latest(max(1, int(limit)))
    except Exception:
        frames = []
    for fr in frames[-max(1, int(limit)):]:
        try:
            raw = base64.b64decode(fr.jpeg_b64)
            fd, path = tempfile.mkstemp(
                prefix=f"mmeval_{int(float(fr.ts) * 1000)}_",
                suffix=".jpg")
            with os.fdopen(fd, "wb") as f:
                f.write(raw)
            out.append(path)
        except Exception as exc:  # noqa: BLE001
            log.debug("[mm-eval] recent frame temp write failed: %s", exc)
    return out


def _cleanup_paths(paths: List[str]) -> None:
    for path in paths or []:
        try:
            os.unlink(path)
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# 视频解码 (cv2, 与 mm_memory_standalone/video_source 同款逻辑; 自包含防依赖 sibling 目录)
# --------------------------------------------------------------------------- #
def _iter_video_frames(source: str, *, capture_fps: float,
                       max_side: int, jpeg_quality: int) -> Iterator[Dict[str, Any]]:
    """惰性解码视频, 按 capture_fps 均匀降采样, yield {ts(视频内秒), jpeg_b64}。

    采样语义与在线 FrameBuffer 一致: 沿时间轴每 1/capture_fps 秒取**一真实帧**
    (非关键帧采样 —— 在线是实时流拿不到关键帧信息, 离线必须给出与在线相同的
    等间隔真实帧序, 不偷工)。

    解码用 PyAV: C 层批量全解码, 全解每一帧但只对采样点做 encode。比 cv2 逐帧
    (每帧一次 Python↔C 往返 + POS_MSEC 调用, Python 开销主导) 快 ~20x —— 83min/
    1080p 视频从 ~166min 降到 ~3min, 采样结果等价 (都是真实中间帧的等间隔采样)。
    """
    import base64
    import av
    import cv2  # 仅用于 imencode / resize (编码), 解码走 PyAV
    import numpy as np  # noqa: F401  (cv2 imencode 需要)

    interval = 1.0 / max(0.1, capture_fps)

    def _encode(img_bgr, ts: float):
        """BGR ndarray -> {ts, jpeg_b64}, 可选降边。编码失败返回 None。"""
        img = img_bgr
        if max_side and max_side > 0:
            h, w = img.shape[:2]
            longest = max(h, w)
            if longest > max_side:
                scale = max_side / float(longest)
                img = cv2.resize(img, (int(w * scale), int(h * scale)),
                                 interpolation=cv2.INTER_AREA)
        ok2, buf = cv2.imencode(
            ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        if not ok2:
            return None
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        return {"ts": ts, "jpeg_b64": b64}

    try:
        container = av.open(source)
    except Exception as exc:
        raise RuntimeError(
            f"无法打开视频: {source!r} (路径不存在? 缺编解码?): {exc}") from exc
    try:
        try:
            stream = container.streams.video[0]
        except (IndexError, KeyError) as exc:
            raise RuntimeError(f"视频无视频流: {source!r}") from exc
        time_base = stream.time_base
        next_emit_ts = 0.0
        idx = 0
        for frame in container.decode(stream):
            if frame.pts is not None and time_base is not None:
                ts = float(frame.pts * time_base)
            else:
                # 极少数流无 pts: 用序号 * native 间隔兜底。
                ts = idx / max(1.0, float(stream.average_rate or 1.0))
            idx += 1
            if ts + 1e-6 < next_emit_ts:
                continue
            next_emit_ts = ts + interval
            out = _encode(frame.to_ndarray(format="bgr24"), ts)
            if out is not None:
                yield out
    finally:
        try:
            container.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# 主编排
# --------------------------------------------------------------------------- #
_VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".m4v", ".ts")


def _find_video_file(folder: str, vid: str) -> Optional[str]:
    """在 folder 里找视频 <vid>.<ext> (常见容器扩展名), 找到返回完整路径, 否则 None。"""
    import os
    for ext in _VIDEO_EXTS:
        p = os.path.join(folder, vid + ext)
        if os.path.isfile(p):
            return p
    return None


def run_mm_memory_eval(
    video: str, json_path: str, *,
    mode: str = "tool",
    source: str = "camera",
    answer_timing: str = "after",
    query_types: Optional[str] = None,
    asr_vtt: Optional[str] = "auto",
    scene_probe: bool = True,
    capture_fps: Optional[float] = None,
    max_side: Optional[int] = None,
    jpeg_quality: int = 80,
    out: Optional[str] = None,
    trace_out: Optional[str] = None,
    log_dir: Optional[str] = None,
    timeout: float = 120.0,
) -> int:
    """返回进程退出码 (0=成功)。

    ``video`` 支持两种形态:
      * **单个视频文件** → 评这一个视频 (从 JSON 命中它对应的 qa_list)。
      * **文件夹** → 评 JSON 里的【所有】视频。开跑前先遍历检查: JSON 数组里每个
        视频 (按 ?v=<id>) 都必须在该文件夹里找到对应 <id>.<ext>, 缺任一 → 报错、
        不开跑。随后逐视频【全新建栈】评测 (记忆互不串场), 单个失败跳过继续、
        末尾汇总, 只要有失败退出码即非 0。answer_predict 写回全量 JSON 对应块。
    """
    import os
    eval_log_path = _setup_eval_logging(video, log_dir)
    print(f"[mm-eval] run log: {eval_log_path}")
    if os.path.isdir(video):
        return _run_folder_eval(
            folder=video, json_path=json_path, mode=mode,
            source=source, answer_timing=answer_timing,
            query_types=query_types,
            asr_vtt=asr_vtt,
            scene_probe=scene_probe,
            capture_fps=capture_fps, max_side=max_side,
            jpeg_quality=jpeg_quality, out=out, trace_out=trace_out,
            timeout=timeout)
    return _eval_one_video(
        video=video, json_path=json_path, mode=mode,
        source=source, answer_timing=answer_timing,
        query_types=query_types,
        asr_vtt=asr_vtt,
        scene_probe=scene_probe,
        capture_fps=capture_fps, max_side=max_side,
        jpeg_quality=jpeg_quality, out=out, trace_out=trace_out,
        timeout=timeout)


def _run_folder_eval(
    *, folder: str, json_path: str, mode: str,
    source: str, answer_timing: str, query_types: Optional[str],
    asr_vtt: Optional[str], scene_probe: bool, capture_fps: Optional[float],
    max_side: Optional[int], jpeg_quality: int, out: Optional[str],
    trace_out: Optional[str], timeout: float,
) -> int:
    """文件夹模式: 遍历检查 + 逐视频全新建栈评测 + 末尾汇总。"""
    from hermes_cli.mm_eval_io import list_eval_video_ids

    # 1) 列出 JSON 里所有视频, 遍历检查它们在文件夹里都存在 (缺任一 → 不开跑)。
    try:
        vids = list_eval_video_ids(json_path)
    except ValueError as e:
        log.error("[mm-eval] JSON 解析失败: %s", e)
        print(f"[mm-eval] JSON 解析失败: {e}")
        return 2

    resolved: List[tuple] = []   # (vid, video_path)
    missing: List[str] = []
    for vid in vids:
        vp = _find_video_file(folder, vid)
        if vp is None:
            missing.append(vid)
        else:
            resolved.append((vid, vp))
    if missing:
        log.error("[mm-eval] 文件夹缺视频, 拒绝开跑: %s", missing)
        print(f"[mm-eval] ❌ 文件夹 {folder!r} 里缺以下 {len(missing)}/{len(vids)} 个视频, "
              f"评测中止 (要求 JSON 里所有视频都在文件夹内):")
        for m in missing:
            print(f"    - {m}  (期望 {m}.mp4 等)")
        return 2

    print(f"[mm-eval] 文件夹模式: JSON 共 {len(vids)} 视频, 全部在 {folder!r} 内找到 ✓ 开始逐个评测")

    # 2) 逐视频全新建栈评测; 单个失败跳过继续。answer_predict 直接写回全量 JSON。
    #    写回目标: 有 out 则每个视频都写进 out (累积到同一份数组); 否则原地写 json_path。
    write_target = out or json_path
    failures: List[tuple] = []   # (vid, code)
    for i, (vid, vp) in enumerate(resolved, 1):
        print(f"\n[mm-eval] ===== 视频 {i}/{len(resolved)}: {vid} =====")
        try:
            cur_trace_out = None
            if trace_out:
                tr_root, tr_ext = os.path.splitext(trace_out)
                cur_trace_out = f"{tr_root}.{vid}{tr_ext or '.json'}"
            code = _eval_one_video(
                video=vp, json_path=write_target, mode=mode,
                source=source, answer_timing=answer_timing,
                query_types=query_types,
                asr_vtt=asr_vtt,
                scene_probe=scene_probe,
                capture_fps=capture_fps, max_side=max_side,
                jpeg_quality=jpeg_quality, out=None, trace_out=cur_trace_out,
                timeout=timeout)
        except Exception as e:  # noqa: BLE001  单视频异常绝不中断整批
            log.warning("[mm-eval] 视频 %s 评测异常: %s", vid, e, exc_info=True)
            print(f"[mm-eval] ❌ 视频 {vid} 异常: {e} — 跳过继续")
            code = 5
        if code != 0:
            failures.append((vid, code))
            print(f"[mm-eval] ⚠ 视频 {vid} 评测未成功 (code={code}) — 已跳过, 继续下一个")

    # 3) 汇总
    ok = len(resolved) - len(failures)
    print(f"\n[mm-eval] ========== 文件夹评测完成: {ok}/{len(resolved)} 成功 ==========")
    if failures:
        print(f"[mm-eval] 失败 {len(failures)}: " + ", ".join(f"{v}(code={c})" for v, c in failures))
        return 6
    return 0


def _eval_one_video(
    *, video: str, json_path: str, mode: str,
    source: str, answer_timing: str, query_types: Optional[str],
    asr_vtt: Optional[str], scene_probe: bool, capture_fps: Optional[float],
    max_side: Optional[int], jpeg_quality: int, out: Optional[str],
    trace_out: Optional[str], timeout: float,
) -> int:
    """评测【单个视频】的完整生命周期 (建栈→ingest→答题→写回该块)。返回退出码。"""
    from hermes_cli.mm_eval_io import resolve_eval_data, validate_title, write_eval_json
    from hermes_cli.config import load_config
    from agent.multimodal.hermes_glue import build_config
    from agent.multimodal._memory import FrameBuffer, Frame
    from agent.multimodal.memory_backend import MemoryBackend

    from hermes_cli.mm_eval_io import is_timed_eval

    # 1) 读 + 校验。resolve_eval_data 同时吃两种格式:
    #    - 单视频对象 {title, qa_list}         (旧, title 由 validate_title 校验)
    #    - 全量数组 [{video_url, qa_list}, ...] (新, 按视频文件名匹配 ?v=<id> 命中一段)
    try:
        data = resolve_eval_data(json_path, video)
        validate_title(data["title"], video)
    except ValueError as e:
        log.error("[mm-eval] 校验失败: %s", e)
        print(f"[mm-eval] 校验失败: {e}")
        return 2
    qa_list: List[Dict[str, Any]] = data["qa_list"]
    query_type_filter = _parse_query_types(query_types)
    if query_type_filter is None:
        eval_pairs: List[Tuple[int, Dict[str, Any]]] = list(enumerate(qa_list))
    else:
        eval_pairs = [
            (i, qa)
            for i, qa in enumerate(qa_list)
            if str(qa.get("query_type") or "").strip().lower() in query_type_filter
        ]
        if not eval_pairs:
            msg = (
                f"[mm-eval] query_type 筛选 {sorted(query_type_filter)} 后没有命中题目; "
                f"本视频 qa_list 共 {len(qa_list)} 题")
            log.error(msg)
            print(msg)
            return 2
    timed = is_timed_eval(data)
    log.info("[mm-eval] title=%r 视频=%s qa=%d eval_qa=%d query_types=%s mode=%s timed=%s",
             data["title"], video, len(qa_list), len(eval_pairs),
             sorted(query_type_filter) if query_type_filter else "all", mode, timed)
    if query_type_filter:
        print(f"[mm-eval] query_type 筛选: {sorted(query_type_filter)} "
              f"→ 评 {len(eval_pairs)}/{len(qa_list)} 题")
    try:
        asr_vtt_path = _resolve_asr_vtt(video, asr_vtt)
        asr_cues = _load_vtt_cues(asr_vtt_path) if asr_vtt_path else []
    except Exception as e:  # noqa: BLE001
        log.error("[mm-eval] ASR VTT 加载失败: %s", e)
        print(f"[mm-eval] ASR VTT 加载失败: {e}")
        return 2
    if asr_vtt_path:
        print(f"[mm-eval] ASR VTT: {asr_vtt_path} cues={len(asr_cues)} "
              "(按 cue 结束时间流式注入 audio_observation)")

    # 2) 建栈
    hermes_cfg = load_config()
    cfg = build_config(hermes_cfg)
    source_type = _normalize_source(source)
    answer_timing = (answer_timing or "after").strip().lower()
    if answer_timing not in {"before", "after"}:
        answer_timing = "after"
    fps = float(capture_fps if capture_fps else getattr(cfg, "buffer_capture_fps", 2.0))
    eff_max_side = int(max_side if max_side is not None else _default_max_side(cfg, source_type))
    buf = FrameBuffer(cfg)
    if hasattr(buf, "set_source_type"):
        buf.set_source_type(source_type)
    backend = MemoryBackend(buf, hermes_cfg=hermes_cfg)
    if not backend.start(offline=True, timeout=60.0):
        print("[mm-eval] 记忆后端启动超时 (writer/recall_agent/loop 60s 内未全就位) — "
              "检查 config 的 model.memory 端点是否可达 / vision_ability=true")
        backend.stop(timeout=5.0)
        return 3

    # answerer (两种模式一致, 时序模式下会在喂帧中途调用)
    if mode == "agent":
        _answer = _make_agent_answerer(backend, buf, timeout)
    else:
        _answer = _make_tool_answerer(backend, timeout)

    trace_doc: Dict[str, Any] = {
        "title": data["title"],
        "video": video,
        "json": json_path,
        "mode": mode,
        "source": source_type,
        "answer_timing": answer_timing,
        "query_types": sorted(query_type_filter) if query_type_filter else None,
        "qa_total": len(qa_list),
        "qa_evaluated": len(eval_pairs),
        "capture_fps": fps,
        "max_side": eff_max_side,
        "jpeg_quality": jpeg_quality,
        "timeout": timeout,
        "asr_vtt": asr_vtt_path,
        "asr_cues_total": len(asr_cues),
        "asr_cues_pumped": 0,
        "qa_traces": [],
    }

    def _do_answer(qa: Dict[str, Any], idx: int, eval_pos: Optional[int] = None,
                   note: str = "") -> None:
        q = str(qa.get("query", "")).strip()
        total_eval = len(eval_pairs)
        eval_label = f" eval#{eval_pos}/{total_eval}" if eval_pos is not None else ""
        log.info("[mm-eval] Q#%d/%d%s%s query_type=%r query=%r",
                 idx, len(qa_list), eval_label, note, qa.get("query_type"), q)
        print(f"[mm-eval] Q#{idx}/{len(qa_list)}{eval_label}{note}: {q}")
        try:
            pred, qtrace = _answer(q)
        except Exception as e:  # noqa: BLE001
            log.warning("[mm-eval] Q#%d 回答异常: %s", idx, e, exc_info=True)
            pred = ""
            qtrace = {"error": str(e)}
        qa["answer_predict"] = pred
        trace_doc["qa_traces"].append({
            "index": idx,
            "eval_index": eval_pos,
            "query": q,
            "query_type": qa.get("query_type"),
            "answer": qa.get("answer"),
            "time": qa.get("time"),
            "answer_predict": pred,
            "trace": qtrace,
        })
        print(f"[mm-eval]   answer_predict={pred[:120]!r}")

    win = max(1, int(round(float(getattr(cfg, "writer_wake_interval", 10.0)) * fps)))
    ocr_interval = max(0.2, float(getattr(cfg, "ocr_worker_interval", 1.0) or 1.0))
    next_ocr_ts = 0.0
    scene_interval = max(1.0, float(getattr(cfg, "scene_probe_interval_s", 20.0) or 20.0))
    next_scene_ts = scene_interval
    asr_cursor = 0
    asr_pumped = 0

    def _pump_asr_until(ts: float) -> int:
        """Append VTT cues whose end time has passed, matching streaming ASR finalization."""
        nonlocal asr_cursor, asr_pumped
        if not asr_cues:
            return 0
        batch: List[Dict[str, Any]] = []
        while asr_cursor < len(asr_cues):
            cue = asr_cues[asr_cursor]
            if float(cue.get("end") or 0.0) > ts + 1e-6:
                break
            batch.append({
                "text": cue.get("text") or "",
                "rel_ts": cue.get("start"),
                "speaker": "asr_vtt",
            })
            asr_cursor += 1
        if not batch:
            return 0
        n = int(backend.append_audio_observations(
            batch, timeout=max(5.0, min(30.0, timeout * 0.25))) or 0)
        asr_pumped += n
        trace_doc["asr_cues_pumped"] = asr_pumped
        return n

    def _pump_ocr_if_due(ts: float, *, force: bool = False) -> int:
        nonlocal next_ocr_ts
        if source_type != "screen":
            return 0
        if not force and ts + 1e-6 < next_ocr_ts:
            return 0
        n = backend.pump_ocr_once(timeout=max(8.0, timeout * 0.5))
        if not force:
            while next_ocr_ts <= ts + 1e-6:
                next_ocr_ts += ocr_interval
        return n

    def _pump_scene_if_due(ts: float, *, force: bool = False) -> bool:
        nonlocal next_scene_ts
        if not scene_probe:
            return False
        if not force and ts + 1e-6 < next_scene_ts:
            return False
        ok = backend.pump_scene_once(timeout=max(10.0, timeout * 0.5))
        if not force:
            while next_scene_ts <= ts + 1e-6:
                next_scene_ts += scene_interval
        return ok

    # ingest 是最慢的阶段 (解码全片 + 每拍跑记忆模型), 之前全程无输出, 后台跑像卡住。
    # 每 _PROGRESS_EVERY_WAKES 拍打一行进度: 已保留帧 / 去重丢帧 / 拍数 / 视频内秒 /
    # 墙钟秒。ingest 结束再打一行"记忆构建完成"汇总 (下方已有)。
    _PROGRESS_EVERY_WAKES = 5

    def _progress(pushed: int, dropped: int, wakes: int, vid_ts: float,
                  t0: float, tail: str = "") -> None:
        print(f"[mm-eval] ingest… 保留 {pushed} 帧 / 去重丢 {dropped} / {wakes} 拍 / "
              f"视频内 {vid_ts:.0f}s / 墙钟 {time.time() - t0:.0f}s{tail}",
              flush=True)

    if not timed:
        # ── 非时序 (原行为): 整片喂完 → 收尾 → 逐题答 ─────────────────────────
        pushed = dropped = wakes = 0
        n_since = 0
        t0 = time.time()
        log.info("[mm-eval] ingest 开始 video=%s fps=%.1f source=%s max_side=%d 一拍窗口=%d 帧",
                 video, fps, source_type, eff_max_side, win)
        print(f"[mm-eval] ingest 开始: 解码 {video} @ {fps}fps source={source_type} "
              f"max_side={eff_max_side}, 每 {win} 帧一拍 "
              f"(此阶段最慢, 会周期打印进度)", flush=True)
        try:
            for df in _iter_video_frames(video, capture_fps=fps, max_side=eff_max_side,
                                         jpeg_quality=jpeg_quality):
                _pump_asr_until(float(df["ts"]))
                before = buf.size
                buf.push(Frame(ts=df["ts"], jpeg_b64=df["jpeg_b64"],
                               source_type=source_type))
                if buf.size > before:
                    pushed += 1
                    n_since += 1
                else:
                    dropped += 1   # 入口 dHash 去重丢弃 (与在线一致)
                _pump_ocr_if_due(float(df["ts"]))
                _pump_scene_if_due(float(df["ts"]))
                if n_since >= win:
                    _pump_ocr_if_due(float(df["ts"]), force=True)
                    backend.pump_one_wake(timeout=max(60.0, timeout * 4))
                    wakes += 1
                    n_since = 0
                    if wakes % _PROGRESS_EVERY_WAKES == 0:
                        _progress(pushed, dropped, wakes, df["ts"], t0)
        except RuntimeError as e:
            print(f"[mm-eval] {e}")
            backend.stop()
            return 4
        _pump_asr_until(float("inf"))
        _pump_ocr_if_due(buf.latest_ts or 0.0, force=True)
        backend.finalize_offline(timeout=max(120.0, timeout * 6))
        log.info("[mm-eval] ingest 完成 pushed=%d deduped_dropped=%d wakes=%d 耗时=%.1fs",
                 pushed, dropped, wakes, time.time() - t0)
        print(f"[mm-eval] 记忆构建完成: 保留 {pushed} 帧 / 去重丢 {dropped} 帧 / "
              f"{wakes} 拍 / 耗时 {time.time() - t0:.1f}s")
        for eval_pos, (orig_i, qa) in enumerate(eval_pairs, 1):
            _do_answer(qa, orig_i + 1, eval_pos=eval_pos)
    else:
        # ── 时序交织: 视频喂到 qa.time 那一刻, 用截至此刻的记忆现状答该题 ───────
        #   决策: 只喂到时间点就答, 不强制补 wake 尾拍 (更接近在线不等满拍的真实态)。
        #   qa 按 time 升序处理; 同一时间点的题一次答完。时间点晚于视频总长的题, 在
        #   喂帧耗尽后用"整片记忆"补答 (不丢题)。
        order = sorted(range(len(eval_pairs)), key=lambda k: eval_pairs[k][1]["_time_sec"])
        cursor = 0                 # 下一个待答 qa 在 order 里的位置
        pushed = dropped = wakes = 0
        n_since = 0
        t0 = time.time()
        log.info("[mm-eval] 时序 ingest 开始 video=%s fps=%.1f source=%s max_side=%d "
                 "一拍窗口=%d 帧 answer_timing=%s (共 %d/%d 题)",
                 video, fps, source_type, eff_max_side, win, answer_timing,
                 len(eval_pairs), len(qa_list))
        print(f"[mm-eval] 时序 ingest 开始: 解码 {video} @ {fps}fps source={source_type} "
              f"max_side={eff_max_side}, 每 {win} 帧一拍, 共 {len(eval_pairs)}/{len(qa_list)} 题按时间点"
              f"{'推帧后' if answer_timing == 'after' else '推帧前'}作答 "
              f"(此阶段最慢, 会周期打印进度)",
              flush=True)

        def _answer_due(upto_ts: float) -> None:
            """喂帧已推进到 upto_ts: 把所有 time<=upto_ts 且未答的题就地答掉。"""
            nonlocal cursor
            while cursor < len(order):
                eval_i = order[cursor]
                orig_i, qa = eval_pairs[eval_i]
                qt = qa["_time_sec"]
                if qt > upto_ts + 1e-6:
                    break
                _do_answer(qa, orig_i + 1, eval_pos=eval_i + 1,
                           note=f" @t={qt:.1f}s(记忆截至此刻)")
                cursor += 1

        try:
            for df in _iter_video_frames(video, capture_fps=fps, max_side=eff_max_side,
                                         jpeg_quality=jpeg_quality):
                ts = float(df["ts"])
                _pump_asr_until(ts)
                # before 保留旧评测语义; after 更贴近线上"用户看到这一帧后提问"。
                if answer_timing == "before":
                    _answer_due(ts)
                before = buf.size
                buf.push(Frame(ts=ts, jpeg_b64=df["jpeg_b64"],
                               source_type=source_type))
                if buf.size > before:
                    pushed += 1
                    n_since += 1
                else:
                    dropped += 1
                _pump_ocr_if_due(ts)
                _pump_scene_if_due(ts)
                if n_since >= win:
                    _pump_ocr_if_due(ts, force=True)
                    backend.pump_one_wake(timeout=max(60.0, timeout * 4))
                    wakes += 1
                    n_since = 0
                    if wakes % _PROGRESS_EVERY_WAKES == 0:
                        _progress(pushed, dropped, wakes, ts, t0,
                                  tail=f" / 已答 {cursor}/{len(order)} 题")
                if answer_timing == "after":
                    _answer_due(ts)
        except RuntimeError as e:
            print(f"[mm-eval] {e}")
            backend.stop()
            return 4
        # 视频喂完: 剩下时间点超过视频总长(或未答)的题, 用整片记忆补答。
        _pump_asr_until(float("inf"))
        _pump_ocr_if_due(buf.latest_ts or 0.0, force=True)
        backend.finalize_offline(timeout=max(120.0, timeout * 6))
        log.info("[mm-eval] 时序 ingest 完成 pushed=%d deduped_dropped=%d wakes=%d 耗时=%.1fs "
                 "已中途答=%d 剩余补答=%d",
                 pushed, dropped, wakes, time.time() - t0, cursor, len(order) - cursor)
        print(f"[mm-eval] 记忆构建完成(时序): 保留 {pushed} 帧 / 去重丢 {dropped} 帧 / "
              f"{wakes} 拍 / 耗时 {time.time() - t0:.1f}s / 中途答 {cursor} 题")
        while cursor < len(order):
            eval_i = order[cursor]
            orig_i, qa = eval_pairs[eval_i]
            _do_answer(qa, orig_i + 1, eval_pos=eval_i + 1,
                       note=" @视频末尾(时间点超出视频长度)")
            cursor += 1

    # 6) 写回
    backend.stop()
    target = write_eval_json(json_path, data, out_path=out)
    log.info("[mm-eval] 写回 %s", target)
    print(f"[mm-eval] 已写回: {target}")
    trace_path = _write_trace_json(
        trace_doc, trace_out=trace_out, result_target=target,
        title=str(data.get("title") or ""))
    if trace_path:
        log.info("[mm-eval] trace 写回 %s", trace_path)
        print(f"[mm-eval] trace 已写回: {trace_path}")
    return 0


def _make_tool_answerer(backend, timeout: float):
    """tool 模式: 直接 backend.recall。"""
    def _answer(query: str) -> Tuple[str, Dict[str, Any]]:
        events: List[Dict[str, Any]] = []

        async def _collect(ev: Dict[str, Any]) -> None:
            if isinstance(ev, dict):
                events.append(ev)

        log.info("[mm-eval] recall.args brief=%r user_text=%r timeout=%.0f",
                 query, query, timeout)
        res = backend.recall(
            brief=query, user_text=query, timeout=timeout,
            on_progress=_collect)
        log.info("[mm-eval] recall.ret ok=%s found=%s rounds=%s elapsed=%.1fs "
                 "frame_ids=%d findings=%r",
                 res.get("ok"), res.get("found"), res.get("rounds"),
                 res.get("elapsed_sec", 0.0), len(res.get("frame_ids") or []),
                 (res.get("findings") or ""))
        if not res.get("ok"):
            # recall 调用本身失败 (提交失败 / 超时 / 召回 agent 内部异常), 不是"没找到"。
            # 不返回空串 (会和"未答"混淆), 而是带上 error 让结果 JSON 可区分。
            err = str(res.get("error") or "unknown").strip()
            trace = _summarize_recall_trace(events, res)
            trace["error"] = err
            trace["ok"] = False
            return f"(记忆召回失败: {err})", trace
        trace = _summarize_recall_trace(events, res)
        trace.update({
            "ok": True,
            "found": bool(res.get("found")),
            "rounds": res.get("rounds"),
            "elapsed_sec": res.get("elapsed_sec"),
            "findings_len": len(res.get("findings") or ""),
        })
        return (res.get("findings") or "").strip(), trace
    return _answer


def _make_agent_answerer(backend, buf, timeout: float):
    """Build the honest headless ``agent`` evaluation path.

    The online ``query_multimodal`` contract requires a gateway-owned answer
    slot and a running QueryWorker dispatcher.  This offline evaluator has
    neither, so it must not register a fake live session or expose the unified
    model tool and silently turn it into Recall-only.  Instead we explicitly
    prefetch RecallWorker evidence, attach the ask-time recent frames, and use a
    real main-agent turn only as the final evidence synthesizer.
    """
    recall_answer = _make_tool_answerer(backend, timeout)

    def _answer(query: str) -> Tuple[str, Dict[str, Any]]:
        from hermes_cli.oneshot import _run_agent
        findings, trace = recall_answer(query)
        trace = dict(trace)
        trace.update({
            "agent_mode": True,
            "execution_path": "offline_prefetched_recall_agent_synthesis",
            "interactive_query_worker": False,
        })
        if not trace.get("ok", False):
            # There is no trustworthy evidence to synthesize. Returning the
            # explicit recall failure is safer than asking the model to fill in
            # a missing video answer.
            trace["agent_synthesis"] = "skipped_recall_error"
            return findings, trace

        image_paths: List[str] = []
        try:
            evidence = findings or "(召回未找到与问题可靠相关的历史记忆。)"
            prompt = (
                "你正在执行离线多模态记忆评测。历史记忆证据和提问时刻"
                "最近画面已由评测器预先取得；这不是一个在线直播会话。请直接"
                "综合下面的证据和附件帧回答原问题，不要调用实时会话、监控或"
                "后台观察工具。如果证据不足，明确说明缺口，不要猜测。\n\n"
                f"原问题：{query}\n\n"
                f"RecallWorker 证据：\n{evidence}"
            )
            image_paths = _recent_frame_image_paths(buf, limit=3)
            resp = _run_agent(
                prompt,
                # Keep this headless synthesis turn away from the live
                # multimodal tool surface. Native image attachments remain
                # visible to a vision-capable main model, while ``web`` keeps
                # external-fact questions possible without pretending that a
                # gateway QueryWorker is running.
                toolsets=["web", "vision"],
                use_config_toolsets=False,
                image_paths=image_paths or None)
            ans = (resp or "").strip() if isinstance(resp, str) else str(resp or "")
            log.info("[mm-eval] agent answer=%r", ans[:120])
            trace.update({
                "agent_synthesis": "complete",
                "attached_recent_frames": len(image_paths),
            })
            return ans, trace
        finally:
            _cleanup_paths(image_paths)
    return _answer
