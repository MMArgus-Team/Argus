# -*- coding: utf-8 -*-
"""mm-memory-eval 的文件 IO 接口 (纯读写 + 校验, 不含任何记忆/召回逻辑).

JSON 结构:
    {
      "title": "<必须 == 去扩展名的视频文件名>",
      "qa_list": [ {"query": "...", "answer": "...", "time": "HH:MM:SS"(可选)}, ... ]
    }

``time`` (提问时间点) 触发【时序评测模式】: 只要 qa_list 里有任一 qa 带 time,
就逐题在"视频喂帧推进到该时间点"那一刻答题 (用截至此刻的记忆现状), 而不是整片喂完
再统一答。此时【每一题都必须有合法 time】, 缺失/无法解析 → 校验失败。都没 time →
保持原行为 (整片喂完再逐题答)。

跑完由编排层给每个 qa 元素追加 "answer_predict", 再经 write_eval_json 写回。
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

# YouTube URL 的 ?v=<id> / &v=<id> 抽取 (id == 下载后的视频文件名, 见
# download_0618_videos.py)。全量数组 0618.json 靠它把"命令行传的视频文件"
# 匹配到数组里对应的视频块。
_VIDEO_ID_RE = re.compile(r"[?&]v=([A-Za-z0-9_-]{6,})")


def video_id_from_url(url: Any) -> Optional[str]:
    """从 youtube URL 抽 ?v=<id>; 抽不到 → None。"""
    m = _VIDEO_ID_RE.search(str(url or ""))
    return m.group(1) if m else None


def parse_timestamp(raw: Any) -> Optional[float]:
    """把提问时间点解析成"视频内秒数"。接受 HH:MM:SS / MM:SS / 纯秒 (int/float/str)。
    解析不了 → None。"""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw) if raw >= 0 else None
    s = str(raw).strip()
    if not s:
        return None
    # 纯数字 (秒)
    try:
        v = float(s)
        return v if v >= 0 else None
    except ValueError:
        pass
    # HH:MM:SS / MM:SS (允许小数秒)
    parts = s.split(":")
    if not (2 <= len(parts) <= 3):
        return None
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return None
    if any(n < 0 for n in nums):
        return None
    if len(parts) == 3:
        h, m, sec = nums
    else:
        h, (m, sec) = 0.0, nums
    if m >= 60 or sec >= 60:
        return None
    return h * 3600 + m * 60 + sec


def _validate_qa_list(qa: Any, *, where: str = "") -> None:
    """校验 qa_list 结构 + 时序模式一致性 (就地给命中题写 _time_sec 缓存)。
    ``where`` 只用于报错定位 (如 "视频 <id> 的 ")。结构不合法 → ValueError。"""
    if not isinstance(qa, list) or not qa:
        raise ValueError(f"{where}缺少非空数组 'qa_list'")
    for i, item in enumerate(qa):
        if not isinstance(item, dict):
            raise ValueError(f"{where}qa_list[{i}] 必须是对象 {{query, answer}}")
        if not isinstance(item.get("query"), str) or not item["query"].strip():
            raise ValueError(f"{where}qa_list[{i}] 缺少非空字符串 'query'")
        if "answer" not in item:
            raise ValueError(f"{where}qa_list[{i}] 缺少 'answer' 字段")

    # ── 时序评测模式校验 ──────────────────────────────────────────────────────
    # 只要有任一 qa 带非空 time, 就进入时序模式; 此时【每一题都必须有合法 time】。
    has_any_time = any(
        str(item.get("time") or "").strip() or isinstance(item.get("time"), (int, float))
        for item in qa)
    if has_any_time:
        for i, item in enumerate(qa):
            secs = parse_timestamp(item.get("time"))
            if secs is None:
                raise ValueError(
                    f"时序评测模式 ({where}qa_list 里存在 time 字段) 下, qa_list[{i}] 的 "
                    f"time={item.get('time')!r} 缺失或无法解析。请用 HH:MM:SS / MM:SS / "
                    f"纯秒, 且每一题都要有 time。")
            # 缓存解析结果, 供编排层直接用 (不改动原始字段, 只加一个下划线私有键)。
            item["_time_sec"] = secs


def load_eval_json(path: str) -> Dict[str, Any]:
    """读并校验【单视频对象】eval JSON {title, qa_list}。结构不合法 → ValueError。
    数组格式 (全量 0618.json) 请用 resolve_eval_data()。"""
    if not os.path.isfile(path):
        raise ValueError(f"JSON 文件不存在: {path!r}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("JSON 顶层必须是对象 {title, qa_list} (数组请用 resolve_eval_data)")
    if "title" not in data or not isinstance(data.get("title"), str):
        raise ValueError("JSON 缺少字符串字段 'title'")
    _validate_qa_list(data.get("qa_list"))
    return data


def resolve_eval_data(json_path: str, video_path: str) -> Dict[str, Any]:
    """把评测 JSON 解析成单视频 {title, qa_list}, 供编排层直接用。

    两种输入格式都吃:
      * **单视频对象** {title, qa_list}  → 走 load_eval_json (title 会在上层校验)。
      * **全量数组** [{video_url, qa_list, ...}, ...] (如 convert 产出的 0618.json)
        → 用命令行传的【视频文件名(去扩展名)】匹配数组里某元素 video_url 的
          ?v=<id>, 命中则取该块的 qa_list 组装成 {title:<id>, qa_list}。
          命中不到 → ValueError (报错, 不静默跳过)。

    数组格式下不需要人工拆分单视频 JSON —— 直接把全量 0618.json 传进来即可。"""
    if not os.path.isfile(json_path):
        raise ValueError(f"JSON 文件不存在: {json_path!r}")
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # 单视频对象 → 老路径 (向后兼容)。
    if isinstance(raw, dict):
        return load_eval_json(json_path)

    if not isinstance(raw, list) or not raw:
        raise ValueError("JSON 顶层必须是对象 {title, qa_list} 或非空数组 [{video_url, qa_list}]")

    stem = _video_stem(video_path)   # 命令行视频文件名去扩展名 (== 期望的 ?v=<id>)
    hit = None
    seen_ids = []
    for blk in raw:
        if not isinstance(blk, dict):
            continue
        vid = video_id_from_url(blk.get("video_url"))
        if vid:
            seen_ids.append(vid)
        if vid == stem:
            hit = blk
            break
    if hit is None:
        raise ValueError(
            f"在数组 JSON {json_path!r} 里找不到视频 {stem!r} 对应的块 "
            f"(按 video_url 的 ?v=<id> 匹配)。\n"
            f"  视频: {video_path}\n"
            f"  JSON 内含 {len(seen_ids)} 个可识别视频: "
            f"{', '.join(seen_ids[:10])}{' …' if len(seen_ids) > 10 else ''}")

    _validate_qa_list(hit.get("qa_list"), where=f"视频 {stem!r} 的 ")
    return {"title": stem, "qa_list": hit["qa_list"]}


def list_eval_video_ids(json_path: str) -> List[str]:
    """列出全量数组 JSON (0618.json) 里所有可识别的视频 id (video_url 的 ?v=<id>),
    保序去重。用于【文件夹模式】开跑前的遍历检查。

    单视频对象格式 → 返回 [title] (让上层也能统一走"逐视频"流程)。
    结构非法 / 无任何可识别视频 → ValueError。"""
    if not os.path.isfile(json_path):
        raise ValueError(f"JSON 文件不存在: {json_path!r}")
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        t = str(raw.get("title") or "").strip()
        if not t:
            raise ValueError("单视频对象 JSON 缺少字符串字段 'title'")
        return [t]
    if not isinstance(raw, list) or not raw:
        raise ValueError("JSON 顶层必须是对象 {title, qa_list} 或非空数组 [{video_url, qa_list}]")
    ids: List[str] = []
    seen = set()
    for blk in raw:
        if not isinstance(blk, dict):
            continue
        vid = video_id_from_url(blk.get("video_url"))
        if vid and vid not in seen:
            seen.add(vid)
            ids.append(vid)
    if not ids:
        raise ValueError(
            f"数组 JSON {json_path!r} 里没有任何可识别视频 (video_url 缺失或无 ?v=<id>)。")
    return ids


def is_timed_eval(data: Dict[str, Any]) -> bool:
    """load_eval_json 之后判断是否走时序模式 (校验时已给每题写了 _time_sec)。"""
    return any("_time_sec" in item for item in data.get("qa_list", []))


def _video_stem(video_path: str) -> str:
    """去扩展名的视频文件名 (basename, 不含目录/后缀)。"""
    return os.path.splitext(os.path.basename(video_path))[0]


def validate_title(title: str, video_path: str) -> None:
    """校验 JSON.title 与视频文件名一致 (严格: 去扩展名 basename 精确匹配)。
    不一致 → ValueError, 让上层报错退出。"""
    stem = _video_stem(video_path)
    t = (title or "").strip()
    if t != stem:
        raise ValueError(
            f"title 校验失败: JSON.title={t!r} 与视频文件名(去扩展名)={stem!r} 不一致。"
            f"\n  视频: {video_path}\n  请确保 JSON 的 title 与视频文件名对应。")


def write_eval_json(path: str, data: Dict[str, Any],
                    out_path: str | None = None) -> str:
    """把带 answer_predict 的 data (单视频 {title, qa_list}) 写回。

    自动适配输入格式:
      * 输入 `path` 是【全量数组】(0618.json) → 只把命中视频块的 qa_list 更新回
        数组里对应元素 (按 video_url 的 ?v=<id> == data['title'] 定位), 其它视频块
        原样保留, 整个数组写回 (默认原地覆盖, 或 out_path 另存)。
      * 输入 `path` 是【单视频对象】 → 原行为 (直接写 data)。
    返回实际写入路径。"""
    # Drop the internal `_time_sec` cache added during load-time validation so it
    # doesn't leak into the written JSON (the user-facing `time` field stays).
    for item in data.get("qa_list", []):
        if isinstance(item, dict):
            item.pop("_time_sec", None)

    # 判定输入是否数组格式 (决定写回的 payload 形状)。
    src = None
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                src = json.load(f)
        except Exception:
            src = None

    if isinstance(src, list):
        # 全量数组: 把 data['qa_list'] 合并回命中块, 整个数组写出。
        title = data.get("title")
        merged = False
        for blk in src:
            if isinstance(blk, dict) and video_id_from_url(blk.get("video_url")) == title:
                blk["qa_list"] = data["qa_list"]
                merged = True
                break
        if not merged:
            # 理论上 resolve 命中过就一定能合并回; 兜底: 不静默丢结果。
            raise ValueError(
                f"写回失败: 数组 JSON 里找不到视频 {title!r} 的块 (video_url ?v=<id>)。")
        payload: Any = src
    else:
        payload = data

    target = out_path or path
    parent = os.path.dirname(os.path.abspath(target))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return target
