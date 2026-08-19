# -*- coding: utf-8 -*-
"""Scene-understanding → FrameBuffer entry-dedup dHash threshold controller.

The FrameBuffer does entry-level dHash dedup on every pushed frame (drop a new
frame that is near-identical to a recently kept one). How aggressive that dedup
is depends on the SCENE:

  • text / slides / video meeting / code review / static desktop → the picture
    barely changes; a LARGE dHash distance cutoff (e.g. 11) drops more
    near-duplicates, saving memory + downstream token cost.
  • outdoor / sports / movie / two-person conversation → the picture changes a
    lot and small differences matter; a SMALLER cutoff (e.g. 3) keeps more
    frames so the temporal detail isn't lost.

This controller periodically (every cfg.scene_probe_interval_s) samples a few
small frames from the recent buffer and asks the auxiliary vision model to
classify the scene. The dHash threshold and pacing are then derived together
from local tables. Keeping both policy values in code prevents a valid scene
label from being paired with a contradictory model-generated threshold (for
example, ``live`` plus an aggressive static-scene cutoff). On any
failure/timeout it leaves the current threshold unchanged — this is strictly
best-effort and must never stall frame ingestion.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
from io import BytesIO
from typing import Any, Optional

log = logging.getLogger("hermes.multimodal.scene")


# ── Pace tiers: 场景变化快慢 → (ttl 秒, watcher 一轮目标帧数) ──────────────────
# 每档: 到达 target_frames 或到达 ttl_sec 就执行一轮 watcher 分析。
#   slow   200s/100帧: 会议、桌面办公、文档/代码阅读、监控大屏 —— 缓慢变化
#   medium 60s/60帧  : 电影剧集、人物交谈、教学演示 —— 中速叙事
#   fast   30s/40帧  : 体育比赛、游戏直播、户外旅行 —— 快速动态
#   live   10s/15帧  : 实时交互 (视频通话、直播连麦、实时操作演示、需立即响应) —— 最高实时性
_PACE_TIERS = {
    "slow":   {"ttl_sec": 200, "target_frames": 100, "label": "缓慢变化(会议/办公/文档)"},
    "medium": {"ttl_sec": 60,  "target_frames": 60,  "label": "中速叙事(影视/交谈/教学)"},
    "fast":   {"ttl_sec": 30,  "target_frames": 40,  "label": "快速动态(比赛/游戏/户外)"},
    "live":   {"ttl_sec": 10,  "target_frames": 15,  "label": "实时交互(通话/连麦/实操)"},
}
_DEFAULT_PACE = "medium"   # 还没判过场景时的默认档

# dHash cutoff is a Hamming distance over a 64-bit perceptual hash, NOT fps.
# ``distance < cutoff`` is dropped: slow=11 drops distances 0..10, medium=7
# drops 0..6, fast=4 drops 0..3, and live=2 drops only 0..1. It must stay
# coupled to the selected scene pace so slow/static scenes deduplicate more
# aggressively while fast/live scenes preserve small visual changes.
_PACE_DHASH_THRESHOLDS = {
    "slow": 11,
    "medium": 7,
    "fast": 4,
    "live": 2,
}

# ── scene 大类 → pace 映射 ────────────────────────────────────────────────────
# 场景分类器现在只输出一个【受约束的大类】(不再自由文本, 也不再让模型单独猜 pace);
# pace 由本表硬映射得出, 保证"场景标签"与"攒帧节奏"永远一致。新增/调整大类只改这里 +
# SCENE_DHASH_SYSTEM 的枚举说明即可。未知/兜底 → _DEFAULT_PACE。
_SCENE_TO_PACE = {
    # slow — 缓慢变化 (画面长时间基本不动)
    "会议": "slow", "办公": "slow", "编程": "slow", "阅读": "slow", "监控": "slow",
    # medium — 中速叙事 (镜头/内容稳定推进)
    "影视": "medium", "对话": "medium", "教学": "medium", "新闻": "medium",
    "音乐": "medium", "绘画": "medium",
    # fast — 快速动态 (画面频繁变化)
    "体育": "fast", "游戏": "fast", "户外": "fast", "棋牌": "fast",
    "美食": "fast", "驾驶": "fast",
    # live — 实时交互 (需最高实时性)
    "直播": "live", "实时竞技": "live", "通话": "live", "实操": "live",
    # 兜底
    "其他": "medium",
}
# 供 SCENE_DHASH_SYSTEM 展示 & _parse 校验用的合法大类集合。
_SCENE_LABELS = list(_SCENE_TO_PACE.keys())
_SCENE_LABELS_EN = [
    "meeting", "office", "coding", "reading", "surveillance",
    "film", "conversation", "tutorial", "news", "music", "drawing",
    "sports", "game", "outdoor", "board_game", "food", "driving",
    "livestream", "real_time_competition", "call", "hands_on",
    "other",
]
_SCENE_ALIASES = {
    "meeting": "会议",
    "office": "办公",
    "coding": "编程",
    "programming": "编程",
    "reading": "阅读",
    "surveillance": "监控",
    "monitoring": "监控",
    "film": "影视",
    "movie": "影视",
    "video": "影视",
    "conversation": "对话",
    "dialogue": "对话",
    "tutorial": "教学",
    "teaching": "教学",
    "news": "新闻",
    "music": "音乐",
    "drawing": "绘画",
    "sports": "体育",
    "game": "游戏",
    "gaming": "游戏",
    "outdoor": "户外",
    "board_game": "棋牌",
    "boardgame": "棋牌",
    "chess": "棋牌",
    "food": "美食",
    "cooking": "美食",
    "driving": "驾驶",
    "livestream": "直播",
    "live_stream": "直播",
    "live": "直播",
    "real_time_competition": "实时竞技",
    "realtime_competition": "实时竞技",
    "call": "通话",
    "video_call": "通话",
    "hands_on": "实操",
    "operation": "实操",
    "other": "其他",
}


def pace_from_scene(scene: Optional[str]) -> Optional[str]:
    """Scene category → pace tier. Not in the enum → None (caller falls back)."""
    return _SCENE_TO_PACE.get(normalize_scene_label(scene))


def normalize_scene_label(scene: Optional[str]) -> str:
    """Map public/English scene labels to the internal legacy Chinese labels."""
    raw = (scene or "").strip()
    if raw in _SCENE_TO_PACE:
        return raw
    key = raw.lower().replace("-", "_").replace(" ", "_")
    return _SCENE_ALIASES.get(key, raw)


def pace_to_pacing(pace: Optional[str]) -> dict:
    """Map a pace tier label → {ttl_sec, target_frames, label}. Unknown → default."""
    t = _PACE_TIERS.get((pace or "").strip().lower())
    return dict(t) if t else dict(_PACE_TIERS[_DEFAULT_PACE])


def threshold_from_pace(pace: Optional[str]) -> int:
    """Return the 64-bit dHash Hamming-distance cutoff, not a frame rate."""
    return int(_PACE_DHASH_THRESHOLDS.get(
        (pace or "").strip().lower(),
        _PACE_DHASH_THRESHOLDS[_DEFAULT_PACE],
    ))


def pace_from_threshold(thr: int) -> str:
    """Derive a pace tier from the dHash threshold, so pacing (ttl/target) stays
    COUPLED to dedup strength even when the model gives a threshold but no pace.
    The threshold is a distance cutoff: distance < threshold is dropped. Thus a
    large threshold means aggressive dedup for a slow/static scene, while a
    small threshold preserves detail for a fast/live scene. Ranges mirror the
    local deterministic policy (slow≈10-12+, medium≈6-9, fast≈4-5,
    live≈2-3)."""
    try:
        t = int(thr)
    except (TypeError, ValueError):
        return _DEFAULT_PACE
    if t >= 10:
        return "slow"
    if t >= 6:
        return "medium"
    if t >= 4:
        return "fast"
    return "live"


SCENE_DHASH_SYSTEM = (
    "You are a video scene classifier. You will receive a few thumbnail frames "
    "sampled uniformly from the same video stream in chronological order. "
    "Classify the scene type.\n"
    "The `scene` value must be exactly one label from this fixed list; do not invent labels:\n"
    "   meeting, office, coding, reading, surveillance      (slow/static: the view barely changes)\n"
    "   film, conversation, tutorial, news, music, drawing   (medium narrative: content progresses steadily)\n"
    "   sports, game, outdoor, board_game, food, driving     (fast dynamic: frames change frequently)\n"
    "   livestream, real_time_competition, call, hands_on    (live interaction: low latency matters)\n"
    "   other                                                (only if none of the above fit)\n"
    "The program derives dHash dedup and frame pacing from `scene`; do not output those values.\n"
    "Output one JSON line only, with no explanation:"
    "{\"scene\": \"<one label from the fixed list>\"}"
)


class SceneDhashController:
    """Periodically probe the scene and tune the FrameBuffer entry-dedup dHash
    threshold. Runs as one async loop on the MemoryBackend event loop.

    Parameters
    ----------
    cfg : Config
    frame_buffer : FrameBuffer  (must expose sample_uniform + set_dhash_threshold)
    vision_client : an OpenAI-compatible ASYNC client (auxiliary.vision) or None.
    vision_model : model name for that client.
    stop_event : threading.Event to end the loop.
    """

    def __init__(self, cfg: Any, frame_buffer: Any, vision_client: Any,
                 vision_model: str, stop_event: Any):
        self.cfg = cfg
        self.buf = frame_buffer
        self.client = vision_client
        self.model = vision_model
        self._stop = stop_event

    # ------------------------------------------------------------------ #
    @staticmethod
    def _shrink_jpeg_b64(jpeg_b64: str, max_side: int, quality: int) -> str:
        """Downscale a JPEG (base64) to max_side longest edge, low quality —
        cheap probe images. Returns the original on any failure."""
        try:
            from PIL import Image
            raw = base64.b64decode(jpeg_b64)
            im = Image.open(BytesIO(raw))
            im = im.convert("RGB")
            w, h = im.size
            long_side = max(w, h)
            if long_side > max_side:
                scale = max_side / float(long_side)
                im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                               Image.LANCZOS)
            out = BytesIO()
            im.save(out, format="JPEG", quality=int(quality))
            return base64.b64encode(out.getvalue()).decode("ascii")
        except Exception:
            return jpeg_b64

    async def _probe_once(self) -> None:
        cfg = self.cfg
        frames = self.buf.sample_uniform(
            float(getattr(cfg, "scene_probe_window_s", 20.0) or 20.0),
            int(getattr(cfg, "scene_probe_frames", 3) or 3))
        if not frames:
            return  # nothing to look at yet
        max_side = int(getattr(cfg, "scene_probe_maxside", 256) or 256)
        quality = int(getattr(cfg, "scene_probe_quality", 50) or 50)
        probe_text = (
            f"These {len(frames)} thumbnails were sampled uniformly from the "
            f"last {getattr(cfg, 'scene_probe_window_s', 20)} seconds of the "
            "video stream, in chronological order. Classify the scene."
        )
        parts = [{"type": "text", "text": probe_text}]
        for f in frames:
            small = self._shrink_jpeg_b64(f.jpeg_b64, max_side, quality)
            parts.append({"type": "image_url",
                          "image_url": {"url": f"data:image/jpeg;base64,{small}"}})
        messages = [
            {"role": "system", "content": SCENE_DHASH_SYSTEM},
            {"role": "user", "content": parts},
        ]
        timeout = float(getattr(cfg, "scene_probe_timeout_s", 20.0) or 20.0)
        resp = await asyncio.wait_for(
            self._create_scene_completion(messages),
            timeout=timeout)
        raw = self._msg_text(resp)
        # ★ DEBUG: 记录场景分类模型的原始输出 (查长尾标签 / 自造标签的第一手证据)。
        log.info("[mm-scene][classify] raw model output: %s",
                 " ".join(str(raw or "").split())[:300])
        scene, legacy_thr = self._parse(raw)
        applied = None
        # scene 同时决定 pace 和 dHash cutoff，二者不再分别相信模型，避免出现
        # "直播 + threshold=11" 这种会把实时画面大量丢掉的矛盾组合。
        eff_pace = pace_from_scene(scene)
        _src = "scene"
        if eff_pace is not None:
            applied = self.buf.set_dhash_threshold(threshold_from_pace(eff_pace))
        # Rolling-upgrade fallback: old prompts/models may still emit only a
        # threshold. Use it only when there is no valid scene classification.
        elif legacy_thr is not None:
            applied = self.buf.set_dhash_threshold(legacy_thr)
            eff_pace = pace_from_threshold(applied)
            _src = "legacy-threshold"
        if eff_pace:
            pacing = pace_to_pacing(eff_pace)
            self.buf.set_current_scene({
                "scene": scene or "", "pace": eff_pace,
                "ttl_sec": pacing["ttl_sec"],
                "target_frames": pacing["target_frames"],
                "label": pacing["label"],
                "dhash_threshold": applied if applied is not None else self.buf.dhash_threshold,
            })
            log.info("[mm-scene] scene=%s pace=%s(%s) → dhash=%s ttl=%ss frames=%s",
                     scene or "?", eff_pace, _src,
                     applied, pacing["ttl_sec"], pacing["target_frames"])
        else:
            log.info("[mm-scene] scene probe: no usable scene/threshold, unchanged")

    async def _create_scene_completion(self, messages):
        """Call the scene-probe vision model with provider-compatible params.

        Some GPT-5-compatible endpoints reject the legacy Chat Completions
        ``max_tokens`` parameter and only allow the default temperature. Try the
        stable scene-probe shape first, then fall back without changing the
        public config contract.
        """
        base = {
            "model": self.model or None,
            "messages": messages,
            "stream": False,
        }
        # Kimi K2.6 (MaaS / vLLM 后端) 默认走 thinking mode, max_tokens 会被 reasoning_content 全部占满
        # 导致 content 输出空 → scene JSON 解析失败。显式关 thinking, 响应速度也从 ~3s 降到 ~0.4s。
        # vLLM/SGLang 的正确格式是 chat_template_kwargs.thinking=False (布尔), 不是 enable_thinking。
        if "kimi-k2" in (self.model or "").lower():
            base["extra_body"] = {"chat_template_kwargs": {"thinking": False}}
        # 采样参数一律不发 (见 transports/chat_completions.py build_kwargs)。
        default_attempt = dict(base, max_tokens=120)
        portable_attempt = dict(base, max_completion_tokens=256)
        if "gpt-5.6-luna" in (self.model or "").lower():
            attempts = [portable_attempt]
        else:
            attempts = [default_attempt, portable_attempt]
        last_exc = None
        for i, kwargs in enumerate(attempts):
            try:
                return await self.client.chat.completions.create(**kwargs)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                msg = str(exc)
                retryable = (
                    "max_tokens" in msg
                    or "max_completion_tokens" in msg
                    or "temperature" in msg
                    or "unsupported_parameter" in msg
                    or "unsupported_value" in msg
                )
                if i == len(attempts) - 1 or not retryable:
                    raise
                log.info("[mm-scene] scene probe retrying with compatible params: %s",
                         " ".join(msg.split())[:240])
        raise last_exc

    @staticmethod
    def _msg_text(resp) -> str:
        try:
            from agent.auxiliary_client import extract_content_or_reasoning
            return (extract_content_or_reasoning(resp) or "").strip()
        except Exception:
            try:
                return (resp.choices[0].message.content or "").strip()
            except Exception:
                return ""

    @staticmethod
    def _parse(raw: str):
        """Extract (scene, threshold) from the model's JSON line. Tolerates code
        fences / stray prose. scene=None when absent or not in the enum (caller
        falls back); threshold=None on parse failure (keep current)."""
        if not raw:
            return None, None
        s = raw.strip()
        # Strip ```json fences if present.
        if s.startswith("```"):
            s = s.strip("`")
            nl = s.find("\n")
            if nl >= 0:
                s = s[nl + 1:]
        # Find the first {...} block.
        i, j = s.find("{"), s.rfind("}")
        if i >= 0 and j > i:
            s = s[i:j + 1]
        try:
            data = json.loads(s)
        except Exception:
            return None, None
        if not isinstance(data, dict):
            return None, None
        scene = str(data.get("scene") or "").strip()
        scene = normalize_scene_label(scene)
        # 只接受枚举内的大类; 模型自造/越界 → None (走阈值兜底)。
        # ★ DEBUG: 记录枚举匹配结果 —— 模型自造的长尾标签在这里被丢弃, 日志留证。
        if scene and scene not in _SCENE_TO_PACE:
            log.info("[mm-scene][classify] OUT-OF-ENUM scene=%r 被丢弃 (枚举无此项, 走阈值兜底)", scene)
            scene = None
        elif scene:
            log.info("[mm-scene][classify] scene=%r 命中枚举 → pace=%s",
                     scene, _SCENE_TO_PACE.get(scene))
        else:
            log.info("[mm-scene][classify] 模型未给出 scene 字段")
        thr = data.get("dhash_threshold")
        try:
            thr = int(thr)
        except (TypeError, ValueError):
            log.info("[mm-scene][classify] dhash_threshold 解析失败: %r", data.get("dhash_threshold"))
            thr = None
        return scene, thr

    async def run(self) -> None:
        cfg = self.cfg
        if not bool(getattr(cfg, "scene_probe_use_llm", True)):
            log.info("[mm-scene] scene_probe_use_llm=False → threshold stays fixed")
            return
        if self.client is None:
            log.info("[mm-scene] no vision client → threshold stays fixed")
            return
        interval = float(getattr(cfg, "scene_probe_interval_s", 20.0) or 20.0)
        # Small initial delay so the buffer has a little content to look at.
        await asyncio.sleep(min(5.0, interval))
        while not self._stop.is_set():
            try:
                await self._probe_once()
            except asyncio.TimeoutError:
                log.debug("[mm-scene] probe timed out; keeping current threshold")
            except Exception as exc:
                log.debug("[mm-scene] probe failed: %s", exc)
            await asyncio.sleep(interval)
