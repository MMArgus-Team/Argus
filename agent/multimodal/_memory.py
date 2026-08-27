# -*- coding: utf-8 -*-
"""AUTO-SPLIT from the monolithic engine — see agent/multimodal/__init__.py.

One slice of the ported TML Always-On Video Agent engine. The public surface is
re-exported from agent.multimodal.core for backward compatibility.
"""
from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import logging
import os
import re
import sqlite3
import tempfile
import threading
import time
import unicodedata
import uuid
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from io import BytesIO
from typing import (
    Any, AsyncIterator, Awaitable, Callable, Deque, Dict, List,
    Optional, Sequence, Set, Tuple,
)

import numpy as np

from ._embedding import (
    EmbeddingClient, MultimodalEmbeddingClient,
    decode_matrix, decode_vector, encode_vector,
)

try:
    import aiohttp  # optional: only needed by the local HTTP speech/Gemini backends
except ImportError:  # pragma: no cover
    aiohttp = None  # type: ignore
import httpx
from openai import AsyncOpenAI

log = logging.getLogger("hermes.multimodal")

from ._config import Config, DEFAULT_SEARCH_TOOL_PATH


# =========================================================================== #
# Keyword recall helpers (tokenize + synonym expansion + relevance scoring)
#
# The memory graph is queried by substring LIKE only. Verbatim `%query%` fails
# hard on: (1) synonyms/translation ("耳机" vs "headphones"/"AirPods"), and
# (2) multi-word queries ("红色 耳机" → requires that exact contiguous string).
# These helpers split a query into OR-terms, expand each with a small synonym
# table, and let callers score candidates by how many DISTINCT terms hit (×
# field weight) blended with recency — instead of pure recency truncation.
# =========================================================================== #

# Small, curated CN/EN synonym+alias groups for the common on-camera object
# vocabulary. Intentionally compact (precision over coverage); every token in a
# group expands to all others in that group. Extend as real misses show up.
_MM_SYNONYM_GROUPS: List[List[str]] = [
    ["耳机", "耳麦", "headphone", "headphones", "earphone", "earphones",
     "earbud", "earbuds", "耳塞", "airpods"],
    ["手机", "smartphone", "phone", "iphone", "cellphone", "移动电话"],
    ["电脑", "laptop", "computer", "notebook", "笔记本", "macbook", "pc"],
    ["相机", "camera", "摄像机", "camcorder", "单反", "微单", "dslr"],
    ["手表", "watch", "smartwatch", "腕表", "apple watch"],
    ["水杯", "杯子", "cup", "mug", "水瓶", "bottle", "tumbler"],
    ["书", "book", "书本", "书籍"],
    ["猫", "cat", "kitten", "猫咪"],
    ["狗", "dog", "puppy", "狗狗"],
    ["键盘", "keyboard"],
    ["鼠标", "mouse"],
    ["平板", "tablet", "ipad", "pad"],
]

# Reverse index: token(lower) -> set of all tokens in its group(s).
_MM_SYNONYM_INDEX: Dict[str, Set[str]] = {}
for _grp in _MM_SYNONYM_GROUPS:
    _low = [w.lower() for w in _grp]
    for _w in _low:
        _MM_SYNONYM_INDEX.setdefault(_w, set()).update(_low)

# Chinese stop tokens that add no retrieval value as standalone terms.
_MM_STOP_TOKENS: Set[str] = {
    "的", "了", "吗", "呢", "啊", "个", "这", "那", "是", "有", "在",
    "和", "跟", "与", "还", "也", "我", "你", "他", "她", "它",
    "刚才", "之前", "现在", "什么", "多少", "一下", "那个", "这个",
}

# Match CJK runs OR ASCII word runs. CJK is segmented per-character-run so we
# don't need a full word segmenter; combined with substring LIKE this still
# matches meaningfully (a 2-char CJK noun run is a good LIKE term).
_MM_TOKEN_RE = re.compile(r"[一-鿿]+|[A-Za-z0-9]+")


def mm_tokenize_query(query: str) -> List[str]:
    """Split a recall query into distinct lower-cased terms (stopwords dropped).

    CJK runs are kept whole (e.g. "红色耳机" → ["红色耳机"]); the caller also
    gets synonym expansion via mm_expand_terms. Returns [] for empty/degenerate
    input so callers can guard the `%`-matches-everything degeneration.
    """
    q = (query or "").strip().lower()
    if not q:
        return []
    raw = _MM_TOKEN_RE.findall(q)
    out: List[str] = []
    seen: Set[str] = set()
    for tok in raw:
        if tok in _MM_STOP_TOKENS or len(tok) < 1:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out


#: 超过这么多字的 CJK 整块 token 才启用二元组兜底 ("红色耳机" 这种 4 字词
#: 直接 LIKE 是准的, 不需要退化)。
_MM_CJK_WHOLE_MAX = 5
_MM_CJK_RUN_RE = re.compile(r"[一-鿿]{2,}")


def mm_cjk_bigram_groups(terms: List[str]) -> List[List[str]]:
    """Character bigrams for over-long CJK tokens, as a keyword-recall fallback.

    :func:`mm_tokenize_query` keeps a CJK run whole, which is correct for short
    noun phrases but fatal for a natural-language question: "红色外套的男人把背包
    放哪了" becomes ONE token, and the keyword arm then requires that entire
    string to appear verbatim inside description/subject/object/action. It never
    does, so the keyword arm silently returns nothing and hybrid retrieval
    degenerates to vector-only for every unspaced Chinese query.

    Returning bigrams as separate concept groups lets the existing
    distinct-concept scoring rank by how many of them a row covers, so noise
    from any single bigram stays diluted. Only used as a retry when the strict
    pass finds nothing.
    """
    groups: List[List[str]] = []
    seen: Set[str] = set()
    for t in terms:
        for run in _MM_CJK_RUN_RE.findall(t or ""):
            if len(run) <= _MM_CJK_WHOLE_MAX:
                continue
            for i in range(len(run) - 1):
                bg = run[i:i + 2]
                if bg in _MM_STOP_TOKENS or bg in seen:
                    continue
                seen.add(bg)
                groups.append([bg])
    return groups


def mm_keyword_pool_sql(
    fields: Tuple[str, ...],
    field_weights: Dict[str, float],
    groups: List[List[str]],
    all_variants: List[str],
    *,
    recency_col: str,
    fixed_params: int = 2,
    var_budget: int = 900,
) -> Tuple[str, List[str], str, List[str]]:
    """Build the shared WHERE/ORDER BY for a keyword candidate pool.

    Returns ``(where_or, likes, order_by, rank_params)``; bind order is
    ``(..., *likes, *rank_params, pool_cap)``.

    ★ 2026-08-19: 抽出来是因为 search_micro_by_keyword 与
    search_entity_by_keyword 本该是同一套检索, 实际却各写了一份, 然后**漂了**:
    micro 侧把相关性表达式下推到了 ORDER BY, entity 侧还停在
    ``ORDER BY {recency} DESC LIMIT pool_cap`` —— 于是 entity 的字段权重打分只在
    "最近 pool_cap 条命中"里生效, 一条久远但精确命中的实体根本进不了池子, 真正
    的排序信号是"新"而不是"像"。同理 CJK 二元组兜底也只加在了 micro 一侧。
    两份实现 = 两倍的漂移面, 所以合并成一个 builder, 差异只留 recency_col。

    相关性下推的代价: 原来 ``ORDER BY {recency} DESC LIMIT n`` 可借索引提前收敛,
    现在要把所有 WHERE 命中行都算一遍分。但命中集本来就得全扫 (``LIKE '%..%'``
    用不上索引), 多出来的只是每行一次表达式求值。

    ``var_budget``: SQLite 默认变量上限 999。词表爆炸时放弃 rank 表达式、退回
    时间序, 别让查询直接报错 (纯词表的 where_or 一定保留)。
    """
    if not all_variants:
        return "", [], "", []
    where_or = " OR ".join(f"{f} LIKE ?" for f in fields for _ in all_variants)
    likes = [f"%{v}%" for _ in fields for v in all_variants]

    rank_sql = ""
    rank_params: List[str] = []
    for f in fields:
        for grp in groups:
            ors = " OR ".join(f"{f} LIKE ?" for _ in grp)
            rank_params.extend(f"%{v}%" for v in grp)
            rank_sql += (
                f" + (CASE WHEN ({ors}) THEN {field_weights[f]} ELSE 0 END)")
    if not rank_sql or (len(likes) + len(rank_params)
                        + fixed_params) > var_budget:
        rank_sql, rank_params = "", []

    order_by = (f"({rank_sql.lstrip(' +')}) DESC, {recency_col} DESC"
                if rank_sql else f"{recency_col} DESC")
    return where_or, likes, order_by, rank_params


def mm_expand_terms(terms: List[str]) -> List[str]:
    """Expand each term with its synonym group. Preserves order, dedups.

    Flat list — used to build the SQL candidate pool (OR of every variant).
    For *scoring*, use mm_expand_groups so synonyms of one concept count as a
    single distinct hit, not N.
    """
    out: List[str] = []
    seen: Set[str] = set()
    for t in terms:
        variants = _MM_SYNONYM_INDEX.get(t, {t})
        for v in ([t] + sorted(variants - {t})):
            if v not in seen:
                seen.add(v)
                out.append(v)
    return out


def mm_expand_groups(terms: List[str]) -> List[List[str]]:
    """Like mm_expand_terms but keeps each original token's variants grouped.

    Returns one variant-list per original query token. Scoring counts a token
    as "hit" if ANY of its variants match — so "耳机" and "headphones" (same
    concept) count as one distinct hit, and a candidate matching two different
    concepts (e.g. 红色 + 耳机) correctly out-ranks one matching only 耳机 (even
    across the synonym variants headphone/headphones).
    """
    groups: List[List[str]] = []
    for t in terms:
        variants = _MM_SYNONYM_INDEX.get(t, {t})
        groups.append([t] + sorted(variants - {t}))
    return groups


def mm_doc_ref_terms(query: str) -> List[str]:
    """Extract explicit document anchors such as Table 1 / Figure 3.

    These anchors are much stronger than their tokenized pieces ("table", "1").
    Keeping them as phrases prevents screen-text retrieval from being dominated
    by weak terms and recency when the user asks about a specific table/figure.
    """
    terms: List[str] = []

    def _add(*vals: str) -> None:
        for val in vals:
            val = str(val or "").strip().lower()
            if val and val not in terms:
                terms.append(val)

    q = str(query or "")
    for m in re.finditer(
            r"(?<![A-Za-z])(?:table|fig(?:ure)?)\s*\.?\s*(\d+)(?![A-Za-z])",
            q, re.IGNORECASE):
        kind = "figure" if m.group(0).lower().startswith(("fig", "figure")) else "table"
        n = m.group(1)
        _add(f"{kind}{n}", f"{kind} {n}", f"{kind}.{n}")
        if kind == "table":
            _add(f"表{n}", f"表 {n}")
        else:
            _add(f"图{n}", f"图 {n}", f"fig {n}", f"fig.{n}")
    for m in re.finditer(r"(表|图)\s*(\d+)", q, re.IGNORECASE):
        prefix, n = m.group(1), m.group(2)
        if prefix == "表":
            _add(f"表{n}", f"表 {n}", f"table{n}", f"table {n}")
        else:
            _add(f"图{n}", f"图 {n}", f"figure{n}", f"figure {n}", f"fig{n}", f"fig {n}")
    return terms


def mm_identifier_variants(query: str, terms: Optional[List[str]] = None) -> List[str]:
    """Return conservative OCR/search variants for plate-like identifiers.

    Vehicle plates and on-screen IDs often mix a short CJK prefix with
    alpha-numeric text. OCR may drop the province character or one edge
    character, so exact LIKE on the full user query misses useful rows such as
    "70463D" for "粤B70463D". Keep the expansion narrow: only mixed
    letter+digit tokens produce variants, and only reasonably long fragments
    are emitted.
    """
    raw_terms = list(terms or [])
    raw_terms.extend(re.findall(r"[A-Za-z0-9]{4,}", str(query or "")))
    out: List[str] = []
    seen: Set[str] = set()

    def _add(val: str) -> None:
        val = re.sub(r"[^a-z0-9]+", "", str(val or "").lower())
        if len(val) >= 4 and val not in seen:
            seen.add(val)
            out.append(val)

    for term in raw_terms:
        tok = re.sub(r"[^a-z0-9]+", "", str(term or "").lower())
        if len(tok) < 4:
            continue
        if not (re.search(r"[a-z]", tok) and re.search(r"\d", tok)):
            continue
        _add(tok)
        leading = re.match(r"^[a-z]+(.+)$", tok)
        if leading:
            suffix = leading.group(1)
            _add(suffix)
            if suffix and suffix[-1].isalpha():
                _add(suffix[:-1])
        if tok[-1].isalpha():
            _add(tok[:-1])
        for digits in re.findall(r"\d{4,}", tok):
            _add(digits)
    return out


# =========================================================================== #
# Frame + FrameBuffer
# =========================================================================== #
@dataclass
class Frame:
    ts: float
    jpeg_b64: str
    # Origin travels with the frame itself instead of living only as mutable
    # FrameBuffer-wide state. This is required when a writer wake spans a source
    # switch (camera → screen): every persisted key frame remains auditable.
    source_type: str = ""
    # Desktop screen-share only: the specific window/screen the user picked in
    # the Electron source picker. `source_id` is the Chromium desktopCapturer id
    # (e.g. 'window:12345:0' / 'screen:0:0') — its numeric window number is what
    # Part 3's AX/UIA抓取 uses to look up the target window. `source_name` is the
    # human-readable app/window title shown in the picker. Both empty for
    # camera / web-client / legacy frames.
    source_id: str = ""
    source_name: str = ""


def frame_to_image_content(frame: Frame) -> dict:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{frame.jpeg_b64}"},
    }


def fmt_ts(ts: float) -> str:
    if ts is None:
        return "N/A"
    m, s = divmod(int(ts), 60)
    return f"{m:02d}:{s:02d}"


def new_response_id() -> str:
    return uuid.uuid4().hex[:12]


# Monitor detection must not inherit the long-term memory stream's perceptual
# dedup policy: a phone or cup can appear for only one 2fps capture tick. Keep a
# short lossless (with respect to server-received frames) side buffer instead of
# duplicating the full 30-minute ring. 60s also covers the monitor's 30s request
# timeout plus one congested retry window while staying small (~120 frames).
_MONITOR_RAW_BUFFER_SECONDS = 60.0


class FrameBuffer:
    """Thread-safe ring buffer of video frames.

    Entry-level dedup (pre-sampling): the capture side pushes at a fixed fps,
    but in many scenes (text/meetings/static desktop) adjacent frames are nearly
    identical; storing every one wastes memory and makes downstream re-analyze
    the same picture. So dedup happens at push: a new frame is compared against
    the last <=2 retained frames by dHash; if any Hamming distance is below the
    current threshold it's treated as a duplicate and dropped from the long-term
    buffer. Watcher / memory writer / main-agent snapshots read that sparse
    stream. Monitor detection reads a separate short ring containing every frame
    received from the 2fps capture path, so a brief target is never hidden by
    memory-oriented dHash policy.

    Dynamic threshold: self._dhash_threshold is adjusted by SceneDhashController
    (~every 20s, classifying the scene from a few small probe frames). It is a
    Hamming-distance cutoff: distance < threshold is dropped, so a larger
    threshold deduplicates more aggressively (static text/meeting) while a
    smaller threshold preserves more motion detail (sports/live conversation).
    Falls back to the init default when no controller is running.

    A decode failure is represented by ``None`` and excluded from dedup. Integer
    dHash 0 is a valid hash (for example, a flat black/white frame) and must take
    part in comparison like every other hash value.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        maxlen = max(8, int(cfg.buffer_seconds * cfg.buffer_capture_fps) + 4)
        self._dq: Deque[Frame] = deque(maxlen=maxlen)
        monitor_capture_fps = max(2.0, float(cfg.buffer_capture_fps or 0.0))
        monitor_maxlen = max(
            8,
            int(_MONITOR_RAW_BUFFER_SECONDS * monitor_capture_fps) + 4,
        )
        self._monitor_dq: Deque[Frame] = deque(maxlen=monitor_maxlen)
        self._lock = threading.Lock()
        # 入口去重状态。阈值可被 set_dhash_threshold 动态调整 (场景理解)。
        self._dhash_threshold: int = int(
            getattr(cfg, "framebuffer_dhash_threshold_init", 6) or 6)
        # 最近"已保留"帧的 dHash (最多 2 张), 用于和新帧比对。
        self._recent_dhash: Deque[int] = deque(maxlen=2)
        # ★ 当前场景 (由 SceneDhashController 每 ~20s 判定后写入)。FrameBuffer 是
        #   控制器 / watcher 引擎 / set_live_watcher 工具三方共享的唯一对象, 所以场景
        #   存这里 = 三方零改管道即可读到。形如
        #   {"scene": "会议", "pace": "slow", "ttl_sec": 360, "target_frames": 300, ...}。
        #   scene 为受约束的大类 (见 scene_dhash._SCENE_TO_PACE), pace 由大类硬映射。
        #   None 表示还没判过 (用默认档)。
        self._current_scene: Optional[dict] = None
        # ★ Server-authoritative time anchors (owned by the buffer, maintained
        #   under _lock in push_live). Previously these were attached to the
        #   instance by the gateway push path (封装泄漏 + check-then-set 竞态 +
        #   非-gateway 喂帧路径误判). Now they live here.
        #   _mono_epoch : time.monotonic() at first live push → frame ts base.
        #   _wall_epoch : time.time() at that same instant → wall_of(ts)=epoch+ts.
        #   _last_push_wall: time.time() of the most recent push → stream liveness.
        self._mono_epoch: Optional[float] = None
        self._wall_epoch: Optional[float] = None
        self._last_push_wall: Optional[float] = None
        self._current_source_type: str = ""
        # Monotonic source epoch. Monitor snapshots it before an async vision
        # call and discards a late verdict if the camera/screen source changed.
        self._source_generation: int = 0

    def _dedup_and_store(self, frame: Frame) -> bool:
        """Entry dHash-dedup + append. MUST be called under self._lock. Returns
        True if the frame was stored, False if dropped as a near-duplicate.
        An undecodable frame (dHash is None) is stored fail-safe without
        polluting the recent-dHash window."""
        cur = _compute_dhash(frame.jpeg_b64)
        # None → 解码失败, 判不了相似, 直接存。注意 0 是合法 dHash,
        # 常见于纯色/黑屏/白屏, 必须正常参与判重。
        if cur is not None:
            thr = self._dhash_threshold
            for prev in self._recent_dhash:
                if _hamming(cur, prev) < thr:
                    return False  # 与最近某帧几乎一样 → 丢弃, 不入 buffer
            self._recent_dhash.append(cur)
        self._dq.append(frame)
        return True

    def push(self, frame: Frame) -> None:
        # Direct push with a caller-supplied ts (tests / non-live paths). Live
        # gateway frames should use push_live for server-authoritative stamping.
        cur = _compute_dhash(frame.jpeg_b64)
        with self._lock:
            self._monitor_dq.append(frame)
            if cur is not None:
                thr = self._dhash_threshold
                for prev in self._recent_dhash:
                    if _hamming(cur, prev) < thr:
                        return
                self._recent_dhash.append(cur)
            self._dq.append(frame)

    def push_live(self, jpeg_b64: str, source_type: str = "",
                  source_id: str = "", source_name: str = "") -> dict:
        """Ingest one LIVE frame (gateway multimodal.frame path). Stamps a
        SERVER-AUTHORITATIVE monotonic ts (client ts ignored — a 0 / non-monotonic
        client clock used to strand the monitor/watcher cursor), maintains the
        mono/wall epochs + last-push wall marker, and runs entry dedup. All under
        _lock so the epoch check-then-set can't race across concurrent pushes.

        Returns {stored: bool, monitor_stored: bool, ts: float, size: int}.
        ``stored`` describes the dHash-deduped long-term buffer;
        ``monitor_stored`` is true for every accepted live frame."""
        import time as _t
        with self._lock:
            # Timestamp inside the same lock as append. Concurrent gateway
            # handlers can otherwise sample t1/t2 and acquire the lock in reverse,
            # producing a raw deque whose order disagrees with its cursor times.
            now_mono = _t.monotonic()
            now_wall = _t.time()
            if self._mono_epoch is None:
                self._mono_epoch = now_mono
                self._wall_epoch = now_wall
            ts = now_mono - self._mono_epoch
            # ★ 严格单调: Windows time.monotonic() 粒度 ~15.6ms, 同一 tick 内连续
            #   push 会拿到相同 ts → 下游 monitor/watcher 游标(按 ts 严格递增取增量帧)
            #   会漏帧/乱序。clamp 到 max(ts, 上次+ε) 兑现 docstring 的 "monotonic ts"
            #   承诺。getattr 兜底: __new__ 半构造对象(测试)无此字段也不崩。
            _last = getattr(self, "_last_live_ts", None)
            if _last is not None and ts <= _last:
                ts = _last + 1e-6
            self._last_live_ts = ts
            self._last_push_wall = now_wall
            st = (source_type or self._current_source_type or "").strip().lower()
            if st and st != self._current_source_type:
                # Never deduplicate across source boundaries. A camera frame and
                # a shared-screen frame that happen to hash alike are still two
                # distinct observations and both must enter all-scene memory.
                self._recent_dhash.clear()
                self._monitor_dq.clear()
                _prev_src = self._current_source_type
                self._current_source_type = st
                self._source_generation += 1
                try:
                    import logging as _l
                    _l.getLogger("hermes.multimodal").info(
                        "[frame-buffer] source_type transition %r → %r "
                        "(gen=%d, id=%s)",
                        _prev_src, st, self._source_generation, id(self),
                    )
                except Exception:
                    pass
            frame = Frame(
                ts=ts, jpeg_b64=jpeg_b64, source_type=st,
                source_id=str(source_id or ""),
                source_name=str(source_name or ""),
            )
            self._monitor_dq.append(frame)
            stored = self._dedup_and_store(frame)
            return {
                "stored": stored,
                "monitor_stored": True,
                "ts": ts,
                "size": len(self._dq),
                "monitor_size": len(self._monitor_dq),
            }

    def set_source_type(self, source_type: str) -> None:
        st = (source_type or "").strip().lower()
        with self._lock:
            if st != self._current_source_type:
                _prev_src = self._current_source_type
                self._recent_dhash.clear()
                self._monitor_dq.clear()
                self._source_generation += 1
                try:
                    import logging as _l
                    _l.getLogger("hermes.multimodal").info(
                        "[frame-buffer] set_source_type %r → %r (gen=%d, id=%s)",
                        _prev_src, st, self._source_generation, id(self),
                    )
                except Exception:
                    pass
            self._current_source_type = st

    @property
    def current_source_type(self) -> str:
        with self._lock:
            return self._current_source_type

    @property
    def source_generation(self) -> int:
        with self._lock:
            return self._source_generation

    def is_screen_source(self) -> bool:
        st = self.current_source_type
        return st in {
            "screen", "screenshare", "screen_share", "desktop",
            "display", "window", "tab",
        }

    @property
    def last_push_wall(self) -> Optional[float]:
        with self._lock:
            return self._last_push_wall

    @property
    def wall_epoch(self) -> Optional[float]:
        with self._lock:
            return self._wall_epoch

    def now_ts(self) -> Optional[float]:
        """Current time on the buffer's SERVER-authoritative frame timeline
        (monotonic - epoch), so other modalities (env audio) can be stamped on
        the SAME clock as video frames. Returns None if no frame pushed yet
        (epoch not anchored) → caller should fall back."""
        import time as _t
        with self._lock:
            if self._mono_epoch is None:
                return None
            return _t.monotonic() - self._mono_epoch

    def set_dhash_threshold(self, value: int) -> int:
        """Update the entry-dedup dHash threshold (from SceneDhashController).
        Clamped to [framebuffer_dhash_threshold_min, _max]. Returns the applied
        value. Since distance < threshold is dropped, larger means stronger
        dedup/fewer retained frames; smaller means weaker dedup/more detail."""
        lo = int(getattr(self.cfg, "framebuffer_dhash_threshold_min", 2) or 2)
        hi = int(getattr(self.cfg, "framebuffer_dhash_threshold_max", 20) or 20)
        v = max(lo, min(hi, int(value)))
        with self._lock:
            self._dhash_threshold = v
        return v

    @property
    def dhash_threshold(self) -> int:
        with self._lock:
            return self._dhash_threshold

    def set_current_scene(self, scene: Optional[dict]) -> None:
        """Store the latest scene classification (from SceneDhashController).
        Read by set_live_watcher (to pick ttl) + the watcher loop (per-round ttl
        / target-frames refresh)."""
        with self._lock:
            self._current_scene = dict(scene) if isinstance(scene, dict) else None

    @property
    def current_scene(self) -> Optional[dict]:
        with self._lock:
            return dict(self._current_scene) if self._current_scene else None

    def sample_uniform(self, window_s: float, n: int) -> List[Frame]:
        """Uniformly pick up to n frames from the last window_s seconds of the
        buffer (chronological). Used by SceneDhashController to probe the scene.
        Fewer than n available → returns what there is."""
        with self._lock:
            if n <= 0 or not self._dq:
                return []
            newest = self._dq[-1].ts
            lo = newest - float(window_s)
            window = [f for f in self._dq if f.ts >= lo]
        if not window:
            return []
        if len(window) <= n:
            return list(window)
        # Even indices across the window (inclusive of first + last).
        step = (len(window) - 1) / (n - 1) if n > 1 else 0
        idxs = sorted({int(round(i * step)) for i in range(n)})
        return [window[i] for i in idxs]

    def clear(self) -> int:
        """Drop all buffered frames. Called when the video SOURCE changes (e.g.
        camera → screen share) so a freshly-launched deep-analysis run doesn't
        analyze the previous source's leftover frames as the new task's opening.
        Returns how many frames were dropped."""
        with self._lock:
            n = len(self._dq)
            self._dq.clear()
            self._monitor_dq.clear()
            self._recent_dhash.clear()
            self._source_generation += 1
            return n

    def latest(self, n: int) -> List[Frame]:
        with self._lock:
            if n <= 0 or not self._dq:
                return []
            return list(self._dq)[-n:]

    def latest_one(self) -> Optional[Frame]:
        with self._lock:
            return self._dq[-1] if self._dq else None

    def all_after(self, ts: float) -> List[Frame]:
        with self._lock:
            return [f for f in self._dq if f.ts >= ts]

    def writer_all_after(self, ts: float) -> List[Frame]:
        """Snapshot frames not yet consumed by MemoryWriter.

        The long-term deque is dHash-sparse while the monitor deque preserves
        every recent capture frame. Merge both under one lock: the raw side
        prevents static/camera scenes from starving Writer, and the long-term
        side retains coarse coverage if a slow model falls more than the raw
        queue's retention window behind. The cursor is exclusive so a
        successfully processed boundary frame is never replayed.
        """
        cursor = float(ts)
        with self._lock:
            merged: Dict[float, Frame] = {}
            for frame in self._dq:
                if frame.ts > cursor:
                    merged[float(frame.ts)] = frame
            for frame in self._monitor_dq:
                if frame.ts > cursor:
                    merged[float(frame.ts)] = frame
        return [merged[key] for key in sorted(merged)]

    def monitor_all_after(self, ts: float) -> List[Frame]:
        """Every server-received capture frame after ``ts`` (no dHash dedup)."""
        with self._lock:
            return [f for f in self._monitor_dq if f.ts >= ts]

    def raw_all_le(self, ts: float, n: Optional[int] = None) -> List[Frame]:
        """Return server-received frames at/before ``ts`` without dHash dedup.

        This is the ask-time snapshot contract for one-shot QueryWorker VQA:
        the model must see the last camera/screen captures that actually
        reached the server, not the sparse long-term-memory representatives in
        ``_dq``.  ``n`` is applied while holding the same lock as ingestion so
        callers receive one atomic, time-ordered snapshot.
        """
        boundary = float(ts)
        with self._lock:
            eligible = [f for f in self._monitor_dq if f.ts <= boundary]
            if n is None:
                return eligible
            limit = max(0, int(n))
            return eligible[-limit:] if limit else []


    def all_le(self, ts: float) -> List[Frame]:
        with self._lock:
            return [f for f in self._dq if f.ts <= ts]

    @property
    def latest_ts(self) -> Optional[float]:
        with self._lock:
            return self._dq[-1].ts if self._dq else None

    @property
    def monitor_latest_ts(self) -> Optional[float]:
        with self._lock:
            return self._monitor_dq[-1].ts if self._monitor_dq else None

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._dq)

    @property
    def monitor_size(self) -> int:
        with self._lock:
            return len(self._monitor_dq)


# =========================================================================== #
# =========================================================================== #
# FrameStore: 关键帧持久化 (内存 LRU + 磁盘落库)
#
# ★ 跟 SQLite (MemoryStore) 解耦: 图像数据**不进 SQLite**, 只在内存里 LRU.
#   每个 finalized micro_event 关联 1+ 张代表帧 (frame_id), SQLite 只记 frame_id.
#
# ★ 工作流:
#   - MemoryWriter._finalize_micro 时, 调用 frame_store.maybe_store(代表帧, micro_id=mid)
#     拿到 frame_id, 回填到 MicroEvent.frame_ids 写入 SQLite.
#   - Recall 调 search_events / get_entity_context 等工具, 输出里把 frame_ids 透传出来.
#   - RecallWorker 在 raw_obs 里正则提取 frame_ids → 累积到 RecallResult.frame_ids.
#   - FrontWorker._stream_continuation 续写时, 从 frame_store 拉真图加进续写 prompt.
# =========================================================================== #
def _compute_dhash(jpeg_b64: str) -> Optional[int]:
    """Simple 9x8 difference hash → 64-bit int for key-frame dedup.

    Returns ``None`` on decode/PIL failure. Zero is a valid 64-bit dHash (common
    for flat-color frames), so it must never be used as the failure sentinel.
    """
    try:
        from PIL import Image
        raw = base64.b64decode(jpeg_b64)
        img = Image.open(BytesIO(raw)).convert("L").resize((9, 8), Image.LANCZOS)
        px = list(img.getdata())
        bits = 0
        bit_pos = 0
        for row in range(8):
            for col in range(8):
                left = px[row * 9 + col]
                right = px[row * 9 + col + 1]
                if left > right:
                    bits |= (1 << bit_pos)
                bit_pos += 1
        return bits
    except Exception:
        return None


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


@dataclass
class StoredFrame:
    frame_id: str
    ts: float                      # frame.ts (相对秒)
    wall_ts: float                 # 入库时刻
    jpeg_b64: str
    dhash: int = 0
    micro_id: Optional[str] = None     # 反向关联到 MicroEvent
    note: str = ""                     # debug 注释 (例如 "finalize_micro_anchor")
    source_type: str = ""              # camera | screen | ... (per-frame)


class FrameStore:
    """Key-frame persistence: in-memory LRU of JPEGs + on-disk persistence.

    Thread-safe via RLock. Written by MemoryWriter (maybe_store) and read by
    FrontWorker._stream_continuation (get / get_many) to re-attach real images.

    ★ Disk persistence: every stored frame is also written to
      ``<mem_db_dir>/frames/<session_stem>/<frame_id>.jpg`` (full) +
      ``<frame_id>_thumb.jpg`` (128px). LRU eviction only drops the in-memory
      copy — the disk copies stay put. ``get()`` transparently falls back to
      disk on memory miss, so a process restart can still resolve frame_ids
      that were persisted this session (or any prior session).
    """

    _ID_RE = re.compile(r"f_[0-9a-f]{10}")  # frame_id 格式约束 (跟 _new_id 同步)

    # ★ Disk-persistence tunables. Kept as class constants (not exposed via
    #   Config for now) — quality/side match "small enough to be cheap, big
    #   enough for humans to eyeball later".
    _DISK_THUMB_MAX_SIDE = 128
    _DISK_THUMB_QUALITY = 60

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._frames: Dict[str, StoredFrame] = {}
        self._order: Deque[str] = deque()       # LRU 队列 (按入库顺序)
        self._lock = threading.RLock()
        # Resolve the per-session disk directory once. Derived from mem_db_path
        # so it lives alongside the SQLite file (same session ⇒ same folder).
        # ``mem_db_path`` looks like ``…/memories/multimodal/<ts>.sqlite`` →
        # frames land in ``…/memories/multimodal/frames/<ts>/``.
        self._disk_dir: Optional[str] = None
        self._db_path: Optional[str] = None
        try:
            db_path = getattr(cfg, "mem_db_path", "") or ""
            if db_path:
                self._db_path = db_path
                base = os.path.dirname(db_path)
                stem = os.path.splitext(os.path.basename(db_path))[0]
                self._disk_dir = os.path.join(base, "frames", stem)
                os.makedirs(self._disk_dir, exist_ok=True)
        except Exception as e:
            # Best-effort: disk persistence off if we can't resolve/create the
            # directory. In-memory LRU still works, matching the old behavior.
            log.warning("[framestore] disk dir setup failed: %s", e)
            self._disk_dir = None
        self._init_index()

    def _init_index(self) -> None:
        """Create the source-neutral key-frame index used by Memory Debug.

        Image bytes remain as JPEG files; SQLite only stores metadata. Unlike
        ``screen_texts`` this table is written for camera, screen, window/tab,
        and any future visual source, so the debug timeline reflects the actual
        all-scene memory rather than only OCR-capable frames.
        """
        if not self._db_path or self._db_path == ":memory:":
            return
        try:
            with sqlite3.connect(self._db_path, timeout=5.0) as c:
                c.execute("""
                    CREATE TABLE IF NOT EXISTS memory_frames (
                        frame_id     TEXT PRIMARY KEY,
                        t_observed   REAL NOT NULL,
                        wall_ts      REAL NOT NULL,
                        micro_id     TEXT,
                        source_type  TEXT,
                        note         TEXT,
                        created_at   REAL NOT NULL
                    )
                """)
                c.execute(
                    "CREATE INDEX IF NOT EXISTS idx_memory_frames_t "
                    "ON memory_frames(t_observed)")
                c.execute(
                    "CREATE INDEX IF NOT EXISTS idx_memory_frames_source "
                    "ON memory_frames(source_type, t_observed)")
                # One-time compatibility backfill: old DBs already have visual
                # key frames referenced by micro_events but no source-neutral
                # index. Surface them immediately instead of showing only frames
                # written after this upgrade. OCR-linked frames can be labelled
                # screen; older camera frames remain honest legacy_unknown.
                tables = {
                    str(r[0]) for r in c.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'")
                }
                if "micro_events" in tables:
                    screen_ids: Set[str] = set()
                    if "screen_texts" in tables:
                        screen_ids = {
                            str(r[0]) for r in c.execute(
                                "SELECT frame_id FROM screen_texts")
                        }
                    for row in c.execute(
                        "SELECT id,t_end,frame_ids,created_at FROM micro_events"
                    ).fetchall():
                        try:
                            fids = json.loads(row[2] or "[]")
                        except Exception:
                            fids = []
                        for fid in fids if isinstance(fids, list) else []:
                            fid_s = str(fid or "").strip()
                            if not fid_s:
                                continue
                            c.execute(
                                """INSERT OR IGNORE INTO memory_frames
                                   (frame_id,t_observed,wall_ts,micro_id,source_type,note,created_at)
                                   VALUES (?,?,?,?,?,?,?)""",
                                (fid_s, float(row[1] or 0.0), 0.0, str(row[0]),
                                 "screen" if fid_s in screen_ids else "legacy_unknown",
                                 "legacy micro-event backfill", float(row[3] or time.time())),
                            )
                c.commit()
        except Exception as e:
            log.warning("[framestore] frame index setup failed: %s", e)

    def _write_index(self, sf: StoredFrame) -> None:
        if not self._db_path or self._db_path == ":memory:":
            return
        try:
            with sqlite3.connect(self._db_path, timeout=5.0) as c:
                c.execute(
                    """INSERT OR REPLACE INTO memory_frames
                       (frame_id,t_observed,wall_ts,micro_id,source_type,note,created_at)
                       VALUES (?,?,?,?,?,?,?)""",
                    (sf.frame_id, float(sf.ts), float(sf.wall_ts), sf.micro_id,
                     sf.source_type or "unknown", sf.note or "", time.time()),
                )
                c.commit()
        except Exception as e:
            log.warning("[framestore] frame index write failed fid=%s: %s",
                        sf.frame_id, e)

    def _read_index(self, frame_id: str) -> Optional[Dict[str, Any]]:
        if not self._db_path or self._db_path == ":memory:":
            return None
        try:
            with sqlite3.connect(self._db_path, timeout=5.0) as c:
                c.row_factory = sqlite3.Row
                row = c.execute(
                    "SELECT * FROM memory_frames WHERE frame_id=?", (frame_id,)
                ).fetchone()
                return dict(row) if row else None
        except Exception:
            return None

    def nearby_index(
        self, t_center: float, *, window_sec: float = 12.0,
        ask_ts: Optional[float] = None, limit: int = 8,
    ) -> List[Dict[str, Any]]:
        """Return persisted key-frame metadata nearest to ``t_center``.

        This reads the source-neutral SQLite index rather than the in-memory LRU,
        so old frames remain discoverable after RAM eviction or process restart.
        """
        if not self._db_path or self._db_path == ":memory:":
            return []
        lo = max(0.0, float(t_center) - max(0.0, float(window_sec)))
        hi = float(t_center) + max(0.0, float(window_sec))
        if ask_ts is not None:
            hi = min(hi, float(ask_ts))
        if hi < lo:
            return []
        try:
            with sqlite3.connect(self._db_path, timeout=5.0) as c:
                c.row_factory = sqlite3.Row
                rows = c.execute(
                    """SELECT * FROM memory_frames
                       WHERE t_observed >= ? AND t_observed <= ?
                       ORDER BY ABS(t_observed - ?), t_observed DESC
                       LIMIT ?""",
                    (lo, hi, float(t_center), max(1, int(limit))),
                ).fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            log.debug("[framestore] nearby index query failed: %s", e)
            return []

    # ── Disk persistence helpers ─────────────────────────────────────────────
    def _disk_paths(self, frame_id: str) -> Optional[Tuple[str, str]]:
        """Return (full_path, thumb_path) for a frame_id, or None if disk
        persistence is off (no _disk_dir resolved)."""
        if not self._disk_dir or not frame_id:
            return None
        full = os.path.join(self._disk_dir, f"{frame_id}.jpg")
        thumb = os.path.join(self._disk_dir, f"{frame_id}_thumb.jpg")
        return full, thumb

    def _write_disk(self, frame_id: str, jpeg_b64: str) -> None:
        """Write both full JPEG and 128px thumbnail to disk. Best-effort — a
        failure logs a warning but doesn't break the in-memory store."""
        paths = self._disk_paths(frame_id)
        if paths is None:
            return
        full_path, thumb_path = paths
        try:
            # Skip if already on disk (idempotency: e.g. re-store of dup fid).
            if not os.path.exists(full_path):
                with open(full_path, "wb") as f:
                    f.write(base64.b64decode(jpeg_b64))
            if not os.path.exists(thumb_path):
                thumb_b64 = self.thumbnail_b64(
                    jpeg_b64, max_side=self._DISK_THUMB_MAX_SIDE,
                    quality=self._DISK_THUMB_QUALITY)
                with open(thumb_path, "wb") as f:
                    f.write(base64.b64decode(thumb_b64))
        except Exception as e:
            log.warning("[framestore] disk write failed fid=%s: %s", frame_id, e)

    def _read_disk(self, frame_id: str) -> Optional[str]:
        """Read a frame back from disk (prefers full, falls back to thumb).
        Returns base64-encoded JPEG or None if not on disk."""
        paths = self._disk_paths(frame_id)
        if paths is None:
            return None
        full_path, thumb_path = paths
        for path in (full_path, thumb_path):
            try:
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        return base64.b64encode(f.read()).decode("ascii")
            except Exception as e:
                log.warning("[framestore] disk read failed %s: %s", path, e)
        return None

    @staticmethod
    def _new_id() -> str:
        return f"f_{uuid.uuid4().hex[:10]}"

    @classmethod
    def extract_frame_ids(cls, text: str) -> List[str]:
        """Extract all frame_ids from arbitrary text, deduped in first-seen
        order (used by RecallWorker to harvest fids embedded in tool obs)."""
        if not text:
            return []
        seen: Set[str] = set()
        out: List[str] = []
        for m in cls._ID_RE.finditer(text):
            fid = m.group(0)
            if fid not in seen:
                seen.add(fid); out.append(fid)
        return out

    def maybe_store(self, frame: Frame, *,
                    micro_id: Optional[str] = None,
                    note: str = "",
                    force: bool = False) -> Optional[str]:
        """Store frame if not a duplicate; return its frame_id (or the existing
        fid if merged as a duplicate).

        Dedup scans the last frame_store_dedup_scan_n stored frames and merges
        on the SINGLE surviving criterion:
          - Exact-same-instant: abs(frame.ts - stored.ts) < frame_store_ts_exact_eps
            → physically the same moment; merge regardless of pixels. This is
            idempotency protection against overlapping wake windows repeatedly
            picking the same real frame. If micro_id is given and the existing
            row has none, it's backfilled before returning.

        The former layer-2 dHash near-duplicate merge was REMOVED: picture-level
        dedup now happens upstream at FrameBuffer entry, so frames reaching here
        are already a sparse deduped stream. No dHash is computed or stored
        (StoredFrame.dhash stays 0), saving a PIL decode.

        force=True skips dedup entirely (always-store path).
        """
        with self._lock:
            # ★ 层2 dHash 已删 → 不再为落库算/存 dHash (StoredFrame.dhash 无人读,
            #   省一次 PIL 解码)。去重只用层1 ts。
            if not force:
                scan_n = max(1, int(self.cfg.frame_store_dedup_scan_n))
                eps = self.cfg.frame_store_ts_exact_eps
                # 扫最近 N 张 (旧实现只看 1 张, 跨 wake 抓不到 ~4s 前的重复)
                tail = list(self._order)[-scan_n:]
                for fid in reversed(tail):
                    sf = self._frames.get(fid)
                    if sf is None:
                        continue
                    dt = abs(frame.ts - sf.ts)
                    # 层1: 精确同帧 (物理同一瞬间) → 合并, 无视画面。
                    #   防的是"跨重叠 wake 窗, LLM 反复挑中同一张真实帧反复入库"
                    #   (dt≈0)。这是持久化幂等保护, 与画面级去重无关, 必须保留。
                    if dt < eps:
                        if micro_id and sf.micro_id is None:
                            sf.micro_id = micro_id
                            sf.note = note or sf.note
                            sf.source_type = (
                                getattr(frame, "source_type", "") or sf.source_type)
                            self._write_index(sf)
                        return sf.frame_id
                    # ★ 层2 (dHash 画面近似去重) 已删: 画面级去重上移到 FrameBuffer 入口,
                    #   进到这里的帧已是去重后的稀疏关键帧, 无需再按 dHash 合并。
                    # 否则: 这张候选不算重复, 继续看下一张。
            fid = self._new_id()
            sf = StoredFrame(
                frame_id=fid, ts=frame.ts, wall_ts=time.time(),
                jpeg_b64=frame.jpeg_b64,
                micro_id=micro_id, note=note,
                source_type=(getattr(frame, "source_type", "") or "unknown"),
            )
            self._frames[fid] = sf
            self._order.append(fid)
            # ★ Disk persistence (survives process restart). Best-effort:
            #   failure is logged but doesn't break the in-memory store.
            self._write_disk(fid, frame.jpeg_b64)
            self._write_index(sf)
            # LRU 淘汰 (only drops in-memory copy; on-disk copies stay)
            while len(self._order) > self.cfg.frame_store_max:
                old = self._order.popleft()
                self._frames.pop(old, None)
            return fid

    def get(self, frame_id: str) -> Optional[StoredFrame]:
        with self._lock:
            sf = self._frames.get(frame_id)
        if sf is not None:
            return sf
        # ★ Memory miss (LRU-evicted or process just restarted): try disk.
        #   We don't repopulate the in-memory LRU on a disk hit — that would
        #   thrash LRU on background recall scans of old frames. Metadata is
        #   restored from memory_frames, including the original event time.
        b64 = self._read_disk(frame_id)
        if b64 is None:
            return None
        meta = self._read_index(frame_id) or {}
        return StoredFrame(
            frame_id=frame_id,
            ts=float(meta.get("t_observed") or 0.0),
            wall_ts=float(meta.get("wall_ts") or 0.0),
            jpeg_b64=b64,
            micro_id=meta.get("micro_id"),
            note=str(meta.get("note") or ""),
            source_type=str(meta.get("source_type") or ""),
        )

    def get_many(self, frame_ids: List[str]) -> List[StoredFrame]:
        out: List[StoredFrame] = []
        with self._lock:
            # First pass: memory hits (cheap, keeps the lock briefly).
            mem_hits = {fid: self._frames.get(fid) for fid in frame_ids}
        for fid in frame_ids:
            sf = mem_hits.get(fid)
            if sf is not None:
                out.append(sf)
                continue
            # ★ Memory miss → disk fallback (same policy as get()).
            b64 = self._read_disk(fid)
            if b64 is not None:
                meta = self._read_index(fid) or {}
                out.append(StoredFrame(
                    frame_id=fid,
                    ts=float(meta.get("t_observed") or 0.0),
                    wall_ts=float(meta.get("wall_ts") or 0.0),
                    jpeg_b64=b64,
                    micro_id=meta.get("micro_id"),
                    note=str(meta.get("note") or ""),
                    source_type=str(meta.get("source_type") or ""),
                ))
        return out

    def thumbnail_b64(self, jpeg_b64: str, *,
                      max_side: int = 0, quality: int = 70) -> str:
        """Downscale to a JPEG thumbnail (saves bandwidth when pushing to UI).
        max_side<=0 returns the original untouched; falls back to the original
        on any decode error."""
        if max_side <= 0:
            return jpeg_b64
        try:
            from PIL import Image
            raw = base64.b64decode(jpeg_b64)
            img = Image.open(BytesIO(raw)).convert("RGB")
            if max(img.size) > max_side:
                img.thumbnail((max_side, max_side), Image.LANCZOS)
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            return base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:
            return jpeg_b64

    def size(self) -> int:
        with self._lock:
            return len(self._frames)

    def reset(self) -> int:
        """Clear the LRU (called on video-source switch). Returns the number of
        frames dropped, for driver logging."""
        with self._lock:
            n = len(self._frames)
            self._frames.clear()
            self._order.clear()
        return n


# =========================================================================== #
# SearchFactStore (session-scoped external-search evidence cache)
# =========================================================================== #
@dataclass(frozen=True)
class SearchFact:
    """One externally retrieved fact/evidence item.

    ``value`` is evidence returned by the search provider, not a claim invented
    by the Router.  Provenance and freshness travel with the value so callers
    can decide whether it is still safe to reuse.  Instances are immutable;
    updates replace the whole item atomically inside :class:`SearchFactStore`.
    """

    key: str
    query: str
    value: str
    source_tool: str
    source_urls: Tuple[str, ...] = ()
    fetched_at: float = 0.0
    expires_at: float = 0.0
    confidence: float = 0.0

    def is_expired(self, now: Optional[float] = None) -> bool:
        return self.expires_at <= (time.time() if now is None else float(now))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "query": self.query,
            "value": self.value,
            "source_tool": self.source_tool,
            "source_urls": list(self.source_urls),
            "fetched_at": self.fetched_at,
            "expires_at": self.expires_at,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class SearchFactSnapshot:
    version: int = 0
    facts: Dict[str, SearchFact] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "facts": {k: v.to_dict() for k, v in self.facts.items()},
        }

    def display_values(self) -> Dict[str, str]:
        """Legacy/UI projection that contains only JSON-safe string values."""
        return {fact.query: fact.value for fact in self.facts.values()}

    def render_full(self, *, max_items: int = 8,
                    value_max_chars: int = 800) -> str:
        if not self.facts:
            ft = "  (暂无)"
        else:
            # Recent evidence is the most useful and keeps the Router prompt
            # bounded even when the session cache itself is larger.
            items = list(self.facts.values())[-max(1, int(max_items)):]
            lines: List[str] = []
            for fact in items:
                value = fact.value
                if value_max_chars > 0 and len(value) > value_max_chars:
                    value = value[:value_max_chars] + "…"
                src = ", ".join(fact.source_urls[:3]) or fact.source_tool
                lines.append(
                    f"  - query={fact.query!r} confidence={fact.confidence:.2f} "
                    f"expires_at={fact.expires_at:.3f} source={src}\n    {value}"
                )
            ft = "\n".join(lines)
        return (
            f"### SearchFactStore v{self.version}\n"
            f"#### session-scoped external-search evidence\n{ft}\n"
        )


class SearchFactStore:
    """Thread-safe, bounded, in-memory cache of external search evidence.

    The store is deliberately session-scoped (no disk persistence).  All
    mutations happen through one synchronous ``upsert_many`` critical section,
    which makes it safe for the MemoryBackend and Watcher event-loop threads to
    share the same instance.  Readers receive immutable ``SearchFact`` values in
    a copied snapshot and never touch the live dictionary.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._facts: Dict[str, SearchFact] = {}
        self._version = 0
        self._lock = threading.RLock()
        self._listeners: List[Callable[[SearchFactSnapshot], None]] = []
        self._max_items = max(1, int(getattr(
            cfg, "search_facts_max", getattr(cfg, "facts_max", 64)) or 64))
        self._value_max_chars = max(1, int(getattr(
            cfg, "search_fact_value_max_chars", 4000) or 4000))

    @staticmethod
    def normalize_query(query: str) -> str:
        text = unicodedata.normalize("NFKC", str(query or ""))
        text = re.sub(r"\s+", " ", text).strip().casefold()
        return text.rstrip("?？!！。.,，;；:：").strip()

    def _prune_expired_locked(self, now: float) -> bool:
        expired = [key for key, fact in self._facts.items()
                   if fact.expires_at <= now]
        for key in expired:
            self._facts.pop(key, None)
        if expired:
            self._version += 1
        return bool(expired)

    def add_listener(self, cb: Callable[[SearchFactSnapshot], None]) -> None:
        """Register a synchronous change listener.

        Callbacks always run after the store lock is released.  They must stay
        lightweight; loop-bound async callbacks are intentionally unsupported.
        """
        if not callable(cb):
            raise TypeError("SearchFactStore listener must be callable")
        with self._lock:
            self._listeners.append(cb)

    def _notify(self, snapshot: SearchFactSnapshot) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for cb in listeners:
            try:
                cb(snapshot)
            except Exception as exc:
                log.warning("[search facts listener] %s", exc)

    def _normalize_fact(self, fact: SearchFact,
                        now: float) -> Optional[SearchFact]:
        if not isinstance(fact, SearchFact):
            return None
        query = " ".join(str(fact.query or "").split()).strip()
        key = self.normalize_query(query)
        value = str(fact.value or "").strip()
        source_tool = str(fact.source_tool or "").strip()
        if not key or not query or not value or not source_tool:
            return None
        try:
            fetched_at = float(fact.fetched_at or now)
            expires_at = float(fact.expires_at or 0.0)
            confidence = max(0.0, min(1.0, float(fact.confidence or 0.0)))
        except (TypeError, ValueError):
            return None
        if expires_at <= max(now, fetched_at):
            return None
        value = value[:self._value_max_chars]
        urls: List[str] = []
        for raw_url in (fact.source_urls or ()):
            url = str(raw_url or "").strip()
            if url and url not in urls:
                urls.append(url)
            if len(urls) >= 12:
                break
        return SearchFact(
            key=key, query=query, value=value, source_tool=source_tool,
            source_urls=tuple(urls), fetched_at=fetched_at,
            expires_at=expires_at, confidence=confidence,
        )

    def snapshot(self, *, now: Optional[float] = None) -> SearchFactSnapshot:
        current = time.time() if now is None else float(now)
        with self._lock:
            changed = self._prune_expired_locked(current)
            snapshot = SearchFactSnapshot(
                version=self._version, facts=dict(self._facts))
        if changed:
            self._notify(snapshot)
        return snapshot

    def get_by_query(self, query: str, *,
                     now: Optional[float] = None) -> Optional[SearchFact]:
        key = self.normalize_query(query)
        if not key:
            return None
        current = time.time() if now is None else float(now)
        with self._lock:
            changed = self._prune_expired_locked(current)
            fact = self._facts.get(key)
            snapshot = (SearchFactSnapshot(
                version=self._version, facts=dict(self._facts))
                if changed else None)
        if snapshot is not None:
            self._notify(snapshot)
        return fact

    def upsert_many(self, facts: List[SearchFact], *,
                    now: Optional[float] = None) -> SearchFactSnapshot:
        """Validate and atomically upsert a batch, newest value winning by key."""
        current = time.time() if now is None else float(now)
        normalized = [self._normalize_fact(fact, current) for fact in facts]
        accepted = [fact for fact in normalized if fact is not None]
        with self._lock:
            changed = self._prune_expired_locked(current)
            mutated = False
            if accepted:
                for fact in accepted:
                    existing = self._facts.get(fact.key)
                    # Different QueryWorker tasks may run the same normalized
                    # query concurrently.  Completion/answer order is not
                    # retrieval order, so a slower old result must never replace
                    # evidence fetched later.  Equal timestamps intentionally
                    # retain ordinary last-commit-wins semantics.
                    if (existing is not None
                            and existing.fetched_at > fact.fetched_at):
                        continue
                    self._facts.pop(fact.key, None)
                    self._facts[fact.key] = fact
                    mutated = True
                while mutated and len(self._facts) > self._max_items:
                    self._facts.pop(next(iter(self._facts)))
                if mutated:
                    self._version += 1
                    changed = True
            snapshot = SearchFactSnapshot(
                version=self._version, facts=dict(self._facts))
        if changed:
            self._notify(snapshot)
        return snapshot

    def reset(self) -> None:
        with self._lock:
            changed = bool(self._facts)
            if self._facts:
                self._facts.clear()
                self._version += 1
            snapshot = SearchFactSnapshot(
                version=self._version, facts=dict(self._facts))
        if changed:
            self._notify(snapshot)


# Import compatibility for callers that have not migrated yet.  These aliases
# intentionally point at the new search-only types; MemoryWriter no longer reads
# from or writes to them.
SharedContext = SearchFactSnapshot
ContextStore = SearchFactStore


# =========================================================================== #
# ConversationLog (user / assistant / observation / audio_observation)
# =========================================================================== #
@dataclass
class Turn:
    """One conversation-log entry.

    role: 'user' | 'assistant' | 'system'
    kind:
      'normal'             user question / assistant reply
      'proactive'          front-driven proactive SPEAK
      'observation'        [visual observation] written by MemoryWriter
      'audio_observation'  [audio observation] from screen-share/ambient ASR
    speaker (v6): meaningful only for audio_observation — the WhisperX
      diarization speaker label (e.g. "SPEAKER_00"), rendered into the
      [audio observation mm:ss SPK] tag so the LLM knows who spoke. None for
      legacy single-speaker ASR.
    rel_ts: frame timeline seconds (frame.ts), when known."""
    role: str
    content: str
    wall_ts: float = field(default_factory=time.time)
    kind: str = "normal"
    rel_ts: Optional[float] = None   # 帧时间戳 (frame.ts)
    speaker: Optional[str] = None    # ★ v6: WhisperX diarize 的 speaker 标签
    row_id: Optional[int] = None     # durable audio_observations.id (recall dedup/RRF)


class ConversationLog:
    def __init__(self, max_chars: int = 100_000, min_turns: int = 1,
                 max_bg_obs: int = 200, audio_store: Optional[Any] = None):
        self.max_chars = max_chars
        self.min_turns = max(1, min_turns)
        self.max_bg_obs = max(20, max_bg_obs)
        # ConversationLog stays bounded for prompt/RAM safety, while every ASR
        # observation is also persisted by MemoryStore for long-horizon recall.
        self.audio_store = audio_store
        self._turns: List[Turn] = []
        # ★ C5: 该实例被 MemoryBackend loop 和 WatcherAgent loop 两个不同事件循环/
        #   线程共享 (router_engine 复用 mb.conversation). asyncio.Lock 跨 loop 无效,
        #   改用 threading.RLock: 对 _turns 的所有读写都在同一把 OS 级锁内, 真正互斥.
        self._lock = threading.RLock()

    @staticmethod
    def _est(text: str) -> int:
        return len(text or "")

    def _trim_obs_locked(self) -> int:
        obs_idx = [i for i, t in enumerate(self._turns) if t.kind == "observation"]
        if len(obs_idx) <= self.max_bg_obs:
            return 0
        drop = len(obs_idx) - self.max_bg_obs
        drop_set = set(obs_idx[:drop])
        self._turns = [t for i, t in enumerate(self._turns) if i not in drop_set]
        return drop

    def _trim_chars_locked(self) -> int:
        total = sum(self._est(t.content) for t in self._turns)
        trimmed = 0
        if total > self.max_chars:
            keep: List[Turn] = []
            removed = 0
            for t in self._turns:
                if total <= self.max_chars:
                    keep.append(t); continue
                if t.kind == "observation":
                    total -= self._est(t.content)
                    removed += 1
                else:
                    keep.append(t)
            if removed:
                self._turns = keep
                trimmed += removed
        keep_min = self.min_turns * 2
        while total > self.max_chars and len(self._turns) > keep_min:
            removed_turn = self._turns.pop(0)
            total -= self._est(removed_turn.content)
            trimmed += 1
        return trimmed

    async def append(self, role: str, content: str, kind: str = "normal",
                     rel_ts: Optional[float] = None,
                     speaker: Optional[str] = None) -> None:
        """Append a turn (auto-trimmed by obs count then char budget).

        speaker (v6): set only for audio_observation, from WhisperX diarize.

        C5: keeps the `async def` signature (callers all `await ...append(...)`)
        but locks with a synchronous `with self._lock` (threading.RLock, shared
        across event loops), not `async with`. The critical section is pure
        in-memory list work and doesn't block the event loop.
        """
        turn = Turn(role=role, content=content, kind=kind,
                    rel_ts=rel_ts, speaker=speaker)
        with self._lock:
            self._turns.append(turn)
            self._trim_obs_locked()
            self._trim_chars_locked()
        if kind == "audio_observation" and self.audio_store is not None:
            try:
                turn.row_id = self.audio_store.insert_audio_observation(turn)
            except Exception as exc:
                # Persistence must not interrupt live ASR delivery. The in-memory
                # turn remains available and the failure is visible in logs.
                log.warning("[mem] persist audio_observation failed: %s", exc)

    def snapshot(self) -> List[Turn]:
        with self._lock:
            return list(self._turns)

    def recent_n(self, n: int) -> List[Turn]:
        if n <= 0:
            return []
        with self._lock:
            return list(self._turns[-n:])

    def latest_obs(self, n: int) -> List[Turn]:
        if n <= 0:
            return []
        with self._lock:
            obs = [t for t in self._turns if t.kind == "observation"]
        return obs[-n:]

    def latest_audio_obs(self, n: int) -> List[Turn]:
        if n <= 0:
            return []
        with self._lock:
            obs = [t for t in self._turns if t.kind == "audio_observation"]
        return obs[-n:]

    @staticmethod
    def _fmt_stamp(t: Turn, *, time_format: str = "mm_ss") -> str:
        """Render one Turn's timestamp. Uses rel_ts when set, else wall_ts % 3600.
        time_format:
          - "mm_ss" (default): "02:13" (fmt_ts) — UI / Front / Router / Recall
          - "sec"            : "133s"           — MemoryWriter (matches the
              [Frame i | ts=XX.Xs] seconds format so the LLM aligns turns to
              frames with zero conversion)
        """
        ts = t.rel_ts if t.rel_ts is not None else (t.wall_ts % 3600)
        if time_format == "sec":
            return f"{ts:.0f}s"
        return fmt_ts(ts)

    @classmethod
    def _fmt_turn(cls, t: Turn, *, time_format: str = "mm_ss") -> str:
        if t.kind == "observation":
            stamp = cls._fmt_stamp(t, time_format=time_format)
            return f"[画面观察 {stamp}] {t.content}"
        if t.kind == "audio_observation":
            stamp = cls._fmt_stamp(t, time_format=time_format)
            # ★ v6: WhisperX 给出 speaker 标签时拼进 tag, 让 LLM 知道"这段是谁说的"
            spk = f" {t.speaker}" if t.speaker else ""
            return f"[音频观察 {stamp}{spk}] {t.content}"
        # ★ user / assistant / proactive: 有 rel_ts (video 内时间) 就带上, 让模型
        #   能精确知道"用户提问"和"画面/音频观察"的相对时序 (E 题时序敏感).
        #   无则不带 (向后兼容老数据 / 没传 rel_ts 的边缘场景).
        if t.rel_ts is not None:
            stamp = " " + cls._fmt_stamp(t, time_format=time_format)
        else:
            stamp = ""
        if t.kind == "proactive":
            return f"[助手(主动){stamp}] {t.content}"
        tag = "用户" if t.role == "user" else "助手"
        return f"[{tag}{stamp}] {t.content}"

    def as_dump_text(self, *, time_format: str = "mm_ss",
                     audio_exclude_after: Optional[float] = None,
                     exclude_audio: bool = False,
                     exclude_observation: bool = False) -> str:
        """Render the conversation history into one big string.

        Optional filters:
          - time_format: "mm_ss" (default) | "sec" (MemoryWriter — aligns with
              the ts=XX.Xs Frame-tag seconds for zero conversion).
          - audio_exclude_after: drop audio_observation turns with rel_ts >= this
              (MemoryWriter routes in-window subtitles through a separate ASR
              block while leaving older subtitles in the dump).
          - exclude_audio (E8): True → drop ALL audio_observation turns, for
              roles that don't need ASR subtitles (Front cruise / Router),
              reducing context noise. Subtitles are recalled on demand via
              Recall's search_audio tool.
          - exclude_observation (E10): True → drop ALL [visual observation]
              turns. Used by MemoryWriter in event-timeline mode (macro+micro
              events replace raw obs), avoiding duplicate context + token waste
              from event text and raw observations coexisting.
        """
        with self._lock:
            turns = list(self._turns)
        if not turns:
            return "(对话尚未开始)"
        lines: List[str] = []
        for t in turns:
            if exclude_audio and t.kind == "audio_observation":
                continue
            if exclude_observation and t.kind == "observation":
                continue
            if (audio_exclude_after is not None
                    and t.kind == "audio_observation"
                    and t.rel_ts is not None
                    and t.rel_ts >= audio_exclude_after):
                continue
            lines.append(self._fmt_turn(t, time_format=time_format))
        return "\n".join(lines)

    def reset(self) -> None:
        with self._lock:
            self._turns = []


# =========================================================================== #
# MemoryStore (SQLite, 4 张表)
#   micro_events / macro_events / super_events / entities / edges
#   全部带时间戳, Recall 查询时 attach WHERE t<=ask_ts (D3)
# =========================================================================== #
SCHEMA_SQL = """
-- ★ #6: 库级元信息 (key-value). summary=首个 macro 回填的简短摘要 (取代改名活库,
--   避免并发 rename 引发 no such table race)。库文件名固定为 <时间戳>.sqlite。
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS micro_events (
    id           TEXT PRIMARY KEY,
    t_start      REAL NOT NULL,
    t_end        REAL NOT NULL,
    description  TEXT NOT NULL,
    subject      TEXT,
    object       TEXT,
    action       TEXT,
    macro_id     TEXT,
    facts_keys   TEXT,
    frame_ids    TEXT,
    created_at   REAL NOT NULL,
    -- ★ E4 (evolve): 软删除 + 修订追溯
    superseded_by TEXT,           -- 被 Reviewer merge/split 后指向新 micro id
    revised_at   REAL,            -- 最后一次被改写时间
    revision_count INTEGER DEFAULT 0,
    -- ★ Hybrid retrieval (一期): description 的文本 embedding (float16 BLOB).
    --   Writer 写入后异步算, Reviewer revise_micro_desc/merge/split 后刷新;
    --   NULL 表示尚未算 (等待 backfill 任务或 embedding 端点未配置).
    text_embedding BLOB
);
CREATE INDEX IF NOT EXISTS idx_micro_t     ON micro_events(t_end);
CREATE INDEX IF NOT EXISTS idx_micro_macro ON micro_events(macro_id);
CREATE INDEX IF NOT EXISTS idx_micro_super ON micro_events(superseded_by);

CREATE TABLE IF NOT EXISTS macro_events (
    id           TEXT PRIMARY KEY,
    t_start      REAL NOT NULL,
    t_end        REAL NOT NULL,
    label        TEXT,
    summary      TEXT NOT NULL,
    super_id     TEXT,
    key_entities TEXT,
    created_at   REAL NOT NULL,
    -- ★ E3 (evolve): 叙事弧 + 实体弧 (带图聚合时由 LLM 生成)
    narrative_arc TEXT,           -- JSON list of {"phase","t","desc"}, setup/rising/climax/resolution
    entity_arcs  TEXT,            -- JSON dict {entity_name: [phase desc, ...]}
    -- ★ E4: 软删除 + 修订追溯
    superseded_by TEXT,
    revised_at   REAL,
    revision_count INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_macro_t ON macro_events(t_end);

CREATE TABLE IF NOT EXISTS super_events (
    id           TEXT PRIMARY KEY,
    t_start      REAL NOT NULL,
    t_end        REAL NOT NULL,
    label        TEXT,
    description  TEXT NOT NULL,
    macro_ids    TEXT,
    is_root      INTEGER DEFAULT 0,
    created_at   REAL NOT NULL,
    -- ★ E3 (evolve): super 也有 narrative_arc (更高层叙事)
    narrative_arc TEXT,
    superseded_by TEXT,
    revised_at   REAL,
    revision_count INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_super_t ON super_events(t_end);

CREATE TABLE IF NOT EXISTS entities (
    id                     TEXT PRIMARY KEY,
    name                   TEXT NOT NULL,
    type                   TEXT NOT NULL,
    attributes             TEXT NOT NULL,
    aliases                TEXT,
    first_seen             REAL NOT NULL,
    last_seen              REAL NOT NULL,
    seen_count             INTEGER DEFAULT 1,
    representative_frame_id TEXT,
    updated_at             REAL NOT NULL,
    -- ★ E4 (evolve): 软删除 (被 merge_entities 时指向 winner; split 时指向第一个子 entity)
    merged_into            TEXT,
    revised_at             REAL,
    revision_count         INTEGER DEFAULT 0,
    -- ★ Hybrid retrieval (一期): name+type+attrs+aliases 的文本 embedding (float16 BLOB).
    --   Writer upsert 后 / Reviewer merge/refine 后异步更新;
    --   NULL 表示尚未算.
    text_embedding         BLOB
);
CREATE INDEX IF NOT EXISTS idx_ent_merged ON entities(merged_into);
CREATE INDEX IF NOT EXISTS idx_ent_name        ON entities(name);
CREATE INDEX IF NOT EXISTS idx_ent_first_seen  ON entities(first_seen);
CREATE INDEX IF NOT EXISTS idx_ent_last_seen   ON entities(last_seen);

-- ★ object(entity) ←→ event(micro) 多对多关联表 (用户构想: 一个 object 可出现在
--   多个事件, 一个事件可含多个 object). finalize_micro 时按"该 micro 时间窗内
--   upsert 过的 entity 都算"(方案a) 批量建联. 查 object 的帧 = 经此表找到 events
--   → 取各 event 的 frame_ids.
CREATE TABLE IF NOT EXISTS entity_event (
    entity_id    TEXT NOT NULL,
    micro_id     TEXT NOT NULL,
    t_observed   REAL NOT NULL,
    PRIMARY KEY (entity_id, micro_id)
);
CREATE INDEX IF NOT EXISTS idx_ee_entity ON entity_event(entity_id);
CREATE INDEX IF NOT EXISTS idx_ee_micro  ON entity_event(micro_id);

-- ★ object(entity) ←→ frame 帧级精确关联表. 跟 entity_event(段级) 互补:
--   entity_event 记 "object 出现在哪个 micro 段", entity_frame 记 "object 清晰出现在
--   具体哪张关键帧". 由 LLM 每拍 key_frames[i].entities 显式指认 → 修复旧版
--   "本拍所有 object 都绑同一张第一帧(tick_fids[0])" 的对不上问题. 一个 object 可挂多帧.
CREATE TABLE IF NOT EXISTS entity_frame (
    entity_id    TEXT NOT NULL,
    frame_id     TEXT NOT NULL,
    micro_id     TEXT,
    t_observed   REAL NOT NULL,
    PRIMARY KEY (entity_id, frame_id)
);
CREATE INDEX IF NOT EXISTS idx_ef2_entity ON entity_frame(entity_id);
CREATE INDEX IF NOT EXISTS idx_ef2_frame  ON entity_frame(frame_id);

CREATE TABLE IF NOT EXISTS edges (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    src_id       TEXT NOT NULL,
    dst_id       TEXT NOT NULL,
    label        TEXT NOT NULL,
    rel_type     TEXT NOT NULL,
    micro_id     TEXT,
    t_observed   REAL NOT NULL,
    metadata     TEXT
);
CREATE INDEX IF NOT EXISTS idx_edge_src ON edges(src_id);
CREATE INDEX IF NOT EXISTS idx_edge_dst ON edges(dst_id);
CREATE INDEX IF NOT EXISTS idx_edge_t   ON edges(t_observed);

-- ★ E1 (evolve): entity 演化时间线
--   双轨制: entities 表存"当前最权威态" (快查), entity_states 存"每次变化的轨迹".
--   写入时机:
--     Writer: 每次 upsert_entity 命中合并 → append 一条 (state_label="refined",
--             attributes_delta = 这次新发现的属性 diff).
--     Writer: 新建 entity → append 一条 (state_label="first_seen").
--     Reviewer: refine_entity / merge_entities / split_entity 时 → append 对应 state_label.
CREATE TABLE IF NOT EXISTS entity_states (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id          TEXT NOT NULL,
    t_observed         REAL NOT NULL,
    state_label        TEXT NOT NULL,      -- first_seen | refined | merged_into | split_from | left_scene | identified
    attributes_delta   TEXT,               -- JSON dict: 本次新发现/变化的属性
    new_aliases        TEXT,               -- JSON list: 本次新增的别名
    confidence         REAL DEFAULT 1.0,
    evidence_frame_ids TEXT,               -- JSON list of frame_id
    micro_id           TEXT,
    source             TEXT NOT NULL,      -- "writer" | "reviewer"
    note               TEXT
);
CREATE INDEX IF NOT EXISTS idx_es_entity ON entity_states(entity_id);
CREATE INDEX IF NOT EXISTS idx_es_t      ON entity_states(t_observed);

-- ★ E2 (evolve): 审校动作日志 (Reviewer 修订审计 + UI Revision Log 数据源)
--   每次 Reviewer 落地一个 action 都追加一条. 支持回溯 "这条 micro 什么时候被改的, 为什么".
CREATE TABLE IF NOT EXISTS revision_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    t_applied       REAL NOT NULL,
    reviewer_round  INTEGER,                -- 第几轮 wake (累计)
    op              TEXT NOT NULL,          -- merge_micros | split_micro | revise_micro_desc | merge_entities | split_entity | refine_entity | rewrite_macro_summary
    target_ids      TEXT NOT NULL,          -- JSON list 被影响的 id
    new_ids         TEXT,                   -- JSON list 新产生的 id (merge/split 才有)
    payload         TEXT,                   -- JSON: action 详细参数 (新 desc / 拆分点 / 新属性 等)
    reason          TEXT,                   -- LLM 的修订理由
    success         INTEGER DEFAULT 1,
    error           TEXT,
    actor           TEXT DEFAULT 'reviewer' -- reviewer | writer (Writer 也可能加 revision_log 比如 entity 合并)
);
CREATE INDEX IF NOT EXISTS idx_rev_t   ON revision_log(t_applied);
CREATE INDEX IF NOT EXISTS idx_rev_op  ON revision_log(op);

-- ★ OMNI-Q: PERSON 说的话 (Writer omni 从原始音频直接产出)
--   text 是转写内容, 归属到某个 PERSON entity_id (含 ent_unknown_speaker).
--   Reviewer 后续 remap 时不 UPDATE 老行, insert 新行并把老行 superseded_by 指向新 id.
CREATE TABLE IF NOT EXISTS entity_quotes (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id          TEXT NOT NULL,        -- 必须是 PERSON 类型 entity (含 ent_unknown_speaker)
    t_start            REAL NOT NULL,
    t_end              REAL NOT NULL,
    text               TEXT NOT NULL,        -- omni 转写的对白文本 (简体中文)
    confidence         REAL DEFAULT 1.0,     -- omni 对 speaker 归属的把握 (0~1)
    evidence_frame_ids TEXT,                 -- JSON list: 嘴部张合那几帧的 frame_id
    micro_id           TEXT,                 -- 关联 micro_event
    macro_id           TEXT,                 -- 关联 macro_event (聚合时回填)
    source             TEXT DEFAULT 'omni',  -- omni | reviewer_remap | manual
    created_at         REAL NOT NULL,
    superseded_by      INTEGER               -- Reviewer 重新归属 entity 时指向新 quote id
);
CREATE INDEX IF NOT EXISTS idx_quote_entity ON entity_quotes(entity_id);
CREATE INDEX IF NOT EXISTS idx_quote_t      ON entity_quotes(t_start);
CREATE INDEX IF NOT EXISTS idx_quote_micro  ON entity_quotes(micro_id);

-- Raw ASR observations are durable and independent of ConversationLog's RAM
-- limits. Recall reads this table first, so hour-long sessions do not lose the
-- earliest transcript when conv_max_audio_obs / conv_max_chars trims the log.
CREATE TABLE IF NOT EXISTS audio_observations (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    t_observed   REAL,
    wall_ts      REAL NOT NULL,
    speaker      TEXT,
    text         TEXT NOT NULL,
    source       TEXT DEFAULT 'asr',
    created_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audio_obs_t ON audio_observations(t_observed);
CREATE INDEX IF NOT EXISTS idx_audio_obs_wall ON audio_observations(wall_ts);

-- ★ OMNI-R: Entity 多代表帧 (替代 entities.representative_frame_id 单帧)
--   每个 PERSON entity 最多保留 top-K (cfg.entity_rep_frames_max_n), �� quality DESC.
--   老�� representative_frame_id 字段保留, 写入时同步取 top-1 帧 fid 给老 Recall 兜底.
CREATE TABLE IF NOT EXISTS entity_rep_frames (
    entity_id     TEXT NOT NULL,
    frame_id      TEXT NOT NULL,            -- 引用 FrameStore
    quality_score REAL DEFAULT 0.5,         -- omni 打分 (0~1): "这帧多清晰能认出此人"
    note          TEXT,                     -- "正脸特写"/"侧脸全身"/"远景" 等
    added_at      REAL NOT NULL,
    added_by      TEXT DEFAULT 'writer',    -- writer | reviewer
    PRIMARY KEY (entity_id, frame_id)
);
CREATE INDEX IF NOT EXISTS idx_rep_entity  ON entity_rep_frames(entity_id);
CREATE INDEX IF NOT EXISTS idx_rep_quality ON entity_rep_frames(entity_id, quality_score DESC);

-- ★ 二期 (frame image embedding): 关键帧的图像向量 (multimodal-embedding-v1,
--   float16 BLOB 1024 维 = 2KB). 与 micro/entity 的 text_embedding 是**独立语义
--   空间**, 仅用于 T→I 跨模态检索 (search_frames_by_text).
--   写入时机: Writer 拍级 maybe_store 成功后异步算 (FrameStore 是纯内存 LRU,
--   帧被淘汰后无法补算, 所以必须落盘即刻算). 与帧生命周期解耦: 即使 JPEG 已被
--   LRU 淘汰, 向量仍可在召回侧提供"该帧曾匹配"的语义线索.
CREATE TABLE IF NOT EXISTS frame_embeddings (
    frame_id     TEXT PRIMARY KEY,
    t_observed   REAL NOT NULL,             -- 帧 ts (D3 防脏读 + UI 展示)
    micro_id     TEXT,                      -- 关联 micro (join 拿上下文描述)
    embedding    BLOB NOT NULL,
    created_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_femb_t ON frame_embeddings(t_observed);
"""


@dataclass
class MicroEvent:
    id: str
    t_start: float
    t_end: float
    description: str
    subject: str = ""
    object: str = ""
    action: str = ""
    macro_id: Optional[str] = None
    facts_keys: List[str] = field(default_factory=list)
    frame_ids: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    # ★ E4 (evolve): 软删除 + 修订追溯
    superseded_by: Optional[str] = None
    revised_at: Optional[float] = None
    revision_count: int = 0


@dataclass
class MacroEvent:
    id: str
    t_start: float
    t_end: float
    label: str
    summary: str
    super_id: Optional[str] = None
    key_entities: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    # ★ E3 (evolve): 叙事弧 / 实体弧
    narrative_arc: List[Dict[str, Any]] = field(default_factory=list)
    entity_arcs: Dict[str, List[str]] = field(default_factory=dict)
    # ★ E4: 软删除 + 修订追溯
    superseded_by: Optional[str] = None
    revised_at: Optional[float] = None
    revision_count: int = 0


@dataclass
class SuperEvent:
    id: str
    t_start: float
    t_end: float
    label: str
    description: str
    macro_ids: List[str] = field(default_factory=list)
    is_root: bool = False
    created_at: float = field(default_factory=time.time)
    narrative_arc: List[Dict[str, Any]] = field(default_factory=list)
    superseded_by: Optional[str] = None
    revised_at: Optional[float] = None
    revision_count: int = 0


# ★ E1 (evolve): entity 演化时间线一行的 dataclass
@dataclass
class EntityState:
    id: int = 0                              # AUTOINCREMENT, 写入后回填
    entity_id: str = ""
    t_observed: float = 0.0
    state_label: str = "refined"             # first_seen | refined | identified | merged_into | split_from | left_scene
    attributes_delta: Dict[str, str] = field(default_factory=dict)
    new_aliases: List[str] = field(default_factory=list)
    confidence: float = 1.0
    evidence_frame_ids: List[str] = field(default_factory=list)
    micro_id: Optional[str] = None
    source: str = "writer"                   # writer | reviewer
    note: str = ""


# ★ E2 (evolve): Reviewer 修订动作审计的 dataclass
@dataclass
class RevisionRecord:
    id: int = 0
    t_applied: float = 0.0
    reviewer_round: int = 0
    op: str = ""                             # merge_micros | split_micro | ...
    target_ids: List[str] = field(default_factory=list)
    new_ids: List[str] = field(default_factory=list)
    payload: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    success: bool = True
    error: str = ""
    actor: str = "reviewer"


@dataclass
class Entity:
    id: str
    name: str
    type: str
    attributes: Dict[str, str] = field(default_factory=dict)
    aliases: List[str] = field(default_factory=list)
    first_seen: float = 0.0
    last_seen: float = 0.0
    seen_count: int = 1
    representative_frame_id: str = ""   # ★ object 的代表帧 (整帧, crop 交给 search)
    updated_at: float = field(default_factory=time.time)
    # ★ E4 (evolve): 软删除 + 修订追溯 (entity 被 merge_entities 时 merged_into 指向 winner)
    merged_into: Optional[str] = None
    revised_at: Optional[float] = None
    revision_count: int = 0


@dataclass
class Edge:
    src_id: str
    dst_id: str
    label: str
    rel_type: str          # 'spatial' | 'subject_object' | 'temporal_causal'
    micro_id: Optional[str] = None
    t_observed: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ★ OMNI-Q: PERSON 对白 dataclass (对应 entity_quotes 表)
@dataclass
class EntityQuote:
    id: int = 0                              # AUTOINCREMENT, 写入后回填
    entity_id: str = ""                      # 必须非空, 至少是 ent_unknown_speaker
    t_start: float = 0.0
    t_end:   float = 0.0
    text: str = ""
    confidence: float = 1.0
    evidence_frame_ids: List[str] = field(default_factory=list)
    micro_id: Optional[str] = None
    macro_id: Optional[str] = None
    source: str = "omni"                     # omni | reviewer_remap | manual
    created_at: float = field(default_factory=time.time)
    superseded_by: Optional[int] = None      # Reviewer 改归属时指向新 quote.id


# ★ OMNI-R: Entity 代表帧 dataclass (对应 entity_rep_frames 表)
@dataclass
class RepFrame:
    entity_id: str = ""
    frame_id:  str = ""
    quality_score: float = 0.5
    note: str = ""
    added_at: float = field(default_factory=time.time)
    added_by: str = "writer"


@dataclass
class ScreenTextBlock:
    text: str = ""
    bbox: List[float] = field(default_factory=list)
    confidence: float = 1.0
    region_type: str = "unknown"


@dataclass
class ScreenTextRecord:
    frame_id: str
    t_observed: float
    app: str = ""
    window_title: str = ""
    ocr_blocks: List[ScreenTextBlock] = field(default_factory=list)
    raw_text: str = ""
    source: str = "writer_vlm"
    created_at: float = field(default_factory=time.time)


@dataclass
class ScreenTableRecord:
    table_id: str
    frame_id: str
    t_observed: float
    title: str = ""
    columns: List[str] = field(default_factory=list)
    rows: List[Any] = field(default_factory=list)
    app: str = ""
    window_title: str = ""
    raw_text: str = ""
    source: str = "writer_vlm"
    confidence: float = 1.0
    created_at: float = field(default_factory=time.time)


@dataclass
class TaskStateRecord:
    id: int = 0
    task_id: str = ""
    t_observed: float = 0.0
    active_task: str = ""
    goal: str = ""
    current_artifact: str = ""
    open_questions: List[str] = field(default_factory=list)
    decisions: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)
    next_actions: List[str] = field(default_factory=list)
    evidence_frame_ids: List[str] = field(default_factory=list)
    source: str = "writer"
    confidence: float = 1.0
    raw: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


def _apply_conn_pragmas(c: sqlite3.Connection, *, set_wal: bool = False) -> None:
    """Per-connection PRAGMA setup. MUST be called on every new connection.

    ``synchronous`` is a per-connection setting that is NOT persisted in the
    database file, so setting it once during schema creation or migration only
    affects that one short-lived connection -- every later connection silently
    falls back to the default. Under WAL that default is FULL, i.e. an fsync on
    every single commit. This DB holds rebuildable observation memory written by
    a 10s-interval writer, so per-commit fsync is not worth its cost; NORMAL
    only fsyncs at checkpoints.

    ``busy_timeout`` is set here as an explicit guard only. Python's
    ``sqlite3.connect()`` already defaults to ``timeout=5.0``, which maps to
    busy_timeout=5000ms, so this line is a no-op under the default -- it exists
    so that behaviour stays pinned if a caller ever passes ``timeout=0``. (The
    four pre-existing ``PRAGMA busy_timeout=5000`` calls elsewhere in this file
    are no-ops for the same reason.)

    ``journal_mode`` IS persisted at the database level, so it only needs to be
    set once per process (pass ``set_wal=True`` on the first connection) as a
    safety net in case this store is the first thing to open a fresh DB file.
    """
    try:
        c.execute("PRAGMA synchronous=NORMAL")
        c.execute("PRAGMA busy_timeout=5000")
        if set_wal:
            c.execute("PRAGMA journal_mode=WAL")
    except sqlite3.Error as e:
        log.warning("[memory] PRAGMA setup failed: %s", e)


class _DesktopSQLiteMixin:
    """Small helper for desktop-memory side stores sharing MemoryStore's DB."""

    def __init__(self, cfg: Config, *, db_path: Optional[str] = None):
        self.cfg = cfg
        self.db_path = db_path or cfg.mem_db_path
        if not self.db_path:
            tmp = tempfile.NamedTemporaryFile(
                prefix="tml_desktop_mem_", suffix=".sqlite", delete=False,
            )
            tmp.close()
            self.db_path = tmp.name
        parent = os.path.dirname(os.path.abspath(self.db_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._lock = threading.RLock()
        self._wal_ready = False

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        _apply_conn_pragmas(c, set_wal=not self._wal_ready)
        self._wal_ready = True
        return c

    @contextmanager
    def _connect(self):
        c = self._conn()
        try:
            yield c
            c.commit()
        except Exception:
            try:
                c.rollback()
            except Exception:
                pass
            raise
        finally:
            try:
                c.close()
            except Exception:
                pass


def _json_loads_safe(s: Optional[str], default: Any) -> Any:
    if not s:
        return default
    try:
        return json.loads(s)
    except Exception:
        return default


class ScreenTextStore(_DesktopSQLiteMixin):
    """OCR/text layer for desktop sharing frames.

    The store is intentionally provider-neutral. A real OCR/window-title
    extractor can call upsert_frame_text directly; until then MemoryWriter can
    persist VLM-extracted text evidence with source='writer_vlm'.
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS screen_texts (
        frame_id     TEXT PRIMARY KEY,
        t_observed   REAL NOT NULL,
        app          TEXT,
        window_title TEXT,
        ocr_blocks   TEXT,
        raw_text     TEXT,
        source       TEXT,
        created_at   REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_screen_text_t
        ON screen_texts(t_observed);
    CREATE INDEX IF NOT EXISTS idx_screen_text_app
        ON screen_texts(app);
    """

    def __init__(self, cfg: Config, *, db_path: Optional[str] = None):
        super().__init__(cfg, db_path=db_path)
        with self._lock, self._connect() as c:
            c.executescript(self._SCHEMA)
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA busy_timeout=5000")

    @staticmethod
    def normalize_blocks(raw_blocks: Any, raw_text: str = "") -> List[ScreenTextBlock]:
        blocks: List[ScreenTextBlock] = []
        if isinstance(raw_blocks, list):
            for b in raw_blocks:
                if isinstance(b, str):
                    txt = b.strip()
                    if txt:
                        blocks.append(ScreenTextBlock(text=txt))
                    continue
                if not isinstance(b, dict):
                    continue
                txt = str(b.get("text", "") or "").strip()
                if not txt:
                    continue
                bbox_raw = b.get("bbox") or []
                bbox: List[float] = []
                if isinstance(bbox_raw, list):
                    for v in bbox_raw[:4]:
                        try:
                            bbox.append(float(v))
                        except (TypeError, ValueError):
                            pass
                try:
                    conf = float(b.get("confidence", 1.0) or 1.0)
                except (TypeError, ValueError):
                    conf = 1.0
                region = str(b.get("region_type", "") or "unknown").strip()
                blocks.append(ScreenTextBlock(
                    text=txt, bbox=bbox, confidence=conf, region_type=region))
        elif raw_text:
            for line in str(raw_text).splitlines():
                txt = line.strip()
                if txt:
                    blocks.append(ScreenTextBlock(text=txt))
        return blocks

    def upsert_frame_text(self, rec: ScreenTextRecord) -> bool:
        if not rec.frame_id:
            return False
        raw_text = (rec.raw_text or "\n".join(
            b.text for b in rec.ocr_blocks if b.text)).strip()
        blocks_json = json.dumps([
            {
                "text": b.text,
                "bbox": b.bbox,
                "confidence": b.confidence,
                "region_type": b.region_type,
            }
            for b in rec.ocr_blocks
        ], ensure_ascii=False)
        with self._lock, self._connect() as c:
            c.execute(
                """INSERT INTO screen_texts
                   (frame_id, t_observed, app, window_title, ocr_blocks,
                    raw_text, source, created_at)
                   VALUES (?,?,?,?,?,?,?,?)
                   ON CONFLICT(frame_id) DO UPDATE SET
                     t_observed=excluded.t_observed,
                     app=COALESCE(NULLIF(excluded.app,''), screen_texts.app),
                     window_title=COALESCE(NULLIF(excluded.window_title,''), screen_texts.window_title),
                     ocr_blocks=CASE
                       WHEN screen_texts.source LIKE 'ocr:%'
                            AND excluded.source='writer_vlm'
                       THEN screen_texts.ocr_blocks
                       ELSE excluded.ocr_blocks
                     END,
                     raw_text=CASE
                       WHEN screen_texts.source LIKE 'ocr:%'
                            AND excluded.source='writer_vlm'
                       THEN screen_texts.raw_text
                       ELSE excluded.raw_text
                     END,
                     source=CASE
                       WHEN screen_texts.source LIKE 'ocr:%'
                            AND excluded.source='writer_vlm'
                       THEN screen_texts.source
                       ELSE excluded.source
                     END,
                     created_at=excluded.created_at""",
                (rec.frame_id, float(rec.t_observed or 0.0),
                 rec.app or "", rec.window_title or "", blocks_json,
                 raw_text, rec.source or "ocr", rec.created_at or time.time()),
            )
        return True

    def get_by_frame_ids(self, frame_ids: List[str]) -> List[ScreenTextRecord]:
        ids = [str(fid).strip() for fid in frame_ids if str(fid).strip()]
        if not ids:
            return []
        qmarks = ",".join("?" for _ in ids)
        with self._connect() as c:
            rows = c.execute(
                f"SELECT * FROM screen_texts WHERE frame_id IN ({qmarks})",
                ids,
            ).fetchall()
        by_id = {r["frame_id"]: self._row_to_record(r) for r in rows}
        return [by_id[fid] for fid in ids if fid in by_id]

    def search(
        self, query: str, ask_ts: float, *,
        t_window: Optional[Tuple[float, float]] = None,
        app: str = "",
        limit: int = 10,
    ) -> List[ScreenTextRecord]:
        raw_limit = max(1, int(limit or 10))
        doc_refs = mm_doc_ref_terms(query)
        terms = mm_expand_terms(mm_tokenize_query(query))
        identifier_terms = mm_identifier_variants(query, terms)
        weak_terms = {
            "table", "figure", "fig", "paper", "pdf", "slide",
            "method", "methods", "model", "models", "benchmark",
            "benchmarks", "comparison", "result", "results", "main",
            "row", "rows", "column", "columns", "accuracy", "acc",
            "刚才", "之前", "论文", "表格", "图表", "方法", "模型",
        }
        # Single digits from "Table 1" are poisonous as LIKE terms: they match
        # almost every paper page. Keep explicit anchors as phrases instead.
        meaningful_terms = [
            t for t in terms
            if len(str(t)) > 1 and str(t).lower() not in weak_terms
        ]
        sql_terms: List[str] = []
        for term in doc_refs + identifier_terms + meaningful_terms:
            if term not in sql_terms:
                sql_terms.append(term)
        if not sql_terms:
            sql_terms = [t for t in terms if len(str(t)) > 1][:12]
        where = ["t_observed <= ?"]
        args: List[Any] = [ask_ts]
        if t_window:
            where.append("t_observed >= ? AND t_observed <= ?")
            args.extend([float(t_window[0]), min(float(t_window[1]), ask_ts)])
        if app:
            where.append("LOWER(app) LIKE ?")
            args.append(f"%{app.lower()}%")
        if sql_terms:
            or_parts: List[str] = []
            for term in sql_terms[:16]:
                pat = f"%{term}%"
                or_parts.append(
                    "(LOWER(raw_text) LIKE ? OR LOWER(window_title) LIKE ? OR LOWER(app) LIKE ?)")
                args.extend([pat, pat, pat])
            where.append("(" + " OR ".join(or_parts) + ")")
        # A real query must rank the complete matching corpus.  The previous
        # `ORDER BY time DESC LIMIT <=250` candidate gate made an old exact OCR
        # match unreachable whenever enough newer weak OR-term matches existed.
        # Empty-query callers only want a recent time slice and keep the limit.
        sql = (f"SELECT * FROM screen_texts WHERE {' AND '.join(where)} "
               "ORDER BY t_observed DESC")
        if not query:
            sql += " LIMIT ?"
            args.append(raw_limit)
        with self._connect() as c:
            rows = c.execute(sql, args).fetchall()
        records = [self._row_to_record(r) for r in rows]
        if not records or not query:
            return records[:raw_limit]

        query_l = str(query or "").lower()
        query_norm = re.sub(r"[^a-z0-9]+", "", query_l)
        table_like = bool(re.search(
            r"(?<![A-Za-z])(?:table|fig(?:ure)?)\s*\.?\d*(?![A-Za-z])|"
            r"表\s*\d*|图\s*\d*|benchmark|benchmarks|row|rows|column|columns",
            query_l, re.IGNORECASE))

        def _contains_phrase(text_l: str, text_norm: str, phrase: str) -> bool:
            p = str(phrase or "").lower()
            if not p:
                return False
            if p in text_l:
                return True
            p_norm = re.sub(r"[^a-z0-9]+", "", p)
            return bool(len(p_norm) >= 3 and p_norm in text_norm)

        def _score(rec: ScreenTextRecord) -> Tuple[float, float]:
            text = "\n".join([
                rec.raw_text or "",
                rec.window_title or "",
                rec.app or "",
            ])
            text_l = text.lower()
            text_norm = re.sub(r"[^a-z0-9]+", "", text_l)
            score = 0.0
            ref_hits = 0
            for ref in doc_refs:
                if _contains_phrase(text_l, text_norm, ref):
                    ref_hits += 1
            if doc_refs:
                score += ref_hits * 1000.0
                if ref_hits == 0:
                    score -= 500.0
            for term in (identifier_terms + meaningful_terms)[:32]:
                term_l = str(term).lower()
                if _contains_phrase(text_l, text_norm, term_l):
                    if term_l in identifier_terms:
                        score += 120.0 + min(len(term_l), 24)
                    else:
                        score += 20.0 + min(len(term_l), 24)
            if table_like and re.search(
                    r"\b(table|figure|fig|method|model|benchmark|accuracy|acc)\b|"
                    r"表\s*\d+|图\s*\d+",
                    text_l, re.IGNORECASE):
                score += 40.0
            if query_norm and len(query_norm) >= 8 and query_norm in text_norm:
                score += 120.0
            # Recency is a tie-breaker, not the primary signal for explicit doc refs.
            age = max(0.0, float(ask_ts) - float(rec.t_observed or 0.0))
            recency = max(0.0, 20.0 - age / 30.0)
            return score + recency, float(rec.t_observed or 0.0)

        ranked = sorted(records, key=_score, reverse=True)
        if doc_refs:
            ref_ranked = [
                r for r in ranked
                if any(_contains_phrase(
                    (r.raw_text + "\n" + r.window_title + "\n" + r.app).lower(),
                    re.sub(r"[^a-z0-9]+", "",
                           (r.raw_text + "\n" + r.window_title + "\n" + r.app).lower()),
                    ref)
                    for ref in doc_refs)
            ]
            if ref_ranked:
                ranked = ref_ranked + [r for r in ranked if r not in ref_ranked]
        return ranked[:raw_limit]

    @staticmethod
    def _row_to_record(r: sqlite3.Row) -> ScreenTextRecord:
        blocks_raw = _json_loads_safe(r["ocr_blocks"], [])
        blocks = ScreenTextStore.normalize_blocks(blocks_raw)
        return ScreenTextRecord(
            frame_id=r["frame_id"],
            t_observed=float(r["t_observed"] or 0.0),
            app=r["app"] or "",
            window_title=r["window_title"] or "",
            ocr_blocks=blocks,
            raw_text=r["raw_text"] or "",
            source=r["source"] or "ocr",
            created_at=float(r["created_at"] or 0.0),
        )


class ScreenTableStore(_DesktopSQLiteMixin):
    """Structured table memory for text-heavy desktop/PDF frames.

    ``screen_texts`` remains the source of truth for raw OCR. This table stores
    the MemoryWriter's parsed table view so recall can answer row/column
    questions without asking an LLM to rediscover table structure from prose.
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS screen_tables (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        table_id     TEXT NOT NULL,
        frame_id     TEXT NOT NULL,
        t_observed   REAL NOT NULL,
        title        TEXT,
        columns_json TEXT,
        rows_json    TEXT,
        app          TEXT,
        window_title TEXT,
        raw_text     TEXT,
        source       TEXT,
        confidence   REAL DEFAULT 1.0,
        created_at   REAL NOT NULL,
        UNIQUE(frame_id, table_id)
    );
    CREATE INDEX IF NOT EXISTS idx_screen_table_t
        ON screen_tables(t_observed);
    CREATE INDEX IF NOT EXISTS idx_screen_table_id
        ON screen_tables(table_id);
    """

    def __init__(self, cfg: Config, *, db_path: Optional[str] = None):
        super().__init__(cfg, db_path=db_path)
        with self._lock, self._connect() as c:
            c.executescript(self._SCHEMA)
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA busy_timeout=5000")

    @staticmethod
    def _norm_alnum(raw: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(raw or "").lower())

    @staticmethod
    def _table_ref_terms(query: str) -> List[str]:
        terms: List[str] = []

        def _add(*vals: str) -> None:
            for val in vals:
                val = str(val or "").strip().lower()
                if val and val not in terms:
                    terms.append(val)

        q = str(query or "")
        for m in re.finditer(
                r"(?<![A-Za-z])(?:table|fig(?:ure)?)\s*\.?\s*(\d+)(?![A-Za-z])",
                q, re.IGNORECASE):
            kind = "figure" if m.group(0).lower().startswith(("fig", "figure")) else "table"
            n = m.group(1)
            _add(f"{kind}{n}", f"{kind} {n}", f"{kind}.{n}")
            if kind == "table":
                _add(f"表{n}", f"表 {n}")
            else:
                _add(f"图{n}", f"图 {n}", f"fig {n}", f"fig.{n}")
        for m in re.finditer(r"(表|图)\s*(\d+)", q, re.IGNORECASE):
            prefix, n = m.group(1), m.group(2)
            if prefix == "表":
                _add(f"表{n}", f"表 {n}", f"table{n}", f"table {n}")
            else:
                _add(f"图{n}", f"图 {n}", f"figure{n}", f"figure {n}", f"fig{n}", f"fig {n}")
        return terms

    @staticmethod
    def _clean_str_list(raw: Any, *, max_n: int = 80) -> List[str]:
        out: List[str] = []
        if isinstance(raw, str):
            raw = [raw]
        if isinstance(raw, list):
            for item in raw[:max_n]:
                text = str(item or "").strip()
                if text and text not in out:
                    out.append(text)
        return out

    @staticmethod
    def _clean_cell(raw: Any) -> str:
        if raw in (None, [], {}):
            return ""
        if isinstance(raw, (dict, list)):
            return json.dumps(raw, ensure_ascii=False)
        return str(raw).strip()

    @classmethod
    def normalize_rows(
        cls, raw: Any, *, columns: Optional[List[str]] = None,
        max_n: int = 80,
    ) -> List[Any]:
        rows: List[Any] = []
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, list):
            return rows
        cols = cls._clean_str_list(columns or [], max_n=120)
        for item in raw[:max_n]:
            if isinstance(item, dict):
                source = {
                    str(k).strip(): cls._clean_cell(v)
                    for k, v in item.items()
                    if str(k).strip()
                }
                if cols:
                    cleaned = {col: source.get(col, "") for col in cols}
                    for k, v in source.items():
                        if k not in cleaned and v:
                            cleaned[k] = v
                else:
                    cleaned = {k: v for k, v in source.items() if v}
                if cleaned:
                    rows.append(cleaned)
                continue
            if isinstance(item, list):
                vals = [cls._clean_cell(v) for v in item[:120]]
                while vals and not vals[-1]:
                    vals.pop()
                if vals:
                    if cols:
                        mapped = {
                            col: (vals[i] if i < len(vals) else "")
                            for i, col in enumerate(cols)
                        }
                        if len(vals) > len(cols):
                            mapped["_extra_cells"] = " | ".join(vals[len(cols):])
                        rows.append(mapped)
                    else:
                        rows.append(vals)
                continue
            text = str(item or "").strip()
            if text:
                if cols:
                    row = {col: "" for col in cols}
                    row["_raw"] = text
                    rows.append(row)
                else:
                    rows.append(text)
        return rows

    def upsert_table(self, rec: ScreenTableRecord) -> bool:
        table_id = (rec.table_id or "").strip()
        frame_id = (rec.frame_id or "").strip()
        if not table_id or not frame_id:
            return False
        columns = self._clean_str_list(rec.columns)
        rows = self.normalize_rows(rec.rows, columns=columns)
        raw_text = (rec.raw_text or "").strip()
        if not (rec.title or columns or rows or raw_text):
            return False
        with self._lock, self._connect() as c:
            c.execute(
                """INSERT INTO screen_tables
                   (table_id, frame_id, t_observed, title, columns_json,
                    rows_json, app, window_title, raw_text, source,
                    confidence, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(frame_id, table_id) DO UPDATE SET
                     t_observed=excluded.t_observed,
                     title=COALESCE(NULLIF(excluded.title,''), screen_tables.title),
                     columns_json=CASE
                       WHEN excluded.columns_json='[]'
                       THEN screen_tables.columns_json ELSE excluded.columns_json END,
                     rows_json=CASE
                       WHEN excluded.rows_json='[]'
                       THEN screen_tables.rows_json ELSE excluded.rows_json END,
                     app=COALESCE(NULLIF(excluded.app,''), screen_tables.app),
                     window_title=COALESCE(NULLIF(excluded.window_title,''), screen_tables.window_title),
                     raw_text=COALESCE(NULLIF(excluded.raw_text,''), screen_tables.raw_text),
                     source=excluded.source,
                     confidence=excluded.confidence,
                     created_at=excluded.created_at""",
                (
                    table_id, frame_id, float(rec.t_observed or 0.0),
                    rec.title or "",
                    json.dumps(columns, ensure_ascii=False),
                    json.dumps(rows, ensure_ascii=False),
                    rec.app or "", rec.window_title or "", raw_text,
                    rec.source or "writer_vlm",
                    float(rec.confidence if rec.confidence is not None else 1.0),
                    rec.created_at or time.time(),
                ),
            )
        return True

    def search(
        self, query: str, ask_ts: float, *,
        t_window: Optional[Tuple[float, float]] = None,
        limit: int = 10,
    ) -> List[ScreenTableRecord]:
        terms = mm_expand_terms(mm_tokenize_query(query))
        for term in self._table_ref_terms(query):
            if term not in terms:
                terms.append(term)
        where = ["t_observed <= ?"]
        args: List[Any] = [ask_ts]
        if t_window:
            where.append("t_observed >= ? AND t_observed <= ?")
            args.extend([float(t_window[0]), min(float(t_window[1]), ask_ts)])
        if terms:
            or_parts: List[str] = []
            for term in terms[:12]:
                pat = f"%{term}%"
                or_parts.append(
                    "(LOWER(table_id) LIKE ? OR LOWER(title) LIKE ? "
                    "OR LOWER(columns_json) LIKE ? OR LOWER(rows_json) LIKE ? "
                    "OR LOWER(raw_text) LIKE ?)")
                args.extend([pat, pat, pat, pat, pat])
            where.append("(" + " OR ".join(or_parts) + ")")
        # As with raw OCR, score all global matches before truncating.  Recent
        # weak table hits must not hide an old exact table id/cell match.
        sql = (f"SELECT * FROM screen_tables WHERE {' AND '.join(where)} "
               "ORDER BY t_observed DESC")
        if not query:
            sql += " LIMIT ?"
            args.append(max(1, int(limit or 10)))
        with self._connect() as c:
            rows = c.execute(sql, args).fetchall()
        records = [self._row_to_record(r) for r in rows]
        if not terms:
            return records[:max(1, int(limit or 10))]

        table_refs = self._table_ref_terms(query)

        def _score(rec: ScreenTableRecord) -> Tuple[int, float]:
            table_id = (rec.table_id or "").lower()
            title = (rec.title or "").lower()
            hay = " ".join([
                table_id, title, json.dumps(rec.columns, ensure_ascii=False).lower(),
                json.dumps(rec.rows, ensure_ascii=False).lower(),
                (rec.raw_text or "").lower(),
            ])
            hay_norm = self._norm_alnum(hay)
            score = 0
            for term in terms[:24]:
                low = term.lower()
                norm = self._norm_alnum(low)
                if not low:
                    continue
                if low in table_id:
                    score += 120
                elif norm and norm == self._norm_alnum(table_id):
                    score += 120
                elif low in title:
                    score += 50
                elif low in hay:
                    score += 12
                elif len(norm) >= 3 and norm in hay_norm:
                    score += 8
            for ref in table_refs:
                ref_norm = self._norm_alnum(ref)
                tid_norm = self._norm_alnum(table_id)
                if ref in table_id or (ref_norm and ref_norm == tid_norm):
                    score += 250
            return score, float(rec.t_observed or 0.0)

        ranked = sorted(records, key=_score, reverse=True)
        ranked = [r for r in ranked if _score(r)[0] > 0]
        if table_refs:
            ref_norms = {self._norm_alnum(ref) for ref in table_refs if ref}
            exact = [
                r for r in ranked
                if self._norm_alnum(r.table_id) in ref_norms
            ]
            if exact:
                ranked = exact
        return ranked[:max(1, int(limit or 10))]

    @staticmethod
    def _row_to_record(r: sqlite3.Row) -> ScreenTableRecord:
        return ScreenTableRecord(
            table_id=r["table_id"] or "",
            frame_id=r["frame_id"] or "",
            t_observed=float(r["t_observed"] or 0.0),
            title=r["title"] or "",
            columns=_json_loads_safe(r["columns_json"], []),
            rows=_json_loads_safe(r["rows_json"], []),
            app=r["app"] or "",
            window_title=r["window_title"] or "",
            raw_text=r["raw_text"] or "",
            source=r["source"] or "writer_vlm",
            confidence=float(r["confidence"] or 0.0),
            created_at=float(r["created_at"] or 0.0),
        )


def reconstruct_screen_tables_from_ocr(
    rec: ScreenTextRecord, *, max_tables: int = 4,
) -> List[ScreenTableRecord]:
    """Best-effort deterministic table reconstruction from OCR bboxes.

    This is intentionally model-free. It handles the common PDF/paper case where
    OCR correctly detects small text boxes, but raw_text loses row/column order.
    The output is conservative: if geometry does not provide a plausible grid,
    return [] and let raw OCR/writer memory remain the fallback.
    """

    def _numish(text: str) -> bool:
        s = str(text or "").strip()
        if not s:
            return False
        if re.fullmatch(r"[-–—/]+", s):
            return True
        return bool(re.search(r"\d", s)) and bool(re.fullmatch(
            r"[+\-−–—]?\d[\d,]*(?:\.\d+)?(?:\s*/\s*[+\-−–—]?\d[\d,]*(?:\.\d+)?)?"
            r"|[+\-−–—]?\d[\d,]*(?:\.\d+)?%?"
            r"|[+\-−–—]?\d[\d,]*(?:\.\d+)?\s*[x×]",
            s.replace(" ", ""),
        ))

    def _headerish(text: str) -> bool:
        t = str(text or "").lower()
        return bool(re.search(
            r"\b(method|model|benchmark|dataset|venue|anno|answer|ans|"
            r"accuracy|acc|cas|avg|average|tok|tool|q/vid|dur|long|"
            r"str|omni|mt|pro|bcvl|imeb|fvqa|livevqa|mmsearch)\b|"
            r"#|↑|↓|✓|✗|✔|✘|✅|❌",
            t,
        ))

    def _primary_headerish(text: str) -> bool:
        t = str(text or "").lower()
        return bool(re.search(
            r"\b(method|model|benchmark|dataset|venue|name|item|metric|system)\b|"
            r"模型|方法|基准|数据集|名称|指标|系统",
            t,
        ))

    def _section_labelish(text: str) -> bool:
        s = _clean_text(text)
        if not s or _numish(s) or len(s) > 120:
            return False
        t = s.lower()
        has_alpha_or_cjk = bool(re.search(r"[A-Za-z\u4e00-\u9fff]", s))
        if not has_alpha_or_cjk or re.search(r"\d", s):
            return False
        if re.search(
            r"\b(direct|answer|workflow|agentic|agents?|baselines?|ours|"
            r"methods?|models?|open[- ]source|closed[- ]source|ablation|"
            r"overall|results?|retrieval|multimodal|section|phase)\b|"
            r"小计|总计|汇总|结果|基线|我们的方法",
            t,
        ):
            return True
        # Short centered labels in paper tables are often section separators.
        tokens = re.findall(r"[A-Za-z]+|[\u4e00-\u9fff]", s)
        return 0 < len(tokens) <= 5 and not _primary_headerish(s) and not _headerish(s)

    def _table_caption(text: str) -> Optional[str]:
        t = str(text or "")
        m = re.search(r"(?<![A-Za-z])Table\s*\.?\s*(\d+)", t, re.IGNORECASE)
        if m:
            return f"Table {m.group(1)}"
        m = re.search(r"表\s*(\d+)", t)
        if m:
            return f"表{m.group(1)}"
        return None

    def _clean_text(text: str) -> str:
        return " ".join(str(text or "").replace("\n", " ").split())

    cells: List[Dict[str, Any]] = []
    for b in rec.ocr_blocks or []:
        text = _clean_text(b.text)
        if not text:
            continue
        box = list(b.bbox or [])
        if len(box) < 4:
            continue
        try:
            x, y, w, h = [float(v) for v in box[:4]]
        except (TypeError, ValueError):
            continue
        if w <= 0 or h <= 0:
            continue
        cells.append({
            "text": text,
            "x0": x,
            "y0": y,
            "x1": x + w,
            "y1": y + h,
            "cx": x + w / 2.0,
            "cy": y + h / 2.0,
            "w": w,
            "h": h,
            "confidence": float(b.confidence or 0.0),
        })
    if len(cells) < 8:
        return []

    heights = sorted(c["h"] for c in cells if c["h"] > 0)
    median_h = heights[len(heights) // 2] if heights else 12.0
    # OCR boxes on dense PDF tables often have row heights around 12-16px while
    # adjacent rows are only ~9-11px apart. A loose line threshold merges two
    # table rows and destroys row/column alignment, so keep it deliberately
    # tight; same-row OCR boxes usually differ by only a few pixels in center-y.
    row_tol = max(3.5, min(8.0, median_h * 0.38))
    page_w = max(c["x1"] for c in cells) - min(c["x0"] for c in cells)

    rows_raw: List[Dict[str, Any]] = []
    for c in sorted(cells, key=lambda z: (z["cy"], z["x0"])):
        target: Optional[Dict[str, Any]] = None
        for row in rows_raw[-3:]:
            if abs(c["cy"] - row["cy"]) <= row_tol:
                target = row
                break
        if target is None:
            target = {"cells": [], "cy": c["cy"]}
            rows_raw.append(target)
        target["cells"].append(c)
        target["cy"] = sum(x["cy"] for x in target["cells"]) / len(target["cells"])

    line_rows: List[Dict[str, Any]] = []
    for row in rows_raw:
        rcells = sorted(row["cells"], key=lambda z: z["x0"])
        text = _clean_text(" ".join(c["text"] for c in rcells))
        if not text:
            continue
        line_rows.append({
            "cells": rcells,
            "text": text,
            "x0": min(c["x0"] for c in rcells),
            "x1": max(c["x1"] for c in rcells),
            "y0": min(c["y0"] for c in rcells),
            "y1": max(c["y1"] for c in rcells),
            "cy": sum(c["cy"] for c in rcells) / len(rcells),
        })
    if len(line_rows) < 4:
        return []

    def _line_score(row: Dict[str, Any]) -> int:
        rcells = row["cells"]
        text = row["text"]
        nums = sum(1 for c in rcells if _numish(c["text"]))
        heads = sum(1 for c in rcells if _headerish(c["text"]))
        span = row["x1"] - row["x0"]
        score = nums * 2 + heads * 3
        if len(rcells) >= 3:
            score += 2
        if span >= max(120.0, page_w * 0.35):
            score += 2
        if _table_caption(text):
            score -= 3
        return score

    captions = [
        (i, _table_caption(r["text"]))
        for i, r in enumerate(line_rows)
        if _table_caption(r["text"])
    ]

    regions: List[Tuple[str, int, int, int]] = []
    used: Set[Tuple[int, int]] = set()
    for cap_i, table_id in captions:
        if not table_id:
            continue
        start = max(0, cap_i - 35)
        header_i: Optional[int] = None
        for j in range(cap_i - 1, start - 1, -1):
            row = line_rows[j]
            text_l = row["text"].lower()
            if len(row["cells"]) >= 2 and (
                re.search(r"\b(method|model|benchmark|dataset)\b", text_l)
                or (sum(1 for c in row["cells"] if _headerish(c["text"])) >= 2)
            ):
                header_i = j
                break
        if header_i is None:
            for j in range(cap_i - 1, start - 1, -1):
                if _line_score(line_rows[j]) >= 8:
                    header_i = j
                    break
        if header_i is None:
            continue
        # If there are multiple plausible header rows immediately above the
        # chosen one, include them so group headers like BCVL/IMEB survive.
        while header_i > start:
            prev = line_rows[header_i - 1]
            gap = line_rows[header_i]["y0"] - prev["y1"]
            if gap <= median_h * 1.6 and (
                    sum(1 for c in prev["cells"] if _headerish(c["text"])) >= 2
                    or len(prev["cells"]) >= 3):
                header_i -= 1
                continue
            break
        end = cap_i
        if (header_i, end) in used:
            continue
        used.add((header_i, end))
        regions.append((table_id, header_i, end, cap_i))

    # Fallback: table-looking regions without an explicit caption.
    if not regions:
        i = 0
        auto_idx = 1
        while i < len(line_rows):
            if _line_score(line_rows[i]) < 9:
                i += 1
                continue
            start = i
            j = i + 1
            while j < len(line_rows) and _line_score(line_rows[j]) >= 5:
                if line_rows[j]["y0"] - line_rows[j - 1]["y1"] > median_h * 2.8:
                    break
                j += 1
            if j - start >= 3:
                regions.append((f"auto_table_{auto_idx}", start, j, j - 1))
                auto_idx += 1
            i = max(j, i + 1)

    def _cluster_positions(values: List[float], tol: float) -> List[float]:
        clusters: List[List[float]] = []
        for x in sorted(values):
            if not clusters or abs(x - (sum(clusters[-1]) / len(clusters[-1]))) > tol:
                clusters.append([x])
            else:
                clusters[-1].append(x)
        return [sum(c) / len(c) for c in clusters]

    def _assign(x: float, anchors: List[float]) -> Optional[int]:
        if not anchors:
            return None
        distances = [abs(x - a) for a in anchors]
        idx = min(range(len(distances)), key=distances.__getitem__)
        return idx

    out: List[ScreenTableRecord] = []
    for table_id, start, end, cap_i in regions[:max_tables]:
        region_rows = line_rows[start:end]
        if len(region_rows) < 2:
            continue
        data_rows = [
            r for r in region_rows
            if sum(1 for c in r["cells"] if _numish(c["text"])) >= 2
        ]
        if not data_rows:
            continue
        x_values: List[float] = []
        first_text_x: List[float] = []
        for r in data_rows:
            for c in r["cells"]:
                if _numish(c["text"]):
                    x_values.append(c["cx"])
                elif c["x0"] < (min(x["x0"] for x in r["cells"]) + page_w * 0.25):
                    first_text_x.append(c["cx"])
        if len(x_values) < 2:
            continue
        widths = sorted(c["w"] for r in data_rows for c in r["cells"] if _numish(c["text"]))
        median_w = widths[len(widths) // 2] if widths else 32.0
        x_tol = max(12.0, min(34.0, median_w * 0.75))
        numeric_anchors = _cluster_positions(x_values, x_tol)
        if len(numeric_anchors) < 2:
            continue
        first_anchor = (
            sum(first_text_x) / len(first_text_x)
            if first_text_x else min(r["x0"] for r in region_rows)
        )
        anchors = [first_anchor] + numeric_anchors

        first_data_i = min(region_rows.index(r) for r in data_rows)
        header_rows = region_rows[:max(1, first_data_i)]
        if not header_rows:
            header_rows = region_rows[:1]

        minor_labels = [""] * len(anchors)
        group_labels = [""] * len(anchors)
        for hi, hr in enumerate(header_rows):
            for c in hr["cells"]:
                txt = _clean_text(c["text"])
                if not txt or _numish(txt):
                    continue
                idx = _assign(c["cx"], anchors)
                if idx is None:
                    continue
                # Wide/group headers usually sit above several numeric columns.
                covered = [
                    k for k, a in enumerate(anchors)
                    if c["x0"] - 8 <= a <= c["x1"] + 8
                ]
                if len(covered) >= 2:
                    for k in covered:
                        group_labels[k] = txt
                elif hi < len(header_rows) - 1 and idx > 0:
                    group_labels[idx] = txt
                else:
                    minor_labels[idx] = txt

        # If group headers are centered over column groups rather than spanning
        # them, assign each numeric column to the nearest group center.
        group_cells: List[Tuple[float, str]] = []
        for hr in header_rows:
            for c in hr["cells"]:
                txt = _clean_text(c["text"])
                if txt and not _numish(txt) and _headerish(txt):
                    if not re.search(r"method|model|benchmark|dataset|#|tok|tool|acc|cas",
                                     txt, re.IGNORECASE):
                        group_cells.append((c["cx"], txt))
        if group_cells:
            group_cells = sorted(group_cells)
            for k, a in enumerate(anchors[1:], start=1):
                nearest = min(group_cells, key=lambda it: abs(a - it[0]))
                if abs(a - nearest[0]) <= page_w * 0.35 and not group_labels[k]:
                    group_labels[k] = nearest[1]

        columns: List[str] = []
        for i, _a in enumerate(anchors):
            minor = minor_labels[i].strip()
            group = group_labels[i].strip()
            if i == 0:
                columns.append(minor or "Method")
            elif group and minor and group.lower() not in minor.lower():
                columns.append(f"{group} {minor}")
            else:
                columns.append(minor or group or f"col_{i}")

        # De-duplicate equal labels without losing column count.
        seen_cols: Dict[str, int] = {}
        unique_cols: List[str] = []
        for col in columns:
            base = col or "col"
            n = seen_cols.get(base, 0)
            seen_cols[base] = n + 1
            unique_cols.append(base if n == 0 else f"{base}_{n + 1}")
        columns = unique_cols

        parsed_rows: List[Dict[str, str]] = []
        for r in region_rows[first_data_i:]:
            vals = [""] * len(columns)
            text_cells = []
            value_count = 0
            for c in r["cells"]:
                idx = _assign(c["cx"], anchors)
                if idx is None:
                    continue
                txt = _clean_text(c["text"])
                if idx == 0 and not _numish(txt):
                    text_cells.append(txt)
                    continue
                if idx > 0 and _numish(txt):
                    value_count += 1
                if vals[idx]:
                    vals[idx] = f"{vals[idx]} {txt}".strip()
                else:
                    vals[idx] = txt
            if text_cells:
                vals[0] = " ".join(text_cells)
            if not vals[0] and value_count < max(2, min(4, len(columns) // 3)):
                continue
            if value_count < 2:
                continue
            parsed_rows.append({columns[i]: vals[i] for i in range(len(columns))})

        if len(parsed_rows) < 2 or len(columns) < 3:
            continue
        caption = line_rows[cap_i]["text"] if 0 <= cap_i < len(line_rows) else ""
        raw_region = "\n".join(r["text"] for r in region_rows)
        avg_conf = sum(c["confidence"] for r in region_rows for c in r["cells"]) / max(
            1, sum(len(r["cells"]) for r in region_rows))
        confidence = max(0.2, min(0.95, avg_conf))

        # Deterministic table reconstruction is useful for compact, regular
        # grids. Multi-section paper tables often contain centered group labels
        # and repeated headers; if we force them into one grid, the reconstructed
        # columns can become shifted evidence. Keep the record, but mark it as
        # low confidence so recall falls back to raw OCR/frame evidence first.
        quality_penalty = 0.0
        first_col = columns[0] if columns else ""
        if len(columns) >= 4 and not _primary_headerish(first_col):
            quality_penalty += 0.40
        suspicious_columns = [
            c for c in columns
            if _numish(c) or _section_labelish(c)
        ]
        if suspicious_columns:
            quality_penalty += min(0.32, 0.08 * len(suspicious_columns))
        generic_columns = [
            c for c in columns[1:]
            if re.fullmatch(r"col_\d+", str(c or "").strip().lower())
        ]
        if generic_columns:
            quality_penalty += min(0.24, 0.06 * len(generic_columns))
        section_rows = [
            r for r in region_rows
            if len(r["cells"]) <= max(2, len(columns) // 4)
            and _section_labelish(r["text"])
        ]
        if len(section_rows) >= 2:
            quality_penalty += min(0.34, 0.16 + 0.06 * (len(section_rows) - 2))
        if len(region_rows) >= 18 and len(section_rows) >= 2:
            quality_penalty += 0.22
        first_col_empty = sum(
            1 for row in parsed_rows
            if not _clean_text(row.get(first_col, ""))
        ) / max(1, len(parsed_rows))
        if first_col_empty > 0.25:
            quality_penalty += 0.16
        sparse_rows = 0
        for row in parsed_rows:
            non_empty = sum(1 for v in row.values() if _clean_text(v))
            if non_empty < max(3, len(columns) // 2):
                sparse_rows += 1
        sparse_ratio = sparse_rows / max(1, len(parsed_rows))
        if sparse_ratio > 0.45:
            quality_penalty += 0.14
        confidence = max(0.2, min(0.95, confidence - quality_penalty))
        source = (
            "ocr_table_rebuilder"
            if confidence >= 0.60
            else "ocr_table_rebuilder_low_confidence"
        )
        out.append(ScreenTableRecord(
            table_id=table_id,
            frame_id=rec.frame_id,
            t_observed=rec.t_observed,
            title=caption,
            columns=columns,
            rows=parsed_rows,
            app=rec.app,
            window_title=rec.window_title,
            raw_text=raw_region,
            source=source,
            confidence=confidence,
            created_at=time.time(),
        ))
    return out


class TaskStateStore(_DesktopSQLiteMixin):
    """Append-only task-state memory for desktop office workflows."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS task_states (
        id                 INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id            TEXT NOT NULL,
        t_observed         REAL NOT NULL,
        active_task        TEXT,
        goal               TEXT,
        current_artifact   TEXT,
        open_questions     TEXT,
        decisions          TEXT,
        blockers           TEXT,
        next_actions       TEXT,
        evidence_frame_ids TEXT,
        source             TEXT,
        confidence         REAL DEFAULT 1.0,
        raw_json           TEXT,
        created_at         REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_task_state_id
        ON task_states(task_id);
    CREATE INDEX IF NOT EXISTS idx_task_state_t
        ON task_states(t_observed);
    """

    def __init__(self, cfg: Config, *, db_path: Optional[str] = None):
        super().__init__(cfg, db_path=db_path)
        with self._lock, self._connect() as c:
            c.executescript(self._SCHEMA)
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA busy_timeout=5000")

    def insert(self, rec: TaskStateRecord) -> Optional[int]:
        if not (rec.task_id or rec.active_task or rec.goal
                or rec.current_artifact or rec.decisions or rec.blockers
                or rec.next_actions or rec.open_questions):
            return None
        task_id = rec.task_id or self.make_task_id(
            rec.active_task, rec.goal, rec.current_artifact)
        with self._lock, self._connect() as c:
            cur = c.execute(
                """INSERT INTO task_states
                   (task_id, t_observed, active_task, goal, current_artifact,
                    open_questions, decisions, blockers, next_actions,
                    evidence_frame_ids, source, confidence, raw_json, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (task_id, float(rec.t_observed or 0.0),
                 rec.active_task or "", rec.goal or "",
                 rec.current_artifact or "",
                 json.dumps(rec.open_questions or [], ensure_ascii=False),
                 json.dumps(rec.decisions or [], ensure_ascii=False),
                 json.dumps(rec.blockers or [], ensure_ascii=False),
                 json.dumps(rec.next_actions or [], ensure_ascii=False),
                 json.dumps(rec.evidence_frame_ids or [], ensure_ascii=False),
                 rec.source or "writer",
                 float(rec.confidence or 1.0),
                 json.dumps(rec.raw or {}, ensure_ascii=False),
                 rec.created_at or time.time()),
            )
            return int(cur.lastrowid)

    @staticmethod
    def make_task_id(active_task: str, goal: str = "",
                     current_artifact: str = "") -> str:
        seed = "|".join(
            p.strip().lower() for p in
            [active_task or "", goal or "", current_artifact or ""]
            if p and p.strip())
        if not seed:
            seed = f"task_{int(time.time())}"
        return "task_" + uuid.uuid5(uuid.NAMESPACE_URL, seed).hex[:10]

    def latest(self, ask_ts: float, limit: int = 5) -> List[TaskStateRecord]:
        with self._connect() as c:
            rows = c.execute(
                """SELECT * FROM task_states
                   WHERE t_observed <= ?
                   ORDER BY t_observed DESC LIMIT ?""",
                (ask_ts, max(1, int(limit or 5))),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def get(self, task_id: str, ask_ts: float, limit: int = 10) -> List[TaskStateRecord]:
        if not task_id:
            return []
        with self._connect() as c:
            rows = c.execute(
                """SELECT * FROM task_states
                   WHERE task_id=? AND t_observed <= ?
                   ORDER BY t_observed ASC LIMIT ?""",
                (task_id, ask_ts, max(1, int(limit or 10))),
            ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def search(self, query: str, ask_ts: float, limit: int = 8) -> List[TaskStateRecord]:
        terms = mm_expand_terms(mm_tokenize_query(query))
        if not terms:
            return self.latest(ask_ts, limit=limit)
        where = ["t_observed <= ?"]
        args: List[Any] = [ask_ts]
        ors: List[str] = []
        for term in terms[:12]:
            pat = f"%{term}%"
            ors.append(
                "(LOWER(active_task) LIKE ? OR LOWER(goal) LIKE ? "
                "OR LOWER(current_artifact) LIKE ? OR LOWER(raw_json) LIKE ?)")
            args.extend([pat, pat, pat, pat])
        where.append("(" + " OR ".join(ors) + ")")
        sql = (f"SELECT * FROM task_states WHERE {' AND '.join(where)} "
               "ORDER BY t_observed DESC LIMIT ?")
        args.append(max(1, int(limit or 8)))
        with self._connect() as c:
            rows = c.execute(sql, args).fetchall()
        return [self._row_to_record(r) for r in rows]

    @staticmethod
    def _row_to_record(r: sqlite3.Row) -> TaskStateRecord:
        return TaskStateRecord(
            id=int(r["id"] or 0),
            task_id=r["task_id"] or "",
            t_observed=float(r["t_observed"] or 0.0),
            active_task=r["active_task"] or "",
            goal=r["goal"] or "",
            current_artifact=r["current_artifact"] or "",
            open_questions=_json_loads_safe(r["open_questions"], []),
            decisions=_json_loads_safe(r["decisions"], []),
            blockers=_json_loads_safe(r["blockers"], []),
            next_actions=_json_loads_safe(r["next_actions"], []),
            evidence_frame_ids=_json_loads_safe(r["evidence_frame_ids"], []),
            source=r["source"] or "writer",
            confidence=float(r["confidence"] or 1.0),
            raw=_json_loads_safe(r["raw_json"], {}),
            created_at=float(r["created_at"] or 0.0),
        )


def _fuzzy_ratio(a: str, b: str) -> float:
    """String similarity in [0, 1]. Case-insensitive, whitespace-stripped;
    boosts substring containment (a short string wholly inside a long one, of
    length >= 2, scores 0.85+) before falling back to SequenceMatcher ratio."""
    a = (a or "").strip().lower()
    b = (b or "").strip().lower()
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    # 包含关系加分
    if a in b or b in a:
        short_len = min(len(a), len(b))
        long_len = max(len(a), len(b))
        if short_len >= 2:
            return 0.85 + 0.15 * (short_len / long_len)
    return SequenceMatcher(None, a, b).ratio()


class MemoryStore:
    """SQLite-backed 3-tier memory (micro/macro/super events) + entity/edge graph.

    D3 anti-dirty-read: Recall read paths take an ask_ts and constrain results to
    t_end <= ask_ts, so a query never sees events from after the moment it asked
    about (a time snapshot). Writes hold self._lock; pure reads do NOT (WAL gives
    concurrent readers), letting Recall's asyncio.gather tools run in parallel."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.db_path = cfg.mem_db_path
        if not self.db_path:
            tmp = tempfile.NamedTemporaryFile(
                prefix="tml_mem_", suffix=".sqlite", delete=False,
            )
            tmp.close()
            self.db_path = tmp.name
            log.info("[mem] 使用临时 SQLite (session-scoped): %s", self.db_path)
        else:
            parent = os.path.dirname(os.path.abspath(self.db_path))
            if parent:
                os.makedirs(parent, exist_ok=True)
            log.info("[mem] 使用 SQLite: %s", self.db_path)
        self._lock = threading.RLock()
        # 必须在 _init_db() 之前置位: _init_db 会走 _conn(), 让第一条连接顺手
        # 兜底把 journal_mode=WAL 落到库上; 同时每条连接都补 synchronous=NORMAL
        # (per-connection 设置, 别处设过的不算)。
        self._wal_ready = False
        self._audio_embedding_storage_removed = False
        self._init_db()
        if self._audio_embedding_storage_removed:
            self._vacuum_after_audio_embedding_removal()
        # in-memory 缓存最近的 micro 边界, 用于 finalize
        self._pending_micros: List[MicroEvent] = []   # 已 finalize 但未聚合到 macro 的 L1
        self._pending_macros: List[MacroEvent] = []   # 已 finalize 但未聚合到 super 的 L2
        # ★ Hybrid retrieval (一期): 文本 embedding 客户端 (总是存在, 内部按
        #   base_url 判断是否 enabled). 未配置端点时 enabled=False, 所有 embedding
        #   路径 (写入、更新、检索) 自动跳过, 系统退化为纯关键词兼容原行为.
        self.embedding_client = EmbeddingClient(
            base_url=getattr(cfg, "embedding_base_url", "") or "",
            api_key=getattr(cfg, "embedding_api_key", "") or "",
            model=getattr(cfg, "embedding_model", "") or "text-embedding-v3",
            dimensions=int(getattr(cfg, "embedding_dimensions", 1024) or 1024),
            timeout_sec=float(
                getattr(cfg, "embedding_timeout_sec", 4.0) or 4.0),
        )
        if self.embedding_client.enabled:
            log.info(
                "[mem] embedding enabled: model=%s dim=%d endpoint=%s",
                self.embedding_client.model, self.embedding_client.dimensions,
                self.embedding_client.base_url)
        else:
            log.info(
                "[mem] embedding disabled (empty base_url or api_key); "
                "hybrid recall degrades to pure keyword")
        # ★ 二期: 帧图像 embedding 客户端 (multimodal-embedding-v1, 独立语义空间).
        #   mm_embedding_model 空 → 关闭; api_key 空 → 复用文本 embedding 的 key
        #   (同为 DashScope 账号). enabled=False 时 Writer 不算 / 检索工具提示未启用.
        _mm_model = (getattr(cfg, "mm_embedding_model", "") or "").strip()
        self.mm_embedding_client = MultimodalEmbeddingClient(
            base_url=getattr(cfg, "mm_embedding_base_url", "") or "",
            api_key=(getattr(cfg, "mm_embedding_api_key", "") or ""
                     or getattr(cfg, "embedding_api_key", "") or ""),
            model=_mm_model,
            timeout_sec=float(
                getattr(cfg, "mm_embedding_timeout_sec", 6.0) or 6.0),
            dimensions=int(getattr(cfg, "mm_embedding_dimensions", 0) or 0),
            res_level=int(getattr(cfg, "mm_embedding_res_level", -1)
                          if getattr(cfg, "mm_embedding_res_level", -1) is not None
                          else -1),
        )
        if self.mm_embedding_client.enabled:
            log.info(
                "[mem] mm-embedding enabled: model=%s endpoint=%s",
                self.mm_embedding_client.model, self.mm_embedding_client.base_url)
        else:
            log.info(
                "[mem] mm-embedding disabled (empty mm_model or api_key); "
                "search_frames_by_text unavailable")

    def _init_db(self) -> None:
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                conn.executescript(SCHEMA_SQL)
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA busy_timeout=5000")

                def _try_alter(sql: str) -> None:
                    try:
                        conn.execute(sql)
                    except sqlite3.OperationalError:
                        pass

                # 兼容旧 db: 老版本没有 frame_ids 列, 试探性 ALTER
                _try_alter("ALTER TABLE micro_events ADD COLUMN frame_ids TEXT")
                # 兼容旧 db: entities 加 representative_frame_id 列
                _try_alter("ALTER TABLE entities ADD COLUMN representative_frame_id TEXT")
                # ★ E3/E4 (evolve): 老 db 升级到新 schema, 给 micro/macro/super/entities
                #   补上新字段. 都用 try-pass, 已有列则忽略.
                for sql in (
                    # micro_events evolve fields
                    "ALTER TABLE micro_events ADD COLUMN superseded_by TEXT",
                    "ALTER TABLE micro_events ADD COLUMN revised_at REAL",
                    "ALTER TABLE micro_events ADD COLUMN revision_count INTEGER DEFAULT 0",
                    # macro_events evolve fields
                    "ALTER TABLE macro_events ADD COLUMN narrative_arc TEXT",
                    "ALTER TABLE macro_events ADD COLUMN entity_arcs TEXT",
                    "ALTER TABLE macro_events ADD COLUMN superseded_by TEXT",
                    "ALTER TABLE macro_events ADD COLUMN revised_at REAL",
                    "ALTER TABLE macro_events ADD COLUMN revision_count INTEGER DEFAULT 0",
                    # super_events evolve fields
                    "ALTER TABLE super_events ADD COLUMN narrative_arc TEXT",
                    "ALTER TABLE super_events ADD COLUMN superseded_by TEXT",
                    "ALTER TABLE super_events ADD COLUMN revised_at REAL",
                    "ALTER TABLE super_events ADD COLUMN revision_count INTEGER DEFAULT 0",
                    # entities evolve fields
                    "ALTER TABLE entities ADD COLUMN merged_into TEXT",
                    "ALTER TABLE entities ADD COLUMN revised_at REAL",
                    "ALTER TABLE entities ADD COLUMN revision_count INTEGER DEFAULT 0",
                    # ★ Hybrid retrieval (一期): 文本 embedding BLOB 列
                    "ALTER TABLE micro_events ADD COLUMN text_embedding BLOB",
                    "ALTER TABLE entities ADD COLUMN text_embedding BLOB",
                ):
                    _try_alter(sql)
                self._audio_embedding_storage_removed = (
                    self._remove_audio_embedding_storage(conn))
                self._audio_fts_enabled = self._ensure_audio_fts(conn)
                self._repair_merged_entity_links(conn)
                conn.commit()

    @staticmethod
    def _remove_audio_embedding_storage(conn: sqlite3.Connection) -> bool:
        """Delete the retired per-ASR vectors from databases made by vNext.

        New databases never create this column.  For an already-upgraded DB we
        prefer ``DROP COLUMN``; older SQLite builds fall back to NULLing the
        BLOBs.  The return value tells ``__init__`` whether a one-time VACUUM is
        worthwhile to return the freed vector pages to the filesystem.
        """
        columns = {
            str(row[1])
            for row in conn.execute(
                "PRAGMA table_info(audio_observations)").fetchall()
        }
        if "text_embedding" not in columns:
            return False
        had_vectors = conn.execute(
            "SELECT 1 FROM audio_observations "
            "WHERE text_embedding IS NOT NULL LIMIT 1").fetchone() is not None
        try:
            conn.execute(
                "ALTER TABLE audio_observations DROP COLUMN text_embedding")
        except sqlite3.OperationalError as exc:
            if had_vectors:
                conn.execute(
                    "UPDATE audio_observations SET text_embedding=NULL "
                    "WHERE text_embedding IS NOT NULL")
            log.info(
                "[mem] audio embedding column retained empty "
                "(SQLite DROP COLUMN unavailable: %s)", exc)
        return had_vectors

    def _vacuum_after_audio_embedding_removal(self) -> None:
        """One-time compaction after deleting legacy per-ASR vector BLOBs."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA busy_timeout=5000")
                conn.execute("VACUUM")
            log.info("[mem] reclaimed SQLite pages after removing audio embeddings")
        except Exception as exc:
            # The pages remain reusable inside SQLite even if an active reader
            # prevents shrinking the file at this moment.
            log.warning("[mem] audio embedding cleanup VACUUM skipped: %s", exc)

    @staticmethod
    def _ensure_audio_fts(conn: sqlite3.Connection) -> bool:
        """Create and backfill the global BM25 index for durable ASR text.

        FTS5's trigram tokenizer is preferred because it indexes CJK and noisy
        ASR substrings without an external segmenter.  Minimal SQLite builds may
        lack trigram while still providing FTS5, so unicode61 is a safe fallback.
        If FTS5 itself is unavailable, recall continues through keyword
        matching instead of making the database unusable.
        """
        tokenizer = "trigram"
        try:
            conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS audio_observations_fts "
                "USING fts5(text, content='audio_observations', "
                "content_rowid='id', tokenize='trigram')")
        except sqlite3.OperationalError:
            tokenizer = "unicode61"
            try:
                conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS audio_observations_fts "
                    "USING fts5(text, content='audio_observations', "
                    "content_rowid='id', tokenize='unicode61')")
            except sqlite3.OperationalError as exc:
                log.info("[mem] audio FTS5 unavailable; using non-FTS recall: %s", exc)
                return False

        conn.executescript("""
            CREATE TRIGGER IF NOT EXISTS audio_observations_fts_ai
            AFTER INSERT ON audio_observations BEGIN
              INSERT INTO audio_observations_fts(rowid, text)
              VALUES (new.id, new.text);
            END;
            CREATE TRIGGER IF NOT EXISTS audio_observations_fts_ad
            AFTER DELETE ON audio_observations BEGIN
              INSERT INTO audio_observations_fts(audio_observations_fts, rowid, text)
              VALUES ('delete', old.id, old.text);
            END;
            CREATE TRIGGER IF NOT EXISTS audio_observations_fts_au
            AFTER UPDATE OF text ON audio_observations BEGIN
              INSERT INTO audio_observations_fts(audio_observations_fts, rowid, text)
              VALUES ('delete', old.id, old.text);
              INSERT INTO audio_observations_fts(rowid, text)
              VALUES (new.id, new.text);
            END;
        """)
        meta_key = f"audio_fts_{tokenizer}_v1"
        indexed = conn.execute(
            "SELECT value FROM meta WHERE key=?", (meta_key,)).fetchone()
        if indexed is None:
            # One-time migration for rows that predate the INSERT trigger.
            conn.execute(
                "INSERT INTO audio_observations_fts(audio_observations_fts) "
                "VALUES ('rebuild')")
            conn.execute(
                "INSERT OR REPLACE INTO meta(key,value) VALUES (?,?)",
                (meta_key, "1"))
        return True

    @staticmethod
    def _resolve_entity_row(
        conn: sqlite3.Connection, entity_id: str, *, max_hops: int = 32,
    ) -> Tuple[Optional[sqlite3.Row], List[str]]:
        """Resolve a soft-merged entity id to its current canonical row.

        ``entities`` is the authoritative current-state table.  Reviewer merge
        operations deliberately retain loser rows for audit/debug, so every
        recall-facing by-id lookup must follow ``merged_into``.  ``PRUNED`` is
        a terminal tombstone.  The returned chain always starts with the
        requested id; a missing row/cycle/pruned target returns ``None``.
        """
        current = str(entity_id or "").strip()
        if not current:
            return None, []
        chain: List[str] = []
        seen: Set[str] = set()
        for _ in range(max(1, max_hops)):
            if current in seen:
                chain.append(current)
                return None, chain
            seen.add(current)
            chain.append(current)
            row = conn.execute(
                "SELECT * FROM entities WHERE id=?", (current,),
            ).fetchone()
            if row is None:
                return None, chain
            target = str(row["merged_into"] or "").strip()
            if not target:
                return row, chain
            if target.upper() == "PRUNED":
                chain.append("PRUNED")
                return None, chain
            current = target
        return None, chain

    def resolve_entity(
        self, entity_id: str, *, max_hops: int = 32,
    ) -> Tuple[Optional[Entity], List[str]]:
        """Public canonical-id resolver for Recall tools.

        Raw Reviewer/debug code can continue to use :meth:`peek_entity` when it
        intentionally needs the historical loser row.  User-answering paths
        should use this resolver and then query with the canonical id.
        """
        with self._connect() as c:
            row, chain = self._resolve_entity_row(
                c, entity_id, max_hops=max_hops)
        return (self._row_to_entity(row) if row is not None else None), chain

    def _repair_merged_entity_links(self, conn: sqlite3.Connection) -> None:
        """One-time repair for links stranded by the old UPDATE OR IGNORE merge.

        When winner and loser shared a frame/event primary key, SQLite ignored
        the UPDATE and left the loser link query-visible.  Move every historical
        loser link to the final canonical winner using insert-then-delete, while
        keeping loser entity/state rows for the revision inspector.
        """
        migration_key = "entity_merge_links_v2"
        done = conn.execute(
            "SELECT value FROM meta WHERE key=?", (migration_key,),
        ).fetchone()
        if done is not None and str(done["value"] or "") == "1":
            return

        rows = conn.execute(
            "SELECT id FROM entities WHERE merged_into IS NOT NULL "
            "AND merged_into != '' AND UPPER(merged_into) != 'PRUNED'"
        ).fetchall()
        touched_winners: Set[str] = set()
        for row in rows:
            loser_id = str(row["id"] or "")
            winner_row, _chain = self._resolve_entity_row(conn, loser_id)
            if winner_row is None:
                continue
            winner_id = str(winner_row["id"] or "")
            if not winner_id or winner_id == loser_id:
                continue
            touched_winners.add(winner_id)
            conn.execute(
                "INSERT OR IGNORE INTO entity_event(entity_id,micro_id,t_observed) "
                "SELECT ?,micro_id,t_observed FROM entity_event WHERE entity_id=?",
                (winner_id, loser_id),
            )
            conn.execute("DELETE FROM entity_event WHERE entity_id=?", (loser_id,))
            conn.execute(
                "INSERT OR IGNORE INTO entity_frame(entity_id,frame_id,micro_id,t_observed) "
                "SELECT ?,frame_id,micro_id,t_observed FROM entity_frame WHERE entity_id=?",
                (winner_id, loser_id),
            )
            conn.execute("DELETE FROM entity_frame WHERE entity_id=?", (loser_id,))
            conn.execute(
                "INSERT OR IGNORE INTO entity_rep_frames"
                "(entity_id,frame_id,quality_score,note,added_at,added_by) "
                "SELECT ?,frame_id,quality_score,note,added_at,added_by "
                "FROM entity_rep_frames WHERE entity_id=?",
                (winner_id, loser_id),
            )
            conn.execute("DELETE FROM entity_rep_frames WHERE entity_id=?", (loser_id,))
            conn.execute(
                "UPDATE entity_quotes SET entity_id=? WHERE entity_id=?",
                (winner_id, loser_id),
            )
            conn.execute("UPDATE edges SET src_id=? WHERE src_id=?", (winner_id, loser_id))
            conn.execute("UPDATE edges SET dst_id=? WHERE dst_id=?", (winner_id, loser_id))
            # Path compression keeps future direct-id resolution cheap.
            conn.execute(
                "UPDATE entities SET merged_into=? WHERE id=?",
                (winner_id, loser_id),
            )

        for winner_id in touched_winners:
            top = conn.execute(
                "SELECT frame_id FROM entity_rep_frames WHERE entity_id=? "
                "ORDER BY quality_score DESC LIMIT 1",
                (winner_id,),
            ).fetchone()
            if top is not None:
                conn.execute(
                    "UPDATE entities SET representative_frame_id=? WHERE id=?",
                    (top["frame_id"], winner_id),
                )
        conn.execute(
            "INSERT INTO meta(key,value) VALUES(?, '1') "
            "ON CONFLICT(key) DO UPDATE SET value='1'",
            (migration_key,),
        )

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        _apply_conn_pragmas(c, set_wal=not self._wal_ready)
        self._wal_ready = True
        return c

    @contextmanager
    def _connect(self):
        """C6 fix: a context manager that actually CLOSES the connection.

        Using an sqlite3.Connection directly as a `with` target only commits/
        rolls back on exit — it does NOT close the connection, so under WAL each
        call opened a fresh connection that was never reclaimed, leaking over a
        long run. Here: yield → commit on success / rollback+raise on error →
        finally always close.
        """
        c = self._conn()
        try:
            yield c
            c.commit()
        except Exception:
            try:
                c.rollback()
            except Exception:
                pass
            raise
        finally:
            try:
                c.close()
            except Exception:
                pass

    # ----- 写入 (MemoryWriter / 聚合 worker 用) -----
    def insert_audio_observation(self, turn: Turn, *, source: str = "asr") -> int:
        """Persist one raw ASR turn independently of ConversationLog limits."""
        text = str(turn.content or "").strip()
        if not text:
            return 0
        with self._lock, self._connect() as c:
            cur = c.execute(
                """INSERT INTO audio_observations
                   (t_observed, wall_ts, speaker, text, source, created_at)
                   VALUES (?,?,?,?,?,?)""",
                (turn.rel_ts, float(turn.wall_ts), turn.speaker, text,
                 str(source or "asr"), time.time()),
            )
            return int(cur.lastrowid or 0)

    @staticmethod
    def _row_to_audio_turn(r: sqlite3.Row) -> Turn:
        return Turn(
            role="system", content=str(r["text"] or ""),
            wall_ts=float(r["wall_ts"] or 0.0), kind="audio_observation",
            rel_ts=(None if r["t_observed"] is None
                    else float(r["t_observed"])),
            speaker=(str(r["speaker"]) if r["speaker"] else None),
            row_id=int(r["id"]),
        )

    def get_audio_observations_after_id(
        self, after_id: int, ask_ts: float,
    ) -> Tuple[List[Turn], int]:
        """Return the contiguous durable ASR batch not yet consumed by Writer.

        The cursor is the SQLite row id, not a timestamp, so multiple cues with
        the same ``t_observed`` cannot be lost. We stop at the first future cue
        instead of skipping over it; advancing the returned cursor therefore
        never creates a hole even if ingestion and Writer run concurrently.
        """
        with self._connect() as c:
            rows = c.execute(
                """SELECT id, t_observed, wall_ts, speaker, text
                   FROM audio_observations
                   WHERE id > ? ORDER BY id""",
                (max(0, int(after_id)),),
            ).fetchall()
        turns: List[Turn] = []
        cursor = max(0, int(after_id))
        for r in rows:
            rel_ts = (None if r["t_observed"] is None
                      else float(r["t_observed"]))
            if rel_ts is not None and rel_ts > float(ask_ts) + 1e-3:
                break
            turns.append(Turn(
                role="system", content=str(r["text"] or ""),
                wall_ts=float(r["wall_ts"] or 0.0),
                kind="audio_observation", rel_ts=rel_ts,
                speaker=(str(r["speaker"]) if r["speaker"] else None),
                row_id=int(r["id"]),
            ))
            cursor = int(r["id"])
        return turns, cursor

    def get_audio_observations(self, ask_ts: float) -> List[Turn]:
        """Return all durable ASR observations visible at the ask snapshot."""
        with self._connect() as c:
            rows = c.execute(
                """SELECT id, t_observed, wall_ts, speaker, text
                   FROM audio_observations
                   WHERE t_observed IS NULL OR t_observed <= ?
                   ORDER BY COALESCE(t_observed, 0.0), id""",
                (float(ask_ts) + 1e-3,),
            ).fetchall()
        return [self._row_to_audio_turn(r) for r in rows]

    def get_audio_observations_in_range(
        self, t_start: float, t_end: float, ask_ts: float,
    ) -> List[Turn]:
        """Return durable ASR cues in one bounded event-time window."""
        safe_end = min(float(t_end), float(ask_ts))
        if safe_end < float(t_start):
            return []
        with self._connect() as c:
            rows = c.execute(
                """SELECT id, t_observed, wall_ts, speaker, text
                   FROM audio_observations
                   WHERE t_observed >= ? AND t_observed <= ?
                   ORDER BY t_observed, id""",
                (float(t_start), safe_end),
            ).fetchall()
        return [self._row_to_audio_turn(r) for r in rows]

    def search_audio_fts(
        self, query: str, ask_ts: float, *, top_k: int = 40,
    ) -> List[Tuple[Turn, float]]:
        """Global ASR FTS5/BM25 recall; lower raw BM25 is better.

        Quoted query segments keep identifiers and CJK phrases intact.  Terms
        shorter than three characters are left to the keyword arm because
        the preferred trigram tokenizer cannot index them reliably.
        """
        if not getattr(self, "_audio_fts_enabled", False):
            return []
        parts = re.findall(
            r"[\u4e00-\u9fff]+|[a-z0-9][a-z0-9._+-]*",
            str(query or "").lower(),
        )
        searchable = [p for p in parts if len(p) >= 3][:16]
        if not searchable:
            return []
        match = " OR ".join(
            '"' + p.replace('"', '""') + '"' for p in searchable)
        try:
            with self._connect() as c:
                rows = c.execute(
                    """SELECT a.id,a.t_observed,a.wall_ts,a.speaker,a.text,
                              bm25(audio_observations_fts) AS bm25_score
                       FROM audio_observations_fts
                       JOIN audio_observations AS a
                         ON a.id=audio_observations_fts.rowid
                       WHERE audio_observations_fts MATCH ?
                         AND (a.t_observed IS NULL OR a.t_observed <= ?)
                       ORDER BY bm25_score ASC, a.t_observed DESC
                       LIMIT ?""",
                    (match, float(ask_ts) + 1e-3, max(1, int(top_k))),
                ).fetchall()
        except sqlite3.OperationalError as exc:
            log.debug("[mem] audio FTS query failed for %r: %s", query, exc)
            return []
        return [
            (self._row_to_audio_turn(r), float(r["bm25_score"] or 0.0))
            for r in rows
        ]

    def set_meta(self, key: str, value: str) -> bool:
        key = str(key or "").strip()
        if not key:
            return False
        with self._lock, self._connect() as c:
            c.execute(
                """INSERT INTO meta(key, value) VALUES(?, ?)
                   ON CONFLICT(key) DO UPDATE SET value=excluded.value""",
                (key, str(value)),
            )
        return True

    def get_meta(self, key: str, default: str = "") -> str:
        key = str(key or "").strip()
        if not key:
            return default
        try:
            with self._connect() as c:
                row = c.execute(
                    "SELECT value FROM meta WHERE key=?", (key,),
                ).fetchone()
            return str(row["value"]) if row and row["value"] is not None else default
        except Exception:
            return default

    def insert_micro(self, mev: MicroEvent) -> None:
        with self._lock, self._connect() as c:
            c.execute(
                """INSERT OR REPLACE INTO micro_events
                   (id, t_start, t_end, description, subject, object, action,
                    macro_id, facts_keys, frame_ids, created_at,
                    superseded_by, revised_at, revision_count)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (mev.id, mev.t_start, mev.t_end, mev.description,
                 mev.subject, mev.object, mev.action, mev.macro_id,
                 json.dumps(mev.facts_keys, ensure_ascii=False),
                 json.dumps(mev.frame_ids, ensure_ascii=False),
                 mev.created_at,
                 mev.superseded_by, mev.revised_at, mev.revision_count),
            )
            c.commit()
        # ★ E4: 软删除的 micro 不再进 pending (避免被聚合)
        if not mev.superseded_by:
            self._pending_micros.append(mev)

    def update_micro_frame_ids(self, micro_id: str,
                                frame_ids: List[str]) -> None:
        """Backfill the associated frame_ids after finalize_micro."""
        with self._lock, self._connect() as c:
            c.execute(
                "UPDATE micro_events SET frame_ids=? WHERE id=?",
                (json.dumps(frame_ids, ensure_ascii=False), micro_id),
            )
            c.commit()

    def insert_macro(self, mac: MacroEvent) -> None:
        with self._lock, self._connect() as c:
            c.execute(
                """INSERT OR REPLACE INTO macro_events
                   (id, t_start, t_end, label, summary, super_id, key_entities,
                    created_at, narrative_arc, entity_arcs,
                    superseded_by, revised_at, revision_count)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (mac.id, mac.t_start, mac.t_end, mac.label, mac.summary,
                 mac.super_id,
                 json.dumps(mac.key_entities, ensure_ascii=False),
                 mac.created_at,
                 json.dumps(mac.narrative_arc, ensure_ascii=False),
                 json.dumps(mac.entity_arcs, ensure_ascii=False),
                 mac.superseded_by, mac.revised_at, mac.revision_count),
            )
            # 把对应 micro 的 macro_id 也回填 (排除软删除的 micro)
            c.execute(
                "UPDATE micro_events SET macro_id=? "
                "WHERE id IN (SELECT id FROM micro_events "
                "WHERE t_start >= ? AND t_end <= ? AND macro_id IS NULL "
                "AND (superseded_by IS NULL OR superseded_by=''))",
                (mac.id, mac.t_start, mac.t_end),
            )
            c.commit()
        if not mac.superseded_by:
            self._pending_macros.append(mac)
        # 清空已聚合的 pending micro
        self._pending_micros = [m for m in self._pending_micros
                                if m.t_end > mac.t_end]

    def insert_super(self, sev: SuperEvent) -> None:
        with self._lock, self._connect() as c:
            c.execute(
                """INSERT OR REPLACE INTO super_events
                   (id, t_start, t_end, label, description, macro_ids,
                    is_root, created_at, narrative_arc,
                    superseded_by, revised_at, revision_count)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (sev.id, sev.t_start, sev.t_end, sev.label, sev.description,
                 json.dumps(sev.macro_ids, ensure_ascii=False),
                 1 if sev.is_root else 0, sev.created_at,
                 json.dumps(sev.narrative_arc, ensure_ascii=False),
                 sev.superseded_by, sev.revised_at, sev.revision_count),
            )
            c.execute(
                "UPDATE macro_events SET super_id=? "
                "WHERE id IN (SELECT id FROM macro_events "
                "WHERE t_start >= ? AND t_end <= ? AND super_id IS NULL "
                "AND (superseded_by IS NULL OR superseded_by=''))",
                (sev.id, sev.t_start, sev.t_end),
            )
            c.commit()
        self._pending_macros = [m for m in self._pending_macros
                                if m.t_end > sev.t_end]

    def try_reuse_entity(
        self, entity_id: str, *,
        type_required: Optional[str] = None,
        new_attributes: Optional[Dict[str, str]] = None,
        new_aliases: Optional[List[str]] = None,
        new_last_seen: Optional[float] = None,
    ) -> Tuple[Optional[Entity], Optional[EntityState]]:
        """E9 fast path: taken when the Writer LLM fills reused_entity_id in
        entities_mentioned, i.e. "I'm 100% sure this observed entity IS the known
        entity with this id — merge the delta directly, skip fuzzy match".

        Returns (None, None) on any validation failure; the caller
        (Writer._upsert_entities) should fall back to the fuzzy path (upsert_entity
        by name). Failure conditions:
          - entity_id doesn't exist
          - merged_into is non-empty (already pruned/merged, must not be reused)
          - type_required doesn't match the DB row's type (guards a mis-copied id)

        On success the side effects match upsert_entity's merge branch (union
        attrs/aliases, bump last_seen, seen_count+1); returns (merged Entity,
        EntityState diff or None when nothing changed).
        """
        if not entity_id:
            return None, None
        with self._lock, self._connect() as c:
            row = c.execute(
                "SELECT * FROM entities WHERE id=?", (entity_id,)
            ).fetchone()
            if row is None:
                return None, None
            merged_into = (row["merged_into"] or "").strip()
            if merged_into:
                return None, None
            if type_required and str(row["type"] or "").upper() != str(type_required).upper():
                return None, None

            old_attrs = json.loads(row["attributes"] or "{}")
            old_aliases = set(json.loads(row["aliases"] or "[]"))
            attrs_delta: Dict[str, str] = {}
            new_alias_diff: List[str] = []
            if new_attributes:
                for k, v in new_attributes.items():
                    if old_attrs.get(k) != v:
                        attrs_delta[k] = v
                        old_attrs[k] = v
            if new_aliases:
                for al in new_aliases:
                    al = str(al).strip()
                    if al and al not in old_aliases:
                        new_alias_diff.append(al)
                        old_aliases.add(al)
            new_last = max(float(row["last_seen"] or 0.0),
                            float(new_last_seen or row["last_seen"] or 0.0))
            c.execute(
                """UPDATE entities SET attributes=?, aliases=?,
                   last_seen=?, seen_count=seen_count+1, updated_at=?
                   WHERE id=?""",
                (json.dumps(old_attrs, ensure_ascii=False),
                 json.dumps(sorted(old_aliases), ensure_ascii=False),
                 new_last, time.time(), entity_id),
            )
            c.commit()
            try:
                rep_fid = row["representative_frame_id"] or ""
            except (IndexError, KeyError):
                rep_fid = ""
            merged = Entity(
                id=row["id"], name=row["name"], type=row["type"],
                attributes=old_attrs, aliases=sorted(old_aliases),
                first_seen=float(row["first_seen"] or 0.0),
                last_seen=new_last,
                seen_count=int(row["seen_count"] or 1) + 1,
                representative_frame_id=rep_fid,
                updated_at=time.time(),
            )
            state: Optional[EntityState] = None
            if attrs_delta or new_alias_diff:
                state = EntityState(
                    entity_id=entity_id, t_observed=new_last,
                    state_label="refined",
                    attributes_delta=attrs_delta,
                    new_aliases=new_alias_diff,
                    source="writer",
                    note="reused_entity_id 命中",
                )
            return merged, state

    def upsert_entity(self, ent: Entity) -> Tuple[Entity, Optional[EntityState]]:
        """Fuzzy-match an existing entity of the same type (by name + aliases,
        threshold cfg.mem_entity_alias_threshold); create a new one on no match.

        E1: returns (entity, entity_state):
          - entity is the effective entity (its id may differ from the input's)
          - entity_state is the evolution state this call produced (caller
            decides whether to append it to the timeline):
            * new         → state_label="first_seen", attributes_delta = all attrs
            * merge hit   → state_label="refined", attributes_delta = the newly
                            added/changed attrs, new_aliases = newly added aliases
            * hit, no change → None (avoids polluting the timeline)
        """
        with self._lock, self._connect() as c:
            cur = c.execute(
                "SELECT * FROM entities WHERE type=? "
                "AND (merged_into IS NULL OR merged_into='')",
                (ent.type,),
            )
            best: Optional[sqlite3.Row] = None
            best_score = 0.0
            for row in cur.fetchall():
                cand_names = [row["name"]] + json.loads(row["aliases"] or "[]")
                for cn in cand_names:
                    s = _fuzzy_ratio(cn, ent.name)
                    if s > best_score:
                        best_score = s
                        best = row
                    for al in ent.aliases:
                        s2 = _fuzzy_ratio(cn, al)
                        if s2 > best_score:
                            best_score = s2
                            best = row
            if best is not None and best_score >= self.cfg.mem_entity_alias_threshold:
                # 合并: 更新 attributes / aliases / last_seen
                old_aliases = set(json.loads(best["aliases"] or "[]"))
                new_alias_diff: List[str] = []
                for al in list(ent.aliases) + ([ent.name] if best["name"] != ent.name else []):
                    if al and al not in old_aliases:
                        new_alias_diff.append(al)
                        old_aliases.add(al)
                old_attrs = json.loads(best["attributes"] or "{}")
                # ★ 算 attributes_delta: 真正新增/变化的字段
                attrs_delta: Dict[str, str] = {}
                for k, v in ent.attributes.items():
                    if old_attrs.get(k) != v:
                        attrs_delta[k] = v
                old_attrs.update(ent.attributes)
                new_last = max(best["last_seen"], ent.last_seen)
                c.execute(
                    """UPDATE entities SET attributes=?, aliases=?,
                       last_seen=?, seen_count=seen_count+1, updated_at=?
                       WHERE id=?""",
                    (json.dumps(old_attrs, ensure_ascii=False),
                     json.dumps(sorted(old_aliases), ensure_ascii=False),
                     new_last, time.time(), best["id"]),
                )
                c.commit()
                try:
                    rep_fid = best["representative_frame_id"] or ""
                except (IndexError, KeyError):
                    rep_fid = ""
                merged = Entity(
                    id=best["id"], name=best["name"], type=best["type"],
                    attributes=old_attrs, aliases=sorted(old_aliases),
                    first_seen=best["first_seen"], last_seen=new_last,
                    seen_count=best["seen_count"] + 1,
                    representative_frame_id=rep_fid, updated_at=time.time(),
                )
                # 有真正变化 (新属性 / 新别名) 才生成 state record
                state: Optional[EntityState] = None
                if attrs_delta or new_alias_diff:
                    state = EntityState(
                        entity_id=merged.id, t_observed=new_last,
                        state_label="refined",
                        attributes_delta=attrs_delta,
                        new_aliases=new_alias_diff,
                        source="writer",
                    )
                return merged, state
            # 新建
            c.execute(
                """INSERT OR REPLACE INTO entities
                   (id, name, type, attributes, aliases, first_seen, last_seen,
                    seen_count, representative_frame_id, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (ent.id, ent.name, ent.type,
                 json.dumps(ent.attributes, ensure_ascii=False),
                 json.dumps(ent.aliases, ensure_ascii=False),
                 ent.first_seen, ent.last_seen, ent.seen_count,
                 ent.representative_frame_id or "", time.time()),
            )
            c.commit()
            state = EntityState(
                entity_id=ent.id, t_observed=ent.first_seen,
                state_label="first_seen",
                attributes_delta=dict(ent.attributes),
                new_aliases=list(ent.aliases),
                source="writer",
            )
            return ent, state

    def insert_edge(self, edge: Edge) -> None:
        with self._lock, self._connect() as c:
            c.execute(
                """INSERT INTO edges
                   (src_id, dst_id, label, rel_type, micro_id, t_observed, metadata)
                   VALUES (?,?,?,?,?,?,?)""",
                (edge.src_id, edge.dst_id, edge.label, edge.rel_type,
                 edge.micro_id, edge.t_observed,
                 json.dumps(edge.metadata, ensure_ascii=False)),
            )
            c.commit()

    # ==================================================================== #
    # ★ Hybrid retrieval (一期): 文本 embedding 读写 + 向量检索原语.
    #
    # 契约:
    #   - embedding 输入文本的构造 (build_micro_embed_text / build_entity_embed_text)
    #     由本类持有, 保证 Writer 首次算 / Reviewer 更新算 / 未来 backfill 三处口径一致.
    #   - update_*_embedding 由 Writer/Reviewer 在异步后台任务里调用, 单条写入.
    #   - vector_search_* 由 MemoryToolBox 在同步 tool.call 里调用 (跑在
    #     asyncio.to_thread 中, 不卡 event loop). 只做纯余弦相似度排序;
    #     RRF 融合逻辑放在 MemoryToolBox 内, 与关键词结果拼接.
    #   - 全链路对 embedding 端点不可用容错: enabled=False 时所有方法 no-op
    #     / 返回空列表, 上层自动降级到关键词兜底.
    # ==================================================================== #
    @staticmethod
    def build_micro_embed_text(mev: "MicroEvent") -> str:
        """Canonical text used to embed a micro event.

        We compose description + SVO triple so a semantic query like "listening
        to something" can match "picking up headphones" even when the exact
        keyword misses. Bounded to 2000 chars for API budget.
        """
        parts: List[str] = []
        desc = (mev.description or "").strip()
        if desc:
            parts.append(desc)
        triple = " ".join(
            x for x in (mev.subject or "", mev.action or "",
                        mev.object or "") if x)
        triple = triple.strip()
        if triple:
            parts.append(triple)
        text = "\n".join(parts).strip()
        return text[:2000] if text else ""

    @staticmethod
    def build_entity_embed_text(ent: "Entity") -> str:
        """Canonical text used to embed an entity: name(type) + aliases + attrs.

        Concatenating attributes ("color=red, material=metal") lets the T→T
        vector channel bridge queries phrased differently from what the LLM
        wrote (e.g. "红色物件" ↔ 'name="unknown", attributes={"color":"red"}').
        """
        name = (ent.name or "").strip()
        etype = (ent.type or "").strip()
        head = f"{name} ({etype})" if etype else name
        parts: List[str] = []
        if head:
            parts.append(head)
        aliases = [a for a in (ent.aliases or []) if a][:8]
        if aliases:
            parts.append("aliases: " + ", ".join(aliases))
        if ent.attributes:
            attrs_str = ", ".join(
                f"{k}={v}" for k, v in list(ent.attributes.items())[:12]
                if v is not None)
            if attrs_str:
                parts.append(attrs_str)
        text = "\n".join(parts).strip()
        return text[:2000] if text else ""

    def update_micro_embedding(self, micro_id: str,
                                vec: Optional[np.ndarray]) -> bool:
        """Persist a float16 BLOB text_embedding on a micro row. Idempotent.

        Returns True when a row was updated (vec valid + row existed). Silently
        no-ops on None / empty / db errors so upstream fire-and-forget callers
        can ignore the return value.
        """
        blob = encode_vector(vec)
        if blob is None or not micro_id:
            return False
        try:
            with self._lock, self._connect() as c:
                cur = c.execute(
                    "UPDATE micro_events SET text_embedding=? WHERE id=?",
                    (blob, micro_id))
                return cur.rowcount > 0
        except Exception as e:
            log.debug("[mem] update_micro_embedding %s failed: %s",
                      micro_id, e)
            return False

    def update_entity_embedding(self, entity_id: str,
                                 vec: Optional[np.ndarray]) -> bool:
        """Persist a float16 BLOB text_embedding on an entity row. Idempotent."""
        blob = encode_vector(vec)
        if blob is None or not entity_id:
            return False
        try:
            with self._lock, self._connect() as c:
                cur = c.execute(
                    "UPDATE entities SET text_embedding=? WHERE id=?",
                    (blob, entity_id))
                return cur.rowcount > 0
        except Exception as e:
            log.debug("[mem] update_entity_embedding %s failed: %s",
                      entity_id, e)
            return False

    def vector_search_micro(
        self, query_vec: np.ndarray, ask_ts: float,
        top_k: int = 30, pool_cap: int = 0,
    ) -> List[Tuple["MicroEvent", float]]:
        """Exact cosine search over the complete visible micro vector corpus.

        ``pool_cap`` is retained for call compatibility but intentionally
        ignored.  The old implementation limited candidates to the 800 most
        recent rows, which made old semantic matches unreachable.  Rows are
        streamed in bounded batches and only the running top-K is retained, so
        recall is complete without materializing every vector at once.
        """
        if query_vec is None or not self.embedding_client.enabled:
            return []
        try:
            with self._connect() as c:
                cur = c.execute(
                    """SELECT * FROM micro_events
                       WHERE t_end <= ? AND text_embedding IS NOT NULL
                         AND (superseded_by IS NULL OR superseded_by='')""",
                    (ask_ts,),
                )
                hits = self._stream_exact_vector_topk(
                    cur, query_vec, top_k=top_k)
        except Exception as e:
            log.debug("[mem] vector_search_micro sql failed: %s", e)
            return []
        return [
            (self._row_to_micro(row), sim) for row, sim in hits
        ]

    def vector_search_entity(
        self, query_vec: np.ndarray, ask_ts: float,
        top_k: int = 30, pool_cap: int = 0,
    ) -> List[Tuple["Entity", float]]:
        """Complete exact entity-vector search (``pool_cap`` is ignored)."""
        if query_vec is None or not self.embedding_client.enabled:
            return []
        try:
            with self._connect() as c:
                cur = c.execute(
                    """SELECT * FROM entities
                       WHERE first_seen <= ? AND text_embedding IS NOT NULL
                         AND (merged_into IS NULL OR merged_into='')""",
                    (ask_ts,),
                )
                hits = self._stream_exact_vector_topk(
                    cur, query_vec, top_k=top_k)
        except Exception as e:
            log.debug("[mem] vector_search_entity sql failed: %s", e)
            return []
        return [(self._row_to_entity(row), sim) for row, sim in hits]

    @staticmethod
    def _stream_exact_vector_topk(
        cursor: sqlite3.Cursor, query_vec: np.ndarray, *, top_k: int,
        embedding_col: str = "text_embedding", batch_size: int = 512,
    ) -> List[Tuple[sqlite3.Row, float]]:
        """Stream a complete SQLite vector corpus and retain exact top-K."""
        q = np.asarray(query_vec, dtype=np.float32)
        qn = float(np.linalg.norm(q))
        if qn > 1e-8:
            q = q / qn
        keep_n = max(1, int(top_k))
        candidates: List[Tuple[sqlite3.Row, float]] = []
        while True:
            rows = cursor.fetchmany(max(1, int(batch_size)))
            if not rows:
                break
            vecs: List[np.ndarray] = []
            kept: List[sqlite3.Row] = []
            for row in rows:
                vec = decode_vector(row[embedding_col])
                # A model/dimension migration can leave old incompatible BLOBs.
                if vec is None or vec.size != q.size:
                    continue
                vecs.append(vec)
                kept.append(row)
            if not vecs:
                continue
            sims = np.stack(vecs, axis=0).astype(np.float32) @ q
            local_idx = np.argsort(-sims)[:keep_n]
            candidates.extend(
                (kept[int(i)], float(sims[int(i)])) for i in local_idx)
            if len(candidates) > keep_n * 4:
                candidates.sort(key=lambda item: item[1], reverse=True)
                del candidates[keep_n:]
        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates[:keep_n]

    # ==================================================================== #
    # ★ 二期 (frame image embedding): frame_embeddings 读写 + T→I 向量检索.
    #   与上面 text 向量同套约定: float16 BLOB / numpy 暴力余弦 / 防脏读 /
    #   失败静默返回空. 独立语义空间 (multimodal-embedding-v1), query 必须走
    #   mm_embedding_client.embed_text, 不能混用 text-embedding-v3 的向量.
    # ==================================================================== #
    def has_frame_embedding(self, frame_id: str) -> bool:
        """该帧是否已有图像向量 (Writer 据此避免重复算 → 省钱)."""
        if not frame_id:
            return False
        try:
            with self._connect() as c:
                row = c.execute(
                    "SELECT 1 FROM frame_embeddings WHERE frame_id=? LIMIT 1",
                    (frame_id,)).fetchone()
                return row is not None
        except Exception:
            return False

    def insert_frame_embedding(
        self, frame_id: str, t_observed: float,
        micro_id: Optional[str], vec: Optional[np.ndarray],
    ) -> bool:
        """写入一帧的图像向量 (INSERT OR IGNORE: 已存在不重写)."""
        blob = encode_vector(vec)
        if blob is None or not frame_id:
            return False
        try:
            with self._lock, self._connect() as c:
                cur = c.execute(
                    """INSERT OR IGNORE INTO frame_embeddings
                       (frame_id, t_observed, micro_id, embedding, created_at)
                       VALUES (?,?,?,?,?)""",
                    (frame_id, float(t_observed or 0.0), micro_id or None,
                     blob, time.time()))
                return cur.rowcount > 0
        except Exception as e:
            log.debug("[mem] insert_frame_embedding %s failed: %s", frame_id, e)
            return False

    def vector_search_frames(
        self, query_vec: np.ndarray, ask_ts: float,
        top_k: int = 8, pool_cap: int = 0,
    ) -> List[Dict[str, Any]]:
        """T→I 跨模态检索: query 文本向量 vs 全部关键帧图像向量.

        返回 [{"frame_id", "t_observed", "micro_id", "sim"}] 按相似度降序.
        防脏读: t_observed <= ask_ts. 帧向量与 JPEG 生命周期解耦 (帧被 LRU
        淘汰后向量仍可召回, 由调用方自行处理"命中但图没了"的情况).
        """
        if query_vec is None or not self.mm_embedding_client.enabled:
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        qn = float(np.linalg.norm(q))
        if qn > 1e-8:
            q = q / qn
        keep_n = max(1, int(top_k))
        candidates: List[Dict[str, Any]] = []
        try:
            with self._connect() as c:
                sql = (
                    "SELECT frame_id,t_observed,micro_id,embedding "
                    "FROM frame_embeddings WHERE t_observed <= ? "
                    "ORDER BY t_observed DESC"
                )
                args: Tuple[Any, ...] = (float(ask_ts),)
                if int(pool_cap or 0) > 0:
                    sql += " LIMIT ?"
                    args = (float(ask_ts), int(pool_cap))
                cur = c.execute(sql, args)
                while True:
                    rows = cur.fetchmany(512)
                    if not rows:
                        break
                    vecs: List[np.ndarray] = []
                    kept: List[sqlite3.Row] = []
                    for r in rows:
                        v = decode_vector(r["embedding"])
                        if v is None or v.size != q.size:
                            continue
                        vecs.append(v)
                        kept.append(r)
                    if not vecs:
                        continue
                    sims = np.stack(vecs, axis=0).astype(np.float32) @ q
                    local_idx = np.argsort(-sims)[:keep_n]
                    for i in local_idx:
                        row = kept[int(i)]
                        candidates.append({
                            "frame_id": row["frame_id"],
                            "t_observed": float(row["t_observed"] or 0.0),
                            "micro_id": row["micro_id"],
                            "sim": float(sims[int(i)]),
                        })
                    if len(candidates) > keep_n * 4:
                        candidates.sort(key=lambda h: h["sim"], reverse=True)
                        del candidates[keep_n:]
        except Exception as e:
            log.debug("[mem] vector_search_frames sql failed: %s", e)
            return []
        candidates.sort(key=lambda h: h["sim"], reverse=True)
        return candidates[:keep_n]

    # ==================================================================== #
    # ★ OMNI-Q: entity_quotes CRUD (PERSON 对白, omni 从原始音频产出)
    # ==================================================================== #
    def insert_quote(self, q: EntityQuote, *,
                     dedup_ts_tol: float = 1.0,
                     dedup_text_sim: float = 0.85) -> Optional[int]:
        """Insert one quote with dedup. On duplicate, returns the existing id
        without inserting. Duplicate criterion: same entity_id, |t_start diff|
        <= dedup_ts_tol, and text similarity >= dedup_text_sim. Returns None if
        entity_id or text is empty, else the row id."""
        if not q.entity_id or not q.text:
            return None
        with self._lock, self._connect() as c:
            rows = c.execute(
                """SELECT id, text FROM entity_quotes
                   WHERE entity_id=? AND ABS(t_start - ?) <= ?""",
                (q.entity_id, q.t_start, dedup_ts_tol),
            ).fetchall()
            for r in rows:
                if _fuzzy_ratio(r["text"], q.text) >= dedup_text_sim:
                    return int(r["id"])   # 已存在等价 quote, skip
            cur = c.execute(
                """INSERT INTO entity_quotes
                   (entity_id, t_start, t_end, text, confidence,
                    evidence_frame_ids, micro_id, macro_id, source, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (q.entity_id, q.t_start, q.t_end, q.text, q.confidence,
                 json.dumps(q.evidence_frame_ids or [], ensure_ascii=False),
                 q.micro_id, q.macro_id, q.source,
                 q.created_at or time.time()),
            )
            c.commit()
            return int(cur.lastrowid)

    def has_any_quotes(self) -> bool:
        """True iff entity_quotes has at least one row (ignores ask_ts / D3).

        Speaker-attributed quotes are a **reserved** capability: the schema,
        indexes and merge-time entity_id rewrite are all in place, but the only
        writer (`insert_quote`) has no caller yet — it waits on face-ID /
        voiceprint-ID landing, which is what will supply the speaker entity per
        utterance. Until then the table is structurally empty, and the two quote
        recall tools can only ever return "(empty)".

        MemoryToolBox uses this to tell the Recall LLM the difference between
        "no quote matched your query" (worth rewording) and "this capability is
        not wired yet" (never worth retrying). Cheap: LIMIT 1 on a rowid table.
        """
        try:
            with self._lock, self._connect() as c:
                row = c.execute(
                    "SELECT 1 FROM entity_quotes LIMIT 1").fetchone()
            return row is not None
        except Exception:
            # Missing table / closed db must not break a recall round; behave
            # as "not populated" so the caller emits the do-not-retry hint.
            return False

    def get_quotes_by_entity(self, entity_id: str, ask_ts: float, *,
                             t_window: Optional[Tuple[float, float]] = None,
                             top_k: int = 10) -> List[EntityQuote]:
        """Quotes said by an entity (ordered t_start DESC). D3 anti-dirty-read:
        WHERE t_end <= ask_ts; superseded quotes excluded. Optional t_window
        further bounds t_start."""
        if not entity_id:
            return []
        with self._lock, self._connect() as c:
            where = ["entity_id=?", "t_end <= ?",
                     "(superseded_by IS NULL OR superseded_by=0)"]
            args: List[Any] = [entity_id, ask_ts]
            if t_window:
                where.append("t_start >= ? AND t_start <= ?")
                args.extend([t_window[0], t_window[1]])
            sql = (f"SELECT * FROM entity_quotes WHERE {' AND '.join(where)} "
                   f"ORDER BY t_start DESC LIMIT ?")
            args.append(top_k)
            rows = c.execute(sql, args).fetchall()
        return [self._row_to_quote(r) for r in rows]

    def search_quotes_by_text(self, query: str, ask_ts: float, *,
                              top_k: int = 5,
                              t_window: Optional[Tuple[float, float]] = None,
                              exclude_unknown: bool = False) -> List[Dict[str, Any]]:
        """Search quotes by text (LIKE), joining the speaker entity. D3:
        WHERE t_end <= ask_ts; excludes superseded quotes and merged entities.
        exclude_unknown drops ent_unknown_speaker. Returns
        [{"quote": EntityQuote, "entity_name", "entity_type"}, ...]."""
        if not query:
            return []
        like = f"%{query}%"
        with self._lock, self._connect() as c:
            where = ["q.text LIKE ?", "q.t_end <= ?",
                     "(q.superseded_by IS NULL OR q.superseded_by=0)",
                     "(e.merged_into IS NULL OR e.merged_into='')"]
            args: List[Any] = [like, ask_ts]
            if exclude_unknown:
                where.append("q.entity_id != 'ent_unknown_speaker'")
            if t_window:
                where.append("q.t_start >= ? AND q.t_start <= ?")
                args.extend([t_window[0], t_window[1]])
            sql = (f"SELECT q.*, e.name AS e_name, e.type AS e_type "
                   f"FROM entity_quotes q JOIN entities e ON q.entity_id = e.id "
                   f"WHERE {' AND '.join(where)} "
                   f"ORDER BY q.t_start DESC LIMIT ?")
            args.append(top_k)
            rows = c.execute(sql, args).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            q = self._row_to_quote(r)
            out.append({
                "quote": q, "entity_name": r["e_name"],
                "entity_type": r["e_type"],
            })
        return out

    @staticmethod
    def _row_to_quote(r: sqlite3.Row) -> EntityQuote:
        def _g(name: str, default=None):
            try:
                return r[name]
            except (IndexError, KeyError):
                return default
        return EntityQuote(
            id=int(_g("id", 0) or 0),
            entity_id=_g("entity_id", "") or "",
            t_start=float(_g("t_start", 0.0) or 0.0),
            t_end=float(_g("t_end", 0.0) or 0.0),
            text=_g("text", "") or "",
            confidence=float(_g("confidence", 1.0) or 1.0),
            evidence_frame_ids=json.loads(_g("evidence_frame_ids") or "[]"),
            micro_id=_g("micro_id"),
            macro_id=_g("macro_id"),
            source=_g("source", "omni") or "omni",
            created_at=float(_g("created_at", 0.0) or 0.0),
            superseded_by=_g("superseded_by"),
        )

    # ==================================================================== #
    # ★ OMNI-R: entity_rep_frames CRUD (PERSON 多代表帧 top-K)
    # ==================================================================== #
    def upsert_rep_frame(self, rf: RepFrame, *, max_per_entity: int = 5) -> bool:
        """Add/update a representative frame; when at max_per_entity, evict the
        lowest-quality one (only if the new frame beats the worst by >0.05, an
        anti-flap margin). An existing (entity_id, frame_id) has its quality
        raised to max(old, new). Returns True if written (incl. update/replace),
        False if dropped (quality not high enough, or empty ids)."""
        if not rf.entity_id or not rf.frame_id:
            return False
        with self._lock, self._connect() as c:
            old = c.execute(
                "SELECT quality_score FROM entity_rep_frames "
                "WHERE entity_id=? AND frame_id=?",
                (rf.entity_id, rf.frame_id),
            ).fetchone()
            if old is not None:
                new_q = max(float(old["quality_score"] or 0.0), rf.quality_score)
                c.execute(
                    "UPDATE entity_rep_frames SET quality_score=?, note=? "
                    "WHERE entity_id=? AND frame_id=?",
                    (new_q, rf.note or "", rf.entity_id, rf.frame_id),
                )
                self._sync_legacy_rep_frame_inline(c, rf.entity_id)
                c.commit()
                return True
            rows = c.execute(
                "SELECT frame_id, quality_score FROM entity_rep_frames "
                "WHERE entity_id=? ORDER BY quality_score DESC",
                (rf.entity_id,),
            ).fetchall()
            if len(rows) < max_per_entity:
                c.execute(
                    """INSERT INTO entity_rep_frames
                       (entity_id, frame_id, quality_score, note, added_at, added_by)
                       VALUES (?,?,?,?,?,?)""",
                    (rf.entity_id, rf.frame_id, rf.quality_score,
                     rf.note or "", rf.added_at, rf.added_by),
                )
                self._sync_legacy_rep_frame_inline(c, rf.entity_id)
                c.commit()
                return True
            worst_score = float(rows[-1]["quality_score"] or 0.0)
            if rf.quality_score > worst_score + 0.05:   # 防抖
                c.execute(
                    "DELETE FROM entity_rep_frames WHERE entity_id=? AND frame_id=?",
                    (rf.entity_id, rows[-1]["frame_id"]),
                )
                c.execute(
                    """INSERT INTO entity_rep_frames
                       (entity_id, frame_id, quality_score, note, added_at, added_by)
                       VALUES (?,?,?,?,?,?)""",
                    (rf.entity_id, rf.frame_id, rf.quality_score,
                     rf.note or "", rf.added_at, rf.added_by),
                )
                self._sync_legacy_rep_frame_inline(c, rf.entity_id)
                c.commit()
                return True
        return False

    @staticmethod
    def _sync_legacy_rep_frame_inline(c: sqlite3.Connection, entity_id: str) -> None:
        """Within the same transaction, sync the top-1 rep_frame into
        entities.representative_frame_id, as a fallback for legacy Recall paths
        that read that single field."""
        r = c.execute(
            "SELECT frame_id FROM entity_rep_frames WHERE entity_id=? "
            "ORDER BY quality_score DESC LIMIT 1",
            (entity_id,),
        ).fetchone()
        if r is None:
            return
        c.execute(
            "UPDATE entities SET representative_frame_id=? WHERE id=?",
            (r["frame_id"], entity_id),
        )

    def get_rep_frames(self, entity_id: str, top_k: int = 3) -> List[RepFrame]:
        """Top-K representative frames for an entity (ordered quality DESC)."""
        if not entity_id:
            return []
        with self._lock, self._connect() as c:
            rows = c.execute(
                "SELECT * FROM entity_rep_frames WHERE entity_id=? "
                "ORDER BY quality_score DESC LIMIT ?",
                (entity_id, top_k),
            ).fetchall()
        out: List[RepFrame] = []
        for r in rows:
            out.append(RepFrame(
                entity_id=r["entity_id"], frame_id=r["frame_id"],
                quality_score=float(r["quality_score"] or 0.5),
                note=r["note"] or "",
                added_at=float(r["added_at"] or 0.0),
                added_by=r["added_by"] or "writer",
            ))
        return out

    # ----- entity ↔ event 关联 (用户构想: object 多对多 event) -----
    def link_entity_event(self, entity_id: str, micro_id: str,
                          t_observed: float) -> None:
        """Record an "entity appears in micro" link (idempotent, INSERT OR
        IGNORE). No-op if either id is empty."""
        if not entity_id or not micro_id:
            return
        with self._lock, self._connect() as c:
            c.execute(
                """INSERT OR IGNORE INTO entity_event
                   (entity_id, micro_id, t_observed) VALUES (?,?,?)""",
                (entity_id, micro_id, t_observed),
            )
            c.commit()

    def link_entity_frame(self, entity_id: str, frame_id: str,
                          micro_id: Optional[str], t_observed: float) -> None:
        """Record a frame-level "entity clearly appears in this key frame" link
        (idempotent). Called by MemoryWriter per the LLM's key_frames[i].entities
        to bind object↔frame precisely. No-op if entity_id or frame_id is empty."""
        if not entity_id or not frame_id:
            return
        with self._lock, self._connect() as c:
            c.execute(
                """INSERT OR IGNORE INTO entity_frame
                   (entity_id, frame_id, micro_id, t_observed) VALUES (?,?,?,?)""",
                (entity_id, frame_id, micro_id, t_observed),
            )
            c.commit()

    def set_entity_representative_frame(self, entity_id: str, frame_id: str,
                                        *, only_if_empty: bool = True) -> None:
        """Set an entity's representative frame (full-frame frame_id).
        only_if_empty=True writes only when none is set yet (keeps the object's
        first-appearance image); False forces overwrite to the latest. No-op if
        either id is empty."""
        if not entity_id or not frame_id:
            return
        with self._lock, self._connect() as c:
            if only_if_empty:
                c.execute(
                    """UPDATE entities SET representative_frame_id=?
                       WHERE id=? AND (representative_frame_id IS NULL
                                       OR representative_frame_id='')""",
                    (frame_id, entity_id),
                )
            else:
                c.execute(
                    "UPDATE entities SET representative_frame_id=? WHERE id=?",
                    (frame_id, entity_id),
                )
            c.commit()

    def get_events_by_entity(self, entity_id: str, ask_ts: float,
                             limit: int = 20) -> List[MicroEvent]:
        """object → micro events it appeared in (t_end DESC, D3: t_end <= ask_ts).

        Pure-read path, holds no self._lock: WAL supports concurrent readers, and
        a Python-level lock would serialize Recall's parallel tools (see
        RecallWorker.run's asyncio.gather). Write paths + _pending_* still lock."""
        with self._connect() as c:
            cur = c.execute(
                """SELECT m.* FROM micro_events m
                   JOIN entity_event ee ON ee.micro_id = m.id
                   WHERE ee.entity_id = ? AND m.t_end <= ?
                     AND (m.superseded_by IS NULL OR m.superseded_by='')
                   ORDER BY m.t_end DESC LIMIT ?""",
                (entity_id, ask_ts, limit),
            )
            return [self._row_to_micro(r) for r in cur.fetchall()]

    def get_frames_by_entity(self, entity_id: str, ask_ts: float,
                             limit_events: int = 20) -> List[str]:
        """object → the key-frame frame_ids it actually appeared in (deduped).
        This is the query that fixes "search_entity found the object but returns
        no image".

        v2 (frame-level precise): prefers the entity_frame links (frames the LLM
        explicitly marked as clearly showing this object), returning only frames
        truly bound to it — not, as the old version did, every frame of the whole
        micro (many not containing the object). representative_frame_id is placed
        first. Fallback: if the entity has no entity_frame rows (legacy data /
        no frame bound this tick), falls back to the segment-level frame_ids from
        get_events_by_entity so it never comes back empty.

        Pure-read path, holds no self._lock (same reason as get_events_by_entity)."""
        out: List[str] = []
        seen: Set[str] = set()
        # 代表帧优先
        with self._connect() as c:
            r = c.execute(
                "SELECT representative_frame_id FROM entities WHERE id=?",
                (entity_id,),
            ).fetchone()
            rep = (r["representative_frame_id"] if r else None) or ""
            if rep:
                seen.add(rep); out.append(rep)
            # 帧级精确关联 (新): 该 object 被显式指认的关键帧, 按时间倒序
            cur = c.execute(
                """SELECT frame_id FROM entity_frame
                   WHERE entity_id = ? AND t_observed <= ?
                   ORDER BY t_observed DESC LIMIT ?""",
                (entity_id, ask_ts, max(1, limit_events) * 4),
            )
            for row in cur.fetchall():
                fid = row["frame_id"]
                if fid and fid not in seen:
                    seen.add(fid); out.append(fid)
        # 回退: 没有帧级关联 → 走旧的段级 frame_ids (兼容旧数据)
        if len(out) <= (1 if rep else 0):
            events = self.get_events_by_entity(entity_id, ask_ts,
                                               limit=limit_events)
            for ev in events:
                for fid in ev.frame_ids:
                    if fid not in seen:
                        seen.add(fid); out.append(fid)
        return out

    def get_entities_by_micro(self, micro_id: str, ask_ts: float,
                              limit: int = 30) -> List[Entity]:
        """event → the objects appearing in it (reverse lookup, for Inspector/
        tools). D3: e.first_seen <= ask_ts. Pure-read path, holds no self._lock."""
        with self._connect() as c:
            cur = c.execute(
                """SELECT e.* FROM entities e
                   JOIN entity_event ee ON ee.entity_id = e.id
                   WHERE ee.micro_id = ? AND e.first_seen <= ?
                     AND (e.merged_into IS NULL OR e.merged_into='')
                   ORDER BY e.last_seen DESC LIMIT ?""",
                (micro_id, ask_ts, limit),
            )
            return [self._row_to_entity(r) for r in cur.fetchall()]

    # ----- 读取 (D3: 所有读 API 强制 ask_ts) -----
    # ★ 锁策略: 全部纯读 API 都不再持 self._lock, 让 Recall 的 asyncio.gather
    #   多 tool 真正并发到 SQLite (WAL 模式下天生多读并发). 写 path 仍持锁.
    def get_micro_by_time(self, t_start: float, t_end: float,
                          ask_ts: float, limit: int = 50) -> List[MicroEvent]:
        assert t_end <= ask_ts + 1e-3, f"防脏读: t_end={t_end} > ask_ts={ask_ts}"
        with self._connect() as c:
            cur = c.execute(
                """SELECT * FROM micro_events
                   WHERE t_start >= ? AND t_end <= ? AND t_end <= ?
                     AND (superseded_by IS NULL OR superseded_by='')
                   ORDER BY t_end DESC LIMIT ?""",
                (t_start, t_end, ask_ts, limit),
            )
            return [self._row_to_micro(r) for r in cur.fetchall()]

    def get_micros_overlapping_time(
        self, t_start: float, t_end: float, ask_ts: float, limit: int = 20,
    ) -> List[MicroEvent]:
        """Return valid micros overlapping a temporal evidence window."""
        safe_end = min(float(t_end), float(ask_ts))
        with self._connect() as c:
            rows = c.execute(
                """SELECT * FROM micro_events
                   WHERE t_start <= ? AND t_end >= ? AND t_end <= ?
                     AND (superseded_by IS NULL OR superseded_by='')
                   ORDER BY ABS(((t_start + t_end) / 2.0) - ?) ASC
                   LIMIT ?""",
                (safe_end, max(0.0, float(t_start)), float(ask_ts),
                 (max(0.0, float(t_start)) + safe_end) / 2.0,
                 max(1, int(limit))),
            ).fetchall()
        return [self._row_to_micro(r) for r in rows]

    def search_micro_by_keyword(self, query: str, ask_ts: float,
                                top_k: int = 5) -> List[MicroEvent]:
        """Keyword recall over micro events.

        ★ Upgraded from single verbatim `%query%` to OR-tokenized + synonym-
        expanded matching with relevance ranking (distinct-term hits × field
        weight, blended with recency) instead of pure recency truncation.
        Empty/degenerate query → falls back to the most recent events (guards
        the old `%`-matches-everything behavior against noise).
        """
        base_terms = mm_tokenize_query(query)
        if not base_terms:
            # Degenerate query: return most recent as a mild fallback (no LIKE
            # explosion). Callers rarely hit this (Router writes real queries).
            with self._connect() as c:
                cur = c.execute(
                    """SELECT * FROM micro_events WHERE t_end <= ?
                       AND (superseded_by IS NULL OR superseded_by='')
                       ORDER BY t_end DESC LIMIT ?""",
                    (ask_ts, top_k),
                )
                return [self._row_to_micro(r) for r in cur.fetchall()]
        # Candidate pool: any variant hitting any field. Cap generously so
        # scoring has room but SQLite stays cheap.
        fields = ("description", "subject", "object", "action")
        # Field weights: description/subject/object are richer signal than action.
        fw = {"description": 3.0, "subject": 2.5, "object": 2.0, "action": 1.0}
        pool_cap = max(top_k * 8, 120)

        def _fetch_pool(groups: List[List[str]],
                        all_variants: List[str]) -> List[sqlite3.Row]:
            """候选池: 任一变体命中任一字段, 按粗相关性 (而非时间) 截断。

            SQL 构造见 mm_keyword_pool_sql (与 search_entity_by_keyword 共用,
            那里有为什么必须把相关性下推到 ORDER BY 的说明)。
            """
            where_or, likes, order_by, rank_params = mm_keyword_pool_sql(
                fields, fw, groups, all_variants, recency_col="t_end")
            if not where_or:
                return []
            with self._connect() as c:
                return c.execute(
                    f"""SELECT * FROM micro_events
                        WHERE t_end <= ?
                          AND (superseded_by IS NULL OR superseded_by='')
                          AND ({where_or})
                        ORDER BY {order_by} LIMIT ?""",
                    (ask_ts, *likes, *rank_params, pool_cap),
                ).fetchall()

        groups = mm_expand_groups(base_terms)        # scoring by concept
        rows = _fetch_pool(groups, mm_expand_terms(base_terms))
        if not rows:
            # ★ CJK 兜底: mm_tokenize_query 把整块中文保留为一个 token, 所以
            #   "红色外套的男人把背包放哪了" 这种自然提问会要求整句原样出现在
            #   字段里 —— 永不命中, 关键词路静默返回空, 混合检索退化成纯向量。
            #   严格匹配无果时降级到字二元组, 让打分按覆盖了几个二元组来排序。
            bigram_groups = mm_cjk_bigram_groups(base_terms)
            if bigram_groups:
                groups = bigram_groups
                rows = _fetch_pool(
                    bigram_groups, [v for g in bigram_groups for v in g])
                if rows:
                    log.debug(
                        "[mem kw] 严格匹配为空, 已降级到 CJK 二元组: %d 组 → %d 行",
                        len(bigram_groups), len(rows))
        scored = []
        for r in rows:
            blob = {f: str(r[f] or "").lower() for f in fields}
            hit_groups = 0
            score = 0.0
            for grp in groups:
                # A concept is "hit" if ANY of its synonym variants match ANY
                # field; count it once (not once per variant).
                grp_hit = False
                for f in fields:
                    if any(v in blob[f] for v in grp):
                        score += fw[f]
                        grp_hit = True
                if grp_hit:
                    hit_groups += 1
            # Distinct-concept coverage dominates; recency is a light tiebreaker.
            score += 2.0 * hit_groups
            scored.append((hit_groups, score, r["t_end"], r))
        scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        return [self._row_to_micro(r) for _, _, _, r in scored[:top_k]]

    def search_entity_by_keyword(self, query: str, ask_ts: float,
                                 top_k: int = 5) -> List[Entity]:
        """Keyword recall over entities — OR-tokenized + synonym-expanded +
        relevance-ranked (see search_micro_by_keyword)."""
        base_terms = mm_tokenize_query(query)
        if not base_terms:
            with self._connect() as c:
                cur = c.execute(
                    """SELECT * FROM entities WHERE first_seen <= ?
                       AND (merged_into IS NULL OR merged_into='')
                       ORDER BY last_seen DESC LIMIT ?""",
                    (ask_ts, top_k),
                )
                return [self._row_to_entity(r) for r in cur.fetchall()]
        fields = ("name", "aliases", "attributes")
        fw = {"name": 3.0, "aliases": 2.5, "attributes": 1.5}
        pool_cap = max(top_k * 6, 60)

        def _fetch_pool(groups: List[List[str]],
                        all_variants: List[str]) -> List[sqlite3.Row]:
            where_or, likes, order_by, rank_params = mm_keyword_pool_sql(
                fields, fw, groups, all_variants, recency_col="last_seen")
            if not where_or:
                return []
            with self._connect() as c:
                return c.execute(
                    f"""SELECT * FROM entities
                        WHERE first_seen <= ?
                          AND (merged_into IS NULL OR merged_into='')
                          AND ({where_or})
                        ORDER BY {order_by} LIMIT ?""",
                    (ask_ts, *likes, *rank_params, pool_cap),
                ).fetchall()

        groups = mm_expand_groups(base_terms)
        rows = _fetch_pool(groups, mm_expand_terms(base_terms))
        if not rows:
            # ★ FIX 2026-08-19: CJK 二元组兜底。此前只有 search_micro_by_keyword
            #   有这一层, 而 search_entity 是 RECALL_SYSTEM "Standard object
            #   lookup" 的**第一步** —— mm_tokenize_query 把整块中文保留为一个
            #   token, 所以"红色外套的男人把背包放哪了"这种自然提问会要求整句原样
            #   出现在 name/aliases/attributes 里, 永不命中。结果是整条实体链的
            #   关键词臂静默返回空, RRF 只剩向量单臂 (且 _RRF_VEC_MIN_SIM=0.25
            #   还会再滤掉一批), 而调用方看不到任何降级信号。
            bigram_groups = mm_cjk_bigram_groups(base_terms)
            if bigram_groups:
                groups = bigram_groups
                rows = _fetch_pool(
                    bigram_groups, [v for g in bigram_groups for v in g])
                if rows:
                    log.debug(
                        "[mem kw] entity 严格匹配为空, 已降级到 CJK 二元组: "
                        "%d 组 → %d 行", len(bigram_groups), len(rows))
        scored = []
        for r in rows:
            blob = {f: str(r[f] or "").lower() for f in fields}
            hit_groups = 0
            score = 0.0
            for grp in groups:
                grp_hit = False
                for f in fields:
                    if any(v in blob[f] for v in grp):
                        score += fw[f]
                        grp_hit = True
                if grp_hit:
                    hit_groups += 1
            score += 2.0 * hit_groups
            scored.append((hit_groups, score, r["last_seen"], r))
        scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        return [self._row_to_entity(r) for _, _, _, r in scored[:top_k]]

    def get_recent_entities(self, ask_ts: float, limit: int = 20) -> List[Entity]:
        with self._connect() as c:
            cur = c.execute(
                """SELECT * FROM entities WHERE first_seen <= ?
                   AND (merged_into IS NULL OR merged_into='')
                   ORDER BY last_seen DESC LIMIT ?""",
                (ask_ts, limit),
            )
            return [self._row_to_entity(r) for r in cur.fetchall()]

    def get_recent_macros(self, ask_ts: float, limit: int = 3) -> List[MacroEvent]:
        with self._connect() as c:
            cur = c.execute(
                """SELECT * FROM macro_events WHERE t_end <= ?
                   AND (superseded_by IS NULL OR superseded_by='')
                   ORDER BY t_end DESC LIMIT ?""",
                (ask_ts, limit),
            )
            return [self._row_to_macro(r) for r in cur.fetchall()]

    # ----- ★ L3 召回读者。写入侧 (aggregate_l3) 一直在花 agg_l3_frames 张图的
    #   LLM 成本产出 super_events, 但读者只有 dashboard 的裸 SQL
    #   (hermes_cli/web_server.py) 和 dump_all (无调用方) —— 召回链路上
    #   零读者, 等于纯成本。下面三个 reader 让 L3 进入召回, 并沿用与
    #   L1/L2 完全一致的两条不变量: D3 防脏读 (t_end <= ask_ts) +
    #   软删过滤 (superseded_by 为空)。
    def get_supers_overlapping_time(
        self, t_start: float, t_end: float, ask_ts: float,
        limit: int = 3,
    ) -> List[SuperEvent]:
        """L3 super_events overlapping [t_start, t_end], newest first.

        Overlap (not containment): a super spans minutes, so a point-in-time
        question almost never falls inside one by containment alone.
        """
        with self._connect() as c:
            rows = c.execute(
                """SELECT * FROM super_events
                   WHERE t_end <= ? AND t_start <= ? AND t_end >= ?
                     AND (superseded_by IS NULL OR superseded_by='')
                   ORDER BY t_end DESC LIMIT ?""",
                (ask_ts, float(t_end), float(t_start), max(1, limit)),
            ).fetchall()
        return [self._row_to_super(r) for r in rows]

    def get_recent_supers(self, ask_ts: float,
                          limit: int = 2) -> List[SuperEvent]:
        """The most recent valid L3 narratives (session-level "what's going on")."""
        with self._connect() as c:
            rows = c.execute(
                """SELECT * FROM super_events WHERE t_end <= ?
                     AND (superseded_by IS NULL OR superseded_by='')
                   ORDER BY t_end DESC LIMIT ?""",
                (ask_ts, max(1, limit)),
            ).fetchall()
        return [self._row_to_super(r) for r in rows]

    def get_supers_for_macro_ids(
        self, macro_ids: Sequence[str], ask_ts: float,
        limit: int = 3,
    ) -> List[SuperEvent]:
        """The L3 narratives that own the given L2 macros, newest first.

        Joins on ``macro_events.super_id`` rather than on the ``super_events
        .macro_ids`` JSON blob: ``insert_super`` back-fills that FK for every
        macro inside the super's span, it is a real indexed column, and a LIKE
        over the JSON text would false-positive on id prefixes.
        """
        wanted = [str(m) for m in (macro_ids or []) if m]
        if not wanted:
            return []
        with self._connect() as c:
            ph = ",".join("?" for _ in wanted)
            sids = [r[0] for r in c.execute(
                f"""SELECT DISTINCT super_id FROM macro_events
                    WHERE id IN ({ph}) AND super_id IS NOT NULL
                      AND super_id != ''""",
                wanted,
            ).fetchall()]
            if not sids:
                return []
            ph2 = ",".join("?" for _ in sids)
            rows = c.execute(
                f"""SELECT * FROM super_events
                    WHERE id IN ({ph2}) AND t_end <= ?
                      AND (superseded_by IS NULL OR superseded_by='')
                    ORDER BY t_end DESC LIMIT ?""",
                (*sids, ask_ts, max(1, limit)),
            ).fetchall()
        return [self._row_to_super(r) for r in rows]

    def get_macro(self, macro_id: str, ask_ts: float) -> Optional[MacroEvent]:
        with self._connect() as c:
            cur = c.execute(
                "SELECT * FROM macro_events WHERE id=? AND t_end <= ? "
                "AND (superseded_by IS NULL OR superseded_by='')",
                (macro_id, ask_ts),
            )
            r = cur.fetchone()
            return self._row_to_macro(r) if r else None

    # ----- ★ E9/E10: Writer prompt 注入专用查询 -----
    def get_macros_for_writer(self, ask_ts: float,
                               limit: int = 15) -> List[MacroEvent]:
        """E10: fetch the most recent N valid macros (Reviewer-superseded ones
        filtered out), returned in ascending t_start order.

        Feeds the "history timeline" block of the Writer prompt. Aligned with
        Recall's D3 anti-dirty-read (t_end <= ask_ts).
        """
        with self._connect() as c:
            rows = c.execute(
                """SELECT * FROM macro_events
                   WHERE t_end <= ? AND (superseded_by IS NULL OR superseded_by='')
                   ORDER BY t_end DESC LIMIT ?""",
                (ask_ts, max(1, limit)),
            ).fetchall()
        out = [self._row_to_macro(r) for r in rows]
        out.sort(key=lambda m: m.t_start)
        return out

    def get_micros_for_writer(self, ask_ts: float, *,
                               exclude_t_start_ge: Optional[float] = None,
                               limit: int = 30) -> List[MicroEvent]:
        """E10: fetch the most recent N valid micros (Reviewer-superseded ones
        filtered out), returned in ascending t_start order.

        exclude_t_start_ge: exclude micros in the current pending segment (a
            safety net; the pending segment isn't yet inserted into SQLite so
            this rarely matches, but the guard stays).
        """
        with self._connect() as c:
            sql = """SELECT * FROM micro_events
                     WHERE t_end <= ?
                       AND (superseded_by IS NULL OR superseded_by='')"""
            args: List[Any] = [ask_ts]
            if exclude_t_start_ge is not None:
                sql += " AND t_start < ?"
                args.append(exclude_t_start_ge)
            sql += " ORDER BY t_end DESC LIMIT ?"
            args.append(max(1, limit))
            rows = c.execute(sql, tuple(args)).fetchall()
        out = [self._row_to_micro(r) for r in rows]
        out.sort(key=lambda m: m.t_start)
        return out

    def get_entities_for_writer(
        self, ask_ts: float, *,
        min_seen: int = 2, macro_lookback: int = 8,
        tier1_n: int = 5, tier2_n: int = 10, tier3_n: int = 15,
    ) -> Tuple[List[Entity], List[Entity], List[Entity]]:
        """E9: score + bucket entities into 3 tiers (for Writer prompt injection).

        Admission (loose): merged_into empty + seen_count >= min_seen +
        first_seen <= ask_ts. PRUNED is filtered too (merged_into='PRUNED' is not
        empty).

        Ranking signals (macro's independent LLM endorsement dominates, guarding
        against "ghost entity" positive-feedback pollution):
          - macro_hits = times name/aliases appear in the last macro_lookback
            macro.key_entities  (× 3.0)
          - log(entity_frame_count + 1) × 1.0   (Writer's explicit frame binding)
          - exp(-(ask_ts - last_seen) / 1800) × 1.5   (~30-min recency decay)
          - log(seen_count + 1) × 0.5   (fallback score before any macro exists)

        Returns (tier1, tier2, tier3), each already sorted by total score DESC,
        of sizes up to tier1_n / tier2_n / tier3_n respectively.
        """
        import math
        with self._connect() as c:
            # 1) 候选: 门槛 + 已过滤 merged_into / PRUNED
            rows = c.execute(
                """SELECT * FROM entities
                   WHERE first_seen <= ? AND seen_count >= ?
                     AND (merged_into IS NULL OR merged_into='')
                   ORDER BY last_seen DESC""",
                (ask_ts, max(1, min_seen)),
            ).fetchall()
            if not rows:
                return [], [], []
            cands = [self._row_to_entity(r) for r in rows]

            # 2) 最近 N 个有效 macro 的 key_entities 池 (主信号)
            mac_rows = c.execute(
                """SELECT key_entities FROM macro_events
                   WHERE t_end <= ? AND (superseded_by IS NULL OR superseded_by='')
                   ORDER BY t_end DESC LIMIT ?""",
                (ask_ts, max(1, macro_lookback)),
            ).fetchall()
            macro_keys_pool: List[str] = []
            for r in mac_rows:
                try:
                    ks = json.loads(r["key_entities"] or "[]")
                    if isinstance(ks, list):
                        macro_keys_pool.extend(
                            str(k).strip() for k in ks if str(k).strip())
                except Exception:
                    continue

            # 3) entity_frame 计数 (帧级关联强度)
            ef_counts: Dict[str, int] = {}
            try:
                for r in c.execute(
                    """SELECT entity_id, COUNT(*) AS n FROM entity_frame
                       WHERE t_observed <= ? GROUP BY entity_id""",
                    (ask_ts,),
                ).fetchall():
                    ef_counts[r["entity_id"]] = int(r["n"])
            except Exception:
                pass

        # 4) 打分
        def _score(e: Entity) -> float:
            ent_names = [e.name] + list(e.aliases or [])
            macro_hits = 0
            for k in macro_keys_pool:
                for en in ent_names:
                    if _fuzzy_ratio(k, en) >= 0.85:
                        macro_hits += 1
                        break
            s = macro_hits * 3.0
            s += math.log(ef_counts.get(e.id, 0) + 1) * 1.0
            dt = max(0.0, ask_ts - e.last_seen)
            s += math.exp(-dt / 1800.0) * 1.5
            s += math.log(e.seen_count + 1) * 0.5
            return s

        scored = [(e, _score(e)) for e in cands]
        scored.sort(key=lambda p: p[1], reverse=True)
        n1, n2, n3 = max(0, tier1_n), max(0, tier2_n), max(0, tier3_n)
        tier1 = [e for e, _ in scored[:n1]]
        tier2 = [e for e, _ in scored[n1:n1 + n2]]
        tier3 = [e for e, _ in scored[n1 + n2:n1 + n2 + n3]]
        return tier1, tier2, tier3

    # ----- peek_* by-id (不带 ask_ts 约束, 仅给 Reviewer read-back / debug dump 用) -----
    # ★ Reviewer 在执行修订前需要把"旧值"读出来塞进 action payload, 让前端能精确画
    #   diff (旧 desc → 新 desc 这种). 这些 peek 跟 get_micro_by_time 等读 API 走同
    #   一个 _conn 池, 多读并发安全 (WAL).
    def peek_micro(self, micro_id: str) -> Optional[MicroEvent]:
        with self._connect() as c:
            cur = c.execute(
                "SELECT * FROM micro_events WHERE id=?", (micro_id,))
            r = cur.fetchone()
            return self._row_to_micro(r) if r else None

    def peek_macro(self, macro_id: str) -> Optional[MacroEvent]:
        with self._connect() as c:
            cur = c.execute(
                "SELECT * FROM macro_events WHERE id=?", (macro_id,))
            r = cur.fetchone()
            return self._row_to_macro(r) if r else None

    def peek_entity(self, entity_id: str) -> Optional[Entity]:
        with self._connect() as c:
            cur = c.execute(
                "SELECT * FROM entities WHERE id=?", (entity_id,))
            r = cur.fetchone()
            return self._row_to_entity(r) if r else None

    def get_subgraph_for_macro(self, macro_id: str, ask_ts: float) -> Dict[str, Any]:
        """Return a macro's full subgraph: its micros + entities alive during the
        macro's time span + edges with t_observed inside that span. Pure-read
        path, holds no self._lock."""
        mac = self.get_macro(macro_id, ask_ts)
        if mac is None:
            return {"macro": None, "micros": [], "entities": [], "edges": []}
        micros = self.get_micro_by_time(mac.t_start, mac.t_end, ask_ts, limit=100)
        with self._connect() as c:
            cur = c.execute(
                """SELECT * FROM entities
                   WHERE first_seen <= ? AND last_seen >= ?
                     AND (merged_into IS NULL OR merged_into='')
                   ORDER BY last_seen DESC LIMIT 20""",
                (mac.t_end, mac.t_start),
            )
            entities = [self._row_to_entity(r) for r in cur.fetchall()]
            cur = c.execute(
                """SELECT * FROM edges
                   WHERE t_observed >= ? AND t_observed <= ?
                   ORDER BY t_observed""",
                (mac.t_start, mac.t_end),
            )
            edges = [self._row_to_edge(r) for r in cur.fetchall()]
        return {"macro": mac, "micros": micros,
                "entities": entities, "edges": edges}

    def get_relations(self, node_id: str, ask_ts: float,
                      max_hops: int = 1, limit_per_hop: int = 10
                      ) -> List[Edge]:
        with self._connect() as c:
            cur = c.execute(
                """SELECT * FROM edges
                   WHERE t_observed <= ? AND (src_id=? OR dst_id=?)
                   ORDER BY t_observed DESC LIMIT ?""",
                (ask_ts, node_id, node_id, limit_per_hop * max(1, max_hops)),
            )
            return [self._row_to_edge(r) for r in cur.fetchall()]

    # pending getters (聚合 worker 用) ★ 纯读 path 不再持 self._lock
    # ★ E4: pending 都排除"已软删除"(superseded_by 非空) 的, 避免被重复聚合.
    def pending_micros_for_l2(self, ask_ts: float) -> List[MicroEvent]:
        """Micros not yet aggregated into a macro (macro_id IS NULL), in time
        order. D3: t_end <= ask_ts; soft-deleted (superseded) rows excluded."""
        with self._connect() as c:
            cur = c.execute(
                """SELECT * FROM micro_events
                   WHERE macro_id IS NULL AND t_end <= ?
                   AND (superseded_by IS NULL OR superseded_by='')
                   ORDER BY t_start""",
                (ask_ts,),
            )
            return [self._row_to_micro(r) for r in cur.fetchall()]

    def pending_macros_for_l3(self, ask_ts: float) -> List[MacroEvent]:
        with self._connect() as c:
            cur = c.execute(
                """SELECT * FROM macro_events
                   WHERE super_id IS NULL AND t_end <= ?
                   AND (superseded_by IS NULL OR superseded_by='')
                   ORDER BY t_start""",
                (ask_ts,),
            )
            return [self._row_to_macro(r) for r in cur.fetchall()]

    # ========================================================== #
    # ★ E1/E2 (evolve) 新增表的读写: entity_states / revision_log
    # ========================================================== #
    def append_entity_state(self, state: EntityState) -> int:
        """Append one entity evolution-timeline row. Returns the new row id (0 if
        entity_id is empty)."""
        if not state.entity_id:
            return 0
        with self._lock, self._connect() as c:
            cur = c.execute(
                """INSERT INTO entity_states
                   (entity_id, t_observed, state_label, attributes_delta,
                    new_aliases, confidence, evidence_frame_ids, micro_id,
                    source, note)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (state.entity_id, state.t_observed, state.state_label,
                 json.dumps(state.attributes_delta, ensure_ascii=False),
                 json.dumps(state.new_aliases, ensure_ascii=False),
                 state.confidence,
                 json.dumps(state.evidence_frame_ids, ensure_ascii=False),
                 state.micro_id, state.source, state.note),
            )
            c.commit()
            return cur.lastrowid or 0

    def get_entity_states(self, entity_id: str,
                          ask_ts: Optional[float] = None,
                          limit: int = 100) -> List[EntityState]:
        """Read an entity's evolution timeline (ascending t_observed). If ask_ts
        is given, constrains to t_observed <= ask_ts (D3)."""
        with self._connect() as c:
            if ask_ts is not None:
                cur = c.execute(
                    """SELECT * FROM entity_states
                       WHERE entity_id=? AND t_observed <= ?
                       ORDER BY t_observed ASC LIMIT ?""",
                    (entity_id, ask_ts, limit),
                )
            else:
                cur = c.execute(
                    """SELECT * FROM entity_states
                       WHERE entity_id=?
                       ORDER BY t_observed ASC LIMIT ?""",
                    (entity_id, limit),
                )
            return [self._row_to_entity_state(r) for r in cur.fetchall()]

    def append_revision_log(self, rec: RevisionRecord) -> int:
        """Append one Reviewer revision-action record. Returns the new row id."""
        with self._lock, self._connect() as c:
            cur = c.execute(
                """INSERT INTO revision_log
                   (t_applied, reviewer_round, op, target_ids, new_ids,
                    payload, reason, success, error, actor)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (rec.t_applied, rec.reviewer_round, rec.op,
                 json.dumps(rec.target_ids, ensure_ascii=False),
                 json.dumps(rec.new_ids, ensure_ascii=False),
                 json.dumps(rec.payload, ensure_ascii=False),
                 rec.reason, 1 if rec.success else 0,
                 rec.error, rec.actor),
            )
            c.commit()
            return cur.lastrowid or 0

    def get_recent_revisions(self, ask_ts: Optional[float] = None,
                             limit: int = 50,
                             op_filter: Optional[List[str]] = None
                             ) -> List[RevisionRecord]:
        """Read the most recent N revision-log rows (t_applied DESC). If ask_ts
        is given, constrains to t_applied <= ask_ts; op_filter limits to those ops."""
        with self._connect() as c:
            sql = "SELECT * FROM revision_log WHERE 1=1"
            params: List[Any] = []
            if ask_ts is not None:
                sql += " AND t_applied <= ?"; params.append(ask_ts)
            if op_filter:
                placeholders = ",".join("?" * len(op_filter))
                sql += f" AND op IN ({placeholders})"
                params.extend(op_filter)
            sql += " ORDER BY t_applied DESC LIMIT ?"
            params.append(limit)
            cur = c.execute(sql, params)
            return [self._row_to_revision(r) for r in cur.fetchall()]

    # ========================================================== #
    # ★ Reviewer 用的修订动作 (merge / split / refine / rewrite)
    # ★ 所有动作都是"软删除 + 新建", 不真正 DELETE 行, 留链供回溯.
    # ========================================================== #
    def revise_micro_desc(self, micro_id: str, new_desc: str, *,
                          new_subject: Optional[str] = None,
                          new_object: Optional[str] = None,
                          new_action: Optional[str] = None,
                          reviewer_round: int = 0,
                          reason: str = "") -> bool:
        """Edit a micro's description in place (+ optional subject/object/action;
        omitted fields keep their old value). Does NOT move macro_id (keeps its
        original assignment). Bumps revision_count and logs a revision. Returns
        False if the micro doesn't exist."""
        with self._lock, self._connect() as c:
            r = c.execute("SELECT * FROM micro_events WHERE id=?",
                          (micro_id,)).fetchone()
            if r is None:
                return False
            new_subj = new_subject if new_subject is not None else r["subject"]
            new_obj = new_object if new_object is not None else r["object"]
            new_act = new_action if new_action is not None else r["action"]
            old_desc = r["description"] or ""
            c.execute(
                """UPDATE micro_events
                   SET description=?, subject=?, object=?, action=?,
                       revised_at=?, revision_count=COALESCE(revision_count,0)+1
                   WHERE id=?""",
                (new_desc, new_subj, new_obj, new_act, time.time(), micro_id),
            )
            c.commit()
        self.append_revision_log(RevisionRecord(
            t_applied=time.time(), reviewer_round=reviewer_round,
            op="revise_micro_desc", target_ids=[micro_id],
            payload={"old_desc": old_desc[:300], "new_desc": new_desc[:300],
                     "new_subject": new_subj, "new_object": new_obj,
                     "new_action": new_act},
            reason=reason,
        ))
        return True

    def merge_micros(self, micro_ids: List[str], *,
                     new_description: str,
                     new_subject: str = "", new_object: str = "",
                     new_action: str = "",
                     reviewer_round: int = 0,
                     reason: str = "") -> Optional[str]:
        """Merge >=2 micros into one new micro (span = min t_start .. max t_end;
        frame_ids/facts_keys unioned; macro_id = the most common non-empty one).
        The old micros are soft-deleted (superseded_by = new id) and their
        entity_event / entity_frame / edges are rebound to the new id. Returns
        the new micro_id, or None if fewer than 2 valid micros are found."""
        if not micro_ids or len(micro_ids) < 2:
            return None
        with self._lock, self._connect() as c:
            placeholders = ",".join("?" * len(micro_ids))
            rows = c.execute(
                f"SELECT * FROM micro_events WHERE id IN ({placeholders}) "
                "AND (superseded_by IS NULL OR superseded_by='')",
                micro_ids,
            ).fetchall()
            if len(rows) < 2:
                return None
            rows = sorted(rows, key=lambda x: x["t_start"])
            t_start = rows[0]["t_start"]; t_end = rows[-1]["t_end"]
            # 合并 frame_ids / facts_keys / macro_id (取多数)
            merged_fids: List[str] = []
            seen_fid: Set[str] = set()
            merged_facts: List[str] = []
            seen_fact: Set[str] = set()
            macro_ids: List[Optional[str]] = []
            for r in rows:
                for fid in json.loads(r["frame_ids"] or "[]"):
                    if fid not in seen_fid:
                        seen_fid.add(fid); merged_fids.append(fid)
                for k in json.loads(r["facts_keys"] or "[]"):
                    if k not in seen_fact:
                        seen_fact.add(k); merged_facts.append(k)
                macro_ids.append(r["macro_id"])
            # 取出现最多的非空 macro_id
            mid_counter: Dict[str, int] = {}
            for m in macro_ids:
                if m:
                    mid_counter[m] = mid_counter.get(m, 0) + 1
            chosen_macro = (max(mid_counter, key=mid_counter.get)
                            if mid_counter else None)
            new_id = f"micro_merged_{int(t_end * 1000)}_{uuid.uuid4().hex[:6]}"
            c.execute(
                """INSERT INTO micro_events
                   (id, t_start, t_end, description, subject, object, action,
                    macro_id, facts_keys, frame_ids, created_at,
                    revised_at, revision_count)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (new_id, t_start, t_end, new_description,
                 new_subject, new_object, new_action, chosen_macro,
                 json.dumps(merged_facts, ensure_ascii=False),
                 json.dumps(merged_fids, ensure_ascii=False),
                 time.time(), time.time(), 0),
            )
            # 软删除老 micros
            c.execute(
                f"UPDATE micro_events SET superseded_by=?, revised_at=?, "
                f"revision_count=COALESCE(revision_count,0)+1 "
                f"WHERE id IN ({placeholders})",
                [new_id, time.time()] + list(micro_ids),
            )
            # 把老 micros 的 entity_event / entity_frame 关联 rebind 到新 id
            c.execute(
                f"UPDATE OR IGNORE entity_event SET micro_id=? "
                f"WHERE micro_id IN ({placeholders})",
                [new_id] + list(micro_ids),
            )
            c.execute(
                f"UPDATE OR IGNORE entity_frame SET micro_id=? "
                f"WHERE micro_id IN ({placeholders})",
                [new_id] + list(micro_ids),
            )
            c.execute(
                f"UPDATE edges SET micro_id=? "
                f"WHERE micro_id IN ({placeholders})",
                [new_id] + list(micro_ids),
            )
            c.commit()
        self.append_revision_log(RevisionRecord(
            t_applied=time.time(), reviewer_round=reviewer_round,
            op="merge_micros", target_ids=list(micro_ids),
            new_ids=[new_id],
            payload={"new_description": new_description[:300],
                     "new_subject": new_subject, "new_object": new_object,
                     "new_action": new_action, "t_start": t_start,
                     "t_end": t_end, "n_frames": len(merged_fids)},
            reason=reason,
        ))
        return new_id

    @staticmethod
    def _rebind_refs_after_split(
        c: sqlite3.Connection, old_micro_id: str,
        split_specs: List[Tuple[str, float, float, List[str]]],
    ) -> Dict[str, int]:
        """Rebind entity_event / entity_frame / edges from a split micro onto the
        specific new micro each row belongs to.

        Assignment policy, most specific first:
          1. entity_frame rows whose frame_id was handed to a split go to that
             split (frame ownership is authoritative);
          2. otherwise the row goes to the split whose [t_start, t_end] contains
             its t_observed;
          3. otherwise the split with the nearest midpoint (LLM-provided ranges
             may leave gaps, and dropping the row would lose the link).

        The decision is made in Python rather than as a chain of time-ranged
        UPDATEs on purpose: each row must be evaluated against the ORIGINAL
        micro_id, but the first UPDATE would already have rewritten it, so a
        cascade would only ever rebind to the first split.

        ``UPDATE OR IGNORE`` is used where a primary key could collide, matching
        :meth:`merge_micros`. Returns per-table rebound row counts.
        """
        if not split_specs:
            return {}
        # frame_id → new micro id (先到先得: 同一 fid 被多个 split 声明时归第一个)
        fid_owner: Dict[str, str] = {}
        for nid, _ts, _te, fids in split_specs:
            for fid in fids:
                fid_owner.setdefault(str(fid), nid)

        def _pick(t_observed: float, frame_id: Optional[str] = None) -> str:
            if frame_id:
                owner = fid_owner.get(str(frame_id))
                if owner:
                    return owner
            for nid, ts, te, _f in split_specs:
                if ts <= t_observed <= te:
                    return nid
            return min(
                split_specs,
                key=lambda s: abs(((s[1] + s[2]) / 2.0) - t_observed),
            )[0]

        counts: Dict[str, int] = {}

        ef_rows = c.execute(
            "SELECT entity_id, frame_id, t_observed FROM entity_frame "
            "WHERE micro_id=?", (old_micro_id,)).fetchall()
        for row in ef_rows:
            c.execute(
                "UPDATE OR IGNORE entity_frame SET micro_id=? "
                "WHERE entity_id=? AND frame_id=?",
                (_pick(float(row["t_observed"]), row["frame_id"]),
                 row["entity_id"], row["frame_id"]),
            )
        counts["entity_frame"] = len(ef_rows)

        # entity_event 的 PK 是 (entity_id, micro_id), 每个 entity 对同一个老
        # micro 只有一行, 所以按 entity_id 逐行改是安全的。
        ee_rows = c.execute(
            "SELECT entity_id, t_observed FROM entity_event WHERE micro_id=?",
            (old_micro_id,)).fetchall()
        for row in ee_rows:
            c.execute(
                "UPDATE OR IGNORE entity_event SET micro_id=? "
                "WHERE entity_id=? AND micro_id=?",
                (_pick(float(row["t_observed"])), row["entity_id"], old_micro_id),
            )
        counts["entity_event"] = len(ee_rows)

        ed_rows = c.execute(
            "SELECT id, t_observed FROM edges WHERE micro_id=?",
            (old_micro_id,)).fetchall()
        for row in ed_rows:
            c.execute(
                "UPDATE edges SET micro_id=? WHERE id=?",
                (_pick(float(row["t_observed"])), row["id"]),
            )
        counts["edges"] = len(ed_rows)
        return counts

    def split_micro(self, micro_id: str, *,
                    splits: List[Dict[str, Any]],
                    reviewer_round: int = 0,
                    reason: str = "") -> List[str]:
        """Split one micro into N (N>=2). Each splits item:
          {"t_start": float, "t_end": float, "description": str,
           "subject"?: str, "object"?: str, "action"?: str,
           "frame_ids"?: [fid, ...]}
        Missing per-split fields inherit the old micro's; missing frame_ids are
        divided evenly across splits. facts_keys go to split 0 only. The old
        micro is soft-deleted (superseded_by = the first new id). Returns the new
        ids, or [] if fewer than 2 splits or the micro isn't found.

        The old micro's entity_event / entity_frame / edges rows are rebound to
        the specific split they belong to (see :meth:`_rebind_refs_after_split`),
        mirroring what :meth:`merge_micros` does. Without that rebind those rows
        keep pointing at the soft-deleted micro and the entity-event links are
        silently lost.
        """
        if not splits or len(splits) < 2:
            return []
        with self._lock, self._connect() as c:
            r = c.execute("SELECT * FROM micro_events WHERE id=? "
                          "AND (superseded_by IS NULL OR superseded_by='')",
                          (micro_id,)).fetchone()
            if r is None:
                return []
            macro_id = r["macro_id"]
            old_facts = json.loads(r["facts_keys"] or "[]")
            old_fids = json.loads(r["frame_ids"] or "[]")
            new_ids: List[str] = []
            # (new_id, t_start, t_end, frame_ids) —— 供下面重绑外键时判定归属
            split_specs: List[Tuple[str, float, float, List[str]]] = []
            for i, sp in enumerate(splits):
                ts = float(sp.get("t_start", r["t_start"]))
                te = float(sp.get("t_end", r["t_end"]))
                desc = str(sp.get("description") or "")
                subj = str(sp.get("subject", r["subject"] or "") or "")
                obj = str(sp.get("object", r["object"] or "") or "")
                act = str(sp.get("action", r["action"] or "") or "")
                fids = sp.get("frame_ids") or [
                    fid for fid in old_fids
                    # 拆分时未指定 fids → 简单按 i 均分 (兜底)
                    if (i * len(old_fids) // max(1, len(splits)) <= old_fids.index(fid)
                        < (i + 1) * len(old_fids) // max(1, len(splits)))
                ] if old_fids else []
                nid = f"micro_split_{int(te * 1000)}_{i}_{uuid.uuid4().hex[:6]}"
                new_ids.append(nid)
                split_specs.append((nid, ts, te, list(fids)))
                c.execute(
                    """INSERT INTO micro_events
                       (id, t_start, t_end, description, subject, object, action,
                        macro_id, facts_keys, frame_ids, created_at,
                        revised_at, revision_count)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (nid, ts, te, desc, subj, obj, act, macro_id,
                     json.dumps(old_facts if i == 0 else [], ensure_ascii=False),
                     json.dumps(fids, ensure_ascii=False),
                     time.time(), time.time(), 0),
                )
            # ★ 把老 micro 的 entity_event / entity_frame / edges 重绑到具体 split
            #   (merge_micros 早就这么做了, split 这边原先漏了)
            n_rebound = self._rebind_refs_after_split(c, micro_id, split_specs)
            # 软删除老 micro, 指向第一个新 id
            c.execute(
                """UPDATE micro_events SET superseded_by=?, revised_at=?,
                   revision_count=COALESCE(revision_count,0)+1 WHERE id=?""",
                (new_ids[0], time.time(), micro_id),
            )
            c.commit()
        self.append_revision_log(RevisionRecord(
            t_applied=time.time(), reviewer_round=reviewer_round,
            op="split_micro", target_ids=[micro_id], new_ids=new_ids,
            payload={"n_splits": len(splits),
                     "rebound_refs": n_rebound,
                     "split_briefs": [str(s.get("description", ""))[:120] for s in splits]},
            reason=reason,
        ))
        return new_ids

    def merge_entities(self, loser_ids: List[str], winner_id: str, *,
                       reviewer_round: int = 0,
                       reason: str = "",
                       t_observed: Optional[float] = None) -> bool:
        """Merge N loser entities into winner_id. Losers are soft-deleted
        (merged_into = winner) and their entity_event / entity_frame / edges are
        rebound to the winner; the winner absorbs their attributes (winner wins
        conflicts), aliases (loser names become aliases), min first_seen, max
        last_seen, and summed seen_count. Appends a merged_into EntityState — with
        t_observed on the frame-ts timeline (fallback wall) so Recall's
        get_entity_states(ask_ts=frame-ts) won't filter it out. Returns False if
        winner or all losers are missing (self-merge ids removed)."""
        if not loser_ids or not winner_id:
            return False
        requested_loser_ids = [str(x or "").strip() for x in loser_ids if str(x or "").strip()]
        if not requested_loser_ids:
            return False
        with self._lock, self._connect() as c:
            # Reviewer may repeat an action or reference an already-merged id.
            # Resolve both sides first so the operation is idempotent and never
            # inflates seen_count by merging the same loser multiple times.
            win, _winner_chain = self._resolve_entity_row(c, winner_id)
            if win is None:
                return False
            winner_id = str(win["id"] or "")
            canonical_loser_ids: List[str] = []
            already_merged_to_winner = False
            for requested_id in requested_loser_ids:
                loser_row, chain = self._resolve_entity_row(c, requested_id)
                if loser_row is None:
                    continue
                canonical_id = str(loser_row["id"] or "")
                if canonical_id == winner_id:
                    already_merged_to_winner = already_merged_to_winner or len(chain) > 1
                    continue
                if canonical_id not in canonical_loser_ids:
                    canonical_loser_ids.append(canonical_id)
            if not canonical_loser_ids:
                return already_merged_to_winner

            placeholders = ",".join("?" * len(canonical_loser_ids))
            losers = c.execute(
                f"SELECT * FROM entities WHERE id IN ({placeholders})",
                canonical_loser_ids,
            ).fetchall()
            if not losers:
                return False
            # 合并 attributes / aliases
            attrs = json.loads(win["attributes"] or "{}")
            aliases = set(json.loads(win["aliases"] or "[]"))
            for l in losers:
                lat = json.loads(l["attributes"] or "{}")
                for k, v in lat.items():
                    attrs.setdefault(k, v)   # winner 优先
                for al in json.loads(l["aliases"] or "[]"):
                    if al: aliases.add(al)
                if l["name"]:
                    aliases.add(l["name"])
            new_last = max(
                [float(win["last_seen"])] +
                [float(l["last_seen"]) for l in losers]
            )
            new_first = min(
                [float(win["first_seen"])] +
                [float(l["first_seen"]) for l in losers]
            )
            seen_count_sum = (
                int(win["seen_count"] or 1)
                + sum(int(l["seen_count"] or 1) for l in losers)
            )
            c.execute(
                """UPDATE entities SET attributes=?, aliases=?,
                   first_seen=?, last_seen=?, seen_count=?,
                   updated_at=?, revised_at=?,
                   revision_count=COALESCE(revision_count,0)+1
                   WHERE id=?""",
                (json.dumps(attrs, ensure_ascii=False),
                 json.dumps(sorted(aliases), ensure_ascii=False),
                 new_first, new_last, seen_count_sum,
                 time.time(), time.time(), winner_id),
            )
            # 软删除 losers
            c.execute(
                f"UPDATE entities SET merged_into=?, revised_at=?, "
                f"revision_count=COALESCE(revision_count,0)+1 "
                f"WHERE id IN ({placeholders})",
                [winner_id, time.time()] + list(canonical_loser_ids),
            )
            # Rebind relation tables with INSERT-then-DELETE.  The previous
            # UPDATE OR IGNORE left loser rows behind whenever winner+loser
            # shared the same primary key (the exact stale-frame recall bug).
            c.execute(
                f"INSERT OR IGNORE INTO entity_event(entity_id,micro_id,t_observed) "
                f"SELECT ?,micro_id,t_observed FROM entity_event "
                f"WHERE entity_id IN ({placeholders})",
                [winner_id] + list(canonical_loser_ids),
            )
            c.execute(
                f"DELETE FROM entity_event WHERE entity_id IN ({placeholders})",
                list(canonical_loser_ids),
            )
            c.execute(
                f"INSERT OR IGNORE INTO entity_frame(entity_id,frame_id,micro_id,t_observed) "
                f"SELECT ?,frame_id,micro_id,t_observed FROM entity_frame "
                f"WHERE entity_id IN ({placeholders})",
                [winner_id] + list(canonical_loser_ids),
            )
            c.execute(
                f"DELETE FROM entity_frame WHERE entity_id IN ({placeholders})",
                list(canonical_loser_ids),
            )
            c.execute(
                f"INSERT OR IGNORE INTO entity_rep_frames"
                f"(entity_id,frame_id,quality_score,note,added_at,added_by) "
                f"SELECT ?,frame_id,quality_score,note,added_at,added_by "
                f"FROM entity_rep_frames WHERE entity_id IN ({placeholders})",
                [winner_id] + list(canonical_loser_ids),
            )
            c.execute(
                f"DELETE FROM entity_rep_frames WHERE entity_id IN ({placeholders})",
                list(canonical_loser_ids),
            )
            c.execute(
                f"UPDATE entity_quotes SET entity_id=? "
                f"WHERE entity_id IN ({placeholders})",
                [winner_id] + list(canonical_loser_ids),
            )
            c.execute(
                f"UPDATE edges SET src_id=? WHERE src_id IN ({placeholders})",
                [winner_id] + list(canonical_loser_ids),
            )
            c.execute(
                f"UPDATE edges SET dst_id=? WHERE dst_id IN ({placeholders})",
                [winner_id] + list(canonical_loser_ids),
            )
            top_rep = c.execute(
                "SELECT frame_id FROM entity_rep_frames WHERE entity_id=? "
                "ORDER BY quality_score DESC LIMIT 1", (winner_id,),
            ).fetchone()
            if top_rep is not None:
                c.execute(
                    "UPDATE entities SET representative_frame_id=? WHERE id=?",
                    (top_rep["frame_id"], winner_id),
                )
            c.commit()
        # 给 winner 加一条 timeline。t_observed 用 frame-ts anchor (回退 wall);
        # 若用 wall-clock, recall 的 get_entity_states(ask_ts=frame-ts) 会把它滤掉。
        self.append_entity_state(EntityState(
            entity_id=winner_id,
            t_observed=(t_observed if t_observed is not None else time.time()),
            state_label="merged_into",
            note=(f"merged {len(canonical_loser_ids)} losers: "
                  f"{','.join(canonical_loser_ids)}"),
            source="reviewer",
        ))
        self.append_revision_log(RevisionRecord(
            t_applied=time.time(), reviewer_round=reviewer_round,
            op="merge_entities", target_ids=list(canonical_loser_ids),
            new_ids=[winner_id],
            payload={"winner_id": winner_id,
                     "loser_ids": list(canonical_loser_ids),
                     "requested_loser_ids": requested_loser_ids},
            reason=reason,
        ))
        return True

    def refine_entity(self, entity_id: str, *,
                      attributes_patch: Optional[Dict[str, str]] = None,
                      add_aliases: Optional[List[str]] = None,
                      remove_aliases: Optional[List[str]] = None,
                      new_name: Optional[str] = None,
                      new_representative_frame_id: Optional[str] = None,
                      evidence_frame_ids: Optional[List[str]] = None,
                      reviewer_round: int = 0,
                      reason: str = "",
                      t_observed: Optional[float] = None) -> bool:
        """Refine an entity (Reviewer): patch attributes, add/remove aliases,
        rename, or set representative_frame_id (omitted fields unchanged). Also
        appends a "refined" EntityState and a revision_log row. Returns False if
        the entity doesn't exist."""
        with self._lock, self._connect() as c:
            r = c.execute("SELECT * FROM entities WHERE id=?",
                          (entity_id,)).fetchone()
            if r is None:
                return False
            attrs = json.loads(r["attributes"] or "{}")
            aliases = set(json.loads(r["aliases"] or "[]"))
            delta: Dict[str, str] = {}
            new_alias_diff: List[str] = []
            if attributes_patch:
                for k, v in attributes_patch.items():
                    if attrs.get(k) != v:
                        delta[k] = str(v)
                        attrs[k] = str(v)
            if add_aliases:
                for al in add_aliases:
                    if al and al not in aliases:
                        aliases.add(al); new_alias_diff.append(al)
            if remove_aliases:
                for al in remove_aliases:
                    aliases.discard(al)
            name = new_name if new_name else r["name"]
            rep_fid = (new_representative_frame_id
                       if new_representative_frame_id
                       else (r["representative_frame_id"] or ""))
            c.execute(
                """UPDATE entities SET name=?, attributes=?, aliases=?,
                   representative_frame_id=?, updated_at=?, revised_at=?,
                   revision_count=COALESCE(revision_count,0)+1
                   WHERE id=?""",
                (name, json.dumps(attrs, ensure_ascii=False),
                 json.dumps(sorted(aliases), ensure_ascii=False),
                 rep_fid, time.time(), time.time(), entity_id),
            )
            c.commit()
        self.append_entity_state(EntityState(
            entity_id=entity_id,
            t_observed=(t_observed if t_observed is not None else time.time()),
            state_label="refined",
            attributes_delta=delta, new_aliases=new_alias_diff,
            evidence_frame_ids=list(evidence_frame_ids or []),
            source="reviewer", note=(reason or "")[:200],
        ))
        self.append_revision_log(RevisionRecord(
            t_applied=time.time(), reviewer_round=reviewer_round,
            op="refine_entity", target_ids=[entity_id],
            payload={"attributes_patch": dict(attributes_patch or {}),
                     "add_aliases": list(add_aliases or []),
                     "remove_aliases": list(remove_aliases or []),
                     "new_name": new_name,
                     "new_representative_frame_id": new_representative_frame_id},
            reason=reason,
        ))
        return True

    def rewrite_macro_summary(self, macro_id: str, *,
                              new_summary: Optional[str] = None,
                              new_label: Optional[str] = None,
                              new_narrative_arc: Optional[List[Dict[str, Any]]] = None,
                              new_entity_arcs: Optional[Dict[str, List[str]]] = None,
                              new_key_entities: Optional[List[str]] = None,
                              reviewer_round: int = 0,
                              reason: str = "") -> bool:
        """Rewrite a macro's summary / label / narrative_arc / entity_arcs /
        key_entities (omitted fields unchanged). Bumps revision_count and logs a
        revision. Returns False if the macro doesn't exist."""
        with self._lock, self._connect() as c:
            r = c.execute("SELECT * FROM macro_events WHERE id=?",
                          (macro_id,)).fetchone()
            if r is None:
                return False
            label = new_label if new_label is not None else (r["label"] or "")
            summary = new_summary if new_summary is not None else r["summary"]
            arc = (new_narrative_arc if new_narrative_arc is not None
                   else json.loads(r["narrative_arc"] or "[]"))
            earcs = (new_entity_arcs if new_entity_arcs is not None
                     else json.loads(r["entity_arcs"] or "{}"))
            key_ents = (new_key_entities if new_key_entities is not None
                        else json.loads(r["key_entities"] or "[]"))
            c.execute(
                """UPDATE macro_events SET label=?, summary=?, narrative_arc=?,
                   entity_arcs=?, key_entities=?, revised_at=?,
                   revision_count=COALESCE(revision_count,0)+1
                   WHERE id=?""",
                (label, summary,
                 json.dumps(arc, ensure_ascii=False),
                 json.dumps(earcs, ensure_ascii=False),
                 json.dumps(key_ents, ensure_ascii=False),
                 time.time(), macro_id),
            )
            c.commit()
        self.append_revision_log(RevisionRecord(
            t_applied=time.time(), reviewer_round=reviewer_round,
            op="rewrite_macro_summary", target_ids=[macro_id],
            payload={"new_label": label,
                     "new_summary_preview": (new_summary or "")[:200],
                     "n_arc_phases": len(arc) if isinstance(arc, list) else 0,
                     "n_entity_arcs": len(earcs) if isinstance(earcs, dict) else 0},
            reason=reason,
        ))
        return True

    def prune_entity(self, entity_id: str, *,
                     reviewer_round: int = 0,
                     reason: str = "",
                     force: bool = False,
                     t_observed: Optional[float] = None) -> bool:
        """E9: Reviewer prunes a "noise" entity.

        Unlike merge_entities' soft-delete, prune has no winner: it judges the
        entity to be a Writer mis-extraction (background passerby, blurry ghost).
        Marks merged_into="PRUNED" so Writer-prompt injection filters it out;
        Recall can still find it (audit trail), and entity_event/entity_frame
        links are left untouched.

        Typical case: high seen_count but never in any macro.key_entities → likely
        the Writer repeatedly mis-extracting the same noise, not a real subject.

        P0 protagonist protection — 2 gates guard against the LLM killing a real
        subject (force=True bypasses all gates; UI / migration scripts only):
          1. PERSON type can never be LLM-pruned (stable facial features; a wrong
             kill is costly and unrecoverable)
          2. appeared in any valid macro.key_entities → protected (an L2 LLM has
             endorsed it) — matched with LIKE '%"name"%'
        Hitting either gate → reject + warning log; on rejection NO revision_log
        is written (a wrong LLM proposal isn't a real "revision action"). An empty
        reason is also rejected. On success, appends a "pruned" EntityState +
        revision_log. Returns True only when the prune actually lands.
        (Note: the omni variant has a 3rd gate — entity_quotes protection — omitted
        here since this evolve variant has no such table.)
        """
        if not entity_id:
            return False
        if not reason:
            # ★ 没 reason 直接拒绝 (跟 _execute_action 的检查重复, 多一层防御)
            log.warning("[mem] 拒 prune %s: 必须给 reason", entity_id)
            return False
        with self._lock, self._connect() as c:
            r = c.execute("SELECT * FROM entities WHERE id=?",
                          (entity_id,)).fetchone()
            if r is None:
                return False
            prev_name = r["name"]
            prev_seen = int(r["seen_count"] or 0)
            prev_type = (r["type"] or "").upper()

            # ★ P0 门槛 (force=True 全部旁路)
            if not force:
                # 门槛 ①: PERSON 永不可被 LLM prune
                if prev_type == "PERSON":
                    log.warning(
                        "[mem] 拒 prune PERSON %s (name=%r, seen=%d): %s",
                        entity_id, prev_name, prev_seen, reason[:120],
                    )
                    return False
                # 门槛 ②: 在任何有效 macro.key_entities 出现过 → 保护
                #   key_entities 列存的是 JSON list ["name1","name2",...],
                #   用 LIKE '%"name"%' 粗匹配 (name 含特殊字符的极少, 够用).
                macro_hits = c.execute(
                    """SELECT COUNT(*) AS n FROM macro_events
                       WHERE (superseded_by IS NULL OR superseded_by='')
                         AND key_entities LIKE ?""",
                    (f'%"{prev_name}"%',),
                ).fetchone()["n"]
                if macro_hits > 0:
                    log.warning(
                        "[mem] 拒 prune %s (name=%r, type=%s): "
                        "在 %d 个 macro.key_entities 中出现过",
                        entity_id, prev_name, prev_type, macro_hits,
                    )
                    return False

            # 门槛全过 → 真正落地 prune
            c.execute(
                """UPDATE entities SET merged_into=?, revised_at=?,
                   revision_count=COALESCE(revision_count,0)+1
                   WHERE id=?""",
                ("PRUNED", time.time(), entity_id),
            )
            c.commit()
        self.append_entity_state(EntityState(
            entity_id=entity_id,
            t_observed=(t_observed if t_observed is not None else time.time()),
            state_label="pruned",
            note=(reason or "")[:200],
            source="reviewer",
        ))
        self.append_revision_log(RevisionRecord(
            t_applied=time.time(), reviewer_round=reviewer_round,
            op="prune_entity", target_ids=[entity_id],
            payload={"prev_name": prev_name, "prev_seen": prev_seen,
                     "prev_type": prev_type, "force": force},
            reason=reason,
        ))
        return True

    @staticmethod
    def _row_to_micro(r: sqlite3.Row) -> MicroEvent:
        # 兼容旧 db: 新字段(frame_ids/superseded_by/...)可能不存在
        def _g(k, d=None):
            try:
                return r[k]
            except (IndexError, KeyError):
                return d
        return MicroEvent(
            id=r["id"], t_start=r["t_start"], t_end=r["t_end"],
            description=r["description"], subject=r["subject"] or "",
            object=r["object"] or "", action=r["action"] or "",
            macro_id=r["macro_id"],
            facts_keys=json.loads(r["facts_keys"] or "[]"),
            frame_ids=json.loads(_g("frame_ids") or "[]"),
            created_at=r["created_at"],
            superseded_by=_g("superseded_by"),
            revised_at=_g("revised_at"),
            revision_count=int(_g("revision_count") or 0),
        )

    @staticmethod
    def _row_to_macro(r: sqlite3.Row) -> MacroEvent:
        def _g(k, d=None):
            try:
                return r[k]
            except (IndexError, KeyError):
                return d
        return MacroEvent(
            id=r["id"], t_start=r["t_start"], t_end=r["t_end"],
            label=r["label"] or "", summary=r["summary"],
            super_id=r["super_id"],
            key_entities=json.loads(r["key_entities"] or "[]"),
            created_at=r["created_at"],
            narrative_arc=json.loads(_g("narrative_arc") or "[]"),
            entity_arcs=json.loads(_g("entity_arcs") or "{}"),
            superseded_by=_g("superseded_by"),
            revised_at=_g("revised_at"),
            revision_count=int(_g("revision_count") or 0),
        )

    @staticmethod
    def _row_to_super(r: sqlite3.Row) -> SuperEvent:
        def _g(k, d=None):
            try:
                return r[k]
            except (IndexError, KeyError):
                return d
        return SuperEvent(
            id=r["id"], t_start=r["t_start"], t_end=r["t_end"],
            label=r["label"] or "", description=r["description"],
            macro_ids=json.loads(r["macro_ids"] or "[]"),
            is_root=bool(r["is_root"]), created_at=r["created_at"],
            narrative_arc=json.loads(_g("narrative_arc") or "[]"),
            superseded_by=_g("superseded_by"),
            revised_at=_g("revised_at"),
            revision_count=int(_g("revision_count") or 0),
        )

    @staticmethod
    def _row_to_entity(r: sqlite3.Row) -> Entity:
        def _g(k, d=None):
            try:
                return r[k]
            except (IndexError, KeyError):
                return d
        return Entity(
            id=r["id"], name=r["name"], type=r["type"],
            attributes=json.loads(r["attributes"] or "{}"),
            aliases=json.loads(r["aliases"] or "[]"),
            first_seen=r["first_seen"], last_seen=r["last_seen"],
            seen_count=r["seen_count"],
            representative_frame_id=_g("representative_frame_id") or "",
            updated_at=r["updated_at"],
            merged_into=_g("merged_into"),
            revised_at=_g("revised_at"),
            revision_count=int(_g("revision_count") or 0),
        )

    @staticmethod
    def _row_to_entity_state(r: sqlite3.Row) -> EntityState:
        return EntityState(
            id=int(r["id"]), entity_id=r["entity_id"],
            t_observed=r["t_observed"], state_label=r["state_label"] or "refined",
            attributes_delta=json.loads(r["attributes_delta"] or "{}"),
            new_aliases=json.loads(r["new_aliases"] or "[]"),
            confidence=float(r["confidence"] or 1.0),
            evidence_frame_ids=json.loads(r["evidence_frame_ids"] or "[]"),
            micro_id=r["micro_id"], source=r["source"] or "writer",
            note=r["note"] or "",
        )

    @staticmethod
    def _row_to_revision(r: sqlite3.Row) -> RevisionRecord:
        return RevisionRecord(
            id=int(r["id"]), t_applied=r["t_applied"],
            reviewer_round=int(r["reviewer_round"] or 0),
            op=r["op"] or "",
            target_ids=json.loads(r["target_ids"] or "[]"),
            new_ids=json.loads(r["new_ids"] or "[]"),
            payload=json.loads(r["payload"] or "{}"),
            reason=r["reason"] or "",
            success=bool(r["success"]),
            error=r["error"] or "",
            actor=r["actor"] or "reviewer",
        )

    @staticmethod
    def _row_to_edge(r: sqlite3.Row) -> Edge:
        return Edge(
            src_id=r["src_id"], dst_id=r["dst_id"], label=r["label"],
            rel_type=r["rel_type"], micro_id=r["micro_id"],
            t_observed=r["t_observed"],
            metadata=json.loads(r["metadata"] or "{}"),
        )

    def dump_all(self, *, limit_each: int = 200) -> Dict[str, List[Any]]:
        """Debug: full dump of current memory (NO ask_ts anti-dirty-read limit;
        UI Inspector only). Returns {"micros", "macros", "supers", "entities"},
        each the most recent limit_each rows in time-descending order. Pure-read
        path, holds no self._lock."""
        with self._connect() as c:
            micros = [self._row_to_micro(r) for r in c.execute(
                "SELECT * FROM micro_events ORDER BY t_end DESC LIMIT ?",
                (limit_each,)).fetchall()]
            macros = [self._row_to_macro(r) for r in c.execute(
                "SELECT * FROM macro_events ORDER BY t_end DESC LIMIT ?",
                (limit_each,)).fetchall()]
            supers = [self._row_to_super(r) for r in c.execute(
                "SELECT * FROM super_events ORDER BY t_end DESC LIMIT ?",
                (limit_each,)).fetchall()]
            entities = [self._row_to_entity(r) for r in c.execute(
                "SELECT * FROM entities ORDER BY last_seen DESC LIMIT ?",
                (limit_each,)).fetchall()]
        return {"micros": micros, "macros": macros,
                "supers": supers, "entities": entities}

    #: Tables that must SURVIVE reset(). Everything else in the DB file is
    #: session-scoped observation data and gets wiped on video-source switch.
    #: A denylist (rather than the old hardcoded allowlist) is deliberate: the
    #: previous allowlist named 10 tables while the schema had grown to 17, so
    #: screen_texts / screen_tables / frame_embeddings / memory_frames /
    #: entity_quotes / entity_rep_frames / task_states all leaked across
    #: sessions -- last video's OCR text and frame vectors stayed queryable by
    #: the next video's recall. New tables are now covered automatically.
    _RESET_KEEP_TABLES = frozenset({
        "meta",             # migration bookkeeping; wiping it re-runs migrations
    })

    def reset(self) -> Dict[str, int]:
        """Clear all memory (called on video-source switch to guarantee session
        isolation).

        Enumerates the live schema from sqlite_master and wipes every table
        except :attr:`_RESET_KEEP_TABLES`, so tables added later are covered
        without touching this method. Also clears the pending L1/L2 caches.
        Returns per-table deleted row counts, for driver logging.
        """
        deleted: Dict[str, int] = {}
        with self._lock, self._connect() as c:
            rows = c.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
            tables = sorted(
                str(r[0]) for r in rows
                if str(r[0]) not in self._RESET_KEEP_TABLES
                # External-content FTS rows are removed by the DELETE trigger
                # on audio_observations.  Deleting its shadow/config tables
                # directly would corrupt the virtual table for the next session.
                and not str(r[0]).startswith("audio_observations_fts")
            )
            for tbl in tables:
                # 表名来自 sqlite_master, 不是外部输入, 拼接安全
                try:
                    cur = c.execute(f'SELECT COUNT(*) FROM "{tbl}"')
                    n = int(cur.fetchone()[0])
                except sqlite3.OperationalError:
                    n = 0
                try:
                    c.execute(f'DELETE FROM "{tbl}"')
                except sqlite3.OperationalError:
                    log.warning("[mem reset] 清表失败, 跳过: %s", tbl)
                    continue
                deleted[tbl] = n
            c.commit()
        self._pending_micros.clear()
        self._pending_macros.clear()
        nonzero = {k: v for k, v in deleted.items() if v}
        log.info("[mem reset] 清空 %d 张表, 其中有数据的: %s",
                 len(deleted), nonzero or "(none)")
        return deleted

    def close(self) -> None:
        pass   # SQLite 每次连接是临时打开关闭, 无需显式 close

    def cleanup(self) -> None:
        """On session end, delete the temp db file (only if a tmp file, i.e. no
        cfg.mem_db_path was configured)."""
        if not self.cfg.mem_db_path and self.db_path and os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
                log.info("[mem] 清理临时 db: %s", self.db_path)
            except Exception as e:
                log.warning("[mem] 清理 db 失败: %s", e)

    def set_summary(self, summary: str) -> bool:
        """Write the first macro's short summary into the db's meta table
        (key='summary'). No-op / False for an empty summary.

        No longer renames the live db — an early version renamed
        <ts>_pending.sqlite → <ts>_<summary>.sqlite, which raced under the live
        watcher: mid-rename, another reader thread (get_recent_macros etc., which
        use lock-free _connect) would sqlite3.connect the old path and silently
        create an empty db → "no such table: macro_events". Now the db name is
        fixed as <ts>.sqlite and the summary lives only in the meta table, read
        via get_summary() when UI/retrieval needs it. Idempotent: overwrites.

        (Method named set_summary; semantically = write summary — it does not
        touch self.db_path or the file.)"""
        s = (summary or "").strip()
        if not s:
            return False
        try:
            with self._lock, self._connect() as c:
                c.execute(
                    "INSERT INTO meta(key, value) VALUES('summary', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (s,))
            log.info("[mem] 摘要写入 meta: %s", s[:40])
            return True
        except Exception as e:
            log.warning("[mem] set_summary 失败: %s", e)
            return False

    def set_session_id(self, session_id: str) -> bool:
        """Bind this memory database to its owning Hermes conversation.

        The dashboard uses this stable id to select the database for the
        currently open conversation.  Older databases do not have the key and
        remain readable; their UI fallback is the newest database.
        """
        sid = (session_id or "").strip()
        if not sid:
            return False
        try:
            with self._lock, self._connect() as c:
                c.execute(
                    "INSERT INTO meta(key, value) VALUES('hermes_session_id', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (sid,))
            return True
        except Exception as e:
            log.warning("[mem] set_session_id 失败: %s", e)
            return False

    def get_summary(self) -> str:
        """Read meta.summary (the first macro's short summary); "" if unset."""
        try:
            with self._connect() as c:
                r = c.execute(
                    "SELECT value FROM meta WHERE key='summary'").fetchone()
                return (r["value"] if r else "") or ""
        except Exception:
            return ""

    def tokens_txt_path(self) -> str:
        """The .tokens.txt path for the current db (same name + dir, .sqlite →
        .tokens.txt; stable since the db name never changes). "" if no db path."""
        if not self.db_path:
            return ""
        return self.db_path[:-len(".sqlite")] + ".tokens.txt" \
            if self.db_path.endswith(".sqlite") else self.db_path + ".tokens.txt"


# =========================================================================== #
# JSON 解析 helpers
# =========================================================================== #
_JSON_OBJ_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_JSON_ARR_BLOCK_RE = re.compile(r"```(?:json)?\s*(\[.*?\])\s*```", re.DOTALL)
# ★ ts 字段两层兜底设计:
#   - prompt schema 示例写的是 `"ts": "13.0s"` 字符串带单位 (跟图标签格式一致,
#     让模型对"秒"有清晰认知). 但模型偶尔会:
#       (a) 写成 `"ts": 13.0s` (非法 JSON, 数字带 s 字面值)
#       (b) 写成 `"ts": 13.0` 纯数字 (合法 JSON 数字)
#       (c) 写成 `"ts": "13.0s"` (合法 JSON 字符串, 期望写法)
#       (d) 写成 `"ts": "13.0"` (合法 JSON 字符串, 漏 s)
#   - 第一层 (本正则): JSON parse 层. 解决 (a), 把 `"ts": 13.0s` 改成 `"ts": 13.0`
#     再 parse. 只匹配 `"ts":` 字段, 不会误伤其他字段里含 s 的字符串值.
#   - 第二层 (_parse_ts_value): 字段消费层. 解决 (c)(d) — parse 出来如果是字符串,
#     剥单位后 float. (b) parse 出来已是数字, float 直接通过.
# ★ 兼容 "ts"(帧标签/micro) 与 "t"(L2/L3 narrative_arc 的 phase 时间字段):
#   Gemini 偶尔把单位 s 也抄进 JSON (如 {"t": 130.0s}), 两种字段名都要能兜底剥单位,
#   否则 json.loads 整段失败 → 聚合退化成空摘要。
_TS_UNIT_FIX_RE = re.compile(r'("t(?:s)?"\s*:\s*-?\d+(?:\.\d+)?)s\b')


def _repair_json_candidate(raw: str) -> str:
    """Best-effort repair for LLM JSON that is complete in spirit but slightly
    invalid/truncated: strip fences, remove trailing commas, close an unfinished
    string, and append missing braces/brackets. It intentionally does not invent
    field values; it only makes the already generated prefix parseable.
    """
    s = (raw or "").strip()
    l = s.find("{")
    if l >= 0:
        s = s[l:]
    if not s:
        return s
    out: List[str] = []
    stack: List[str] = []
    in_str = False
    esc = False
    for ch in s:
        out.append(ch)
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack and ch == stack[-1]:
                stack.pop()
            else:
                out.pop()
                break
    if in_str:
        out.append('"')
    repaired = "".join(out).rstrip()
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    while stack:
        closer = stack.pop()
        repaired = re.sub(r",\s*$", "", repaired.rstrip()) + closer
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    return repaired


def _parse_ts_value(v: Any) -> Optional[float]:
    """Tolerantly parse a ts field value into float seconds, or None.
    Accepts: int / float / plain numeric string "13.0" / unit-suffixed string
    "13.0s" / "13s" (also "sec"/"seconds"). Run key_frames[i]["ts"] through this
    instead of float(v) — it's far more robust to LLM formatting."""
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        # 剥末尾的 s / sec / seconds 等单位 (常见 LLM 写法)
        for suffix in ("seconds", "sec", "s"):
            if s.lower().endswith(suffix):
                s = s[: -len(suffix)].rstrip()
                break
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _try_parse_json_obj(raw: str) -> Optional[Dict[str, Any]]:
    """Internal: the same 3-candidate parse as extract_json_obj, without the
    sanitize retry. Returns the first candidate that parses to a dict, or None."""
    cand: List[str] = []
    if raw.startswith("{"):
        cand.append(raw)
    m = _JSON_OBJ_BLOCK_RE.search(raw)
    if m:
        cand.append(m.group(1))
    l, r = raw.find("{"), raw.rfind("}")
    if 0 <= l < r:
        cand.append(raw[l:r + 1])
    repaired = _repair_json_candidate(raw)
    if repaired:
        cand.append(repaired)
    for c in cand:
        c = re.sub(r",\s*([}\]])", r"\1", c.strip())
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def extract_json_obj(raw: str) -> Optional[Dict[str, Any]]:
    """Extract the first valid JSON object from raw model output.
    Tries: a bare `{...}`, a ```json```-fenced block, and the substring from the
    first `{` to the last `}`.

    Gemini sanitize fallback: if all three candidates fail, run _TS_UNIT_FIX_RE
    once over raw (rewrites `"ts": 13.0s` → `"ts": 13.0`) and retry. This works
    around Gemini occasionally copying the 's' unit from the ts=XX.Xs frame tag
    into JSON field values (a known instruction-following bug).
    """
    if not raw:
        return None
    raw = raw.strip()
    obj = _try_parse_json_obj(raw)
    if obj is not None:
        return obj
    # ★ 兜底: sanitize "ts": 13.0s → "ts": 13.0, 再重试. 不变就跳过省 1 次 parse.
    fixed = _TS_UNIT_FIX_RE.sub(r"\1", raw)
    if fixed != raw:
        obj = _try_parse_json_obj(fixed)
        if obj is not None:
            return obj
    return None


def extract_json_arr(raw: str) -> Optional[List[Any]]:
    if not raw:
        return None
    raw = raw.strip()
    cand: List[str] = []
    if raw.startswith("["):
        cand.append(raw)
    m = _JSON_ARR_BLOCK_RE.search(raw)
    if m:
        cand.append(m.group(1))
    l, r = raw.find("["), raw.rfind("]")
    if 0 <= l < r:
        cand.append(raw[l:r + 1])
    for c in cand:
        try:
            obj = json.loads(c)
            if isinstance(obj, list):
                return obj
        except json.JSONDecodeError:
            continue
    wrapped = extract_json_obj(raw)
    if isinstance(wrapped, dict):
        inner = wrapped.get("tool_calls") or wrapped.get("calls")
        if isinstance(inner, list):
            return inner
    return None


# =========================================================================== #
# Token estimator (粗略, 用于 prompt 体积监控)
# =========================================================================== #
def estimate_msg_tokens(messages: List[Dict[str, Any]]) -> Tuple[int, int, int]:
    """Estimate (text_chars, n_images, est_tokens) for a message list.

    Rough formula (logging/monitoring only, not billing):
      - mixed CN/EN: 1 token ≈ 2 chars
      - 1 image_url ≈ 320 tokens (qwen3.5 vision medium-resolution rule of thumb)
    """
    text_chars = 0
    n_images = 0
    for m in messages or []:
        content = m.get("content")
        if isinstance(content, str):
            text_chars += len(content)
        elif isinstance(content, list):
            for it in content:
                if not isinstance(it, dict):
                    text_chars += len(str(it))
                    continue
                if it.get("type") == "image_url":
                    n_images += 1
                elif it.get("type") == "text":
                    text_chars += len(it.get("text", "") or "")
                else:
                    text_chars += len(str(it))
    est = text_chars // 2 + n_images * 320
    return text_chars, n_images, est


def fmt_tok(messages: List[Dict[str, Any]]) -> str:
    """Log-friendly token summary: 'tok≈3450 (txt=6900 img=4)'."""
    tc, ni, est = estimate_msg_tokens(messages)
    return f"tok≈{est} (txt={tc} img={ni})"
