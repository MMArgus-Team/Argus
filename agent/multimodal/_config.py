# -*- coding: utf-8 -*-
"""Config dataclass for the multimodal Always-On Video Agent.

Auto-split from the original monolithic engine (see agent/multimodal/__init__.py);
the public surface is re-exported from agent.multimodal.core for backward
compatibility. Holds the tunable knobs for every worker (MemoryWriter,
MemoryReviewer, WatcherWorker, Search/Recall) plus the ASR/TTS and FrameBuffer
settings.

★ SOURCE OF TRUTH — DO NOT EDIT VALUES HERE TO CONFIGURE THE APP.
   This dataclass is a SCHEMA + FALLBACK only. The single user-facing input is
   the project ``config.yaml`` (v33 module-aggregated layout: each submodule's
   endpoint + behavior under ``model.<role>``; cross-module knobs under
   ``settings:``; the speech interface under ``audio:``). At startup
   config.yaml is synced into HERMES_HOME and ``hermes_glue.build_config`` /
   ``flatten_mm_config`` translate the nested yaml back onto these flat field
   names. A field's default here is used ONLY when config.yaml omits it.

   Field-name ↔ yaml-path contract (so an edit always takes effect):
     * yaml ``model.watcher.*``  → cfg.watcher_* / router_* / search_* / watch_*
     * yaml ``model.memory.*``   → cfg.memory_* / writer_* / reviewer_* /
                                    recall_* / mem_* / agg_* / frame_store_*
     * yaml ``model.monitor.*``  → cfg.monitor_* (+ raw mm.get for tick/merge)
     * yaml ``model.embedding.*``→ cfg.embedding_* / mm_embedding_* / recall_*
     * yaml ``model.ocr.*``      → cfg.ocr_*
   A mistyped / mis-nested yaml key is WARNED at startup
   ("[config] unknown ... IGNORED") by flatten_mm_config's unknown-key guard —
   it never silently vanishes. When adding a NEW field: add it here, wire its
   yaml path in hermes_glue (_DEEP_PATH_* / _NUMERIC_KEYS), and document it in
   config.yaml.
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
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from io import BytesIO
from typing import (
    Any, AsyncIterator, Awaitable, Callable, Deque, Dict, List,
    Optional, Set, Tuple,
)

try:
    import aiohttp  # optional: only needed by the local HTTP speech/Gemini backends
except ImportError:  # pragma: no cover
    aiohttp = None  # type: ignore
import httpx
from openai import AsyncOpenAI

log = logging.getLogger("hermes.multimodal")


# Optional external *image* search tool. Argus ships no implementation: the
# module is a deployment-specific extension point (it has to talk to whatever
# image-retrieval service you run), so the default is empty and the image_search_*
# tools degrade to a clean "not configured" observation instead of failing hard.
# To enable them, point multimodal.search_tool_path at a Python file exposing
#   async unified_image_search(image, keys=[...], threshold=0.6) -> list[dict]
# (or the older ``image_search_observation(image, image_search_keys=[...])``).
# NOTE: text search does NOT need this — it goes through the AnySearch backend,
# see the anysearch_* fields below.
DEFAULT_SEARCH_TOOL_PATH = ""


# ★ Edge rel_type 白名单 (writer 建图去污染): LLM 乱编的 rel_type 会污染图谱,
#   _insert_edges_for_micro 用它把非白名单值降级到 subject_object.
#   person_relation 额外要求两端都是 PERSON entity (见 _workers._insert_edges_for_micro).
EDGE_REL_TYPES = frozenset({
    "subject_object", "spatial", "temporal_causal",
    "person_relation", "social", "semantic",
})


# =========================================================================== #
# Config
# =========================================================================== #
@dataclass
class Config:
    # ---- vLLM 底座 ----
    # 主 endpoint: Writer / Router / Search / Recall 四个角色共用.
    base_url: str = "http://localhost:12347/v1"
    api_key: str = "EMPTY"
    model: str = "qwen3.5"
    # True when the user pinned multimodal.worker_model in config → build-time
    # "follow the main agent's resolved model" is skipped (glue set model above).
    _worker_model_explicit: bool = False
    # ★ watcher/worker 的模型兜底: 空则沿用主 model。(原名 front_model, FrontWorker
    #   已删, 改名对齐实际用途; front_base_url 是死配置已删。见 watcher_engine 用它。)
    worker_fallback_model: str = ""
    # ★ 可选: MemoryWriter 独立 backend (类似 front_*, 但允许 **跨协议** 切换).
    #   memory_provider 取值:
    #     ""        → 沿用主 endpoint (OpenAI 兼容 vLLM, 原行为).
    #     "openai"  → 显式用 OpenAI 兼容协议, base_url/api_key/model 走 memory_*.
    #     "gemini"  → 走 Gemini (runway) generateContent, thinking=HIGH 硬编码,
    #                 media_resolution=HIGH 硬编码, 用 vision/JSON 更强的模型
    #                 来理解长画面 + 严格 JSON 输出.
    memory_provider: str = ""
    memory_base_url: str = ""
    memory_api_key: str = ""
    memory_model: str = ""
    # ★ 可选: 多模态复杂 query 深度分析 (WatcherAgent 的 Router/Search/Recall +
    #   deep-analysis 汇总) 的【独立模型端点】。与主 agent 机制对齐 (仿 memory_*):
    #     watcher_base_url/api_key 都留空 → fallback 主 agent 解析的模型 (原行为)。
    #     填了 watcher_base_url → 用这个独立 OpenAI 兼容端点跑 watcher (+ worker_model)。
    #   worker_model 单独填只改模型名 (端点仍跟主 agent); 想完全独立就三个都填。
    #   ★ 命名: yaml 里叫 model.watcher.*, dataclass 字段前缀统一为 watcher_ (与 yaml
    #     对齐, 避免"配置后不生效"; worker_model 沿用 cfg.model 不改名)。
    watcher_provider: str = ""
    watcher_base_url: str = ""
    watcher_api_key: str = ""
    # (worker_model 已在上方定义为 cfg.model 的 override 入口)
    # ★ 可选: MonitorAgent (always-on 视频监控 SPEAK/SILENT 判定) 的
    #   【独立模型端点】。同样与主 agent 对齐:
    #     monitor_base_url 留空 → fallback 主 agent 的 client/model (原行为)。
    #     填了 → 用独立 OpenAI 兼容端点 (sync) 跑监控判定。
    monitor_provider: str = ""
    monitor_base_url: str = ""
    monitor_api_key: str = ""
    monitor_model: str = ""
    # Some OpenAI-compatible MaaS gateways require the same credential in both
    # the standard Authorization bearer token and an ``api-key`` header. Keep
    # this opt-in and Monitor-only so every existing endpoint stays unchanged.
    monitor_send_api_key_header: bool = False
    # ★ Recall decide/distill 【独立端点】(v33 之后新增, 允许 recall 与 memory
    #   writer 分家): 空 → 回退到 model.memory (跟 writer 共用 client, 老行为)。
    #   典型用法: writer 走 K2.6 MaaS (无审查, non-thinking 快, 10s 高频便宜),
    #   recall 走 Luna @ /v1/chat/completions (视觉多图+文, tools 不
    #   需要, 命中率优先)。
    recall_provider: str = ""
    recall_base_url: str = ""
    recall_api_key: str = ""
    recall_model: str = ""
    # Recall 末尾视觉验收也可独立端点, 隔离 Writer/Reviewer 长请求对正确性 gate 的
    # 抢占。verify_base_url 空 → 回退到 recall_client (再回退 memory)。
    recall_verify_provider: str = ""
    recall_verify_base_url: str = ""
    recall_verify_api_key: str = ""
    recall_verify_model: str = ""
    # ★ Embedding backend (混合关键词+向量召回一期: 只做文本 embedding).
    #   四件套 + 数值参数, 与 memory/monitor/recall 对齐:
    #     embedding_base_url 空 → 全局关闭 embedding: Writer/Reviewer 不算,
    #     MemoryToolBox.search_* 走纯关键词兜底 (与改造前行为一致).
    #     填了 → 启用混合检索: Writer/Reviewer 后台算并落库, search_events/search_entity
    #     用关键词×向量 RRF 融合排序. text-embedding-v3 支持 dimensions 参数
    #     (256/512/768/1024), 默认 1024 与 DashScope 官方推荐一致.
    embedding_provider: str = ""              # "openai" / "custom" / "" (空=关闭)
    embedding_base_url: str = ""              # e.g. dashscope compatible-mode /v1
    embedding_api_key: str = ""
    embedding_model: str = "text-embedding-v3"
    embedding_dimensions: int = 1024
    embedding_timeout_sec: float = 4.0
    embedding_batch_size: int = 16            # backfill / Reviewer 批处理上限
    # ★ 二期: 帧图像 embedding (DashScope multimodal-embedding-v1, 与文本向量
    #   不同语义空间, 独立客户端). mm_embedding_model 空 → 关闭帧向量路径,
    #   search_frames_by_text 工具返回"未启用"提示. mm_embedding_api_key 空 →
    #   复用 embedding_api_key (都是 DashScope key).
    mm_embedding_model: str = "multimodal-embedding-v1"
    mm_embedding_base_url: str = ""           # 空 → DashScope 默认端点
    mm_embedding_api_key: str = ""            # 空 → 复用 embedding_api_key
    mm_embedding_timeout_sec: float = 6.0
    # ★ tongyi-embedding-vision-*-2026-03-06 专属调参 (老模型须保持 0/-1 不传):
    #   dimensions=0 → 不传, 用模型默认 (plus-2026-03-06 默认 1152);
    #   res_level=-1 → 不传, 用默认 1; 0/1/2/3 对应单图 127/402/578/1026 token,
    #   res_level=3 对"小物体在大场景"的帧检索 +5~10% 效果 (成本×2.5 但单价减半).
    mm_embedding_dimensions: int = 0
    mm_embedding_res_level: int = -1
    recall_frame_vector_topk: int = 8         # search_frames_by_text 默认召回数
    frame_vector_pool_cap: int = 0          # 帧向量候选池上限 (帧量≈micro×3)
    # ★ OCR: screen-text extraction before MemoryWriter (ALWAYS ON, v33 — no
    #   ocr_enabled gate). Local-only: RapidOCR/PP-OCR on onnxruntime, no
    #   endpoint/credential needed (rapidocr+onnxruntime are core deps). The
    #   remote/cloud VLM OCR path (ocr_use_local / ocr_provider/base_url/api_key/
    #   model) was removed — ocr_backend is the local backend name (rapidocr).
    #   ocr_max_threads: 单实例共享引擎的并发线程上限 (全忙 → 跳过本次识别不重试);
    #   ocr_worker_interval: 后台 worker 触发周期 (3s); ocr_timeout_sec: 单次超时.
    ocr_backend: str = "rapidocr"
    ocr_timeout_sec: float = 8.0
    ocr_max_tokens: int = 1200
    ocr_frames_per_wake: int = 4
    ocr_max_side: int = 0
    ocr_max_threads: int = 8
    ocr_worker_interval: float = 3.0
    ocr_worker_backlog_limit: int = 12
    ocr_worker_max_attempts: int = 3
    # MemoryReviewer may use a dedicated endpoint/model. Empty fields fall back
    # to model.memory so deployments that want Writer/Recall/Reviewer unified
    # keep the old behavior.
    reviewer_provider: str = ""
    reviewer_base_url: str = ""
    reviewer_base_urls: List[str] = field(default_factory=list)
    reviewer_api_key: str = ""
    reviewer_model: str = ""
    # 混合检索融合参数 (只在 embedding 启用时生效)
    recall_hybrid_enabled: bool = True        # False → 只用关键词 (即便 embedding 已算好)
    recall_vector_topk: int = 30              # 关键词 / 向量各路先取 top_k, 再 RRF 融合
    # RRF 常数。原论文的 60 是在 ~1000 条结果的 TREC run 上调的; 这里两路各
    # recall_vector_topk=30 条, k=60 会把 rank1 与 rank30 的差距压到只有 1.47x,
    # 等于把两路的排序信息大半抹平。k=20 时该比值回到 2.43x。
    recall_rrf_k: int = 20
    # ★ Gemini 慢 + runway 偶尔挂: 连续 N 次 wake 失败就停掉 MemoryWriter loop,
    #   推 memory_writer_dead 事件给前端弹错 (避免大规模测试时 agent 假活).
    memory_max_consecutive_failures: int = 10

    # (删: FrontWorker (1s 自驱) 整段死配置 — FrontWorker 已删除, 这些字段无消费者:
    #  front_loop_interval/temperature/max_tokens/recent_frames/min_speak_interval
    #  + cruise_macro_summary_max_chars。)

    # 续写 (handle_ask 触发, search+recall findings 都到位后)
    cont_max_tokens: int = 1536
    cont_recent_frames: int = 12
    cont_now_frames: int = 2
    cont_recent_history_turns: int = 16   # ★ 续写/Router answer 附最近 N 条对话历史

    # QueryWorker jobs share one Watcher loop per session.  Limit simultaneous
    # Recall/Search executions and bound the waiting queue so a burst of chat
    # turns cannot create an unbounded number of worker tasks/LLM requests.
    query_worker_max_concurrency: int = 2
    query_worker_max_pending: int = 8

    # ---- ③ Router.answer() 一次性回答模式 (v3 架构: 不再用 _stream_continuation) ----
    watcher_answer_max_tokens: int = 1536

    # ---- ② MemoryWriter (5s wake-up, 跟 env_audio_window_sec 严格对齐) ----
    # ★ wake 周期 4s→5s: 跟 env_audio_window_sec=5.0 对齐, 让"每拍刚好等到一段
    #   完整 ASR ready"; 帧窗仍 30 帧 (15s), 前 10s 字幕大概率已 ready, 后 5s
    #   没 ready 下一拍重叠区会补 (跟帧重叠同套容错). 副带 LLM 调用频率 -20%.
    # ★ 省 token 调优 (2026-07, 从 mm_memory_standalone 对齐): wake 5→10s、帧窗
    #   30→20. 权衡: 精度略降、LLM 调用频率再 -50%. 想回精度优先改回 5 / 30.
    writer_wake_interval: float = 10.0
    writer_max_tokens: int = 5000              # event_boundary + OCR/table/task 结构化输出
    writer_recent_frames: int = 20
    writer_image_max_side: int = 0             # L1 writer 请求前单独缩图; <=0 使用原始帧
    writer_image_jpeg_quality: int = 85        # L1 writer 缩图 JPEG 质量
    # ★ MemoryWriter 看的"最近 N 秒字幕"窗口 (跟 writer_recent_frames 时段对齐, 默认 15s).
    #   wake 时拉 [now-N, now] 区间内的 audio_observation 拼成独立 ASR block,
    #   插在画面前面让 LLM 跟图严格对齐. 无字幕时整个 block 不输出.
    writer_asr_window_sec: float = 15.0

    # ★ 记忆抽取是视觉任务, 记忆模型必须能看图。writer_vision_ability=False 视为
    #   配置错误: MemoryBackend 启动时直接报错、不建 backend (视频流也无法正常带记忆
    #   开启), 而不是静默按"有视觉"跑一个看不了图的模型。默认 True。
    writer_vision_ability: bool = True                 # 记忆模型能否看图 (硬性前置)
    # (v33: 整条 OMNI 原始音频写入路径已删 — writer_audio_ability /
    #  writer_omni_audio_window_sec / writer_omni_thinking_level / media_resolution /
    #  audio_buffer_seconds 全部移除。音频记忆统一走外部 ASR 字幕。)
    entity_rep_frames_max_n: int = 5                   # 每个 PERSON entity 最多留几张代表帧

    # ---- ★ E9 (entity tier): Writer prompt 注入"已知关键 Entity" 三档分桶 ----
    # 设计核心: 防"鬼魂 entity"(角落一闪而过被打高分) 进 Tier 1 形成正反馈污染.
    #   - 准入门槛 (低): merged_into IS NULL + seen_count >= writer_entity_min_seen
    #   - 排序主信号: macro_hits = 在最近 N 个 macro.key_entities 中出现次数
    #     (macro 是独立 L2 聚合 LLM 挑的, 不会把模糊路人放进 key_entities → 鬼魂 entity
    #      macro_hits=0, 自然挤不进 Tier 1)
    #   - Tier 1 (带代表帧缩略图 + 强复用约束): top N 最重要
    #   - Tier 2 (纯文字详情): 中等重要, 保留 aliases/attrs/seen
    #   - Tier 3 (极简一行): 仅 name + type, 防漏掉但不占太多 token
    writer_entity_enabled: bool = True
    writer_entity_min_seen: int = 2                  # 单次出现的不参选 (防一次性误抽)
    writer_entity_macro_lookback: int = 8            # 打分时看最近 N 个 macro 的 key_entities
    writer_entity_tier1_n: int = 5                   # 带图档: top N
    writer_entity_tier2_n: int = 10                  # 文字详情档
    writer_entity_tier3_n: int = 15                  # 极简一行档
    # ★ 代表帧分辨率: 0 表示用原图 (跳过 thumbnail_b64 的二次 JPEG 重编码, 0 损失).
    #   前端送进来的摄像头帧本身 max 边就 = 720 (state.maxSide=720, q=0.7), 屏幕共享 1280;
    #   设 0 让 Tier 1 跟主帧 30 张完全同源同质, LLM 做"已知 entity vs 当前画面" 视觉对比时
    #   有最大细粒度信号 (代价: 多 ~5×1300 token; 想缩成本就改成 >0 走 thumbnail).
    writer_entity_tier1_thumb_side: int = 0          # 0 = 原图; >0 = thumbnail 边长 (px)
    writer_entity_tier1_thumb_quality: int = 80      # 仅 thumb_side>0 时生效
    # tier1 是否限制 type 才挂图 (空字符串 → 全 type 都挂; 例 "PERSON,OBJECT,LOCATION")
    writer_entity_tier1_visual_types: str = ""
    # ★ 软硬结合: prompt 鼓励 LLM 在 entities_mentioned 里填 reused_entity_id 复用已知 entity,
    #   后端校验 id 存在则直接走精确路径绕过 fuzzy match; 否则 fallback 老逻辑.
    writer_reused_id_enabled: bool = True

    # ---- ★ E10 (event timeline): Writer 历史 [画面观察] 换成 macro+micro 事件 ----
    # 改造前: conv_dump 全量塞 [画面观察 mm:ss] 流水, 受 conv_max_bg_obs=200 cap 后丢早期
    # 改造后: 历史段用结构化事件 (macro/micro), 当前 pending 段才用 raw [画面观察]
    #   - macros 按 t_start 升序输出 (summary + key_entities), 已被 Reviewer superseded 的过滤
    #   - micros 跟 macros 混在同一时间轴 (按 t_start 升序), 不区分"是否已聚合"
    #   - 当前 pending micro 段内的 obs → raw [画面观察] (Writer 仍需细颗粒度判 event_boundary)
    #   - super 暂不接入 (先看 macro+micro 效果)
    # 注意: user/assistant/audio_observation turn 仍走 conv_dump (用户提问的上下文不能丢).
    writer_event_timeline_enabled: bool = True
    writer_event_max_macros: int = 15                # 时间轴最多保留 N 个 macro
    writer_event_max_micros: int = 30                # 时间轴最多保留 N 个 micro

    # ---- ③ WatcherWorker ----
    react_max_tokens: int = 512
    react_recent_frames: int = 8
    # ★ v5: 12→16, 跟原 router_answer 阶段的 cont_recent_history_turns 对齐
    #   (Router 现在最后一轮要直接出 answer, 需要更完整的对话上下文).
    react_recent_history_turns: int = 16
    # ---- ③b WatcherWorker ReAct 多轮编排 (v4: Router 当唯一大脑, 多轮派 search/recall) ----
    react_max_rounds: int = 4         # ReAct 循环硬上限 (防死循环)
    # ★ 成本闸: 一轮里并发派出的 search 子任务数硬上限 (每条=一次带图付费检索)。
    #   Router 偶尔一口气派十几条, 截断到这个上限。recall 不受限(本地 SQLite, 便宜)。
    react_search_tasks_max: int = 5
    # ★ v5: 768→2048. Router 收尾轮要同时输出 thought/can_answer/tasks **+ 完整 answer**,
    #   省掉 LIVE_RESEARCH_ANSWER_SYSTEM 那次额外 LLM 调用. 中间轮不带 answer 时也用这个上限,
    #   多出来的 budget 模型不会硬吃, 按实际 output 长度计费, 无成本浪费.
    react_round_max_tokens: int = 2048

    # ---- ③c 深度分析 watch (持续 live-watcher) ----
    # set_live_watcher 统一走 WatcherAgent 的 delegation 循环, 且 ALWAYS 持续:
    # 从 buffer 头开始逐批遍历视频, 只要视频流不停就持续调研 + 写 log (看画面 vs
    # 检索的算力分配由 subagent 自主决定), 直到视频源停止 / 用户停止 / 画面长时间
    # 无变化自动收尾。没有'看到能答就提前退出'的一次性(qa)模式。
    # ★ 每轮攒帧现在走【TTL + 帧数双门】, 由场景 pace 决定 (set_live_watcher 的 ttl
    #   或 FrameBuffer.current_scene): 200s/100 · 60s/60 · 30s/40 · 10s/15。
    #   到达目标帧数 或 到达 ttl(取手头全部)就跑一轮。下面这些是【无场景/无 ttl 时的
    #   兜底/测试】旋钮:
    watch_frame_batch: int = 64         # (兼容保留) 每轮送 LLM 的帧上限参考值
    watch_min_batch: int = 64           # 无场景/无 ttl 时的兜底目标帧数
    watch_round_ttl_sec: float = 120.0  # 无场景/无 ttl 时的兜底轮 ttl (秒)
    watch_poll_interval: float = 2.0    # 攒帧时的轮询/进度推送间隔 (秒)
    # If raw capture keeps advancing but the dHash-novel tail stays unchanged,
    # run one before/after raw-frame completion check after this interval instead
    # of waiting for the normal 10-200s scene TTL. Raw timestamps must keep
    # advancing, so a dead capture cannot be mistaken for an ended video.
    watch_static_tail_flush_sec: float = 2.0
    # Provider accepts at most 50 images. Keep two slots of headroom for
    # provider-side wrappers and share this budget between recalled/current
    # frames in every Watcher ReAct request.
    watch_request_max_images: int = 48
    # A normal segment may flag a plausible ending without terminating the
    # watcher.  If no dHash-novel frame arrives for this grace period, a small
    # dedicated visual call re-checks the raw capture tail before task_complete.
    # Keep this aligned with static_tail_flush_sec so either detection path
    # responds after the same two-second static boundary.
    watch_completion_confirm_delay_sec: float = 2.0
    # Advanced compatibility knob used only when max_attempts is explicitly
    # raised above the default 1. It is a TOTAL no-novel-scene duration measured
    # from candidate detection, so prior waits/model latency also count.
    watch_completion_confirm_retry_total_sec: float = 8.0
    watch_completion_confirm_max_attempts: int = 1
    watch_completion_confirm_frames: int = 8
    # The verifier uses 0.6-0.79 for a likely (but not explicit) ending. Product
    # policy prefers returning the accumulated report over waiting indefinitely
    # once ending evidence is more likely than playback continuation.
    watch_completion_confirm_min_confidence: float = 0.6
    # watch_stream_idle_stop: DELETED — 帧空闲启发式判"流停止"已废弃 (直播正常间隙会
    #   误判)。改用前端显式 multimodal.source_stopped 信号 (见 watcher_engine._source_stopped)。
    watch_max_rounds: int = 0           # 持续型硬上限 (轮数); 0/负=不限 (默认)。
    #   结束只靠 源显式停止 / 用户停。>0 时才作为"防跑飞"轮数上限 (代码保留能力,
    #   暂不提供开启旋钮)。
    #   #2 周期增量推送: 每 N 批把累积报告推给前端 (不等结束就先给用户看)。
    #   ★ v33: watcher 不产出最终汇总报告 (删 watch_summary_max_tokens/timeout +
    #     watch_silent_stop_rounds) —— 只有过程报告 (每轮 on_round_report), hook 每轮触发。
    watch_report_every_rounds: int = 3

    # ---- ④ SearchWorker (外部检索) ----
    search_max_tokens: int = 4096
    search_max_tool_rounds: int = 3
    # ★ Search 看的帧数: 30 → 1 (只看 anchor). 理由:
    #   - Search 的 LLM 用图只做两件事: 决定调哪个工具 / 写 image_search_crop 的 bbox,
    #     两件事都只需 anchor 那一张
    #   - 多余的帧反而干扰 bbox 决策 (LLM 可能基于几帧前的物体位置画 bbox, 但工具用的是 anchor)
    #   - "物体动态过程"对外部图搜无价值, 图搜是对单张图
    #   - 跟下面 _spawn_delegation 里 rf=[a_frame] 一致, 不再用 bg_frames 兜底
    search_recent_frames: int = 1

    # ---- ⑤ RecallWorker (记忆查询) ----
    recall_max_tokens: int = 2048
    recall_max_rounds: int = 4
    # ★ 深度分析路径的召回 top_k (从 5 提到 12): 配合 OR 分词 + 相关性排序,
    #   更大候选面避免相关旧条目被 recency 截断。
    recall_topk_micro: int = 12
    recall_topk_entity: int = 12
    recall_distill_max_tokens: int = 512
    recall_decide_frames: int = 4            # ★ Recall 决策时附 N 帧画面 (fix #4)
    recall_verify_enabled: bool = True       # ★ 召回末尾视觉验收, 过滤掉不含目标的噪声帧
    recall_verify_max_frames: int = 8        # 单次 verify 最多验几张召回帧
    recall_verify_retries: int = 1           # verify 格式错误/临时过载时额外重试次数
    recall_verify_retry_delay_sec: float = 0.5

    # ---- 对话历史 ----
    conv_max_chars: int = 100_000
    conv_min_turns: int = 1
    conv_max_bg_obs: int = 200

    # ---- 工具 (search_tool 外接) ----
    enable_search: bool = True
    search_tool_path: str = DEFAULT_SEARCH_TOOL_PATH   # 图搜(暂废弃)仍用它 importlib 加载
    # 检索源标签, 逗号分隔。原样透传给外接 search tool, 由它决定认哪些 key。
    text_search_keys: str = "google"
    image_search_keys: str = "google"
    search_threshold: float = 0.6
    text_search_topk: int = 10
    image_search_max_side: int = 720
    # ---- ★ AnySearch (text_search 的可选外部检索后端) ----
    #   POST {anysearch_endpoint} JSON-RPC 2.0 tools/call name="search"。
    #   key 优先级: 环境变量 ANYSEARCH_API_KEY > 这里的 anysearch_api_key。明文写入沿用项目惯例。
    anysearch_endpoint: str = "https://api.anysearch.com/mcp"
    anysearch_api_key: str = ""            # 由 config.yaml 填 as_sk_...(或走 env)
    anysearch_max_results: int = 8         # <=10
    anysearch_timeout: float = 30.0
    # ★ 单条 text_search 结果注入 ReAct 上下文前的字符上限。AnySearch 一条 query 常
    #   返回数十~数百 KB 的正文 (实测 >300KB), 一轮最多 5 条 search 又逐轮累积进
    #   search_log → 直接喂回 react_step 会瞬间撑爆上下文 (费钱 + 拖慢 + 挤掉画面)。
    #   截断到这个上限 (保留头部, 尾部标注被截), 背景补充够用。0/负 = 不截断。
    anysearch_result_max_chars: int = 4000

    # ---- FrameBuffer ----
    # ★ E8 (evolve): buffer_seconds 60 → 1800 (30min).
    #   原版只存 60s 滑动窗 → L2/L3/Reviewer 看历史时大部分帧已被淘汰, 只能拿
    #   FrameStore 里 Writer 自己挑过的关键帧 (有损). 30min 原始 2fps 流全保留 →
    #   长程聚合 / Reviewer 真能拿到接近"原始所有帧"的密集采样, 仍由 _collect 内
    #   的均匀采样收敛到 agg_l2_frames / agg_l3_frames / reviewer_total_frames 上限.
    #   内存估算: 1800s × 2fps × ~80KB/张 (720p JPEG b64) ≈ 280MB, 接受范围.
    #   场景超过 30min 后, 早于 30min 的原始帧仍按 deque maxlen 自动淘汰, 不会爆.
    #   想再省内存可调 buffer_capture_fps=1.0 (减半至 ~140MB).
    buffer_seconds: float = 1800.0
    buffer_capture_fps: float = 2.0

    # Monitor detection separately keeps the latest 60s of every server-received
    # capture frame at this rate. It intentionally bypasses the long-term dHash
    # queue so brief visual targets are not removed as memory duplicates.

    # ---- FrameBuffer 入口去重 (前置抽帧) ----
    # 采集固定 fps=2, 但 push 入口按 dHash 去重: 新帧与最近 <=2 张保留帧海明距 < 阈值
    #   → 丢弃。阈值由 SceneDhashController 每 scene_probe_interval_s 秒用 auxiliary.vision
    #   看几张小图判定场景后动态调。这是“重复距离截止值”: 阈值越大去重越激进;
    #   文字/会议/静态桌面 → 大, 户外/比赛/直播/通话 → 小。
    framebuffer_dhash_threshold_init: int = 6    # 起始/兜底阈值 (居中)
    framebuffer_dhash_threshold_min: int = 2     # 剧烈动态下限 (弱去重, 保留更多帧)
    framebuffer_dhash_threshold_max: int = 20    # 静态场景上限 (强去重, 保留更少帧)

    # ---- SceneDhashController (场景理解 → dHash 阈值) ----
    scene_probe_interval_s: float = 20.0   # 每隔多久判一次场景
    scene_probe_window_s: float = 20.0     # 从近多少秒的 buffer 里抽样
    scene_probe_frames: int = 3            # 均匀抽几张小图给 MLLM 看
    scene_probe_maxside: int = 256         # 抽样图压缩到的最长边 (省 token/带宽)
    scene_probe_quality: int = 50          # 抽样图 JPEG 质量
    scene_probe_use_llm: bool = True       # false → 不调 MLLM, 阈值恒为 init 值
    scene_probe_timeout_s: float = 20.0    # MLLM 调用超时, 超时保持当前阈值
    # (v33: scene_probe_model 删除 — scene probe 直接用 model.memory 端点+模型。)

    # ---- SearchFactStore (session-scoped external-search evidence cache) ----
    # Search facts are produced only from successful external retrievals.  They
    # are deliberately separate from MemoryWriter's observed video/audio memory.
    search_facts_max: int = 64
    search_fact_ttl_sec: float = 3600.0
    search_fact_value_max_chars: int = 4000
    # Deprecated config compatibility.  New code uses ``search_facts_max``;
    # keeping this field lets older profiles load without an unknown-key cliff.
    facts_max: int = 600

    # ---- MemoryStore ----
    mem_db_path: str = ""                              # 空则用临时文件 (session-scoped)
    mem_l2_macro_min_micro: int = 5                    # 攒满 5 个 micro 触发 L2 聚合
    mem_l2_macro_max_duration: float = 180.0           # 或 3min 时长 cap 触发
    mem_l3_super_min_macro: int = 4                    # 攒满 4 个 macro 触发 L3 聚合
    mem_l3_super_max_duration: float = 900.0           # 或 15min 时长 cap 触发
    mem_entity_alias_threshold: float = 0.85           # entity canonicalize 模糊匹配阈值
    mem_aggregator_max_tokens: int = 2048              # ★ evolve: 加大让 narrative_arc 装得下
    mem_aggregator_image_max_side: int = 512           # L2/L3 多图聚合专用缩略图, 防请求体超限
    mem_aggregator_image_jpeg_quality: int = 55
    # ★ E5/E6: L2/L3 聚合升级带图 — 把覆盖时段内的帧均匀采样 N 张给 LLM 看,
    #   出"带图的" macro/super summary + narrative_arc (setup/rising/climax/resolution).
    agg_l2_frames: int = 50         # L2 聚合采样帧数 (默认 50, 兼容常见 50 图上限)
    agg_l3_frames: int = 128        # L3 聚合采样帧数 (默认 128)
    # ---- ⑥ MemoryReviewer (E7: configurable wake / macro 事件触发) ----
    # ★ Reviewer 看 128 帧 (复合采样: 60% 最近 + 40% 全程, 兼顾近期细节和全局演进).
    #   每次 wake 输出 actions: merge_micros / split_micro / revise_micro_desc /
    #   merge_entities / refine_entity / prune_entity / rewrite_macro_summary.
    #   写到 revision_log.
    reviewer_enabled: bool = True
    # ★ 省 token 调优 (2026-07, 从 mm_memory_standalone 对齐): wake 60→120s、
    #   总帧 128→64. 想回精度优先改回 60 / 128.
    reviewer_wake_interval: float = 120.0
    reviewer_total_frames: int = 64
    reviewer_recent_ratio: float = 0.6              # 128 帧里 60% 给最近, 40% 给全程
    reviewer_recent_window_sec: float = 300.0       # "最近" 窗长 (默认 5min)
    reviewer_min_micros: int = 3                    # 积满 N micro 才值得 review
    reviewer_max_tokens: int = 3072                 # 审校思考与 actions JSON 共享输出预算
    reviewer_max_consecutive_failures: int = 10     # 跟 Writer 同款兜底
    reviewer_max_actions_per_round: int = 8         # 单轮最多落 N 个 action, 防 LLM 暴走
    reviewer_min_seg_dur_for_split: float = 4.0     # split_micro 后两段都至少 4s 才允许
    reviewer_image_max_side: int = 512              # reviewer 多图复盘专用缩略图, 避免请求体超限
    reviewer_image_jpeg_quality: int = 55
    reviewer_max_concurrency: int = 1               # 每个 Reviewer endpoint 的最大在途请求数
    reviewer_single_endpoint_interval_sec: float = 0.5  # 只有一个 endpoint 时相邻请求起始间隔
    reviewer_overload_retries: int = 1              # 仅对 429/overload/临时网关错误重试
    reviewer_retry_backoff_sec: float = 2.0         # 指数退避基数: 2s, 4s, ...
    # ★ P1: 专项 Reviewer 开关 + EntityReviewer 省 token 帧预算.
    #   默认只开 Entity + gated Event. Edge 目前是 no-op 骨架, 默认关闭以免空耗 token.
    reviewer_entity_enabled: bool = True
    reviewer_event_enabled: bool = True
    reviewer_edge_enabled: bool = False
    reviewer_entity_frames: int = 12                # EntityReviewer 帧预算 (省 token, 远小于 total)
    reviewer_event_frames: int = 80                 # EventReviewer 最高帧数
    # EventReviewer is expensive because it audits event/macro text with many
    # frames. Run it only for risky macro windows, plus a light periodic sample.
    reviewer_event_gate_enabled: bool = True
    reviewer_event_sample_every_macros: int = 4
    reviewer_event_min_micro_count: int = 8
    reviewer_event_min_entity_state_changes: int = 60
    reviewer_event_min_distinct_entities: int = 15
    reviewer_event_min_asr_cues: int = 12
    reviewer_event_min_asr_chars: int = 600
    # ★ micro 段强制 finalize 兜底 (A): 防 boundary 长期 continue 导致段永不落地.
    #   注意: 帧/entity 关联已在每拍实时落地 (C), 这里只是让 micro "文本行"按时切段,
    #   保证 L2 能正常聚合 + 段内记忆及时可查.
    mem_micro_max_duration: float = 30.0   # 段最长 30s 强制切一个 micro
    # ★ 8→6: 同步 wake_interval 4s→5s 后, 6 拍 * 5s = 30s, 跟 mem_micro_max_duration
    #   严格对齐, 两套兜底语义一致 (老配 8*4=32s, duration 先触发, ticks 沦为冗余).
    mem_micro_max_ticks: int = 6           # 或累积 6 拍 (=30s) 也强制切

    # ---- FrameStore (关键帧持久化, 突破 FrameBuffer 60s 滑动窗) ----
    # 每个 finalized micro 存一张代表帧, 让 Recall/续写能拿到原图.
    frame_store_max: int = 4000                        # LRU 上限 (默认能容纳几小时)
    frame_store_dedup_scan_n: int = 8                  # 入库前扫最近 N 张做去重对比
    # ★ 入库去重: 只保留【层1 精确同帧】(两帧 ts 差 < ts_exact_eps → 物理同一瞬间, 合并,
    #   无视画面)。它防的是"跨重叠 wake 窗, LLM 反复挑中同一张真实帧反复入库"(dt≈0),
    #   是持久化幂等保护。【层2 dHash 画面近似去重已删】: 画面级去重上移到 FrameBuffer 入口。
    frame_store_ts_exact_eps: float = 0.3              # 层1: 同一瞬间的 ts 容差 (秒)
    # 续写时, 把 Recall 召回到的历史帧最多塞几张给 LLM (避免上下文爆掉)
    cont_recall_frames_max: int = 4
    # 推给前端 UI debug 用的缩略图最大边长 (px); 0 表示不缩放
    ui_event_thumb_max_side: int = 480
    ui_event_thumb_jpeg_quality: int = 70

    # ---- in-flight brief (避免 cruise 重复 BRIEF 同一件事) ----
    inflight_brief_ttl: float = 30.0

    # ---- STT (ASR) ----
    # ★ Qwen3-ASR (老 ASR): 用户对 agent 说话用 (单 speaker, 要快). 不动.
    # 端点/密钥不硬编码 —— 通过 ~/.argus/config.yaml 的 multimodal.asr_* 覆盖。
    # 兼容各类云 ASR:
    #   * 自托管服务: multipart file → {"text": ...}, 可不鉴权。
    #   * OpenAI 兼容 (/v1/audio/transcriptions, Groq, Azure OpenAI, 硅基流动等):
    #     设 asr_api_key (→ Authorization: Bearer) + asr_model (→ form 里的 model
    #     字段, 例 "whisper-1" / "FunAudioLLM/SenseVoiceSmall")。返回体解析
    #     {"text"} 或 OpenAI verbose_json 的 {"text"}, 皆已兼容。
    asr_url: str = ""
    asr_timeout: float = 30.0
    asr_run_vllm: bool = False
    asr_api_key: str = ""      # 非空 → 加 Authorization: Bearer 头 (云 API)
    asr_model: str = ""        # 非空 → 加 model form 字段 (OpenAI 兼容 ASR)

    # ---- 环境音 ASR (视频里的人说话) ----
    # ★ 独立的端点/密钥/模型三元组 (env_url / env_api_key / env_asr_model)。
    #   全部留空时回退到用户语音的 asr_* (env 与 user 共用一个 ASR 服务),
    #   保留旧行为; 想给环境音单独配一个云 ASR 就填这三个。
    # ★ env_asr_backend:
    #   - "dashscope"/"qwen" (默认): 整段一条 text, 无 speaker。DashScope
    #     compatible-mode uses input_audio; env_api_key empty falls back to
    #     dashscope_api_key. Plain qwen/internal services still use multipart.
    #   - "whisperx": 多 speaker diarization + word-level ts。用 env_url + 下面的
    #     whisperx_* 调参。实测 5s 短窗下准确率不如 qwen, 暂不默认开。
    env_asr_backend: str = "dashscope"
    env_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"  # 空 → 回退 asr_url
    env_api_key: str = ""                     # 空 → 回退 asr_api_key
    env_asr_model: str = "qwen3-asr-flash"    # 空 → 回退 asr_model (仅 qwen 后端用)
    whisperx_timeout: float = 60.0           # WhisperX 三步 (asr+align+diarize) 比单 ASR 慢

    # ---- DashScope (通义千问) 实时语音: 流式麦克风 ASR + 流式 TTS ----
    # 走 DashScope realtime WebSocket (wss://dashscope.aliyuncs.com/api-ws/v1/realtime)。
    # dashscope_api_key 是按账号的百炼 API Key, 不随仓库分发 —— 留空则实时 ASR/TTS
    # 静默禁用 (功能优雅降级, 不报错)。ASR/TTS 共用这一个 key。
    dashscope_api_key: str = ""
    realtime_asr_enabled: bool = True         # 有 key 才真正生效
    realtime_asr_model: str = "qwen3-asr-flash-realtime"
    realtime_asr_language: str = "zh"
    realtime_asr_sample_rate: int = 16000     # 前端上传 PCM16 的采样率
    # 服务端 VAD 阈值/静音时长 — 决定"说话结束"的判定灵敏度。
    #   threshold 越高越不敏感(不容易被噪音触发,但轻声说话可能被吞);
    #   silence_ms 越大越"耐心"(容忍说话中间的停顿,不会一停就切句)。
    # 默认 0.5 / 1200ms ≈ 听人自然说话的容忍度;想更快断句可调 0.7 / 600ms。
    realtime_asr_vad_threshold: float = 0.5
    realtime_asr_vad_silence_ms: int = 1200
    realtime_tts_enabled: bool = True         # 有 key 才真正生效
    realtime_tts_model: str = "qwen3-tts-flash-realtime"
    realtime_tts_voice: str = "Cherry"
    realtime_tts_sample_rate: int = 24000
    # Playback speed for realtime TTS. 1.0 = default (sounds slow), 1.3 ≈
    # natural conversational pace, up to ~2.0. Deployment honors `speech_rate`.
    realtime_tts_speech_rate: float = 1.3
    whisperx_language: str = "zh"
    whisperx_diarize: bool = True
    whisperx_min_speakers: int = 0           # 0 = 不限 (None 在 Form 里传不方便)
    whisperx_max_speakers: int = 0           # 0 = 不限

    # ---- TTS ----
    tts_url: str = ""
    tts_voice: str = "Vivian"
    tts_language: str = "Chinese"
    tts_instruct: str = ""
    tts_connect_timeout: float = 5.0
    tts_read_timeout: float = 120.0
    tts_min_sentence_chars: int = 6
    tts_max_sentence_chars: int = 120

    # ---- 环境音频 ----
    env_audio_enabled: bool = True
    env_audio_window_sec: float = 5.0
    env_audio_min_rms: float = 0.005
    env_audio_min_text_chars: int = 2
    env_audio_filter_fillers: bool = True
    conv_max_audio_obs: int = 300

    # ---- 调试 / 持久化 ----
    dump_raw: bool = False
    history_log_enabled: bool = True
    history_log_path: str = "ours_results/agent_with_memory_history.jsonl"
