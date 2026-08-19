"""VoiceAgent v2 — 常驻语音交互 Agent (设计 .plans/voice_agent_proactive_upgrade.md).

线程模型 = 1 后台 daemon 线程 + 私有 asyncio loop (run_forever)
  ├─ 主线程 (submit_user 入口): ASR final → 直接派发主 Agent
  ├─ 理解 worker (长驻, 可重入=异步并发): 取用户作业 → create_task 起子协程 →
  │   提交主 Agent → 阻塞等结果 → 结果最高优先入回播队列
  └─ 交互 worker (长驻): 回播队列新条目触发 → 拉四源快照 → 生成口播措辞 → TTS

Phase-2 骨架版: 只实现常驻 loop + 队列 + 空转 worker + start/stop 生命周期.
决策/分诊/口播 LLM 在 Phase 3-6 逐步填充.

工程骨架照抄 MonitorEngine (monitor_engine.py:255-329).
TTS 出口走引擎 (WatcherAgent) 的 enqueue_tts + finish_tts 串行队列。
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, List

from agent.multimodal.voice_agent_context import (
    build_world_snapshot,
    judge_intent_eou_local,
    judge_intent_eou_remote,
    phrase_utterance,
)
from agent.multimodal.voice_trace import vtrace

log = logging.getLogger("hermes.multimodal.voice_agent")


# ─── 优先级常量 (回播队列插入用) ────────────────────────────────────────
# 数字越大越优先。**三级严格隔离**, 同级追加 (FIFO):
#   L1 fast: 秒回类("好的"/self_answer) —— 最高。语音交互体验的核心, 用户话音一落就该听到。
#   L2 task: 主 Agent 作业结果 —— 次之。用户明确要的东西。
#   L3 watcher: 深度分析每轮报告 —— 有 context, 该播报。
#   L4 monitor: 事件命中告知 —— 最低。可静默/延后, 过期即丢。
# PriorityQueue 天然满足: 低优先级可以先出(队列空时任何都行), 高优先级到了下次先出。
PRI_FAST_REPLY  = 100    # L1: fast 回复 (self_answer, "好的", 用户直答)
PRI_TASK_RESULT = 50     # L2: 主 Agent 作业结果 (user_task_result / main_agent_reply)
PRI_WATCHER     = 20     # L3: watcher 每轮报告
PRI_MONITOR     = 10     # L4: monitor 事件命中
PRI_AMBIENT     = PRI_MONITOR  # 兼容旧引用 (最低档)
PRI_MIN         = 0

# 兼容旧名 (代码内已用到, 保持迁移期不断)
PRI_USER_TASK_RESULT = PRI_TASK_RESULT
PRI_HIGH             = PRI_FAST_REPLY
PRI_MID              = PRI_WATCHER
PRI_LOW              = PRI_MONITOR


@dataclass(order=False)
class SpeakItem:
    """回播队列的一个待播条目."""
    priority: int                            # 越大越优先
    seq: int                                 # 单调递增, 同优先级 FIFO
    source: str                              # user_task_result / monitor / watcher / main_agent_reply / self_answer
    text: str                                # 原始文本 (交互 worker 会改写)
    task_id: str = ""                        # 关联的 rid (可选)
    skip_phrase: bool = False                # 跳过拟词 LLM (秒回"好的"等短确认用: 立即播原文, 不等 LLM)
    # 委派发起时用户提问的原文 (只对 user_task_result / main_agent_reply 有意义).
    # 供拟词 LLM 对齐"这条回复是回哪个问题": 主 Agent 多并发 → 结果逆序到达时若无这条,
    # 拟词只能按 recent_dialogue 猜, 极易张冠李戴。带上就是"该结果的原问题=X, 请围绕它口播"。
    origin_query: str = ""
    # 用户开口计数快照，仅保留给旧委派结果的兼容元数据。
    user_seq_at_submit: int = 0
    created_at: float = field(default_factory=time.time)


@dataclass
class UserTask:
    """理解 worker 队列的一个作业条目."""
    seq: int                                 # 提交顺序 (作为 tiebreaker, 用户作业按此排)
    user_text: str                           # 用户说的话 (原样转交给主 Agent)
    submitted_at: float = field(default_factory=time.time)
    # 委派发起时的用户开口计数快照（兼容字段）。
    user_seq_at_submit: int = 0


@dataclass(frozen=True)
class _MainReplyResult:
    """Result delivered to a VoiceAgent waiter, with an explicit speak policy."""

    text: str
    speak: bool = True


class VoiceAgent:
    """常驻语音交互 Agent (Phase-2 骨架).

    生命周期 (照抄 MonitorEngine):
        va = VoiceAgent(engine=..., session=..., submit_main_agent_cb=..., ...)
        va.start()   # 起线程 + 私有 loop + 2 个 worker
        ...
        va.stop()    # 优雅停机
    """

    # (已废弃) 快速回复/分诊逻辑已移除，所有用户话直接派发主 Agent。

    def __init__(
        self,
        *,
        engine: Any,                                  # 有 enqueue_tts/finish_tts 的对象 (复用)
        session: dict,                                # 会话字典 (读 _mm_tts_on / _mm_asr_on)
        sid: str,
        # 主 Agent 提交回调 (由 gateway 注入, 侧信道):
        #   submit_main_agent(text, task_seq) -> None
        # VoiceAgent 内部只管把用户作业塞进主 Agent 输入队列, 不直接调 _run_prompt_submit.
        # task_seq 由 VoiceAgent 维护 (UserTask.seq), gateway 记住它,
        # 主 Agent turn 结束时调 va.notify_main_reply(task_seq, final_text) 回传结果.
        submit_main_agent_cb: Optional[Callable[[str, int], None]] = None,
        # 会话忙判定 (可选): 主 Agent 是否 running (影响作业提交时序)
        is_session_busy: Optional[Callable[[], bool]] = None,
        # Voice 专用远端 LLM：拟词、是否开口、self/main_agent 分诊。
        # Gateway 当前与 intent_client 注入同一个 auxiliary.voice_intent client。
        aux_client: Any = None,
        aux_model: str = "",
        # 层3 意图分类 LLM (config auxiliary.voice_intent). 如果 None →
        # 只走层2规则过滤; 有 client 则规则放行的还要过一遍 LLM 判 "是否在跟我说话".
        intent_client: Any = None,
        intent_model: str = "",
        # 配置读取 (settings.voice_interact_* 那组)
        cfg: Optional[dict] = None,
        emit_cb: Optional[Callable[[str, dict], None]] = None,
    ) -> None:
        self._engine = engine
        self._session = session
        self._sid = sid
        self._submit_main_agent = submit_main_agent_cb or (lambda _text, _seq: None)
        self._is_session_busy = is_session_busy or (lambda: False)
        self._aux_client = aux_client
        self._aux_model = aux_model
        self._intent_client = intent_client
        self._intent_model = intent_model
        self._emit_cb = emit_cb
        # 接收【完整 config】; interaction 段 (voice_* 扁平) + auxiliary.text 都要读。
        self._cfg = cfg or {}
        # 缓存常用子节点。★ v33: voice_* 在顶层 interaction 段; 文本改写/意图分类端点
        #   在 auxiliary.text (含 use_local + local_backend + remote_backend)。
        self._settings: dict = self._cfg.get("interaction") or {}
        self._aux_voice_intent: dict = (
            (self._cfg.get("auxiliary") or {}).get("text") or {}
        )
        # 层2 dedup 用: 上一句被处理的用户话 + 时间戳
        self._last_user_text: str = ""
        self._last_user_ts: float = 0.0
        # ★ 过期校验(方案B)的"用户开口计数": 只在层2+层3 都放行、确认是真用户话时 +1
        #   (见 _admit_user_with_intent_check)。委派任务快照它, 结果回来比对判过期。
        #   刻意不复用 _seq_counter(那个是任务+回播条目混用的全局序号, 不纯是开口次数)。
        self._user_utterance_seq: int = 0
        # ★ EOU 拼接状态机:
        #   _eou_listening: True = 正在拼接（已确认 speak_to_me=true 且 is_end=false）
        #     → 新 final 直接追加 buffer，不再调 LLM
        #   _eou_buffer: 已追加但还未 flush 的各段文本
        #   _eou_timer: 监听中超时句柄（本地 1.5s / 远端 2s 无新话 → 强制 flush）
        #   _eou_interrupted: 本轮是否已打断过 TTS（只打断一次）
        self._eou_listening: bool = False
        self._eou_buffer: List[str] = []
        self._eou_timer: Optional[asyncio.TimerHandle] = None
        self._eou_interrupted: bool = False

        # ── 线程 / loop 生命周期 (对齐 MonitorEngine) ────────────
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._healthy = False

        # ── 三个队列 (都在 loop 线程里创建, 见 _run) ────────────
        # 理解队列: 用户作业 (UserTask). 理解 worker 循环取.
        self._understand_q: Optional[asyncio.Queue] = None
        # 回播队列: SpeakItem 优先级队列. 交互 worker 循环取.
        # asyncio.PriorityQueue 按 tuple (priority_neg, seq) 排序 (取负=大的优先).
        self._speak_q: Optional[asyncio.PriorityQueue] = None
        # 序号发生器 (同优先级 FIFO tiebreaker)
        self._seq_counter = 0

        # ── worker 句柄 ─────────────────────────────────────────
        self._understand_task: Optional[asyncio.Task] = None
        self._interact_task: Optional[asyncio.Task] = None
        # 理解 worker 的子协程集合 (可重入并发, 每作业一个 task)
        # stop 时需要一并 cancel, 防止 "Task was destroyed but it is pending".
        self._pending_subtasks: set = set()

        # ── 状态 (供决策/交互层读) ──────────────────────────────
        # 自己说过的最近 N 句 (已经播出的). 播出完成后追加.
        self._self_recent: List[str] = []
        # ★ 独立 fast-reply QA 对话队列: route=self 的"用户问↔我答"成对入队.
        #   不进主 Agent history (不污染主 Agent), 仅作 VoiceAgent 后续对话的 context
        #   source (与主 Agent history 在快照里是两个独立字段, 不合并)。上限 ~6 轮.
        self._qa_dialogue: List[dict] = []
        # 已入回播队列但还没播的 (seq → text). 决策/拟词时应把它们并入 voice_last_2,
        # 否则 "刚入队还没播出" 的重复内容会被漏判 (S7 bug).
        self._scheduled_by_seq: dict = {}
        self._last_spoke_ts: float = 0.0
        # ★ #2 播放 ack: 记录已交给 TTS 的话 rid → 它在 _self_recent 里的引用信息。
        #   前端打断时回传 record_tts_played(rid, played_ms, total_ms), 据此把
        #   _self_recent 里对应那条**按已播时长比例截断到字符**, 让下游"我说过什么"
        #   对齐到用户真正听到的部分 (而不是把整句当已说)。只留最近一条 (串行 TTS,
        #   同时只有一句在播)。{rid: {"text": 完整文本}}。
        self._flush_by_rid: dict = {}
        # 主 Agent 侧关联表 (Phase 5 用: 提交作业时记 rid, 主 Agent 返回时匹配)
        self._pending_main_tasks: dict = {}   # rid → UserTask

    def _emit_progress(self, phase: str, **payload: Any) -> None:
        if self._emit_cb is None:
            return
        try:
            self._emit_cb("multimodal.trajectory", {
                "worker": "VoiceAgent",
                "phase": phase,
                **payload,
            })
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════
    # 生命周期 (照抄 MonitorEngine:255-329)
    # ══════════════════════════════════════════════════════════════
    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name=f"mm-voice-agent-{self._sid}", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=10.0)
        if not self._healthy:
            log.warning("[voice] start: loop not healthy (sid=%s)", self._sid)
        # Local intent model (BitCPM4-0.5B) lifecycle is owned by the readiness
        # module (agent/multimodal/readiness.py) — same place that runs every
        # other MM startup check + preload + endpoint probe. Here we only
        # consult its verdict to decide whether the runtime should route
        # through the local path or fall back to the remote endpoint.
        try:
            from agent.multimodal.readiness import should_use_local_aux_text
            if not should_use_local_aux_text(self._cfg):
                # Weights missing / load failed / config says remote. Force
                # the in-memory use_local flag off so downstream branches
                # take the remote path immediately without probing again.
                self._aux_voice_intent = dict(self._aux_voice_intent)
                self._aux_voice_intent["use_local"] = False
        except Exception as exc:
            log.debug("[voice] readiness consult skipped: %s", exc)

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        # 队列必须在 loop 线程里创建 (asyncio.Queue 绑定当前 loop).
        self._understand_q = asyncio.Queue()
        self._speak_q = asyncio.PriorityQueue()
        # 起 2 个长驻 worker.
        self._understand_task = loop.create_task(
            self._understand_worker(), name="voice-understand")
        self._interact_task = loop.create_task(
            self._interact_worker(), name="voice-interact")
        self._healthy = True
        self._ready.set()
        log.info("[voice] engine ready (sid=%s)", self._sid)
        try:
            loop.run_forever()
        except Exception as exc:
            log.debug("[voice] loop ended: %s", exc)
        finally:
            try:
                loop.close()
            except Exception:
                pass

    def stop(self) -> None:
        self._stop.set()
        loop = self._loop
        if loop is None:
            return

        async def _teardown():
            # 收集所有要 cancel 的 task, 用 gather 等它们干净结束
            self._cancel_eou_timer()   # 防 loop 关闭后 EOU timer 还 fire
            all_tasks = list(self._pending_subtasks) + [
                t for t in (self._understand_task, self._interact_task) if t is not None
            ]
            self._pending_subtasks.clear()
            for t in all_tasks:
                try:
                    t.cancel()
                except Exception:
                    pass
            # 等 cancel 落地 (每个 task 内部的 finally 走完), 抑制异常
            if all_tasks:
                try:
                    await asyncio.gather(*all_tasks, return_exceptions=True)
                except Exception:
                    pass
            loop.stop()

        try:
            loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(_teardown(), loop=loop))
        except Exception:
            pass
        # 等线程退出 (best effort)
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    # ══════════════════════════════════════════════════════════════
    # TTS output gate
    # ══════════════════════════════════════════════════════════════
    def is_interactive(self) -> bool:
        """Compatibility guard for retired input-routing code paths."""
        return False

    def is_speaker_on(self) -> bool:
        """Return whether automatic spoken output is enabled."""
        try:
            return bool(self._session.get("_mm_tts_on"))
        except Exception:
            return False

    # ══════════════════════════════════════════════════════════════
    # 主线程入口 (跨线程投递, 照抄 MonitorEngine call_soon_threadsafe 模式)
    # ══════════════════════════════════════════════════════════════
    def submit_user(self, text: str) -> None:
        """ASR final 用户说话进来.

        流程:
          1) ★ 层2 本地规则过滤 (太短/纯语气词/连字符/短窗口重复 → 直接丢, 无副作用)
          2) ★ 层3 LLM 意图分类 (async): 判"是否在跟我说话", 不是 → 丢
             ★ 只有层2+层3都放行, 才认为是真的用户输入 → 打断 TTS + 走 route
          3) 打断当前 TTS + 清空回播队列 (让出发声通道)
          4) 直接派发主 Agent (不做分诊快速回复)
        """
        text = (text or "").strip()
        if not text:
            return
        self._emit_progress("user_received", text=text)
        loop = self._loop
        if loop is None:
            log.warning("[voice] submit_user before loop ready")
            return

        # ── 层2: 本地规则过滤 (零成本, 直接丢) ──
        drop_reason = self._layer2_filter(text)
        if drop_reason:
            log.info("[voice] L2 drop: %s | text=%r", drop_reason, text[:60])
            # 用户说了话但被本地规则吞掉 —— 排查"我说了它没反应"的头号嫌疑。
            vtrace("intent.l2_drop", sid=self._sid, reason=drop_reason, text=text)
            self._emit_progress("user_filtered", text=text, reason=drop_reason)
            return

        # ── 层3 + 后续: 放到 async task 里 (LLM 意图 + 打断 + route) ──
        try:
            loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(
                    self._admit_user_with_intent_check(text), loop=loop))
        except Exception as exc:
            log.debug("[voice] submit_user failed: %s", exc)

    def _layer2_filter(self, text: str) -> Optional[str]:
        """层2 本地规则过滤. 返回 None=放行, 返回 str=丢弃原因."""
        min_chars = int(self._settings.get("voice_filter_min_chars", 3))
        if len(text) < min_chars:
            return f"too_short ({len(text)}<{min_chars})"
        # 纯语气词/单字重复 (如 "嗯嗯嗯"/"啊呀"/"哦哦哦")
        _fillers = {"嗯", "啊", "哦", "呃", "唉", "喂", "咦", "唔", "哈", "哎"}
        _stripped = text.strip("。！？，、,.?!… ")
        if _stripped and all(c in _fillers for c in _stripped):
            return "pure_filler"
        # 全同一字符 ("啊啊啊"/"哈哈哈哈")
        if len(_stripped) >= 2 and len(set(_stripped)) == 1:
            return "single_char_repeat"
        # 短窗口内 dedup (ASR 打字机幻觉复述前一句)
        dedup_win = float(self._settings.get("voice_filter_dedup_window_sec", 2.0))
        dedup_ratio = float(self._settings.get("voice_filter_dedup_ratio", 0.8))
        if self._last_user_text and (time.time() - self._last_user_ts) < dedup_win:
            sim = _text_similarity(text, self._last_user_text)
            if sim >= dedup_ratio:
                return f"dedup_recent (sim={sim:.2f})"
        return None

    async def _admit_user_with_intent_check(self, text: str) -> None:
        """EOU 拼接状态机入口 (本地/远端统一逻辑):

        1) 若当前是【监听中】(_eou_listening=True):
             → 直接追加 buffer，(重)置超时，不调 LLM。
        2) 若当前【非监听中】:
             → 调 LLM 判 {speak_to_me, is_end}
               speak_to_me=false → 丢
               speak_to_me=true, is_end=true → 打断TTS + 直接 flush
               speak_to_me=true, is_end=false → 打断TTS + 进入监听中 + 追加 + 置超时

        本地模式 (use_local=true): LLM=本地 BitCPM，超时 1.5s
        远端模式 (use_local=false): LLM=远端 deepseek，超时 2s

        EOU 始终开启，无关闭开关。两种模式都走同一套状态机。
        """
        use_local = bool(self._aux_voice_intent.get("use_local", False))

        # ── 监听中: 直接追加，不调 LLM ──────────────────────────────────────
        if self._eou_listening:
            self._eou_buffer.append(text.strip())
            log.info("[voice] EOU append (listening) seg=%d text=%r",
                     len(self._eou_buffer), text[:60])
            self._notify_buffer_updated()
            max_segs = int(self._settings.get("voice_eou_max_buffer_segs", 5))
            if len(self._eou_buffer) >= max_segs:
                log.info("[voice] EOU buffer full → flush")
                self._flush_eou_buffer()
                return
            # 每次新段进来都重置超时 (从本次算起)
            self._arm_eou_timer(use_local=use_local)
            return

        # ── 非监听中: 调 LLM 判意图 + EOU ────────────────────────────────────
        hint = ""
        silence = time.time() - self._last_spoke_ts if self._last_spoke_ts else 999.0
        if silence < 15.0 and self._self_recent:
            hint = f"助手刚说过: {self._self_recent[-1][:60]}"

        if use_local:
            r = await judge_intent_eou_local(text, hint)   # 纯本地, 兜底 {True, True}
            speak_to_me = bool(r.get("speak_to_me", True))
            is_end = bool(r.get("is_end", True))
        else:
            # ★ 远端也【真正判 is_end】(合并一次调用出 {speak_to_me, is_end}),
            #   与本地同逻辑, 不再硬编码 is_end=True。超时/失败兜底 {True, True}
            #   (保守放行 + 当说完, 剩下交给监听超时机制 2s 兜底 flush)。
            timeout = float(self._settings.get("voice_intent_check_timeout_sec", 2.0))
            r = await judge_intent_eou_remote(
                client=self._intent_client, model=self._intent_model,
                user_text=text, hint=hint, timeout_sec=timeout,
            )
            speak_to_me = bool(r.get("speak_to_me", True))
            is_end = bool(r.get("is_end", True))

        if not speak_to_me:
            log.info("[voice] EOU drop: not_addressed | text=%r", text[:60])
            vtrace("intent.drop", sid=self._sid, reason="not_addressed", text=text)
            return

        # speak_to_me=true → 确认是跟我说的, 打断 TTS (只一次)
        if not self._eou_interrupted:
            self._interrupt_current_playback()
            self._eou_interrupted = True

        self._eou_buffer.append(text.strip())
        self._notify_buffer_updated()

        if is_end:
            log.info("[voice] EOU is_end=true → flush immediately")
            self._flush_eou_buffer()
        else:
            # is_end=false → 进入监听中, 置超时
            self._eou_listening = True
            log.info("[voice] EOU is_end=false → entering listening state")
            self._arm_eou_timer(use_local=use_local)

    def _notify_buffer_updated(self) -> None:
        """通知 gateway 侧当前 buffer 内容已更新 (供前端显示已拼接段)。
        VoiceAgent 持有 _session, gateway 侧注册了 _mm_eou_buffer_cb 就回调。"""
        cb = self._session.get("_mm_eou_buffer_cb")
        if callable(cb):
            try:
                cb(list(self._eou_buffer))
            except Exception as exc:
                log.debug("[voice] eou_buffer_cb err: %s", exc)

    def _arm_eou_timer(self, *, use_local: bool = True) -> None:
        """(重)设监听中超时: 超时无新话 → 强制 flush。
        本地 1.5s / 远端 2s。每次新 final 到来都重置。"""
        self._cancel_eou_timer()
        if self._loop is None:
            return
        delay_key = "voice_eou_timeout_sec" if use_local else "voice_eou_remote_timeout_sec"
        delay = float(self._settings.get(delay_key, 1.5 if use_local else 2.0))
        def _fire():
            self._eou_timer = None
            if self._eou_buffer:
                log.info("[voice] EOU timeout %.1fs → flush (%d segs)",
                         delay, len(self._eou_buffer))
                self._flush_eou_buffer()
        self._eou_timer = self._loop.call_later(delay, _fire)

    def _cancel_eou_timer(self) -> None:
        if self._eou_timer is not None:
            try:
                self._eou_timer.cancel()
            except Exception:
                pass
            self._eou_timer = None

    def _flush_eou_buffer(self) -> None:
        """整句拼接完成 → 清状态 + 提交。
        按 is_interactive() 分流:
          对话模式 → _route_user (进 VoiceAgent 分诊队列)
          单独开麦 → 走现有语音 turn 路径 (主 Agent, 不进 VoiceAgent 队列)
        """
        self._cancel_eou_timer()
        self._eou_listening = False
        self._eou_interrupted = False
        if not self._eou_buffer:
            return
        _segs = len(self._eou_buffer)   # 链路日志用, 下一行就清空了
        full = " ".join(s for s in self._eou_buffer if s).strip()
        self._eou_buffer = []
        self._notify_buffer_updated()   # buffer 清空, 通知前端
        if not full:
            return
        self._last_user_text = full
        self._last_user_ts = time.time()
        self._user_utterance_seq += 1   # ★ 过期校验(方案B): 整句才 +1
        log.info("[voice] EOU flush: %r → is_interactive=%s", full[:80], self.is_interactive())
        # ★ 环节1 出口: ASR 的整句 transcript。partial / 逐段 final / EOU 累积
        #   都是中间过程, 不记 —— 这一句才是驱动下游所有 LLM 决策的输入。
        vtrace("asr", sid=self._sid, segs=_segs, seq=self._user_utterance_seq,
               interactive=self.is_interactive(), text=full)
        if self.is_interactive():
            # 对话模式: 进 VoiceAgent 分诊
            if self._loop is not None:
                asyncio.ensure_future(self._route_user(full), loop=self._loop)
        else:
            # 单独开麦: 走现有语音 turn 路径 (不进 VoiceAgent 队列)
            cb = self._session.get("_mm_voice_turn_cb")
            if callable(cb):
                try:
                    cb(full)
                except Exception as exc:
                    log.debug("[voice] voice_turn_cb err: %s", exc)

    def _interrupt_current_playback(self) -> None:
        """用户开口 → 立即打断当前 TTS + 清空回播队列.

        - engine.interrupt_tts(): 取消底层 TTS 消费任务, drain 底层 TTS 队列
        - _speak_q.get_nowait() 循环: 清空所有等待中的 SpeakItem
        - _scheduled_by_seq 清空: 让决策层的 recent_all 不再看到被打断的条目
        (只清"未播"的; 已播过的 self_recent 保留, 因为用户是听到那些才反应的)
        """
        # 1. 底层 TTS 打断 (触发引擎 cancel _run_tts + drain 引擎侧队列)
        try:
            if hasattr(self._engine, "interrupt_tts"):
                self._engine.interrupt_tts()
        except Exception as exc:
            log.debug("[voice] engine.interrupt_tts failed: %s", exc)
        # 2. 清 v2 侧回播队列 (asyncio.PriorityQueue, 需在 loop 线程做; 用
        #    call_soon_threadsafe 转发)
        loop = self._loop
        if loop is None:
            return
        def _clear_q():
            q = self._speak_q
            drained = 0
            if q is not None:
                while not q.empty():
                    try:
                        q.get_nowait()
                        drained += 1
                    except Exception:
                        break
            self._scheduled_by_seq.clear()
            if drained:
                log.info("[voice] user interrupt: dropped %d queued items", drained)
            # 打断即使 drained=0 也要记: "用户开口了但队列本来就空"和"没触发打断"
            # 是两种完全不同的故障, 日志里必须能分开。
            vtrace("tts.interrupt", sid=self._sid, drained=drained)
        try:
            loop.call_soon_threadsafe(_clear_q)
        except Exception:
            pass

    def _record_qa(self, user_text: str, self_answer: str) -> None:
        """把一轮 fast-reply QA (route=self) 成对记入独立 QA 对话队列.

        ★ 只在 route=self 调用 (VoiceAgent 自己��的轮次)。route=main_agent 的轮次
          不记这里 —— 那些交主 Agent 处理, 会落到 session.history。
          该队列不进主 Agent, 仅作 VoiceAgent 后续对话的 context (与 history 独立)。
        """
        u = (user_text or "").strip()
        a = (self_answer or "").strip()
        if not u and not a:
            return
        if u:
            self._qa_dialogue.append({"role": "user", "content": u})
        if a:
            self._qa_dialogue.append({"role": "assistant", "content": a})
        # 上限保留最近 ~6 轮 (12 项), 防无限增长.
        if len(self._qa_dialogue) > 12:
            self._qa_dialogue = self._qa_dialogue[-12:]

    async def _route_user(self, text: str) -> None:
        """对话模式入口: 用户话直接派发主 Agent（不做分诊快速回复）.

        快速回复（decide_route → route=self）已移除：LLM 超时/不稳定时会产出
        低质量兜底（如"让Wall-E去办"），体验不可控。现在统一由主 Agent 处理
        所有用户话（含问候）统一由主 Agent 处理。
        """
        if not self.is_interactive():
            log.debug("[voice._route_user] session no longer interactive, skip")
            return
        log.info("[voice._route_user] → main_agent (direct) text=%r", text[:80])
        # ★ 环节3: 路由决策。这里不截断 —— 上面那行的 text[:80] 是给 agent.log
        #   扫的, 链路日志要的是完整决策输入。
        vtrace("route.user", sid=self._sid, route="main_agent", text=text)
        self._emit_progress(
            "route_decision", route="main_agent", local=False,
            fallback=False, user_text=text, answer="")
        self._enqueue_user_task(text)

    def _enqueue_user_task(self, text: str) -> None:
        """在 loop 线程里入理解队列 (Phase-5 会由子协程消费并提交主 Agent)."""
        if self._understand_q is None:
            return
        self._seq_counter += 1
        # ★ 过期校验(方案B): 记下发起时的用户开口计数, 结果回来比对。
        task = UserTask(seq=self._seq_counter, user_text=text,
                        user_seq_at_submit=self._user_utterance_seq)
        try:
            self._understand_q.put_nowait(task)
            self._emit_progress(
                "main_agent_queued", task_seq=task.seq, user_text=text,
                queue_size=self._understand_q.qsize())
        except Exception as exc:
            log.warning("[voice] understand_q put failed: %s", exc)

    def submit(self, source: str, text: str, *, task_id: str = "") -> None:
        """播报旁路入口: server.py 的 monitor/watcher/assistant hooks 调用。"""
        # hook 用 "assistant" 表示主 Agent 回复; 内部规范化为 main_agent_reply.
        src = "main_agent_reply" if source == "assistant" else source
        self.submit_output(src, text, task_id=task_id)

    def submit_output(self, source: str, text: str, *, task_id: str = "") -> None:
        """输出侧 hook 入口；喇叭开时直接入回播队列。"""
        text = (text or "").strip()
        if not text:
            return
        self._emit_progress(
            "output_received", source=source, task_id=task_id,
            text=text)
        loop = self._loop
        if loop is None:
            return
        try:
            loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(
                    self._route_output(source, text, task_id), loop=loop))
        except Exception as exc:
            log.debug("[voice] submit_output failed: %s", exc)

    async def _route_output(self, source: str, text: str, task_id: str) -> None:
        """Queue output whenever the independent speaker switch is on."""
        if not self.is_speaker_on():
            log.debug("[voice._route_output] speaker OFF, drop source=%s", source)
            return
        self._enqueue_speak(source, text, _default_priority(source), task_id)

    def _enqueue_speak(self, source: str, text: str, priority: int, task_id: str,
                       *, skip_phrase: bool = False,
                       origin_query: str = "",
                       user_seq_at_submit: Optional[int] = None) -> None:
        """在 loop 线程里入回播队列.

        同时把 text 登记到 _scheduled_by_seq (占位), 供决策层判"是否重复". 播出时
        (交互 worker 里) 从该 dict 移除 seq. 这修 S7 bug: 决策层看到的 voice_last_2
        原本只含"已播出的话", 队列里还没播的重复条目会漏判.

        skip_phrase=True: 秒回短确认 (如"好的"). **不入队, 不经交互 worker, 立即直送
        TTS**. 因为交互 worker 串行处理: 前面还有 monitor/watcher 的拟词在跑, "好的"
        入队后要等它们播完才轮到, 完全违背"秒回"的设计目的.
        """
        if self._speak_q is None:
            return
        # Fast path: 秒回类 (skip_phrase=True) 直接开一个 fire-and-forget task 送 TTS,
        # 不入队, 不排队, 不拟词. 保证用户话音一落, 立刻听到响应.
        if skip_phrase:
            self._seq_counter += 1
            item_seq = self._seq_counter
            # 登记到 scheduled (决策层的 recent_all 会读到)
            self._scheduled_by_seq[item_seq] = text
            async def _fire_and_forget():
                try:
                    # 秒回快路径也受喇叭门节制 (堵住"喇叭没开但秒回仍出声"的旁路)。
                    # _flush_to_tts 里还有终门兜底, 这里提前 return 省一次 TTS 调用。
                    if not self.is_speaker_on():
                        return
                    await self._flush_to_tts(text, source)
                    self._self_recent.append(text)
                    if len(self._self_recent) > 16:
                        self._self_recent = self._self_recent[-16:]
                    self._last_spoke_ts = time.time()
                finally:
                    self._scheduled_by_seq.pop(item_seq, None)
            # loop 线程内, 直接起 task 立即执行 (不入队, 不受交互 worker 阻塞).
            asyncio.ensure_future(_fire_and_forget(), loop=self._loop)
            return
        self._seq_counter += 1
        # Snapshot the current user-utterance counter so decide/phrase can tell
        # whether the user has moved on since this speak item was queued.
        _use_seq = (user_seq_at_submit if user_seq_at_submit is not None
                    else self._user_utterance_seq)
        item = SpeakItem(priority=priority, seq=self._seq_counter,
                         source=source, text=text, task_id=task_id,
                         skip_phrase=skip_phrase,
                         origin_query=origin_query,
                         user_seq_at_submit=_use_seq)
        try:
            self._speak_q.put_nowait((-item.priority, item.seq, item))
            # 登记占位 (决策层 recent_all() 会读)
            self._scheduled_by_seq[item.seq] = text
            # ★ 环节5 入队: text 此时还是原文, 拟词发生在 _interact_worker 里。
            #   qsize 是排查"为什么播报延迟"的第一现场。
            vtrace("speak.enqueue", sid=self._sid, source=source, pri=priority,
                   seq=item.seq, task_id=task_id or None,
                   qsize=self._speak_q.qsize(), text=text)
        except Exception as exc:
            log.warning("[voice] speak_q put failed: %s", exc)

    def _recent_all(self) -> List[str]:
        """已播过的话 (已确认) + 队列里未播的占位, 按时间顺序.

        决策层/拟词层都用它做 voice_last_2 的源, 避免漏判"刚入队还没播出的重复".
        """
        # scheduled 是 dict{seq → text}, seq 单调递增, 按 seq 升序即时间序.
        scheduled = [t for _, t in sorted(self._scheduled_by_seq.items())]
        # 组合: 已播的在前 (更早), 队列里未播的在后 (更近). 后 N 项就是"最近的话".
        return self._self_recent + scheduled

    # Normalized text form for fuzzy duplicate detection — strip whitespace,
    # punctuation and case so "有人进来了。" ≈ "有人进来了" ≈ "有人进来".
    _DEDUP_NORM_STRIP = "，。！？!?.,;；：: \t\n\r\"'“”‘’()（）[]【】"

    @classmethod
    def _dedup_norm(cls, text: str) -> str:
        s = (text or "").strip()
        for ch in cls._DEDUP_NORM_STRIP:
            s = s.replace(ch, "")
        return s.lower()

    def _is_ambient_stale_or_duplicate(self, item: SpeakItem) -> bool:
        """播报前的 ambient (monitor / watcher) 过期 + 去重裁剪.

        过期规则 (基于 _user_utterance_seq 的推进程度, 用 SpeakItem.user_seq_at_submit
        当基线):
          • monitor: 用户又开口了 (drift ≥ 1) 即丢弃 —— 事件通知的时效性最强, 场景已变.
          • watcher: 用户开口 ≥ 2 次才丢 —— 报告有 context 价值, 只在明显被埋没时才裁.
          • main_agent_reply / user_task_result 不走此函数 (用 _speak_task_result_with_expiry_check).

        同质化规则:
          • 与最近 6 条 _self_recent 归一化后 exact match → 静默丢. 换措辞的相似告警
            (dHash/字面差异) 走拟词层的"不重复"约束, 这里只挡精确复读.
        """
        cur_seq = self._user_utterance_seq
        drift = max(0, cur_seq - int(item.user_seq_at_submit or 0))
        src = (item.source or "").lower()
        if src == "monitor" and drift >= 1:
            log.info("[voice] ambient drop STALE monitor drift=%d text=%r",
                     drift, item.text[:60])
            vtrace("speak.drop", sid=self._sid, reason="stale",
                   source=src, drift=drift, text=item.text)
            return True
        if src in ("watcher", "watcher_report") and drift >= 2:
            log.info("[voice] ambient drop STALE watcher drift=%d text=%r",
                     drift, item.text[:60])
            vtrace("speak.drop", sid=self._sid, reason="stale",
                   source=src, drift=drift, text=item.text)
            return True
        # 精确同质化 (归一化后完全一致) 直接丢。近似相似依赖拟词层判断。
        norm = self._dedup_norm(item.text)
        if norm:
            for prev in self._self_recent[-6:]:
                if self._dedup_norm(prev) == norm:
                    log.info("[voice] ambient drop DUPLICATE %s text=%r", src, item.text[:60])
                    vtrace("speak.drop", sid=self._sid, reason="duplicate",
                           source=src, text=item.text)
                    return True
        return False

    # ══════════════════════════════════════════════════════════════
    # Worker: 理解 (可重入 = 异步并发, 见设计 §0)
    #   worker 本身循环取作业, 每来一个 create_task 起子协程处理, 立刻回到取下一个.
    #   子协程: 提交主 Agent → 阻塞等结果 (通过 Future) → 结果最高优先入回播队列.
    #   → worker 不被单个作业等待阻塞, 多作业并发在等各自主 Agent 结果.
    # ══════════════════════════════════════════════════════════════
    async def _understand_worker(self) -> None:
        """长驻循环: 每取一个委派就起【子协程】并发处理, 不阻塞取下一个.

        ★ 为什么并发 (不能串行): 用户说完一句话不该等结果回来才能说下一句。串行
          await 会让第二句语音一直卡到第一句的主 Agent turn 全跑完 (最长 300s), 体验
          不可接受。所以每个委派起一个子协程立即 await 各自的 Future; 第二个委派进来
          时, 它的 _submit_main 发现主 Agent 忙 → 进 queued_prompts FIFO 排队, turn
          完成后按 session["_voice_active_seq"] 精确配对回传 (排队+精确配对地基已支持
          并发, 见 server.py _submit_main / _drain_queued_prompt)。

        ★ 竞态防护 (取代旧串行的作用): 不再靠"同时只有一个委派"防错配, 改由
          _voice_active_seq 单值精确配对保证 (抢门后主 Agent 仍串行执行一个 turn,
          回传按精确 seq 而非 FIFO pop(0))。并发的只是"等待", 主 Agent 执行本身串行。

        ★ 纯 FIFO, 不丢旧委派: 后一句不一定取代前一句。
        """
        log.info("[voice] understand worker started (sid=%s)", self._sid)
        try:
            while not self._stop.is_set():
                try:
                    task: UserTask = await self._understand_q.get()
                except asyncio.CancelledError:
                    break
                # 并发: 起子协程立即处理, worker 马上回去取下一个 (不阻塞)。
                #   子协程句柄存进 _pending_subtasks, stop() 时统一 cancel;
                #   done 回调自动移除, 防集合无限增长。
                sub = self._loop.create_task(self._handle_user_task(task))
                self._pending_subtasks.add(sub)
                sub.add_done_callback(self._pending_subtasks.discard)
        except Exception as exc:
            log.warning("[voice] understand worker crashed: %s", exc, exc_info=True)

    async def _handle_user_task(self, task: UserTask) -> None:
        """单个用户作业的处理协程 (由 _understand_worker 起【子协程】并发运行, 各等各的).

        1) 创建 Future 用于等主 Agent 结果 (由 notify_main_reply 按 seq 精确唤醒)
        2) 调 submit_main_agent_cb 把用户话丢给主 Agent (忙则进 queued_prompts 排队)
        3) await Future (可能等几十秒到几分钟, 主 Agent 跑完才回来)
        4) 结果经过期校验 (方案B) 后入回播队列
        """
        try:
            fut: asyncio.Future = self._loop.create_future()
            self._pending_main_tasks[task.seq] = fut
            log.info("[voice._handle_user_task] seq=%d → submit_main_agent: %r",
                     task.seq, task.user_text[:60])
            # 提交主 Agent (submit_main_agent_cb 是同步回调, 由 gateway 注入)
            # cb 收到 (text, task.seq), 记住 task.seq, 主 Agent turn 结束时通过
            # notify_main_reply(task.seq, result) 回调唤醒 fut.
            try:
                self._submit_main_agent(task.user_text, task.seq)
            except Exception as exc:
                log.warning("[voice._handle_user_task] submit_main_agent err: %s", exc)
                fut.set_exception(exc)
            # 等主 Agent 结果 (最长等 5 分钟 — 长任务超时保底)
            should_speak = True
            try:
                result = await asyncio.wait_for(fut, timeout=300.0)
                if isinstance(result, _MainReplyResult):
                    should_speak = result.speak
                    result = result.text
            except asyncio.TimeoutError:
                log.warning("[voice._handle_user_task] seq=%d timeout waiting main agent",
                            task.seq)
                result = "任务处理超时了"
            except Exception as exc:
                log.warning("[voice._handle_user_task] seq=%d err: %s", task.seq, exc)
                result = "任务处理出错了"
            finally:
                self._pending_main_tasks.pop(task.seq, None)
            # 结果回来后只尊重显式 speak=False 的内部控制信号。
            result_text = (result or "").strip() or "任务完成"
            if not should_speak:
                log.info(
                    "[voice._handle_user_task] seq=%d resolved silently",
                    task.seq,
                )
                return
            await self._speak_task_result_with_expiry_check(task, result_text)
            log.info("[voice._handle_user_task] seq=%d done", task.seq)
        except asyncio.CancelledError:
            self._pending_main_tasks.pop(task.seq, None)
            raise
        except Exception as exc:
            log.warning("[voice._handle_user_task] crashed seq=%d: %s",
                        task.seq, exc, exc_info=True)

    async def _speak_task_result_with_expiry_check(
        self, task: UserTask, result_text: str
    ) -> None:
        """Queue a delegated result without an LLM speak/silence decision."""
        task_id = f"utask_{task.seq}"
        if not self.is_speaker_on():
            log.info("[voice] task_result dropped: speaker off seq=%d", task.seq)
            return
        self._enqueue_speak("user_task_result", result_text,
                            PRI_USER_TASK_RESULT, task_id=task_id,
                            origin_query=task.user_text,
                            user_seq_at_submit=task.user_seq_at_submit)

    def notify_main_reply(
        self,
        task_seq: int,
        result_text: str,
        *,
        speak: bool = True,
    ) -> None:
        """主 Agent 完成一轮回复时, 由 gateway 侧回调这里唤醒对应等待的作业.

        任一 handle_user_task 通过 _pending_main_tasks[task_seq] 的 Future 等结果.
        gateway 侧关联: submit_main_agent_cb 提交时把 task.seq 记进 session["_voice_pending_seq"],
        主 Agent turn 结束时取出该 seq 调 notify_main_reply(seq, final_text).

        若无匹配 seq (说明这轮 turn 不是 VoiceAgent 触发的) → 走 submit_output("main_agent_reply", ...).
        """
        loop = self._loop
        if loop is None:
            return
        self._emit_progress(
            "main_agent_reply", task_seq=task_seq,
            result_text=(result_text or "")[:8000])
        def _resolve():
            fut = self._pending_main_tasks.get(task_seq)
            if fut is None or fut.done():
                # 不是 VoiceAgent 触发的 turn → 当普通 main_agent_reply 走决策层
                log.debug("[voice.notify_main_reply] no pending seq=%d, fallback submit_output",
                          task_seq)
                return
            try:
                fut.set_result(_MainReplyResult(result_text, speak=speak))
            except Exception as exc:
                log.debug("[voice.notify_main_reply] set_result err: %s", exc)
        try:
            loop.call_soon_threadsafe(_resolve)
        except Exception as exc:
            log.debug("[voice.notify_main_reply] failed: %s", exc)

    # ══════════════════════════════════════════════════════════════
    # Worker: 交互 (Phase-2 骨架 — Phase-6 会填 LLM 拟词, 现在直接 passthrough)
    # ══════════════════════════════════════════════════════════════
    async def _interact_worker(self) -> None:
        """长驻循环: 从回播队列取条目, 拉四源快照, LLM 拟词, 交给 TTS.

        流程 (每个条目):
          1) 取 SpeakItem (阻塞 get)
          2) Post-await liveness re-check (is_interactive?)
          3) 拉世界快照 (触发事件 = 当前 SpeakItem 内容)
          4) LLM 拟词 (超时兜底原文)
          5) TTS 播出 + 更新 self_recent + last_spoke_ts
        """
        log.info("[voice] interact worker started (sid=%s)", self._sid)
        max_chars = int(self._settings.get("voice_interact_output_max_chars", 60))
        timeout_sec = float(self._settings.get("voice_interact_llm_timeout_sec", 6.0))
        try:
            while not self._stop.is_set():
                try:
                    _neg_pri, _seq, item = await self._speak_q.get()
                    item: SpeakItem
                except asyncio.CancelledError:
                    break
                # 无论播出/跳过/异常, 最后必须从 scheduled 移除该 seq (S7 修复配套).
                try:
                    # Post-await liveness re-check.
                    # ★ 用 is_speaker_on (不是 is_interactive): 交互 worker 是所有
                    #   入队条目的唯一消费者。若 gate 在 is_interactive, 则"喇叭开+
                    #   对话关"时深研/monitor 结果入了队却永远不被消费 (Gate-3 死支)。
                    #   收在 speaker 上 → 喇叭开就播, 与 _flush_to_tts 终门一致。
                    if not self.is_speaker_on():
                        log.debug("[voice] interact: skip (speaker off) source=%s",
                                  item.source)
                        continue
                    # ── Pre-phrase filter: ambient (monitor / watcher) 过期裁剪 + 同质化去重 ──
                    #   1) 用户开口计数已推进过阈值 → monitor 直接丢, watcher 至少丢一半
                    #      (监控信息过期没意义 — 场景已改; watcher 报告仍有 context 价值)
                    #   2) 与最近 self_recent 高度雷同 → 静默处理, 不重复念
                    if item.source in ("monitor", "watcher", "watcher_report") \
                            and self._is_ambient_stale_or_duplicate(item):
                        self._scheduled_by_seq.pop(item.seq, None)
                        continue
                    # ★ Fast path: skip_phrase=True (秒回"好的"这类短确认) 跳过拟词 LLM,
                    #   立即送 TTS. 拟词 LLM 那几秒延迟违背 "秒回" 的设计目的.
                    if item.skip_phrase:
                        self._scheduled_by_seq.pop(item.seq, None)
                        await self._flush_to_tts(item.text, item.source)
                        self._self_recent.append(item.text)
                        if len(self._self_recent) > 16:
                            self._self_recent = self._self_recent[-16:]
                        self._last_spoke_ts = time.time()
                        continue
                    # 拉四源快照 (触发事件 = 本条要播的内容).
                    # ★ 关键: recent_all() 组合 已播 + 队列里未播的占位, 让拟词层
                    #   也能看到"刚要说但还没说的话"避免重复措辞.
                    #   注意: 因为本条 item 也在 scheduled dict 里 (入队时登记), 拉快照前
                    #   先移除自身占位, 免得看到自己. finally 里已经会兜底再移除, 这里
                    #   先手动 pop 保证快照干净.
                    self._scheduled_by_seq.pop(item.seq, None)
                    # Carry the origin_query into the trigger so phrase_utterance
                    # can pin "this reply is for THAT question" — critical when
                    # multiple main-agent replies arrive out of order.
                    _trigger: dict = {"kind": item.source, "text": item.text,
                                      "task_id": item.task_id}
                    if item.origin_query:
                        _trigger["original_query"] = item.origin_query
                    snap = build_world_snapshot(
                        session=self._session,
                        trigger=_trigger,
                        self_recent=self._recent_all(),
                        last_spoke_ts=self._last_spoke_ts,
                        convo_turns=self._settings.get("voice_interact_ctx_convo_turns", 6),
                        convo_max_chars=self._settings.get("voice_interact_ctx_convo_max_chars", 1200),
                        self_recent_n=self._settings.get("voice_interact_ctx_self_recent", 2),
                        qa_dialogue=self._qa_dialogue,
                        qa_turns=self._settings.get("voice_qa_dialogue_turns", 6),
                    )
                    # 队列里还剩几条 (提示 LLM 长话短说)
                    pending = self._speak_q.qsize()
                    # LLM 拟词; 失败/超时 → 兜底原文
                    spoken = await phrase_utterance(
                        client=self._aux_client, model=self._aux_model,
                        snapshot=snap, source=item.source, raw_text=item.text,
                        pending_queue_depth=pending, max_output_chars=max_chars,
                        timeout_sec=timeout_sec,
                    )
                    if not spoken:
                        spoken = item.text   # 兜底: 原文 passthrough (设计 §5.5 on_timeout=passthrough)
                    # 再次 liveness re-check (LLM 调用可能几秒, 期间用户可能已关喇叭)
                    if not self.is_speaker_on():
                        log.debug("[voice] interact: skip after phrase (speaker off)")
                        continue
                    await self._flush_to_tts(spoken, item.source)
                    # 记录自己说过的话 (真实拟出的措辞, 可能与入队时的 text 不同).
                    # 保留 16 条: 决策层看 8 条判重复, 拟词层看 2 条防措辞, 16 足够两者用.
                    self._self_recent.append(spoken)
                    if len(self._self_recent) > 16:
                        self._self_recent = self._self_recent[-16:]
                    self._last_spoke_ts = time.time()
                finally:
                    # 双保险: 无论何种路径退出, 确保 seq 已从 scheduled 清理.
                    self._scheduled_by_seq.pop(item.seq, None)
        except Exception as exc:
            log.warning("[voice] interact worker crashed: %s", exc, exc_info=True)

    async def _flush_to_tts(self, spoken: str, source: str) -> None:
        """把一句话交给 TTS (engine.enqueue_tts + finish_tts + hold 占用估计).

        ★ 统一强制门 (唯一出口): 全 v2 的 TTS 都从这里出去 —— 秒回快路径
          (_fire_and_forget)、交互 worker、决策层放行, 无一例外。所以把"喇叭
          是否开"的最终判定收在这一行, 杜绝任何"喇叭没开但仍能出声"的旁路
          (skip_phrase / user_task_result / decide 放行 都被这道墙兜住)。
          语义: 喇叭 OFF (_mm_tts_on=False) 一律不出声。对话模式下前端强制
          _mm_tts_on=True, 故对话模式亦生效。
        """
        if not self.is_speaker_on():
            log.debug("[voice._flush_to_tts] speaker OFF, drop source=%s", source)
            vtrace("tts.drop", sid=self._sid, reason="speaker_off",
                   source=source, text=spoken)
            return
        import secrets as _sec
        rid = "voice_" + source + "_" + _sec.token_hex(3)
        # ★ #2: 记 rid → 完整文本, 供打断时按已播比例截断 _self_recent。只留最近一条
        #   (串行 TTS 同时只有一句在播); 顺带清理旧条防无限增长。
        self._flush_by_rid = {rid: {"text": spoken}}
        try:
            self._engine.enqueue_tts(spoken, rid)
            self._engine.finish_tts(rid)
            # ★ 环节5 唯一出口: spoken 是拟词之后、真正要播的文本。链路的终点。
            vtrace("tts.flush", sid=self._sid, rid=rid, source=source,
                   qsize=(self._speak_q.qsize() if self._speak_q is not None else 0),
                   text=spoken)
        except Exception as exc:
            log.warning("[voice] enqueue_tts failed: %s", exc)
            vtrace("tts.drop", sid=self._sid, reason="enqueue_failed",
                   source=source, err=str(exc), text=spoken)
            return
        # 占用估计: 让"播放中"覆盖大致语音时长
        _SEC_PER_CHAR = 0.18
        _MAX_HOLD_SEC = 30.0
        hold = min(_MAX_HOLD_SEC, max(0.5, len(spoken) * _SEC_PER_CHAR))
        try:
            await asyncio.sleep(hold)
        except asyncio.CancelledError:
            pass

    def record_tts_played(self, rid: str, played_ms: float, total_ms: float) -> None:
        """★ #2 播放 ack (线程安全入口, 由 gateway 在打断时回调)。

        前端打断时回传"这条 rid 实际播了 played_ms / 总时长 total_ms"。据此把
        _self_recent 里对应那句**按已播时长比例截断到字符** —— 让"我说过什么"对齐到
        用户真正听到的部分, 而不是把整句当已说 (否则下轮模型会以为讲过用户没听到的内容)。

        比例截断是近似 (未做文本↔音频精确对齐), 但对"判重复 / 下轮上下文"已足够。
        """
        loop = self._loop
        if loop is None:
            return
        try:
            loop.call_soon_threadsafe(self._apply_tts_played, rid, played_ms, total_ms)
        except Exception as exc:
            log.debug("[voice] record_tts_played schedule failed: %s", exc)

    def _apply_tts_played(self, rid: str, played_ms: float, total_ms: float) -> None:
        """在 loop 线程里按已播比例截断 _self_recent 中对应 rid 的那句。"""
        info = self._flush_by_rid.get(rid)
        if not info:
            return
        full = str(info.get("text") or "")
        if not full:
            return
        try:
            ratio = 1.0 if total_ms <= 0 else max(0.0, min(1.0, played_ms / total_ms))
        except Exception:
            ratio = 1.0
        if ratio >= 0.999:
            return   # 基本播完, 不必截断
        keep = int(len(full) * ratio)
        heard = full[:keep].rstrip()
        # 在 _self_recent 里找到那条完整文本 (通常是最后一条), 替换成"实际听到的部分"。
        for i in range(len(self._self_recent) - 1, -1, -1):
            if self._self_recent[i] == full:
                if heard:
                    self._self_recent[i] = heard
                else:
                    # 一个字都没播到 → 视为没说过, 移除。
                    self._self_recent.pop(i)
                log.info("[voice] tts_played truncate rid=%s ratio=%.2f kept=%d/%d chars",
                         rid, ratio, keep, len(full))
                break
        self._flush_by_rid.pop(rid, None)


# ═══════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════

def _text_similarity(a: str, b: str) -> float:
    """粗略字符集重合度 [0,1]. 用于层2 dedup: ASR 打字机幻觉常复述前一句.
    不做 Levenshtein (对短句/中文分词无优势, 字集重合已够挡典型重复)."""
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def _default_priority(source: str) -> int:
    """source → 四级优先级映射:
      L1 fast (100): self_answer / chitchat (用户直答, 秒回, 不走队列)
      L2 task (50):  user_task_result / main_agent_reply (主 Agent 回给用户提问)
      L3 watcher (20): watcher / watcher_report (深度分析每轮报告)
      L4 monitor (10): monitor (事件命中告知, 最低)
    """
    s = (source or "").lower()
    if s in ("self_answer", "chitchat"):
        return PRI_FAST_REPLY
    if s in ("user_task_result", "main_agent_reply", "assistant"):
        return PRI_TASK_RESULT
    if s in ("watcher", "watcher_report"):
        return PRI_WATCHER
    if s == "monitor":
        return PRI_MONITOR
    return PRI_MIN
