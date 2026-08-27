"""VoiceAgent v2 — 世界快照 (context_provider) + 决策层 (decide_speak) + 分诊层 (decide_route).

按设计 .plans/voice_agent_proactive_upgrade.md §5:
  build_world_snapshot 拉四源: recent_dialogue / trigger_event / voice_last_2 / silence_sec
  Voice 专用 LLM 调用: judge_addressed (是否在跟我说话) / decide_route
  (self 直答或 main_agent 委派) / decide_speak (要不要说) / phrase_utterance
  (口播拟词)。Gateway 将它们统一路由到 auxiliary.text.remote_backend。

设计原则:
- LLM 调用超时严守 (拟词是热路径, 超时兜底走原文, 不卡口播).
- system prompt 直接抄 .plans/voice_agent_proactive_upgrade.md §5 我写的两段.
- 返回结构化 dict, 上层根据 speak/route 做决策.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, List, Optional

log = logging.getLogger("hermes.multimodal.voice_agent.ctx")


def _model_is_kimi_k3(model: str) -> bool:
    return "kimi-k3" in (model or "").replace(" ", "-").lower()


def _log_voice_prompt(call: str, model: str, messages: List[dict],
                      **fields: Any) -> None:
    """把本次真正上线的 system / user 原文写进 voice_chain.log.

    ``[VA_LLM] in=`` 记的是结构化 payload (世界快照 dict), 不是拼装后的 prompt,
    而 system prompt 从来没被记过 —— 想核对"到底给模型看了什么"只能去读源码。
    这里在唯一的出网口按角色 (decide_speak / decide_route / phrase / intent /
    intent_eou) dump 一次, 不截断、不转义。默认关, 与 vtrace 同一个开关。
    """
    try:
        from agent.multimodal.voice_trace import vtrace_prompt
        system = "\n".join(
            str(m.get("content") or "") for m in messages
            if m.get("role") == "system")
        user = "\n".join(
            str(m.get("content") or "") for m in messages
            if m.get("role") == "user")
        vtrace_prompt(f"{call}.prompt", model=model, system=system,
                      user=user, **fields)
    except Exception:
        pass


async def _voice_chat(
    *,
    client: Any,
    model: str,
    messages: List[dict],
    max_tokens: int,
    temperature: float,
    timeout_sec: float,
    call: str = "voice",
) -> Any:
    """chat.completions.create for the voice roles — one attempt, no sampling params.

    ``temperature`` is still accepted so the five call sites keep their intent
    documented, but it is never sent: sampling params are dropped everywhere now
    (see agent/transports/chat_completions.py build_kwargs). That removes the
    reason this helper used to exist — a default→portable retry that reacted to
    "invalid temperature: only 1 is allowed for this model" 400s. With nothing
    to be rejected there is nothing to retry, so a single call is enough.

    ``max_completion_tokens`` is kept for routes that only accept that spelling,
    and widened well past voice's tiny ``max_tokens`` (50–200) because a
    reasoning model's thinking tokens share the completion budget — at 200 the
    reply comes back empty.

    asyncio.TimeoutError propagates untouched to the caller's timeout fallback.
    """
    kwargs = {
        "model": model,
        "messages": messages,
        "stream": False,
        "max_completion_tokens": max(int(max_tokens or 0), 8192),
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    # ★ Cap reasoning on the thinking routes or voice never gets an answer at
    #   all. The budgets are brutal (intent/EOU 2s, decide/phrase 6s) because
    #   they sit in front of a live conversation, while Kimi K3 at default
    #   effort reasons until the gateway's own 60s deadline — the same override
    #   _workers._post applies, whose smoke tests found min/medium run to that
    #   deadline while "low returns usable JSON". Without it every call burns
    #   its whole timeout and falls back (speak anyway / raw passthrough /
    #   assume the sentence ended), silently disabling the decision layer.
    if _model_is_kimi_k3(model):
        kwargs["reasoning_effort"] = "low"
    # Dumped before the wire so the log shows what the model was actually shown
    # even when the call then fails.
    _log_voice_prompt(call, model, messages, max_tokens=max_tokens)
    return await asyncio.wait_for(
        client.chat.completions.create(**kwargs), timeout=timeout_sec)


# ══════════════════════════════════════════════════════════════════
# 1. 世界快照 (context_provider, 四源)
# ══════════════════════════════════════════════════════════════════

@dataclass
class WorldSnapshot:
    """决策/分诊/拟词 三个 LLM 都读它."""
    recent_dialogue: List[dict]   # 主 Agent history 最近 N 轮 (user↔assistant), 每项 {role, content}
    trigger_event: dict           # 本次触发: {"kind": "user_utter"/"monitor"/"watcher"/"main_reply", "text": str, ...}
    voice_last_2: List[str]       # 自己最近说过的 2 句 (防重复)
    silence_sec: float            # 距上次开口秒数
    voice_qa_dialogue: List[dict] = None  # VoiceAgent 独立 QA 队列最近几轮 (fast-reply 快问快答, 不进主 Agent). 与 recent_dialogue 并列、不合并.

    def __post_init__(self):
        if self.voice_qa_dialogue is None:
            self.voice_qa_dialogue = []

    def as_prompt_dict(self) -> dict:
        """全量版 (五源). 目前三个 LLM 都各用精简版 (as_*_prompt_dict), 这个保留作
        调试/兼容参考: recent_dialogue 与 voice_qa_dialogue 是两个独立字段, 不合并。
        """
        return {
            "recent_dialogue": self.recent_dialogue,
            "voice_qa_dialogue": self.voice_qa_dialogue,
            "trigger_event": self.trigger_event,
            "voice_last_2": self.voice_last_2,
            "silence_sec": round(self.silence_sec, 1),
        }

    def as_phrase_prompt_dict(self) -> dict:
        """phrase_utterance (口播编辑) 专用: **去掉 recent_dialogue**, 保留 trigger_event.

        拟词的核心任务是把 trigger_event 的内容改造成一句口播, 主 Agent 的完整对话史
        (recent_dialogue) 对"怎么措辞"意义不大反而占 token, 去掉。留下的:
          - trigger_event: **要改造播报的正主** (raw_text 即它的 text)
          - voice_qa_dialogue: 你和用户的快问快答 (帮语气接得上)
          - voice_last_2: 自己刚说的话 (防措辞重复)
          - silence_sec: 很久没说→可带衔接词; 刚说完→直接切正题
        """
        return {
            "trigger_event": self.trigger_event,
            "voice_qa_dialogue": self.voice_qa_dialogue,
            "voice_last_2": self.voice_last_2,
            "silence_sec": round(self.silence_sec, 1),
        }

    def as_speak_prompt_dict(self) -> dict:
        """decide_speak (决策"要不要说") 专用: 去掉 voice_last_2 (冗余).

        留下的:
          - recent_dialogue / voice_qa_dialogue: 判"是否新信息/是否已说过"
          - trigger_event: 本次要判断的新增内容
          - silence_sec: 判"刚说完没多久 + 不紧急 → 缓一缓别打扰"
        去掉 voice_last_2: 自己说过的话在两条对话历史里已能体现, 单列冗余。
        """
        return {
            "recent_dialogue": self.recent_dialogue,
            "voice_qa_dialogue": self.voice_qa_dialogue,
            "trigger_event": self.trigger_event,
            "silence_sec": round(self.silence_sec, 1),
        }

    def as_route_prompt_dict(self) -> dict:
        """decide_route (分诊) 专用的**精简**版: 只留判 self/main_agent 真正需要的两项.

        分诊只需知道"用户这句话说了啥"(trigger_event) + "我俩刚聊过啥"(voice_qa_dialogue)
        就够判了。刻意去掉:
          - recent_dialogue: 主 Agent 对话对"self 还是 main_agent"无用 (甚至误导)
          - silence_sec:     只跟"要不要打扰"有关, 分诊不看
          - voice_last_2:    自己最近说的话已含在 voice_qa_dialogue 里, 冗余
        字段更少 → prompt 更短 → 分诊更快更准 (它是秒回热路径)。
        """
        return {
            "voice_qa_dialogue": self.voice_qa_dialogue,
            "trigger_event": self.trigger_event,
        }


def build_world_snapshot(
    *,
    session: dict,
    trigger: dict,
    self_recent: List[str],
    last_spoke_ts: float,
    convo_turns: int = 6,
    convo_max_chars: int = 1200,
    self_recent_n: int = 2,
    qa_dialogue: Optional[List[dict]] = None,
    qa_turns: int = 6,
) -> WorldSnapshot:
    """拉五源快照.

    Args:
        session: 会话字典, 需含 "history": List[dict] (角色对话).
        trigger: 本次触发事件 dict, 至少含 "kind" 和 "text".
        self_recent: VoiceAgent 自己说过的话 (FIFO 最新在尾).
        last_spoke_ts: 上次开口时间戳 (time.time()); 0 表示从未开口.
        convo_turns: 主 Agent history 最近对话轮数 (默认 6).
        convo_max_chars: 每轮内容字符上限 (超出取头尾).
        self_recent_n: 自己最近说过的话取几句 (默认 2, 按意见 8).
        qa_dialogue: VoiceAgent 独立 QA 队列 (成对 user↔assistant, 最新在尾).
        qa_turns: QA 队列取最近几轮 (默认 6 轮 = 12 项). 与 history **不合并**.
    """
    history = session.get("history") or []
    recent = _last_n_turns(history, n=convo_turns, max_chars=convo_max_chars)
    now = time.time()
    silence = (now - last_spoke_ts) if last_spoke_ts > 0 else 999999.0
    # QA 队列: 独立字段, 只取最近 qa_turns 轮 (每轮 2 项), 不与 history 合并.
    qa = list((qa_dialogue or [])[-(qa_turns * 2):])
    return WorldSnapshot(
        recent_dialogue=recent,
        trigger_event=trigger,
        voice_last_2=list(self_recent[-self_recent_n:]),
        silence_sec=silence,
        voice_qa_dialogue=qa,
    )


def _last_n_turns(history: List[dict], *, n: int, max_chars: int) -> List[dict]:
    """从 session.history 抽最近 n 项 (role + content 简化, content 截断).

    history 里每项形如 {role: user/assistant/system/tool, content: str/list...}.
    本函数只保留 role/content, 且丢掉 system/tool 消息 (对话上下文不需要).
    content 若是 list (多模态) → 提取其中 text 部分拼接; 超长截 max_chars.
    """
    out: List[dict] = []
    for item in history[-max(n * 3, n + 10):]:   # 多抓一些以过滤 system/tool
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "")
        if role not in ("user", "assistant"):
            continue
        content = item.get("content")
        text = _flatten_content(content)
        if not text:
            continue
        if len(text) > max_chars:
            head = text[: max_chars // 2]
            tail = text[-max_chars // 2:]
            text = f"{head}...[省略]...{tail}"
        out.append({"role": role, "content": text})
    return out[-n:]


def _flatten_content(content: Any) -> str:
    """把 str 或 [{type:text|image_url|...}, ...] 扁平化成纯文本."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                t = p.get("type")
                if t == "text":
                    parts.append(str(p.get("text") or ""))
                elif t in ("image_url", "image", "input_image"):
                    parts.append("[image]")
                elif t in ("audio_url", "audio", "input_audio"):
                    parts.append("[audio]")
        return "\n".join(x for x in parts if x).strip()
    return str(content).strip()


# ══════════════════════════════════════════════════════════════════
# 2. LLM 调用: decide_speak (决策层) + decide_route (主线程分诊)
# ══════════════════════════════════════════════════════════════════

# system prompt 来自 .plans/voice_agent_proactive_upgrade.md §5 (决策层 + 主线程分诊)
_DECIDE_SPEAK_SYSTEM = """You are the decision layer of the Argus voice assistant in a face-to-face voice conversation. A new item has arrived. Decide only whether it should be spoken now; a downstream speech editor will handle wording.

Input fields:
- recent_dialogue: recent exchanges between the main agent and the user.
- voice_qa_dialogue: the voice assistant's own quick conversation with the user. Treat it separately from recent_dialogue.
- trigger_event.kind: the source of the new item.
  - watcher or monitor: a live task explicitly started by the user. Default to speak=true, even when the content is long or technical.
  - main_agent_reply or user_task_result: the result of work delegated by the user. Default to speak=true.
  - other sources such as ambient noise: be conservative.
- trigger_event.text: the item itself. It may be in any language.
- silence_sec: seconds since the assistant last spoke.

Decision policy:
- For watcher, monitor, and task-result items, return speak=false only when the same information has already been spoken and the new item adds nothing. A raw watcher report present in recent_dialogue does not by itself mean the voice assistant already spoke it. Do not suppress an item merely because its topic is similar to earlier video analysis.
- The lack of a new user question is not a reason to suppress watcher or monitor updates; these are proactive tasks.
- Length or formal writing style is not a reason to suppress an item; the speech editor will shorten it.
- For other sources, speak only when the item contains clear new information the user should know now.

Stale task results:
When trigger_event.expiry_check is true, the user spoke again after delegating the task. Compare trigger_event.original_query and trigger_event.text with the later dialogue:
- If the user changed or cancelled the request, or moved to an unrelated topic, return speak=false.
- If the user only clarified or followed up and is still waiting for the result, return speak=true.
- If uncertain, prefer speak=true so a requested result is not silently discarded.

Output JSON only, with no explanation outside the object:
{"speak": true, "reason": "brief reason for debugging"}"""


# ★ #4 语义 EOU (end-of-utterance): 判断"用户这句话在语义上说完了吗"。
#   声学 VAD 只按静音时长切句, 用户"帮我查一下……(停顿思考)……北京明天"中间的停顿
#   会被误判成说完。这个判定补上语义层: 明显没说完 (半截话/悬而未决) → done=false,
#   上游可再等一会儿把后续合并进来, 减少误切。
_DECIDE_ROUTE_SYSTEM = """You are the front line of the Argus voice assistant and must respond immediately to the user's utterance.

Speaking style:
Sound like a familiar friend: natural, warm, empathetic, concise, and conversational. Avoid stiff customer-service acknowledgements. Write the answer in the same language as the user's utterance unless the user requests another language.

Context:
- trigger_event: the user's current utterance; this is what you must handle.
- voice_qa_dialogue: the voice assistant's own recent quick conversation with the user. Use it to resolve follow-ups and references.

Choose route=self when:
- the user is greeting, chatting, or expressing an emotion;
- the question is simple and unrelated to the active task;
- a task-related question can be answered accurately in one short sentence without tools, visual inspection, memory retrieval, external research, computer actions, or deep analysis;
- a follow-up can be answered directly from voice_qa_dialogue.

Choose route=main_agent when:
- the request is complex or requires multi-step reasoning;
- it requires tools, visual details, memory retrieval, external research, computer actions, or deep analysis;
- you are not confident you can answer accurately yourself.

The answer field must never be empty:
- For route=self, provide the direct first-person spoken answer. Keep it natural and warm, with no preamble, and within 60 CJK characters or equivalently brief in other languages.
- For route=main_agent, provide a natural handoff sentence saying that Argus is handling the request in the background and the result will follow shortly. Keep it warm and within 40 CJK characters or equivalently brief. This is only an immediate bridge; the main agent's result will be spoken later.

Output JSON only:
{"route": "self", "answer": "non-empty spoken answer"}
or
{"route": "main_agent", "answer": "non-empty handoff sentence"}"""


@dataclass
class SpeakDecision:
    speak: bool
    reason: str = ""
    # ★ priority + text 已删: decide_speak 只判"说不说"(闸门)。
    #   - 播放优先级固定三级 (由 _default_priority(source) 定)。
    #   - 具体措辞由下游 phrase_utterance (口播编辑) 独家改造, 避免"decide 一遍 phrase
    #     再一遍"的重复改写。


@dataclass
class RouteDecision:
    route: str          # "self" 或 "main_agent"
    answer: str = ""    # self 时的口语答案


# ══════════════════════════════════════════════════════════════════
# 3. 交互 worker 拟词 (§5.5.5, "口播编辑")
# ══════════════════════════════════════════════════════════════════
# 上游已决定"要说这条", 拟词只负责怎么说才自然/口语/有逻辑/说重点.
_PHRASE_UTTERANCE_SYSTEM = """You are the voice assistant's speech editor. An upstream decision has already determined that this item should be spoken now. Your only task is to rewrite it as one natural utterance suitable for direct TTS playback: warm, focused, and concise.

The content to rewrite is trigger_event.text, also provided as raw_text. Other fields are context only and must not be recited:
- voice_qa_dialogue helps preserve conversational continuity and forms of address;
- recently spoken messages help avoid repetition;
- silence duration may justify a light transition after a long pause, while a short pause calls for getting directly to the point;
- pending_queue_depth greater than zero means the result should be especially brief.

Use source to choose tone:
- monitor: a brief, caring notification;
- watcher: a conclusion-focused summary;
- main_agent_reply or user_task_result: a warm first-person spoken rendering of a written result;
- self_answer: a natural conversational response.

For main_agent_reply or user_task_result, original_query is the user request that this result answers. Prefer it over the latest item in recent_dialogue because results may arrive out of order. If original_query is absent, use recent_dialogue only as fallback context.

Rules:
1. Use natural spoken language in the same language as trigger_event.text or original_query unless explicitly requested otherwise. Do not use Markdown, formal prose, acknowledgements, or introductory filler.
2. Keep the result within 60 CJK characters or equivalently brief in other languages.
3. State only the one or two most important points from trigger_event.text.
4. Use a natural transition only when it genuinely continues the conversation.
5. Do not repeat recently spoken content; rephrase only when the meaning is still necessary.
6. Do not refuse, ask a question, request more information, invent details, or expand beyond the available content. The upstream layer has already decided to speak this item.

Output JSON only:
{"text": "one final utterance for TTS"}"""


async def phrase_utterance(
    *,
    client: Any,
    model: str,
    snapshot: WorldSnapshot,
    source: str,
    raw_text: str,
    pending_queue_depth: int = 0,
    max_output_chars: int = 60,
    timeout_sec: float = 6.0,
) -> Optional[str]:
    """交互 worker 拟词 LLM: 把 raw_text 改写成一句口播 (≤max_output_chars 字).

    返回:
        改写后的一句话 (成功且不为空).
        None → 调用方兜底: 直接播 raw_text (passthrough).
    """
    if client is None or not model:
        return None
    payload = {
        "context": snapshot.as_phrase_prompt_dict(),
        "source": source,
        "raw_text": raw_text,
        "pending_queue_depth": pending_queue_depth,
        "max_chars": max_output_chars,
    }
    user_msg = "Reference:\n" + json.dumps(payload, ensure_ascii=False, indent=None)
    _t0 = time.monotonic()
    def _ms() -> float:
        return (time.monotonic() - _t0) * 1000.0
    try:
        resp = await _voice_chat(
            client=client, model=model,
            messages=[
                {"role": "system", "content": _PHRASE_UTTERANCE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=100, temperature=0.6, timeout_sec=timeout_sec,
            call="phrase",
        )
    except asyncio.TimeoutError:
        log.warning("[voice.phrase] timeout after %.1fs", timeout_sec)
        _va_llm_log("phrase", loc="remote", ok=False, ms=_ms(),
                    payload_in=payload, out=None, err=f"timeout>{timeout_sec}s")
        return None
    except Exception as exc:
        log.warning("[voice.phrase] llm err: %s", exc)
        _va_llm_log("phrase", loc="remote", ok=False, ms=_ms(),
                    payload_in=payload, out=None, err=f"llm_err:{exc}")
        return None
    raw = _extract_msg_text(resp)
    obj = _parse_json_relaxed(raw)
    if not obj or not isinstance(obj, dict):
        _va_llm_log("phrase", loc="remote", ok=False, ms=_ms(),
                    payload_in=payload, out=raw[:200], err="parse_fail")
        return None
    text = str(obj.get("text") or "").strip()
    if not text:
        _va_llm_log("phrase", loc="remote", ok=False, ms=_ms(),
                    payload_in=payload, out=obj, err="empty_text")
        return None
    # 硬规则兜底: 超长截断
    if len(text) > max_output_chars:
        text = text[:max_output_chars].rstrip() + "..."
    _va_llm_log("phrase", loc="remote", ok=True, ms=_ms(),
                payload_in=payload, out={"text": text})
    return text


async def decide_speak(
    *,
    client: Any,           # AsyncOpenAI-兼容 client
    model: str,
    snapshot: WorldSnapshot,
    timeout_sec: float = 6.0,
) -> Optional[SpeakDecision]:
    """决策层 LLM: 判本次触发**要不要说** (speak). 不产出措辞 (那是 phrase_utterance 的活).

    超时/失败 → 返回 None (调用方兜底: 直接入队, 交给 phrase 改造).
    ★ 用精简 payload as_speak_prompt_dict (去掉 voice_last_2 —— 冗余). silence_sec
      仍留 (判"刚说完别频繁打扰"), recent/qa 对话仍留 (判重复/新信息)。
    """
    if client is None or not model:
        return None
    _payload = snapshot.as_speak_prompt_dict()
    user_msg = "Reference:\n" + json.dumps(_payload, ensure_ascii=False, indent=None)
    _t0 = time.monotonic()
    def _ms() -> float:
        return (time.monotonic() - _t0) * 1000.0
    try:
        resp = await _voice_chat(
            client=client, model=model,
            messages=[
                {"role": "system", "content": _DECIDE_SPEAK_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=200, temperature=0.4, timeout_sec=timeout_sec,
            call="decide_speak",
        )
    except asyncio.TimeoutError:
        log.warning("[voice.decide_speak] timeout after %.1fs", timeout_sec)
        _va_llm_log("decide_speak", loc="remote", ok=False, ms=_ms(),
                    payload_in=_payload, out=None, err=f"timeout>{timeout_sec}s")
        return None
    except Exception as exc:
        log.warning("[voice.decide_speak] llm err: %s", exc)
        _va_llm_log("decide_speak", loc="remote", ok=False, ms=_ms(),
                    payload_in=_payload, out=None, err=f"llm_err:{exc}")
        return None
    raw = _extract_msg_text(resp)
    obj = _parse_json_relaxed(raw)
    if not obj or not isinstance(obj, dict):
        log.debug("[voice.decide_speak] parse fail raw=%r", raw[:200])
        _va_llm_log("decide_speak", loc="remote", ok=False, ms=_ms(),
                    payload_in=_payload, out=raw[:200], err="parse_fail")
        return None
    dec = SpeakDecision(
        speak=bool(obj.get("speak", False)),
        reason=str(obj.get("reason") or "").strip(),
    )
    _va_llm_log("decide_speak", loc="remote", ok=True, ms=_ms(),
                payload_in=_payload, out={"speak": dec.speak, "reason": dec.reason})
    return dec


async def decide_route(
    *,
    client: Any,
    model: str,
    snapshot: WorldSnapshot,
    timeout_sec: float = 5.0,
) -> Optional[RouteDecision]:
    """主线程分诊 LLM: 判用户话 self 直答 / main_agent 委派.

    trigger 必须是 {"kind": "user_utter", "text": <用户说的话>}.
    超时/失败 → 返回 None (调用方兜底: 默认 route=main_agent 安全 fallback).
    """
    if client is None or not model:
        return None
    # ★ 分诊用精简 payload (只 voice_qa_dialogue + trigger_event), 见
    #   WorldSnapshot.as_route_prompt_dict —— 去掉 recent_dialogue/silence_sec/voice_last_2.
    _payload = snapshot.as_route_prompt_dict()
    user_msg = "Reference:\n" + json.dumps(_payload, ensure_ascii=False, indent=None)
    _t0 = time.monotonic()
    def _ms() -> float:
        return (time.monotonic() - _t0) * 1000.0
    try:
        resp = await _voice_chat(
            client=client, model=model,
            messages=[
                {"role": "system", "content": _DECIDE_ROUTE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=200, temperature=0.3, timeout_sec=timeout_sec,
            call="decide_route",
        )
    except asyncio.TimeoutError:
        log.warning("[voice.decide_route] timeout after %.1fs", timeout_sec)
        _va_llm_log("decide_route", loc="remote", ok=False, ms=_ms(),
                    payload_in=_payload, out=None, err=f"timeout>{timeout_sec}s")
        return None
    except Exception as exc:
        log.warning("[voice.decide_route] llm err: %s", exc)
        _va_llm_log("decide_route", loc="remote", ok=False, ms=_ms(),
                    payload_in=_payload, out=None, err=f"llm_err:{exc}")
        return None
    raw = _extract_msg_text(resp)
    obj = _parse_json_relaxed(raw)
    if not obj or not isinstance(obj, dict):
        log.debug("[voice.decide_route] parse fail raw=%r", raw[:200])
        _va_llm_log("decide_route", loc="remote", ok=False, ms=_ms(),
                    payload_in=_payload, out=raw[:200], err="parse_fail")
        return None
    route = str(obj.get("route") or "").strip().lower()
    if route not in ("self", "main_agent"):
        route = "main_agent"   # 不合法值兜底: 转交更安全
    dec = RouteDecision(
        route=route,
        answer=str(obj.get("answer") or "").strip(),
    )
    _va_llm_log("decide_route", loc="remote", ok=True, ms=_ms(),
                payload_in=_payload, out={"route": dec.route, "answer": dec.answer})
    return dec


# ══════════════════════════════════════════════════════════════════
# 4. 层3 意图前置分类: 这句话是不是在跟 VoiceAgent 说话
# ══════════════════════════════════════════════════════════════════
# 远场语音容易误识别 (电视里的对话/别人聊天/环境噪声被 ASR 识别成用户输入),
# 这一层用最轻量 LLM 秒判"这句话是不是在对我说", 挡在决策/分诊层之前, 避免
# 无效打断 + 无效 LLM 调用。目标延迟 <500ms, 输出 <30 tokens.
_INTENT_ADDRESSED_SYSTEM = """Determine whether the transcribed utterance is addressed to the voice assistant. The utterance may be in any language.

Return addressed=true when it is a command, request, question, greeting, explicit address to Argus or an assistant, or a clear continuation of the user's conversation with the assistant.

Return addressed=false when it is self-talk, an exclamation not directed at the assistant, television or video dialogue, background conversation between other people, meaningless filler or isolated sounds, or speech clearly addressed to someone else.

Use the context hint only to decide whether the utterance continues the assistant conversation.

Output JSON only:
{"addressed": true, "reason": "brief reason"}"""

_INTENT_EOU_SYSTEM = """Classify one ASR-transcribed utterance for a voice assistant. The utterance may be in any language. Make both decisions and output JSON only.

1. speak_to_me: whether the utterance is addressed to the AI voice assistant.
- true for a question, command, greeting, casual conversation with the assistant, or a continuation of the preceding assistant dialogue.
- false for television or background speech, conversation between other people, self-talk, meaningless filler, or an isolated sound or word not directed at the assistant.

2. is_end: whether the utterance is semantically complete and ready to process.
- true when it expresses a complete meaning, question, or instruction, even if informal.
- false when it is clearly unfinished, suspended, or only the beginning of a request.

Output exactly one JSON object with no explanation:
{"speak_to_me": true, "is_end": true}"""


async def judge_addressed_to_me(
    *,
    client: Any,
    model: str,
    user_text: str,
    context_hint: str = "",
    timeout_sec: float = 1.5,
) -> Optional[bool]:
    """层3 前置分类: 这句 ASR final 是不是在跟 VoiceAgent 说话.

    使用 ``auxiliary.text.remote_backend`` 对应的远端 aux LLM。

    Args:
        client: AsyncOpenAI 客户端 (来自 auxiliary.text.remote_backend)
        model: 远端模型名
        user_text: ASR final 文本
        context_hint: 可选的对话上下文提示 (如 "刚刚助手说了 xxx", 帮 LLM 判断延续)
        timeout_sec: 远端超时秒 (超时 → 返回 None, 调用方兜底放行)
    Returns:
        True  = 是在跟我说话, 应处理
        False = 不是, 应丢弃
        None  = 所有路径都失败 → 调用方兜底 (保守: 放行, 宁误接不误拒)
    """
    if not user_text:
        return None
    _in = {"user_text": user_text, "hint": context_hint}
    if client is None or not model or not user_text:
        return None
    payload = f"User utterance: {user_text}"
    if context_hint:
        payload += f"\nContext hint: {context_hint}"
    _t0 = time.monotonic()
    def _ms() -> float:
        return (time.monotonic() - _t0) * 1000.0
    try:
        resp = await _voice_chat(
            client=client, model=model,
            messages=[
                {"role": "system", "content": _INTENT_ADDRESSED_SYSTEM},
                {"role": "user", "content": payload},
            ],
            max_tokens=50, temperature=0.1, timeout_sec=timeout_sec, call="intent",
        )
    except asyncio.TimeoutError:
        log.warning("[voice.intent] timeout after %.1fs", timeout_sec)
        _va_llm_log("intent", loc="remote", ok=False, ms=_ms(),
                    payload_in=_in, out=None, err=f"timeout>{timeout_sec}s")
        return None
    except Exception as exc:
        log.warning("[voice.intent] llm err: %s", exc)
        _va_llm_log("intent", loc="remote", ok=False, ms=_ms(),
                    payload_in=_in, out=None, err=f"llm_err:{exc}")
        return None
    raw = _extract_msg_text(resp)
    obj = _parse_json_relaxed(raw)
    if not obj or not isinstance(obj, dict):
        _va_llm_log("intent", loc="remote", ok=False, ms=_ms(),
                    payload_in=_in, out=raw[:200], err="parse_fail")
        return None
    if "addressed" not in obj:
        _va_llm_log("intent", loc="remote", ok=False, ms=_ms(),
                    payload_in=_in, out=obj, err="no_addressed_key")
        return None
    result = bool(obj.get("addressed"))
    reason = str(obj.get("reason") or "")[:80]
    log.info("[voice.intent] addressed=%s reason=%s", result, reason)
    _va_llm_log("intent", loc="remote", ok=True, ms=_ms(),
                payload_in=_in, out={"addressed": result, "reason": reason})
    return result


async def judge_intent_eou_remote(
    *, client: Any, model: str, user_text: str, hint: str = "",
    timeout_sec: float = 2.0,
) -> dict:
    """用远端 aux LLM 一次判断意图和语义 EOU。

    返回 {"speak_to_me": bool, "is_end": bool}。
    超时 / 解析失败 / 无 client → 保守兜底 {"speak_to_me": True, "is_end": True}
      (宁误接不误拒 + 当作说完, 剩下交给监听超时兜底 flush)。
    """
    _in = {"user_text": user_text, "hint": hint}
    _t0 = time.monotonic()
    def _ms() -> float:
        return (time.monotonic() - _t0) * 1000.0
    _fallback = {"speak_to_me": True, "is_end": True}
    if not user_text or not user_text.strip():
        return {"speak_to_me": False, "is_end": True}
    if client is None or not model:
        _va_llm_log("intent_eou", loc="remote", ok=False, ms=_ms(),
                    payload_in=_in, out=_fallback, err="no_client")
        return _fallback
    payload = f"User utterance: {user_text}"
    if hint:
        payload += f"\nContext hint: {hint}"
    # 日志要记"发给 LLM 决策前最后一刻"的东西 —— user_text/hint 是结构化镜像,
    # payload 才是真正上线的 user 消息。两者都留: 前者好机器解析, 后者是真相。
    _in = {**_in, "user_msg": payload}
    try:
        resp = await _voice_chat(
            client=client, model=model,
            messages=[
                {"role": "system", "content": _INTENT_EOU_SYSTEM},
                {"role": "user", "content": payload},
            ],
            max_tokens=50, temperature=0.1, timeout_sec=timeout_sec, call="intent_eou",
        )
    except asyncio.TimeoutError:
        log.warning("[voice.intent_eou] remote timeout after %.1fs", timeout_sec)
        _va_llm_log("intent_eou", loc="remote", ok=False, ms=_ms(),
                    payload_in=_in, out=_fallback, err=f"timeout>{timeout_sec}s")
        return _fallback
    except Exception as exc:
        log.warning("[voice.intent_eou] remote llm err: %s", exc)
        _va_llm_log("intent_eou", loc="remote", ok=False, ms=_ms(),
                    payload_in=_in, out=_fallback, err=f"llm_err:{exc}")
        return _fallback
    raw = _extract_msg_text(resp)
    obj = _parse_json_relaxed(raw)
    if not obj or not isinstance(obj, dict) \
            or "speak_to_me" not in obj or "is_end" not in obj:
        _va_llm_log("intent_eou", loc="remote", ok=False, ms=_ms(),
                    payload_in=_in, out=(obj or raw[:200]), err="parse_fail")
        return _fallback
    out = {"speak_to_me": bool(obj.get("speak_to_me")),
           "is_end": bool(obj.get("is_end"))}
    log.info("[voice.intent_eou] remote speak_to_me=%s is_end=%s",
             out["speak_to_me"], out["is_end"])
    _va_llm_log("intent_eou", loc="remote", ok=True, ms=_ms(), payload_in=_in, out=out)
    return out


# ══════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════

def _va_llm_log(
    call: str, *, loc: str, ok: bool, ms: float,
    payload_in: Any = None, out: Any = None, err: str = "",
) -> None:
    """统一的 VoiceAgent LLM 调用日志 (单行 key=value, 易解析).

    格式 (一行):
      [VA_LLM] call=<名> loc=remote ok=<0|1> ms=<耗时> in=<json串> out=<json串> err=<json串>
    - call: decide_route / decide_speak / phrase / intent (调用点名)
    - loc:  remote (远端 aux LLM)
    - ok:   1=成功拿到可用结果, 0=超时/报错/解析失败/空
    - ms:   本次调用耗时 (毫秒, 含网络/推理)
    - in:   完整输入 payload (json.dumps → 一行内, 转义安全, 不破坏单行解析)
    - out:  结构化输出 (dict/str/bool → json.dumps); 失败时可为 null
    - err:  错误/失败原因 (无则空串 "")
    解析示例: grep '\\[VA_LLM\\]' log | 按 ' <key>=' 切; in/out 是合法 JSON 可再 parse。
    """
    def _j(v: Any) -> str:
        try:
            return json.dumps(v, ensure_ascii=False)
        except Exception:
            return json.dumps(str(v), ensure_ascii=False)
    try:
        log.info(
            "[VA_LLM] call=%s loc=%s ok=%d ms=%.0f in=%s out=%s err=%s",
            call, loc, 1 if ok else 0, ms,
            _j(payload_in), _j(out), _j(err or ""),
        )
    except Exception:
        pass
    # 同一条记录再进语音链路专用文件 (~/.argus/logs/voice_chain.log), 让 ASR →
    # 意图 → 路由 → 拟词 → 播报 能在一个 tail -f 里按时序对齐。默认关, 由
    # ARGUS_TRACE=1 / config logging.voice_trace 打开。上面的 agent.log 行为不变。
    try:
        from agent.multimodal.voice_trace import vtrace
        vtrace(f"{call}.llm", loc=loc, ok=ok, ms=round(ms),
               **{"in": payload_in, "out": out, "err": err or None})
    except Exception:
        pass


def _extract_msg_text(resp: Any) -> str:
    """OpenAI 风格响应 → assistant.content 字符串."""
    try:
        return resp.choices[0].message.content or ""
    except Exception:
        return ""


def _parse_json_relaxed(raw: str) -> Optional[dict]:
    """LLM 输出 JSON 松解析: 剥 ``` 围栏、抓第一个 { ... } 块."""
    if not raw:
        return None
    s = raw.strip()
    # 剥 markdown 围栏 (```json ... ```)
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    # 抓第一个 { ... } (含嵌套 → 简单实现: 找匹配的最后一个 })
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        return None
    frag = m.group(0)
    try:
        return json.loads(frag)
    except Exception:
        return None


def _coerce_int(v: Any, *, lo: int, hi: int, default: int) -> int:
    """强制成 [lo, hi] 整数, 失败给 default."""
    try:
        n = int(v)
    except Exception:
        return default
    return max(lo, min(hi, n))
