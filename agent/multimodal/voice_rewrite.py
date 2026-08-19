# -*- coding: utf-8 -*-
"""TTS spoken-rewrite layer (part of the main-agent-independent side channel).

Rewrites one piece of to-be-read text (a main-agent assistant / monitor / watcher
bubble) into natural spoken language via a lightweight LLM (auxiliary.llm), then
hands it to TTS. The rewrite output never enters the main agent's history/context.

- The rewrite rules and the current text are merged into a SINGLE user message
  (the proxy endpoint rejects a system-only request with no user turn).
- On LLM failure/timeout, or when the input text is empty, an empty string is
  returned — the caller simply plays nothing. It never raises.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

log = logging.getLogger("hermes.multimodal.voice_rewrite")

# ── 专用日志: 只记 tts 改写的【完整 LLM 请求 JSON】+【输出结果】, 写到独立文件
#    ~/.argus/logs/voice_rewrite.log, 不混进 agent.log。tail -f 该文件即可单独看。 ──
_io_log = logging.getLogger("hermes.multimodal.voice_rewrite.io")
_io_log.propagate = False   # 不冒泡到 root → 不进 agent.log


def _ensure_io_handler() -> None:
    if _io_log.handlers:
        return
    try:
        from hermes_constants import get_hermes_home
        d = get_hermes_home() / "logs"
        d.mkdir(parents=True, exist_ok=True)
        h = logging.FileHandler(d / "voice_rewrite.log", encoding="utf-8")
        h.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        _io_log.addHandler(h)
        _io_log.setLevel(logging.INFO)
    except Exception:
        # 取不到 home / 建不了文件 → 退化到普通 logger (不阻断改写)。
        _io_log.addHandler(logging.NullHandler())

# Instruction head used by the spoken rewrite request. The source text is
# appended by the request builder and sent as a single user message.
_SYSTEM_HEAD = """You are the Wall-E voice module. Rewrite the supplied text as a concise, informative utterance that can be played directly to the user in a face-to-face conversation.
Treat the text as third-party input, but speak from the assistant's first-person perspective. Preserve the input language unless the text explicitly requests another language. Do not add an introduction, restate or evaluate the user's instruction, or describe your reasoning or progress.
Keep the output within 60 characters for CJK text, or equivalently brief in other languages. The user can already see the underlying information, so surface only the most important point.

## Rules (follow strictly)
- Do not use acknowledgement or process language such as confirming receipt, restating requirements, decomposing the task, or announcing analysis steps.
- Do not output reasoning, task interpretation, prefatory filler, or an opening greeting.
- Do not invent, infer, or expand beyond the supplied text.
- Do not generate lists, numbering, alphanumeric IDs, LaTeX, Markdown, or symbols that are difficult to read aloud.

## Speaking style
- Use natural, relaxed spoken language, like a warm and positive person on a video call.

Text to rewrite:"""


def build_system_prompt() -> str:
    """Return the persona/rules head ending with the text-rewrite cue.

    The current text itself is supplied separately by ``build_user_prompt``.
    No previous turn or history is included.
    """
    return _SYSTEM_HEAD


def build_user_prompt(cur_text: str) -> str:
    """The current text to rewrite (follows on from the system head's cue)."""
    return f"{(cur_text or '').strip()}"


async def rewrite_for_tts(client, model, text: str,
                          *, history: Optional[list] = None,
                          timeout: float = 12.0) -> str:
    """Rewrite the current text into a spoken announcement via the LLM. Rules and
    text are merged into a SINGLE user message. history is currently unused.
    Empty input text → returns "" immediately (no LLM call). LLM failure / timeout
    / no client → returns "". Never raises."""
    text = (text or "").strip()
    if not text:
        return ""   # 文本为空 → 直接返回空, 不发 LLM

    # ★ 规则(system 文案) + 本轮正文 合并成【一条 user message】, 不发 system。
    #   代理端点拒绝"只有 system 无 user"的请求 (400 messages cannot be empty), 且
    #   与 scripts/voice_rewrite_replay.py 对齐 —— 信息全放 user。
    merged = f"{build_system_prompt()}\n{build_user_prompt(text)}"

    # ★ 就是【原样发给 LLM 的请求体】—— 建一次, 既用于日志也用于调用, 保证日志=实发。
    payload = {
        "model": model or None,
        "messages": [
            {"role": "user", "content": merged},
        ],
        "max_tokens": 256,
        "temperature": 1.0,
        "stream": False,
    }
    # ── 专用文件日志: 完整请求 JSON, 写到独立的 voice_rewrite.log (不混进 agent.log)。 ──
    _ensure_io_handler()
    _io_log.info("===== REQUEST =====\n%s",
                 json.dumps(payload, ensure_ascii=False, indent=2))

    # ★ 关键修复: 用【同步 client 在线程池里跑】, 而不是把 async client 绑在 VoiceAgent
    #   的引擎 loop 上。async OpenAI/httpx client 跨 event loop 使用会死锁 (第一次可能
    #   成功、后续卡死), 且卡在原生 IO 时连 asyncio.wait_for 都取消不掉 → _play 永不返回
    #   → _playing 卡在 True → "播到一半再也不播"。同步调用放 executor + wait_for 超时,
    #   即使卡住也只占一个线程、能干净超时兜底, 绝不会 wedge 引擎 loop。
    def _blocking_call():
        return client.chat.completions.create(**payload)
    try:
        import asyncio
        loop = asyncio.get_event_loop()
        resp = await asyncio.wait_for(
            loop.run_in_executor(None, _blocking_call), timeout=timeout)
        out = ""
        try:
            out = (resp.choices[0].message.content or "").strip()
        except Exception:
            out = ""
        _io_log.info("===== RESPONSE =====\n%s\n", out or "(空)")
        return out
    except Exception as exc:  # noqa: BLE001
        _io_log.warning("===== ERROR =====\nLLM 改写失败(超时/异常): %s\n", exc)
        return ""


# ── 差异判断: 当前口播内容 vs 历史(过去5轮说过的) 是否有足够差异 ─────────────────
_JUDGE_SYSTEM = (
    "You decide whether a proposed spoken message is meaningfully different "
    "from recently spoken messages.\n"
    "Return True when it contains a clear new fact, change, or conclusion.\n"
    "Return False when it repeats or closely paraphrases prior content without "
    "adding information.\n"
    "Return True when the history is empty.\n"
    "Output exactly one word, True or False, with no explanation or punctuation."
)


async def judge_speak(client, model, spoken: str, history: Optional[list] = None,
                      *, timeout: float = 12.0) -> bool:
    """Second LLM pass: judge whether spoken differs enough from history to be
    worth saying. Returns True (worth saying) / False (too similar to history,
    skip). Empty spoken → False. Empty history or no client → True (no LLM call).
    Failure / timeout / unparsable output → defaults to True (say it)."""
    spoken = (spoken or "").strip()
    if not spoken:
        return False
    hist = [str(h).strip() for h in (history or []) if str(h or "").strip()]
    if not hist:
        return True   # 无历史 → 第一条, 直接说
    if client is None:
        return True   # 无判断模型 → 默认说

    hist_block = "\n".join(f"{i}. {h}" for i, h in enumerate(hist, 1))
    user = (f"Recently spoken messages:\n{hist_block}\n\n"
            f"Proposed message:\n{spoken}\n\n"
            "Is the proposed message meaningfully different and worth saying? "
            "Reply with True or False only.")
    payload = {
        "model": model or None,
        "messages": [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
        "max_tokens": 8,
        "temperature": 0.0,
        "stream": False,
    }
    _ensure_io_handler()
    _io_log.info("===== JUDGE-REQUEST =====\n%s",
                 json.dumps(payload, ensure_ascii=False, indent=2))

    def _blocking_call():
        return client.chat.completions.create(**payload)
    try:
        import asyncio
        loop = asyncio.get_event_loop()
        resp = await asyncio.wait_for(
            loop.run_in_executor(None, _blocking_call), timeout=timeout)
        out = ""
        try:
            out = (resp.choices[0].message.content or "").strip().lower()
        except Exception:
            out = ""
        # 解析: 只有【明确出现 false 且没出现 true】才判 False; 其余(明确 true /
        #   模糊 / 解析不出) 一律 True (默认说)。
        if "false" in out and "true" not in out:
            should = False
        else:
            should = True
        _io_log.info("===== JUDGE-RESPONSE =====\nout=%r → should_speak=%s\n",
                     out or "(空)", should)
        return should
    except Exception as exc:  # noqa: BLE001
        _io_log.warning("===== JUDGE-ERROR =====\n差异判断失败(超时/异常), 默认 True(说): %s\n",
                        exc)
        return True
