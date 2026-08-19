# -*- coding: utf-8 -*-
"""对话模式语音链路的专用链路日志 (ASR → 意图/EOU → 路由 → 拟词 → 播报队列)。

一条命令看全链路::

    tail -f ~/.argus/logs/voice_chain.log

按环节看::

    tail -f ~/.argus/logs/voice_chain.log | grep 'ARGUS| asr'
    tail -f ~/.argus/logs/voice_chain.log | grep 'ARGUS| tts'

设计取舍:

- **独立文件 + propagate=False**: 不冒泡到 root → 不混进 ``agent.log``。语音
  链路的信噪比才是这套日志的全部价值; 和主 Agent/uvicorn/工具日志混在一起
  就等于没有。同 ``voice_rewrite.py`` 的既有范式。
- **只记"发给 LLM 决策前最后一刻"的 transcript/payload**: ASR partial、逐段
  final、EOU 累积、SSE chunk 这些中间过程一概不记 —— 它们只会把真正的决策
  输入淹掉。
- **默认关**: ``vtrace()`` 第一行就 return, payload 字符串完全不构建, 热路径
  零开销。
- **永不抛异常**: 日志失败就静默丢, 绝不打断语音链路。
"""
from __future__ import annotations

import json
import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Any, Optional

log = logging.getLogger("hermes.multimodal.voice_trace")
log.propagate = False   # 不冒泡到 root → 不进 agent.log

# 单个字段值的字符上限。带完整 transcript 的行可以很长 (拟词的 in 里裹着
# 世界快照), 无上限会让 grep 输出彻底不可读。
_DEFAULT_MAXLEN = 2000

# grep 锚点。所有行都以它开头, 所以 `grep 'ARGUS|'` 拿全链路。
_ANCHOR = "ARGUS|"

# stage 名左对齐宽度 —— 让 tail 出来的多行 key=value 目视对齐。
_STAGE_WIDTH = 14

_enabled_cache: Optional[bool] = None


def _config_flag() -> bool:
    """读 config.yaml 的 ``logging.voice_trace``。

    环境变量之外必须有这条路: gateway 是 desktop app / dashboard fork 出来的
    子进程, ``ARGUS_TRACE`` 不一定继承得到; config 是稳的。
    """
    try:
        import yaml
        from hermes_constants import get_config_path
        config_path = get_config_path()
        if not config_path.exists():
            return False
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        log_cfg = cfg.get("logging") or {}
        if isinstance(log_cfg, dict):
            return bool(log_cfg.get("voice_trace"))
    except Exception:
        pass
    return False


def _truthy(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def is_enabled() -> bool:
    """``ARGUS_TRACE=1`` 或 config ``logging.voice_trace: true`` 任一即开启。

    结果缓存: 这个判断在每条链路日志上都要过一遍, 不能每次都去读 yaml。
    改开关需要重启进程 (和 ``logging.level`` 的语义一致)。
    """
    global _enabled_cache
    if _enabled_cache is None:
        env = os.environ.get("ARGUS_TRACE")
        _enabled_cache = _truthy(env) if env is not None else _config_flag()
    return _enabled_cache


def _maxlen() -> int:
    try:
        return max(80, int(os.environ.get("ARGUS_TRACE_MAXLEN", "")))
    except Exception:
        return _DEFAULT_MAXLEN


def _ensure_handler() -> None:
    if log.handlers:
        return
    try:
        from hermes_constants import get_hermes_home
        d = get_hermes_home() / "logs"
        d.mkdir(parents=True, exist_ok=True)
        h = RotatingFileHandler(
            d / "voice_chain.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        h.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        log.addHandler(h)
        log.setLevel(logging.INFO)
    except Exception:
        # 取不到 home / 建不了文件 → 静默降级, 不阻断语音链路。
        log.addHandler(logging.NullHandler())


def _fmt(value: Any) -> str:
    """值 → 单行、转义安全的字符串。

    走 ``json.dumps(ensure_ascii=False)``: 中文保持可读, 换行/引号被转义, 单行
    解析不会被 payload 里的内容破坏。抄 ``_va_llm_log._j``。
    """
    if isinstance(value, bool):
        # bool 在 str() 下是 True/False, 但链路日志里 ok=1/ok=0 更好扫 ——
        # 和既有 [VA_LLM] 的 ok=%d 对齐。注意这条必须在 int 判断之前。
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return str(value)
    try:
        s = json.dumps(value, ensure_ascii=False)
    except Exception:
        s = json.dumps(str(value), ensure_ascii=False)
    limit = _maxlen()
    if len(s) > limit:
        # 截断后补真实长度, 免得"看着完整其实被砍了"。
        s = s[:limit] + "…(+%d chars)" % (len(s) - limit)
    return s


def vtrace_prompt(
    stage: str, *, model: str = "", system: str = "", user: str = "",
    **fields: Any,
) -> None:
    """Dump the exact system/user prompt text sent to an LLM — verbatim.

    Why this is not just another :func:`vtrace` field: prompts are the one thing
    you cannot debug from a summary. ``vtrace`` renders every value through
    ``json.dumps`` and caps it at ``ARGUS_TRACE_MAXLEN`` (2000), which turns a
    system prompt into one 2 KB escaped line with ``\\n`` between every
    sentence — and silently clips the tail of the longer ones. Neither is
    readable, and the payload dict that ``[VA_LLM] in=`` already logs is the
    *structured input*, never the assembled prompt.

    So: a grep-able header line (``ARGUS| <stage>`` + char counts), then the raw
    prompt bodies inside delimiters, unescaped, untruncated, exactly as sent.

    Tradeoff, deliberately taken: the body lines do NOT carry the ``ARGUS|``
    anchor, so ``grep 'ARGUS|'`` shows the headers but not the prompt text. Use
    the delimiters to read a prompt::

        grep -A200 'decide_speak.prompt' ~/.argus/logs/voice_chain.log

    Gated by the same ``is_enabled()`` switch, so it costs nothing when off.
    """
    if not is_enabled():
        return
    try:
        _ensure_handler()
        head = ["%s %-*s" % (_ANCHOR, _STAGE_WIDTH, stage)]
        if model:
            head.append("model=%s" % _fmt(model))
        head.append("system_chars=%d" % len(system or ""))
        head.append("user_chars=%d" % len(user or ""))
        for key, value in fields.items():
            if value is not None:
                head.append("%s=%s" % (key, _fmt(value)))
        block = [" ".join(head)]
        if system:
            block.append("%s ---8<--- %s SYSTEM ---" % (_ANCHOR, stage))
            block.append(system)
        if user:
            block.append("%s ---8<--- %s USER ---" % (_ANCHOR, stage))
            block.append(user)
        block.append("%s --->8--- %s END ---" % (_ANCHOR, stage))
        log.info("\n".join(block))
    except Exception:
        pass


def vtrace(stage: str, **fields: Any) -> None:
    """写一行 grep-able 链路日志。

    ``stage`` 用点号分层 (``tts.flush`` / ``tts.drop``), 这样 ``grep 'ARGUS| tts'``
    能拿到一整个环节。字段顺序即调用时的 kwargs 顺序 (Python 3.7+ 保序), 所以
    把最重要的放前面、``text``/``in``/``out`` 这类长字段放最后。
    """
    if not is_enabled():
        return
    try:
        _ensure_handler()
        parts = ["%s %-*s" % (_ANCHOR, _STAGE_WIDTH, stage)]
        for key, value in fields.items():
            if value is None:
                continue
            parts.append("%s=%s" % (key, _fmt(value)))
        log.info(" ".join(parts))
    except Exception:
        pass
