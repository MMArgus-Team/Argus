"""VoiceAgent v2 层3 意图分类 — 本地 Qwen2.5-0.5B 推理 (transformers + torch).

替代远端 auxiliary.voice_intent LLM (P50 ~800ms) 用本地小模型 (P50 ~200-400ms CPU,
<100ms 有 GPU/MPS 时). 只做二分类: 用户 ASR final 是不是在跟 VoiceAgent 说话.

跨平台: transformers + torch 三大 OS 都有官方 PyPI wheel, 无需 build.

设计要点:
- transformers AutoModelForCausalLM + AutoTokenizer 加载 Qwen2.5-0.5B-Instruct
- 权重 safetensors, HF hub 自动缓存 (~1GB); 支持 HF_ENDPOINT 镜像
- 推理: chat template + generate max_new_tokens=1 + 抽首 token → 看 "是"/"否" 分类
- 单例常驻 + 后台预热 (fire-and-forget), 首次调用不阻塞语音链路
- 一切失败静默返回 None → 上游 fallback 远端 (never break voice pipeline)

对外 API:
    ensure_ready_async()             # 启动时触发后台预热
    is_ready() -> bool               # 模型是否已加载
    judge_addressed(text, hint) -> Optional[bool]
        True  = 在跟我说话
        False = 不是 (环境语音)
        None  = 未加载完 / 边界模糊 / 出错 → 让上游走远端裁决

配置 (从 config.auxiliary.voice_intent 读, 方案X合并):
    local_enabled : 总开关 (关 → 直接走远端 fallback)
    local_path    : 本地权重目录 (相对路径 → HERMES_HOME 为根; 不存在则兜底 HF repo)
    local_device  : auto / cpu / cuda / mps
    local_dtype   : auto / float16 / bfloat16 / float32
    local_hf_repo : 下载脚本用 (从此 repo 拉权重到 local_path)
远端 fallback 同节: provider / model / base_url / api_key (走 auxiliary_client).
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Callable, Optional, Tuple

log = logging.getLogger("hermes.multimodal.voice_intent_local")

# ── 模块级单例 ────────────────────────────────────────────────────────
_MODEL: Any = None                              # AutoModelForCausalLM 实例
_TOKENIZER: Any = None                          # AutoTokenizer 实例
_DEVICE: str = ""                               # 实际选定的 device (cuda/mps/cpu)
_LLM_LOADING = False                            # 预热进行中 flag (防重入)
_LLM_LOAD_ERROR: Optional[str] = None           # 最后一次加载错误 (调试用)
_LOCK = threading.Lock()                        # 单例构建/推理互斥

# 默认配置 (config 里可覆盖)
_DEFAULT_DEVICE = "auto"
_DEFAULT_DTYPE = "auto"
# 默认本地目录名。★ 现在指向项目 weights/ 里的 BitCPM4-0.5B (启动时软链到
#   HERMES_HOME/weights, 见 config_sync.sync_project_weights)。config 里 local_path
#   建议直接写相对路径 "weights/bitcpm4-0.5b"; 留空则回退到这个默认目录名 (在
#   HERMES_HOME/models/ 下), 再不存在才回退 HF repo。
_DEFAULT_LOCAL_DIRNAME = "qwen2.5-0.5b-instruct"
# HF repo id 兜底 (config local_path 空/无效路径 且 本地目录也不存在时)
_DEFAULT_HF_REPO = "Qwen/Qwen2.5-0.5B-Instruct"


def default_local_model_dir() -> str:
    """默认本地模型目录: $HERMES_HOME/models/qwen2.5-0.5b-instruct.

    公开给下载脚本用: 下载后要放到这里, 才与运行时加载路径一致.
    """
    hh = os.environ.get("ARGUS_HOME")
    if not hh:
        try:
            from hermes_constants import get_hermes_home
            hh = str(get_hermes_home())
        except Exception:
            hh = os.path.expanduser("~/.argus")
    return os.path.join(hh, "models", _DEFAULT_LOCAL_DIRNAME)


def _resolve_hf_cache_layout(path: str) -> str:
    """自动识别 HF hub cache 布局并解析出真实模型目录.

    HF cache 结构:
        <cache>/models--Org--Repo/
            blobs/       (真实文件)
            snapshots/<hash>/config.json  (软链或副本)
            refs/main    (指向 hash 的引用)
    transformers 的 from_pretrained 需要含 config.json 的目录, 所以自动解析到
    snapshots/<hash>/ 那层. 如果 path 本身就直接含 config.json → 原样返回.
    """
    if not path or not os.path.isdir(path):
        return path
    # 1. path 本身就是含 config.json 的目录 → 直接用
    if os.path.isfile(os.path.join(path, "config.json")):
        return path
    # 2. HF cache 布局: 有 snapshots/ 子目录
    snap_root = os.path.join(path, "snapshots")
    if os.path.isdir(snap_root):
        # 优先: refs/main 里记的 hash
        refs_main = os.path.join(path, "refs", "main")
        if os.path.isfile(refs_main):
            try:
                with open(refs_main, encoding="utf-8") as f:
                    h = f.read().strip()
                cand = os.path.join(snap_root, h)
                if os.path.isfile(os.path.join(cand, "config.json")):
                    log.info("[voice_intent_local] resolved HF cache via refs/main → %s", cand)
                    return cand
            except Exception:
                pass
        # 兜底: snapshots/ 下第一个含 config.json 的子目录
        try:
            for name in os.listdir(snap_root):
                cand = os.path.join(snap_root, name)
                if os.path.isdir(cand) and os.path.isfile(os.path.join(cand, "config.json")):
                    log.info("[voice_intent_local] resolved HF cache via snapshots/* → %s", cand)
                    return cand
        except Exception:
            pass
    # 无法解析 → 原样返回, 让 transformers 自己报错
    return path


def _hermes_home_root() -> str:
    """HERMES_HOME 根目录 (config.yaml 里相对路径的解析根).

    优先环境变量 HERMES_HOME; 否则 get_hermes_home() (Windows 默认
    ~/AppData/Local/hermes, macOS/Linux 默认 ~/.argus).

    ★ 项目约定: config.yaml 里的相对路径以 HERMES_HOME 为根解析
    (不是 CWD, 也不是项目安装目录). 这样用户跨机器部署时只需拷贝
    HERMES_HOME (包含 config + 模型权重 + session db 等), 项目代码可
    单独更新.
    """
    hh = os.environ.get("ARGUS_HOME")
    if hh:
        return hh
    try:
        from hermes_constants import get_hermes_home
        return str(get_hermes_home())
    except Exception:
        return os.path.expanduser("~/.argus")


def _resolve_maybe_relative(path: str) -> str:
    """把 config 里的路径按 HERMES_HOME 为根解析成绝对路径.

    - 绝对路径: 原样返回
    - 相对路径: 拼到 HERMES_HOME 下 (符合项目约定, 见 _hermes_home_root docstring)
    """
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(_hermes_home_root(), path))


def _resolve_config() -> Tuple[str, str, str]:
    """从 argus config 拉 (model_id_or_path, device, dtype). 失败用默认.

    读 auxiliary.text.local_backend.{local_path, local_device, local_dtype}
    (v33: auxiliary.llm→text; 本地小模型配置放 local_backend 子块, use_local 决定启用).

    model_id_or_path:
      - config 有值:
        * 相对路径 → 按 HERMES_HOME 为根解析 (项目约定); 目录存在 → HF cache 布局解析.
        * 绝对路径 且目录存在 → HF cache 布局解析.
        * 目录不存在 → 当作 HF repo id (兜底 HF hub).
      - config 空 → 用 default_local_model_dir(); 若该目录也不存在 → HF _DEFAULT_HF_REPO 兜底.
    """
    try:
        from hermes_cli.config import load_config
        aux = (load_config() or {}).get("auxiliary") or {}
        text = aux.get("text") or {}
        vi = text.get("local_backend") or {}
    except Exception:
        vi = {}
    raw = str(vi.get("local_path") or "").strip()
    device = str(vi.get("local_device") or _DEFAULT_DEVICE).lower()
    dtype = str(vi.get("local_dtype") or _DEFAULT_DTYPE).lower()
    if raw:
        # 相对路径按项目根解析; 绝对路径不变.
        resolved = _resolve_maybe_relative(raw)
        if os.path.isdir(resolved):
            return _resolve_hf_cache_layout(resolved), device, dtype
        # 目录不存在 → 当 HF repo id (原样传, 不用绝对化)
        return raw, device, dtype
    # config 空 → 用默认本地路径; 不存在则兜底 HF repo id.
    default_dir = default_local_model_dir()
    if os.path.isdir(default_dir):
        return _resolve_hf_cache_layout(default_dir), device, dtype
    return _DEFAULT_HF_REPO, device, dtype


def _resolve_device(want: str) -> str:
    """want=auto → 自动选最好的 (cuda > mps > cpu). 显式 want → 尊重但按可用性回退."""
    try:
        import torch
    except ImportError:
        return "cpu"
    want = (want or "auto").lower()
    if want == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if want == "mps":
        return "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    if want == "cpu":
        return "cpu"
    # auto
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(want: str, device: str) -> Any:
    """want=auto → GPU 用 float16, CPU 用 float32 (CPU 上 float16 慢反而)."""
    try:
        import torch
    except ImportError:
        return None
    want = (want or "auto").lower()
    _map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if want in _map:
        return _map[want]
    # auto
    if device == "cuda":
        return torch.float16
    if device == "mps":
        return torch.float16   # MPS 支持 fp16 且更快
    return torch.float32       # CPU 上 fp32 比 fp16 快 (无原生 fp16 硬件)


def _load_model() -> bool:
    """(阻塞) 加载模型 + warmup. 只在预热线程里调用. 返回是否成功."""
    global _MODEL, _TOKENIZER, _DEVICE, _LLM_LOAD_ERROR
    # 1. 懒装依赖
    try:
        from tools.lazy_deps import ensure as _lazy_ensure
        _lazy_ensure("local_llm.qwen05b_transformers", prompt=False)
    except Exception as exc:
        _LLM_LOAD_ERROR = f"deps unavailable: {exc}"
        log.warning("[voice_intent_local] lazy install failed: %s", exc)
        return False
    # 2. import
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        _LLM_LOAD_ERROR = f"import failed: {exc}"
        log.warning("[voice_intent_local] transformers/torch import failed: %s", exc)
        return False
    # 3. 配置解析
    model_id, want_device, want_dtype = _resolve_config()
    device = _resolve_device(want_device)
    dtype = _resolve_dtype(want_dtype, device)
    _src = "local dir" if os.path.isdir(model_id) else "HF hub"
    log.info("[voice_intent_local] loading %s (%s) on %s (dtype=%s)...",
             model_id, _src, device, dtype)
    # 4. 加载 (transformers 会自动 HF cache 到 ~/.cache/huggingface/hub, 首次~1GB)
    try:
        t0 = time.time()
        # ★ BitCPM4-0.5B 是 custom_code (BitNet 伪量化), tokenizer + model 都必须
        #   trust_remote_code=True, 否则 from_pretrained 直接报错。对普通模型
        #   (如旧 Qwen2.5) 传这个也无害。
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        model = model.to(device)
        model.eval()
        log.info("[voice_intent_local] loaded in %.2fs", time.time() - t0)
        _MODEL = model
        _TOKENIZER = tok
        _DEVICE = device
    except Exception as exc:
        _LLM_LOAD_ERROR = f"load failed: {exc}"
        log.warning("[voice_intent_local] load failed: %s", exc, exc_info=True)
        return False
    # 5. warmup (跑一次合并推理, 让 kernel/编译一次到位)
    try:
        _ = _infer_intent_eou("hello", "")
        log.info("[voice_intent_local] warmup done (total %.2fs)", time.time() - t0)
    except Exception as exc:
        log.debug("[voice_intent_local] warmup ignored err: %s", exc)
    return True


def _kick_load_background(
    on_done: Optional[Callable[[bool, str], None]] = None,
) -> None:
    """在后台线程发起加载 (fire-and-forget). 已加载/加载中 → noop.

    这样第一次 judge_addressed() 不会同步阻塞语音链路: 未加载完就返回 None,
    上游 fallback 到远端; 后台加载完成后, 后续调用才用本地.

    ``on_done`` fires ONCE after the first load attempt (success or failure)
    so callers can surface the outcome (e.g. via a UI toast). If a prior load
    already succeeded, the callback fires immediately with (True, "").
    """
    global _LLM_LOADING
    with _LOCK:
        already_ready = _MODEL is not None
        if already_ready:
            if on_done is not None:
                try:
                    on_done(True, "")
                except Exception:
                    pass
            return
        if _LLM_LOADING:
            # A load is already in flight; the earlier caller owns the callback.
            return
        _LLM_LOADING = True

    def _worker():
        global _LLM_LOADING
        ok = False
        err = ""
        try:
            ok = _load_model()
            if not ok:
                err = _LLM_LOAD_ERROR or "load returned False"
        except Exception as exc:
            log.warning("[voice_intent_local] load thread crashed: %s", exc, exc_info=True)
            err = f"crash: {exc}"
        with _LOCK:
            _LLM_LOADING = False
        log.info("[voice_intent_local] background load %s", "OK" if ok else "FAILED")
        if on_done is not None:
            try:
                on_done(ok, err)
            except Exception as _exc:
                log.debug("[voice_intent_local] on_done callback failed: %s", _exc)

    threading.Thread(target=_worker, daemon=True,
                     name="voice-intent-local-load").start()


def ensure_ready_async(on_done: Optional[Callable[[bool, str], None]] = None) -> None:
    """VoiceAgent 启动时调用: 触发后台预热, 不阻塞. 幂等.

    ``on_done(ok, err)`` — optional callback fired ONCE from the background
    load thread after the first attempt finishes. ``ok`` is True on success;
    on failure ``err`` carries the human reason (from _LLM_LOAD_ERROR). Used
    by VoiceAgent to surface load failures to the frontend via
    multimodal.toast instead of silently degrading to remote-only.
    """
    _kick_load_background(on_done=on_done)


def is_ready() -> bool:
    """True 表示模型已加载, 可以调 judge_addressed."""
    with _LOCK:
        return _MODEL is not None


def get_load_error() -> str:
    """Return the most recent load-failure reason (empty when never failed)."""
    return _LLM_LOAD_ERROR or ""


# NOTE: weight-presence checks + downgrade decisions live in
# agent/multimodal/readiness.py (the one MM readiness module — covers every
# preload / check / endpoint probe). This file is purely the BitCPM inference
# kernel — call ensure_ready_async() to have readiness orchestrate the load.


def _log_local_prompt(call: str, messages: list, templated: str) -> None:
    """Dump the on-device prompt verbatim, same format as the remote path.

    ``templated`` is what the model literally consumes — the chat template has
    already wrapped the messages in its role markers, so it can differ from a
    naive system+user concatenation. Both are recorded: ``system``/``user`` for
    reading, ``templated_chars`` so a template that mangled something is
    visible as a length mismatch. Tagged ``loc="local"`` so a local decision is
    never mistaken for a remote one in the same file.
    """
    try:
        from agent.multimodal.voice_trace import vtrace_prompt
        system = "\n".join(
            str(m.get("content") or "") for m in messages
            if m.get("role") == "system")
        user = "\n".join(
            str(m.get("content") or "") for m in messages
            if m.get("role") == "user")
        vtrace_prompt(f"{call}.prompt", model="local", system=system,
                      user=user, loc="local",
                      templated_chars=len(templated or ""))
    except Exception:
        pass


def decide_route_local(text: str, hint: str = "") -> Optional[Tuple[str, str]]:
    """本地 Qwen2.5-0.5B 分诊: 判 self (自己答) / main_agent (委派主 Agent).

    比 judge_addressed 更进一步 —— 不只判 "是否跟我说话", 还判"自己答还是委派".
    远端 decide_route (qwen3.7-plus) 要 2-3s, 本地 0.5B 只要 200ms → 秒回"好的"成为可能.

    Args:
        text: ASR final 文本 (调用方已过 L2+L3, 是"跟我说话")
        hint: 可选对话上下文
    Returns:
        (route, answer) 元组:
            route="self",       answer=<一句简短口语回答>
            route="main_agent", answer=""
        None → 未加载完 / 判定模糊 / 出错 → 上游 fallback 远端精判
    """
    if not text or not text.strip():
        return None
    with _LOCK:
        if _MODEL is None:
            need_kick = not _LLM_LOADING
        else:
            need_kick = False
    if _MODEL is None:
        if need_kick:
            _kick_load_background()
        return None
    try:
        with _LOCK:
            r = _infer_route(text.strip(), hint.strip() if hint else "")
        return r
    except Exception as exc:
        log.warning("[voice_intent_local.decide_route] infer err: %s", exc)
        return None


def _infer_route(text: str, hint: str) -> Optional[Tuple[str, str]]:
    """本地 decide_route 推理: 判 self / main_agent, self 时同时给出简短口语回答.

    ★ prompt 与解析与远端 decide_route **完全统一** (用户要求): 直接复用远端
      _DECIDE_ROUTE_SYSTEM 整段 system prompt + 输出 JSON {"route","answer"} + JSON 解析。
      user 消息也对齐远端: 精简 payload {voice_qa_dialogue, trigger_event} 的等价物
      (本地只拿到 text + hint, 无完整 qa 队列, 故用 trigger_event.text=text + hint 塞
      voice_qa_dialogue 占位, 保持字段名一致让 0.5B 见到熟悉结构)。

    风险: 0.5B 大概率产不出稳定 JSON → _parse_json_relaxed 失败 → 返回 None → 上游
      fallback 远端 (功能不坏, 只是本地命中率可能下降; 真机命中太低可回退首字格式)。
    """
    import time as _time
    _t0 = _time.monotonic()
    model = _MODEL
    tok = _TOKENIZER
    device = _DEVICE
    if model is None or tok is None:
        return None
    try:
        import torch
    except ImportError:
        return None
    # ★ 复用远端分诊的 system prompt + JSON 解析 + 统一日志 (lazy import 防循环)。
    try:
        from agent.multimodal.voice_agent_context import (
            _DECIDE_ROUTE_SYSTEM, _parse_json_relaxed, _va_llm_log,
        )
    except Exception as exc:
        log.debug("[voice_intent_local.decide_route] import remote prompt failed: %s", exc)
        return None
    import json as _json
    # user 消息对齐远端 decide_route 的精简 payload (as_route_prompt_dict):
    #   {voice_qa_dialogue, trigger_event}. 本地无完整 QA 队列, 用 hint 作最近一轮占位。
    qa = []
    if hint:
        qa = [{"role": "assistant", "content": hint}]
    payload = {
        "voice_qa_dialogue": qa,
        "trigger_event": {"kind": "user_utter", "text": text},
    }
    def _ms() -> float:
        return (_time.monotonic() - _t0) * 1000.0
    user = "Reference:\n" + _json.dumps(payload, ensure_ascii=False)
    messages = [
        {"role": "system", "content": _DECIDE_ROUTE_SYSTEM},
        {"role": "user", "content": user},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False,
                                     add_generation_prompt=True)
    _log_local_prompt("decide_route", messages, prompt)
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=200,      # 与远端一致 (JSON 可能比首字格式长)
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    text_out = tok.decode(new_tokens, skip_special_tokens=True).strip()
    if not text_out:
        _va_llm_log("decide_route", loc="local", ok=False, ms=_ms(),
                    payload_in=payload, out=None, err="empty_output")
        return None
    # ★ JSON 解析 (与远端一致): {"route": "self"/"main_agent", "answer": "..."}
    obj = _parse_json_relaxed(text_out)
    if not obj or not isinstance(obj, dict):
        log.debug("[voice_intent_local.decide_route] JSON parse fail: %r", text_out[:80])
        _va_llm_log("decide_route", loc="local", ok=False, ms=_ms(),
                    payload_in=payload, out=text_out[:200], err="parse_fail")
        return None
    route = str(obj.get("route") or "").strip()
    if route == "self":
        answer = str(obj.get("answer") or "").strip() or "嗯"
        _va_llm_log("decide_route", loc="local", ok=True, ms=_ms(),
                    payload_in=payload, out={"route": "self", "answer": answer})
        return ("self", answer)
    if route == "main_agent":
        _va_llm_log("decide_route", loc="local", ok=True, ms=_ms(),
                    payload_in=payload, out={"route": "main_agent", "answer": ""})
        return ("main_agent", "")
    # route 字段缺失/异常 → None, 上游 fallback
    log.debug("[voice_intent_local.decide_route] unexpected route=%r out=%r",
              route, text_out[:80])
    _va_llm_log("decide_route", loc="local", ok=False, ms=_ms(),
                payload_in=payload, out=text_out[:200], err=f"bad_route:{route}")
    return None


# ★ 意图 + 语义 EOU 合并判定的 system prompt (本地 BitCPM 专用)。一次调用同时判两件事:
#   speak_to_me = 这句是不是在跟语音助手说话; is_end = 这句语义上说完整了没。
_INTENT_EOU_SYSTEM = """Classify one ASR-transcribed utterance for a voice assistant. The utterance may be in any language. Make both decisions and output JSON only.

1. speak_to_me: whether the utterance is addressed to the AI voice assistant.
- true for a question, command, greeting, casual conversation with the assistant, or a continuation of the preceding assistant dialogue.
- false for television or background speech, conversation between other people, self-talk, meaningless filler, or an isolated sound or word not directed at the assistant.

2. is_end: whether the utterance is semantically complete and ready to process.
- true when it expresses a complete meaning, question, or instruction, even if informal.
- false when it is clearly unfinished, suspended, or only the beginning of a request.

Output exactly one JSON object with no explanation:
{"speak_to_me": true, "is_end": true}"""


def _infer_intent_eou(text: str, hint: str) -> Optional[dict]:
    """本地 BitCPM 推理: 一���判 {speak_to_me, is_end}。只允许在 _LOCK 已持有时调用。

    与 _infer_route 同款: 组 chat message → generate → _parse_json_relaxed 解析。
    输出非法 JSON / 缺字段 → 返回 None, 上游按纯本地策略兜底 {True, True}。
    """
    import time as _time
    _t0 = _time.monotonic()
    model = _MODEL
    tok = _TOKENIZER
    device = _DEVICE
    if model is None or tok is None:
        return None
    try:
        import torch
    except ImportError:
        return None
    try:
        from agent.multimodal.voice_agent_context import (
            _parse_json_relaxed, _va_llm_log,
        )
    except Exception as exc:
        log.debug("[voice_intent_local.eou] import remote helpers failed: %s", exc)
        return None

    def _ms() -> float:
        return (_time.monotonic() - _t0) * 1000.0

    user = f"User utterance: {text}"
    if hint:
        user += f"\nContext: {hint}"
    messages = [
        {"role": "system", "content": _INTENT_EOU_SYSTEM},
        {"role": "user", "content": user},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False,
                                     add_generation_prompt=True)
    _log_local_prompt("intent_eou", messages, prompt)
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=32,       # 一个小 JSON 足够
            do_sample=False,         # 贪心, 稳定
            pad_token_id=tok.eos_token_id,
        )
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    text_out = tok.decode(new_tokens, skip_special_tokens=True).strip()
    if not text_out:
        _va_llm_log("intent_eou", loc="local", ok=False, ms=_ms(),
                    payload_in={"text": text, "hint": hint, "user_msg": user}, out=None, err="empty")
        return None
    obj = _parse_json_relaxed(text_out)
    if not obj or not isinstance(obj, dict) \
            or "speak_to_me" not in obj or "is_end" not in obj:
        log.debug("[voice_intent_local.eou] JSON parse/field fail: %r", text_out[:80])
        _va_llm_log("intent_eou", loc="local", ok=False, ms=_ms(),
                    payload_in={"text": text, "hint": hint, "user_msg": user},
                    out=text_out[:200], err="parse_fail")
        return None
    result = {"speak_to_me": bool(obj.get("speak_to_me")),
              "is_end": bool(obj.get("is_end"))}
    _va_llm_log("intent_eou", loc="local", ok=True, ms=_ms(),
                payload_in={"text": text, "hint": hint, "user_msg": user}, out=result)
    return result


def judge_addressed_and_eou(text: str, hint: str = "") -> Optional[dict]:
    """本地 BitCPM4-0.5B 一次判 {speak_to_me, is_end}.

    Returns:
        {"speak_to_me": bool, "is_end": bool} = 判定成功
        None = 未加载完 / 解析失败 / 出错 (调用方按纯本地策略兜底为 speak_to_me=true, is_end=true)
    """
    if not text or not text.strip():
        return None
    with _LOCK:
        need_kick = (_MODEL is None) and (not _LLM_LOADING)
    if _MODEL is None:
        if need_kick:
            _kick_load_background()
        return None
    try:
        with _LOCK:
            return _infer_intent_eou(text.strip(), hint.strip() if hint else "")
    except Exception as exc:
        log.warning("[voice_intent_local.eou] infer err: %s", exc)
        return None
