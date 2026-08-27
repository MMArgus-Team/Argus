"""Multimodal readiness — the single source of truth for "can the user
actually use each multimodal capability, and if not, what's missing / how to
fix it / and load whatever needs loading".

MM is the project's main line; every subsystem (auxiliary.text/ocr/vision,
memory, voice, watcher, monitor, live-research) is part of one lifecycle.
This module runs every check + kicks every preload + tracks every load state
in one place so the answer never diverges across surfaces:
  * ``hermes mm doctor``          — CLI self-check.
  * ``argus setup multimodal``   — onboarding wizard.
  * the gateway ``mm.readiness`` RPC — web/desktop persistent banner.

Two probe modes:
  * ``deep=False`` — pure, cheap (config/import/filesystem checks only). Safe
    for CLI use where TCP + preloads are undesirable.
  * ``deep=True`` (default in the gateway) — the pure probes AND:
      - Auxiliary local-weight presence + one-shot background preload,
      - Bounded TCP CONNECT reachability for every configured LLM endpoint
        (main / monitor / watcher / memory / embedding / auxiliary.text
        remote / auxiliary.vision),
      - `should_use_local_aux_text()` verdict that VoiceAgent consults.

The report shape is stable (it's a cross-surface contract):

    {
      "ready": bool,                     # all REQUIRED capabilities are ok
      "capabilities": [
        {
          "key": "voice",               # stable id
          "label": "语音 (ASR/TTS)",     # human label
          "status": "ok"|"missing"|"broken"|"unknown",
          "required": bool,             # False → optional; won't block `ready`
          "reason": str,                # empty when ok
          "fix": str,                   # empty when ok
          # optional extras (endpoints[], weights_path, error, url, ...)
        }, ...
      ],
    }

Status semantics:
  ok      — configured/installed and expected to work.
  missing — a prerequisite (key/weight) is absent; capability is off.
  broken  — a prerequisite is present but wrong (e.g. weights loaded but the
            model init crashed, or TCP probe failed).
  unknown — can't determine (e.g. OS capture perms, or a preload still running).
"""

from __future__ import annotations

import importlib.util
import logging
import os
import socket
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

log = logging.getLogger("hermes.multimodal.readiness")

# Status constants (also the contract for the three consumers).
OK = "ok"
MISSING = "missing"
BROKEN = "broken"
UNKNOWN = "unknown"

# Bounded TCP CONNECT timeout for endpoint probes.
_TCP_DEFAULT_TIMEOUT = 3.0


def _has(*values: Optional[str]) -> bool:
    """True if any of the given values is a non-empty string once stripped."""
    for v in values:
        if isinstance(v, str) and v.strip():
            return True
    return False


def _module_installed(name: str) -> bool:
    """True if an importable module exists WITHOUT importing it (no heavy load,
    no side effects like torchvision registering ops)."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        # find_spec can raise if a parent package is itself broken.
        return False


def _installed_version(name: str) -> Optional[str]:
    """Return an installed distribution's version string, or None. Uses
    importlib.metadata so we don't import the package itself."""
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            return version(name)
        except PackageNotFoundError:
            return None
    except Exception:
        return None


def _cap(key: str, label: str, status: str, *, required: bool,
         reason: str = "", fix: str = "") -> Dict[str, Any]:
    return {
        "key": key,
        "label": label,
        "status": status,
        "required": bool(required),
        "reason": reason,
        "fix": fix,
    }


def _get(cfg: Any, name: str, default: Any = None) -> Any:
    """Read a field from either a Config dataclass or a plain dict."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


# --------------------------------------------------------------------------- #
# Per-capability probes. Each returns one capability dict.
# --------------------------------------------------------------------------- #

def _probe_voice(cfg: Any) -> Dict[str, Any]:
    """Realtime ASR/TTS via DashScope. Runtime uses cfg.dashscope_api_key
    directly (watcher_engine.py); we also accept the DASHSCOPE_API_KEY env as a
    hint since users commonly set it there."""
    key = _get(cfg, "dashscope_api_key", "")
    env_key = os.environ.get("DASHSCOPE_API_KEY", "")
    asr_on = bool(_get(cfg, "realtime_asr_enabled", True))
    tts_on = bool(_get(cfg, "realtime_tts_enabled", True))

    if not (asr_on or tts_on):
        return _cap(
            "voice", "语音 (ASR/TTS)", MISSING, required=False,
            reason="realtime_asr_enabled / realtime_tts_enabled 均为 false",
            fix="在 config.yaml 打开 realtime_asr_enabled / realtime_tts_enabled")
    if _has(key, env_key):
        return _cap("voice", "语音 (ASR/TTS)", OK, required=False)
    return _cap(
        "voice", "语音 (ASR/TTS)", MISSING, required=False,
        reason="缺 DashScope 百炼 API key (dashscope_api_key)",
        fix="设 env DASHSCOPE_API_KEY=... 或填 config.yaml 的 dashscope_api_key")


def _probe_deep_research(cfg: Any) -> Dict[str, Any]:
    """Deep-research web search via AnySearch. Runtime prefers env
    ANYSEARCH_API_KEY over cfg.anysearch_api_key (_workers.py:465)."""
    key = _get(cfg, "anysearch_api_key", "")
    env_key = os.environ.get("ANYSEARCH_API_KEY", "")
    if _has(env_key, key):
        return _cap("deep_research", "深研搜索 (AnySearch)", OK, required=False)
    return _cap(
        "deep_research", "深研搜索 (AnySearch)", MISSING, required=False,
        reason="缺 AnySearch API key",
        fix="设 env ANYSEARCH_API_KEY=as_sk_... 或填 config.yaml 的 anysearch_api_key")


def _probe_memory(cfg: Any) -> Dict[str, Any]:
    """Memory backend needs a local OCR path. RapidOCR (rapidocr + onnxruntime)
    is the only OCR backend — remote/cloud VLM OCR was removed — and both
    packages are now core dependencies. A missing package means a broken
    install: MemoryBackend refuses to start (silently, in the logs only)."""
    if _module_installed("rapidocr"):
        return _cap("memory", "记忆 (本地 OCR)", OK, required=True)
    return _cap(
        "memory", "记忆 (本地 OCR)", BROKEN, required=True,
        reason="rapidocr 未安装 → 记忆后端拒绝启动",
        fix='uv pip install -e ".[web]"  (rapidocr/onnxruntime 是必装依赖)')


def _voice_intent_local_path(raw_cfg: Any) -> str:
    """Resolve the configured local voice-intent weights path the SAME way the
    runtime does: ``config.auxiliary.voice_intent.local_path`` (a nested dict —
    NOT a flat Config field), else empty. This is the real source; the flat
    Config dataclass does not expose it."""
    if isinstance(raw_cfg, dict):
        aux = raw_cfg.get("auxiliary")
        if isinstance(aux, dict):
            vi = aux.get("voice_intent")
            if isinstance(vi, dict):
                p = vi.get("local_path")
                if isinstance(p, str) and p.strip():
                    return p.strip()
    return ""


def _probe_local_models(cfg: Any, raw_cfg: Any = None) -> Dict[str, Any]:
    """Local voice-intent model weights (BitCPM4-0.5B). Optional: absent →
    intent classification falls back to cloud/heuristic.

    The configured path lives at ``auxiliary.voice_intent.local_path`` in the
    nested config (read from ``raw_cfg``), matching voice_intent_local.py. When
    unset the runtime falls back to ``$HERMES_HOME/models/bitcpm4-0.5b`` and then
    to the project ``weights/bitcpm4-0.5b`` — we check the same candidates."""
    configured = _voice_intent_local_path(raw_cfg)
    home = os.environ.get("ARGUS_HOME", "")

    candidates: List[str] = []
    if configured:
        candidates.append(configured)
        if home and not os.path.isabs(configured):
            candidates.append(os.path.join(home, configured))
    # Runtime fallbacks (voice_intent_local.default_local_model_dir + project weights).
    if home:
        candidates.append(os.path.join(home, "models", "bitcpm4-0.5b"))
        candidates.append(os.path.join(home, "weights", "bitcpm4-0.5b"))
    candidates.append("weights/bitcpm4-0.5b")

    for c in candidates:
        try:
            if c and os.path.isdir(c) and os.listdir(c):
                return _cap("local_models", "本地权重 (BitCPM4)", OK, required=False)
        except OSError:
            continue
    shown = configured or "weights/bitcpm4-0.5b (默认)"
    return _cap(
        "local_models", "本地权重 (BitCPM4)", MISSING, required=False,
        reason=f"未找到本地权重目录 ({shown})",
        fix="python download_weights.py")


def _probe_vision_deps() -> Dict[str, Any]:
    """torch/torchvision must be version-matched or torchvision fails to
    register ops (torchvision::nms) and ALL YOLO/DINOv3 tracking silently dies.
    This is the documented crash trap — surface it explicitly."""
    if not _module_installed("torch"):
        return _cap(
            "vision_deps", "追踪 (torch/torchvision)", MISSING, required=False,
            reason="未安装 torch (实体追踪不可用)",
            fix="pip install torch==2.5.1 torchvision==0.20.1")
    if not _module_installed("torchvision"):
        return _cap(
            "vision_deps", "追踪 (torch/torchvision)", MISSING, required=False,
            reason="未安装 torchvision",
            fix="pip install torchvision==0.20.1  (须匹配 torch 版本)")
    tv = _installed_version("torchvision")
    tc = _installed_version("torch")
    # Known-good pairing for this project: torch 2.5.x ↔ torchvision 0.20.x.
    if tc and tv:
        tc_mm = ".".join(tc.split(".")[:2])
        tv_mm = ".".join(tv.split(".")[:2])
        if tc_mm == "2.5" and tv_mm != "0.20":
            return _cap(
                "vision_deps", "追踪 (torch/torchvision)", BROKEN, required=False,
                reason=f"torch {tc} 需要 torchvision 0.20.x,实际 {tv} → nms 不注册,追踪全崩",
                fix="pip install torchvision==0.20.1")
    return _cap("vision_deps", "追踪 (torch/torchvision)", OK, required=False)


def _probe_capture_perms() -> Dict[str, Any]:
    """OS-level screen/mic/camera permissions can't be reliably introspected
    here (and can't be granted programmatically). Report unknown with a
    per-OS pointer so onboarding can guide the user."""
    import sys
    if sys.platform == "darwin":
        fix = ("系统设置 → 隐私与安全性 → 屏幕录制 / 麦克风 / 摄像头,"
               "为终端或本 App 授权")
    elif sys.platform.startswith("win"):
        fix = "设置 → 隐私和安全性 → 麦克风 / 相机,允许桌面应用访问"
    else:
        fix = "确保系统已授予屏幕采集 / 麦克风 / 摄像头权限"
    return _cap(
        "capture_perms", "采集权限 (麦/摄/屏)", UNKNOWN, required=False,
        reason="OS 采集权限无法自动检测,请手动确认已授权", fix=fix)


def probe_mm_readiness(cfg: Any = None, raw_cfg: Any = None,
                       deep: bool = False) -> Dict[str, Any]:
    """Build the full multimodal readiness report.

    ``cfg`` may be a Config dataclass, a plain dict of the same flat fields, or
    None (falls back to field defaults). ``raw_cfg`` is the ORIGINAL nested
    argus config dict (before flattening) — needed for values that live only in
    the nested layout (e.g. ``auxiliary.text.*``, ``auxiliary.vision.base_url``).

    ``deep=True`` runs the extended probes: bounded TCP endpoint reachability,
    auxiliary local-weight presence, one-shot background preload of the local
    BitCPM4 model. ``deep=False`` (the default) is pure — safe for CLI /
    unit-test contexts where we don't want network I/O or heavy imports.
    """
    caps: List[Dict[str, Any]] = [
        _probe_voice(cfg),
        _probe_deep_research(cfg),
        _probe_memory(cfg),
        _probe_local_models(cfg, raw_cfg),
        _probe_vision_deps(),
        _probe_capture_perms(),
    ]
    if deep:
        caps.extend(_probe_deep_caps(cfg, raw_cfg))
    # ready = every REQUIRED capability is ok. Optional caps never block.
    ready = all(c["status"] == OK for c in caps if c["required"])
    return {"ready": ready, "capabilities": caps}


# =============================================================================
# Deep probes: network + preload (run only when deep=True). Everything below
# is what used to live in preflight.py — now folded into the one readiness
# module so there's a single mental model: "MM readiness = this file".
# =============================================================================

def _nested(raw_cfg: Any, *path: str) -> Any:
    node: Any = raw_cfg
    for p in path:
        if not isinstance(node, dict):
            return None
        node = node.get(p)
    return node


def _nested_str(raw_cfg: Any, *path: str) -> str:
    v = _nested(raw_cfg, *path)
    return v.strip() if isinstance(v, str) else ""


def _nested_bool(raw_cfg: Any, *path: str, default: bool = False) -> bool:
    v = _nested(raw_cfg, *path)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "1", "on")
    return default


def _cap(key: str, label: str, status: str, *, required: bool,
         reason: str = "", fix: str = "", **extra: Any) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "key": key, "label": label, "status": status,
        "required": required, "reason": reason, "fix": fix,
    }
    entry.update(extra)
    return entry


def _module_installed(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def _tcp_reachable(url: str, timeout: float) -> Tuple[bool, str]:
    """Bounded TCP CONNECT. (ok, reason). Never raises."""
    try:
        parsed = urlparse(url)
    except Exception as exc:
        return False, f"unparseable URL: {exc}"
    host = (parsed.hostname or "").strip()
    if not host:
        return False, "URL has no host"
    scheme = (parsed.scheme or "").lower()
    port = parsed.port or (443 if scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        return False, f"DNS resolution failed: {exc}"
    except Exception as exc:
        return False, f"address lookup error: {exc}"
    last_err = ""
    for family, socktype, proto, _cn, sa in infos:
        s: Optional[socket.socket] = None
        try:
            s = socket.socket(family, socktype, proto)
            s.settimeout(timeout)
            s.connect(sa)
            return True, ""
        except socket.timeout:
            last_err = f"connect timed out after {timeout:.1f}s"
        except ConnectionRefusedError:
            last_err = "connection refused (port closed)"
        except OSError as exc:
            last_err = str(exc) or type(exc).__name__
        finally:
            if s is not None:
                try:
                    s.close()
                except Exception:
                    pass
    return False, last_err or "no address reachable"


# ── Aux.text local-weight resolution (shared with voice_intent_local) ─────

def _hermes_home() -> str:
    hh = os.environ.get("ARGUS_HOME")
    if hh:
        return hh
    try:
        from hermes_constants import get_hermes_home
        return str(get_hermes_home())
    except Exception:
        return os.path.expanduser("~/.argus")


def _relative_to_home(path: str) -> str:
    if not path or os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(_hermes_home(), path))


def resolve_aux_text_weights_path(raw_cfg: Any) -> str:
    """Return absolute weights path iff config.json exists there, else ''.
    Pure disk probe — no transformers/torch import, no HF hub network."""
    configured = _nested_str(raw_cfg, "auxiliary", "text",
                              "local_backend", "local_path")
    candidates: List[str] = []
    if configured:
        candidates.append(_relative_to_home(configured))
    home = _hermes_home()
    candidates.append(os.path.join(home, "models", "bitcpm4-0.5b"))
    candidates.append(os.path.join(home, "weights", "bitcpm4-0.5b"))
    candidates.append(os.path.abspath("weights/bitcpm4-0.5b"))
    for c in candidates:
        try:
            if c and os.path.isdir(c) and os.path.isfile(os.path.join(c, "config.json")):
                return c
        except OSError:
            continue
    return ""


# ── Preload state (module-global, one-shot per process) ───────────────────

_LOAD_LOCK = threading.Lock()
_LOAD_STATE: Dict[str, Dict[str, str]] = {
    "aux_text": {"state": "not_started", "error": ""},
}
_LOAD_LISTENERS: Dict[str, List[Callable[[bool, str], None]]] = {"aux_text": []}


def _set_load_state(kind: str, state: str, error: str = "") -> None:
    with _LOAD_LOCK:
        _LOAD_STATE.setdefault(kind, {})
        _LOAD_STATE[kind]["state"] = state
        _LOAD_STATE[kind]["error"] = error
        listeners = list(_LOAD_LISTENERS.get(kind, []))
        _LOAD_LISTENERS[kind] = []  # one-shot per registration
    if state in ("ready", "failed"):
        for cb in listeners:
            try:
                cb(state == "ready", error)
            except Exception as exc:
                log.debug("[readiness] load listener err (%s): %s", kind, exc)


def observe_aux_text_load(cb: Callable[[bool, str], None]) -> None:
    """Register a callback that fires ONCE when aux_text load resolves.
    If already ready/failed, fires immediately with the last outcome."""
    with _LOAD_LOCK:
        state = _LOAD_STATE.get("aux_text", {}).get("state", "not_started")
        error = _LOAD_STATE.get("aux_text", {}).get("error", "")
        if state == "ready":
            _fire, args = True, (True, "")
        elif state == "failed":
            _fire, args = True, (False, error)
        else:
            _fire, args = False, (False, "")
            _LOAD_LISTENERS.setdefault("aux_text", []).append(cb)
    if _fire:
        try:
            cb(*args)
        except Exception as exc:
            log.debug("[readiness] observe cb err: %s", exc)


def _kick_aux_text_preload() -> None:
    with _LOAD_LOCK:
        state = _LOAD_STATE.get("aux_text", {}).get("state", "not_started")
        if state in ("loading", "ready"):
            return
        _LOAD_STATE["aux_text"]["state"] = "loading"
        _LOAD_STATE["aux_text"]["error"] = ""

    def _on_done(ok: bool, err: str) -> None:
        _set_load_state("aux_text", "ready" if ok else "failed", err if not ok else "")

    try:
        from agent.multimodal.voice_intent_local import ensure_ready_async
        ensure_ready_async(on_done=_on_done)
    except Exception as exc:
        _set_load_state("aux_text", "failed", f"import failed: {exc}")


def get_aux_text_state() -> Dict[str, str]:
    """Peek at current aux_text load state without triggering a load."""
    with _LOAD_LOCK:
        s = _LOAD_STATE.get("aux_text", {})
        return {"state": s.get("state", "not_started"),
                "error": s.get("error", "")}


def should_use_local_aux_text(raw_cfg: Any) -> bool:
    """Definitive answer for VoiceAgent (and any future caller):
    'should this session route through the local BitCPM path?'
    Consolidates use_local=true AND weights present AND load didn't fail.
    """
    if not _nested_bool(raw_cfg, "auxiliary", "text", "use_local"):
        return False
    if not resolve_aux_text_weights_path(raw_cfg):
        return False
    return get_aux_text_state()["state"] != "failed"


# ── Deep capability builders ──────────────────────────────────────────────

def _probe_aux_text(cfg: Any, raw_cfg: Any) -> List[Dict[str, Any]]:
    caps: List[Dict[str, Any]] = []
    use_local = _nested_bool(raw_cfg, "auxiliary", "text", "use_local")
    weights_path = resolve_aux_text_weights_path(raw_cfg)
    weights_present = bool(weights_path)

    if use_local:
        if weights_present:
            _kick_aux_text_preload()
            with _LOAD_LOCK:
                st = _LOAD_STATE.get("aux_text", {}).get("state", "not_started")
                err = _LOAD_STATE.get("aux_text", {}).get("error", "")
            if st == "ready":
                caps.append(_cap(
                    "aux_text_local", "本地意图模型 (BitCPM4)", OK,
                    required=True, weights_path=weights_path))
            elif st == "failed":
                caps.append(_cap(
                    "aux_text_local", "本地意图模型 (BitCPM4)", BROKEN,
                    required=True,
                    reason=f"加载失败: {err[:200]}",
                    fix="检查 transformers/torch 依赖 (uv pip install "
                        "'transformers==4.46.3' 'torch==2.5.1' 'accelerate==1.2.1'),"
                        "或改 config.yaml auxiliary.text.use_local=false 切换到远端。",
                    weights_path=weights_path, error=err))
            else:
                caps.append(_cap(
                    "aux_text_local", "本地意图模型 (BitCPM4)", UNKNOWN,
                    required=False,
                    reason=f"加载中 (状态: {st})",
                    fix="等待加载完成; 若卡在 loading > 30s 请查后台日志。",
                    weights_path=weights_path))
        else:
            caps.append(_cap(
                "aux_text_local", "本地意图模型 (BitCPM4)", MISSING,
                required=True,
                reason="config auxiliary.text.use_local=true 但本地权重缺失。"
                       "已自动降级到远端,请检查远端可达性 (下方)。",
                fix="运行 `python download_weights.py` 下载 BitCPM4-0.5B 权重,"
                    "或改 config.yaml auxiliary.text.use_local=false。"))
            caps.append(_probe_aux_text_remote(raw_cfg, downgraded=True))
    else:
        caps.append(_probe_aux_text_remote(raw_cfg, downgraded=False))
    return caps


def _probe_aux_text_remote(raw_cfg: Any, *, downgraded: bool) -> Dict[str, Any]:
    base_url = _nested_str(raw_cfg, "auxiliary", "text",
                            "remote_backend", "base_url")
    if not base_url:
        return _cap(
            "aux_text_endpoint", "文本/意图 远端端点", MISSING,
            required=True,
            reason=("本地权重降级后需要远端兜底,但 auxiliary.text.remote_backend.base_url 为空"
                    if downgraded else
                    "auxiliary.text.use_local=false 但 remote_backend.base_url 为空"),
            fix="在 config.yaml 填入 auxiliary.text.remote_backend.base_url/api_key/model,"
                "或改 use_local=true 并安装本地权重。")
    ok, reason = _tcp_reachable(base_url, _TCP_DEFAULT_TIMEOUT)
    if ok:
        return _cap(
            "aux_text_endpoint", "文本/意图 远端端点", OK,
            required=True, url=base_url)
    return _cap(
        "aux_text_endpoint", "文本/意图 远端端点", BROKEN,
        required=True,
        reason=f"{base_url} 不可达: {reason}",
        fix="检查 config.yaml auxiliary.text.remote_backend.base_url 或本地服务是否启动。",
        url=base_url, tcp_error=reason)


def _probe_aux_ocr(cfg: Any, raw_cfg: Any) -> List[Dict[str, Any]]:
    """Local rapidocr availability. Remote/cloud VLM OCR was removed — there is
    no endpoint path anymore; rapidocr+onnxruntime are core dependencies."""
    del raw_cfg
    caps: List[Dict[str, Any]] = []
    if _module_installed("rapidocr") or _module_installed("rapidocr_onnxruntime"):
        caps.append(_cap(
            "aux_ocr_local", "本地 OCR (rapidocr)", OK, required=True))
    else:
        caps.append(_cap(
            "aux_ocr_local", "本地 OCR (rapidocr)", MISSING,
            required=True,
            reason="rapidocr 未安装。MemoryBackend 会跳过 OCR 但仍启动。",
            fix="uv pip install rapidocr onnxruntime (rapidocr/onnxruntime 是必装依赖)"))
    return caps


# Roles whose base_url actually lives on the flat Config dataclass (populated
# by flatten_mm_config from ``model.<role>.base_url``). NOTE: "main" is NOT
# here — the main-agent endpoint does NOT flow into cfg.base_url (that field
# is a dead default; the real main client is resolved via auxiliary_client).
# We read the main endpoint straight from raw yaml (``model.base_url``) below.
_MAIN_LLM_ROLES: List[Tuple[str, str]] = [
    ("monitor", "monitor_base_url"),
    ("watcher", "watcher_base_url"),
    ("memory", "memory_base_url"),
    ("embedding", "embedding_base_url"),
    ("mm_embedding", "mm_embedding_base_url"),
]

# Human-visible yaml path for each role, used in fix strings so the user knows
# EXACTLY which key to open. Kept in sync with hermes_glue's flatten map.
_ROLE_YAML_PATH: Dict[str, str] = {
    "main": "model.base_url",
    "monitor": "model.monitor.base_url",
    "watcher": "model.watcher.base_url",
    "memory": "model.memory.base_url",
    "embedding": "model.embedding.base_url",
    "mm_embedding": "model.mm_embedding.base_url",
    "auxiliary.vision": "auxiliary.vision.base_url",
    "auxiliary.text": "auxiliary.text.remote_backend.base_url",
}


def _probe_llm_endpoints(cfg: Any, raw_cfg: Any) -> Dict[str, Any]:
    entries: List[Dict[str, str]] = []

    def _get(key: str) -> str:
        if cfg is None:
            return ""
        try:
            v = getattr(cfg, key, "") if not isinstance(cfg, dict) else cfg.get(key, "")
        except Exception:
            v = ""
        return str(v or "").strip()

    # Main agent endpoint reads DIRECTLY from raw yaml (model.base_url) — its
    # value never lands on cfg.base_url (see comment on _MAIN_LLM_ROLES). Skip
    # when unset: some deployments run entirely on nested per-role endpoints.
    main_url = _nested_str(raw_cfg, "model", "base_url")
    if main_url:
        entries.append({"role": "main", "url": main_url})
    for role, key in _MAIN_LLM_ROLES:
        url = _get(key)
        if url:
            entries.append({"role": role, "url": url})
    vision_url = _nested_str(raw_cfg, "auxiliary", "vision", "base_url")
    if vision_url:
        entries.append({"role": "auxiliary.vision", "url": vision_url})

    if not entries:
        return _cap(
            "llm_endpoints", "LLM 端点连通性", UNKNOWN,
            required=True,
            reason="no LLM endpoints configured to probe",
            endpoints=[])

    seen: Dict[str, Tuple[bool, str]] = {}
    details: List[Dict[str, Any]] = []
    for e in entries:
        url = e["url"]
        if url not in seen:
            seen[url] = _tcp_reachable(url, _TCP_DEFAULT_TIMEOUT)
        ok, reason = seen[url]
        details.append({"role": e["role"], "url": url, "ok": ok, "reason": reason})

    bad = [d for d in details if not d["ok"]]
    if not bad:
        return _cap(
            "llm_endpoints", "LLM 端点连通性", OK,
            required=True, endpoints=details)
    summary = ", ".join(f"{d['role']} ({d['reason']})" for d in bad[:4])
    if len(bad) > 4:
        summary += f", ... +{len(bad) - 4} more"
    # Fix hint: list the ACTUAL yaml paths of the failed roles so the user can
    # ctrl-F directly to the key (no more "<role>" placeholder confusion).
    bad_paths_unique: List[str] = []
    for d in bad:
        yaml_path = _ROLE_YAML_PATH.get(str(d["role"]), str(d["role"]))
        if yaml_path not in bad_paths_unique:
            bad_paths_unique.append(yaml_path)
    fix_paths = " / ".join(bad_paths_unique[:6])
    return _cap(
        "llm_endpoints", "LLM 端点连通性", BROKEN,
        required=True,
        reason=f"{len(bad)}/{len(details)} 端点不可达: {summary}",
        fix=f"检查 config.yaml 里以下 key 的 URL 是否正确、对应服务是否启动: {fix_paths}",
        endpoints=details)


def _probe_deep_caps(cfg: Any, raw_cfg: Any) -> List[Dict[str, Any]]:
    """Assemble the deep-mode capability list. Failures never raise —
    a probe crash becomes an unknown capability rather than an RPC error."""
    caps: List[Dict[str, Any]] = []
    try:
        caps.extend(_probe_aux_text(cfg, raw_cfg))
    except Exception as exc:
        log.warning("[readiness] aux_text probe failed: %s", exc, exc_info=True)
        caps.append(_cap(
            "aux_text_local", "本地意图模型 (BitCPM4)", UNKNOWN,
            required=False, reason=f"probe error: {exc}"))
    try:
        caps.extend(_probe_aux_ocr(cfg, raw_cfg))
    except Exception as exc:
        log.warning("[readiness] aux_ocr probe failed: %s", exc, exc_info=True)
    try:
        caps.append(_probe_llm_endpoints(cfg, raw_cfg))
    except Exception as exc:
        log.warning("[readiness] endpoints probe failed: %s", exc, exc_info=True)
    return caps
