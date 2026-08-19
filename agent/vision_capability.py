"""Known-model vision-capability classification.

Answers one question for config validation: *does this model accept image
input?* Used by :func:`hermes_cli.config.validate_config_structure` to warn when

  1. a model is declared ``supports_vision: true`` but is a known text-only model
     (likely a config typo), or
  2. a vision-REQUIRED role (main agent, multimodal monitor / deep-research
     worker / memory writer+reviewer) is pointed at a known text-only model.

Resolution order (first definitive hit wins):

  1. Built-in family/pattern table (:data:`_VISION_KNOWN`) matched on the MODEL
     NAME — the most reliable signal, because config endpoints are usually
     ``provider: custom`` (models.dev can't resolve them) and often sit behind
     proxy hostnames that :func:`_infer_provider_from_url` doesn't recognise.
  2. models.dev capability lookup, using the config ``provider`` (when it's a
     real models.dev provider, not ``custom``) or a provider inferred from the
     ``base_url``.
  3. ``None`` — genuinely unknown. Callers must NOT warn on ``None`` (avoids
     false alarms every time the user switches to a model we don't recognise).

This module is pure/offline (the models.dev fallback uses the bundled snapshot +
disk cache) and never raises — any internal error resolves to ``None``.
"""

from __future__ import annotations

import re
from typing import List, Optional, Pattern, Tuple

__all__ = ["classify_vision"]


def _bare(model: str) -> str:
    """Lowercased last path segment (drops aggregator prefixes)."""
    return (model or "").strip().lower().rsplit("/", 1)[-1]


# Ordered (regex, supports_vision) rules matched against the bare model name.
# FIRST match wins, so put the more specific text-only carve-outs BEFORE the
# broad vision families they'd otherwise be caught by (e.g. glm-4-9b before
# glm vision, qwen non-vl before a hypothetical broad qwen rule).
#
# True  = known to accept image input.
# False = known text-only.
# Anything not matched here → fall through to models.dev, then None.
_RULES: List[Tuple[str, bool]] = [
    # ── Known TEXT-ONLY (checked first so they win over broad families) ──
    (r"^deepseek", False),                 # entire DeepSeek line is text-only
    (r"^gpt-oss", False),                  # OpenAI open-weights, text-only
    (r"^qwq", False),                      # Qwen QwQ reasoning, text-only
    (r"^glm-4-9b", False),
    (r"^glm-4-flash", False),
    (r"^glm-4-air(?!.*v)", False),
    (r"(?:^|[-_])embedding", False),
    (r"^text-", False),
    (r"-instruct-text\b", False),

    # ── Qwen (Alibaba Tongyi) ──────────────────────────────────────────────
    # Modern Qwen chat models are NATIVELY multimodal (text + image): the
    # qwen3.5 / qwen3.6 / qwen3.7 generations and the current bare aliases
    # (qwen-plus / -flash / -turbo / -max), plus every explicit -vl slug. Only
    # QwQ (reasoning) and the legacy qwen2 / qwen2.5 non-vl chat models are
    # text-only. Anything else Qwen → not judged here (fall through), so we
    # never wrongly warn about a vision-capable Qwen (the qwen3.5-flash bug).
    (r"^qwen[\w.]*-vl", True),             # any qwen*-vl explicit vision
    (r"^qwen3\.[5-9]", True),              # qwen3.5 / 3.6 / 3.7 …
    (r"^qwen3-max", True),
    (r"^qwen-(?:plus|flash|turbo|max)\b", True),
    (r"^qwen2(?:\.5)?-(?!.*vl)", False),   # legacy qwen2 / 2.5 non-vl → text-only

    # ── GLM (Zhipu) ────────────────────────────────────────────────────────
    # Only the explicit vision slugs (…v…) are known vision. Legacy small chat
    # (glm-4-9b / glm-4-flash) is text-only. The base chat lines (glm-4.5 /
    # 4.6 / 5) are NOT judged here (fall through → silent), since their vision
    # support is version-dependent and we must not false-warn.
    (r"glm-\d+(?:\.\d+)?v", True),         # glm-4v / glm-4.5v / glm-5v …
    (r"^glm-.*v-", True),

    # ── Known VISION-CAPABLE families ──
    (r"-vl(?:$|[-_])", True),              # generic *-vl vision slugs
    (r"^qwen-vl", True),
    (r"^kimi", True),                      # Kimi K2.x accept image input (moonshot platform)
    (r"^moonshot-v1", True),               # moonshot-v1-*-vision & vision-capable v1 chat
    (r"^gemini", True),                    # all Gemini are multimodal
    (r"^gpt-4o", True),
    (r"^gpt-4\.1", True),
    (r"^gpt-4-turbo", True),
    (r"^gpt-5", True),                     # gpt-5.x are multimodal
    (r"^chatgpt-4o", True),
    (r"^o3", True),                        # o3 / o3-mini reasoning accept images
    (r"^o4", True),
    (r"^claude-3", True),                  # Claude 3.x all vision
    (r"^claude-(?:sonnet|opus|haiku)", True),  # claude-sonnet-4/opus-4/... vision
    (r"^claude-.*-4", True),
    (r"^claude-fable", True),
    (r"^llama-3\.2-.*vision", True),
    (r"vision", True),                     # generic *vision* slug (grok-*-vision, etc.)
    (r"^pixtral", True),
    (r"^llava", True),
    (r"^internvl", True),
    (r"^step-1v", True),
    (r"^step-1o", True),
    (r"^grok-4", True),                    # Grok 4 multimodal
    (r"^grok-2-vision", True),
    (r"^mimo-v", True),                    # Xiaomi MiMo-V vision
]

def _compile_rules() -> List[Tuple[Pattern[str], Optional[bool]]]:
    out: List[Tuple[Pattern[str], Optional[bool]]] = []
    for pat, val in _RULES:
        try:
            out.append((re.compile(pat), val))
        except re.error:
            continue
    return out


_PATTERNS: List[Tuple[Pattern[str], Optional[bool]]] = _compile_rules()


def _match_table(model: str) -> Optional[bool]:
    """Return True/False from the built-in table, or None if unmatched/ambiguous.

    A rule value of ``None`` in the table means "explicitly ambiguous — stop
    treating this as a broad-family hit and let a more specific rule or the
    models.dev fallback decide". We implement that by skipping ``None`` rules
    here (they exist only to document intent) and relying on ordering.
    """
    bare = _bare(model)
    if not bare:
        return None
    for pat, val in _PATTERNS:
        if val is None:
            continue
        if pat.search(bare):
            return val
    return None


def _models_dev_lookup(provider: str, model: str, base_url: str) -> Optional[bool]:
    """models.dev fallback. Resolve a real provider (config often says 'custom')
    from the given provider or the base_url, then read capabilities."""
    try:
        from agent.models_dev import get_model_capabilities, PROVIDER_TO_MODELS_DEV
    except Exception:
        return None

    candidates: List[str] = []
    p = (provider or "").strip().lower()
    if p and p not in ("custom", "openai", "auto", "main", ""):
        candidates.append(p)
    # Infer from base_url (handles standard hosts like api.deepseek.com,
    # dashscope.aliyuncs.com; returns None for proxy hostnames).
    try:
        from agent.model_metadata import _infer_provider_from_url
        inferred = _infer_provider_from_url(base_url or "")
        if inferred:
            candidates.append(inferred)
    except Exception:
        pass

    for cand in dict.fromkeys(filter(None, candidates)):
        # Only bother if models.dev knows this provider id.
        try:
            if cand not in PROVIDER_TO_MODELS_DEV and cand not in PROVIDER_TO_MODELS_DEV.values():
                continue
            caps = get_model_capabilities(cand, model)
        except Exception:
            caps = None
        if caps is not None:
            return bool(caps.supports_vision)
    return None


def classify_vision(
    provider: str, model: str, base_url: str = "",
) -> Optional[bool]:
    """Classify whether *model* accepts image input.

    Returns:
      * ``True``  — known vision-capable.
      * ``False`` — known text-only.
      * ``None``  — unknown (caller must NOT warn).

    Never raises.
    """
    try:
        if not (model or "").strip():
            return None
        table = _match_table(model)
        if table is not None:
            return table
        return _models_dev_lookup(provider, model, base_url)
    except Exception:
        return None
