"""Alibaba Cloud DashScope provider profile.

DashScope's OpenAI-compatible endpoint gates reasoning on the non-standard
``enable_thinking`` parameter, passed through ``extra_body`` (Python SDK). Hybrid
thinking models (Qwen3 / Qwen3.5 / Qwen3.6 / Qwen3.7 — Max/Plus/Flash/Turbo and
open-source variants) default to thinking OFF for -max and ON for others, so the
flag must be set explicitly to honor Hermes' unified reasoning toggle. Without
this override, ``reasoning_config`` never reached the wire and the thinking
switch silently did nothing (defect D6).

Wire shape (per Alibaba Model Studio docs):

    {"extra_body": {"enable_thinking": true | false,
                    "thinking_budget": <int, optional>}}

``enable_thinking`` is top-level in ``extra_body`` — NOT nested under
``chat_template_kwargs`` (that's the vLLM/self-hosted convention). Thinking-only
models (QwQ, ``*-thinking-*``) ignore the flag, so we omit it for them; models
that aren't recognizably hybrid-thinking are left untouched to avoid perturbing
older qwen-vl / qwen-max-2024 wire formats.
"""

from __future__ import annotations

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


def _model_supports_thinking_toggle(model: str | None) -> bool:
    """True for DashScope Qwen models whose reasoning can be toggled per request.

    Covers the Qwen3+ hybrid families (qwen3-*, qwen3.5-*, qwen3.6-*, qwen3.7-*,
    and the bare qwen-plus/qwen-flash/qwen-turbo/qwen-max aliases which currently
    route to a Qwen3 hybrid backend). Thinking-only models (``*-thinking*``,
    ``qwq*``) are excluded — the flag is a no-op / rejected there. Older
    non-thinking models (qwen-vl-*, qwen-max-2024-*, qwen2*) are excluded so we
    don't perturb their wire format.
    """
    m = (model or "").strip().lower().rsplit("/", 1)[-1]
    if not m:
        return False
    # Thinking-only → no toggle (always reasons).
    if "thinking" in m or m.startswith("qwq"):
        return False
    # Explicit Qwen3+ generations.
    if m.startswith(("qwen3", "qwen-3")):
        return True
    # Bare current aliases that route to Qwen3 hybrid backends.
    if m in ("qwen-plus", "qwen-flash", "qwen-turbo", "qwen-max",
             "qwen-plus-latest", "qwen-flash-latest",
             "qwen-turbo-latest", "qwen-max-latest"):
        return True
    return False


def _effort_budget(effort: str | None) -> int | None:
    """Map a Hermes reasoning tier onto a ``thinking_budget`` token cap.

    DashScope takes a token count rather than a named level, so the tier has to
    be translated. The numbers are deliberately the same ladder
    ``agent.anthropic_adapter.THINKING_BUDGET`` uses for Anthropic's
    ``budget_tokens``, so "High" means a comparable amount of reasoning
    whichever vendor is behind the session.

    ``None`` (unknown/absent tier) means "send no cap", which is DashScope's own
    default (the model's maximum chain-of-thought length) — the right behaviour
    when the user has expressed no preference.

    ``minimal`` gets a deliberately small floor rather than 0: 0 would read as
    "no reasoning at all", and the off state is already expressed by
    ``enable_thinking: False``.
    """
    tier = (effort or "").strip().lower()

    return {
        "minimal": 1024,
        "low": 4000,
        "medium": 8000,
        "high": 16000,
        "xhigh": 32000,
        "max": 32000,
    }.get(tier)


class AlibabaProfile(ProviderProfile):
    """Alibaba DashScope — extra_body.enable_thinking (+ thinking_budget depth)."""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None,
        model: str | None = None, **context,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}

        if not _model_supports_thinking_toggle(model):
            # Thinking-only / non-thinking / unknown → leave wire untouched.
            return extra_body, top_level

        # Default enabled (matches most hybrid Qwen3 non-max defaults); an
        # explicit reasoning_config.enabled=False disables it.
        enabled = True
        if isinstance(reasoning_config, dict) and reasoning_config.get("enabled") is False:
            enabled = False

        extra_body["enable_thinking"] = enabled

        if enabled and isinstance(reasoning_config, dict):
            # Depth. DashScope caps reasoning with `thinking_budget` — a TOKEN
            # COUNT, not a named tier ("Use thinking_budget to cap reasoning
            # tokens... When the limit is reached, the model stops reasoning and
            # responds immediately"). Supported by "Qwen3 (in thinking mode) and
            # Kimi models", and its default is the model's own maximum, so
            # omitting it means "no cap".
            #
            # ★ An explicit numeric budget wins; otherwise DERIVE one from the
            #   effort tier. Only the explicit branch existed, and nothing in
            #   Hermes ever puts `thinking_budget`/`budget_tokens` into
            #   reasoning_config (parse_reasoning_effort emits `enabled` +
            #   `effort` only) — so it was unreachable, and every "on" tier from
            #   Min to Max produced a byte-identical request. The dial was
            #   decoration for DashScope.
            budget = reasoning_config.get("thinking_budget")
            if budget is None:
                budget = reasoning_config.get("budget_tokens")
            if budget is None:
                budget = _effort_budget(reasoning_config.get("effort"))
            try:
                if budget is not None:
                    extra_body["thinking_budget"] = int(budget)
            except (TypeError, ValueError):
                pass

        return extra_body, top_level


alibaba = AlibabaProfile(
    name="alibaba",
    aliases=("dashscope", "alibaba-cloud", "qwen-dashscope"),
    env_vars=("DASHSCOPE_API_KEY",),
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)

register_provider(alibaba)
