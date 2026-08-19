"""Kimi / Moonshot provider profiles.

Kimi has dual endpoints:
  - sk-kimi-* keys → api.kimi.com/coding (Anthropic Messages API)
  - legacy keys → api.moonshot.ai/v1 (OpenAI chat completions)

This module covers the chat_completions path (/v1 endpoint).
"""

from typing import Any

from providers import register_provider
from providers.base import OMIT_TEMPERATURE, ProviderProfile


def _is_thinking_only(model: str | None) -> bool:
    """True for Moonshot models that REJECT an explicit thinking-off.

    Delegates to the existing helper so the model list lives in one place.
    """
    try:
        from agent.moonshot_schema import is_thinking_only_moonshot_model

        return bool(is_thinking_only_moonshot_model(model))
    except Exception:
        return False


class KimiProfile(ProviderProfile):
    """Kimi/Moonshot — temperature omitted, thinking xor reasoning_effort."""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None,
        model: str | None = None, **context
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Kimi reasoning controls.

        Moonshot's wire shape treats ``extra_body.thinking`` (a binary toggle)
        and a top-level ``reasoning_effort`` as mutually exclusive — sending
        both is at best redundant and risks "cannot specify both 'thinking' and
        'reasoning_effort'" (HTTP 400). This mirrors the kimi-k2 handling on the
        opencode-go relay: send effort when one is requested, otherwise fall
        back to ``extra_body.thinking`` — never both.
        """
        extra_body = {}
        top_level = {}

        if not reasoning_config or not isinstance(reasoning_config, dict):
            # No config → thinking enabled, let the server pick the depth.
            # (Previously also sent reasoning_effort="medium", which paired
            # thinking + effort on every default call.)
            extra_body["thinking"] = {"type": "enabled"}
            return extra_body, top_level

        enabled = reasoning_config.get("enabled", True)
        if enabled is False:
            # ★ Thinking-only models (k2.7-code family) answer HTTP 400 to an
            #   explicit "off" — "invalid thinking: only type=enabled is allowed
            #   for this model". Omit the field entirely; the vendor default is
            #   enabled, which is the closest legal state to the request.
            #   The multimodal gateway already guarded this
            #   (agent.moonshot_schema.is_thinking_only_moonshot_model); the main
            #   chat path did not, so turning thinking off on k2.7-code failed the
            #   whole request instead of merely being impossible.
            if not _is_thinking_only(model):
                extra_body["thinking"] = {"type": "disabled"}
            return extra_body, top_level

        # Enabled: prefer an explicit effort; only fall back to extra_body
        # thinking when no recognized effort is requested.
        effort = (reasoning_config.get("effort") or "").strip().lower()
        if effort in {"low", "medium", "high"}:
            top_level["reasoning_effort"] = effort
        else:
            extra_body["thinking"] = {"type": "enabled"}

        return extra_body, top_level


kimi = KimiProfile(
    name="kimi-coding",
    aliases=("kimi", "moonshot", "kimi-for-coding"),
    env_vars=("KIMI_API_KEY", "KIMI_CODING_API_KEY"),
    base_url="https://api.moonshot.ai/v1",
    fixed_temperature=OMIT_TEMPERATURE,
    default_max_tokens=32000,
    default_headers={"User-Agent": "argus-agent/1.0"},
    default_aux_model="kimi-k2-turbo-preview",
)

kimi_cn = KimiProfile(
    name="kimi-coding-cn",
    aliases=("kimi-cn", "moonshot-cn"),
    env_vars=("KIMI_CN_API_KEY",),
    base_url="https://api.moonshot.cn/v1",
    fixed_temperature=OMIT_TEMPERATURE,
    default_max_tokens=32000,
    default_headers={"User-Agent": "argus-agent/1.0"},
    default_aux_model="kimi-k2-turbo-preview",
)

register_provider(kimi)
register_provider(kimi_cn)
