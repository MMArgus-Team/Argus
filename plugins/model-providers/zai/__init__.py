"""ZAI / GLM provider profile.

GLM's thinking-capable models (GLM-4.5+, GLM-5+) default to thinking-mode ON and
gate it with a ``thinking: {"type": "enabled" | "disabled"}`` parameter — the
same wire shape as DeepSeek. On the OpenAI-compatible endpoint that non-standard
field is passed via ``extra_body``. Without an explicit flag GLM keeps reasoning
on and emits reasoning content, which (combined with Hermes' history replay) can
trip the "reasoning must be echoed back" contract on later turns, and Hermes'
unified reasoning toggle silently did nothing (defect D3).

Wire shape:

    {"extra_body": {"thinking": {"type": "enabled" | "disabled"},
                    "reasoning_effort": "high" | "max"}}   # GLM-5.2 only

Non-thinking GLM models (glm-4-9b, glm-4-flash, glm-4v-*) are left as no-ops so
their wire format is untouched.

Depth (``reasoning_effort``) is separate from the on/off toggle and much more
narrowly supported — see ``_effort_param`` for the exact rules and why the
Hermes tier is remapped rather than passed through.
"""

from __future__ import annotations

import re
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


def _model_supports_thinking(model: str | None) -> bool:
    """GLM thinking-capable model families.

    Zhipu gates the ``thinking`` parameter on the model *generation*: "仅
    GLM-4.5 及以上模型支持此参数配置". So this is a minimum-version test, not a
    hand-maintained list of releases — an allow-list of known versions silently
    excludes each new one (``glm-4.7`` was missing and therefore could not have
    its thinking turned off, even though it is a thinking model).

    Older ``glm-4-9b`` / ``glm-4-flash`` / vision ``glm-4v-*`` predate the
    parameter and are excluded so their wire format stays untouched. Tolerant of
    the dot-vs-dash version separator aggregators sometimes use.
    """
    m = (model or "").strip().lower().rsplit("/", 1)[-1]
    if not m or not m.startswith("glm"):
        return False
    # Pre-4.5 families: no thinking parameter. Checked before the version parse
    # because `glm-4v-plus` / `glm-4-9b` would otherwise read as major 4.
    if m.startswith(("glm-4v", "glm-4-9b", "glm-4-flash", "glm-4-air", "glm-4-plus")):
        return False

    match = re.match(r"^glm-?(\d+)(?:[.\-](\d+))?", m)
    if not match:
        return False

    major = int(match.group(1))
    minor = int(match.group(2) or 0)

    # GLM-4.5 and newer. GLM-5/6/… are all newer by major version.
    return (major, minor) >= (4, 5)


def _model_supports_effort(model: str | None) -> bool:
    """True for GLM models accepting ``reasoning_effort``.

    Zhipu documents this as "仅 GLM-5.2 支持" — far narrower than the ``thinking``
    toggle (GLM-4.5+). It is also absent from the VISION request schema
    (``ChatCompletionVisionRequest`` has ``thinking`` but no ``reasoning_effort``),
    so ``glm-5.2v``-style vision variants are excluded too.

    Deliberately a minimum-version test with an explicit vision exclusion rather
    than a literal ``== "glm-5.2"``: aggregators append suffixes
    (``glm-5.2-preview``, ``glm-5.2-0930``) that an equality check would miss.
    """
    m = (model or "").strip().lower().rsplit("/", 1)[-1]
    if not m.startswith("glm"):
        return False

    match = re.match(r"^glm-?(\d+)(?:[.\-](\d+))?(v)?", m)
    if not match:
        return False
    # Vision variants (glm-5.2v) take `thinking` but not `reasoning_effort`.
    if match.group(3):
        return False

    major = int(match.group(1))
    minor = int(match.group(2) or 0)

    return (major, minor) >= (5, 2)


def _effort_param(effort: str) -> str | None:
    """Map a Hermes reasoning tier onto GLM's ``reasoning_effort`` value.

    Zhipu collapses the incoming scale on its side "为保持和其他协议兼容":
    ``low``/``medium`` → ``high``, ``xhigh`` → ``max``, and ``none``/``minimal``
    make the model skip reasoning. Its own default is ``max``.

    ★ We send the ALREADY-REMAPPED value rather than passing the raw tier
      through. Both reach the same model behaviour, but the remapped form is what
      the request actually means — so logs, replays and any future strict
      validation show the truth instead of implying a distinction (low vs medium)
      that the vendor does not honour.

    ★ ``minimal`` is the awkward one. It is Hermes' LOWEST "thinking on" tier
      (``parse_reasoning_effort`` returns ``enabled: True``), but Zhipu documents
      ``minimal`` as making the model skip reasoning entirely — the two scales
      disagree. We honour Zhipu: pass it through so the shallowest tier really is
      shallowest there. Remapping it up to ``high`` would make Min and High
      identical, which is the "dial does nothing" class of bug this whole change
      exists to remove.

    ``none`` never reaches here — thinking-off is expressed by the caller as
    ``thinking: {"type": "disabled"}``, a documented enum value, rather than as a
    side effect of the depth knob.
    """
    tier = (effort or "").strip().lower()
    if tier == "minimal":
        return "minimal"
    if tier in {"low", "medium", "high"}:
        return "high"
    if tier in {"xhigh", "max"}:
        return "max"
    return None


class ZaiProfile(ProviderProfile):
    """Z.AI / GLM — extra_body.thinking:{type} (+ reasoning_effort on GLM-5.2)."""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None,
        model: str | None = None, **context,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}

        if not _model_supports_thinking(model):
            return extra_body, top_level

        enabled = True
        if isinstance(reasoning_config, dict) and reasoning_config.get("enabled") is False:
            enabled = False

        extra_body["thinking"] = {"type": "enabled" if enabled else "disabled"}

        # Depth, only where the vendor accepts it. Sent alongside the toggle
        # (Zhipu treats them as complementary — "thinking 开启时生效" — unlike
        # Moonshot, which rejects the pair). Skipped when thinking is off, where
        # a depth would be meaningless.
        if enabled and _model_supports_effort(model):
            tier = ""
            if isinstance(reasoning_config, dict):
                tier = str(reasoning_config.get("effort") or "")
            value = _effort_param(tier)
            if value:
                extra_body["reasoning_effort"] = value

        return extra_body, top_level


zai = ZaiProfile(
    name="zai",
    aliases=("glm", "z-ai", "z.ai", "zhipu"),
    env_vars=("GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY"),
    display_name="Z.AI (GLM)",
    description="Z.AI / GLM — Zhipu AI models",
    signup_url="https://z.ai/",
    fallback_models=(
        "glm-5.2",
        "glm-5",
        "glm-4-9b",
    ),
    base_url="https://api.z.ai/api/paas/v4",
    default_aux_model="glm-4.5-flash",
)

register_provider(zai)
