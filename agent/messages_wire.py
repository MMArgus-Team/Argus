"""Wire-format classification for custom ``/v1/messages`` gateways.

The URL path identifies the transport endpoint, not one uniform JSON schema
for every payload dimension.  In particular, the internal Luna gateway uses a
``/v1/messages`` leaf, accepts OpenAI-style ``image_url`` parts, and requires
Messages-style ``tool_use``/``tool_result`` history plus Anthropic top-level
tool declarations.  Keep this decision shared by the main-agent and
multimodal clients so the two call paths cannot silently diverge.
"""

from __future__ import annotations

import copy
import json
from urllib.parse import urlparse
from typing import Any


def uses_anthropic_messages_wire(
    base_url: str,
    *,
    api_mode: str | None = None,
) -> bool:
    """Return whether a messages endpoint explicitly speaks Anthropic JSON.

    ``/v1/messages`` alone is deliberately not treated as proof: hybrid and
    internal gateways commonly reuse that path with OpenAI-compatible content
    parts.  Explicit API mode wins; otherwise only well-known Anthropic hosts
    or an ``/anthropic`` route select Anthropic content blocks.
    """
    mode = str(api_mode or "").strip().lower()
    if mode:
        return mode == "anthropic_messages"

    normalized = str(base_url or "").strip().lower().rstrip("/")
    if not normalized:
        return False
    parsed = urlparse(normalized)
    host = (parsed.hostname or "").lower()
    path = parsed.path.rstrip("/")
    if host == "api.anthropic.com":
        return True
    if host == "api.kimi.com" and "/coding" in path:
        return True
    return any(
        path == prefix or path.startswith(prefix + "/")
        for prefix in ("/anthropic", "/v1/anthropic")
    ) or "/anthropic/" in path


def is_local_hybrid_messages_endpoint(base_url: str) -> bool:
    """The local development proxy keeps ordinary Chat Completions messages."""
    parsed = urlparse(str(base_url or "").strip().lower())
    return (parsed.hostname or "") in {"127.0.0.1", "localhost"} and (
        parsed.port == 8080)


def uses_anthropic_tools_wire(
    base_url: str,
    *,
    api_mode: str | None = None,
) -> bool:
    """Return whether top-level tools use ``name``/``input_schema``.

    Tool declarations, image content, and historical tool exchanges are
    separate protocol dimensions on the internal Luna gateway.  The direct
    ``/v1/messages`` transport expects Anthropic tool declarations; its image
    blocks remain OpenAI-compatible and historical tool pairs are normalized by
    :func:`hybrid_messages_from_chat`.  The localhost development proxy is the
    one known exception and keeps OpenAI tool wrappers end to end.
    """
    if is_local_hybrid_messages_endpoint(base_url):
        return False
    if uses_anthropic_messages_wire(base_url, api_mode=api_mode):
        return True

    parsed = urlparse(str(base_url or "").strip().lower().rstrip("/"))
    return parsed.path.rstrip("/").endswith("/v1/messages")


def tools_wire_label(base_url: str, *, api_mode: str | None = None) -> str:
    """Stable diagnostic label for the top-level tool declaration schema."""
    return (
        "anthropic"
        if uses_anthropic_tools_wire(base_url, api_mode=api_mode)
        else "openai"
    )


def openai_messages_from_chat(
    messages: Any,
    *,
    lift_system: bool = True,
) -> tuple[str | None, list]:
    """Preserve OpenAI content/tool shapes while optionally lifting system."""
    if not lift_system:
        return None, copy.deepcopy(list(messages or []))
    system_parts = []
    out = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        if message.get("role") != "system":
            out.append(copy.deepcopy(message))
            continue
        content = message.get("content")
        if isinstance(content, str) and content:
            system_parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text")
                    if isinstance(text, str) and text:
                        system_parts.append(text)
    return "\n\n".join(system_parts).strip() or None, out


def hybrid_messages_from_chat(
    messages: Any,
    *,
    lift_system: bool = True,
) -> tuple[str | None, list]:
    """Build history for the internal hybrid ``/v1/messages`` gateway.

    The gateway accepts OpenAI ``image_url`` content parts, but its historical
    tool exchange follows the Messages protocol: assistant ``tool_use`` blocks
    followed by user ``tool_result`` blocks.  Sending an otherwise valid OpenAI
    ``assistant.tool_calls`` / ``role=tool`` pair is rejected by the gateway as
    an orphan tool result.  Keep the two protocol dimensions independent here:
    ordinary content parts are copied verbatim while only completed tool
    exchanges are converted.
    """
    system_parts = []
    out = []

    def _content_blocks(content: Any) -> list:
        if isinstance(content, list):
            return copy.deepcopy(content)
        if content is None or content == "":
            return []
        return [{"type": "text", "text": str(content)}]

    def _tool_input(tool_call: dict) -> tuple[str, dict]:
        function = (
            tool_call.get("function")
            if isinstance(tool_call.get("function"), dict)
            else {}
        )
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except Exception:
                arguments = {"arguments": arguments}
        if not isinstance(arguments, dict):
            arguments = {}
        return str(function.get("name") or ""), arguments

    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "user")
        content = message.get("content")
        if lift_system and role == "system":
            if isinstance(content, str) and content:
                system_parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text = part.get("text")
                        if isinstance(text, str) and text:
                            system_parts.append(text)
            continue
        if role == "assistant":
            blocks = _content_blocks(content)
            for index, tool_call in enumerate(message.get("tool_calls") or []):
                if not isinstance(tool_call, dict):
                    continue
                name, arguments = _tool_input(tool_call)
                blocks.append({
                    "type": "tool_use",
                    "id": (
                        tool_call.get("id")
                        or f"call_messages_hybrid_{index}"
                    ),
                    "name": name,
                    "input": arguments,
                })
            out.append({
                "role": "assistant",
                "content": blocks or [{"type": "text", "text": ""}],
            })
            continue
        if role == "tool":
            result_content = copy.deepcopy(content)
            if not isinstance(result_content, (str, list)):
                try:
                    result_content = json.dumps(
                        result_content, ensure_ascii=False)
                except Exception:
                    result_content = str(result_content)
            out.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": (
                        message.get("tool_call_id")
                        or message.get("id")
                        or ""
                    ),
                    "content": result_content,
                }],
            })
            continue
        out.append(copy.deepcopy(message))

    return "\n\n".join(system_parts).strip() or None, out
