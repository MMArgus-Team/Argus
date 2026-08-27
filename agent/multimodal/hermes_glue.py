"""Glue between Hermes' config / model resolution and the multimodal subsystem.

This is the fusion layer:

  * :func:`build_config` reads Hermes' ``multimodal:`` config section and produces
    a fully-populated :class:`agent.multimodal.core.Config` (numeric hyperparameters
    only — endpoints/models come from Hermes' own resolution).

  * :class:`HermesClientFactory` hands the workers an OpenAI-compatible async client
    resolved through Hermes' provider/model machinery (``agent.auxiliary_client``),
    so the multimodal agent shares the user's ``argus model`` selection, providers,
    credential pool, and fallbacks. Vision-capable resolution is preferred because
    every worker sends video frames.

  * :func:`build_speech_factory` returns the ASR/TTS backend. Default is the local
    HTTP placeholder; cloud APIs are slotted in by implementing a SpeechFactory in
    :mod:`agent.multimodal.speech`.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import re
import time
import uuid
from dataclasses import fields
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent.messages_wire import (
    hybrid_messages_from_chat,
    is_local_hybrid_messages_endpoint,
    openai_messages_from_chat,
    uses_anthropic_messages_wire,
    uses_anthropic_tools_wire,
)

from .core import (
    Config,
    GeminiMemoryClient,
    LLMClientFactory,
    LocalSpeechFactory,
    MemoryLLMClient,
    MessagesMemoryClient,
    OpenAIMemoryClient,
    SpeechFactory,
)

log = logging.getLogger("hermes.multimodal.glue")


def _memory_prefers_messages_endpoint(model: str) -> bool:
    """Whether memory calls should use /v1/messages instead of chat completions."""
    m = (model or "").strip().lower()
    return "gpt-5.6 luna" in m or m == "kimi/kimi-k3"


def _prefers_messages_transport(model: str, base_url: str) -> bool:
    """Whether an explicit backend should use the Messages client.

    Transport signal precedence (endpoint > model name):
      1. endpoint ends with ``/chat/completions`` (or ``/v1/chat/completions``)
         → OpenAI wire, HARD OVERRIDE. Even when the model is Luna/K3, the
         user has explicitly targeted the chat_completions leaf and expects
         OpenAI-compatible request format (e.g. a hosted Luna endpoint 在 chat/
         completions 端点支持 multi-image + text via OpenAI wire, but the
         same model on /v1/messages proxy strips vision — the endpoint is
         what actually differs).
      2. endpoint ends with ``/v1/messages`` → Messages wire (Anthropic).
      3. otherwise fall back to the historical model-name heuristic (Luna/K3
         default to Messages when the endpoint doesn't disambiguate).
    """
    endpoint = (base_url or "").strip().rstrip("/").lower()
    # ★ Endpoint HARD OVERRIDE: explicit chat_completions leaf wins over any
    #   model-name-based routing. Fixes recall @ Luna @ chat/completions being
    #   silently redirected to MessagesMemoryClient by the "gpt-5.6 luna" in
    #   model-name substring match, which would then fail on the wrong wire.
    if endpoint.endswith("/chat/completions"):
        return False
    if endpoint.endswith("/v1/messages"):
        return True
    return _memory_prefers_messages_endpoint(model)


def _messages_endpoint(base_url: str) -> str:
    base = (base_url or "").rstrip("/")
    if base.endswith("/v1/messages"):
        return base
    if base.endswith("/v1"):
        return base + "/messages"
    return base + "/v1/messages"


def _messages_uses_anthropic_wire(base_url: str) -> bool:
    return uses_anthropic_messages_wire(base_url)


def _anthropic_tool_choice(tool_choice: Any, payload: Dict[str, Any]) -> Any:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, dict):
        choice_type = str(tool_choice.get("type") or "").strip().lower()
        if choice_type == "function":
            name = ((tool_choice.get("function") or {}).get("name") or "").strip()
            return {"type": "tool", "name": name} if name else None
        if choice_type in {"auto", "any", "tool"}:
            return tool_choice
        if choice_type == "required":
            return {"type": "any"}
        if choice_type == "none":
            payload.pop("tools", None)
            return None
        return tool_choice
    if isinstance(tool_choice, str):
        value = tool_choice.strip()
        lower = value.lower()
        if lower == "required":
            return {"type": "any"}
        if lower == "auto":
            return {"type": "auto"}
        if lower == "none":
            payload.pop("tools", None)
            return None
        if value:
            return {"type": "tool", "name": value}
    return None


def _anthropic_tools_from_openai(tools: Any) -> list:
    result = []
    seen = set()
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        if tool.get("name") and tool.get("input_schema"):
            converted = copy.deepcopy(tool)
        else:
            fn = tool.get("function") if isinstance(tool.get("function"), dict) else {}
            name = str(fn.get("name") or "").strip()
            if not name:
                continue
            schema = fn.get("parameters")
            if not isinstance(schema, dict):
                schema = {"type": "object", "properties": {}}
            schema = copy.deepcopy(schema)
            schema.setdefault("type", "object")
            if schema.get("type") == "object" and not isinstance(schema.get("properties"), dict):
                schema["properties"] = {}
            converted = {
                "name": name,
                "description": fn.get("description") or "",
                "input_schema": schema,
            }
        name = converted.get("name")
        if name in seen:
            continue
        seen.add(name)
        result.append(converted)
    return result


def _anthropic_image_source(url: str) -> Dict[str, str]:
    url = str(url or "")
    if url.startswith("data:"):
        header, _, data = url.partition(",")
        media_type = "image/jpeg"
        if header.startswith("data:") and ";" in header:
            media_type = header[5:].split(";", 1)[0] or media_type
        return {
            "type": "base64",
            "media_type": media_type,
            "data": data,
        }
    return {"type": "url", "url": url}


def _anthropic_content_blocks(content: Any) -> list:
    if content is None:
        return [{"type": "text", "text": ""}]
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        return [{"type": "text", "text": str(content)}]
    blocks = []
    for part in content:
        if not isinstance(part, dict):
            blocks.append({"type": "text", "text": str(part)})
            continue
        ptype = part.get("type")
        if ptype == "text":
            blocks.append({"type": "text", "text": str(part.get("text") or "")})
        elif ptype == "image":
            blocks.append(copy.deepcopy(part))
        elif ptype == "image_url":
            image_url = part.get("image_url")
            url = image_url.get("url") if isinstance(image_url, dict) else image_url
            if url:
                blocks.append({"type": "image", "source": _anthropic_image_source(str(url))})
        elif ptype in {"tool_use", "tool_result"}:
            blocks.append(copy.deepcopy(part))
        else:
            text = part.get("text")
            if text is not None:
                blocks.append({"type": "text", "text": str(text)})
    return blocks or [{"type": "text", "text": ""}]


def _tool_result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False)
    except Exception:
        return str(content)


def _merge_anthropic_messages(messages: list) -> list:
    merged = []
    for msg in messages:
        if not merged or merged[-1].get("role") != msg.get("role"):
            merged.append(msg)
            continue
        prev = merged[-1]
        prev_content = prev.get("content")
        cur_content = msg.get("content")
        if not isinstance(prev_content, list):
            prev_content = [{"type": "text", "text": str(prev_content or "")}]
        if not isinstance(cur_content, list):
            cur_content = [{"type": "text", "text": str(cur_content or "")}]
        prev["content"] = prev_content + cur_content
    return merged


def _anthropic_messages_from_openai(messages: Any) -> Tuple[Any, list]:
    system_parts = []
    out = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role") or "user"
        content = msg.get("content")
        if role == "system":
            system_parts.extend(_anthropic_content_blocks(content))
            continue
        if role == "tool":
            out.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id") or msg.get("id") or "",
                    "content": _tool_result_text(content),
                }],
            })
            continue
        if role == "assistant":
            blocks = _anthropic_content_blocks(content)
            tool_blocks = []
            for idx, tool_call in enumerate(msg.get("tool_calls") or []):
                if not isinstance(tool_call, dict):
                    continue
                fn = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except Exception:
                        args = {"arguments": args}
                if not isinstance(args, dict):
                    args = {}
                tool_blocks.append({
                    "type": "tool_use",
                    "id": tool_call.get("id") or f"call_messages_{idx}",
                    "name": str(fn.get("name") or ""),
                    "input": args,
                })
            if tool_blocks:
                blocks = [b for b in blocks if b.get("type") != "text" or b.get("text")]
                blocks.extend(tool_blocks)
            out.append({"role": "assistant", "content": blocks})
            continue
        out.append({"role": "user", "content": _anthropic_content_blocks(content)})
    system: Any = None
    if system_parts:
        text_parts = [b.get("text", "") for b in system_parts if b.get("type") == "text"]
        system = "\n".join(t for t in text_parts if t)
        if not system:
            system = system_parts
    return system, _merge_anthropic_messages(out)


def _messages_preserve_image_url_from_openai(messages: Any) -> Tuple[Any, list]:
    """Build a /v1/messages payload that keeps OpenAI-style image_url parts.

    Some messages gateways use the endpoint path while retaining OpenAI content
    parts.  System text is still lifted to top-level ``system`` for the direct
    messages transport.
    """
    system_parts = []
    out = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "user")
        content = msg.get("content")
        if role == "system":
            if isinstance(content, str) and content:
                system_parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text = part.get("text")
                        if isinstance(text, str) and text:
                            system_parts.append(text)
            continue
        if role not in {"assistant", "user"}:
            role = "user"
        out.append({"role": role, "content": copy.deepcopy(content)})
    system = "\n\n".join(system_parts).strip() or None
    return system, _merge_anthropic_messages(out)


def _messages_payload(kwargs: Dict[str, Any], *, base_url: str = "") -> Dict[str, Any]:
    kwargs = kwargs or {}
    model = str(kwargs.get("model") or "").strip().lower()
    if _messages_uses_anthropic_wire(base_url):
        try:
            if model == "kimi/kimi-k3":
                system, messages = _messages_preserve_image_url_from_openai(
                    kwargs.get("messages") or [])
            else:
                system, messages = _anthropic_messages_from_openai(
                    kwargs.get("messages") or [])
            payload: Dict[str, Any] = {
                "model": kwargs.get("model"),
                "messages": messages,
            }
            if system:
                payload["system"] = system
            tools = _anthropic_tools_from_openai(kwargs.get("tools") or [])
            if tools:
                payload["tools"] = tools
            converted_choice = _anthropic_tool_choice(
                kwargs.get("tool_choice"), payload)
            if converted_choice is not None and payload.get("tools"):
                payload["tool_choice"] = converted_choice
        except Exception as exc:
            log.warning(
                "messages payload Anthropic conversion failed; "
                "falling back to raw OpenAI payload: %s", exc)
            payload = {
                key: kwargs[key]
                for key in ("model", "messages", "tools", "tool_choice")
                if key in kwargs
            }
    else:
        if uses_anthropic_tools_wire(base_url):
            system, messages = hybrid_messages_from_chat(
                kwargs.get("messages") or [],
                lift_system=not is_local_hybrid_messages_endpoint(base_url),
            )
        else:
            system, messages = openai_messages_from_chat(
                kwargs.get("messages") or [],
                lift_system=not is_local_hybrid_messages_endpoint(base_url),
            )
        payload = {"model": kwargs.get("model"), "messages": messages}
        if system:
            payload["system"] = system
        if uses_anthropic_tools_wire(base_url):
            tools = _anthropic_tools_from_openai(kwargs.get("tools") or [])
            if tools:
                payload["tools"] = tools
            converted_choice = _anthropic_tool_choice(
                kwargs.get("tool_choice"), payload)
            if converted_choice is not None and payload.get("tools"):
                payload["tool_choice"] = converted_choice
        else:
            for key in ("tools", "tool_choice"):
                if key in kwargs:
                    payload[key] = kwargs[key]
    max_out = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens")
    if max_out:
        payload["max_tokens"] = int(max_out)
    if model == "kimi/kimi-k3":
        payload["reasoning_effort"] = "low"
    return payload


def _normalize_messages_response(data: Any) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise RuntimeError(f"non-object /v1/messages response: {data!r}")
    if "choices" in data:
        return data
    if data.get("type") != "message":
        return data

    text_parts = []
    tool_calls = []
    for idx, part in enumerate(data.get("content") or []):
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            text = part.get("text")
            if isinstance(text, str):
                text_parts.append(text)
        elif part.get("type") == "tool_use":
            args = part.get("input")
            try:
                arguments = json.dumps(
                    args if isinstance(args, dict) else {},
                    ensure_ascii=False,
                )
            except Exception:
                arguments = "{}"
            tool_calls.append({
                "id": part.get("id") or f"call_messages_{idx}",
                "type": "function",
                "function": {
                    "name": str(part.get("name") or ""),
                    "arguments": arguments,
                },
            })

    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    stop_reason = str(data.get("stop_reason") or "").strip()
    finish_reason = "tool_calls" if tool_calls else (
        "length" if stop_reason == "max_tokens" else "stop"
    )
    return {
        "id": data.get("id") or f"chatcmpl_messages_{int(time.time() * 1000)}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": data.get("model"),
        "choices": [{
            "index": 0,
            "finish_reason": finish_reason,
            "message": {
                "role": data.get("role") or "assistant",
                "content": "".join(text_parts) if text_parts else None,
                "tool_calls": tool_calls or None,
                "refusal": None,
            },
        }],
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens", usage.get("input_tokens", 0)),
            "completion_tokens": usage.get("completion_tokens", usage.get("output_tokens", 0)),
            "total_tokens": usage.get("total_tokens", (
                (usage.get("input_tokens", 0) or 0)
                + (usage.get("output_tokens", 0) or 0)
            )),
        },
    }


def _to_attr(obj: Any) -> Any:
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_attr(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_attr(v) for v in obj]
    return obj


class _OneShotAsyncStream:
    def __init__(self, response: Any):
        self._response = response
        self._sent = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._sent:
            raise StopAsyncIteration
        self._sent = True
        choice = self._response.choices[0]
        msg = choice.message
        delta = SimpleNamespace(
            content=getattr(msg, "content", None),
            reasoning_content=getattr(msg, "reasoning_content", None),
            reasoning=getattr(msg, "reasoning", None),
            tool_calls=[],
        )
        for idx, tc in enumerate(getattr(msg, "tool_calls", None) or []):
            fn = getattr(tc, "function", SimpleNamespace())
            delta.tool_calls.append(SimpleNamespace(
                index=idx,
                id=getattr(tc, "id", None),
                function=SimpleNamespace(
                    name=getattr(fn, "name", ""),
                    arguments=getattr(fn, "arguments", ""),
                ),
            ))
        return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


class MessagesChatCompletionsClient:
    """AsyncOpenAI-shaped client that posts directly to /v1/messages."""

    def __init__(self, *, base_url: str, api_key: str, model: str):
        import httpx
        self.endpoint = _messages_endpoint(base_url)
        self.api_key = api_key or "EMPTY"
        self.model = model
        self._client = httpx.AsyncClient(timeout=None)
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    async def aclose(self) -> None:
        try:
            await self._client.aclose()
        except Exception:
            pass

    async def close(self) -> None:
        await self.aclose()

    async def _create(self, **kwargs):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload = _messages_payload(kwargs, base_url=self.endpoint)
        if not payload.get("model"):
            payload["model"] = self.model
        request_kwargs: Dict[str, Any] = {
            "headers": headers,
            "json": payload,
        }
        if kwargs.get("timeout") is not None:
            request_kwargs["timeout"] = kwargs["timeout"]
        resp = await self._client.post(self.endpoint, **request_kwargs)
        if resp.status_code >= 400:
            raise RuntimeError(resp.text)
        data = _normalize_messages_response(resp.json())
        out = _to_attr(data)
        if kwargs.get("stream"):
            return _OneShotAsyncStream(out)
        return out


# Config keys that map 1:1 from the Hermes ``multimodal:`` dict onto Config fields.
# (Everything in DEFAULT_CONFIG["multimodal"] except ``enabled``, which gates the
# subsystem rather than parameterizing a worker.)
_NUMERIC_KEYS = (
    # (删: front_* / cruise_macro_summary_max_chars — FrontWorker 已删的死配置)
    "cont_max_tokens", "cont_recent_frames",
    "cont_now_frames", "cont_recent_history_turns",
    "query_worker_max_concurrency", "query_worker_max_pending",
    "writer_wake_interval", "writer_max_tokens",
    "writer_recent_frames", "writer_image_max_side",
    "writer_image_jpeg_quality", "writer_asr_window_sec",
    # E9 (entity tier injection) + E10 (event timeline) — new in the latest engine
    "writer_entity_enabled", "writer_entity_min_seen",
    "writer_entity_macro_lookback", "writer_entity_tier1_n",
    "writer_entity_tier2_n", "writer_entity_tier3_n",
    "writer_entity_tier1_thumb_side", "writer_entity_tier1_thumb_quality",
    "writer_entity_tier1_visual_types", "writer_reused_id_enabled",
    "writer_event_timeline_enabled", "writer_event_max_macros",
    "writer_event_max_micros",
    "react_max_tokens", "react_recent_frames",
    "react_recent_history_turns", "react_max_rounds",
    "react_search_tasks_max",
    "react_round_max_tokens", 
    "watcher_answer_max_tokens",
    # ★ merge 修复: 这些 watcher pacing 兜底旋钮之前漏在 flatten 白名单外 →
    #   config.yaml 里改了不生效 (静默回落 dataclass 默认)。补上。
    "watch_min_batch", "watch_round_ttl_sec", "watch_poll_interval",
    "watch_static_tail_flush_sec", "watch_request_max_images",
    "watch_report_every_rounds", "watch_completion_confirm_delay_sec",
    "watch_completion_confirm_retry_total_sec",
    "watch_completion_confirm_max_attempts",
    "watch_completion_confirm_frames",
    "watch_completion_confirm_min_confidence",
    "search_max_tokens", "search_max_tool_rounds",
    "search_recent_frames",
    "recall_max_tokens", "recall_max_rounds",
    "recall_topk_micro", "recall_topk_entity", "recall_distill_max_tokens",
    "recall_decide_frames", "recall_verify_enabled", "recall_verify_max_frames",
    "recall_verify_retries", "recall_verify_retry_delay_sec",
    "conv_max_chars", "conv_min_turns", "conv_max_bg_obs", "conv_max_audio_obs",
    "buffer_seconds", "buffer_capture_fps",
    "search_facts_max", "search_fact_ttl_sec",
    "search_fact_value_max_chars", "facts_max",
    "mem_l2_macro_min_micro", "mem_l2_macro_max_duration",
    "mem_l3_super_min_macro", "mem_l3_super_max_duration",
    "mem_entity_alias_threshold", 
    "mem_aggregator_max_tokens", "mem_aggregator_image_max_side",
    "mem_aggregator_image_jpeg_quality", "agg_l2_frames", "agg_l3_frames",
    "mem_micro_max_duration", "mem_micro_max_ticks",
    "reviewer_enabled", "reviewer_wake_interval", "reviewer_total_frames",
    "reviewer_recent_ratio", "reviewer_recent_window_sec", "reviewer_min_micros",
    "reviewer_max_tokens",
    "reviewer_max_actions_per_round", "reviewer_min_seg_dur_for_split",
    "reviewer_image_max_side", "reviewer_image_jpeg_quality",
    "reviewer_max_concurrency", "reviewer_single_endpoint_interval_sec",
    "reviewer_overload_retries",
    "reviewer_retry_backoff_sec",
    "frame_store_max", "frame_store_dedup_scan_n", "frame_store_ts_exact_eps",
    # (frame_store_dhash_lookback/threshold 已废弃: FrameStore 层2 dHash 删除,
    #  画面级去重上移到 FrameBuffer 入口。不再从 config 透传。)
    # FrameBuffer 入口去重 + 场景理解 → dHash 阈值
    "framebuffer_dhash_threshold_init", "framebuffer_dhash_threshold_min",
    "framebuffer_dhash_threshold_max",
    "scene_probe_interval_s", "scene_probe_window_s", "scene_probe_frames",
    "scene_probe_maxside", "scene_probe_quality", "scene_probe_use_llm",
    "scene_probe_timeout_s",
    "cont_recall_frames_max", "ui_event_thumb_max_side",
    "ui_event_thumb_jpeg_quality", "inflight_brief_ttl",
    "memory_max_consecutive_failures", "reviewer_max_consecutive_failures",
    "env_audio_enabled", "env_audio_window_sec", "env_audio_min_rms",
    "env_audio_min_text_chars", "env_audio_filter_fillers",
    # Audio backends (str/bool/int — generic setattr handles all types)
    "asr_url", "asr_timeout", "asr_run_vllm", "asr_api_key", "asr_model",
    "env_asr_backend", "env_url", "env_api_key", "env_asr_model",
    "whisperx_timeout", "whisperx_language",
    "whisperx_diarize", "whisperx_min_speakers", "whisperx_max_speakers",
    # DashScope realtime speech (streaming mic ASR + streaming TTS)
    "dashscope_api_key",
    "realtime_asr_enabled", "realtime_asr_model", "realtime_asr_language",
    "realtime_asr_sample_rate",
    "realtime_asr_vad_threshold", "realtime_asr_vad_silence_ms",
    "realtime_tts_enabled", "realtime_tts_model", "realtime_tts_voice",
    "realtime_tts_sample_rate", "realtime_tts_speech_rate",
    # AnySearch (text_search 后端) 数值项
    "anysearch_max_results", "anysearch_timeout",
    # ★ P1: 3 专项 Reviewer 开关 + Entity 帧预算 (bool/int, _coerce 处理)
    "reviewer_entity_enabled", "reviewer_event_enabled",
    "reviewer_edge_enabled", "reviewer_entity_frames", "reviewer_event_frames",
    "reviewer_event_gate_enabled", "reviewer_event_sample_every_macros",
    "reviewer_event_min_micro_count",
    "reviewer_event_min_entity_state_changes",
    "reviewer_event_min_distinct_entities",
    "reviewer_event_min_asr_cues", "reviewer_event_min_asr_chars",
    # 记忆模型视觉能力开关 (v33: audio_ability/omni 音频路径已删)
    "writer_vision_ability", "entity_rep_frames_max_n",
    # ★ Embedding backend + 混合检索融合数值参数 (一期文本 embedding).
    "embedding_dimensions", "embedding_timeout_sec", "embedding_batch_size",
    "recall_hybrid_enabled", "recall_vector_topk", "recall_rrf_k",
    # ★ 二期: 帧图像 embedding 数值参数
    "mm_embedding_timeout_sec", "recall_frame_vector_topk",
    "frame_vector_pool_cap",
    "mm_embedding_dimensions", "mm_embedding_res_level",
    # ★ OCR 垂类模型: writer 前置屏幕文字抽取 (v33: ocr_enabled 删 → 必开;
    #   local-only rapidocr — remote/cloud VLM OCR 已移除)
    "ocr_timeout_sec", "ocr_max_tokens",
    "ocr_max_side", "ocr_max_threads", "ocr_frames_between_ocr",
    "ocr_worker_backlog_limit",
    # Dedicated Monitor MaaS auth compatibility. Default False preserves the
    # exact client kwargs used by all existing endpoints.
    "monitor_send_api_key_header",
)
# AnySearch 字符串配置项 (endpoint / api key)
_ANYSEARCH_KEYS = ("anysearch_endpoint", "anysearch_api_key")

# Optional dedicated MemoryWriter/Reviewer backend (string settings, not numeric).
# Default empty → memory writing reuses the main resolved Hermes model. Set
# memory_provider="gemini" (+ base_url/api_key/model) to route memory through a
# vision/JSON-strong Gemini generateContent endpoint, or "openai" for a separate
# OpenAI-compatible endpoint. Endpoints/keys are NOT hardcoded.
_MEMORY_BACKEND_KEYS = (
    "memory_provider", "memory_base_url", "memory_api_key", "memory_model",
)

# Optional dedicated backends for the deep-analysis workers and the monitor.
# Default empty → both fall back to the main resolved Hermes model (original
# behavior). Filling base_url (+ model) points that role at an independent
# OpenAI-compatible endpoint. Same shape/mechanism as _MEMORY_BACKEND_KEYS.
_WORKER_BACKEND_KEYS = (
    "watcher_provider", "watcher_base_url", "watcher_api_key",  # worker_model via cfg.model
)
_MONITOR_BACKEND_KEYS = (
    "monitor_provider", "monitor_base_url", "monitor_api_key", "monitor_model",
)
# RecallAgent loads directly from model.memory, like MemoryWriter. Reviewer can
# optionally override the same 4-tuple under model.memory.reviewer.
_REVIEWER_BACKEND_KEYS = (
    "reviewer_provider", "reviewer_base_url", "reviewer_base_urls",
    "reviewer_api_key", "reviewer_model",
)
# ★ Recall decide/distill 【独立端点】4 件套 (post-v33 新增)。空 → recall_client
#   fallback 到 memory_client (老行为)。用于 writer/recall 分家: writer 保留
#   K2.6 MaaS (高频便宜), recall 切 Luna @ chat/completions (视觉多图+文)。
_RECALL_BACKEND_KEYS = (
    "recall_provider", "recall_base_url",
    "recall_api_key", "recall_model",
)
_RECALL_VERIFY_BACKEND_KEYS = (
    "recall_verify_provider", "recall_verify_base_url",
    "recall_verify_api_key", "recall_verify_model",
)
# ★ Embedding backend 四件套 (与 memory/monitor/recall 同款契约):
#   base_url 空 → 全局关闭 embedding, 混合检索退化为纯关键词。
_EMBEDDING_BACKEND_KEYS = (
    "embedding_provider", "embedding_base_url", "embedding_api_key",
    "embedding_model",
    # ★ 二期: 帧图像 embedding 字符串项 (DashScope multimodal-embedding-v1)
    "mm_embedding_model", "mm_embedding_base_url", "mm_embedding_api_key",
)
# OCR backend string keys (local-only rapidocr; remote 4-tuple removed).
_OCR_BACKEND_KEYS = (
    "ocr_backend",
)


# Sentinel: coercion failed → skip the key and keep the dataclass default.
_COERCE_FAILED = object()

# Flat keys that legitimately appear in the flattened dict but are NOT Config
# dataclass fields (consumed by build_config directly, or gate the subsystem).
# The unknown-key guard must not warn about these.
# ── Retired keys ────────────────────────────────────────────────────────────
# Keys that WERE real Config fields and have since been removed on purpose.
# They stay listed here so an existing config.yaml that still carries them
# loads silently: the unknown-key guard below exists to catch typos ("your
# edit was IGNORED"), and shouting that at someone whose file predates a
# deliberate removal is just noise they can do nothing useful about.
#
# The *_temperature family went away when sampling params stopped being sent
# at all — see agent/transports/chat_completions.py build_kwargs for why.
_RETIRED_KEYS = frozenset({
    "cont_temperature", "writer_temperature", "react_temperature",
    "watcher_answer_temperature", "search_temperature", "recall_temperature",
    "mem_aggregator_temperature", "reviewer_temperature",
    # Remote/cloud VLM OCR removed (local rapidocr is the only backend):
    # model.ocr.use_local + model.ocr.remote_backend.{provider,base_url,
    # api_key,model} → flat ocr_use_local / ocr_provider / ocr_base_url /
    # ocr_api_key / ocr_model no longer map to any Config field.
    "ocr_use_local", "ocr_provider", "ocr_base_url", "ocr_api_key", "ocr_model",
})


_NON_CFG_PASSTHROUGH = frozenset({
    "enabled",            # subsystem gate (multimodal_enabled), not a Config field
    "worker_model",       # legacy alias → cfg.model pin (read in build_config)
    "watcher_model",      # v33 name → cfg.model pin (read in build_config)
    "history_log_path",   # re-rooted to HERMES_HOME in build_config
    "mem_db_path",        # re-rooted / per-stream in build_config
    "speech_provider",    # speech factory selection, not a Config field
    "vision_model",       # main-agent vision override, consumed by gateway
    # legacy pre-v33 flat worker_* names (aliased to watcher_* with their own
    # migration warning — don't double-warn via the unknown-key guard).
    "worker_provider", "worker_base_url", "worker_api_key", "worker_model",
})

# Flat-name PREFIXES read raw from the flattened dict (mm.get(...)) by the
# gateway / monitor engine / VoiceAgent — legitimately not Config dataclass
# fields, so the unknown-key guard must not warn about them.
#   voice_*   → VoiceAgent (self._cfg.get("settings"))
#   vision_*  → main-agent per-turn vision-model switch (vision_model/provider)
#   monitor_* → monitor_engine + gateway (tick/merge/overload/enabled)
_RAW_FLAT_PREFIXES = ("voice_", "vision_", "monitor_")


# --------------------------------------------------------------------------- #
# Schema translation: new nested layout → legacy flat `multimodal.*` dict.
#
# config.yaml (v32+) splits the old flat ``multimodal:`` block into three homes:
#   * model.monitor / model.watcher / model.memory[.recall]  — the per-role
#     LLM endpoints (4-tuple {provider, base_url, api_key, model}).
#   * settings.*  — every non-audio behavior knob (renamed from ``multimodal``).
#   * audio.*     — the ASR/TTS/env-audio/DashScope interface keys.
#
# The multimodal Config dataclass (and every downstream worker) still reads the
# ORIGINAL flat field names (cfg.watcher_provider / cfg.monitor_model /
# cfg.memory_* / cfg.recall_* / cfg.asr_url / …). So we translate the new nested
# layout back into one flat dict keyed by the legacy names, then the existing
# build_config logic populates the dataclass unchanged.
#
# Backward compatible: if a value isn't found in the new location we fall back to
# the legacy flat ``multimodal.<key>`` so an old config.yaml still loads.
# --------------------------------------------------------------------------- #

# (nested-role-key, dest-prefix): model.<role> 4-tuple → cfg.<prefix>_<field>.
# watcher maps onto the watcher_* fields (watcher == the deep-research worker;
# dataclass prefix renamed worker_→watcher_ to match the yaml role name).
_MODEL_ROLE_MAP = (
    ("monitor", "monitor"),
    ("watcher", "watcher"),
    ("memory", "memory"),
    ("embedding", "embedding"),
    # (ocr removed: remote VLM OCR 4-tuple deleted — OCR is local rapidocr only.)
)
_ROLE_TUPLE_FIELDS = ("provider", "base_url", "api_key", "model")

# Audio interface keys that live under the new ``audio:`` section (moved out of
# ``multimodal``). Everything else stays a ``settings.*`` behavior knob.
_AUDIO_KEYS = (
    "asr_url", "asr_timeout", "asr_run_vllm", "asr_api_key", "asr_model",
    "env_asr_backend", "env_url", "env_api_key", "env_asr_model",
    "whisperx_timeout", "whisperx_language", "whisperx_diarize",
    "whisperx_min_speakers", "whisperx_max_speakers",
    "dashscope_api_key",
    "realtime_asr_enabled", "realtime_asr_model", "realtime_asr_language",
    "realtime_asr_sample_rate", "realtime_asr_vad_threshold",
    "realtime_asr_vad_silence_ms",
    "realtime_tts_enabled", "realtime_tts_model", "realtime_tts_voice",
    "realtime_tts_sample_rate", "realtime_tts_speech_rate",
    "env_audio_enabled", "env_audio_window_sec", "env_audio_min_rms",
    "env_audio_min_text_chars", "env_audio_filter_fillers", "conv_max_audio_obs",
    # TTS (local/http) knobs also belong to the audio interface.
    "tts_url", "tts_voice", "tts_language", "tts_instruct",
    "tts_connect_timeout", "tts_read_timeout",
    "tts_min_sentence_chars", "tts_max_sentence_chars",
    "speech_provider",
)


# --------------------------------------------------------------------------- #
# Deep-path flattening (v33: module aggregation).
#
# config.yaml v33 groups every submodule's BEHAVIOR knobs under its
# model.<role> block (not just the 4-tuple endpoint). E.g.
#   model.memory.writer.wake_interval  → cfg.writer_wake_interval
#   model.watcher.react.max_rounds → cfg.react_max_rounds
#   settings.framebuffer.dhash_threshold_init → cfg.framebuffer_dhash_threshold_init
#
# The dataclass field names are NOT a clean prefix+leaf (writer entity fields are
# writer_entity_*, aggregator fields are mem_aggregator_*/agg_l2_frames, etc.), so
# we use an explicit ``nested-path-prefix → flat-name-builder`` table. Each entry
# maps a dotted nested location to either:
#   * a string prefix  → flat name is f"{prefix}_{leaf}" (e.g. "writer" → writer_x)
#   * a dict           → exact per-leaf overrides {leaf: flat_name}; leaves not in
#                        the dict fall through to the prefix (dict must carry a
#                        "" key holding the default prefix, or None to drop)
#
# Anything read here is written into the flat ``out`` dict keyed by dataclass
# field names, exactly like the legacy path. Unknown leaves are collected and
# warned about (治"改了不生效": a typo'd key never silently vanishes).
# --------------------------------------------------------------------------- #

# Dotted nested path (under the top-level section) → flat prefix.
# The walker joins any DEEPER nesting with "_" onto the leaf before prefixing,
# so model.memory.writer.entity.min_seen resolves via the "model.memory.writer"
# entry: leaf="entity_min_seen", prefix="writer" → "writer_entity_min_seen".
_DEEP_PATH_PREFIX = {
    # ── model.monitor.* (behavior scalars directly under the role) → monitor_* ──
    "model.monitor": "monitor",
    # ── model.watcher.* (deep-research worker) ──
    "model.watcher.react": "react",
    "model.watcher.search": "search",
    "model.watcher.watch": "watch",
    # ── model.memory.* ──
    "model.memory.writer": "writer",              # writer.entity.* → writer_entity_*
    "model.memory.reviewer": "reviewer",          # reviewer.event_gate.* → reviewer_event_*
    "model.memory.recall": "recall",
    "model.memory.micro": "mem_micro",            # micro.max_duration → mem_micro_max_duration
    "model.memory.frame_store": "frame_store",
    # ── model.embedding.* (scalars directly under role) → embedding_* ──
    "model.embedding": "embedding",
    "model.embedding.hybrid": "recall",           # hybrid.vector_topk → recall_vector_topk
    # ── model.ocr.* (scalars directly under role) → ocr_* ──
    "model.ocr": "ocr",
    # ── settings.* nested groups ──
    "settings.framebuffer": "framebuffer",
    "settings.scene_probe": "scene_probe",
    "settings.conv": "conv",
    "settings.ui_event_thumb": "ui_event_thumb",
    # NOTE: settings.voice_* stays FLAT (VoiceAgent reads the raw settings dict
    # directly via self._cfg.get("settings"), bypassing flatten — nesting would
    # silently break its reads). Do NOT add settings.voice here.
    # ── audio.* nested groups ──
    "audio.realtime_asr": "realtime_asr",
    "audio.realtime_tts": "realtime_tts",
    "audio.env_audio": "env_audio",
    "audio.whisperx": "whisperx",
    "audio.tts": "tts",
    "audio.asr": "asr",
}

# Per-path EXACT leaf→flat overrides (win over the prefix rule above). Used where
# the dataclass name doesn't follow prefix_leaf. Value None drops the key.
_DEEP_PATH_EXACT = {
    # (settings.vision.inject_* removed in v33 — passive frame injection deleted;
    #  main agent uses the get_current_frame tool instead.)
    "settings.framebuffer": {
        "seconds": "buffer_seconds",
        "capture_fps": "buffer_capture_fps",
        "dhash_threshold_init": "framebuffer_dhash_threshold_init",
        "dhash_threshold_min": "framebuffer_dhash_threshold_min",
        "dhash_threshold_max": "framebuffer_dhash_threshold_max",
    },
    "settings.scene_probe": {
        "interval_s": "scene_probe_interval_s",
        "window_s": "scene_probe_window_s",
        "frames": "scene_probe_frames",
        "maxside": "scene_probe_maxside",
        "quality": "scene_probe_quality",
        "use_llm": "scene_probe_use_llm",
        "timeout_s": "scene_probe_timeout_s",
    },
    "settings.conv": {
        "max_chars": "conv_max_chars",
        "min_turns": "conv_min_turns",
        "max_bg_obs": "conv_max_bg_obs",
        "max_audio_obs": "conv_max_audio_obs",
    },
    "model.memory.recall": {
        # recall endpoint 4-tuple is handled separately; behavior knobs here.
        "decide_frames": "recall_decide_frames",
        "verify_max_frames": "recall_verify_max_frames",
        "cont_frames_max": "cont_recall_frames_max",
    },
    "model.memory": {
        # memory-level knobs that aren't under a subblock.
        "entity_alias_threshold": "mem_entity_alias_threshold",
        "max_failures": "memory_max_consecutive_failures",
    },
    # (v33: omni tuning moved to memory top-level keys thinking_level /
    #  media_resolution / audio_window_sec — handled by _MEM_ABILITY_MAP, not here.)
    "model.memory.writer.entity": {
        # E9 entity tier — mostly writer_entity_* but rep_frames has a _n suffix.
        "enabled": "writer_entity_enabled",
        "min_seen": "writer_entity_min_seen",
        "macro_lookback": "writer_entity_macro_lookback",
        "tier1_n": "writer_entity_tier1_n",
        "tier2_n": "writer_entity_tier2_n",
        "tier3_n": "writer_entity_tier3_n",
        "rep_frames_max": "entity_rep_frames_max_n",
    },
    "model.memory.writer.event_timeline": {
        # inconsistent dataclass names: enabled keeps 'timeline', counts drop it.
        "enabled": "writer_event_timeline_enabled",
        "max_macros": "writer_event_max_macros",
        "max_micros": "writer_event_max_micros",
    },
    "model.memory.reviewer": {
        "max_failures": "reviewer_max_consecutive_failures",
    },
    "model.memory.reviewer.event_gate": {
        # Keep the EventReviewer master switch independent from its expensive
        # gate. Without this exact override, the residual event_gate→event
        # rename aliases both `event_enabled` and `event_gate.enabled` onto
        # reviewer_event_enabled, leaving reviewer_event_gate_enabled at its
        # dataclass default.
        "enabled": "reviewer_event_gate_enabled",
    },
    "model.memory.aggregator": {
        "l2_min_micro": "mem_l2_macro_min_micro",
        "l2_max_duration": "mem_l2_macro_max_duration",
        "l3_min_macro": "mem_l3_super_min_macro",
        "l3_max_duration": "mem_l3_super_max_duration",
        "l2_frames": "agg_l2_frames",
        "l3_frames": "agg_l3_frames",
        "max_tokens": "mem_aggregator_max_tokens",
        "image_max_side": "mem_aggregator_image_max_side",
        "image_jpeg_quality": "mem_aggregator_image_jpeg_quality",
    },
    "model.embedding": {
        "dimensions": "embedding_dimensions",
        "batch_size": "embedding_batch_size",
        "timeout_sec": "embedding_timeout_sec",
    },
    "model.embedding.frame": {
        "model": "mm_embedding_model",
        "base_url": "mm_embedding_base_url",
        "api_key": "mm_embedding_api_key",
        "dimensions": "mm_embedding_dimensions",
        "res_level": "mm_embedding_res_level",
        "timeout_sec": "mm_embedding_timeout_sec",
    },
    "model.embedding.hybrid": {
        "enabled": "recall_hybrid_enabled",
        "vector_topk": "recall_vector_topk",
        "rrf_k": "recall_rrf_k",
        "frame_topk": "recall_frame_vector_topk",
        "frame_pool_cap": "frame_vector_pool_cap",
    },
    "settings": {
        # top-level settings knobs that keep their exact flat name (identity).
        "anysearch": None,   # handled as its own nested block below
    },
}

# Residual-segment renames: when a deeper subblock's residual path doesn't match
# the dataclass name. Keyed by the matched prefix path; maps residual → renamed.
#   model.memory.writer.event_timeline.max_macros: prefix "model.memory.writer",
#   residual "event_timeline" → rename to "event" → writer_event_max_macros.
#   model.memory.reviewer.event_gate.min_micro_count: residual "event_gate" →
#   "event" → reviewer_event_min_micro_count.
#   NOTE: writer.event_timeline / writer.entity are handled by EXACT overrides
#   (see _DEEP_PATH_EXACT) because their dataclass names are inconsistent, so no
#   residual rename is needed for them. reviewer.event_gate → event is uniform.
_DEEP_PATH_RESIDUAL_RENAME = {
    "model.memory.reviewer": {"event_gate": "event"},
}


def _walk_leaves(d: Dict[str, Any], base: str, out_pairs: list, unknown: list):
    """Recursively yield (dotted_path, leaf_suffix, value) for scalar leaves.

    dotted_path is the full path (e.g. "model.memory.writer.entity"); leaf_suffix
    is the trailing key(s) joined by "_" relative to the nearest mapped prefix —
    resolved by the caller. Here we just flatten to (full_parent_path, key, value)
    and let the resolver match the longest known prefix.
    """
    for k, v in d.items():
        path = f"{base}.{k}"
        if isinstance(v, dict):
            _walk_leaves(v, path, out_pairs, unknown)
        else:
            out_pairs.append((base, k, v))


def _resolve_deep_leaf(parent_path: str, leaf: str) -> Optional[str]:
    """Map (parent nested path, leaf key) → flat dataclass field name, or None.

    Longest-prefix match against _DEEP_PATH_PREFIX/_EXACT so deeper subblocks
    (writer.entity) win over shallower (writer).
    """
    # Try exact per-path override first (exact parent path match).
    exact = _DEEP_PATH_EXACT.get(parent_path)
    if exact is not None and leaf in exact:
        return exact[leaf]
    # Longest-matching prefix entry, joining the residual path onto the leaf.
    best = None
    for p in _DEEP_PATH_PREFIX:
        if parent_path == p or parent_path.startswith(p + "."):
            if best is None or len(p) > len(best):
                best = p
    if best is not None:
        prefix = _DEEP_PATH_PREFIX[best]
        residual = parent_path[len(best):].lstrip(".")   # e.g. "entity"
        # Apply residual-segment renames (event_timeline→event, event_gate→event).
        renames = _DEEP_PATH_RESIDUAL_RENAME.get(best)
        if residual and renames:
            parts = [renames.get(seg, seg) for seg in residual.split(".")]
            residual = "_".join(parts)
        else:
            residual = residual.replace(".", "_")
        suffix = f"{residual}_{leaf}" if residual else leaf
        return f"{prefix}_{suffix}"
    # exact-only paths (no prefix) already tried exact above → unknown.
    return None


def flatten_mm_config(hermes_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Merge the new nested config layout into one legacy-flat ``multimodal`` dict.

    Precedence: new nested location wins; legacy flat ``multimodal.<key>`` is the
    fallback so pre-v32 configs keep working. The returned dict is keyed by the
    ORIGINAL flat field names the Config dataclass expects.

    v33: also flattens per-module BEHAVIOR subblocks (model.<role>.<sub>.<leaf>,
    settings.<group>.<leaf>, audio.<group>.<leaf>) via _DEEP_PATH_* tables, and
    warns on any nested leaf that doesn't map to a known dataclass field (治"改了
    不生效": a typo'd key surfaces as a startup warning instead of silently
    vanishing).
    """
    hermes_cfg = hermes_cfg or {}
    legacy = dict(hermes_cfg.get("multimodal") or {})   # pre-v32 fallback source
    settings = dict(hermes_cfg.get("settings") or {})
    audio = dict(hermes_cfg.get("audio") or {})
    model = dict(hermes_cfg.get("model") or {}) if isinstance(
        hermes_cfg.get("model"), dict) else {}

    out: Dict[str, Any] = dict(legacy)  # start from legacy, override below

    # ── Legacy alias: pre-v33 flat worker_* → watcher_* (dataclass renamed). ──
    #    A1 hard-cut transition: honor the old name for 1~2 versions but WARN so
    #    the user migrates. Only aliases keys the user actually set.
    for _legacy_k, _new_k in (
        ("worker_provider", "watcher_provider"),
        ("worker_base_url", "watcher_base_url"),
        ("worker_api_key", "watcher_api_key"),
        ("worker_model", "watcher_model"),
    ):
        if _legacy_k in out and _new_k not in out:
            out[_new_k] = out[_legacy_k]
            log.warning("[config] %r is renamed to model.watcher.%s — the old "
                        "name still works for now but please migrate.",
                        _legacy_k, _new_k.split("_", 1)[1])

    # ── settings.* / audio.* SCALAR knobs override legacy (unchanged behavior).
    #    Nested dicts are handled by the deep-path walker below, not copied raw.
    for k, v in settings.items():
        if not isinstance(v, dict):
            out[k] = v
    for k, v in audio.items():
        if not isinstance(v, dict):
            out[k] = v

    # ── Deep-path nested subblocks → flat dataclass names (v33). ──
    unknown_leaves: list = []
    _deep_pairs: list = []
    # settings.<group>.* / audio.<group>.* nested dicts (anysearch handled below).
    for section, base in (("settings", "settings"), ("audio", "audio")):
        sect = hermes_cfg.get(section)
        if isinstance(sect, dict):
            for k, v in sect.items():
                if k == "anysearch":
                    continue   # special-cased below
                if isinstance(v, dict):
                    _walk_leaves(v, f"{base}.{k}", _deep_pairs, unknown_leaves)
    # model.<role>.* — BOTH scalars directly under the role (monitor.tick_sec,
    # ocr.enabled, embedding.dimensions) AND deeper sub-dicts. Keys consumed by
    # the 4-tuple / ability / mm-embedding / ocr-backend special handling below
    # are skipped here so they aren't double-processed or mis-warned.
    # Reserved ONLY at the top role level (these scalars directly under the role
    # are consumed by the 4-tuple / ability / mm / backend special handling
    # below). Nested sub-dicts (recall/reviewer/frame/writer/…) ARE walked; their
    # 4-tuple leaves resolve to the same flat name the special-case writes, so a
    # duplicate write is harmless.
    _ROLE_TOP_RESERVED = set(_ROLE_TUPLE_FIELDS) | {
        # memory vision ability (handled by _MEM_ABILITY_MAP)
        "vision_ability",
        # embedding mm_* + ocr backend (handled by their special-cases)
        "mm_model", "mm_base_url", "mm_api_key", "backend",
    }
    # Sub-dicts handled by their own special-case (not the generic walker).
    # NOTE: model.ocr.remote_backend is deliberately NOT skipped here — its
    # leaves fall through to the generic walker and resolve to flat ocr_* keys
    # that are now retired (remote VLM OCR removed), loading silently via
    # _RETIRED_KEYS instead of being mapped to a Config field.
    _ROLE_SUBDICT_SKIP = {("watcher", "anysearch"),
                          ("ocr", "local_backend")}
    for role in ("monitor", "watcher", "memory", "embedding", "ocr"):
        sub = model.get(role)
        if not isinstance(sub, dict):
            continue
        for k, v in sub.items():
            if (role, k) in _ROLE_SUBDICT_SKIP:
                continue
            if isinstance(v, dict):
                _walk_leaves(v, f"model.{role}.{k}", _deep_pairs, unknown_leaves)
            elif k not in _ROLE_TOP_RESERVED:
                # scalar directly under the role → parent_path = model.<role>
                _deep_pairs.append((f"model.{role}", k, v))
    # anysearch nested block (v33: moved to model.watcher.anysearch.* — it's the
    # deep-research watcher's external text-search backend). Legacy
    # settings.anysearch.* still accepted as a fallback.
    _as = None
    _as_src = ""
    _watcher_blk = model.get("watcher")
    if isinstance(_watcher_blk, dict) and isinstance(_watcher_blk.get("anysearch"), dict):
        _as, _as_src = _watcher_blk["anysearch"], "model.watcher.anysearch"
    elif isinstance(settings.get("anysearch"), dict):
        _as, _as_src = settings["anysearch"], "settings.anysearch"
    if isinstance(_as, dict):
        _AS_MAP = {
            "endpoint": "anysearch_endpoint", "api_key": "anysearch_api_key",
            "max_results": "anysearch_max_results",
            "result_max_chars": "anysearch_result_max_chars",
            "timeout": "anysearch_timeout",
        }
        for k, v in _as.items():
            flat = _AS_MAP.get(k)
            if flat:
                out[flat] = v
            else:
                unknown_leaves.append(f"{_as_src}.{k}")

    # ocr local_backend nested block (v33): model.ocr.local_backend.backend →
    # ocr_backend (the local rapidocr backend name). The remote_backend block
    # (model.ocr.remote_backend.{provider,base_url,api_key,model}) was REMOVED —
    # remote/cloud VLM OCR no longer exists; those leaves now fall through the
    # generic walker to retired flat ocr_* keys (see _RETIRED_KEYS) so old
    # configs load silently.
    _ocr_blk = model.get("ocr")
    if isinstance(_ocr_blk, dict):
        _lb = _ocr_blk.get("local_backend")
        if isinstance(_lb, dict):
            if _lb.get("backend") is not None:
                out["ocr_backend"] = _lb["backend"]
            for k in _lb:
                if k != "backend":
                    unknown_leaves.append(f"model.ocr.local_backend.{k}")

    for parent_path, leaf, value in _deep_pairs:
        if value is None:
            continue
        flat = _resolve_deep_leaf(parent_path, leaf)
        if flat is None:
            unknown_leaves.append(f"{parent_path}.{leaf}")
            continue
        out[flat] = value

    # ── Unknown-key guard (治"改了不生效"). Validate every flat key we produced
    #    against the real dataclass fields; anything unmatched is a typo/stale
    #    key → warn once so the user knows their edit was IGNORED, not applied.
    try:
        _valid = {f.name for f in fields(Config)}
        stray = [
            k for k in out
            if k not in _valid
            and k not in _NON_CFG_PASSTHROUGH
            and k not in _RETIRED_KEYS
            # These prefixes are read RAW from the flattened dict (mm.get(...)) by
            # the gateway / monitor engine / VoiceAgent — NOT Config fields:
            #   voice_*    → VoiceAgent (self._cfg.get("settings"))
            #   vision_*   → main-agent per-turn vision-model switch
            #   monitor_*  → monitor_engine + gateway (tick/merge/overload)
            and not k.startswith(_RAW_FLAT_PREFIXES)
        ]
        for u in unknown_leaves:
            log.warning("[config] unknown nested key %r does not map to any "
                        "Config field — IGNORED (check spelling/nesting).", u)
        for k in stray:
            log.warning("[config] key %r has no matching Config field — "
                        "IGNORED (check spelling; did the field get renamed?).", k)
    except Exception as e:  # pragma: no cover - validation must never block startup
        log.debug("[config] unknown-key validation skipped: %s", e)

    # model.<role> 4-tuple → flat cfg.<prefix>_<field>.
    for role, prefix in _MODEL_ROLE_MAP:
        sub = model.get(role)
        if not isinstance(sub, dict):
            continue
        for field_name in _ROLE_TUPLE_FIELDS:
            if field_name in sub and sub[field_name] is not None:
                out[f"{prefix}_{field_name}"] = sub[field_name]
        # (v33: ocr backend/endpoint moved to model.ocr.local_backend/remote_backend,
        #  handled by the ocr special-case above — not the 4-tuple/backend logic here.)
        # ★ memory 角色的模型能力声明 (跟 provider/model 同级):
        #   model.memory.vision_ability → writer_vision_ability (记忆抽取必须能看图)。
        #   (v33: audio_ability / audio_window_sec / omni tuning 全删 — omni 原始音频
        #    写入路径整体移除; 音频记忆走外部 ASR 字幕。)
        if role == "memory":
            _MEM_ABILITY_MAP = (
                ("vision_ability", "writer_vision_ability"),
            )
            for src_key, flat_key in _MEM_ABILITY_MAP:
                if src_key in sub and sub[src_key] is not None:
                    out[flat_key] = sub[src_key]
            # ★ post-v33: memory.recall 【可选】endpoint 4-tuple (provider/
            #   base_url/api_key/model) — 通过 _DEEP_PATH_PREFIX["model.memory.
            #   recall"]="recall" 前缀规则平铺成 recall_*, 由 recall_client() 消费。
            #   memory.reviewer 依旧没有独立 4-tuple (由下方 _REVIEWER_BACKEND_KEYS
            #   走 reviewer_base_urls 多路径专有解析)。行为 knobs (recall.topk_micro
            #   / reviewer.total_frames / …) 依旧走 deep-path walker。
        # ★ embedding 角色除 4-tuple 外, 二期带 mm_* 三件套 (帧图像 embedding,
        #   DashScope multimodal-embedding-v1, 与文本向量独立客户端):
        #     model.embedding.mm_model/mm_base_url/mm_api_key
        #     → cfg.mm_embedding_model/mm_embedding_base_url/mm_embedding_api_key.
        if role == "embedding":
            _EMB_MM_MAP = (
                ("mm_model", "mm_embedding_model"),
                ("mm_base_url", "mm_embedding_base_url"),
                ("mm_api_key", "mm_embedding_api_key"),
            )
            for src_key, flat_key in _EMB_MM_MAP:
                if src_key in sub and sub[src_key] is not None:
                    out[flat_key] = sub[src_key]
    return out


def _coerce_config_value(key: str, raw: Any, ann: Any) -> Any:
    """Coerce a raw YAML value to the Config field's annotated type.

    ``Config`` is annotated with plain builtin types (via ``from __future__
    import annotations`` the annotation arrives as a string, e.g. ``"int"``).
    We only handle the four scalar types the numeric-hyperparameter fields use
    (bool/int/float/str). Anything else is passed through unchanged. Returns
    ``_COERCE_FAILED`` when the value can't be converted.
    """
    tname = ann.__name__ if isinstance(ann, type) else str(ann or "").strip()
    # bool must be checked before int (bool is a subclass of int).
    if tname == "bool":
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, (int, float)):
            return bool(raw)
        if isinstance(raw, str):
            s = raw.strip().lower()
            if s in ("true", "1", "yes", "on", "y"):
                return True
            if s in ("false", "0", "no", "off", "n", ""):
                return False
        return _COERCE_FAILED
    if tname == "int":
        if isinstance(raw, bool):
            return int(raw)
        try:
            return int(raw)
        except (TypeError, ValueError):
            return _COERCE_FAILED
    if tname == "float":
        if isinstance(raw, bool):
            return float(raw)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return _COERCE_FAILED
    if tname == "str":
        return raw if isinstance(raw, str) else str(raw)
    # Unknown/unhandled annotation → pass through unchanged (old behavior).
    return raw


def _hermes_config() -> Dict[str, Any]:
    """Load the full Hermes config dict (best-effort; empty on failure)."""
    try:
        from hermes_cli.config import load_config
        return load_config() or {}
    except Exception as e:  # pragma: no cover - config load is environment-dependent
        log.warning("[multimodal] load_config failed (%s); using defaults", e)
        return {}


def _default_memory_db_name(*, session_id: str = "",
                            runtime_id: str = "") -> str:
    """Return a readable, collision-resistant per-runtime SQLite filename.

    The old second-resolution ``<timestamp>.sqlite`` name made two live
    sessions (and, before the ready barrier, two independently-built stacks)
    silently share one database.  Keep the sortable timestamp prefix used by
    the debug UI, but add a privacy-preserving session hash and a runtime nonce.
    Passing the same runtime id is deterministic so one backend can precompute
    the Config once and share it with every owned store.
    """
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    sid = (session_id or "").strip()
    session_token = (
        hashlib.sha256(sid.encode("utf-8", errors="replace")).hexdigest()[:10]
        if sid else "anon"
    )
    raw_run = (runtime_id or uuid.uuid4().hex).strip().lower()
    run_token = "".join(ch for ch in raw_run if ch.isalnum())[:12]
    if not run_token:
        run_token = uuid.uuid4().hex[:12]
    return f"{stamp}_{session_token}_{run_token}.sqlite"


def build_config(hermes_cfg: Optional[Dict[str, Any]] = None, *,
                 session_id: str = "", runtime_id: str = "") -> Config:
    """Build a multimodal :class:`Config` from Hermes' ``multimodal:`` section.

    Only numeric hyperparameters are taken from config; endpoints/models are left
    at their dataclass defaults and overridden at runtime by the client factory.
    """
    if hermes_cfg is None:
        hermes_cfg = _hermes_config()
    # New nested layout (model.<role> / settings.* / audio.*) is flattened back
    # onto the legacy flat field names the Config dataclass expects; a pre-v32
    # flat multimodal.* config still loads via the fallback inside flatten.
    mm = flatten_mm_config(hermes_cfg)
    # One-way compatibility for profiles created before SearchFactStore was
    # named explicitly.  The new key wins when both are present; otherwise the
    # old facts_max value becomes the cache capacity instead of being silently
    # ignored behind Config.search_facts_max's default.
    if "search_facts_max" not in mm and mm.get("facts_max") is not None:
        mm = dict(mm)
        mm["search_facts_max"] = mm["facts_max"]

    cfg = Config()
    _field_types = {f.name: f.type for f in fields(Config)}
    for key in _NUMERIC_KEYS:
        if key in mm and mm[key] is not None:
            raw = mm[key]
            coerced = _coerce_config_value(key, raw, _field_types.get(key))
            if coerced is _COERCE_FAILED:
                log.warning(
                    "[multimodal] config key %r=%r cannot be coerced to %s; "
                    "keeping dataclass default %r",
                    key, raw, _field_types.get(key), getattr(cfg, key, None))
                continue
            setattr(cfg, key, coerced)
    # Dedicated MemoryWriter backend (optional; default reuses the main model).
    # These are string settings, kept separate from the numeric hyperparameters.
    for key in (_MEMORY_BACKEND_KEYS + _WORKER_BACKEND_KEYS
                + _MONITOR_BACKEND_KEYS
                + _REVIEWER_BACKEND_KEYS
                + _RECALL_BACKEND_KEYS
                + _RECALL_VERIFY_BACKEND_KEYS
                + _EMBEDDING_BACKEND_KEYS + _OCR_BACKEND_KEYS
                + _ANYSEARCH_KEYS):
        if key in mm and mm[key] is not None:
            setattr(cfg, key, mm[key])

    # Project-local paths: the dataclass defaults are from the original
    # streaming_demo environment (relative ours_results/, an /mnt/... tool path).
    # Re-root them to the Hermes home so they're writable cross-platform.
    try:
        from hermes_constants import get_hermes_home
        mm_dir = get_hermes_home() / "multimodal"
        mm_dir.mkdir(parents=True, exist_ok=True)
        # history log: allow explicit override, else Hermes-home location
        hist = mm.get("history_log_path")
        cfg.history_log_path = hist or str(mm_dir / "history.jsonl")
        # memory sqlite: explicit override, else a per-runtime file under
        # memories/multimodal. The timestamp remains human-sortable while the
        # session hash + runtime nonce prevent same-second cross-session
        # collisions. The live file is still never renamed; the first macro's
        # summary lives in meta.summary.
        db = mm.get("mem_db_path")
        if db:
            cfg.mem_db_path = db
        else:
            mem_root = get_hermes_home() / "memories" / "multimodal"
            mem_root.mkdir(parents=True, exist_ok=True)
            cfg.mem_db_path = str(mem_root / _default_memory_db_name(
                session_id=session_id, runtime_id=runtime_id))
    except Exception as e:  # pragma: no cover - fall back to dataclass defaults
        log.debug("[multimodal] path re-root skipped: %s", e)
    # Optional external image-search module. Nothing is bundled — this is a pure
    # extension point, so an unset multimodal.search_tool_path leaves it empty and
    # the image_search_* tools report "not configured" (they are deprecated and no
    # longer declared in the system prompt anyway, so the LLM won't dispatch them).
    explicit = mm.get("search_tool_path")
    cfg.search_tool_path = str(explicit).strip() if explicit else ""
    # enable_search gates the whole SearchWorker, and text_search runs on the
    # AnySearch backend, which needs no local module. So it must NOT be ANDed with
    # search_tool_path — doing that would silently kill deep-research web search
    # for everyone who hasn't wired up an image-search module.
    enable = mm.get("enable_search", True)
    cfg.enable_search = bool(enable)

    # WatcherAgent / Memory workers call the LLM with cfg.model. By default that
    # should FOLLOW the main agent's resolved model (set at engine/backend build
    # time via main_client() → cfg.model). An explicit multimodal.worker_model
    # overrides that: when set, workers use it verbatim and the build-time
    # follow is skipped. cfg._worker_model_explicit records whether the user
    # pinned it, so the engine's build-time sync knows not to overwrite.
    # model.watcher.model flattens to "watcher_model" (role prefix renamed
    # worker_→watcher_); accept legacy "worker_model" as a fallback so a pre-v33
    # flat config still pins the model.
    _wm = (mm.get("watcher_model") or mm.get("worker_model") or "").strip()
    if _wm:
        cfg.model = _wm
    try:
        cfg._worker_model_explicit = bool(_wm)
    except Exception:
        pass
    return cfg


def multimodal_enabled(hermes_cfg: Optional[Dict[str, Any]] = None) -> bool:
    if hermes_cfg is None:
        hermes_cfg = _hermes_config()
    # ``enabled`` moved into settings: (v32); fall back to legacy multimodal.enabled.
    settings = hermes_cfg.get("settings")
    if isinstance(settings, dict) and "enabled" in settings:
        return bool(settings.get("enabled", True))
    mm = hermes_cfg.get("multimodal") or {}
    return bool(mm.get("enabled", True))


# --------------------------------------------------------------------------- #
# Kimi/Moonshot parameter adapter for the multimodal DIRECT-LLM callers.
#
# The multimodal workers + monitor call chat.completions.create() directly (they
# bypass the main agent's transport, which already sanitizes Moonshot requests).
# When those clients point at a Kimi/Moonshot endpoint, two hardcoded params
# break every call:
#   * temperature: Kimi allows ONLY temperature=1 ("invalid temperature: only 1
#     is allowed for this model" → HTTP 400). Workers/monitor hardcode 0.0-0.4.
#   * enable_thinking: Kimi ignores it; the toggle is `thinking:{"type":...}`.
# We wrap create() once at client-build time so ALL downstream calls are fixed
# transparently, without editing every call site. Non-Kimi clients are returned
# unchanged (wrapper only rewrites when the per-call model is a Moonshot slug).
# --------------------------------------------------------------------------- #
def kimi_fix_create_kwargs(kwargs: dict) -> dict:
    """Coerce chat.completions.create kwargs to be Kimi/Moonshot-compatible.

    No-op unless kwargs['model'] is a Moonshot slug. Mutates + returns kwargs.
    Standalone (not only inside the wrapper) so call sites that use a SHARED
    client (e.g. the monitor's agent.client, which must not be mutated globally)
    can fix params per-call instead.
    """
    try:
        from agent.moonshot_schema import (
            is_moonshot_model, is_thinking_only_moonshot_model,
            sanitize_moonshot_tools)
    except Exception:
        return kwargs
    model = kwargs.get("model")
    if not is_moonshot_model(model):
        return kwargs
    # Sanitize tool schemas to Moonshot's stricter JSON-Schema subset (missing
    # `type`, anyOf-parent-type) so tool-carrying worker/monitor calls don't 400
    # with "not a valid moonshot flavored json schema". No-op when no tools /
    # already compliant. (The main agent path sanitizes at its transport layer;
    # these direct chat.completions callers didn't until now — defect D8.)
    tools = kwargs.get("tools")
    if tools:
        try:
            kwargs["tools"] = sanitize_moonshot_tools(tools)
        except Exception:
            pass
    # ★ Some Kimi models (k2.7-code) are thinking-ONLY: they REJECT
    #   thinking:{type:disabled} with HTTP 400. For those we must never emit a
    #   disabled flag — omit the thinking key entirely (default = enabled).
    thinking_only = is_thinking_only_moonshot_model(model)
    # Translate enable_thinking → thinking:{type}. Kimi ignores enable_thinking.
    eb = kwargs.get("extra_body")
    if isinstance(eb, dict):
        think = None
        if "enable_thinking" in eb:
            think = bool(eb.pop("enable_thinking"))
        ctk = eb.get("chat_template_kwargs")
        if isinstance(ctk, dict) and "enable_thinking" in ctk:
            if think is None:
                think = bool(ctk.get("enable_thinking"))
            ctk.pop("enable_thinking", None)
            if not ctk:
                eb.pop("chat_template_kwargs", None)
        eb.pop("top_k", None)
        if thinking_only:
            # Thinking-only model: never send a thinking flag (disabled → 400,
            # enabled is the default anyway). Strip any that a caller set.
            eb.pop("thinking", None)
        elif think is not None and "thinking" not in eb:
            eb["thinking"] = {"type": "enabled" if think else "disabled"}
        if not eb:
            kwargs.pop("extra_body", None)
    return kwargs


def _detect_thinking_provider(provider: str, base_url: str, model: str) -> str:
    """Classify a submodule endpoint into a thinking-param dialect.

    The multimodal workers/monitor hardcode the vLLM/self-hosted Qwen shape
    ``extra_body.chat_template_kwargs.enable_thinking``. Real hosted providers
    expect different wire shapes:

      * ``deepseek``   → ``extra_body.thinking:{"type": enabled|disabled}``
                         (+ round-trip contract; see the deepseek profile).
      * ``dashscope``  → ``extra_body.enable_thinking`` (top-level, per Alibaba
                         Model Studio OpenAI-compat docs — NOT under
                         chat_template_kwargs).
      * ``moonshot``   → handled separately by kimi_fix_create_kwargs.
      * ``openai``     → strip vLLM-only keys.  This also covers GPT-named
                         models behind a generic ``custom`` proxy.
      * ``vllm``       → keep chat_template_kwargs.enable_thinking (self-hosted
                         Qwen / generic OpenAI-compat default).

    Detection is provider-first, then base_url heuristics for the ``custom``/
    ``openai`` generic providers.
    """
    p = (provider or "").strip().lower()
    if p in ("deepseek",):
        return "deepseek"
    if p in ("alibaba", "dashscope", "qwen-dashscope", "alibaba-cloud"):
        return "dashscope"
    if p in ("moonshot", "kimi"):
        return "moonshot"
    # Generic provider ("", custom, openai): sniff the base_url.  URL-specific
    # dialects win over the generic provider label (for example provider=openai
    # pointed at api.moonshot.ai).
    url = (base_url or "").strip().lower()
    if "deepseek" in url:
        return "deepseek"
    if "dashscope" in url or "aliyuncs" in url:
        return "dashscope"
    if "moonshot" in url:
        return "moonshot"
    if "openai" in url:
        return "openai"
    if p == "openai":
        return "openai"
    # A generic/custom gateway can proxy an OpenAI-family model without an
    # identifying hostname.  The multimodal workers historically treated every
    # such endpoint as vLLM and sent ``extra_body.top_k`` plus
    # ``chat_template_kwargs``.  Strict GPT-compatible proxies reject those
    # fields (the RecallAgent failure was: ``Unknown parameter: 'top_k'``).
    # Model identity is the only signal available for these private gateways,
    # and omitting non-standard knobs is the portable choice for GPT/o-series.
    model_l = (model or "").strip().lower()
    if re.match(r"^(?:gpt(?:-|$)|o[134](?:-|$))", model_l):
        return "openai"
    # Default: leave the hardcoded vLLM/self-hosted Qwen shape untouched.
    return "vllm"


def normalize_thinking_kwargs(kwargs: dict, dialect: str) -> dict:
    """Translate the workers' hardcoded ``chat_template_kwargs.enable_thinking``
    (and any pre-set ``enable_thinking``) into the target provider's native
    thinking wire shape. Mutates + returns kwargs.

    ``dialect`` comes from :func:`_detect_thinking_provider`. ``moonshot`` and
    ``vllm`` are no-ops here (Moonshot is handled by kimi_fix_create_kwargs; vLLM
    already uses the hardcoded shape). Non-thinking-capable models are still
    translated (a disabled/enabled flag is harmless), except DeepSeek V3 where we
    strip the flag entirely to avoid perturbing its wire format.
    """
    # ★ Kimi K2.6 MaaS (vLLM 部署, 非标准 chat_template) 特殊情况:
    #   不认 chat_template_kwargs.enable_thinking, 只认 chat_template_kwargs.thinking (布尔).
    #   实测: enable_thinking=false → thinking 依然 on; thinking=false → 立即生效, 3s→0.4s.
    #   dialect 会被判成 "vllm", 但 vllm 分支直接 return 不翻译 → memory writer 一直是
    #   thinking mode. 这里在 vllm 直返之前手动翻译 kimi-k2* 模型.
    if dialect == "vllm":
        _model_l = (kwargs.get("model") or "").strip().lower()
        if "kimi-k2" in _model_l:
            eb = kwargs.get("extra_body")
            if isinstance(eb, dict):
                ctk = eb.get("chat_template_kwargs")
                if isinstance(ctk, dict) and "enable_thinking" in ctk:
                    _think = bool(ctk.pop("enable_thinking"))
                    ctk["thinking"] = _think
                    if not ctk:
                        eb.pop("chat_template_kwargs", None)
                # 顶层 enable_thinking 也翻译一下
                if "enable_thinking" in eb:
                    _think = bool(eb.pop("enable_thinking"))
                    _ctk = eb.get("chat_template_kwargs") or {}
                    _ctk.setdefault("thinking", _think)
                    eb["chat_template_kwargs"] = _ctk
                eb.pop("top_k", None)
                if not eb:
                    kwargs.pop("extra_body", None)
            return kwargs
    if dialect in ("vllm", "moonshot"):
        return kwargs
    eb = kwargs.get("extra_body")
    if not isinstance(eb, dict):
        return kwargs
    # Extract the intended enable flag from either location.
    think = None
    if "enable_thinking" in eb:
        think = bool(eb.pop("enable_thinking"))
    ctk = eb.get("chat_template_kwargs")
    if isinstance(ctk, dict) and "enable_thinking" in ctk:
        if think is None:
            think = bool(ctk.get("enable_thinking"))
        ctk.pop("enable_thinking", None)
        if not ctk:
            eb.pop("chat_template_kwargs", None)
    # vLLM-only sampling knob DashScope/DeepSeek OpenAI-compat don't accept.
    eb.pop("top_k", None)
    if think is None:
        # Nothing to translate; just cleaned vLLM-only keys.
        if not eb:
            kwargs.pop("extra_body", None)
        return kwargs
    if dialect == "openai":
        # GPT/OpenAI-compatible endpoints have no portable request flag for the
        # vLLM chat-template toggle.  Removing it preserves the model/provider
        # defaults and, crucially, avoids turning a Recall transport error into
        # a false "memory not found" result.
        pass
    elif dialect == "deepseek":
        model = (kwargs.get("model") or "").strip().lower()
        # DeepSeek V3 (deepseek-chat) has no thinking mode — don't perturb it.
        is_v3 = model.startswith("deepseek-v3") or model == "deepseek-chat"
        if not is_v3:
            eb["thinking"] = {"type": "enabled" if think else "disabled"}
    elif dialect == "dashscope":
        # Alibaba Model Studio OpenAI-compat: top-level enable_thinking in body.
        eb["enable_thinking"] = think
    if not eb:
        kwargs.pop("extra_body", None)
    return kwargs


def wrap_kimi_client(client: Any, *, dialect: str = "moonshot") -> Any:
    """Wrap an OpenAI-compatible client so chat.completions.create() coerces
    provider-incompatible params (Kimi temperature/thinking + cross-provider
    thinking-flag translation). Idempotent; safe for both sync and async
    clients. Only use on OWNED clients (mutates the client in place) — never on a
    client shared with the main agent; use kimi_fix_create_kwargs per-call for
    those. Returns the same client (mutated) for chaining.

    ``dialect`` (from :func:`_detect_thinking_provider`) selects the thinking
    wire shape for the endpoint. ``moonshot``/``vllm`` skip the extra
    translation; ``deepseek``/``dashscope`` rewrite the workers' hardcoded
    ``chat_template_kwargs.enable_thinking`` into the provider's native shape.
    kimi_fix_create_kwargs still runs unconditionally (no-op unless the per-call
    model is a Moonshot slug).
    """
    if client is None:
        return client
    try:
        completions = client.chat.completions
    except Exception:
        return client
    if getattr(completions, "_kimi_wrapped", False):
        return client
    orig_create = completions.create

    def wrapped_create(*args, **kwargs):
        kwargs = normalize_thinking_kwargs(kwargs, dialect)
        return orig_create(*args, **kimi_fix_create_kwargs(kwargs))

    try:
        completions.create = wrapped_create  # type: ignore[attr-defined]
        completions._kimi_wrapped = True      # type: ignore[attr-defined]
    except Exception:
        # Some client objects are frozen; fall back to the unwrapped client
        # (callers that pinned temperature will still 400, but we never crash).
        log.warning("[multimodal] could not wrap client for param adaptation")
    return client


def _submodule_http_client(provider: str, model: str):
    """Build an httpx.AsyncClient whose request timeout follows the MAIN agent's
    config — resolved via ``get_provider_request_timeout``, which honors (in
    order) the per-model ``timeout_seconds``, the per-provider
    ``request_timeout_seconds``, then the universal ``llm_timeout_seconds``
    fallback shared by the main agent and every submodule.

    Returns ``None`` when no timeout is configured, so the OpenAI SDK applies its
    own default exactly like the main agent does (which passes no http_client in
    that case). This keeps multimodal submodules (recall/monitor/worker/memory)
    byte-for-byte aligned with the main agent instead of hardcoding a value.
    """
    try:
        from hermes_cli.timeouts import get_provider_request_timeout
        secs = get_provider_request_timeout((provider or "").strip(),
                                            (model or "").strip() or None)
    except Exception:
        secs = None
    if not secs:
        return None
    import httpx
    return httpx.AsyncClient(timeout=httpx.Timeout(float(secs), connect=10.0))


def _needs_dual_auth_header(base_url: str) -> bool:
    """Whether ``base_url`` points at a gateway that wants the credential in BOTH
    the ``Authorization: Bearer`` header and a separate ``api-key`` header.

    Some enterprise LLM gateways reject Bearer-only requests with 401. Because the
    hostnames are deployment-specific, the match list is not hardcoded: set
    ``MM_DUAL_AUTH_URL_PATTERNS`` to a comma-separated list of case-insensitive
    substrings (e.g. ``gateway.example.com,/tenant-``) to opt those endpoints in.
    Empty/unset (the default) disables URL-based detection entirely; the monitor
    submodule can still opt in explicitly via
    ``multimodal.monitor_send_api_key_header``.
    """
    patterns = [p.strip().lower()
                for p in os.environ.get("MM_DUAL_AUTH_URL_PATTERNS", "").split(",")
                if p.strip()]
    if not patterns:
        return False
    url_lower = (base_url or "").lower()
    return any(p in url_lower for p in patterns)



def build_submodule_client(
    *, provider: str, base_url: str, api_key: str, model: str,
    resolve_main: "Callable[[], Tuple[Any, str]]",
    label: str,
    send_api_key_header: bool = False,
) -> Tuple[Any, str]:
    """Unified per-submodule LLM endpoint resolver — the ONE contract shared by
    the WatcherAgent worker and the MonitorAgent (memory has its own richer
    variant with Gemini-native client, kept separate).

    Contract (identical across submodules so config stays consistent):
      * base_url EMPTY  → follow the main agent: return resolve_main() unchanged
        (a bare `<sub>_model` just renames the model on the main endpoint via
        build_config → cfg.model, handled by the caller).
      * base_url SET, provider in ("", "custom", "openai") → an OpenAI-compatible
        endpoint: AsyncOpenAI(base_url, api_key), owned → wrapped for Kimi.
      * base_url SET, provider == "gemini" → not supported for these submodules
        yet (they call chat.completions directly); raise a clear error so a
        misconfig is loud instead of silently broken.
      * send_api_key_header=True is honored only for label="monitor": reuse the
        resolved bearer credential as ``default_headers["api-key"]`` for MaaS
        gateways that require both forms of authentication.

    Returns (client, model). ``model`` falls back to the main resolved name when
    the caller didn't pin one.
    """
    provider = (provider or "").strip().lower()
    base_url = (base_url or "").strip()
    model = (model or "").strip()
    if not base_url:
        # Follow main agent (no dedicated endpoint). Do NOT wrap: the returned
        # client may be shared with the main/aux stack.
        return resolve_main()
    if provider == "gemini":
        raise ValueError(
            f"multimodal.{label}_provider=gemini is not supported (the {label} "
            f"path calls chat.completions directly). Use an OpenAI-compatible "
            f"endpoint (provider custom/openai) or route via memory_provider.")
    # OpenAI-compatible ("", "custom", "openai", or any other → treat as OAI).
    from openai import AsyncOpenAI
    if not model:
        try:
            model = resolve_main()[1]
        except Exception:
            pass
    if _prefers_messages_transport(model, base_url):
        client = MessagesChatCompletionsClient(
            base_url=base_url,
            api_key=(api_key or "").strip() or "EMPTY",
            model=model,
        )
        client._hermes_submodule_owned = True  # type: ignore[attr-defined]
        log.info("[multimodal] %s client: messages endpoint=%s model=%s provider=%s",
                 label, client.endpoint, model, provider or "(messages)")
        return client, model
    # ★ LLM request timeout: unify with the main agent. Read the same
    #   providers.<id>.request_timeout_seconds (or model-level timeout_seconds)
    #   the main chat path uses; None → let the SDK apply its own default (do NOT
    #   invent a submodule-specific value). "主 Agent 有它就有, 主 Agent 没有它也不自作主张."
    _http_client = _submodule_http_client(provider, model)
    resolved_api_key = (api_key or "").strip() or "EMPTY"
    client_kwargs: Dict[str, Any] = {
        "base_url": base_url,
        "api_key": resolved_api_key,
    }
    if _http_client is not None:
        client_kwargs["http_client"] = _http_client
    # Some enterprise gateways require the credential in both auth locations.
    # Monitor path is explicit opt-in via config; any other submodule pointed at
    # an endpoint matching MM_DUAL_AUTH_URL_PATTERNS auto-adds the api-key header.
    _dual_auth = _needs_dual_auth_header(base_url)
    if (label == "monitor" and send_api_key_header) or _dual_auth:
        client_kwargs["default_headers"] = {"api-key": resolved_api_key}
        if _dual_auth and label != "monitor":
            log.info("[multimodal] %s client: gateway auto-added api-key header", label)
    client = AsyncOpenAI(**client_kwargs)
    dialect = _detect_thinking_provider(provider, base_url, model)
    log.info("[multimodal] %s client: dedicated endpoint=%s model=%s provider=%s dialect=%s",
             label, base_url, model, provider or "(oai)", dialect)
    # Owned client → wrap for provider param adaptation (Kimi temperature/thinking
    # + cross-provider thinking-flag translation for DeepSeek/DashScope).  Keep
    # ownership on the concrete AsyncOpenAI instance: callers that receive the
    # main/aux shared client through the no-base-url branch must never close it,
    # while resident submodules can deterministically close a dedicated pool on
    # their owning event loop.
    client = wrap_kimi_client(client, dialect=dialect)
    client._hermes_submodule_owned = True  # type: ignore[attr-defined]
    return client, model


# --------------------------------------------------------------------------- #
# LLM client factory backed by Hermes provider/model resolution
# --------------------------------------------------------------------------- #
class HermesClientFactory(LLMClientFactory):
    """Resolve worker LLM clients through Hermes' auxiliary/vision client stack.

    All multimodal workers send video frames, so we prefer a vision-capable
    backend (``resolve_vision_provider_client``). When no dedicated vision
    backend resolves, we fall back to the text auxiliary client (the user's main
    model), which is correct for text/vision-capable main models.
    """

    def __init__(self, cfg: Optional[Config] = None):
        # cfg carries the optional dedicated memory-backend settings
        # (memory_provider / memory_base_url / memory_api_key / memory_model).
        self.cfg = cfg
        self._cached: Optional[Tuple[Any, str]] = None

    def _resolve(self) -> Tuple[Any, str]:
        if self._cached is not None:
            return self._cached
        client: Any = None
        model: str = ""
        try:
            from agent.auxiliary_client import resolve_vision_provider_client
            # provider=None (not "auto") so a configured auxiliary.vision.base_url
            # direct endpoint is honored; passing "auto" explicitly makes the
            # resolver skip the config base_url and only probe known aggregators.
            _provider, vclient, vmodel = resolve_vision_provider_client(
                async_mode=True)
            if vclient is not None:
                client, model = vclient, (vmodel or "")
                log.info("[multimodal] vision client resolved: provider=%s model=%s",
                         _provider, model)
        except Exception as e:
            log.warning("[multimodal] vision client resolution failed: %s", e)

        if client is None:
            try:
                from agent.auxiliary_client import get_async_text_auxiliary_client
                tclient, tmodel = get_async_text_auxiliary_client(task="multimodal")
                if tclient is not None:
                    client, model = tclient, (tmodel or "")
                    log.info("[multimodal] text auxiliary client resolved: model=%s",
                             model)
            except Exception as e:
                log.warning("[multimodal] text client resolution failed: %s", e)

        if client is None:
            raise RuntimeError(
                "multimodal: no LLM provider could be resolved from Argus config. "
                "Configure a model with `argus model` (a vision-capable model is "
                "recommended for the video stream).")
        self._cached = (client, model)
        return self._cached

    def main_client(self) -> Tuple[Any, str]:
        return self._resolve()

    def worker_client(self) -> Tuple[Any, str]:
        """Resolve the deep-analysis watcher client (Router/Search/Recall + deep
        query summary), via the unified per-submodule contract
        (model.watcher.{provider,base_url,api_key,model}):
          * watcher_base_url set → dedicated endpoint (provider custom/openai =
            OpenAI-compatible, owned+Kimi-wrapped).
          * empty → follow the main resolved Hermes model (a bare worker_model
            just renames the model on the main endpoint via build_config →
            cfg.model, resolved below).
        Returns (async_client, model).
        """
        cfg = self.cfg
        # worker_model flows into cfg.model via build_config; treat the dataclass
        # default "qwen3.5" as "not a real pin" so the helper falls back to main.
        model = (getattr(cfg, "model", "") or "").strip() if cfg else ""
        if model == "qwen3.5":
            model = ""
        return build_submodule_client(
            provider=getattr(cfg, "watcher_provider", "") if cfg else "",
            base_url=getattr(cfg, "watcher_base_url", "") if cfg else "",
            api_key=getattr(cfg, "watcher_api_key", "") if cfg else "",
            model=model,
            resolve_main=self._resolve,
            label="watcher",
        )

    def monitor_client(self) -> Tuple[Any, str]:
        """Resolve the MonitorAgent client (always-on video SPEAK/SILENT + event
        merge), via the SAME unified contract as worker_client
        (multimodal.monitor_{provider,base_url,api_key,model}):
          * monitor_base_url set → dedicated endpoint.
          * empty → follow the main resolved Hermes model.
        Returns (async_client, model). Replaces the ad-hoc client construction
        that used to live in tui_gateway/server.py so worker & monitor share one
        provider-aware path.
        """
        cfg = self.cfg
        return build_submodule_client(
            provider=getattr(cfg, "monitor_provider", "") if cfg else "",
            base_url=getattr(cfg, "monitor_base_url", "") if cfg else "",
            api_key=getattr(cfg, "monitor_api_key", "") if cfg else "",
            model=(getattr(cfg, "monitor_model", "") if cfg else ""),
            resolve_main=self._resolve,
            label="monitor",
            send_api_key_header=(
                bool(getattr(cfg, "monitor_send_api_key_header", False))
                if cfg else False
            ),
        )

    def recall_client(self) -> Tuple[Any, str]:
        """Resolve the RecallAgent client (multimodal memory-recall sub-agent).

        Dedicated endpoint (post-v33): if model.memory.recall.base_url is set,
        the recall sub-agent uses its OWN backend (built by
        _memory_client_from_config, same path as reviewer/verify). Otherwise it
        falls back to model.memory (same client as MemoryWriter, historical
        behavior).

        Typical split:
          * memory writer @ K2.6 MaaS  (无审查, non-thinking 快, 10s 高频便宜)
          * recall decide/distill @ Luna @ /v1/chat/completions
            (多图+文视觉, 命中率优先, tools 不需要)

        Returns (client_or_memory_wrapper, model). RecallAgent supports both a
        raw AsyncOpenAI client and a MemoryLLMClient wrapper (e.g.
        MessagesMemoryClient for Luna's /v1/messages proxy)."""
        cfg = self.cfg
        recall_base = str(getattr(cfg, "recall_base_url", "") or "").strip() if cfg else ""
        if recall_base:
            client = self._memory_client_from_config(
                provider=str(getattr(cfg, "recall_provider", "") or ""),
                base_url=recall_base,
                api_key=str(getattr(cfg, "recall_api_key", "") or ""),
                model=str(getattr(cfg, "recall_model", "") or ""),
                role_label="recall",
            )
            return client, getattr(client, "model", "") or str(
                getattr(cfg, "recall_model", "") or "")
        # Fallback: share MemoryWriter's client/model (historical v33 behavior).
        mc = self.memory_client(None)
        model = getattr(mc, "model", "") or ""
        return mc, model

    def recall_verify_client(
        self, *, recall_client: Any = None, recall_model: str = "",
    ) -> Tuple[Any, str]:
        """Resolve the visual verifier endpoint, falling back to Recall.

        The verifier is the final correctness gate for exact text and IDs. A
        dedicated endpoint keeps it independent from long Writer requests while
        preserving the old single-client behavior when no override is set.
        """
        cfg = self.cfg
        if not cfg or not str(getattr(
                cfg, "recall_verify_base_url", "") or "").strip():
            if recall_client is not None:
                return recall_client, recall_model
            return self.recall_client()
        provider = (
            str(getattr(cfg, "recall_verify_provider", "") or "").strip()
            or str(getattr(cfg, "memory_provider", "") or "").strip()
        )
        api_key = (
            str(getattr(cfg, "recall_verify_api_key", "") or "").strip()
            or str(getattr(cfg, "memory_api_key", "") or "").strip()
        )
        model = (
            str(getattr(cfg, "recall_verify_model", "") or "").strip()
            or str(recall_model or "").strip()
            or str(getattr(cfg, "memory_model", "") or "").strip()
        )
        client = self._memory_client_from_config(
            provider=provider,
            base_url=str(getattr(
                cfg, "recall_verify_base_url", "") or "").strip(),
            api_key=api_key,
            model=model,
            role_label="recall.verify",
        )
        return client, getattr(client, "model", "") or model

    def memory_client(self, shared_main: Any) -> MemoryLLMClient:
        """Resolve the MemoryWriter/Reviewer backend.

        Optional dedicated backend (when the multimodal config sets these):
          * memory_provider == "gemini" → GeminiMemoryClient against
            memory_base_url (a runway/Gemini generateContent endpoint) with
            memory_api_key + memory_model. Vision/JSON-strong; ported from the
            original streaming_demo MemoryWriter design.
          * memory_provider == "openai"/"custom" (or any non-gemini value with a
            base_url) → a separate OpenAI-compatible endpoint
            (memory_base_url/api_key/model) for memory writing.
        Otherwise (default, empty base_url) → reuse the main resolved Hermes model.
        """
        cfg = self.cfg
        return self._memory_client_from_config(
            provider=(getattr(cfg, "memory_provider", "") or ""),
            base_url=(getattr(cfg, "memory_base_url", "") or ""),
            api_key=(getattr(cfg, "memory_api_key", "") or ""),
            model=(getattr(cfg, "memory_model", "") or ""),
            role_label="memory",
        )

    def reviewer_client(self) -> MemoryLLMClient:
        """Resolve the first configured MemoryReviewer backend."""
        return self.reviewer_clients()[0]

    def reviewer_clients(self) -> List[MemoryLLMClient]:
        """Resolve the MemoryReviewer endpoint pool.

        If model.memory.reviewer.* is unset, keep the historical behavior and
        reuse the model.memory backend shape. Individual reviewer fields fall
        back to model.memory. ``base_urls`` enables one independent client per
        endpoint; ``base_url`` remains the single-endpoint compatibility form.
        """
        cfg = self.cfg
        if not cfg:
            return [self.memory_client(None)]
        has_override = any(
            bool(getattr(cfg, key, None))
            for key in (
                "reviewer_provider", "reviewer_base_url", "reviewer_base_urls",
                "reviewer_api_key", "reviewer_model",
            )
        )
        if not has_override:
            return [self.memory_client(None)]

        provider = (
            (getattr(cfg, "reviewer_provider", "") or "")
            or (getattr(cfg, "memory_provider", "") or "")
        )
        api_key = (
            (getattr(cfg, "reviewer_api_key", "") or "")
            or (getattr(cfg, "memory_api_key", "") or "")
        )
        model = (
            (getattr(cfg, "reviewer_model", "") or "")
            or (getattr(cfg, "memory_model", "") or "")
        )
        urls_raw = getattr(cfg, "reviewer_base_urls", None) or []
        if isinstance(urls_raw, str):
            urls_raw = [urls_raw]
        urls: List[str] = []
        for value in urls_raw:
            url = str(value or "").strip()
            if url and url not in urls:
                urls.append(url)
        legacy_url = str(getattr(cfg, "reviewer_base_url", "") or "").strip()
        if not urls and legacy_url:
            urls.append(legacy_url)
        if not urls:
            memory_url = str(getattr(cfg, "memory_base_url", "") or "").strip()
            if memory_url:
                urls.append(memory_url)
        if not urls:
            return [self.memory_client(None)]
        return [
            self._memory_client_from_config(
                provider=provider,
                base_url=url,
                api_key=api_key,
                model=model,
                role_label=f"reviewer[{idx}]",
            )
            for idx, url in enumerate(urls)
        ]

    def _memory_client_from_config(
        self, *, provider: str, base_url: str, api_key: str,
        model: str, role_label: str,
    ) -> MemoryLLMClient:
        cfg = self.cfg
        provider = (provider or "").strip().lower()
        if provider in ("gemini", "gemini_omni"):
            if not base_url or not api_key:
                raise ValueError(
                    f"multimodal.{role_label}_provider=gemini[_omni] requires "
                    f"{role_label}_base_url + {role_label}_api_key")
            try:
                import aiohttp  # noqa: F401
            except ImportError as e:
                raise RuntimeError(
                    "multimodal.memory_provider=gemini needs the 'aiohttp' package "
                    "(pip install aiohttp).") from e
            # (v33: gemini_omni provider 已删 — omni 原始音频路径整体移除。gemini
            #  provider 走纯 GeminiMemoryClient; 音频记忆统一走外部 ASR 字幕。)
            log.info("[multimodal] %s backend: gemini endpoint=%s model=%s",
                     role_label, base_url, model or "gemini-3.5-flash")
            return GeminiMemoryClient(
                api_url=base_url.strip(),
                api_key=api_key.strip(),
                model=(model or "gemini-3.5-flash").strip())
        # ★ OpenAI 兼容 (openai / custom / 其它非 gemini), 且给了专用 base_url:
        #   按主 Agent 标准对齐 —— 走跟 monitor/watcher (build_submodule_client) 同一套
        #   dashscope/moonshot dialect 检测 + wrap_kimi_client. 这样 qwen3-omni
        #   (阿里云百炼 compatible-mode, provider=custom) 等能正确适配 thinking/温度线格式,
        #   且 OpenAIMemoryClient 会把 audio_url → input_audio 让 omni 吃原始音频.
        if cfg and (base_url or "").strip():
            from openai import AsyncOpenAI
            mem_model = (model or "").strip()
            if not mem_model:
                # Fall back to the main resolved model, which may be a VISION
                # model (see _resolve prefers resolve_vision_provider_client).
                mem_model = self._resolve()[1]
                log.warning(
                    "[multimodal] %s provider=%r but %s_model is empty; "
                    "falling back to the main resolved model %r against endpoint %s. "
                    "Set model.memory%s.model to a model served by that endpoint.",
                    role_label, provider or "(oai)", role_label, mem_model, base_url,
                    ".reviewer" if role_label == "reviewer" else "")
            if _prefers_messages_transport(mem_model, base_url):
                log.info("[multimodal] %s backend: messages endpoint=%s model=%s",
                         role_label, base_url, mem_model)
                return MessagesMemoryClient(
                    base_url=base_url.strip(),
                    api_key=(api_key or "EMPTY"),
                    model=mem_model)
            # ★ LLM request timeout: unify with the main agent (same
            #   providers.<id>.request_timeout_seconds / model timeout_seconds).
            #   None → SDK default, matching how the main agent builds its client.
            _http_client = _submodule_http_client(
                provider, mem_model)
            _mem_key = (api_key or "EMPTY")
            _client_kwargs: Dict[str, Any] = {
                "base_url": base_url.strip(),
                "api_key": _mem_key,
            }
            if _http_client is not None:
                _client_kwargs["http_client"] = _http_client
            # ★ 部分企业网关要求 Bearer + api-key 双 header 认证.
            #   build_submodule_client 里的 send_api_key_header 只对
            #   label='monitor' 生效, memory 段完全被忽略 → 401.
            #   这里按 URL 自动补 api-key header (匹配列表见
            #   MM_DUAL_AUTH_URL_PATTERNS, 默认关闭, config 无需 opt-in).
            if _needs_dual_auth_header(base_url):
                _client_kwargs["default_headers"] = {"api-key": _mem_key.strip()}
                log.info("[multimodal] %s backend: dual-auth gateway detected, adding api-key header", role_label)
            mem_oai = AsyncOpenAI(**_client_kwargs)
            # dialect 检测 + wrap: 跟 build_submodule_client 完全同款 (主 Agent 标准).
            mem_dialect = _detect_thinking_provider(
                provider, base_url, mem_model)
            mem_oai = wrap_kimi_client(mem_oai, dialect=mem_dialect)
            log.info("[multimodal] %s backend: %s endpoint=%s model=%s dialect=%s",
                     role_label, provider or "(oai)", base_url,
                     mem_model or "(main)", mem_dialect)
            return OpenAIMemoryClient(mem_oai, model=mem_model, owned=True)
        # Default: reuse the main resolved Hermes model.
        client, model = self._resolve()
        return OpenAIMemoryClient(client, model=model)


def build_speech_factory(hermes_cfg: Optional[Dict[str, Any]] = None,
                         *, cfg: Optional[Config] = None) -> SpeechFactory:
    """Return the ASR/TTS backend factory.

    Default = local HTTP placeholder (:class:`LocalSpeechFactory`). To plug in
    cloud ASR/TTS, implement a :class:`SpeechFactory` in
    :mod:`agent.multimodal.speech` and return it here based on config.
    """
    if cfg is None:
        cfg = build_config(hermes_cfg)
    try:
        from .speech import build_cloud_speech_factory
        factory = build_cloud_speech_factory(hermes_cfg, cfg=cfg)
        if factory is not None:
            return factory
    except Exception as e:  # pragma: no cover - optional
        log.debug("[multimodal] cloud speech factory not active: %s", e)
    return LocalSpeechFactory(cfg)
