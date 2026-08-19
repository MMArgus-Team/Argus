# -*- coding: utf-8 -*-
"""Backward-compatible facade for the multimodal engine.

The engine was originally a single 8000-line module. For maintainability it is
now split into focused submodules; this module re-exports the full public
surface so existing imports (``from agent.multimodal.core import ...``) keep
working unchanged.

Submodules:
  * :mod:`agent.multimodal._config`     — the :class:`Config` dataclass.
  * :mod:`agent.multimodal._memory`     — frames, conversation log, and the
    SQLite-backed layered :class:`MemoryStore` (+ event/entity dataclasses).
  * :mod:`agent.multimodal._workers`    — the 6 workers, prompts, ToolBox,
    memory LLM clients, and history recorder.
  * :mod:`agent.multimodal._dual_agent` — speech IO clients plus the ASR/TTS
    and LLM-client factories.

Prefer importing from the focused submodule when writing new code; this facade
exists for compatibility and for callers that just want "the engine".
"""

from __future__ import annotations

from ._config import Config, DEFAULT_SEARCH_TOOL_PATH

from ._memory import (
    Frame,
    FrameBuffer,
    StoredFrame,
    FrameStore,
    SearchFact,
    SearchFactSnapshot,
    SearchFactStore,
    SharedContext,
    ContextStore,
    SearchFact,
    SearchFactSnapshot,
    SearchFactStore,
    Turn,
    ConversationLog,
    MicroEvent,
    MacroEvent,
    SuperEvent,
    EntityState,
    RevisionRecord,
    Entity,
    Edge,
    MemoryStore,
    ScreenTextRecord,
    ScreenTextStore,
    ScreenTableRecord,
    ScreenTableStore,
    TaskStateRecord,
    TaskStateStore,
    frame_to_image_content,
    fmt_ts,
    new_response_id,
    extract_json_obj,
    extract_json_arr,
    estimate_msg_tokens,
    fmt_tok,
)

from ._workers import (
    HistoryRecorder,
    ToolBox,
    MemoryLLMClient,
    OpenAIMemoryClient,
    MessagesMemoryClient,
    GeminiMemoryClient,
    WriterResult,
    ScreenOCRWorker,
    MemoryWriter,
    ReviewerResult,
    MemoryReviewer,
    EventReviewer,
    EntityReviewer,
    EdgeReviewer,
    ReactStep,
    WatcherWorker,
    RecallResult,
    MemoryToolBox,
    RecallAgent,
    RecallWorker,   # backward-compat alias of RecallAgent
    StreamSink,
)

from ._dual_agent import (
    SentenceAccumulator,
    STTClient,
    WhisperXSegment,
    WhisperXClient,
    TTSClient,
    TTSChunkSink,
    append_audio_observation,
    SpeechFactory,
    LocalSpeechFactory,
    LLMClientFactory,
)

__all__ = [
    "Config",
    "DEFAULT_SEARCH_TOOL_PATH",
    "Frame",
    "FrameBuffer",
    "StoredFrame",
    "FrameStore",
    "SearchFact",
    "SearchFactSnapshot",
    "SearchFactStore",
    "SharedContext",
    "ContextStore",
    "SearchFact",
    "SearchFactSnapshot",
    "SearchFactStore",
    "Turn",
    "ConversationLog",
    "MicroEvent",
    "MacroEvent",
    "SuperEvent",
    "EntityState",
    "RevisionRecord",
    "Entity",
    "Edge",
    "MemoryStore",
    "ScreenTextRecord",
    "ScreenTextStore",
    "ScreenTableRecord",
    "ScreenTableStore",
    "TaskStateRecord",
    "TaskStateStore",
    "frame_to_image_content",
    "fmt_ts",
    "new_response_id",
    "extract_json_obj",
    "extract_json_arr",
    "estimate_msg_tokens",
    "fmt_tok",
    "HistoryRecorder",
    "ToolBox",
    "MemoryLLMClient",
    "OpenAIMemoryClient",
    "MessagesMemoryClient",
    "GeminiMemoryClient",
    "WriterResult",
    "ScreenOCRWorker",
    "MemoryWriter",
    "ReviewerResult",
    "MemoryReviewer",
    "EventReviewer",
    "EntityReviewer",
    "EdgeReviewer",
    "ReactStep",
    "WatcherWorker",
    "RecallResult",
    "MemoryToolBox",
    "RecallAgent",
    "RecallWorker",
    "StreamSink",
    "SentenceAccumulator",
    "STTClient",
    "WhisperXSegment",
    "WhisperXClient",
    "TTSClient",
    "TTSChunkSink",
    "append_audio_observation",
    "SpeechFactory",
    "LocalSpeechFactory",
    "LLMClientFactory",
]
