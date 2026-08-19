"""Hermes multimodal subsystem — always-on video/audio streaming agent.

The mainline ``/multimodal`` page drives the main Hermes agent plus three
resident modules sharing one FrameBuffer + MemoryStore:

  * :class:`agent.multimodal.memory_backend.MemoryBackend` —
    MemoryWriter / MemoryReviewer; writes/reviews layered memory whenever the
    video stream is active, decoupled from the main agent.
  * :class:`agent.multimodal.watcher_engine.WatcherAgent` —
    live-research ReAct worker + RecallAgent; invoked on-demand via the
    ``set_live_watcher`` tool for complex multimodal analysis
    (background Router-ReAct + proactive result bubble). Simple visual
    questions are answered by the main agent directly from its injected frames.
  * The independent MonitorAgent daemon (``tui_gateway.server`` +
    :mod:`agent.multimodal.monitor_agent`) — event monitoring (see-and-record
    to a per-monitor file, report on sight / per period / silent).

The worker classes (WatcherWorker, RecallAgent,
MemoryWriter, MemoryReviewer, ToolBox, STTClient, WhisperXClient, TTSClient, …)
are the implementation those modules wrap.
"""

from .core import Config, Frame, fmt_ts, new_response_id

__all__ = [
    "Config",
    "Frame",
    "fmt_ts",
    "new_response_id",
]
