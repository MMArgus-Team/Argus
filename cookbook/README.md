# Argus Cookbook: A Multimodal Agent That Sees, Hears, Thinks, and Speaks

> An engineering narrative documenting the four-layer architecture of see, hear, think, and speak.

This is the **design handbook** for the Argus multimodal subsystem (a real-time video Agent built on Hermes). It does not cover installation or configuration (see the project root [README](../README.md) for that). Instead, it explains **how this "watch-and-chat" system is architected, why it is built that way, and what breaks if you do it differently**.

The book is organized around four capability layers — **see, hear, think, speak** — plus one chapter of cross-cutting engineering lessons. Each chapter tries to: explain **why** first, then land on **where the code lives and how it works** (references use `file:line` and are clickable). Every design principle and every "pitfall we hit" comes from real project conversations, logs, and user feedback.

## Table of Contents

| Chapter | Topic | One-liner |
| --- | --- | --- |
| [Prologue](00-序章-一个会看会听会想会说的Agent.md) | Big picture & layering philosophy | Why the system is split into four layers — see / hear / think / speak — and the lifecycle of one video frame |
| [Ch. 1 · See](01-看-让Agent拥有眼睛.md) | Visual perception | FrameBuffer, dHash ingress dedup, scene-adaptive thresholds, QueryWorker question-time frames, watcher TTL+frame dual gate, OCR reflow + window-text bridge |
| [Ch. 2 · Hear](02-听-让Agent拥有耳朵.md) | Speech input | Streaming ASR protocol, dead-socket self-healing, local intent model, barge-in, audio/video time-base alignment |
| [Ch. 3 · Think](03-想-让Agent拥有大脑.md) | Reasoning & memory | Three-layer memory, ReAct deep research, event-monitor hooks into the main Agent, system-prompt orchestration, empty-image 400 |
| [Ch. 4 · Speak](04-说-让Agent开口说话.md) | Speech output | VoiceAgent, `_flush_to_tts` as the single gate, per-segment TTS, real interruption, warm-tone post-processing |
| [Ch. 5 · Engineering](05-贯穿全局的工程经验.md) | Cross-cutting | Long-session performance, frontend/backend state consistency, config architecture, eval pipeline, desktop UX traps, prompt i18n, eleven design principles |

## Suggested Reading Order

- **Quick global mental model**: Read the [Prologue](00-序章-一个会看会听会想会说的Agent.md), especially the "lifecycle of one frame" data-flow diagram.
- **Deep dive by capability**: The four core chapters stand alone, but start with the [Prologue](00-序章-一个会看会听会想会说的Agent.md) + [Ch. 1 · See](01-看-让Agent拥有眼睛.md) — `FrameBuffer` is shared ground every later chapter builds on.
- **Pitfalls and design taste only**: Jump straight to [Ch. 5 · Engineering](05-贯穿全局的工程经验.md).

## Architecture in One Sentence

The main Agent handles **user text and semantic routing only**; it does not passively receive live frames. One-shot requests about the present, the past, frame retrieval, or "on-screen entity + external facts" all go through `query_multimodal` to QueryWorker, which reads frames at question time and uses Recall/Search as needed. Watcher and Monitor handle ongoing deep research and event monitoring respectively. All roles share one `FrameBuffer` + `MemoryStore` — one perception substrate, many consumers.

## Tech Stack at a Glance

```
Backend          Python · asyncio · DashScope Realtime ASR/TTS
Frontend (web)   React · Vite · Tailwind
Frontend (desktop) Electron · nanostores · assistant-ui
Protocols        WebSocket (gateway) · JSON-RPC (tools) · PCM16 (audio)
AI               qwen3-asr-flash-realtime · qwen3-tts-flash-realtime · deepseek-v4-flash
                 + main routing model + vision-capable QueryWorker/memory/monitor/deep-research models (independently configurable)
```

---

*Last updated: August 2026*
