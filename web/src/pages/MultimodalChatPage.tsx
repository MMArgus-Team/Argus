/**
 * Multimodal Chat — camera/screen capture wired into one Hermes session.
 *
 * Phase 2 of the multimodal integration. Unlike the standalone MultimodalPage
 * (which talks the legacy /api/multimodal/ws DualAgent protocol), this page
 * drives the MAIN chat agent over the gateway JSON-RPC WebSocket:
 *
 *   - Owns one gateway session (session.create).
 *   - Streams camera/screen frames at ~2fps via the `multimodal.frame` RPC into
 *     that session agent's FrameBuffer.
 *   - Sends text questions via `prompt.submit`; the main agent routes one-shot
 *     visual questions through `query_multimodal`, whose QueryWorker reads the
 *     ask-time frames and chooses direct VQA, Recall, or Search as needed.
 *   - Renders the streamed answer from `message.start/delta/complete` events.
 *
 * Frames + questions share ONE session, so workers resolve the same buffer.
 */
import { lazy, memo, Suspense, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type FC, type UIEvent } from "react";
import { ArrowDown, Camera, ChevronDown, Database, Loader2, Mic, Monitor, NotebookPen, Play, Send, Square, Terminal, Volume2 } from "lucide-react";
import { ChatModelPill } from "@/components/ChatModelPill";
import { MonitorEvidenceStrip } from "@/components/MonitorEvidenceStrip";
import { ChatSessionContext } from "@/contexts/chat-session-context";
import { useSearchParams } from "react-router-dom";
import { Button } from "@nous-research/ui/ui/components/button";
import { Card, CardContent, CardHeader, CardTitle } from "@nous-research/ui/ui/components/card";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { GatewayClient } from "@/lib/gatewayClient";
import { SlashPopover, type SlashPopoverHandle } from "@/components/SlashPopover";
import { executeSlash } from "@/lib/slashExec";
import { HERMES_BASE_PATH, api } from "@/lib/api";
import { Markdown } from "@/components/Markdown";
import { MmReadinessBanner, type MmReadinessReport } from "@/components/MmReadinessBanner";
// Debug-only panel behind the header's "Memory" button: lazy so its ~170 kB of
// inspector UI stays out of the multimodal chunk until someone actually opens it.
const MemoryDebugPanel = lazy(() =>
  import("./MemoryDebugPanel").then((m) => ({ default: m.MemoryDebugPanel })));
import { useCliDrawer } from "@/contexts/cli-drawer-context";
import { usePageHeader } from "@/contexts/usePageHeader";
import { useProfileScope } from "@/contexts/useProfileScope";
import { preferLightCapture } from "@/lib/perf-hints";
import { formatElapsed, useElapsedSeconds } from "@/hooks/useElapsedSeconds";
import { visualCaptureProfile } from "@/lib/visual-capture-profile";
import { RECALL_NO_CLUES, isSynthSaw } from "@/lib/mm-sentinels";
import { normalizeMonitorEvidence, type MonitorEvidence } from "@/lib/monitor-evidence";
import {
  isEphemeralControl,
  monitorPresentation,
  removeEphemeralControlTurn,
  resolveRegistryPull,
  type MonitorRegistryItem,
} from "@/lib/monitor-control";
import {
  AsrTurnTransport,
  asrFinishFailureMessage,
  ownsAsrStopUi,
  type AsrStopDisposition,
  type AsrTurnMode,
} from "@/lib/asr-turn-transport";
import {
  VoiceDialogRecovery,
  type VoiceDialogActivation,
} from "@/lib/voice-dialog-recovery";
import { useI18n, useLocaleRevision } from "@/i18n";
import { translateNow } from "@/i18n/runtime";

type SourceType = "camera" | "screen" | null;

const QUERY_MULTIMODAL_TOOL_NAME = "query_multimodal";
const LEGACY_QUERY_MULTIMODAL_TOOL_NAME = "recall_multimodal_memory";

/** Match the model-visible tool used by new live QueryWorker handoffs. */
// eslint-disable-next-line react-refresh/only-export-components
export function isQueryMultimodalToolName(toolName: unknown): boolean {
  return toolName === QUERY_MULTIMODAL_TOOL_NAME;
}

/**
 * Match persisted QueryWorker handoffs. Older sessions keep their original
 * tool rows, so hydration accepts the pre-rename name without exposing it to
 * the new live event path.
 */
// eslint-disable-next-line react-refresh/only-export-components
export function isQueryMultimodalHistoryToolName(toolName: unknown): boolean {
  return isQueryMultimodalToolName(toolName)
    || toolName === LEGACY_QUERY_MULTIMODAL_TOOL_NAME;
}

interface ChatMsg {
  id: string;
  role: "user" | "assistant" | "system";
  text: string;
  streaming?: boolean;
  queued?: boolean;
  queuePosition?: number;
  voice?: boolean;
  // Inline progress entries (kind!=="chat") interleave with chat bubbles.
  kind?: "chat" | "tool" | "status" | "clarify";
  // clarify-entry fields (kind==="clarify"): a blocking clarify.request from a
  // tool (e.g. set_monitor silent-mode). Rendered inline in the waterfall as a
  // question with option buttons; answered via clarify.respond.
  clarifyReqId?: string;
  clarifyQuestion?: string;
  clarifyChoices?: string[];
  clarifyAnswer?: string;   // set once answered → buttons freeze, show choice
  // tool-entry fields
  toolId?: string;
  toolName?: string;
  toolCtx?: string;
  toolDone?: boolean;
  toolSummary?: string;
  toolDurationMs?: number;
  toolDetail?: string;   // result_text / inline_diff (expandable)
  // A failed call (`{"error": ...}` in the tool result). Kept separate from the
  // bubble-level `isError` so the row can be styled as a failure and say WHY
  // without the user having to expand it.
  toolIsError?: boolean;
  toolError?: string;
  // Structured, privacy-classified call args from tool.start's `args_fields`.
  // Ships in every mode (not just verbose) so a tool row always has something
  // to expand — a bare tool name can't tell the user what the model did.
  toolArgs?: ToolArgField[];
  recallTrace?: RecallTraceEntry[];
  recallFindings?: string;
  // QueryWorker owns the answer after query_multimodal hands off.
  // Its live trajectory is folded back into this same tool card by task id.
  workerTaskId?: string;
  workerStatus?: "running" | "complete" | "error" | "cancelled";
  workerProgress?: QueryWorkerProgressStep[];
  // assistant reasoning (kept separate from the answer text)
  reasoning?: string;
  // Auxiliary-LLM-summarised label of the latest reasoning segment
  // (~10 chars). If empty and streaming reasoning exists, the raw tail is
  // shown instead. Cleared once the answer body starts streaming.
  reasoningSummary?: string;
  // Streaming state machine flags (for the AssistantMessage "first line"):
  //   awaitingFirstDelta = true → 显示 "Waiting response…"
  //   hasReasoning       = true → 显示 "Thinking…" / reasoning 内容
  //   收到 message.delta 后, streaming 保持 true 但 hasReasoning 已经不重要
  //     (第一行整体消失, 让位给正文)
  awaitingFirstDelta?: boolean;
  hasReasoning?: boolean;
  // error styling
  isError?: boolean;
  // marks proactive bubbles from the multimodal RouterEngine
  deepResearch?: boolean;
  // Phase 10: concurrent-instance routing. proactive bubbles from
  // RouterEngine carry request_id; monitor SPEAK bubbles carry monitor_id.
  requestId?: string;
  monitorId?: string;
  monitorLabel?: string;   // user-facing short label (id is never shown)
  // Phase 13: which background worker produced this bubble. Drives a
  // distinct color so sub-agent output reads differently from both the
  // real user's turns and the main agent's replies.
  //   "monitor" → monitor daemon proactive alert (amber)
  //   "router"  → RouterEngine deep-research result (violet)
  //   "query_worker" → one-shot Recall/Search answer owner (cyan)
  //   undefined → main agent reply (or real user turn on role="user")
  subRole?: "monitor" | "router" | "watcher_report" | "query_worker";
  // Post-Clarify thread-back: shown in the center chat (not the left sub-window).
  threadback?: boolean;
  // Deep-research event name (the brief the user asked) — shown as the router
  // badge instead of the old "已回传主对话" text.
  brief?: string;
  // watcher_report: 本段画面时段区间 (mm:ss–mm:ss), 展示在头部行。
  deepRange?: string;
  // Client-local creation time (epoch ms) → absolute HH:MM:SS beside role name.
  createdAt?: number;
}

export interface QueryWorkerProgressStep {
  id: string;
  seq: number;
  ts: number;
  worker: string;
  phase: string;
  title: string;
  detail?: string;
  metrics?: string[];
  plannedTools?: RecallTraceToolCall[];
  toolResults?: RecallTraceToolObs[];
  frames?: MmTrajectoryFrame[];
  ocrRecords?: QueryWorkerOcrRecord[];
  ocrState?: "available" | "empty" | "skipped" | "timeout" | "error";
  ocrReason?: string;
  ocrRecordCount?: number;
  ocrElapsedSec?: number;
  taskRef?: string;
  callState?: "planned" | "called";
  terminal?: boolean;
  status?: "running" | "complete" | "error" | "cancelled";
}

export interface QueryWorkerOcrRecord {
  frameTs?: number;
  sourceType?: string;
  evidenceSource?: string;
  app?: string;
  windowTitle?: string;
  rawText: string;
}

const QUERY_WORKER_PROGRESS_LIMIT = 80;
const QUERY_WORKER_TASK_CACHE_LIMIT = 48;
const QUERY_WORKER_IMAGE_TASK_LIMIT = 4;
const QUERY_WORKER_IMAGE_CHAR_BUDGET = 4_000_000;
const QUERY_WORKER_OCR_RECORD_LIMIT = 3;
const QUERY_WORKER_OCR_TEXT_LIMIT = 1_800;
// 切换/恢复会话时向后端要的轨迹行数 (后端硬上限 2000)。见 fetchTrajectory 的注释:
// 拉满会在切换瞬间搬运一大堆随后就被 compactQueryWorkerTrajectory 丢掉的 base64 图。
const TRAJECTORY_RESUME_LIMIT = 200;

function frameImageChars(frame: MmTrajectoryFrame): number {
  return (typeof frame.jpeg_b64 === "string" ? frame.jpeg_b64.length : 0)
    + (typeof frame.thumb_b64 === "string" ? frame.thumb_b64.length : 0);
}

function withoutFrameImage(frame: MmTrajectoryFrame): MmTrajectoryFrame {
  if (frame.jpeg_b64 == null && frame.thumb_b64 == null) return frame;
  const metadata = { ...frame };
  delete metadata.jpeg_b64;
  delete metadata.thumb_b64;
  return metadata;
}

function compactFrames(
  frames: MmTrajectoryFrame[] | undefined,
  remaining: { chars: number },
  protectedInput = false,
): MmTrajectoryFrame[] | undefined {
  if (!frames?.length) return frames;
  if (protectedInput) return frames;
  return frames.map((frame) => {
    const chars = frameImageChars(frame);
    if (chars === 0) return frame;
    if (chars <= remaining.chars) {
      remaining.chars -= chars;
      return frame;
    }
    return withoutFrameImage(frame);
  });
}

function recentTaskIds(taskOrder: string[]): Set<string> {
  return new Set(taskOrder.slice(-QUERY_WORKER_IMAGE_TASK_LIMIT));
}

/**
 * Keep QueryWorker progress globally bounded while preserving textual/frame
 * metadata. The latest task's frozen ``started`` inputs are protected; older
 * debug images are evicted before their timestamps, source labels, or steps.
 */
// eslint-disable-next-line react-refresh/only-export-components
export function updateQueryWorkerProgressCache(
  existing: Map<string, QueryWorkerProgressStep[]>,
  taskId: string,
  incoming: QueryWorkerProgressStep | QueryWorkerProgressStep[],
): Map<string, QueryWorkerProgressStep[]> {
  const next = new Map(existing);
  const merged = mergeQueryWorkerProgress(next.get(taskId) || [], incoming);
  // Delete + set makes Map insertion order the LRU order.
  next.delete(taskId);
  next.set(taskId, merged);
  while (next.size > QUERY_WORKER_TASK_CACHE_LIMIT) {
    const oldest = next.keys().next().value;
    if (typeof oldest !== "string") break;
    next.delete(oldest);
  }

  const order = Array.from(next.keys());
  const newest = order.at(-1) || "";
  const imageTasks = recentTaskIds(order);
  const protectedChars = (next.get(newest) || [])
    .filter((step) => step.phase.startsWith("started:"))
    .flatMap((step) => step.frames || [])
    .reduce((total, frame) => total + frameImageChars(frame), 0);
  const remaining = {
    chars: Math.max(0, QUERY_WORKER_IMAGE_CHAR_BUDGET - protectedChars),
  };

  for (const id of order.slice().reverse()) {
    const steps = next.get(id) || [];
    const keepTaskImages = imageTasks.has(id);
    const compacted = steps.slice().reverse().map((step) => {
      const protectedInput = id === newest && step.phase.startsWith("started:");
      const frames = keepTaskImages
        ? compactFrames(step.frames, remaining, protectedInput)
        : step.frames?.map(withoutFrameImage);
      return frames === step.frames ? step : { ...step, frames };
    }).reverse();
    next.set(id, compacted);
  }
  return next;
}

export function mergeQueryWorkerProgress(
  existing: QueryWorkerProgressStep[],
  incoming: QueryWorkerProgressStep | QueryWorkerProgressStep[],
): QueryWorkerProgressStep[] {
  const byId = new Map<string, QueryWorkerProgressStep>();
  for (const step of existing) byId.set(step.id, step);
  for (const step of Array.isArray(incoming) ? incoming : [incoming]) {
    byId.set(step.id, step);
  }
  return Array.from(byId.values())
    .sort((a, b) => a.seq - b.seq || a.ts - b.ts || a.id.localeCompare(b.id))
    .slice(-QUERY_WORKER_PROGRESS_LIMIT);
}

/** Guard an async trajectory hydrate against session switches and stale pulls. */
// eslint-disable-next-line react-refresh/only-export-components
export function isCurrentTrajectoryHydration(
  requestedSessionId: string,
  requestedGeneration: number,
  currentSessionId: string,
  currentGeneration: number,
): boolean {
  return Boolean(requestedSessionId)
    && requestedSessionId === currentSessionId
    && requestedGeneration === currentGeneration;
}

/** 恢复历史时首屏先渲染的气泡条数 (剩下的下一帧补齐到 MAX_MESSAGES 窗口)。 */
export const HISTORY_FIRST_PAINT = 60;

/**
 * 把恢复出来的历史切成"首屏 + 补齐"两段的下标。
 *
 * 聊天列表是非虚拟化的普通 div (见 MAX_MESSAGES 处的注释), 所以一次性 setMessages
 * 400 条会在切换会话时同步解析 400 份 Markdown、一帧内交付 —— 这就是"切换后要等一下
 * 才出内容"的前端那一半原因。改成先渲染尾部 HISTORY_FIRST_PAINT 条 (用户视线本来就
 * 落在最新消息上), 下一帧再把窗口补到 MAX_MESSAGES。
 *
 * 返回的 firstStart / fullStart 都是对同一个 restored 数组的下标, 且 firstStart >=
 * fullStart。当历史本来就不超过首屏时两者相等, 调用方可据此跳过第二段 (不多余渲染一次)。
 */
// eslint-disable-next-line react-refresh/only-export-components
export function historyPaintWindow(
  total: number,
  maxMessages: number,
  firstPaint: number = HISTORY_FIRST_PAINT,
): { firstStart: number; fullStart: number; needsSecondPaint: boolean } {
  const fullStart = Math.max(0, total - maxMessages);
  const firstStart = Math.max(fullStart, total - Math.max(1, firstPaint));
  return { firstStart, fullStart, needsSecondPaint: firstStart > fullStart };
}

/** Apply the bounded cache to already-rendered tool cards without losing steps. */
// eslint-disable-next-line react-refresh/only-export-components
export function compactQueryWorkerMessageProgress(
  messages: ChatMsg[],
  cache: Map<string, QueryWorkerProgressStep[]>,
): ChatMsg[] {
  let changed = false;
  const compacted = messages.map((message) => {
    if (!message.workerTaskId || !message.workerProgress?.length) return message;
    const cached = cache.get(message.workerTaskId);
    const progress = cached || message.workerProgress.map((step) => {
      if (!step.frames?.some((frame) => frameImageChars(frame) > 0)) return step;
      return { ...step, frames: step.frames.map(withoutFrameImage) };
    });
    if (progress === message.workerProgress) return message;
    changed = true;
    return { ...message, workerProgress: progress };
  });
  return changed ? compacted : messages;
}

interface RecallTraceToolObs {
  name?: string;
  args?: Record<string, unknown>;
  obs_len?: number;
  elapsed_sec?: number;
  obs_summary?: string;
  frame_ids?: string[];
  evidence_segments?: RecallEvidenceSegment[];
  source_urls?: string[];
  cache_hit?: boolean;
  anchor?: string;
  anchor_ts?: number;
}

interface RecallTraceToolCall {
  name?: string;
  args?: Record<string, unknown>;
  anchor?: string;
  anchor_ts?: number;
}

interface RecallEvidenceSegment {
  kind?: string;
  t_start?: number;
  t_end?: number;
  frame_ids?: string[];
  preview?: string;
}

/** One classified tool-call argument (backend: agent.display.describe_arg_fields).
 *  - literal    → `value` is safe to display (identifiers, enums, paths, intent prose)
 *  - freeform   → payload the call is writing/sending; only `chars` is sent, never
 *                 the content, so a DM body or file content can't surface in the UI
 *  - shape      → array/object; only `count` is sent
 *  - credential → a secret (password/token/ssn/…): key ONLY, no value and no
 *                 length, since a secret's length is itself a hint
 *  - elided     → synthetic trailing entry (`key` is ""), `count` = how many
 *                 fields were dropped past the backend's per-call cap */
interface ToolArgField {
  key: string;
  kind: "literal" | "freeform" | "shape" | "credential" | "elided";
  value?: string;
  chars?: number;
  count?: number;
}

interface RecallTraceEntry {
  phase?: string;
  round?: number;
  can_answer?: boolean;
  next_tool_calls?: RecallTraceToolCall[];
  tools?: RecallTraceToolObs[];
  thought?: string;
  decision_summary?: string;
  useful_info?: string;
  clue?: string;
  query?: string;
  findings_len?: number;
  frame_ids?: string[];
  parallel_elapsed_sec?: number;
  elapsed_sec?: number;
  error?: string;
  stage?: string;
}

interface CropItem {
  label: string;
  bbox?: number[];
  width?: number;
  height?: number;
  jpeg_b64: string;
}

// One readable "segment" of a deep-research run — a single analysis round,
// rendered as a card: 🎬 第N段 [mm:ss–mm:ss] → 👁 看到 → 🔎/🧩 检索 → 📝 就绪.
export interface BgSegment {
  seg: number;                    // segment/round index (1-based for display)
  tsRange?: [number, number];     // frame time range for the header
  scene?: string;                 // 场景标记 (后端从本段 thought 廉价提取, 标题行展示)
  saw?: string;                   // 👁 what the model saw this round (from `thought`)
  thinking?: string;              // 💭 model's raw reasoning trace (thinking models)
  // 🔧 tool calls the model issued this round (name + a short arg preview).
  toolCalls?: { name: string; arg?: string }[];
  // ⚠️ tool failures this round (which tool + why).
  toolErrors?: { name: string; error: string }[];
  // 🔎 search / 🧩 recall lines: query → result summary.
  lookups: { kind: "search" | "recall"; query: string; result?: string; done?: boolean }[];
  ready?: boolean;                // 📝 this segment's 解读 is generated
  readyChars?: number;
  answer?: string;                // 📝 this segment's interpretation text (for folding)
  crops?: CropItem[];             // 🖼 crop thumbnails (image search)
}

export interface BgItem {
  id: string;                     // one item per request_id
  requestId?: string;             // which RouterEngine delegation
  label?: string;                 // UI label (lightweight summary) for the card title
  segments: BgSegment[];          // ordered segment cards (one per productive round)
  // frame-accumulation status: current/target frames + ttl countdown.
  waiting?: { have: number; need: number; ttlSec?: number; ttlRemaining?: number; seg?: number; paused?: boolean } | null;
  done?: boolean;
  report?: string;                // ★ latest incremental deep-research report (progress_report)
  reportBatches?: number;
  // Final consolidated report (summarize_watch) pushed once on completion via
  // watcher.final — the authoritative result, shown in-panel. The main agent
  // chat is never touched by the watcher.
  finalReport?: string;
}

interface ObsItem {
  ts: string;       // mm:ss timestamp
  speaker?: string; // audio-observation speaker label
  text: string;
}

interface CtxState {
  version: number;
  obs: ObsItem[];         // 画面观察(时间轴)
  audioObs: ObsItem[];    // 音频观察(时间轴)
  facts: Record<string, string>;  // SearchFactStore 的 UI 字符串投影
}

interface TtsRefs {
  audioCtx: AudioContext | null;
  audioNextStart: number;
  active: AudioBufferSourceNode[];
  currentRid: string | null;
  cancelled: Set<string>;
  // Barge-in guard: mute the mic (drop PCM, don't send to ASR) until this
  // epoch-ms deadline, so speaker-played TTS isn't re-captured and looped.
  ttsMuteUntil: number;
  // ★ #2 播放 ack: 追踪当前 rid 的播放进度, 打断时回传"实际听了多少"给后端,
  //   后端据此把"我说过什么"截断到用户真听到的部分。
  ctxStartTime: number;   // 当前 rid 首块的 AudioContext 起播时刻
  scheduledSec: number;   // 当前 rid 已排定的总播放时长 (秒)
}

interface Refs {
  gw: GatewayClient | null;
  sessionId: string;
  stream: MediaStream | null;
  sourceType: SourceType;
  capFps: number;
  capTimer: number | null;
  startTs: number;
  captureAttemptId: string;
  sentFrames: number;
  // Frames skipped because the WS out-buffer was over the backpressure
  // threshold. Surfaced in the diag log so we can tell "capture is throttling"
  // apart from "capture is broken".
  droppedFrames: number;
  // Last time (performance.now ms) the frameCount state was pushed — throttles
  // the display-only count to ~1/s so screen-share capture doesn't re-render
  // the page every tick.
  _lastCountPush?: number;
  // DEPRECATED bookkeeping: tracks whether any assistant bubble is streaming.
  // It NO LONGER gates frame capture — capture is always-on now (pausing it
  // dropped frames + staled the stream-liveness signal). Kept only because a few
  // error-recovery paths still reset it; safe to remove in a later cleanup.
  isAnswering: boolean;
  // mic (user speech → streaming realtime ASR → ask). Uses AudioWorklet
  // (off-main-thread) so PCM downsampling + base64 doesn't block the UI thread.
  micStream: MediaStream | null;
  micAudioCtx: AudioContext | null;
  micNode: AudioWorkletNode | null;
  micSource: MediaStreamAudioSourceNode | null;
  isRecording: boolean;
  asrTransport: AsrTurnTransport | null;
  micGeneration: number;
  micFlushResolve: (() => void) | null;
  micStopPromise: Promise<void> | null;
  // A disconnect/session boundary cancels the exact old turn before a
  // continuous Voice Dialog is allowed to bind to the replacement sid.
  micBoundaryPromise: Promise<void> | null;
  // env audio (screen/people speaking → audio_observation in memory)
  envStream: MediaStream | null;
  envRecorder: MediaRecorder | null;
  envStop: boolean;
  envMime: string;
  envWindowSec: number;
  envSliceTimer: number | null;
  envCaptureId: string;
  envChunkSeq: number;
  envLastError: string;
  // Set by the gateway effect so the ?mm= watcher can switch sessions in place
  // (resume a different id + restore its transcript) without a full remount.
  resumeSessionById?: (sid: string, restoreHistory: boolean) => Promise<boolean>;
  // Set by the gateway effect so the ?mm=new (新建) handler can create a fresh
  // session on demand (returns the new persisted id).
  createSession?: () => Promise<string>;
  // ★ The PERSISTED session id (stored_session_id / session_key), distinct from
  //   `sessionId` (the live runtime id RPCs route by). This is what `?mm=`, the
  //   sidebar list, and localStorage use — it survives auto-compress rotation.
  storedSid: string;
  // Stashed so the monitor/watcher toggle (render scope) can re-pull the
  // authoritative registry after a toggle (confirm the optimistic flip).
  fetchRegistries?: (sid: string) => void;
}

type MicLifecycleState = "idle" | "connecting" | "recording" | "finalizing";

/**
 * Stop browser capture immediately. For a user-initiated finish we first ask
 * the worklet to emit its sub-200ms tail, then close the audio graph. Cancel
 * paths deliberately discard that tail and never submit it.
 */
async function stopLocalMic(r: Refs, flushTail: boolean): Promise<boolean> {
  const node = r.micNode;
  let flushPromise: Promise<boolean> = Promise.resolve(!flushTail);
  if (flushTail && node && r.isRecording) {
    flushPromise = new Promise<boolean>((resolve) => {
      let settled = false;
      let timeout = 0;
      const finish = (confirmed: boolean) => {
        if (settled) return;
        settled = true;
        if (timeout) window.clearTimeout(timeout);
        r.micFlushResolve = null;
        resolve(confirmed);
      };
      r.micFlushResolve = () => finish(true);
      timeout = window.setTimeout(() => finish(false), 300);
      try {
        node.port.postMessage({ type: "flush" });
      } catch {
        finish(false);
      }
    });
  }
  // Queue the worklet flush first, then stop the physical track in the same
  // synchronous turn. Capture ends immediately, while the already-processed
  // tail remains available to the worklet's message handler.
  try { r.micStream?.getTracks().forEach((track) => track.stop()); } catch { /* noop */ }
  const flushConfirmed = await flushPromise;
  r.isRecording = false;
  r.micFlushResolve = null;
  try {
    if (node) {
      node.port.onmessage = null;
      node.port.close();
      node.disconnect();
    }
  } catch { /* noop */ }
  try { r.micSource?.disconnect(); } catch { /* noop */ }
  try {
    if (r.micAudioCtx && r.micAudioCtx.state !== "closed") await r.micAudioCtx.close();
  } catch { /* noop */ }
  r.micNode = null;
  r.micSource = null;
  r.micAudioCtx = null;
  r.micStream = null;
  return flushConfirmed;
}

/** Cancel the transport before tearing down PCM so late worklet messages drop. */
function cancelActiveMic(r: Refs): Promise<void> {
  r.micGeneration += 1;
  const turn = r.asrTransport?.current();
  const cancel = turn
    ? r.asrTransport?.stop(turn.sessionId, turn.turnId, "cancel").catch(() => undefined)
    : Promise.resolve();
  const operation = Promise.allSettled([cancel, stopLocalMic(r, false)]).then(() => undefined);
  r.micBoundaryPromise = operation;
  void operation.finally(() => {
    if (r.micBoundaryPromise === operation) r.micBoundaryPromise = null;
  });
  return operation;
}

function pickMicMime(): string {
  const c = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg"];
  for (const m of c) {
    if (typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported(m)) return m;
  }
  return "";
}

function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => {
      const s = (r.result as string) || "";
      const i = s.indexOf(",");
      resolve(i >= 0 ? s.slice(i + 1) : s);
    };
    r.onerror = reject;
    r.readAsDataURL(blob);
  });
}

let _seq = 0;
const nid = () => `m${++_seq}_${Date.now()}`;

// ── Backend history → ChatMsg[] (resume restoration) ────────────────────────
// Convert the flat message array returned by `session.resume` (gateway) or
// `getSessionMessages` (REST) into the page's ChatMsg bubbles, so reopening a
// session restores its transcript instead of a blank waterfall.
//
// Field shapes differ between the two sources — we accept both:
//   gateway resume: { role, text, name?, context?, subRole?, monitorLabel?, ... }
//   REST messages : { role, content, tool_name?, tool_call_id?, reasoning?, tool_calls? }
// so we read text from `text || content`, tool name from `name || tool_name`, etc.
interface RawHistoryMsg {
  role?: string;
  text?: string;
  content?: unknown;
  timestamp?: number;   // DB 消息时间戳 (秒); 恢复气泡用它还原 createdAt
  name?: string;
  tool_name?: string;
  context?: string;      // 调用侧: 命令/参数预览 (≠ 工具返回值)
  args_fields?: ToolArgField[];  // 调用侧结构化入参 (与实时 tool.start 同构)
  summary?: string;      // 工具结果摘要 (exit code / error 首行)
  tool_call_id?: string;
  tool_calls?: unknown;
  reasoning?: unknown;
  reasoning_content?: unknown;
  subRole?: string;
  requestId?: string;
  monitorLabel?: string;
  eventId?: string;
  brief?: string;
  deepRange?: string;
  deepReportRid?: string;
  deepRound?: number;
  history_policy?: unknown;
  ephemeral_control?: unknown;
  ephemeral?: unknown;
}

/** Coerce a message `content`/`text` field to a plain display string.
 *  Backend flattens most content to a string already; array content (rare,
 *  multimodal blocks) is reduced to its text parts + [image]/[audio] markers. */
function _coerceHistoryText(m: RawHistoryMsg): string {
  if (typeof m.text === "string") return m.text;
  const c = m.content;
  if (typeof c === "string") return c;
  if (Array.isArray(c)) {
    const parts: string[] = [];
    for (const part of c) {
      if (typeof part === "string") { parts.push(part); continue; }
      if (part && typeof part === "object") {
        const p = part as Record<string, unknown>;
        if (typeof p.text === "string") parts.push(p.text);
        else if (p.type === "image_url" || p.type === "image") parts.push("[image]");
        else if (p.type === "input_audio" || p.type === "audio") parts.push("[audio]");
      }
    }
    return parts.join("");
  }
  return "";
}

/** Convert a backend history array into ChatMsg bubbles for the waterfall.
 *   orphanIds: monitor/watcher event ids that are NOT on this session's disk
 *  (磁盘为权威) → their bubbles are dropped (not rendered). The caller toasts. */
// eslint-disable-next-line react-refresh/only-export-components
export function historyToMmMessages(raw: unknown, orphanIds?: Set<string>): ChatMsg[] {
  if (!Array.isArray(raw)) return [];
  const out: ChatMsg[] = [];
  const orphans = orphanIds || new Set<string>();

  // ── Per-turn aggregation (对齐实时流式观感) ─────────────────────────────
  // 实时时, 主 agent 一轮内的所有普通文本 delta 都追加到【同一条】assistant
  // 气泡 (ensureBubble 按 stream-key 复用 id), tool 气泡穿插其后 → 观感是
  // "一条文本 + 其后一串工具"。但 DB 把一轮存成多行:
  //   assistant(文本1+tool_calls) → tool → assistant(文本2+tool_calls) → tool
  // 逐行重建会拆成 [文本1,工具1,文本2,工具2] 交替多条, 与实时不一致 (用户报的
  // "退出重进后展开成多个交替消息")。这里按轮聚合: 把同一轮的多段主 agent 文本
  // 合并成一条气泡, 该轮的 tool 气泡跟在其后, 复刻实时的分组。
  //
  // 轮边界: 一条 user 消息开启新轮 (monitor/watcher 触发的 user 注入同样算新轮)。
  // 只聚合"普通主 agent 文本" (无特殊 subRole); monitor/watcher/router 子气泡与
  // reasoning 保持各自独立, 不并入合并文本。
  let turnTextSegs: string[] = [];
  let turnReasoning: string | undefined;
  let turnTools: ChatMsg[] = [];
  const flushTurn = () => {
    if (turnTextSegs.length > 0 || turnReasoning) {
      const merged = turnTextSegs.filter(Boolean).join("\n\n").trim();
      if (merged || turnReasoning) {
        out.push({
          id: nid(), role: "assistant", text: merged,
          ...(turnReasoning ? { reasoning: turnReasoning } : {}),
        });
      }
    }
    for (const t of turnTools) out.push(t);
    turnTextSegs = [];
    turnReasoning = undefined;
    turnTools = [];
  };

  for (const item of raw as RawHistoryMsg[]) {
    if (!item || typeof item !== "object") continue;
    // Defensive compatibility for sessions written during a staggered
    // backend rollout. New pure Monitor controls are not persisted at all;
    // if an older writer did persist a marked row, never resurrect it here.
    if (isEphemeralControl(item)) continue;

    // ★ mm_notice 双形态兜底 (对齐 desktop toChatMessages):
    //   两条恢复路径返回的形态不同 ——
    //     gateway session.resume → _history_to_messages 已重建: 顶层带 subRole /
    //       monitorLabel / eventId / deepReportRid (下面 subRole 分支认)。
    //     REST /api/sessions/{id}/messages → 未重建, 还是原始 dict:
    //       { role, content:{ type:"mm_notice", mm_kind, mm_event_id, mm_label,
    //         mm_round?, text } } —— 无 subRole。
    //   历史恢复实际可能吃到任一条 (gateway 或 REST/兜底), 所以这里【两种都认】,
    //   把原始 dict 也重建成 monitor/watcher_report 气泡, 否则该条会因 text 取不到
    //   (content 是 dict) 被当空消息丢弃 (web 端曾丢 monitor/watcher 通知的真因)。
    const _c = (item as { content?: unknown }).content;
    if (_c && typeof _c === "object" && !Array.isArray(_c)
        && (_c as { type?: string }).type === "mm_notice") {
      // Monitor + watcher notices no longer live in the center chat — they
      // hydrate the right multimodal panel from mm_monitor_alerts +
      // mm_watcher_reports sidechannel tables (see list_monitor_alerts /
      // list_watcher_content RPCs). Legacy rows from before the split are
      // simply skipped here; the query/query_user notices don't take this raw
      // branch (they come through as subRole:query_worker in the reshaped
      // branch below).
      continue;
    }

    // ★ 孤儿丢弃: 该 monitor/watcher 气泡的 event id 不在磁盘 → 不渲染。
    const _eid = String(
      (item as { monitorId?: string; deepReportRid?: string; eventId?: string })
        .monitorId
      || (item as { deepReportRid?: string }).deepReportRid
      || (item as { eventId?: string }).eventId || "");
    if (_eid && orphans.has(_eid)) continue;
    const role = String(item.role || "");
    // Tool result rows → a completed tool bubble, buffered into the current
    // turn so it renders after this turn's merged assistant text (matching the
    // realtime "one text bubble + trailing tool bubbles" grouping).
    if (role === "tool") {
      const toolName = String(item.name || item.tool_name || "tool");
      // ★ ctx 与 detail 是【两个不同东西】, 不能互相兜底:
      //     context = 调用侧 (命令/参数预览, 后端 _tool_ctx 截到 80 字)
      //     content = 工具真正的返回值 (后端 _history_tool_result 截断后的投影)
      //   旧代码 `_coerceHistoryText(item) || item.context` 在没有 content 时
      //   回落到 context, 把"命令预览"塞进 toolDetail → 摘要行只剩 "✓ terminal",
      //   同时凭空多出一层 <details>, 点开只有那 80 字命令 (用户报的"点开很冗余
      //   又没信息")。现在各归各位: 没有真实输出就不给 toolDetail, 不出折叠层。
      const detail = _coerceHistoryText(item);
      const ctx = typeof item.context === "string" ? item.context : "";
      const recallDebug = isQueryMultimodalHistoryToolName(toolName)
        ? extractRecallDebug(null, detail)
        : null;
      const parsedToolResult = safeJsonParse(detail);
      const workerTaskId = isQueryMultimodalHistoryToolName(toolName)
        && isRecord(parsedToolResult)
        && parsedToolResult.reply_owner === "query_worker"
        && typeof parsedToolResult.task_id === "string"
          ? parsedToolResult.task_id : "";
      turnTools.push({
        id: nid(), role: "assistant", text: "", kind: "tool",
        toolName, toolDone: true,
        ...(ctx ? { toolCtx: ctx } : {}),
        ...(item.args_fields?.length ? { toolArgs: item.args_fields } : {}),
        ...(detail ? { toolDetail: detail } : {}),
        ...(item.summary ? { toolSummary: String(item.summary) } : {}),
        ...(recallDebug?.trace?.length ? { recallTrace: recallDebug.trace } : {}),
        ...(recallDebug?.findings ? { recallFindings: recallDebug.findings } : {}),
        ...(workerTaskId ? {
          workerTaskId,
          workerStatus: "running" as const,
          workerProgress: [],
        } : {}),
      });
      continue;
    }
    if (role !== "user" && role !== "assistant" && role !== "system") continue;
    const text = _coerceHistoryText(item);
    // Monitor + watcher_report legacy rows are dropped here — they hydrate
    // the right multimodal panel from the sidechannel RPCs instead. Only
    // query_worker (main-agent-driven Recall) still shows as a center bubble.
    const subRole = item.subRole;
    if (subRole === "monitor" || subRole === "watcher_report") {
      continue;
    }
    if (subRole === "query_worker") {
      // Special sub-agent bubble ends the current main-agent turn aggregation.
      flushTurn();
      const _ts2 = typeof item.timestamp === "number" && item.timestamp > 0
        ? item.timestamp * 1000
        : undefined;
      out.push({
        id: nid(), role: "assistant", text,
        subRole: "query_worker",
        requestId: item.requestId || item.eventId || undefined,
        brief: item.brief,
        deepRange: item.deepRange,
        ...(_ts2 ? { createdAt: _ts2 } : {}),
      });
      continue;
    }
    // An assistant turn that only carried tool_calls (no visible text) is a
    // routing placeholder — skip it (the tool rows above already show the work).
    if (role === "assistant" && !text && item.tool_calls) continue;
    if (!text) continue;
    const reasoning =
      typeof item.reasoning === "string" ? item.reasoning
      : typeof item.reasoning_content === "string" ? item.reasoning_content
      : undefined;
    if (role === "assistant") {
      // Accumulate this turn's main-agent text; merged into one bubble at the
      // turn boundary (flushTurn) so multi-tool turns render as a single
      // message + trailing tool bubbles, matching the realtime stream.
      turnTextSegs.push(text);
      if (reasoning && !turnReasoning) turnReasoning = reasoning;
      continue;
    }
    // user / system: a real turn boundary — flush the prior turn first, then
    // push this message independently.
    flushTurn();
    out.push({
      id: nid(), role: role as ChatMsg["role"], text,
      ...(reasoning ? { reasoning } : {}),
    });
  }
  flushTurn();
  return out;
}

/** Format an epoch-ms timestamp as local HH:MM:SS for a message header. */
const fmtClock = (ms?: number): string => {
  if (!ms) return "";
  const d = new Date(ms);
  const p = (n: number) => String(n).padStart(2, "0");
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
};


async function blitVideoToCanvas(
  v: HTMLVideoElement,
  cvs: HTMLCanvasElement,
  w: number,
  h: number,
  resizeQuality: "low" | "medium" | "high" = "low",
): Promise<void> {
  if (cvs.width !== w) cvs.width = w;
  if (cvs.height !== h) cvs.height = h;
  const ctx = cvs.getContext("2d");
  if (!ctx) return;
  // createImageBitmap(resize*) offloads downscale off the sync drawImage path;
  // Retina / HiDPI screen-share tracks are often 2–4× the logical size (e.g. a
  // 2560-wide native capture). Frames arrive here already clamped by their
  // visual-capture profile, so this is a light blit; "medium" keeps text crisp
  // on the normal 1080p screen tier while 720p camera/light tiers stay cheap.
  if (typeof createImageBitmap === "function") {
    try {
      const bitmap = await createImageBitmap(v, {
        resizeWidth: w,
        resizeHeight: h,
        resizeQuality,
      });
      ctx.drawImage(bitmap, 0, 0);
      bitmap.close();
      return;
    } catch {
      /* fall through */
    }
  }
  ctx.drawImage(v, 0, 0, w, h);
}

// Input row (textarea + mic + send) as an isolated leaf. It owns `askText`
// locally so each keystroke re-renders ONLY this component — not the whole
// page and its (up to 120) message bubbles. onSend receives the text and the
// composer clears itself; the parent never sees per-keystroke state.
const _MM_SESSION_KEY = "mm.sessionId";

// 新会话/切换会话时置顶的"系统"引导气泡。★ 用工厂函数 (每次给新 id), 避免多处共用同
//   一 object 引用。之前性能重构把它只放进 useState 初值, resetSessionUi 清成 [] 后不再
//   补回 → 新建/切换会话就没有这条置顶引导了。
const _mmWelcomeMsg = (): ChatMsg => ({
  id: nid(), role: "system",
  text: "Turn on the camera or share your screen, then just ask. One-shot visual questions go to QueryWorker, which reads the frames from the moment you asked and, when needed, recalls history or searches for reference material.",
});
const ChatComposer = memo(function ChatComposer({
  micState, onSend, onSlash, gw, onMicToggle, generating, onStop,
  ttsEnabled, onTtsToggle, composerApiRef,
}: {
  micState: MicLifecycleState;
  onSend: (text: string) => void;
  // `/`-prefixed lines run the gateway slash pipeline instead of a chat turn.
  onSlash: (command: string) => void;
  gw: GatewayClient | null;
  onMicToggle: () => void;
  generating: boolean;
  onStop: () => void;
  ttsEnabled: boolean;
  onTtsToggle: () => void;
  // Lets the parent (slash pipeline /undo prefill) set the composer text.
  composerApiRef?: React.MutableRefObject<{
    setText: (text: string) => void;
  } | null>;
}) {
  const { t } = useI18n();
  const [askText, setAskText] = useState("");
  const taRef = useRef<HTMLTextAreaElement | null>(null);
  const slashRef = useRef<SlashPopoverHandle>(null);
  // Register the stable setter so the parent can prefill the composer
  // (e.g. /undo drops the backed-up user turn back in for editing).
  useLayoutEffect(() => {
    if (composerApiRef) composerApiRef.current = { setText: setAskText };
  }, [composerApiRef]);
  const submit = () => {
    const txt = askText.trim();
    if (!txt) return;
    // A leading "/" is a system command → gateway slash pipeline; everything
    // else is a normal question. Mirrors the desktop composer / Ink TUI.
    if (txt.startsWith("/")) onSlash(txt);
    else onSend(txt);
    setAskText("");
  };
  // 单行起步, 内容换行时长高到 max-h-24 为止 —— 否则 rows={1} 的可视高度固定,
  // 第二行会被裁掉(overflow 能滚, 但用户看不见自己刚打的字)。清空后缩回一行。
  // 在 layout effect 里做, 避免先按旧高度绘制再跳一帧。
  useLayoutEffect(() => {
    const el = taRef.current;
    if (!el) return;
    el.style.height = "auto";              // 先塌陷, 再按真实内容量取值
    el.style.height = `${el.scrollHeight}px`;
  }, [askText]);
  const connecting = micState === "connecting";
  const recording = micState === "recording";
  const finalizing = micState === "finalizing";
  return (
    // relative wrapper: SlashPopover pops up `bottom-full` above the composer.
    <div className="relative">
      <SlashPopover ref={slashRef} input={askText} gw={gw} onApply={setAskText} />
    {/* items-end: composer surface 现在比两个 icon 按钮高, 按钮贴底对齐才不会飘在
        输入框中部 (desktop 同样用 items-end)。 */}
    <div className="flex items-end gap-2 border-t p-3">
      <Button size="icon"
        // Red only when actually recording. Connecting stays clickable so a
        // slow permission/backend handshake can be cancelled. Finalizing is
        // disabled to make the one-submit boundary explicit and idempotent.
        // ★ 对话模式开时: 按钮态保持不变(不禁用/不联动高亮), 但点击无效 —— 拦截+
        //   提示在父级 onMicToggle 里做 (对话独占麦, 后台已联动)。
        destructive={recording}
        outlined={!recording}
        disabled={finalizing}
        title={recording ? t.multimodal.voice.stopRecording
          : connecting ? t.multimodal.voice.connecting
          : finalizing ? t.multimodal.voice.connecting
          : t.multimodal.voice.startSpeaking}
        onClick={onMicToggle}>
        {connecting || finalizing ? <Loader2 className="animate-spin" /> : <Mic />}
      </Button>
      <Button size="icon"
        // 独立 TTS 语音播报开关 (与麦克风解耦): 开 = 实心高亮, 关 = 描边。
        // ★ 对话模式开时: 喇叭按钮态保持不变(后台已强制 TTS 生效), 点击无效 ——
        //   拦截+提示在父级 onTtsToggle 里做。
        outlined={!ttsEnabled}
        title={ttsEnabled
          ? t.multimodal.voice.ttsOnTitle
          : t.multimodal.voice.ttsOffTitle}
        onClick={onTtsToggle}>
        <Volume2 />
      </Button>
      {/* ★ Composer surface —— 单行: 输入区 + pill 同处一行、同一个 border 内。
          发送/停止 在 surface 之外的右侧 (和左侧那两个 toggle 一样是框外控件);
          框内只放"编辑相关"的东西, 动作按钮不进编辑框。
          文字在 pill 处截止 —— 靠 flex 分栏而非 padding 预留: 输入区是
          `min-w-0 flex-1`, pill 是 `shrink-0`, 所以文字天然写到 pill 左边缘就
          换行, 不会跑到 pill 底下。
          注意不要加 overflow-hidden: ChatModelPill 的面板是 `absolute
          bottom-full` 向上弹出的, 会被裁掉。 */}
      <div className="flex min-w-0 flex-1 items-center gap-2 rounded-md border bg-background px-3 py-1 focus-within:border-foreground/30">
        <textarea
          ref={taRef}
          value={askText}
          onChange={(e) => setAskText(e.target.value)}
          onKeyDown={(e) => {
            // Let the slash popover claim ↑/↓/Tab/Esc first; it lets Enter fall
            // through so Enter still submits the (possibly completed) command.
            if (slashRef.current?.handleKey(e)) return;
            if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); }
          }}
          rows={1}
          placeholder={t.multimodal.composer.placeholder}
          // min-w-0 让 flex 子项可以真正收缩(否则 textarea 的默认固有宽度会把
          // pill 挤出去); max-h-24 是长高的上限, 超过就内部滚动。
          className="max-h-24 min-w-0 flex-1 resize-none self-center overflow-y-auto border-0 bg-transparent p-0 text-sm leading-snug outline-none" />
        <ChatModelPill className="shrink-0" />
      </div>
      {/* A live foreground turn no longer disables intake. New sends are
          accepted into the backend FIFO; Stop remains an explicit, separate
          action for cancelling the current turn + its queued successors. */}
      {generating && (
        <Button className="shrink-0" size="icon" destructive title={t.multimodal.composer.stop} onClick={onStop}>
          <Square />
        </Button>
      )}
      <Button className="shrink-0" size="sm" prefix={<Send />} onClick={submit}>
        {generating ? t.multimodal.composer.queued : t.multimodal.composer.send}
      </Button>
    </div>
    </div>
  );
});

// ASR live preview — isolated so partial transcript updates don't re-render
// the chat list / video column (Mac Chrome was stuttering during voice input).
// buffer: already-stitched EOU segments shown behind the current partial.
const AsrBar = memo(function AsrBar({
  recording, partial, buffer,
}: { recording: boolean; partial: string; buffer: string[] }) {
  // Re-render on language switch: this component's labels come from
  // translateNow(), which React cannot see. Must precede the early return so
  // the hook order stays stable across renders.
  useLocaleRevision();
  if (!recording && !partial && buffer.length === 0) return null;
  const buffered = buffer.join(" ").trim();
  return (
    <div className="flex items-center gap-2 border-t px-3 pt-2 text-xs text-muted-foreground">
      {recording && <span className="h-2 w-2 flex-shrink-0 animate-pulse rounded-full bg-red-500" />}
      <span className="truncate">
        {buffered ? (
          <>
            <span className="opacity-60">{buffered}</span>
            {partial ? <span className="ml-1">{partial}</span> : null}
          </>
        ) : (
          partial || "Listening..."
        )}
      </span>
    </div>
  );
});

function safeJsonParse(text: string): unknown {
  try { return JSON.parse(text); } catch { return null; }
}

function isRecord(x: unknown): x is Record<string, unknown> {
  return !!x && typeof x === "object" && !Array.isArray(x);
}

function extractRecallDebug(result: unknown, detail?: string): {
  trace: RecallTraceEntry[];
  findings?: string;
} | null {
  const direct = isRecord(result) ? result : null;
  const parsed = !direct && detail ? safeJsonParse(detail) : null;
  const obj = direct || (isRecord(parsed) ? parsed : null);
  if (!obj) return null;
  const traceValue = obj.recall_trace ?? obj.trace;
  const trace = Array.isArray(traceValue)
    ? traceValue.filter(isRecord) as RecallTraceEntry[]
    : [];
  const findings = typeof obj.findings === "string"
    ? obj.findings
    : typeof obj.partial_findings === "string"
      ? obj.partial_findings
      : undefined;
  if (!trace.length && !findings) return null;
  return { trace, findings };
}

function argPreview(args: Record<string, unknown> | undefined, max = 180): string {
  if (!args) return "";
  const q = args.query ?? args.entity_id ?? args.task_id ?? args.frame_id ?? args.target;
  const raw = typeof q === "string" ? q : JSON.stringify(args);
  return String(raw || "").replace(/\s+/g, " ").slice(0, max);
}

export function formatTraceTime(seconds: unknown): string {
  const value = Number(seconds);
  if (!Number.isFinite(value) || value < 0) return "";
  const tenths = Math.round(value * 10);
  const whole = Math.floor(tenths / 10);
  const mins = Math.floor(whole / 60);
  const secs = whole % 60;
  const tenth = tenths % 10;
  const fraction = tenth ? `.${tenth}` : "";
  return `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}${fraction}`;
}

function normalizeQueryWorkerOcrRecords(value: unknown): QueryWorkerOcrRecord[] {
  if (!Array.isArray(value)) return [];
  return value.filter(isRecord).slice(0, QUERY_WORKER_OCR_RECORD_LIMIT).map((record) => {
    const frameTsRaw = Number(record.frame_ts ?? record.frameTs);
    const bounded = (snake: string, camel: string, limit: number): string => {
      const raw = record[snake] ?? record[camel];
      return typeof raw === "string" ? raw.slice(0, limit) : "";
    };
    return {
      ...(Number.isFinite(frameTsRaw) ? { frameTs: frameTsRaw } : {}),
      ...(bounded("source_type", "sourceType", 80)
        ? { sourceType: bounded("source_type", "sourceType", 80) } : {}),
      ...(bounded("evidence_source", "evidenceSource", 120)
        ? { evidenceSource: bounded("evidence_source", "evidenceSource", 120) } : {}),
      ...(bounded("app", "app", 160) ? { app: bounded("app", "app", 160) } : {}),
      ...(bounded("window_title", "windowTitle", 240)
        ? { windowTitle: bounded("window_title", "windowTitle", 240) } : {}),
      rawText: bounded("raw_text", "rawText", QUERY_WORKER_OCR_TEXT_LIMIT),
    };
  });
}

function queryOcrSourceLabel(sourceType?: string): string {
  const value = String(sourceType || "").trim().toLowerCase();
  if (value === "camera" || value === "webcam") return translateNow("multimodal.ocr.cameraLive");
  if (["screen", "screenshare", "screen_share", "desktop", "display", "window", "tab"].includes(value)) {
    return translateNow("multimodal.ocr.screenLive");
  }
  return sourceType?.trim() || translateNow("multimodal.ocr.methodUnknown");
}

function queryOcrMethodLabel(evidenceSource?: string): string {
  const value = String(evidenceSource || "").trim().toLowerCase();
  if (value === "background_screen_texts" || value.includes("background") || value.includes("cache")) {
    return translateNow("multimodal.ocr.backgroundCache");
  }
  if (value === "synchronous_camera_ocr") return translateNow("multimodal.ocr.cameraLive");
  if (value === "synchronous_screen_fallback") return translateNow("multimodal.ocr.screenLive");
  return evidenceSource?.trim() || translateNow("multimodal.ocr.methodUnknown");
}

function queryOcrStateMessage(step: QueryWorkerProgressStep): string {
  if (step.ocrState === "timeout") {
    return translateNow("multimodal.ocr.timeout");
  }
  if (step.ocrState === "error") {
    return translateNow("multimodal.ocr.error");
  }
  if (step.ocrState === "skipped") {
    if (step.ocrReason === "no_frozen_frames") return translateNow("multimodal.ocr.skippedNoFrames");
    if (step.ocrReason === "ocr_unavailable") return translateNow("multimodal.ocr.skippedUnavailable");
    return step.ocrReason ? translateNow("multimodal.ocr.skippedWithReason", step.ocrReason) : translateNow("multimodal.ocr.skippedGeneric");
  }
  return translateNow("multimodal.ocr.noTextFound");
}

function QueryWorkerOcrEvidence({ step }: { step: QueryWorkerProgressStep }) {
  const records = step.ocrRecords || [];
  const count = step.ocrRecordCount ?? records.length;
  return (
    <details open className="mt-1.5 rounded border border-sky-300/20 bg-sky-300/5 px-2 py-1.5">
      <summary className="cursor-pointer select-none list-none text-[10px] font-medium text-sky-100">
        {translateNow("multimodal.ocr.helperTitle")}
        <span className="ml-1.5 font-normal text-muted-foreground/70">
          {records.length ? translateNow("multimodal.ocr.countItems", count) : translateNow("multimodal.ocr.noText")}
          {step.ocrElapsedSec != null ? ` · ${step.ocrElapsedSec.toFixed(2)}s` : ""}
        </span>
      </summary>
      {records.length ? (
        <div className="mt-1.5 space-y-1.5">
          {records.map((record, index) => (
            <div key={`${record.frameTs ?? "unknown"}-${record.evidenceSource || "ocr"}-${index}`} className="rounded border border-sky-300/15 bg-black/15 p-1.5">
              <div className="flex flex-wrap gap-1">
                <span className="rounded border border-sky-300/20 px-1.5 py-0.5 font-mono text-[9px] text-sky-100/75">
                  {record.frameTs != null ? formatTraceTime(record.frameTs) : translateNow("multimodal.ocr.timeUnknown")}
                </span>
                <span className="rounded border border-sky-300/20 px-1.5 py-0.5 text-[9px] text-sky-100/75">
                  {queryOcrSourceLabel(record.sourceType)}
                </span>
                <span className="rounded border border-sky-300/20 px-1.5 py-0.5 text-[9px] text-sky-100/75">
                  {queryOcrMethodLabel(record.evidenceSource)}
                </span>
              </div>
              {(record.app || record.windowTitle) && (
                <div className="mt-1 truncate text-[9px] text-muted-foreground/60" title={[record.app, record.windowTitle].filter(Boolean).join(" · ")}>
                  {[record.app, record.windowTitle].filter(Boolean).join(" · ")}
                </div>
              )}
              <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap break-words rounded bg-black/20 p-1.5 text-[10px] text-foreground/80">
                {record.rawText || translateNow("multimodal.ocr.noTextInFrame")}
              </pre>
            </div>
          ))}
        </div>
      ) : (
        <div className="mt-1.5 rounded bg-black/15 px-2 py-1 text-[10px] text-muted-foreground/75">
          {queryOcrStateMessage(step)}
        </div>
      )}
      <div className="mt-1 text-[9px] text-sky-100/45">
        {translateNow("multimodal.misc.ocrDisclaimer")}
      </div>
    </details>
  );
}

function sourceClipMetric(value: unknown): string | undefined {
  if (!isRecord(value)) return undefined;
  const start = formatTraceTime(value.t_start);
  const end = formatTraceTime(value.t_end);
  if (!start || !end) return undefined;
  const count = Number(value.n_frames || 0);
  return translateNow("multimodal.evidence.sourceClip", start, end, count);
}

function evidenceSegmentLabel(segment: RecallEvidenceSegment): string {
  const start = formatTraceTime(segment.t_start);
  const end = formatTraceTime(segment.t_end);
  if (!start) return "";
  const time = end && end !== start ? `${start}–${end}` : start;
  const kind = segment.kind === "audio" ? translateNow("multimodal.evidence.audio")
    : segment.kind === "quote" ? translateNow("multimodal.evidence.quote")
      : segment.kind === "screen" ? translateNow("multimodal.evidence.screen")
        : segment.kind === "frame" ? translateNow("multimodal.evidence.frame") : translateNow("multimodal.evidence.memory");
  return `${kind} ${time}`;
}

function RecallTracePanel({
  trace, findings,
}: {
  trace?: RecallTraceEntry[];
  findings?: string;
}) {
  const items = trace || [];
  if (!items.length && !findings) return null;
  const toolCount = items.reduce((n, e) => n + (e.tools?.length || 0), 0);
  return (
    <details className="mt-1 rounded border border-emerald-400/30 bg-emerald-400/5 p-2 text-[11px] text-emerald-100/90">
      <summary className="cursor-pointer select-none font-medium text-emerald-200">
        {translateNow("multimodal.recall.traceTitle", items.length, toolCount)}
      </summary>
      <div className="mt-1 text-[10px] text-emerald-100/55">
        {translateNow("multimodal.recall.traceSubtitle")}
      </div>
      <div className="mt-2 space-y-2">
        {findings && (
          <div className="rounded border border-emerald-400/20 bg-background/40 p-2">
            <div className="mb-1 text-emerald-300/90">findings</div>
            <div className="whitespace-pre-wrap break-words text-foreground/85">{findings}</div>
          </div>
        )}
        {items.map((e, idx) => {
          const phase = String(e.phase || "step");
          return (
            <div key={`${phase}-${idx}`} className="rounded border border-border/60 bg-background/40 p-2">
              <div className="mb-1 flex flex-wrap items-center gap-1.5 text-emerald-200">
                <span className="font-medium">{phase}</span>
                {e.round != null && <span className="text-muted-foreground">r{e.round}</span>}
                {e.can_answer != null && (
                  <span className={e.can_answer ? "text-emerald-300" : "text-amber-300"}>
                    can_answer={String(e.can_answer)}
                  </span>
                )}
                {e.parallel_elapsed_sec != null && (
                  <span className="text-muted-foreground">
                    {Number(e.parallel_elapsed_sec).toFixed(2)}s
                  </span>
                )}
              </div>
              {e.decision_summary && (
                <div className="mb-1 whitespace-pre-wrap break-words text-muted-foreground">
                  {translateNow("multimodal.recall.decisionSummary", e.decision_summary)}
                </div>
              )}
              {e.error && (
                <div className="mb-1 whitespace-pre-wrap break-words text-red-300">
                  {e.stage ? `${e.stage}: ` : ""}{e.error}
                </div>
              )}
              {e.useful_info && <div className="mb-1 whitespace-pre-wrap break-words text-foreground/80">useful: {e.useful_info}</div>}
              {e.clue && <div className="mb-1 whitespace-pre-wrap break-words text-foreground/80">clue: {e.clue}</div>}
              {e.query && <div className="mb-1 break-words text-muted-foreground">query: {e.query}</div>}
              {Array.isArray(e.next_tool_calls) && e.next_tool_calls.length > 0 && (
                <div className="space-y-1">
                  <div className="text-emerald-300/90">planned tools</div>
                  {e.next_tool_calls.map((tc, i) => (
                    <div key={`planned-${i}`} className="rounded bg-muted/30 px-2 py-1">
                      <span className="font-medium text-foreground/90">{tc.name || "tool"}</span>
                      {tc.args && <span className="text-muted-foreground"> · {argPreview(tc.args)}</span>}
                    </div>
                  ))}
                </div>
              )}
              {Array.isArray(e.tools) && e.tools.length > 0 && (
                <div className="space-y-1">
                  <div className="text-emerald-300/90">tool results</div>
                  {e.tools.map((tool, i) => (
                    <details key={`tool-${i}`} className="rounded bg-muted/30 px-2 py-1">
                      <summary className="cursor-pointer select-none list-none">
                        <span className="font-medium text-foreground/90">{tool.name || "tool"}</span>
                        {tool.args && <span className="text-muted-foreground"> · {argPreview(tool.args)}</span>}
                        {tool.obs_len != null && <span className="text-muted-foreground/70"> · {tool.obs_len} chars</span>}
                        {tool.frame_ids?.length ? <span className="text-muted-foreground/70"> · {tool.frame_ids.length} frames</span> : null}
                      </summary>
                      {tool.obs_summary && (
                        <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap break-words rounded border bg-background/50 p-2 text-foreground/80">
                          {tool.obs_summary}
                        </pre>
                      )}
                    </details>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </details>
  );
}

export const QueryWorkerProgressPanel = memo(function QueryWorkerProgressPanel({
  taskId, status, steps,
}: {
  taskId: string;
  status?: ChatMsg["workerStatus"];
  steps: QueryWorkerProgressStep[];
}) {
  useLocaleRevision();  // labels come from translateNow — see AsrBar
  const active = !status || status === "running";
  const visible = steps.slice(-QUERY_WORKER_PROGRESS_LIMIT);
  return (
    <div className="mt-2 rounded border border-cyan-400/35 bg-cyan-950/20 p-2 text-[11px] text-muted-foreground">
      <div className="flex items-center gap-1.5 text-cyan-200">
        {active
          ? <span className="inline-block animate-spin">◌</span>
          : status === "complete" ? <span className="text-emerald-400">✓</span>
            : <span className="text-red-400">!</span>}
        <span className="font-semibold">{translateNow("multimodal.queryWorker.title")}</span>
        <span className="font-mono text-[10px] text-cyan-300/60">#{taskId}</span>
        <span className="ml-auto text-[10px] text-muted-foreground/70">
          {active ? translateNow("multimodal.queryWorker.working") : status === "complete" ? translateNow("multimodal.queryWorker.completed") : status || translateNow("multimodal.queryWorker.ended")}
        </span>
      </div>
      <div className="mt-1 text-[10px] leading-snug text-cyan-100/55">
        {translateNow("multimodal.queryWorker.subtitle")}
      </div>
      {visible.length === 0 ? (
        <div className="mt-1.5 animate-pulse text-muted-foreground/70">{translateNow("multimodal.queryWorker.waitingProgress")}</div>
      ) : (
        <div className="mt-2 space-y-1 border-l border-cyan-400/25 pl-2">
          {visible.map((step, idx) => {
            const latest = idx === visible.length - 1;
            const askTimeFrames = step.phase.startsWith("started:");
            const ocrEvidence = step.phase.startsWith("ocr_evidence:");
            return (
              <div key={step.id} className="relative rounded bg-background/25 px-2 py-1.5">
                <span className={`absolute -left-[13px] top-2.5 h-1.5 w-1.5 rounded-full ${
                  step.status === "error" ? "bg-red-400"
                    : step.status === "complete" ? "bg-emerald-400"
                      : latest && active ? "animate-pulse bg-cyan-300" : "bg-cyan-500/70"
                }`} />
                <div className="flex min-w-0 items-center gap-1.5">
                  <span className="shrink-0 font-medium text-cyan-100">{step.worker}</span>
                  <span className="min-w-0 flex-1 break-words text-foreground/85">{step.title}</span>
                  {step.taskRef && (
                    <span className="shrink-0 rounded border border-cyan-400/20 px-1 font-mono text-[9px] text-cyan-200/55">
                      {step.taskRef}
                    </span>
                  )}
                  <span className="shrink-0 font-mono text-[9px] text-muted-foreground/45">#{step.seq}</span>
                </div>
                {ocrEvidence && <QueryWorkerOcrEvidence step={step} />}
                {step.detail && step.detail.length > 180 && (
                  <details className="mt-1" open={latest && active}>
                    <summary className="cursor-pointer select-none break-words text-muted-foreground/80">
                      {`${step.detail.slice(0, 180)}…`}
                    </summary>
                    <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap break-words rounded border bg-black/20 p-1.5 text-[10px] text-foreground/75">
                      {step.detail}
                    </pre>
                  </details>
                )}
                {step.detail && step.detail.length <= 180 && (
                  <div className="mt-1 whitespace-pre-wrap break-words text-muted-foreground/80">
                    {step.detail}
                  </div>
                )}
                {!!step.metrics?.length && (
                  <div className="mt-1.5 flex flex-wrap gap-1">
                    {step.metrics.map((metric, i) => (
                      <span key={`${metric}-${i}`} className="rounded border border-cyan-400/20 bg-cyan-400/5 px-1.5 py-0.5 font-mono text-[9px] text-cyan-100/70">
                        {metric}
                      </span>
                    ))}
                  </div>
                )}
                {!!step.plannedTools?.length && (
                  <div className="mt-1.5 space-y-1">
                    <div className="text-[10px] text-cyan-200/70">
                      {step.callState === "called" ? translateNow("multimodal.queryWorker.actualCall") : translateNow("multimodal.queryWorker.plannedCall")}
                    </div>
                    {step.plannedTools.map((tool, i) => (
                      <details
                        key={`${tool.name || "tool"}-${i}`}
                        open={step.callState === "called"}
                        className="rounded border border-cyan-400/15 bg-black/10 px-2 py-1"
                      >
                        <summary className="cursor-pointer select-none break-words text-foreground/80">
                          <span className="font-medium text-cyan-100">{tool.name || "memory tool"}</span>
                          {tool.args ? <span className="text-muted-foreground"> · {argPreview(tool.args)}</span> : null}
                          {tool.anchor ? (
                            <span className="text-muted-foreground/70">
                              {` · anchor=${tool.anchor}${tool.anchor_ts != null ? ` @${formatTraceTime(tool.anchor_ts)}` : ""}`}
                            </span>
                          ) : null}
                        </summary>
                        {tool.args && (
                          <pre className="mt-1 max-h-32 overflow-auto whitespace-pre-wrap break-words rounded bg-black/20 p-1.5 text-[10px] text-foreground/70">
                            {JSON.stringify(tool.args, null, 2)}
                          </pre>
                        )}
                      </details>
                    ))}
                  </div>
                )}
                {!!step.toolResults?.length && (
                  <div className="mt-1.5 space-y-1">
                    <div className="text-[10px] text-emerald-200/70">{translateNow("multimodal.queryWorker.toolReturned")}</div>
                    {step.toolResults.map((tool, i) => (
                      <details
                        key={`${tool.name || "tool-result"}-${i}`}
                        open
                        className="rounded border border-emerald-400/15 bg-emerald-400/5 px-2 py-1"
                      >
                        <summary className="cursor-pointer select-none break-words text-foreground/80">
                          <span className="font-medium text-emerald-200">{tool.name || "memory tool"}</span>
                          {tool.args ? <span className="text-muted-foreground"> · {argPreview(tool.args)}</span> : null}
                          {tool.obs_len != null ? <span className="text-muted-foreground/70"> · {translateNow("multimodal.queryWorker.chars", tool.obs_len)}</span> : null}
                          {tool.elapsed_sec != null ? <span className="text-muted-foreground/70"> · {tool.elapsed_sec.toFixed(2)}s</span> : null}
                          {tool.frame_ids?.length ? <span className="text-muted-foreground/70"> · {translateNow("multimodal.queryWorker.frames", tool.frame_ids.length)}</span> : null}
                          {tool.cache_hit ? <span className="text-amber-200/70"> · cache hit</span> : null}
                          {tool.anchor ? (
                            <span className="text-muted-foreground/70">
                              {` · anchor=${tool.anchor}${tool.anchor_ts != null ? ` @${formatTraceTime(tool.anchor_ts)}` : ""}`}
                            </span>
                          ) : null}
                        </summary>
                        {tool.args && (
                          <pre className="mt-1 max-h-28 overflow-auto whitespace-pre-wrap break-words rounded bg-black/20 p-1.5 text-[10px] text-foreground/65">
                            {JSON.stringify(tool.args, null, 2)}
                          </pre>
                        )}
                        {tool.obs_summary && (
                          <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap break-words rounded border border-emerald-400/10 bg-black/20 p-1.5 text-[10px] text-foreground/75">
                            {tool.obs_summary}
                          </pre>
                        )}
                        {!!tool.frame_ids?.length && (
                          <div className="mt-1 break-all font-mono text-[9px] text-muted-foreground/60">
                            frame_ids: {tool.frame_ids.join(", ")}
                          </div>
                        )}
                        {!!tool.evidence_segments?.length && (
                          <div className="mt-1 flex flex-wrap gap-1">
                            {tool.evidence_segments.slice(0, 12).map((segment, idx) => {
                              const label = evidenceSegmentLabel(segment);
                              if (!label) return null;
                              return (
                                <span
                                  key={`${label}-${idx}`}
                                  title={segment.preview || label}
                                  className="rounded border border-amber-300/20 bg-amber-300/5 px-1.5 py-0.5 font-mono text-[9px] text-amber-100/80"
                                >
                                  {label}{segment.frame_ids?.length
                                    ? ` · ${translateNow("multimodal.queryWorker.frames", segment.frame_ids.length)}`
                                    : ""}
                                </span>
                              );
                            })}
                          </div>
                        )}
                        {!!tool.source_urls?.length && (
                          <div className="mt-1 space-y-0.5 text-[9px] text-cyan-200/65">
                            {tool.source_urls.slice(0, 5).map((url) => (
                              <a key={url} href={url} target="_blank" rel="noreferrer" className="block truncate hover:underline">
                                {url}
                              </a>
                            ))}
                          </div>
                        )}
                      </details>
                    ))}
                  </div>
                )}
                {!!step.frames?.length && (
                  <div className="mt-1.5">
                    <div className="mb-1 text-[10px] text-cyan-200/70">
                      {askTimeFrames
                        ? translateNow("multimodal.queryWorker.frozenInputTitle")
                        : translateNow("multimodal.queryWorker.recallEvidence")}
                    </div>
                    <div className={askTimeFrames
                      ? "grid grid-cols-1 gap-1 sm:grid-cols-3"
                      : "grid grid-cols-2 gap-1 sm:grid-cols-4"}
                    >
                      {step.frames.slice(0, 8).map((fr, i) => {
                        const b64 = fr.thumb_b64 || fr.jpeg_b64 || "";
                        const usable = b64 && !b64.startsWith("<omitted");
                        const dataUrl = usable ? `data:image/jpeg;base64,${b64}` : "";
                        return (
                          <figure key={`${fr.frame_id || fr.ts || i}-${i}`} className="overflow-hidden rounded border border-cyan-400/20 bg-black/20">
                            {usable ? (
                              <a
                                href={dataUrl}
                                target="_blank"
                                rel="noreferrer"
                                title={translateNow("multimodal.queryWorker.clickToEnlarge")}
                                className="block"
                              >
                                <img
                                  src={dataUrl}
                                  alt={fr.frame_id || `${askTimeFrames ? "ask-time input" : "recall evidence"} ${i + 1}`}
                                  className="h-20 w-full cursor-zoom-in object-contain"
                                />
                              </a>
                            ) : (
                              <div className="flex h-20 items-center justify-center text-[9px]">no thumbnail</div>
                            )}
                            <figcaption className="truncate px-1 py-0.5 font-mono text-[9px] text-cyan-200/70">
                              {fr.frame_id || `${askTimeFrames ? translateNow("multimodal.queryWorker.inputFrame") : "frame"} ${i + 1}`}
                              {fr.ts != null ? ` · ${formatTraceTime(fr.ts)}` : ""}
                              {fr.source_type ? ` · ${fr.source_type}` : ""}
                            </figcaption>
                          </figure>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
});

// Tool-call arguments, rendered as key/value rows inside an expanded tool row.
//
// Why this exists: the collapsed line only has room for a one-line preview, so
// a call like `computer_use` used to be a dead end — the user could see the
// tool ran but never what it was told to do. The backend now ships classified
// args on every tool.start (see agent.display.describe_arg_fields), and this
// panel is where they land.
//
// Privacy is enforced BACKEND-side, not here: `freeform` fields (message
// bodies, file contents) arrive as a character count and `credential` fields
// (password/token/ssn/…) as a bare key, both with no value attached, so there
// is nothing to accidentally render. This component therefore never has to
// decide what is safe — it just can't display what it wasn't given.
export const ToolArgsPanel = memo(function ToolArgsPanel({
  fields,
}: {
  fields?: ToolArgField[];
}) {
  useLocaleRevision();  // labels come from translateNow — see AsrBar
  if (!fields || !fields.length) return null;
  return (
    <div className="mt-1">
      <div className="mb-0.5 text-[10px] text-muted-foreground/70">{translateNow("multimodal.misc.toolArgsLabel")}</div>
      <div className="space-y-px rounded border bg-background/50 p-2">
        {fields.map((f, i) =>
          // `elided` carries no key (it is the "+N more" tail, not a field), so
          // it can't use f.key as its React key and renders as a bare note.
          f.kind === "elided" ? (
            <div key="elided" className="font-mono text-[10px] leading-relaxed text-muted-foreground/50 italic">
              {translateNow("multimodal.misc.moreFields", f.count)}
            </div>
          ) : (
            <div key={`${f.key}-${i}`} className="flex gap-2 font-mono text-[10px] leading-relaxed">
              <span className="shrink-0 text-violet-300/80">{f.key}</span>
              {f.kind === "credential" ? (
                // 凭证类字段连长度都不发 —— 密码的长度本身就是线索。
                <span className="text-amber-400/70 italic">{translateNow("multimodal.misc.redactedCredentials")}</span>
              ) : f.kind === "freeform" ? (
                // 正文类字段只有长度 —— 后端不发内容, 这里也就无从渲染。
                <span className="text-muted-foreground/60 italic">{translateNow("multimodal.misc.charsNotShown", f.chars)}</span>
              ) : f.kind === "shape" ? (
                <span className="text-muted-foreground/80">{translateNow("multimodal.misc.items", f.count)}</span>
              ) : (
                <span className="min-w-0 break-all text-foreground/80">{f.value}</span>
              )}
            </div>
          )
        )}
      </div>
    </div>
  );
});

// ── Segment summary ─────────────────────────────────────────────────────────
// What a step block DID, in one line, derived purely from the tool calls — so
// it works with thinking mode OFF, where there is no reasoning to show and the
// header was the content-free "Processing" no matter what ran.
//
// Deliberately family-based rather than per-tool: the backend's `_tool_summary`
// only produces text for `web_search` and `web_extract` (2 of ~74 tools), so
// per-tool summaries are empty for almost everything. Families are derived from
// the tool NAME, which every call has.
// Full literal key paths, not `multimodal.misc.${family}` — src/i18n/
// key-integrity.test.ts scans for string-literal keys at the call site, so a
// template literal would silently opt these keys out of the guard that catches
// a key missing from one locale. (These constants are still outside its regex;
// the work-segments test asserts every one of them resolves in both locales.)
const STEP_FAMILY_KEYS = {
  read: "multimodal.misc.stepRead",
  edit: "multimodal.misc.stepEdit",
  run: "multimodal.misc.stepRun",
  search: "multimodal.misc.stepSearch",
  browse: "multimodal.misc.stepBrowse",
  look: "multimodal.misc.stepLook",
} as const;

const STEP_OTHER_KEY = "multimodal.misc.stepCall";
const STEP_FAILED_KEY = "multimodal.misc.stepFailed";

type StepFamily = keyof typeof STEP_FAMILY_KEYS;

function stepFamilyOf(toolName: string): StepFamily | null {
  const n = toolName.toLowerCase();

  if (/^(read_file|view_file|read|cat|list_dir|glob|find)/.test(n)) return "read";
  if (/^(edit_file|write_file|apply_patch|patch|str_replace)/.test(n)) return "edit";
  if (/^(terminal|bash|shell|execute_code|run)/.test(n)) return "run";
  if (/^(web_search|search|grep|ripgrep|recall)/.test(n)) return "search";
  if (/^(browser_|web_extract|fetch|open_url)/.test(n)) return "browse";
  if (/^(get_current_frame|query_multimodal|show_memory_frame|screenshot)/.test(n)) return "look";

  return null;
}

/** One-line "what this step did", e.g. `读了 2 个文件 · 执行了 1 条命令`. */
export function summarizeStep(
  items: readonly { kind?: string; toolName?: string; toolDone?: boolean; toolIsError?: boolean }[],
  tr: (key: string, n: number) => string,
): string {
  const counts = new Map<StepFamily | "other", number>();
  let failed = 0;

  for (const it of items) {
    if (it.kind !== "tool" || !it.toolName) continue;
    if (it.toolIsError) failed += 1;
    const fam = stepFamilyOf(it.toolName) ?? "other";
    counts.set(fam, (counts.get(fam) ?? 0) + 1);
  }

  const parts: string[] = [];

  // Stable, meaningful order — not Map insertion order, so the same set of
  // calls always reads the same way.
  for (const fam of ["read", "edit", "run", "search", "browse", "look"] as const) {
    const n = counts.get(fam);
    if (n) parts.push(tr(STEP_FAMILY_KEYS[fam], n));
  }

  const other = counts.get("other");
  if (other) parts.push(tr(STEP_OTHER_KEY, other));
  if (failed) parts.push(tr(STEP_FAILED_KEY, failed));

  return parts.join(" · ");
}

// Grouped "background" card (tools + status). Memoized so it only re-renders
// when its `items` array identity changes (the parent rebuilds `rows` from a
// new `messages` array only when a message actually changes).
// Exported for the render test that pins disclosure nesting depth.
export const BgBlock = memo(function BgBlock({ items, thinking }: {
  items: ChatMsg[];
  // Reasoning that preceded THIS segment's tool calls (see buildRows). Kept on
  // the segment so "what it was thinking" stays next to "what it then did" —
  // previously reasoning was dropped outright once the turn stopped streaming,
  // so a finished turn showed tool names with no rationale anywhere.
  thinking?: string;
  // `seg` (round index) is still carried on the Row and compared by the memo
  // below, but is intentionally NOT rendered: with one segment per turn being
  // the common case a constant "#1" was pure noise, and the `#` read as an id
  // next to the neighbouring `#req_…` request ids.
}) {
  useLocaleRevision();  // labels come from translateNow — see AsrBar
  const running = items.some((it) => it.kind === "tool"
    && (!it.toolDone || it.workerStatus === "running"));
  // ★ 计时器搬到这里 (从上方那条纯思考行迁来): 工具卡一出就吞掉思考行后, 卡片自己
  //   就是这段耗时的唯一归属。key 绑本段第一个条目 id → 走 useElapsedSeconds 的模块
  //   注册表, 虚拟滚动卸载/重挂不会倒退; running=false 后停表, 定格总时长。
  const timerKey = items[0]?.id ? `bg:${items[0].id}` : undefined;
  const elapsed = useElapsedSeconds(running, timerKey);
  const containerRef = useRef<HTMLDivElement>(null);
  const [allExpanded, setAllExpanded] = useState(false);
  // Collapsed by default: the header line already names the step count, so the
  // rationale is one click away without spending vertical space every turn.
  const [thinkingOpen, setThinkingOpen] = useState(false);

  const hasExpandable = items.some(
    (it) => it.kind === "tool" && (it.toolDetail || (it.recallTrace && it.recallTrace.length) || it.recallFindings || (it.toolArgs && it.toolArgs.length))
  );

  // Drives the native <details> children directly rather than lifting each
  // row's open state into React — the rows are rendered in a .map() so per-row
  // state would need a component split, and the DOM is the source of truth for
  // individually-toggled disclosures anyway.
  const toggleAll = useCallback(() => {
    const el = containerRef.current;
    if (!el) return;
    const next = !allExpanded;
    el.querySelectorAll("details").forEach((d) => { d.open = next; });
    setAllExpanded(next);
  }, [allExpanded]);

  // "Read 2 files · Ran 1 command" instead of a bare "Processing". Derived from
  // the calls themselves, so it carries real information with thinking mode OFF
  // too. While the step is still running we keep "Processing" — the counts are
  // not final yet and would visibly churn.
  const anyFailed = items.some((it) => it.kind === "tool" && it.toolIsError);
  const stepLabel = running
    ? translateNow("multimodal.misc.processing")
    : summarizeStep(items, (key, n) => translateNow(key, n))
      || translateNow("multimodal.misc.processing");

  return (
    <div className="ml-9 rounded-md border border-dashed border-violet-400/40 bg-violet-400/5 px-2.5 py-1.5 text-[11px]">
      <div className="mb-1 flex items-center gap-1.5 font-medium text-violet-400">
        {/* A step containing a failed call must not report itself as all-clear —
            the header is what the user scans when collapsed. */}
        {running
          ? <span className="inline-block animate-spin">◌</span>
          : anyFailed
            ? <span className="text-red-400">✕</span>
            : <span className="text-emerald-500">✓</span>}
        {stepLabel}
        {elapsed > 0 && (
          <span className="font-normal tabular-nums text-violet-400/70">
            {formatElapsed(elapsed)}
          </span>
        )}
        {hasExpandable && (
          <span
            className="ml-auto cursor-pointer select-none text-[11px] font-medium text-violet-400 hover:text-violet-300"
            onClick={toggleAll}
          >
            {allExpanded
              ? translateNow("multimodal.misc.collapseAll")
              : translateNow("multimodal.misc.expandAll")}
          </span>
        )}
      </div>
      {/* 💭 This segment's rationale, above the calls it produced.
          Raw chain-of-thought is long, usually English even in a zh UI, and is
          NOT an answer — it must never compete with the prose or push the tool
          rows off screen. So: always exactly one line when collapsed (which is
          the default), and a bounded scroll area when opened. */}
      {thinking && (
        <div className="mb-1">
          <button
            type="button"
            aria-expanded={thinkingOpen}
            className="flex w-full cursor-pointer items-center gap-1 text-left text-[11px] text-muted-foreground/80 hover:text-muted-foreground"
            onClick={() => setThinkingOpen((v) => !v)}
          >
            <span className="shrink-0">💭</span>
            {/* Keep the one-line preview even when open, so the row never
                collapses to a bare chevron and the header stays a fixed height
                (an empty label here is why the block looked headless). */}
            <span className="min-w-0 flex-1 truncate">
              {thinking.replace(/\s+/g, " ")}
            </span>
            <span className="shrink-0 text-muted-foreground/50">{thinkingOpen ? "▾" : "▸"}</span>
          </button>
          {thinkingOpen && (
            <div className="mt-0.5 max-h-40 overflow-auto whitespace-pre-wrap break-words border-l border-violet-400/25 pl-2 text-[11px] leading-relaxed text-muted-foreground/85">
              {thinking}
            </div>
          )}
        </div>
      )}
      <div className="space-y-0.5" ref={containerRef}>
        {items.map((it) => {
          if (it.kind === "status") {
            return (
              <div key={it.id} className="italic text-muted-foreground/80">
                {it.text}
              </div>
            );
          }
          // tool entry — collapsible if it has detail
          const head = (
            <>
              {/* A failure must not wear the success checkmark. */}
              {it.toolIsError
                ? <span className="text-red-400">✕</span>
                : it.toolDone
                  ? <span className="text-emerald-500">✓</span>
                  : <span className="inline-block animate-spin text-violet-400">◌</span>}
              {" "}
              {/* 未完成的工具: 名字 + 参数摘要一起走 .shimmer 流光, 与上方思考行同一
                  套动效语言。完成后换成静态实色 + ✓ + 耗时, 一眼区分"在跑"和"跑完"。
                  这里不挂计时器: 行在 map 里, 每行一个 hook 需要拆组件, 而完成态本就
                  显示 toolDurationMs, 运行态的时长由上方思考行的计时器代表。 */}
              <span className={
                it.toolIsError ? "font-medium text-red-300"
                  : it.toolDone ? "font-medium text-foreground/90"
                    : "shimmer font-medium text-violet-300"
              }>
                {it.toolName}
              </span>
              {it.toolCtx ? (
                <span className={it.toolDone ? "text-muted-foreground" : "shimmer text-violet-300/80"}>
                  {" · "}{it.toolCtx.slice(0, 80)}
                </span>
              ) : null}
              {/* Why it failed, inline — no expand needed. This is the single
                  highest-value line on a failed row and it used to be hidden. */}
              {it.toolError ? (
                <span className="text-red-400"> ↳ {it.toolError.split("\n")[0].slice(0, 160)}</span>
              ) : it.toolDone && it.toolSummary ? (
                <span className="text-muted-foreground"> ↳ {it.toolSummary.slice(0, 120)}</span>
              ) : null}
              {it.toolDurationMs != null ? <span className="text-muted-foreground/60"> · {(it.toolDurationMs / 1000).toFixed(1)}s</span> : null}
            </>
          );
          const hasRecallTrace = !!(it.recallTrace && it.recallTrace.length) || !!it.recallFindings;
          const hasArgs = !!(it.toolArgs && it.toolArgs.length);
          return (
            <div key={it.id} className="break-words text-muted-foreground">
              {it.toolDetail || hasRecallTrace || hasArgs ? (
                <details>
                  <summary className="cursor-pointer select-none break-words">
                    {head}
                    {hasRecallTrace ? (
                      <span className="ml-2 rounded border border-emerald-400/30 px-1.5 py-0.5 text-[10px] text-emerald-300">
                        {translateNow("multimodal.misc.expandRecallTrace")}
                      </span>
                    ) : null}
                  </summary>
                  {/* 入参放最上面: 用户点开一个工具行, 第一个问题总是"它是拿什么参数
                      调的", 而不是"它返回了什么"。 */}
                  <ToolArgsPanel fields={it.toolArgs} />
                  <RecallTracePanel trace={it.recallTrace} findings={it.recallFindings} />
                  {/* ★ 输出直接平铺, 不再套第二层 "Raw tool result" <details>: 用户已经
                      点开一层才看到它, 再折一层等于两次点击才见内容。只有 Recall 那种
                      "结构化轨迹 + 原始输出"并存的工具才需要区分两块 —— 此时给原始输出
                      加个轻标题即可, 层级仍是一层。 */}
                  {it.toolDetail && (
                    <div className="mt-1">
                      {hasRecallTrace && (
                        <div className="mb-0.5 text-[10px] text-muted-foreground/70">{translateNow("multimodal.misc.rawOutputLabel")}</div>
                      )}
                      <pre className="max-h-48 overflow-auto whitespace-pre-wrap break-words rounded border bg-background/50 p-2">{it.toolDetail}</pre>
                    </div>
                  )}
                </details>
              ) : <div>{head}</div>}
              {it.workerTaskId && (
                <QueryWorkerProgressPanel
                  taskId={it.workerTaskId}
                  status={it.workerStatus}
                  steps={it.workerProgress || []}
                />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}, (a, b) =>
  // ★ 性能(#6): 按 items 逐元素引用比较 (不是数组引用)。rows useMemo 每次都 new 一个
  //   items 数组, 但纯 chat 流式期间 tool/status 消息对象 identity 不变 → 内容相同的
  //   bg 块在这里判等、跳过重渲染。只有本块的 tool/status 真变 (新增/patch) 才重渲染。
  //   `thinking` 也要比 —— 否则 interleaved reasoning 折进本段后画面不更新。
  //   (`seg` 不再是 prop: 段号已不渲染, 比较它也就没有意义了。)
  a.thinking === b.thinking &&
  a.items.length === b.items.length && a.items.every((it, i) => it === b.items[i]),
);

// Stable no-op play handler for contexts that render ChatBubble without TTS
// (the deep-research sub-window). A module-level constant keeps ChatBubble's
// memo intact — an inline `() => {}` would be a new identity every render and
// force every sub-window bubble to re-render on each parent tick.
const NOOP_PLAY = (_text: string) => { /* no TTS in sub-window */ };

// Inline clarify bubble: a blocking clarify.request from a tool, rendered in
// the chat waterfall as a question + option buttons (Claude-Code-desktop
// style). Once answered, the buttons freeze and the picked answer is shown.
// Memoized so unrelated stream ticks don't re-render every clarify row.
const ClarifyBubble = memo(function ClarifyBubble({
  m, onAnswer,
}: {
  m: ChatMsg;
  onAnswer: (reqId: string, answer: string) => void;
}) {
  useLocaleRevision();  // labels come from translateNow — see AsrBar
  const reqId = m.clarifyReqId || "";
  const answered = m.clarifyAnswer !== undefined;
  const choices = m.clarifyChoices || [];
  const openEnded = choices.length === 0;
  // Local draft for open-ended (no-choices) clarify — the answer is free text.
  const [draft, setDraft] = useState("");
  const submitText = () => {
    const t = draft.trim();
    if (!t) return;
    onAnswer(reqId, t);
    setDraft("");
  };
  // ★ Once answered, the question box + option buttons collapse into ONE compact
  // system line ("✓ 已选择：<choice>"). This is nicer than freezing the dialog
  // in place: the prompt disappears and the choice reads as a settled step in
  // the conversation. (The answer already reached the tool via clarify.respond;
  // this is purely the front-end presentation.)
  if (answered) {
    return (
      <div className="flex gap-2">
        <div className="flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-full bg-muted text-xs text-foreground">
          ✓
        </div>
        <div className="min-w-0 flex-1 self-center text-xs text-muted-foreground">
          {translateNow("multimodal.clarify.selected")}<span className="text-foreground">{m.clarifyAnswer || translateNow("multimodal.clarify.emptySelection")}</span>
        </div>
      </div>
    );
  }
  return (
    <div className="flex gap-2">
      <div className="flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-full bg-amber-500 text-xs font-semibold text-black">
        ?
      </div>
      <div className="min-w-0 flex-1">
        <div className="mb-1 text-xs font-medium text-amber-300">{translateNow("multimodal.clarify.needsConfirmation")}</div>
        <div className="rounded-md border border-amber-400/40 bg-amber-400/5 p-2.5">
          <div className="mb-2 whitespace-pre-wrap text-sm text-amber-100">{m.clarifyQuestion}</div>
          {/* Only the unanswered state renders here — the answered case is
              handled by the compact early return above. */}
          {!openEnded ? (
            <div className="flex flex-wrap gap-1.5">
              {choices.map((c) => (
                <Button key={c} size="sm" outlined
                  onClick={() => onAnswer(reqId, c)}>
                  {c}
                </Button>
              ))}
            </div>
          ) : (
            <div className="flex items-center gap-1.5">
              <input
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter") submitText(); }}
                placeholder={translateNow("multimodal.clarify.placeholder")}
                className="min-w-0 flex-1 rounded border bg-background px-2 py-1 text-xs" />
              <Button size="sm" onClick={submitText}>{translateNow("multimodal.clarify.submit")}</Button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
});

// 深度回传气泡: 头部行 (🔬 label·第N段 + [时段区间] + 时间, #事件id 右对齐) + 正文受控折叠。
// 默认折叠 (三角 ▸ + 正文首行 line-clamp-1 省略号); 点三角展开 (▾ + 全文)。与桌面端一致。
const WatcherReportBubble = memo(function WatcherReportBubble({
  m, onPlay,
}: { m: ChatMsg; onPlay?: (text: string) => void }) {
  const { t } = useI18n();
  const [open, setOpen] = useState(false);
  const body = m.text.trim();
  // brief 形如 "腾讯会议新消息监控 · 第1段" → 拆成 标签(进紫框) + 段号(做正文折叠头)。
  const rawBrief = m.brief || t.multimodal.deepAnalysis.title;
  const sepIdx = rawBrief.lastIndexOf(" · ");
  const label = sepIdx >= 0 ? rawBrief.slice(0, sepIdx) : rawBrief;
  const segment = sepIdx >= 0 ? rawBrief.slice(sepIdx + 3) : "";
  // 正文第一行预览 (去掉 markdown 标题/列表符号), 折叠时作为 "第N段" 后的灰字提示。
  const firstLine = (body.split("\n").find((l) => l.trim()) || "")
    .replace(/^#+\s*/, "").replace(/^[-*>]\s*/, "").trim();
  // 与 monitor 气泡同构: [笔记本头像] [紫框标签] [时间] [播放] ...... [#事件id 右对齐]
  return (
    <div className="flex gap-2">
      <div className="flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-full bg-violet-500 text-white">
        <NotebookPen className="h-4 w-4 -rotate-12" />
      </div>
      <div className="min-w-0 flex-1">
        <div className="mb-1 flex items-center gap-1.5 text-xs text-muted-foreground">
          <span className="font-medium text-violet-300">{label}</span>
          {m.createdAt != null && (
            <span className="tabular-nums text-muted-foreground/60">{fmtClock(m.createdAt)}</span>
          )}
          {body && onPlay && (
            <button
              onClick={() => onPlay(body)}
              title={t.multimodal.chat.playVoice}
              className="ml-1 inline-flex items-center gap-1 rounded border border-border/50 px-1.5 py-0.5 text-[10px] text-muted-foreground hover:border-primary/60 hover:text-primary">
              <Play className="h-3 w-3" /> {t.multimodal.chat.play}
            </button>
          )}
          {m.requestId && (
            <span className="ml-auto font-mono text-muted-foreground/50">#{m.requestId}</span>
          )}
        </div>
        {/* 正文卡: 折叠时 "第N段" + 一行灰色正文预览 (≤75% 宽, 溢出 " ..." 收尾),
            右上角 "点击展开" 图标; 展开后显示该段全文。 */}
        <div className="rounded-md border-l-2 border-violet-400/50 bg-violet-950/30 px-3 py-2">
          <div
            className="flex w-full cursor-pointer select-none items-center gap-1.5 text-left text-sm"
            onClick={() => setOpen((o) => !o)}
          >
            <span className="shrink-0 font-medium text-violet-200">{segment || t.multimodal.chat.viewAnalysis}</span>
            {m.deepRange && (
              <span className="shrink-0 tabular-nums text-xs font-normal text-violet-300/70">{m.deepRange}</span>
            )}
            {!open && firstLine && (
              // ≤75% 宽, 溢出用 " ..." 收尾 (overflow-hidden 不带原生 "…", 显式三点)。
              <span className="flex min-w-0 max-w-[75%] items-baseline text-xs text-muted-foreground/70">
                <span className="min-w-0 overflow-hidden whitespace-nowrap">{firstLine}</span>
                <span className="shrink-0">{" ..."}</span>
              </span>
            )}
            {/* 展开/收起 小胶囊按钮 (对齐头部 "▷ 播放" 形态), 紫色调。 */}
            <button
              type="button"
              onClick={(e) => { e.stopPropagation(); setOpen((o) => !o); }}
              title={open ? t.multimodal.chat.collapse : t.multimodal.chat.expandFull}
              className="ml-auto inline-flex shrink-0 items-center gap-1 rounded border border-violet-400/50 px-1.5 py-0.5 text-[10px] text-violet-300 hover:border-violet-300 hover:text-violet-200">
              <ChevronDown className={`h-3 w-3 transition-transform ${open ? "rotate-180" : ""}`} />
              {open ? t.multimodal.chat.collapse : t.multimodal.chat.expandFull}
            </button>
          </div>
          {open && (
            <div className="mt-2 border-t border-violet-400/20 pt-2">
              <Markdown content={body} streaming={false} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
});

// 判定"纯思考态" chat 行: assistant 流式中、还没有正文 (且非错误/监控/深度/QueryWorker)。
// 这一形态只渲染一行 💭 状态 (ThinkingLine), 不出完整 AssistantMessage。renderRow 与
// ChatBubble 共用此判据, 保证"是否走思考行"两处完全一致。
function isPureThinkingChat(m: ChatMsg): boolean {
  return (
    m.role === "assistant" && !!m.streaming && !m.text?.trim() && !m.isError &&
    !m.monitorId && !m.deepResearch && m.subRole !== "query_worker"
  );
}

// ── 思考行 ↔ 处理过程卡 的分工 ────────────────────────────────────────────────
// 流式一轮里两者【同时】呈现, 各司其职, 不再互相顶替:
//   * 下方"处理过程"卡 (BgBlock): 工具的全部细节 —— 运行中是 ◌ 旋转 + 流光名 +
//     参数摘要, 完成后换 ✓ + 耗时 + 结果摘要, 外加本段的 💭 思考块。
//   * 上方 💭 思考行: 一行状态 + 计时器。BgBlock 的逐行工具【故意不挂计时器】
//     (见其注释: "运行态的时长由上方思考行的计时器代表"), 所以这一行是卡片的
//     计时伴侣, 两者本就设计为共存。
//
// ★ 曾经这里有一套"二选一"仲裁: 工具运行期间隐藏卡片, 只留一行 💭。它带来两个
//   问题 —— (1) 工具正在跑的时候用户看不到工具框, 得等它【跑完】才出卡, 迟了一整个
//   工具的时长; (2) 隐藏条件是"本 bg 含 tool", 而顶替条件是"有【正在运行】的 tool",
//   二者在"工具已完成、正文未到"的窗口里不一致 → 两边都不出, 只剩 "Waiting
//   response…"。仲裁本身就是这两个 bug 的共同来源, 所以整个删掉: 卡片永不让位,
//   tool.start 一到就出卡, 也不存在"谁该让谁"的不一致。
export type TurnToolPresentation = {
  /** 本轮已产生 tool 条目 (ChatBubble 用它决定不出空气泡)。 */
  inToolCall: boolean;
  /** 当前运行中工具的一句话活动; "" = 无运行中工具 → 思考行回落到 reasoning 摘要。 */
  toolActivity: string;
};

export function deriveTurnToolPresentation(
  items: readonly ChatMsg[] | undefined,
): TurnToolPresentation {
  const none: TurnToolPresentation = { inToolCall: false, toolActivity: "" };
  if (!items?.length) return none;
  const tools = items.filter((it) => it.kind === "tool");
  if (!tools.length) return none;

  // 最新的运行中工具胜出 (从尾部找第一个 !toolDone), 对齐 desktop
  // CurrentActivityLine 的"最新动作胜出"。
  let running: ChatMsg | undefined;
  for (let k = tools.length - 1; k >= 0; k--) {
    if (!tools[k].toolDone) { running = tools[k]; break; }
  }
  const toolActivity = running?.toolName
    ? (running.toolCtx ? `${running.toolName} · ${running.toolCtx.slice(0, 60)}` : running.toolName)
    : "";

  return { inToolCall: true, toolActivity };
}

// 一行"思考中"提示行 —— 事件驱动状态机 (跟 desktop CurrentActivityLine 同思路)。
//   0. streaming + 有正在运行的工具 (toolActivity)   → 显示工具活动 ("toolName · ctx")
//   1. streaming + awaitingFirstDelta               → "Waiting response…"
//   2. streaming + hasReasoning + reasoningSummary  → 显示 aux 生成的 ~10 字 label
//   3. streaming + hasReasoning + 无 summary + 有 reasoning 正文 → 原文最后 1 行滚动
//   4. streaming + hasReasoning + 无 summary + 无正文 (只是 Luna 那类 signal-only)
//                                                    → "Thinking…"
//   5. 兜底: 流开始 3s 都还没任何 delta                → 自动切 "Thinking…"
// ★ 工具活动优先于思维链: 工具在跑时显示工具, 工具间隙 (无运行中工具) 回落到 reasoning
//   摘要 → 二者随本轮进展【交替显示】(对齐 desktop CurrentActivityLine "最新动作胜出")。
// ★ 纵轴对齐: 用隐形头像列占位 (h-7 w-7) + gap-2 + px-3, 让 💭 文字左缘与 user/assistant
//   气泡正文左缘精确对齐 —— 不再顶到 UserMessage 边界前面。
// 首个正文 delta 落地后, 父组件不再渲染这一行 (整体消失)。
const ThinkingLine: FC<{ msg: ChatMsg; toolActivity?: string }> = ({ msg, toolActivity }) => {
  const [fallbackThinking, setFallbackThinking] = useState(false);
  // 计时器 key 绑消息 id: 本轮内 label 在 工具↔思维链 之间切换【不】重置总时长,
  // 换一轮 (新 assistant 消息) 才从 0 开始。本组件只在 streaming 时被渲染
  // (isPureThinkingChat), 所以恒 active —— 正文落地后父级直接不渲染这一行。
  const elapsed = useElapsedSeconds(true, `activity:${msg.id}`);
  useEffect(() => {
    if (!msg.awaitingFirstDelta) return;
    const timer = window.setTimeout(() => setFallbackThinking(true), 3000);
    return () => window.clearTimeout(timer);
  }, [msg.awaitingFirstDelta]);
  useEffect(() => {
    if (!msg.awaitingFirstDelta) setFallbackThinking(false);
  }, [msg.awaitingFirstDelta]);

  // ★ This line reports STATUS, not content. It used to scroll the tail of the
  //   raw reasoning text, which re-rendered on every 80ms delta flush — combined
  //   with the shimmer sweep and the pulsing emoji, a fast model made it visibly
  //   twitch several times a second. Unreadable as content AND useless as a
  //   "still alive" signal, which is all a one-line indicator can honestly be.
  //
  //   So: only stable labels here. The reasoning text itself is not lost — it is
  //   kept on the turn and shown, per work segment, in the collapsed 💭 block
  //   inside BgBlock (see buildRows), where it can be read at rest.
  //
  //   `reasoningSummary` is the one exception: it is an aux-model ~10-char label
  //   that changes at most once per reasoning segment, not per token.
  let label = "Waiting response…";
  if (toolActivity) {
    label = toolActivity;
  } else if (msg.hasReasoning) {
    label = msg.reasoningSummary || "Thinking…";
  } else if (fallbackThinking) {
    label = "Thinking…";
  }
  return (
    <div className="flex gap-2">
      <div className="h-7 w-7 flex-shrink-0" aria-hidden="true" />
      {/* 对齐基准 = 气泡【正文首字】的左缘 (不是 "You" 头部行的左缘), 所以这里用气泡
          自身的 px-3 (12px) 而非 0。再减 3px 是实测视觉微调: 12px 时 💭 的墨迹看起来
          比正文首字偏右一点 (emoji 字形自带左侧留白), 减 3px 后两者视觉左缘齐平。
          ★ 原先 label 与 emoji 之间靠 "💭 " 的尾随空格分隔, 现已改为 flex + gap-1.5
          (加计时器后需要 flex 布局), 该空格不再存在 —— 但 -3px 仍只补 emoji 自身的
          左侧留白, 故保持不变。改动这里请同时目视核对一次。 */}
      <div className="min-w-0 flex-1 pl-[calc(0.75rem-3px)] pr-3">
        {/* ★ 动效 (对齐 desktop CurrentActivityLine): 💭 脉冲 + label 流光 + 计时。
            label 用 .shimmer 让一道高光沿文字扫过 —— 工具执行/思考期间这一行是
            页面上唯一在动的元素, 用它告诉用户"没卡住, 正在跑"。此前只有 emoji
            在 pulse, 文字完全静止, 长工具 (curl / 大文件读) 看着像 UI 假死。
            flex + min-w-0 让 label 独占剩余宽度并 truncate, 计时器 shrink-0
            永不被挤掉。 */}
        <div className="mb-1 flex items-center gap-1.5 text-xs text-muted-foreground">
          <span className="shrink-0 animate-pulse">💭</span>
          <span className="shimmer min-w-0 flex-1 truncate">{label}</span>
          <span className="shrink-0 font-mono text-[10px] tabular-nums text-muted-foreground/55">
            {formatElapsed(elapsed)}
          </span>
        </div>
      </div>
    </div>
  );
};

// A single chat bubble (user / assistant / system / sub-agent). Memoized with
// the default shallow prop compare: `m` is a fresh object only when THAT
// message changes (delta append maps to a new object), and `model`/`onPlay`
// are stable refs. So a streaming bubble re-renders alone; idle bubbles are
// skipped — the parent can re-render freely (frame ticks, ctx updates) without
// re-diffing the whole list. Markdown inside stays memoized on top of this.
const ChatBubble = memo(function ChatBubble({
  m, model, onPlay, onReopenDeep, inToolCall, toolActivity,
}: {
  m: ChatMsg;
  model: string;
  onPlay?: (text: string) => void;   // undefined = 自动播报开着, 隐藏逐条 ▶
  onReopenDeep?: (rid: string) => void;
  // 由父级 rows 映射判断: 当前 streaming assistant 消息紧邻其后有 tool 条目。
  // ★ 不再让它翻到"完整空气泡"分支 —— 工具调用期间仍走思考行 (ThinkingLine),
  //   把工具活动写进 💭 位置, 不出空的 AssistantMessage; 待正文落地再出完整气泡。
  inToolCall?: boolean;
  // 当前正在运行的工具的一句话活动 (由父级从相邻 bg 行的运行中 tool 派生), 传给
  // ThinkingLine 作为最高优先级 label。无运行中工具时为空 → 回落到 reasoning 摘要。
  toolActivity?: string;
}) {
  const { t } = useI18n();
  // Monitor SPEAK bubbles are labelled with the short event label
  // (never "Assistant" / raw id) so the user sees what fired.
  const roleName = m.role === "user" ? "You"
    : m.monitorId ? (m.monitorLabel || t.multimodal.monitor.alert)
    : m.subRole === "query_worker" ? "QueryWorker"
    : m.threadback ? t.multimodal.deepAnalysis.title
    : m.role === "assistant" ? "Assistant" : "System";
  // Trim leading/trailing blank lines from the answer (the model often emits a
  // leading newline). Keep interior whitespace; while streaming, only trim the
  // start so the cursor doesn't jump.
  const body = m.streaming ? m.text.replace(/^\s+/, "") : m.text.trim();
  // ★ 丝滑修复: 不再对 body 叠加第二层节流。body 本身已由统一 flush (80ms) 限速,
  //   token 只在每次 flush 时批量并入 → Markdown 天然 ~12.5fps 重解析。再套一层
  //   120ms useThrottledValue 会与 flush 相位错开, 产生"一段段"拍频。直接用 body。
  // ── Watcher per-round report: 头部行 (🔬 label·第N段 + [时段区间] + 时间, #id 右对齐) +
  //    正文受控折叠 (默认折叠露第一行行末省略, 点三角展开全文)。Ephemeral, 不入 history。
  if (m.subRole === "watcher_report") {
    return <WatcherReportBubble m={m} onPlay={onPlay} />;
  }
  // ★ 纯思考态: assistant 流式中、还没有正文 → 只渲染一小行 💭 状态文字, 用隐形头像列
  //   占位与 UserMessage/正文气泡左缘对齐。工具卡/监控/深度/QueryWorker 各有形态均不走
  //   这条。★ inToolCall (本轮已产生 tool 条目) 时【也走这条】: 工具调用提示 (toolActivity)
  //   直接写进 💭 位置, 不再翻到"完整空气泡 + ▍"分支 (不出空 AssistantMessage); 待正文
  //   落地或本轮结束后思考行消失, 再由完整正文气泡一次性渲染。判据与 isPureThinkingChat
  //   一致 (见其定义), 保证 renderRow 的 spacing/派生与此处渲染同步。
  if (isPureThinkingChat(m)) {
    return <ThinkingLine msg={m} toolActivity={toolActivity} />;
  }
  return (
    <div className="flex gap-2">
      <div className={`flex h-7 w-7 flex-shrink-0 items-center justify-center rounded-full text-xs font-semibold ${
        m.subRole === "monitor" ? "bg-amber-500 text-black"
          : m.subRole === "router" ? "bg-violet-500 text-white"
          : m.subRole === "query_worker" ? "bg-cyan-400 text-black"
          : m.role === "user" ? "bg-emerald-500 text-black"
          : m.role === "assistant" ? "bg-sky-400 text-black"
          : "bg-muted text-foreground"}`}>
        {m.subRole === "monitor" ? "👁" : m.subRole === "router" ? "🔬"
          : m.subRole === "query_worker" ? "Q"
          : m.role === "user" ? "U" : m.role === "assistant" ? "A" : "i"}
      </div>
      <div className="min-w-0 flex-1">
        <div className="mb-1 flex items-center gap-1 text-xs text-muted-foreground">
          {/* 监控/深度气泡: 只显示 [事件tab #id] + 时间, 不显示角色名/模型名 (对齐桌面端形态)。
              普通 user/assistant: 角色名 + 时间 + 模型徽标。 */}
          {m.deepResearch ? (
            <Badge tone="outline"
              className={`border-violet-400/60 text-violet-300${
                m.requestId && onReopenDeep ? " cursor-pointer hover:bg-violet-400/15" : ""}`}
              title={m.requestId ? t.multimodal.deepAnalysis.reopenReadonly : ""}
              onClick={m.requestId && onReopenDeep ? () => onReopenDeep(m.requestId as string) : undefined}>
              {`🔬 ${m.brief || t.multimodal.deepAnalysis.title}${m.requestId ? ` #${m.requestId}` : ""}`}
            </Badge>
          ) : m.monitorId ? (
            <Badge tone="outline" className="border-amber-400/60 text-amber-300">
              {m.monitorLabel || t.multimodal.monitor.title}
            </Badge>
          ) : (
            roleName
          )}
          {m.createdAt != null && (
            <span className="tabular-nums text-muted-foreground/60">{fmtClock(m.createdAt)}</span>
          )}
          {/* 模型名只在普通 assistant 回复显示; 监控/深度气泡不显示。 */}
          {m.role === "assistant" && !m.isError && !m.monitorId
            && !m.deepResearch && m.subRole !== "query_worker" && model && (
            <Badge tone="secondary" className="ml-1">{model}</Badge>
          )}
          {m.voice && <Badge tone="secondary" className="ml-1">{t.multimodal.chat.voiceBadge}</Badge>}
          {m.queued && (
            <Badge tone="outline" className="ml-1">
              {m.queuePosition ? t.multimodal.chat.queuePosition(m.queuePosition) : t.multimodal.chat.queued}
            </Badge>
          )}
          {onPlay && m.role === "assistant" && !m.isError && !m.streaming && m.text.trim() && (
            <button
              onClick={() => onPlay(m.text)}
              title={t.multimodal.chat.playVoice}
              className="ml-1 inline-flex items-center gap-1 rounded border border-border/50 px-1.5 py-0.5 text-[10px] text-muted-foreground hover:border-primary/60 hover:text-primary">
              <Play className="h-3 w-3" /> {t.multimodal.chat.play}
            </button>
          )}
          {/* 监控事件 id 挪到框外最右侧右对齐 (对齐 watcher 汇报形态)。 */}
          {m.monitorId && (
            <span className="ml-auto font-mono text-muted-foreground/50">#{m.monitorId}</span>
          )}
        </div>
        {/* AssistantMessage 第一行 —— 事件驱动状态机:
            - streaming + 未收到任何 delta        → "Waiting response…"
            - streaming + 收到 reasoning 但无内容 → "Thinking…"
            - streaming + 有 reasoning 内容
                * 有 reasoningSummary (aux 生成的 ~10 字 label) → 显示 summary
                * 无 aux (失败/未启用) → 原 reasoning 最后一行滚动
            - 首个 message.delta 落地后 → 整行消失 (但 m.reasoning 后台保留)。
            推理完成 (!streaming) 也一并隐藏, 后台记录仅供下一轮 API 回传。 */}
        {m.role === "assistant" && m.streaming && !body && !m.isError && !inToolCall && (
          <ThinkingLine msg={m} />
        )}
        {(body || m.streaming) && (
          <div className={`break-words rounded-md px-3 py-2 text-sm ${
            m.isError ? "bg-red-500/15 text-red-400"
              : m.subRole === "monitor" ? "bg-amber-950/40 border-l-2 border-amber-400/50"
              : m.subRole === "router" ? "bg-violet-950/40 border-l-2 border-violet-400/50"
              : m.subRole === "query_worker" ? "bg-cyan-950/30 border-l-2 border-cyan-400/50"
              : m.role === "user" ? "bg-emerald-950/40" : "bg-muted/50"}`}>
            {m.queued && !body ? (
              <span className="inline-flex items-center gap-1 text-muted-foreground">
                {t.multimodal.chat.waitingPrevious}
                <span className="animate-pulse">…</span>
              </span>
            ) : m.role === "assistant" && !m.isError ? (
              // 流式渲染直接吃 body (唯一节流是 80ms 统一 flush, 见上); Markdown 的
              // streaming prop 画尾部光标并容忍半开语法; 完成后 body 即最终全文。
              // ★ 工具调用进行中 (inToolCall) 或还没落字的空 body 流式态 → Markdown
              // 空内容不画光标, 手动补一个 ▍ 让"气泡已就位、正文在路上"这件事看得见。
              m.streaming && !body ? (
                <span className="animate-pulse text-primary">▍</span>
              ) : m.streaming ? (
                <Markdown content={body} streaming={true} />
              ) : (
                <Markdown content={body} streaming={false} />
              )
            ) : (
              <span className="whitespace-pre-wrap">
                {body}
                {m.streaming && <span className="animate-pulse text-primary">▍</span>}
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
});

// mm:ss formatter for frame time ranges.
function fmtTs(s?: number): string {
  if (s == null || !isFinite(s)) return "";
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
}

// 深度分析面板的段解读 / 最终报告 Markdown。
// ★ 丝滑修复: 不再叠加第二层节流 (原来套 150ms useThrottledValue, 与统一 80ms flush
//   相位错开 → "一段段"拍频)。content 已由 answer_delta 入队合并 + 80ms 统一 flush 限速,
//   这里直接渲染即可; memo 保证只有 content 变化的段才重解析, 不会全列重渲。
const LiveMarkdown = memo(function LiveMarkdown({ content }: { content: string }) {
  return <Markdown content={content} streaming={false} />;
});

// One readable analysis-round card: 🎬 第N段 [mm:ss–mm:ss] → 👁 看到 →
// 🔎/🧩 检索 → 🖼 crops → 📝 就绪. Mirrors the desktop SegmentCard.
export const SegmentCard = memo(function SegmentCard({ s, defaultOpen, terminal }: {
  s: BgSegment; defaultOpen?: boolean;
  /** 整个深度研究已结束 → 空段不能再写"分析中…"(它永远不会再有内容了)。 */
  terminal?: boolean;
}) {
  useLocaleRevision();  // labels come from translateNow — see AsrBar
  const range = s.tsRange ? ` ${fmtTs(s.tsRange[0])}–${fmtTs(s.tsRange[1])}` : "";
  // 真实的段描述: 排除后端合成的占位句 (isSynthSaw, 见 lib/mm-sentinels)。
  const desc = s.saw && !isSynthSaw(s.saw) ? s.saw : "";
  const empty = !desc && s.lookups.length === 0 && !s.ready && !(s.crops && s.crops.length)
    && !(s.toolCalls && s.toolCalls.length) && !(s.toolErrors && s.toolErrors.length);
  // req ④: only the current/active segment is expanded by default; older ones
  // fold to a one-line summary the user can click to expand.
  const [open, setOpen] = useState(!!defaultOpen);
  useEffect(() => { setOpen(!!defaultOpen); }, [defaultOpen]);
  return (
    <div className="flex flex-col gap-1 rounded border border-violet-400/30 bg-background/40 px-2 py-1.5 text-[11px]">
      {/* 标题行 (唯一可点击行): ▸/▾ + 第N段 + 时间戳 + 场景标记。 */}
      <button onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-1.5 text-left font-semibold text-violet-300">
        <span className="shrink-0">{open ? "▾" : "▸"}</span>
        <span className="shrink-0">🎬 {translateNow("multimodal.deepAnalysis.segment", s.seg)}</span>
        {range && <span className="shrink-0 font-normal text-violet-300/70 tabular-nums">{range}</span>}
        {s.scene && (
          <span className="ml-0.5 truncate rounded bg-violet-400/15 px-1.5 py-0.5 font-normal text-violet-200/90">
            {s.scene}
          </span>
        )}
      </button>
      {/* 标题行下方固定一行文本描述 (来自 s.saw, 不限字数, 可换行成多行)。
          合成占位句已被 desc 过滤掉。 */}
      {desc ? (
        <div className="break-words leading-snug text-muted-foreground">
          {desc}
        </div>
      ) : empty && !terminal ? (
        <div className="leading-snug text-violet-200/60">{translateNow("multimodal.deepAnalysis.analyzing")}</div>
      ) : null}
      {/* 💭 思考: 默认折叠 (<details>, 对齐主 Agent), 点击展开看全文。思考中 (本段还没
          ready) 时图标带脉冲动画, 避免被误认为界面卡死。
          🔧 工具调用 / ⚠️ 错误 / 🔎 检索: 始终展示 (过程事实, 非内心独白), 见下方。 */}
      {s.thinking && (
        <details className="text-violet-200/70">
          <summary className="flex cursor-pointer list-none select-none items-center gap-1 leading-snug">
            <span className={s.ready || terminal ? "" : "animate-pulse"}>💭</span>
            <span>{s.ready || terminal ? translateNow("multimodal.deepAnalysis.thinking") : translateNow("multimodal.deepAnalysis.thinkingInProgress")}</span>
            {!s.ready && !terminal && (
              <span className="ml-0.5 inline-flex gap-0.5">
                <span className="h-1 w-1 animate-bounce rounded-full bg-violet-400 [animation-delay:-0.3s]" />
                <span className="h-1 w-1 animate-bounce rounded-full bg-violet-400 [animation-delay:-0.15s]" />
                <span className="h-1 w-1 animate-bounce rounded-full bg-violet-400" />
              </span>
            )}
          </summary>
          <div className="mt-1 whitespace-pre-wrap break-words rounded bg-violet-500/5 px-1.5 py-1 leading-snug">
            {s.thinking}
          </div>
        </details>
      )}
      {s.toolCalls && s.toolCalls.map((c, i) => (
        <div key={`tc${i}`} className="break-words leading-snug text-sky-300/80">
          {translateNow("multimodal.deepAnalysis.toolCall", c.name, c.arg || undefined)}
        </div>
      ))}
      {s.toolErrors && s.toolErrors.map((e, i) => (
        <div key={`te${i}`} className="break-words leading-snug text-red-400">
          {translateNow("multimodal.deepAnalysis.toolFailed", e.name, e.error)}
        </div>
      ))}
      {s.lookups.map((l, i) => (
        <div key={i} className="break-words leading-snug text-muted-foreground">
          {l.kind === "search" ? translateNow("multimodal.deepAnalysis.searchLookup", l.query) : translateNow("multimodal.deepAnalysis.memoryLookup", l.query)}
          {l.result ? ` → ${l.result}` : "…"}
        </div>
      ))}
      {open && (<>
        {s.crops && s.crops.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {s.crops.map((c, i) => (
              <img key={i} src={`data:image/jpeg;base64,${c.jpeg_b64}`} alt={c.label || `crop ${i}`}
                className="h-12 w-auto rounded border object-cover" title={c.label} />
            ))}
          </div>
        )}
        {s.ready && (
          <div className="leading-snug text-violet-100/90">
            {s.answer ? (
              // ★ 段解读走 Markdown 渲染 (支持表格 / LaTeX 公式), 不再纯文本。
              <div className="flex gap-1">
                <span className="shrink-0">📝</span>
                <div className="min-w-0 flex-1"><LiveMarkdown content={s.answer} /></div>
              </div>
            ) : (
              // s.ready 但没有流式 answer 文本: 本段没有独立解读 (通常是模型直接跳到
              // 回答/无额外发现), 内容已并入下方的累积报告 —— 如实说明, 不摆"就绪"空架子。
              <span className="whitespace-pre-wrap text-muted-foreground">
                {translateNow("multimodal.deepAnalysis.noIndependentInterpretation")}
              </span>
            )}
          </div>
        )}
      </>)}
    </div>
  );
});

// One RouterEngine deep-research sub-window (left column). Renders that
// delegation's streamed bubbles (reusing ChatBubble) + live search/recall
// progress + an optional Clarify follow-up input. Collapsible: only the
// expanded one shows its body.
// 攒帧进度条 (下一段的实时帧计数 + ttl 倒数)。抽成独立组件, 供【固定顶栏】复用 ——
// 现固定渲染在"深度分析 · 标签"标题下方的一行, 不再夹在段卡片之间随内容滚走。
function WaitingBanner({ waiting }: { waiting: NonNullable<BgItem["waiting"]> }) {
  const segPrefix = typeof waiting.seg === "number" ? `Seg ${waiting.seg} · ` : "";
  const paused = !!waiting.paused;
  // ★ paused: ttl 已耗尽、零新帧 → 后端在等画面变化 (不烧 VLM)。以前这里把整条
  //   攒帧条塌成一行、恢复时又整块重挂 → 观感就是"倒计时藏了又突然跳出个小数字"。
  //   现在【统一布局】: paused/active 都渲染同一套 (帧计数行 + 帧进度条 + ttl 条),
  //   paused 时只把 ⏱ 换成 ⏸ 文案、ttl 条置 0, 数字原位更新, 不再 mount/unmount 闪。
  //   注: 屏幕一动"倒计时跳到很小 + 秒完成"是场景重分级 (slow 200s→live 10s) 的
  //   自适应节奏, 属设计行为, 不在本显示修复范围内。
  const hasTtl = typeof waiting.ttlSec === "number"
    && typeof waiting.ttlRemaining === "number" && waiting.ttlSec > 0;
  const framePct = Math.min(100, waiting.need ? (waiting.have / waiting.need) * 100 : 0);
  const ttlPct = hasTtl
    ? Math.min(100, Math.max(0, (waiting.ttlRemaining! / waiting.ttlSec!)) * 100)
    : 0;
  return (
    <div className="flex flex-col gap-1 text-[11px] text-violet-200/80">
      <div className="flex items-center gap-1.5">
        <span className="inline-flex gap-0.5">
          {paused ? (
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-violet-400/60" />
          ) : (
            <>
              <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-violet-400 [animation-delay:-0.3s]" />
              <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-violet-400 [animation-delay:-0.15s]" />
              <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-violet-400" />
            </>
          )}
        </span>
        <span>
          {segPrefix}
          {paused ? (
            <>Waiting for new frames… <span className="ml-1 text-violet-300/70">⏸ 等待画面变化</span></>
          ) : (
            <>Buffering frames… {waiting.have}/{waiting.need}
              {typeof waiting.ttlRemaining === "number" && (
                <span className="ml-1 text-violet-300/70">· ⏱ {Math.max(0, Math.ceil(waiting.ttlRemaining))}s left</span>
              )}</>
          )}
        </span>
      </div>
      <div className="h-1 w-full overflow-hidden rounded bg-violet-400/15">
        <div className="h-full rounded bg-violet-400/70 transition-all duration-300"
          style={{ width: `${framePct}%` }} />
      </div>
      <div className="h-0.5 w-full overflow-hidden rounded bg-amber-400/15">
        <div className="h-full rounded bg-amber-400/60 transition-all duration-300"
          style={{ width: `${ttlPct}%` }} />
      </div>
    </div>
  );
}

export const DeepWindow = memo(function DeepWindow({
  rid, item, msgs, model, expanded, onToggle,
}: {
  rid: string;
  item: BgItem | undefined;
  msgs: ChatMsg[];
  model: string;
  expanded: boolean;
  onToggle: (rid: string) => void;
}) {
  const { t } = useI18n();
  const bodyScrollRef = useRef<HTMLDivElement | null>(null);
  // 用户是否停在底部。仅当停在底部时才随新内容自动下拉; 用户一旦向上翻 (atBottom
  // 变 false) 就不再打断他, 直到他自己滚回底部。onScroll 里用 24px 容差判定。
  const atBottomRef = useRef(true);
  const onBodyScroll = useCallback((e: UIEvent<HTMLDivElement>) => {
    const el = e.currentTarget;
    atBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
  }, []);
  const streaming = msgs.some((m) => m.streaming);
  const shortId = rid.replace(/^req_/, "").slice(0, 6);
  const label = item?.label || "";
  const segments = item?.segments || [];
  // ★ 渲染层兜底: 已结束的任务一律不显示攒帧条 —— 收尾时最后一个 Seg 的 waiting 常是
  //   "下一段" 的预告心跳 (seg = 末轮+1), 那一段永远不会真正开始。即使上游漏清, 这里也
  //   不该把 "已完成" 和一个转圈的 Seg 同时呈现。展开的进度条与收起的一行预览共用它。
  const waiting = item?.done ? null : item?.waiting;
  // Collapsed one-line preview: the waiting banner, else the newest segment's
  // most-informative line, else any streamed answer text — so a folded window is
  // never a blank title bar during long ReAct phases.
  const lastSeg = segments[segments.length - 1];
  const _wSeg = waiting && typeof waiting.seg === "number" ? `Seg ${waiting.seg} · ` : "";
  // ★ 已结束的任务不能再显示"第N段分析中…" —— 那是 lastSeg 缺真实描述时的进行中占位符,
  //   末段常因收尾提前退出而没有 saw, 于是收起的窗口会在"已完成"旁边写"分析中"。终态下
  //   优先用最终报告首句当预览, 没有则留空 (交给 answerPreview), 绝不复用进行中文案。
  const segDone = !!item?.done;
  const lastSegDesc = lastSeg && lastSeg.saw && !isSynthSaw(lastSeg.saw) ? lastSeg.saw : "";
  const segPreview = waiting
    ? (waiting.paused ? `⏳ ${_wSeg}Waiting for new frames…` : `⏳ ${_wSeg}Buffering frames… (${waiting.have}/${waiting.need})`)
    : lastSegDesc
      ? `👁 ${lastSegDesc}`
      : segDone
        ? (item?.finalReport || "").replace(/[#*`>\-\s]+/g, " ").trim().slice(0, 120)
        : lastSeg
          ? translateNow("multimodal.deepAnalysis.segmentAnalyzing", lastSeg.seg)
          : "";
  // ★ 性能: answerPreview 只在真需要时算 (没有 segPreview 且窗口收起才显示 preview)。
  //   旧代码每次渲染都 msgs.map/join/replace 全量拼接一遍, 展开时根本用不到。
  const answerPreview = useMemo(
    () => (segPreview ? "" : msgs.map((m) => m.text).join(" ").replace(/\s+/g, " ").trim()),
    [segPreview, msgs],
  );
  const preview = segPreview || answerPreview;
  const hasLiveWork = streaming || (item ? !item.done : false);

  useEffect(() => {
    const el = bodyScrollRef.current;
    // 只有用户仍停在底部时才自动下拉; 展开切换时重置为跟随底部。
    if (el && expanded && atBottomRef.current) el.scrollTop = el.scrollHeight;
  }, [msgs, segments, expanded]);

  return (
    <div className="rounded-md border border-violet-400/40 bg-violet-400/5">
      <button onClick={() => onToggle(rid)}
        className="flex w-full items-center gap-1.5 px-2.5 py-1.5 text-left text-[11px] font-medium text-violet-300">
        <span>{expanded ? "▾" : "▸"}</span>
        <span>🔬 {t.multimodal.deepAnalysis.title}{label ? ` · ${label}` : ` #${shortId}`}</span>
        {hasLiveWork ? <span className="animate-pulse">…</span>
          : <span className="ml-auto text-[10px] font-normal text-violet-300/60">{t.multimodal.deepAnalysis.completed}</span>}
      </button>
      {!expanded && preview && (
        <div className="truncate px-2.5 pb-1.5 text-[11px] text-violet-200/60">
          {preview}
        </div>
      )}
      {/* 攒帧进度条固定在标题正下方一行 (展开时), 不随段卡片滚动而离场。 */}
      {expanded && waiting && (
        <div className="border-t border-violet-400/20 px-2.5 py-1.5">
          <WaitingBanner waiting={waiting} />
        </div>
      )}
      {expanded && (
        <div className="border-t border-violet-400/20 px-2 py-2">
          <div ref={bodyScrollRef} onScroll={onBodyScroll} className="max-h-64 space-y-2 overflow-y-auto">
            {/* 段卡片: 旧段折叠 + 当前段展开。攒帧进度条已移到固定顶栏 (标题下方),
                不再夹在段之间随滚动离场。 */}
            {(() => {
              const older = segments.slice(0, Math.max(0, segments.length - 1));
              const last = segments.length > 0 ? segments[segments.length - 1] : null;
              return (
                <>
                  {older.length > 0 && (
                    <div className="space-y-1.5">
                      {older.map((s) => (
                        <SegmentCard key={s.seg} s={s} defaultOpen={false} terminal={segDone} />
                      ))}
                    </div>
                  )}
                  {last && (
                    <div className="space-y-1.5">
                      <SegmentCard key={last.seg} s={last} defaultOpen={true} terminal={segDone} />
                    </div>
                  )}
                </>
              );
            })()}
            {!answerPreview && hasLiveWork && segments.length === 0 && !waiting && (
              <div className="text-[11px] text-violet-200/70">
                {t.multimodal.deepAnalysis.starting}
              </div>
            )}
            {/* Final consolidated report (watcher.final) — the authoritative
                result, shown in-panel; the main agent chat is never touched. */}
            {item?.finalReport && (
              <div className="rounded-md border border-violet-400/50 bg-violet-400/10 p-2">
                <div className="mb-1 text-[11px] font-medium text-violet-200">📋 {t.multimodal.deepAnalysis.finalReport}</div>
                <div className="text-[12px] leading-relaxed text-violet-50">
                  {/* ★ 最终报告走 Markdown (表格 / LaTeX 公式), 不再纯文本。
                      节流解析 (LiveMarkdown): 长报告在同帧与主 agent 并发时不再双 O(n²) 撑爆主线程。 */}
                  <LiveMarkdown content={item.finalReport} />
                </div>
              </div>
            )}
            {msgs.map((m) => (
              <ChatBubble key={m.id} m={m} model={model} onPlay={NOOP_PLAY} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
});

/* ================================================================== */
/*  Column components (perf: isolate re-renders)                       */
/*  三列各自 memo 化: 任一 setState 只重渲染其数据切片所在的那一列, 而不是                */
/*  整个 3500 行 render body。父组件仍持有全部 state + 派生值, 只负责把                  */
/*  【引用稳定】的 props 传下去 —— 无关的 setState (frameCount 1/s /                  */
/*  anchor / ctx / bgItems) 不改动某列的 props → 该列 memo 命中、跳过。                */
/* ================================================================== */

// grouped waterfall row (chat bubble | background progress block). Hoisted to
// module scope so the column components can reference it.
type Row =
  | { type: "chat"; msg: ChatMsg }
  | { type: "bg"; id: string; items: ChatMsg[]; thinking?: string; seg?: number };

// ── Work segments ────────────────────────────────────────────────────────────
// A model turn is really `think → act → think → act → answer`. Rendering it as
// "all the thinking in one pile, all the tools in another" is regular but drops
// the causal chain — you can no longer tell WHICH thought led to WHICH call.
// Rendering every part strictly inline is faithful but noisy (that is why the
// piles existed in the first place).
//
// The fix is to group by TIME, not by TYPE — the same shelf model ui-tui
// already ships (`lib/liveProgress.ts::appendToolShelfMessage`):
//
//   * assistant prose is a BARRIER — it closes the current segment.
//   * consecutive tool/status rows MERGE into the open segment (so ten calls
//     are one block, not ten cards).
//   * the reasoning that preceded those calls rides on the SAME segment, so
//     "what it was thinking" sits with "what it therefore did".
//
// Result: one block per segment, in true order, each carrying its own thinking.
export function buildRows(messages: readonly ChatMsg[]): Row[] {
  const out: Row[] = [];
  // Reasoning seen since the last barrier, waiting to be attached to the
  // segment its tool calls belong to.
  let pendingThinking = "";
  let segCount = 0;

  for (const m of messages) {
    // Live deep-research streams live in the left sub-windows; post-Clarify
    // thread-backs (threadback=true) render here in the center chat.
    if (m.subRole === "router" && !m.threadback) continue;
    // Safety net: monitor + watcher_report never render in the center chat any
    // more (they live in the right multimodal panel). Any legacy message that
    // slipped through is filtered here.
    if (m.subRole === "monitor" || m.subRole === "watcher_report") continue;

    if (m.kind === "tool" || m.kind === "status") {
      const last = out[out.length - 1];

      if (last && last.type === "bg") {
        last.items.push(m);
        // Reasoning can arrive after the segment opened (interleaved thinking);
        // fold it in rather than dropping it.
        if (pendingThinking && !last.thinking) last.thinking = pendingThinking;
      } else {
        segCount += 1;
        out.push({
          type: "bg",
          id: `bg_${m.id}`,
          items: [m],
          seg: segCount,
          ...(pendingThinking ? { thinking: pendingThinking } : {}),
        });
      }

      pendingThinking = "";
      continue;
    }

    // ★ `ensureBubble` reuses ONE assistant message per turn: reasoning
    //   accumulates onto it first, then the final answer text lands on that same
    //   object. So by the time the turn finishes, the bubble carrying the answer
    //   is ALSO the only carrier of that turn's reasoning. Claim its reasoning
    //   for the segments that follow before treating it as a barrier, otherwise
    //   the rationale is dropped the instant the answer arrives — which is the
    //   old "reasoning only exists while streaming" bug in a new place.
    const carriesReasoning = m.role === "assistant" && !!m.reasoning?.trim() && !m.isError;

    // An assistant turn with reasoning but no visible prose yet is not a
    // barrier — it is the "thinking" half of the segment that is about to open.
    // Hold its reasoning and let the tool rows that follow claim it.
    const isThinkingOnly = carriesReasoning && !m.text?.trim();

    if (isThinkingOnly) {
      const reasoning = m.reasoning!.trim();
      const last = out[out.length - 1];

      // Interleaved thinking (reasoning between two calls of the same round):
      // fold it into the open segment and emit NO row, so the following tool
      // still merges into that segment instead of starting a new one. A
      // thinking-only row must never act as a barrier.
      if (last && last.type === "bg") {
        if (!last.thinking) last.thinking = reasoning;
        pendingThinking = "";
        continue;
      }

      // No segment open yet — hold the reasoning for the segment about to open,
      // and emit the row so it renders as the live 💭 line while streaming.
      pendingThinking = reasoning;
      out.push({ type: "chat", msg: m });
      continue;
    }

    // Prose + reasoning on the same object (the finished-turn shape): the prose
    // is still a barrier for what came BEFORE it, but its reasoning belongs to
    // the segments that come AFTER, so carry it forward instead of clearing.
    if (carriesReasoning) {
      pendingThinking = m.reasoning!.trim();
      out.push({ type: "chat", msg: m });
      continue;
    }

    // ★ An assistant bubble that ended up carrying NOTHING must not split the
    //   work segments. Prose is a deliberate barrier (it marks "the model said
    //   something, a new round starts after this"), but a finished bubble with
    //   no text and no reasoning says nothing — it is invisible on screen, yet
    //   emitting a chat row for it still breaks bg adjacency, so the tool calls
    //   before and after it render as two separate 处理过程 cards with no
    //   visible reason between them. Streaming bubbles are kept: with no text
    //   yet they are the live 💭 ThinkingLine and must still render.
    const isInvisibleAssistant = (
      m.role === "assistant"
      && !m.streaming
      && !m.text?.trim()
      && !m.reasoning?.trim()
      && !m.isError
      && !m.kind            // not tool/status/clarify — those return earlier
      && !m.monitorId
      && !m.deepResearch
      && !m.subRole
    );
    if (isInvisibleAssistant) continue;

    // Real prose (or an error / user turn) — barrier. Close the segment.
    if (m.text?.trim() || m.role === "user") pendingThinking = "";
    out.push({ type: "chat", msg: m });
  }

  return out;
}

type WatcherReg = { watcher_id: string; label?: string; task_instruction?: string; status?: string };
type MonitorReg = MonitorRegistryItem;
// One proactive alert emitted by a monitor. Alerts render inline under their
// monitor row in the right registry (never as center-chat bubbles).
type MonitorAlert = {
  id: string;
  text: string;
  ts: number;
  streaming?: boolean;
  evidence?: MonitorEvidence;
};
type MmToast = { id: string; level: string; text: string };
type AnchorFrame = { ts: number | null; jpeg_b64: string };

/* ── LEFT column: 视频 + 注入帧 + 画面/音频观察 + 搜索事实 ────────────────── */
const LeftPanels = memo(function LeftPanels({
  sourceType, frameCount, anchorFrames, ctxVersion, obs, audioObs, factsList,
  videoRef, obsScrollRef, audioObsScrollRef,
  onStartCamera, onStopStream, onStartScreen,
}: {
  sourceType: SourceType;
  frameCount: number;
  anchorFrames: AnchorFrame[];
  ctxVersion: number;
  obs: ObsItem[];
  audioObs: ObsItem[];
  factsList: [string, string][];
  videoRef: React.RefObject<HTMLVideoElement | null>;
  obsScrollRef: React.RefObject<HTMLDivElement | null>;
  audioObsScrollRef: React.RefObject<HTMLDivElement | null>;
  onStartCamera: () => void;
  onStopStream: () => void;
  onStartScreen: () => void;
}) {
  const { t } = useI18n();
  return (
    <div className="flex min-h-0 flex-col gap-3 overflow-y-auto">
      <Card>
        <CardContent className="p-3">
          <div className="relative aspect-[4/3] overflow-hidden rounded-md bg-black [contain:strict]">
            {/* Live preview. NOTE: hiding this / throttling it does NOT reduce the
                screen-share mouse lag — that lag is macOS ScreenCaptureKit
                capturing the whole Retina display contending with the WindowServer
                compositor (cursor), which the web layer can't touch. So keep the
                smoothest, simplest preview: the raw <video> at the source's 4fps. */}
            <video ref={videoRef} autoPlay playsInline muted
              className="h-full w-full object-cover [transform:translateZ(0)]" />
            {!sourceType && (
              <div className="absolute left-2 top-2 rounded bg-black/60 px-2 py-1 text-xs text-white">{t.multimodal.video.notStarted}</div>
            )}
            {sourceType && (
              <div className="absolute right-2 top-2 flex items-center gap-1 rounded bg-black/60 px-2 py-1 text-xs text-white">
                <span className="h-2 w-2 animate-pulse rounded-full bg-red-500" />
                {t.multimodal.video.recFrames(sourceType === "camera" ? t.multimodal.video.camera : t.multimodal.video.screenShare, frameCount)}
              </div>
            )}
          </div>
          <div className="mt-3 grid grid-cols-2 gap-2">
            <Button size="sm" prefix={<Camera />}
              destructive={sourceType === "camera"} disabled={sourceType === "screen"}
              onClick={() => (sourceType === "camera" ? onStopStream() : onStartCamera())}>
              {sourceType === "camera" ? t.multimodal.video.stopCamera : t.multimodal.video.startCamera}
            </Button>
            <Button size="sm" prefix={<Monitor />}
              destructive={sourceType === "screen"} outlined={sourceType !== "screen"}
              disabled={sourceType === "camera"}
              onClick={() => (sourceType === "screen" ? onStopStream() : onStartScreen())}>
              {sourceType === "screen" ? t.multimodal.video.stopScreen : t.multimodal.video.startScreen}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* ② Anchor debug: frames the vision model saw this turn */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-xs uppercase tracking-wide text-muted-foreground">
            🎯 {t.multimodal.observations.injectedFrames} {anchorFrames.length > 0 && <span className="text-primary">· {anchorFrames.length}</span>}
          </CardTitle>
        </CardHeader>
        <CardContent className="pt-0">
          {anchorFrames.length === 0
            ? <div className="text-xs italic text-muted-foreground">{t.multimodal.observations.injectedFramesHint}</div>
            : (
              <div className="flex gap-1.5 overflow-x-auto">
                {anchorFrames.map((f, i) => (
                  <div key={i} className="flex-shrink-0">
                    <img src={`data:image/jpeg;base64,${f.jpeg_b64}`} alt={`frame ${i}`}
                      className="h-16 w-auto cursor-zoom-in rounded border"
                      onClick={() => window.open(`data:image/jpeg;base64,${f.jpeg_b64}`, "_blank")} />
                    {f.ts != null && <div className="mt-0.5 text-center text-[10px] text-muted-foreground">{f.ts.toFixed(1)}s</div>}
                  </div>
                ))}
              </div>
            )}
        </CardContent>
      </Card>

      {/* ③ 画面观察 */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-xs uppercase tracking-wide text-muted-foreground">
            {t.multimodal.observations.videoObs} <span className="text-primary">{t.multimodal.observations.videoObsVersion(ctxVersion)}</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="pt-0">
          <div ref={obsScrollRef} className="max-h-52 space-y-2 overflow-y-auto rounded border bg-background/50 p-2 text-xs">
            {obs.length === 0
              ? <span className="italic text-muted-foreground">{t.multimodal.observations.empty}</span>
              : obs.map((o, i) => (
                  <div key={`obs-${i}`} className="rounded border border-border/60 bg-background/60 p-2">
                    <div className="mb-1 flex items-center gap-1">
                      <span className="rounded bg-violet-500/15 px-1.5 py-0.5 font-mono text-[10px] text-violet-400">{o.ts}</span>
                      {o.speaker ? <span className="text-[10px] text-muted-foreground">{o.speaker}</span> : null}
                    </div>
                    <div className="whitespace-pre-wrap leading-snug">{o.text}</div>
                  </div>
                ))}
          </div>
        </CardContent>
      </Card>

      {/* ④ 音频观察 */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-xs uppercase tracking-wide text-muted-foreground">{t.multimodal.observations.audioObs}</CardTitle>
        </CardHeader>
        <CardContent className="pt-0">
          <div ref={audioObsScrollRef} className="max-h-44 space-y-2 overflow-y-auto rounded border bg-background/50 p-2 text-xs">
            {audioObs.length === 0
              ? <span className="italic text-muted-foreground">{t.multimodal.observations.audioObsHint}</span>
              : audioObs.map((o, i) => (
                  <div key={`aobs-${i}`} className="rounded border border-border/60 bg-background/60 p-2">
                    <div className="mb-1 flex items-center gap-1">
                      <span className="rounded bg-sky-500/15 px-1.5 py-0.5 font-mono text-[10px] text-sky-400">{o.ts}</span>
                      {o.speaker ? <span className="text-[10px] text-muted-foreground">🗣 {o.speaker}</span> : null}
                    </div>
                    <div className="whitespace-pre-wrap leading-snug">{o.text}</div>
                  </div>
                ))}
          </div>
        </CardContent>
      </Card>

      {/* ⑤ SearchFactStore: 外部检索证据 */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-xs uppercase tracking-wide text-muted-foreground">{t.multimodal.observations.searchFacts}</CardTitle>
        </CardHeader>
        <CardContent className="pt-0">
          <div className="max-h-48 overflow-y-auto rounded border bg-background/50 p-2 text-xs">
            {factsList.length === 0
              ? <span className="italic text-muted-foreground">{t.multimodal.observations.noneYet}</span>
              : <ul className="space-y-1">{factsList.map(([k, v]) => (
                  <li key={k}><span className="text-violet-400">{k}</span>: {String(v)}</li>
                ))}</ul>}
          </div>
        </CardContent>
      </Card>
    </div>
  );
});

/* ── MIDDLE column: 聊天列 + AsrBar + ChatComposer ────────────────────── */
const ChatColumn = memo(function ChatColumn({
  rows, renderRow, itemKey, atBottom, chatScrollRef, onChatScroll, scrollChatToBottom,
  chatAtBottomRef, isRecordingUI, asrPartial, asrBuffer,
  micState, ttsEnabled, onTtsToggle,
  generating, onStop, onSend, onSlash, gw, onMicToggle, composerApiRef,
}: {
  rows: Row[];
  renderRow: (i: number, row: Row) => React.ReactNode;
  itemKey: (i: number, row: Row) => string;
  atBottom: boolean;
  chatScrollRef: React.RefObject<HTMLDivElement | null>;
  onChatScroll: () => void;
  scrollChatToBottom: (smooth?: boolean) => void;
  chatAtBottomRef: React.RefObject<boolean>;
  isRecordingUI: boolean;
  asrPartial: string;
  asrBuffer: string[];
  micState: MicLifecycleState;
  ttsEnabled: boolean;
  onTtsToggle: () => void;
  generating: boolean;
  onStop: () => void;
  onSend: (text: string) => void;
  onSlash: (command: string) => void;
  gw: GatewayClient | null;
  onMicToggle: () => void;
  composerApiRef?: React.MutableRefObject<{
    setText: (text: string) => void;
  } | null>;
}) {
  useLocaleRevision();  // labels come from translateNow — see AsrBar
  // ★ 自动跟随底部 (替代 Virtuoso followOutput)。仅当用户原本就在底部才下拉。
  //   移进 ChatColumn: 只在 rows 变(消息/流式)时跑, 不再被 ctx/anchor/frameCount 触发。
  useLayoutEffect(() => {
    if (chatAtBottomRef.current) scrollChatToBottom(false);
  }, [rows, scrollChatToBottom, chatAtBottomRef]);
  return (
    <Card className="relative flex min-h-0 min-w-0 flex-col">
      {/* ★ 普通滚动 div 全量渲染 (取代 react-virtuoso —— 它对流式大表格会测量死循环)。
          消息量由 capMsgs 软上限兜底。min-w-0: 让列可收缩, 超长不换行内容(长英文
          标题/表格)不撑破列宽 → 消除切屏后右侧溢出。 */}
      <div
        ref={chatScrollRef}
        onScroll={onChatScroll}
        className="min-h-0 min-w-0 flex-1 space-y-3 overflow-y-auto px-3 pb-24 pt-3"
      >
        {rows.map((row, _i) => {
          // renderRow 可能返回 null (如流式期间被隐藏的"处理过程"bg 行) —— 此时不发出
          // 空 wrapper div, 避免 space-y-3 在其位置留下一段幽灵间距。
          const el = renderRow(_i, row);
          return el == null ? null : <div key={itemKey(_i, row)}>{el}</div>;
        })}
      </div>
      {!atBottom && (
        <button
          type="button"
          onClick={() => scrollChatToBottom(true)}
          title={translateNow("multimodal.misc.jumpToLatest")}
          className="absolute bottom-20 right-4 z-10 flex h-8 w-8 items-center justify-center rounded-full border border-border bg-background/90 text-muted-foreground shadow-md backdrop-blur hover:text-foreground">
          <ArrowDown className="h-4 w-4" />
        </button>
      )}
      <AsrBar recording={isRecordingUI} partial={asrPartial} buffer={asrBuffer} />
      <ChatComposer
        micState={micState}
        ttsEnabled={ttsEnabled}
        onTtsToggle={onTtsToggle}
        generating={generating}
        onStop={onStop}
        onSend={onSend}
        onSlash={onSlash}
        gw={gw}
        onMicToggle={onMicToggle}
        composerApiRef={composerApiRef}
      />
    </Card>
  );
});

/* ── 监控 / 深度分析 注册表 (右列顶部) ─────────────────────────────────── */
const RegistryPanels = memo(function RegistryPanels({
  monitors, watchers, onToggleMonitor, onToggleWatcher,
}: {
  monitors: MonitorReg[];
  watchers: WatcherReg[];
  onToggleMonitor: (m: MonitorReg) => void;
  onToggleWatcher: (w: WatcherReg) => void;
}) {
  const { t } = useI18n();
  return (
    <>
      {/* ① Monitor registry (multi-instance, set_monitor CRUD) */}
      {monitors.some((m) => m.status !== "deleted") && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs uppercase tracking-wide text-muted-foreground">
              {t.multimodal.monitor.titleWithCount(monitors.filter((m) => m.status !== "deleted").length)}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 pt-2 text-[11px]">
            {monitors.filter((m) => m.status !== "deleted").map((m) => {
              const { active: on, canToggle, done, mode, statusToken } = monitorPresentation(m);
              const modeLabel = mode === "once" ? t.multimodal.monitor.once : t.multimodal.monitor.continuous;
              const statusLabel = statusToken === "done" ? t.multimodal.monitor.statusDone
                : statusToken === "active" ? t.multimodal.monitor.statusActive
                : t.multimodal.monitor.statusInterrupted;
              const label = (m.label && m.label.trim()) || m.brief.slice(0, 10) || t.multimodal.monitor.title;
              return (
                <div key={m.monitor_id}
                  className={`flex items-center justify-between gap-2 rounded border px-2.5 py-2 ${
                    done ? "border-emerald-400/30 bg-emerald-400/5"
                    : on ? "border-amber-400/30 bg-amber-400/5"
                    : "border-border/40 bg-muted/20 opacity-60"}`}>
                  {/* label · 触发模式 · 状态 · #事件号 同一行。 */}
                  <span className={`flex min-w-0 flex-1 items-baseline gap-1 break-words leading-tight ${
                    done ? "text-emerald-200" : on ? "text-amber-200" : "text-muted-foreground"}`}
                    title={`${label} · ${modeLabel} · ${statusLabel} · #${m.monitor_id}`}>
                    <span className="truncate">{label}</span>
                    <span className="shrink-0 rounded border border-current/20 px-1 text-[9px] opacity-75">{modeLabel}</span>
                    <span className="shrink-0 text-[10px] text-muted-foreground/60">· {statusLabel}</span>
                    <span className="shrink-0 font-mono text-[10px] text-muted-foreground/45">· #{m.monitor_id}</span>
                  </span>
                  <button
                    type="button"
                    disabled={!canToggle}
                    onClick={() => canToggle && onToggleMonitor(m)}
                    title={done
                      ? t.multimodal.monitor.completedOnce
                      : on ? t.multimodal.monitor.pauseHint : t.multimodal.monitor.resumeHint}
                    aria-label={`${label}：${statusLabel}`}
                    className={`relative h-4 w-7 flex-shrink-0 rounded-full transition-colors ${
                      done ? "cursor-not-allowed bg-emerald-400/35"
                      : on ? "bg-amber-400/70" : "bg-muted-foreground/30"}`}>
                    <span className={`absolute top-0.5 h-3 w-3 rounded-full bg-background transition-all ${
                      on ? "left-3.5" : "left-0.5"}`} />
                  </button>
                </div>
              );
            })}
          </CardContent>
        </Card>
      )}

      {/* ①b Active watchers registry (set_live_watcher CRUD + reopen). */}
      {watchers.length > 0 && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs uppercase tracking-wide text-muted-foreground">
              🔬 {t.multimodal.deepAnalysis.title} <span className="text-violet-300">· {watchers.length}</span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 pt-2 text-[11px]">
            {watchers.filter((w) => w.status !== "deleted").map((w) => {
              const on = w.status === "running";
              const label = (w.label && w.label.trim())
                || (w.task_instruction || "").slice(0, 12) || t.multimodal.deepAnalysis.title;
              // ★ 五态标签: running=进行中 / done=已完成 / stopping=正在停止 /
              //   interrupted=已中断(需开流重启)。
              const statusLabel =
                w.status === "running" ? t.multimodal.deepAnalysis.inProgress
                : w.status === "done" ? t.multimodal.deepAnalysis.completed
                : w.status === "stopping" ? t.multimodal.deepAnalysis.statusStopping
                : t.multimodal.monitor.statusInterrupted;
              return (
                <div key={w.watcher_id}
                  className={`flex items-center justify-between gap-2 rounded border px-2.5 py-2 ${
                    on ? "border-violet-400/30 bg-violet-400/5" : "border-border/40 bg-muted/20 opacity-60"}`}>
                  {/* label · 状态 · #事件号 同一行 (溢出截断)。 */}
                  <span className={`flex min-w-0 flex-1 items-baseline gap-1 break-words leading-tight ${
                    on ? "text-violet-200" : "text-muted-foreground"}`}
                    title={`${label} · ${statusLabel} · #${w.watcher_id}`}>
                    <span className="truncate">{label}</span>
                    <span className="shrink-0 text-[10px] text-muted-foreground/60">· {statusLabel}</span>
                    <span className="shrink-0 font-mono text-[10px] text-muted-foreground/45">· #{w.watcher_id}</span>
                  </span>
                  <button
                    onClick={() => onToggleWatcher(w)}
                    title={on ? t.multimodal.deepAnalysis.pauseHint : t.multimodal.deepAnalysis.resumeHint}
                    className={`relative h-4 w-7 flex-shrink-0 rounded-full transition-colors ${
                      on ? "bg-violet-400/70" : "bg-muted-foreground/30"}`}>
                    <span className={`absolute top-0.5 h-3 w-3 rounded-full bg-background transition-all ${
                      on ? "left-3.5" : "left-0.5"}`} />
                  </button>
                </div>
              );
            })}
          </CardContent>
        </Card>
      )}
    </>
  );
});

/* ── Per-monitor alert panel: title bar (collapsible) + up to 2 latest hits +
   "展开更多" expander to reveal all history. Amber-themed to match the monitor
   registry row. Display-only (the enable/disable toggle stays on the registry
   card above). */
const MONITOR_ALERTS_VISIBLE = 2;
const MonitorPanel = memo(function MonitorPanel({
  monitor, alerts, collapsed, expanded, onToggleCollapsed, onToggleExpanded,
}: {
  monitor: MonitorReg;
  alerts: MonitorAlert[];
  collapsed: boolean;                 // title-bar arrow
  expanded: boolean;                  // "展开更多" → show all vs. only latest 2
  onToggleCollapsed: (mid: string) => void;
  onToggleExpanded: (mid: string) => void;
}) {
  const { t } = useI18n();
  const mid = monitor.monitor_id;
  const label = (monitor.label && monitor.label.trim())
    || monitor.brief.slice(0, 12) || t.multimodal.monitor.title;
  const { active: on, done, mode, statusToken } = monitorPresentation(monitor);
  const modeLabel = mode === "once" ? t.multimodal.monitor.once : t.multimodal.monitor.continuous;
  const statusLabel = statusToken === "done" ? t.multimodal.monitor.statusDone
    : statusToken === "active" ? t.multimodal.monitor.statusActive
    : t.multimodal.monitor.statusInterrupted;
  const streaming = alerts.some((a) => a.streaming);
  const hiddenCount = Math.max(0, alerts.length - MONITOR_ALERTS_VISIBLE);
  const shown = expanded ? alerts : alerts.slice(-MONITOR_ALERTS_VISIBLE);
  return (
    <div className={`rounded-md border ${
      done ? "border-emerald-400/35 bg-emerald-400/5"
      : on ? "border-amber-400/40 bg-amber-400/5"
      : "border-border/40 bg-muted/20 opacity-70"}`}>
      {/* Title bar — click to collapse/expand the whole panel. */}
      <button onClick={() => onToggleCollapsed(mid)}
        className="flex w-full items-center gap-1.5 px-2.5 py-1.5 text-left text-[11px] font-medium text-amber-200">
        <span className="shrink-0 text-amber-300/70">{collapsed ? "▸" : "▾"}</span>
        <span>👁</span>
        <span className="min-w-0 truncate">{t.multimodal.monitor.label(label)}</span>
        <span className="shrink-0 rounded border border-current/20 px-1 text-[9px] font-normal opacity-70">{modeLabel}</span>
        {streaming
          ? <span className="animate-pulse text-amber-300/70">…</span>
          : <span className="shrink-0 text-[10px] font-normal text-amber-300/60">· {statusLabel}</span>}
        <span className="ml-auto shrink-0 font-mono text-[10px] text-amber-300/45">#{mid}</span>
      </button>
      {!collapsed && (
        <div className="border-t border-amber-400/20 px-2 py-2">
          {alerts.length === 0 && (
            <div className="text-[11px] text-amber-200/70">
              {done ? t.multimodal.monitor.completedOnceShort : on ? t.multimodal.monitor.noHitsYet : t.multimodal.monitor.paused}
            </div>
          )}
          {hiddenCount > 0 && (
            <button onClick={() => onToggleExpanded(mid)}
              className="mb-1.5 text-[10px] text-amber-300/70 hover:text-amber-200">
              {expanded ? t.multimodal.monitor.collapseEarly : t.multimodal.monitor.showMore(hiddenCount)}
            </button>
          )}
          <div className={`space-y-1.5 ${expanded ? "max-h-64 overflow-y-auto" : ""}`}>
            {shown.map((a) => (
              <div key={a.id}
                className="rounded border-l-2 border-amber-400/50 bg-amber-950/30 px-2 py-1.5">
                <div className="mb-0.5 flex items-center gap-1.5 text-[10px] text-amber-300/60">
                  <span className="tabular-nums">{fmtClock(a.ts)}</span>
                  {a.streaming && <span className="animate-pulse">…</span>}
                </div>
                <div className="whitespace-pre-wrap break-words text-[12px] leading-relaxed text-amber-50">
                  {a.text || (a.streaming ? "…" : "")}
                </div>
                {a.evidence && <MonitorEvidenceStrip evidence={a.evidence} />}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
});

/* ── RIGHT column: 监控/深研注册表 + 监控 panel + 深研窗口 + toast ────────── */
const DeepColumn = memo(function DeepColumn({
  showDeepCol, mmToasts, monitors, watchers, onToggleMonitor, onToggleWatcher,
  visibleDeep, bgByRid, deepExpanded, model, onToggleDeep,
  monitorAlerts, monitorCollapsed, monitorExpanded,
  onToggleMonitorCollapsed, onToggleMonitorExpanded,
}: {
  showDeepCol: boolean;
  mmToasts: MmToast[];
  monitors: MonitorReg[];
  watchers: WatcherReg[];
  onToggleMonitor: (m: MonitorReg) => void;
  onToggleWatcher: (w: WatcherReg) => void;
  visibleDeep: { rid: string; msgs: ChatMsg[] }[];
  bgByRid: Map<string, BgItem>;
  deepExpanded: string | null;
  model: string;
  onToggleDeep: (rid: string) => void;
  monitorAlerts: Map<string, MonitorAlert[]>;
  monitorCollapsed: Set<string>;
  monitorExpanded: Set<string>;
  onToggleMonitorCollapsed: (mid: string) => void;
  onToggleMonitorExpanded: (mid: string) => void;
}) {
  useLocaleRevision();  // labels come from translateNow — see AsrBar (before the early return)
  if (!showDeepCol) return null;
  // Monitor panels: display in the order the monitors were created (registry
  // is already sorted by created_at asc — the earliest sits at the top). Users
  // asked for this stacking so a session with multiple monitors reads as a
  // stable timeline.
  const activeMonitors = monitors
    .filter((m) => m.status !== "deleted")
    .slice()
    .sort((a, b) => (a.created_at || 0) - (b.created_at || 0));
  return (
    <div className="relative flex min-h-0 min-w-0 flex-col gap-2 overflow-y-auto">
      {/* 底部 toast 小框栈 (监控/深度研究过程失败/停用, 3s 淡出)。 */}
      {mmToasts.length > 0 && (
        <div className="pointer-events-none absolute inset-x-2 bottom-2 z-30 flex flex-col gap-1.5">
          {mmToasts.map((t) => (
            <div key={t.id}
              className={`mm-toast-in pointer-events-auto rounded-md border px-2.5 py-1.5 text-[11px] leading-snug shadow-md backdrop-blur-sm ${
                t.level === "warning"
                  ? "border-amber-400/40 bg-amber-500/10 text-amber-300"
                  : t.level === "info"
                    ? "border-border/50 bg-muted/40 text-muted-foreground"
                    : "border-red-400/40 bg-red-500/10 text-red-300"}`}>
              {t.text}
            </div>
          ))}
        </div>
      )}
      {/* 监控 / 深度分析 注册表: 置于深度研究窗口之上 (对齐桌面端 deep-panel)。 */}
      <RegistryPanels
        monitors={monitors}
        watchers={watchers}
        onToggleMonitor={onToggleMonitor}
        onToggleWatcher={onToggleWatcher}
      />
      {/* Monitor alert panels — one per active monitor, stacked in creation
          order. Each shows latest 2 alerts by default, expandable to full
          history; the title bar arrow collapses the whole panel. */}
      {activeMonitors.map((m) => (
        <MonitorPanel
          key={m.monitor_id}
          monitor={m}
          alerts={monitorAlerts.get(m.monitor_id) || []}
          collapsed={monitorCollapsed.has(m.monitor_id)}
          expanded={monitorExpanded.has(m.monitor_id)}
          onToggleCollapsed={onToggleMonitorCollapsed}
          onToggleExpanded={onToggleMonitorExpanded}
        />
      ))}
      {visibleDeep.map(({ rid, msgs }, i) => {
        const streaming = msgs.some((m) => m.streaming);
        const ridBg = bgByRid.get(rid);
        const ridBusy = streaming || (ridBg ? !ridBg.done : false);
        // Explicit user choice for this rid wins (open or the "" collapse
        // sentinel). Otherwise: expand while streaming/busy, else newest.
        const userChoice = deepExpanded === rid;
        const userCollapsed = deepExpanded === "";
        const expanded = userChoice
          || (!userCollapsed && (ridBusy || (deepExpanded === null && i === 0)));
        return (
          <DeepWindow
            key={rid}
            rid={rid}
            msgs={msgs}
            item={ridBg}
            model={model}
            expanded={expanded}
            onToggle={onToggleDeep}
          />
        );
      })}
    </div>
  );
});

export interface MmTrajectoryFrame {
  frame_id?: string;
  ts?: number;
  jpeg_b64?: string;
  thumb_b64?: string;
  source_type?: string;
}

export interface MmTrajectoryEntry {
  id: string;
  seq: number;
  ts: number;
  event: string;
  worker: string;
  phase: string;
  payload: Record<string, unknown>;
}

/**
 * Bound image bytes in the inspector's trajectory copy. QueryWorker entries
 * keep their structured payload and frame metadata; only old base64 fields are
 * evicted. This is separate from the progress-card cache because the Debug
 * inspector also retains the normalized trajectory rows.
 */
// eslint-disable-next-line react-refresh/only-export-components
export function compactQueryWorkerTrajectory(
  entries: MmTrajectoryEntry[],
): MmTrajectoryEntry[] {
  const taskOrder: string[] = [];
  const seen = new Set<string>();
  for (const entry of entries) {
    const taskId = typeof entry.payload?.task_id === "string"
      ? entry.payload.task_id : "";
    if (!taskId) continue;
    if (seen.has(taskId)) {
      taskOrder.splice(taskOrder.indexOf(taskId), 1);
    } else {
      seen.add(taskId);
    }
    taskOrder.push(taskId);
  }
  const newest = taskOrder.at(-1) || "";
  const imageTasks = recentTaskIds(taskOrder);
  const protectedChars = entries.reduce((total, entry) => {
    const p = entry.payload || {};
    if (p.task_id !== newest || String(entry.phase || p.phase) !== "started") {
      return total;
    }
    const frames = Array.isArray(p.frames) ? p.frames.filter(isRecord) : [];
    return total + frames.reduce(
      (sum, frame) => sum + frameImageChars(frame as MmTrajectoryFrame), 0,
    );
  }, 0);
  const remaining = {
    chars: Math.max(0, QUERY_WORKER_IMAGE_CHAR_BUDGET - protectedChars),
  };

  return entries.slice().reverse().map((entry) => {
    const p = entry.payload || {};
    const taskId = typeof p.task_id === "string" ? p.task_id : "";
    const rawFrames = Array.isArray(p.frames) ? p.frames : null;
    if (!taskId || !rawFrames?.length) return entry;
    const protectedInput = taskId === newest
      && String(entry.phase || p.phase) === "started";
    const frames = rawFrames.map((value) => {
      if (!isRecord(value)) return value;
      return imageTasks.has(taskId)
        ? compactFrames([value as MmTrajectoryFrame], remaining, protectedInput)?.[0] || value
        : withoutFrameImage(value as MmTrajectoryFrame);
    });
    return { ...entry, payload: { ...p, frames } };
  }).reverse();
}

export function queryWorkerProgressFromTrajectory(item: MmTrajectoryEntry): {
  taskId: string;
  step: QueryWorkerProgressStep;
} | null {
  const p = item.payload || {};
  const taskId = typeof p.task_id === "string" ? p.task_id : "";
  const parentId = typeof p.parent_user_message_id === "string"
    ? p.parent_user_message_id : "";
  if (!taskId || (!taskId.startsWith("qry_") && !parentId)) return null;
  const ev = isRecord(p.event) ? p.event : {};
  const outerPhase = String(item.phase || p.phase || "progress");
  const phase = String(ev.phase || ev.type || outerPhase);
  const channel = String(ev.channel || p.channel || "").toLowerCase();
  // Raw token-level thinking would add hundreds of rows and is not part of the
  // public execution trace.  Decisions below are synthesized from structured
  // fields (can_answer/tool calls/evidence), while every Recall/Search action
  // remains inspectable.
  if (outerPhase === "router_thinking" || phase === "router_thinking") return null;

  let worker = String(item.worker || "QueryWorker");
  if (channel === "recall" || outerPhase === "recall_done") worker = "RecallWorker";
  if (channel === "search" || outerPhase === "search_done") worker = "SearchWorker";
  let title = phase;
  let detail = "";
  let metrics: string[] = [];
  let plannedTools: RecallTraceToolCall[] = [];
  let toolResults: RecallTraceToolObs[] = [];
  let ocrRecords: QueryWorkerOcrRecord[] = [];
  let ocrState: QueryWorkerProgressStep["ocrState"];
  let ocrReason: string | undefined;
  let ocrRecordCount: number | undefined;
  let ocrElapsedSec: number | undefined;
  let status: QueryWorkerProgressStep["status"] = "running";
  let terminal = false;
  const roundRaw = Number(ev.round);
  const decisionRound = /^r(\d+)_decision$/.exec(phase);
  const round = Number.isFinite(roundRaw)
    ? roundRaw + 1
    : decisionRound ? Number(decisionRound[1]) + 1 : undefined;
  const observations = Array.isArray(ev.observations)
    ? ev.observations.filter(isRecord) : [];
  const recallTasks = Array.isArray(ev.recall_tasks) ? ev.recall_tasks : [];
  const toolCalls = Array.isArray(ev.tool_calls) ? ev.tool_calls : [];
  const rawFrames = Array.isArray(p.frames) ? p.frames : [];
  const elapsedRaw = Number(ev.elapsed_sec ?? p.elapsed_sec);
  const elapsed = Number.isFinite(elapsedRaw) ? elapsedRaw : undefined;
  const clipMetric = sourceClipMetric(ev.source_clip ?? p.source_clip);
  const taskRef = typeof ev.task_id === "string" && ev.task_id !== taskId
    ? ev.task_id : undefined;
  let callState: QueryWorkerProgressStep["callState"];
  const addMetric = (value: string | undefined) => {
    if (value) metrics.push(value);
  };

  if (outerPhase === "started") {
    title = translateNow("multimodal.recall.started", Number(p.n_frames || 0));
    addMetric(Number(p.ask_ts) ? `ask_ts ${Number(p.ask_ts).toFixed(1)}s` : undefined);
  } else if (outerPhase === "ocr_evidence") {
    worker = "OCR";
    ocrRecords = normalizeQueryWorkerOcrRecords(
      p.evidence ?? p.records ?? ev.evidence ?? ev.records,
    );
    const countRaw = Number(p.record_count ?? ev.record_count ?? ocrRecords.length);
    ocrRecordCount = Number.isFinite(countRaw) && countRaw >= 0
      ? Math.floor(countRaw) : ocrRecords.length;
    const elapsedOcrRaw = Number(p.elapsed_sec ?? ev.elapsed_sec);
    ocrElapsedSec = Number.isFinite(elapsedOcrRaw) && elapsedOcrRaw >= 0
      ? elapsedOcrRaw : undefined;
    ocrReason = String(p.reason ?? ev.reason ?? "").trim() || undefined;
    const stateRaw = String(
      p.evidence_state ?? ev.evidence_state ?? p.status ?? ev.status ?? "",
    ).trim().toLowerCase();
    if (stateRaw === "skipped") {
      ocrState = "skipped";
    } else if (stateRaw === "timeout" || ocrReason === "deadline_exceeded") {
      ocrState = "timeout";
    } else if (stateRaw === "error" || stateRaw === "failed") {
      ocrState = "error";
    } else if (ocrRecords.length || ocrRecordCount > 0 || stateRaw === "available") {
      ocrState = "available";
    } else {
      ocrState = "empty";
    }
    status = ocrState === "timeout" || ocrState === "error" ? "error" : "complete";
    title = ocrState === "available"
      ? translateNow("multimodal.ocr.helperAvailable", ocrRecordCount)
      : ocrState === "skipped" ? translateNow("multimodal.ocr.helperSkipped")
        : ocrState === "timeout" ? translateNow("multimodal.ocr.helperTimeout")
          : ocrState === "error" ? translateNow("multimodal.ocr.helperError")
            : translateNow("multimodal.ocr.helperEmpty");
  } else if (outerPhase === "delegate_start") {
    title = translateNow("multimodal.recall.analysisStart");
  } else if (outerPhase === "router_react") {
    const noTools = recallTasks.length === 0 && toolCalls.length === 0;
    title = noTools
      ? translateNow("multimodal.recall.planRoundNoTools")
      : translateNow("multimodal.recall.planRound", recallTasks.length, toolCalls.length);
    plannedTools = [
      ...toolCalls.filter(isRecord).map((call) => ({
        name: String(call.name || "search tool"),
        ...(isRecord(call.args) ? { args: call.args } : {}),
        ...(typeof call.anchor === "string" ? { anchor: call.anchor } : {}),
      })),
      ...recallTasks.filter(isRecord).map((call) => ({
        name: "recall_memory",
        args: { brief: String(call.brief || "") },
      })),
    ];
    callState = "planned";
    addMetric(round ? translateNow("multimodal.recall.outerRound", round) : undefined);
    addMetric(elapsed != null ? `${elapsed.toFixed(2)}s` : undefined);
  } else if (outerPhase === "recall_skipped") {
    worker = "RecallWorker";
    title = ev.reason === "retry_limit_after_two_failures"
      ? translateNow("multimodal.recall.retryLimitStop")
      : translateNow("multimodal.recall.skipDuplicate");
    detail = String(ev.brief || "");
    addMetric(round ? translateNow("multimodal.recall.outerRound", round) : undefined);
  } else if (outerPhase === "bg_progress" && channel === "recall") {
    if (phase === "bg_progress") {
      title = translateNow("multimodal.recall.recallCall", String(ev.tool_name || "recall_memory"), round || undefined);
      detail = String(ev.brief || ev.obs_summary || "");
      if (typeof ev.tool_name === "string") {
        plannedTools = [{
          name: ev.tool_name,
          ...(isRecord(ev.args) ? { args: ev.args } : {}),
        }];
        callState = "called";
      }
    } else if (phase === "start") {
      title = translateNow("multimodal.recall.startRecall");
      detail = String(ev.brief || "");
      addMetric(typeof ev.model === "string" ? `model ${ev.model}` : undefined);
      addMetric(Number(ev.ask_ts) ? `ask_ts ${Number(ev.ask_ts).toFixed(1)}s` : undefined);
    } else if (phase === "tool_obs") {
      title = translateNow("multimodal.recall.roundRead", round || "?", observations.length);
      toolResults = observations.map((obs) => ({
        name: String(obs.name || "memory tool"),
        ...(isRecord(obs.args) ? { args: obs.args } : {}),
        ...(Number.isFinite(Number(obs.obs_len)) ? { obs_len: Number(obs.obs_len) } : {}),
        ...(Number.isFinite(Number(obs.elapsed_sec)) ? { elapsed_sec: Number(obs.elapsed_sec) } : {}),
        ...(typeof obs.obs_summary === "string" ? { obs_summary: obs.obs_summary } : {}),
        ...(Array.isArray(obs.frame_ids)
          ? { frame_ids: obs.frame_ids.map(String).filter(Boolean) }
          : {}),
        ...(Array.isArray(obs.evidence_segments)
          ? {
              evidence_segments: obs.evidence_segments
                .filter(isRecord)
                .slice(0, 12)
                .map((segment) => ({
                  ...(typeof segment.kind === "string" ? { kind: segment.kind } : {}),
                  ...(Number.isFinite(Number(segment.t_start)) ? { t_start: Number(segment.t_start) } : {}),
                  ...(Number.isFinite(Number(segment.t_end)) ? { t_end: Number(segment.t_end) } : {}),
                  ...(Array.isArray(segment.frame_ids)
                    ? { frame_ids: segment.frame_ids.map(String).filter(Boolean) }
                    : {}),
                  ...(typeof segment.preview === "string" ? { preview: segment.preview } : {}),
                })),
            }
          : {}),
      }));
      addMetric(Number.isFinite(Number(ev.parallel_elapsed_sec))
        ? translateNow("multimodal.recall.parallelRead", Number(ev.parallel_elapsed_sec).toFixed(2)) : undefined);
      addMetric(Array.isArray(ev.new_frame_ids) && ev.new_frame_ids.length
        ? translateNow("multimodal.recall.newEvidenceFrames", ev.new_frame_ids.length) : undefined);
    } else if (phase === "distill") {
      title = translateNow("multimodal.recall.roundDistill", round || "?");
      detail = String(ev.clue || "");
    } else if (decisionRound) {
      const canAnswer = ev.can_answer === true;
      const nNext = Number(ev.n_next_calls || 0);
      const verdict = canAnswer
        ? translateNow("multimodal.recall.decisionEnough")
        : nNext ? translateNow("multimodal.recall.decisionContinueTools", nNext)
          : translateNow("multimodal.recall.decisionNoTools");
      title = translateNow("multimodal.recall.roundDecision", round || "?", verdict);
      detail = String(ev.decision_summary || ev.useful_info || "");
      const nextCalls = Array.isArray(ev.next_tool_calls)
        ? ev.next_tool_calls.filter(isRecord) : [];
      plannedTools = nextCalls.map((call) => ({
        name: String(call.name || "memory tool"),
        ...(isRecord(call.args) ? { args: call.args } : {}),
      }));
      callState = "planned";
      addMetric(`can_answer ${String(canAnswer)}`);
      addMetric(Number(ev.n_clues_so_far || 0)
        ? translateNow("multimodal.recall.cluesSoFar", Number(ev.n_clues_so_far)) : undefined);
      if (ev.useful_info && ev.decision_summary) {
        detail += `${detail ? "\n" : ""}${translateNow("multimodal.recall.evidenceSummary", String(ev.useful_info))}`;
      }
    } else if (phase === "tool_skipped") {
      title = translateNow("multimodal.recall.roundSkipDuplicate", round || "?");
      detail = String(ev.name || "memory tool");
      plannedTools = [{
        name: String(ev.name || "memory tool"),
        ...(isRecord(ev.args) ? { args: ev.args } : {}),
      }];
    } else if (phase === "verify") {
      title = translateNow("multimodal.recall.visualReview", Number(ev.n_kept || 0), Number(ev.n_in || 0));
      detail = String(ev.visual_correction || translateNow("multimodal.recall.noVisualConflict"));
    } else if (phase === "fast_table") {
      status = "complete";
      const toolName = String(ev.tool_name || "search_screen_text");
      title = translateNow("multimodal.recall.quickTool", toolName, Number(ev.findings_len || 0));
      detail = String(ev.findings_preview || ev.obs_summary || "");
      toolResults = [{
        name: toolName,
        ...(isRecord(ev.args) ? { args: ev.args } : {}),
        ...(Number.isFinite(Number(ev.obs_len ?? ev.findings_len))
          ? { obs_len: Number(ev.obs_len ?? ev.findings_len) } : {}),
        ...(elapsed != null ? { elapsed_sec: elapsed } : {}),
        ...(typeof ev.obs_summary === "string"
          ? { obs_summary: ev.obs_summary }
          : typeof ev.findings_preview === "string"
            ? { obs_summary: ev.findings_preview } : {}),
        ...(Array.isArray(ev.frame_ids)
          ? { frame_ids: ev.frame_ids.map(String).filter(Boolean) }
          : {}),
        ...(Array.isArray(ev.evidence_segments)
          ? {
              evidence_segments: ev.evidence_segments
                .filter(isRecord)
                .slice(0, 12)
                .map((segment) => ({
                  ...(typeof segment.kind === "string" ? { kind: segment.kind } : {}),
                  ...(Number.isFinite(Number(segment.t_start)) ? { t_start: Number(segment.t_start) } : {}),
                  ...(Number.isFinite(Number(segment.t_end)) ? { t_end: Number(segment.t_end) } : {}),
                  ...(Array.isArray(segment.frame_ids)
                    ? { frame_ids: segment.frame_ids.map(String).filter(Boolean) }
                    : {}),
                  ...(typeof segment.preview === "string" ? { preview: segment.preview } : {}),
                })),
            }
          : {}),
      }];
      addMetric(elapsed != null ? `${elapsed.toFixed(2)}s` : undefined);
    } else if (phase === "error") {
      status = "error";
      title = translateNow("multimodal.recall.failed", ev.stage ? String(ev.stage) : undefined);
      detail = String(ev.error || translateNow("multimodal.errors.unknown"));
      addMetric(typeof ev.model === "string" ? `model ${ev.model}` : undefined);
      addMetric(elapsed != null ? `${elapsed.toFixed(2)}s` : undefined);
    } else if (phase === "done") {
      status = "complete";
      const found = Number(ev.n_clues || 0) > 0
        || (String(ev.findings_preview || "")
          && String(ev.findings_preview || "") !== RECALL_NO_CLUES);
      title = found
        ? translateNow("multimodal.recall.completeFound", Number(ev.n_clues || 0))
        : translateNow("multimodal.recall.completeNotFound");
      detail = String(ev.findings_preview || "");
      addMetric(Number(ev.rounds || 0) ? translateNow("multimodal.recall.innerRounds", Number(ev.rounds)) : undefined);
      addMetric(Number(ev.findings_len || 0) ? translateNow("multimodal.recall.evidenceChars", Number(ev.findings_len)) : undefined);
      addMetric(elapsed != null ? `${elapsed.toFixed(2)}s` : undefined);
    } else {
      title = translateNow("multimodal.recall.phase", phase, round || undefined);
      detail = String(ev.clue || ev.obs_summary || "");
    }
  } else if (outerPhase === "bg_progress" && channel === "search") {
    const toolName = String(ev.tool_name || "text_search");
    title = phase === "bg_progress"
      ? translateNow("multimodal.recall.searchCall", toolName)
      : translateNow("multimodal.recall.searchPhase", phase === "start" ? translateNow("multimodal.recall.searchStart") : phase);
    detail = String(ev.brief || ev.obs_summary || ev.findings || "");
    plannedTools = [{
      name: toolName,
      ...(isRecord(ev.args) ? { args: ev.args } : {}),
      ...(typeof ev.anchor === "string" ? { anchor: ev.anchor } : {}),
      ...(ev.anchor_ts != null && Number.isFinite(Number(ev.anchor_ts))
        ? { anchor_ts: Number(ev.anchor_ts) } : {}),
    }];
    callState = "called";
  } else if (outerPhase === "recall_done") {
    status = "complete";
    const found = ev.found !== false
      && String(ev.findings_preview || "") !== RECALL_NO_CLUES;
    title = found
      ? translateNow("multimodal.recall.returnFound", Number(ev.n_clues || 0), rawFrames.length)
      : translateNow("multimodal.recall.returnNotFound", rawFrames.length);
    detail = String(ev.findings_preview || "");
    toolResults = [{
      name: String(ev.tool_name || "recall_memory"),
      ...(isRecord(ev.args) ? { args: ev.args } : {}),
      ...(Number.isFinite(Number(ev.findings_len)) ? { obs_len: Number(ev.findings_len) } : {}),
      ...(elapsed != null ? { elapsed_sec: elapsed } : {}),
      ...(typeof ev.findings_preview === "string" ? { obs_summary: ev.findings_preview } : {}),
      ...(Array.isArray(ev.frame_ids)
        ? { frame_ids: ev.frame_ids.map(String).filter(Boolean) }
        : {}),
    }];
    addMetric(Number(ev.rounds || 0) ? translateNow("multimodal.recall.innerRounds", Number(ev.rounds)) : undefined);
    addMetric(Number(ev.findings_len || 0) ? translateNow("multimodal.recall.evidenceChars", Number(ev.findings_len)) : undefined);
    addMetric(elapsed != null ? `${elapsed.toFixed(2)}s` : undefined);
  } else if (outerPhase === "search_done") {
    status = "complete";
    title = translateNow("multimodal.recall.searchReturn", Number(ev.findings_len || 0));
    detail = String(ev.findings_preview || "");
    toolResults = [{
      name: String(ev.tool_name || "text_search"),
      ...(isRecord(ev.args) ? { args: ev.args } : {}),
      ...(Number.isFinite(Number(ev.findings_len)) ? { obs_len: Number(ev.findings_len) } : {}),
      ...(elapsed != null ? { elapsed_sec: elapsed } : {}),
      ...(typeof ev.findings_preview === "string" ? { obs_summary: ev.findings_preview } : {}),
      ...(Array.isArray(ev.source_urls)
        ? { source_urls: ev.source_urls.map(String).filter(Boolean).slice(0, 12) }
        : {}),
      ...(typeof ev.cache_hit === "boolean" ? { cache_hit: ev.cache_hit } : {}),
      ...(typeof ev.anchor === "string" ? { anchor: ev.anchor } : {}),
      ...(ev.anchor_ts != null && Number.isFinite(Number(ev.anchor_ts))
        ? { anchor_ts: Number(ev.anchor_ts) } : {}),
    }];
    addMetric(ev.cache_hit === true ? "cache hit" : undefined);
    addMetric(elapsed != null ? `${elapsed.toFixed(2)}s` : undefined);
  } else if (outerPhase === "answer_ready") {
    title = translateNow("multimodal.recall.composingAnswer");
    detail = String(ev.text_preview || "");
  } else if (["complete", "error", "cancelled"].includes(outerPhase)) {
    terminal = true;
    status = outerPhase as QueryWorkerProgressStep["status"];
    title = outerPhase === "complete" ? translateNow("multimodal.recall.answerComplete")
      : outerPhase === "cancelled" ? translateNow("multimodal.recall.taskCancelled") : translateNow("multimodal.recall.taskFailed");
    detail = String(p.answer_preview || "");
    addMetric(elapsed != null ? `${elapsed.toFixed(2)}s` : undefined);
  } else if (outerPhase === "tool_error") {
    status = "error";
    worker = channel === "recall" ? "RecallWorker"
      : channel === "search" ? "SearchWorker" : worker;
    title = translateNow("multimodal.recall.subtaskFailed", channel === "search" ? "Search" : "Recall", String(ev.target || "unknown"));
    detail = String(ev.findings || translateNow("multimodal.errors.unknown"));
    if (typeof ev.tool_name === "string") {
      plannedTools = [{
        name: ev.tool_name,
        ...(isRecord(ev.args) ? { args: ev.args } : {}),
        ...(typeof ev.anchor === "string" ? { anchor: ev.anchor } : {}),
        ...(ev.anchor_ts != null && Number.isFinite(Number(ev.anchor_ts))
          ? { anchor_ts: Number(ev.anchor_ts) } : {}),
      }];
    }
  } else {
    // Keep the chat card readable; the full unfiltered event remains available
    // in Memory → Debug → worker trajectory.
    return null;
  }

  addMetric(clipMetric);

  const frames = rawFrames.filter(isRecord) as MmTrajectoryFrame[];
  return {
    taskId,
    step: {
      id: item.id,
      seq: item.seq,
      ts: item.ts,
      worker,
      phase: `${outerPhase}:${phase}`,
      title,
      ...(detail.trim() ? { detail: detail.trim() } : {}),
      ...(metrics.length ? { metrics } : {}),
      ...(plannedTools.length ? { plannedTools } : {}),
      ...(toolResults.length ? { toolResults } : {}),
      ...(frames.length ? { frames } : {}),
      ...(ocrState ? { ocrState } : {}),
      ...(ocrRecords.length ? { ocrRecords } : {}),
      ...(ocrReason ? { ocrReason } : {}),
      ...(ocrRecordCount != null ? { ocrRecordCount } : {}),
      ...(ocrElapsedSec != null ? { ocrElapsedSec } : {}),
      ...(taskRef ? { taskRef } : {}),
      ...(callState ? { callState } : {}),
      ...(terminal ? { terminal: true } : {}),
      status,
    },
  };
}


export default function MultimodalChatPage() {
  const { t } = useI18n();
  const refs = useRef<Refs>({
    gw: null, sessionId: "", storedSid: "", stream: null, sourceType: null,
    capFps: 2, capTimer: null,
    startTs: 0, captureAttemptId: "", sentFrames: 0, droppedFrames: 0,
    isAnswering: false,
    micStream: null, micAudioCtx: null, micNode: null, micSource: null,
    isRecording: false, asrTransport: null, micGeneration: 0, micFlushResolve: null,
    micStopPromise: null, micBoundaryPromise: null,
    envStream: null, envRecorder: null, envStop: false, envMime: "audio/webm",
    envWindowSec: 5, envSliceTimer: null, envCaptureId: "", envChunkSeq: 0,
    envLastError: "",
  });
  // The mount-time establish path and the ?mm= watcher must never both create
  // a session for the same `?mm=new` navigation.
  const sessionEstablishedRef = useRef(false);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  // Slash pipeline /undo prefill → ChatComposer's setAskText (registered on
  // mount via composerApiRef).
  const composerApiRef = useRef<{ setText: (text: string) => void } | null>(null);
  // Phase 10: route message.* events by key (request_id / monitor_id / __main__)
  // so concurrent RouterEngine delegations + multi-monitor SPEAKs land in
  // distinct bubbles. Key "__main__" is the regular main-agent turn bubble.
  const curAssistantId = useRef<Map<string, string>>(new Map());
  // Per-key monitor-alert routing. Keys mirror curAssistantId (via keyOf), but
  // point to {monitorId, alertId} pairs so message.delta / message.complete can
  // append to monitorAlerts[monitorId] instead of the center-chat messages list.
  const curMonitorAlertId = useRef<Map<string, { monitorId: string; alertId: string }>>(new Map());
  const queryProgressByTaskRef = useRef<
    Map<string, QueryWorkerProgressStep[]>
  >(new Map());
  // Invalidates trajectory.list responses that race a session switch or a
  // newer hydrate request for the same live session.
  const trajectoryHydrationGenerationRef = useRef(0);
  // Voice Dialog is continuous user intent layered over an exact live-session
  // ASR owner. The recovery state survives transport/session replacement while
  // each activation remains bound to one sid + attempt.
  const voiceDialogRecoveryRef = useRef(new VoiceDialogRecovery());
  const runVoiceDialogActivationRef = useRef<
    (activation: VoiceDialogActivation) => void
  >(() => undefined);
  const ttsRefs = useRef<TtsRefs>({
    audioCtx: null, audioNextStart: 0, active: [], ttsMuteUntil: 0,
    currentRid: null, cancelled: new Set(),
    ctxStartTime: 0, scheduledSec: 0,
  });

  const { setAfterTitle, setEnd } = usePageHeader();
  const { setOpen: setCliDrawerOpen } = useCliDrawer();
  // ?mm=<id> selects which session to open (set by the sidebar session list).
  // scopedProfile scopes the "default = newest session" lookup on first load.
  const [searchParams, setSearchParams] = useSearchParams();
  const mmParam = searchParams.get("mm");
  const { profile: scopedProfile } = useProfileScope();
  const [connected, setConnected] = useState(false);
  // Multimodal readiness advisory (soft, non-blocking) — fetched once on connect
  // over the page's own gateway connection (no extra WS), rendered as a banner.
  const [mmReadiness, setMmReadiness] = useState<MmReadinessReport | null>(null);
  // Raw connection state for a 3-way badge (已连接 / 重连中 / 未连接).
  const [connState, setConnState] = useState<string>("connecting");
  const [model, setModel] = useState("");
  const [sourceType, setSourceType] = useState<SourceType>(null);
  const [frameCount, setFrameCount] = useState(0);
  // Mic lifecycle: local capture begins while the backend connects; a manual
  // second click stops the physical track immediately, then waits in
  // ``finalizing`` until the backend has flushed and submitted exactly once.
  const [micState, setMicState] = useState<MicLifecycleState>("idle");
  const isRecordingUI = micState === "recording";
  // 独立 TTS 语音播报开关 (与麦克风解耦)。默认关; 切换时通知后端 announcer。
  // toggleTts 定义在 pushTopToast 之后 (需引用它做"对话托管"拦截提示)。
  const [ttsEnabled, setTtsEnabled] = useState(false);
  // 对话模式状态仍由 session 恢复 / 麦喇叭托管逻辑使用; UI 不再提供入口按钮。
  const [voiceDialogEnabled, setVoiceDialogEnabled] = useState(false);
  const [messages, setMessages] = useState<ChatMsg[]>(() => [_mmWelcomeMsg()]);
  // ★ 聊天列表已改普通 div 全量渲染 (非虚拟化) —— 渲染成本随消息数线性增长, 所以这个
  //   软上限从 5000 降到 400: 只保留最近 400 条 chat+tool+status 气泡, 更早的 tail-slice
  //   丢弃, 防长会话全量渲染卡顿 / 堆增长 (气泡可能带 base64 图)。400 对单次工作会话
  //   足够, 超出的历史仍在后端 DB, 重开某更早节点可再看。
  const MAX_MESSAGES = 400;
  const HISTORY_PAGE = 200;   // 向上翻到顶时每次补进渲染窗的历史条数
  // ★ 历史回看: reopen 时后端一次性返回**全部**历史。之前 capMsgs 直接 slice 掉头部
  //   400 条外的, 用户翻到顶就再也看不到 → 现在把全量历史存进 ref(不渲染), 只渲染尾部
  //   窗口; 滚到顶再从 ref 预取上一段补进窗口 (见 loadOlderHistory)。
  const fullHistoryRef = useRef<ChatMsg[]>([]);   // reopen 的完整历史 (含已划出窗口的头部)
  const [hasMoreHistory, setHasMoreHistory] = useState(false);  // 窗口上方还有没有更早历史
  const capMsgs = (list: ChatMsg[]) => {
    // Stamp a client-local creation time on any message that lacks one (new
    // items are appended last; already-stamped ones are untouched) so each
    // bubble can show an absolute HH:MM:SS timestamp.
    const now = Date.now();
    for (const m of list) if (m.createdAt == null) m.createdAt = now;
    return list.length > MAX_MESSAGES ? list.slice(list.length - MAX_MESSAGES) : list;
  };
  const [ctx, setCtx] = useState<CtxState>({ version: 0, obs: [], audioObs: [], facts: {} });
  const [anchorFrames, setAnchorFrames] = useState<{ ts: number | null; jpeg_b64: string }[]>([]);
  const [bgItems, setBgItems] = useState<BgItem[]>([]);
  // Per-monitor alert history. Keyed by monitor_id. Rendered in the right
  // multimodal panel (never as center-chat bubbles). Hydrated on session
  // resume via the multimodal.list_monitor_alerts RPC.
  const [monitorAlerts, setMonitorAlerts] = useState<Map<string, MonitorAlert[]>>(() => new Map());
  // Title-bar collapse for a monitor's panel (default expanded: not in set).
  const [monitorCollapsed, setMonitorCollapsed] = useState<Set<string>>(() => new Set());
  // "展开更多" — reveal older alerts beyond the default 2 (default off).
  const [monitorExpanded, setMonitorExpanded] = useState<Set<string>>(() => new Set());
  // Which deep-research sub-window is expanded (request_id). Only the newest is
  // open by default; clicking a title toggles. null = default (newest open).
  const [deepExpanded, setDeepExpanded] = useState<string | null>(null);
  // ★ 用户一旦显式点击过窗口头, 后台 bg flush 的"自动展开最新窗口"就不再抢焦点 ——
  //   否则完成前的拖尾事件会把用户刚点开的旧窗口又顶回最新窗口 ("点开又被关回去")。
  const deepExpandedUserPinned = useRef(false);
  const [monitors, setMonitors] = useState<MonitorReg[]>([]);
  // 右侧面板底部 toast (监控/深度研究过程失败/停用), 3s 后自动移除。不进 history、不发主气泡。
  const [mmToasts, setMmToasts] = useState<{ id: string; level: string; text: string }[]>([]);
  // 顶部居中 toast (页面级操作提示, 如"未开启视频流无法恢复监控"), 3s 淡出。
  const [topToasts, setTopToasts] = useState<{ id: string; level: string; text: string }[]>([]);
  const [memoryDebugOpen, setMemoryDebugOpen] = useState(false);
  const [trajectory, setTrajectory] = useState<MmTrajectoryEntry[]>([]);
  const pushTopToast = useCallback((text: string, level: string = "warning") => {
    const id = nid();
    setTopToasts((prev) => [...prev, { id, level, text }]);
    setTimeout(() => setTopToasts((prev) => prev.filter((x) => x.id !== id)), 2000);
  }, []);
  // TTS 播报开关切换。★ 对话模式开时喇叭由对话托管 (后端 is_speaker_on OR 对话态
  //   已强制 TTS 生效), 单独点喇叭无效 → 拦截 + 顶部小提示 (按钮态不变)。
  const toggleTts = useCallback(() => {
    if (voiceDialogEnabled) {
      pushTopToast(t.multimodal.toasts.dialogModeTtsAutoEnabled, "info");
      return;
    }
    setTtsEnabled((prev) => {
      const next = !prev;
      const r = refs.current;
      try {
        r.gw?.request("multimodal.tts_toggle",
          { session_id: r.sessionId, enabled: next }).catch(() => {});
      } catch { /* noop */ }
      return next;
    });
  }, [voiceDialogEnabled, pushTopToast]);
  // Watcher (set_live_watcher) registry — mirrors monitors. A reopened session
  // re-registers interrupted watchers (disabled) so this list + on/off toggle
  // can surface + re-enable them (parity with desktop WatcherList).
  // (WatcherReg type hoisted to module scope.)
  const [watchers, setWatchers] = useState<WatcherReg[]>([]);
  // Ref so the (event-handler-scoped) watcher.report_append handler can read the
  // latest registry for a report's label without re-subscribing on every change.
  const watchersRef = useRef<WatcherReg[]>([]);
  useEffect(() => { watchersRef.current = watchers; }, [watchers]);
  const [ttsPlaying, setTtsPlaying] = useState(false);
  const [asrPartial, setAsrPartial] = useState("");
  // EOU 监听中状态下已拼接但未 flush 的各段文本 (由 asr_buffer 事件更新)
  const [asrBuffer, setAsrBuffer] = useState<string[]>([]);
  // ★ 聊天列表改用普通滚动 div (去掉 react-virtuoso) —— Virtuoso 遇到流式中突然变高
  //   的 item(大表格) + followOutput:"auto" 会进 measure→scroll→remeasure 同步死循环,
  //   把主线程占死 (大型 Markdown 工具结果已用 F12 复现)。普通 div 全量渲染无此问题;
  //   消息量由 capMsgs 软上限兜底 (见下)。
  const chatScrollRef = useRef<HTMLDivElement | null>(null);
  const chatAtBottomRef = useRef(true);        // 用户是否停在底部 (ref, 不触发重渲)
  const [atBottom, setAtBottom] = useState(true);  // 同值 state, 仅驱动"跳到最新"按钮显隐
  // ★ 翻到顶补历史: 从 fullHistoryRef 取当前渲染窗上方的一段 (HISTORY_PAGE 条) prepend
  //   进 messages。prepend 会撑高内容 → 视口会跳; 用 scrollHeight 差值补 scrollTop 保持
  //   用户视线锚定在原来那条上 (无跳动)。同步桥用 useLayoutEffect (见下 pendingPrependRef)。
  const pendingPrependRef = useRef<number>(0);   // 本次 prepend 前的 scrollHeight, 供布局后补偿
  const loadOlderHistory = useCallback(() => {
    const el = chatScrollRef.current;
    const full = fullHistoryRef.current;
    if (!el || full.length === 0) return;
    setMessages((cur) => {
      // 置顶欢迎气泡不在 fullHistoryRef 里 (纯前端), 定位/长度都要跳过它。
      const welcomeAtHead = cur.length > 0 && cur[0]?.role === "system"
        && cur[0]?.text === _mmWelcomeMsg().text ? 1 : 0;
      const realCurLen = cur.length - welcomeAtHead;
      if (realCurLen >= full.length) { setHasMoreHistory(false); return cur; }
      // cur 的头部 (跳过 welcome) 对应 full 里的某个位置: 用第一条真实历史 id 定位 (id 稳定)。
      const firstId = cur[welcomeAtHead]?.id;
      let headIdx = firstId ? full.findIndex((m) => m.id === firstId) : full.length - realCurLen;
      if (headIdx < 0) headIdx = Math.max(0, full.length - realCurLen);
      if (headIdx <= 0) { setHasMoreHistory(false); return cur; }
      const newStart = Math.max(0, headIdx - HISTORY_PAGE);
      pendingPrependRef.current = el.scrollHeight;   // 记录撑高前高度, 布局后补偿
      const older = full.slice(newStart, headIdx);
      setHasMoreHistory(newStart > 0);
      // welcome 仍留在顶: [welcome?, older..., realCur...]
      return welcomeAtHead
        ? [cur[0], ...older, ...cur.slice(1)]
        : [...older, ...cur];
    });
  }, []);
  // prepend 后校正 scrollTop: 新内容撑高了 scrollHeight, 加上差值让视线不跳。
  useLayoutEffect(() => {
    const el = chatScrollRef.current;
    if (!el || pendingPrependRef.current === 0) return;
    const delta = el.scrollHeight - pendingPrependRef.current;
    pendingPrependRef.current = 0;
    if (delta > 0) el.scrollTop = el.scrollTop + delta;
  });
  // 滚动监听: 更新"是否在底部"。阈值 40px 容差。翻到顶(≤80px)且还有更早历史 → 补一页。
  const onChatScroll = useCallback(() => {
    const el = chatScrollRef.current;
    if (!el) return;
    const bottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
    chatAtBottomRef.current = bottom;
    setAtBottom((prev) => (prev === bottom ? prev : bottom));
    if (el.scrollTop <= 80 && hasMoreHistory) loadOlderHistory();
  }, [hasMoreHistory, loadOlderHistory]);
  const scrollChatToBottom = useCallback((smooth = false) => {
    const el = chatScrollRef.current;
    if (!el) return;
    const behavior: ScrollBehavior = smooth ? "smooth" : "auto";
    el.scrollTo({ top: el.scrollHeight, behavior });
    if (!smooth) {
      requestAnimationFrame(() => {
        const next = chatScrollRef.current;
        if (next) next.scrollTop = next.scrollHeight;
      });
    }
  }, []);
  const obsScrollRef = useRef<HTMLDivElement | null>(null);
  const audioObsScrollRef = useRef<HTMLDivElement | null>(null);
  // Auto-scroll the observation timelines to the bottom whenever new items
  // arrive — newest observation is now rendered at the bottom (natural
  // chronological order), so the interesting content is what we scroll to.
  useEffect(() => {
    const el = obsScrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [ctx.obs.length, ctx.version]);
  useEffect(() => {
    const el = audioObsScrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [ctx.audioObs.length]);
  const factsList = useMemo(() => Object.entries(ctx.facts), [ctx.facts]);

  // Group consecutive progress entries (tool/status) into one "background" block,
  // so the chat reads as: chat bubble → [background card] → chat bubble.
  // (Row type hoisted to module scope so the column components can use it.)
  //
  // ★ 性能(#6 BgBlock memo 稳定): 每次 messages 变(哪怕只有一个 chat 气泡在流式
  //   追加 text), rows 都会重建, 每个 bg 行都 new 一个 items 数组 → 若 BgBlock 用默认
  //   浅比 (比 items 引用) 则全部失效重渲染。解决放在 BgBlock 自己的 memo 比较器里
  //   (按 items 【逐元素引用】比较, 见 BgBlock 定义处) —— tool/status 对象在纯 chat
  //   流式期间 identity 不变, 故内容相同的 bg 块 memo 命中、跳过。��处保持纯函数、
  //   不在 render 期读写 ref。
  const rows = useMemo<Row[]>(() => buildRows(messages), [messages]);

  // Live gateway+session handed to leaf controls that issue session-scoped RPCs
  // (the composer's thinking dial). Identity is stable for the page's lifetime —
  // it reads refs.current on call instead of capturing gw/sessionId, so a
  // reconnect or session switch can't invalidate the memoized composer subtree.
  const chatSessionCtx = useMemo(
    () => ({
      resolve: () => {
        const r = refs.current;
        return r.gw && r.sessionId ? { gw: r.gw, sessionId: r.sessionId } : null;
      },
    }),
    [],
  );

  // (Chat auto-scroll useLayoutEffect moved into <ChatColumn> so it only runs
  //  when `rows` changes — not on every ctx/anchor/frameCount setState.)

  // RouterEngine deep-research sub-windows: group the router bubbles by
  // requestId (each request_id = one delegation, possibly multi-round via the
  // Clarify loop). Newest first. Rendered in the left column under the video.
  const deepWindows = useMemo(() => {
    const byRid = new Map<string, ChatMsg[]>();
    for (const m of messages) {
      if (m.subRole !== "router" || m.threadback) continue;
      const rid = m.requestId || "__deep__";
      const arr = byRid.get(rid);
      if (arr) arr.push(m); else byRid.set(rid, [m]);
    }
    // Progress (multimodal.bg) can lead message.start by a tick — still show a
    // sub-window shell so the user sees incremental search/recall updates.
    for (const b of bgItems) {
      if (!b.requestId || byRid.has(b.requestId)) continue;
      byRid.set(b.requestId, []);
    }
    // Insertion order = chronological; reverse for newest-first display.
    return Array.from(byRid.entries()).reverse().map(([rid, msgs]) => ({ rid, msgs }));
  }, [messages, bgItems]);

  // One bg item per request_id now (the reducer groups internally), so map
  // rid → its single BgItem for O(1) lookup by the windows below.
  const bgByRid = useMemo(() => {
    const byRid = new Map<string, BgItem>();
    for (const b of bgItems) {
      const rid = b.requestId || "__deep__";
      if (!byRid.has(rid)) byRid.set(rid, b);
    }
    return byRid;
  }, [bgItems]);

  // A deep-research delegation is "active" while it is still streaming, has
  // unfinished background search/recall, or is waiting on a Clarify follow-up.
  // The right RouteEngine column only opens for active delegations (plus any the
  // user explicitly re-opened for review via a thread-back bubble). Once a rid
  // finishes AND isn't the re-opened one, it drops out → the column closes.
  // A window is shown if it is still working OR it has produced any content
  // (segments / final report). req A-fix: a FINISHED watcher must NOT vanish —
  // its process segments + final report stay visible in-panel (the whole point
  // of the panel). It only disappears once evicted by the bgItems cap (last 8).
  const ridIsActive = useCallback((rid: string, msgs: ChatMsg[]) => {
    if (msgs.some((m) => m.streaming)) return true;
    const b = bgByRid.get(rid);
    if (!b) return false;
    if (!b.done) return true;                 // still working
    if (b.finalReport) return true;           // finished — keep the result visible
    if (b.segments && b.segments.length) return true;  // keep the process visible
    return false;
  }, [bgByRid]);

  const visibleDeep = useMemo(
    () => deepWindows.filter(({ rid, msgs }) => ridIsActive(rid, msgs)),
    [deepWindows, ridIsActive],
  );
  // 右侧列承载: 监控注册表 + 深度分析注册表 + 深度研究窗口 (与桌面端 deep-panel 一致)。
  // 只按【未完成】任务自动打开: watcher status ∈ {running,interrupted,disabled} 都算未完成;
  // monitor 在册即算; visibleDeep 含实时进度 + 用户手动重开的只读窗口。
  // ★ 五态统一: "未完成"= running / interrupted / stopping (disabled 已并入 interrupted;
  //   done/deleted 视为已结束, 不撑开面板)。
  const hasIncompleteWatcher = watchers.some((w) =>
    ["running", "interrupted", "stopping"].includes(String(w.status || "")));
  // 有 toast 时也保持右列可见 (监控停用后可能已无活跃任务, 否则 toast 无处可显)。
  const showDeepCol = visibleDeep.length > 0 || monitors.length > 0 || hasIncompleteWatcher || mmToasts.length > 0;
  const generating = useMemo(() => messages.some((m) => m.streaming), [messages]);

  // 监控 / 深度分析 注册表 toggle 回调 (稳定引用 → <RegistryPanels> memo 命中)。
  // ★ I: 不再用客户端 r.stream 硬拦。后端 is_source_live() 判据是 _source_stopped
  //   (前端 source_stopped{started} RPC 驱动) + "从未采集过" 兜底, 不是 buffer 有无帧。
  //   以后端为准: 直接发, 后端拒了走 catch → rollback + toast。
  const onToggleMonitor = useCallback((m: MonitorReg) => {
    const on = m.enabled !== false;
    const label = (m.label && m.label.trim()) || m.brief.slice(0, 10) || t.multimodal.monitor.title;
    const r = refs.current;
    if (!r.gw || !r.sessionId) return;
    setMonitors((prev) => prev.map((x) =>
      x.monitor_id === m.monitor_id ? { ...x, enabled: !on } : x));
    r.gw.request("multimodal.monitor_toggle", {
      session_id: r.sessionId, monitor_id: m.monitor_id, enabled: !on,
    }).then(() => {
      // ★ H: 成功后拉权威注册表对账 (push best-effort, 丢了会永久 desync)。
      refs.current.fetchRegistries?.(r.sessionId);
    }).catch((e: { error?: string; message?: string }) => {
      setMonitors((prev) => prev.map((x) =>
        (x.monitor_id === m.monitor_id && x.enabled === !on)
          ? { ...x, enabled: on } : x));
      // ★ M: 开启 AND 关闭失败都提示。
      pushTopToast(
        t.multimodal.toasts.monitorToggleFailed(!on, label, e?.error || e?.message || t.multimodal.errors.unknown),
        "error");
    });
  }, [pushTopToast, t]);
  const onToggleWatcher = useCallback((w: WatcherReg) => {
    const on = w.status === "running";
    const label = (w.label && w.label.trim())
      || (w.task_instruction || "").slice(0, 12) || t.multimodal.deepAnalysis.title;
    const r = refs.current;
    if (!r.gw || !r.sessionId) return;
    const want = !on;
    // Optimistic: 开→running; 关→stopping (当前轮收尾, 后端收尾后落 interrupted)。
    setWatchers((prev) => prev.map((x) =>
      x.watcher_id === w.watcher_id
        ? { ...x, status: want ? "running" : "stopping" } : x));
    r.gw.request("multimodal.watcher_toggle", {
      session_id: r.sessionId, watcher_id: w.watcher_id, enabled: want,
    }).then(() => {
      refs.current.fetchRegistries?.(r.sessionId);
    }).catch((e: { error?: string; message?: string }) => {
      setWatchers((prev) => prev.map((x) =>
        (x.watcher_id === w.watcher_id
          && x.status === (want ? "running" : "stopping"))
          ? { ...x, status: on ? "running" : "interrupted" } : x));
      pushTopToast(
        t.multimodal.toasts.watcherToggleFailed(want, label, e?.error || e?.message || t.multimodal.errors.unknown),
        "error");
    });
  }, [pushTopToast, t]);

  // (Chat auto-scroll: see the useLayoutEffect on `rows` above — scrolls the
  // plain scroll div to bottom on new content when the user is already at bottom.)

  const addMsg = useCallback((m: ChatMsg) => setMessages((p) => capMsgs([...p, m])), []);

  const offerVoiceDialogSession = useCallback((sessionId: string) => {
    const recovery = voiceDialogRecoveryRef.current;
    const activation = recovery.sessionAvailable(sessionId);
    if (activation) {
      runVoiceDialogActivationRef.current(activation);
      return;
    }
    // A user may switch OFF while the socket has no live sid. Reconcile the
    // newly resumed backend to the authoritative OFF intent so a durable
    // session cannot retain a stale VoiceAgent-routing bit.
    const r = refs.current;
    if (!recovery.wantsVoiceDialog() && r.gw && r.sessionId === sessionId) {
      void r.gw.request("multimodal.voice_dialog_toggle", {
        session_id: sessionId,
        enabled: false,
      }).catch(() => undefined);
    }
  }, []);

  const markVoiceDialogBoundary = useCallback(() => {
    voiceDialogRecoveryRef.current.boundary();
  }, []);

  const leaveVoiceDialogSession = useCallback((): Promise<void> => {
    const r = refs.current;
    const oldOwnerSid = voiceDialogRecoveryRef.current.boundary();
    // Deliberate A→B/New navigation still has a valid owner transport. Clear
    // A's durable routing bit before B is allowed to rearm continuous ASR.
    const disableOld = oldOwnerSid && r.gw && r.sessionId === oldOwnerSid
      ? r.gw.request("multimodal.voice_dialog_toggle", {
          session_id: oldOwnerSid,
          enabled: false,
        }).catch(() => undefined)
      : Promise.resolve();
    const cancelOldTurn = cancelActiveMic(r);
    const operation = Promise.allSettled([disableOld, cancelOldTurn]).then(() => undefined);
    r.micBoundaryPromise = operation;
    void operation.finally(() => {
      if (r.micBoundaryPromise === operation) r.micBoundaryPromise = null;
    });
    return operation;
  }, []);

  const failWaitingVoiceDialog = useCallback(() => {
    if (!voiceDialogRecoveryRef.current.sessionUnavailable()) return;
    setVoiceDialogEnabled(false);
    const text = translateNow("multimodal.errors.voiceDialogNoSession");
    pushTopToast(text, "error");
    addMsg({ id: nid(), role: "system", text });
  }, [addMsg, pushTopToast]);

  // ★ 切换会话时清空 ALL 上一会话的 UI 状态 (对齐 desktop resetDeepUi + 更全)。
  //   只清 messages/curAssistantId 会让旧会话的深研窗/注入帧/观察面板/监控列表/
  //   toast/帧计数残留到新会话。这里一次清干净; 新会话的 registries 由 resume 后的
  //   fetchRegistries + push 重新填充。
  const resetSessionUi = useCallback(() => {
    trajectoryHydrationGenerationRef.current += 1;
    curAssistantId.current.clear();
    curMonitorAlertId.current.clear();
    queryProgressByTaskRef.current.clear();
    // ★ 清历史回看窗口状态 (切换/新建会话不能串到上一会话的历史)。
    fullHistoryRef.current = [];
    setHasMoreHistory(false);
    // ★ 不清成空 —— 补回置顶"系统"引导气泡 (新建/切换会话都保留)。
    setMessages([_mmWelcomeMsg()]);
    setBgItems([]);
    setMonitors([]);
    setWatchers([]);
    setMonitorAlerts(new Map());
    setMonitorCollapsed(new Set());
    setMonitorExpanded(new Set());
    setMmToasts([]);
    setTopToasts([]);
    setAnchorFrames([]);
    setCtx({ version: 0, obs: [], audioObs: [], facts: {} });
    setDeepExpanded(null);
    deepExpandedUserPinned.current = false;
    setFrameCount(0);
    setAsrPartial("");
    setAsrBuffer([]);
    setTrajectory([]);
    refs.current.sentFrames = 0;
  }, []);

  // ★ 性能: 稳定的 rid 折叠回调 (每个 DeepWindow 复用同一个函数引用 → 不破坏 memo)。
  //   旧代码在 .map 里为每个 window 现造 () => setDeepExpanded(...), 每次父渲染都换
  //   新 onToggle identity → 所有 DeepWindow memo 失效、全部重渲染。
  //   ★ 用户显式交互: 置 pin, 此后后台自动展开不再覆盖 (见 runBgFlush)。
  const toggleDeepWindow = useCallback((rid: string) => {
    deepExpandedUserPinned.current = true;
    setDeepExpanded((cur) => (cur === rid ? "" : rid));
  }, []);
  const toggleMonitorCollapsed = useCallback((mid: string) => {
    setMonitorCollapsed((prev) => {
      const next = new Set(prev);
      if (next.has(mid)) next.delete(mid); else next.add(mid);
      return next;
    });
  }, []);
  const toggleMonitorExpanded = useCallback((mid: string) => {
    setMonitorExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(mid)) next.delete(mid); else next.add(mid);
      return next;
    });
  }, []);

  // ── TTS playback (WebAudio, gapless scheduling) ────────────────────────
  const ensureAudioCtx = useCallback(() => {
    const r = ttsRefs.current;
    if (!r.audioCtx) {
      const AC = (window as any).AudioContext || (window as any).webkitAudioContext;
      r.audioCtx = new AC();
      r.audioNextStart = r.audioCtx!.currentTime;
    }
    if (r.audioCtx!.state === "suspended") r.audioCtx!.resume().catch(() => {});
    return r.audioCtx!;
  }, []);

  const stopAllTts = useCallback((resetRid: boolean) => {
    const r = ttsRefs.current;
    // ★ #2 播放 ack: 停播前先算"当前 rid 实际听了多少", 回传后端截断"我说过什么"。
    //   played = min(经过的挂钟时长, 已排定总时长); total = 已排定总时长。
    if (resetRid && r.currentRid && r.audioCtx && r.scheduledSec > 0) {
      const elapsed = Math.max(0, r.audioCtx.currentTime - r.ctxStartTime);
      const playedSec = Math.min(elapsed, r.scheduledSec);
      const rr = refs.current;
      if (rr.gw && rr.sessionId) {
        rr.gw.request("multimodal.tts_played", {
          session_id: rr.sessionId,
          response_id: r.currentRid,
          played_ms: playedSec * 1000,
          total_ms: r.scheduledSec * 1000,
        }).catch(() => { /* best-effort */ });
      }
    }
    for (const src of r.active) { try { src.stop(); } catch { /* noop */ } }
    r.active = [];
    // Playback stopped early → lift the mic mute now (keep only a short tail for
    // speaker/AEC decay) so the user can talk again immediately.
    r.ttsMuteUntil = Math.min(r.ttsMuteUntil, Date.now() + 300);
    if (resetRid && r.currentRid) {
      r.cancelled.add(r.currentRid);
      // Cap the cancelled Set so it can't grow unbounded over a long session.
      if (r.cancelled.size > 64) {
        r.cancelled = new Set(Array.from(r.cancelled).slice(-32));
      }
      r.currentRid = null;
      r.scheduledSec = 0;
      if (r.audioCtx) r.audioNextStart = r.audioCtx.currentTime;
      setTtsPlaying(false);
    }
  }, []);

  const onTtsChunk = useCallback((msg: {
    response_id?: string; pcm_b64?: string; sample_rate?: number; is_final?: boolean;
  }) => {
    const r = ttsRefs.current;
    const rid = msg.response_id || "";
    // ★ Barge-in sentinel: 后端 interrupt_tts 发 rid="__interrupt__" + is_final=true
    //   通知前端立即停播。之前只按 rid 匹配, 这个 sentinel 匹配不上任何当前 rid → 忽略,
    //   前端已收到的 PCM 继续在 WebAudio 里播完 = "打断没效果"。识别它 → 全停。
    if (rid === "__interrupt__") {
      stopAllTts(true);
      return;
    }
    if (r.cancelled.has(rid)) return;
    if (msg.is_final) {
      if (r.currentRid === rid) {
        // Let queue drain; just clear the playing badge.
        setTtsPlaying(false);
      }
      return;
    }
    if (!msg.pcm_b64) return;
    const ctx = ensureAudioCtx();
    if (r.currentRid !== rid) {
      for (const s of r.active) { try { s.stop(); } catch { /* noop */ } }
      r.active = [];
      r.currentRid = rid;
      r.audioNextStart = ctx.currentTime;
      // ★ #2: 新 rid 开播 → 记起播时刻 + 清零已排定时长 (用于打断时算"实际听了多少")。
      r.ctxStartTime = ctx.currentTime;
      r.scheduledSec = 0;
      setTtsPlaying(true);
    }
    try {
      const bin = atob(msg.pcm_b64);
      const bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      // PCM16 is 2 bytes/sample; `new Int16Array(buffer)` throws RangeError if
      // the byte length is odd (a truncated chunk at a boundary). Drop the
      // trailing odd byte so a malformed frame degrades to a tiny gap instead
      // of a swallowed exception.
      const evenLen = bytes.byteLength & ~1;
      const i16 = new Int16Array(bytes.buffer, 0, evenLen >> 1);
      const f32 = new Float32Array(i16.length);
      for (let i = 0; i < i16.length; i++) f32[i] = i16[i] / 32768.0;
      const sr = msg.sample_rate || 24000;
      const buf = ctx.createBuffer(1, f32.length, sr);
      buf.copyToChannel(f32, 0);
      const src = ctx.createBufferSource();
      src.buffer = buf;
      src.connect(ctx.destination);
      const startAt = Math.max(ctx.currentTime, r.audioNextStart);
      src.start(startAt);
      r.active.push(src);
      r.audioNextStart = startAt + buf.duration;
      r.scheduledSec += buf.duration;   // ★ #2: 累计当前 rid 已排定总时长
      // Mute the mic through this chunk's playback (AudioContext time → wall
      // clock) + a short tail, so speaker output isn't re-captured into ASR.
      const playoutMs = Math.max(0, r.audioNextStart - ctx.currentTime) * 1000;
      r.ttsMuteUntil = Math.max(r.ttsMuteUntil, Date.now() + playoutMs + 300);
      src.onended = () => {
        const i = r.active.indexOf(src);
        if (i >= 0) r.active.splice(i, 1);
      };
    } catch { /* drop chunk */ }
  }, [ensureAudioCtx, stopAllTts]);

  // ── Gateway session lifecycle ──────────────────────────────────────────
  useEffect(() => {
    const gw = new GatewayClient();
    refs.current.gw = gw;
    const asrTransport = new AsrTurnTransport(gw);
    refs.current.asrTransport = asrTransport;

    // ★ Session-scoped guard (对齐 desktop mine()). Backend _emit stamps every
    //   event with session_id = the LIVE sid. After a ?mm= switch (or any
    //   session change) refs.current.sessionId points at the NEW session, so
    //   straggler events from the OLD session (still finishing server-side)
    //   carry a different sid → dropped here instead of polluting the new
    //   session's waterfall/panels. Untagged events (no session_id) are treated
    //   as ours (legacy/broadcast).
    const isMine = (ev: { session_id?: string }): boolean =>
      !ev.session_id || ev.session_id === refs.current.sessionId;

    // Phase 10: derive a routing key from payload (request_id > monitor_id >
    // "__main__"). The frontend keeps one bubble per key so concurrent
    // RouterEngine delegations + multi-monitor SPEAKs don't share streams.
    const keyOf = (p: any): string =>
      (p && (p.request_id || p.monitor_id)) || "__main__";
    let activeForegroundKey = "__main__";
    // Trace helper — logs one milestone the first time it's seen this turn.
    // Cleared when sendAsk fires a new turn (via __mmTraceLast reset).
    const traceOnce = (stage: string, color = "#0d6efd") => {
      const w = window as any;
      const t = w.__mmTraceLast;
      if (!t || t.seen[stage]) return;
      t.seen[stage] = true;
      // eslint-disable-next-line no-console
      console.log(
        `%c[mm-trace-fe] +${(performance.now() - t.t0).toFixed(0)}ms ${stage}`,
        `color:${color}`,
      );
    };

    // Create the assistant bubble for a stream key and register its id. Shared
    // by message.start AND message.delta/.complete: if a delta/complete arrives
    // for a key that has no bubble yet (events can arrive out of order — a
    // watcher delta racing ahead of its message.start would otherwise be
    // dropped, leaving the sub-window title present but body empty), we lazily
    // synthesize the bubble here from the event's own payload so no token is
    // lost. Returns the bubble id.
    // Monitor alerts are routed to the right multimodal panel (monitorAlerts
    // state), NOT to the center chat. Everything else (main agent, query
    // worker, deep-research router bubbles) still creates a center-chat bubble.
    // Watcher no longer streams message.* into the chat at all — its content
    // arrives through watcher.report_append + multimodal.bg (right panel only).
    const ensureBubble = (p: {
      source?: string; request_id?: string;
      monitor_id?: string; monitor_label?: string; brief?: string;
    }): string => {
      const key = keyOf(p);
      const isMonitor = p.source === "monitor" || !!p.monitor_id;
      if (isMonitor) {
        const monitorId = p.monitor_id || key;
        const existing = curMonitorAlertId.current.get(key);
        if (existing) return existing.alertId;
        const alertId = nid();
        curMonitorAlertId.current.set(key, { monitorId, alertId });
        // Seed the alert into monitorAlerts as a streaming placeholder. Text
        // fills in via message.delta; message.complete flips streaming → false.
        setMonitorAlerts((prev) => {
          const next = new Map(prev);
          const list = next.get(monitorId) ? next.get(monitorId)!.slice() : [];
          list.push({ id: alertId, text: "", ts: Date.now(), streaming: true });
          next.set(monitorId, list);
          return next;
        });
        return alertId;
      }
      const existing = curAssistantId.current.get(key);
      if (existing) return existing;
      const id = nid();
      curAssistantId.current.set(key, id);
      setMessages((prev) => capMsgs([...prev, {
        id, role: "assistant", text: "", streaming: true,
        awaitingFirstDelta: true,
        hasReasoning: false,
        brief: p.brief,
        requestId: p.request_id,
      }]));
      refs.current.isAnswering = true;
      return id;
    };

    // A user turn owns its answer slot before the backend starts. When the
    // main agent transfers reply ownership, QueryWorker reuses that same slot;
    // tag it in place so out-of-order completion remains visibly attributable
    // without moving the bubble away from its originating question.
    const markQueryWorker = (p: { source?: string; request_id?: string }) => {
      if (p.source !== "query_worker") return;
      const id = curAssistantId.current.get(keyOf(p));
      if (!id) return;
      setMessages((prev) => prev.map((m) => (
        m.id === id && m.subRole !== "query_worker"
          ? { ...m, subRole: "query_worker" }
          : m
      )));
    };

    const offStart = gw.on<{ source?: string; request_id?: string; monitor_id?: string; monitor_label?: string; brief?: string }>(
      "message.start", (ev) => {
        if (!isMine(ev)) return;
        const p = ev.payload || {};
        traceOnce(`message_start (source=${p.source || "-"})`);
        markQueryWorker(p);
        const key = keyOf(p);
        if (!p.monitor_id && p.source !== "monitor") activeForegroundKey = key;
        // User-originated turns already own a preallocated answer slot directly
        // below their query. Mark that slot active without moving it. Backend-
        // originated turns still create lazily on delta/complete so user_echo
        // remains before their assistant bubble.
        const id = curAssistantId.current.get(key);
        if (id) {
          setMessages((prev) => prev.map((m) => (
            m.id === id && m.queued
              ? { ...m, queued: false, queuePosition: undefined }
              : m
          )));
        }
      });
    // ── Streaming batcher ────────────────────────────────────────────────
    // The old code called setMessages + prev.map() PER TOKEN for message.delta,
    // reasoning.delta and thinking.delta. On qwen3.x with thinking ON, one turn
    // easily emits 200+ tokens; after 3-4 turns the chat list is ~30 items and
    // Markdown+syntax-highlighter re-parse every assistant bubble each frame.
    // React couldn't keep up → main thread wedged (hover unresponsive).
    //
    // Fix: buffer text/reasoning deltas into refs keyed by bubble id, and flush
    // them in a single setMessages() on a throttled timer (~10 fps). rAF-only
    // batching still hit 60 updates/s — too many when combined with screen
    // capture JPEG encode + JSON.stringify on the same main thread.
    // ★ 80ms ≈ 12.5fps 的统一 flush (主 agent message.delta + 深度分析 bg 共用)。
    //   ★ 丝滑修复: 下游渲染【不再叠加第二层节流】—— 主气泡 Markdown 直接吃 body、
    //   面板 LiveMarkdown 直接吃 content。原来在 flush 之上又套 120/150ms 的
    //   useThrottledValue, 与 flush 相位错开产生"一段段"拍频, 已移除。现在唯一的节流
    //   就是这个 flush, 帧率稳定无拍频。
    const STREAM_FLUSH_MS = 80;
    const streamBuf = { text: new Map<string, string>(), reasoning: new Map<string, string>() };
    // ★ multimodal.bg 事件队列 (answer_delta 流式高频)。
    let bgQueue: any[] = [];
    // Deferred deepExpanded update: raw bg handler records the last rid, flush
    // applies it once (one setState per flush cycle instead of per-event).
    let bgPendingExpandRid: string | null = null;
    // ★ 消息队列: tool.start / tool.complete / status.update / watcher.report_append
    //   等非流式但高频的事件。之前直接调 setMessages() → 深度研究时 watcher ReAct 循环
    //   密集 tool 调用 (每轮 2-4 个, 10 轮 = 40-80 次) 绕过 80ms 节流系统, 每次都触发
    //   全量重渲染与已节流的 flush 竞争主线程 → livelock → 界面卡死。现在统一入队, 由
    //   runUnifiedFlush 一帧 drain 一次, 与 stream/bg 合并为一次 React 批量更新。
    type MsgQueueEntry =
      | { action: "append"; msg: ChatMsg }
      | { action: "patch_tool"; toolId: string; toolName?: string; patch: Partial<ChatMsg> }
      | { action: "patch_query_worker"; taskId: string; step: QueryWorkerProgressStep }
      | { action: "collapse_status"; text: string };
    let msgQueue: MsgQueueEntry[] = [];
    // Progress can race ahead of tool.complete because QueryWorker is scheduled
    // before the tool handler returns. Buffer by qry_* so the complete event can
    // attach every already-seen step to the correct tool card in one patch.
    // ★ 统一节流器: message.delta (主 agent) 与 multimodal.bg (深度分析) 共用 ONE
    //   timer + ONE rAF。两个 Agent 并发输出时, 同一帧内把两条流一次性 drain
    //   (runStreamFlush + runBgFlush 背靠背), React 自动把两次 setState 批处理成
    //   一次重渲染 —— 避免"长 markdown 报告重渲 + deep-panel 重渲"在同一主线程各自
    //   触发把线程撑爆 (页面无响应)。
    let flushTimer: number | null = null;
    let flushRaf: number | null = null;   // ★ C26: 卸载时取消, 防卸载后 setState。
    let flushDisposed = false;
    const runStreamFlush = () => {
      if (streamBuf.text.size === 0 && streamBuf.reasoning.size === 0) return;
      _mmAct("streamFlush", `keys=${streamBuf.text.size}`);
      const textDrain = streamBuf.text; streamBuf.text = new Map();
      const reasonDrain = streamBuf.reasoning; streamBuf.reasoning = new Map();
      setMessages((prev) => {
        // Single pass over the list; only patch objects that actually have
        // pending deltas (avoid new-referencing the entire list per turn).
        let dirty = false;
        const next = prev.map((m) => {
          const td = textDrain.get(m.id);
          const rd = reasonDrain.get(m.id);
          if (td === undefined && rd === undefined) return m;
          dirty = true;
          const patch: Partial<ChatMsg> = {};
          if (td !== undefined) {
            patch.text = m.text + td;
            // 一旦正文开始, 第一行整体让位 —— 清 awaiting/hasReasoning。
            //   m.reasoning 后台保留, 供下一轮 API 回传。
            patch.awaitingFirstDelta = false;
            patch.hasReasoning = false;
            patch.reasoningSummary = undefined;
          }
          if (rd !== undefined) {
            patch.reasoning = (m.reasoning || "") + rd;
            patch.awaitingFirstDelta = false;
            patch.hasReasoning = true;
          }
          if (td !== undefined && m.queued) patch.queued = false;
          return { ...m, ...patch };
        });
        try { (window as any).__mmN = { ...(window as any).__mmN, msgs: next.length }; } catch { /* noop */ }
        return dirty ? next : prev;
      });
    };
    // ── ★ 主线程卡顿看门狗 + 面包屑 (诊断"整界面卡死", F12 一卡就死拿不到 profile) ──
    //   _mmAct("动作名"): 记录最近一次热路径动作 + 时间戳到 window.__mmAct。
    //   看门狗每 250ms tick 一次; 若两次 tick 间隔 >> 250ms, 说明主线程刚被占死过,
    //   console.warn 报出"卡了多久 + 卡死前最后动作 + 当前数据规模"。卡死缓过来后
    //   Console 里最后一条 [mm-watchdog] 就是现场。生产可留 (开销极小)。
    const _mmAct = (name: string, extra?: string) => {
      (window as any).__mmAct = { name, extra: extra || "", t: performance.now() };
    };
    let _wdLast = performance.now();
    const _watchdog = window.setInterval(() => {
      const now = performance.now();
      const gap = now - _wdLast;
      _wdLast = now;
      // 期望间隔 250ms; 超过 600ms 视为主线程被占死过一段。
      if (gap > 600) {
        const a = (window as any).__mmAct || {};
        const sinceAct = a.t ? Math.round(now - a.t) : -1;
        // eslint-disable-next-line no-console
        console.warn(
          `%c[mm-watchdog] 主线程卡顿 ${Math.round(gap)}ms | 卡死前最后动作=${a.name || "?"}` +
          `${a.extra ? "(" + a.extra + ")" : ""} 距今${sinceAct}ms`,
          "color:#dc3545;font-weight:bold",
        );
      }
    }, 250);

    // 批量 flush: 把 msgQueue 里累积的 tool/status/report 事件一次性 reduce 进
    // messages, 只触发 1 次 setMessages。由统一节流器 runUnifiedFlush 调用。
    const runMsgQueueFlush = () => {
      if (msgQueue.length === 0) return;
      const drain = msgQueue; msgQueue = [];
      _mmAct("msgQueueFlush", `queued=${drain.length}`);
      setMessages((prev) => {
        let list = prev;
        for (const entry of drain) {
          if (entry.action === "append") {
            list = capMsgs([...list, entry.msg]);
          } else if (entry.action === "patch_tool") {
            let idx = entry.toolId
              ? list.findIndex((m) => m.kind === "tool" && m.toolId === entry.toolId && !m.toolDone)
              : -1;
            if (idx < 0 && entry.toolName) {
              for (let i = list.length - 1; i >= 0; i--) {
                const m = list[i];
                if (m.kind === "tool" && m.toolName === entry.toolName && !m.toolDone) { idx = i; break; }
              }
            }
            if (idx >= 0) {
              const next = list.slice();
              next[idx] = { ...next[idx], ...entry.patch };
              list = next;
            } else {
              list = capMsgs([...list, {
                id: nid(), role: "assistant", text: "", kind: "tool",
                toolId: entry.toolId, toolName: entry.toolName || "tool",
                ...entry.patch,
              } as ChatMsg]);
            }
          } else if (entry.action === "patch_query_worker") {
            const idx = list.findIndex((m) =>
              m.kind === "tool" && m.workerTaskId === entry.taskId);
            if (idx >= 0) {
              const current = list[idx];
              const existing = current.workerProgress || [];
              if (!existing.some((step) => step.id === entry.step.id)) {
                const nextProgress = queryProgressByTaskRef.current.get(entry.taskId)
                  || mergeQueryWorkerProgress(existing, entry.step);
                const next = list.slice();
                next[idx] = {
                  ...current,
                  workerProgress: nextProgress,
                  workerStatus: entry.step.terminal
                    ? entry.step.status
                    : current.workerStatus || "running",
                };
                list = next;
              }
            }
          } else if (entry.action === "collapse_status") {
            const last = list[list.length - 1];
            if (last && last.kind === "status") {
              const next = list.slice();
              next[next.length - 1] = { ...last, text: entry.text };
              list = next;
            } else {
              list = capMsgs([...list, { id: nid(), role: "assistant", text: entry.text, kind: "status" }]);
            }
          }
        }
        return compactQueryWorkerMessageProgress(
          list, queryProgressByTaskRef.current,
        );
      });
    };

    // 统一 flush: 同一帧内同时 drain 主 agent 流 + 深度分析 bg 流 + 消息队列。
    const runUnifiedFlush = () => {
      const _t0 = performance.now();
      _mmAct("unifiedFlush");
      runStreamFlush();
      runBgFlush();
      runMsgQueueFlush();
      const _dt = performance.now() - _t0;
      // 单次 flush >50ms = 一帧预算 (16ms) 的 3 倍以上, 该帧必掉。报出规模帮定位。
      if (_dt > 50) {
        // eslint-disable-next-line no-console
        console.warn(
          `%c[mm-watchdog] flush 耗时 ${Math.round(_dt)}ms | msgs=${(window as any).__mmN?.msgs ?? "?"}` +
          ` bg=${(window as any).__mmN?.bg ?? "?"} seg=${(window as any).__mmN?.seg ?? "?"}`,
          "color:#dc3545",
        );
      }
    };
    const scheduleFlush = () => {
      if (flushTimer !== null || flushDisposed) return;
      flushTimer = window.setTimeout(() => {
        flushTimer = null;
        if (flushDisposed) return;
        flushRaf = requestAnimationFrame(() => {
          flushRaf = null;
          if (flushDisposed) return;
          runUnifiedFlush();
        });
      }, STREAM_FLUSH_MS);
    };
    const appendText = (id: string, delta: string) => {
      if (!id || !delta) return;
      streamBuf.text.set(id, (streamBuf.text.get(id) || "") + delta);
      scheduleFlush();
    };
    const offDelta = gw.on<{ text?: string; source?: string; request_id?: string; monitor_id?: string; monitor_label?: string }>(
      "message.delta", (ev) => {
        if (!isMine(ev)) return;
        const p = ev.payload || {};
        traceOnce("first_message_delta", "#28a745");
        if (!p.text) return;
        const isMonitor = p.source === "monitor" || !!p.monitor_id;
        if (isMonitor) {
          // Route monitor deltas to the right-panel alert list (NOT chat).
          // Lazily create the alert if the delta beat message.start.
          ensureBubble(p);
          const key = keyOf(p);
          const rec = curMonitorAlertId.current.get(key);
          if (!rec) return;
          setMonitorAlerts((prev) => {
            const next = new Map(prev);
            const list = next.get(rec.monitorId);
            if (!list) return prev;
            const at = list.findIndex((a) => a.id === rec.alertId);
            if (at < 0) return prev;
            const cp = list.slice();
            cp[at] = { ...cp[at], text: cp[at].text + p.text! };
            next.set(rec.monitorId, cp);
            return next;
          });
          return;
        }
        // Lazily create the bubble if the delta beat its message.start (out of
        // order delivery) — otherwise the token would be silently dropped.
        const id = ensureBubble(p);
        markQueryWorker(p);
        appendText(id, p.text);
      });
    const offComplete = gw.on<{
      text?: string; source?: string; request_id?: string;
      monitor_id?: string; monitor_label?: string; status?: string; brief?: string;
      history_policy?: unknown; ephemeral_control?: unknown; ephemeral?: unknown;
      evidence?: unknown;
    }>(
      "message.complete", (ev) => {
        if (!isMine(ev)) return;
        const p = ev.payload || {};
        const isMonitor = p.source === "monitor" || !!p.monitor_id;
        if (isMonitor) {
          // Finalize the monitor alert on the right-panel list. If the whole
          // stream arrived as one complete (no prior start/delta), synthesize
          // the alert now so its text isn't lost. Then flip streaming → false.
          ensureBubble(p);
          const key = keyOf(p);
          const rec = curMonitorAlertId.current.get(key);
          curMonitorAlertId.current.delete(key);
          if (!rec) return;
          const finalText = (p.text || "").toString();
          setMonitorAlerts((prev) => {
            const next = new Map(prev);
            const list = next.get(rec.monitorId);
            if (!list) return prev;
            const at = list.findIndex((a) => a.id === rec.alertId);
            if (at < 0) return prev;
            const cp = list.slice();
            const cur = cp[at];
            // If deltas already accumulated, keep that; else use the payload's
            // full text (single-complete case).
            const text = cur.text.trim() ? cur.text : finalText;
            cp[at] = {
              ...cur,
              text,
              streaming: false,
              evidence: normalizeMonitorEvidence(p.evidence),
            };
            next.set(rec.monitorId, cp);
            return next;
          });
          return;
        }
        // (Watcher no longer streams into the center chat via message.* — the dead
        // source=watcher/watcher_threadback branches were removed. Watcher
        // process/report live in the right panel + arrive as folded bubbles via
        // watcher.report_append.)
        const key = keyOf(p);
        const ephemeralControl = isEphemeralControl(p);
        // If the whole stream arrived as a single complete (no start/delta seen),
        // synthesize the bubble so its text isn't lost, then finalize it.
        const id = curAssistantId.current.get(key) ?? ensureBubble(p);
        markQueryWorker(p);
        curAssistantId.current.delete(key);
        if (activeForegroundKey === key) activeForegroundKey = "__main__";
        refs.current.isAnswering = curAssistantId.current.size > 0;
        traceOnce("message_complete", "#28a745");
        if (!id) return;
        // Cancel any pending throttled flush — we drain inline below.
        if (flushTimer !== null) {
          clearTimeout(flushTimer);
          flushTimer = null;
        }
        // Final flush + clear streaming flag in ONE setMessages call. Running
        // flush inline ensures the completed bubble's last tokens land before
        // we mark it not-streaming (otherwise the tail could be lost if a rAF
        // was still pending).
        const textDrain = streamBuf.text; streamBuf.text = new Map();
        const reasonDrain = streamBuf.reasoning; streamBuf.reasoning = new Map();
        // tool.complete precedes message.complete on the wire, but its throttled
        // card update may still be queued. Flush it first so the final ephemeral
        // turn cleanup runs after every tool card has been correlated/rendered.
        if (ephemeralControl) runMsgQueueFlush();
        setMessages((prev) => {
          const completed = prev.map((m) => {
            if (m.id !== id) {
              // Still apply any pending drains to non-target bubbles.
              const td = textDrain.get(m.id); const rd = reasonDrain.get(m.id);
              if (td === undefined && rd === undefined) return m;
              return { ...m, ...(td !== undefined ? { text: m.text + td } : {}),
                       ...(rd !== undefined ? { reasoning: (m.reasoning || "") + rd } : {}),
                       ...(td !== undefined && m.queued
                         ? { queued: false, queuePosition: undefined } : {}) };
            }
            const td = textDrain.get(id) || ""; const rd = reasonDrain.get(id) || "";
            const streamed = m.text + td;
            // Thread-back completes carry the full body in payload.text with no
            // prior deltas — don't leave the bubble empty.
            const text = streamed.trim() ? streamed : (p.text || streamed);
            return { ...m, text,
                     reasoning: rd ? (m.reasoning || "") + rd : m.reasoning,
                     streaming: false, queued: false, queuePosition: undefined };
          });
          // Pure Monitor control turns live in the right-side registry and
          // sidechannel, not in canonical chat. The final turn-level marker is
          // authoritative: drop every center item carrying this request id,
          // including all tool cards in the batch. Legacy uncorrelated events
          // still remove only the known assistant bubble.
          return ephemeralControl
            ? removeEphemeralControlTurn(completed, key === "__main__" ? "" : key, id)
            : completed;
        });
        // ★ 统一节流器: message.complete 取消了共享 flushTimer, 若此刻有排队的 bg/msg
        //   事件 (深度分析并发) 会被搁置 → 这里补 drain 一次, 不丢进度。
        runBgFlush();
        runMsgQueueFlush();
      });
    // Observation panels (画面观察/音频观察/搜索事实) pushed by the memory backend.
    let ctxPending: CtxState | null = null;
    let ctxFlushScheduled = false;
    let ctxRaf: number | null = null;
    const offCtx = gw.on<{
      obs?: ObsItem[]; audio_obs?: ObsItem[];
      facts?: Record<string, string>; version?: number;
    }>("multimodal.ctx", (ev) => {
      if (!isMine(ev)) return;
      const c = ev.payload || {};
      // Cap the observation arrays — backend pushes the full log each time and
      // over a long session these grow unbounded, making each obs-panel render
      // slower. Keep the most recent 200 (matches the visible scrollable view).
      const rawObs = c.obs || [];
      const rawAObs = c.audio_obs || [];
      ctxPending = {
        version: c.version || 0,
        obs: rawObs.length > 200 ? rawObs.slice(rawObs.length - 200) : rawObs,
        audioObs: rawAObs.length > 200 ? rawAObs.slice(rawAObs.length - 200) : rawAObs,
        facts: c.facts || {},
      };
      if (ctxFlushScheduled) return;
      ctxFlushScheduled = true;
      ctxRaf = requestAnimationFrame(() => {
        ctxFlushScheduled = false;
        ctxRaf = null;
        // Guard against a rAF firing after unmount (setState on dead component).
        if (flushDisposed) return;
        if (ctxPending) setCtx(ctxPending);
      });
    });

    // ── Tool / status / reasoning progress (so long requests show feedback) ──
    // ★ tool.start / tool.complete 走 msgQueue + 统一节流 flush, 不再直接
    //   setMessages。深度研究 watcher ReAct 循环密集 tool 调用绕过节流系统是
    //   "界面卡死"的主因。
    const offToolStart = gw.on<{
      tool_id?: string; name?: string; context?: any; args_text?: string;
      args_fields?: ToolArgField[]; request_id?: string;
    }>(
      "tool.start", (ev) => {
        if (!isMine(ev)) return;
        const p = ev.payload || {};
        const turnRequestId = p.request_id
          || (activeForegroundKey !== "__main__" ? activeForegroundKey : undefined);
        // ★ 每个分支都要能落到下一个兜底。之前写成
        //   `typeof p.context === "string" ? p.context : (…) || p.args_text`,
        //   于是 context 为空字符串时三元的第一支直接返回 "" —— 后面的
        //   args_text 兜底永远不生效 (verbose 模式对这行完全无效)。
        const ctxStr =
          (typeof p.context === "string" ? p.context : "")
          || (p.context && typeof p.context === "object"
            ? (p.context.summary || p.context.text || "") : "")
          || p.args_text
          || "";
        msgQueue.push({ action: "append", msg: {
          id: nid(), role: "assistant", text: "", kind: "tool",
          toolId: p.tool_id, toolName: p.name || "tool", toolCtx: String(ctxStr || ""),
          ...(p.args_fields?.length ? { toolArgs: p.args_fields } : {}),
          toolDone: false, requestId: turnRequestId,
        }});
        scheduleFlush();
      });
    const offToolComplete = gw.on<{
      tool_id?: string; name?: string; summary?: string; duration_s?: number;
      result_text?: string; inline_diff?: string; render_hint?: string;
      dispatch_label?: string; dispatch_note?: string; request_id?: string; task_id?: string;
      result?: unknown;
      recall_debug?: unknown;
    }>("tool.complete", (ev) => {
      if (!isMine(ev)) return;
      const p = ev.payload || {};
      const turnRequestId = p.request_id
        || (activeForegroundKey !== "__main__" ? activeForegroundKey : undefined);
      const detail = p.dispatch_note || p.inline_diff || p.result_text || "";
      const durMs = p.duration_s != null ? Math.round(p.duration_s * 1000) : undefined;
      const summary = p.dispatch_label || p.summary || "";
      const recallDebug = isQueryMultimodalToolName(p.name)
        ? (extractRecallDebug(p.recall_debug, detail) || extractRecallDebug(p.result, detail))
        : null;
      const workerTaskId = isQueryMultimodalToolName(p.name)
        ? String(p.task_id || "") : "";
      const bufferedWorkerProgress = workerTaskId
        ? (queryProgressByTaskRef.current.get(workerTaskId) || [])
          .slice(-QUERY_WORKER_PROGRESS_LIMIT) : [];
      const terminalWorkerStep = [...bufferedWorkerProgress]
        .reverse().find((step) => step.terminal);
      // A failed call used to render EXACTLY like a successful one (green ✓,
      // same styling) — the only trace was the error text buried inside the
      // collapsed detail. `tools/registry.py` returns failures as
      // `{"error": "..."}`, so read that and let the row show it.
      const resultObj = p.result && typeof p.result === "object" && !Array.isArray(p.result)
        ? (p.result as Record<string, unknown>) : null;
      const toolError = typeof resultObj?.error === "string" ? resultObj.error.trim() : "";
      msgQueue.push({ action: "patch_tool", toolId: p.tool_id || "", toolName: p.name, patch: {
        toolDone: true, toolSummary: summary, toolDurationMs: durMs, toolDetail: detail,
        ...(toolError ? { toolIsError: true, toolError } : {}),
        ...(recallDebug?.trace?.length ? { recallTrace: recallDebug.trace } : {}),
        ...(recallDebug?.findings ? { recallFindings: recallDebug.findings } : {}),
        ...(turnRequestId ? { requestId: turnRequestId } : {}),
        ...(workerTaskId ? {
          workerTaskId,
          workerStatus: (terminalWorkerStep?.status || "running") as ChatMsg["workerStatus"],
          workerProgress: bufferedWorkerProgress,
        } : {}),
      }});
      scheduleFlush();
    });
    // Live-watcher is FULLY decoupled from the main agent chat: per-round process
    // arrives via multimodal.bg (DeepPanel segment cards) and the final
    // consolidated report via watcher.final. NOTHING is appended to the center
    // chat — no turn-2 mutation, no lock contention with user/agent turns.
    const offWatcherFinal = gw.on<{ request_id?: string; brief?: string; text?: string }>(
      "watcher.final", (ev) => {
        if (!isMine(ev)) return;
        const p = ev.payload || {};
        const rid = p.request_id || "";
        const text = (p.text || "").trim();
        if (!rid || !text) return;
        setBgItems((prev) => {
          const idx = prev.findIndex((b) => b.requestId === rid);
          if (idx >= 0) {
            const next = prev.slice();
            // ★ waiting: null —— 最终报告到达即代表整轮结束, 必须同时撤掉攒帧条。
            //   watcher.final 是内联送达、multimodal.bg 走 80ms 节流队列, 所以本事件
            //   总是先于 delegation_done 到; 若这里只置 done, 后到的 delegation_done
            //   会被它自己的 !b.done 判断跳过 → waiting 永远留在上一个 Seg 上 (显示
            //   "已完成" 却仍挂着 "Seg N · 攒帧 5/40 · 26s left")。与桌面端
            //   setWatcherFinal 对齐。
            next[idx] = { ...next[idx], finalReport: text, done: true, waiting: null };
            return next;
          }
          // No bg item yet (run produced no visible events) → create one so the
          // final report is still shown in the panel.
          return [...prev, { id: rid, requestId: rid, segments: [], finalReport: text, done: true }].slice(-8);
        });
      });
    // Per-round report: the live DeepWindow already streams each segment's
    // answer via multimodal.bg answer_delta events, and the backend persists
    // each round to mm_watcher_reports for reopen restore. So this event is a
    // no-op on the frontend now — no center-chat bubble, no double-append.
    const offReportAppend = gw.on<{ request_id?: string; round?: number; text?: string }>(
      "watcher.report_append", () => { /* no-op: live=bg, reopen=list_watcher_content */ });
    // Terminal marker: the delegation finished. The final report is delivered via
    // watcher.final; this is kept only for potential future UI cues.
    const offDeepComplete = gw.on<{ request_id?: string; brief?: string }>(
      "watcher.complete", () => { /* no-op: final report delivered via watcher.final */ });
    const offStatus = gw.on<{ kind?: string; text?: string }>("status.update", (ev) => {
      if (!isMine(ev)) return;
      const text = (ev.payload?.text || "").trim();
      if (!text) return;
      // kind=process carries a background-process notification (async delegation
      // batch complete, bash process done, ...). The gateway also submits that
      // same text as a prompt, so it arrives again via message.user_echo and is
      // persisted to history as a user message — rendering it here too showed it
      // twice (once in the 处理过程 card, once as a "You" bubble). The bubble is
      // the durable one (it replays from history after a refresh, this card does
      // not), so drop the card and keep the bubble. The event itself must keep
      // flowing: the desktop app uses it to refresh its background-process list.
      if ((ev.payload?.kind || "") === "process") return;
      msgQueue.push({ action: "collapse_status", text });
      scheduleFlush();
    });
    const appendReasoning = (delta: string) => {
      // reasoning.delta has no id → attach to the main agent's open bubble
      // (the only bubble that produces reasoning in the mainline path).
      // Routes through the same rAF-batched streamBuf as message.delta so
      // reasoning tokens don't trigger a re-render each.
      const id = curAssistantId.current.get(activeForegroundKey)
        || curAssistantId.current.get("__main__");
      if (!id || !delta) return;
      streamBuf.reasoning.set(id, (streamBuf.reasoning.get(id) || "") + delta);
      scheduleFlush();
    };
    const offReasoning = gw.on<{ text?: string }>("reasoning.delta",
      (ev) => { if (!isMine(ev)) return; traceOnce("first_reasoning_delta"); appendReasoning(ev.payload?.text || ""); });
    const offThinking = gw.on<{ text?: string }>("thinking.delta",
      (ev) => { if (!isMine(ev)) return; traceOnce("first_thinking_delta"); appendReasoning(ev.payload?.text || ""); });
    // Auxiliary-LLM 生成的 ~10 字段级 label. 后端每攒够一段 reasoning 就
    // fire 一次; 前端把最新一条塞进 m.reasoningSummary, 供 AssistantMessage
    // 第一行滚动展示。失败/超时时后端不推 —— FE 自动 fallback 到"reasoning
    // 原文最后一行"呈现 (在渲染层判断)。
    const offReasoningSummary = gw.on<{ text?: string }>("reasoning.summary",
      (ev) => {
        if (!isMine(ev)) return;
        const t = (ev.payload?.text || "").trim();
        if (!t) return;
        const id = curAssistantId.current.get(activeForegroundKey)
          || curAssistantId.current.get("__main__");
        if (!id) return;
        setMessages((prev) => prev.map((m) => (
          m.id === id ? { ...m, reasoningSummary: t, hasReasoning: true, awaitingFirstDelta: false } : m
        )));
      });
    const offError = gw.on<{ message?: string; request_id?: string }>("error", (ev) => {
      if (!isMine(ev)) return;
      const p = ev.payload || {};
      const msg = p.message || translateNow("multimodal.errors.unknown");
      if (p.request_id) {
        const id = curAssistantId.current.get(p.request_id);
        curAssistantId.current.delete(p.request_id);
        if (activeForegroundKey === p.request_id) activeForegroundKey = "__main__";
        if (id) {
          setMessages((prev) => prev.map((m) => (
            m.id === id
              ? { ...m, text: `${translateNow("multimodal.misc.errorPrefix")}: ${msg}`, streaming: false,
                  queued: false, queuePosition: undefined, isError: true }
              : m
          )));
          refs.current.isAnswering = curAssistantId.current.size > 0;
          return;
        }
      }
      // An error aborts whatever streams were in flight — no message.complete
      // will arrive to clear their curAssistantId keys. Clear the key map and
      // close any still-streaming bubbles so stale streaming state doesn't
      // linger. (isAnswering is deprecated no-op bookkeeping now.)
      curAssistantId.current.clear();
      refs.current.isAnswering = false;
      setMessages((prev) => capMsgs([
        ...prev.map((m) => (m.streaming
          ? { ...m, streaming: false, queued: false, queuePosition: undefined }
          : m)),
        { id: nid(), role: "system", text: `${translateNow("multimodal.misc.errorPrefix")}: ${msg}`, isError: true },
      ]));
    });
    const offSessionInfo = gw.on<{ model?: string }>("session.info", (ev) => {
      const m = ev.payload?.model;
      if (m) setModel(m);
    });
    // 监控/深度研究过程失败/停用 → 右侧面板底部 toast (10s 淡出, 多条自然堆叠 —— 见 setMmToasts
    //   的 prev+push, 不做互相踢)。不进 history、不发主气泡。
    const offToast = gw.on<{ level?: string; text?: string }>("multimodal.toast", (ev) => {
      if (!isMine(ev)) return;
      const text = (ev.payload?.text || "").trim();
      if (!text) return;
      const id = nid();
      const level = ev.payload?.level || "error";
      setMmToasts((prev) => [...prev, { id, level, text }]);
      setTimeout(() => setMmToasts((prev) => prev.filter((x) => x.id !== id)), 10000);
    });
    // Multimodal RouterEngine background progress (search / recall / crop).
    const offBg = gw.on<{
      type?: string; channel?: string; task_id?: string; brief?: string;
      label?: string; phase?: string; frame_ts?: number; target?: string;
      crops?: CropItem[]; anchor_ts?: number; anchor_jpeg_b64?: string;
      observations?: { name?: string; obs_summary?: string }[];
      findings_len?: number; n_frames?: number; rounds?: number;
      elapsed_sec?: number; findings?: string; thought?: string;
      can_answer?: boolean; text_len?: number; text_preview?: string;
      have?: number; need?: number;
      report?: string; batches?: number; obs_summary?: string; n_clues?: number;
      frame_ts_range?: [number, number]; seg?: number; delegation_done?: boolean;
    }>("multimodal.bg", (ev) => {
      if (!isMine(ev)) return;
      const p: any = ev.payload || {};
      const rid = p.request_id || "";
      // ★ 性能: setDeepExpanded 移到 runBgFlush 末尾统一处理, 不再在每个 bg 事件
      //   上直接 setState → 减少一个与 flush 竞争主线程的重渲染源。
      if (rid) bgPendingExpandRid = rid;
      // ★ 性能 (livelock 根治): deep research 现在是真实 LLM token 流式 (每 chunk 一条
      //   answer_delta), 一段几千字仍是成百上千条事件。策略与主 Agent 一致 —— 后端不合并,
      //   靠前端吸收: (1) 这里【入队时合并】—— 若队尾已是同一 (rid, seg) 的 answer_delta,
      //   直接把本次 delta 追加到队尾那条上, 不新增队列项 → 无论后端发多快, answer_delta
      //   在队列里对每个段最多占 1 条, 队列长度不随 token 数增长 (根治雪崩);
      //   (2) 下游走与 message.delta 共用的 100ms 节流 flush (scheduleFlush)。双重吸收。
      if (p.type === "answer_delta" && typeof p.delta === "string") {
        const tail = bgQueue[bgQueue.length - 1];
        if (tail && tail.type === "answer_delta"
            && (tail.request_id || "") === rid
            && tail.seg === p.seg
            && (tail.channel || "bg") === (p.channel || "bg")) {
          tail.delta = String(tail.delta || "") + p.delta;
          scheduleFlush();
          return;
        }
      }
      bgQueue.push(p);
      scheduleFlush();
    });
    // reduceBg: 把一个 bg payload 折进 bgItems 列表 (纯函数, 逻辑与原 setBgItems 内联体
    //   一字不差, 只是抽出来供批量 flush 复用)。
    const reduceBg = (prevList: BgItem[], p: any): BgItem[] => {
      const ch = p.channel || "bg";
      const rid = p.request_id || "";
      if (p.delegation_done && rid) {
        // ★ 幂等: 不再用 !b.done 作为前置条件。watcher.final 可能已先把 done 置真,
        //   那时旧写法会整条跳过 → 连 waiting 也清不掉。收尾清理必须无条件执行。
        return prevList.map((b) => (b.requestId === rid && (!b.done || b.waiting)
          ? { ...b, done: true, waiting: null } : b));
      }
      const itemId = rid || `_:${ch}`;
      const prev = prevList;
      {
        const idx = prev.findIndex((b) => b.id === itemId);
        // ★ 性能: 只浅拷贝 BgItem + segments 数组本身, 不深拷贝每个 segment。旧代码
        //   segments.map(s => ({...s})) 每个事件都给【所有】段换新对象 identity →
        //   SegmentCard 的 memo 全部失效 → 每 100ms flush 整列段重渲染。改成: 数组浅拷,
        //   只有被本次修改的那个段 (segFor 里) 才 clone-on-write, 其余段保持原引用 →
        //   memo 生效, 只有变化的那张卡重渲染。
        const cur: BgItem = idx >= 0
          ? { ...prev[idx], segments: prev[idx].segments.slice() }
          : { id: itemId, requestId: rid || undefined, segments: [] };
        if (p.label) cur.label = p.label;

        const t = p.type || "";
        // ★ L: 若本 item 已 done (watcher.final 已到), 忽略迟到的攒帧/新段类事件 ——
        //   它们会重设 cur.waiting / 加空段, 导致"完成报告"下方又冒出"攒帧中…"。
        //   (bg 队列 80ms flush, watcher.final 内联 → final 可能先于最后一批 bg 到。)
        if (idx >= 0 && prev[idx].done &&
            (t === "waiting" || t === "batch_ready" || t === "segment_start")) {
          return prev;
        }
        const clip = (s: any, n: number) => String(s || "").replace(/\s+/g, " ").trim().slice(0, n);
        // clone-on-write: 把 cur.segments[i] 换成一个新对象 (仅这一个段变 identity),
        // 返回它供修改。其余段引用不变 → 它们的 SegmentCard memo 命中、不重渲染。
        const cowAt = (i: number): BgSegment => {
          const copy = { ...cur.segments[i], lookups: cur.segments[i].lookups.slice() };
          cur.segments[i] = copy;
          return copy;
        };
        const curSeg = (): BgSegment => {
          if (cur.segments.length === 0) cur.segments.push({ seg: 1, lookups: [] });
          return cowAt(cur.segments.length - 1);
        };
        // The backend stamps `seg` (1-based round index) on every round event.
        // Route to the matching segment (create if new) so out-of-order deltas
        // still land in the right card.
        const segNo = typeof p.seg === "number" && p.seg > 0 ? p.seg : undefined;
        const segFor = (): BgSegment => {
          if (segNo === undefined) return curSeg();
          const at = cur.segments.findIndex((x) => x.seg === segNo);
          if (at < 0) {
            const s = { seg: segNo, lookups: [] } as BgSegment;
            cur.segments.push(s); cur.segments.sort((a, b) => a.seg - b.seg);
            return cur.segments.find((x) => x.seg === segNo)!;
          }
          return cowAt(at);
        };

        if (t === "waiting") {
          // Frame-accumulation: current/target frames + ttl countdown (so the
          // panel shows a live "攒帧 N/target · ttl 余 Ns" instead of freezing).
          cur.waiting = {
            have: p.have ?? 0, need: p.need ?? 0,
            ttlSec: typeof p.ttl_sec === "number" ? p.ttl_sec : undefined,
            ttlRemaining: typeof p.ttl_remaining === "number" ? p.ttl_remaining : undefined,
            seg: typeof p.seg === "number" ? p.seg : undefined,
            paused: !!p.paused,
          };
        } else if (t === "answer_delta") {
          // Live streaming of THIS segment's interpretation, token by token.
          const s = segFor();
          if (p.delta) s.answer = (s.answer || "") + String(p.delta);
        } else if (t === "batch_ready") {
          // 攒够/开始分析: ★ 不再把 waiting 置 null (那会让攒帧条整块卸载→下次心跳再挂载,
          //   产生"消失又重现"闪烁)。改成原位标记为满额 (have=need), 保持组件挂载;
          //   分析期间的心跳 waiting 会继续原位更新数字, 直到 done 才真正清除。
          if (cur.waiting) cur.waiting = { ...cur.waiting, have: cur.waiting.need };
        } else if (t === "segment_start") {
          // New analysis round → a fresh segment card with its frame time range.
          // 后端在首次拿到 thought 后会补发一条带 scene_label 的 segment_start, 刷新场景标记。
          // ★ 不清 waiting: 保持攒帧条原位, 由后续 waiting 事件原位更新 (见 batch_ready)。
          const s = segFor();
          if (p.frame_ts_range && p.frame_ts_range.length === 2) s.tsRange = p.frame_ts_range;
          if (p.scene_label) s.scene = String(p.scene_label);
        } else if (t === "router_react") {
          // The model's per-round reasoning → 👁 看到 (most human-readable) +
          // 💭 thinking trace + 🔧 tool calls (req ②: show thinking & tool calls).
          const s = segFor();
          // 固定文本描述 (标题行下方): 用户要求不限制字数, 只去首尾空白/换行折叠展示。
          if (p.thought) s.saw = String(p.thought).replace(/\s+/g, " ").trim();
          const tc = (p.tool_calls || []) as { name?: string; args?: Record<string, unknown> }[];
          if (tc.length) {
            s.toolCalls = tc.map((c) => ({
              name: String(c.name || "tool"),
              arg: clip((c.args && (c.args.query ?? c.args.target ?? JSON.stringify(c.args))) as string, 60),
            }));
          }
        } else if (t === "router_thinking") {
          // Raw reasoning trace from a thinking model (req ②).
          const s = segFor();
          if (p.text) s.thinking = ((s.thinking || "") + String(p.text)).slice(-2000);
        } else if (t === "tool_error") {
          // A tool call failed (req ③: surface failures, don't swallow).
          const s = segFor();
          s.toolErrors = s.toolErrors || [];
          s.toolErrors.push({ name: String(p.target || p.brief || "tool"), error: clip(p.findings || p.obs_summary || translateNow("multimodal.misc.callFailed"), 120) });
        } else if (t === "bg_progress") {
          // A search/recall dispatched — show it "in flight" (query, no result yet).
          const s = segFor();
          const kind = ch === "recall" ? "recall" : "search";
          const query = clip(p.brief, 80);
          if (query && !s.lookups.some((l) => l.kind === kind && l.query === query)) {
            s.lookups.push({ kind, query });
          }
        } else if (t === "search_done" || t === "recall_done") {
          const s = segFor();
          const kind = t === "recall_done" ? "recall" : "search";
          const query = clip(p.brief, 80);
          const clues = t === "recall_done" && p.n_clues ? translateNow("multimodal.misc.clues", p.n_clues) : "";
          const result = translateNow("multimodal.misc.foundChars", p.findings_len || 0, clues)
            + (p.elapsed_sec != null ? ` · ${Number(p.elapsed_sec).toFixed(1)}s` : "");
          const existing = s.lookups.find((l) => l.kind === kind && l.query === query && !l.done);
          if (existing) { existing.result = result; existing.done = true; }
          else s.lookups.push({ kind, query, result, done: true });
        } else if (t === "answer_ready") {
          const s = segFor();
          s.ready = true;
          s.readyChars = p.text_len || 0;
          // Store this segment's interpretation text so the expanded card shows
          // it (req ④ folding). Only use the preview if we DIDN'T already stream
          // the full answer via answer_delta (else we'd truncate it to 400 chars).
          if (p.text_preview && !(s.answer && s.answer.length >= (p.text_len || 0)))
            s.answer = clip(p.text_preview, 400);
          // Fallback "看到" when the round jumped straight to answering with an
          // empty thought (self-explanatory scene) — use the answer preview so
          // the segment card is never just "📝就绪".
          if (!s.saw && p.text_preview) s.saw = clip(p.text_preview, 140);
        } else if (t === "progress_report") {
          cur.report = String(p.report || "");
          cur.reportBatches = p.batches || 0;
        } else if (p.phase === "crop_images") {
          const s = segFor();
          s.crops = (p.crops || []).filter((c: CropItem) => c.jpeg_b64);
        } else if (p.phase === "done") {
          cur.done = true;
          cur.waiting = null;   // 整个深度研究结束 → 才真正撤掉攒帧条
        }
        // Note: writer_start / distill / tool_obs / rN_decision / start are
        // intentionally ignored — internal ReAct steps, not user-facing progress.

        const next = idx >= 0 ? prev.slice() : [...prev, cur];
        if (idx >= 0) next[idx] = cur;
        // Cap segments per item so a very long run doesn't grow unbounded.
        if (cur.segments.length > 40) cur.segments = cur.segments.slice(-40);
        return next.slice(-8);
      }
    };
    // 批量 flush: 把队列里累积的 bg 事件一次性 reduce 进 bgItems, 只触发 1 次重渲染。
    // 由统一节流器 runUnifiedFlush 调用 (跟主 agent 流同一帧), 不再单独排 timer/rAF。
    const runBgFlush = () => {
      if (bgQueue.length === 0 && bgPendingExpandRid === null) return;
      const n = bgQueue.length;
      const drain = bgQueue; bgQueue = [];
      if (n > 0) {
        _mmAct("bgFlush", `queued=${n}`);
        setBgItems((prev) => {
          const next = drain.reduce((acc, p) => reduceBg(acc, p), prev);
          try {
            const seg = next.reduce((mx: number, b: BgItem) => Math.max(mx, b.segments.length), 0);
            (window as any).__mmN = { ...(window as any).__mmN, bg: next.length, seg };
          } catch { /* noop */ }
          return next;
        });
      }
      // Auto-open the matching sub-window (deferred from raw bg handler).
      if (bgPendingExpandRid !== null) {
        const rid = bgPendingExpandRid;
        bgPendingExpandRid = null;
        // ★ 用户已显式点过窗口头 → 保持用户选择 (包括 "" 折叠哨兵), 不再自动追新。
        //   未交互时自动展开最新窗口, 仅在用户折叠过 (cur === "") 时尊重折叠。
        if (!deepExpandedUserPinned.current) {
          setDeepExpanded((cur) => (cur === "" ? cur : rid));
        }
      }
    };
    // Multimodal monitor registry push (set_monitor CRUD result).
    const offMonitors = gw.on<{
      monitors?: MonitorReg[];
    }>("multimodal.monitors", (ev) => {
      if (!isMine(ev)) return;
      setMonitors(ev.payload?.monitors || []);
    });
    // Multimodal watcher registry push (set_live_watcher CRUD + reopen re-register).
    const offWatchers = gw.on<{ watchers?: WatcherReg[] }>(
      "multimodal.watchers", (ev) => {
        if (!isMine(ev)) return;
        setWatchers(ev.payload?.watchers || []);
      });
    // Generic blocking clarify.request from a tool (e.g. set_monitor silent-
    // mode). The backend blocks the tool thread until we answer via
    // clarify.respond, so we MUST surface it — otherwise the tool hangs ~300s
    // and its tool.complete never fires. Render inline in the chat waterfall as
    // a question + option buttons (Claude-Code-desktop style). Dedup by
    // request_id so a re-emit doesn't stack duplicate bubbles.
    const offClarify = gw.on<{ request_id?: string; question?: string; choices?: string[] | null }>(
      "clarify.request", (ev) => {
        if (!isMine(ev)) return;
        const p = ev.payload || {};
        const reqId = p.request_id || "";
        if (!reqId) return;
        const choices = Array.isArray(p.choices)
          ? p.choices.filter((c): c is string => typeof c === "string") : [];
        setMessages((prev) => {
          if (prev.some((m) => m.kind === "clarify" && m.clarifyReqId === reqId)) return prev;
          return capMsgs([...prev, {
            id: nid(), role: "assistant", text: "", kind: "clarify",
            clarifyReqId: reqId,
            clarifyQuestion: p.question || t.multimodal.misc.pleaseSelect,
            clarifyChoices: choices,
          }]);
        });
      });
    // Multimodal TTS chunk (legacy PCM streaming → WebAudio).
    const offTts = gw.on<{
      response_id?: string; pcm_b64?: string;
      sample_rate?: number; is_final?: boolean;
    }>("multimodal.tts", (ev) => onTtsChunk(ev.payload || {}));
    // Streaming realtime ASR: live partial preview + EOU buffer + final.
    const offAsrPartial = gw.on<{ text?: string; turn_id?: string }>(
      "multimodal.asr_partial", (ev) => {
        if (!asrTransport.ownsEvent(ev.session_id, ev.payload?.turn_id)) return;
        setAsrPartial(ev.payload?.text || "");
      });
    // EOU listening state: already-stitched segments (shown as dimmed prefix in AsrBar).
    const offAsrBuffer = gw.on<{ segments?: string[]; turn_id?: string }>(
      "multimodal.asr_buffer", (ev) => {
        if (!asrTransport.ownsEvent(ev.session_id, ev.payload?.turn_id)) return;
        setAsrBuffer(ev.payload?.segments ?? []);
      });
    const offAsrFinal = gw.on<{ text?: string; request_id?: string; turn_id?: string }>(
      "multimodal.asr_final", (ev) => {
        const turnId = ev.payload?.turn_id;
        if (!asrTransport.ownsEvent(ev.session_id, turnId)) return;
        const t = (ev.payload?.text || "").trim();
        if (t) {
          // ★ Preallocate this turn's answer slot, exactly as sendAsk does for
          //   the typed composer. Without it a voice turn behaved like a
          //   backend-originated one: `ensureBubble` only runs on the first
          //   message.delta / message.complete, so the answer bubble was
          //   appended AFTER whatever tools had already run. The array became
          //   `user, tool, assistant, tool` instead of `user, assistant, tool,
          //   tool` — which renders the first 处理过程 card ABOVE the answer and
          //   splits the turn's tool calls across two cards, since buildRows
          //   merges only adjacent tool rows. Typed turns never showed this
          //   because their slot is claimed at send time.
          const rid = ev.payload?.request_id;
          const key = rid || "__main__";
          const answerId = nid();
          const claim = !curAssistantId.current.get(key);
          if (claim) curAssistantId.current.set(key, answerId);
          setMessages((prev) => capMsgs([
            ...prev,
            { id: nid(), role: "user", text: t, voice: true, requestId: rid },
            ...(claim ? [{
              id: answerId,
              role: "assistant" as const,
              text: "",
              streaming: true,
              awaitingFirstDelta: true,
              hasReasoning: false,
              requestId: rid,
            }] : []),
          ]));
          refs.current.isAnswering = true;
        }
        setAsrPartial("");
        setAsrBuffer([]);
        if (typeof ev.session_id === "string" && typeof turnId === "string") {
          asrTransport.noteFinal(ev.session_id, turnId);
        }
      });
    // Anchor debug: the exact frames injected into the vision model this turn.
    const offAnchor = gw.on<{ frames?: { ts: number | null; jpeg_b64: string }[] }>(
      "multimodal.anchor", (ev) => {
        if (!isMine(ev)) return;
        const frames = ev.payload?.frames || [];
        if (frames.length) setAnchorFrames(frames);
      });
    // ★ 后端发起的 user turn 回显 (watcher/monitor hook 完成 → 把 hook 指令作为正式
    //   UserMessage 注入主 agent)。普通用户输入由前端本地 addMsg, 不走这里; 只有
    //   后端注入的 (前端没本地加过) 才靠这个 echo 显示 user 气泡, 否则用户看不到
    //   触发这轮的那条指令。
    const offUserEcho = gw.on<{
      text?: string; request_id?: string;
      history_policy?: unknown; ephemeral_control?: unknown; ephemeral?: unknown;
    }>("message.user_echo", (ev) => {
      if (!isMine(ev)) return;
      if (isEphemeralControl(ev.payload)) return;
      const t = (ev.payload?.text || "").trim();
      if (t) addMsg({
        id: nid(), role: "user", text: t,
        requestId: ev.payload?.request_id,
      });
    });
    // LLM latency diagnostic: chat_completion_helpers pushes SEND/RECV/
    // FIRST_BYTE events with model / msg count / image count / image bytes /
    // prompt tokens / reasoning tokens / elapsed seconds. Dump to F12 so you
    // can see "why is this turn slow" without SSHing to the gateway box.
    const offDiag = gw.on<any>("multimodal.diag", (ev) => {
      const p = ev.payload || {};
      // Green for SEND, blue for RECV/FIRST_BYTE, red if slow (>10s).
      const slow = typeof p.elapsed_s === "number" && p.elapsed_s > 10;
      const color = p.phase === "SEND" ? "#28a745"
        : slow ? "#dc3545" : "#0d6efd";
      // eslint-disable-next-line no-console
      console.log("%c[mm-llm] " + (p.phase || "?"),
        `color:${color};font-weight:bold`, p);
      // Cross-reference: mark this LLM event on the per-turn trace so a
      // single glance at `[mm-trace-fe]` tells you where the LLM SEND
      // happened relative to prompt.submit and where FIRST_BYTE arrived.
      if (p.phase === "SEND") traceOnce(`llm_SEND (msgs=${p.msgs ?? "?"} imgs=${p.imgs ?? 0})`);
      else if (p.phase === "FIRST_BYTE") traceOnce("llm_FIRST_BYTE");
      else if (p.phase === "RECV") traceOnce(`llm_RECV (elapsed_s=${p.elapsed_s ?? "?"})`);
    });
    // Unified, bounded worker trajectory. This is intentionally one typed event
    // rather than the old onAny logger that copied every chat token and made the
    // page stutter. Backend entries already include Writer/OCR/Recall/Search/
    // Watcher/Monitor/MainScheduler phases and recalled frame thumbnails.
    const offTrajectory = gw.on<MmTrajectoryEntry>("multimodal.trajectory", (ev) => {
      if (!isMine(ev)) return;
      const item = ev.payload;
      if (!item?.id) return;
      const queryProgress = queryWorkerProgressFromTrajectory(item);
      if (queryProgress) {
        const current = queryProgressByTaskRef.current.get(queryProgress.taskId) || [];
        if (!current.some((step) => step.id === queryProgress.step.id)) {
          queryProgressByTaskRef.current = updateQueryWorkerProgressCache(
            queryProgressByTaskRef.current,
            queryProgress.taskId,
            queryProgress.step,
          );
          msgQueue.push({
            action: "patch_query_worker",
            taskId: queryProgress.taskId,
            step: queryProgress.step,
          });
          scheduleFlush();
        }
      }
      setTrajectory((prev) => {
        if (prev.some((x) => x.id === item.id)) return prev;
        const next = [...prev, item];
        return compactQueryWorkerTrajectory(
          next.length > 2000 ? next.slice(next.length - 2000) : next,
        );
      });
    });


    // ★ 拉注册表 (monitor/watcher): 有未完成任务 → 右侧面板自动打开。
    //   抽为独立函数, 供 resume 和 create 两条路径复用。
    const fetchRegistries = (sid: string) => {
      if (!sid) return;
      gw.request<{
        monitors?: MonitorReg[];
        watchers?: WatcherReg[];
        ready?: boolean;
      }>(
        "multimodal.list_registries", { session_id: sid },
      ).then((r) => {
        // ★ K: list_registries 用 _sess_nowait, 冷 resume 时 agent 还没 build 完 →
        //   ready=false/legacy 空 pull 不覆盖已到的 push。agent 就绪后 ready=true，
        //   空数组也是权威快照，可清掉断线期间已删除的最后一张旧卡。
        setMonitors((prev) => resolveRegistryPull(prev, r?.monitors, r?.ready));
        setWatchers((prev) => resolveRegistryPull(prev, r?.watchers, r?.ready));
      }).catch(() => { /* best-effort */ });
    };
    refs.current.fetchRegistries = fetchRegistries;

    // Hydrate the multimodal sidechannel state (monitor alerts + watcher
    // content) on session resume. The main-agent history no longer carries
    // these — they live in dedicated DB tables.
    const fetchMmSidechannel = (sid: string) => {
      if (!sid) return;
      gw.request<{
        alerts?: {
          monitor_id: string;
          text: string;
          label?: string;
          wall_ts: number;
          evidence?: unknown;
        }[];
      }>(
        "multimodal.list_monitor_alerts", { session_id: sid },
      ).then((r) => {
        const list = Array.isArray(r?.alerts) ? r!.alerts! : [];
        if (list.length === 0) return;
        const grouped = new Map<string, MonitorAlert[]>();
        for (const a of list) {
          if (!a.monitor_id || !a.text) continue;
          const cur = grouped.get(a.monitor_id) || [];
          cur.push({
            id: `${a.monitor_id}_${a.wall_ts}_${cur.length}`,
            text: a.text,
            ts: Math.round((a.wall_ts || 0) * 1000),
            evidence: normalizeMonitorEvidence(a.evidence),
          });
          grouped.set(a.monitor_id, cur);
        }
        setMonitorAlerts(grouped);
      }).catch(() => { /* best-effort */ });
      gw.request<{
        reports?: { watcher_id: string; round_idx: number; text: string; label?: string; wall_ts: number }[];
        finals?: { watcher_id: string; text: string; wall_ts: number }[];
      }>("multimodal.list_watcher_content", { session_id: sid }).then((r) => {
        const reports = Array.isArray(r?.reports) ? r!.reports! : [];
        const finals = Array.isArray(r?.finals) ? r!.finals! : [];
        if (reports.length === 0 && finals.length === 0) return;
        // Reconstruct one BgItem per watcher_id from persisted reports+finals.
        // Each report becomes a BgSegment (seg = round_idx, answer = text). The
        // final report (if any) attaches to the BgItem's finalReport. Segments
        // are ordered by round_idx asc so the newest sits at the bottom (same
        // as live streaming order).
        const byRid = new Map<string, BgItem>();
        for (const rp of reports) {
          if (!rp.watcher_id || !rp.text) continue;
          const cur = byRid.get(rp.watcher_id)
            || { id: rp.watcher_id, requestId: rp.watcher_id, segments: [], done: true };
          cur.segments.push({ seg: rp.round_idx || cur.segments.length + 1,
                              lookups: [], answer: rp.text });
          if (rp.label && !cur.label) cur.label = rp.label;
          byRid.set(rp.watcher_id, cur);
        }
        for (const f of finals) {
          if (!f.watcher_id || !f.text) continue;
          const cur = byRid.get(f.watcher_id)
            || { id: f.watcher_id, requestId: f.watcher_id, segments: [], done: true };
          cur.finalReport = f.text;
          byRid.set(f.watcher_id, cur);
        }
        if (byRid.size === 0) return;
        // Merge with any bgItems already populated by live events (rare on
        // fresh resume, but be defensive): live wins if same rid + live has
        // richer segments (crops/lookups); else replace with restored one.
        setBgItems((prev) => {
          const live = new Map(prev.map((b) => [b.requestId || b.id, b]));
          for (const [rid, restored] of byRid) {
            const cur = live.get(rid);
            if (!cur) { live.set(rid, restored); continue; }
            // If the live item has any segment with crops/lookups (streaming
            // detail), keep it; otherwise adopt the restored segments.
            const richLive = cur.segments.some((s) =>
              (s.crops && s.crops.length > 0) || (s.lookups && s.lookups.length > 0));
            if (!richLive) cur.segments = restored.segments;
            if (!cur.finalReport && restored.finalReport) cur.finalReport = restored.finalReport;
            if (!cur.label && restored.label) cur.label = restored.label;
            live.set(rid, cur);
          }
          return Array.from(live.values()).slice(-8);
        });
      }).catch(() => { /* best-effort */ });
    };

    const fetchTrajectory = (sid: string) => {
      if (!sid) return;
      const generation = ++trajectoryHydrationGenerationRef.current;
      // ★ 切换会话只取尾部一段, 不要一次拉满 2000 条。后端上限仍是 2000, 但轨迹行
      //   会带 base64 证据缩略图 (每行最多 12 帧), 拉满时单帧 WS payload 实测 ~8.5MB
      //   → 切换瞬间白等一大段网络 + json 解析。而前端 compactQueryWorkerTrajectory
      //   本来就只保留最近几个 task 的图 (QUERY_WORKER_IMAGE_CHAR_BUDGET=4MB), 多拉的
      //   那部分图解析完立刻被丢掉, 纯浪费。往回翻/开 Debug 面板时再按需补。
      gw.request<{ entries?: MmTrajectoryEntry[] }>(
        "multimodal.trajectory.list", { session_id: sid, limit: TRAJECTORY_RESUME_LIMIT },
      ).then((res) => {
        if (!isCurrentTrajectoryHydration(
          sid,
          generation,
          refs.current.sessionId,
          trajectoryHydrationGenerationRef.current,
        )) return;
        const pulled = Array.isArray(res?.entries) ? res.entries : [];
        for (const item of pulled) {
          const qp = queryWorkerProgressFromTrajectory(item);
          if (!qp) continue;
          const current = queryProgressByTaskRef.current.get(qp.taskId) || [];
          if (current.some((step) => step.id === qp.step.id)) continue;
          queryProgressByTaskRef.current = updateQueryWorkerProgressCache(
            queryProgressByTaskRef.current,
            qp.taskId,
            qp.step,
          );
          msgQueue.push({ action: "patch_query_worker", taskId: qp.taskId, step: qp.step });
        }
        if (msgQueue.length) scheduleFlush();
        setTrajectory((live) => {
          const merged = new Map<string, MmTrajectoryEntry>();
          for (const it of [...pulled, ...live]) if (it?.id) merged.set(it.id, it);
          return compactQueryWorkerTrajectory(Array.from(merged.values())
            .sort((a, b) => (a.seq || 0) - (b.seq || 0))
            .slice(-2000));
        });
      }).catch(() => { /* best-effort */ });
    };

    // ★ resume 一个指定 session 并把历史灌进 waterfall。返回是否成功。
    //   restoreHistory=true 时把后端返回的 transcript 转成气泡显示 (只在 waterfall
    //   还是初始态时灌, 避免覆盖用户已输入 / 重复灌)。
    const resumeSessionById = async (
      targetSid: string, restoreHistory: boolean,
    ): Promise<boolean> => {
      if (!targetSid) return false;
      try {
        const res = await gw.request<{
          session_id?: string; session_key?: string; resumed?: string;
          messages?: unknown; orphan_event_ids?: string[];
        }>("session.resume", {
          session_id: targetSid,
          // A persisted session can predate the dedicated source value. The
          // multimodal page is authoritative about the runtime it needs when
          // reopening it.
          source: "multimodal",
          close_on_disconnect: false,
        });
        // ★ 两个 id: session_id=live runtime key (RPC 路由用); session_key/resumed=
        //   持久 DB id (跨 auto-compress 稳定, ?mm=/侧边栏/localStorage 用)。之前错把
        //   live sid 存进 localStorage → 下次 resume 必 404。
        const liveSid = res?.session_id || targetSid;
        const storedSid = res?.session_key || res?.resumed || targetSid;
        refs.current.sessionId = liveSid;
        refs.current.storedSid = storedSid;
        try { localStorage.setItem(_MM_SESSION_KEY, storedSid); } catch { /* noop */ }
        offerVoiceDialogSession(liveSid);
        if (restoreHistory) {
          // ★ F: 孤儿 monitor/watcher (history 有、本 session 磁盘无) → 丢弃气泡 + 提示。
          const orphans = new Set(
            Array.isArray(res?.orphan_event_ids) ? res!.orphan_event_ids : []);
          const restored = historyToMmMessages(res?.messages, orphans);
          if (restored.length > 0) {
            // ★ 窗口化: 全量历史存 ref (含头部), 只渲染尾部 MAX_MESSAGES 条; 头部留给
            //   "翻到顶再取"(loadOlderHistory)。补 createdAt (capMsgs 会截头, 这里要全量)。
            const now = Date.now();
            for (const m of restored) if (m.createdAt == null) m.createdAt = now;
            fullHistoryRef.current = restored;
            const { firstStart, fullStart, needsSecondPaint } = historyPaintWindow(
              restored.length, MAX_MESSAGES);
            // ★ 置顶欢迎气泡 (纯前端引导, 不入 backend history)。老 session 刷新走这里
            //   恢复历史, 若不主动 prepend 会被 restored 冲掉。
            // ★ 两段式首屏: 先只交付尾部一屏 (用户视线本来就落在最新那条上), 让气泡立刻
            //   可见; 下一帧再把渲染窗补到 MAX_MESSAGES。列表是非虚拟化的普通 div, 一次
            //   交付 400 条要同步解析 400 份 Markdown —— 这是"切换后要等一下才出内容"的
            //   前端那一半原因。
            setMessages([_mmWelcomeMsg(), ...restored.slice(firstStart)]);
            // 首屏之上还有内容 → 先允许上翻 (第二段补齐后由 loadOlderHistory 自己更新)。
            setHasMoreHistory(firstStart > 0);
            if (needsSecondPaint) {
              requestAnimationFrame(() => {
                // 期间用户可能又切走了 → 别把上一会话的历史补进新会话。
                if (refs.current.sessionId !== liveSid) return;
                // 不用像 loadOlderHistory 那样补 scrollTop: 那是"用户停在顶部往上翻"的
                // 场景 (要锚住视线); 这里刚切完会话, 用户在底部, ChatColumn 里跟随底部的
                // useLayoutEffect 会因 rows 变化再拉一次到底 —— 视线始终在最新那条上。
                const older = restored.slice(fullStart, firstStart);
                setMessages((cur) => {
                  // 把 older prepend 到【首屏那批之前】, 保留 cur 里已有的一切 (欢迎气泡在
                  // 头部, 尾部可能已经流进来了新的实时气泡)。按 id 去重防重复补。
                  const have = new Set(cur.map((m) => m.id));
                  const add = older.filter((m) => !have.has(m.id));
                  if (add.length === 0) return cur;
                  const welcomeAtHead = cur[0]?.role === "system"
                    && cur[0]?.text === _mmWelcomeMsg().text ? 1 : 0;
                  return welcomeAtHead
                    ? [cur[0], ...add, ...cur.slice(1)]
                    : [...add, ...cur];
                });
                setHasMoreHistory(fullStart > 0);
              });
            }
          } else {
            // 空 history 也要留一条欢迎气泡 (与 resetSessionUi 一致)。
            setMessages([_mmWelcomeMsg()]);
          }
          if (orphans.size > 0) {
            pushTopToast(
              `error msg: monitor / watcher event id ${Array.from(orphans).join(", ")} not found in local files.`,
              "error");
          }
        }
        fetchRegistries(liveSid);
        fetchMmSidechannel(liveSid);
        fetchTrajectory(liveSid);
        return true;
      } catch {
        return false;
      }
    };

    // ★ 新建一个持久化 session (close_on_disconnect: false → WS 断开不销毁;
    //   source: "multimodal" → 区分 TUI("tui")/子 agent("tool"), session.list 可见)。
    //   返回新会话的持久 id (供"新建"路径把 URL ?mm= 换成它)。
    const createSession = async (): Promise<string> => {
      try {
        const res = await gw.request<{ session_id: string; stored_session_id?: string }>(
          "session.create", { close_on_disconnect: false, source: "multimodal" },
        );
        const liveSid = res?.session_id || "";
        const storedSid = res?.stored_session_id || liveSid;
        refs.current.sessionId = liveSid;
        refs.current.storedSid = storedSid;
        offerVoiceDialogSession(liveSid);
        if (storedSid) {
          try { localStorage.setItem(_MM_SESSION_KEY, storedSid); } catch { /* noop */ }
        }
        // 新会话是空的: 清掉上一会话残留的 UI 状态 + 灌入初始欢迎气泡。
        resetSessionUi();
        fetchRegistries(liveSid);
        fetchMmSidechannel(liveSid);
        fetchTrajectory(liveSid);
        return storedSid;
      } catch (e) {
        addMsg({ id: nid(), role: "system",
          text: translateNow("multimodal.errors.sessionFailed",
            e instanceof Error ? e.message : String(e)) });
        return "";
      }
    };
    // Expose create so the ?mm=new (新建) handler can call it.
    refs.current.createSession = createSession;

    // ★ 建会话的决策链 (首次连接 / 重连时都走它):
    //   0) URL ?mm=new (侧边栏加号) → 强制新建一个空会话
    //   1) URL ?mm=<id> 指定 → resume 它 (侧边栏点选)
    //   2) localStorage 上次打开的 → resume 它 (刷新保持)
    //   3) 都没有 → 拉最近会话列表, resume 最上面那条 (默认打开最新)
    //   4) 一条都没有 → 新建
    const establishSession = async () => {
      try {
        const urlSid = new URLSearchParams(window.location.search).get("mm");
        if (urlSid === "new") {
          const newSid = await createSession();
          if (newSid) setSearchParams({ mm: newSid }, { replace: true });
          return;
        }
        if (urlSid && await resumeSessionById(urlSid, true)) return;

        let savedSid: string | null = null;
        try { savedSid = localStorage.getItem(_MM_SESSION_KEY); } catch { /* noop */ }
        if (savedSid && await resumeSessionById(savedSid, true)) return;
        // resume 失败 → 清掉过期 id
        if (savedSid) { try { localStorage.removeItem(_MM_SESSION_KEY); } catch { /* noop */ } }

        // 默认打开最近一条会话 (session list 最上面)。exclude tool/cron 子会话,
        // 否则可能 resume 到子 agent/cron 会话当主对话。
        try {
          const list = await api.getSessions(1, 0, scopedProfile ?? "", "recent", "tool,cron");
          const top = list?.sessions?.[0]?.id;
          if (top && await resumeSessionById(top, true)) return;
        } catch { /* best-effort → 落到 create */ }

        await createSession();
      } finally {
        const stillOpen = gw.state === "open";
        sessionEstablishedRef.current = stillOpen;
        if (stillOpen && !refs.current.sessionId) failWaitingVoiceDialog();
      }
    };
    // Expose the resume core so the ?mm= watcher effect can switch sessions.
    refs.current.resumeSessionById = resumeSessionById;

    // ★ Connection state drives the UI + capture. Without this the badge was set
    //   true once and never cleared — a dropped WS looked "connected" forever
    //   while frames were black-holed. Now: badge follows real state, capture
    //   pauses off-line, and an auto-reconnect rebuilds the session.
    const offGwState = gw.onState((s) => {
      setConnected(s === "open");
      setConnState(s);
      if (s !== "open") {
        sessionEstablishedRef.current = false;
        markVoiceDialogBoundary();
        // Pause frame capture while disconnected — pushing into a dead socket
        // just burns CPU/encoding for frames that go nowhere.
        try { stopCapture(); } catch { /* noop */ }
        // ★ E: WS 断了, 正在流式的那轮不会再收到 message.complete/error → streaming
        //   标志永远清不掉, composer 卡在"停止"。这里主动清所有 streaming + 打开的
        //   bubble 映射, 让 composer 回到"发送"。
        curAssistantId.current.clear();
        setMessages((prev) => {
          if (!prev.some((m) => m.streaming)) return prev;
          return prev.map((m) => (m.streaming
            ? { ...m, streaming: false, queued: false, queuePosition: undefined }
            : m));
        });
        // ★ J: 断线时后端 ASR 会话被回收, 但前端仍以为在录音 (红点常亮, PCM 打到死
        //   socket)。这里同步拆本地 mic (inline, 因 stopMic 定义在此 effect 之后)。
        const r = refs.current;
        if (r.asrTransport?.current() || r.isRecording || r.micStream) {
          void cancelActiveMic(r);
          setMicState("idle");
          setAsrPartial("");
          setAsrBuffer([]);
        }
        // Never route post-disconnect work through the stale runtime id. The
        // durable id remains in storedSid/localStorage for establishSession.
        r.sessionId = "";
      }
    });
    const offReconnect = gw.onReconnect(() => {
      // WS came back — session 还在 (close_on_disconnect: false), establishSession
      // 先 try resume 再 fallback create。
      void establishSession().then(() => {
        if (refs.current.stream) { try { startCapture(); } catch { /* noop */ } }
      });
    });

    gw.connect()
      .then(() => establishSession())
      .then(() => {
        // Reuse this connection to fetch the multimodal readiness advisory once
        // (no separate WS). probe_endpoints=true opts in to a bounded TCP probe
        // of each configured LLM endpoint so the banner can warn "endpoint
        // unreachable — requests will hang" instead of the user hitting the
        // mysterious "agent initialization timed out" wall.
        gw.request<MmReadinessReport>("mm.readiness", { probe_endpoints: true })
          .then((r) => { if (r && typeof r.ready === "boolean") setMmReadiness(r); })
          .catch(() => { /* advisory is best-effort */ });
      })
      .catch((e: Error) => addMsg({ id: nid(), role: "system",
        text: translateNow("multimodal.errors.connectionFailed", e.message) }));

    return () => {
      flushDisposed = true;
      clearInterval(_watchdog);
      if (flushTimer !== null) clearTimeout(flushTimer);
      if (flushRaf !== null) cancelAnimationFrame(flushRaf);
      if (ctxRaf !== null) cancelAnimationFrame(ctxRaf);
      msgQueue = [];
      bgPendingExpandRid = null;
      offGwState();
      offReconnect();
      offStart();
      offDelta();
      offComplete();
      offCtx();
      offToolStart();
      offToolComplete();
      offWatcherFinal();
      offReportAppend();
      offDeepComplete();
      offStatus();
      offReasoning();
      offThinking();
      offReasoningSummary();
      offError();
      offSessionInfo();
      offToast();
      offBg();
      offMonitors();
      offWatchers();
      offClarify();
      offTts();
      offAsrPartial();
      offAsrBuffer();
      offAsrFinal();
      offAnchor();
      offUserEcho();
      offDiag();
      offTrajectory();
      stopAllTts(true);
      // Close the AudioContext (browsers cap ~6 per page; leaking one per
      // mount eventually throws on new AudioContext()).
      try {
        const ac = ttsRefs.current.audioCtx;
        if (ac && ac.state !== "closed") ac.close();
      } catch { /* noop */ }
      ttsRefs.current.audioCtx = null;
      // Stop the streaming mic (not covered by stopStream, which only tears
      // down the video + env-audio path). Unmount is always cancel, never send.
      cancelActiveMic(refs.current);
      refs.current.asrTransport = null;
      // Clear the per-key bubble map so it can't leak across a remount.
      curAssistantId.current.clear();
      refs.current.isAnswering = false;
      stopStream();
      gw.close();
      refs.current.gw = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── ?mm= 切换: 用户在侧边栏点了另一条 session → resume 它 + 恢复历史 ──────────
  //   与上面的 mount-once 建会话逻辑分开: mount 时那次已按初始 ?mm= 处理, 这里只
  //   处理【挂载后】param 的变化。切换前清空 waterfall (旧会话气泡不能串到新会话),
  //   由 resumeSessionById(restoreHistory=true) 灌入新会话的 transcript。
  useEffect(() => {
    if (!mmParam) return;
    if (!connected || !sessionEstablishedRef.current) return;
    const resume = refs.current.resumeSessionById;
    if (!resume) return;                       // 会话地基还没建好, mount 那次会处理
    // ★ 新建 (?mm=new, 侧边栏加号): 强制建一个空会话, 再把 URL 换成新会话 id
    //   (replace, 不留 new 在历史)。切换前先关流 + 清 UI。
    if (mmParam === "new") {
      sessionEstablishedRef.current = false;
      const oldOwnerClosed = leaveVoiceDialogSession();
      // Session navigation is an intentional synchronous UI boundary.
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setMicState("idle");
      if (refs.current.stream) { try { stopStream(); } catch { /* noop */ } }
      refs.current.sessionId = "";
      // 切换会话必须【同步】清空上一会话 UI (气泡/深研窗/面板), 否则旧内容会串到新
      // 会话——这是有意的同步 reset, 不是 cascading-render bug。
      // eslint-disable-next-line react-hooks/set-state-in-effect
      resetSessionUi();
      void oldOwnerClosed.then(() => refs.current.createSession?.() || "").then((newSid) => {
        if (newSid) setSearchParams({ mm: newSid }, { replace: true });
      }).finally(() => {
        const stillOpen = refs.current.gw?.state === "open";
        sessionEstablishedRef.current = stillOpen;
        if (stillOpen && !refs.current.sessionId) failWaitingVoiceDialog();
      });
      return;
    }
    // ★ 守卫比【持久 id】(storedSid), 不是 live sid —— mmParam 是持久 id, live sid
    //   会因 auto-compress 变化, 用它比永不相等 (死守卫)。
    if (mmParam === refs.current.storedSid) return;  // 已经是当前会话, 免重复
    // ★ G2: 切换 session 前先关掉视频流 —— 采集属于【旧会话】, 不能带进新会话。
    //   stopStream() 会给旧 session 发 source_stopped{started:false} (让旧会话的
    //   monitor/watcher 停止等帧) + 停本地采集。新会话默认无流, 用户按需重开;
    //   因此不需要给新会话补 started:true 握手 (本来就没流)。
    const oldOwnerClosed = leaveVoiceDialogSession();
    // Session navigation is an intentional synchronous UI boundary.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setMicState("idle");
    if (refs.current.stream) { try { stopStream(); } catch { /* noop */ } }
    refs.current.sessionId = "";
    // 切换前清空上一会话的【全部】UI 状态 (气泡/深研窗/注入帧/观察/监控/toast/帧数),
    // 再由 resumeSessionById(restoreHistory=true) 灌入新会话的 transcript + registries。
    resetSessionUi();
    void oldOwnerClosed.then(() => resume(mmParam, true)).then((ok) => {
      if (!ok && refs.current.gw?.state === "open") failWaitingVoiceDialog();
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mmParam, connected]);

  // ── Frame capture ──────────────────────────────────────────────────────
  // Reuse ONE offscreen canvas across ticks. The old code did
  // document.createElement("canvas") every 500ms — cheap allocation but
  // 2 fps × long session = thousands of throwaway canvases + GC pressure.
  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null);
  // ASYNC frame capture. The old sync path called canvas.toDataURL() which
  // BLOCKS the main thread for 20-60 ms per 720p JPEG encode — every 500 ms
  // at 2 fps that's 5-15% of every second frozen. Symptom: user sends a
  // query while screen-sharing, WS delta events arrive on time but React
  // never repaints because setState-triggered rAFs sit behind the next
  // toDataURL. Chrome/Edge exposes canvas.convertToBlob() which offloads
  // the encode to a compositor thread; combined with FileReader (also
  // async) the whole capture stays off the main thread. Fallback to
  // toDataURL only on ancient browsers where convertToBlob is missing.
  const captureFrame = useCallback(async (): Promise<string | null> => {
    const r = refs.current;
    const v = videoRef.current;
    if (!v || !v.videoWidth) return null;
    let w = v.videoWidth, h = v.videoHeight;
    const isScreen = r.sourceType === "screen";
    const profile = visualCaptureProfile(
      isScreen ? "screen" : "camera",
      preferLightCapture(),
    );
    const maxSide = profile.maxSide;
    if (maxSide > 0 && Math.max(w, h) > maxSide) {
      const scale = maxSide / Math.max(w, h);
      w = Math.round(w * scale); h = Math.round(h * scale);
    }
    let cvs = captureCanvasRef.current;
    if (!cvs) {
      cvs = document.createElement("canvas");
      captureCanvasRef.current = cvs;
    }
    if (cvs.width !== w) cvs.width = w;
    if (cvs.height !== h) cvs.height = h;
    await blitVideoToCanvas(v, cvs, w, h, profile.resizeQuality);
    const quality = profile.jpegQuality;

    // Async path: convertToBlob offloads JPEG encode off the main thread.
    // Available in Chrome 66+/Edge 79+/Firefox 105+ — safe for a dashboard.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const cAny = cvs as any;
    if (typeof cAny.convertToBlob === "function"
        || typeof (cvs as HTMLCanvasElement).toBlob === "function") {
      const blob = await new Promise<Blob | null>((resolve) => {
        if (typeof cAny.convertToBlob === "function") {
          cAny.convertToBlob({ type: "image/jpeg", quality })
            .then(resolve, () => resolve(null));
        } else {
          (cvs as HTMLCanvasElement).toBlob(resolve, "image/jpeg", quality);
        }
      });
      if (!blob) return null;
      // FileReader.readAsDataURL is async & runs the base64 encode off-thread.
      // Slice off the "data:image/jpeg;base64," prefix — the gateway wants
      // the raw base64 body (it strips data: prefixes but the extra bytes
      // are pure waste on a WS notify at 2 fps).
      const dataUrl = await new Promise<string | null>((resolve) => {
        const fr = new FileReader();
        fr.onload = () => resolve(typeof fr.result === "string" ? fr.result : null);
        fr.onerror = () => resolve(null);
        fr.readAsDataURL(blob);
      });
      if (!dataUrl) return null;
      const i = dataUrl.indexOf(",");
      return i >= 0 ? dataUrl.slice(i + 1) : dataUrl;
    }
    // Legacy sync fallback.
    return cvs.toDataURL("image/jpeg", quality).split(",")[1] || null;
  }, []);

  const startCapture = useCallback(() => {
    const r = refs.current;
    const period = Math.max(50, Math.round(1000 / r.capFps));
    if (r.capTimer) clearInterval(r.capTimer);
    // Backpressure: if the WS send buffer has more than this many bytes
    // waiting to hit the wire, skip this tick. Screen frames are ~200 KB
    // base64 apiece; anything above ~2 frames pending means we're already
    // saturating the WS and chat SSE would queue behind us. 512 KB gives
    // one frame of headroom without letting the buffer balloon.
    const BUF_LIMIT = 512 * 1024;
    // Reentrancy guard: if a capture is still in flight when the next tick
    // fires (slow encode / disk-IO stall), skip rather than pile up
    // parallel encodes on the compositor. Without this a 2 fps interval
    // could balloon into 5-10 concurrent encodes under load.
    let inFlight = false;
    r.capTimer = window.setInterval(() => {
      const gw = r.gw;
      const sessionId = r.sessionId;
      const stream = r.stream;
      const sourceType = r.sourceType;
      const captureAttemptId = r.captureAttemptId;
      if (!gw || !sessionId || !stream || !sourceType || !captureAttemptId) return;
      if (inFlight) return;
      // ★ Capture NEVER pauses — not even while the agent is answering. Pausing
      //   on isAnswering dropped a run of frames every time the agent spoke,
      //   leaving holes in the video the monitor / deep-analysis rely on, and
      //   staling _last_push_wall (which wrongly read as "source stopped"). The
      //   starvation this once guarded against is already handled: the inFlight
      //   reentrancy guard, off-thread JPEG encode (convertToBlob/FileReader),
      //   the setTimeout(0) yield below, and bufferedAmount backpressure — so at
      //   2fps the per-tick cost is negligible and must not cost us frames.
      inFlight = true;
      void (async () => {
        try {
      const data = await captureFrame();
      if (!data) return;
      if (r.gw !== gw || r.sessionId !== sessionId || r.stream !== stream
        || r.sourceType !== sourceType || r.captureAttemptId !== captureAttemptId) return;
      const ts = (performance.now() - r.startTs) / 1000;
      // Yield so stream flushes + UI events run before JSON.stringify blocks.
      await new Promise<void>((resolve) => { setTimeout(resolve, 0); });
      // Fire-and-forget notify (no ACK): the frame is best-effort, we
      // don't need a per-frame roundtrip and blocking on ACK meant every
      // qwen SSE token had to queue behind the frame reply on the single
      // shared WS. `notify` returns the post-send bufferedAmount so we
      // can drop the NEXT tick if the pipe is backed up.
      const buffered = gw.notify(
        "multimodal.frame",
        {
          session_id: sessionId,
          ts,
          jpeg_b64: data,
          source_type: sourceType,
          capture_attempt_id: captureAttemptId,
        },
      );
      if (buffered < 0) return; // socket not open
      if (buffered > BUF_LIMIT) {
        r.droppedFrames = (r.droppedFrames || 0) + 1;
        return;
      }
      r.sentFrames++;
      // Throttle the frame-count state push to ~1/s. It only feeds a display
      // number in the local video overlay; pushing every frame (2 fps) forces a
      // MultimodalChatPage re-render each tick for no visible benefit. The final
      // count is flushed by stopCapture.
      {
        const _now = performance.now();
        if (_now - (r._lastCountPush || 0) >= 1000) {
          r._lastCountPush = _now;
          setFrameCount(r.sentFrames);
        }
      }
        } finally {
          inFlight = false;
        }
      })();
    }, period);
  }, [captureFrame]);

  const stopCapture = useCallback(() => {
    const r = refs.current;
    if (r.capTimer) { clearInterval(r.capTimer); r.capTimer = null; }
    // Flush the final count (throttled pushes may have skipped the last frames).
    r._lastCountPush = 0;
    setFrameCount(r.sentFrames);
  }, []);

  const attachStream = useCallback(async (stream: MediaStream, st: SourceType) => {
    const r = refs.current;
    r.stream = stream; r.sourceType = st;
    const random = typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID()
      : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
    r.captureAttemptId = `webcap_${random}`;
    setSourceType(st);
    const v = videoRef.current;
    if (v) { v.srcObject = stream; await v.play().catch(() => {}); }
    r.startTs = performance.now();
    startCapture();
    // Tell the backend a video source is LIVE, so a continuous deep-analysis
    // run knows to keep waiting for frames (vs one-shot when no source).
    if (r.gw && r.sessionId) {
      r.gw.request("multimodal.source_stopped", {
        session_id: r.sessionId,
        started: true,
        source_type: st,
        capture_attempt_id: r.captureAttemptId,
      }).catch(() => { /* best-effort */ });
    }
  }, [startCapture]);

  const stopStream = useCallback(() => {
    const r = refs.current;
    const captureAttemptId = r.captureAttemptId;
    stopCapture();
    // Inline env-audio teardown (avoid const-ordering coupling with stopEnvAudio).
    r.envStop = true;
    if (r.envSliceTimer != null) {
      clearTimeout(r.envSliceTimer);
      r.envSliceTimer = null;
    }
    if (r.envRecorder) {
      try { if (r.envRecorder.state === "recording") r.envRecorder.stop(); } catch { /* noop */ }
      r.envRecorder = null;
    }
    if (r.envStream) { r.envStream.getTracks().forEach((t) => t.stop()); r.envStream = null; }
    r.envCaptureId = "";
    r.envChunkSeq = 0;
    r.envLastError = "";
    if (r.stream) r.stream.getTracks().forEach((t) => t.stop());
    r.stream = null; r.sourceType = null;
    r.captureAttemptId = "";
    setSourceType(null);
    if (videoRef.current) videoRef.current.srcObject = null;
    // Tell the backend the video source CLOSED, so any continuous deep-analysis
    // run stops waiting for new frames and finishes after draining the buffer.
    if (r.gw && r.sessionId) {
      r.gw.request("multimodal.source_stopped", {
        session_id: r.sessionId,
        started: false,
        capture_attempt_id: captureAttemptId,
      }).catch(() => { /* best-effort */ });
    }
  }, [stopCapture]);

  const startCamera = useCallback(async () => {
    if (refs.current.stream) return;
    try {
      const profile = visualCaptureProfile("camera", preferLightCapture());
      // ★ frameRate ideal:24 → smooth local preview. The vision pipeline still
      //   only SAMPLES the buffer at capFps (~2fps); the higher source rate only
      //   affects the on-screen <video> mirror's smoothness, not the push rate.
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: profile.width, max: profile.width },
          height: { ideal: profile.height, max: profile.height },
          frameRate: { ideal: profile.sourceFrameRate },
          facingMode: "user",
        },
        audio: false,
      });
      await attachStream(stream, "camera");
    } catch (e: any) {
      addMsg({ id: nid(), role: "system",
        text: translateNow("multimodal.errors.cameraFailed", String(e?.message)) });
    }
  }, [attachStream, addMsg]);

  // ── Mic: streaming realtime ASR (DashScope). Ordinary mic is one
  //    manual turn: click to capture + preview, click again to flush and submit
  //    one combined utterance. Voice-dialog remains continuous/auto-segmented.
  //
  //    Downsampling runs in an AudioWorklet (dedicated audio thread) so main-
  //    thread stays free for UI; the worklet batches ~200ms of PCM before
  //    posting to us, cutting RPC rate ~2.5x vs the old 85ms cadence. Server-
  //    side VAD (silence_ms=1200) doesn't care about packet cadence.
  // ────────────────────────────────────────────────────────────────────────
  const startMic = useCallback(async (
    mode: AsrTurnMode = "manual_turn",
  ): Promise<boolean> => {
    const r = refs.current;
    const transport = r.asrTransport;
    if (r.isRecording || r.micStopPromise || transport?.current()) return false;
    if (!r.gw || !r.sessionId || !transport) return false;
    const sessionId = r.sessionId;
    const generation = ++r.micGeneration;
    setMicState("connecting");
    setAsrPartial("");
    setAsrBuffer([]);

    let mediaPromise: Promise<MediaStream>;
    try {
      // Start local permission/capture first. Backend startup begins in this
      // same task and the transport buffers PCM until it is ready.
      mediaPromise = navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          channelCount: 1,
        },
      });
    } catch (error) {
      setMicState("idle");
      if (mode === "manual_turn") {
        addMsg({ id: nid(), role: "system",
          text: translateNow("multimodal.errors.micFailed",
            error instanceof Error ? error.message : String(error)) });
      }
      return false;
    }

    const turn = transport.begin(sessionId, mode);
    const backendReady = turn.ready.then(() => true).catch((error) => {
      const current = transport.current();
      if (refs.current.micGeneration !== generation || current?.turnId !== turn.turnId) return false;
      refs.current.micGeneration += 1;
      void transport.stop(sessionId, turn.turnId, "cancel").catch(() => undefined);
      void stopLocalMic(refs.current, false);
      setMicState("idle");
      setAsrPartial("");
      setAsrBuffer([]);
      if (mode === "manual_turn") {
        addMsg({ id: nid(), role: "system",
          text: translateNow("multimodal.errors.streamingVoiceFailed",
            error instanceof Error ? error.message : String(error)) });
      }
      return false;
    });

    try {
      const stream = await mediaPromise;
      if (r.micGeneration !== generation || transport.current()?.turnId !== turn.turnId) {
        stream.getTracks().forEach((track) => track.stop());
        return false;
      }
      r.micStream = stream;
      const Ctx = window.AudioContext || (window as any).webkitAudioContext;
      const ctx = new Ctx();
      r.micAudioCtx = ctx;
      if (ctx.state === "suspended") await ctx.resume();
      if (r.micGeneration !== generation || transport.current()?.turnId !== turn.turnId) {
        await stopLocalMic(r, false);
        return false;
      }
      // Worklet lives at /pcm-worklet.js served from public/. Prefix with the
      // deployed base path so it resolves correctly under a non-root mount.
      await ctx.audioWorklet.addModule(`${HERMES_BASE_PATH}/pcm-worklet.js`);
      if (r.micGeneration !== generation || transport.current()?.turnId !== turn.turnId) {
        await stopLocalMic(r, false);
        return false;
      }
      const source = ctx.createMediaStreamSource(stream);
      r.micSource = source;
      const node = new AudioWorkletNode(ctx, "pcm-downsample-processor", {
        numberOfInputs: 1, numberOfOutputs: 1,
        processorOptions: { inRate: ctx.sampleRate, batchMs: 200 },
      });
      r.micNode = node;
      // Worklet posts Int16 ArrayBuffer batches. Main thread only base64-
      // encodes + fires one RPC per batch — no per-sample math here.
      node.port.onmessage = (ev: MessageEvent) => {
        const rr = refs.current;
        if (ev.data && !(ev.data instanceof ArrayBuffer) && ev.data.type === "flushed") {
          rr.micFlushResolve?.();
          return;
        }
        if (!rr.isRecording) return;
        // Barge-in guard: while the assistant's TTS is audible (+ tail), drop the
        // mic PCM so speaker output isn't re-captured and looped back into ASR.
        if (Date.now() < ttsRefs.current.ttsMuteUntil) return;
        const buf = ev.data as ArrayBuffer;
        if (!buf || !buf.byteLength) return;
        // Encode ArrayBuffer → base64 without going through a string first
        // (fromCharCode.apply blows the stack on large buffers).
        const bytes = new Uint8Array(buf);
        let bin = "";
        for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
        const pcm_b64 = btoa(bin);
        transport.pushPcm(sessionId, turn.turnId, pcm_b64);
      };
      source.connect(node);
      node.connect(ctx.destination);
      r.isRecording = true;
      // Physical capture is live. If backend start is still pending, the
      // bounded pre-roll retains these packets in order.
      setMicState("recording");
      return await backendReady;
    } catch (error) {
      if (r.micGeneration !== generation) return false;
      r.micGeneration += 1;
      await stopLocalMic(r, false);
      await transport.stop(sessionId, turn.turnId, "cancel").catch(() => undefined);
      setMicState("idle");
      if (mode === "manual_turn") {
        addMsg({ id: nid(), role: "system",
          text: translateNow("multimodal.errors.micFailed",
            error instanceof Error ? error.message : String(error)) });
      }
      return false;
    }
  }, [addMsg]);

  const stopMic = useCallback(async (disposition: AsrStopDisposition = "finish") => {
    const r = refs.current;
    const transport = r.asrTransport;
    if (r.micStopPromise) {
      if (disposition !== "cancel") return r.micStopPromise;
      // New/profile/session/disconnect is a stronger boundary than a manual
      // finish already waiting on the provider. Invalidate the old UI owner,
      // stop the physical input, and send a distinct exact-turn cancellation.
      const finishingTurn = transport?.current();
      r.micGeneration += 1;
      setMicState("idle");
      setAsrPartial("");
      setAsrBuffer([]);
      const cancelled = finishingTurn && transport
        ? transport.stop(finishingTurn.sessionId, finishingTurn.turnId, "cancel")
        : Promise.resolve();
      await stopLocalMic(r, false);
      await cancelled.catch(() => undefined);
      return;
    }
    const turn = transport?.current();
    if (!turn || !transport) {
      await stopLocalMic(r, false);
      setMicState("idle");
      return;
    }
    const stopGeneration = ++r.micGeneration;
    const stillOwnsStopUi = () => ownsAsrStopUi(
      r.micGeneration,
      stopGeneration,
      r.sessionId,
      turn.sessionId,
    );
    setMicState(disposition === "finish" ? "finalizing" : "idle");
    const anchorParams = disposition === "finish" && r.stream
      && r.captureAttemptId && r.startTs > 0
      ? {
          anchor_ts: Math.max(0, (performance.now() - r.startTs) / 1000),
          capture_attempt_id: r.captureAttemptId,
        }
      : {};

    const operation = (async () => {
      try {
        if (disposition === "cancel") {
          // Mark the turn cancelled first so late worklet packets fail closed.
          const cancelled = transport.stop(turn.sessionId, turn.turnId, "cancel");
          await stopLocalMic(r, false);
          await cancelled;
        } else {
          // Physical capture stops synchronously inside stopLocalMic. Its
          // already-processed tail is flushed before the one backend finish.
          const flushConfirmed = await stopLocalMic(r, true);
          if (!flushConfirmed) {
            await transport.stop(turn.sessionId, turn.turnId, "cancel")
              .catch(() => undefined);
            throw new Error(translateNow("multimodal.errors.micFlushFailed"));
          }
          const result = await transport.stop(
            turn.sessionId,
            turn.turnId,
            "finish",
            anchorParams,
          );
          const failure = asrFinishFailureMessage(result);
          if (failure) throw new Error(failure);
        }
      } catch (error) {
        if (disposition === "finish" && stillOwnsStopUi()) {
          addMsg({ id: nid(), role: "system",
            text: translateNow("multimodal.errors.voiceSendFailed",
              error instanceof Error ? error.message : String(error)) });
        }
      } finally {
        // A session switch/newer capture owns the UI now; this old stop must
        // not clear its state or live transcript when it eventually settles.
        if (stillOwnsStopUi()) {
          setMicState("idle");
          setAsrPartial("");
          setAsrBuffer([]);
        }
      }
    })();
    r.micStopPromise = operation;
    try {
      await operation;
    } finally {
      if (r.micStopPromise === operation) r.micStopPromise = null;
    }
  }, [addMsg]);

  const runVoiceDialogActivation = useCallback(async (
    activation: VoiceDialogActivation,
  ) => {
    const recovery = voiceDialogRecoveryRef.current;
    const failClosed = (message: string) => {
      if (!recovery.activationFailed(activation)) return;
      setVoiceDialogEnabled(false);
      const r = refs.current;
      void cancelActiveMic(r);
      if (r.gw && r.sessionId === activation.sessionId) {
        void r.gw.request("multimodal.voice_dialog_toggle", {
          session_id: activation.sessionId,
          enabled: false,
        }).catch(() => undefined);
      }
      pushTopToast(message, "error");
      addMsg({ id: nid(), role: "system", text: message });
    };

    try {
      // A reconnect/session switch first tombstones the exact old ASR turn. Do
      // not let the replacement turn race that cancellation in the transport.
      const boundary = refs.current.micBoundaryPromise;
      if (boundary) await boundary;
      if (!recovery.owns(activation)) return;

      const r = refs.current;
      const gw = r.gw;
      if (!gw || r.sessionId !== activation.sessionId) {
        failClosed(translateNow("multimodal.errors.voiceDialogUnbound"));
        return;
      }

      const toggleResult = await gw.request<{ ok?: boolean; enabled?: boolean }>(
        "multimodal.voice_dialog_toggle",
        { session_id: activation.sessionId, enabled: true },
      );
      if (toggleResult?.ok === false || toggleResult?.enabled !== true) {
        throw new Error("voice_dialog_toggle_rejected");
      }
      if (!recovery.owns(activation)
        || refs.current.gw !== gw
        || refs.current.sessionId !== activation.sessionId) return;

      const started = await startMic("continuous");
      if (!started) throw new Error("continuous_asr_start_failed");
      if (!recovery.activationSucceeded(activation)) {
        // OFF/boundary won while getUserMedia or backend ASR was starting.
        // The exact turn must not survive that stale activation.
        void cancelActiveMic(refs.current);
      }
    } catch {
      failClosed(translateNow("multimodal.errors.voiceDialogStartFailed"));
    }
  }, [addMsg, pushTopToast, startMic]);
  // Gateway/session callbacks are registered by the mount-once effect above;
  // a ref gives them the latest activation implementation without rebuilding
  // the WebSocket connection or capturing stale React state.
  useEffect(() => {
    runVoiceDialogActivationRef.current = (activation) => {
      void runVoiceDialogActivation(activation);
    };
  }, [runVoiceDialogActivation]);

  // ── Env audio: screen/people speaking → multimodal.env_audio → memory ────
  const startEnvRecorder = useCallback(() => {
    const recordChunk = () => {
      const r = refs.current;
      const envStream = r.envStream;
      if (r.envStop || !envStream || !envStream.getAudioTracks().some((t) => t.readyState === "live")) return;

      const requestedMime = r.envMime || "audio/webm";
      let rec: MediaRecorder;
      try {
        rec = new MediaRecorder(envStream, { mimeType: requestedMime });
      } catch {
        try {
          rec = new MediaRecorder(envStream);
        } catch (e) {
          const key = `recorder:${e instanceof Error ? e.message : String(e)}`;
          if (r.envLastError !== key) {
            r.envLastError = key;
            pushTopToast(translateNow("multimodal.errors.sharedAudioRecordFailed",
              e instanceof Error ? e.message : String(e)), "error");
          }
          return;
        }
      }

      const chunkSeq = ++r.envChunkSeq;
      const captureId = r.envCaptureId;
      const chunkId = `${captureId}:${chunkSeq}`;
      const chunkStartedAt = performance.now();
      const timelineOrigin = r.startTs;
      const gw = r.gw;
      const sessionId = r.sessionId;
      const parts: Blob[] = [];
      let blobTimecode = 0;
      let chunkStoppedAt: number | null = null;
      r.envRecorder = rec;

      rec.ondataavailable = (ev: BlobEvent) => {
        if (ev.data.size > 0) parts.push(ev.data);
        if (Number.isFinite(ev.timecode)) blobTimecode = ev.timecode;
      };
      rec.onstop = () => {
        const rr = refs.current;
        if (rr.envRecorder === rec) rr.envRecorder = null;
        const chunkEndedAt = chunkStoppedAt ?? performance.now();
        const payloadMime = parts[0]?.type || rec.mimeType || requestedMime;
        const blob = parts.length === 1 ? parts[0] : new Blob(parts, { type: payloadMime });
        if (!blob || blob.size < 1000 || !gw || !sessionId) return;

        const clientStartTs = Math.max(0, (chunkStartedAt - timelineOrigin) / 1000);
        const clientEndTs = Math.max(clientStartTs, (chunkEndedAt - timelineOrigin) / 1000);
        const clientDurationSec = Math.max(0, (chunkEndedAt - chunkStartedAt) / 1000);
        void blobToBase64(blob).then((b64) => {
          console.debug("[mm-env-asr-fe] sending complete audio chunk", {
            capture_id: captureId, chunk_id: chunkId, chunk_seq: chunkSeq,
            bytes: blob.size, mime: payloadMime,
            client_start_ts: clientStartTs, client_end_ts: clientEndTs,
            client_duration_sec: clientDurationSec, blob_timecode: blobTimecode,
          });
          return gw.request<{ ingested?: boolean; reason?: string }>("multimodal.env_audio", {
            session_id: sessionId, data_b64: b64, mime: payloadMime,
            // Compatibility field: this is the beginning of the audio window,
            // not the time at which its upload happened.
            window_ts: clientStartTs,
            capture_id: captureId,
            chunk_id: chunkId,
            chunk_seq: chunkSeq,
            client_start_ts: clientStartTs,
            client_end_ts: clientEndTs,
            client_duration_sec: clientDurationSec,
            blob_timecode: blobTimecode,
          });
        }).then((res) => {
          if (res?.ingested !== false) {
            refs.current.envLastError = "";
            return;
          }
          const reason = res.reason || "unknown";
          if (reason === "too_short") return;
          const latest = refs.current;
          if (latest.envLastError !== reason) {
            latest.envLastError = reason;
            pushTopToast(translateNow("multimodal.errors.sharedAudioAsrNotReceived", reason), "error");
          }
        }).catch((e) => {
          const reason = e instanceof Error ? e.message : String(e);
          const latest = refs.current;
          if (latest.envLastError !== reason) {
            latest.envLastError = reason;
            pushTopToast(translateNow("multimodal.errors.sharedAudioAsrFailed", reason), "error");
          }
        });
      };

      try {
        // A MediaRecorder timeslice is a continuation fragment on some browsers
        // (only the first fragment has a WebM/MP4 header), so it is not safe to
        // decode every dataavailable Blob as a standalone ASR file. Stop this
        // recorder and start a fresh one per window: every upload is then a
        // complete, independently decodable media container.
        rec.start();
        const timer = window.setTimeout(() => {
          const latest = refs.current;
          if (latest.envSliceTimer !== timer) return;
          latest.envSliceTimer = null;
          if (latest.envRecorder !== rec || rec.state === "inactive") return;

          chunkStoppedAt = performance.now();
          const shouldRestart = !latest.envStop
            && latest.envStream === envStream
            && envStream.getAudioTracks().some((t) => t.readyState === "live");
          try {
            rec.stop();
          } catch (e) {
            const key = `stop:${e instanceof Error ? e.message : String(e)}`;
            if (latest.envLastError !== key) {
              latest.envLastError = key;
              pushTopToast(translateNow("multimodal.errors.sharedAudioSliceFailed",
                e instanceof Error ? e.message : String(e)), "error");
            }
            return;
          }
          if (latest.envRecorder === rec) latest.envRecorder = null;
          if (shouldRestart) recordChunk();
        }, Math.max(1000, Math.round(r.envWindowSec * 1000)));
        r.envSliceTimer = timer;
      } catch (e) {
        if (r.envRecorder === rec) r.envRecorder = null;
        const key = `start:${e instanceof Error ? e.message : String(e)}`;
        if (r.envLastError !== key) {
          r.envLastError = key;
          pushTopToast(translateNow("multimodal.errors.sharedAudioStartFailed",
            e instanceof Error ? e.message : String(e)), "error");
        }
      }
    };

    recordChunk();
  }, [pushTopToast]);

  const startEnvAudio = useCallback((stream: MediaStream) => {
    const r = refs.current;
    const tracks = stream.getAudioTracks();
    if (tracks.length === 0) return;
    // Defensive restart cleanup: normally this runs once per screen share, but
    // a repeated start must not orphan the previous recorder/timer.
    r.envStop = true;
    if (r.envSliceTimer != null) {
      clearTimeout(r.envSliceTimer);
      r.envSliceTimer = null;
    }
    if (r.envRecorder) {
      try { if (r.envRecorder.state === "recording") r.envRecorder.stop(); } catch { /* noop */ }
      r.envRecorder = null;
    }
    r.envStream = new MediaStream(tracks);
    r.envMime = pickMicMime() || "audio/webm";
    r.envStop = false;
    r.envCaptureId = `cap_${nid()}`;
    r.envChunkSeq = 0;
    r.envLastError = "";
    startEnvRecorder();
  }, [startEnvRecorder]);

  // ── Ask ─────────────────────────────────────────────────────────────────
  // Text is passed in from <ChatComposer> (which owns the input state) rather
  // than read from a parent-level `askText` state — so keystrokes only
  // re-render the composer leaf, never the whole page / message list.
  const startScreen = useCallback(async () => {
    if (refs.current.stream) return;
    if (!navigator.mediaDevices?.getDisplayMedia) {
      addMsg({ id: nid(), role: "system",
        text: translateNow("multimodal.errors.screenNotSupported") });
      return;
    }
    try {
      // ★ frameRate ideal:4 — 后端只 2fps 采样，MediaStream 开得越高 OS 合成器
      //   就要以那个频率刷新捕获管道，直接和鼠标渲染/UI 合成抢 GPU 资源 → 鼠标卡。
      //   4fps 给 2fps 采样留足余量，OS 开销降到 15fps 的 1/4，鼠标立刻流畅。
      // Normal mode uses 1080p for OCR; Mac/HiDPI keeps the original 720p
      // light profile to avoid compositor and base64/JSON serialization stalls.
      const profile = visualCaptureProfile("screen", preferLightCapture());
      const screenVideo = {
        frameRate: { ideal: profile.sourceFrameRate, max: profile.sourceFrameRate },
        width: { ideal: profile.width, max: profile.width },
        height: { ideal: profile.height, max: profile.height },
      };
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: screenVideo, audio: true,
      });
      stream.getVideoTracks().forEach((t) =>
        t.addEventListener("ended", () => stopStream(), { once: true }));
      await attachStream(stream, "screen");
      const audioTracks = stream.getAudioTracks();
      if (audioTracks.length > 0) {
        startEnvAudio(stream);
        pushTopToast(translateNow("multimodal.toasts.sharedAudioConnected", audioTracks.length), "info");
      } else {
        pushTopToast(
          translateNow("multimodal.toasts.sharedAudioNoTracks"),
          "warning",
        );
      }
    } catch (e: unknown) {
      const err = e instanceof Error ? e : new Error(String(e));
      if (err.name !== "NotAllowedError") {
        addMsg({ id: nid(), role: "system",
          text: translateNow("multimodal.errors.screenShareFailed", err.message) });
      }
    }
  }, [addMsg, attachStream, pushTopToast, startEnvAudio, stopStream]);

  const sendAsk = useCallback((raw: string) => {
    const text = raw.trim();
    const r = refs.current;
    if (!text || !r.gw || !r.sessionId) return;
    // Allocate a stable turn id and its assistant slot before sending. This
    // keeps every answer directly below its own query even when Q2/Q3 are
    // accepted while Q1 is still streaming; gateway events route back by the
    // same id instead of all competing for a single "__main__" bubble.
    const clientRequestId = `turn_${nid()}`;
    const answerId = nid();
    curAssistantId.current.set(clientRequestId, answerId);
    r.isAnswering = true;
    // NB: do NOT preset queued=true. If prompt.submit later returns
    // status="queued" we flip it below; otherwise queued stays false so the
    // bubble renders as a normal streaming answer (blinking cursor). Presetting
    // queued=true used to leak a misleading "等待前一条回答完成 ..." message
    // whenever the LLM endpoint was unreachable — message.start never fired
    // (the call was stuck in TCP timeout), so the flag was never cleared and
    // the user saw an imaginary "previous turn is running" state.
    setMessages((prev) => capMsgs([
      ...prev,
      { id: nid(), role: "user", text, requestId: clientRequestId },
      {
        id: answerId,
        role: "assistant",
        text: "",
        streaming: true,
        awaitingFirstDelta: true,
        hasReasoning: false,
        requestId: clientRequestId,
      },
    ]));
    // Per-turn front-end trace. Complements the gateway [mm-trace] lines —
    // together they give you: send from browser → server enter →
    // agent_run_start → LLM SEND → LLM FIRST_BYTE → first delta reaches
    // browser. Any wide gap = the stall's location.
    const traceStart = performance.now();
    // eslint-disable-next-line no-console
    console.log(`%c[mm-trace-fe] +0ms sendAsk ("${text.slice(0, 40)}")`,
      "color:#28a745");
    (window as any).__mmTraceLast = { text, t0: traceStart, seen: {} };
    r.gw.request<{ status?: string; queue_position?: number }>("prompt.submit", {
      session_id: r.sessionId,
      text,
      client_request_id: clientRequestId,
      // Keep the current direct answer intact. The server accepts this message
      // as a distinct FIFO turn while watcher/monitor workers continue in their
      // own loops.
      queue_if_busy: true,
    })
      .then((res) => {
        if (res?.status === "queued" && res.queue_position != null) {
          setMessages((prev) => prev.map((m) => (
            m.id === answerId
              ? { ...m, queued: true, queuePosition: res.queue_position }
              : m
          )));
        }
        // eslint-disable-next-line no-console
        console.log(`%c[mm-trace-fe] +${(performance.now() - traceStart).toFixed(0)}ms prompt_submit_ack`,
          "color:#28a745");
      })
      .catch((e: Error) => {
        // eslint-disable-next-line no-console
        console.log(`%c[mm-trace-fe] +${(performance.now() - traceStart).toFixed(0)}ms prompt_submit_error ${e.message}`,
          "color:#dc3545");
        curAssistantId.current.delete(clientRequestId);
        refs.current.isAnswering = curAssistantId.current.size > 0;
        setMessages((prev) => prev.map((m) => (
          m.id === answerId
            ? { ...m, text: translateNow("multimodal.errors.sendFailed", e.message), streaming: false,
                queued: false, queuePosition: undefined, isError: true }
            : m
        )));
      });
  }, []);

  // Slash commands: a `/`-prefixed composer line runs the SAME gateway pipeline
  // the desktop composer / Ink TUI use — try slash.exec (the full command
  // registry: /help, /model, /compress, /resume, …), then fall back to
  // command.dispatch (exec/plugin/alias/skill/send). Output renders as a system
  // bubble; skill/send directives submit a normal turn via sendAsk. Non-slash
  // text never reaches here (ChatComposer.submit routes it to onSend).
  const runSlash = useCallback((command: string) => {
    const r = refs.current;
    if (!r.gw) return;
    void executeSlash({
      command,
      sessionId: r.sessionId || "",
      gw: r.gw,
      callbacks: {
        sys: (text: string) => addMsg({ id: nid(), role: "system", text }),
        send: (message: string) => sendAsk(message),
        // /undo returns {type:"prefill"}: drop the backed-up user turn into
        // the composer so it can be edited and resubmitted.
        prefill: (message: string) => composerApiRef.current?.setText(message),
      },
    });
  }, [addMsg, sendAsk]);

  // Stop button: interrupt the in-flight turn (session.interrupt aborts the live
  // turn + clears any queued prompt). Optimistically clear streaming flags so the
  // composer flips Stop→Send immediately even if the server's final events race.
  const stopAsk = useCallback(() => {
    const r = refs.current;
    setMessages((prev) => prev.some((m) => m.streaming)
      ? prev.map((m) => (m.streaming
        ? {
            ...m,
            text: m.text || (m.queued ? translateNow("multimodal.errors.cancelled") : m.text),
            streaming: false,
            queued: false,
            queuePosition: undefined,
          }
        : m)) : prev);
    r.isAnswering = false;
    if (!r.gw || !r.sessionId) return;
    r.gw.request("session.interrupt", { session_id: r.sessionId })
      .catch(() => { /* best-effort — turn may have already finished */ });
  }, []);

  // Mic toggle (稳定引用给 <ChatColumn>)。Connecting 可点击取消;
  // recording 的第二击才 finish + 提交; finalizing 期防重复。
  // ★ 对话模式开时麦由对话托管, 单独点麦无效 —— 拦截 + 顶部小提示 (按钮态不变)。
  const onMicToggle = useCallback(() => {
    if (voiceDialogEnabled) {
      pushTopToast(t.multimodal.toasts.dialogModeMicBlocked, "info");
      return;
    }
    if (micState === "finalizing") return;
    if (micState === "connecting") void stopMic("cancel");
    else if (micState === "recording") void stopMic("finish");
    else void startMic("manual_turn");
  }, [micState, stopMic, startMic, voiceDialogEnabled, pushTopToast, t]);

  // ▶ 播放 button on assistant bubbles for text-input turns (voice-input turns)
  // auto-speak in the backend hook, so no button appears there).
  const playAssistantAudio = useCallback((text: string) => {
    const r = refs.current;
    if (!r.gw || !r.sessionId || !text.trim()) return;
    r.gw.request("multimodal.tts_speak", {
      session_id: r.sessionId, text,
    }).catch(() => { /* best-effort; TTS failure never blocks chat */ });
  }, []);

  // Virtuoso item renderer + stable key. Defined AFTER playAssistantAudio so
  // the callback closes over the stable ref (not a TDZ ref). Per-item px-4 pb-3
  // reproduces the old container p-4 + space-y-3 (container spacing utilities
  // don't apply to virtualized rows, which sit in their own positioned wrappers).
  // Answer a generic blocking clarify.request (e.g. set_monitor silent-mode).
  // Freezes the inline bubble to show the chosen answer, then unblocks the
  // waiting tool via clarify.respond. Idempotent: ignores clicks on an
  // already-answered bubble. Declared before renderRow (which lists it as a
  // useCallback dep) to avoid a temporal-dead-zone reference.
  const answerClarify = useCallback((reqId: string, answer: string) => {
    const r = refs.current;
    let already = false;
    setMessages((prev) => prev.map((m) => {
      if (m.kind !== "clarify" || m.clarifyReqId !== reqId) return m;
      if (m.clarifyAnswer !== undefined) { already = true; return m; }
      return { ...m, clarifyAnswer: answer };
    }));
    if (already || !r.gw || !r.sessionId) return;
    r.gw.request("clarify.respond", {
      session_id: r.sessionId, request_id: reqId, answer,
    }).catch(() => { /* best-effort; tool will time out server-side */ });
  }, []);

  // 点击历史深度研究气泡 → 从 analyse 文件读回, 在右侧重建一个【只读】BgItem
  // (分段 + 最终报告), 使 ridIsActive 命中 → visibleDeep 纳入 → 右侧窗口打开。
  const reopenDeepReport = useCallback((rid: string) => {
    const r = refs.current;
    if (!r.gw || !r.sessionId || !rid) return;
    r.gw.request<{
      found?: boolean;
      status?: string;
      query?: string;
      rounds?: { n: number; frame_range?: string; sub_queries?: string[]; findings?: string }[];
      final_report?: string;
    }>("multimodal.get_watcher_report", { session_id: r.sessionId, request_id: rid })
      .then((res) => {
        if (!res || res.found === false) return;
        const segments: BgSegment[] = (res.rounds || []).map((rd) => ({
          seg: rd.n,
          saw: "",
          answer: (rd.findings || "").trim(),
          ready: true,
          lookups: (rd.sub_queries || []).map((q) => ({ kind: "search" as const, query: q, done: true })),
        }));
        const label = watchersRef.current.find((w) => w.watcher_id === rid)?.label
          || res.query || t.multimodal.deepAnalysis.title;
        // done 按文件真实 status 判定, 不硬编码 (否则进行中的任务被点开会错显"已完成")。
        const st = String(res.status || "").toLowerCase();
        const isDone = ["completed", "complete", "done", "stopped"].includes(st);
        const item: BgItem = {
          id: rid, requestId: rid, label, segments, done: isDone, waiting: null,
          finalReport: (res.final_report || "").trim() || undefined,
        };
        setBgItems((prev) => {
          // ★ 若该 rid 已在面板里【实时进行中】(existing.done !== true), 不用只读快照顶替
          //   (否则把"进行中"错显成"已完成"并盖掉实时流)。只置顶已有条目即可。
          const existing = prev.find((b) => (b.requestId || b.id) === rid);
          if (existing && existing.done !== true) {
            return [existing, ...prev.filter((b) => (b.requestId || b.id) !== rid)];
          }
          return [item, ...prev.filter((b) => (b.requestId || b.id) !== rid)].slice(-8);
        });
        setDeepExpanded(rid);   // 强制展开该窗口
        deepExpandedUserPinned.current = true;  // 用户显式重开 → 不再被自动展开顶走
      })
      .catch(() => { /* best-effort */ });
  }, []);

  // ★ 逐条 ▶ 手动播放按钮只在"无自动播报"时显示 (喇叭关 且 对话关)。任一自动播报
  //   开着 → 传 undefined 让 ChatBubble 隐藏 ▶, 防"自动念 + 手动点"双重播放。
  const autoSpeakOn = ttsEnabled || voiceDialogEnabled;
  const renderRow = useCallback((_i: number, row: Row) => {
    // ── bg (工具/状态) 行 ──
    // ★ 卡片【永不隐藏】: tool.start 一到就出卡, 于是用户在工具【开始】调用时就看到
    //   思考块 + 旋转中的工具行, 而不是等它跑完。上方 💭 思考行同时在跑计时器, 两者
    //   分工见 deriveTurnToolPresentation 的注释 —— 这里曾有一套"运行期间隐藏卡片"
    //   的仲裁, 既让工具框迟到一整个工具时长, 又在"工具完成、正文未到"时与思考行的
    //   顶替条件不一致导致两边都不出。删掉仲裁后这两种情况一起消失。
    if (row.type === "bg") {
      return (
        <div className="px-4 pb-3">
          <BgBlock items={row.items} thinking={row.thinking} />
        </div>
      );
    }

    // ── chat 行 ──
    const msg = row.msg;
    const pureThinking = isPureThinkingChat(msg);
    // ── 从相邻 bg 行派生本轮的工具呈现 (见 deriveTurnToolPresentation 的注释):
    //    inToolCall 决定不出空气泡, toolActivity 是 💭 位置的最高优先级 label ——
    //    无运行中工具时为空 → 回落到 reasoning 摘要, 与思维链【交替显示】。
    //    工具是独立 kind:"tool" 消息, 相邻的会合并进同一个 bg row。
    //    注意这【不】影响下方卡片的可见性: 卡片永不隐藏, 这里只决定上方那一行的文字。
    const nextRow = pureThinking ? rows[_i + 1] : undefined;
    const { inToolCall, toolActivity } = deriveTurnToolPresentation(
      nextRow?.type === "bg" ? nextRow.items : undefined,
    );
    // ★ 工具卡一旦出现 (相邻 bg 行含 tool → inToolCall), 上方这条纯思考状态行就整体
    //   不再显示 —— 即使正文气泡还没落地。理由: 工具卡本身已经实时呈现进度 (◌ 旋转 /
    //   ✓ 耗时 / 结果摘要), 再在它上面叠一行 "Thinking…" / "Waiting response…" 既冗余,
    //   又会在"工具已全部跑完、正文未到"的空窗期被误读成"模型正在思考推理"。卡片仍由
    //   它自己的 bg 行渲染, 这里只吞掉这条状态行 (pureThinking 时 ChatBubble 只渲染
    //   ThinkingLine, 所以直接不出这一行即可)。
    if (pureThinking && inToolCall) return null;
    // 紧凑收纳: 纯思考行 (含工具调用态) 只是一行小状态文字, 上下都不需要 12px 大间距,
    // 用 -mt-3 抵消 space-y-3 让它紧贴上一条 UserMessage, 用 pb-0 拿掉底部 padding ——
    // 下一条 (最终气泡) 靠 space-y-3 自己拿间距。
    return (
      <div className={pureThinking ? "-mt-3 px-4 pb-0" : "px-4 pb-3"}>
        {msg.kind === "clarify"
          ? <ClarifyBubble m={msg} onAnswer={answerClarify} />
          : <ChatBubble m={msg} model={model}
              onPlay={autoSpeakOn ? undefined : playAssistantAudio}
              onReopenDeep={reopenDeepReport}
              inToolCall={inToolCall}
              toolActivity={toolActivity} />}
      </div>
    );
  }, [rows, model, playAssistantAudio, answerClarify, reopenDeepReport, autoSpeakOn]);
  const itemKey = useCallback(
    (_i: number, row: Row) => (row.type === "bg" ? row.id : row.msg.id),
    [],
  );

  // Publish the model / source / TTS badges next to the "Multimodal" page title,
  // and the connection status (+ stop-TTS control) into the header's end slot.
  //
  // ★ PERF: this effect updates the App-level PageHeaderProvider state
  // (setAfterTitle/setEnd), whose children include the ENTIRE current page. So
  // anything in this effect's deps that changes frequently re-renders the whole
  // page from the top. `frameCount` updates ~2×/s during screen share → that
  // was re-rendering the full page (video + chat + all panels) twice a second,
  // making the whole UI janky and typing laggy. Do NOT depend on frameCount
  // here — the live frame count is shown in the local video overlay instead.
  useEffect(() => {
    setAfterTitle(
      <div className="flex flex-wrap items-center gap-1.5">
        {sourceType && (
          <Badge tone="secondary">
            {sourceType === "camera" ? t.multimodal.video.camera : t.multimodal.evidence.screen}
          </Badge>
        )}
        {ttsPlaying && (
          <Badge tone="outline" className="gap-1 border-violet-400/60 text-violet-300">
            <Volume2 className="h-3 w-3" /> TTS
          </Badge>
        )}
      </div>,
    );
    setEnd(
      <div className="flex items-center gap-2">
        {ttsPlaying && (
          <Button size="sm" outlined prefix={<Square />} onClick={() => stopAllTts(true)}>
            {t.multimodal.voice.stopSpeaking}
          </Button>
        )}
        <button
          type="button"
          onClick={() => setCliDrawerOpen(true)}
          className="inline-flex h-7 items-center gap-1.5 rounded border border-violet-300/40 bg-background/80 px-2 text-xs text-violet-200 backdrop-blur hover:border-violet-200"
        >
          <Terminal className="h-3.5 w-3.5" />
          CLI
        </button>
        <button
          type="button"
          onClick={() => setMemoryDebugOpen(true)}
          className="inline-flex h-7 items-center gap-1.5 rounded border border-emerald-300/40 bg-background/80 px-2 text-xs text-emerald-200 backdrop-blur hover:border-emerald-200"
        >
          <Database className="h-3.5 w-3.5" />
          Memory
        </button>
        <Badge tone={connected ? "success"
          : connState === "reconnecting" || connState === "connecting" ? "warning"
          : "destructive"}>
          {connected ? t.multimodal.status.connected
            : connState === "reconnecting" ? t.multimodal.status.reconnecting
            : connState === "connecting" ? t.multimodal.status.connecting
            : t.multimodal.status.disconnected}
        </Badge>
      </div>,
    );
    return () => { setAfterTitle(null); setEnd(null); };
  }, [sourceType, ttsPlaying, connected, connState, stopAllTts, setAfterTitle, setEnd, setCliDrawerOpen, setMemoryDebugOpen, t]);

  // ── Render ───────────────────────────────────────────────────────────────
  return (
    <div className="relative flex h-full min-h-0 flex-col gap-3 p-4">
      <MmReadinessBanner report={mmReadiness} />
      {/* Gate on memoryDebugOpen so the lazy chunk is only fetched once the user
          opens the panel. The panel itself also returns null when !open, so this
          only changes WHEN the module loads, not the rendered result. */}
      {memoryDebugOpen && (
        <Suspense fallback={null}>
          <MemoryDebugPanel
            open={memoryDebugOpen}
            onClose={() => setMemoryDebugOpen(false)}
            currentSessionId={refs.current.storedSid || refs.current.sessionId}
            trajectory={trajectory}
          />
        </Suspense>
      )}
      {/* 顶部居中 toast (页面级操作提示, 如未开视频流恢复监控失败), 2s 淡出。
          fixed 贴视口顶部 (不受页面 p-4 内边距下压), 悬浮在最上层。 */}
      {topToasts.length > 0 && (
        <div className="pointer-events-none fixed inset-x-0 top-2 z-[100] flex flex-col items-center gap-1.5">
          {topToasts.map((tt) => (
            <div key={tt.id}
              className={`mm-toast-in pointer-events-auto max-w-[90%] rounded-md border px-3 py-2 text-xs leading-snug shadow-lg backdrop-blur-sm ${
                tt.level === "warning"
                  ? "border-amber-400/40 bg-amber-500/15 text-amber-200"
                  : tt.level === "info"
                    ? "border-border/50 bg-muted/50 text-muted-foreground"
                    : "border-red-400/40 bg-red-500/15 text-red-200"}`}>
              {tt.text}
            </div>
          ))}
        </div>
      )}
      {/* ★ 列宽用 minmax(0,1fr) 而非 1fr: CSS grid 的 1fr 默认 min-width:auto =
          内容宽度, 中/右列一旦有超长不换行内容(长英文标题/表格/代码)就会把列撑破,
          整行溢出、右列冲出容器右边界 (切屏后偶发的"没有右边界")。minmax(0,1fr)
          让列可收缩到 0 以下由内层 overflow/break 兜住, 消除溢出。 */}
      <div className={`grid min-h-0 flex-1 grid-cols-1 gap-3 ${
        showDeepCol ? "lg:grid-cols-[360px_minmax(0,1fr)_minmax(0,1fr)]" : "lg:grid-cols-[360px_minmax(0,1fr)]"}`}>
        {/* LEFT: 视频 + 注入帧 + 画面/音频观察 + 搜索事实。frameCount(1/s)/anchor/ctx
            的 setState 只重渲染这一列, 不再牵动中间聊天列与右侧深研列。 */}
        <LeftPanels
          sourceType={sourceType}
          frameCount={frameCount}
          anchorFrames={anchorFrames}
          ctxVersion={ctx.version}
          obs={ctx.obs}
          audioObs={ctx.audioObs}
          factsList={factsList}
          videoRef={videoRef}
          obsScrollRef={obsScrollRef}
          audioObsScrollRef={audioObsScrollRef}
          onStartCamera={() => void startCamera()}
          onStopStream={stopStream}
          onStartScreen={() => void startScreen()}
        />

        {/* MIDDLE: 聊天列。messages(rows) 变才重渲染; anchor/ctx/frameCount 变时
            rows 引用不变 → 此列 memo 命中、跳过。
            ★ Provider 包在外层而不是往下传 props: chatSessionCtx 身份恒定, 所以
              ChatColumn 的 memo 不会因它失效 (加两个 props 就会)。 */}
        <ChatSessionContext.Provider value={chatSessionCtx}>
        <ChatColumn
          rows={rows}
          renderRow={renderRow}
          itemKey={itemKey}
          atBottom={atBottom}
          chatScrollRef={chatScrollRef}
          onChatScroll={onChatScroll}
          scrollChatToBottom={scrollChatToBottom}
          chatAtBottomRef={chatAtBottomRef}
          isRecordingUI={isRecordingUI}
          asrPartial={asrPartial}
          asrBuffer={asrBuffer}
          micState={micState}
          ttsEnabled={ttsEnabled}
          onTtsToggle={toggleTts}
          generating={generating}
          onStop={stopAsk}
          onSend={sendAsk}
          onSlash={runSlash}
          gw={refs.current.gw}
          onMicToggle={onMicToggle}
          composerApiRef={composerApiRef}
        />
        </ChatSessionContext.Provider>

        {/* RIGHT: 监控/深研注册表 + 深研窗口 + toast。bgItems/visibleDeep/monitors/
            watchers 变才重渲染; 主 agent 纯文本流 (messages 变但不涉 router) 不必然
            触及此列 (deepWindows 依赖 messages, 保持现状——router 气泡本就该更新)。 */}
        <DeepColumn
          showDeepCol={showDeepCol}
          mmToasts={mmToasts}
          monitors={monitors}
          watchers={watchers}
          onToggleMonitor={onToggleMonitor}
          onToggleWatcher={onToggleWatcher}
          visibleDeep={visibleDeep}
          bgByRid={bgByRid}
          deepExpanded={deepExpanded}
          model={model}
          onToggleDeep={toggleDeepWindow}
          monitorAlerts={monitorAlerts}
          monitorCollapsed={monitorCollapsed}
          monitorExpanded={monitorExpanded}
          onToggleMonitorCollapsed={toggleMonitorCollapsed}
          onToggleMonitorExpanded={toggleMonitorExpanded}
        />
      </div>
    </div>
  );
}
