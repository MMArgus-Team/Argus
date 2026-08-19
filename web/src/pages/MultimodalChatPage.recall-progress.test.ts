import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import {
  compactQueryWorkerMessageProgress,
  compactQueryWorkerTrajectory,
  formatTraceTime,
  BgBlock,
  historyToMmMessages,
  isCurrentTrajectoryHydration,
  isQueryMultimodalHistoryToolName,
  isQueryMultimodalToolName,
  mergeQueryWorkerProgress,
  QueryWorkerProgressPanel,
  queryWorkerProgressFromTrajectory,
  updateQueryWorkerProgressCache,
} from "./MultimodalChatPage";

describe("QueryWorker tool rename compatibility", () => {
  it("uses query_multimodal exclusively for new live events", () => {
    expect(isQueryMultimodalToolName("query_multimodal")).toBe(true);
    expect(isQueryMultimodalToolName("recall_multimodal_memory")).toBe(false);
    expect(isQueryMultimodalToolName("get_current_frame")).toBe(false);
  });

  it("hydrates both current and legacy QueryWorker tool cards", () => {
    expect(isQueryMultimodalHistoryToolName("query_multimodal")).toBe(true);
    expect(isQueryMultimodalHistoryToolName("recall_multimodal_memory")).toBe(true);
    expect(isQueryMultimodalHistoryToolName("get_current_frame")).toBe(false);

    const makeResult = (taskId: string) => JSON.stringify({
      reply_owner: "query_worker",
      task_id: taskId,
      recall_trace: [{ phase: "recall_done", round: 1 }],
      findings: `findings-${taskId}`,
    });
    const restored = historyToMmMessages([
      {
        role: "tool",
        name: "query_multimodal",
        content: makeResult("qry_current"),
      },
      {
        role: "tool",
        tool_name: "recall_multimodal_memory",
        content: makeResult("qry_legacy"),
      },
    ]);

    expect(restored).toHaveLength(2);
    expect(restored[0]).toMatchObject({
      kind: "tool",
      toolName: "query_multimodal",
      workerTaskId: "qry_current",
      workerStatus: "running",
      recallFindings: "findings-qry_current",
    });
    expect(restored[0]?.recallTrace).toHaveLength(1);
    expect(restored[1]).toMatchObject({
      kind: "tool",
      toolName: "recall_multimodal_memory",
      workerTaskId: "qry_legacy",
      workerStatus: "running",
      recallFindings: "findings-qry_legacy",
    });
    expect(restored[1]?.recallTrace).toHaveLength(1);
  });

  it("does not attach QueryWorker state to unrelated historical tools", () => {
    const [restored] = historyToMmMessages([{
      role: "tool",
      name: "get_current_frame",
      content: JSON.stringify({
        reply_owner: "query_worker",
        task_id: "qry_wrong_tool",
        findings: "must stay private to the owning tool",
      }),
    }]);

    expect(restored).toMatchObject({
      kind: "tool",
      toolName: "get_current_frame",
    });
    expect(restored?.workerTaskId).toBeUndefined();
    expect(restored?.recallFindings).toBeUndefined();
  });
});

function trajectory(
  phase: string,
  event: Record<string, unknown>,
  extras: Record<string, unknown> = {},
) {
  return {
    id: `event-${phase}`,
    seq: 1,
    ts: 10,
    event: "multimodal.trajectory",
    worker: "QueryWorker",
    phase,
    payload: {
      task_id: "qry_test",
      event,
      ...extras,
    },
  };
}

describe("QueryWorker Recall progress mapping", () => {
  it("renders the exact three frozen ask-time inputs as clickable debug frames", () => {
    const mapped = queryWorkerProgressFromTrajectory(trajectory("started", {}, {
      n_frames: 3,
      ask_ts: 21.5,
      frames: [
        { ts: 19.5, source_type: "screen", jpeg_b64: "frame-one" },
        { ts: 20.5, source_type: "screen", jpeg_b64: "frame-two" },
        { ts: 21.5, source_type: "screen", jpeg_b64: "frame-three" },
      ],
    }));

    expect(mapped?.step.title).toContain("frozen input frames 3");
    expect(mapped?.step.frames).toHaveLength(3);
    expect(mapped?.step.frames?.map((frame) => frame.ts)).toEqual([19.5, 20.5, 21.5]);

    const html = renderToStaticMarkup(createElement(QueryWorkerProgressPanel, {
      taskId: "qry_test",
      status: "running",
      steps: [mapped!.step],
    }));

    expect(html).toContain("Frozen input frames at ask-time (actual thumbnails seen by QueryWorker)");
    expect(html.match(/<img/g)).toHaveLength(3);
    expect(html.match(/target="_blank"/g)).toHaveLength(3);
    expect(html).toContain("data:image/jpeg;base64,frame-one");
    expect(html).toContain("Input frame 3");
    expect(html).toContain("screen");
  });

  it("keeps an empty frozen snapshot explicit without inventing thumbnails", () => {
    const mapped = queryWorkerProgressFromTrajectory(trajectory("started", {}, {
      n_frames: 0,
      ask_ts: 21.5,
      frames: [],
    }));

    expect(mapped?.step.title).toContain("frozen input frames 0");
    expect(mapped?.step.frames).toBeUndefined();
  });

  it("maps and renders the same three bounded OCR records for live and hydrated trajectories", () => {
    const liveEntry = trajectory("ocr_evidence", {}, {
      evidence_state: "available",
      record_count: 3,
      elapsed_sec: 0.317,
      evidence: [
        {
          frame_ts: 19.5,
          source_type: "camera",
          evidence_source: "synchronous_camera_ocr",
          raw_text: "东方树叶\n茉莉花茶",
        },
        {
          frame_ts: 20.5,
          source_type: "screen",
          evidence_source: "background_screen_texts",
          app: "Chrome",
          window_title: "商品页",
          raw_text: "500 mL",
        },
        {
          frame_ts: 21.5,
          source_type: "screen_share",
          evidence_source: "synchronous_screen_fallback",
          raw_text: "售价 5 元",
        },
      ],
    });
    const live = queryWorkerProgressFromTrajectory(liveEntry);
    const hydrated = queryWorkerProgressFromTrajectory(
      JSON.parse(JSON.stringify(liveEntry)),
    );

    expect(live?.step).toEqual(hydrated?.step);
    expect(live?.step.ocrState).toBe("available");
    expect(live?.step.ocrRecords).toHaveLength(3);
    expect(live?.step.ocrElapsedSec).toBe(0.317);

    const html = renderToStaticMarkup(createElement(QueryWorkerProgressPanel, {
      taskId: "qry_test",
      status: "running",
      steps: [live!.step],
    }));
    expect(html).toContain("OCR helper text");
    expect(html).toContain("Camera live OCR");
    expect(html).toContain("Screen live OCR");
    expect(html).toContain("Background OCR cache");
    expect(html).toContain("00:19.5");
    // Backend OCR text stays verbatim — it is captured content, not UI copy.
    expect(html).toContain("东方树叶");
    expect(html).toContain("500 mL");
    expect(html).toContain("售价 5 元");
  });

  it("accepts the records alias and keeps OCR text plain instead of executable HTML", () => {
    const mapped = queryWorkerProgressFromTrajectory(trajectory("ocr_evidence", {}, {
      evidence_state: "available",
      record_count: 1,
      records: [{
        frame_ts: 8,
        source_type: "camera",
        evidence_source: "synchronous_camera_ocr",
        app: "<script>alert('app')</script>",
        raw_text: "<img src=x onerror=alert('ocr')>东方树叶",
      }],
    }));
    const html = renderToStaticMarkup(createElement(QueryWorkerProgressPanel, {
      taskId: "qry_test",
      status: "running",
      steps: [mapped!.step],
    }));

    expect(mapped?.step.ocrRecords).toHaveLength(1);
    expect(html).toContain("&lt;img src=x onerror=alert(&#x27;ocr&#x27;)&gt;东方树叶");
    expect(html).toContain("&lt;script&gt;alert(&#x27;app&#x27;)&lt;/script&gt;");
    expect(html).not.toContain("<img src=x");
    expect(html).not.toContain("<script>");
  });

  it.each([
    ["empty", "", "OCR completed but no usable text was recognized."],
    ["skipped", "ocr_unavailable", "OCR skipped: OCR service currently unavailable."],
    ["error", "deadline_exceeded", "OCR timed out; QueryWorker continued with the original frame"],
  ])("renders a clear %s OCR state", (evidenceState, reason, expected) => {
    const mapped = queryWorkerProgressFromTrajectory(trajectory("ocr_evidence", {}, {
      evidence_state: evidenceState,
      reason,
      record_count: 0,
      evidence: [],
      elapsed_sec: 4.25,
    }));
    const html = renderToStaticMarkup(createElement(QueryWorkerProgressPanel, {
      taskId: "qry_test",
      status: "running",
      steps: [mapped!.step],
    }));

    expect(html).toContain(expected);
    expect(html).toContain("4.25s");
  });

  it("renders structured decision data without exposing raw model thought", () => {
    const mapped = queryWorkerProgressFromTrajectory(trajectory("bg_progress", {
      channel: "recall",
      phase: "r0_decision",
      round: 0,
      can_answer: false,
      n_next_calls: 1,
      n_clues_so_far: 2,
      decision_summary: "证据不足，继续调用 1 个记忆工具",
      useful_info: "已找到探店话题",
      thought: "hidden raw reasoning must not be rendered",
      next_tool_calls: [{
        name: "search_audio",
        args: { query: "探店", top_k: 8 },
      }],
    }));

    expect(mapped?.step.worker).toBe("RecallWorker");
    expect(mapped?.step.title).toContain("continue with 1 more tool");
    // detail is backend-authored (decision_summary / useful_info) — stays verbatim.
    expect(mapped?.step.detail).toContain("证据不足");
    expect(mapped?.step.detail).toContain("已找到探店话题");
    expect(mapped?.step.detail).not.toContain("hidden raw reasoning");
    expect(mapped?.step.plannedTools).toEqual([{
      name: "search_audio",
      args: { query: "探店", top_k: 8 },
    }]);
    expect(mapped?.step.title).toContain("Recall round 1 decision");
    expect(mapped?.step.metrics).toContain("clues so far 2");
  });

  it("renders bounded tool observations and ignores legacy full observations", () => {
    const mapped = queryWorkerProgressFromTrajectory(trajectory("bg_progress", {
      channel: "recall",
      phase: "tool_obs",
      round: 0,
      parallel_elapsed_sec: 0.42,
      observations: [{
        name: "search_audio",
        args: { query: "探店" },
        obs_len: 4200,
        elapsed_sec: 0.18,
        obs_summary: "84.5s 提到联系李维刚来探店",
        obs_full: "legacy full observation must stay out of the UI",
        frame_ids: ["f_84"],
        evidence_segments: [{
          kind: "audio",
          t_start: 84.5,
          t_end: 84.5,
          frame_ids: ["f_84"],
        }],
      }],
    }));

    expect(mapped?.step.toolResults).toEqual([{
      name: "search_audio",
      args: { query: "探店" },
      obs_len: 4200,
      elapsed_sec: 0.18,
      obs_summary: "84.5s 提到联系李维刚来探店",
      frame_ids: ["f_84"],
      evidence_segments: [{
        kind: "audio",
        t_start: 84.5,
        t_end: 84.5,
        frame_ids: ["f_84"],
      }],
    }]);
    expect(JSON.stringify(mapped)).not.toContain("legacy full observation");
  });

  it("shows the exact Search and Recall calls selected by QueryRouter", () => {
    const mapped = queryWorkerProgressFromTrajectory(trajectory("router_react", {
      type: "router_react",
      round: 0,
      source_clip: { t_start: 142, t_end: 148, n_frames: 12 },
      tool_calls: [{
        name: "text_search",
        args: { query: "SCARPA 攀岩鞋品牌" },
        anchor: "current",
      }],
      recall_tasks: [{ brief: "金发男子的攀岩鞋品牌" }],
      thought: "hidden raw reasoning",
    }));

    expect(mapped?.step.plannedTools).toEqual([
      {
        name: "text_search",
        args: { query: "SCARPA 攀岩鞋品牌" },
        anchor: "current",
      },
      {
        name: "recall_memory",
        args: { brief: "金发男子的攀岩鞋品牌" },
      },
    ]);
    expect(mapped?.step.metrics).toContain("Trigger clip 02:22–02:28 · 12 frames");
    expect(JSON.stringify(mapped)).not.toContain("hidden raw reasoning");
  });

  it("explicitly says when QueryRouter made no Recall or Search call", () => {
    const mapped = queryWorkerProgressFromTrajectory(trajectory("router_react", {
      type: "router_react",
      round: 0,
      tool_calls: [],
      recall_tasks: [],
      thought: "hidden raw reasoning",
    }));

    expect(mapped?.step.title).toContain("no Recall / Search this round");
    expect(mapped?.step.plannedTools).toBeUndefined();
    expect(JSON.stringify(mapped)).not.toContain("hidden raw reasoning");
  });

  it("marks dispatches as actual calls and correlates child task ids", () => {
    const mapped = queryWorkerProgressFromTrajectory(trajectory("bg_progress", {
      type: "bg_progress",
      channel: "search",
      task_id: "r0_s1",
      tool_name: "text_search",
      args: { query: "SCARPA 攀岩鞋品牌" },
      anchor: "current",
      anchor_ts: 148,
    }));

    expect(mapped?.step.callState).toBe("called");
    expect(mapped?.step.taskRef).toBe("r0_s1");
    expect(mapped?.step.title).toContain("Search call · text_search");
  });

  it("shows Search parameters, result preview, source URLs and cache state", () => {
    const mapped = queryWorkerProgressFromTrajectory(trajectory("search_done", {
      type: "search_done",
      tool_name: "text_search",
      args: { query: "SCARPA 攀岩鞋品牌" },
      anchor: "current",
      anchor_ts: 148,
      source_clip: { t_start: 142, t_end: 148, n_frames: 12 },
      brief: "原始搜索任务",
      findings_preview: "SCARPA 是意大利户外鞋品牌。",
      findings_len: 18,
      source_urls: ["https://example.test/scarpa"],
      cache_hit: true,
      elapsed_sec: 0.35,
    }));

    expect(mapped?.step.detail).toBe("SCARPA 是意大利户外鞋品牌。");
    expect(mapped?.step.detail).not.toBe("原始搜索任务");
    expect(mapped?.step.toolResults).toEqual([{
      name: "text_search",
      args: { query: "SCARPA 攀岩鞋品牌" },
      obs_len: 18,
      elapsed_sec: 0.35,
      obs_summary: "SCARPA 是意大利户外鞋品牌。",
      source_urls: ["https://example.test/scarpa"],
      cache_hit: true,
      anchor: "current",
      anchor_ts: 148,
    }]);
    expect(mapped?.step.metrics).toContain("cache hit");
    expect(mapped?.step.metrics).toContain("Trigger clip 02:22–02:28 · 12 frames");
  });

  it("formats Recall evidence timestamps for display", () => {
    expect(formatTraceTime(84.5)).toBe("01:24.5");
    expect(formatTraceTime(148)).toBe("02:28");
    expect(formatTraceTime(-1)).toBe("");
  });

  it("shows the fast-table Recall tool and its returned rows", () => {
    const mapped = queryWorkerProgressFromTrajectory(trajectory("bg_progress", {
      channel: "recall",
      task_id: "r0_r0",
      phase: "fast_table",
      tool_name: "search_screen_text",
      args: { query: "表 2 Argus Score", limit: 8 },
      obs_len: 80,
      obs_summary: "[00:42-00:45] Argus | 91.2",
      findings_len: 120,
      findings_preview: "已命中表 2：Argus | 91.2",
      elapsed_sec: 0.12,
      frame_ids: ["f_1234567890"],
      evidence_segments: [{
        kind: "screen",
        t_start: 42,
        t_end: 45,
        frame_ids: ["f_1234567890"],
      }],
    }));

    expect(mapped?.step.title).toContain("search_screen_text");
    expect(mapped?.step.detail).toContain("Argus | 91.2");
    expect(mapped?.step.taskRef).toBe("r0_r0");
    expect(mapped?.step.toolResults?.[0]).toMatchObject({
      name: "search_screen_text",
      args: { query: "表 2 Argus Score", limit: 8 },
      obs_summary: "[00:42-00:45] Argus | 91.2",
      frame_ids: ["f_1234567890"],
    });
    expect(mapped?.step.metrics).toContain("0.12s");
  });

  it("shows request failures as errors instead of memory misses", () => {
    const mapped = queryWorkerProgressFromTrajectory(trajectory("bg_progress", {
      channel: "recall",
      phase: "error",
      stage: "decision",
      model: "GPT-5.6 Luna",
      elapsed_sec: 4.2,
      error: "HTTP 400 Unknown parameter: top_k",
    }));

    expect(mapped?.step.status).toBe("error");
    expect(mapped?.step.title).toContain("Recall request failed");
    expect(mapped?.step.detail).toBe("HTTP 400 Unknown parameter: top_k");
    expect(mapped?.step.metrics).toContain("model GPT-5.6 Luna");
    expect(mapped?.step.terminal).toBeUndefined();
  });

  it("marks only the outer QueryWorker completion as terminal", () => {
    const childDone = queryWorkerProgressFromTrajectory(trajectory("search_done", {
      type: "search_done",
      task_id: "r0_s0",
      tool_name: "text_search",
      findings_preview: "result",
      findings_len: 6,
    }));
    const workerDone = queryWorkerProgressFromTrajectory(trajectory("complete", {}, {
      elapsed_sec: 1.2,
      answer_preview: "final answer",
    }));

    expect(childDone?.step.status).toBe("complete");
    expect(childDone?.step.terminal).toBeUndefined();
    expect(workerDone?.step.status).toBe("complete");
    expect(workerDone?.step.terminal).toBe(true);
  });

  it("uses findings_preview for the final return rather than repeating brief", () => {
    const mapped = queryWorkerProgressFromTrajectory(trajectory("recall_done", {
      brief: "店主准备找谁来探店",
      found: true,
      n_clues: 1,
      findings_preview: "店主打算联系李维刚来探店",
      findings_len: 18,
      rounds: 1,
    }));

    expect(mapped?.step.status).toBe("complete");
    expect(mapped?.step.detail).toBe("店主打算联系李维刚来探店");
    expect(mapped?.step.detail).not.toBe("店主准备找谁来探店");
    expect(mapped?.step.toolResults).toEqual([{
      name: "recall_memory",
      obs_len: 18,
      obs_summary: "店主打算联系李维刚来探店",
    }]);
  });

  it("distinguishes duplicate completion from the retry ceiling", () => {
    const duplicate = queryWorkerProgressFromTrajectory(trajectory("recall_skipped", {
      phase: "recall_skipped",
      reason: "duplicate_completed_brief",
      brief: "视频中店主找谁探店",
    }));
    const exhausted = queryWorkerProgressFromTrajectory(trajectory("recall_skipped", {
      phase: "recall_skipped",
      reason: "retry_limit_after_two_failures",
      brief: "店主找谁探店",
    }));

    expect(duplicate?.step.title).toContain("Skipped duplicate Recall");
    expect(exhausted?.step.title).toContain("failed twice in a row");
  });

  it("keeps a long out-of-order trace sorted, deduplicated, and bounded", () => {
    const steps = Array.from({ length: 90 }, (_, seq) => ({
      id: `step-${seq}`,
      seq,
      ts: seq,
      worker: "RecallWorker",
      phase: "tool_obs",
      title: `step ${seq}`,
    }));
    const merged = mergeQueryWorkerProgress(
      [steps[89], steps[10], steps[50]],
      [...steps.slice().reverse(), { ...steps[89], title: "latest copy" }],
    );

    expect(merged).toHaveLength(80);
    expect(merged[0].seq).toBe(10);
    expect(merged.at(-1)?.seq).toBe(89);
    expect(merged.at(-1)?.title).toBe("latest copy");
  });

  it("renders actual parameters and returned evidence expanded by default", () => {
    const dispatch = queryWorkerProgressFromTrajectory(trajectory("bg_progress", {
      type: "bg_progress",
      channel: "search",
      task_id: "r0_s0",
      tool_name: "text_search",
      args: { query: "SCARPA 攀岩鞋品牌" },
      anchor: "current",
      anchor_ts: 148,
    }));
    const result = queryWorkerProgressFromTrajectory({
      ...trajectory("search_done", {
        type: "search_done",
        task_id: "r0_s0",
        tool_name: "text_search",
        args: { query: "SCARPA 攀岩鞋品牌" },
        findings_preview: "SCARPA 是意大利户外鞋品牌。",
        findings_len: 18,
        source_urls: ["https://example.test/scarpa"],
      }),
      id: "event-search-result",
      seq: 2,
    });
    const html = renderToStaticMarkup(createElement(QueryWorkerProgressPanel, {
      taskId: "qry_test",
      status: "running",
      steps: [dispatch!.step, result!.step],
    }));

    expect(html).toContain("Actual call");
    expect(html).toContain("Tool returned");
    expect(html).toContain("r0_s0");
    expect(html).toContain("SCARPA 是意大利户外鞋品牌。");
    expect(html).toContain("https://example.test/scarpa");
    expect(html.match(/<details open=""/g)?.length).toBeGreaterThanOrEqual(2);
  });
});

describe("QueryWorker debug retention", () => {
  const step = (task: number, image = `image-${task}`) => ({
    id: `started-${task}`,
    seq: task,
    ts: task,
    worker: "QueryWorker",
    phase: "started:started",
    title: `task ${task}`,
    frames: [{ ts: task, source_type: "screen", jpeg_b64: image }],
  });

  it("bounds the task LRU globally while preserving the newest frozen frames", () => {
    let cache = new Map();
    for (let task = 0; task < 55; task += 1) {
      cache = updateQueryWorkerProgressCache(cache, `qry_${task}`, step(task));
    }

    expect(cache.size).toBe(48);
    expect(cache.has("qry_0")).toBe(false);
    expect(cache.has("qry_6")).toBe(false);
    expect(cache.has("qry_7")).toBe(true);
    expect(cache.get("qry_54")?.[0].frames?.[0].jpeg_b64).toBe("image-54");
    // Old steps remain inspectable, but their image bytes are evicted.
    expect(cache.get("qry_7")?.[0].frames?.[0]).toEqual({
      ts: 7,
      source_type: "screen",
    });
  });

  it("keeps only recent trajectory image bytes and retains old frame metadata", () => {
    const entries = Array.from({ length: 6 }, (_, task) => ({
      id: `tr-${task}`,
      seq: task,
      ts: task,
      event: "multimodal.trajectory",
      worker: "QueryWorker",
      phase: "started",
      payload: {
        task_id: `qry_${task}`,
        n_frames: 1,
        frames: [{
          ts: task,
          source_type: "camera",
          jpeg_b64: `image-${task}`,
        }],
      },
    }));

    const compacted = compactQueryWorkerTrajectory(entries);
    expect(compacted[0].payload.frames).toEqual([{ ts: 0, source_type: "camera" }]);
    expect(compacted[1].payload.frames).toEqual([{ ts: 1, source_type: "camera" }]);
    expect(compacted.at(-1)?.payload.frames).toEqual([{
      ts: 5,
      source_type: "camera",
      jpeg_b64: "image-5",
    }]);
  });

  it("evicts image bytes from old message cards without deleting their steps", () => {
    const old = step(1);
    const current = step(2);
    const cache = new Map([["qry_2", [current]]]);
    const compacted = compactQueryWorkerMessageProgress([
      { id: "old", role: "assistant", text: "", workerTaskId: "qry_1", workerProgress: [old] },
      { id: "new", role: "assistant", text: "", workerTaskId: "qry_2", workerProgress: [current] },
    ], cache);

    expect(compacted[0].workerProgress?.[0].frames?.[0]).toEqual({
      ts: 1,
      source_type: "screen",
    });
    expect(compacted[0].workerProgress?.[0].title).toBe("task 1");
    expect(compacted[1].workerProgress?.[0].frames?.[0].jpeg_b64).toBe("image-2");
  });

  it("rejects trajectory hydrate responses after a session or generation change", () => {
    expect(isCurrentTrajectoryHydration("live-a", 4, "live-a", 4)).toBe(true);
    expect(isCurrentTrajectoryHydration("live-a", 4, "live-b", 4)).toBe(false);
    expect(isCurrentTrajectoryHydration("live-a", 4, "live-a", 5)).toBe(false);
  });
});

describe("history tool card: call preview vs result body", () => {
  it("keeps context as the arg preview and never promotes it to the detail body", () => {
    // Regression: history rebuild used `content || context`, so a tool row with
    // no stored output put the 80-char command preview into toolDetail. That
    // made the summary line collapse to a bare "terminal" AND spawned an empty
    // disclosure whose only content was that same command.
    const [card] = historyToMmMessages([
      { role: "tool", name: "terminal", context: "curl -L --max-time 15 wttr.in" },
    ]);

    expect(card.toolName).toBe("terminal");
    expect(card.toolCtx).toBe("curl -L --max-time 15 wttr.in");
    expect(card.toolDetail).toBeUndefined();
  });

  it("carries real tool output and summary through as distinct fields", () => {
    const [card] = historyToMmMessages([
      {
        role: "tool",
        name: "terminal",
        context: "curl wttr.in",
        content: "curl: (28) Operation timed out",
        summary: "exit 28",
      },
    ]);

    expect(card.toolCtx).toBe("curl wttr.in");
    expect(card.toolDetail).toBe("curl: (28) Operation timed out");
    expect(card.toolSummary).toBe("exit 28");
  });
});

describe("tool card disclosure depth", () => {
  const html = (items: Parameters<typeof BgBlock>[0]["items"]) =>
    renderToStaticMarkup(createElement(BgBlock, { items }));

  it("renders output one click deep, with no nested Raw-tool-result layer", () => {
    const markup = html([{
      id: "t1", role: "assistant", text: "", kind: "tool",
      toolName: "terminal", toolDone: true,
      toolCtx: "curl wttr.in",
      toolDetail: "curl: (28) Operation timed out",
    }]);

    // Exactly one <details>: the user opens the tool row and sees the output.
    expect(markup.match(/<details/g)?.length).toBe(1);
    expect(markup).not.toContain("Raw tool result");
    expect(markup).toContain("curl: (28) Operation timed out");
  });

  it("renders a plain row (no disclosure) when a tool stored no output", () => {
    const markup = html([{
      id: "t2", role: "assistant", text: "", kind: "tool",
      toolName: "terminal", toolDone: true, toolCtx: "echo hi",
    }]);

    expect(markup).not.toContain("<details");
    expect(markup).toContain("terminal");
    expect(markup).toContain("echo hi");
  });
});

describe("tool call argument visibility", () => {
  const html = (items: Parameters<typeof BgBlock>[0]["items"]) =>
    renderToStaticMarkup(createElement(BgBlock, { items }));

  it("opens a disclosure for a tool that has args but no output yet", () => {
    // The gap this closes: `computer_use` previewed as a bare tool name with
    // nothing expandable, so the user could see a skill ran but never what it
    // was told to do. Args alone must be enough to make the row openable.
    const markup = html([{
      id: "t1", role: "assistant", text: "", kind: "tool",
      toolName: "computer_use", toolDone: true,
      toolCtx: "capture som · Google Chrome",
      toolArgs: [
        { key: "action", kind: "literal", value: "capture" },
        { key: "mode", kind: "literal", value: "som" },
        { key: "app", kind: "literal", value: "Google Chrome" },
      ],
    }]);

    expect(markup).toContain("<details");
    expect(markup).toContain("action");
    expect(markup).toContain("capture");
    expect(markup).toContain("Google Chrome");
  });

  it("shows a payload field's length and never its content", () => {
    // Privacy boundary. The backend sends `chars` with no value for payload
    // fields, so a DM body cannot reach the DOM even if this panel tried.
    const markup = html([{
      id: "t2", role: "assistant", text: "", kind: "tool",
      toolName: "yb_send_dm", toolDone: true,
      toolArgs: [
        { key: "group_code", kind: "literal", value: "G9" },
        { key: "message", kind: "freeform", chars: 22 },
      ],
    }]);

    expect(markup).toContain("G9");
    expect(markup).toContain("22 chars (not shown)");
    expect(markup).not.toContain("hunter2");
  });

  it("withholds a credential field entirely, including its length", () => {
    // Stricter than `freeform`: a password's length is itself a clue, so the
    // backend sends the key alone — no value, no `chars`. Nothing to render.
    const markup = html([{
      id: "t2b", role: "assistant", text: "", kind: "tool",
      toolName: "login", toolDone: true,
      toolArgs: [
        { key: "user", kind: "literal", value: "alice" },
        { key: "password", kind: "credential" },
      ],
    }]);

    expect(markup).toContain("password");
    expect(markup).toContain("Redacted (credentials)");
    expect(markup).not.toContain("hunter2");
    // No length leak: the credential row must not borrow the `chars` wording.
    expect(markup).not.toContain("chars (not shown)");
  });

  it("renders the elided tail as a bare count with no key", () => {
    const markup = html([{
      id: "t2c", role: "assistant", text: "", kind: "tool",
      toolName: "mcp_call", toolDone: true,
      toolArgs: [
        { key: "name", kind: "literal", value: "search" },
        { key: "", kind: "elided", count: 4 },
      ],
    }]);

    expect(markup).toContain("4 more fields (not shown)");
  });

  it("renders array/object args as a count", () => {
    const markup = html([{
      id: "t3", role: "assistant", text: "", kind: "tool",
      toolName: "web_extract", toolDone: true,
      toolArgs: [{ key: "urls", kind: "shape", count: 3 }],
    }]);

    expect(markup).toContain("urls");
    expect(markup).toContain("3 items");
  });

  it("still renders a plain row when a tool has neither args nor output", () => {
    const markup = html([{
      id: "t4", role: "assistant", text: "", kind: "tool",
      toolName: "list_apps", toolDone: true,
    }]);

    expect(markup).not.toContain("<details");
    expect(markup).toContain("list_apps");
  });

  it("carries args_fields from history so reopened sessions expand too", () => {
    const [card] = historyToMmMessages([
      {
        role: "tool",
        name: "computer_use",
        context: "click #7 · Google Chrome",
        args_fields: [
          { key: "action", kind: "literal", value: "click" },
          { key: "element", kind: "literal", value: "7" },
        ],
      },
    ]);

    expect(card.toolArgs).toHaveLength(2);
    expect(card.toolArgs?.[0]).toEqual({ key: "action", kind: "literal", value: "click" });
  });
});
