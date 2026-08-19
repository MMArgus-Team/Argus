import { describe, expect, it, vi } from "vitest";

import {
  ASR_CONTROL_TIMEOUT_MS,
  ASR_PRE_ROLL_SECONDS,
  AsrTurnTransport,
  asrFinishFailureMessage,
  decodedBase64Bytes,
  ownsAsrStopUi,
  type AsrRpcClient,
} from "./asr-turn-transport";

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
}

function rpcWithStartGate() {
  const start = deferred<Record<string, unknown>>();
  const calls: {
    method: string;
    params: Record<string, unknown>;
    timeoutMs?: number;
  }[] = [];
  const rpc: AsrRpcClient = {
    request: vi.fn(async (
      method: string,
      params: Record<string, unknown>,
      timeoutMs?: number,
    ) => {
      calls.push({ method, params, timeoutMs });
      if (method === "multimodal.asr_start") return start.promise;
      return { ok: true };
    }),
  };
  return { rpc, calls, start };
}

describe("AsrTurnTransport", () => {
  it("keeps a rejected A finish from mutating B after a session boundary", async () => {
    const finish = deferred<never>();
    const stopGeneration = 10;
    let currentGeneration = stopGeneration;
    let currentSessionId = "session-a";
    let micState = "finalizing";
    let partial = "A partial";
    let buffer = ["A segment"];
    const errors: string[] = [];
    const stillOwnsA = () => ownsAsrStopUi(
      currentGeneration,
      stopGeneration,
      currentSessionId,
      "session-a",
    );

    const pendingA = finish.promise.catch((error: Error) => {
      if (stillOwnsA()) errors.push(error.message);
    }).finally(() => {
      if (!stillOwnsA()) return;
      micState = "idle";
      partial = "";
      buffer = [];
    });

    // New/session switch invalidates A, then B owns the live UI.
    currentGeneration += 1;
    currentSessionId = "session-b";
    micState = "recording";
    partial = "B partial";
    buffer = ["B segment"];
    finish.reject(new Error("old A finish failed"));
    await pendingA;

    expect(errors).toEqual([]);
    expect({ micState, partial, buffer }).toEqual({
      micState: "recording",
      partial: "B partial",
      buffer: ["B segment"],
    });
  });

  it("captures locally first and flushes pre-roll to the backend in order", async () => {
    const { rpc, calls, start } = rpcWithStartGate();
    const transport = new AsrTurnTransport(rpc);
    const turn = transport.begin("session-a", "manual_turn", "turn-a");

    expect(transport.pushPcm("session-a", "turn-a", "YQ==")).toBe(true);
    expect(transport.pushPcm("session-a", "turn-a", "YmI=")).toBe(true);
    expect(calls.map((call) => call.method)).toEqual(["multimodal.asr_start"]);

    start.resolve({ enabled: true, turn_id: "turn-a" });
    await turn.ready;
    await transport.stop("session-a", "turn-a", "finish");

    expect(calls.map((call) => call.method)).toEqual([
      "multimodal.asr_start",
      "multimodal.asr_audio",
      "multimodal.asr_audio",
      "multimodal.asr_stop",
    ]);
    expect(calls.slice(1, 3).map((call) => call.params.pcm_b64)).toEqual(["YQ==", "YmI="]);
    expect(calls.at(-1)?.params).toMatchObject({
      session_id: "session-a",
      turn_id: "turn-a",
      disposition: "finish",
    });
    expect(calls[0]?.timeoutMs).toBe(ASR_CONTROL_TIMEOUT_MS);
    expect(calls.slice(1, 3).every((call) => call.timeoutMs === undefined)).toBe(true);
    expect(calls.at(-1)?.timeoutMs).toBe(ASR_CONTROL_TIMEOUT_MS);
  });

  it("dispatches every ready audio packet without serializing on RPC ACKs", async () => {
    const audioGates = [deferred<unknown>(), deferred<unknown>()];
    const calls: { method: string; params: Record<string, unknown> }[] = [];
    let audioIndex = 0;
    const rpc: AsrRpcClient = { request: vi.fn(async (method, params) => {
      calls.push({ method, params });
      if (method === "multimodal.asr_start") return { enabled: true, turn_id: "turn-fast" };
      if (method === "multimodal.asr_audio") return audioGates[audioIndex++].promise;
      return { ok: true };
    }) };
    const transport = new AsrTurnTransport(rpc);
    const turn = transport.begin("session-a", "manual_turn", "turn-fast");
    await turn.ready;

    transport.pushPcm("session-a", "turn-fast", "YQ==");
    transport.pushPcm("session-a", "turn-fast", "Yg==");
    const stopping = transport.stop("session-a", "turn-fast", "finish");

    expect(calls.filter((call) => call.method === "multimodal.asr_audio")).toHaveLength(2);
    expect(calls.filter((call) => call.method === "multimodal.asr_stop")).toHaveLength(0);
    audioGates[0].resolve({ ok: true });
    audioGates[1].resolve({ ok: true });
    await stopping;
    expect(calls.at(-1)?.method).toBe("multimodal.asr_stop");
  });

  it("cancels instead of submitting a manual turn with a failed PCM upload", async () => {
    const calls: { method: string; params: Record<string, unknown> }[] = [];
    const rpc: AsrRpcClient = { request: vi.fn(async (method, params) => {
      calls.push({ method, params });
      if (method === "multimodal.asr_start") {
        return { enabled: true, turn_id: "turn-upload-failed" };
      }
      if (method === "multimodal.asr_audio") {
        return { ok: false, reason: "stale_transport" };
      }
      return { ok: true };
    }) };
    const transport = new AsrTurnTransport(rpc);
    const turn = transport.begin(
      "session-a",
      "manual_turn",
      "turn-upload-failed",
    );
    await turn.ready;
    transport.pushPcm("session-a", "turn-upload-failed", "YXVkaW8=");

    const result = await transport.stop(
      "session-a",
      "turn-upload-failed",
      "finish",
    );

    expect(result).toMatchObject({
      ok: false,
      submitted: false,
      reason: "audio_delivery_failed",
    });
    expect(calls.filter((call) => call.method === "multimodal.asr_stop")
      .map((call) => call.params.disposition)).toEqual(["cancel"]);
    expect(asrFinishFailureMessage(result)).toContain("Audio upload was interrupted");
  });

  it("finishes an empty manual recording exactly once", async () => {
    const calls: { method: string; params: Record<string, unknown> }[] = [];
    const rpc: AsrRpcClient = { request: vi.fn(async (method, params) => {
      calls.push({ method, params });
      return method === "multimodal.asr_start" ? { enabled: true } : { submitted: false };
    }) };
    const transport = new AsrTurnTransport(rpc);
    const turn = transport.begin("session-a", "manual_turn", "turn-empty");
    await turn.ready;

    const first = transport.stop("session-a", "turn-empty", "finish");
    const second = transport.stop("session-a", "turn-empty", "finish");
    expect(first).toBe(second);
    await first;
    expect(calls.filter((call) => call.method === "multimodal.asr_stop")).toHaveLength(1);
  });

  it("lets an exact cancel overtake an in-flight finish", async () => {
    const finishGate = deferred<unknown>();
    const calls: { method: string; params: Record<string, unknown> }[] = [];
    const rpc: AsrRpcClient = { request: vi.fn(async (method, params) => {
      calls.push({ method, params });
      if (method === "multimodal.asr_start") {
        return { enabled: true, turn_id: "turn-boundary" };
      }
      if (method === "multimodal.asr_stop" && params.disposition === "finish") {
        return finishGate.promise;
      }
      return { ok: true, reason: "cancelled" };
    }) };
    const transport = new AsrTurnTransport(rpc);
    const turn = transport.begin("session-a", "manual_turn", "turn-boundary");
    await turn.ready;

    const finishing = transport.stop("session-a", "turn-boundary", "finish");
    await vi.waitFor(() => {
      expect(calls.some((call) => call.method === "multimodal.asr_stop"
        && call.params.disposition === "finish")).toBe(true);
    });
    const cancelling = transport.stop("session-a", "turn-boundary", "cancel");

    expect(cancelling).not.toBe(finishing);
    expect(calls.filter((call) => call.method === "multimodal.asr_stop")
      .map((call) => call.params.disposition)).toEqual(["finish", "cancel"]);
    await cancelling;
    finishGate.resolve({ ok: false, reason: "cancelled" });
    await finishing;
  });

  it("passes a frozen visual anchor only on manual finish", async () => {
    const calls: { method: string; params: Record<string, unknown> }[] = [];
    const rpc: AsrRpcClient = { request: vi.fn(async (method, params) => {
      calls.push({ method, params });
      return method === "multimodal.asr_start"
        ? { enabled: true, turn_id: "turn-anchor" }
        : { ok: true, submitted: true };
    }) };
    const transport = new AsrTurnTransport(rpc);
    const turn = transport.begin("session-a", "manual_turn", "turn-anchor");
    await turn.ready;

    await transport.stop("session-a", "turn-anchor", "finish", {
      anchor_ts: 12.5,
      capture_attempt_id: "capture-a",
    });

    expect(calls.at(-1)?.params).toMatchObject({
      session_id: "session-a",
      turn_id: "turn-anchor",
      disposition: "finish",
      anchor_ts: 12.5,
      capture_attempt_id: "capture-a",
    });
  });

  it("turns empty and upstream failures into actionable diagnostics", () => {
    expect(asrFinishFailureMessage({ ok: true, submitted: false, reason: "empty" }))
      .toContain("No speech recognized");
    expect(asrFinishFailureMessage({
      ok: false,
      submitted: false,
      reason: "upstream_error",
      error: "provider unavailable",
    })).toContain("provider unavailable");
    expect(asrFinishFailureMessage({ ok: true, submitted: true })).toBe("");
  });

  it("cancels a connecting turn after a late start without uploading pre-roll", async () => {
    const { rpc, calls, start } = rpcWithStartGate();
    const transport = new AsrTurnTransport(rpc);
    transport.begin("session-a", "manual_turn", "turn-cancel");
    transport.pushPcm("session-a", "turn-cancel", "YXVkaW8=");

    const stopped = transport.stop("session-a", "turn-cancel", "cancel");
    expect(calls.filter((call) => call.method === "multimodal.asr_stop")).toHaveLength(1);
    expect(calls.filter((call) => call.method === "multimodal.asr_audio")).toHaveLength(0);
    start.resolve({ enabled: true });
    await stopped;
    await Promise.resolve();

    expect(calls.filter((call) => call.method === "multimodal.asr_audio")).toHaveLength(0);
    expect(calls.filter((call) => call.method === "multimodal.asr_stop")).toEqual([{
      method: "multimodal.asr_stop",
      params: { session_id: "session-a", turn_id: "turn-cancel", disposition: "cancel" },
      timeoutMs: undefined,
    }]);
    expect(transport.current()).toBeNull();
    expect(transport.pushPcm("session-a", "turn-cancel", "bGF0ZQ==")).toBe(false);
    expect(transport.ownsEvent("session-a", "turn-cancel")).toBe(false);
  });

  it("fails closed for missing, old-session, and old-turn events", async () => {
    const rpc: AsrRpcClient = { request: vi.fn(async () => ({ enabled: true })) };
    const transport = new AsrTurnTransport(rpc);
    const turn = transport.begin("session-new", "manual_turn", "turn-new");
    await turn.ready;

    expect(transport.ownsEvent("session-new", undefined)).toBe(false);
    expect(transport.ownsEvent(undefined, "turn-new")).toBe(false);
    expect(transport.ownsEvent("session-old", "turn-new")).toBe(false);
    expect(transport.ownsEvent("session-new", "turn-old")).toBe(false);
    expect(transport.ownsEvent("session-new", "turn-new")).toBe(true);
    transport.noteFinal("session-new", "turn-new");
    expect(transport.ownsEvent("session-new", "turn-new")).toBe(false);
  });

  it("keeps continuous mode event ownership across segment finals until cancel", async () => {
    const rpc: AsrRpcClient = { request: vi.fn(async () => ({ enabled: true })) };
    const transport = new AsrTurnTransport(rpc);
    const turn = transport.begin("session-a", "continuous", "turn-dialog");
    await turn.ready;

    transport.noteFinal("session-a", "turn-dialog");
    expect(transport.ownsEvent("session-a", "turn-dialog")).toBe(true);
    await transport.stop("session-a", "turn-dialog", "cancel");
    expect(transport.ownsEvent("session-a", "turn-dialog")).toBe(false);
  });

  it("keeps its rolling pre-roll bounded", async () => {
    const { rpc, calls, start } = rpcWithStartGate();
    const transport = new AsrTurnTransport(rpc, 4);
    transport.begin("session-a", "manual_turn", "turn-bounded");

    expect(decodedBase64Bytes("YQ==")).toBe(1);
    expect(ASR_PRE_ROLL_SECONDS).toBeGreaterThan(210);
    for (const pcm of ["YQ==", "Yg==", "Yw==", "ZA==", "ZQ=="]) {
      expect(transport.pushPcm("session-a", "turn-bounded", pcm)).toBe(true);
    }
    // Stale-turn audio is rejected rather than entering another turn's queue.
    expect(transport.pushPcm("session-a", "turn-old", "Zg==")).toBe(false);
    start.resolve({ enabled: true });
    await transport.stop("session-a", "turn-bounded", "finish");
    expect(calls.filter((call) => call.method === "multimodal.asr_audio")
      .map((call) => call.params.pcm_b64)).toEqual(["Yg==", "Yw==", "ZA==", "ZQ=="]);
  });

  it("rejects missing enablement and a mismatched start turn id", async () => {
    for (const response of [{}, { enabled: true, turn_id: "wrong" }]) {
      const rpc: AsrRpcClient = { request: vi.fn(async () => response) };
      const transport = new AsrTurnTransport(rpc);
      const turn = transport.begin("session-a", "manual_turn", "turn-right");
      await expect(turn.ready).rejects.toThrow();
      await transport.stop("session-a", "turn-right", "cancel");
    }
  });
});
