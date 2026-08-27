import { describe, expect, it, vi } from "vitest";

import type { GatewayClient } from "@/lib/gatewayClient";
import {
  executeSlash,
  parseCommandDispatch,
  parseSlash,
  rpcErrorMessage,
} from "./slashExec";

function gw(request: ReturnType<typeof vi.fn>): GatewayClient {
  return { request } as unknown as GatewayClient;
}

function callbacks() {
  return {
    sys: vi.fn(),
    send: vi.fn(),
    prefill: vi.fn(),
  };
}

describe("parseSlash", () => {
  it("splits name and arg", () => {
    expect(parseSlash("/model opus-4.6")).toEqual({
      name: "model",
      arg: "opus-4.6",
    });
    expect(parseSlash("/help")).toEqual({ name: "help", arg: "" });
    expect(parseSlash("//undo 3")).toEqual({ name: "undo", arg: "3" });
    expect(parseSlash("")).toEqual({ name: "", arg: "" });
  });
});

describe("parseCommandDispatch", () => {
  it("parses exec and plugin directives", () => {
    expect(parseCommandDispatch({ type: "exec", output: "hi" })).toEqual({
      type: "exec",
      output: "hi",
    });
    expect(parseCommandDispatch({ type: "plugin", output: "p" })).toEqual({
      type: "plugin",
      output: "p",
    });
    expect(parseCommandDispatch({ type: "exec" })).toEqual({
      type: "exec",
      output: undefined,
    });
  });

  it("parses alias, skill, send, and prefill", () => {
    expect(parseCommandDispatch({ type: "alias", target: "help" })).toEqual({
      type: "alias",
      target: "help",
    });
    expect(
      parseCommandDispatch({ type: "skill", name: "x", message: "do" }),
    ).toEqual({ type: "skill", name: "x", message: "do" });
    expect(
      parseCommandDispatch({ type: "send", message: "hello world" }),
    ).toEqual({ type: "send", message: "hello world" });
    expect(
      parseCommandDispatch({
        type: "send",
        message: "hello world",
        notice: "⊙ Goal set",
      }),
    ).toEqual({
      type: "send",
      message: "hello world",
      notice: "⊙ Goal set",
    });
    expect(
      parseCommandDispatch({ type: "prefill", message: "edit me" }),
    ).toEqual({ type: "prefill", message: "edit me" });
    expect(
      parseCommandDispatch({
        type: "prefill",
        message: "edit me",
        notice: "↶ Undid 1 turn",
      }),
    ).toEqual({
      type: "prefill",
      message: "edit me",
      notice: "↶ Undid 1 turn",
    });
  });

  it("rejects malformed payloads", () => {
    expect(parseCommandDispatch(null)).toBeNull();
    expect(parseCommandDispatch("output")).toBeNull();
    expect(parseCommandDispatch([])).toBeNull();
    expect(parseCommandDispatch({})).toBeNull();
    expect(parseCommandDispatch({ type: "alias" })).toBeNull();
    expect(parseCommandDispatch({ type: "skill", name: 1 })).toBeNull();
    expect(parseCommandDispatch({ type: "send" })).toBeNull();
    expect(parseCommandDispatch({ type: "send", message: 42 })).toBeNull();
    expect(parseCommandDispatch({ type: "prefill" })).toBeNull();
    expect(parseCommandDispatch({ type: "prefill", message: 42 })).toBeNull();
    expect(parseCommandDispatch({ type: "bogus" })).toBeNull();
  });
});

describe("rpcErrorMessage", () => {
  it("unwraps Error, string, and unknown rejections", () => {
    expect(rpcErrorMessage(new Error("boom"))).toBe("boom");
    expect(rpcErrorMessage("broken")).toBe("broken");
    expect(rpcErrorMessage("  ")).toBe("request failed");
    expect(rpcErrorMessage({ code: 500 })).toBe("request failed");
  });
});

describe("executeSlash", () => {
  it("renders plain slash.exec output as a system line", async () => {
    const g = gw(vi.fn().mockResolvedValue({ output: "ok" }));
    const cb = callbacks();
    const result = await executeSlash({
      command: "/help",
      sessionId: "s1",
      gw: g,
      callbacks: cb,
    });
    expect(result).toBe("done");
    expect(cb.sys).toHaveBeenCalledWith("ok");
    expect(cb.send).not.toHaveBeenCalled();
  });

  it("renders slash.exec warning + output", async () => {
    const g = gw(
      vi.fn().mockResolvedValue({ output: "body", warning: "careful" }),
    );
    const cb = callbacks();
    await executeSlash({ command: "/model", sessionId: "s1", gw: g, callbacks: cb });
    expect(cb.sys).toHaveBeenCalledWith("warning: careful\nbody");
  });

  it("falls back to /name: no output when slash.exec returns nothing", async () => {
    const g = gw(vi.fn().mockResolvedValue({}));
    const cb = callbacks();
    const result = await executeSlash({
      command: "/mystery",
      sessionId: "s1",
      gw: g,
      callbacks: cb,
    });
    expect(result).toBe("done");
    expect(cb.sys).toHaveBeenCalledWith("/mystery: no output");
  });

  it("parses a structured directive returned straight from slash.exec (/goal)", async () => {
    const g = gw(
      vi.fn().mockResolvedValue({
        type: "send",
        notice: "⊙ Goal set (20-turn budget)",
        message: "build a parser",
      }),
    );
    const cb = callbacks();
    const result = await executeSlash({
      command: "/goal build a parser",
      sessionId: "s1",
      gw: g,
      callbacks: cb,
    });
    expect(result).toBe("sent");
    expect(cb.sys).toHaveBeenCalledWith("⊙ Goal set (20-turn budget)");
    expect(cb.send).toHaveBeenCalledWith("build a parser");
  });

  it("handles /undo prefill from slash.exec: notice + composer prefill", async () => {
    const g = gw(
      vi.fn().mockResolvedValue({
        type: "prefill",
        notice: "↶ Undid 1 turn (3 message(s)).",
        message: "my earlier question",
      }),
    );
    const cb = callbacks();
    const result = await executeSlash({
      command: "/undo",
      sessionId: "s1",
      gw: g,
      callbacks: cb,
    });
    expect(result).toBe("done");
    expect(cb.sys).toHaveBeenCalledWith("↶ Undid 1 turn (3 message(s)).");
    expect(cb.prefill).toHaveBeenCalledWith("my earlier question");
    expect(cb.send).not.toHaveBeenCalled();
  });

  it("falls back to a sys line when no prefill callback is wired", async () => {
    const g = gw(
      vi.fn().mockResolvedValue({ type: "prefill", message: "edit me" }),
    );
    const cb = callbacks();
    delete cb.prefill;
    const result = await executeSlash({
      command: "/undo",
      sessionId: "s1",
      gw: g,
      callbacks: cb,
    });
    expect(result).toBe("done");
    expect(cb.sys).toHaveBeenCalledWith("/undo: edit me");
  });

  it("falls back to command.dispatch when slash.exec rejects", async () => {
    const request = vi
      .fn()
      .mockRejectedValueOnce(new Error("slash worker failed"))
      .mockResolvedValueOnce({ type: "send", message: "queued prompt" });
    const cb = callbacks();
    const result = await executeSlash({
      command: "/q queued prompt",
      sessionId: "s1",
      gw: gw(request),
      callbacks: cb,
    });
    expect(request).toHaveBeenNthCalledWith(
      1,
      "slash.exec",
      expect.objectContaining({ command: "q queued prompt" }),
    );
    expect(request).toHaveBeenNthCalledWith(
      2,
      "command.dispatch",
      expect.objectContaining({ name: "q", arg: "queued prompt" }),
    );
    expect(result).toBe("sent");
    expect(cb.send).toHaveBeenCalledWith("queued prompt");
  });

  it("recurses through alias directives with the original arg tail", async () => {
    const request = vi
      .fn()
      .mockRejectedValueOnce(new Error("nope"))
      .mockResolvedValueOnce({ type: "alias", target: "help" })
      .mockResolvedValueOnce({ output: "help text" });
    const cb = callbacks();
    const result = await executeSlash({
      command: "/h something",
      sessionId: "s1",
      gw: gw(request),
      callbacks: cb,
    });
    expect(request).toHaveBeenNthCalledWith(
      3,
      "slash.exec",
      expect.objectContaining({ command: "help something" }),
    );
    expect(result).toBe("done");
    expect(cb.sys).toHaveBeenCalledWith("help text");
  });

  it("handles skill directives from command.dispatch", async () => {
    const request = vi
      .fn()
      .mockRejectedValueOnce(new Error("nope"))
      .mockResolvedValueOnce({
        type: "skill",
        name: "computer-use",
        message: "[SKILL] computer-use loaded",
      });
    const cb = callbacks();
    const result = await executeSlash({
      command: "/computer-use",
      sessionId: "s1",
      gw: gw(request),
      callbacks: cb,
    });
    expect(result).toBe("sent");
    expect(cb.sys).toHaveBeenCalledWith("⚡ loading skill: computer-use");
    expect(cb.send).toHaveBeenCalledWith("[SKILL] computer-use loaded");
  });

  it("errors when a send directive carries an empty message", async () => {
    const request = vi
      .fn()
      .mockRejectedValueOnce(new Error("nope"))
      .mockResolvedValueOnce({ type: "send", message: "  " });
    const cb = callbacks();
    const result = await executeSlash({
      command: "/steer",
      sessionId: "s1",
      gw: gw(request),
      callbacks: cb,
    });
    expect(result).toBe("error");
    expect(cb.sys).toHaveBeenCalledWith("/steer: empty message");
    expect(cb.send).not.toHaveBeenCalled();
  });

  it("errors when command.dispatch returns an invalid payload", async () => {
    const request = vi
      .fn()
      .mockRejectedValueOnce(new Error("nope"))
      .mockResolvedValueOnce({ type: "prefill" });
    const cb = callbacks();
    const result = await executeSlash({
      command: "/undo",
      sessionId: "s1",
      gw: gw(request),
      callbacks: cb,
    });
    expect(result).toBe("error");
    expect(cb.sys).toHaveBeenCalledWith("error: invalid response: command.dispatch");
  });

  it("surfaces the command.dispatch error message with priority", async () => {
    const request = vi
      .fn()
      .mockRejectedValueOnce(new Error("slash worker failed"))
      .mockRejectedValueOnce(new Error("session busy — /interrupt first"));
    const cb = callbacks();
    const result = await executeSlash({
      command: "/retry",
      sessionId: "s1",
      gw: gw(request),
      callbacks: cb,
    });
    expect(result).toBe("error");
    expect(cb.sys).toHaveBeenCalledWith(
      "error: session busy — /interrupt first",
    );
  });

  it("rejects empty commands up front", async () => {
    const cb = callbacks();
    const result = await executeSlash({
      command: "///",
      sessionId: "s1",
      gw: gw(vi.fn()),
      callbacks: cb,
    });
    expect(result).toBe("error");
    expect(cb.sys).toHaveBeenCalledWith("empty slash command");
  });
});
