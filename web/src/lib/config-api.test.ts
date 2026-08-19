/**
 * Session-scoped model switching contract.
 *
 * The payload shape here is load-bearing, so it gets a test rather than trust:
 * `--session` is what keeps a composer pick from rewriting the GLOBAL profile
 * default (the backend's resolve_persist_behavior() falls through to
 * `model.persist_switch_by_default`, which DEFAULTS TO TRUE — so omitting the
 * flag silently persists globally).
 */

import { describe, expect, it, vi } from "vitest";

import { setSessionModel, setSessionReasoningEffort } from "@/lib/config-api";
import type { GatewayClient } from "@/lib/gatewayClient";

function fakeGateway(reply: unknown = {}) {
  const request = vi.fn(async () => reply);

  return { gw: { request } as unknown as GatewayClient, request };
}

describe("setSessionModel", () => {
  it("scopes the switch to one session with --session", async () => {
    const { gw, request } = fakeGateway({ value: "claude-sonnet-4.6" });

    await setSessionModel(gw, "session-1", {
      model: "claude-sonnet-4.6",
      provider: "anthropic",
    });

    expect(request).toHaveBeenCalledWith("config.set", {
      confirm_expensive_model: false,
      key: "model",
      session_id: "session-1",
      value: "claude-sonnet-4.6 --provider anthropic --session",
    });
  });

  it("forwards the expensive-model confirmation on retry", async () => {
    const { gw, request } = fakeGateway({ value: "expensive-opus" });

    await setSessionModel(gw, "session-1", {
      confirmExpensive: true,
      model: "expensive-opus",
      provider: "anthropic",
    });

    expect(request).toHaveBeenCalledWith(
      "config.set",
      expect.objectContaining({ confirm_expensive_model: true }),
    );
  });

  it("returns confirm_required so callers can undo their optimistic update", async () => {
    // The backend returns this BEFORE calling agent.switch_model() — the switch
    // did NOT happen, so a caller that ignores it shows a model the agent isn't
    // running.
    const { gw } = fakeGateway({
      confirm_message: "costs a lot",
      confirm_required: true,
    });

    const result = await setSessionModel(gw, "session-1", {
      model: "expensive-opus",
      provider: "anthropic",
    });

    expect(result.confirm_required).toBe(true);
    expect(result.confirm_message).toBe("costs a lot");
  });
});

describe("setSessionReasoningEffort", () => {
  // Model and effort must stay symmetrical: both session-scoped, both landing
  // on the live agent rather than config.yaml.
  it("uses scope:session so it hits the live agent", async () => {
    const { gw, request } = fakeGateway({});

    await setSessionReasoningEffort(gw, "session-1", "high");

    expect(request).toHaveBeenCalledWith("config.set", {
      key: "reasoning",
      scope: "session",
      session_id: "session-1",
      value: "high",
    });
  });

  it("rejects an invalid effort before touching the gateway", async () => {
    const { gw, request } = fakeGateway({});

    await expect(
      setSessionReasoningEffort(gw, "session-1", "turbo"),
    ).rejects.toThrow(/invalid reasoning effort/);
    expect(request).not.toHaveBeenCalled();
  });
});
