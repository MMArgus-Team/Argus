import { describe, expect, it } from "vitest";

import {
  isEphemeralControl,
  monitorPresentation,
  removeEphemeralControlTurn,
  resolveRegistryPull,
  type MonitorRegistryItem,
} from "./monitor-control";

function monitor(overrides: Partial<MonitorRegistryItem> = {}): MonitorRegistryItem {
  return {
    monitor_id: "mon_1",
    brief: "监控视频标题",
    created_at: 1,
    ...overrides,
  };
}

describe("monitorPresentation", () => {
  it("treats legacy rows as continuous without changing their running behavior", () => {
    expect(monitorPresentation(monitor())).toMatchObject({
      active: true,
      done: false,
      canToggle: true,
      mode: "continuous",
      statusToken: "active",
    });
  });

  it("marks a completed one-shot monitor as terminal and non-toggleable", () => {
    expect(monitorPresentation(monitor({
      trigger_mode: "once",
      status: "done",
      enabled: true,
    }))).toEqual({
      active: false,
      done: true,
      canToggle: false,
      mode: "once",
      statusToken: "done",
    });
  });

  it("keeps the legacy complete status terminal", () => {
    expect(monitorPresentation(monitor({ status: "complete" }))).toMatchObject({
      active: false,
      done: true,
      canToggle: false,
      statusToken: "done",
    });
  });

  it("keeps a paused continuous monitor resumable", () => {
    expect(monitorPresentation(monitor({
      trigger_mode: "continuous",
      status: "interrupted",
      enabled: false,
    }))).toMatchObject({
      active: false,
      canToggle: true,
      mode: "continuous",
      statusToken: "interrupted",
    });
  });
});

describe("resolveRegistryPull", () => {
  it("accepts an authoritative empty registry after the backend is ready", () => {
    const current = [{ id: "stale-monitor" }];

    expect(resolveRegistryPull(current, [], true)).toEqual([]);
  });

  it("keeps a newer push when a cold non-blocking pull is temporarily empty", () => {
    const current = [{ id: "pushed-monitor" }];

    expect(resolveRegistryPull(current, [], false)).toBe(current);
    expect(resolveRegistryPull(current, [], undefined)).toBe(current);
  });

  it("accepts a non-empty pull regardless of readiness", () => {
    const incoming = [{ id: "restored-monitor" }];

    expect(resolveRegistryPull([{ id: "stale-monitor" }], incoming, false)).toBe(incoming);
  });

  it("ignores a response that omits the registry field", () => {
    const current = [{ id: "current-monitor" }];

    expect(resolveRegistryPull(current, undefined, true)).toBe(current);
  });
});

describe("ephemeral Monitor controls", () => {
  it("accepts both rollout-compatible completion markers", () => {
    expect(isEphemeralControl({ ephemeral: true })).toBe(true);
    expect(isEphemeralControl({ ephemeral_control: true })).toBe(true);
    expect(isEphemeralControl({ history_policy: "ephemeral_control" })).toBe(true);
    expect(isEphemeralControl({ history_policy: "persist" })).toBe(false);
    expect(isEphemeralControl(undefined)).toBe(false);
  });

  it("drops every center item owned by the finalized request", () => {
    const messages = [
      { id: "user-monitor", requestId: "turn-monitor" },
      { id: "assistant-monitor", requestId: "turn-monitor" },
      { id: "set-monitor-tool", requestId: "turn-monitor", kind: "tool" },
      { id: "monitor-helper-tool", requestId: "turn-monitor", kind: "tool" },
      { id: "user-other", requestId: "turn-other" },
      { id: "other-tool", requestId: "turn-other", kind: "tool" },
      { id: "assistant-other", requestId: "turn-other" },
    ];

    expect(removeEphemeralControlTurn(
      messages,
      "turn-monitor",
      "assistant-monitor",
    )).toEqual([
      { id: "user-other", requestId: "turn-other" },
      { id: "other-tool", requestId: "turn-other", kind: "tool" },
      { id: "assistant-other", requestId: "turn-other" },
    ]);
  });

  it("falls back to removing only the known assistant when request_id is absent", () => {
    const messages = [
      { id: "legacy-user" },
      { id: "legacy-tool", kind: "tool" },
      { id: "legacy-assistant" },
    ];

    expect(removeEphemeralControlTurn(messages, "", "legacy-assistant")).toEqual([
      { id: "legacy-user" },
      { id: "legacy-tool", kind: "tool" },
    ]);
  });
});
