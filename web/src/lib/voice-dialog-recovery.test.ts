import { describe, expect, it } from "vitest";

import { VoiceDialogRecovery } from "./voice-dialog-recovery";

describe("VoiceDialogRecovery", () => {
  it("waits without a session, then starts exactly once when one becomes live", () => {
    const recovery = new VoiceDialogRecovery();

    expect(recovery.enable("")).toBeNull();
    expect(recovery.snapshot()).toMatchObject({
      desired: true,
      phase: "waiting_for_session",
      sessionId: "",
    });

    const activation = recovery.sessionAvailable("live-a");
    expect(activation).toEqual({ sessionId: "live-a", attempt: expect.any(Number) });
    expect(recovery.sessionAvailable("live-a")).toBeNull();
    expect(recovery.activationSucceeded(activation!)).toBe(true);
    expect(recovery.sessionAvailable("live-a")).toBeNull();
    expect(recovery.snapshot()).toMatchObject({
      desired: true,
      phase: "active",
      sessionId: "live-a",
    });
  });

  it("keeps continuous intent across a boundary and rearms once on the replacement sid", () => {
    const recovery = new VoiceDialogRecovery();
    const first = recovery.enable("live-a")!;
    expect(recovery.activationSucceeded(first)).toBe(true);

    recovery.boundary();
    expect(recovery.snapshot()).toMatchObject({
      desired: true,
      phase: "waiting_for_session",
      sessionId: "",
    });
    const replacement = recovery.sessionAvailable("live-b")!;
    expect(replacement.sessionId).toBe("live-b");
    expect(replacement.attempt).not.toBe(first.attempt);
    expect(recovery.sessionAvailable("live-b")).toBeNull();
    expect(recovery.activationSucceeded(replacement)).toBe(true);
  });

  it("exposes the exact old owner for A-off before the one B-on rearm", () => {
    const recovery = new VoiceDialogRecovery();
    const calls: string[] = [];
    const first = recovery.enable("live-a")!;
    recovery.activationSucceeded(first);

    const oldOwner = recovery.boundary();
    if (oldOwner) calls.push(`${oldOwner}:false`);
    const replacement = recovery.sessionAvailable("live-b")!;
    calls.push(`${replacement.sessionId}:true`);
    recovery.activationSucceeded(replacement);

    expect(calls).toEqual(["live-a:false", "live-b:true"]);
    expect(recovery.sessionAvailable("live-b")).toBeNull();
  });

  it("ignores late success/failure from the old transport", () => {
    const recovery = new VoiceDialogRecovery();
    const oldActivation = recovery.enable("live-a")!;

    recovery.boundary();
    const current = recovery.sessionAvailable("live-b")!;
    expect(recovery.activationSucceeded(oldActivation)).toBe(false);
    expect(recovery.activationFailed(oldActivation)).toBe(false);
    expect(recovery.snapshot()).toMatchObject({
      desired: true,
      phase: "starting",
      sessionId: "live-b",
    });
    expect(recovery.activationSucceeded(current)).toBe(true);
  });

  it("fails closed after a current activation failure", () => {
    const recovery = new VoiceDialogRecovery();
    const activation = recovery.enable("live-a")!;

    expect(recovery.activationFailed(activation)).toBe(true);
    expect(recovery.snapshot()).toMatchObject({
      desired: false,
      phase: "off",
      sessionId: "",
    });
    expect(recovery.sessionAvailable("live-b")).toBeNull();
  });

  it("does not leave an ON flag behind when session establishment settles empty", () => {
    const recovery = new VoiceDialogRecovery();
    recovery.enable("");

    expect(recovery.sessionUnavailable()).toBe(true);
    expect(recovery.wantsVoiceDialog()).toBe(false);
    expect(recovery.sessionUnavailable()).toBe(false);
  });

  it("OFF invalidates a pending start and does not auto-rearm", () => {
    const recovery = new VoiceDialogRecovery();
    const activation = recovery.enable("live-a")!;

    recovery.disable();
    expect(recovery.activationSucceeded(activation)).toBe(false);
    recovery.boundary();
    expect(recovery.sessionAvailable("live-b")).toBeNull();
    expect(recovery.snapshot().phase).toBe("off");
  });
});
