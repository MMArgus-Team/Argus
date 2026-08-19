import { describe, expect, it } from "vitest";

import { visualCaptureProfile } from "./visual-capture-profile";

describe("visualCaptureProfile", () => {
  it("matches camera fidelity to the successful light screen-share tier", () => {
    const camera = visualCaptureProfile("camera", true);
    const screen = visualCaptureProfile("screen", true);

    expect(camera).toMatchObject({
      width: 1280,
      height: 720,
      maxSide: 1280,
      jpegQuality: 0.72,
      resizeQuality: "low",
    });
    expect(camera).toMatchObject({
      width: screen.width,
      height: screen.height,
      maxSide: screen.maxSide,
      jpegQuality: screen.jpegQuality,
      resizeQuality: screen.resizeQuality,
    });
  });

  it("does not downscale a normal camera below 720p", () => {
    expect(visualCaptureProfile("camera", false)).toMatchObject({
      width: 1280,
      height: 720,
      maxSide: 1280,
      jpegQuality: 0.72,
    });
  });

  it("preserves the normal 1080p screen-share profile", () => {
    expect(visualCaptureProfile("screen", false)).toEqual({
      width: 1920,
      height: 1080,
      sourceFrameRate: 4,
      maxSide: 1920,
      jpegQuality: 0.8,
      resizeQuality: "medium",
    });
  });
});
