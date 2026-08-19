export type VisualCaptureSource = "camera" | "screen";

export interface VisualCaptureProfile {
  width: number;
  height: number;
  sourceFrameRate: number;
  maxSide: number;
  jpegQuality: number;
  resizeQuality: "low" | "medium";
}

/** Keep live camera labels as legible as the equivalent screen-share tier. */
export function visualCaptureProfile(
  source: VisualCaptureSource,
  lightCapture: boolean,
): VisualCaptureProfile {
  if (source === "camera" || lightCapture) {
    return {
      width: 1280,
      height: 720,
      sourceFrameRate: source === "camera" ? 24 : 4,
      maxSide: 1280,
      jpegQuality: 0.72,
      resizeQuality: "low",
    };
  }
  return {
    width: 1920,
    height: 1080,
    sourceFrameRate: 4,
    maxSide: 1920,
    jpegQuality: 0.8,
    resizeQuality: "medium",
  };
}
