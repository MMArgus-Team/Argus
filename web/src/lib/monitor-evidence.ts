// ★ 2026-08-19: 实现已搬到 @hermes/shared/monitor-evidence (apps/shared/src)。
// 这里原本是和 apps/desktop/src/store/multimodal-deep.ts 逐字相同的一份副本,
// 两边各自内联 6 / 600_000 / 32 / 100_000 四个字面量。保留本文件做 re-export,
// 所以 `@/lib/monitor-evidence` 的既有 import 面不变
// (MonitorEvidenceStrip.tsx / MultimodalChatPage.tsx / 对应的 vitest)。
export {
  MONITOR_EVIDENCE_MAX_B64_CHARS,
  MONITOR_EVIDENCE_MAX_FRAMES,
  type MonitorEvidence,
  type MonitorEvidenceFrame,
  normalizeMonitorEvidence,
} from "@hermes/shared";
