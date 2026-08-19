import { useI18n } from "@/i18n";
import type { MonitorEvidence } from "@/lib/monitor-evidence";

const formatFrameTs = (ts: number): string => {
  const sec = Math.max(0, Math.round(ts));
  return `${String(Math.floor(sec / 60)).padStart(2, "0")}:${String(sec % 60).padStart(2, "0")}`;
};

export function MonitorEvidenceStrip({ evidence }: { evidence: MonitorEvidence }) {
  const { t } = useI18n();

  return (
    <div className="mt-1.5">
      <div className="mb-1 text-[9px] text-amber-300/55">
        {t.multimodal.monitor.evidenceSummary(evidence.input_count, evidence.frames.length)}
      </div>
      <div className="grid grid-cols-3 gap-1">
        {evidence.frames.map((frame, index) => {
          const src = `data:image/jpeg;base64,${frame.thumb_b64}`;
          return (
            <button
              key={`${frame.ts}_${index}`}
              type="button"
              className="group relative overflow-hidden rounded border border-amber-400/20 bg-black"
              onClick={() => window.open(src, "_blank")}
              title={`${formatFrameTs(frame.ts)}${frame.source_type ? ` · ${frame.source_type}` : ""}`}
            >
              <img
                alt={t.multimodal.monitor.evidenceFrame(index + 1)}
                className="aspect-video w-full object-cover"
                src={src}
              />
              <span className="absolute bottom-0 right-0 bg-black/70 px-1 py-px font-mono text-[8px] text-white">
                {formatFrameTs(frame.ts)}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
