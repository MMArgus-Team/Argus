import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { MonitorEvidenceStrip } from "@/components/MonitorEvidenceStrip";
import { normalizeMonitorEvidence } from "@/lib/monitor-evidence";

describe("Monitor evidence", () => {
  it("renders the real model-input count and bounded evidence thumbnails", () => {
    const evidence = normalizeMonitorEvidence({
      input_count: 18,
      shown_count: 99,
      frames: [
        { ts: 2, source_type: "screen", thumb_b64: "dGh1bWIx" },
        { ts: 8, source_type: "screen", thumb_b64: "dGh1bWIy" },
      ],
    });

    const html = renderToStaticMarkup(createElement(MonitorEvidenceStrip, {
      evidence: evidence!,
    }));

    expect(html).toContain("Model input: 18 frames · showing 2 evidence thumbnails");
    expect(html).toContain("data:image/jpeg;base64,dGh1bWIx");
    expect(html).toContain("data:image/jpeg;base64,dGh1bWIy");
  });

  it("drops excess and malformed image rows instead of trusting the payload", () => {
    const evidence = normalizeMonitorEvidence({
      input_count: 20,
      frames: [
        ...Array.from({ length: 8 }, (_, index) => ({
          ts: index,
          thumb_b64: `dGh1bWI${index}`,
        })),
        { ts: 99, thumb_b64: "" },
      ],
    });

    expect(evidence?.input_count).toBe(20);
    expect(evidence?.shown_count).toBe(6);
    expect(evidence?.frames).toHaveLength(6);
  });
});
