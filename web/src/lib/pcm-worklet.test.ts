import { readFileSync } from "node:fs";
import vm from "node:vm";

import { describe, expect, it, vi } from "vitest";

describe("pcm-downsample worklet manual flush", () => {
  it("posts the pending sub-batch before acknowledging flush", () => {
    let Processor: new (options?: unknown) => {
      _batch: Int16Array;
      _batchFill: number;
      port: {
        onmessage: ((event: { data: unknown }) => void) | null;
        postMessage: ReturnType<typeof vi.fn>;
      };
    };
    const postMessage = vi.fn();
    class AudioWorkletProcessorStub {
      port = { onmessage: null, postMessage };
    }
    const source = readFileSync(
      new URL("../../public/pcm-worklet.js", import.meta.url),
      "utf8",
    );
    vm.runInNewContext(source, {
      AudioWorkletProcessor: AudioWorkletProcessorStub,
      Int16Array,
      Math,
      sampleRate: 48_000,
      registerProcessor: (_name: string, ctor: typeof Processor) => { Processor = ctor; },
    });

    const processor = new Processor!({ processorOptions: { inRate: 48_000, batchMs: 200 } });
    processor._batch.set([101, -202, 303]);
    processor._batchFill = 3;
    processor.port.onmessage?.({ data: { type: "flush" } });

    expect(postMessage).toHaveBeenCalledTimes(2);
    const pcm = postMessage.mock.calls[0][0] as ArrayBuffer;
    expect(Array.from(new Int16Array(pcm))).toEqual([101, -202, 303]);
    expect(postMessage.mock.calls[1][0]).toEqual({ type: "flushed" });
    expect(processor._batchFill).toBe(0);
  });
});
