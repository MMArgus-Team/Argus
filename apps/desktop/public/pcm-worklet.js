// PCM-16 downsample worklet.
//
// Runs on a dedicated audio thread (NOT the main JS thread) — the previous
// ScriptProcessor implementation blocked the UI thread every 85ms with base64
// + JSON.stringify + gw.request, causing "task queue exceeded allotted
// deadline" and hover-freeze under video + ASR + streaming reply load.
//
// Contract: each incoming input frame (128 samples at the AudioContext's rate,
// typically 48 kHz) is downsampled to 16 kHz mono int16. Samples are appended
// to a batch until we've accumulated >= `batchMs` milliseconds of audio, then
// posted to the main thread as a single ArrayBuffer via `port.postMessage`.
// Batching cuts message rate ~2-3x vs the old 85ms cadence; the DashScope
// realtime ASR's server-side VAD (silence_ms=1200) doesn't care about the
// packet cadence.
//
// The main thread only has to: base64-encode the buffer + send one RPC. No
// per-sample math in the UI thread anymore.

class PcmDownsampleProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = (options && options.processorOptions) || {};
    this.inRate = Number(opts.inRate) || sampleRate;   // AudioContext rate
    this.outRate = 16000;
    this.ratio = this.inRate / this.outRate;
    // 200ms at 16 kHz mono int16 = 3200 samples = 6400 bytes per batch.
    this.batchTargetSamples = Math.max(
      1600, Math.floor((Number(opts.batchMs) || 200) * this.outRate / 1000));
    this._batch = new Int16Array(this.batchTargetSamples);
    this._batchFill = 0;
    // Fractional-sample cursor for continuous downsampling across process()
    // calls (128-sample frames don't align to any integer decimation, so we
    // carry the sub-sample phase between frames — otherwise the downsampled
    // waveform gets tiny clicks at frame boundaries).
    this._phase = 0.0;
    // A manual turn may stop between 200ms batches. Flush the already-processed
    // tail before the renderer sends asr_stop(disposition=finish).
    this.port.onmessage = (event) => {
      if (event && event.data && event.data.type === "flush") {
        this._flushBatch();
        this.port.postMessage({ type: "flushed" });
      }
    };
  }

  _flushBatch() {
    if (this._batchFill <= 0) return;
    const out = new Int16Array(this._batchFill);
    out.set(this._batch.subarray(0, this._batchFill));
    this._batchFill = 0;
    this.port.postMessage(out.buffer, [out.buffer]);
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const ch = input[0];
    if (!ch || ch.length === 0) return true;

    // Downsample this 128-sample frame using linear picking with fractional
    // phase carry-over.
    let phase = this._phase;
    const ratio = this.ratio;
    let batch = this._batch;
    let fill = this._batchFill;
    const cap = this.batchTargetSamples;

    while (phase < ch.length) {
      const i = phase | 0;  // floor
      let s = ch[i];
      if (s > 1) s = 1; else if (s < -1) s = -1;
      batch[fill++] = s < 0 ? Math.round(s * 0x8000) : Math.round(s * 0x7fff);
      if (fill >= cap) {
        // Copy into a fresh ArrayBuffer so we can transfer ownership to the
        // main thread without holding onto it here.
        const out = new Int16Array(fill);
        out.set(batch.subarray(0, fill));
        this.port.postMessage(out.buffer, [out.buffer]);
        fill = 0;
        batch = this._batch;  // reuse the working buffer
      }
      phase += ratio;
    }
    // Carry the remaining fractional phase into the next frame.
    this._phase = phase - ch.length;
    this._batchFill = fill;
    return true;
  }
}

registerProcessor("pcm-downsample-processor", PcmDownsampleProcessor);
