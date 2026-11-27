import type { TimeSpan } from "./types";

export function activeSpans(samples: Float32Array, rate = 8000): TimeSpan[] {
  const frameSize = Math.round(rate / 20);
  const frames: number[] = [];
  for (let i = 0; i < samples.length; i += frameSize) {
    const frame = samples.subarray(i, i + frameSize);
    frames.push(
      Math.sqrt(frame.reduce((sum, x) => sum + x * x, 0) / frame.length),
    );
  }
  const sorted = [...frames].sort((a, b) => a - b);
  const threshold = Math.max(
    10 ** (-55 / 20),
    (sorted[Math.floor(sorted.length * 0.9)] ?? 0) * 10 ** (-24 / 20),
  );
  const spans: TimeSpan[] = [];
  for (const [i, rms] of frames.entries()) {
    if (rms < threshold) continue;
    const end = Math.min((i + 1) / 20, samples.length / rate);
    const last = spans.at(-1);
    if (last && i / 20 - last.end <= 0.051) last.end = end;
    else spans.push({ start: i / 20, end });
  }
  return spans;
}

