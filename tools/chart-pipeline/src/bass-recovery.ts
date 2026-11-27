import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { ProcessRunner } from "./process-runner";
import { activeSpans } from "./stem-activity";
import type { BassRecovery, RawChartEvent } from "./types";

// Onsets and pitch changes are independent: a bassist can strike the same
// string several times. Require a regular low-frequency oscillation after the attack.
export function bassRestrikes(
  onsets: readonly number[],
  pcm: Float32Array,
  active: readonly { start: number; end: number }[],
): RawChartEvent[] {
  return onsets.flatMap((time) => {
    const span = active.find((s) => s.start <= time && time < s.end);
    if (!span) return [];
    const start = Math.round((time + 0.01) * 8000);
    const end = Math.min(start + 960, Math.floor(span.end * 8000), pcm.length);
    const periods: number[] = [];
    let prior = 0;
    for (let i = start + 1; i < end; i++) {
      if ((pcm[i - 1] ?? 0) <= 0 && (pcm[i] ?? 0) > 0) {
        if (prior) periods.push(i - prior);
        prior = i;
      }
    }
    periods.sort((a, b) => a - b);
    const period = periods[Math.floor(periods.length / 2)] ?? 0;
    const hz = 8000 / period;
    if (
      periods.length < 2 ||
      hz < 25 ||
      hz > 180 ||
      periods.filter((v) => Math.abs(v - period) <= period * 0.25).length <
        periods.length * 0.65
    )
      return [];
    return [
      { time, pitch: 69 + 12 * Math.log2(hz / 440), duration: 0, strength: 1 },
    ];
  });
}

export async function analyzeLowBass(
  audio: string,
  runner: ProcessRunner,
  detect: (file: string) => Promise<readonly RawChartEvent[]>,
  detectAttacks: (file: string) => Promise<readonly number[]>,
): Promise<BassRecovery> {
  const dir = await mkdtemp(join(tmpdir(), "chart-bass-"));
  try {
    const wav = join(dir, "octave.wav");
    const { FFMPEG = "ffmpeg" } = process.env;
    const ffmpeg = [
      FFMPEG,
      "-v",
      "error",
      "-nostdin",
      "-i",
      audio,
      "-ac",
      "1",
      "-af",
    ];
    await runner.text([
      ...ffmpeg,
      "aresample=44100,lowpass=f=180,asetrate=88200,aresample=44100",
      wav,
    ]);
    const [shifted, pcm, onsets] = await Promise.all([
      detect(wav),
      runner.bytes([
        ...ffmpeg,
        "highpass=f=25,lowpass=f=180",
        "-ar",
        "8000",
        "-f",
        "f32le",
        "pipe:1",
      ]),
      detectAttacks(audio),
    ]);
    const samples = new Float32Array(pcm);
    const spans = activeSpans(samples);
    const events = shifted.flatMap((event) => {
      const time = event.time * 2;
      const pitch = event.pitch - 12;
      const hz = 440 * 2 ** ((pitch - 69) / 12);
      const span = spans.find((s) => time >= s.start && time < s.end);
      if (!span || hz < 25 || hz > 180) return [];
      const duration = Math.min(event.duration * 2, span.end - time);
      return duration >= 0.1 ? [{ ...event, time, pitch, duration }] : [];
    });
    return {
      events,
      active: spans,
      restrikes: bassRestrikes(onsets, samples, spans),
    };
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}
