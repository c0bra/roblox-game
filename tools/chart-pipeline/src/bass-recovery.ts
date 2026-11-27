import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { ProcessRunner } from "./process-runner";
import { activeSpans } from "./stem-activity";
import type { BassRecovery, RawChartEvent } from "./types";

export async function analyzeLowBass(
  audio: string,
  runner: ProcessRunner,
  detect: (file: string) => Promise<readonly RawChartEvent[]>,
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
    const [shifted, pcm] = await Promise.all([
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
    ]);
    const spans = activeSpans(new Float32Array(pcm));
    const events = shifted.flatMap((event) => {
      const time = event.time * 2;
      const pitch = event.pitch - 12;
      const hz = 440 * 2 ** ((pitch - 69) / 12);
      const span = spans.find((s) => time >= s.start && time < s.end);
      if (!span || hz < 25 || hz > 180) return [];
      const duration = Math.min(event.duration * 2, span.end - time);
      return duration >= 0.1 ? [{ ...event, time, pitch, duration }] : [];
    });
    return { events, active: spans };
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}
