import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { z } from "zod";
import { classifyDrumHits } from "./drum-hit-classifier";
import { drumTransients } from "./instrument-recovery";
import type { ProcessRunner } from "./process-runner";
import { systemProcessRunner } from "./process-runner";
import type { InstrumentRecovery, RawChartEvent } from "./types";

export { classifyDrumHits } from "./drum-hit-classifier";

export type DrumOnsetInput = {
  readonly audio: string;
  readonly clip: { readonly start: number; readonly duration: number };
  readonly ffmpeg?: string;
  readonly runner?: ProcessRunner;
};

const onsetSchema = z.coerce.number().nonnegative();

const readDrums = async (input: DrumOnsetInput) => {
  const directory = await mkdtemp(join(tmpdir(), "drum-onsets-"));
  const wav = join(directory, "drums.wav");
  const runner = input.runner ?? systemProcessRunner;
  const { FFMPEG: configuredFfmpeg } = process.env;
  const ffmpeg = input.ffmpeg ?? configuredFfmpeg ?? "ffmpeg";
  try {
    await runner.text([
      ffmpeg,
      "-hide_banner",
      "-loglevel",
      "error",
      "-y",
      "-ss",
      String(input.clip.start),
      "-t",
      String(input.clip.duration),
      "-i",
      input.audio,
      "-ac",
      "1",
      "-ar",
      "48000",
      "-c:a",
      "pcm_f32le",
      wav,
    ]);
    const csv = await runner.text([
      "sonic-annotator",
      "-d",
      "vamp:vamp-aubio:aubioonset:onsets",
      "-w",
      "csv",
      "--csv-stdout",
      "--csv-omit-filename",
      "--force",
      wav,
    ]);
    const onsetTimes = csv
      .split("\n")
      .filter((line) => line.trim().length > 0)
      .map((line) => onsetSchema.parse(line))
      .filter((time) => time > 0 && time < input.clip.duration);
    const pcm = await runner.bytes([
      ffmpeg,
      "-hide_banner",
      "-loglevel",
      "error",
      "-i",
      wav,
      "-f",
      "f32le",
      "-ac",
      "1",
      "-ar",
      "48000",
      "pipe:1",
    ]);
    const samples = new Float32Array(pcm);
    const events = classifyDrumHits({
      onsetTimes,
      samples,
      sampleRate: 48_000,
    }).map((event) => ({ ...event, time: event.time + input.clip.start }));
    return { events, samples };
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
};

export async function extractDrumOnsets(
  input: DrumOnsetInput,
): Promise<readonly RawChartEvent[]> {
  return (await readDrums(input)).events;
}
export async function analyzeDrums(
  input: DrumOnsetInput,
): Promise<{ events: readonly RawChartEvent[]; recovery: InstrumentRecovery }> {
  const { events, samples } = await readDrums(input);
  return { events, recovery: drumTransients(samples, 48000, input.clip.start) };
}
