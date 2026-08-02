import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { z } from "zod";
import type { RawChartEvent } from "./chart-compiler";
import { classifyDrumHits } from "./drum-hit-classifier";

export { classifyDrumHits } from "./drum-hit-classifier";

type DrumOnsetInput = {
  readonly audio: string;
  readonly clip: {
    readonly start: number;
    readonly duration: number;
  };
  readonly ffmpeg: string;
};

class OnsetCommandFailure extends Error {
  override readonly name = "OnsetCommandFailure";

  constructor(
    readonly command: readonly string[],
    readonly exitCode: number,
    readonly stderr: string,
  ) {
    super(`${command[0] ?? "command"} exited ${exitCode}: ${stderr.trim()}`);
  }
}

const run = async (command: readonly string[]): Promise<string> => {
  const child = Bun.spawn([...command], { stdout: "pipe", stderr: "pipe" });
  const [stdout, stderr, exitCode] = await Promise.all([
    new Response(child.stdout).text(),
    new Response(child.stderr).text(),
    child.exited,
  ]);
  if (exitCode !== 0) throw new OnsetCommandFailure(command, exitCode, stderr);
  return stdout;
};

const runBytes = async (command: readonly string[]): Promise<ArrayBuffer> => {
  const child = Bun.spawn([...command], { stdout: "pipe", stderr: "pipe" });
  const [stdout, stderr, exitCode] = await Promise.all([
    new Response(child.stdout).arrayBuffer(),
    new Response(child.stderr).text(),
    child.exited,
  ]);
  if (exitCode !== 0) throw new OnsetCommandFailure(command, exitCode, stderr);
  return stdout;
};

const onsetSchema = z.coerce.number().nonnegative();

export const extractDrumOnsets = async (
  input: DrumOnsetInput,
): Promise<readonly RawChartEvent[]> => {
  const directory = await mkdtemp(join(tmpdir(), "drum-onsets-"));
  const wav = join(directory, "drums.wav");
  try {
    await run([
      input.ffmpeg,
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
    const csv = await run([
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
    const pcm = await runBytes([
      input.ffmpeg,
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
    return classifyDrumHits({
      onsetTimes,
      samples: new Float32Array(pcm),
      sampleRate: 48_000,
    }).map((event) => ({
      ...event,
      time: event.time + input.clip.start,
    }));
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
};
