import { copyFile, mkdir } from "node:fs/promises";
import { basename, join } from "node:path";
import type { ProcessRunner } from "./process-runner";

export const separateSong = async (input: {
  readonly song: string;
  readonly output: string;
  readonly model: string;
  readonly runner: ProcessRunner;
}): Promise<string> => {
  const inputDirectory = join(input.output, "analysis", "input");
  const stems = join(input.output, "analysis", "separated");
  await Promise.all([
    mkdir(inputDirectory, { recursive: true }),
    mkdir(stems, { recursive: true }),
  ]);
  const songName = basename(input.song);
  await copyFile(input.song, join(inputDirectory, songName));
  await input.runner.text([
    "docker",
    "run",
    "--rm",
    "-v",
    `${inputDirectory}:/input:ro`,
    "-v",
    `${stems}:/output`,
    "beveradb/audio-separator",
    "--output_dir",
    "/output",
    "--output_format",
    "wav",
    "--model_filename",
    input.model,
    `/input/${songName}`,
  ]);
  return stems;
};
