import { copyFile, mkdir } from "node:fs/promises";
import { basename, extname, join, resolve } from "node:path";
import { findAudioFiles, selectRequiredStems } from "./audio-files";
import { createBundleManifest, validateBundle } from "./bundle";
import { compileDifficulties } from "./chart-compiler";
import {
  type BundleManifest,
  chartDifficulties,
  type Instrument,
  instruments,
} from "./chart-format";
import { extractDrumOnsets } from "./drum-onsets";
import type { BuildOptions } from "./options";
import type { ProcessRunner } from "./process-runner";
import { systemProcessRunner } from "./process-runner";
import { analyzeSong } from "./sonic-analysis";
import { separateSong } from "./stem-separator";
import type { CompileResult, RawChartEvent } from "./types";

type StemMap = Record<Instrument, string>;
type EventMap = Record<Instrument, readonly RawChartEvent[]>;

export type WriteChartBundleInput = {
  readonly output: string;
  readonly source: string;
  readonly model: string;
  readonly clip: { readonly start: number; readonly duration: number };
  readonly maxSnapSeconds: number;
  readonly sourceStems: StemMap;
  readonly beatTimes: readonly number[];
  readonly events: EventMap;
};

const copyStems = async (
  output: string,
  sourceStems: StemMap,
): Promise<BundleManifest["stems"]> => {
  const stemsDirectory = join(output, "audio", "stems");
  await mkdir(stemsDirectory, { recursive: true });
  const entries = await Promise.all(
    instruments.map(async (instrument) => {
      const source = sourceStems[instrument];
      const extension = extname(source).toLowerCase() || ".wav";
      const relative = `audio/stems/${instrument}${extension}`;
      const destination = join(output, relative);
      if (resolve(source) !== resolve(destination))
        await copyFile(source, destination);
      return [instrument, relative] as const;
    }),
  );
  return Object.fromEntries(entries) as BundleManifest["stems"];
};

export const writeChartBundle = async (
  input: WriteChartBundleInput,
): Promise<BundleManifest> => {
  const output = resolve(input.output);
  const chartsDirectory = join(output, "charts");
  await mkdir(chartsDirectory, { recursive: true });
  const stems = await copyStems(output, input.sourceStems);
  const compiled = Object.fromEntries(
    instruments.map((instrument) => [
      instrument,
      compileDifficulties({
        instrument,
        events: input.events[instrument],
        beatTimes: input.beatTimes,
        clip: input.clip,
        maxSnapSeconds: input.maxSnapSeconds,
      }),
    ]),
  ) as Record<Instrument, CompileResult>;
  for (const instrument of instruments) {
    for (const difficulty of chartDifficulties) {
      const chart = { ...compiled[instrument].charts[difficulty], attacks: [] };
      await Bun.write(
        join(chartsDirectory, `${instrument}-${difficulty}.json`),
        `${JSON.stringify(chart, null, 2)}\n`,
      );
    }
  }
  await Bun.write(
    join(chartsDirectory, "validation.json"),
    `${JSON.stringify(
      {
        clip: input.clip,
        beatCount: input.beatTimes.length,
        maxSnapMs: input.maxSnapSeconds * 1_000,
        instruments: Object.fromEntries(
          instruments.map((instrument) => [
            instrument,
            compiled[instrument].report,
          ]),
        ),
      },
      null,
      2,
    )}\n`,
  );
  const manifest = createBundleManifest({
    source: basename(input.source),
    start: input.clip.start,
    duration: input.clip.duration,
    model: input.model,
    stems,
  });
  await Bun.write(
    join(output, "manifest.json"),
    `${JSON.stringify(manifest, null, 2)}\n`,
  );
  return validateBundle(output);
};

export const buildChartBundle = async (
  options: BuildOptions,
  runner: ProcessRunner = systemProcessRunner,
): Promise<BundleManifest> => {
  const output = resolve(options.output);
  const source =
    "song" in options ? resolve(options.song) : resolve(options.stems);
  const stemsDirectory =
    "song" in options
      ? await separateSong({
          song: source,
          output,
          model: options.model,
          runner,
        })
      : source;
  const sourceStems = selectRequiredStems(await findAudioFiles(stemsDirectory));
  const analysis = await analyzeSong(sourceStems, runner);
  const duration =
    options.duration ??
    Math.max(1, (analysis.beatTimes.at(-1) ?? 1) - options.start);
  const clip = { start: options.start, duration };
  const events: EventMap = {
    drums: await extractDrumOnsets({ audio: sourceStems.drums, clip, runner }),
    vocals: analysis.vocals,
    guitar: analysis.guitar,
    bass: analysis.bass,
  };
  return writeChartBundle({
    output,
    source,
    model: options.model,
    clip,
    maxSnapSeconds: options.snapMs / 1_000,
    sourceStems,
    beatTimes: analysis.beatTimes,
    events,
  });
};
