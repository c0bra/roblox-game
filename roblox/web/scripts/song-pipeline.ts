import { mkdir, readdir } from "node:fs/promises";
import { basename, dirname, extname, join, resolve } from "node:path";
import { parseArgs } from "node:util";
import { z } from "zod";
import {
  chartDifficulties,
  type Instrument,
  instruments,
} from "../src/data/level";
import { compileDifficulties, type RawChartEvent } from "./chart-compiler";
import { extractDrumOnsets } from "./drum-onsets";
import { selectStemCandidate } from "./stem-selection";

const audioExtensions = new Set([".wav", ".mp3", ".flac", ".m4a", ".ogg"]);
const ffmpeg = process.env.FFMPEG ?? "ffmpeg";
const argsSchema = z
  .object({
    song: z.string().min(1).optional(),
    stems: z.string().min(1).optional(),
    output: z.string().min(1),
    start: z.coerce.number().nonnegative().default(0),
    duration: z.coerce.number().positive().optional(),
    model: z.string().min(1).default("htdemucs.yaml"),
    "snap-ms": z.coerce.number().positive().max(250).default(80),
  })
  .refine((value) => Boolean(value.song) !== Boolean(value.stems), {
    message: "Provide exactly one of --song or --stems",
  });

type PipelineArgs = z.infer<typeof argsSchema>;

class CommandFailure extends Error {
  override readonly name = "CommandFailure";

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
  if (exitCode !== 0) throw new CommandFailure(command, exitCode, stderr);
  return stdout;
};

const findAudioFiles = async (
  directory: string,
): Promise<readonly string[]> => {
  const files: string[] = [];
  const visit = async (current: string): Promise<void> => {
    for (const entry of await readdir(current, { withFileTypes: true })) {
      const path = join(current, entry.name);
      if (entry.isDirectory()) await visit(path);
      else if (audioExtensions.has(extname(entry.name).toLowerCase()))
        files.push(path);
    }
  };
  await visit(directory);
  return files;
};

const selectStems = (files: readonly string[]): Record<Instrument, string> => {
  const select = (instrument: Instrument): string => {
    const selected = selectStemCandidate(files, instrument);
    if (!selected)
      throw new CommandFailure(
        ["find-stem", instrument],
        1,
        `No ${instrument} stem found`,
      );
    return selected;
  };
  return {
    drums: select("drums"),
    vocals: select("vocals"),
    guitar: select("guitar"),
    bass: select("bass"),
  };
};

const sonicCsv = async (
  plugin: string,
  audio: string,
): Promise<readonly (readonly number[])[]> => {
  const stdout = await run([
    "sonic-annotator",
    "-d",
    plugin,
    "-w",
    "csv",
    "--csv-stdout",
    "--csv-omit-filename",
    "--force",
    audio,
  ]);
  return stdout
    .split("\n")
    .map((line) => line.split(",").map(Number))
    .filter((row) => row.length > 0 && row.every(Number.isFinite));
};

const hzToMidi = (frequency: number): number =>
  69 + 12 * Math.log2(frequency / 440);

const analyzeEvents = async (
  stems: Record<Instrument, string>,
): Promise<{
  readonly beatTimes: readonly number[];
  readonly vocals: readonly RawChartEvent[];
  readonly guitar: readonly RawChartEvent[];
  readonly bass: readonly RawChartEvent[];
}> => {
  const [beats, vocals, guitar, bass] = await Promise.all([
    sonicCsv("vamp:beatroot-vamp:beatroot:beats", stems.drums),
    sonicCsv("vamp:pyin:pyin:notes", stems.vocals),
    sonicCsv("vamp:pyin:pyin:notes", stems.guitar),
    sonicCsv("vamp:pyin:pyin:notes", stems.bass),
  ]);
  const melodic = (rows: readonly (readonly number[])[]): RawChartEvent[] =>
    rows.flatMap((row) => {
      const time = row[0];
      const frequency = row[2];
      return time !== undefined && frequency !== undefined && frequency > 0
        ? [{ time, pitch: hzToMidi(frequency), strength: 1 }]
        : [];
    });
  return {
    beatTimes: beats.flatMap((row) => (row[0] === undefined ? [] : [row[0]])),
    vocals: melodic(vocals),
    guitar: melodic(guitar),
    bass: melodic(bass),
  };
};

const separate = async (
  song: string,
  output: string,
  model: string,
): Promise<string> => {
  const stems = join(output, "analysis", "stems");
  await mkdir(stems, { recursive: true });
  await run([
    "docker",
    "run",
    "--rm",
    "-v",
    `${dirname(song)}:/input:ro`,
    "-v",
    `${stems}:/output`,
    "beveradb/audio-separator",
    "--output_dir",
    "/output",
    "--output_format",
    "wav",
    "--model_filename",
    model,
    `/input/${basename(song)}`,
  ]);
  return stems;
};

const writeCharts = async (
  args: PipelineArgs,
  stems: Record<Instrument, string>,
): Promise<void> => {
  const analysis = await analyzeEvents(stems);
  const duration =
    args.duration ?? Math.max(1, (analysis.beatTimes.at(-1) ?? 1) - args.start);
  const events: Record<Instrument, readonly RawChartEvent[]> = {
    drums: await extractDrumOnsets({
      audio: stems.drums,
      clip: { start: args.start, duration },
      ffmpeg,
    }),
    vocals: analysis.vocals,
    guitar: analysis.guitar,
    bass: analysis.bass,
  };
  const compiled = Object.fromEntries(
    instruments.map((instrument) => [
      instrument,
      compileDifficulties({
        instrument,
        events: events[instrument],
        beatTimes: analysis.beatTimes,
        clip: { start: args.start, duration },
        maxSnapSeconds: args["snap-ms"] / 1_000,
      }),
    ]),
  );
  const chartsDir = join(resolve(args.output), "charts");
  await mkdir(chartsDir, { recursive: true });
  for (const instrument of instruments) {
    for (const difficulty of chartDifficulties) {
      const result = compiled[instrument];
      if (!result) continue;
      const chart = { ...result.charts[difficulty], attacks: [] };
      await Bun.write(
        join(chartsDir, `${instrument}-${difficulty}.json`),
        `${JSON.stringify(chart, null, 2)}\n`,
      );
    }
  }
  await Bun.write(
    join(chartsDir, "validation.json"),
    `${JSON.stringify(
      {
        clip: { start: args.start, duration },
        stems,
        instruments: Object.fromEntries(
          instruments.map((instrument) => [
            instrument,
            compiled[instrument]?.report,
          ]),
        ),
      },
      null,
      2,
    )}\n`,
  );
};

const parsed = parseArgs({
  options: {
    song: { type: "string" },
    stems: { type: "string" },
    output: { type: "string" },
    start: { type: "string" },
    duration: { type: "string" },
    model: { type: "string" },
    "snap-ms": { type: "string" },
  },
});
const args = argsSchema.parse(parsed.values);
const output = resolve(args.output);
const stemsDirectory = args.stems
  ? resolve(args.stems)
  : await separate(resolve(args.song ?? ""), output, args.model);
await writeCharts(args, selectStems(await findAudioFiles(stemsDirectory)));
