import type { Instrument } from "./chart-format";
import type { ProcessRunner } from "./process-runner";
import type { RawChartEvent } from "./types";

export type SongAnalysis = {
  readonly beatTimes: readonly number[];
  readonly vocals: readonly RawChartEvent[];
  readonly guitar: readonly RawChartEvent[];
  readonly bass: readonly RawChartEvent[];
};

const parseCsv = (stdout: string): readonly (readonly number[])[] =>
  stdout
    .split("\n")
    .filter((line) => line.trim().length > 0)
    .map((line) => line.split(",").map(Number))
    .filter((row) => row.length > 0 && row.every(Number.isFinite));

const sonicCsv = async (
  runner: ProcessRunner,
  plugin: string,
  audio: string,
): Promise<readonly (readonly number[])[]> =>
  parseCsv(
    await runner.text([
      "sonic-annotator",
      "-d",
      plugin,
      "-w",
      "csv",
      "--csv-stdout",
      "--csv-omit-filename",
      "--force",
      audio,
    ]),
  );

const melodicEvents = (
  rows: readonly (readonly number[])[],
): readonly RawChartEvent[] =>
  rows.flatMap((row) => {
    const time = row[0];
    const duration = row[1];
    const frequency = row[2];
    return time !== undefined &&
      duration !== undefined &&
      frequency !== undefined &&
      frequency > 0
      ? [
          {
            time,
            pitch: 69 + 12 * Math.log2(frequency / 440),
            strength: 1,
            duration,
          },
        ]
      : [];
  });

export const analyzeSong = async (
  stems: Record<Instrument, string>,
  runner: ProcessRunner,
): Promise<SongAnalysis> => {
  const [beats, vocals, guitar, bass] = await Promise.all([
    sonicCsv(runner, "vamp:beatroot-vamp:beatroot:beats", stems.drums),
    sonicCsv(runner, "vamp:pyin:pyin:notes", stems.vocals),
    sonicCsv(runner, "vamp:pyin:pyin:notes", stems.guitar),
    sonicCsv(runner, "vamp:pyin:pyin:notes", stems.bass),
  ]);
  return {
    beatTimes: beats.flatMap((row) => (row[0] === undefined ? [] : [row[0]])),
    vocals: melodicEvents(vocals),
    guitar: melodicEvents(guitar),
    bass: melodicEvents(bass),
  };
};
