import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { analyzeLowBass } from "./bass-recovery";
import type { Instrument } from "./chart-format";
import { guitarAttacks, vocalPhrases } from "./instrument-recovery";
import type { ProcessRunner } from "./process-runner";
import type { BassRecovery, InstrumentRecovery, RawChartEvent } from "./types";
import { vocalAccents } from "./vocal-accents";

export type SongAnalysis = {
  readonly recovery: Record<"bass" | "guitar" | "vocals", InstrumentRecovery>;
  readonly bassRecovery: BassRecovery;
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

async function instrumentOnsets(
  audio: string,
  runner: ProcessRunner,
): Promise<number[]> {
  const { FFMPEG = "ffmpeg" } = process.env;
  const detect = async (file: string) =>
    (
      await sonicCsv(
        runner,
        "vamp:qm-vamp-plugins:qm-onsetdetector:onsets",
        file,
      )
    ).flatMap((row) => (row[0] === undefined ? [] : [row[0]]));
  const directory = await mkdtemp(join(tmpdir(), "chart-strums-"));
  try {
    const wav = join(directory, "levelled.wav");
    await runner.text([
      FFMPEG,
      "-v",
      "error",
      "-nostdin",
      "-i",
      audio,
      "-ac",
      "1",
      "-af",
      "dynaudnorm=f=250:g=9:p=0.8:m=12:t=0.002",
      "-ar",
      "44100",
      wav,
    ]);
    const [original, levelled] = await Promise.all([
      detect(audio),
      detect(wav),
    ]);
    // Keep the original attack when both passes hear the same strum.
    let index = 0;
    return [
      ...original,
      ...levelled.filter((t) => {
        while ((original[index] ?? Infinity) < t - 0.06) index++;
        return (original[index] ?? Infinity) > t + 0.06;
      }),
    ].sort((a, b) => a - b);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
}

export const analyzeSong = async (
  stems: Record<Instrument, string>,
  runner: ProcessRunner,
): Promise<SongAnalysis> => {
  const { FFMPEG = "ffmpeg" } = process.env;
  const band = async (file: string, filter: string) =>
    new Float32Array(
      await runner.bytes([
        FFMPEG,
        "-v",
        "error",
        "-nostdin",
        "-i",
        file,
        "-ac",
        "1",
        "-af",
        filter,
        "-ar",
        "8000",
        "-f",
        "f32le",
        "pipe:1",
      ]),
    );
  const [
    beats,
    vocals,
    guitar,
    bass,
    bassRecovery,
    strums,
    guitarPcm,
    voiceTrack,
    voicePcm,
    vocalOnsets,
  ] = await Promise.all([
    sonicCsv(runner, "vamp:beatroot-vamp:beatroot:beats", stems.drums),
    sonicCsv(runner, "vamp:pyin:pyin:notes", stems.vocals),
    sonicCsv(runner, "vamp:pyin:pyin:notes", stems.guitar),
    sonicCsv(runner, "vamp:pyin:pyin:notes", stems.bass),
    analyzeLowBass(
      stems.bass,
      runner,
      async (file) =>
        melodicEvents(await sonicCsv(runner, "vamp:pyin:pyin:notes", file)),
      (file) => instrumentOnsets(file, runner),
    ),
    instrumentOnsets(stems.guitar, runner),
    band(stems.guitar, "highpass=f=70,lowpass=f=3000"),
    sonicCsv(runner, "vamp:pyin:pyin:smoothedpitchtrack", stems.vocals),
    band(stems.vocals, "highpass=f=75,lowpass=f=3500"),
    instrumentOnsets(stems.vocals, runner),
  ]);
  return {
    recovery: {
      bass: bassRecovery,
      guitar: guitarAttacks(strums, guitarPcm),
      vocals: {
        ...vocalPhrases(voiceTrack, voicePcm),
        hardAccents: vocalAccents(vocalOnsets, voiceTrack, voicePcm),
      },
    },
    bassRecovery,
    beatTimes: beats.flatMap((row) => (row[0] === undefined ? [] : [row[0]])),
    vocals: melodicEvents(vocals),
    guitar: melodicEvents(guitar),
    bass: melodicEvents(bass),
  };
};
