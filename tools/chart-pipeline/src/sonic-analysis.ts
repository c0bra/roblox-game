import { guitarAttacks, vocalPhrases } from "./instrument-recovery";
import { analyzeLowBass } from "./bass-recovery";
import type { Instrument } from "./chart-format";
import type { ProcessRunner } from "./process-runner";
import type { BassRecovery, InstrumentRecovery, RawChartEvent } from "./types";

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

export const analyzeSong = async (
  stems: Record<Instrument, string>,
  runner: ProcessRunner,
): Promise<SongAnalysis> => {
  const { FFMPEG = "ffmpeg" } = process.env;
  const band = async (file:string,filter:string) => new Float32Array(await runner.bytes([FFMPEG,"-v","error","-nostdin","-i",file,"-ac","1","-af",filter,"-ar","8000","-f","f32le","pipe:1"]));
  const [beats, vocals, guitar, bass, bassRecovery, strums, guitarPcm, voiceTrack, voicePcm] = await Promise.all([
    sonicCsv(runner, "vamp:beatroot-vamp:beatroot:beats", stems.drums),
    sonicCsv(runner, "vamp:pyin:pyin:notes", stems.vocals),
    sonicCsv(runner, "vamp:pyin:pyin:notes", stems.guitar),
    sonicCsv(runner, "vamp:pyin:pyin:notes", stems.bass),
    analyzeLowBass(stems.bass, runner, async (file) =>
      melodicEvents(await sonicCsv(runner, "vamp:pyin:pyin:notes", file)),
    ),
    sonicCsv(runner,"vamp:qm-vamp-plugins:qm-onsetdetector:onsets",stems.guitar),
    band(stems.guitar,"highpass=f=70,lowpass=f=3000"),
    sonicCsv(runner,"vamp:pyin:pyin:smoothedpitchtrack",stems.vocals),
    band(stems.vocals,"highpass=f=75,lowpass=f=3500"),
  ]);
  return {
    recovery:{bass:bassRecovery, guitar:guitarAttacks(strums.flatMap(row=>row[0] === undefined ? [] : [row[0]]),guitarPcm),vocals:vocalPhrases(voiceTrack,voicePcm)},
    bassRecovery,
    beatTimes: beats.flatMap((row) => (row[0] === undefined ? [] : [row[0]])),
    vocals: melodicEvents(vocals),
    guitar: melodicEvents(guitar),
    bass: melodicEvents(bass),
  };
};
