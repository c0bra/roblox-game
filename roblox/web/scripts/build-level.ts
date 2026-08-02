import { mkdir } from "node:fs/promises";
import { join, resolve } from "node:path";
import { Midi } from "@tonejs/midi";
import {
  attackWindows,
  chartDifficulties,
  type Instrument,
  instruments,
} from "../src/data/level";
import { compileDifficulties, type RawChartEvent } from "./chart-compiler";
import { extractDrumOnsets } from "./drum-onsets";

const songDir = resolve(import.meta.dir, "../../../audio/Heavens_Edge");
const publicDir = resolve(import.meta.dir, "../public/levels/heavens-edge");
const clipStart = 60;
const duration = 90;
const maxSnapSeconds = 0.08;
const ffmpeg = process.env.FFMPEG ?? "ffmpeg";
const stems: Record<Instrument, string> = {
  drums: "Heaven's Edge (Drums).mp3",
  vocals: "Heaven's Edge (Vocals).mp3",
  guitar: "Heaven's Edge (Guitar).mp3",
  bass: "Heaven's Edge (Bass).mp3",
};
const allStemFiles = [
  "Heaven's Edge (Backing Vocals).mp3",
  "Heaven's Edge (Bass).mp3",
  "Heaven's Edge (Drums).mp3",
  "Heaven's Edge (FX).mp3",
  "Heaven's Edge (Guitar).mp3",
  "Heaven's Edge (Keyboard).mp3",
  "Heaven's Edge (Percussion).mp3",
  "Heaven's Edge (Strings).mp3",
  "Heaven's Edge (Synth).mp3",
  "Heaven's Edge (Vocals).mp3",
];

const readText = async (file: string): Promise<string> => Bun.file(file).text();

const loadMidi = async (file: string): Promise<RawChartEvent[]> => {
  const midi = new Midi(await Bun.file(file).arrayBuffer());
  return midi.tracks
    .flatMap((track) => track.notes)
    .map((note) => ({
      time: note.time,
      pitch: note.midi,
      strength: note.velocity,
      duration: note.duration,
    }));
};

const loadBeatTimes = async (): Promise<number[]> =>
  (await readText(join(songDir, "stems/drum_beats.csv")))
    .split("\n")
    .map(Number)
    .filter(Number.isFinite);

const loadVocals = async (): Promise<RawChartEvent[]> => {
  const csv = await readText(
    join(songDir, "stems/input_(Vocals)_htdemucs_vamp_pyin_pyin_notes.csv"),
  );
  return csv
    .trim()
    .split("\n")
    .map((line) => {
      const [timeText, durationText, frequencyText] = line.split(",");
      const frequency = Number(frequencyText);
      return {
        time: Number(timeText),
        pitch: 69 + 12 * Math.log2(frequency / 440),
        strength: 1,
        duration: Number(durationText),
      };
    })
    .filter(
      (note) =>
        Number.isFinite(note.time) &&
        Number.isFinite(note.pitch) &&
        Number.isFinite(note.duration),
    );
};

const buildCharts = async (): Promise<void> => {
  const beatTimes = await loadBeatTimes();
  const sources: Record<Instrument, readonly RawChartEvent[]> = {
    drums: await extractDrumOnsets({
      audio: join(songDir, stems.drums),
      clip: { start: clipStart, duration },
      ffmpeg,
    }),
    vocals: await loadVocals(),
    guitar: await loadMidi(
      join(songDir, "Heaven's Edge (Guitar)_basic_pitch.mid"),
    ),
    bass: await loadMidi(join(songDir, "stems/bass_notes.mid")),
  };
  const compiled = {
    drums: compileDifficulties({
      instrument: "drums",
      events: sources.drums,
      beatTimes,
      clip: { start: clipStart, duration },
      maxSnapSeconds,
    }),
    vocals: compileDifficulties({
      instrument: "vocals",
      events: sources.vocals,
      beatTimes,
      clip: { start: clipStart, duration },
      maxSnapSeconds,
    }),
    guitar: compileDifficulties({
      instrument: "guitar",
      events: sources.guitar,
      beatTimes,
      clip: { start: clipStart, duration },
      maxSnapSeconds,
    }),
    bass: compileDifficulties({
      instrument: "bass",
      events: sources.bass,
      beatTimes,
      clip: { start: clipStart, duration },
      maxSnapSeconds,
    }),
  };
  for (const instrument of instruments) {
    for (const difficulty of chartDifficulties) {
      const chart = {
        ...compiled[instrument].charts[difficulty],
        attacks: attackWindows,
      };
      await Bun.write(
        join(publicDir, "charts", `${instrument}-${difficulty}.json`),
        `${JSON.stringify(chart, null, 2)}\n`,
      );
      if (difficulty === "medium")
        await Bun.write(
          join(publicDir, "charts", `${instrument}.json`),
          `${JSON.stringify(chart, null, 2)}\n`,
        );
      console.info(`${instrument}/${difficulty}: ${chart.notes.length} notes`);
    }
  }
  await Bun.write(
    join(publicDir, "charts", "validation.json"),
    `${JSON.stringify(
      {
        clip: { start: clipStart, duration },
        beatCount: beatTimes.length,
        maxSnapMs: maxSnapSeconds * 1_000,
        instruments: {
          drums: compiled.drums.report,
          vocals: compiled.vocals.report,
          guitar: compiled.guitar.report,
          bass: compiled.bass.report,
        },
      },
      null,
      2,
    )}\n`,
  );
};

const run = async (args: string[]): Promise<void> => {
  const process = Bun.spawn(args, { stdout: "inherit", stderr: "inherit" });
  const exitCode = await process.exited;
  if (exitCode !== 0)
    throw new Error(`Command failed (${exitCode}): ${args[0]}`);
};

const buildAudio = async (): Promise<void> => {
  for (const instrument of instruments) {
    const selected = stems[instrument];
    const backing = allStemFiles.filter((file) => file !== selected);
    const inputArgs = backing.flatMap((file) => ["-i", join(songDir, file)]);
    const trimmedBacking = backing
      .map(
        (_, index) =>
          `[${index}:a:0]atrim=start=${clipStart}:duration=${duration},asetpts=PTS-STARTPTS[b${index}]`,
      )
      .join(";");
    await run([
      ffmpeg,
      "-hide_banner",
      "-loglevel",
      "error",
      "-y",
      ...inputArgs,
      "-filter_complex",
      `${trimmedBacking};${backing.map((_, index) => `[b${index}]`).join("")}amix=inputs=${backing.length}:normalize=0,volume=0.48,afade=t=out:st=88:d=2[a]`,
      "-map",
      "[a]",
      "-vn",
      "-c:a",
      "aac",
      "-b:a",
      "160k",
      join(publicDir, "audio", `${instrument}-backing.m4a`),
    ]);
    await run([
      ffmpeg,
      "-hide_banner",
      "-loglevel",
      "error",
      "-y",
      "-i",
      join(songDir, selected),
      "-map",
      "0:a:0",
      "-vn",
      "-af",
      `atrim=start=${clipStart}:duration=${duration},asetpts=PTS-STARTPTS,volume=0.9,afade=t=out:st=88:d=2`,
      "-ac",
      "1",
      "-c:a",
      "aac",
      "-b:a",
      "96k",
      join(publicDir, "audio", `${instrument}-stem.m4a`),
    ]);
  }
};

await mkdir(join(publicDir, "charts"), { recursive: true });
await mkdir(join(publicDir, "audio"), { recursive: true });
await buildCharts();
await buildAudio();
