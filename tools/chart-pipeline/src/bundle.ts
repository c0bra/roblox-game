import { access, readFile } from "node:fs/promises";
import { join } from "node:path";
import {
  type BundleManifest,
  bundleManifestSchema,
  type ChartDifficulty,
  chartDifficulties,
  chartSchema,
  type Instrument,
  instruments,
} from "./chart-format";

const chartPaths = {
  drums: {
    easy: "charts/drums-easy.json",
    medium: "charts/drums-medium.json",
    hard: "charts/drums-hard.json",
  },
  vocals: {
    easy: "charts/vocals-easy.json",
    medium: "charts/vocals-medium.json",
    hard: "charts/vocals-hard.json",
  },
  guitar: {
    easy: "charts/guitar-easy.json",
    medium: "charts/guitar-medium.json",
    hard: "charts/guitar-hard.json",
  },
  bass: {
    easy: "charts/bass-easy.json",
    medium: "charts/bass-medium.json",
    hard: "charts/bass-hard.json",
  },
} as const;

const defaultStemPaths = {
  drums: "audio/stems/drums.wav",
  vocals: "audio/stems/vocals.wav",
  guitar: "audio/stems/guitar.wav",
  bass: "audio/stems/bass.wav",
} as const;

type BundleManifestInput = {
  readonly source: string;
  readonly start: number;
  readonly duration: number;
  readonly model: string;
  readonly stems?: BundleManifest["stems"];
};

export const createBundleManifest = (
  input: BundleManifestInput,
): BundleManifest =>
  bundleManifestSchema.parse({
    schemaVersion: 1,
    source: { name: input.source, separationModel: input.model },
    duration: input.duration,
    timing: { unit: "seconds", sourceOffset: input.start },
    stems: input.stems ?? defaultStemPaths,
    charts: chartPaths,
    validation: "charts/validation.json",
  });

export const validateBundle = async (
  directory: string,
): Promise<BundleManifest> => {
  const manifest = bundleManifestSchema.parse(
    JSON.parse(await readFile(join(directory, "manifest.json"), "utf8")),
  );
  await Promise.all(
    instruments.map((instrument) =>
      access(join(directory, manifest.stems[instrument])),
    ),
  );
  for (const instrument of instruments) {
    for (const difficulty of chartDifficulties) {
      const chart = chartSchema.parse(
        JSON.parse(
          await readFile(
            join(directory, manifest.charts[instrument][difficulty]),
            "utf8",
          ),
        ),
      );
      if (chart.instrument !== instrument || chart.difficulty !== difficulty) {
        throw new BundleChartMismatch(instrument, difficulty);
      }
    }
  }
  JSON.parse(await readFile(join(directory, manifest.validation), "utf8"));
  return manifest;
};

export class BundleChartMismatch extends Error {
  override readonly name = "BundleChartMismatch";

  constructor(
    readonly instrument: Instrument,
    readonly difficulty: ChartDifficulty,
  ) {
    super(`Chart metadata does not match ${instrument}/${difficulty}`);
  }
}
