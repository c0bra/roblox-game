import { randomUUID } from "node:crypto";
import { existsSync } from "node:fs";
import {
  copyFile,
  mkdir,
  readFile,
  rename,
  rm,
  writeFile,
} from "node:fs/promises";
import { join } from "node:path";
import { validateBundle } from "@bands-battle/chart-pipeline/bundle";
import {
  chartDifficulties,
  type Instrument,
  instruments,
} from "@bands-battle/chart-pipeline/format";
import { z } from "zod";
import { levelCatalogSchema, levelIdSchema } from "../src/data/level-catalog";
import {
  type CommandRunner,
  encodeBackingCommand,
  encodeStemCommand,
} from "./ffmpeg-audio";

const importInputSchema = z.object({
  bundle: z.string().min(1),
  levelId: levelIdSchema,
  title: z.string().trim().min(1),
  levelsDirectory: z.string().min(1),
  catalogFile: z.string().min(1),
  ffmpeg: z.string().min(1),
});

export type ImportWebLevelInput = {
  readonly bundle: string;
  readonly levelId: string;
  readonly title: string;
  readonly levelsDirectory: string;
  readonly catalogFile: string;
  readonly runner: CommandRunner;
  readonly ffmpeg?: string;
};

export class ExistingWebLevelError extends Error {
  override readonly name = "ExistingWebLevelError";

  constructor(readonly levelId: string) {
    super(`Web level already exists: ${levelId}`);
  }
}

const backingInstruments = (
  instrument: Instrument,
): readonly [Instrument, Instrument, Instrument] => {
  switch (instrument) {
    case "drums":
      return ["vocals", "guitar", "bass"];
    case "vocals":
      return ["drums", "guitar", "bass"];
    case "guitar":
      return ["drums", "vocals", "bass"];
    case "bass":
      return ["drums", "vocals", "guitar"];
  }
};

export const importWebLevel = async (
  rawInput: ImportWebLevelInput,
): Promise<void> => {
  const input = importInputSchema.parse({
    ...rawInput,
    ffmpeg: rawInput.ffmpeg ?? process.env.FFMPEG ?? "ffmpeg",
  });
  const manifest = await validateBundle(input.bundle);
  const catalog = levelCatalogSchema.parse(
    JSON.parse(await readFile(input.catalogFile, "utf8")),
  );
  const destination = join(input.levelsDirectory, input.levelId);
  if (
    existsSync(destination) ||
    catalog.levels.some((level) => level.id === input.levelId)
  ) {
    throw new ExistingWebLevelError(input.levelId);
  }

  const importId = randomUUID();
  const staging = join(
    input.levelsDirectory,
    `.import-${input.levelId}-${importId}`,
  );
  const catalogStaging = `${input.catalogFile}.${importId}.tmp`;
  try {
    await Promise.all([
      mkdir(join(staging, "charts"), { recursive: true }),
      mkdir(join(staging, "audio"), { recursive: true }),
    ]);
    await Promise.all(
      instruments.flatMap((instrument) =>
        chartDifficulties.map((difficulty) =>
          copyFile(
            join(input.bundle, manifest.charts[instrument][difficulty]),
            join(staging, "charts", `${instrument}-${difficulty}.json`),
          ),
        ),
      ),
    );

    for (const instrument of instruments) {
      const stem = join(input.bundle, manifest.stems[instrument]);
      await rawInput.runner.run(
        encodeStemCommand(
          input.ffmpeg,
          stem,
          join(staging, "audio", `${instrument}-stem.m4a`),
        ),
      );
      const backing = backingInstruments(instrument).map((part) =>
        join(input.bundle, manifest.stems[part]),
      ) as [string, string, string];
      await rawInput.runner.run(
        encodeBackingCommand(
          input.ffmpeg,
          backing,
          join(staging, "audio", `${instrument}-backing.m4a`),
        ),
      );
    }

    const nextCatalog = levelCatalogSchema.parse({
      ...catalog,
      levels: [...catalog.levels, { id: input.levelId, title: input.title }],
    });
    await writeFile(
      catalogStaging,
      `${JSON.stringify(nextCatalog, null, 2)}\n`,
    );
    await rename(staging, destination);
    await rename(catalogStaging, input.catalogFile);
  } catch (error) {
    await Promise.all([
      rm(staging, { recursive: true, force: true }),
      rm(catalogStaging, { force: true }),
    ]);
    if (existsSync(destination)) {
      await rm(destination, { recursive: true, force: true });
    }
    throw error;
  }
};
