import type {
  ChartDifficulty,
  Instrument,
} from "@bands-battle/chart-pipeline/format";
import { z } from "zod";
import catalogJson from "./levels.json";

export const levelIdSchema = z
  .string()
  .regex(/^[a-z0-9]+(?:-[a-z0-9]+)*$/)
  .brand("LevelId");

const levelEntrySchema = z.object({
  id: levelIdSchema,
  title: z.string().trim().min(1),
});

export const levelCatalogSchema = z
  .object({
    defaultLevelId: levelIdSchema,
    levels: z.array(levelEntrySchema).min(1),
  })
  .superRefine((catalog, context) => {
    const ids = catalog.levels.map((level) => level.id);
    if (new Set(ids).size !== ids.length) {
      context.addIssue({
        code: "custom",
        path: ["levels"],
        message: "Level IDs must be unique",
      });
    }
    if (!ids.includes(catalog.defaultLevelId)) {
      context.addIssue({
        code: "custom",
        path: ["defaultLevelId"],
        message: "Default level must exist in the catalog",
      });
    }
  });

export type LevelId = z.infer<typeof levelIdSchema>;
export type LevelEntry = z.infer<typeof levelEntrySchema>;
export type LevelCatalog = z.infer<typeof levelCatalogSchema>;
export type LevelAudioUrls = {
  readonly backing: string;
  readonly stem: string;
};

export class UnknownLevelError extends Error {
  override readonly name = "UnknownLevelError";

  constructor(readonly levelId: LevelId) {
    super(`Unknown level: ${levelId}`);
  }
}

export const levelCatalog = levelCatalogSchema.parse(catalogJson);

export const resolveLevel = (
  catalog: LevelCatalog,
  levelId: LevelId,
): LevelEntry => {
  const level = catalog.levels.find((candidate) => candidate.id === levelId);
  if (!level) throw new UnknownLevelError(levelId);
  return level;
};

export const chartUrl = (
  level: LevelEntry,
  instrument: Instrument,
  difficulty: ChartDifficulty,
): string => `/levels/${level.id}/charts/${instrument}-${difficulty}.json`;

export const audioUrls = (
  level: LevelEntry,
  instrument: Instrument,
): LevelAudioUrls => ({
  backing: `/levels/${level.id}/audio/${instrument}-backing.m4a`,
  stem: `/levels/${level.id}/audio/${instrument}-stem.m4a`,
});
