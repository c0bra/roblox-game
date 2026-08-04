import { z } from "zod";

export const instruments = ["drums", "vocals", "guitar", "bass"] as const;
export type Instrument = (typeof instruments)[number];

export const chartDifficulties = ["easy", "medium", "hard"] as const;
export type ChartDifficulty = (typeof chartDifficulties)[number];

export const laneSchema = z.union([z.literal(0), z.literal(1), z.literal(2)]);
export type Lane = z.infer<typeof laneSchema>;

export const noteSchema = z.object({
  time: z.number().nonnegative(),
  lane: laneSchema,
  duration: z.number().nonnegative(),
});

export const attackSchema = z.object({
  start: z.number().nonnegative(),
  end: z.number().positive(),
  threshold: z.number().min(0).max(1),
});

export const chartSchema = z.object({
  instrument: z.enum(instruments),
  difficulty: z.enum(chartDifficulties),
  duration: z.number().positive(),
  notes: z.array(noteSchema),
  attacks: z.array(attackSchema),
});

export type ChartNote = z.infer<typeof noteSchema>;
export type AttackWindow = z.infer<typeof attackSchema>;
export type LevelChart = z.infer<typeof chartSchema>;

const difficultyPathsSchema = z.object({
  easy: z.string().regex(/^charts\/[a-z]+-easy\.json$/),
  medium: z.string().regex(/^charts\/[a-z]+-medium\.json$/),
  hard: z.string().regex(/^charts\/[a-z]+-hard\.json$/),
});

const stemPathsSchema = z.object({
  drums: z.string().regex(/^audio\/stems\/drums\.[a-z0-9]+$/),
  vocals: z.string().regex(/^audio\/stems\/vocals\.[a-z0-9]+$/),
  guitar: z.string().regex(/^audio\/stems\/guitar\.[a-z0-9]+$/),
  bass: z.string().regex(/^audio\/stems\/bass\.[a-z0-9]+$/),
});

export const bundleManifestSchema = z.object({
  schemaVersion: z.literal(1),
  source: z.object({
    name: z.string().min(1),
    separationModel: z.string().min(1),
  }),
  duration: z.number().positive(),
  timing: z.object({
    unit: z.literal("seconds"),
    sourceOffset: z.number().nonnegative(),
  }),
  stems: stemPathsSchema,
  charts: z.object({
    drums: difficultyPathsSchema,
    vocals: difficultyPathsSchema,
    guitar: difficultyPathsSchema,
    bass: difficultyPathsSchema,
  }),
  validation: z.literal("charts/validation.json"),
});

export type BundleManifest = z.infer<typeof bundleManifestSchema>;
