import { z } from "zod";

export const instruments = ["drums", "vocals", "guitar", "bass"] as const;
export type Instrument = (typeof instruments)[number];
export const chartDifficulties = ["easy", "medium", "hard"] as const;
export type ChartDifficulty = (typeof chartDifficulties)[number];
export const defaultDifficulty: ChartDifficulty = "easy";
export type Lane = 0 | 1 | 2;

const laneSchema = z.union([z.literal(0), z.literal(1), z.literal(2)]);
const noteSchema = z.object({
  time: z.number().nonnegative(),
  lane: laneSchema,
});
const attackSchema = z.object({
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

export const instrumentLabels: Record<Instrument, string> = {
  drums: "War Drums",
  vocals: "Seraph Voice",
  guitar: "Edge Guitar",
  bass: "Void Bass",
};

export const difficultyLabels: Record<ChartDifficulty, string> = {
  easy: "Easy",
  medium: "Medium",
  hard: "Hard",
};

export const difficultyDensity: Record<ChartDifficulty, string> = {
  easy: "1 / beat",
  medium: "2 / beat",
  hard: "4 / beat",
};

export const chartPath = (
  instrument: Instrument,
  difficulty: ChartDifficulty,
): string => `/levels/heavens-edge/charts/${instrument}-${difficulty}.json`;

export const attackWindows: AttackWindow[] = [
  { start: 14.64, end: 22.68, threshold: 0.6 },
  { start: 34.31, end: 42.14, threshold: 0.65 },
  { start: 54.54, end: 63.01, threshold: 0.7 },
  { start: 72.25, end: 80.4, threshold: 0.75 },
];
