import {
  type AttackWindow,
  type ChartDifficulty,
  type ChartNote,
  chartDifficulties,
  chartSchema,
  type Instrument,
  instruments,
  type Lane,
  type LevelChart,
} from "@bands-battle/chart-pipeline/format";

export type {
  AttackWindow,
  ChartDifficulty,
  ChartNote,
  Instrument,
  Lane,
  LevelChart,
};
export { chartDifficulties, chartSchema, instruments };
export const defaultDifficulty: ChartDifficulty = "easy";

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

export const attackWindows: AttackWindow[] = [
  { start: 14.64, end: 22.68, threshold: 0.6 },
  { start: 34.31, end: 42.14, threshold: 0.65 },
  { start: 54.54, end: 63.01, threshold: 0.7 },
  { start: 72.25, end: 80.4, threshold: 0.75 },
];
