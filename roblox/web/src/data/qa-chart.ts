import type { LevelChart } from "@bands-battle/chart-pipeline/format";

export const createQaChart = (chart: LevelChart): LevelChart => ({
  ...chart,
  duration: 12,
  notes: chart.notes.slice(0, 12).map((note, index) => ({
    ...note,
    time: 1.2 + index * 0.82,
    duration: Math.min(note.duration, 0.7),
  })),
  attacks: [
    { start: 2.8, end: 4.8, threshold: 0.35 },
    { start: 6.5, end: 8.5, threshold: 0.35 },
  ],
});
