import type { ChartDifficulty } from "./chart-format";
import { chooseDifficulty } from "./compile/difficulty";
import { quantize } from "./compile/grid";
import { assignLanes } from "./compile/lanes";
import type { CompiledChart, CompileInput, CompileResult } from "./types";

export type {
  CompiledChart,
  CompiledNote,
  CompileInput,
  CompileReport,
  CompileResult,
  DrumHitLabel,
  RawChartEvent,
} from "./types";
export { drumHitLabels } from "./types";

export const compileDifficulties = (input: CompileInput): CompileResult => {
  const quantized = quantize(input);
  const chartFor = (difficulty: ChartDifficulty): CompiledChart => ({
    instrument: input.instrument,
    difficulty,
    duration: input.clip.duration,
    notes: assignLanes({
      events: chooseDifficulty(quantized.events, difficulty),
      instrument: input.instrument,
      clipStart: input.clip.start,
      clipDuration: input.clip.duration,
    }),
  });
  const charts = {
    easy: chartFor("easy"),
    medium: chartFor("medium"),
    hard: chartFor("hard"),
  };
  const errors = quantized.events
    .map((event) => event.gridError * 1_000)
    .sort((left, right) => left - right);
  const medianGridErrorMs = errors[Math.floor(errors.length / 2)] ?? 0;
  return {
    charts,
    report: {
      inputEvents: input.events.length,
      acceptedEvents: quantized.events.length,
      rejectedOffGrid: quantized.rejected,
      duplicateGridEvents: quantized.duplicates,
      maxGridErrorMs: Number((errors.at(-1) ?? 0).toFixed(1)),
      medianGridErrorMs: Number(medianGridErrorMs.toFixed(1)),
      noteCounts: {
        easy: charts.easy.notes.length,
        medium: charts.medium.notes.length,
        hard: charts.hard.notes.length,
      },
    },
  };
};
