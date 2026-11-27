import type { ChartDifficulty } from "./chart-format";
import { chooseDifficulty } from "./compile/difficulty";
import { quantize } from "./compile/grid";
import { assignLanes } from "./compile/lanes";
import {
  activeGaps,
  protectExisting,
  recoveryCandidates,
} from "./compile/recovery";
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

export const compileDifficulties = (source: CompileInput): CompileResult => {
  const recovery =
    source.recovery ??
    (source.instrument === "bass" ? source.bassRecovery : undefined);
  const input = recovery ? { ...source, recovery } : source;
  const quantized = quantize(input);
  const candidates = recoveryCandidates(input, quantized.events);
  const added = { easy: 0, medium: 0, hard: 0 };
  const chartFor = (difficulty: ChartDifficulty): CompiledChart => {
    const settings = {
      instrument: input.instrument,
      clipStart: input.clip.start,
      clipDuration: input.clip.duration,
    };
    const primary = assignLanes({
      ...settings,
      events: chooseDifficulty(quantized.events, difficulty),
    });
    const extra = assignLanes({
      ...settings,
      events: chooseDifficulty(
        protectExisting(candidates, primary, input.clip.start),
        difficulty,
      ),
    });
    added[difficulty] = extra.length;
    return {
      instrument: input.instrument,
      difficulty,
      duration: input.clip.duration,
      notes: extra.length
        ? [...primary, ...extra].sort((a, b) => a.time - b.time)
        : primary,
    };
  };
  const charts = {
    easy: chartFor("easy"),
    medium: chartFor("medium"),
    hard: chartFor("hard"),
  };
  const errors = quantized.events
    .map((event) => event.gridError * 1_000)
    .sort((left, right) => left - right);
  const medianGridErrorMs = errors[Math.floor(errors.length / 2)] ?? 0;
  const review = recovery
    ? {
        added,
        activeGaps: activeGaps(recovery.active, input.clip, [
          ...(input.reserved ?? []),
          ...charts.medium.notes.map((n) => ({
            start: n.time + input.clip.start,
            end: n.time + input.clip.start + n.duration,
          })),
        ]),
      }
    : undefined;
  return {
    charts,
    report: {
      ...(review
        ? {
            recovery: review,
            ...(input.instrument === "bass" ? { bassRecovery: review } : {}),
          }
        : {}),
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
