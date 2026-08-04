import type { ChartDifficulty, Instrument, Lane } from "./chart-format";

export const drumHitLabels = ["kick", "snare", "hats"] as const;
export type DrumHitLabel = (typeof drumHitLabels)[number];

export type RawChartEvent = {
  readonly time: number;
  readonly pitch: number;
  readonly strength: number;
  readonly duration: number;
  readonly label?: DrumHitLabel;
};

export type GridPoint = {
  readonly time: number;
  readonly beatIndex: number;
  readonly slot: number;
};

export type QuantizedEvent = RawChartEvent &
  GridPoint & {
    readonly gridError: number;
  };

export type CompileInput = {
  readonly instrument: Instrument;
  readonly events: readonly RawChartEvent[];
  readonly beatTimes: readonly number[];
  readonly clip: {
    readonly start: number;
    readonly duration: number;
  };
  readonly maxSnapSeconds: number;
};

export type CompiledNote = {
  readonly time: number;
  readonly lane: Lane;
  readonly duration: number;
};

export type CompiledChart = {
  readonly instrument: Instrument;
  readonly difficulty: ChartDifficulty;
  readonly duration: number;
  readonly notes: readonly CompiledNote[];
};

export type CompileReport = {
  readonly inputEvents: number;
  readonly acceptedEvents: number;
  readonly rejectedOffGrid: number;
  readonly duplicateGridEvents: number;
  readonly maxGridErrorMs: number;
  readonly medianGridErrorMs: number;
  readonly noteCounts: Record<ChartDifficulty, number>;
};

export type CompileResult = {
  readonly charts: Record<ChartDifficulty, CompiledChart>;
  readonly report: CompileReport;
};
