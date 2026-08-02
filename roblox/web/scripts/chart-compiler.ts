import type { ChartDifficulty, Instrument, Lane } from "../src/data/level";

export const drumHitLabels = ["kick", "snare", "hats"] as const;
export type DrumHitLabel = (typeof drumHitLabels)[number];

export interface RawChartEvent {
  readonly time: number;
  readonly pitch: number;
  readonly strength: number;
  readonly label?: DrumHitLabel;
}

interface GridPoint {
  readonly time: number;
  readonly beatIndex: number;
  readonly slot: number;
}

interface QuantizedEvent extends RawChartEvent, GridPoint {
  readonly gridError: number;
}

interface CompileInput {
  readonly instrument: Instrument;
  readonly events: readonly RawChartEvent[];
  readonly beatTimes: readonly number[];
  readonly clip: {
    readonly start: number;
    readonly duration: number;
  };
  readonly maxSnapSeconds: number;
}

interface CompiledNote {
  readonly time: number;
  readonly lane: Lane;
}

interface CompiledChart {
  readonly instrument: Instrument;
  readonly difficulty: ChartDifficulty;
  readonly duration: number;
  readonly notes: readonly CompiledNote[];
}

export interface CompileResult {
  readonly charts: Record<ChartDifficulty, CompiledChart>;
  readonly report: {
    readonly inputEvents: number;
    readonly acceptedEvents: number;
    readonly rejectedOffGrid: number;
    readonly duplicateGridEvents: number;
    readonly maxGridErrorMs: number;
    readonly medianGridErrorMs: number;
    readonly noteCounts: Record<ChartDifficulty, number>;
  };
}

const subdivisionsPerBeat = 4;
const notesPerBeat = { easy: 1, medium: 2, hard: 4 } as const;
const drumLaneByLabel: Record<DrumHitLabel, Lane> = {
  kick: 0,
  snare: 1,
  hats: 2,
};

class UnclassifiedDrumEvent extends Error {
  override readonly name = "UnclassifiedDrumEvent";

  constructor(readonly time: number) {
    super("Drum events must be classified as kick, snare, or hats");
  }
}

const roundSeconds = (seconds: number): number => Number(seconds.toFixed(3));

const buildGrid = (beats: readonly number[]): readonly GridPoint[] => {
  const sorted = [...new Set(beats)].sort((left, right) => left - right);
  const grid: GridPoint[] = [];
  for (let beatIndex = 0; beatIndex < sorted.length - 1; beatIndex += 1) {
    const start = sorted[beatIndex];
    const end = sorted[beatIndex + 1];
    if (start === undefined || end === undefined || end <= start) continue;
    for (let slot = 0; slot < subdivisionsPerBeat; slot += 1) {
      grid.push({
        time: start + ((end - start) * slot) / subdivisionsPerBeat,
        beatIndex,
        slot,
      });
    }
  }
  return grid;
};

const nearestGridPoint = (
  grid: readonly GridPoint[],
  time: number,
): GridPoint | undefined => {
  let low = 0;
  let high = grid.length;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    const point = grid[middle];
    if (point && point.time < time) low = middle + 1;
    else high = middle;
  }
  const before = grid[Math.max(0, low - 1)];
  const after = grid[Math.min(grid.length - 1, low)];
  if (!before) return after;
  if (!after) return before;
  return time - before.time <= after.time - time ? before : after;
};

const quantize = (
  input: CompileInput,
): {
  readonly events: readonly QuantizedEvent[];
  readonly rejected: number;
  readonly duplicates: number;
} => {
  const grid = buildGrid(input.beatTimes);
  const clipEnd = input.clip.start + input.clip.duration;
  const byGridTime = new Map<number, QuantizedEvent>();
  let rejected = 0;
  let duplicates = 0;
  for (const event of input.events) {
    const point = nearestGridPoint(grid, event.time);
    const gridError = point ? Math.abs(point.time - event.time) : Infinity;
    if (!point || gridError > input.maxSnapSeconds) {
      rejected += 1;
      continue;
    }
    const noteTime = input.instrument === "drums" ? event.time : point.time;
    if (noteTime < input.clip.start || noteTime >= clipEnd) continue;
    const existing = byGridTime.get(point.time);
    const quantized = {
      ...event,
      beatIndex: point.beatIndex,
      slot: point.slot,
      gridError,
      time: noteTime,
    };
    if (existing) {
      duplicates += 1;
      if (quantized.strength > existing.strength)
        byGridTime.set(point.time, quantized);
    } else {
      byGridTime.set(point.time, quantized);
    }
  }
  return {
    events: [...byGridTime.values()].sort(
      (left, right) => left.time - right.time,
    ),
    rejected,
    duplicates,
  };
};

const chooseDifficulty = (
  events: readonly QuantizedEvent[],
  difficulty: ChartDifficulty,
): readonly QuantizedEvent[] => {
  const grouped = new Map<number, QuantizedEvent[]>();
  for (const event of events) {
    const beatEvents = grouped.get(event.beatIndex) ?? [];
    beatEvents.push(event);
    grouped.set(event.beatIndex, beatEvents);
  }
  const selected: QuantizedEvent[] = [];
  for (const beatEvents of grouped.values()) {
    const limit = notesPerBeat[difficulty];
    selected.push(
      ...[...beatEvents]
        .sort((left, right) => {
          const leftAnchor = left.slot === 0 ? 2 : left.slot === 2 ? 1 : 0;
          const rightAnchor = right.slot === 0 ? 2 : right.slot === 2 ? 1 : 0;
          return rightAnchor - leftAnchor || right.strength - left.strength;
        })
        .slice(0, limit),
    );
  }
  return selected.sort((left, right) => left.time - right.time);
};

const assignLanes = (
  events: readonly QuantizedEvent[],
  instrument: Instrument,
  clipStart: number,
): readonly CompiledNote[] => {
  const pitches = events
    .map((event) => event.pitch)
    .sort((left, right) => left - right);
  const lower = pitches[Math.floor(pitches.length / 3)] ?? 48;
  const upper = pitches[Math.floor((pitches.length * 2) / 3)] ?? 67;
  let previousLane: Lane = 1;
  let streak = 0;
  return events.map((event) => {
    if (instrument === "drums") {
      if (!event.label) throw new UnclassifiedDrumEvent(event.time);
      return {
        time: roundSeconds(event.time - clipStart),
        lane: drumLaneByLabel[event.label],
      };
    }
    let lane: Lane = event.pitch < lower ? 0 : event.pitch > upper ? 2 : 1;
    streak = lane === previousLane ? streak + 1 : 1;
    if (streak > 4) {
      lane = previousLane === 0 ? 1 : previousLane === 1 ? 2 : 1;
      streak = 1;
    }
    previousLane = lane;
    return { time: roundSeconds(event.time - clipStart), lane };
  });
};

export const compileDifficulties = (input: CompileInput): CompileResult => {
  const quantized = quantize(input);
  const chartFor = (difficulty: ChartDifficulty): CompiledChart => ({
    instrument: input.instrument,
    difficulty,
    duration: input.clip.duration,
    notes: assignLanes(
      chooseDifficulty(quantized.events, difficulty),
      input.instrument,
      input.clip.start,
    ),
  });
  const charts = {
    easy: chartFor("easy"),
    medium: chartFor("medium"),
    hard: chartFor("hard"),
  };
  const errors = quantized.events
    .map((event) => event.gridError * 1_000)
    .sort((a, b) => a - b);
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
