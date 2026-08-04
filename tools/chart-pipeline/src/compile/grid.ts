import type { CompileInput, GridPoint, QuantizedEvent } from "../types";

const subdivisionsPerBeat = 4;

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

export type QuantizeResult = {
  readonly events: readonly QuantizedEvent[];
  readonly rejected: number;
  readonly duplicates: number;
};

export const quantize = (input: CompileInput): QuantizeResult => {
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
      if (quantized.strength > existing.strength) {
        byGridTime.set(point.time, quantized);
      }
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
