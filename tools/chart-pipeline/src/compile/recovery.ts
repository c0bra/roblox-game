import type {
  CompiledNote,
  CompileInput,
  QuantizedEvent,
  TimeSpan,
} from "../types";
import { quantize } from "./grid";

export function recoveryCandidates(
  input: CompileInput,
  primary: readonly QuantizedEvent[],
) {
  const recovery = input.recovery;
  if (!recovery) return [];
  const reserved = input.reserved ?? [];
  const occupied = new Set(primary.map((e) => e.beatIndex));
  const events = recovery.events.filter(
    (e) => !reserved.some((w) => e.time >= w.start && e.time < w.end),
  );
  return quantize({ ...input, events }).events.flatMap((e) => {
    if (
      occupied.has(e.beatIndex) ||
      reserved.some((w) => e.time >= w.start && e.time < w.end)
    )
      return [];
    const span = recovery.active.find(
      (w) => e.time >= w.start && e.time < w.end,
    );
    if (!span) return [];
    const stop = Math.min(
      span.end,
      ...reserved.filter((w) => w.start > e.time).map((w) => w.start - 0.08),
    );
    return [
      { ...e, duration: Math.min(e.duration, Math.max(0, stop - e.time)) },
    ];
  });
}

// Reserve the complete hold on this difficulty, including its release gap.
// Times in the chart are clip-relative, while detections and windows use source time.
export function protectExisting(
  events: readonly QuantizedEvent[],
  notes: readonly CompiledNote[],
  offset: number,
) {
  let index = 0;
  return events.flatMap((e) => {
    const time = e.time - offset;
    while (notes[index] && (notes[index]?.time ?? Infinity) <= time) index++;
    const before = notes[index - 1],
      after = notes[index];
    if (before && time < before.time + before.duration + 0.081) return [];
    if (after && after.time - time < 0.081) return [];
    return [
      {
        ...e,
        duration: Math.min(
          e.duration,
          (after?.time ?? Infinity) - time - 0.081,
        ),
      },
    ];
  });
}

export function activeGaps(
  active: readonly TimeSpan[],
  clip: { start: number; duration: number },
  coverage: readonly TimeSpan[],
): TimeSpan[] {
  const covered = [...coverage].sort((a, b) => a.start - b.start);
  const passages: TimeSpan[] = [];
  // Nearby attacks belong to one passage even when a drum decays between hits.
  for (const span of active) {
    const previous = passages.at(-1);
    if (previous && span.start - previous.end <= 0.35) previous.end = span.end;
    else passages.push({ ...span });
  }
  const gaps: TimeSpan[] = [];
  for (const span of passages) {
    let cursor = Math.max(clip.start, span.start);
    const end = Math.min(clip.start + clip.duration, span.end);
    for (const note of covered) {
      if (note.start >= end) break;
      if (note.start - cursor >= 2)
        gaps.push({ start: cursor, end: note.start });
      cursor = Math.max(cursor, note.end);
    }
    if (end - cursor >= 2) gaps.push({ start: cursor, end });
  }
  return gaps;
}
