import type { ChartDifficulty } from "../chart-format";
import type { QuantizedEvent } from "../types";

const notesPerBeat = { easy: 1, medium: 2, hard: 4 } as const;

export const chooseDifficulty = (
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
    selected.push(
      ...[...beatEvents]
        .sort((left, right) => {
          const leftAnchor = left.slot === 0 ? 2 : left.slot === 2 ? 1 : 0;
          const rightAnchor = right.slot === 0 ? 2 : right.slot === 2 ? 1 : 0;
          return rightAnchor - leftAnchor || right.strength - left.strength;
        })
        .slice(0, notesPerBeat[difficulty]),
    );
  }
  return selected.sort((left, right) => left.time - right.time);
};
