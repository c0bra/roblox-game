import type { Instrument, Lane } from "../chart-format";
import type { CompiledNote, DrumHitLabel, QuantizedEvent } from "../types";

const minimumSustainSeconds = 0.35;
const releaseGapSeconds = 0.08;
const drumLaneByLabel: Record<DrumHitLabel, Lane> = {
  kick: 0,
  snare: 1,
  hats: 2,
};

const roundSeconds = (seconds: number): number => Number(seconds.toFixed(3));

export class UnclassifiedDrumEvent extends Error {
  override readonly name = "UnclassifiedDrumEvent";

  constructor(readonly time: number) {
    super("Drum events must be classified as kick, snare, or hats");
  }
}

export type LaneAssignmentInput = {
  readonly events: readonly QuantizedEvent[];
  readonly instrument: Instrument;
  readonly clipStart: number;
  readonly clipDuration: number;
};

export const assignLanes = (
  input: LaneAssignmentInput,
): readonly CompiledNote[] => {
  const pitches = input.events
    .map((event) => event.pitch)
    .sort((left, right) => left - right);
  const lower = pitches[Math.floor(pitches.length / 3)] ?? 48;
  const upper = pitches[Math.floor((pitches.length * 2) / 3)] ?? 67;
  let previousLane: Lane = 1;
  let streak = 0;
  const notes = input.events.map((event): CompiledNote => {
    if (input.instrument === "drums") {
      if (!event.label) throw new UnclassifiedDrumEvent(event.time);
      return {
        time: roundSeconds(event.time - input.clipStart),
        lane: drumLaneByLabel[event.label],
        duration: 0,
      };
    }
    let lane: Lane = event.pitch < lower ? 0 : event.pitch > upper ? 2 : 1;
    streak = lane === previousLane ? streak + 1 : 1;
    if (streak > 4) {
      lane = previousLane === 0 ? 1 : previousLane === 1 ? 2 : 1;
      streak = 1;
    }
    previousLane = lane;
    return {
      time: roundSeconds(event.time - input.clipStart),
      lane,
      duration: event.duration,
    };
  });
  return notes.map((note, index) => {
    if (input.instrument === "drums") return note;
    const next = notes[index + 1];
    const available = Math.max(
      0,
      Math.min(
        input.clipDuration - note.time,
        next ? next.time - note.time - releaseGapSeconds : input.clipDuration,
      ),
    );
    const duration = Math.min(note.duration, available);
    return {
      ...note,
      duration: duration >= minimumSustainSeconds ? roundSeconds(duration) : 0,
    };
  });
};
