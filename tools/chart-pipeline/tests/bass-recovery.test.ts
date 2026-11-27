import { expect, test } from "bun:test";
import { compileDifficulties } from "../src/chart-compiler";
import { chartDifficulties } from "../src/chart-format";

const note = (time: number, duration: number) => ({
  time,
  duration,
  pitch: 42,
  strength: 1,
});
test("bass recovery preserves existing notes and holds, respects reviewed windows and reports active gaps in source seconds", () => {
  const input = {
    instrument: "bass",
    clip: { start: 10, duration: 8 },
    beatTimes: Array.from({ length: 17 }, (_, i) => 10 + i / 2),
    maxSnapSeconds: 0.04,
    events: [note(10, 1.2), note(16, 0.4)],
  } as const;
  const base = compileDifficulties(input);
  const result = compileDifficulties({
    ...input,
    reserved: [{ start: 12, end: 13 }],
    bassRecovery: {
      active: [{ start: 10, end: 18 }],
      events: [
        note(10.25, 0.1),
        note(10.75, 0.3),
        note(11.5, 2),
        note(12.5, 0.2),
        note(15.75, 1),
      ],
    },
  });
  for (const d of chartDifficulties) {
    const notes = result.charts[d].notes;
    expect(notes.filter((n) => n.time === 0 || n.time === 6)).toEqual([
      ...base.charts[d].notes,
    ]);
    expect(notes.map((n) => n.time)).toEqual([0, 1.5, 5.75, 6]);
    expect(notes[1]?.duration).toBe(0.42);
    expect(notes[2]?.duration).toBe(0);
  }
  expect(result.report.bassRecovery?.activeGaps).toEqual([
    { start: 13, end: 15.75 },
  ]);
});
