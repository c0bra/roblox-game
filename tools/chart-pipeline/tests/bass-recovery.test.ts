import { expect, test } from "bun:test";
import { bassRestrikes } from "../src/bass-recovery";
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

test("a measured bass restrike fits an unused subdivision while preserving the hold and the difficulty cap", () => {
  const result = compileDifficulties({
    instrument: "bass",
    clip: { start: 4, duration: 2 },
    beatTimes: [4, 4.5, 5, 5.5, 6],
    events: [note(4, 0), note(5, 0.45)],
    recovery: {
      events: [],
      active: [{ start: 4, end: 6 }],
      restrikes: [note(4.25, 0), note(4.375, 0), note(5.25, 0)],
    },
    maxSnapSeconds: 0.05,
  });
  expect(result.charts.easy.notes.map((n) => n.time)).toEqual([0, 1]);
  expect(result.charts.medium.notes.map((n) => n.time)).toEqual([0, 0.25, 1]);
  expect(result.charts.hard.notes.map((n) => n.time)).toEqual([
    0, 0.25, 0.375, 1,
  ]);
});

test("a bass attack needs periodic energy after the onset and cannot cross the source rest", () => {
  const pcm = Float32Array.from({ length: 8000 }, (_, i) =>
    i < 4000 ? 0.2 * Math.sin((2 * Math.PI * 40 * i) / 8000) : 0,
  );
  const accepted = bassRestrikes([0.1, 0.25, 0.49, 0.7], pcm, [
    { start: 0, end: 0.5 },
    { start: 0.65, end: 0.9 },
  ]);
  expect(accepted.map((n) => n.time)).toEqual([0.1, 0.25]);
  expect(accepted[0]?.pitch).toBeCloseTo(69 + 12 * Math.log2(40 / 440), 1);
});
