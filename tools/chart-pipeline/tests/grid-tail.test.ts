import { expect, test } from "bun:test";
import { compileDifficulties } from "../src/chart-compiler";

test("a guitar attack after the final drum beat survives in a full-song chart", () => {
  const input = {
    instrument: "guitar",
    continueFinalTempo: true,
    beatTimes: [10, 10.5, 11],
    events: [
      { time: 11.019, pitch: 64, duration: 0, strength: 1 },
      { time: 12.013, pitch: 67, duration: 0.43, strength: 1 },
    ],
    clip: { start: 10, duration: 4 },
    maxSnapSeconds: 0.04,
  } as const;
  const chart = compileDifficulties(input).charts.hard;
  expect(
    compileDifficulties({ ...input, continueFinalTempo: false }).charts.hard
      .notes,
  ).toEqual([]);
  expect(chart.duration).toBe(4);
  expect(chart.notes.map(({ time, duration }) => [time, duration])).toEqual([
    [1, 0],
    [2, 0.43],
  ]);
  expect(
    compileDifficulties({ ...input, events: [] }).charts.hard.notes,
  ).toEqual([]);
  expect(
    compileDifficulties({
      ...input,
      events: [{ ...input.events[1], time: 12.06 }],
    }).report.rejectedOffGrid,
  ).toBe(1);
});
