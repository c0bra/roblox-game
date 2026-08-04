import { describe, expect, test } from "bun:test";
import { chartSchema } from "../src/data/level";
import { createQaChart } from "../src/data/qa-chart";

describe("QA chart", () => {
  test("Given a full chart, when shortened for QA, then timing and attacks are deterministic", () => {
    const chart = chartSchema.parse({
      instrument: "vocals",
      difficulty: "easy",
      duration: 90,
      notes: Array.from({ length: 14 }, (_, index) => ({
        time: index,
        lane: index % 3,
        duration: 1.2,
      })),
      attacks: [],
    });

    const result = createQaChart(chart);

    expect(result.duration).toBe(12);
    expect(result.notes).toHaveLength(12);
    expect(result.notes[0]).toEqual({ time: 1.2, lane: 0, duration: 0.7 });
    expect(result.notes.at(-1)?.time).toBeCloseTo(10.22);
    expect(result.attacks).toEqual([
      { start: 2.8, end: 4.8, threshold: 0.35 },
      { start: 6.5, end: 8.5, threshold: 0.35 },
    ]);
  });

  test("Given a short chart, when shortened for QA, then it does not invent notes", () => {
    const chart = chartSchema.parse({
      instrument: "drums",
      difficulty: "easy",
      duration: 5,
      notes: [{ time: 1, lane: 0, duration: 0 }],
      attacks: [],
    });

    expect(createQaChart(chart).notes).toHaveLength(1);
  });
});
