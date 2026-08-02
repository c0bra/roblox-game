import { describe, expect, test } from "bun:test";
import type { ChartNote } from "../src/data/level";
import { judgeTap, resolveAttackWindow } from "../src/game/judgement";

const note = (time: number, lane: 0 | 1 | 2): ChartNote => ({ time, lane });

describe("rhythm judgement", () => {
  test("Given a note 55ms away, when its lane is tapped, then it is Perfect", () => {
    expect(judgeTap([note(10, 1)], new Set(), 10.055, 1)).toEqual({
      grade: "perfect",
      noteIndex: 0,
      offsetMs: 55,
    });
  });

  test("Given a note 165ms away, when its lane is tapped, then it is Good", () => {
    expect(judgeTap([note(10, 2)], new Set(), 9.835, 2).grade).toBe("good");
  });

  test("Given only another lane or a late note, when tapped, then it misses", () => {
    expect(judgeTap([note(10, 0)], new Set(), 10, 1).grade).toBe("miss");
    expect(judgeTap([note(10, 0)], new Set(), 10.18, 0).grade).toBe("miss");
  });
});

describe("boss attack resolution", () => {
  test("Given enough phrase accuracy, when the window closes, then it blocks", () => {
    expect(resolveAttackWindow(7, 10, 0.65)).toBe("blocked");
  });

  test("Given accuracy under its threshold, when the window closes, then it strikes", () => {
    expect(resolveAttackWindow(6, 10, 0.65)).toBe("struck");
  });
});
