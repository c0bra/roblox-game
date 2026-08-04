import { describe, expect, test } from "bun:test";
import { levelIdSchema } from "../src/data/level-catalog";
import {
  moveSelectionIndex,
  type RunSelection,
  reduceRunSelection,
} from "../src/game/run-selection";

const initial: RunSelection = {
  levelId: levelIdSchema.parse("heavens-edge"),
  instrument: "drums",
  difficulty: "easy",
};

describe("run selection", () => {
  test("Given vocals on hard, when the level changes, then instrument and difficulty remain selected", () => {
    const configured = reduceRunSelection(
      reduceRunSelection(initial, {
        type: "instrument",
        instrument: "vocals",
      }),
      { type: "difficulty", difficulty: "hard" },
    );

    const result = reduceRunSelection(configured, {
      type: "level",
      levelId: levelIdSchema.parse("blackened-crown"),
    });

    expect(result.levelId).toBe(levelIdSchema.parse("blackened-crown"));
    expect(result.instrument).toBe("vocals");
    expect(result.difficulty).toBe("hard");
  });

  test("Given a radio group edge, when keyboard movement occurs, then it wraps or selects the exact edge", () => {
    expect(moveSelectionIndex({ current: 0, count: 2, step: -1 })).toBe(1);
    expect(moveSelectionIndex({ current: 1, count: 2, step: 1 })).toBe(0);
    expect(moveSelectionIndex({ current: 1, count: 3, edge: "first" })).toBe(0);
    expect(moveSelectionIndex({ current: 0, count: 3, edge: "last" })).toBe(2);
  });
});
