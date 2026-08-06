import { describe, expect, test } from "bun:test";
import {
  createArenaRun,
  performArena,
  syncArenaRun,
} from "../src/arena/combat";
import { parseArenaEncounter } from "../src/arena/encounter";
import { validArenaEncounter } from "./fixtures/arena-encounter";

const encounter = parseArenaEncounter(validArenaEncounter);

describe("Arena perform judgment", () => {
  test.each([
    [10, "perfect"],
    [10.09, "great"],
    [10.16, "good"],
  ] as const)(
    "Given the first step, when Perform occurs at %p, then it earns %p",
    (time, grade) => {
      const result = performArena(encounter, createArenaRun(encounter), time);

      expect(result.state.lastJudgment?.grade).toBe(grade);
    },
  );

  test("Given no eligible step, when Perform occurs, then a flub acknowledges immediately", () => {
    const result = performArena(encounter, createArenaRun(encounter), 9.8);

    expect(result.state.lastJudgment?.grade).toBe("miss");
    expect(result.effects).toEqual([
      { type: "input-ack", action: "perform" },
      { type: "perform-flub", time: 9.8 },
    ]);
  });

  test("Given an early successful input, when judged, then acknowledgement is immediate and contact waits for the beat", () => {
    const result = performArena(encounter, createArenaRun(encounter), 9.9);

    expect(result.effects).toContainEqual({
      type: "perform-contact",
      stepId: "opening-1",
      grade: "great",
      contactTime: 10,
      timing: "scheduled",
      offsetMilliseconds: -100,
    });
  });

  test("Given a late successful input, when judged, then contact is compressed to now rather than presented in the past", () => {
    const result = performArena(encounter, createArenaRun(encounter), 10.1);

    expect(result.effects).toContainEqual({
      type: "perform-contact",
      stepId: "opening-1",
      grade: "great",
      contactTime: 10.1,
      timing: "immediate",
      offsetMilliseconds: 100,
    });
  });

  test("Given an already resolved step, when Perform repeats, then the step cannot resolve twice", () => {
    const first = performArena(encounter, createArenaRun(encounter), 10);
    const second = performArena(encounter, first.state, 10.02);

    expect(second.state.resolvedStepIds).toEqual(["opening-1"]);
    expect(second.effects[1]?.type).toBe("perform-flub");
  });

  test("Given Spotlight, when a Perfect step resolves, then position multiplier increases damage and exposure", () => {
    const state = {
      ...createArenaRun(encounter),
      position: "spotlight" as const,
    };
    const result = performArena(encounter, state, 10);

    expect(result.state.bossResolve).toBe(89.2);
    expect(result.state.exposure).toBe(1.6);
  });

  test("Given an unresolved step passes its Good window, when time jumps ahead, then it becomes one deterministic Miss", () => {
    const result = syncArenaRun(encounter, createArenaRun(encounter), 10.3);

    expect(result.state.resolvedStepIds).toEqual(["opening-1"]);
    expect(result.state.totalJudgments).toBe(1);
    expect(result.state.hitCount).toBe(0);
  });
});
