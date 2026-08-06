import { describe, expect, test } from "bun:test";
import { createArenaRun } from "../src/arena/combat";
import { parseArenaEncounter } from "../src/arena/encounter";
import { deriveArenaPresentation } from "../src/arena/presentation";
import { validArenaEncounter } from "./fixtures/arena-encounter";

const encounter = parseArenaEncounter(validArenaEncounter);

describe("Arena presentation derivation", () => {
  test("Given ordinary combat time, when presentation is derived, then beat progress exists without a scrolling lane", () => {
    const result = deriveArenaPresentation(
      encounter,
      createArenaRun(encounter),
      12.25,
    );

    expect(result.beat).toEqual({ index: 12, progress: 0.25, downbeat: true });
    expect(result.activeAttack).toBeUndefined();
  });

  test.each([
    [14.5, "prepare"],
    [17, "impact"],
    [18, "recovery"],
  ] as const)(
    "Given Rift Sweep at %p, when presentation is derived, then phase is %p",
    (time, phase) => {
      const result = deriveArenaPresentation(
        encounter,
        createArenaRun(encounter),
        time,
      );

      expect(result.activeAttack?.phase).toBe(phase);
      expect(result.activeAttack?.affectedPositionIds).toEqual([
        "midline",
        "spotlight",
      ]);
      expect(result.positions.find(({ id }) => id === "midline")).toEqual({
        id: "midline",
        current: true,
        state: "danger",
      });
    },
  );

  test("Given post-recovery time, when presentation is derived, then target geometry clears", () => {
    const result = deriveArenaPresentation(
      encounter,
      createArenaRun(encounter),
      20.1,
    );

    expect(result.activeAttack).toBeUndefined();
  });
});
