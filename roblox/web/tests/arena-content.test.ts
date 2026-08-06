import { describe, expect, test } from "bun:test";
import { parseArenaEncounter } from "../src/arena/encounter";

describe("authored Arena content", () => {
  test("Given the production encounter file, when parsed, then it covers the selected 42-second slice", async () => {
    const source = await Bun.file(
      "public/levels/heavens-edge/arena/drums-easy.json",
    ).json();
    const encounter = parseArenaEncounter(source);

    expect(encounter.duration).toBe(42);
    expect(encounter.rehearsal.duration).toBe(8);
    expect(encounter.phrases.some(({ steps }) => steps.length >= 3)).toBe(true);
    expect(new Set(encounter.bossEvents.map(({ type }) => type))).toEqual(
      new Set(["sweep", "burst"]),
    );
  });

  test("Given the QA encounter file, when parsed, then it compactly exercises both attacks and all positions", async () => {
    const source = await Bun.file(
      "public/levels/heavens-edge/arena/drums-easy.qa.json",
    ).json();
    const encounter = parseArenaEncounter(source);

    expect(encounter.duration).toBeLessThan(30);
    expect(encounter.positions.map(({ id }) => id)).toEqual([
      "shelter",
      "midline",
      "spotlight",
    ]);
    expect(encounter.bossEvents).toHaveLength(2);
    expect(encounter.repositionWindows[0]?.choices).toHaveLength(3);
  });
});
