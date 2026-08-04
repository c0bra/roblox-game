import { describe, expect, test } from "bun:test";
import { levelCatalog, levelIdSchema } from "../src/data/level-catalog";
import { loadRunAssets } from "../src/data/level-loader";

const chart = {
  instrument: "vocals" as const,
  difficulty: "hard" as const,
  duration: 10,
  notes: [],
  attacks: [],
};

describe("level loader", () => {
  test("Given Blackened Crown vocals on hard, when loaded, then its chart and audio URLs are used", async () => {
    const requested: string[] = [];
    const prepared: { backing: string; stem: string }[] = [];

    await loadRunAssets({
      catalog: levelCatalog,
      selection: {
        levelId: levelIdSchema.parse("blackened-crown"),
        instrument: "vocals",
        difficulty: "hard",
      },
      json: async (url) => {
        requested.push(url);
        return chart;
      },
      audio: {
        prepare: async (urls) => {
          prepared.push(urls);
        },
      },
      qa: false,
    });

    expect(requested).toEqual([
      "/levels/blackened-crown/charts/vocals-hard.json",
    ]);
    expect(prepared).toEqual([
      {
        backing: "/levels/blackened-crown/audio/vocals-backing.m4a",
        stem: "/levels/blackened-crown/audio/vocals-stem.m4a",
      },
    ]);
  });

  test("Given chart metadata for another run, when loaded, then audio is not prepared", async () => {
    let prepareCount = 0;
    await expect(
      loadRunAssets({
        catalog: levelCatalog,
        selection: {
          levelId: levelIdSchema.parse("blackened-crown"),
          instrument: "drums",
          difficulty: "easy",
        },
        json: async () => chart,
        audio: {
          prepare: async () => {
            prepareCount += 1;
          },
        },
        qa: false,
      }),
    ).rejects.toThrow("Chart metadata does not match drums/easy");
    expect(prepareCount).toBe(0);
  });
});
