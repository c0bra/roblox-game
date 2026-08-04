import { describe, expect, test } from "bun:test";
import {
  audioUrls,
  chartUrl,
  levelCatalogSchema,
  levelIdSchema,
  resolveLevel,
} from "../src/data/level-catalog";

const fixture = {
  defaultLevelId: "heavens-edge",
  levels: [
    { id: "heavens-edge", title: "Heaven's Edge" },
    { id: "blackened-crown", title: "Blackened Crown" },
  ],
};

describe("level catalog", () => {
  test("Given two levels, when assets resolve, then their URLs stay isolated", () => {
    const catalog = levelCatalogSchema.parse(fixture);
    const heavensEdge = resolveLevel(
      catalog,
      levelIdSchema.parse("heavens-edge"),
    );
    const blackenedCrown = resolveLevel(
      catalog,
      levelIdSchema.parse("blackened-crown"),
    );

    expect(chartUrl(heavensEdge, "drums", "easy")).toBe(
      "/levels/heavens-edge/charts/drums-easy.json",
    );
    expect(chartUrl(blackenedCrown, "drums", "easy")).toBe(
      "/levels/blackened-crown/charts/drums-easy.json",
    );
    expect(audioUrls(blackenedCrown, "vocals")).toEqual({
      backing: "/levels/blackened-crown/audio/vocals-backing.m4a",
      stem: "/levels/blackened-crown/audio/vocals-stem.m4a",
    });
  });

  test("Given malformed catalog data, when parsed, then it is rejected", () => {
    const malformed = [
      { defaultLevelId: "heavens-edge", levels: [] },
      {
        defaultLevelId: "heavens-edge",
        levels: [
          { id: "heavens-edge", title: "One" },
          { id: "heavens-edge", title: "Two" },
        ],
      },
      {
        defaultLevelId: "missing",
        levels: [{ id: "heavens-edge", title: "Heaven's Edge" }],
      },
      {
        defaultLevelId: "../escape",
        levels: [{ id: "../escape", title: "Escape" }],
      },
      {
        defaultLevelId: "/absolute",
        levels: [{ id: "/absolute", title: "Absolute" }],
      },
      {
        defaultLevelId: "space id",
        levels: [{ id: "space id", title: "Space" }],
      },
      {
        defaultLevelId: "blank-title",
        levels: [{ id: "blank-title", title: "   " }],
      },
    ];

    expect(
      malformed.every((value) => !levelCatalogSchema.safeParse(value).success),
    ).toBe(true);
  });
});
