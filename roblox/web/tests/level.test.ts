import { describe, expect, test } from "bun:test";
import { chartPath, defaultDifficulty } from "../src/data/level";

describe("level difficulty", () => {
  test("Given a new run, when difficulty is initialized, then Easy is selected", () => {
    expect(defaultDifficulty).toBe("easy");
  });

  test("Given War Drums on Easy, when the chart path is built, then the Easy chart is requested", () => {
    expect(chartPath("drums", "easy")).toBe(
      "/levels/heavens-edge/charts/drums-easy.json",
    );
  });
});
