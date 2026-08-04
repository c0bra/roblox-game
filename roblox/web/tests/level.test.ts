import { describe, expect, test } from "bun:test";
import { defaultDifficulty } from "../src/data/level";

describe("level difficulty", () => {
  test("Given a new run, when difficulty is initialized, then Easy is selected", () => {
    expect(defaultDifficulty).toBe("easy");
  });
});
