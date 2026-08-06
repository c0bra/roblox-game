import { describe, expect, test } from "bun:test";
import { arenaShell, arenaShowcase } from "../src/arena/arena-template";

describe("Arena templates", () => {
  test("Given Arena setup, when rendered, then mode choice and explicit demo setup are available", () => {
    const html = arenaShell();

    expect(html).toContain('data-mode="classic"');
    expect(html).toContain('data-mode="arena"');
    expect(html).toContain('id="arena-use-demo"');
  });

  test("Given Arena battle, when rendered, then stable labelled controls and no note highway exist", () => {
    const html = arenaShell();

    expect(html).toContain('data-arena-action="retreat"');
    expect(html).toContain('data-arena-action="perform"');
    expect(html).toContain('data-arena-action="advance"');
    expect(html).not.toContain("highway-canvas");
    expect(html).not.toContain("data-lane");
  });

  test("Given the primitive showcase, when rendered, then required UI states are represented", () => {
    const html = arenaShowcase();

    for (const state of [
      "mode-selector",
      "unsupported",
      "phrase-preview",
      "phrase-current-next",
      "positions",
      "reposition",
      "sweep-telegraph",
      "burst-telegraph",
      "meters",
      "loading",
      "fallback",
      "results",
    ]) {
      expect(html).toContain(`data-showcase="${state}"`);
    }
  });
});
