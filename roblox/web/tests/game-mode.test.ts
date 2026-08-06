import { describe, expect, test } from "bun:test";
import { levelIdSchema } from "../src/data/level-catalog";
import {
  arenaDemoSelection,
  arenaSelectionSupport,
  modeFromSearchParams,
  reduceModeSelection,
  selectionFromSearchParams,
} from "../src/game/game-mode";
import type { RunSelection } from "../src/game/run-selection";

const selected: RunSelection = {
  levelId: levelIdSchema.parse("blackened-crown"),
  instrument: "vocals",
  difficulty: "hard",
};

describe("game mode", () => {
  test("Given no mode query, when mode is parsed, then Classic is selected", () => {
    expect(modeFromSearchParams(new URLSearchParams())).toBe("classic");
  });

  test("Given an explicit Arena query, when mode is parsed, then Arena is selected", () => {
    expect(modeFromSearchParams(new URLSearchParams("mode=arena"))).toBe(
      "arena",
    );
  });

  test("Given an invalid mode query, when mode is parsed, then Classic is selected", () => {
    expect(modeFromSearchParams(new URLSearchParams("mode=unknown"))).toBe(
      "classic",
    );
  });

  test("Given a configured run, when mode changes, then the run selection is preserved", () => {
    const result = reduceModeSelection(
      { mode: "classic", selection: selected },
      { type: "select-mode", mode: "arena" },
    );

    expect(result).toEqual({ mode: "arena", selection: selected });
  });

  test("Given a complete run query, when selection is parsed, then all choices are preserved", () => {
    expect(
      selectionFromSearchParams(
        new URLSearchParams(
          "level=blackened-crown&instrument=vocals&difficulty=hard",
        ),
        arenaDemoSelection,
      ),
    ).toEqual(selected);
  });

  test("Given invalid run query values, when selection is parsed, then the fallback is retained", () => {
    expect(
      selectionFromSearchParams(
        new URLSearchParams("level=bad!&instrument=keys&difficulty=nightmare"),
        arenaDemoSelection,
      ),
    ).toEqual(arenaDemoSelection);
  });

  test("Given an unsupported Arena selection, when support is checked, then it remains unchanged and names the demo action", () => {
    const result = arenaSelectionSupport(selected);

    expect(result).toEqual({
      type: "unsupported",
      selection: selected,
      demoSelection: arenaDemoSelection,
    });
  });

  test("Given the supported Arena selection, when support is checked, then it is playable", () => {
    expect(arenaSelectionSupport(arenaDemoSelection)).toEqual({
      type: "supported",
      selection: arenaDemoSelection,
    });
  });

  test("Given an unsupported selection, when the explicit demo action is chosen, then the demo setup replaces it", () => {
    const result = reduceModeSelection(
      { mode: "arena", selection: selected },
      { type: "use-arena-demo" },
    );

    expect(result).toEqual({ mode: "arena", selection: arenaDemoSelection });
  });
});
