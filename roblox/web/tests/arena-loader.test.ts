import { describe, expect, test } from "bun:test";
import {
  ArenaEncounterFetchError,
  ArenaEncounterLoadError,
  encounterUrl,
  loadArenaEncounter,
} from "../src/arena/encounter-loader";
import { arenaDemoSelection } from "../src/game/game-mode";
import { validArenaEncounter } from "./fixtures/arena-encounter";

describe("Arena encounter loader", () => {
  test("Given the supported selection, when encounter data loads, then it is parsed once at the boundary", async () => {
    let calls = 0;
    const result = await loadArenaEncounter({
      selection: arenaDemoSelection,
      qa: false,
      json: async () => {
        calls += 1;
        return validArenaEncounter;
      },
    });

    expect(result.id).toBe("heavens-edge-drums-easy-arena");
    expect(calls).toBe(1);
  });

  test("Given QA mode, when the URL is resolved, then the deterministic encounter is selected", () => {
    expect(encounterUrl(arenaDemoSelection, true)).toBe(
      "/levels/heavens-edge/arena/drums-easy.qa.json",
    );
  });

  test("Given mismatched instrument metadata, when loaded, then a selection failure keeps Classic recovery data", async () => {
    const mismatched = { ...validArenaEncounter, instrument: "vocals" };

    const error = await captureLoadError(() =>
      loadArenaEncounter({
        selection: arenaDemoSelection,
        qa: false,
        json: async () => mismatched,
      }),
    );

    expect(error.kind).toBe("selection");
    expect(error.recovery.classicSelection).toBe(arenaDemoSelection);
    expect(error.recovery.canRetry).toBe(false);
  });

  test("Given a missing document, when loaded, then missing recovery can retry or return to Classic", async () => {
    const error = await captureLoadError(() =>
      loadArenaEncounter({
        selection: arenaDemoSelection,
        qa: false,
        json: async () => {
          throw new ArenaEncounterFetchError(404);
        },
      }),
    );

    expect(error.kind).toBe("missing");
    expect(error.recovery.canRetry).toBe(true);
    expect(error.userMessage).toContain("not available");
  });

  test("Given a network failure, when loaded, then it is distinguished from missing data", async () => {
    const error = await captureLoadError(() =>
      loadArenaEncounter({
        selection: arenaDemoSelection,
        qa: false,
        json: async () => {
          throw new ArenaEncounterFetchError(503);
        },
      }),
    );

    expect(error.kind).toBe("network");
  });

  test("Given malformed JSON data, when loaded, then it maps to a parse failure", async () => {
    const error = await captureLoadError(() =>
      loadArenaEncounter({
        selection: arenaDemoSelection,
        qa: false,
        json: async () => ({ version: 1 }),
      }),
    );

    expect(error.kind).toBe("parse");
  });

  test("Given an unknown version, when loaded, then it maps to a version failure", async () => {
    const error = await captureLoadError(() =>
      loadArenaEncounter({
        selection: arenaDemoSelection,
        qa: false,
        json: async () => ({ ...validArenaEncounter, version: 9 }),
      }),
    );

    expect(error.kind).toBe("version");
  });

  test("Given structurally valid but unsorted data, when loaded, then it maps to a semantic failure", async () => {
    const error = await captureLoadError(() =>
      loadArenaEncounter({
        selection: arenaDemoSelection,
        qa: false,
        json: async () => ({ ...validArenaEncounter, beats: [0, 2, 1] }),
      }),
    );

    expect(error.kind).toBe("semantic");
    expect(error.diagnostic).toContain("beats");
  });
});

const captureLoadError = async (
  load: () => Promise<unknown>,
): Promise<ArenaEncounterLoadError> => {
  try {
    await load();
  } catch (error) {
    if (error instanceof ArenaEncounterLoadError) return error;
    throw error;
  }
  throw new Error("Expected Arena encounter load to fail");
};
