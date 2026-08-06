import { z } from "zod";
import type { RunSelection } from "../game/run-selection";
import {
  type ArenaEncounter,
  ArenaEncounterValidationError,
  parseArenaEncounter,
} from "./encounter";

export type ArenaEncounterLoadFailureKind =
  | "missing"
  | "network"
  | "parse"
  | "version"
  | "semantic"
  | "selection";

export type ArenaEncounterRecovery = {
  readonly canRetry: boolean;
  readonly classicSelection: RunSelection;
};

export class ArenaEncounterLoadError extends Error {
  override readonly name = "ArenaEncounterLoadError";

  constructor(
    readonly kind: ArenaEncounterLoadFailureKind,
    readonly userMessage: string,
    readonly diagnostic: string,
    readonly recovery: ArenaEncounterRecovery,
  ) {
    super(userMessage);
  }
}

export class ArenaEncounterFetchError extends Error {
  override readonly name = "ArenaEncounterFetchError";

  constructor(readonly status: number) {
    super(`Arena encounter request failed with status ${status}`);
  }
}

export type ArenaEncounterLoaderInput = {
  readonly selection: RunSelection;
  readonly qa: boolean;
  readonly json: (url: string) => Promise<unknown>;
};

export const encounterUrl = (selection: RunSelection, qa: boolean): string =>
  `/levels/${selection.levelId}/arena/${selection.instrument}-${selection.difficulty}${qa ? ".qa" : ""}.json`;

const recoveryFor = (
  selection: RunSelection,
  canRetry: boolean,
): ArenaEncounterRecovery => ({ canRetry, classicSelection: selection });

const isUnknownVersion = (value: unknown): boolean => {
  if (typeof value !== "object" || value === null) return false;
  return "version" in value && value.version !== 1;
};

const loadFailure = (
  kind: ArenaEncounterLoadFailureKind,
  selection: RunSelection,
  diagnostic: string,
): ArenaEncounterLoadError => {
  switch (kind) {
    case "missing":
      return new ArenaEncounterLoadError(
        kind,
        "Arena is not available for this selection yet.",
        diagnostic,
        recoveryFor(selection, true),
      );
    case "network":
      return new ArenaEncounterLoadError(
        kind,
        "Arena could not reach the encounter data.",
        diagnostic,
        recoveryFor(selection, true),
      );
    case "parse":
      return new ArenaEncounterLoadError(
        kind,
        "Arena encounter data could not be read.",
        diagnostic,
        recoveryFor(selection, true),
      );
    case "version":
      return new ArenaEncounterLoadError(
        kind,
        "This Arena encounter needs a newer game version.",
        diagnostic,
        recoveryFor(selection, false),
      );
    case "semantic":
      return new ArenaEncounterLoadError(
        kind,
        "Arena encounter timing is invalid.",
        diagnostic,
        recoveryFor(selection, false),
      );
    case "selection":
      return new ArenaEncounterLoadError(
        kind,
        "Arena data does not match the selected song setup.",
        diagnostic,
        recoveryFor(selection, false),
      );
  }
};

export const loadArenaEncounter = async (
  input: ArenaEncounterLoaderInput,
): Promise<ArenaEncounter> => {
  let raw: unknown;
  try {
    raw = await input.json(encounterUrl(input.selection, input.qa));
    const encounter = parseArenaEncounter(raw);
    if (
      encounter.levelId !== input.selection.levelId ||
      encounter.instrument !== input.selection.instrument ||
      encounter.difficulty !== input.selection.difficulty
    ) {
      throw loadFailure(
        "selection",
        input.selection,
        `${encounter.levelId}/${encounter.instrument}/${encounter.difficulty}`,
      );
    }
    return encounter;
  } catch (error) {
    if (error instanceof ArenaEncounterLoadError) throw error;
    if (error instanceof ArenaEncounterFetchError) {
      throw loadFailure(
        error.status === 404 ? "missing" : "network",
        input.selection,
        error.message,
      );
    }
    if (error instanceof ArenaEncounterValidationError) {
      throw loadFailure("semantic", input.selection, error.message);
    }
    if (error instanceof z.ZodError) {
      throw loadFailure(
        isUnknownVersion(raw) ? "version" : "parse",
        input.selection,
        error.message,
      );
    }
    if (error instanceof SyntaxError) {
      throw loadFailure("parse", input.selection, error.message);
    }
    const diagnostic =
      error instanceof Error ? error.message : "Unknown network error";
    throw loadFailure("network", input.selection, diagnostic);
  }
};

export type ArenaEncounterLoader = (
  selection: RunSelection,
) => Promise<ArenaEncounter>;

export const createBrowserArenaLoader =
  (qa: boolean): ArenaEncounterLoader =>
  async (selection) =>
    loadArenaEncounter({
      selection,
      qa,
      json: async (url) => {
        const response = await fetch(url);
        if (!response.ok) throw new ArenaEncounterFetchError(response.status);
        return response.json();
      },
    });
