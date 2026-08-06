import { type ArenaEncounter, arenaEncounterSchema } from "./encounter-schema";
import {
  type ArenaEncounterIssue,
  validateArenaEncounter,
} from "./encounter-validation";

export class ArenaEncounterValidationError extends Error {
  override readonly name = "ArenaEncounterValidationError";

  constructor(readonly issues: readonly ArenaEncounterIssue[]) {
    super(issues.map(({ path, message }) => `${path}: ${message}`).join("; "));
  }
}

export const parseArenaEncounter = (input: unknown): ArenaEncounter => {
  const encounter = arenaEncounterSchema.parse(input);
  const issues = validateArenaEncounter(encounter);
  if (issues.length > 0) throw new ArenaEncounterValidationError(issues);
  return encounter;
};

export type {
  ArenaAction,
  ArenaBossEvent,
  ArenaEncounter,
  ArenaPhrase,
  ArenaPositionId,
  ArenaRepositionWindow,
} from "./encounter-schema";
