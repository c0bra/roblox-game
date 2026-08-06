import type {
  ArenaPhraseProgress,
  ArenaRunState,
  ArenaTravel,
} from "./combat-types";
import type { ArenaEncounter, ArenaPositionId } from "./encounter";

class ArenaPositionNotFoundError extends Error {
  override readonly name = "ArenaPositionNotFoundError";

  constructor(readonly positionId: string) {
    super(`Arena position ${positionId} was not found`);
  }
}

export const createArenaRun = (encounter: ArenaEncounter): ArenaRunState => ({
  phase: "running",
  songTime: 0,
  position: "midline",
  ward: encounter.initialWard,
  bossResolve: encounter.initialResolve,
  score: 0,
  hitCount: 0,
  totalJudgments: 0,
  accuracy: 0,
  streak: 0,
  bestStreak: 0,
  exposure: 0,
  resolvedStepIds: [],
  preparedEventIds: [],
  resolvedBossEventIds: [],
  openedEventIds: [],
});

export const positionProfile = (
  encounter: ArenaEncounter,
  positionId: ArenaPositionId,
): ArenaEncounter["positions"][number] => {
  const profile = encounter.positions.find(({ id }) => id === positionId);
  if (!profile) throw new ArenaPositionNotFoundError(positionId);
  return profile;
};

export const completeTravel = (
  state: ArenaRunState,
  time: number,
): { readonly state: ArenaRunState; readonly arrived?: ArenaPositionId } => {
  const travel: ArenaTravel | undefined = state.travel;
  if (!travel || time < travel.end) return { state };
  return {
    state: { ...state, position: travel.to, travel: undefined },
    arrived: travel.to,
  };
};

export const stepsForPosition = (
  phrase: ArenaEncounter["phrases"][number],
  position: ArenaPositionId,
): readonly ArenaEncounter["phrases"][number]["steps"][number][] => {
  const bonus = phrase.positionBonusSteps.find(
    ({ positionId }) => positionId === position,
  );
  return [...phrase.steps, ...(bonus?.steps ?? [])].sort(
    (left, right) => left.time - right.time,
  );
};

export const phraseProgressAt = (
  encounter: ArenaEncounter,
  state: ArenaRunState,
  time: number,
): ArenaPhraseProgress | undefined => {
  const phrase = encounter.phrases.find(
    ({ previewStart, end }) => time >= previewStart && time <= end,
  );
  if (!phrase) return undefined;
  const steps = stepsForPosition(phrase, state.position);
  const unresolved = steps.filter(
    ({ id }) => !state.resolvedStepIds.includes(id),
  );
  const current = unresolved[0];
  const next = unresolved[1];
  return {
    phraseId: phrase.id,
    status: time < phrase.executionStart ? "preview" : "execution",
    ...(current ? { currentStepId: current.id } : {}),
    ...(next ? { nextStepId: next.id } : {}),
    totalSteps: steps.length,
    resolvedSteps: steps.length - unresolved.length,
  };
};
