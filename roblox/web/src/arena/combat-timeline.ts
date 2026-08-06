import {
  completeTravel,
  phraseProgressAt,
  stepsForPosition,
} from "./combat-state";
import type {
  ArenaEffect,
  ArenaRunState,
  ArenaTransition,
} from "./combat-types";
import type { ArenaEncounter } from "./encounter";

const terminal = (state: ArenaRunState): boolean =>
  state.phase === "victory" ||
  state.phase === "failed-resolve" ||
  state.phase === "ward-defeat";

const resolveMisses = (
  encounter: ArenaEncounter,
  state: ArenaRunState,
  time: number,
): ArenaTransition => {
  const missed = encounter.phrases
    .flatMap((phrase) => stepsForPosition(phrase, state.position))
    .filter(
      ({ id, time: stepTime }) =>
        stepTime < time - 0.17 && !state.resolvedStepIds.includes(id),
    )
    .sort((left, right) => left.time - right.time);
  if (missed.length === 0) return { state, effects: [] };
  return {
    state: {
      ...state,
      streak: 0,
      totalJudgments: state.totalJudgments + missed.length,
      accuracy: state.hitCount / (state.totalJudgments + missed.length),
      lastJudgment: { stepId: missed.at(-1)?.id, grade: "miss" },
      resolvedStepIds: [
        ...state.resolvedStepIds,
        ...missed.map(({ id }) => id),
      ],
    },
    effects: missed.map(({ id }) => ({ type: "phrase-miss", stepId: id })),
  };
};

const resolveBossEvents = (
  encounter: ArenaEncounter,
  state: ArenaRunState,
  time: number,
): ArenaTransition => {
  let next = state;
  const effects: ArenaEffect[] = [];
  for (const event of encounter.bossEvents) {
    if (
      time >= event.telegraphStart &&
      !next.preparedEventIds.includes(event.id)
    ) {
      next = {
        ...next,
        preparedEventIds: [...next.preparedEventIds, event.id],
      };
      effects.push({
        type: "boss-prepare",
        eventId: event.id,
        attackType: event.type,
      });
    }
    if (
      time >= event.impactTime &&
      !next.resolvedBossEventIds.includes(event.id)
    ) {
      const avoided = event.safePositionIds.includes(next.position);
      const damage = avoided ? 0 : event.damage;
      next = {
        ...next,
        ward: Math.max(0, next.ward - damage),
        resolvedBossEventIds: [...next.resolvedBossEventIds, event.id],
      };
      effects.push({
        type: "boss-impact",
        eventId: event.id,
        avoided,
        damage,
      });
    }
    if (time >= event.recoveryEnd && !next.openedEventIds.includes(event.id)) {
      next = {
        ...next,
        openedEventIds: [...next.openedEventIds, event.id],
      };
      effects.push({ type: "boss-opening", eventId: event.id });
    }
  }
  return { state: next, effects };
};

const resolveOutcome = (
  encounter: ArenaEncounter,
  state: ArenaRunState,
  time: number,
): ArenaTransition => {
  if (state.ward <= 0) {
    return {
      state: { ...state, phase: "ward-defeat" },
      effects: [{ type: "outcome", outcome: "ward-defeat" }],
    };
  }
  if (time < encounter.finalCadence) return { state, effects: [] };
  const outcome =
    state.bossResolve <= encounter.resolveVictoryThreshold
      ? "victory"
      : "failed-resolve";
  return {
    state: { ...state, phase: outcome },
    effects: [{ type: "outcome", outcome }],
  };
};

export const syncArenaRun = (
  encounter: ArenaEncounter,
  state: ArenaRunState,
  time: number,
): ArenaTransition => {
  if (state.phase === "paused" || terminal(state))
    return { state, effects: [] };
  const travel = completeTravel(state, time);
  const travelEffects: ArenaEffect[] = travel.arrived
    ? [{ type: "move-arrive", position: travel.arrived }]
    : [];
  const activeRepositionWindow = encounter.repositionWindows.find(
    ({ start, deadline }) => time >= start && time <= deadline,
  );
  const timed = {
    ...travel.state,
    songTime: time,
    activeRepositionWindowId: activeRepositionWindow?.id,
  };
  const misses = resolveMisses(encounter, timed, time);
  const boss = resolveBossEvents(encounter, misses.state, time);
  const withPhrase = {
    ...boss.state,
    phraseProgress: phraseProgressAt(encounter, boss.state, time),
  };
  const outcome = resolveOutcome(encounter, withPhrase, time);
  return {
    state: outcome.state,
    effects: [
      ...travelEffects,
      ...misses.effects,
      ...boss.effects,
      ...outcome.effects,
    ],
  };
};
