import {
  type Grade,
  gradeForOffsetMilliseconds,
  scoreForGrade,
} from "../game/judgement";
import { positionProfile, stepsForPosition } from "./combat-state";
import type {
  ArenaEffect,
  ArenaRunState,
  ArenaTransition,
} from "./combat-types";
import type { ArenaEncounter } from "./encounter";

const gradeDamageMultiplier: Record<Exclude<Grade, "miss">, number> = {
  perfect: 1,
  great: 0.8,
  good: 0.6,
};

const eligibleSteps = (encounter: ArenaEncounter, state: ArenaRunState) =>
  encounter.phrases.flatMap((phrase) =>
    stepsForPosition(phrase, state.position),
  );

export const performArena = (
  encounter: ArenaEncounter,
  state: ArenaRunState,
  time: number,
): ArenaTransition => {
  const effects: ArenaEffect[] = [{ type: "input-ack", action: "perform" }];
  if (state.phase !== "running") return { state, effects: [] };
  const step = eligibleSteps(encounter, state)
    .filter(({ id }) => !state.resolvedStepIds.includes(id))
    .map((candidate) => ({
      candidate,
      offset: Math.abs(candidate.time - time),
    }))
    .sort((left, right) => left.offset - right.offset)[0];
  const grade = gradeForOffsetMilliseconds((step?.offset ?? Infinity) * 1_000);
  if (!step || grade === "miss") {
    effects.push({ type: "perform-flub", time });
    return {
      state: {
        ...state,
        songTime: time,
        streak: 0,
        totalJudgments: state.totalJudgments + 1,
        accuracy: state.hitCount / (state.totalJudgments + 1),
        lastJudgment: { grade: "miss" },
      },
      effects,
    };
  }

  const profile = positionProfile(encounter, state.position);
  const damage =
    step.candidate.resolveDamage *
    gradeDamageMultiplier[grade] *
    profile.combatMultiplier;
  const streak = state.streak + 1;
  const contactTime = time < step.candidate.time ? step.candidate.time : time;
  const signedOffsetMilliseconds = Math.round(
    (time - step.candidate.time) * 1_000,
  );
  effects.push({
    type: "perform-contact",
    stepId: step.candidate.id,
    grade,
    contactTime,
    timing: time < step.candidate.time ? "scheduled" : "immediate",
    offsetMilliseconds: signedOffsetMilliseconds,
  });
  return {
    state: {
      ...state,
      songTime: time,
      bossResolve: Math.max(0, state.bossResolve - damage),
      score: state.score + scoreForGrade(grade),
      hitCount: state.hitCount + 1,
      totalJudgments: state.totalJudgments + 1,
      accuracy: (state.hitCount + 1) / (state.totalJudgments + 1),
      streak,
      bestStreak: Math.max(state.bestStreak, streak),
      exposure: state.exposure + profile.exposureMultiplier,
      lastJudgment: {
        stepId: step.candidate.id,
        grade,
        offsetMilliseconds: signedOffsetMilliseconds,
      },
      resolvedStepIds: [...state.resolvedStepIds, step.candidate.id],
    },
    effects,
  };
};
