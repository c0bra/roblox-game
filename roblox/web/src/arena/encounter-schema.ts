import { z } from "zod";
import { chartDifficulties, instruments } from "../data/level";
import { levelIdSchema } from "../data/level-catalog";

const secondsSchema = z.number().finite().nonnegative();
export const positionIdSchema = z.enum(["shelter", "midline", "spotlight"]);
export const arenaActionSchema = z.enum(["retreat", "perform", "advance"]);
const positionReferenceSchema = z.string().min(1);

const performStepSchema = z
  .object({
    id: z.string().min(1),
    time: secondsSchema,
    action: z.literal("perform"),
    resolveDamage: z.number().finite().positive(),
  })
  .strict()
  .readonly();

const positionBonusSchema = z
  .object({
    positionId: positionReferenceSchema,
    steps: z.array(performStepSchema).min(1).readonly(),
  })
  .strict()
  .readonly();

const phraseSchema = z
  .object({
    id: z.string().min(1),
    previewStart: secondsSchema,
    executionStart: secondsSchema,
    end: secondsSchema,
    steps: z.array(performStepSchema).min(1).max(5).readonly(),
    positionBonusSteps: z.array(positionBonusSchema).readonly(),
  })
  .strict()
  .readonly();

const positionSchema = z
  .object({
    id: positionIdSchema,
    label: z.string().min(1),
    order: z.number().int().min(0).max(2),
    combatMultiplier: z.number().finite().positive(),
    songInfluence: z.number().finite().positive(),
    exposureMultiplier: z.number().finite().nonnegative(),
  })
  .strict()
  .readonly();

const repositionChoiceSchema = z
  .object({
    from: positionReferenceSchema,
    to: positionReferenceSchema,
    action: z.enum(["retreat", "hold", "advance"]),
  })
  .strict()
  .readonly();

const repositionWindowSchema = z
  .object({
    id: z.string().min(1),
    start: secondsSchema,
    decisionTime: secondsSchema,
    deadline: secondsSchema,
    travelDuration: secondsSchema.positive(),
    bossEventId: z.string().min(1),
    choices: z.array(repositionChoiceSchema).min(1).readonly(),
  })
  .strict()
  .readonly();

const bossEventSchema = z
  .object({
    id: z.string().min(1),
    type: z.enum(["sweep", "burst"]),
    telegraphStart: secondsSchema,
    criticalStart: secondsSchema,
    impactTime: secondsSchema,
    recoveryEnd: secondsSchema,
    affectedPositionIds: z.array(positionReferenceSchema).min(1).readonly(),
    safePositionIds: z.array(positionReferenceSchema).min(1).readonly(),
    damage: z.number().finite().positive(),
    openingEnd: secondsSchema,
  })
  .strict()
  .readonly();

const rehearsalSchema = z
  .object({
    duration: secondsSchema.positive().max(15),
    phraseStepTimes: z.array(secondsSchema).min(3).max(5).readonly(),
    repositionAt: secondsSchema,
  })
  .strict()
  .readonly();

const phaseSchema = z
  .object({
    id: z.string().min(1),
    type: z.enum(["intro", "combat", "climax"]),
    start: secondsSchema,
  })
  .strict()
  .readonly();

export const arenaEncounterSchema = z
  .object({
    version: z.literal(1),
    id: z.string().min(1),
    levelId: levelIdSchema,
    instrument: z.enum(instruments),
    difficulty: z.enum(chartDifficulties),
    duration: secondsSchema.positive().max(45),
    initialWard: z.number().finite().positive(),
    initialResolve: z.number().finite().positive(),
    resolveVictoryThreshold: z.number().finite().nonnegative(),
    finalCadence: secondsSchema,
    rehearsal: rehearsalSchema,
    beats: z.array(secondsSchema).min(2).readonly(),
    downbeats: z.array(secondsSchema).min(1).readonly(),
    positions: z.array(positionSchema).length(3).readonly(),
    phrases: z.array(phraseSchema).min(1).readonly(),
    repositionWindows: z.array(repositionWindowSchema).min(1).readonly(),
    bossEvents: z.array(bossEventSchema).min(2).readonly(),
    phases: z.array(phaseSchema).min(3).readonly(),
  })
  .strict()
  .readonly();

export type ArenaEncounter = z.infer<typeof arenaEncounterSchema>;
export type ArenaPositionId = z.infer<typeof positionIdSchema>;
export type ArenaAction = z.infer<typeof arenaActionSchema>;
export type ArenaPhrase = ArenaEncounter["phrases"][number];
export type ArenaBossEvent = ArenaEncounter["bossEvents"][number];
export type ArenaRepositionWindow = ArenaEncounter["repositionWindows"][number];
