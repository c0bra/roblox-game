import { describe, expect, test } from "bun:test";
import {
  ArenaEncounterValidationError,
  parseArenaEncounter,
} from "../src/arena/encounter";
import { validArenaEncounter } from "./fixtures/arena-encounter";

const changed = (changes: object): unknown => ({
  ...validArenaEncounter,
  ...changes,
});

describe("Arena encounter parsing", () => {
  test("Given a complete encounter, when parsed, then typed data is returned", () => {
    const result = parseArenaEncounter(validArenaEncounter);

    expect(result.id).toBe("heavens-edge-drums-easy-arena");
    expect(result.positions).toHaveLength(3);
    expect(result.bossEvents).toHaveLength(2);
  });

  test("Given an unknown version, when parsed, then the boundary rejects it", () => {
    expect(() => parseArenaEncounter(changed({ version: 2 }))).toThrowError();
  });

  test("Given an unknown action, when parsed, then the boundary rejects it", () => {
    const phrase = validArenaEncounter.phrases[0];
    if (!phrase) throw new Error("Missing phrase fixture");
    const steps = [{ ...phrase.steps[0], action: "jump" }];

    expect(() =>
      parseArenaEncounter(
        changed({
          phrases: [{ ...phrase, steps }, validArenaEncounter.phrases[1]],
        }),
      ),
    ).toThrowError();
  });

  test("Given unsorted beat times, when parsed, then semantic validation identifies beats", () => {
    expect(() => parseArenaEncounter(changed({ beats: [0, 2, 1] }))).toThrow(
      ArenaEncounterValidationError,
    );
  });

  test("Given a non-finite time, when parsed, then the boundary rejects it", () => {
    expect(() =>
      parseArenaEncounter(changed({ beats: [0, Number.POSITIVE_INFINITY] })),
    ).toThrowError();
  });

  test("Given an unknown attacked position, when parsed, then semantic validation identifies the event", () => {
    const attack = validArenaEncounter.bossEvents[0];
    if (!attack) throw new Error("Missing boss event fixture");

    expect(() =>
      parseArenaEncounter(
        changed({
          bossEvents: [
            { ...attack, affectedPositionIds: ["nowhere"] },
            validArenaEncounter.bossEvents[1],
          ],
        }),
      ),
    ).toThrow(ArenaEncounterValidationError);
  });

  test("Given a phrase with less than two beats of preview, when parsed, then it is rejected", () => {
    const phrase = validArenaEncounter.phrases[0];
    if (!phrase) throw new Error("Missing phrase fixture");

    expect(() =>
      parseArenaEncounter(
        changed({ phrases: [{ ...phrase, previewStart: 9.5 }] }),
      ),
    ).toThrow(ArenaEncounterValidationError);
  });

  test("Given a bonus step for an unknown position, when parsed, then it is rejected", () => {
    const phrase = validArenaEncounter.phrases[0];
    if (!phrase) throw new Error("Missing phrase fixture");

    expect(() =>
      parseArenaEncounter(
        changed({
          phrases: [
            {
              ...phrase,
              positionBonusSteps: [
                { positionId: "nowhere", steps: phrase.steps },
              ],
            },
          ],
        }),
      ),
    ).toThrow(ArenaEncounterValidationError);
  });

  test("Given duplicate base and bonus step ids, when parsed, then it is rejected", () => {
    const phrase = validArenaEncounter.phrases[0];
    const baseStep = phrase?.steps[0];
    if (!phrase || !baseStep) throw new Error("Missing phrase fixture");

    expect(() =>
      parseArenaEncounter(
        changed({
          phrases: [
            {
              ...phrase,
              positionBonusSteps: [
                {
                  positionId: "spotlight",
                  steps: [{ ...baseStep, time: phrase.end }],
                },
              ],
            },
          ],
        }),
      ),
    ).toThrow(ArenaEncounterValidationError);
  });

  test("Given a bonus step before execution, when parsed, then it is rejected", () => {
    const phrase = validArenaEncounter.phrases[0];
    const bonus = phrase?.positionBonusSteps[0];
    const step = bonus?.steps[0];
    if (!phrase || !bonus || !step) throw new Error("Missing bonus fixture");

    expect(() =>
      parseArenaEncounter(
        changed({
          phrases: [
            {
              ...phrase,
              positionBonusSteps: [
                {
                  ...bonus,
                  steps: [{ ...step, time: phrase.executionStart - 0.1 }],
                },
              ],
            },
          ],
        }),
      ),
    ).toThrow(ArenaEncounterValidationError);
  });

  test("Given a bonus step after the phrase, when parsed, then it is rejected", () => {
    const phrase = validArenaEncounter.phrases[0];
    const bonus = phrase?.positionBonusSteps[0];
    const step = bonus?.steps[0];
    if (!phrase || !bonus || !step) throw new Error("Missing bonus fixture");

    expect(() =>
      parseArenaEncounter(
        changed({
          phrases: [
            {
              ...phrase,
              positionBonusSteps: [
                { ...bonus, steps: [{ ...step, time: phrase.end + 0.1 }] },
              ],
            },
          ],
        }),
      ),
    ).toThrow(ArenaEncounterValidationError);
  });

  test("Given movement that cannot finish before impact, when parsed, then it is rejected", () => {
    const reposition = validArenaEncounter.repositionWindows[0];
    if (!reposition) throw new Error("Missing reposition fixture");

    expect(() =>
      parseArenaEncounter(
        changed({
          repositionWindows: [
            { ...reposition, deadline: 16.5, travelDuration: 0.8 },
          ],
        }),
      ),
    ).toThrow(ArenaEncounterValidationError);
  });

  test("Given an Easy phrase preview during a critical telegraph, when parsed, then both cues are rejected", () => {
    const phrase = validArenaEncounter.phrases[0];
    if (!phrase) throw new Error("Missing phrase fixture");

    expect(() =>
      parseArenaEncounter(
        changed({
          phrases: [
            {
              ...phrase,
              id: "collision",
              previewStart: 16,
              executionStart: 18,
              end: 21,
              steps: [
                {
                  id: "collision-step",
                  time: 18,
                  action: "perform",
                  resolveDamage: 5,
                },
              ],
              positionBonusSteps: [],
            },
          ],
        }),
      ),
    ).toThrow(ArenaEncounterValidationError);
  });

  test("Given impact before telegraph or recovery, when parsed, then attack ordering is rejected", () => {
    const attack = validArenaEncounter.bossEvents[0];
    if (!attack) throw new Error("Missing boss event fixture");

    expect(() =>
      parseArenaEncounter(
        changed({
          bossEvents: [
            { ...attack, impactTime: 13 },
            validArenaEncounter.bossEvents[1],
          ],
        }),
      ),
    ).toThrow(ArenaEncounterValidationError);
  });

  test("Given an impossible Resolve threshold, when parsed, then it is rejected", () => {
    expect(() =>
      parseArenaEncounter(changed({ resolveVictoryThreshold: 101 })),
    ).toThrow(ArenaEncounterValidationError);
  });

  test("Given an event beyond duration, when parsed, then it is rejected", () => {
    expect(() => parseArenaEncounter(changed({ finalCadence: 45 }))).toThrow(
      ArenaEncounterValidationError,
    );
  });

  test("Given intentionally coincident downbeat, phrase step, and impact, when parsed, then all are preserved", () => {
    const phrase = validArenaEncounter.phrases[0];
    const attack = validArenaEncounter.bossEvents[0];
    if (!phrase || !attack) throw new Error("Missing fixture event");
    const result = parseArenaEncounter(
      changed({
        phrases: [
          {
            ...phrase,
            steps: [
              {
                id: "coincident",
                time: 16,
                action: "perform",
                resolveDamage: 8,
              },
            ],
            end: 17,
          },
        ],
        bossEvents: [
          {
            ...attack,
            criticalStart: 15.5,
            impactTime: 16,
            recoveryEnd: 18,
          },
          validArenaEncounter.bossEvents[1],
        ],
        repositionWindows: [
          {
            ...validArenaEncounter.repositionWindows[0],
            deadline: 15,
            travelDuration: 0.8,
          },
        ],
      }),
    );

    expect(result.downbeats).toContain(16);
    expect(result.phrases[0]?.steps[0]?.time).toBe(16);
    expect(result.bossEvents[0]?.impactTime).toBe(16);
  });
});
