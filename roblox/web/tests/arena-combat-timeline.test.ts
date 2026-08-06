import { describe, expect, test } from "bun:test";
import {
  createArenaRun,
  performArena,
  syncArenaRun,
} from "../src/arena/combat";
import { parseArenaEncounter } from "../src/arena/encounter";
import { validArenaEncounter } from "./fixtures/arena-encounter";

const encounter = parseArenaEncounter(validArenaEncounter);

describe("Arena timeline", () => {
  test("Given phrase preview and execution times, when synchronized, then complete stationary phrase progress is derived", () => {
    const preview = syncArenaRun(encounter, createArenaRun(encounter), 8);
    const execution = syncArenaRun(encounter, preview.state, 10);
    const cleared = syncArenaRun(encounter, execution.state, 14.2);

    expect(preview.state.phraseProgress).toEqual({
      phraseId: "opening-riff",
      status: "preview",
      currentStepId: "opening-1",
      nextStepId: "opening-2",
      totalSteps: 3,
      resolvedSteps: 0,
    });
    expect(execution.state.phraseProgress?.status).toBe("execution");
    expect(cleared.state.phraseProgress).toBeUndefined();
  });

  test("Given a boss telegraph, when synchronized before impact, then prepare is emitted once", () => {
    const first = syncArenaRun(encounter, createArenaRun(encounter), 14.1);
    const second = syncArenaRun(encounter, first.state, 15);

    expect(first.effects).toContainEqual({
      type: "boss-prepare",
      eventId: "rift-sweep",
      attackType: "sweep",
    });
    expect(second.effects).not.toContainEqual({
      type: "boss-prepare",
      eventId: "rift-sweep",
      attackType: "sweep",
    });
  });

  test("Given an unsafe position at impact, when a dropped frame jumps past impact, then ward damage resolves exactly once", () => {
    const first = syncArenaRun(encounter, createArenaRun(encounter), 17.2);
    const second = syncArenaRun(encounter, first.state, 18);

    expect(first.state.ward).toBe(65);
    expect(first.effects).toContainEqual({
      type: "boss-impact",
      eventId: "rift-sweep",
      avoided: false,
      damage: 35,
    });
    expect(second.state.ward).toBe(65);
  });

  test("Given a safe position at impact, when synchronized, then the attack is avoided without damage", () => {
    const state = {
      ...createArenaRun(encounter),
      position: "shelter" as const,
    };
    const result = syncArenaRun(encounter, state, 17);

    expect(result.state.ward).toBe(100);
    expect(result.effects).toContainEqual({
      type: "boss-impact",
      eventId: "rift-sweep",
      avoided: true,
      damage: 0,
    });
  });

  test("Given enough Resolve before the ending, when time advances before final cadence, then the run does not end early", () => {
    const state = { ...createArenaRun(encounter), bossResolve: 20 };
    const result = syncArenaRun(encounter, state, 40);

    expect(result.state.phase).toBe("running");
  });

  test("Given final cadence with enough Resolve, when synchronized, then victory is deterministic", () => {
    const state = { ...createArenaRun(encounter), bossResolve: 20 };
    const result = syncArenaRun(encounter, state, 41.5);

    expect(result.state.phase).toBe("victory");
  });

  test("Given final cadence without enough Resolve, when synchronized, then failed Resolve is defeat", () => {
    const result = syncArenaRun(encounter, createArenaRun(encounter), 41.5);

    expect(result.state.phase).toBe("failed-resolve");
  });

  test("Given lethal impact damage, when synchronized, then ward defeat interrupts the encounter", () => {
    const state = { ...createArenaRun(encounter), ward: 20 };
    const result = syncArenaRun(encounter, state, 17);

    expect(result.state.phase).toBe("ward-defeat");
    expect(result.state.ward).toBe(0);
  });

  test("Given pause and replay are pure state transitions, when reapplied, then time freezes and reset is deterministic", () => {
    const progressed = performArena(encounter, createArenaRun(encounter), 10);
    const paused = { ...progressed.state, phase: "paused" as const };
    const frozen = syncArenaRun(encounter, paused, 20);
    const replay = createArenaRun(encounter);

    expect(frozen.state.songTime).toBe(10);
    expect(replay).toEqual(createArenaRun(encounter));
  });
});
