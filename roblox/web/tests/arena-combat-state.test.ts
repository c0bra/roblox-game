import { describe, expect, test } from "bun:test";
import {
  createArenaRun,
  moveArena,
  positionProfile,
  syncArenaRun,
} from "../src/arena/combat";
import { parseArenaEncounter } from "../src/arena/encounter";
import { validArenaEncounter } from "./fixtures/arena-encounter";

const encounter = parseArenaEncounter(validArenaEncounter);

describe("Arena run state", () => {
  test("Given a new encounter, when the run is created, then it starts at Midline with configured resources", () => {
    const state = createArenaRun(encounter);

    expect(state.position).toBe("midline");
    expect(state.ward).toBe(100);
    expect(state.bossResolve).toBe(100);
    expect(state.phase).toBe("running");
  });

  test("Given each tactical position, when its profile is read, then authored risk and reward are preserved", () => {
    expect(positionProfile(encounter, "shelter").combatMultiplier).toBe(0.75);
    expect(positionProfile(encounter, "midline").combatMultiplier).toBe(1);
    expect(positionProfile(encounter, "spotlight").exposureMultiplier).toBe(
      1.6,
    );
  });

  test("Given an open reposition window at Midline, when retreat is chosen, then visible travel begins toward Shelter", () => {
    const result = moveArena(encounter, createArenaRun(encounter), {
      type: "move",
      direction: "retreat",
      time: 15,
    });

    expect(result.state.travel).toEqual({
      from: "midline",
      to: "shelter",
      start: 15,
      end: 15.8,
    });
    expect(result.effects).toContainEqual({
      type: "move-start",
      direction: "retreat",
      from: "midline",
      to: "shelter",
      end: 15.8,
    });
    expect(syncArenaRun(encounter, result.state, 15.8).state.position).toBe(
      "shelter",
    );
  });

  test("Given Shelter, when retreat is pressed, then the boundary is acknowledged without movement", () => {
    const state = {
      ...createArenaRun(encounter),
      position: "shelter" as const,
    };
    const result = moveArena(encounter, state, {
      type: "move",
      direction: "retreat",
      time: 15,
    });

    expect(result.state.travel).toBeUndefined();
    expect(result.effects).toEqual([
      { type: "boundary", direction: "retreat" },
    ]);
  });

  test("Given the first authored decision, when valid choices are inspected, then retreat, hold, and advance remain player-selected", () => {
    const choices = encounter.repositionWindows[0]?.choices.map(
      ({ action }) => action,
    );

    expect(choices).toEqual(["retreat", "hold", "advance"]);
  });

  test("Given movement after the deadline, when requested, then it cannot change the resolved position", () => {
    const result = moveArena(encounter, createArenaRun(encounter), {
      type: "move",
      direction: "retreat",
      time: 17.1,
    });

    expect(result.state.position).toBe("midline");
    expect(result.state.travel).toBeUndefined();
    expect(result.effects[0]?.type).toBe("move-unavailable");
  });

  test("Given travel is already committed, when another direction is pressed, then the destination is preserved", () => {
    const first = moveArena(encounter, createArenaRun(encounter), {
      type: "move",
      direction: "retreat",
      time: 15,
    });
    const second = moveArena(encounter, first.state, {
      type: "move",
      direction: "advance",
      time: 15.1,
    });

    expect(second.state.travel).toEqual(first.state.travel);
    expect(second.effects).toEqual([
      { type: "move-unavailable", direction: "advance" },
    ]);
  });
});
