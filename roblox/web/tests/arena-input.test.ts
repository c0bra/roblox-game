import { describe, expect, test } from "bun:test";
import { arenaActionForCode } from "../src/arena/arena-input";

describe("Arena keyboard input", () => {
  test("Given movement controls, when W or Left Arrow is pressed, then the performer retreats", () => {
    expect(arenaActionForCode("KeyW")).toBe("retreat");
    expect(arenaActionForCode("ArrowLeft")).toBe("retreat");
  });

  test("Given movement controls, when D or Right Arrow is pressed, then the performer advances", () => {
    expect(arenaActionForCode("KeyD")).toBe("advance");
    expect(arenaActionForCode("ArrowRight")).toBe("advance");
  });

  test("Given performance controls, when F or Space is pressed, then the performer performs", () => {
    expect(arenaActionForCode("KeyF")).toBe("perform");
    expect(arenaActionForCode("Space")).toBe("perform");
  });

  test("Given the retired K binding, when K is pressed, then no Arena action is triggered", () => {
    expect(arenaActionForCode("KeyK")).toBeUndefined();
  });
});
