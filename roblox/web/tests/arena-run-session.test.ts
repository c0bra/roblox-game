import { describe, expect, test } from "bun:test";
import {
  ArenaRunSession,
  type ArenaRunSessionPorts,
} from "../src/arena/arena-run-session";
import type { ArenaEffect } from "../src/arena/combat";
import { parseArenaEncounter } from "../src/arena/encounter";
import { validArenaEncounter } from "./fixtures/arena-encounter";

class FakeFrameScheduler {
  readonly frames = new Map<number, FrameRequestCallback>();
  private nextId = 1;

  requestAnimationFrame(callback: FrameRequestCallback): number {
    const id = this.nextId;
    this.nextId += 1;
    this.frames.set(id, callback);
    return id;
  }

  cancelAnimationFrame(id: number): void {
    this.frames.delete(id);
  }

  runNext(timestamp: number): void {
    const entry = this.frames.entries().next().value;
    if (!entry) throw new Error("No Arena frame is scheduled");
    const [id, callback] = entry;
    this.frames.delete(id);
    callback(timestamp);
  }
}

const makeHarness = () => {
  const audio = {
    time: 0,
    startCalls: [] as number[],
    pauseCalls: 0,
    resumeCalls: 0,
    stopCalls: 0,
    duckCalls: 0,
    accentCalls: 0,
    start(offset = 0): void {
      this.startCalls.push(offset);
      this.time = offset;
    },
    pause(): void {
      this.pauseCalls += 1;
    },
    resume(): void {
      this.resumeCalls += 1;
    },
    stop(): void {
      this.stopCalls += 1;
    },
    duck(): void {
      this.duckCalls += 1;
    },
    accent(): void {
      this.accentCalls += 1;
    },
  };
  const sound = {
    counts: [] as boolean[],
    effects: [] as ArenaEffect[],
    count(final: boolean): void {
      this.counts.push(final);
    },
    playEffect(effect: ArenaEffect): void {
      this.effects.push(effect);
    },
  };
  const scene = {
    paused: [] as boolean[],
    effects: [] as ArenaEffect[],
    updateTimes: [] as number[],
    update(_encounter: unknown, _state: unknown, time: number): void {
      this.updateTimes.push(time);
    },
    playEffect(effect: ArenaEffect): void {
      this.effects.push(effect);
    },
    setPaused(paused: boolean): void {
      this.paused.push(paused);
    },
  };
  const view = {
    announcements: [] as string[],
    pauseShows: 0,
    pauseHides: 0,
    resultShows: 0,
    renderTimes: [] as number[],
    render(_encounter: unknown, _state: unknown, time: number): void {
      this.renderTimes.push(time);
    },
    announce(message: string): void {
      this.announcements.push(message);
    },
    showPause(): void {
      this.pauseShows += 1;
    },
    hidePause(): void {
      this.pauseHides += 1;
    },
    showResult(): void {
      this.resultShows += 1;
    },
  };
  const scheduler = new FakeFrameScheduler();
  const ports: ArenaRunSessionPorts = {
    audio,
    sound,
    scene,
    view,
    scheduler,
  };
  return {
    audio,
    sound,
    scene,
    view,
    scheduler,
    session: new ArenaRunSession(ports),
  };
};

const encounter = parseArenaEncounter(validArenaEncounter);

describe("Arena run session", () => {
  test("Given a loaded encounter, when the run starts, then audio and note input are live immediately", () => {
    const harness = makeHarness();

    harness.session.start(encounter);
    harness.session.act("perform");

    expect(harness.audio.startCalls).toEqual([0]);
    expect(harness.sound.counts).toEqual([true]);
    expect(harness.scheduler.frames.size).toBe(1);
    expect(harness.sound.effects.map(({ type }) => type)).toEqual([
      "input-ack",
      "perform-flub",
    ]);
    expect(harness.view.announcements).toEqual([
      "Live · break its Resolve",
      "Miss · recover",
    ]);
  });

  test("Given an animation frame timestamp, when audio remains at zero, then no rehearsal judgments are created", () => {
    const harness = makeHarness();
    harness.session.start(encounter);

    harness.scheduler.runNext(8_000);

    expect(harness.view.renderTimes).toEqual([0]);
    expect(harness.scene.updateTimes).toEqual([0]);
    expect(harness.sound.effects).toEqual([]);
  });

  test("Given running audio, when a frame synchronizes, then the song clock alone advances combat", () => {
    const harness = makeHarness();
    harness.session.start(encounter);
    harness.audio.time = 10.3;

    harness.scheduler.runNext(1);

    expect(harness.view.renderTimes).toEqual([10.3]);
    expect(harness.sound.effects.map(({ type }) => type)).toContain(
      "phrase-miss",
    );
  });

  test("Given an early note hit, when its beat arrives, then deferred contact plays exactly once", () => {
    const harness = makeHarness();
    harness.session.start(encounter);
    harness.audio.time = 9.9;

    harness.session.act("perform");
    expect(harness.sound.effects.map(({ type }) => type)).toEqual([
      "input-ack",
    ]);

    harness.audio.time = 10;
    harness.scheduler.runNext(10_000);
    harness.scheduler.runNext(10_001);

    expect(
      harness.sound.effects.filter(({ type }) => type === "perform-contact"),
    ).toHaveLength(1);
  });

  test("Given a live run, when paused and resumed, then audio and one frame follow the same state", () => {
    const harness = makeHarness();
    harness.session.start(encounter);

    expect(harness.session.pause()).toBe(true);
    expect(harness.session.pause()).toBe(false);
    expect(harness.session.status).toBe("paused");
    expect(harness.audio.pauseCalls).toBe(1);
    expect(harness.scene.paused).toEqual([true]);
    expect(harness.scheduler.frames.size).toBe(0);

    harness.session.resume();

    expect(harness.session.status).toBe("running");
    expect(harness.audio.resumeCalls).toBe(1);
    expect(harness.scene.paused).toEqual([true, false]);
    expect(harness.scheduler.frames.size).toBe(1);
    expect(harness.view.pauseShows).toBe(1);
    expect(harness.view.pauseHides).toBe(1);
  });

  test("Given a completed attempt, when replayed, then audio restarts at zero with one fresh frame", () => {
    const harness = makeHarness();
    harness.session.start(encounter);
    harness.audio.time = 42;
    harness.scheduler.runNext(42_000);

    harness.session.replay();

    expect(harness.audio.startCalls).toEqual([0, 0]);
    expect(harness.session.status).toBe("running");
    expect(harness.scheduler.frames.size).toBe(1);
    expect(harness.sound.counts).toEqual([true, true]);
  });

  test("Given a terminal run, when the session stops repeatedly, then playback and frames stop exactly once", () => {
    const harness = makeHarness();
    harness.session.start(encounter);
    harness.audio.time = 42;

    harness.scheduler.runNext(42_000);

    expect(harness.session.status).toBe("result");
    expect(harness.audio.stopCalls).toBe(1);
    expect(harness.view.resultShows).toBe(1);
    expect(harness.scheduler.frames.size).toBe(0);

    harness.session.stop();
    harness.session.stop();

    expect(harness.session.status).toBe("idle");
    expect(harness.audio.stopCalls).toBe(1);
    expect(harness.scheduler.frames.size).toBe(0);
  });
});
