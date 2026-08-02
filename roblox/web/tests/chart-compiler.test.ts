import { describe, expect, test } from "bun:test";
import { compileDifficulties } from "../scripts/chart-compiler";
import { classifyDrumHits } from "../scripts/drum-onsets";

const rawEvent = (time: number, pitch = 60, strength = 1) => ({
  time,
  pitch,
  strength,
});

describe("chart compiler", () => {
  test("Given a changing-tempo beat grid, when events are compiled, then hard notes use the local sub-beat timestamps", () => {
    const result = compileDifficulties({
      instrument: "guitar",
      events: [
        rawEvent(10.13, 48),
        rawEvent(10.37, 55),
        rawEvent(10.65, 60),
        rawEvent(10.96, 64),
      ],
      beatTimes: [10, 10.5, 11.1],
      clip: { start: 10, duration: 1.1 },
      maxSnapSeconds: 0.08,
    });

    expect(result.charts.hard.notes.map((note) => note.time)).toEqual([
      0.125, 0.375, 0.65, 0.95,
    ]);
  });

  test("Given an event outside snap tolerance, when compiled, then it is rejected instead of silently returning raw timing", () => {
    const result = compileDifficulties({
      instrument: "vocals",
      events: [rawEvent(0), rawEvent(0.06), rawEvent(0.13)],
      beatTimes: [0, 0.5, 1],
      clip: { start: 0, duration: 1 },
      maxSnapSeconds: 0.05,
    });

    expect(result.report.rejectedOffGrid).toBe(1);
    expect(result.charts.hard.notes.map((note) => note.time)).toEqual([
      0, 0.125,
    ]);
  });

  test("Given detected drum transients, when compiled, then notes preserve the audible onset timestamps", () => {
    const result = compileDifficulties({
      instrument: "drums",
      events: [
        { ...rawEvent(0.06), label: "kick" },
        { ...rawEvent(0.13), label: "snare" },
        { ...rawEvent(0.49), label: "hats" },
      ],
      beatTimes: [0, 0.5, 1],
      clip: { start: 0, duration: 1 },
      maxSnapSeconds: 0.08,
    });

    expect(result.charts.hard.notes.map((note) => note.time)).toEqual([
      0.06, 0.13, 0.49,
    ]);
  });

  test("Given unlabeled drum transients, when compiled, then a synthetic lane cycle is rejected", () => {
    expect(() =>
      compileDifficulties({
        instrument: "drums",
        events: [rawEvent(0.06), rawEvent(0.13), rawEvent(0.49)],
        beatTimes: [0, 0.5, 1],
        clip: { start: 0, duration: 1 },
        maxSnapSeconds: 0.08,
      }),
    ).toThrow("Drum events must be classified as kick, snare, or hats");
  });

  test("Given low, mid, and high drum bursts, when classified, then they map to kick, snare, and hats", () => {
    const sampleRate = 16_000;
    const samples = new Float32Array(sampleRate);
    const addBurst = (start: number, frequencies: readonly number[]): void => {
      const startSample = Math.round(start * sampleRate);
      const length = Math.round(0.08 * sampleRate);
      for (let offset = 0; offset < length; offset += 1) {
        const envelope = Math.exp(-offset / (sampleRate * 0.025));
        samples[startSample + offset] =
          envelope *
          frequencies.reduce(
            (sum, frequency) =>
              sum + Math.sin((2 * Math.PI * frequency * offset) / sampleRate),
            0,
          );
      }
    };
    addBurst(0.1, [80]);
    addBurst(0.4, [900, 1_800]);
    addBurst(0.7, [5_800, 6_700]);

    expect(
      classifyDrumHits({
        onsetTimes: [0.1, 0.4, 0.7],
        samples,
        sampleRate,
      }).map((event) => event.label),
    ).toEqual(["kick", "snare", "hats"]);
  });

  test("Given four events per beat, when difficulties are compiled, then density increases from easy to hard", () => {
    const beatTimes = [0, 0.5, 1, 1.5, 2];
    const events = Array.from({ length: 16 }, (_, index) =>
      rawEvent(index * 0.125, 48 + index, 1 + index / 100),
    );
    const result = compileDifficulties({
      instrument: "vocals",
      events,
      beatTimes,
      clip: { start: 0, duration: 2 },
      maxSnapSeconds: 0.02,
    });

    expect(result.charts.easy.notes).toHaveLength(4);
    expect(result.charts.medium.notes).toHaveLength(8);
    expect(result.charts.hard.notes).toHaveLength(16);
    expect(result.report.maxGridErrorMs).toBe(0);
  });
});
