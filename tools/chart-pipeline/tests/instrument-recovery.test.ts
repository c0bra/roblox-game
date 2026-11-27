import { expect, test } from "bun:test";
import { compileDifficulties } from "../src/chart-compiler";
import {
  drumTransients,
  guitarAttacks,
  vocalPhrases,
} from "../src/instrument-recovery";

const rate = 8000;
test("two same-pitch strums survive, unsupported attacks in silence do not", () => {
  const pcm = new Float32Array(rate);
  for (const at of [0.1, 0.5])
    for (let i = 0; i < rate * 0.18; i++)
      pcm[Math.round(at * rate) + i] =
        0.3 * Math.sin((2 * Math.PI * 196 * i) / rate);
  const result = guitarAttacks([0.1, 0.5, 0.8], pcm);
  expect(result.events.map((n) => n.time)).toEqual([0.1, 0.5]);
});
test("voiced runs form holds, separating syllables across a breath and ignoring unvoiced noise", () => {
  const pcm = new Float32Array(rate * 2).fill(0.15);
  const track = Array.from({ length: 180 }, (_, i) => [
    i * 0.01,
    i < 70 || (i >= 82 && i < 140) ? 210 + Math.sin(i) * 6 : -210,
  ]);
  const notes = vocalPhrases(track, pcm).events;
  expect(notes).toHaveLength(2);
  expect(notes[0]?.duration).toBeCloseTo(0.7, 1);
  expect(notes[1]?.time).toBeCloseTo(0.82, 2);
  expect(vocalPhrases(track, new Float32Array(rate * 2)).events).toHaveLength(
    0,
  );
});
test("isolated drum transients keep source times, enter only vacant beats and get a drum lane", () => {
  const pcm = new Float32Array(rate * 3);
  for (const at of [0.25, 1.25])
    for (let i = 0; i < 400; i++)
      pcm[rate * at + i] =
        0.5 * Math.exp(-i / 80) * Math.sin((2 * Math.PI * 90 * i) / rate);
  const recovered = drumTransients(pcm, rate, 10);
  expect(recovered.events.map((e) => e.time)).toEqual([10.25, 11.25]);
  const result = compileDifficulties({
    instrument: "drums",
    beatTimes: [10, 10.5, 11, 11.5, 12, 12.5, 13],
    clip: { start: 10, duration: 3 },
    maxSnapSeconds: 0.08,
    events: [
      { time: 10.24, pitch: 38, label: "snare", duration: 0, strength: 1 },
    ],
    recovery: recovered,
  });
  expect(result.charts.medium.notes).toEqual([
    { time: 0.24, lane: 1, duration: 0 },
    { time: 1.25, lane: 0, duration: 0 },
  ]);
});

test("two voiced syllables can share pitch across a brief dip in volume", () => {
  const pcm = new Float32Array(rate).fill(0.2);
  pcm.fill(0.025, 0.4 * rate, 0.5 * rate);
  const track = Array.from({ length: 95 }, (_, i) => [i / 100, 205]);
  const notes = vocalPhrases(track, pcm).events;
  expect(notes).toHaveLength(2);
  expect(notes[0]?.duration).toBeCloseTo(0.4, 1);
  expect(notes[1]?.time).toBeCloseTo(0.5, 1);
});

test("a review finding spans repeated drum decays and stops at a genuine rest", () => {
  const active = Array.from({ length: 10 }, (_, i) => ({
    start: 10 + i * 0.25,
    end: 10.1 + i * 0.25,
  }));
  const result = compileDifficulties({
    instrument: "drums",
    clip: { start: 10, duration: 5 },
    beatTimes: [10, 11, 12, 13, 14, 15],
    maxSnapSeconds: 0.08,
    events: [],
    recovery: { active: [...active, { start: 14, end: 14.15 }], events: [] },
  });
  expect(result.report.recovery?.activeGaps).toEqual([
    { start: 10, end: 12.35 },
  ]);
});
