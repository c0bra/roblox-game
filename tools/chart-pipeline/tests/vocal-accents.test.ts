import { expect, test } from "bun:test";
import { compileDifficulties } from "../src/chart-compiler";
import { vocalAccents } from "../src/vocal-accents";

test("a rearticulation is playable on Hard while a steady vowel, vibrato and unvoiced noise are ignored", () => {
  const pcm = Float32Array.from(
    { length: 16000 },
    (_, i) => 0.15 * Math.sin((2 * Math.PI * 220 * i) / 8000),
  );
  for (let i = 3400; i < 4000; i++) pcm[i] = (pcm[i] ?? 0) * 0.15;
  const track = Array.from({ length: 120 }, (_, i) => [
    i / 100,
    220 + 4 * Math.sin(i * 0.4),
  ]);
  const accents = vocalAccents([0.2, 0.5, 0.9, 1.5], track, pcm);
  expect(accents.map((n) => n.time)).toEqual([0.5]);
  const result = compileDifficulties({
    instrument: "vocals",
    events: [],
    clip: { start: 0, duration: 2 },
    beatTimes: [0, 0.5, 1, 1.5, 2],
    maxSnapSeconds: 0.08,
    recovery: {
      events: [],
      active: [{ start: 0, end: 2 }],
      hardAccents: accents,
    },
  });
  expect(result.charts.easy.notes).toHaveLength(0);
  expect(result.charts.medium.notes).toHaveLength(0);
  expect(result.charts.hard.notes.map((n) => n.time)).toEqual([0.5]);
});

test("a stable pitch step needs an attack and cannot interrupt an established Hard hold", () => {
  const pcm = Float32Array.from(
    { length: 8000 },
    (_, i) =>
      0.16 * Math.sin((2 * Math.PI * (i < 4000 ? 220 : 277) * i) / 8000),
  );
  const track = Array.from({ length: 100 }, (_, i) => [
    i / 100,
    i < 50 ? 220 : 277,
  ]);
  const accents = vocalAccents([0.5], track, pcm);
  expect(accents).toHaveLength(1);
  expect(vocalAccents([], track, pcm)).toHaveLength(0);
  const charts = compileDifficulties({
    instrument: "vocals",
    events: [{ time: 0, duration: 0.9, pitch: 57, strength: 1 }],
    clip: { start: 0, duration: 1 },
    beatTimes: [0, 0.5, 1],
    maxSnapSeconds: 0.08,
    recovery: {
      events: [],
      active: [{ start: 0, end: 1 }],
      hardAccents: accents,
    },
  }).charts;
  expect(charts.hard.notes).toEqual(charts.medium.notes);
  expect(charts.hard.notes[0]?.duration).toBe(0.9);
});
