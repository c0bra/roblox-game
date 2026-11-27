import type { RawChartEvent } from "./types";

const sampleRate = 8000;
function rms(pcm: Float32Array, start: number, end: number) {
  const frame = pcm.subarray(
    Math.max(0, Math.round(start * sampleRate)),
    Math.max(0, Math.round(end * sampleRate)),
  );
  return Math.sqrt(
    frame.reduce((sum, x) => sum + x * x, 0) / Math.max(1, frame.length),
  );
}

// pYIN reports voiced frames on the source clock. Require a settled pitch over
// several frames so a breath, a brief pitch glitch or a passing slide is insufficient.
function settledPitch(
  track: readonly (readonly number[])[],
  index: number,
  start: number,
  end: number,
) {
  const pitches: number[] = [];
  let first: number | undefined, last: number | undefined;
  for (let i = index; i < track.length; i++) {
    const row = track[i];
    if (!row) break;
    const [time, hz] = row;
    if (time === undefined || time < start) continue;
    if (time > end) break;
    if (hz === undefined || hz < 55 || hz > 1600) continue;
    if (last !== undefined && time - last > 0.025) return undefined;
    first ??= time;
    last = time;
    pitches.push(69 + 12 * Math.log2(hz / 440));
  }
  if (
    first === undefined ||
    last === undefined ||
    last - first < 0.035 ||
    pitches.length < 4
  )
    return undefined;
  pitches.sort((a, b) => a - b);
  const low = pitches[Math.floor(0.2 * pitches.length)] ?? 0,
    high = pitches[Math.floor(0.8 * pitches.length)] ?? 0;
  return high - low <= 1.3
    ? pitches[Math.floor(pitches.length / 2)]
    : undefined;
}

export function vocalAccents(
  onsets: readonly number[],
  track: readonly (readonly number[])[],
  pcm: Float32Array,
): RawChartEvent[] {
  let index = 0;
  const events: RawChartEvent[] = [];
  for (const time of onsets) {
    while (
      index < track.length &&
      (track[index]?.[0] ?? -Infinity) < time - 0.09
    )
      index++;
    const pitch = settledPitch(track, index, time + 0.012, time + 0.1);
    if (pitch === undefined) continue;
    const after = rms(pcm, time + 0.012, time + 0.07);
    if (after < 0.004) continue;
    const before = rms(pcm, time - 0.065, time - 0.012);
    const earlier = settledPitch(track, index, time - 0.09, time - 0.012);
    const pitchStep = earlier !== undefined && Math.abs(earlier - pitch) >= 2;
    if (after < 1.5 * before && !pitchStep) continue;
    if (time - (events.at(-1)?.time ?? -Infinity) < 0.105) continue;
    events.push({ time, pitch, strength: after, duration: 0 });
  }
  return events;
}
