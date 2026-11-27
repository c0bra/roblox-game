import { classifyDrumHits } from "./drum-hit-classifier";
import { activeSpans } from "./stem-activity";
import type { InstrumentRecovery, RawChartEvent } from "./types";

const melodicRate = 8000;
const midiPitch = (hz: number) => 69 + 12 * Math.log2(hz / 440);

// A strum can repeat the same chord. Estimate its register only after the pick transient.
function guitarFrequency(samples: Float32Array) {
  let best = 0,
    bestLag = 0;
  if (samples.length < 320) return 0;
  let previous = 1,
    rising = false;
  for (let lag = 6; lag <= 106; lag++) {
    let dot = 0,
      left = 0,
      right = 0;
    for (let i = 106; i < samples.length; i++) {
      const a = samples[i] ?? 0,
        b = samples[i - lag] ?? 0;
      dot += a * b;
      left += a * a;
      right += b * b;
    }
    const correlation = dot / Math.sqrt(left * right || 1);
    // Prefer the shorter period when multiples have nearly identical scores.
    if (rising && correlation < previous && previous > best + 0.015) {
      best = previous;
      bestLag = lag - 1;
    }
    rising = correlation > previous;
    previous = correlation;
  }
  return best >= 0.52 ? melodicRate / bestLag : 0;
}
export function guitarAttacks(
  onsets: readonly number[],
  pcm: Float32Array,
): InstrumentRecovery {
  const active = activeSpans(pcm);
  const events = onsets.flatMap((time) => {
    const span = active.find((s) => s.start <= time && time < s.end);
    if (!span) return [];
    const start = Math.round((time + 0.015) * melodicRate);
    const hz = guitarFrequency(
      pcm.subarray(
        start,
        Math.min(start + 480, Math.round(span.end * melodicRate)),
      ),
    );
    return hz ? [{ time, pitch: midiPitch(hz), duration: 0, strength: 1 }] : [];
  });
  return { active, events };
}

export function vocalPhrases(
  track: readonly (readonly number[])[],
  pcm: Float32Array,
): InstrumentRecovery {
  const active = activeSpans(pcm);
  const events: RawChartEvent[] = [];
  let start = 0,
    last = 0,
    pitches: number[] = [];
  function finish() {
    if (last - start >= 0.09 && pitches.length) {
      pitches.sort((a, b) => a - b);
      events.push({
        time: start,
        duration: last - start,
        pitch: midiPitch(pitches[Math.floor(pitches.length / 2)] ?? 0),
        strength: 1,
      });
    }
    pitches = [];
  }
  let spanIndex = 0,
    envelope = 0,
    dipStart: number | undefined,
    dipPeak = 0;
  let priorTime = -1;

  for (const [time, hz] of track) {
    if (time === undefined || hz === undefined || hz < 55 || hz > 1600)
      continue;
    while (active[spanIndex] && (active[spanIndex]?.end ?? Infinity) <= time) {
      finish();
      spanIndex++;
    }
    const span = active[spanIndex];
    if (!span || time < span.start) continue;
    if (time - priorTime > 0.06 || !pitches.length) {
      finish();
      envelope = 0;
      dipStart = undefined;
    }
    // A voiced consonant can separate two syllables without breaking pYIN voicing.
    const at = Math.round(time * melodicRate);
    const frame = pcm.subarray(at, at + 160);
    const level = Math.sqrt(
      frame.reduce((sum, x) => sum + x * x, 0) / Math.max(1, frame.length),
    );
    if (
      dipStart === undefined &&
      level < envelope * 0.25 &&
      time - start >= 0.09
    ) {
      dipStart = time;
      dipPeak = envelope;
    }
    if (dipStart !== undefined && level >= dipPeak * 0.5) {
      if (time - dipStart >= 0.045) {
        last = dipStart;
        finish();
      }
      dipStart = undefined;
    }
    envelope = Math.max(level, envelope * Math.exp(-(time - priorTime) / 0.2));
    priorTime = time;
    if (!pitches.length) start = time;
    last = time;
    pitches.push(hz);
  }
  finish();
  return { active, events };
}

export function drumTransients(
  samples: Float32Array,
  rate: number,
  offset: number,
): InstrumentRecovery {
  const active = activeSpans(samples, rate);
  const hop = Math.round(rate * 0.01);
  const times: number[] = [];
  let envelope = 0,
    lastHit = -1;
  for (let i = 0; i < samples.length; i += hop) {
    const frame = samples.subarray(i, i + hop);
    const energy = Math.sqrt(
      frame.reduce((sum, x) => sum + x * x, 0) / frame.length,
    );
    const time = i / rate;
    // Attack against the recent envelope. A ringing cymbal or noise floor alone produces no pulse train.
    if (
      energy > Math.max(0.004, envelope * 2.2) &&
      time - lastHit >= 0.09 &&
      active.some((s) => s.start <= time && time < s.end)
    ) {
      times.push(time);
      lastHit = time;
    }
    envelope = Math.max(energy, envelope * 0.82);
  }
  return {
    active: active.map((s) => ({
      start: s.start + offset,
      end: s.end + offset,
    })),
    events: classifyDrumHits({
      samples,
      sampleRate: rate,
      onsetTimes: times,
    }).map((e) => ({ ...e, time: e.time + offset })),
  };
}
