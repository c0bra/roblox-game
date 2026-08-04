import type { DrumHitLabel, RawChartEvent } from "./types";

export type DrumClassificationInput = {
  readonly onsetTimes: readonly number[];
  readonly samples: Float32Array;
  readonly sampleRate: number;
};

type BandProfile = {
  readonly low: number;
  readonly mid: number;
  readonly high: number;
  readonly strength: number;
};

type MeanProfileInput = {
  readonly profiles: readonly BandProfile[];
  readonly assignments: readonly number[];
  readonly cluster: number;
  readonly fallback: BandProfile;
};

export type ClassifiedDrumHit = RawChartEvent & {
  readonly label: DrumHitLabel;
};

const profileAt = (
  input: DrumClassificationInput,
  onsetTime: number,
): BandProfile => {
  const start = Math.max(0, Math.round(onsetTime * input.sampleRate));
  const end = Math.min(
    input.samples.length,
    start + Math.round(input.sampleRate * 0.08),
  );
  const lowAlpha = 1 - Math.exp((-2 * Math.PI * 180) / input.sampleRate);
  const highAlpha = 1 - Math.exp((-2 * Math.PI * 4_500) / input.sampleRate);
  let lowState = 0;
  let highState = 0;
  let lowEnergy = 0;
  let midEnergy = 0;
  let highEnergy = 0;
  for (let index = start; index < end; index += 1) {
    const sample = input.samples[index] ?? 0;
    lowState += lowAlpha * (sample - lowState);
    highState += highAlpha * (sample - highState);
    const mid = highState - lowState;
    const high = sample - highState;
    lowEnergy += lowState * lowState;
    midEnergy += mid * mid;
    highEnergy += high * high;
  }
  const total = lowEnergy + midEnergy + highEnergy || 1;
  return {
    low: lowEnergy / total,
    mid: midEnergy / total,
    high: highEnergy / total,
    strength: Math.sqrt(total / Math.max(1, end - start)),
  };
};

const profileDistance = (left: BandProfile, right: BandProfile): number =>
  (left.low - right.low) ** 2 +
  (left.mid - right.mid) ** 2 +
  (left.high - right.high) ** 2;

const strongestProfile = (
  profiles: readonly BandProfile[],
  score: (profile: BandProfile) => number,
  excluded: ReadonlySet<number>,
): number => {
  let selected = 0;
  let best = Number.NEGATIVE_INFINITY;
  for (const [index, profile] of profiles.entries()) {
    if (excluded.has(index)) continue;
    const value = score(profile);
    if (value > best) {
      selected = index;
      best = value;
    }
  }
  return selected;
};

const meanProfile = (input: MeanProfileInput): BandProfile => {
  const members = input.profiles.filter(
    (_, index) => input.assignments[index] === input.cluster,
  );
  if (members.length === 0) return input.fallback;
  const sum = members.reduce(
    (total, profile) => ({
      low: total.low + profile.low,
      mid: total.mid + profile.mid,
      high: total.high + profile.high,
      strength: total.strength + profile.strength,
    }),
    { low: 0, mid: 0, high: 0, strength: 0 },
  );
  return {
    low: sum.low / members.length,
    mid: sum.mid / members.length,
    high: sum.high / members.length,
    strength: sum.strength / members.length,
  };
};

const clusterProfiles = (
  profiles: readonly BandProfile[],
): readonly DrumHitLabel[] => {
  if (profiles.length < 3) {
    return profiles.map((profile) =>
      profile.low >= profile.mid && profile.low >= profile.high
        ? "kick"
        : profile.high >= profile.mid
          ? "hats"
          : "snare",
    );
  }
  const used = new Set<number>();
  const kick = strongestProfile(profiles, (profile) => profile.low, used);
  used.add(kick);
  const hats = strongestProfile(profiles, (profile) => profile.high, used);
  used.add(hats);
  const snare = strongestProfile(profiles, (profile) => profile.mid, used);
  const labels = ["kick", "snare", "hats"] as const;
  let centers = [profiles[kick], profiles[snare], profiles[hats]].filter(
    (profile): profile is BandProfile => profile !== undefined,
  );
  let assignments = profiles.map(() => 0);
  for (let iteration = 0; iteration < 12; iteration += 1) {
    assignments = profiles.map((profile) => {
      const distances = centers.map((center) =>
        profileDistance(profile, center),
      );
      return distances.indexOf(Math.min(...distances));
    });
    centers = centers.map((center, cluster) =>
      meanProfile({ profiles, assignments, cluster, fallback: center }),
    );
  }
  return assignments.map((cluster) => labels[cluster] ?? "snare");
};

const drumPitch: Record<DrumHitLabel, number> = {
  kick: 36,
  snare: 38,
  hats: 42,
};

export const classifyDrumHits = (
  input: DrumClassificationInput,
): readonly ClassifiedDrumHit[] => {
  const profiles = input.onsetTimes.map((time) => profileAt(input, time));
  const labels = clusterProfiles(profiles);
  const peakStrength = Math.max(
    Number.EPSILON,
    ...profiles.map((profile) => profile.strength),
  );
  return input.onsetTimes.flatMap((time, index) => {
    const label = labels[index];
    const profile = profiles[index];
    return label && profile
      ? [
          {
            time,
            pitch: drumPitch[label],
            strength: profile.strength / peakStrength,
            duration: 0,
            label,
          },
        ]
      : [];
  });
};
