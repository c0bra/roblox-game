import type { ArenaRunState } from "./combat";
import type { ArenaEncounter } from "./encounter";

export type ArenaAttackPresentation = {
  readonly eventId: string;
  readonly type: "sweep" | "burst";
  readonly phase: "prepare" | "impact" | "recovery";
  readonly progress: number;
  readonly affectedPositionIds: readonly string[];
  readonly safePositionIds: readonly string[];
};

export type ArenaPositionPresentation = {
  readonly id: ArenaEncounter["positions"][number]["id"];
  readonly current: boolean;
  readonly state: "safe" | "danger" | "neutral";
};

export type ArenaPresentation = {
  readonly beat: {
    readonly index: number;
    readonly progress: number;
    readonly downbeat: boolean;
  };
  readonly positions: readonly ArenaPositionPresentation[];
  readonly activeAttack?: ArenaAttackPresentation | undefined;
};

const activeAttackAt = (
  encounter: ArenaEncounter,
  time: number,
): ArenaAttackPresentation | undefined => {
  const event = encounter.bossEvents.find(
    ({ telegraphStart, openingEnd }) =>
      time >= telegraphStart && time <= openingEnd,
  );
  if (!event) return undefined;
  const phase =
    time < event.impactTime
      ? "prepare"
      : time <= event.impactTime + 0.12
        ? "impact"
        : "recovery";
  const progress =
    phase === "prepare"
      ? (time - event.telegraphStart) /
        (event.impactTime - event.telegraphStart)
      : phase === "impact"
        ? 1
        : (time - event.impactTime) / (event.openingEnd - event.impactTime);
  return {
    eventId: event.id,
    type: event.type,
    phase,
    progress: Math.max(0, Math.min(1, progress)),
    affectedPositionIds: event.affectedPositionIds,
    safePositionIds: event.safePositionIds,
  };
};

export const deriveArenaPresentation = (
  encounter: ArenaEncounter,
  state: ArenaRunState,
  time: number,
): ArenaPresentation => {
  let index = 0;
  for (const [candidate, beatTime] of encounter.beats.entries()) {
    if (beatTime > time) break;
    index = candidate;
  }
  const current = encounter.beats[index] ?? 0;
  const next = encounter.beats[index + 1] ?? current + 1;
  const activeAttack = activeAttackAt(encounter, time);
  const positions: readonly ArenaPositionPresentation[] =
    encounter.positions.map(({ id }) => ({
      id,
      current: id === state.position,
      state: activeAttack?.affectedPositionIds.includes(id)
        ? "danger"
        : activeAttack?.safePositionIds.includes(id)
          ? "safe"
          : "neutral",
    }));
  return {
    beat: {
      index,
      progress: Math.max(0, Math.min(1, (time - current) / (next - current))),
      downbeat: encounter.downbeats.includes(current),
    },
    positions,
    activeAttack,
  };
};
