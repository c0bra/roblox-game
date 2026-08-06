import type { Grade } from "../game/judgement";
import type { ArenaPositionId } from "./encounter";

export type ArenaRunPhase =
  | "running"
  | "paused"
  | "victory"
  | "failed-resolve"
  | "ward-defeat";

export type ArenaTravel = {
  readonly from: ArenaPositionId;
  readonly to: ArenaPositionId;
  readonly start: number;
  readonly end: number;
};

export type ArenaPhraseProgress = {
  readonly phraseId: string;
  readonly status: "preview" | "execution";
  readonly currentStepId?: string;
  readonly nextStepId?: string;
  readonly totalSteps: number;
  readonly resolvedSteps: number;
};

export type ArenaJudgment = {
  readonly stepId?: string | undefined;
  readonly grade: Grade;
  readonly offsetMilliseconds?: number;
};

export type ArenaRunState = {
  readonly phase: ArenaRunPhase;
  readonly songTime: number;
  readonly position: ArenaPositionId;
  readonly travel?: ArenaTravel | undefined;
  readonly activeRepositionWindowId?: string | undefined;
  readonly ward: number;
  readonly bossResolve: number;
  readonly phraseProgress?: ArenaPhraseProgress | undefined;
  readonly score: number;
  readonly hitCount: number;
  readonly totalJudgments: number;
  readonly accuracy: number;
  readonly streak: number;
  readonly bestStreak: number;
  readonly exposure: number;
  readonly lastJudgment?: ArenaJudgment | undefined;
  readonly resolvedStepIds: readonly string[];
  readonly preparedEventIds: readonly string[];
  readonly resolvedBossEventIds: readonly string[];
  readonly openedEventIds: readonly string[];
};

export type ArenaEffect =
  | {
      readonly type: "input-ack";
      readonly action: "perform" | "retreat" | "advance";
    }
  | { readonly type: "perform-flub"; readonly time: number }
  | {
      readonly type: "perform-contact";
      readonly stepId: string;
      readonly grade: Exclude<Grade, "miss">;
      readonly contactTime: number;
      readonly timing: "scheduled" | "immediate";
      readonly offsetMilliseconds: number;
    }
  | { readonly type: "phrase-miss"; readonly stepId: string }
  | {
      readonly type: "move-start";
      readonly direction: "retreat" | "advance";
      readonly from: ArenaPositionId;
      readonly to: ArenaPositionId;
      readonly end: number;
    }
  | { readonly type: "move-arrive"; readonly position: ArenaPositionId }
  | { readonly type: "boundary"; readonly direction: "retreat" | "advance" }
  | {
      readonly type: "move-unavailable";
      readonly direction: "retreat" | "advance";
    }
  | {
      readonly type: "boss-prepare";
      readonly eventId: string;
      readonly attackType: "sweep" | "burst";
    }
  | {
      readonly type: "boss-impact";
      readonly eventId: string;
      readonly avoided: boolean;
      readonly damage: number;
    }
  | { readonly type: "boss-opening"; readonly eventId: string }
  | {
      readonly type: "outcome";
      readonly outcome: "victory" | "failed-resolve" | "ward-defeat";
    };

export type ArenaTransition = {
  readonly state: ArenaRunState;
  readonly effects: readonly ArenaEffect[];
};

export type ArenaMoveAction = {
  readonly type: "move";
  readonly direction: "retreat" | "advance";
  readonly time: number;
};
