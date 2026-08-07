import type { BattleAudio } from "../audio/battle-audio";
import { browserScheduler, type RuntimeScheduler } from "../game/lifecycle";
import type { ArenaScene } from "./arena-scene";
import type { ArenaSound } from "./arena-sound";
import type { ArenaView } from "./arena-view";
import {
  type ArenaEffect,
  type ArenaRunState,
  createArenaRun,
  moveArena,
  performArena,
  syncArenaRun,
} from "./combat";
import type { ArenaEncounter } from "./encounter";
import type { ArenaAction } from "./encounter-schema";

export type ArenaRunSessionPorts = {
  readonly audio: Pick<
    BattleAudio,
    "start" | "time" | "pause" | "resume" | "duck" | "accent" | "stop"
  >;
  readonly sound: Pick<ArenaSound, "count" | "playEffect">;
  readonly scene: Pick<ArenaScene, "update" | "playEffect" | "setPaused">;
  readonly view: Pick<
    ArenaView,
    "render" | "announce" | "showPause" | "hidePause" | "showResult"
  >;
  readonly scheduler?: Pick<
    RuntimeScheduler,
    "requestAnimationFrame" | "cancelAnimationFrame"
  >;
};

export type ArenaRunSessionStatus = "idle" | "running" | "paused" | "result";

export class ArenaRunSession {
  private runStatus: ArenaRunSessionStatus = "idle";
  private encounter: ArenaEncounter | undefined;
  private state: ArenaRunState | undefined;
  private frameId = 0;
  private pendingContacts: ArenaEffect[] = [];
  private audioActive = false;
  private readonly scheduler: Pick<
    RuntimeScheduler,
    "requestAnimationFrame" | "cancelAnimationFrame"
  >;

  constructor(private readonly ports: ArenaRunSessionPorts) {
    this.scheduler = ports.scheduler ?? browserScheduler;
  }

  get status(): ArenaRunSessionStatus {
    return this.runStatus;
  }

  start(encounter: ArenaEncounter): void {
    this.cancelFrame();
    this.stopPlayback();
    this.encounter = encounter;
    this.state = createArenaRun(encounter);
    this.pendingContacts = [];
    this.runStatus = "running";
    this.ports.audio.start(0);
    this.audioActive = true;
    this.ports.view.announce("Live · break its Resolve");
    this.ports.sound.count(true);
    this.scheduleFrame();
  }

  act(action: ArenaAction): void {
    const encounter = this.encounter;
    const state = this.state;
    if (this.runStatus !== "running" || !encounter || !state) return;
    const time = this.ports.audio.time;
    const transition =
      action === "perform"
        ? performArena(encounter, state, time)
        : moveArena(encounter, state, {
            type: "move",
            direction: action,
            time,
          });
    this.state = transition.state;
    this.handleEffects(transition.effects, time);
  }

  pause(): boolean {
    const state = this.state;
    if (this.runStatus !== "running" || !state) return false;
    this.ports.audio.pause();
    this.ports.scene.setPaused(true);
    this.state = { ...state, phase: "paused" };
    this.runStatus = "paused";
    this.cancelFrame();
    this.ports.view.showPause();
    return true;
  }

  resume(): void {
    const state = this.state;
    if (this.runStatus !== "paused" || !state) return;
    this.state = { ...state, phase: "running" };
    this.ports.scene.setPaused(false);
    this.ports.audio.resume();
    this.runStatus = "running";
    this.ports.view.hidePause();
    this.scheduleFrame();
  }

  replay(): void {
    const encounter = this.encounter;
    if (encounter) this.start(encounter);
  }

  stop(): void {
    this.cancelFrame();
    this.stopPlayback();
    this.encounter = undefined;
    this.state = undefined;
    this.pendingContacts = [];
    this.runStatus = "idle";
  }

  private readonly tick = (): void => {
    this.frameId = 0;
    const encounter = this.encounter;
    const state = this.state;
    if (this.runStatus !== "running" || !encounter || !state) return;
    const time = Math.min(this.ports.audio.time, encounter.duration);
    const transition = syncArenaRun(encounter, state, time);
    this.state = transition.state;
    this.handleEffects(transition.effects, time);
    this.flushContacts(time);
    this.ports.view.render(encounter, transition.state, time);
    this.ports.scene.update(encounter, transition.state, time);
    if (transition.state.phase === "running") this.scheduleFrame();
    else this.finish();
  };

  private handleEffects(effects: readonly ArenaEffect[], time: number): void {
    for (const effect of effects) {
      if (
        effect.type === "perform-contact" &&
        effect.contactTime > time + 0.01
      ) {
        this.pendingContacts.push(effect);
      } else {
        this.presentEffect(effect, time);
      }
    }
  }

  private flushContacts(time: number): void {
    const ready = this.pendingContacts.filter(
      (effect) =>
        effect.type === "perform-contact" && effect.contactTime <= time,
    );
    this.pendingContacts = this.pendingContacts.filter(
      (effect) =>
        effect.type !== "perform-contact" || effect.contactTime > time,
    );
    for (const effect of ready) this.presentEffect(effect, time);
  }

  private presentEffect(effect: ArenaEffect, time: number): void {
    this.ports.sound.playEffect(effect);
    this.ports.scene.playEffect(effect, time);
    if (effect.type === "perform-flub" || effect.type === "phrase-miss") {
      this.ports.audio.duck();
      this.ports.view.announce("Miss · recover", "miss");
    } else if (effect.type === "perform-contact") {
      const timing =
        effect.offsetMilliseconds < -35
          ? `Early · ${effect.grade}`
          : effect.offsetMilliseconds > 35
            ? `Late · ${effect.grade}`
            : `On time · ${effect.grade}`;
      this.ports.audio.accent(effect.grade);
      this.ports.view.announce(timing, effect.grade);
    } else if (effect.type === "boss-prepare") {
      this.ports.view.announce(
        effect.attackType === "sweep"
          ? "Sweep · seek Shelter"
          : "Burst · seek Spotlight",
      );
    } else if (effect.type === "boss-impact") {
      this.ports.view.announce(
        effect.avoided ? "Evaded" : `Ward −${effect.damage}`,
        effect.avoided ? "great" : "miss",
      );
    } else if (effect.type === "move-unavailable") {
      this.ports.view.announce("Wait for a reposition cue", "miss");
    } else if (effect.type === "boundary") {
      this.ports.view.announce("Position boundary", "miss");
    }
  }

  private finish(): void {
    const encounter = this.encounter;
    const state = this.state;
    if (!encounter || !state) return;
    this.runStatus = "result";
    this.cancelFrame();
    this.stopPlayback();
    this.ports.view.showResult(state, encounter.initialResolve);
  }

  private scheduleFrame(): void {
    this.cancelFrame();
    this.frameId = this.scheduler.requestAnimationFrame(this.tick);
  }

  private cancelFrame(): void {
    if (this.frameId === 0) return;
    this.scheduler.cancelAnimationFrame(this.frameId);
    this.frameId = 0;
  }

  private stopPlayback(): void {
    if (!this.audioActive) return;
    this.ports.audio.stop();
    this.audioActive = false;
  }
}
