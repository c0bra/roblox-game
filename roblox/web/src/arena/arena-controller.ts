import type { BattleAudio } from "../audio/battle-audio";
import { audioUrls, levelCatalog, resolveLevel } from "../data/level-catalog";
import { arenaDemoSelection, arenaSelectionSupport } from "../game/game-mode";
import { type GameModeController, LifecycleScope } from "../game/lifecycle";
import type { RunSelection } from "../game/run-selection";
import { ArenaScene } from "./arena-scene";
import { ArenaSound } from "./arena-sound";
import { ArenaView } from "./arena-view";
import {
  type ArenaEffect,
  type ArenaRunState,
  createArenaRun,
  moveArena,
  performArena,
  syncArenaRun,
} from "./combat";
import type { ArenaEncounter } from "./encounter";
import type { ArenaEncounterLoader } from "./encounter-loader";

type ArenaAction = "retreat" | "perform" | "advance";
type RunStage =
  | "setup"
  | "loading"
  | "rehearsal"
  | "running"
  | "paused"
  | "result";

const keyAction = (event: KeyboardEvent): ArenaAction | undefined => {
  if (event.code === "Space" || event.code === "KeyF") return "perform";
  if (event.code === "KeyD" || event.code === "ArrowLeft") return "retreat";
  if (event.code === "KeyK" || event.code === "ArrowRight") return "advance";
  return undefined;
};

export class ArenaController implements GameModeController {
  private readonly lifecycle = new LifecycleScope();
  private readonly view: ArenaView;
  private readonly sound = new ArenaSound();
  private scene: ArenaScene | undefined;
  private encounter: ArenaEncounter | undefined;
  private state: ArenaRunState | undefined;
  private stage: RunStage = "setup";
  private frameId = 0;
  private loadGeneration = 0;
  private rehearsalStartedAt = 0;
  private pendingContacts: ArenaEffect[] = [];
  private disposed = false;

  constructor(
    private readonly root: HTMLElement,
    private readonly canvas: HTMLCanvasElement,
    private readonly audio: BattleAudio,
    private readonly loadEncounter: ArenaEncounterLoader,
    private selection: RunSelection,
  ) {
    this.view = new ArenaView(root);
  }

  async mount(): Promise<void> {
    this.listen("#arena-start", "click", () => void this.start());
    this.listen("#arena-retry", "click", () => void this.start());
    this.listen("#arena-load-cancel", "click", () => this.exit());
    this.listen("#arena-pause", "click", () => this.pause());
    this.listen("#arena-resume", "click", () => this.resume());
    this.listen("#arena-exit", "click", () => this.exit());
    this.listen("#arena-replay", "click", () => this.replay());
    this.listen("#arena-result-exit", "click", () => this.exit());
    this.listen("#arena-use-demo", "click", () => this.useDemoSelection());
    for (const action of ["retreat", "perform", "advance"] as const) {
      this.listen(`[data-arena-action="${action}"]`, "click", () =>
        this.act(action),
      );
    }
    this.listen('[data-mode="classic"]', "click", () => {
      location.href = "?mode=classic";
    });
    this.lifecycle.listen(window, "keydown", (event) =>
      this.onKey(event as KeyboardEvent),
    );
    this.lifecycle.listen(window, "resize", () => this.scene?.resize());
    this.lifecycle.listen(document, "visibilitychange", () => {
      if (document.hidden && this.stage === "running") this.pause();
    });
    this.lifecycle.listen(this.canvas, "webglcontextlost", (event) => {
      event.preventDefault();
      this.loadGeneration += 1;
      this.audio.stop();
      this.stopFrame();
      this.stage = "setup";
      this.view.showFallback(
        "The graphics context was lost. Retry Arena or return to Classic.",
      );
    });
    this.lifecycle.own(() => this.stopFrame());
    this.lifecycle.own(() => this.scene?.dispose());
    this.lifecycle.own(() => this.sound.dispose());
    this.lifecycle.own(() => this.audio.dispose());
    this.lifecycle.own(() => this.view.dispose());
    this.configureSelection();
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.loadGeneration += 1;
    this.lifecycle.dispose();
  }

  private async start(): Promise<void> {
    const generation = ++this.loadGeneration;
    this.sound.unlock();
    this.audio.stop();
    this.stopFrame();
    this.scene?.dispose();
    this.scene = undefined;
    this.stage = "loading";
    this.view.showLoading(8, "Validating encounter…");
    let scene: ArenaScene | undefined;
    try {
      scene = new ArenaScene(this.canvas);
      this.scene = scene;
      const level = resolveLevel(levelCatalog, this.selection.levelId);
      const [encounterResult, audioResult, sceneResult] =
        await Promise.allSettled([
          this.loadEncounter(this.selection),
          this.audio.prepare(audioUrls(level, this.selection.instrument)),
          scene.load((percent, stage) => {
            if (generation === this.loadGeneration && !this.disposed) {
              this.view.showLoading(percent, stage);
            }
          }),
        ]);
      if (generation !== this.loadGeneration || this.disposed) {
        scene.dispose();
        return;
      }
      if (encounterResult.status === "rejected") throw encounterResult.reason;
      if (audioResult.status === "rejected") throw audioResult.reason;
      if (sceneResult.status === "rejected") throw sceneResult.reason;
      this.encounter = encounterResult.value;
      this.state = createArenaRun(encounterResult.value);
      this.beginRehearsal();
    } catch (error) {
      if (generation !== this.loadGeneration || this.disposed) return;
      this.loadGeneration += 1;
      this.stage = "setup";
      scene?.dispose();
      this.scene = undefined;
      console.error("Arena initialization failed", error);
      this.view.showFallback(
        "The Arena could not be initialized. Retry Arena or return to Classic.",
      );
    }
  }

  private beginRehearsal(): void {
    const encounter = this.encounter;
    if (!encounter) return;
    this.stage = "rehearsal";
    this.rehearsalStartedAt = performance.now() / 1_000;
    this.view.showBattle();
    this.view.announce("Rehearsal · watch the phrase");
    this.sound.count(false);
    this.startFrame();
  }

  private tick = (): void => {
    const encounter = this.encounter;
    const state = this.state;
    if (!encounter || !state) return;
    if (this.stage === "rehearsal") {
      const elapsed = performance.now() / 1_000 - this.rehearsalStartedAt;
      const rehearsalTime = Math.min(elapsed, encounter.rehearsal.duration);
      const rehearsalState = syncArenaRun(
        encounter,
        createArenaRun(encounter),
        rehearsalTime,
      ).state;
      this.view.render(encounter, rehearsalState, rehearsalTime);
      this.scene?.update(encounter, rehearsalState, rehearsalTime);
      if (elapsed >= encounter.rehearsal.duration) this.beginScoredRun();
    } else if (this.stage === "running") {
      const time = Math.min(this.audio.time, encounter.duration);
      const transition = syncArenaRun(encounter, state, time);
      this.state = transition.state;
      this.handleEffects(transition.effects, time);
      this.flushContacts(time);
      this.view.render(encounter, transition.state, time);
      this.scene?.update(encounter, transition.state, time);
      if (transition.state.phase !== "running") this.finish();
    }
    if (this.stage === "rehearsal" || this.stage === "running")
      this.startFrame();
  };

  private beginScoredRun(): void {
    const encounter = this.encounter;
    if (!encounter) return;
    this.state = createArenaRun(encounter);
    this.pendingContacts = [];
    this.audio.start(0);
    this.stage = "running";
    this.view.announce("Live · break its Resolve");
    this.sound.count(true);
  }

  private act(action: ArenaAction): void {
    const encounter = this.encounter;
    const state = this.state;
    if (this.stage !== "running" || !encounter || !state) return;
    const time = this.audio.time;
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

  private handleEffects(effects: readonly ArenaEffect[], time: number): void {
    for (const effect of effects) {
      if (
        effect.type === "perform-contact" &&
        effect.contactTime > time + 0.01
      ) {
        this.pendingContacts.push(effect);
        continue;
      }
      this.presentEffect(effect, time);
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
    this.sound.playEffect(effect);
    this.scene?.playEffect(effect, time);
    if (effect.type === "perform-flub" || effect.type === "phrase-miss") {
      this.audio.duck();
      this.view.announce("Miss · recover", "miss");
    } else if (effect.type === "perform-contact") {
      const timing =
        effect.offsetMilliseconds < -35
          ? `Early · ${effect.grade}`
          : effect.offsetMilliseconds > 35
            ? `Late · ${effect.grade}`
            : `On time · ${effect.grade}`;
      this.audio.accent(effect.grade);
      this.view.announce(timing, effect.grade);
    } else if (effect.type === "boss-prepare") {
      this.view.announce(
        effect.attackType === "sweep"
          ? "Sweep · seek Shelter"
          : "Burst · seek Spotlight",
      );
    } else if (effect.type === "boss-impact") {
      this.view.announce(
        effect.avoided ? "Evaded" : `Ward −${effect.damage}`,
        effect.avoided ? "great" : "miss",
      );
    } else if (effect.type === "move-unavailable") {
      this.view.announce("Wait for a reposition cue", "miss");
    } else if (effect.type === "boundary") {
      this.view.announce("Position boundary", "miss");
    }
  }

  private pause(): void {
    if (this.stage !== "running" || !this.state) return;
    this.audio.pause();
    this.scene?.setPaused(true);
    this.state = { ...this.state, phase: "paused" };
    this.stage = "paused";
    this.stopFrame();
    this.view.showPause();
  }

  private resume(): void {
    if (this.stage !== "paused" || !this.state) return;
    this.state = { ...this.state, phase: "running" };
    this.scene?.setPaused(false);
    this.audio.resume();
    this.stage = "running";
    this.view.hidePause();
    this.startFrame();
  }

  private replay(): void {
    if (!this.encounter) return;
    this.view.showBattle();
    this.beginRehearsal();
  }

  private finish(): void {
    const encounter = this.encounter;
    const state = this.state;
    if (!encounter || !state) return;
    this.stage = "result";
    this.audio.stop();
    this.stopFrame();
    this.view.showResult(state, encounter.initialResolve);
  }

  private exit(): void {
    this.loadGeneration += 1;
    this.audio.stop();
    this.stopFrame();
    this.scene?.dispose();
    this.scene = undefined;
    this.encounter = undefined;
    this.state = undefined;
    this.pendingContacts = [];
    this.stage = "setup";
    this.view.showSetup();
  }

  private useDemoSelection(): void {
    this.selection = arenaDemoSelection;
    this.configureSelection();
  }

  private configureSelection(): void {
    const support = arenaSelectionSupport(this.selection);
    const unsupported =
      this.root.querySelector<HTMLElement>("#arena-unsupported");
    const start = this.root.querySelector<HTMLButtonElement>("#arena-start");
    const recovery = this.root.querySelector<HTMLAnchorElement>(
      "#arena-classic-recovery",
    );
    const setupRecovery = this.root.querySelector<HTMLAnchorElement>(
      "#arena-classic-setup",
    );
    if (!unsupported || !start || !recovery || !setupRecovery) return;
    unsupported.hidden = support.type === "supported";
    start.disabled = support.type !== "supported";
    const query = new URLSearchParams({
      mode: "classic",
      level: this.selection.levelId,
      instrument: this.selection.instrument,
      difficulty: this.selection.difficulty,
    });
    recovery.href = `?${query.toString()}`;
    setupRecovery.href = recovery.href;
  }

  private onKey(event: KeyboardEvent): void {
    if (event.code === "Escape" && this.stage === "running") {
      event.preventDefault();
      this.pause();
      return;
    }
    if (event.repeat || event.target instanceof HTMLInputElement) return;
    const action = keyAction(event);
    if (!action) return;
    event.preventDefault();
    this.act(action);
  }

  private listen(selector: string, type: string, handler: () => void): void {
    const element = this.root.querySelector(selector);
    if (!element) throw new Error(`Missing Arena control: ${selector}`);
    this.lifecycle.listen(element, type, handler);
  }

  private startFrame(): void {
    this.stopFrame();
    this.frameId = requestAnimationFrame(this.tick);
  }

  private stopFrame(): void {
    cancelAnimationFrame(this.frameId);
    this.frameId = 0;
  }
}
