import type { BattleAudio } from "../audio/battle-audio";
import { audioUrls, levelCatalog, resolveLevel } from "../data/level-catalog";
import { arenaDemoSelection, arenaSelectionSupport } from "../game/game-mode";
import { type GameModeController, LifecycleScope } from "../game/lifecycle";
import type { RunSelection } from "../game/run-selection";
import { arenaActionForCode } from "./arena-input";
import { ArenaRunSession } from "./arena-run-session";
import { ArenaScene } from "./arena-scene";
import { ArenaSound } from "./arena-sound";
import { ArenaView } from "./arena-view";
import type { ArenaEncounterLoader } from "./encounter-loader";

export class ArenaController implements GameModeController {
  private readonly lifecycle = new LifecycleScope();
  private readonly view: ArenaView;
  private readonly sound = new ArenaSound();
  private scene: ArenaScene | undefined;
  private session: ArenaRunSession | undefined;
  private loadGeneration = 0;
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
    this.listen("#arena-pause", "click", () => this.session?.pause());
    this.listen("#arena-resume", "click", () => this.session?.resume());
    this.listen("#arena-exit", "click", () => this.exit());
    this.listen("#arena-replay", "click", () => this.replay());
    this.listen("#arena-result-exit", "click", () => this.exit());
    this.listen("#arena-use-demo", "click", () => this.useDemoSelection());
    for (const action of ["retreat", "perform", "advance"] as const) {
      this.listen(`[data-arena-action="${action}"]`, "click", () =>
        this.session?.act(action),
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
      if (document.hidden) this.session?.pause();
    });
    this.lifecycle.listen(this.canvas, "webglcontextlost", (event) => {
      event.preventDefault();
      this.loadGeneration += 1;
      this.stopSession();
      this.view.showFallback(
        "The graphics context was lost. Retry Arena or return to Classic.",
      );
    });
    this.lifecycle.own(() => this.scene?.dispose());
    this.lifecycle.own(() => this.sound.dispose());
    this.lifecycle.own(() => this.audio.dispose());
    this.lifecycle.own(() => this.view.dispose());
    this.lifecycle.own(() => this.stopSession());
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
    this.stopSession();
    this.scene?.dispose();
    this.scene = undefined;
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
      const session = new ArenaRunSession({
        audio: this.audio,
        sound: this.sound,
        scene,
        view: this.view,
      });
      this.session = session;
      this.view.showBattle();
      session.start(encounterResult.value);
    } catch (error) {
      if (generation !== this.loadGeneration || this.disposed) return;
      this.loadGeneration += 1;
      scene?.dispose();
      this.scene = undefined;
      console.error("Arena initialization failed", error);
      this.view.showFallback(
        "The Arena could not be initialized. Retry Arena or return to Classic.",
      );
    }
  }

  private replay(): void {
    this.view.showBattle();
    this.session?.replay();
  }

  private exit(): void {
    this.loadGeneration += 1;
    this.stopSession();
    this.scene?.dispose();
    this.scene = undefined;
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
    if (event.code === "Escape") {
      if (this.session?.pause()) event.preventDefault();
      return;
    }
    if (event.repeat || event.target instanceof HTMLInputElement) return;
    const action = arenaActionForCode(event.code);
    if (!action) return;
    event.preventDefault();
    this.session?.act(action);
  }

  private listen(selector: string, type: string, handler: () => void): void {
    const element = this.root.querySelector(selector);
    if (!element) throw new Error(`Missing Arena control: ${selector}`);
    this.lifecycle.listen(element, type, handler);
  }

  private stopSession(): void {
    if (!this.session) {
      this.audio.stop();
      return;
    }
    this.session.stop();
    this.session = undefined;
  }
}
